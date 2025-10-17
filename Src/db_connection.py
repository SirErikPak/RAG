import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from sqlalchemy import text
import pandas as pd
from typing import Optional

# Load variables from .env into environment
load_dotenv()

class AsyncPostgresConnector:
    """
    Async connection manager for PostgreSQL — ideal for RAG/LLM pipelines.
    
    Note: This tool is a specialized component designed to handle the high volume 
    of database interactions required by modern AI applications in a highly efficient, 
    non-blocking way. When this manager is asynchronous (Async), it means it uses 
    Python's asyncio framework to handle connections without waiting for each one to 
    finish before moving to the next.
    """
    def __init__(self):
        # Build the database URL using environment variables for security
        self.db_url = (
            f"postgresql+asyncpg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
            f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME')}"
        )
        self.engine = None
        self.AsyncSessionLocal = None
        # Initialize the engine and session factory
        self.initialize_engine()

    def initialize_engine(self):
        """Initializes the SQLAlchemy engine and connection pool settings."""
        if self.engine is not None:
            return  # Already initialized
        
        print("🔄 Initializing async SQLAlchemy engine...")

        # Create async SQLAlchemy engine with connection pooling settings
        self.engine = create_async_engine(
            self.db_url,
            echo=False,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
        )

        # echo=False	False	Disables query logging. Setting this to True prints every executed SQL query to the console,
        #                       which is helpful for debugging but would overwhelm the console in a high-volume LLM application.
        # pool_size	5	Minimum connections. This sets the number of connections the pool will create and maintain open at all times, 
        #               even if they aren't being used. This ensures your first 5 queries have zero connection setup latency.
        # max_overflow	10	Maximum temporary connections. If all 5 connections in the pool are busy, the pool can temporarily 
        #                   create up to 10 additional connections. This allows the system to absorb traffic spikes 
        #                   (e.g., a sudden rush of users querying the LLM). The total concurrent connections can reach 5+10=15.
        # pool_timeout	30	The number of seconds to wait for a connection to become available before an error is thrown. 
        #                   If the application requests a connection and all 15 are busy, the request will wait for up to 30 seconds.
        # pool_recycle	1800	The number of seconds (30 minutes) after which an unused connection will be gracefully closed and re-opened. 
        #                       This prevents stale connections and helps avoid potential firewall or database server timeouts.

        print("✅ Async SQLAlchemy engine initialized.")

        # Async session factory
        self.AsyncSessionLocal = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        # Setting expire_on_commit=False keeps objects in memory after commit, which is safer and faster 
        # for pipelines where you may immediately read results or use them in downstream async tasks 
        # (like embedding or RAG retrieval).

        print("✅ Session factory initialized.")
        
    async def get_session(self):
        """
        Creates an async database session (use with async context manager).
        """
        async with self.AsyncSessionLocal() as session:
            yield session

    async def close_connection(self):
        """
        Gracefully closes the database engine and disposes of all pooled connections.
        This must be called when the application is shutting down.
        """

        if self.engine:
            print("🛑 Shutting down and gracefully disposing of async engine...")
            await self.engine.dispose()
            self.engine = None
            self.AsyncSessionLocal = None
            print("🗑️ All pooled connections closed.")


    async def test_connection(self):
        """
        Test connection (for diagnostics).
        """
        try:
            async with self.engine.connect() as conn:
                result = await conn.execute(text("SELECT * FROM pg_extension;"))
                rows = result.fetchall()
                columns = result.keys()
                # Dummy pandas import since pandas is not available in all environments
                try:
                    df = pd.DataFrame(rows, columns=columns)
                    print("✅ Connection successful. DataFrame result:")
                    print(df.to_string())
                except NameError:
                    print("✅ Connection successful. First 5 rows of data:")
                    for row in rows[:5]:
                        print(row)

        except Exception as e:
            print(f"❌ Connection failed: {e}")



    async def insert_new_document_metadata(
            self,
            paper_id: str,
            title: str,
            author: str,
            year: int,
            source_path: str,
            embedding_version: Optional[str] = 'default'):
            """
            Inserts a new document's metadata into docs.document_keys and docs.paper_metadata
            within a single, atomic transaction. Returns the generated doc_pk_id.
            """
            print(f"\n📥 Inserting metadata for paper_id: {paper_id}...")
            
            async for session in self.get_session():
                try:
                    # Begin an atomic transaction to ensure both inserts succeed or fail together
                    async with session.begin():
                        # 1. Insert into Document Keys and RETRIEVE the generated Primary Key (doc_pk_id)
                        key_insert_query = text(
                            "INSERT INTO docs.document_keys (paper_id, embedding_model_version) "
                            "VALUES (:p_id, :model_version) RETURNING doc_pk_id;"
                        ).bind_params(p_id=paper_id, model_version=embedding_version)
                        
                        result = await session.execute(key_insert_query)
                        doc_pk_id = result.scalar_one()
                        
                        # 2. Insert into Paper Metadata using the retrieved doc_pk_id
                        metadata_insert_query = text(
                            "INSERT INTO docs.paper_metadata (doc_pk_id, paper_title, author, publication_year, source_path) "
                            "VALUES (:pk_id, :title, :author, :year, :path);"
                        ).bind_params(
                            pk_id=doc_pk_id,
                            title=title,
                            author=author,
                            year=year,
                            path=source_path
                        )
                        
                        await session.execute(metadata_insert_query)
                    
                    # If we reach here, the transaction committed successfully
                    print(f"✅ Metadata inserted successfully. Generated doc_pk_id: {doc_pk_id}")
                    return

                except Exception as e:
                    print(f"❌ Document insertion failed (Transaction rolled back): {e}")
                    # The 'async with session.begin()' automatically handles the rollback on error
                    return None
            




    # Example of a dynamic, parameterized query    
    async def fetch_records_by_status(self, status: str):
        """
        Executes a dynamic, parameterized SQL query to fetch documents 
        based on their processing status.
        """
        print(f"\n🔎 Fetching documents with status: '{status}'...")
        
        # 1. Acquire session safely using the generator
        async for session in self.get_session():
            try:
                # 2. Define the SQL query with a named parameter placeholder (:target_status)
                sql_query = text(
                    "SELECT doc_pk_id, paper_id, processing_status, created_at "
                    "FROM docs.document_keys WHERE processing_status = :target_status;"
                ).bind_params(target_status=status) # 3. Bind the Python variable 'status' to the placeholder

                # Execute the query
                result = await session.execute(sql_query)
                rows = result.fetchall()
                columns = result.keys()

                # Print results (handling environment where pandas might not be installed)
                if rows:
                    print(f"   Found {len(rows)} record(s).")
                    try:
                        df = pd.DataFrame(rows, columns=columns)
                        print(df.to_string(index=False))
                    except NameError:
                        print("   (Pandas not available. Showing first row keys and values):")
                        print(f"   Columns: {columns}")
                        print(f"   Row 1: {rows[0]}")
                else:
                    print("   No records found for this status.")
                
                return rows

            except Exception as e:
                print(f"❌ Dynamic query failed: {e}")
                return []