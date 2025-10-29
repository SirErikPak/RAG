import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnablePassthrough

# --- Configuration (Based on your system paths) ---
LOCAL_MODEL_NAME = "/Users/sir/Downloads/HuggingFace/sentence_transformer/intfloat_e5-large-v2"
LLM_MODEL_NAME = "/Users/sir/Downloads/HuggingFace/LLM/Mistral-7B-Instruct-v0.3"
RERANKER_MODEL_NAME = "/Users/sir/Downloads/HuggingFace/cross_encoder/BAAI_bge-reranker-large"
FAISS_INITIAL_K = 10 
RERANKER_FINAL_K = 5
SAVE_DIR = "/Users/sir/Downloads/HuggingFace/VectorDB/faiss_local_index"
PDF_PATH = "/Users/sir/Desktop/Project/Data/NLP/Book/LLMs-in-Production.pdf"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 25

# --- 1. Load and Chunk PDF Documents ---
print("--- 1. Loading and Chunking Documents ---")
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
documents = text_splitter.split_documents(docs)
print(f"Loaded and split into {len(documents)} chunks.")

# --- 2. Embed and Store in FAISS ---
print("\n--- 2. Setting up FAISS and Embeddings ---")
embedding_model = HuggingFaceEmbeddings(model_name=LOCAL_MODEL_NAME)

if not os.path.exists(SAVE_DIR):
    print("Creating new FAISS vector store...")
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(SAVE_DIR)

vectorstore = FAISS.load_local(SAVE_DIR, embedding_model, allow_dangerous_deserialization=True)
print("FAISS vector store loaded.")

# --- 3. Create Base Retriever ---
retriever = vectorstore.as_retriever(search_kwargs={"k": FAISS_INITIAL_K})

# --- 4. Add Reranker ---
print("\n--- 4. Initializing Reranker ---")
reranker_model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_NAME)
reranker = CrossEncoderReranker(model=reranker_model, top_n=RERANKER_FINAL_K)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=retriever
)
print("Compression Retriever (Reranker) initialized.")

# --- 5. Load LLM (Mistral) ---
print(f"\n--- 5. Loading LLM: {LLM_MODEL_NAME} ---")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, dtype=torch.float16, device_map="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=pipe)
print("LLM pipeline initialized.")

# --- 6. RAG Prompt Template ---
# --- EDITED: Adding stricter instructions to prevent meta-commentary and focus on context ---
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful expert. Answer the question STRICTLY using the context provided below. "
     "Do not use external knowledge or generate code examples. "
     "If the answer is not in the context, you MUST state that you cannot find the answer. "
     "Your response must be a direct answer to the question."
    ),
    ("human", "Question: {input}\n\nContext:\n{context}")
])

# --- 7. Document Chain ---
document_chain = create_stuff_documents_chain(llm, rag_prompt)

# --- 8. History-Aware Retriever Prompt ---
history_aware_prompt = ChatPromptTemplate.from_messages([
    ("placeholder", "{chat_history}"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a concise, standalone search query for the latest user question. Do not include any conversational filler."),
])

# --- 9. History-Aware Retriever ---
history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=compression_retriever,
    prompt=history_aware_prompt
)
print("History-Aware Retriever initialized.")

# --- 10. Define retrieval_chain ---
retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)


# --- 11. Final Conversational Chain Setup (Fixes Included) ---
print("\n--- 11. Setting up Conversational Chain (With Fixes) ---")

# 1. Simplified Answer Cleaning Function (Returns only the string to be assigned to 'output')
def clean_answer_string(output_dict):
    """
    Heuristically cleans the verbose Mistral output and returns only the answer string.
    This resolves the verbose output issue.
    """
    raw_answer = output_dict['answer'].strip()
    
    # Step 1: Clean known model/LangChain prefixes
    clean_answer = raw_answer
    if "Answer: " in raw_answer:
        clean_answer = raw_answer.split("Answer: ", 1)[-1].strip()
        
    # Fallback cleaning for older, more verbose structures
    elif "Context:" in raw_answer:
        clean_answer = raw_answer.split("Context:", 1)[-1].strip()
        
    # Step 2: Final cleanup of any residual human prompt elements
    if "Human: Question:" in clean_answer:
        clean_answer = clean_answer.split("Human: Question:", 1)[0].strip()
        
    # Step 3: CRITICAL FIX: Check for the start of the unwanted template/instruction block
    # This specifically targets the prompt leakage seen in the Turn 3 output.
    instruction_start = clean_answer.find("###Instruction:")
    if instruction_start != -1:
        # Truncate the string right before the instruction block starts
        clean_answer = clean_answer[:instruction_start].strip()
        
    return clean_answer # Returns only the clean string!

# 2. History Factory
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Returns a new in-memory history store for a given session ID."""
    return InMemoryChatMessageHistory(session_id=session_id)


# 3. Create the final runnable with the output mapper attached
# This ADDS a new key 'output' whose value is the cleaned string, resolving KeyError.
final_runnable = retrieval_chain | RunnablePassthrough.assign(
    output=clean_answer_string 
)

# 4. Manually wrap the final_runnable with RunnableWithMessageHistory
final_rag_chain = RunnableWithMessageHistory(
    final_runnable,
    get_session_history,
    input_messages_key="input", 
    history_messages_key="chat_history",
    session_history_key="configurable", 
)

print("Final Conversational RAG Chain ready.")

# --- Execution Example ---
session_id = "user_session_1" 

# 1. First question
question_1 = "What is the primary benefit of using LLMs in a production environment?"
print("\n" + "="*50)
print(f"Session ID: {session_id}")
print(f"--- Turn 1: {question_1} ---")

response_1 = final_rag_chain.invoke(
    {"input": question_1}, 
    config={"configurable": {"session_id": session_id}}
)
# We access the new, safe key 'output'
print(f"A: {response_1['output'].strip()}") 


# 2. Second, context-dependent question
question_2 = "What are the common risks associated with that?" 
print(f"\n--- Turn 2: {question_2} ---")

response_2 = final_rag_chain.invoke(
    {"input": question_2}, 
    config={"configurable": {"session_id": session_id}} 
)
# We access the new, safe key 'output'
print(f"A: {response_2['output'].strip()}")

# 3. Third, context-dependent question
question_3 = "What is LLM?"
print(f"\n--- Turn 3: {question_3} ---")

response_3 = final_rag_chain.invoke(
    {"input": question_3},
    config={"configurable": {"session_id": session_id}}
)
# We access the new, safe key 'output'
print(f"A: {response_3['output'].strip()}")

# 4. Fourth, context-dependent question
question_4 = "Who is one of forst company to adapt LLM?"
print(f"\n--- Turn 4: {question_4} ---")

response_4 = final_rag_chain.invoke(
    {"input": question_4},
    config={"configurable": {"session_id": session_id}}
)
# We access the new, safe key 'output'
print(f"A: {response_4['output'].strip()}")
print("="*50)
