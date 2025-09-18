# RAG (Retrieval-Augmented Generation)

This work presents a LangChain-powered Retrieval-Augmented Generation (RAG) pipeline that enables advanced semantic querying over PDF documents. Through systematic text preprocessing, embedding generation, and vectorized search, the system supports both FAISS and Chroma as interchangeable high-performance vector store backends, enhancing flexibility and retrieval accuracy.

## Create New Environment

- `conda create --name rag_env python=3.13`

- For jupyter notebook
  - `conda install ipykernel`
  - `python -m ipykernel install --user --name=RAG`
  - `pip install -U jupyterlab ipywidgets`

## LangChain Community Package

- `pip install 'langchain[all]'`
- `pip install Pandas numpy`
- `pip install langchain-community`

## Install the Unstructured Library

- `pip install "unstructured[all-docs]"`

## Sentence Transformers for HuggingFace Embeddings

- `pip install sentence-transformers`

## Vector Databases

- Chroma DB is an open-source embedding database. It's designed to make it easy to build LLM applications by providing a simple way to store, query, and manage embeddings.
  - `pip install chromadb`
- Faiss is a library for efficient similarity search and clustering of dense vectors. It's developed by Facebook AI and is highly optimized for performance, especially with large datasets.
- Faiss's primary GPU acceleration is built on NVIDIA CUDA. This means that the faiss-gpu package is designed to leverage NVIDIA GPUs, which are typically found in Linux and Windows machines, not macOS.
  - `pip install faiss-cpu`

## Addtional Information

**Verify installation:**

- `pip list | grep langchain`

You should see both langchain and langchain-community listed. If langchain-community is missing, you haven't installed it in the right place.resents a LangChain-powered Retrieval-Augmented Generation (RAG) pipeline that enables advanced semantic querying over PDF documents. Through systematic text preprocessing, embedding generation, and vectorized search, the system supports both FAISS and Chroma as interchangeable high-performance vector store backends, enhancing flexibility and retrieval accuracy.

## LangChain for building LLM-powered applications in Python

To install LangChain for building LLM-powered applications in Python, you can use pip with different options for core and advanced integrations:

### Core LangChain Installation Options

1. **Basic LangChain** (`pip install langchain`)
   - Installs the core LangChain framework and dependencies
   - Enables building chains, agents, and prompt-based applications with LLMs
   - Standard starting point for most users

2. **LangChain with common LLM integrations** (`pip install langchain[llms]`)
   - Includes dependencies for popular LLM providers (OpenAI, Anthropic, etc.)
   - Saves time with broader out-of-the-box support

3. **LangChain with all integrations** (`pip install langchain[all]`)
   - Most comprehensive option with all available integration modules
   - Covers wide range of vector stores, model providers, and tools
   - Perfect for experimentation or complex projects

You should see both langchain and langchain-community listed there. If langchain-community is missing, you haven't installed it in the right place.

## Generate requirements.txt

- pip freeze > requirements.txt
