# RAG (Retrieval-Augmented Generation)

This work presents a LangChain-powered Retrieval-Augmented Generation (RAG) pipeline that enables advanced semantic querying over PDF documents. Through systematic text preprocessing, embedding generation, and vectorized search, the system supports both FAISS and Chroma as interchangeable high-performance vector store backends, enhancing flexibility and retrieval accuracy.

## Create New Environment

- `conda create --name rag_env python=3.13.5`

- For jupyter notebook
  - `conda install ipykernel`
  - `python -m ipykernel install --user --name=rag_env`
  - `pip install -U jupyterlab ipywidgets`

## LangChain Community Package

- `pip install 'langchain[all]'`

**What it is:**
A standalone package containing community-contributed integrations for LangChain.

**What you get:**

- Connectors and wrappers for third-party LLMs, vector stores, retrievers, and tools
- Examples: Hugging Face endpoints, local Llama models, various document loaders

**When to use:**
If you need integrations not in the core LangChain package or want access to latest community-supported features.

**Installation:**

- `pip install langchain-community`
- `pip install numpy pandas`

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

### Recommended Installation

- `pip install 'langchain[all]'`

What it is:
A standalone package containing community-contributed integrations for LangChain.
What you get:
Connectors and wrappers for a wide variety of third-party LLMs, vector stores, retrievers, and tools not maintained directly by the LangChain core team.
Examples: Hugging Face endpoints, local Llama models, various document loaders, and more.
When to use:
If you need integrations that are not in the core LangChain package or want access to the latest community-supported features.

- `pip install langchain-community`
- `pip install numpy pandas`

You should see both langchain and langchain-community listed there. If langchain-community is missing, you haven't installed it in the right place.

- `pip list | grep langchain`

## Install the Unstructured Library

For most document types (including PDFs), run

- `pip install "unstructured[all-docs]"`

## Poppler PDF Rendering Toolkit

Poppler is a PDF rendering toolkit that includes command-line tools like:

- `pdfinfo` – extracts metadata (used by pdf2image)
- `pdftoppm` – converts PDFs to images (used internally by unstructured, langchain, etc.)

## PDF Processing Libraries

These tools are required if you want to process PDFs with Python libraries like:

- pdf2image
- unstructured  
- langchain

**Installation:**

- `conda install -c conda-forge poppler`
- `pip install pymupdf`

## Additional Dependencies

- Install tesseract and the English language data by default
  - `conda install -c conda-forge tesseract`

**Should be installed already, double check:**

- `pip install -U pypdf`
- `pip install -U unstructured`
- `pip install -U pdfminer.six`
- `pip install -U python-docx`
- `pip install -U langchain-community unstructured openpyxl`

## Sentence Transformers for HuggingFace Embeddings

For embedding models:

- `pip install sentence-transformers`
