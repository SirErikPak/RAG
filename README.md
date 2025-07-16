# RAG (Retrieval-Augmented Generation)

This work presents a LangChain-powered Retrieval-Augmented Generation (RAG) pipeline that enables advanced semantic querying over PDF documents. Through systematic text preprocessing, embedding generation, and vectorized search, the system supports both FAISS and Chroma as interchangeable high-performance vector store backends, enhancing flexibility and retrieval accuracy.

## Create New Environment

`conda create —name reg_env python=3.13.5`

- For jupyter notebook
`conda install ipykernel`
`python -m ipykernel install --user --name=rag_env`
`pip install -U jupyterlab ipywidgets`

## LangChain for building LLM-powered applications in Python

To install LangChain for building LLM-powered applications in Python, you simply use pip, with options for core and advanced integrations.
LangChain (pip install langchain)
This command installs the core LangChain framework and its dependencies, enabling you to build chains, agents, and prompt-based applications with LLMs. It's the standard starting point for most users and ensures you get the latest stable version from PyPI.
LangChain with common LLM integrations (pip install langchain[llms])
This option installs LangChain along with dependencies for the most common LLM providers, saving time if you plan to use popular models like OpenAI, Anthropic, or others. It's ideal if you want broader out-of-the-box support for LLM integrations.
LangChain with all integrations (pip install langchain[all])
This command installs LangChain plus all available integration modules, covering a wide range of vector stores, model providers, and tools. It's the most comprehensive option, perfect for experimentation or complex projects involving multiple integrations.
LangChain integration packages (e.g., langchain-openai, langchain-community)
For advanced or specific integrations, install additional packages like langchain-openai or langchain-community after the core LangChain install. This modular approach keeps your environment lean and lets you add only the features you need.

`pip install 'langchain[all]'`

What it is:
A standalone package containing community-contributed integrations for LangChain.
What you get:
Connectors and wrappers for a wide variety of third-party LLMs, vector stores, retrievers, and tools not maintained directly by the LangChain core team.
Examples: Hugging Face endpoints, local Llama models, various document loaders, and more.
When to use:
If you need integrations that are not in the core LangChain package or want access to the latest community-supported features.

`pip install langchain-community`
`pip install numpy pandas`

You should see both langchain and langchain-community listed there. If langchain-community is missing, you haven't installed it in the right place.

`pip list | grep langchain`

## Install the Unstructured Library

For most document types (including PDFs), run
`pip install "unstructured[all-docs]"`

## Poppler is a PDF rendering toolkit. It includes command-line tools like:

- pdfinfo – extracts metadata (used by pdf2image)
- pdftoppm – converts PDFs to images (used internally by unstructured, langchain, etc.)

## These tools are required if you want to process PDFs with Python libraries like:

- pdf2image
- unstructured
- langchain

`conda install -c conda-forge poppler`
`pip install pymupdf`

## should be install already, double check

`pip install -U pypdf`
`pip install -U unstructured`
`pip install -U pdfminer.six`
`pip install -U python-docx`
`pip install -U langchain-community unstructured openpyxl`

## You'll need the sentence-transformers library for HuggingFaceEmbeddings

`pip install sentence-transformers`
