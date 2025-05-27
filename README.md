# miniRAG_bot

This `README.md` was created using my package [README_genie](https://github.com/browshanravan/README_genie).

A demonstration of a lightweight Retrieval-Augmented Generation (RAG) pipeline using LangChain, Google Vertex AI embeddings, Chroma vector store, and Google’s Gemini LLM (via GenAI). Given a PDF, this bot embeds its contents, indexes them in Chroma, retrieves relevant passages to a user query, and generates an answer with Google’s Gemini model.

---

## About This Project

miniRAG_bot is a minimal, end-to-end RAG example in Python. It shows how to:

- Load and split a PDF into chunks  
- Create embeddings with Google Vertex AI  
- Persist embeddings in Chroma  
- Retrieve context relevant to a question  
- Generate answers with Google’s Gemini LLM  

It is intended as a learning reference for integrating LangChain with Google GenAI tools.

---

## Project Description

This project provides:

1. **PDF Ingestion & Splitting**  
   Uses `langchain_community.document_loaders.PyPDFLoader` and `RecursiveCharacterTextSplitter` to break a PDF into manageable chunks.
2. **Embedding Generation**  
   Calls `VertexAIEmbeddings` to get vector representations via Vertex AI.
3. **Vector Store Persistence**  
   Stores embeddings in a local Chroma database for fast similarity search.
4. **Context Retrieval**  
   Retrieves top-k relevant document chunks for a user’s query via Chroma’s retriever.
5. **LLM Generation**  
   Feeds retrieved passages plus the query to Google’s Gemini model (via `google-genai`) to produce a final answer.

Use cases include PDF Q&A, knowledge-base search over documents, and prototyping RAG workflows with Google Cloud’s GenAI suite.

---

## Features

- Flexible chunk sizing and overlap  
- Persistent vector database on disk  
- Streamed or single-shot LLM generation  
- Configurable embedding & LLM models  
- Simple, zero-boilerplate Python API  

---

## Prerequisites

- Python 3.11
- Google Cloud Project with:
  - Vertex AI API enabled  
  - GenAI (Gemini) API enabled  
- A service account key JSON with permissions for Vertex AI and GenAI  
- Environment variable `GOOGLE_APPLICATION_CREDENTIALS` pointing to your service account JSON  

---

## Installation

1. Clone the repository  
2. (Optional) Create and activate a virtual environment  

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies  

   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

Open `main.py` and adjust the parameters in the “HASH BOX” section:

```python
HOME_PATH = os.environ["HOME"]
PDF_FILE_NAME = "my_cv.pdf"
FULL_PDF_PATH = f"{HOME_PATH}/Downloads/{PDF_FILE_NAME}"

COLLECTION_NAME = "RAG_collection"
CHROMA_DIRECTORY = f"{HOME_PATH}/Downloads/chroma_db"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 10

SERVICE_ACCOUNT_JSON_PATH = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
GCP_PROJECT_NAME = "<your-gcp-project-id>"
LOCATION = "us-central1"

EMBEDDING_MODEL = "text-embedding-004"
GCP_LLM_MODEL = "gemini-2.5-flash-preview-05-20"

STREAM = False
QUERY = "Who is the data scientist?"
```

- **PDF_FILE_NAME** and **FULL_PDF_PATH** point to your PDF.  
- **COLLECTION_NAME** and **CHROMA_DIRECTORY** set your vector store.  
- **CHUNK_SIZE/OVERLAP** control text splitting.  
- **EMBEDDING_MODEL** and **GCP_LLM_MODEL** select Vertex AI and Gemini models.  
- **STREAM** toggles streaming vs. single-shot responses.  
- **QUERY** is the question to ask.

---

## Usage

Run the main script:

```bash
python main.py
```

- On first run, Chroma will index your document.
- The script retrieves passages relevant to `QUERY` and prints the Gemini LLM’s answer.

To ask new questions, either:
1. Change the `QUERY` constant in `main.py` and re-run.  
2. Extend the code for dynamic user input (e.g., via `input()` or CLI flags).

---

## File Structure

```
miniRAG_bot/
├── LICENSE
├── README.md
├── requirements.txt
└── main.py
└── miniRAG_bot/
    └── src/
        └── utils.py         # Core RAG functions
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.