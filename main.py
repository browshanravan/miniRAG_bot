from miniRAG_bot.src.utils import gemini_llm
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from google.oauth2 import service_account
from langchain_chroma import Chroma
import os



HOME_PATH= os.environ["HOME"]
PDF_FILE_NAME= "my_cv.pdf"
FULL_PDF_PATH= f"{HOME_PATH}/Downloads/{PDF_FILE_NAME}"

COLLECTION_NAME= "example_collection"
CHROMA_DIRECTORY= f"{HOME_PATH}/Downloads/chroma_db"

SERVICE_ACCOUNT_JSON_PATH= os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
GCP_PROJECT_NAME= "-".join(SERVICE_ACCOUNT_JSON_PATH.split("/")[-1].split("-")[:-1])
LOCATION= 'us-central1'

EMBEDDING_MODEL= "text-embedding-004"
GCP_LLM_MODEL= "gemini-2.5-flash-preview-05-20"
QUERY= "Who is the data scientist?"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= SERVICE_ACCOUNT_JSON_PATH

gcp_credentials= service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON_PATH)
vertex_ai_embedding = VertexAIEmbeddings(
    model=EMBEDDING_MODEL,
    project= GCP_PROJECT_NAME,
    location= LOCATION,
    credentials= gcp_credentials,
    )


pdf_loader = PyPDFLoader(FULL_PDF_PATH)
pdf_data= pdf_loader.load()
text_splitter= RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0)
documents= text_splitter.split_documents(pdf_data)


vector_db = Chroma(
    collection_name= COLLECTION_NAME,
    embedding_function= vertex_ai_embedding,
    persist_directory= CHROMA_DIRECTORY,
)

document_storage = vector_db.from_documents(documents=documents, embedding=vertex_ai_embedding).as_retriever()
relevant_documents = document_storage.invoke(QUERY)
relevant_contents= [relevant_documents[x].page_content for x in range(len(relevant_documents))]

gemini_llm(
    project= GCP_PROJECT_NAME, 
    location= LOCATION, 
    model= GCP_LLM_MODEL, 
    credentials= gcp_credentials, 
    question= QUERY, 
    contents= relevant_contents
)