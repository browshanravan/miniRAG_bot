from miniRAG_bot.src.utils import (
    generate_gcp_credentials,
    generate_embedding,
    generate_documents,
    store_embedding_chroma,
    retrieve_relevant_contents,
    gemini_llm,
)

import os


##PLEASE CHANGE THE PARAMETERS INSIDE THE HASH BOX BASED ON YOUR SETUP
#######################################################################################
HOME_PATH= os.environ["HOME"]
PDF_FILE_NAME= "my_cv.pdf"
FULL_PDF_PATH= f"{HOME_PATH}/Downloads/{PDF_FILE_NAME}"

COLLECTION_NAME= "RAG_collection"
CHROMA_DIRECTORY= f"{HOME_PATH}/Downloads/chroma_db"
CHUNK_SIZE= 1000
CHUNK_OVERLAP= 10

SERVICE_ACCOUNT_JSON_PATH= os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
GCP_PROJECT_NAME= "-".join(SERVICE_ACCOUNT_JSON_PATH.split("/")[-1].split("-")[:-1])
LOCATION= 'us-central1'

EMBEDDING_MODEL= "text-embedding-004"
GCP_LLM_MODEL= "gemini-2.5-flash-preview-05-20"
STREAM= False
QUERY= "Who is the data scientist?"
#######################################################################################




gcp_credentials= generate_gcp_credentials(
    scopes= ['https://www.googleapis.com/auth/cloud-platform'], 
    service_account_path= SERVICE_ACCOUNT_JSON_PATH,
    )

vertex_ai_embedding = generate_embedding(
    model= EMBEDDING_MODEL,
    project= GCP_PROJECT_NAME,
    location= LOCATION,
    credentials= gcp_credentials,
    )

documents= generate_documents(
    pdf_filepath= FULL_PDF_PATH,
    chunk_size= CHUNK_SIZE,
    chunk_overlap= CHUNK_OVERLAP,
    )

embedded_documents= store_embedding_chroma(
    collection_name= COLLECTION_NAME,
    embedding_function= vertex_ai_embedding,
    persist_directory= CHROMA_DIRECTORY,
    documents= documents
    )

relevant_contents= retrieve_relevant_contents(
    embedded_documents= embedded_documents, 
    query= QUERY)


gemini_llm(
    project= GCP_PROJECT_NAME, 
    location= LOCATION, 
    model= GCP_LLM_MODEL, 
    credentials= gcp_credentials, 
    question= QUERY, 
    contents= relevant_contents,
    stream= STREAM,
)