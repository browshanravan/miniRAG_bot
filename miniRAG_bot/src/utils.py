from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_chroma import Chroma
from google import genai
from google.genai import types
from google.oauth2 import service_account


def generate_gcp_credentials(scopes, service_account_path):
    gcp_credentials= service_account.Credentials.from_service_account_file(
    filename= service_account_path,
    scopes= scopes
    )

    return gcp_credentials


def generate_embedding(model, project, location, credentials):
    vertex_ai_embedding= VertexAIEmbeddings(
        model=model,
        project= project,
        location= location,
        credentials= credentials,
        )

    return vertex_ai_embedding


def generate_documents(pdf_filepath, chunk_size, chunk_overlap):
    pdf_loader = PyPDFLoader(pdf_filepath)
    pdf_data= pdf_loader.load()
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size= chunk_size,
        chunk_overlap= chunk_overlap)
    documents= text_splitter.split_documents(pdf_data)
    
    return documents


def store_embedding_chroma(collection_name, embedding_function, persist_directory, documents):
    vector_db = Chroma(
        collection_name= collection_name,
        embedding_function= embedding_function,
        persist_directory= persist_directory,
        )
    
    embedded_documents = vector_db.from_documents(documents= documents, embedding= embedding_function)

    return embedded_documents


def retrieve_relevant_contents(embedded_documents, query):
    all_documents= embedded_documents.as_retriever()
    relevant_documents= all_documents.invoke(query)
    relevant_contents= [relevant_documents[x].page_content for x in range(len(relevant_documents))]

    return relevant_contents


def gemini_llm(project, location, model, credentials, question, contents, stream):
    client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
        credentials= credentials,
    )

    contents = [
        #Can have multiple types.Content with one being a pre-prompt and one being the retrieved contents shown below and one being the question
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=" ".join(contents))] #since contents retrieved is a list of str they need to become one str
            ),
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=question)]
            )
            ]

    generate_content_config = types.GenerateContentConfig(
        temperature = 1,
        top_p = 1,
        seed = 0,
        max_output_tokens = 65535,
        safety_settings = [types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
        ),types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
        ),types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
        ),types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
        )],
        )

    if stream:
        for response in client.models.generate_content_stream(
            model = model,
            contents = contents,
            config = generate_content_config,
            ):
            print(response.text)
    
    else:
        response= client.models.generate_content(
            model = model,
            contents = contents,
            config = generate_content_config,
            )
        print(response.text)