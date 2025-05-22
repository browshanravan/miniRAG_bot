from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
import os

HOME_PATH= os.environ["HOME"]
PDF_FILE_NAME= "my_cv.pdf"
FULL_PDF_PATH= f"{HOME_PATH}/Downloads/{PDF_FILE_NAME}"


loader = PyPDFLoader(FULL_PDF_PATH)
document= loader.lazy_load()

document_pages= []
for page in document:
    document_pages.append(page)



vector_store = InMemoryVectorStore.from_documents(document_pages, OpenAIEmbeddings())
docs = vector_store.similarity_search("Who is data scientist?", k=2)

for doc in docs:
    print(f'Page {doc.metadata["page"]}: {doc.page_content[:300]}\n')