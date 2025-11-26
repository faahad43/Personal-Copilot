from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import os
import pandas as pd

df = pd.read_csv("../data/programmingConcepts.csv")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents= []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content= row["title"] + " " + row["content"],
            metadata= {"category": row["category"], "example": row["example"], "keywords": row["keywords"], "description": row["description"]},
            id = str(i)
        )

        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name= "Programming_Concepts",
    persist_directory= db_location,
    embedding_function= embeddings
)

if add_documents:
    vector_store.add_documents(documents= documents, ids= ids)

retriever = vector_store.as_retriever(
    search_kwargs = {"k": 5}
)