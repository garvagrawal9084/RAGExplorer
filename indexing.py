from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
import tempfile
import os
import qdrant_client

load_dotenv()


def indexing (file):
    # Load the file

     # Extract collection name from filename
    print("Getting collection name")
    collection_name = os.path.splitext(file.name)[0]

    # Connect to Qdrant
    print("Qdrant connect to qdrant")
    qdrant = qdrant_client.QdrantClient(url="http://localhost:6333")

    # Check if the collection already exists
    print("Check if the collection exist")
    existing_collections = [col.name for col in qdrant.get_collections().collections]
    if collection_name in existing_collections:
        return "✅ This document has already been uploaded and indexed. You can now chat with it."
    
    print("Create temp file")

    with tempfile.NamedTemporaryFile(delete=False , suffix=".pdf") as tmp_file :
        tmp_file.write(file.read())
        tmp_path = tmp_file.name

    print("Loading")
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # split the file
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000 , 
        chunk_overlap = 400 
    )

    split_docs = text_splitter.split_documents(documents=docs)

    # Vector Embedding 

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-large" 
    )

    # Using embedding_model create embedding of split_docs and store in db
    print("Vector store")
    vector_store = QdrantVectorStore.from_documents(
        embedding=embedding,
        documents = split_docs ,
        url = "http://localhost:6333" ,
        collection_name=os.path.splitext(file.name)[0],
        
    )

    os.remove(tmp_path)

    return f"Indexing of the document is done now you can chat"

