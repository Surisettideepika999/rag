from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from dotenv import load_dotenv
import os
load_dotenv()

def ingest(docs_path="docs"):
    print("Loading environment variables...")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"{docs_path} does not exist")
    
    docs = DirectoryLoader(
        docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
    ).load()
    if len(docs) == 0:
        raise ValueError(f"No documents found in {docs_path}")
    
    for i, doc in enumerate(docs):
        print(f"doc - {i+1}")
        print(f"source: {doc.metadata['source']}")
        print(f"metadata: {doc.metadata}")
    return docs

def ingest_chunk(docs, chunk_size=500, chunk_overlap=50):
    chunks = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap).split_documents(docs)
    for i, chunk in enumerate(chunks):
        print(f"chunk - {i+1}")
        print(f"source: {chunk.metadata['source']}")
        print(f"metadata: {chunk.page_content}...")
    return chunks

def create_vector_store(chunks, persistant_path="db/chroma", collection_name="my_collection"):
    print("Creating vector store...")
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        chunks, 
        embedding, 
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=persistant_path)
    print(f"Vector store created and saved to {persistant_path}.")
    return vector_store


docs=ingest()
chunks = ingest_chunk(docs, chunk_size=100, chunk_overlap=0)
vector_store= create_vector_store(chunks)