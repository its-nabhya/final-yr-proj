# rag/ingestion.py

import os

# --- FIX: Disable parallelism to prevent cleanup errors ---
# This MUST be at the top, before other imports
#Remove this for large dataset/high CPU Core machines
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

docs_path = "rag/data"
# persist_directory = "rag/faiss_index"
# os.makedirs(docs_path, exist_ok=True)
# os.makedirs(persist_directory, exist_ok=True)

def load_documents(docs_path = docs_path):
    """"Load all text files from the docs directory."""
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The direcctory {docs_path} does not exist. Please create it and add your company files")
    
    Loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,  # langchain text loader class for text files, we have respective classes for pdf images etc
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = Loader.load()

    # documents = [
    #  Document(page_content="",metadata={'source' : "docs/filename.txt'},...."))]

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}, please add the company documents.")

    for i,doc in enumerate(documents[:2]):
        print(f"\nDocumentt {i+1}:" )
        print(f"Source : {doc.metadata['source']}")
        print(f"Content length : {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}...")
        print(f"metadata: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size = 800, chunk_overlap=50):  # use 1000 for GPT mini embedding. 250 for HF
    """"Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")

    ### Using basic  chunking (charactertextsplitter)
    # text_splitter = CharacterTextSplitter(
    #     separator = "\n\n",
    #     chunk_size = chunck_size, 
    #     chunk_overlap=chunk_overlap
    # )
    # chunks = text_splitter.split_documents(documents)

    ### Using recursiveCTS

    recursive_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n",". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = recursive_splitter.split_documents(documents)

    if chunks: 
        for i,chunk in enumerate(chunks[:5]):
            print(f"\n---Chunk {i+1}---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content: ")
            print(chunk.page_content)
            print("-"*50)

        if len(chunks)>5:
            print(f"\n... and {len(chunks)-5} more chunks")
    
    return chunks

def create_vector_store(chunks, persist_directory="assets/db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
    # embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    #Create ChromaDB vector store
    print("--- Creating Vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )
    print("--- Finished creating vector store---")
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    print("Main function")

    #1. Source : Loading Documents/files
    documents = load_documents(docs_path="rag/data") 

    #2. Chunking the files
    chunks = split_documents(documents)

    #3. Embedding and storing in the Vector DB
    vectorstore = create_vector_store(chunks)



if __name__ == "__main__":
    main()  
