import os
import argparse
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def ingest_pdfs():
    """
    Scans the data/documents directory for agricultural text files, 
    splits the text, and stores offline embeddings into ChromaDB.
    """
    base_dir = Path(__file__).parent.parent.parent
    docs_dir = base_dir / "data" / "documents"
    chroma_dir = base_dir / "data" / "chroma_db"
    
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    txt_files = list(docs_dir.glob("*.txt"))
    if not txt_files:
        print(f"No TXT files found in {docs_dir}. Please run scripts/setup_rag_data.py first.")
        return

    print(f"Found {len(txt_files)} TXT files. Processing...")
    
    documents = []
    for txt_file in txt_files:
        loader = TextLoader(str(txt_file), encoding="utf-8")
        documents.extend(loader.load())
        
    print(f"Loaded {len(documents)} document pages. Splitting...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks. Generating embeddings...")
    
    # Lightweight offline embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize and persist Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_dir)
    )
    
    # Note: In newer versions of ChromaDB / LangChain, persist() is automatic, 
    # but we can explicitly call it if available.
    if hasattr(vectorstore, 'persist'):
        vectorstore.persist()
        
    print(f"Successfully ingested {len(chunks)} chunks into local ChromaDB at {chroma_dir}")

if __name__ == "__main__":
    ingest_pdfs()
