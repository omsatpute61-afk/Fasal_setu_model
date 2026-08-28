import os
import argparse
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def ingest_pdfs():
    """
    Scans the data/documents directory for agricultural PDFs, 
    splits the text, and stores offline embeddings into ChromaDB.
    """
    base_dir = Path(__file__).parent.parent.parent
    docs_dir = base_dir / "data" / "documents"
    chroma_dir = base_dir / "data" / "chroma_db"
    
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {docs_dir}. Please place extension manuals there.")
        # Create a dummy PDF text text file to ensure the script doesn't fail 
        # when running without PDFs. Actually, we'll just return gracefully.
        return

    print(f"Found {len(pdf_files)} PDFs. Processing...")
    
    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
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
