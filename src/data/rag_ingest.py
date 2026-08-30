import os
import ssl

# --- 1. Fix Windows SSL Certificate Verification Issue ---
ssl._create_default_https_context = ssl._create_unverified_context  # pyright: ignore[reportPrivateUsage]
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class VectorDatabaseBuilder:
    def __init__(
        self,
        csv_filename: str = 'pestopia_treatments.csv',
        db_dirname: str = 'chroma_db'
    ) -> None:
        self.base_dir: str = os.path.abspath(os.path.dirname(__file__))
        self.csv_path: str = os.path.join(self.base_dir, csv_filename)
        self.persist_dir: str = os.path.join(self.base_dir, db_dirname)

    def build(self) -> None:
        if not os.path.exists(self.csv_path):
            print(f"❌ Error: Cannot find {self.csv_path}. Please place the CSV in the data folder.")
            return

        print("Loading Pestopia treatment data...")
        loader = CSVLoader(file_path=self.csv_path, encoding='utf-8')
        documents = loader.load()
        
        print(f"Loaded {len(documents)} treatment records. Generating embeddings...")
        
        # Initialize the lightweight embedding model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        print("Building ChromaDB Vector Store. This may take a moment...")
        
        # Create and persist the vector store
        Chroma.from_documents( # pyright: ignore[reportUnknownMemberType]
            documents=documents,
            embedding=embeddings,
            persist_directory=self.persist_dir
        )

        print(f"\n✅ Local RAG database built successfully at: {self.persist_dir}")
        print("The LangChain Advisory Agent is now armed with localized treatments!")


if __name__ == "__main__":
    builder = VectorDatabaseBuilder()
    builder.build()