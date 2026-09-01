# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportMissingTypeArgument=false, reportIndexIssue=false
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class TreatmentAdvisor:
    """
    RAG-powered Knowledge Retrieval Pipeline.
    Queries the local ChromaDB for context-aware agricultural treatment plans.
    """
    def __init__(self):
        base_dir = Path(__file__).parent.parent.parent
        chroma_dir = base_dir / "data" / "chroma_db"
        
        # Load the same lightweight embedding model used for ingestion
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize connection to local ChromaDB
        self.vectorstore = Chroma(
            persist_directory=str(chroma_dir),
            embedding_function=self.embeddings
        )

    def get_treatment(self, disease_name: str, pest_name: str, health_category: str) -> str:
        """
        Constructs a dynamic query and retrieves the most relevant K-chunks.
        """
        # If no disease or pest is detected, return a baseline response
        if disease_name in ["Healthy", "None"] and pest_name in ["None", ""]:
            return "Crop is optimally healthy. Maintain standard watering and fertilization."

        # Construct semantic search query
        query = f"Recommended chemical and organic treatment for {disease_name} and {pest_name} when plant health is {health_category}."
        
        try:
            # Perform similarity search
            docs = self.vectorstore.similarity_search(query, k=2)
            
            if not docs:
                return "No RAG context available in local database. Rely on standard taxonomy registry."
                
            # Formatting the retrieved context
            rag_response = "--- Local Extension Manual Insights ---\n"
            for i, doc in enumerate(docs, 1):
                clean_text = doc.page_content.replace('\n', ' ').strip()
                rag_response += f"* Insight {i}: {clean_text}\n"
                
            return rag_response
            
        except Exception as e:
            # Fallback gracefully if ChromaDB fails or is empty
            return f"RAG Engine Offline: {str(e)}"
