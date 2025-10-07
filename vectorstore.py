from typing import List, Optional
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document


class VectorStoreManager:
    """Manages the vector store for document embeddings."""
    
    def __init__(
        self, 
        embedding_model: str = "text-embedding-ada-002",
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents"
    ):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_model: Name of the OpenAI embedding model
            persist_directory: Directory to persist the vector store
            collection_name: Name of the collection
        """
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore = None
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of documents to add to the vector store
            
        Returns:
            Created vector store
        """
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        self.vectorstore.persist()
        return self.vectorstore
    
    def load_vectorstore(self) -> Chroma:
        """
        Load an existing vector store.
        
        Returns:
            Loaded vector store
        """
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        return self.vectorstore
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to an existing vector store.
        
        Args:
            documents: List of documents to add
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Create or load a vector store first.")
        
        self.vectorstore.add_documents(documents)
        self.vectorstore.persist()
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of similar documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Create or load a vector store first.")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def get_retriever(self, search_type: str = "similarity", k: int = 3):
        """
        Get a retriever for the vector store.
        
        Args:
            search_type: Type of search ("similarity" or "mmr")
            k: Number of documents to retrieve
            
        Returns:
            Retriever object
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Create or load a vector store first.")
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
