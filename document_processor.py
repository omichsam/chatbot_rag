import os
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class DocumentProcessor:
    """Handles loading and processing of documents for the RAG system."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_documents(self, file_path: str) -> List[Document]:
        """
        Load documents from a file or directory.
        
        Args:
            file_path: Path to file or directory
            
        Returns:
            List of loaded documents
        """
        if os.path.isdir(file_path):
            # Load all text files from directory
            txt_loader = DirectoryLoader(
                file_path, 
                glob="**/*.txt", 
                loader_cls=TextLoader
            )
            pdf_loader = DirectoryLoader(
                file_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            documents = txt_loader.load() + pdf_loader.load()
        elif file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
            documents = loader.load()
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        return documents
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of processed document chunks
        """
        return self.text_splitter.split_documents(documents)
    
    def load_and_process(self, file_path: str) -> List[Document]:
        """
        Load and process documents in one step.
        
        Args:
            file_path: Path to file or directory
            
        Returns:
            List of processed document chunks
        """
        documents = self.load_documents(file_path)
        return self.process_documents(documents)
