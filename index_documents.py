#!/usr/bin/env python3
"""
Document Indexing Utility

This script helps you index documents into the vector store.
"""

import os
import sys
import argparse
from document_processor import DocumentProcessor
from vectorstore import VectorStoreManager
from config import load_config, setup_environment


def index_documents(document_path: str, config_path: str = "config.yaml"):
    """
    Index documents into the vector store.
    
    Args:
        document_path: Path to documents (file or directory)
        config_path: Path to configuration file
    """
    print("=" * 60)
    print("Document Indexing Utility")
    print("=" * 60)
    print()
    
    # Load configuration
    try:
        config = load_config(config_path)
        setup_environment(config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Get configuration values
    openai_config = config.get('openai', {})
    vectorstore_config = config.get('vectorstore', {})
    
    # Check if path exists
    if not os.path.exists(document_path):
        print(f"Error: Path not found: {document_path}")
        sys.exit(1)
    
    # Process documents
    print(f"Processing documents from: {document_path}")
    processor = DocumentProcessor()
    
    try:
        documents = processor.load_and_process(document_path)
        print(f"✓ Processed {len(documents)} document chunks")
    except Exception as e:
        print(f"Error processing documents: {e}")
        sys.exit(1)
    
    # Create or update vector store
    persist_dir = vectorstore_config.get('persist_directory', './chroma_db')
    vectorstore = VectorStoreManager(
        embedding_model=openai_config.get('embedding_model', 'text-embedding-ada-002'),
        persist_directory=persist_dir,
        collection_name=vectorstore_config.get('collection_name', 'documents')
    )
    
    if os.path.exists(persist_dir):
        print(f"Updating existing vector store...")
        vectorstore.load_vectorstore()
        vectorstore.add_documents(documents)
        print("✓ Documents added to existing vector store")
    else:
        print("Creating new vector store...")
        vectorstore.create_vectorstore(documents)
        print("✓ Vector store created successfully")
    
    print()
    print("=" * 60)
    print("Indexing complete! You can now run the chatbot with:")
    print("  python main.py")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Index documents into the RAG chatbot vector store"
    )
    parser.add_argument(
        "path",
        help="Path to document file or directory to index"
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    args = parser.parse_args()
    index_documents(args.path, args.config)


if __name__ == "__main__":
    main()
