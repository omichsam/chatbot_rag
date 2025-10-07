#!/usr/bin/env python3
"""
RAG Chatbot Example Usage

This script demonstrates how to use the RAG chatbot system to:
1. Load and process documents
2. Create a vector store
3. Run an interactive chatbot session
"""

import os
import sys
from document_processor import DocumentProcessor
from vectorstore import VectorStoreManager
from chatbot import RAGChatbot
from config import load_config, setup_environment


def main():
    """Main function to run the RAG chatbot."""
    
    print("=" * 60)
    print("RAG Chatbot System")
    print("=" * 60)
    print()
    
    # Load configuration
    try:
        config = load_config()
        setup_environment(config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease create a config.yaml file from config.template.yaml")
        print("and add your OpenAI API key.")
        sys.exit(1)
    
    # Get configuration values
    openai_config = config.get('openai', {})
    vectorstore_config = config.get('vectorstore', {})
    retriever_config = config.get('retriever', {})
    chatbot_config = config.get('chatbot', {})
    
    # Check if vector store exists
    persist_dir = vectorstore_config.get('persist_directory', './chroma_db')
    
    if not os.path.exists(persist_dir):
        print("No existing vector store found.")
        print("\nTo use the chatbot, you need to index some documents first.")
        print("\nOptions:")
        print("1. Add documents to a 'data' directory and run this script")
        print("2. Or use the following code to index your documents:\n")
        print("from document_processor import DocumentProcessor")
        print("from vectorstore import VectorStoreManager")
        print()
        print("processor = DocumentProcessor()")
        print("docs = processor.load_and_process('path/to/your/documents')")
        print()
        print("vectorstore = VectorStoreManager()")
        print("vectorstore.create_vectorstore(docs)")
        print()
        
        # Try to load from data directory
        if os.path.exists('data') and any(os.scandir('data')):
            print("Found 'data' directory. Processing documents...")
            processor = DocumentProcessor()
            documents = processor.load_and_process('data')
            print(f"Processed {len(documents)} document chunks")
            
            print("Creating vector store...")
            vectorstore = VectorStoreManager(
                embedding_model=openai_config.get('embedding_model', 'text-embedding-ada-002'),
                persist_directory=persist_dir,
                collection_name=vectorstore_config.get('collection_name', 'documents')
            )
            vectorstore.create_vectorstore(documents)
            print("Vector store created successfully!")
        else:
            print("No 'data' directory found. Exiting...")
            sys.exit(1)
    else:
        print("Loading existing vector store...")
        vectorstore = VectorStoreManager(
            embedding_model=openai_config.get('embedding_model', 'text-embedding-ada-002'),
            persist_directory=persist_dir,
            collection_name=vectorstore_config.get('collection_name', 'documents')
        )
        vectorstore.load_vectorstore()
        print("Vector store loaded successfully!")
    
    # Create retriever
    retriever = vectorstore.get_retriever(
        search_type=retriever_config.get('search_type', 'similarity'),
        k=retriever_config.get('k', 3)
    )
    
    # Initialize chatbot
    chatbot = RAGChatbot(
        retriever=retriever,
        model=openai_config.get('model', 'gpt-3.5-turbo'),
        temperature=chatbot_config.get('temperature', 0.7),
        max_tokens=chatbot_config.get('max_tokens', 500)
    )
    
    print("\n" + "=" * 60)
    print("Chatbot ready! Type your questions below.")
    print("Commands: 'quit' or 'exit' to stop, 'clear' to clear history")
    print("=" * 60)
    print()
    
    # Interactive chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            
            if user_input.lower() == 'clear':
                chatbot.clear_history()
                print("Conversation history cleared.\n")
                continue
            
            # Get response
            result = chatbot.ask(user_input)
            answer = result['answer']
            sources = result.get('source_documents', [])
            
            print(f"\nBot: {answer}\n")
            
            # Optionally show sources
            if sources:
                print(f"(Based on {len(sources)} source document(s))")
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            continue


if __name__ == "__main__":
    main()
