#!/usr/bin/env python3
"""
Quick Start Guide for RAG Chatbot

This script provides examples of how to use the RAG chatbot system.
"""

def example_1_basic_usage():
    """Example 1: Basic usage with sample document."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    print("""
from document_processor import DocumentProcessor
from vectorstore import VectorStoreManager
from chatbot import RAGChatbot
from config import load_config, setup_environment

# 1. Load configuration
config = load_config()
setup_environment(config)

# 2. Process documents
processor = DocumentProcessor()
documents = processor.load_and_process('sample_document.txt')

# 3. Create vector store
vectorstore = VectorStoreManager()
vectorstore.create_vectorstore(documents)

# 4. Create chatbot
retriever = vectorstore.get_retriever(k=3)
chatbot = RAGChatbot(retriever)

# 5. Ask questions
result = chatbot.ask("What is RAG?")
print(result['answer'])
    """)


def example_2_batch_indexing():
    """Example 2: Index multiple documents from a directory."""
    print("\n" + "=" * 60)
    print("Example 2: Batch Document Indexing")
    print("=" * 60)
    print("""
from document_processor import DocumentProcessor
from vectorstore import VectorStoreManager

# Process all documents in a directory
processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
documents = processor.load_and_process('data/')

# Create vector store
vectorstore = VectorStoreManager(
    persist_directory="./my_knowledge_base",
    collection_name="my_docs"
)
vectorstore.create_vectorstore(documents)

print(f"Indexed {len(documents)} document chunks")
    """)


def example_3_updating_index():
    """Example 3: Add new documents to existing index."""
    print("\n" + "=" * 60)
    print("Example 3: Updating Existing Index")
    print("=" * 60)
    print("""
from document_processor import DocumentProcessor
from vectorstore import VectorStoreManager

# Load existing vector store
vectorstore = VectorStoreManager(persist_directory="./my_knowledge_base")
vectorstore.load_vectorstore()

# Process new documents
processor = DocumentProcessor()
new_documents = processor.load_and_process('new_document.pdf')

# Add to existing store
vectorstore.add_documents(new_documents)
print("Documents added successfully!")
    """)


def example_4_custom_configuration():
    """Example 4: Custom chatbot configuration."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Configuration")
    print("=" * 60)
    print("""
from vectorstore import VectorStoreManager
from chatbot import RAGChatbot

# Load vector store
vectorstore = VectorStoreManager()
vectorstore.load_vectorstore()

# Create retriever with custom settings
retriever = vectorstore.get_retriever(
    search_type="mmr",  # Maximum Marginal Relevance
    k=5  # Retrieve top 5 documents
)

# Create chatbot with custom settings
chatbot = RAGChatbot(
    retriever=retriever,
    model="gpt-4",  # Use GPT-4 for better responses
    temperature=0.3,  # Lower temperature for more focused answers
    max_tokens=1000  # Longer responses
)

# Use the chatbot
answer = chatbot.chat("Explain the concept in detail")
print(answer)
    """)


def example_5_conversation_history():
    """Example 5: Managing conversation history."""
    print("\n" + "=" * 60)
    print("Example 5: Conversation History")
    print("=" * 60)
    print("""
from chatbot import RAGChatbot

# Assuming chatbot is already initialized
# chatbot = RAGChatbot(retriever)

# Have a conversation
chatbot.chat("What is machine learning?")
chatbot.chat("Can you give me an example?")
chatbot.chat("How does it differ from traditional programming?")

# View conversation history
history = chatbot.get_chat_history()
for msg in history:
    print(f"{msg.type}: {msg.content}")

# Clear history to start fresh
chatbot.clear_history()
    """)


def example_6_cli_usage():
    """Example 6: Command-line usage."""
    print("\n" + "=" * 60)
    print("Example 6: Command-Line Usage")
    print("=" * 60)
    print("""
# Index documents using the CLI
$ python index_documents.py ./data

# Run the interactive chatbot
$ python main.py

# The chatbot will:
# - Load the vector store
# - Start an interactive session
# - Accept questions and provide answers
# - Remember conversation context
    """)


def main():
    """Display all examples."""
    print("\n" + "=" * 70)
    print(" " * 20 + "RAG CHATBOT - QUICK START EXAMPLES")
    print("=" * 70)
    
    example_1_basic_usage()
    example_2_batch_indexing()
    example_3_updating_index()
    example_4_custom_configuration()
    example_5_conversation_history()
    example_6_cli_usage()
    
    print("\n" + "=" * 70)
    print("For more information, see README.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
