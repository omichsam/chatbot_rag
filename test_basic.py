"""
Basic tests for the RAG chatbot components.

Note: These tests require a valid OpenAI API key in config.yaml
"""

import os
import sys
from document_processor import DocumentProcessor
from vectorstore import VectorStoreManager


def test_document_processing():
    """Test document loading and processing."""
    print("Testing document processing...")
    
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    
    # Test with sample document
    if os.path.exists('sample_document.txt'):
        documents = processor.load_and_process('sample_document.txt')
        assert len(documents) > 0, "No documents processed"
        print(f"✓ Successfully processed {len(documents)} chunks")
        print(f"  First chunk preview: {documents[0].page_content[:100]}...")
        return True
    else:
        print("✗ Sample document not found")
        return False


def test_vectorstore():
    """Test vector store operations (requires API key)."""
    print("\nTesting vector store...")
    
    try:
        from config import load_config, setup_environment
        config = load_config()
        setup_environment(config)
        
        # Process sample document
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        documents = processor.load_and_process('sample_document.txt')
        
        # Create temporary vector store
        vectorstore = VectorStoreManager(
            persist_directory="./test_chroma_db",
            collection_name="test_collection"
        )
        
        print("  Creating vector store...")
        vectorstore.create_vectorstore(documents)
        print("✓ Vector store created successfully")
        
        # Test similarity search
        print("  Testing similarity search...")
        results = vectorstore.similarity_search("What is RAG?", k=2)
        assert len(results) > 0, "No search results"
        print(f"✓ Found {len(results)} relevant documents")
        print(f"  Top result preview: {results[0].page_content[:100]}...")
        
        # Cleanup
        import shutil
        if os.path.exists("./test_chroma_db"):
            shutil.rmtree("./test_chroma_db")
            print("✓ Cleaned up test files")
        
        return True
    except FileNotFoundError:
        print("✗ Config file not found - skipping vector store test")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running RAG Chatbot Tests")
    print("=" * 60)
    print()
    
    results = []
    
    # Test 1: Document Processing
    results.append(test_document_processing())
    
    # Test 2: Vector Store (optional, requires API key)
    results.append(test_vectorstore())
    
    print()
    print("=" * 60)
    print(f"Tests completed: {sum(results)}/{len(results)} passed")
    print("=" * 60)
    
    return all(results)


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
