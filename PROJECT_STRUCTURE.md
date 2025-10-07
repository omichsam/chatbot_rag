# Project Structure

```
chatbot_rag/
│
├── Core Components
│   ├── chatbot.py              # RAG chatbot implementation
│   ├── document_processor.py   # Document loading and chunking
│   ├── vectorstore.py          # Vector database management
│   └── config.py               # Configuration utilities
│
├── User Scripts
│   ├── main.py                 # Interactive chatbot CLI
│   ├── index_documents.py      # Document indexing utility
│   └── examples.py             # Usage examples
│
├── Configuration
│   ├── config.template.yaml    # Configuration template
│   ├── requirements.txt        # Python dependencies
│   └── .gitignore             # Git ignore rules
│
├── Documentation
│   ├── README.md              # Main documentation
│   ├── ARCHITECTURE.md        # System architecture details
│   ├── CONTRIBUTING.md        # Contribution guidelines
│   └── PROJECT_STRUCTURE.md   # This file
│
├── Utilities
│   ├── setup.sh               # Setup script
│   └── test_basic.py          # Basic functionality tests
│
├── Sample Data
│   └── sample_document.txt    # Example document
│
└── License
    └── LICENSE                # MIT License

Runtime Directories (created at runtime, ignored by git):
│
├── venv/                      # Virtual environment
├── data/                      # User documents
├── chroma_db/                 # Vector database
├── __pycache__/              # Python cache
└── config.yaml               # User configuration
```

## File Descriptions

### Core Components

**chatbot.py** (3.1K)
- `RAGChatbot` class for conversational AI
- Implements RAG (Retrieval-Augmented Generation)
- Manages conversation memory
- Provides ask() and chat() interfaces

**document_processor.py** (2.7K)
- `DocumentProcessor` class for document handling
- Loads PDF and TXT files
- Splits documents into chunks
- Supports both files and directories

**vectorstore.py** (3.5K)
- `VectorStoreManager` class for vector database
- ChromaDB integration
- Embedding generation via OpenAI
- Similarity search and retrieval

**config.py** (892 bytes)
- Configuration loading from YAML
- Environment setup for API keys
- Configuration validation

### User Scripts

**main.py** (5.0K)
- Main entry point for the chatbot
- Interactive CLI interface
- Automatic document indexing from 'data/' directory
- Conversation loop with history

**index_documents.py** (2.9K)
- Command-line tool for indexing documents
- Supports incremental updates
- Progress feedback
- Usage: `python index_documents.py <path>`

**examples.py** (4.8K)
- Code examples for different use cases
- Basic usage patterns
- Advanced configurations
- Best practices

### Configuration

**config.template.yaml** (418 bytes)
- Template for user configuration
- OpenAI API settings
- Vector store settings
- Retriever and chatbot parameters

**requirements.txt** (165 bytes)
- Python package dependencies
- LangChain and related packages
- OpenAI API client
- ChromaDB vector store

**.gitignore** (367 bytes)
- Excludes Python cache files
- Excludes virtual environments
- Excludes user data and configurations
- Excludes vector database

### Documentation

**README.md** (4.7K)
- Project overview and features
- Installation instructions
- Usage examples
- Configuration guide
- API documentation

**ARCHITECTURE.md** (6.7K)
- System architecture diagrams
- Data flow explanation
- Component descriptions
- Technology stack details
- Extension guidelines

**CONTRIBUTING.md** (1.7K)
- Contribution guidelines
- Code style requirements
- Pull request process
- Development setup

### Utilities

**setup.sh** (1.7K)
- Automated setup script
- Creates virtual environment
- Installs dependencies
- Initializes configuration

**test_basic.py** (3.0K)
- Basic functionality tests
- Document processing validation
- Vector store testing
- Does not require API key for all tests

### Sample Data

**sample_document.txt** (1.3K)
- Example document about RAG
- Demonstrates document format
- Used for testing and demos

### License

**LICENSE** (1.1K)
- MIT License
- Open source
- Permissive usage rights

## Total Project Stats

- **Total Files**: 17 source files
- **Total Lines**: ~1,313 lines
- **Code Files**: 7 Python modules
- **Documentation**: 4 markdown files
- **Configuration**: 3 config files
- **Scripts**: 3 executable scripts

## Dependencies

### Python Packages
- langchain (0.1.0) - LLM application framework
- langchain-community (0.0.10) - Community integrations
- langchain-openai (0.0.2) - OpenAI integration
- chromadb (0.4.22) - Vector database
- openai (1.7.2) - OpenAI API client
- python-dotenv (1.0.0) - Environment variables
- tiktoken (0.5.2) - Token counting
- pypdf (3.17.4) - PDF parsing
- pyyaml (6.0.1) - YAML configuration

### System Requirements
- Python 3.8+
- OpenAI API key
- ~500MB disk space for dependencies
- Internet connection for API calls

## Getting Started Quick Reference

```bash
# 1. Setup
./setup.sh

# 2. Configure
# Edit config.yaml with your OpenAI API key

# 3. Add documents
mkdir data
# Copy your PDF/TXT files to data/

# 4. Run
python main.py
```

## Module Import Structure

```python
# Main application
from chatbot import RAGChatbot
from vectorstore import VectorStoreManager
from document_processor import DocumentProcessor
from config import load_config, setup_environment

# LangChain dependencies
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
```

## Typical Workflow

1. **Setup Phase**: Run setup.sh, configure API key
2. **Indexing Phase**: Add documents, run index_documents.py
3. **Query Phase**: Run main.py, ask questions
4. **Maintenance**: Add more documents, update index

## Extension Points

- **New document types**: Extend document_processor.py
- **Different vector stores**: Modify vectorstore.py
- **Custom prompts**: Update chatbot.py
- **Alternative LLMs**: Change model in config.yaml
- **API integration**: Build on top of existing modules
