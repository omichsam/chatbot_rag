# RAG Chatbot Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG Chatbot System                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│   Documents      │
│  (.pdf, .txt)    │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│  Document Processor (document_processor.py)                  │
│  - Loads documents (PDF, TXT)                                │
│  - Splits into chunks (1000 chars default)                   │
│  - Maintains overlap (200 chars default)                     │
└────────┬─────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│  Embeddings (OpenAI text-embedding-ada-002)                  │
│  - Converts text chunks to vectors                           │
│  - 1536 dimensions                                           │
└────────┬─────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│  Vector Store (vectorstore.py)                               │
│  - ChromaDB for efficient storage                            │
│  - Similarity search                                         │
│  - Persistent storage on disk                                │
└────────┬─────────────────────────────────────────────────────┘
         │
         │
         │  User Question
         │  ─────────────────────────┐
         │                           │
         ▼                           ▼
┌──────────────────────┐    ┌──────────────────────┐
│   Retriever          │    │    User Input        │
│  - Similarity search │    │                      │
│  - Fetches top K     │    │                      │
│    documents         │    │                      │
└──────────┬───────────┘    └──────────┬───────────┘
         │                           │
         │                           │
         └───────────┬───────────────┘
                     │
                     ▼
         ┌─────────────────────────────┐
         │  RAG Chain (chatbot.py)     │
         │  - Combines context + query │
         │  - Conversation memory      │
         │  - Prompt template          │
         └──────────┬──────────────────┘
                     │
                     ▼
         ┌─────────────────────────────┐
         │  LLM (ChatGPT)              │
         │  - GPT-3.5-turbo / GPT-4    │
         │  - Generates answer         │
         │  - Uses retrieved context   │
         └──────────┬──────────────────┘
                     │
                     ▼
         ┌─────────────────────────────┐
         │    Response                 │
         │  - Answer text              │
         │  - Source documents         │
         └─────────────────────────────┘
```

## Data Flow

1. **Indexing Phase** (One-time or periodic):
   ```
   Documents → Processor → Chunks → Embeddings → Vector Store
   ```

2. **Query Phase** (Every user question):
   ```
   User Question → Retriever → Relevant Chunks
                                     ↓
   User Question + Context → LLM → Answer
   ```

## Components Detail

### 1. Document Processor (`document_processor.py`)
- **Purpose**: Load and prepare documents
- **Methods**:
  - `load_documents()`: Load from file/directory
  - `process_documents()`: Split into chunks
  - `load_and_process()`: Combined operation

### 2. Vector Store Manager (`vectorstore.py`)
- **Purpose**: Manage embeddings and search
- **Methods**:
  - `create_vectorstore()`: Initialize new store
  - `load_vectorstore()`: Load existing store
  - `add_documents()`: Add more documents
  - `similarity_search()`: Find relevant chunks
  - `get_retriever()`: Get retriever for chain

### 3. RAG Chatbot (`chatbot.py`)
- **Purpose**: Orchestrate RAG conversation
- **Methods**:
  - `ask()`: Get answer with sources
  - `chat()`: Simple chat interface
  - `get_chat_history()`: View conversation
  - `clear_history()`: Reset conversation

### 4. Configuration (`config.py`)
- **Purpose**: Manage settings
- **Methods**:
  - `load_config()`: Load YAML config
  - `setup_environment()`: Set API keys

## Key Technologies

- **LangChain**: Framework for LLM applications
- **OpenAI API**: Embeddings and chat completion
- **ChromaDB**: Vector database
- **PyPDF**: PDF document parsing
- **Python**: Core implementation language

## Workflow Example

### Setup (One-time)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
cp config.template.yaml config.yaml
# Edit config.yaml with your API key

# 3. Index documents
python index_documents.py ./data
```

### Usage (Interactive)
```bash
# Run chatbot
python main.py

# User: "What is machine learning?"
# Bot: [Searches vector store → Retrieves relevant chunks → 
#       Sends to LLM with context → Returns answer]
```

### Programmatic Usage
```python
from chatbot import RAGChatbot
from vectorstore import VectorStoreManager

# Load vector store
vs = VectorStoreManager()
vs.load_vectorstore()

# Create chatbot
chatbot = RAGChatbot(vs.get_retriever())

# Ask questions
result = chatbot.ask("Your question?")
print(result['answer'])
```

## Configuration Options

### Vector Store Settings
- `persist_directory`: Where to save the database
- `collection_name`: Name for document collection
- `embedding_model`: OpenAI embedding model

### Retriever Settings
- `search_type`: "similarity" or "mmr" (max marginal relevance)
- `k`: Number of chunks to retrieve (default: 3)

### Chatbot Settings
- `model`: LLM model (gpt-3.5-turbo, gpt-4, etc.)
- `temperature`: Randomness (0.0-1.0)
- `max_tokens`: Max response length

## Performance Considerations

- **Chunk Size**: Larger chunks = more context but less precision
- **Chunk Overlap**: Prevents splitting related information
- **Top K**: More documents = better context but slower/expensive
- **Temperature**: Lower = more focused, Higher = more creative

## Extending the System

### Add New Document Types
Implement loader in `document_processor.py`:
```python
elif file_path.endswith('.docx'):
    loader = DocxLoader(file_path)
    documents = loader.load()
```

### Custom Prompts
Modify prompt template in `chatbot.py`:
```python
template = """Your custom prompt template here..."""
```

### Different Vector Stores
Replace ChromaDB in `vectorstore.py`:
```python
from langchain_community.vectorstores import Pinecone
# Implement Pinecone-specific methods
```

### Alternative LLMs
Change model in `chatbot.py`:
```python
from langchain_community.llms import Anthropic
self.llm = Anthropic(model="claude-2")
```
