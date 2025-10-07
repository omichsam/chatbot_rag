# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot system that combines document retrieval with language model generation to provide accurate, context-aware responses based on your documents.

## Features

- 📄 **Document Processing**: Load and process PDF and text documents
- 🔍 **Vector Search**: Semantic search using embeddings and vector database
- 💬 **Conversational AI**: Interactive chatbot with conversation memory
- 🔗 **RAG Pipeline**: Retrieves relevant context before generating responses
- ⚙️ **Configurable**: Easy configuration through YAML files

## Architecture

The system consists of several key components:

1. **Document Processor** (`document_processor.py`): Loads and chunks documents
2. **Vector Store Manager** (`vectorstore.py`): Manages document embeddings and similarity search
3. **RAG Chatbot** (`chatbot.py`): Combines retrieval and generation for answering questions
4. **Main Application** (`main.py`): Interactive CLI interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/omichsam/chatbot_rag.git
cd chatbot_rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up configuration:
```bash
cp config.template.yaml config.yaml
```

4. Edit `config.yaml` and add your OpenAI API key:
```yaml
openai:
  api_key: "your-openai-api-key-here"
  model: "gpt-3.5-turbo"
  embedding_model: "text-embedding-ada-002"
```

## Usage

### Quick Start

1. Create a `data` directory and add your documents (PDF or TXT files):
```bash
mkdir data
# Add your documents to the data directory
```

2. Run the chatbot:
```bash
python main.py
```

The system will automatically:
- Process documents from the `data` directory
- Create vector embeddings
- Start an interactive chat session

### Using the Components Programmatically

#### Process Documents
```python
from document_processor import DocumentProcessor

processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
documents = processor.load_and_process('path/to/documents')
```

#### Create Vector Store
```python
from vectorstore import VectorStoreManager

vectorstore = VectorStoreManager(
    embedding_model="text-embedding-ada-002",
    persist_directory="./chroma_db"
)
vectorstore.create_vectorstore(documents)
```

#### Initialize Chatbot
```python
from chatbot import RAGChatbot

retriever = vectorstore.get_retriever(search_type="similarity", k=3)
chatbot = RAGChatbot(
    retriever=retriever,
    model="gpt-3.5-turbo",
    temperature=0.7
)

# Ask questions
result = chatbot.ask("Your question here")
print(result['answer'])
```

## Configuration

The `config.yaml` file allows you to customize:

- **OpenAI Settings**: API key, model selection, embedding model
- **Vector Store**: Storage location, collection name
- **Retriever**: Search type, number of documents to retrieve
- **Chatbot**: Temperature, max tokens

Example configuration:
```yaml
openai:
  api_key: "your-api-key"
  model: "gpt-3.5-turbo"
  embedding_model: "text-embedding-ada-002"

vectorstore:
  type: "chroma"
  persist_directory: "./chroma_db"
  collection_name: "documents"

retriever:
  search_type: "similarity"
  k: 3

chatbot:
  temperature: 0.7
  max_tokens: 500
```

## Project Structure

```
chatbot_rag/
├── main.py                    # Main application entry point
├── document_processor.py      # Document loading and processing
├── vectorstore.py            # Vector store management
├── chatbot.py                # RAG chatbot implementation
├── config.py                 # Configuration utilities
├── requirements.txt          # Python dependencies
├── config.template.yaml      # Configuration template
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## How RAG Works

1. **Document Ingestion**: Documents are loaded and split into chunks
2. **Embedding**: Text chunks are converted to vector embeddings
3. **Indexing**: Embeddings are stored in a vector database
4. **Retrieval**: When a question is asked, relevant chunks are retrieved
5. **Generation**: The LLM generates an answer using the retrieved context

## Requirements

- Python 3.8+
- OpenAI API key
- See `requirements.txt` for Python package dependencies

## Example Interaction

```
You: What is the main topic of the documents?

Bot: Based on the documents, the main topic is...

You: Can you provide more details?

Bot: Certainly! Here are more details...

Commands: 'quit' or 'exit' to stop, 'clear' to clear history
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.