import os
import uuid
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from pydantic import BaseModel, Field
from datetime import datetime
from dotenv import load_dotenv
import secrets

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_DIR = "chroma_db"
API_KEYS = os.getenv("API_KEYS", "").split(",")  # Comma-separated API keys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Improved vectorstore initialization with error handling
def get_vectorstore():
    try:
        # Check if Chroma directory exists and has content
        if not os.path.exists(CHROMA_DIR):
            logger.warning(f"Chroma directory {CHROMA_DIR} does not exist")
            return None
            
        # Check if Chroma directory has the necessary files
        required_files = ['chroma.sqlite3', 'chroma-collections.parquet', 'chroma-embeddings.parquet']
        existing_files = os.listdir(CHROMA_DIR) if os.path.exists(CHROMA_DIR) else []
        
        has_required_files = any(file in existing_files for file in required_files)
        
        if not has_required_files:
            logger.warning(f"Chroma directory {CHROMA_DIR} exists but doesn't contain required database files")
            return None
            
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        logger.info("Vectorstore initialized successfully")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error initializing vectorstore: {e}")
        return None

vectorstore = get_vectorstore()

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Security
security = HTTPBearer(auto_error=False)

app = FastAPI(
    title="Document Q&A API",
    description="A RESTful API for asking questions about documents. Documents are pre-loaded and managed separately.",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    contact={
        "name": "API Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Models
class ChatRequest(BaseModel):
    question: str = Field(..., description="The question to ask about the documents")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation context")

class ChatResponse(BaseModel):
    success: bool = Field(..., description="Whether the request was successful")
    answer: str = Field(..., description="The AI-generated answer")
    session_id: str = Field(..., description="Session ID for continuing the conversation")
    sources: List[str] = Field(..., description="List of document sources used for the answer")
    timestamp: str = Field(..., description="ISO timestamp of the response")

class HealthResponse(BaseModel):
    status: str = Field(..., description="API status (healthy/unhealthy)")
    timestamp: str = Field(..., description="ISO timestamp of the check")
    documents_loaded: bool = Field(..., description="Whether documents are available")
    documents_count: int = Field(..., description="Number of documents loaded")
    version: str = Field(..., description="API version")

class APIKeyResponse(BaseModel):
    success: bool = Field(..., description="Whether the API key is valid")
    message: str = Field(..., description="Response message")

class ErrorResponse(BaseModel):
    success: bool = Field(..., description="Always false for errors")
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")

# Improved helper functions
def get_documents_count():
    """Get the actual number of documents in the vectorstore"""
    try:
        if vectorstore is None:
            return 0
            
        # Try multiple methods to get document count
        methods_to_try = [
            # Method 1: Try to get collection count
            lambda: vectorstore._collection.count(),
            # Method 2: Try similarity search with empty query
            lambda: len(vectorstore.similarity_search("", k=1000)),
            # Method 3: Try get method
            lambda: len(vectorstore.get()['documents']) if vectorstore.get() and 'documents' in vectorstore.get() else 0
        ]
        
        for method in methods_to_try:
            try:
                count = method()
                logger.info(f"Document count retrieved: {count}")
                return count
            except Exception as e:
                logger.debug(f"Document count method failed: {e}")
                continue
                
        return 0
        
    except Exception as e:
        logger.error(f"Error getting document count: {e}")
        return 0

def has_documents():
    """Check if documents are available"""
    count = get_documents_count()
    logger.info(f"Document availability check: {count} documents found")
    return count > 0

def check_chroma_health():
    """Comprehensive ChromaDB health check"""
    health_info = {
        "chroma_directory_exists": os.path.exists(CHROMA_DIR),
        "vectorstore_initialized": vectorstore is not None,
        "document_count": 0,
        "directory_contents": []
    }
    
    if health_info["chroma_directory_exists"]:
        health_info["directory_contents"] = os.listdir(CHROMA_DIR)
        health_info["document_count"] = get_documents_count()
    
    return health_info

async def get_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not API_KEYS or not API_KEYS[0]:  # No API keys configured
        return None
    
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide it in the Authorization header as Bearer token."
        )
    
    if credentials.credentials not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return credentials.credentials

# Routes
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """Serve the chatbot interface"""
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<html><body><h1>Chatbot Interface</h1><p>Static files not found</p></body></html>")

@app.get("/api", include_in_schema=False)
async def api_root():
    """API root endpoint with documentation links"""
    chroma_health = check_chroma_health()
    
    return {
        "message": "Document Q&A API",
        "version": "2.0.0",
        "endpoints": {
            "documentation": "/api/docs",
            "redoc": "/api/redoc",
            "chat": "/api/chat",
            "health": "/api/health",
            "validate_key": "/api/validate-key",
            "debug_chroma": "/api/debug/chroma"  # Added debug endpoint
        },
        "chroma_health": chroma_health,
        "authentication": "API key required for all endpoints (except docs and health)" if API_KEYS and API_KEYS[0] else "No authentication required"
    }

@app.post("/api/chat", 
          response_model=ChatResponse,
          responses={
              200: {"description": "Successful response", "model": ChatResponse},
              400: {"description": "Bad request", "model": ErrorResponse},
              401: {"description": "Unauthorized", "model": ErrorResponse},
              500: {"description": "Internal server error", "model": ErrorResponse}
          },
          summary="Ask a question about documents",
          description="Submit a question and receive an AI-generated answer based on the pre-loaded documents.")
async def chat(
    chat_request: ChatRequest,
    api_key: str = Depends(get_api_key)
):
    """Chat with the document-based AI
    
    - **question**: The question to ask about the documents
    - **session_id**: Optional session ID for maintaining conversation context
    - Returns: AI-generated answer with source documents
    """
    try:
        # Generate session ID if not provided
        session_id = chat_request.session_id or str(uuid.uuid4())
        
        # Check if we have documents with detailed logging
        documents_available = has_documents()
        logger.info(f"Chat request - Documents available: {documents_available}")
        
        if not documents_available:
            chroma_health = check_chroma_health()
            logger.warning(f"No documents available. Chroma health: {chroma_health}")
            
            return ChatResponse(
                success=True,
                answer="I don't have any documents to reference yet. The system is configured but no documents are currently loaded in the database. Please check if documents were properly uploaded to the chroma_db directory.",
                session_id=session_id,
                sources=[],
                timestamp=datetime.now().isoformat()
            )
        
        # Search for relevant content
        try:
            results = vectorstore.similarity_search(chat_request.question, k=5)
            context = "\n\n".join([doc.page_content for doc in results])
            sources = list(set([doc.metadata.get("filename", "Unknown") for doc in results]))
            logger.info(f"Found {len(results)} relevant documents for query: {chat_request.question}")
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return ChatResponse(
                success=True,
                answer="I encountered an error while searching the documents. Please try again later.",
                session_id=session_id,
                sources=[],
                timestamp=datetime.now().isoformat()
            )
        
        prompt = f"""
        You are a helpful document assistant. Based on the following context from documents, 
        provide a helpful and natural response to the user's question.
        
        Context from documents:
        {context}
        
        User's question: {chat_request.question}
        
        Guidelines:
        1. If the information isn't in the context, politely say you don't have that information
        2. Respond in a friendly, conversational tone
        3. Be helpful and informative
        4. If the question is unclear, ask for clarification
        5. Keep responses concise but thorough
        6. Always be polite and professional
        """
        
        answer = ask_gemini(prompt)
        
        return ChatResponse(
            success=True,
            answer=answer,
            session_id=session_id,
            sources=sources,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/api/health", 
         response_model=HealthResponse,
         summary="API health check",
         description="Check the API status and document availability.")
async def health_check():
    """Health check endpoint
    
    Returns: API status, document availability, and version information
    """
    try:
        documents_count = get_documents_count()
        chroma_health = check_chroma_health()
        
        logger.info(f"Health check - Documents count: {documents_count}, Chroma health: {chroma_health}")
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            documents_loaded=documents_count > 0,
            documents_count=documents_count,
            version="2.0.0"
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            documents_loaded=False,
            documents_count=0,
            version="2.0.0"
        )

@app.get("/api/debug/chroma", include_in_schema=False)
async def debug_chroma():
    """Debug endpoint to check ChromaDB status"""
    return check_chroma_health()

@app.get("/api/validate-key", 
         response_model=APIKeyResponse,
         summary="Validate API key",
         description="Check if an API key is valid.")
async def validate_api_key(api_key: str = Depends(get_api_key)):
    """Validate API key
    
    Returns: Validation result
    """
    return APIKeyResponse(
        success=True,
        message="API key is valid"
    )

@app.get("/api/generate-key", 
         response_model=dict,
         include_in_schema=False)  # Not in public docs for security
async def generate_api_key():
    """Generate a new API key (for admin use)"""
    new_key = secrets.token_urlsafe(32)
    return {
        "api_key": new_key,
        "message": "Add this key to your API_KEYS environment variable"
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error=exc.detail,
            detail=getattr(exc, 'detail', None)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    
    # Log startup information
    chroma_health = check_chroma_health()
    logger.info(f"Starting server with Chroma health: {chroma_health}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8006,
        # reload=True
    )