import os
import uuid
import logging
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from pydantic import BaseModel
import aiofiles
from datetime import datetime
from dotenv import load_dotenv
import shutil

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
UPLOAD_DIR = "uploaded_files"
CHROMA_DIR = "chroma_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    timestamp: str

class FileInfo(BaseModel):
    id: str
    filename: str
    upload_time: str
    size: int

# Helpers
def load_file(file_path: str, file_extension: str):
    try:
        if file_extension.lower() == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            # Try different encodings for text files
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    loader = TextLoader(file_path, encoding=encoding)
                    docs = loader.load()
                    return docs
                except UnicodeDecodeError:
                    continue
            raise ValueError("Could not decode text file with any encoding")
        return loader.load()
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load file: {str(e)}")

def ask_gemini(prompt: str):
    if not GOOGLE_API_KEY:
        return "API not configured. Please set GOOGLE_API_KEY environment variable."
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

async def process_file(file_path: str, filename: str):
    try:
        file_extension = os.path.splitext(filename)[1].lower()
        docs = load_file(file_path, file_extension)
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        for chunk in chunks:
            chunk.metadata = {"filename": filename, "file_id": os.path.basename(file_path)}
        
        if chunks:
            vectorstore.add_documents(chunks)
            return True
        return False
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return False

# Routes
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
        
        # Save the file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Process the file
        success = await process_file(file_path, file.filename)
        
        if success:
            return {"status": "success", "file_id": file_id, "filename": file.filename}
        else:
            os.remove(file_path)
            raise HTTPException(status_code=500, detail="File processing failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Check if we have any documents
        try:
            results = vectorstore.similarity_search("test", k=1)
            has_documents = len(results) > 0
        except:
            has_documents = False
        
        if not has_documents:
            return ChatResponse(
                answer="I'm still waiting for information. Please upload some documents first so I can help you!",
                sources=[],
                timestamp=datetime.now().isoformat()
            )
        
        # Search for relevant content
        results = vectorstore.similarity_search(request.query, k=5)
        context = "\n\n".join([doc.page_content for doc in results])
        sources = list(set([doc.metadata.get("filename", "Unknown") for doc in results]))
        
        prompt = f"""
        Based on the following context, provide a helpful and natural response to: {request.query}
        
        Context:
        {context}
        
        If the information isn't in the context, politely say you don't have that information yet.
        Respond in a friendly, conversational tone as if you're having a natural conversation.
        """
        
        answer = ask_gemini(prompt)
        return ChatResponse(
            answer=answer,
            sources=sources,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        return ChatResponse(
            answer=f"I encountered an error while processing your request: {str(e)}",
            sources=[],
            timestamp=datetime.now().isoformat()
        )

@app.get("/files")
async def list_files():
    files = []
    try:
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                # Extract file_id and original filename
                parts = filename.split('_', 1)
                file_id = parts[0]
                original_filename = parts[1] if len(parts) > 1 else "Unknown"
                
                files.append({
                    "id": file_id,
                    "filename": original_filename,
                    "upload_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "size": stat.st_size
                })
    except Exception as e:
        logger.error(f"Error listing files: {e}")
    
    return files

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    try:
        # Find and delete the file
        for filename in os.listdir(UPLOAD_DIR):
            if filename.startswith(file_id + "_"):
                file_path = os.path.join(UPLOAD_DIR, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
                    # Recreate vectorstore to remove the file's content
                    global vectorstore
                    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
                    
                    return {"status": "success", "message": f"File {file_id} deleted"}
        
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/files")
async def clear_all_files():
    try:
        # Clear uploaded files
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Clear vectorstore by recreating it
        global vectorstore
        if os.path.exists(CHROMA_DIR):
            import shutil
            shutil.rmtree(CHROMA_DIR)
        os.makedirs(CHROMA_DIR)
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        
        return {"status": "success", "message": "All files cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running", "endpoints": ["/upload", "/chat", "/files", "/health"]}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", 
                port=8006, 
                # reload=True
                )