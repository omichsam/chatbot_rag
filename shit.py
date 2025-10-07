import streamlit as st
import os
import uuid
import tempfile
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
import shutil
import json
import hashlib
import secrets
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg for static plots

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
UPLOAD_DIR = "uploaded_files"
CHROMA_DIR = "chroma_db"
USERS_FILE = "users.json"
API_KEYS_FILE = "api_keys.json"
ANALYTICS_FILE = "analytics.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize vectorstore
@st.cache_resource
def get_vectorstore():
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

vectorstore = get_vectorstore()

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Page configuration
st.set_page_config(
    page_title="Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    .chat-message.user {
        background-color: #0f4c75;
        color: white;
        margin-left: 20%;
    }
    .chat-message.assistant {
        background-color: #f0f2f6;
        margin-right: 20%;
        color: black;
    }
    .file-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .auth-container {
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        background-color: white;
    }
    .api-key-display {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        font-family: monospace;
        word-break: break-all;
    }
    .admin-only {
        border-left: 4px solid #ff4b4b;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    .analytics-card {
        background-color: white;
        color:dark;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Authentication functions
def hash_password(password: str) -> str:
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users() -> Dict:
    """Load users from JSON file"""
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return {}

def save_users(users: Dict):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def load_api_keys() -> Dict:
    """Load API keys from JSON file"""
    try:
        if os.path.exists(API_KEYS_FILE):
            with open(API_KEYS_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return {}

def save_api_keys(api_keys: Dict):
    """Save API keys to JSON file"""
    with open(API_KEYS_FILE, 'w') as f:
        json.dump(api_keys, f, indent=2)

def load_analytics() -> Dict:
    """Load analytics data from JSON file"""
    try:
        if os.path.exists(ANALYTICS_FILE):
            with open(ANALYTICS_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return {"queries": [], "users": {}, "popular_questions": {}}

def save_analytics(analytics: Dict):
    """Save analytics data to JSON file"""
    with open(ANALYTICS_FILE, 'w') as f:
        json.dump(analytics, f, indent=2)

def track_query(username: str, question: str, answer: str, sources: List[str]):
    """Track user queries for analytics"""
    analytics = load_analytics()
    
    # Track the query
    query_data = {
        "username": username,
        "question": question,
        "answer": answer[:200] + "..." if len(answer) > 200 else answer,  # Store excerpt
        "sources": sources,
        "timestamp": datetime.now().isoformat()
    }
    analytics["queries"].append(query_data)
    
    # Track user activity
    if username not in analytics["users"]:
        analytics["users"][username] = {
            "query_count": 0,
            "last_active": datetime.now().isoformat()
        }
    
    analytics["users"][username]["query_count"] += 1
    analytics["users"][username]["last_active"] = datetime.now().isoformat()
    
    # Track popular questions
    clean_question = question.strip().lower()
    if clean_question in analytics["popular_questions"]:
        analytics["popular_questions"][clean_question] += 1
    else:
        analytics["popular_questions"][clean_question] = 1
    
    save_analytics(analytics)

def generate_api_key() -> str:
    """Generate a new API key"""
    return f"sk_{secrets.token_urlsafe(32)}"

def register_user(username: str, password: str, is_admin: bool = False) -> bool:
    """Register a new user"""
    users = load_users()
    if username in users:
        return False
    
    users[username] = {
        "password_hash": hash_password(password),
        "is_admin": is_admin,
        "created_at": datetime.now().isoformat()
    }
    save_users(users)
    return True

def authenticate_user(username: str, password: str) -> bool:
    """Authenticate a user"""
    users = load_users()
    if username not in users:
        return False
    
    return users[username]["password_hash"] == hash_password(password)

def is_admin(username: str) -> bool:
    """Check if user is admin"""
    users = load_users()
    return users.get(username, {}).get("is_admin", False)

def get_user_api_keys(username: str) -> List[str]:
    """Get API keys for a user"""
    api_keys = load_api_keys()
    return [key for key, key_data in api_keys.items() if key_data["username"] == username]

def create_api_key(username: str, key_name: str) -> str:
    """Create a new API key for a user"""
    api_keys = load_api_keys()
    new_key = generate_api_key()
    
    api_keys[new_key] = {
        "username": username,
        "name": key_name,
        "created_at": datetime.now().isoformat(),
        "is_active": True
    }
    
    save_api_keys(api_keys)
    return new_key

def revoke_api_key(api_key: str):
    """Revoke an API key"""
    api_keys = load_api_keys()
    if api_key in api_keys:
        api_keys[api_key]["is_active"] = False
        save_api_keys(api_keys)

def validate_api_key(api_key: str) -> bool:
    """Validate an API key"""
    api_keys = load_api_keys()
    return api_key in api_keys and api_keys[api_key]["is_active"]

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "chat"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(int(datetime.now().timestamp()))
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "show_login" not in st.session_state:
    st.session_state.show_login = True
if "show_register" not in st.session_state:
    st.session_state.show_register = False

# Helper functions
def load_file(file_path, file_extension):
    try:
        if file_extension.lower() == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            # Try different encodings for text files
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    loader = TextLoader(file_path, encoding=encoding)
                    return loader.load()
                except UnicodeDecodeError:
                    continue
            raise ValueError("Could not decode text file with any encoding")
        return loader.load()
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def ask_gemini(prompt):
    if not GOOGLE_API_KEY:
        return "API not configured. Please set GOOGLE_API_KEY environment variable."
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def process_file(file_path, filename):
    try:
        file_extension = os.path.splitext(filename)[1].lower()
        docs = load_file(file_path, file_extension)
        
        if not docs:
            return False
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        for chunk in chunks:
            chunk.metadata = {"filename": filename}
        
        if chunks:
            vectorstore.add_documents(chunks)
            return True
        return False
    except Exception as e:
        st.error(f"Processing error: {e}")
        return False

def get_files_list():
    files = []
    try:
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                files.append({
                    "id": filename,
                    "filename": filename,
                    "upload_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "size": stat.st_size
                })
    except Exception as e:
        st.error(f"Error listing files: {e}")
    
    return files

def delete_single_file(filename):
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            
            # Recreate vectorstore to remove the file's content
            global vectorstore
            if os.path.exists(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR)
            os.makedirs(CHROMA_DIR)
            vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
            
            # Reprocess remaining files
            for remaining_file in os.listdir(UPLOAD_DIR):
                remaining_file_path = os.path.join(UPLOAD_DIR, remaining_file)
                process_file(remaining_file_path, remaining_file)
            
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting file: {e}")
        return False

def clear_all_files():
    try:
        # Clear uploaded files
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Clear vectorstore
        global vectorstore
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        os.makedirs(CHROMA_DIR)
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        
        return True
    except Exception as e:
        st.error(f"Error clearing files: {e}")
        return False

def chat_with_docs(query):
    try:
        # Check if we have any documents
        try:
            results = vectorstore.similarity_search("test", k=1)
            has_documents = len(results) > 0
        except:
            has_documents = False
        
        if not has_documents:
            return {
                "answer": "I don't have any documents to reference yet. Please ask an admin to upload some documents first.",
                "sources": []
            }
        
        # Search for relevant content
        results = vectorstore.similarity_search(query, k=5)
        context = "\n\n".join([doc.page_content for doc in results])
        sources = list(set([doc.metadata.get("filename", "Unknown") for doc in results]))
        
        prompt = f"""
        Based on the following context, provide a helpful and natural response to: {query}
        
        Context:
        {context}
        
        If the information isn't in the context, politely say you don't have that information yet.
        Respond in a friendly, conversational tone as if you're having a natural conversation.
        """
        
        answer = ask_gemini(prompt)
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        return {
            "answer": f"I encountered an error while processing your request: {str(e)}",
            "sources": []
        }

# Authentication UI
if not st.session_state.authenticated:
    st.markdown('<h1 class="main-header">🔐 Document Assistant</h1>', unsafe_allow_html=True)
    
    if st.session_state.show_login:
        with st.container():
            st.markdown('<div class="auth-container text-success">', unsafe_allow_html=True)
            st.subheader("Login")
            
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                
                if submit:
                    if authenticate_user(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.show_login = False
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
            
            if st.button("Create new account"):
                st.session_state.show_login = False
                st.session_state.show_register = True
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.show_register:
        with st.container():
            st.markdown('<div class="auth-container">', unsafe_allow_html=True)
            st.subheader("Register")
            
            with st.form("register_form"):
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submit = st.form_submit_button("Register")
                
                if submit:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif register_user(new_username, new_password):
                        st.success("Account created successfully! Please login.")
                        st.session_state.show_register = False
                        st.session_state.show_login = True
                        st.rerun()
                    else:
                        st.error("Username already exists")
            
            if st.button("Back to login"):
                st.session_state.show_register = False
                st.session_state.show_login = True
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.stop()

# Main application (only shown if authenticated)
# Sidebar navigation
with st.sidebar:
    st.title(f"📚 Document Assistant")
    st.write(f"Welcome, **{st.session_state.username}**")
    
    if is_admin(st.session_state.username):
        st.success("👑 Admin User")
    else:
        st.info("👤 Regular User")
    
    st.divider()
    
    page_options = ["💬 Chat", "🔑 API Keys", "⚙️ Settings"]
    if is_admin(st.session_state.username):
        page_options.insert(1, "📁 Files")  # Add Files tab only for admins
        page_options.insert(2, "📊 Analytics")  # Add Analytics tab for admins
        page_options.append("👑 Admin")
    
    page = st.radio("Navigation", page_options, index=0)
    
    if "Chat" in page:
        st.session_state.page = "chat"
    elif "Files" in page:
        st.session_state.page = "files"
    elif "Analytics" in page:
        st.session_state.page = "analytics"
    elif "API Keys" in page:
        st.session_state.page = "api_keys"
    elif "Settings" in page:
        st.session_state.page = "settings"
    elif "Admin" in page:
        st.session_state.page = "admin"
    
    st.divider()
    
    # System status
    files_count = len(get_files_list())
    if files_count > 0:
        st.success(f"✅ {files_count} document(s) loaded")
    else:
        st.warning("📝 No documents available")
    
    if not GOOGLE_API_KEY:
        st.error("❌ Google API key not set")
    else:
        st.success("✅ Google API configured")
    
    st.divider()
    
    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()

# Chat page
if st.session_state.page == "chat":
    st.markdown('<h1 class="main-header">💬 Document Chat</h1>', unsafe_allow_html=True)
    
    # File uploader - ONLY FOR ADMINS
    if is_admin(st.session_state.username):
        uploaded_file = st.file_uploader("Upload a document (PDF or TXT)", type=["pdf", "txt"])
        if uploaded_file is not None:
            if uploaded_file.name not in st.session_state.uploaded_files:
                with st.spinner("Uploading and processing document..."):
                    # Save the file
                    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Process the file
                    success = process_file(file_path, uploaded_file.name)
                    
                    if success:
                        st.success(f"✅ Uploaded {uploaded_file.name} successfully!")
                        st.session_state.uploaded_files[uploaded_file.name] = True
                    else:
                        st.error("❌ Failed to process file. Please try another file.")
                        # Remove the file if processing failed
                        if os.path.exists(file_path):
                            os.remove(file_path)
    else:
        # Show message for non-admin users
        st.info("📚 Only administrators can upload documents. Please ask an admin to add documents to the system.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.container():
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant">{message["content"]}</div>', unsafe_allow_html=True)
                if message.get("sources"):
                    st.caption(f"📁 Sources: {', '.join(message['sources'])}")
    
    # Chat input
    if prompt := st.chat_input("Ask something about the documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get AI response
        with st.spinner("🤔 Thinking..."):
            response = chat_with_docs(prompt)
        
        # Add AI response
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response.get("answer", "I'm not sure how to respond to that."),
            "sources": response.get("sources", [])
        })
        
        # Track the query for analytics
        track_query(
            st.session_state.username, 
            prompt, 
            response.get("answer", ""), 
            response.get("sources", [])
        )
        
        st.rerun()

# Files page - ONLY FOR ADMINS
elif st.session_state.page == "files" and is_admin(st.session_state.username):
    st.markdown('<h1 class="main-header">📁 Document Management</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Uploaded Documents")
        
        # Refresh files button
        if st.button("🔄 Refresh Files", use_container_width=True):
            st.rerun()
        
        # Display files
        files = get_files_list()
        if not files:
            st.info("📝 No documents uploaded yet. Use the uploader on the Chat page to add documents.")
        else:
            for file in files:
                filename = file.get('filename', 'Unknown')
                file_size = file.get('size', 0)
                upload_time = file.get('upload_time', 'Unknown')
                
                with st.container():
                    st.markdown(f"""
                    <div class="file-card text-dark">
                        <h4>📄 {filename}</h4>
                        <p>📦 Size: {file_size} bytes</p>
                        <p>⏰ Uploaded: {upload_time}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"🗑️ Delete {filename}", key=f"delete_{filename}", use_container_width=True):
                        with st.spinner("Deleting..."):
                            success = delete_single_file(filename)
                            if success:
                                st.success(f"✅ Deleted {filename}")
                                if filename in st.session_state.uploaded_files:
                                    del st.session_state.uploaded_files[filename]
                                st.rerun()
                            else:
                                st.error("❌ Failed to delete file")
    
    with col2:
        st.subheader("Actions")
        
        if st.button("🗑️ Clear All Files", type="secondary", use_container_width=True):
            if files:
                if st.checkbox("I'm sure I want to delete ALL files"):
                    with st.spinner("Clearing all files..."):
                        success = clear_all_files()
                        if success:
                            st.success("✅ All files cleared!")
                            st.session_state.messages = []
                            st.session_state.uploaded_files = {}
                            st.rerun()
                        else:
                            st.error("❌ Failed to clear files")
            else:
                st.warning("No files to delete")

# Analytics page - ONLY FOR ADMINS
elif st.session_state.page == "analytics" and is_admin(st.session_state.username):
    st.markdown('<h1 class="main-header">📊 Chatbot Analytics</h1>', unsafe_allow_html=True)
    
    analytics = load_analytics()
    
    if not analytics["queries"]:
        st.info("No analytics data available yet. Users need to start asking questions.")
        st.stop()
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(analytics["queries"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    
    # Overall stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Queries", len(analytics["queries"]))
    with col2:
        st.metric("Active Users", len(analytics["users"]))
    with col3:
        most_active_user = max(analytics["users"].items(), key=lambda x: x[1]["query_count"])[0] if analytics["users"] else "N/A"
        st.metric("Most Active User", most_active_user)
    with col4:
        total_queries = len(analytics["queries"])
        unique_questions = len(analytics["popular_questions"])
        st.metric("Unique Questions", unique_questions)
    
    st.divider()
    
    # Popular questions
    st.subheader("Most Popular Questions")
    popular_questions = sorted(analytics["popular_questions"].items(), key=lambda x: x[1], reverse=True)[:10]
    
    if popular_questions:
        for i, (question, count) in enumerate(popular_questions, 1):
            st.write(f"{i}. **{question}** ({count} asks)")
    else:
        st.info("No popular questions data yet.")
    
    st.divider()
    
    # Query activity over time
    st.subheader("Query Activity Over Time")
    
    # Daily activity
    daily_activity = df.groupby('date').size().reset_index(name='count')
    
    if not daily_activity.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(daily_activity['date'], daily_activity['count'], marker='o')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Queries')
        ax.set_title('Daily Query Activity')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Not enough data to show activity charts.")
    
    st.divider()
    
    # User activity
    st.subheader("User Activity")
    user_activity = []
    for username, data in analytics["users"].items():
        user_activity.append({
            "username": username,
            "query_count": data["query_count"],
            "last_active": data["last_active"]
        })
    
    user_df = pd.DataFrame(user_activity)
    if not user_df.empty:
        user_df = user_df.sort_values("query_count", ascending=False)
        st.dataframe(user_df, use_container_width=True)
    else:
        st.info("No user activity data available.")
    
    st.divider()
    
    # Recent queries
    st.subheader("Recent Queries")
    recent_queries = analytics["queries"][-10:]  # Last 10 queries
    for query in reversed(recent_queries):
        with st.expander(f"{query['username']} - {query['timestamp']}"):
            st.write(f"**Question:** {query['question']}")
            st.write(f"**Answer excerpt:** {query['answer']}")
            if query['sources']:
                st.write(f"**Sources:** {', '.join(query['sources'])}")

# API Keys page
elif st.session_state.page == "api_keys":
    st.markdown('<h1 class="main-header">🔑 API Keys</h1>', unsafe_allow_html=True)
    
    # Create new API key
    st.subheader("Create New API Key")
    with st.form("create_api_key"):
        key_name = st.text_input("Key Name", placeholder="e.g., Production Key")
        create_key = st.form_submit_button("Generate API Key")
        
        if create_key and key_name:
            new_key = create_api_key(st.session_state.username, key_name)
            st.success("✅ API Key created successfully!")
            st.markdown(f'<div class="api-key-display">{new_key}</div>', unsafe_allow_html=True)
            st.warning("⚠️ Copy this key now! It won't be shown again.")
    
    st.divider()
    
    # List existing API keys
    st.subheader("Your API Keys")
    api_keys = load_api_keys()
    user_keys = [key for key, key_data in api_keys.items() if key_data["username"] == st.session_state.username]
    
    if not user_keys:
        st.info("You don't have any API keys yet. Create one above.")
    else:
        for key in user_keys:
            key_data = api_keys[key]
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{key_data['name']}**")
                st.caption(f"Created: {key_data['created_at']}")
                st.caption(f"Status: {'✅ Active' if key_data['is_active'] else '❌ Revoked'}")
            
            with col2:
                if key_data['is_active']:
                    if st.button("📋 Copy", key=f"copy_{key}"):
                        st.write(key)  # This would need JavaScript to actually copy to clipboard
                        st.info("Key copied to clipboard")
            
            with col3:
                if key_data['is_active']:
                    if st.button("❌ Revoke", key=f"revoke_{key}"):
                        revoke_api_key(key)
                        st.success("Key revoked successfully")
                        st.rerun()

# Settings page
elif st.session_state.page == "settings":
    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("System Information")
        
        # Document count
        files = get_files_list()
        st.write(f"📊 Documents loaded: {len(files)}")
        
        # Vector store info
        try:
            if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
                st.success("✅ Vector database is ready")
            else:
                st.warning("📝 Vector database is empty")
        except:
            st.error("❌ Vector database error")
        
        # API key status
        if GOOGLE_API_KEY:
            st.success("✅ Google API key is configured")
        else:
            st.error("❌ Google API key not found")
            st.info("Create a .env file with: GOOGLE_API_KEY=your_key_here")
        
        st.subheader("Session Information")
        st.write(f"Session ID: `{st.session_state.session_id}`")
        
        if st.button("🆕 New Session", use_container_width=True):
            st.session_state.session_id = str(int(datetime.now().timestamp()))
            st.session_state.messages = []
            st.success("✅ New session created!")
    
    with col2:
        st.subheader("User Information")
        st.write(f"Username: `{st.session_state.username}`")
        st.write(f"Role: {'👑 Admin' if is_admin(st.session_state.username) else '👤 User'}")
        
        st.subheader("Change Password")
        with st.form("change_password"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            change_pass = st.form_submit_button("Change Password")
            
            if change_pass:
                if not authenticate_user(st.session_state.username, current_password):
                    st.error("Current password is incorrect")
                elif new_password != confirm_password:
                    st.error("New passwords don't match")
                else:
                    users = load_users()
                    users[st.session_state.username]["password_hash"] = hash_password(new_password)
                    save_users(users)
                    st.success("✅ Password changed successfully!")

# Admin page - ONLY FOR ADMINS
elif st.session_state.page == "admin" and is_admin(st.session_state.username):
    st.markdown('<h1 class="main-header">👑 Admin Panel</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["User Management", "API Key Management", "System Stats"])
    
    with tab1:
        st.subheader("User Management")
        users = load_users()
        
        for username, user_data in users.items():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{username}**")
                st.caption(f"Created: {user_data['created_at']}")
                st.caption(f"Role: {'👑 Admin' if user_data['is_admin'] else '👤 User'}")
            
            with col2:
                if username != st.session_state.username:
                    if st.button("Toggle Admin", key=f"admin_{username}"):
                        users[username]["is_admin"] = not users[username]["is_admin"]
                        save_users(users)
                        st.success(f"Updated admin status for {username}")
                        st.rerun()
            
            with col3:
                if username != st.session_state.username:
                    if st.button("Delete User", key=f"delete_{username}"):
                        del users[username]
                        save_users(users)
                        st.success(f"Deleted user {username}")
                        st.rerun()
            
            st.divider()
        
        st.subheader("Create New User")
        with st.form("create_user"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            is_admin_user = st.checkbox("Admin User")
            create_user = st.form_submit_button("Create User")
            
            if create_user:
                if new_username in users:
                    st.error("Username already exists")
                else:
                    register_user(new_username, new_password, is_admin_user)
                    st.success(f"User {new_username} created successfully!")
                    st.rerun()
    
    with tab2:
        st.subheader("API Key Management")
        api_keys = load_api_keys()
        
        for key, key_data in api_keys.items():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{key_data['name']}**")
                st.caption(f"User: {key_data['username']}")
                st.caption(f"Created: {key_data['created_at']}")
                st.caption(f"Status: {'✅ Active' if key_data['is_active'] else '❌ Revoked'}")
            
            with col2:
                if st.button("Toggle Active", key=f"toggle_{key}"):
                    api_keys[key]["is_active"] = not api_keys[key]["is_active"]
                    save_api_keys(api_keys)
                    st.success("Key status updated")
                    st.rerun()
            
            with col3:
                if st.button("Delete", key=f"delete_key_{key}"):
                    del api_keys[key]
                    save_api_keys(api_keys)
                    st.success("Key deleted")
                    st.rerun()
            
            st.divider()
    
    with tab3:
        st.subheader("System Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Users", len(load_users()))
        
        with col2:
            st.metric("Total API Keys", len(load_api_keys()))
        
        with col3:
            st.metric("Active API Keys", len([k for k, v in load_api_keys().items() if v["is_active"]]))
        
        st.subheader("Document Statistics")
        files = get_files_list()
        st.metric("Uploaded Documents", len(files))
        
        if files:
            total_size = sum(f["size"] for f in files)
            st.metric("Total Storage Used", f"{total_size / (1024*1024):.2f} MB")

# Add instructions for first-time users
# if st.session_state.page == "chat" and not get_files_list():
#     if is_admin(st.session_state.username):
#         st.info("""
#         👋 Welcome to Document Assistant!
        
#         To get started:
#         1. Upload a PDF or text file using the uploader above
#         2. Ask questions about your document
#         3. The
        
        
        
# Add instructions for first-time users
if st.session_state.page == "chat" and not get_files_list():
    if is_admin(st.session_state.username):
        st.info("""
        👋 Welcome to Document Assistant!
        
        To get started:
        1. Upload a PDF or text file using the uploader above
        2. Ask questions about your document
        3. The AI will help you analyze and understand your content
        
        Make sure you have a Google API key set in the .env file!
        """)
    else:
        st.info("""
        👋 Welcome to Document Assistant!
        
        This system allows you to ask questions about documents that have been uploaded by administrators.
        
        Currently, there are no documents in the system. Please ask an admin to upload some documents.
        
        Once documents are available, you can ask questions using the chat interface.
        """)

# Display API key instructions if not set
if not GOOGLE_API_KEY:
    st.sidebar.error("""
    **Google API Key Required**
    
    Create a .env file with:
    ```
    GOOGLE_API_KEY=your_actual_key_here
    ```
    
    Get a key from: https://aistudio.google.com/
    """)

# Create default admin user if no users exist
if not load_users():
    register_user("admin", "admin123", True)
    st.sidebar.info("Default admin user created: admin/admin123")