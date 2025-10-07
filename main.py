import streamlit as st
import requests
import time
from datetime import datetime
import os

# Configuration
API_BASE = "http://localhost:8006"

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
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .chat-message.user {
        background-color: #0f4c75;
        color: white;
        margin-left: 20%;
    }
    .chat-message.assistant {
        background-color: #f0f2f6;
        margin-right: 20%;
    }
    .file-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "chat"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(int(time.time()))
if "files" not in st.session_state:
    st.session_state.files = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

# API functions
def api_request(endpoint, method="GET", data=None, files=None):
    try:
        url = f"{API_BASE}{endpoint}"
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files)
            else:
                headers = {"Content-Type": "application/json"}
                response = requests.post(url, json=data, headers=headers)
        elif method == "DELETE":
            response = requests.delete(url)
        
        if response.status_code in [200, 201]:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API server. Make sure it's running on port 8006.")
        return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def upload_file(file):
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = api_request("/upload", "POST", files=files)
        return response
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return None

def send_message(message):
    data = {"query": message, "session_id": st.session_state.session_id}
    response = api_request("/chat", "POST", data=data)
    return response

def get_files():
    response = api_request("/files", "GET")
    if response and isinstance(response, list):
        return response
    return []

def delete_file(file_id):
    response = api_request(f"/files/{file_id}", "DELETE")
    return response

def clear_all_files():
    response = api_request("/files", "DELETE")
    return response

# Sidebar navigation
with st.sidebar:
    st.title("📚 Document Assistant")
    st.divider()
    
    page_options = ["💬 Chat", "📁 Files", "⚙️ Settings"]
    page = st.radio("Navigation", page_options, index=0)
    
    if "Chat" in page:
        st.session_state.page = "chat"
    elif "Files" in page:
        st.session_state.page = "files"
    else:
        st.session_state.page = "settings"
    
    st.divider()
    
    # API status
    try:
        health = requests.get(f"{API_BASE}/health", timeout=2)
        if health.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Not Connected")
    
    st.info("Upload documents and chat with AI about their content!")

# Chat page
if st.session_state.page == "chat":
    st.markdown('<h1 class="main-header">💬 Document Chat</h1>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a document (PDF or TXT)", type=["pdf", "txt"])
    if uploaded_file is not None:
        if uploaded_file.name not in st.session_state.uploaded_files:
            with st.spinner("Uploading and processing document..."):
                result = upload_file(uploaded_file)
                if result and result.get("status") == "success":
                    st.success(f"✅ Uploaded {uploaded_file.name} successfully!")
                    st.session_state.uploaded_files[uploaded_file.name] = True
                    # Refresh files list
                    st.session_state.files = get_files()
                else:
                    st.error("❌ Failed to upload file. Check if API server is running.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.container():
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user">👤 **You:** {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant">🤖 **Assistant:** {message["content"]}</div>', unsafe_allow_html=True)
                if message.get("sources"):
                    st.caption(f"📁 Sources: {', '.join(message['sources'])}")
    
    # Chat input
    if prompt := st.chat_input("Ask something about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get AI response
        with st.spinner("🤔 Thinking..."):
            response = send_message(prompt)
        
        if response:
            # Add AI response
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response.get("answer", "I'm not sure how to respond to that."),
                "sources": response.get("sources", [])
            })
            st.rerun()
        else:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "Sorry, I couldn't connect to the AI service. Please make sure the API server is running.",
                "sources": []
            })
            st.rerun()

# Files page
elif st.session_state.page == "files":
    st.markdown('<h1 class="main-header">📁 Document Management</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Uploaded Documents")
        
        # Refresh files button
        if st.button("🔄 Refresh Files", use_container_width=True):
            st.session_state.files = get_files()
            st.rerun()
        
        # Display files
        files = get_files()
        if not files:
            st.info("📝 No documents uploaded yet. Go to the Chat page to upload some!")
        else:
            for file in files:
                # Safely access file properties with defaults
                filename = file.get('filename', 'Unknown')
                file_id = file.get('id', 'unknown')
                file_size = file.get('size', 0)
                upload_time = file.get('upload_time', 'Unknown')
                
                with st.container():
                    st.markdown(f"""
                    <div class="file-card">
                        <h4>📄 {filename}</h4>
                        <p>🆔 ID: {file_id} | 📦 Size: {file_size} bytes</p>
                        <p>⏰ Uploaded: {upload_time}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"🗑️ Delete {filename}", key=f"delete_{file_id}", use_container_width=True):
                        with st.spinner("Deleting..."):
                            result = delete_file(file_id)
                            if result and result.get("status") == "success":
                                st.success(f"✅ Deleted {filename}")
                                st.session_state.files = get_files()
                                if filename in st.session_state.uploaded_files:
                                    del st.session_state.uploaded_files[filename]
                                st.rerun()
                            else:
                                st.error("❌ Failed to delete file")
    
    with col2:
        st.subheader("Actions")
        
        if st.button("🗑️ Clear All Files", type="secondary", use_container_width=True):
            if st.session_state.files:
                if st.checkbox("I'm sure I want to delete ALL files"):
                    with st.spinner("Clearing all files..."):
                        result = clear_all_files()
                        if result and result.get("status") == "success":
                            st.success("✅ All files cleared!")
                            st.session_state.files = []
                            st.session_state.messages = []
                            st.session_state.uploaded_files = {}
                            st.rerun()
                        else:
                            st.error("❌ Failed to clear files")
            else:
                st.warning("No files to delete")

# Settings page
else:
    st.markdown('<h1 class="main-header">⚙️ Settings & API</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("API Information")
        st.code(f"""
        Base URL: {API_BASE}
        
        Endpoints:
        - POST /upload - Upload a document
        - POST /chat - Send a message
        - GET /files - List uploaded files
        - DELETE /files/{{id}} - Delete a specific file
        - DELETE /files - Delete all files
        - GET /health - Health check
        
        Example usage:
        ```python
        import requests
        
        # Upload a file
        with open("document.pdf", "rb") as f:
            response = requests.post("{API_BASE}/upload", files={{"file": f}})
        
        # Chat with documents
        response = requests.post("{API_BASE}/chat", json={{
            "query": "What is this document about?",
            "session_id": "user123"
        }})
        ```
        """)
    
    with col2:
        st.subheader("Session Information")
        st.write(f"Session ID: `{st.session_state.session_id}`")
        
        if st.button("🆕 New Session", use_container_width=True):
            st.session_state.session_id = str(int(time.time()))
            st.session_state.messages = []
            st.success("✅ New session created!")
        
        st.subheader("System Status")
        try:
            health = requests.get(f"{API_BASE}/health", timeout=2)
            if health.status_code == 200:
                health_data = health.json()
                st.success(f"✅ API Status: {health_data.get('status', 'unknown')}")
                st.success(f"✅ Last check: {health_data.get('timestamp', 'unknown')}")
            else:
                st.error("❌ API Error")
        except:
            st.error("❌ API is not responding")
        
        st.subheader("Quick Actions")
        if st.button("🔄 Refresh All Data", use_container_width=True):
            st.session_state.files = get_files()
            st.success("✅ Data refreshed!")

# Initialize files list on first load
if st.session_state.files == []:
    st.session_state.files = get_files()

# Add instructions for first-time users
if st.session_state.page == "chat" and not st.session_state.files:
    st.info("""
    👋 Welcome to Document Assistant!
    
    To get started:
    1. Make sure the API server is running (check the sidebar status)
    2. Upload a PDF or text file using the uploader above
    3. Ask questions about your document
    4. The AI will help you analyze and understand your content
    
    To start the API server, run: `python api.py`
    """)

# Display API connection instructions if not connected
try:
    health = requests.get(f"{API_BASE}/health", timeout=2)
    if health.status_code != 200:
        st.sidebar.warning("""
        **API Server Not Running**
        
        To start the API:
        1. Open a terminal
        2. Run: `python api.py`
        3. Wait for "Application startup complete"
        4. Refresh this page
        """)
except:
    st.sidebar.warning("""
    **API Server Not Running**
    
    To start the API:
    1. Open a terminal
    2. Run: `python api.py`
    3. Wait for "Application startup complete"
    4. Refresh this page
    """)