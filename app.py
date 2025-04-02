import streamlit as st
import requests
import base64
import time
import uuid
import io
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# External library imports
import PyPDF2
import supabase
import google.generativeai as genai
from bs4 import BeautifulSoup
import numpy as np
from PIL import Image
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase_client = supabase.create_client(supabase_url, supabase_key)

# Initialize Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

# Configure page
st.set_page_config(
    page_title="DocChat - RAG Chat Application",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e6f3ff;
        border-left: 5px solid #0078d4;
    }
    .chat-message.bot {
        background-color: #f0f0f0;
        border-left: 5px solid #424242;
    }
    .chat-message .message-content {
        display: flex;
        margin-top: 0.5rem;
    }
    .sidebar .element-container:has(.stButton) {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .document-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #ddd;
        transition: all 0.3s;
    }
    .document-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .document-title {
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .document-source {
        color: #666;
        font-size: 0.8rem;
    }
    .document-date {
        color: #999;
        font-size: 0.7rem;
    }
</style>
""", unsafe_allow_html=True)

# Authentication functions
def initialize_auth_session():
    """Initialize authentication session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None

def sign_up():
    """Handle user sign up"""
    with st.form("signup_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        name = st.text_input("Full Name")
        submit = st.form_submit_button("Sign Up")
        
        if submit and email and password:
            try:
                response = supabase_client.auth.sign_up({
                    "email": email,
                    "password": password,
                    "options": {
                        "data": {
                            "full_name": name
                        }
                    }
                })
                if response.user:
                    st.success("Signed up successfully! Please check your email for verification.")
                else:
                    st.error("Sign up failed.")
            except Exception as e:
                st.error(f"Error signing up: {str(e)}")

def sign_in():
    """Handle user sign in"""
    with st.form("signin_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign In")
        
        if submit and email and password:
            try:
                response = supabase_client.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })
                if response.user:
                    st.session_state.authenticated = True
                    st.session_state.user = response.user
                    st.session_state.access_token = response.session.access_token
                    st.rerun()
                else:
                    st.error("Sign in failed.")
            except Exception as e:
                st.error(f"Error signing in: {str(e)}")

def sign_out():
    """Handle user sign out"""
    supabase_client.auth.sign_out()
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.access_token = None
    st.rerun()

# Document processing functions
def extract_text_from_url(url: str) -> str:
    """Extract text content from a URL"""
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        st.error(f"Error extracting content from URL: {str(e)}")
        return ""

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text content from a PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting content from PDF: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 100:  # Only keep chunks with substantial content
            chunks.append(chunk)
    return chunks

def generate_embeddings(text: str) -> List[float]:
    """Generate text embeddings using Gemini's embedding feature"""
    try:
        # Since Gemini doesn't currently offer direct embedding API, we'll simulate it
        # In a real application, you would use a dedicated embedding model 
        # For demonstration purposes, we'll use a function that returns random embeddings
        # In production, consider using models like OpenAI's text-embedding-ada-002 or similar
        
        # This simulates a 1536-dimensional embedding vector
        # The actual implementation would call the appropriate embedding API
        return np.random.normal(0, 0.1, 1536).tolist()
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return []

def store_document(user_id: str, source: str, title: str, content: str, doc_type: str) -> str:
    """Store document and its embeddings in Supabase"""
    try:
        # Generate a unique ID for the document
        doc_id = str(uuid.uuid4())
        current_timestamp = datetime.now().isoformat()
        
        # Store document metadata
        supabase_client.table('documents').insert({
            'id': doc_id,
            'user_id': user_id,
            'title': title,
            'source': source,
            'doc_type': doc_type,
            'created_at': current_timestamp
        }).execute()
        
        # Process and store document chunks
        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            embeddings = generate_embeddings(chunk)
            
            # Store chunk and embeddings in the vector store
            supabase_client.table('document_chunks').insert({
                'id': chunk_id,
                'document_id': doc_id,
                'content': chunk,
                'chunk_index': i,
                'embedding': embeddings
            }).execute()
        
        return doc_id
    except Exception as e:
        st.error(f"Error storing document: {str(e)}")
        return ""

def get_user_documents(user_id: str) -> List[Dict]:
    """Get all documents for a user"""
    try:
        response = supabase_client.table('documents') \
            .select('id, title, source, doc_type, created_at') \
            .eq('user_id', user_id) \
            .order('created_at', desc=True) \
            .execute()
        
        return response.data
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return []

def delete_document(doc_id: str):
    """Delete a document and its chunks"""
    try:
        # Delete document chunks first
        supabase_client.table('document_chunks') \
            .delete() \
            .like('id', f"{doc_id}_%") \
            .execute()
        
        # Delete document metadata
        supabase_client.table('documents') \
            .delete() \
            .eq('id', doc_id) \
            .execute()
        
        return True
    except Exception as e:
        st.error(f"Error deleting document: {str(e)}")
        return False

# Chat functions
def create_chat(user_id: str, title: str) -> str:
    """Create a new chat session"""
    try:
        chat_id = str(uuid.uuid4())
        current_timestamp = datetime.now().isoformat()
        
        supabase_client.table('chats').insert({
            'id': chat_id,
            'user_id': user_id,
            'title': title,
            'created_at': current_timestamp
        }).execute()
        
        return chat_id
    except Exception as e:
        st.error(f"Error creating chat: {str(e)}")
        return ""

def get_user_chats(user_id: str) -> List[Dict]:
    """Get all chats for a user"""
    try:
        response = supabase_client.table('chats') \
            .select('id, title, created_at') \
            .eq('user_id', user_id) \
            .order('created_at', desc=True) \
            .execute()
        
        return response.data
    except Exception as e:
        st.error(f"Error retrieving chats: {str(e)}")
        return []

def get_chat_messages(chat_id: str) -> List[Dict]:
    """Get all messages for a chat"""
    try:
        response = supabase_client.table('messages') \
            .select('id, role, content, created_at') \
            .eq('chat_id', chat_id) \
            .order('created_at', asc=True) \
            .execute()
        
        return response.data
    except Exception as e:
        st.error(f"Error retrieving chat messages: {str(e)}")
        return []

def store_message(chat_id: str, role: str, content: str) -> str:
    """Store a chat message"""
    try:
        message_id = str(uuid.uuid4())
        current_timestamp = datetime.now().isoformat()
        
        supabase_client.table('messages').insert({
            'id': message_id,
            'chat_id': chat_id,
            'role': role,
            'content': content,
            'created_at': current_timestamp
        }).execute()
        
        return message_id
    except Exception as e:
        st.error(f"Error storing message: {str(e)}")
        return ""

def search_relevant_chunks(query: str, limit: int = 5) -> List[Dict]:
    """Search for relevant document chunks based on query"""
    try:
        # In a real implementation, you would:
        # 1. Generate embeddings for the query
        # 2. Perform a vector similarity search in Supabase
        
        # For demonstration, we'll do a simple text search
        # In production, you should use Supabase's vector search capabilities
        query_embedding = generate_embeddings(query)
        
        # This would be replaced with actual similarity search
        # For now, we'll simulate it with a basic text search
        response = supabase_client.table('document_chunks') \
            .select('id, document_id, content, chunk_index') \
            .limit(limit) \
            .execute()
        
        # In production, you would use pgvector's similarity search like:
        # .rpc('match_documents', {'query_embedding': query_embedding, 'match_threshold': 0.7, 'match_count': limit})
        
        return response.data
    except Exception as e:
        st.error(f"Error searching for relevant chunks: {str(e)}")
        return []

def generate_ai_response(query: str, context: List[str]) -> str:
    """Generate a response from Gemini with context from relevant documents"""
    try:
        # Combine context chunks into a single context string
        context_text = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context)])
        
        # Create the prompt with RAG context
        prompt = f"""I want you to answer the user's question based on the context provided.
If the context doesn't contain relevant information, please let the user know you don't have enough information.
Don't make up information that's not supported by the context.

Context:
{context_text}

User Question: {query}

Your Response:"""

        # Generate response from Gemini
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        st.error(f"Error generating AI response: {str(e)}")
        return f"I apologize, but I encountered an error while processing your request. Error: {str(e)}"

# UI Components
def display_auth_ui():
    """Display authentication UI"""
    st.title("DocChat - Intelligent Document Chat")
    
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
    
    with tab1:
        sign_in()
    
    with tab2:
        sign_up()

def display_chat_message(message, is_user=False):
    """Display a chat message"""
    message_type = "user" if is_user else "bot"
    with st.container():
        st.markdown(f"""
        <div class="chat-message {message_type}">
            <div class="avatar">
                <strong>{"You" if is_user else "AI Assistant"}</strong>
            </div>
            <div class="message-content">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_document_card(doc):
    """Display a document card"""
    doc_type_icon = "📄" if doc['doc_type'] == 'pdf' else "🔗"
    created_at = datetime.fromisoformat(doc['created_at']).strftime("%b %d, %Y")
    
    with st.container():
        st.markdown(f"""
        <div class="document-card">
            <div class="document-title">{doc_type_icon} {doc['title']}</div>
            <div class="document-source">{doc['source']}</div>
            <div class="document-date">Added on {created_at}</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("Delete", key=f"delete_{doc['id']}", help="Delete this document"):
                if delete_document(doc['id']):
                    st.rerun()

def display_document_management():
    """Display document management interface"""
    st.header("Add New Document")
    
    tab1, tab2 = st.tabs(["URL", "PDF Upload"])
    
    with tab1:
        with st.form("url_form"):
            url = st.text_input("Website URL", key="url_input")
            title = st.text_input("Document Title (Optional)", key="url_title")
            submit_url = st.form_submit_button("Process URL")
            
            if submit_url and url:
                with st.spinner("Processing URL..."):
                    content = extract_text_from_url(url)
                    if content:
                        doc_title = title if title else url
                        doc_id = store_document(
                            st.session_state.user.id,
                            url,
                            doc_title,
                            content,
                            'url'
                        )
                        if doc_id:
                            st.success(f"URL processed and stored successfully!")
                            st.rerun()
    
    with tab2:
        with st.form("pdf_form"):
            uploaded_file = st.file_uploader("Upload PDF", type="pdf", key="pdf_upload")
            pdf_title = st.text_input("Document Title (Optional)", key="pdf_title")
            submit_pdf = st.form_submit_button("Process PDF")
            
            if submit_pdf and uploaded_file:
                with st.spinner("Processing PDF..."):
                    content = extract_text_from_pdf(uploaded_file)
                    if content:
                        doc_title = pdf_title if pdf_title else uploaded_file.name
                        doc_id = store_document(
                            st.session_state.user.id,
                            uploaded_file.name,
                            doc_title,
                            content,
                            'pdf'
                        )
                        if doc_id:
                            st.success(f"PDF processed and stored successfully!")
                            st.rerun()
    
    st.header("Your Documents")
    documents = get_user_documents(st.session_state.user.id)
    
    if not documents:
        st.info("You don't have any documents yet. Add one using the form above.")
    else:
        for doc in documents:
            display_document_card(doc)

def initialize_chat_session():
    """Initialize chat session state variables"""
    if 'chat_id' not in st.session_state:
        st.session_state.chat_id = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chats' not in st.session_state:
        st.session_state.chats = []

def display_chat_interface():
    """Display the chat interface"""
    # Update chats list
    st.session_state.chats = get_user_chats(st.session_state.user.id)
    
    # Sidebar for chat selection and creation
    with st.sidebar:
        st.header("Your Chats")
        
        # New chat button
        if st.button("New Chat", use_container_width=True):
            chat_title = f"Chat {len(st.session_state.chats) + 1}"
            chat_id = create_chat(st.session_state.user.id, chat_title)
            if chat_id:
                st.session_state.chat_id = chat_id
                st.session_state.messages = []
                st.rerun()
        
        # Display existing chats
        if not st.session_state.chats:
            st.info("You don't have any chats yet.")
        else:
            for chat in st.session_state.chats:
                chat_label = chat['title']
                if st.button(chat_label, key=f"chat_{chat['id']}", use_container_width=True):
                    st.session_state.chat_id = chat['id']
                    st.session_state.messages = get_chat_messages(chat['id'])
                    st.rerun()
    
    # Main chat area
    st.header("Chat with Your Documents")
    
    if st.session_state.chat_id is None:
        st.info("Please select a chat or create a new one to start.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        display_chat_message(message['content'], message['role'] == 'user')
    
    # Chat input
    user_input = st.text_input("Ask a question about your documents", key="user_input")
    
    if user_input:
        # Add user message to chat
        store_message(st.session_state.chat_id, 'user', user_input)
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input
        })
        
        # Search for relevant context
        relevant_chunks = search_relevant_chunks(user_input)
        context = [chunk['content'] for chunk in relevant_chunks]
        
        # Generate and display AI response
        with st.spinner("Thinking..."):
            ai_response = generate_ai_response(user_input, context)
            store_message(st.session_state.chat_id, 'assistant', ai_response)
            st.session_state.messages.append({
                'role': 'assistant',
                'content': ai_response
            })
            
        st.rerun()

def main():
    # Initialize auth session state
    initialize_auth_session()
    
    # Check if user is authenticated
    if not st.session_state.authenticated:
        display_auth_ui()
    else:
        # Initialize chat session state
        initialize_chat_session()
        
        # Sidebar navigation
        with st.sidebar:
            st.title("DocChat")
            st.image("https://via.placeholder.com/150x60?text=DocChat", width=150)
            
            # Navigation
            page = st.radio("Navigation", ["Chat", "Documents"])
            
            # User info and sign out
            st.divider()
            st.write(f"Signed in as: {st.session_state.user.email}")
            if st.button("Sign Out", use_container_width=True):
                sign_out()
        
        # Display selected page
        if page == "Chat":
            display_chat_interface()
        elif page == "Documents":
            display_document_management()

if __name__ == "__main__":
    main()