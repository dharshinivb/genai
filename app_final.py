# import streamlit as st
# import requests
# import base64
# import time
# import uuid
# import io
# import os
# import json
# from datetime import datetime
# from typing import List, Dict, Any, Optional, Tuple

# # External library imports
# import PyPDF2
# import supabase
# import google.generativeai as genai
# from bs4 import BeautifulSoup
# import numpy as np
# from PIL import Image
# import pandas as pd
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Initialize Supabase client
# supabase_url = os.getenv("SUPABASE_URL")
# supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
# supabase_client = supabase.create_client(supabase_url, supabase_key)

# # Initialize Gemini API
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# model = genai.GenerativeModel('gemini-2.0-flash')

# # Configure page
# st.set_page_config(
#     page_title="DocChat - RAG Chat Application",
#     page_icon="💬",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better UI
# st.markdown("""
# <style>
#     .main {
#         background-color: #f5f7f9;
#     }
#     .chat-message {
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin-bottom: 1rem;
#         display: flex;
#         flex-direction: column;
#     }
#     .chat-message.user {
#         background-color: #e6f3ff;
#         border-left: 5px solid #0078d4;
#     }
#     .chat-message.bot {
#         background-color: #f0f0f0;
#         border-left: 5px solid #424242;
#     }
#     .chat-message .message-content {
#         display: flex;
#         margin-top: 0.5rem;
#     }
#     .sidebar .element-container:has(.stButton) {
#         padding-top: 1rem;
#         padding-bottom: 1rem;
#     }
#     .document-card {
#         background-color: white;
#         border-radius: 0.5rem;
#         padding: 1rem;
#         margin-bottom: 1rem;
#         border: 1px solid #ddd;
#         transition: all 0.3s;
#     }
#     .document-card:hover {
#         box-shadow: 0 4px 8px rgba(0,0,0,0.1);
#     }
#     .document-title {
#         font-weight: 600;
#         margin-bottom: 0.5rem;
#     }
#     .document-source {
#         color: #666;
#         font-size: 0.8rem;
#     }
#     .document-date {
#         color: #999;
#         font-size: 0.7rem;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Authentication functions
# def initialize_auth_session():
#     """Initialize authentication session state variables"""
#     if 'authenticated' not in st.session_state:
#         st.session_state.authenticated = False
#     if 'user' not in st.session_state:
#         st.session_state.user = None
#     if 'access_token' not in st.session_state:
#         st.session_state.access_token = None

# def sign_up():
#     """Handle user sign up"""
#     with st.form("signup_form"):
#         email = st.text_input("Email")
#         password = st.text_input("Password", type="password")
#         name = st.text_input("Full Name")
#         submit = st.form_submit_button("Sign Up")
        
#         if submit and email and password:
#             try:
#                 response = supabase_client.auth.sign_up({
#                     "email": email,
#                     "password": password,
#                     "options": {
#                         "data": {
#                             "full_name": name
#                         }
#                     }
#                 })
#                 if response.user:
#                     st.success("Signed up successfully! Please check your email for verification.")
#                 else:
#                     st.error("Sign up failed.")
#             except Exception as e:
#                 st.error(f"Error signing up: {str(e)}")

# def sign_in():
#     """Handle user sign in"""
#     with st.form("signin_form"):
#         email = st.text_input("Email")
#         password = st.text_input("Password", type="password")
#         submit = st.form_submit_button("Sign In")
        
#         if submit and email and password:
#             try:
#                 response = supabase_client.auth.sign_in_with_password({
#                     "email": email,
#                     "password": password
#                 })
#                 if response.user:
#                     st.session_state.authenticated = True
#                     st.session_state.user = response.user
#                     st.session_state.access_token = response.session.access_token
#                     st.rerun()
#                 else:
#                     st.error("Sign in failed.")
#             except Exception as e:
#                 st.error(f"Error signing in: {str(e)}")

# def sign_out():
#     """Handle user sign out"""
#     supabase_client.auth.sign_out()
#     st.session_state.authenticated = False
#     st.session_state.user = None
#     st.session_state.access_token = None
#     st.rerun()

# # Document processing functions
# def extract_text_from_url(url: str) -> str:
#     """Extract text content from a URL"""
#     try:
#         response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
#         response.raise_for_status()
#         soup = BeautifulSoup(response.text, 'html.parser')
        
#         # Remove script and style elements
#         for script in soup(["script", "style"]):
#             script.extract()
        
#         # Get text and clean it
#         text = soup.get_text()
#         lines = (line.strip() for line in text.splitlines())
#         chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
#         text = '\n'.join(chunk for chunk in chunks if chunk)
        
#         return text
#     except Exception as e:
#         st.error(f"Error extracting content from URL: {str(e)}")
#         return ""

# def extract_text_from_pdf(pdf_file) -> str:
#     """Extract text content from a PDF file"""
#     try:
#         pdf_reader = PyPDF2.PdfReader(pdf_file)
#         text = ""
#         for page_num in range(len(pdf_reader.pages)):
#             page = pdf_reader.pages[page_num]
#             text += page.extract_text() + "\n"
#         return text
#     except Exception as e:
#         st.error(f"Error extracting content from PDF: {str(e)}")
#         return ""

# def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
#     """Split text into overlapping chunks"""
#     chunks = []
#     for i in range(0, len(text), chunk_size - overlap):
#         chunk = text[i:i + chunk_size]
#         if len(chunk) > 100:  # Only keep chunks with substantial content
#             chunks.append(chunk)
#     return chunks

# def generate_embeddings(text: str) -> List[float]:
#     """Generate text embeddings using Gemini's embedding feature"""
#     try:
#         # Since Gemini doesn't currently offer direct embedding API, we'll simulate it
#         # In a real application, you would use a dedicated embedding model 
#         # For demonstration purposes, we'll use a function that returns random embeddings
#         # In production, consider using models like OpenAI's text-embedding-ada-002 or similar
        
#         # This simulates a 1536-dimensional embedding vector
#         # The actual implementation would call the appropriate embedding API
#         return np.random.normal(0, 0.1, 1536).tolist()
#     except Exception as e:
#         st.error(f"Error generating embeddings: {str(e)}")
#         return []

# def store_document(user_id: str, source: str, title: str, content: str, doc_type: str) -> str:
#     """Store document and its embeddings in Supabase"""
#     try:
#         # Generate a unique ID for the document
#         doc_id = str(uuid.uuid4())
#         current_timestamp = datetime.now().isoformat()
        
#         # Store document metadata
#         supabase_client.table('documents').insert({
#             'id': doc_id,
#             'user_id': user_id,
#             'title': title,
#             'source': source,
#             'doc_type': doc_type,
#             'created_at': current_timestamp
#         }).execute()
        
#         # Process and store document chunks
#         chunks = chunk_text(content)
#         for i, chunk in enumerate(chunks):
#             chunk_id = f"{doc_id}_{i}"
#             embeddings = generate_embeddings(chunk)
            
#             # Store chunk and embeddings in the vector store
#             supabase_client.table('document_chunks').insert({
#                 'id': chunk_id,
#                 'document_id': doc_id,
#                 'content': chunk,
#                 'chunk_index': i,
#                 'embedding': embeddings
#             }).execute()
        
#         return doc_id
#     except Exception as e:
#         st.error(f"Error storing document: {str(e)}")
#         return ""

# def get_user_documents(user_id: str) -> List[Dict]:
#     """Get all documents for a user"""
#     try:
#         response = supabase_client.table('documents') \
#             .select('id, title, source, doc_type, created_at') \
#             .eq('user_id', user_id) \
#             .order('created_at', desc=True) \
#             .execute()
        
#         return response.data
#     except Exception as e:
#         st.error(f"Error retrieving documents: {str(e)}")
#         return []

# def delete_document(doc_id: str):
#     """Delete a document and its chunks"""
#     try:
#         # Delete document chunks first
#         supabase_client.table('document_chunks') \
#             .delete() \
#             .like('id', f"{doc_id}_%") \
#             .execute()
        
#         # Delete document metadata
#         supabase_client.table('documents') \
#             .delete() \
#             .eq('id', doc_id) \
#             .execute()
        
#         return True
#     except Exception as e:
#         st.error(f"Error deleting document: {str(e)}")
#         return False

# # Chat functions
# def create_chat(user_id: str, title: str) -> str:
#     """Create a new chat session"""
#     try:
#         chat_id = str(uuid.uuid4())
#         current_timestamp = datetime.now().isoformat()
        
#         supabase_client.table('chats').insert({
#             'id': chat_id,
#             'user_id': user_id,
#             'title': title,
#             'created_at': current_timestamp
#         }).execute()
        
#         return chat_id
#     except Exception as e:
#         st.error(f"Error creating chat: {str(e)}")
#         return ""

# def get_user_chats(user_id: str) -> List[Dict]:
#     """Get all chats for a user"""
#     try:
#         response = supabase_client.table('chats') \
#             .select('id, title, created_at') \
#             .eq('user_id', user_id) \
#             .order('created_at', desc=True) \
#             .execute()
        
#         return response.data
#     except Exception as e:
#         st.error(f"Error retrieving chats: {str(e)}")
#         return []

# # def get_chat_messages(chat_id: str) -> List[Dict]:
# #     """Get all messages for a chat"""
# #     try:
# #         response = supabase_client.table('messages') \
# #             .select('id, role, content, created_at') \
# #             .eq('chat_id', chat_id) \
# #             .order('created_at', asc=True) \
# #             .execute()
        
# #         return response.data
# #     except Exception as e:
# #         st.error(f"Error retrieving chat messages: {str(e)}")
# #         return []

# def get_chat_messages(chat_id: str) -> List[Dict]:
#     """Get all messages for a chat"""
#     try:
#         response = supabase_client.table('messages') \
#             .select('id, role, content, created_at') \
#             .eq('chat_id', chat_id) \
#             .order('created_at', desc=False) \
#             .execute()
        
#         chat_messages = response.data  # Store retrieved messages
#         st.write(f"Messages for chat {chat_id}: {chat_messages}")  # Debugging output

#         # Store in session state
#         st.session_state["messages"] = chat_messages if chat_messages else []

#         return chat_messages
#     except Exception as e:
#         st.error(f"Error retrieving chat messages: {str(e)}")
#         return []
    

# def store_message(chat_id: str, role: str, content: str) -> str:
#     """Store a chat message"""
#     try:
#         message_id = str(uuid.uuid4())
#         current_timestamp = datetime.now().isoformat()
        
#         supabase_client.table('messages').insert({
#             'id': message_id,
#             'chat_id': chat_id,
#             'role': role,
#             'content': content,
#             'created_at': current_timestamp
#         }).execute()
        
#         return message_id
#     except Exception as e:
#         st.error(f"Error storing message: {str(e)}")
#         return ""

# def search_relevant_chunks(query: str, limit: int = 5) -> List[Dict]:
#     """Search for relevant document chunks based on query"""
#     try:
#         # In a real implementation, you would:
#         # 1. Generate embeddings for the query
#         # 2. Perform a vector similarity search in Supabase
        
#         # For demonstration, we'll do a simple text search
#         # In production, you should use Supabase's vector search capabilities
#         query_embedding = generate_embeddings(query)
        
#         # This would be replaced with actual similarity search
#         # For now, we'll simulate it with a basic text search
#         response = supabase_client.table('document_chunks') \
#             .select('id, document_id, content, chunk_index') \
#             .limit(limit) \
#             .execute()
        
#         # In production, you would use pgvector's similarity search like:
#         # .rpc('match_documents', {'query_embedding': query_embedding, 'match_threshold': 0.7, 'match_count': limit})
        
#         return response.data
#     except Exception as e:
#         st.error(f"Error searching for relevant chunks: {str(e)}")
#         return []

# def generate_ai_response(query: str, context: List[str]) -> str:
#     """Generate a response from Gemini with context from relevant documents"""
#     try:
#         # Combine context chunks into a single context string
#         context_text = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context)])
        
#         # Create the prompt with RAG context
#         prompt = f"""I want you to answer the user's question based on the context provided.
# If the context doesn't contain relevant information, please let the user know you don't have enough information.
# Don't make up information that's not supported by the context.

# Context:
# {context_text}

# User Question: {query}

# Your Response:"""

#         # Generate response from Gemini
#         response = model.generate_content(prompt)
        
#         return response.text
#     except Exception as e:
#         st.error(f"Error generating AI response: {str(e)}")
#         return f"I apologize, but I encountered an error while processing your request. Error: {str(e)}"


# # UI Components
# def display_auth_ui():
#     """Display authentication UI"""
#     st.title("DocChat - Intelligent Document Chat")
    
#     tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
    
#     with tab1:
#         sign_in()
    
#     with tab2:
#         sign_up()

# def display_chat_message(message, is_user=False):
#     """Display a chat message"""
#     message_type = "user" if is_user else "bot"
#     with st.container():
#         st.markdown(f"""
#         <div class="chat-message {message_type}">
#             <div class="avatar">
#                 <strong>{"You" if is_user else "AI Assistant"}</strong>
#             </div>
#             <div class="message-content">
#                 {message}
#             </div>
#         </div>
#         """, unsafe_allow_html=True)

# def display_document_card(doc):
#     """Display a document card"""
#     doc_type_icon = "📄" if doc['doc_type'] == 'pdf' else "🔗"
#     created_at = datetime.fromisoformat(doc['created_at']).strftime("%b %d, %Y")
    
#     with st.container():
#         st.markdown(f"""
#         <div class="document-card">
#             <div class="document-title">{doc_type_icon} {doc['title']}</div>
#             <div class="document-source">{doc['source']}</div>
#             <div class="document-date">Added on {created_at}</div>
#         </div>
#         """, unsafe_allow_html=True)
        
#         col1, col2 = st.columns([4, 1])
#         with col2:
#             if st.button("Delete", key=f"delete_{doc['id']}", help="Delete this document"):
#                 if delete_document(doc['id']):
#                     st.rerun()

# def display_document_management():
#     """Display document management interface"""
#     st.header("Add New Document")
    
#     tab1, tab2 = st.tabs(["URL", "PDF Upload"])
    
#     with tab1:
#         with st.form("url_form"):
#             url = st.text_input("Website URL", key="url_input")
#             title = st.text_input("Document Title (Optional)", key="url_title")
#             submit_url = st.form_submit_button("Process URL")
            
#             if submit_url and url:
#                 with st.spinner("Processing URL..."):
#                     content = extract_text_from_url(url)
#                     if content:
#                         doc_title = title if title else url
#                         doc_id = store_document(
#                             st.session_state.user.id,
#                             url,
#                             doc_title,
#                             content,
#                             'url'
#                         )
#                         if doc_id:
#                             st.success(f"URL processed and stored successfully!")
#                             st.rerun()
    
#     with tab2:
#         with st.form("pdf_form"):
#             uploaded_file = st.file_uploader("Upload PDF", type="pdf", key="pdf_upload")
#             pdf_title = st.text_input("Document Title (Optional)", key="pdf_title")
#             submit_pdf = st.form_submit_button("Process PDF")
            
#             if submit_pdf and uploaded_file:
#                 with st.spinner("Processing PDF..."):
#                     content = extract_text_from_pdf(uploaded_file)
#                     if content:
#                         doc_title = pdf_title if pdf_title else uploaded_file.name
#                         doc_id = store_document(
#                             st.session_state.user.id,
#                             uploaded_file.name,
#                             doc_title,
#                             content,
#                             'pdf'
#                         )
#                         if doc_id:
#                             st.success(f"PDF processed and stored successfully!")
#                             st.rerun()
    
#     st.header("Your Documents")
#     documents = get_user_documents(st.session_state.user.id)
    
#     if not documents:
#         st.info("You don't have any documents yet. Add one using the form above.")
#     else:
#         for doc in documents:
#             display_document_card(doc)

# def initialize_chat_session():
#     """Initialize chat session state variables"""
#     if 'chat_id' not in st.session_state:
#         st.session_state.chat_id = None
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
#     if 'chats' not in st.session_state:
#         st.session_state.chats = []

# def display_chat_interface():
#     """Display the chat interface"""
    
#     # Fetch user chats only once
#     if "chats_loaded" not in st.session_state:
#         st.session_state.chats = get_user_chats(st.session_state.user.id)
#         st.session_state.chats_loaded = True  # Prevent repeated API calls

#     # Sidebar for chat selection and creation
#     with st.sidebar:
#         st.header("Your Chats")

#         # New chat button
#         if st.button("New Chat", use_container_width=True):
#             chat_title = f"Chat {len(st.session_state.chats) + 1}"
#             chat_id = create_chat(st.session_state.user.id, chat_title)
#             if chat_id:
#                 st.session_state.chat_id = chat_id
#                 st.session_state.messages = []
#                 st.session_state.chats.append({"id": chat_id, "title": chat_title})
#                 st.rerun()

#         # Display existing chats
#         if not st.session_state.chats:
#             st.info("You don't have any chats yet.")
#         else:
#             for chat in st.session_state.chats:
#                 chat_label = chat['title']
#                 if st.button(chat_label, key=f"chat_{chat['id']}", use_container_width=True):
#                     if st.session_state.chat_id != chat['id']:  # Prevent duplicate reload
#                         st.session_state.chat_id = chat['id']
#                         st.session_state.messages = get_chat_messages(chat['id'])
#                         st.rerun()

#     # Main chat area
#     st.header("Chat with Your Documents")

#     if st.session_state.chat_id is None:
#         st.info("Please select a chat or create a new one to start.")
#         return

#     # Display chat messages
#     for message in st.session_state.messages:
#         display_chat_message(message['content'], message['role'] == 'user')

#     # Chat input
#     user_input = st.text_input("Ask a question about your documents", key="user_input")

#     if user_input:
#         # Add user message to chat
#         store_message(st.session_state.chat_id, 'user', user_input)
#         st.session_state.messages.append({'role': 'user', 'content': user_input})

#         # Search for relevant context
#         relevant_chunks = search_relevant_chunks(user_input)
#         context = [chunk['content'] for chunk in relevant_chunks]

#         # Generate AI response
#         with st.spinner("Thinking..."):
#             ai_response = generate_ai_response(user_input, context)
#             store_message(st.session_state.chat_id, 'assistant', ai_response)
#             st.session_state.messages.append({'role': 'assistant', 'content': ai_response})

#         # ✅ **Fix: Remove input instead of modifying**
#         del st.session_state["user_input"]  # Clears the input field
#         st.rerun()



# def main():
#     # Initialize auth session state
#     initialize_auth_session()
    
#     # Check if user is authenticated
#     if not st.session_state.authenticated:
#         display_auth_ui()
#     else:
#         # Initialize chat session state
#         initialize_chat_session()
        
#         # Sidebar navigation
#         with st.sidebar:
#             st.title("DocChat")
#             st.image("https://via.placeholder.com/150x60?text=DocChat", width=150)
            
#             # Navigation
#             page = st.radio("Navigation", ["Chat", "Documents"])
            
#             # User info and sign out
#             st.divider()
#             st.write(f"Signed in as: {st.session_state.user.email}")
#             if st.button("Sign Out", use_container_width=True):
#                 sign_out()
        
#         # Display selected page
#         if page == "Chat":
#             display_chat_interface()
#         elif page == "Documents":
#             display_document_management()

# if __name__ == "__main__":
#     main()


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
    .chat-docs-section {
        background-color: #f9f9f9;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #eee;
    }
    .url-input-container {
        background-color: #f0f8ff;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #d1e3fa;
    }
    .pdf-input-container {
        background-color: #fff0f0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #fad1d1;
    }
    .selected-docs-container {
        background-color: #f0fff0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #d1fad1;
    }
    .document-view-container {
        background-color: #f5f5f5;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 1rem;
        border: 1px solid #ddd;
        max-height: 400px;
        overflow-y: auto;
    }
    .document-content {
        white-space: pre-wrap;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .view-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 5px 10px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 12px;
        margin: 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #ddd;
    }
    .sidebar-section {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
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
    """Generate text embeddings using an external API or service"""
    try:
        # Use Gemini's embedding API if available
        # For demonstration, we'll use a simulated embedding
        # In production, replace with an actual embedding API call
        
        # Simulate a 1536-dimensional embedding vector
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
        
        # Store document metadata and full content
        supabase_client.table('documents').insert({
            'id': doc_id,
            'user_id': user_id,
            'title': title,
            'source': source,
            'doc_type': doc_type,
            'full_content': content,  # Store the full content for viewing
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

def get_document_content(doc_id: str) -> str:
    """Get the full content of a document"""
    try:
        response = supabase_client.table('documents') \
            .select('full_content') \
            .eq('id', doc_id) \
            .execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]['full_content']
        return ""
    except Exception as e:
        st.error(f"Error retrieving document content: {str(e)}")
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
            .order('created_at', desc=False) \
            .execute()
        
        chat_messages = response.data
        
        # Store in session state
        st.session_state["messages"] = chat_messages if chat_messages else []

        return chat_messages
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

# Chat-document relationships
def link_documents_to_chat(chat_id: str, document_ids: List[str]):
    """Link documents to a chat session"""
    try:
        for doc_id in document_ids:
            # Check if the link already exists to avoid duplicates
            existing = supabase_client.table('chat_documents') \
                .select('*') \
                .eq('chat_id', chat_id) \
                .eq('document_id', doc_id) \
                .execute()
            
            if not existing.data:  # Only create if it doesn't exist
                supabase_client.table('chat_documents').insert({
                    'id': str(uuid.uuid4()),
                    'chat_id': chat_id,
                    'document_id': doc_id,
                    'created_at': datetime.now().isoformat()
                }).execute()
        
        return True
    except Exception as e:
        st.error(f"Error linking documents to chat: {str(e)}")
        return False

def get_chat_documents(chat_id: str) -> List[Dict]:
    """Get all documents linked to a chat"""
    try:
        # Join chat_documents with documents to get full document info
        response = supabase_client.table('chat_documents') \
            .select('document_id, documents(id, title, source, doc_type, created_at)') \
            .eq('chat_id', chat_id) \
            .execute()
        
        # Extract and format document data
        documents = []
        for item in response.data:
            if item['documents']:
                doc = item['documents']
                doc['id'] = item['document_id']  # Ensure we have the document ID
                documents.append(doc)
        
        return documents
    except Exception as e:
        st.error(f"Error retrieving chat documents: {str(e)}")
        return []

def search_relevant_chunks(query: str, document_ids: List[str] = None, limit: int = 5) -> List[Dict]:
    """Search for relevant document chunks based on query, optionally filtered by document IDs"""
    try:
        # Generate embeddings for the query
        query_embedding = generate_embeddings(query)
        
        # Start with the base query
        query_builder = supabase_client.table('document_chunks') \
            .select('id, document_id, content, chunk_index')
        
        # Filter by document IDs if provided
        if document_ids and len(document_ids) > 0:
            query_builder = query_builder.in_('document_id', document_ids)
        
        # Execute with limit
        response = query_builder.limit(limit).execute()
        
        # In production, use vector similarity search
        # For now, we're using a simple retrieval approach
        
        return response.data
    except Exception as e:
        st.error(f"Error searching for relevant chunks: {str(e)}")
        return []

def generate_ai_response(query: str, context: List[str]) -> str:
    """Generate a response from Gemini with context from relevant documents"""
    try:
        # Check if we have any context
        if not context or len(context) == 0:
            return "I don't have enough context to answer your question accurately. Please upload or link relevant documents first."
        
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

def display_document_card(doc, show_delete=True, select_callback=None, show_view=True):
    """Display a document card, optionally with selection and view capability"""
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
        
        cols = st.columns([3, 1, 1, 1]) if show_delete and show_view else st.columns([3, 1, 1])
        
        button_col = 2 if show_delete else 1
        
        if show_view:
            with cols[1]:
                if st.button("View", key=f"view_{doc['id']}", help="View document content"):
                    # Set the document to view in session state
                    st.session_state.viewing_doc_id = doc['id']
                    st.session_state.viewing_doc_title = doc['title']
                    st.rerun()
        
        if show_delete:
            with cols[button_col]:
                if st.button("Delete", key=f"delete_{doc['id']}", help="Delete this document"):
                    if delete_document(doc['id']):
                        st.rerun()
        
        if select_callback:
            with cols[0]:
                if st.button("Select", key=f"select_{doc['id']}", help="Select this document for chat"):
                    select_callback(doc['id'])

def display_document_viewer():
    """Display the document content viewer"""
    if 'viewing_doc_id' in st.session_state and st.session_state.viewing_doc_id:
        doc_id = st.session_state.viewing_doc_id
        doc_title = st.session_state.viewing_doc_title
        
        st.subheader(f"Viewing: {doc_title}")
        
        # Close button
        if st.button("Close", key="close_doc_viewer"):
            st.session_state.viewing_doc_id = None
            st.session_state.viewing_doc_title = None
            st.rerun()
        
        # Fetch and display document content
        content = get_document_content(doc_id)
        if content:
            st.markdown("<div class='document-view-container'><div class='document-content'>", unsafe_allow_html=True)
            st.text(content[:10000] + "..." if len(content) > 10000 else content)
            st.markdown("</div></div>", unsafe_allow_html=True)
        else:
            st.warning("Could not retrieve document content.")

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
    
    # Document viewer
    if 'viewing_doc_id' in st.session_state and st.session_state.viewing_doc_id:
        display_document_viewer()

def initialize_chat_session():
    """Initialize chat session state variables"""
    if 'chat_id' not in st.session_state:
        st.session_state.chat_id = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chats' not in st.session_state:
        st.session_state.chats = []
    if 'chat_documents' not in st.session_state:
        st.session_state.chat_documents = []
    if 'viewing_doc_id' not in st.session_state:
        st.session_state.viewing_doc_id = None
    if 'viewing_doc_title' not in st.session_state:
        st.session_state.viewing_doc_title = None

def process_url_for_chat(chat_id):
    """Process URL directly within a chat"""
    with st.form(key=f"chat_url_form_{chat_id}"):
        url = st.text_input("Website URL", key="chat_url_input")
        title = st.text_input("Document Title (Optional)", key="chat_url_title")
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
                        # Link to current chat
                        link_documents_to_chat(chat_id, [doc_id])
                        st.success(f"URL processed and added to this chat!")
                        # Update chat documents
                        st.session_state.chat_documents = get_chat_documents(chat_id)
                        st.rerun()

def process_pdf_for_chat(chat_id):
    """Process PDF directly within a chat"""
    with st.form(key=f"chat_pdf_form_{chat_id}"):
        uploaded_file = st.file_uploader("Upload PDF", type="pdf", key="chat_pdf_upload")
        pdf_title = st.text_input("Document Title (Optional)", key="chat_pdf_title")
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
                        # Link to current chat
                        link_documents_to_chat(chat_id, [doc_id])
                        st.success(f"PDF processed and added to this chat!")
                        # Update chat documents
                        st.session_state.chat_documents = get_chat_documents(chat_id)
                        st.rerun()

def select_documents_for_chat():
    """UI for selecting existing documents for the current chat"""
    if not st.session_state.chat_id:
        return
    
    # Get all user documents
    all_documents = get_user_documents(st.session_state.user.id)
    
    # Current chat documents
    current_chat_docs = st.session_state.chat_documents
    current_doc_ids = [doc['id'] for doc in current_chat_docs]
    
    # Filter out documents already in chat
    available_docs = [doc for doc in all_documents if doc['id'] not in current_doc_ids]
    
    if not available_docs:
        st.info("All your documents are already linked to this chat.")
        return
    
    st.subheader("Add Existing Documents to Chat")
    
    # Temporary storage for selected docs
    if 'temp_selected_docs' not in st.session_state:
        st.session_state.temp_selected_docs = []
    
    # Display available documents with checkboxes
    selected_docs = []
    for doc in available_docs:
        doc_type_icon = "📄" if doc['doc_type'] == 'pdf' else "🔗"
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"{doc_type_icon} **{doc['title']}** ({doc['source']})")
        with col2:
            if st.checkbox("", key=f"select_doc_{doc['id']}"):
                selected_docs.append(doc['id'])
    
    if selected_docs:
        if st.button("Add Selected Documents to Chat"):
            link_documents_to_chat(st.session_state.chat_id, selected_docs)
            st.session_state.chat_documents = get_chat_documents(st.session_state.chat_id)
            st.success(f"Added {len(selected_docs)} document(s) to this chat")
            st.rerun()

def display_chat_interface():
    """Display the main chat interface"""
    initialize_chat_session()
    
    # Sidebar for chat selection and management
    with st.sidebar:
        st.markdown('<div class="sidebar-title">DocChat</div>', unsafe_allow_html=True)
        
        # User information
        st.write(f"Logged in as: **{st.session_state.user.email}**")
        if st.button("Sign Out"):
            sign_out()
        
        st.markdown('<div class="sidebar-section">Chats</div>', unsafe_allow_html=True)
        
        # Create new chat
        new_chat_title = st.text_input("New Chat Title", placeholder="Enter chat title...")
        if st.button("Create New Chat"):
            if new_chat_title:
                chat_id = create_chat(st.session_state.user.id, new_chat_title)
                if chat_id:
                    st.session_state.chat_id = chat_id
                    st.session_state.messages = []
                    st.session_state.chat_documents = []
                    st.rerun()
            else:
                st.warning("Please enter a title for the new chat")
        
        # Display existing chats
        st.markdown('<div class="sidebar-section">Your Chats</div>', unsafe_allow_html=True)
        chats = get_user_chats(st.session_state.user.id)
        st.session_state.chats = chats
        
        if not chats:
            st.info("No chats yet. Create a new one!")
        else:
            for chat in chats:
                chat_date = datetime.fromisoformat(chat['created_at']).strftime("%b %d")
                if st.button(f"{chat['title']} ({chat_date})", key=f"chat_{chat['id']}"):
                    st.session_state.chat_id = chat['id']
                    # Load messages and linked documents
                    st.session_state.messages = get_chat_messages(chat['id'])
                    st.session_state.chat_documents = get_chat_documents(chat['id'])
                    st.rerun()
        
        # Document management link
        st.markdown('<div class="sidebar-section">Document Management</div>', unsafe_allow_html=True)
        if st.button("Manage All Documents"):
            st.session_state.page = "document_management"
            st.rerun()
    
    # Main chat area
    if not st.session_state.chat_id:
        st.title("DocChat - Document Powered Conversations")
        st.write("Select or create a chat to get started.")
        st.write("DocChat allows you to have intelligent conversations with your documents.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **Features**:
            - Upload PDF documents
            - Import content from web URLs
            - Chat with your documents
            """)
        with col2:
            st.markdown("""
            **How it works**:
            1. Add documents to your library
            2. Create a new chat
            3. Link documents to your chat
            4. Start asking questions!
            """)
        with col3:
            st.markdown("""
            **Use cases**:
            - Research assistance
            - Document Q&A
            - Information extraction
            - Learning & study aid
            """)
    else:
        # Get current chat information
        current_chat = next((c for c in st.session_state.chats if c['id'] == st.session_state.chat_id), None)
        if current_chat:
            st.title(f"Chat: {current_chat['title']}")
            
            # Chat document management
            with st.expander("Documents for this chat", expanded=False):
                tab1, tab2, tab3 = st.tabs(["Linked Documents", "Add URL", "Add PDF"])
                
                with tab1:
                    chat_docs = st.session_state.chat_documents
                    if not chat_docs:
                        st.info("No documents linked to this chat. Add documents to enhance responses.")
                    else:
                        for doc in chat_docs:
                            display_document_card(doc, show_delete=False, show_view=True)
                    
                    # Button to add existing documents
                    if st.button("Add Existing Documents"):
                        st.session_state.showing_doc_selector = True
                        st.rerun()
                    
                    # Document selector
                    if st.session_state.get('showing_doc_selector', False):
                        select_documents_for_chat()
                        if st.button("Cancel"):
                            st.session_state.showing_doc_selector = False
                            st.rerun()
                    
                with tab2:
                    process_url_for_chat(st.session_state.chat_id)
                
                with tab3:
                    process_pdf_for_chat(st.session_state.chat_id)
            
            # Display chat messages
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.messages:
                    display_chat_message(message['content'], message['role'] == 'user')
            
            # Input for new message
            with st.form(key="chat_input_form", clear_on_submit=True):
                user_input = st.text_area("Your message:", key="user_input", height=100)
                submit_button = st.form_submit_button("Send")
                
                if submit_button and user_input:
                    # Store user message
                    store_message(st.session_state.chat_id, "user", user_input)
                    
                    # Get document IDs for this chat to use in search
                    doc_ids = [doc['id'] for doc in st.session_state.chat_documents]
                    
                    # Search for relevant chunks
                    relevant_chunks = search_relevant_chunks(user_input, doc_ids, limit=5)
                    chunk_contents = [chunk['content'] for chunk in relevant_chunks]
                    
                    # Generate AI response with context
                    with st.spinner("Thinking..."):
                        ai_response = generate_ai_response(user_input, chunk_contents)
                        # Store AI response
                        store_message(st.session_state.chat_id, "assistant", ai_response)
                    
                    # Refresh messages
                    st.session_state.messages = get_chat_messages(st.session_state.chat_id)
                    st.rerun()

# Main application
def main():
    # Initialize session state for authentication
    initialize_auth_session()
    
    # Initialize page selection
    if 'page' not in st.session_state:
        st.session_state.page = "chat"
    
    # Authentication check
    if not st.session_state.authenticated:
        display_auth_ui()
    else:
        # Page routing
        if st.session_state.page == "document_management":
            display_document_management()
            if st.sidebar.button("Back to Chat"):
                st.session_state.page = "chat"
                st.rerun()
        else:
            display_chat_interface()

if __name__ == "__main__":
    main()