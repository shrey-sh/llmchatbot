import os
import streamlit as st
import requests
import pickle
import threading
import json  # Added missing import
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import Docx2txtLoader
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import

class ThreadSafeSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

class DocumentEmbeddingManager(ThreadSafeSingleton):
    def __init__(self, document_path='IGL FAQs.docx'):
        self.document_path = document_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"  # Corrected model name
        )
        self.vectorstore = None
        self.embedding_lock = threading.Lock()
        self.load_or_create_embedding()

    def load_or_create_embedding(self):
        embedding_cache_path = 'document_embedding_cache.pkl'
        
        if os.path.exists(embedding_cache_path):
            try:
                with open(embedding_cache_path, 'rb') as f:
                    self.vectorstore = pickle.load(f)
                print("Loaded existing embedding")
                return
            except Exception as e:
                print(f"Error loading cached embedding: {e}")
        
        with self.embedding_lock:
            # Use Docx2txtLoader for DOCX files
            loader = Docx2txtLoader(self.document_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=350,
                chunk_overlap=50
            )
            texts = text_splitter.split_documents(documents)
            
            self.vectorstore = FAISS.from_documents(texts, self.embeddings)
            
            with open(embedding_cache_path, 'wb') as f:
                pickle.dump(self.vectorstore, f)
            
            print("Created new embedding and cached")

    def retrieve_context(self, query, k=3):
        if not self.vectorstore:
            return ""
        
        retrieval_results = self.vectorstore.similarity_search(query, k=k)
        contexts = [doc.page_content for doc in retrieval_results]
        return " ".join(contexts)

class DeepInfraChatbot:
    def __init__(self, embedding_manager, deepinfra_api_key):
        self.embedding_manager = embedding_manager
        self.deepinfra_api_key = deepinfra_api_key
        self.executor = ThreadPoolExecutor(max_workers=5)

    def is_query_relevant(self, query):
        try:
            retrieval_results = self.embedding_manager.vectorstore.similarity_search(query, k=2)
            return len(retrieval_results) > 0
        except Exception:
            return False

    def generate_response(self, query):
        if not self.is_query_relevant(query):
            yield "I can only answer questions related to the document."
            return
        
        context = self.embedding_manager.retrieve_context(query)
        
        full_prompt = f"""
<SYSTEM>
You are a friendly customer support chatbot for Indraprastha Gas Limited (IGL). Follow these rules:

1. Greetings & General Chat:
- If user says hello/goodbye/thanks, respond politely but briefly
- Example: "Hello! How can I assist you with IGL services today?"

2. Document-Based Queries:
- Strictly use only the following context to answer questions
- Never make up answers
- If answer isn't in context, say: "This information isn't available in our documents. Please contact IGL customer care for assistance."

3. Multi-Turn Conversations:
- Maintain conversation context naturally
- Acknowledge previous questions where relevant
- Keep responses concise

</SYSTEM>

<CONTEXT>
{context}
</CONTEXT>

<User Question>
{query}
</User Question>

<Response Rules>
- If greeting: respond with 1 short sentence
- If document question: provide exact details from context
- If unclear/out-of-scope: use fallback response
- Never mention "context" or "document" to users
</Response Rules>
"""
        
        headers = {
            "Authorization": f"Bearer {self.deepinfra_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mistralai/Mistral-7B-Instruct-v0.1",
            "messages": [{"role": "user", "content": full_prompt}],
            "stream": True
        }
        
        response = ""
        try:
            with requests.post(
                "https://api.deepinfra.com/v1/openai/chat/completions", 
                headers=headers, 
                json=payload, 
                stream=True
            ) as r:
                for line in r.iter_lines():
                    if line:
                        # DeepInfra streaming response parsing
                        if line.startswith(b'data: '):
                            chunk_str = line.decode('utf-8')[1:]
                            try:
                                chunk_data = json.loads(chunk_str)
                                if 'choices' in chunk_data and chunk_data['choices']:
                                    delta = chunk_data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        chunk_content = delta['content']
                                        response += chunk_content
                                        yield response
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            yield f"An error occurred: {str(e)}"

def main():
    st.set_page_config(page_title="Document Chatbot", page_icon="📄")
    
    # Get DeepInfra API key from environment or Streamlit secrets
    deepinfra_api_key = "X0qpHSxphLKoz3vBdXgOaKBgdPbXVcRq"
    
    if not deepinfra_api_key:
        st.error("Please set DEEPINFRA_API_KEY in Streamlit secrets or environment variables")
        return
    
    embedding_manager = DocumentEmbeddingManager()
    
    chatbot = DeepInfraChatbot(embedding_manager, deepinfra_api_key)
    
    st.title("📄 Document Intelligence Chatbot")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Have a question about the document? Ask here!"):
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            for chunk in chatbot.generate_response(prompt):
                full_response = chunk
                response_placeholder.markdown(full_response)
            
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

if __name__ == "__main__":
    main()
