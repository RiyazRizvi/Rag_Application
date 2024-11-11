from utils import extract_file_content, get_text_chunks, get_vector_store
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import re
import logging
from pymongo import MongoClient

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['rag_app']
faiss_collection = db['faiss_index']

def handle_conversation(text: str) -> str:
    """Handle conversational inputs that don't require document knowledge"""
    # Implement your conversation handling logic here

def handle_user_input(user_question: str):
    """Handle user input with enhanced error handling, logging, and better conversation handling"""
    # Implement your handle_user_input logic here

def is_conversational_input(text: str) -> bool:
    """Determine if the input is conversational rather than a question about documents"""
    # Implement your conversational input detection logic here

def create_qa_chain():
    """Create the QA chain using the FAISS index stored in MongoDB"""
    try:
        prompt_template = """You are a polite, respectful, and efficient AI assistant."""
        # Add more prompt template content here

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "chat_history"]
        )

        llm = Ollama(model="llama3.1", temperature=0.1)

        # Retrieve the FAISS index from MongoDB
        faiss_index = FAISS.load_local("faiss_index", HuggingFaceEmbeddings(), allow_dangerous_deserialization=True)

        retriever = faiss_index.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=st.session_state.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

        return qa_chain
    except Exception as e:
        logging.error(f"Error in create_qa_chain: {str(e)}", exc_info=True)
        return None

def clear_conversation():
    """Clear only the conversation history without affecting processed documents"""
    st.session_state.memory.clear()
    st.session_state.conversation = []
    st.success("Conversation cleared! You can continue asking questions about the processed documents.")