import pdfplumber
import csv
import pandas as pd
from langchain_core.documents import Document
from io import StringIO, BytesIO
from PyPDF2 import PdfReader
from pptx import Presentation
from docx import Document as DocxDocument
from pptx.enum.shapes import MSO_SHAPE_TYPE
import re
import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import logging

# Initialize OCR components
try:
    reader = easyocr.Reader(['en'])
    trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    
    # Move models to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trocr_model.to(device)
except Exception as e:
    logging.error(f"Error initializing OCR components: {str(e)}")
    st.error("Error initializing OCR components. Some image processing features may be limited.")

def process_image(image_file):
    """
    Process image files and extract text using multiple OCR engines.
    """
    # Implement your process_image logic here

def extract_tables_from_image(gray_image):
    """
    Extract tables from images using OpenCV.
    """
    # Implement your extract_tables_from_image logic here

def extract_formulas_from_image(gray_image):
    """
    Extract mathematical formulas from images.
    """
    # Implement your extract_formulas_from_image logic here

def get_pdf_text(pdf_docs):
    """Extract text and tables from PDF documents."""
    # Implement your get_pdf_text logic here

def get_non_table_pdf_text(pdf_docs):
    """Extract non-tabular text from PDF documents."""
    # Implement your get_non_table_pdf_text logic here

def get_csv_text(csv_file):
    """Extract text from CSV files."""
    # Implement your get_csv_text logic here

def get_excel_text(excel_files):
    """Extract text from Excel files."""
    # Implement your get_excel_text logic here

def get_ppt_text(ppt_files):
    """Extract text from PowerPoint files."""
    # Implement your get_ppt_text logic here

def get_word_text(word_files):
    """Extract text from Word documents."""
    # Implement your get_word_text logic here

def init_session_state():
    """Initialize session state variables"""
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'user_context' not in st.session_state:
        st.session_state.user_context = {}
    if 'vector_store_path' not in st.session_state:
        st.session_state.vector_store_path = "faiss_index"

def process_documents(uploaded_files):
    """
    Process uploaded documents and create vector store
    
    Args:
        uploaded_files: List of uploaded file objects
        
    Returns:
        bool: True if processing successful, False otherwise
    """
    try:
        all_text = ""
        all_docs = []
        
        # Process each uploaded file
        for uploaded_file in uploaded_files:
            logging.info(f"Processing file: {uploaded_file.name}")
            text, docs = extract_file_content(uploaded_file)
            if text and docs:
                all_text += text + "\n\n"
                all_docs.extend(docs)
        
        if not all_text.strip():
            logging.warning("No text content extracted from uploaded files")
            return False
            
        # Create text chunks with metadata
        text_chunks = []
        metadata_chunks = []
        
        for doc in all_docs:
            chunks, meta_chunks = get_text_chunks(doc.page_content, doc.metadata)
            text_chunks.extend(chunks)
            metadata_chunks.extend(meta_chunks)
        
        if not text_chunks:
            logging.warning("No text chunks created from documents")
            return False
            
        # Create and save vector store
        vector_store = get_vector_store(text_chunks, metadata_chunks)
        if vector_store is None:
            logging.error("Failed to create vector store")
            return False
            
        logging.info("Documents processed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error in process_documents: {str(e)}", exc_info=True)
        return False