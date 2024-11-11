# utils.py
import os
import logging
import re
import pdfplumber
import csv
import pandas as pd
from langchain_core.documents import Document
from io import StringIO, BytesIO
from PyPDF2 import PdfReader
from pptx import Presentation
from docx import Document as DocxDocument
from pptx.enum.shapes import MSO_SHAPE_TYPE
from PIL import Image
import pytesseract
import cv2
import numpy as np
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# Initialize OCR components
try:
    reader = easyocr.Reader(['en'])
    trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trocr_model.to(device)
except Exception as e:
    logging.error(f"Error initializing OCR components: {str(e)}")

def process_image(image_file):
    """Process image files and extract text using multiple OCR engines."""
    # ... (existing image processing code)

def extract_tables_from_image(gray_image):
    """Extract tables from images using OpenCV."""
    # ... (existing table extraction code)

def extract_formulas_from_image(gray_image):
    """Extract mathematical formulas from images."""
    # ... (existing formula extraction code)

def get_pdf_text(pdf_docs):
    """Extract text and tables from PDF documents."""
    # ... (existing PDF processing code)

def get_non_table_pdf_text(pdf_docs):
    """Extract non-tabular text from PDF documents."""
    # ... (existing non-tabular PDF processing code)

def get_csv_text(csv_file):
    """Extract text from CSV files."""
    # ... (existing CSV processing code)

def get_excel_text(excel_files):
    """Extract text from Excel files."""
    # ... (existing Excel processing code)

def get_ppt_text(ppt_files):
    """Extract text from PowerPoint files."""
    # ... (existing PowerPoint processing code)

def get_word_text(word_files):
    """Extract text from Word documents."""
    # ... (existing Word document processing code)

def extract_file_content(uploaded_file):
    """Extract text content from various file types."""
    file_type = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_type in {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}:
            return process_image(uploaded_file)
        elif file_type == 'pdf':
            return get_pdf_text([uploaded_file])
        elif file_type == 'csv':
            return get_csv_text(uploaded_file), [Document(page_content=get_csv_text(uploaded_file), metadata={'source': 'csv', 'filename': uploaded_file.name})]
        elif file_type in {'xls', 'xlsx'}:
            return get_excel_text([uploaded_file])
        elif file_type in {'pptx'}:
            return get_ppt_text([uploaded_file])
        elif file_type in {'doc', 'docx'}:
            return get_word_text([uploaded_file])
        else:
            error_msg = f"Unsupported file format: {file_type}"
            logging.error(error_msg)
            return "", []
    except Exception as e:
        error_msg = f"Error processing file {uploaded_file.name}: {str(e)}"
        logging.error(error_msg)
        return "", []

def create_prompt():
    """Create the prompt for the QA chain."""
    prompt_template = """You are a polite, respectful, and efficient AI assistant.

    IF the user's message matches ANY of these patterns:
    - "Hi", "Hello", "Hey", "Hii", "Hola" (just greeting)
    - "My name is [any name]"
    - "I am [any name]"
    - "[any greeting] my name is [any name]"
    - "[any greeting] I am [any name]"
    THEN respond only with: "Hello! How can I assist you today?"

    OTHERWISE:
    1. Use only the provided information:
    - Context: {context}
    - Chat History: {chat_history}
    - Current Question: {question}

    2. Your response must be:
    - Direct and to-the-point
    - Based only on given context and history
    - Without any explanations about your capabilities
    - Without mentioning sources or references

    3. If the answer cannot be found in context or history:
    Response should be only: "I don't have enough information to answer this question."

    4. Never start responses with:
    - "Based on..."
    - "According to..."
    - "I understand..."
    - "Let me..."

    5. Never end responses with:
    - "Is there anything else..."
    - "Let me know if..."
    - "Feel free to..."

    6. If the user uses any abusive or inappropriate language, respond politely and avoid escalation:
    "I apologize, but I don't engage with that type of language. How else can I assist you today?"

    Question: {question}"""
    return PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "chat_history"]
    )