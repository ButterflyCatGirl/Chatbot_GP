# Fixed app.py
import streamlit as st
from PIL import Image
import torch
from transformers import (
    BlipProcessor, 
    BlipForQuestionAnswering,
    MarianTokenizer,
    MarianMTModel
)
import logging
import time
import gc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models with better memory management
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        # Check GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load BLIP processor and model with better error handling
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        
        # Try custom model first, fallback to base
        try:
            model = BlipForQuestionAnswering.from_pretrained(
                "Salesforce/blip-vqa-base",  # Use base model for stability
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
        except:
            model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        
        model = model.to(device)
        logger.info(f"BLIP model loaded successfully on {device}")
        
        # Load smaller translation models to avoid memory issues
        ar_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
        ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en").to(device)
        en_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
        en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar").to(device)
        
        return processor, model, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model, device
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        st.error(f"Model loading error: {str(e)}")
        st.stop()

# Improved translation with better error handling
def translate_text(text, tokenizer, model, device):
    try:
        if not text.strip():
            return text
            
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        return text

# Enhanced processing function
def process_medical_query(image, question, processor, model, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model, device):
    try:
        # Clear GPU cache
        if device == "cuda":
            torch.cuda.empty_cache()
            
        # Detect language
        is_arabic = any('\u0600' <= c <= '\u06FF' for c in question)
        
        if is_arabic:
            question_ar = question.strip()
            question_en = translate_text(question_ar, ar_en_tokenizer, ar_en_model, device)
        else:
            question_en = question.strip()
            question_ar = translate_text(question_en, en_ar_tokenizer, en_ar_model, device)
        
        # Resize image if too large
        if image.size[0] > 1024 or image.size[1] > 1024:
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Process with BLIP model
        inputs = processor(image, question_en, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50, num_beams=4, early_stopping=True)
        
        answer_en = processor.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Translate answer
        answer_ar = translate_text(answer_en, en_ar_tokenizer, en_ar_model, device)
        
        # Clean up GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()
            
        return question_ar, question_en, answer_ar, answer_en
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return f"Error: {str(e)}", "", "", ""

# Add memory cleanup
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Rest of your UI code remains the same, just update the model loading call:
def main():
    # ... your existing UI code ...
    
    # Initialize models (updated call)
    with st.spinner("Loading medical AI models... Please wait"):
        try:
            processor, model, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model, device = load_models()
            st.success(f"Medical AI models loaded successfully on {device}!")
        except Exception as e:
            st.error(f"Model loading error: {str(e)}")
            st.stop()
    
    # ... rest of your existing code ...
