import os
import subprocess

# Install dependencies directly
dependencies = [
    "torch==2.3.0",
    "torchvision==0.18.0",
    "transformers==4.41.2",
    "Pillow==10.3.0",
    "accelerate==0.31.0",
    "sentencepiece==0.2.0",
    "protobuf==3.20.3"
]

for package in dependencies:
    try:
        __import__(package.split('==')[0])
    except ImportError:
        subprocess.check_call(["pip", "install", package])

# Now the rest of your imports
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


# Setup logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Load models with enhanced error handling
@st.cache_resource(show_spinner=False)
def load_models():
    # Load BLIP processor (always needed)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    
    # Load BLIP model with fallback
    try:
        model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base",  # Using base model for reliability
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        logger.info("BLIP model loaded successfully")
    except Exception as e:
        logger.exception("BLIP model loading failed")
        st.error(f"Critical error: {str(e)}")
        st.stop()
    
    # Load translation models
    try:
        ar_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
        ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
        en_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
        en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
        logger.info("Translation models loaded successfully")
    except Exception as e:
        logger.exception("Translation model loading failed")
        st.error(f"Translation model error: {str(e)}")
        st.stop()
        
    return processor, model, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model

# Translation functions
def translate_ar_to_en(text, tokenizer, model):
    try:
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs)
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception:
        return text  # Return original if translation fails

def translate_en_to_ar(text, tokenizer, model):
    try:
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs)
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception:
        return text  # Return original if translation fails

# Medical term dictionary
medical_terms = {
    "chest x-ray": "أشعة سينية للصدر",
    "x-ray": "أشعة سينية",
    "ct scan": "تصوير مقطعي محوسب",
    "mri": "تصوير بالرنين المغناطيسي",
    "ultrasound": "تصوير بالموجات فوق الصوتية",
    "normal": "طبيعي",
    "abnormal": "غير طبيعي",
    "brain": "الدماغ",
    "fracture": "كسر",
    "no abnormality detected": "لا توجد شذوذات",
    "left lung": "الرئة اليسرى",
    "right lung": "الرئة اليمنى",
    "pneumonia": "التهاب رئوي",
    "tumor": "ورم",
    "cancer": "سرطان",
    "infection": "عدوى",
    "heart": "القلب",
    "liver": "الكبد",
    "kidney": "الكلى",
    "bone": "العظم",
    "blood vessel": "وعاء دموي",
    "artery": "شريان",
    "vein": "وريد",
    "benign": "حميد",
    "malignant": "خبيث"
}

def translate_answer_medical(answer_en, en_ar_tokenizer, en_ar_model):
    key = answer_en.lower().strip()
    # First try exact match
    if key in medical_terms:
        return medical_terms[key]
    
    # Then try partial match
    for term, translation in medical_terms.items():
        if term in key:
            return translation
    
    # Finally, use machine translation
    return translate_en_to_ar(answer_en, en_ar_tokenizer, en_ar_model)

# Main processing function
def process_medical_query(image, question, processor, model, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model):
    try:
        # Detect language
        is_arabic = any('\u0600' <= c <= '\u06FF' for c in question)
        
        if is_arabic:
            question_ar = question.strip()
            question_en = translate_ar_to_en(question_ar, ar_en_tokenizer, ar_en_model)
        else:
            question_en = question.strip()
            question_ar = translate_en_to_ar(question_en, en_ar_tokenizer, en_ar_model)
        
        # Process with BLIP model
        inputs = processor(image, question_en, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs)
        
        answer_en = processor.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Translate answer
        answer_ar = translate_answer_medical(answer_en, en_ar_tokenizer, en_ar_model)
        
        return question_ar, question_en, answer_ar, answer_en
        
    except Exception as e:
        logger.exception("Processing failed")
        return f"Error: {str(e)}", "", "", ""

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Medical VQA Chatbot",
        layout="wide",
        page_icon="🩺",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .stApp {
            background-color: #f0f7ff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .header {
            background: linear-gradient(90deg, #0d47a1 0%, #1976d2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 0 0 20px 20px;
            margin-bottom: 2rem;
        }
        .card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
        }
        .result-box {
            background: #e3f2fd;
            border-radius: 12px;
            padding: 1.2rem;
            margin: 1rem 0;
            border-left: 4px solid #1976d2;
        }
        .footer {
            text-align: center;
            padding: 1.5rem;
            color: #666;
            margin-top: 2rem;
        }
        .stButton>button {
            background: #0d47a1;
            color: white;
            border-radius: 8px;
            padding: 0.8rem 1.5rem;
            font-weight: 500;
            transition: all 0.3s;
            border: none;
            width: 100%;
        }
        .stButton>button:hover {
            background: #1565c0;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="header">
        <h1 style="text-align:center;margin:0;">🩺 Medical VQA Chatbot</h1>
        <p style="text-align:center;margin:0;">Upload medical images and ask questions in Arabic or English</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize models
    with st.spinner("Loading medical AI models... Please wait"):
        try:
            processor, model, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model = load_models()
            st.success("Medical AI models loaded successfully!")
        except Exception as e:
            st.error(f"Model loading error: {str(e)}")
            st.stop()
    
    # Create two columns
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("<div style='font-size:1.2rem;font-weight:bold;margin-bottom:1rem;'>Upload Medical Image</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(" ", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
    
    with col2:
        st.markdown("<div style='font-size:1.2rem;font-weight:bold;margin-bottom:1rem;'>Ask Your Question</div>", unsafe_allow_html=True)
        question = st.text_area(" ", placeholder="Type your question in Arabic or English...", height=150, label_visibility="collapsed")
        
        if st.button("Get Medical Analysis", use_container_width=True):
            if not uploaded_file:
                st.warning("Please upload an image")
            elif not question.strip():
                st.warning("Please enter a question")
            else:
                with st.spinner("Analyzing medical image..."):
                    try:
                        start_time = time.time()
                        image_pil = Image.open(uploaded_file).convert("RGB")
                        q_ar, q_en, a_ar, a_en = process_medical_query(
                            image_pil, question, processor, model, 
                            ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model
                        )
                        processing_time = time.time() - start_time
                        
                        st.divider()
                        st.markdown("<div style='font-size:1.3rem;font-weight:bold;margin-bottom:1rem;'>Analysis Results</div>", unsafe_allow_html=True)
                        
                        # Display results in two columns
                        col_res1, col_res2 = st.columns(2)
                        
                        with col_res1:
                            st.markdown("**Arabic Question:**")
                            st.info(q_ar)
                            
                            st.markdown("**Arabic Answer:**")
                            st.success(a_ar)
                        
                        with col_res2:
                            st.markdown("**English Question:**")
                            st.info(q_en)
                            
                            st.markdown("**English Answer:**")
                            st.success(a_en)
                        
                        st.caption(f"Processed in {processing_time:.2f} seconds")
                        
                    except Exception as e:
                        st.error(f"Processing error: {str(e)}")
    
    # Footer
    st.divider()
    st.markdown("""
    <div class="footer">
        <p><strong>Medical VQA System</strong> • Powered by BLIP medical AI</p>
        <p>Note: This AI assistant provides preliminary medical insights. Always consult a healthcare professional for diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
