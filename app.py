# Install required packages
!pip install -q streamlit transformers torch Pillow pyngrok huggingface_hub
!npm install -g localtunnel

# Write the Streamlit app to a file
import os
if not os.path.exists('app.py'):
    with open('app.py', 'w') as f:
        f.write("""
import streamlit as st
from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    MarianMTModel, 
    MarianTokenizer
)
import time
import base64
import requests
from io import BytesIO
from huggingface_hub import hf_hub_download
import os

# Page configuration
st.set_page_config(
    page_title="Medical VQA - Your Custom BLIP Model",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional medical UI
st.markdown(\"""
<style>
    .stApp {
        background: linear-gradient(135deg, #f0f7ff 0%, #e6f7ff 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        background: linear-gradient(90deg, #0d47a1 0%, #1976d2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        text-align: center;
    }
    .card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0d47a1;
        margin-bottom: 1rem;
        text-align: center;
    }
    .result-box {
        background: #e3f2fd;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        border-left: 4px solid #1976d2;
    }
    .lang-tag {
        display: inline-block;
        background: #1976d2;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin-right: 0.5rem;
    }
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #666;
        font-size: 0.9rem;
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
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background: #1565c0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .model-badge {
        display: inline-block;
        background: #4caf50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    .spinner-container {
        display: flex;
        justify-content: center;
        margin: 2rem 0;
    }
    .history-item {
        transition: all 0.3s;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        background: #f8fbff;
        border: 1px solid #e0e0e0;
    }
    .history-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    .error-box {
        background: #ffebee;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        border-left: 4px solid #f44336;
    }
</style>
\""", unsafe_allow_html=True)

# Load models with caching
@st.cache_resource(show_spinner=False)
def load_models():
    \"\"\"Load all required models with robust error handling\"\"\"
    progress_text = st.empty()
    
    # Try to load BLIP model
    try:
        progress_text.markdown("<div class='spinner-container'><div class='model-badge'>Loading ButterflyCatGirl/Blip-Streamlit-chatbot</div></div>", unsafe_allow_html=True)
        
        # Attempt to download model file directly
        model_path = hf_hub_download(
            repo_id="ButterflyCatGirl/Blip-Streamlit-chatbot",
            filename="pytorch_model.bin",
            cache_dir="model_cache"
        )
        
        # Load processor from base model
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Load model with custom weights
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            state_dict=torch.load(model_path, map_location=torch.device('cpu'))
        
        st.success("Custom BLIP model loaded successfully from direct download!")
        return processor, model
        
    except Exception as e:
        st.warning(f"Custom model loading failed: {str(e)}. Using base BLIP model instead.")
        try:
            # Fallback to base model
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            return processor, model
        except Exception as e2:
            st.error(f"Base model loading failed: {str(e2)}")
            return None, None

# Load translation models
@st.cache_resource(show_spinner=False)
def load_translation_models():
    \"\"\"Load translation models separately\"\"\"
    try:
        ar_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
        ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
        en_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
        en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
        return ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model
    except Exception as e:
        st.error(f"Translation model loading failed: {str(e)}")
        return None, None, None, None

# Translation functions
def translate_ar_to_en(text, tokenizer, model):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs)
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except:
        return text  # Return original text if translation fails

def translate_en_to_ar(text, tokenizer, model):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs)
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except:
        return text  # Return original text if translation fails

# Medical term dictionary - expanded for medical use cases
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
    "pneumonia": "الالتهاب الرئوي",
    "tumor": "ورم",
    "cancer": "سرطان",
    "infection": "عدوى",
    "inflammation": "التهاب",
    "heart": "القلب",
    "liver": "الكبد",
    "kidney": "الكلى",
    "bone": "العظم",
    "blood vessel": "وعاء دموي",
    "artery": "شريان",
    "vein": "وريد",
    "benign": "حميد",
    "malignant": "خبيث",
    "metastasis": "انتشار سرطاني",
    "lesion": "آفة",
    "nodule": "عقيدة",
    "mass": "كتلة",
    "swelling": "تورم",
    "bleeding": "نزيف",
    "clot": "جلطة",
    "embolism": "انصمام",
    "cardiomegaly": "تضخم القلب",
    "pneumothorax": "استرواح الصدر",
    "edema": "وذمة",
    "consolidation": "تصلب"
}

def translate_answer_medical(answer_en, en_ar_tokenizer, en_ar_model):
    \"\"\"Smart translation of medical answers\"\"\"
    # First try exact match
    key = answer_en.lower().strip()
    if key in medical_terms:
        return medical_terms[key]
    
    # Then try partial match
    for term, translation in medical_terms.items():
        if term in key:
            return translation
    
    # Finally, use machine translation
    return translate_en_to_ar(answer_en, en_ar_tokenizer, en_ar_model)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# App header
st.markdown(\"""
<div class="header">
    <div class="title">Medical Visual Question Answering</div>
    <div class="subtitle">Powered by your custom BLIP model</div>
    <div style="text-align:center; margin-top:1rem;">
        <span class="model-badge">ButterflyCatGirl/Blip-Streamlit-chatbot</span>
    </div>
</div>
\""", unsafe_allow_html=True)

# Load models
with st.spinner("Initializing medical AI models... This might take a few minutes"):
    try:
        # Create model cache directory if it doesn't exist
        os.makedirs("model_cache", exist_ok=True)
        
        # Load models
        blip_processor, blip_model = load_models()
        ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model = load_translation_models()
        
        if blip_processor is None or blip_model is None:
            st.error("Critical error: Failed to load vision model. App cannot function.")
            st.stop()
            
    except Exception as e:
        st.error(f"Model initialization failed: {str(e)}")
        st.stop()

# Main app layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div class='card-title'>Upload Medical Image</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Medical Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
    else:
        # Display placeholder image
        st.image("https://images.unsplash.com/photo-1516549655169-df83a0774514?auto=format&fit=crop&w=600&h=400&q=80", 
                 caption="Medical scan placeholder", use_column_width=True)

with col2:
    st.markdown("<div class='card-title'>Ask Your Question</div>", unsafe_allow_html=True)
    question = st.text_area("", placeholder="Type your question in Arabic or English...", 
                            height=150, label_visibility="collapsed")
    
    analyze_btn = st.button("Analyze Image", use_container_width=True)
    
    if analyze_btn:
        if uploaded_file is None or not question.strip():
            st.warning("Please upload an image and enter a question")
        else:
            try:
                image = Image.open(uploaded_file).convert('RGB')
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.stop()
            
            with st.spinner("Analyzing medical image and question..."):
                start_time = time.time()
                
                try:
                    # Detect language
                    is_arabic = any('\\u0600' <= c <= '\\u06FF' for c in question)
                    
                    if is_arabic:
                        question_ar = question.strip()
                        question_en = translate_ar_to_en(question_ar, ar_en_tokenizer, ar_en_model)
                    else:
                        question_en = question.strip()
                        question_ar = translate_en_to_ar(question_en, en_ar_tokenizer, en_ar_model)
                    
                    # Process with BLIP model
                    try:
                        inputs = blip_processor(
                            images=image, 
                            text=question_en, 
                            return_tensors="pt"
                        )
                        
                        # Generate answer
                        out = blip_model.generate(**inputs)
                        answer_en = blip_processor.decode(out[0], skip_special_tokens=True)
                        
                        # Translate answer
                        answer_ar = translate_answer_medical(answer_en, en_ar_tokenizer, en_ar_model)
                        
                        # Save to history
                        st.session_state.history.append({
                            "image": uploaded_file.getvalue(),
                            "question_ar": question_ar,
                            "question_en": question_en,
                            "answer_ar": answer_ar,
                            "answer_en": answer_en,
                            "time": time.strftime("%Y-%m-%d %H:%M")
                        })
                        
                        processing_time = time.time() - start_time
                        
                        # Display results
                        st.markdown("<div class='card-title'>Analysis Results</div>", unsafe_allow_html=True)
                        
                        st.markdown(f"<div class='result-box'>"
                                    f"<span class='lang-tag'>AR</span> <strong>السؤال:</strong> {question_ar}<br>"
                                    f"<span class='lang-tag'>EN</span> <strong>Question:</strong> {question_en}" 
                                    f"</div>", unsafe_allow_html=True)
                        
                        st.markdown(f"<div class='result-box'>"
                                    f"<span class='lang-tag'>AR</span> <strong>الإجابة:</strong> {answer_ar}<br>"
                                    f"<span class='lang-tag'>EN</span> <strong>Answer:</strong> {answer_en}" 
                                    f"</div>", unsafe_allow_html=True)
                        
                        st.caption(f"Processed in {processing_time:.2f} seconds")
                        
                    except Exception as model_error:
                        st.markdown("<div class='error-box'>"
                                    f"<strong>Model Error:</strong> {str(model_error)}<br>"
                                    "Please try a different image or question"
                                    "</div>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown("<div class='error-box'>"
                                f"<strong>System Error:</strong> {str(e)}<br>"
                                "Please try again or report this issue"
                                "</div>", unsafe_allow_html=True)

# History section
if st.session_state.history:
    st.divider()
    st.markdown("<div class='card-title'>Analysis History</div>", unsafe_allow_html=True)
    
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.container():
            st.markdown(f"<div class='history-item'>", unsafe_allow_html=True)
            
            col_img, col_text = st.columns([1, 2])
            
            with col_img:
                try:
                    st.image(item["image"], caption="Medical Image", use_column_width=True)
                except:
                    st.warning("Could not display image from history")
            
            with col_text:
                st.markdown(f"<div><strong>Question (AR):</strong> {item['question_ar']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div><strong>Question (EN):</strong> {item['question_en']}</div>", unsafe_allow_html=True)
                st.divider()
                st.markdown(f"<div><strong>Answer (AR):</strong> {item['answer_ar']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div><strong>Answer (EN):</strong> {item['answer_en']}</div>", unsafe_allow_html=True)
                st.caption(f"Analyzed at {item['time']}")
            
            st.markdown("</div>", unsafe_allow_html=True)

# Clear history button
if st.session_state.history:
    if st.button("Clear History", type="secondary", use_container_width=True):
        st.session_state.history = []
        st.experimental_rerun()

# Footer
st.divider()
st.markdown(\"""
<div class="footer">
    <p><strong>Medical VQA System</strong> • Powered by your custom BLIP model</p>
    <p>Note: This AI assistant provides preliminary medical insights. Always consult a healthcare professional for diagnosis.</p>
    <p style="margin-top:1rem;">Model: ButterflyCatGirl/Blip-Streamlit-chatbot</p>
</div>
\""", unsafe_allow_html=True)
""")

# Set up the environment
import socket
from pyngrok import ngrok
import subprocess
import time

# Find an available port
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

port = find_free_port()
print(f"Using port: {port}")

# Start Streamlit in the background
process = subprocess.Popen([
    "streamlit", "run", "app.py",
    "--server.port", str(port),
    "--server.headless", "true",
    "--browser.serverAddress", "0.0.0.0",
    "--server.enableCORS", "false",
    "--server.enableXsrfProtection", "false"
])

# Wait for app to start
time.sleep(8)

# Set up LocalTunnel
print("Setting up LocalTunnel...")
tunnel_process = subprocess.Popen(
    f"npx localtunnel --port {port}",
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Wait for tunnel URL
time.sleep(8)
tunnel_url = None

# Try to get tunnel URL
try:
    for line in iter(tunnel_process.stdout.readline, ''):
        if "your url is: " in line:
            tunnel_url = line.split("your url is: ")[1].strip()
            break
except:
    pass

if not tunnel_url:
    # Fallback to ngrok if LocalTunnel fails
    print("LocalTunnel failed, using ngrok instead")
    try:
        ngrok.set_auth_token("2g5vF8jz4wjYqH4WY1ZfZg3b9eT_2L5yXbVdZRqPyzR4gXZvV")  # Free public token
        tunnel = ngrok.connect(port)
        tunnel_url = tunnel.public_url
    except Exception as e:
        print(f"Ngrok failed: {str(e)}")
        tunnel_url = "https://example.com (setup failed)"

print("="*80)
print(f"Your Medical VQA App is running at: {tunnel_url}")
print("="*80)
print("Note: It may take 1-2 minutes for models to download on first run")
print("Keep this notebook running to keep the app active")

# Keep the cell running
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("Shutting down server...")
    process.terminate()
    tunnel_process.terminate()
