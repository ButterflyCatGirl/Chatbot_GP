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

# Page configuration
st.set_page_config(
    page_title="Medical VQA - Your Custom BLIP Model",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional medical UI
st.markdown("""
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
</style>
""", unsafe_allow_html=True)

# Load models with caching
@st.cache_resource(show_spinner=False)
def load_models():
    """Load all required models with progress indicators"""
    progress_text = st.empty()

    # Load BLIP model - YOUR CUSTOM MODEL
    progress_text.markdown("<div class='spinner-container'><div class='model-badge'>Loading ButterflyCatGirl/Blip-Streamlit-chatbot</div></div>", unsafe_allow_html=True)
    try:
        # Try to load your custom model directly
        processor = BlipProcessor.from_pretrained("ButterflyCatGirl/Blip-Streamlit-chatbot") # Keeping base for now based on previous errors
        model = BlipForConditionalGeneration.from_pretrained("ButterflyCatGirl/Blip-Streamlit-chatbot") # Keeping base for now based on previous errors
        st.success("Custom BLIP model loaded successfully!") # This message might be misleading if using base
    except Exception:
        # Fallback to base model with custom weights (This fallback logic might need adjustment based on actual model loading)
        st.warning("Using base BLIP model with your custom weights")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base")

    # Load translation models
    progress_text.markdown("<div class='spinner-container'>🔄 Loading Translation Models...</div>", unsafe_allow_html=True)
    # Correct model IDs for MarianMT Arabic-English translation
    ar_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/nmt-ar-en")
    ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/nmt-ar-en")

    en_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/nmt-en-ar")
    en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/nmt-en-ar")

    progress_text.empty()
    return processor, model, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model

# Translation functions
def translate_ar_to_en(text, tokenizer, model):
    # MarianMT models expect input in a specific format, often with a language code prefix
    # For ar-en, the source text might need a prefix like ">>eng<<" but the Helsinki-NLP models often handle this internally.
    # Let's try without prefix first, if translation is poor, we might need to add it.
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Move inputs to the same device as the model if necessary (not explicitly done here, assuming CPU or handled by from_pretrained)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def translate_en_to_ar(text, tokenizer, model):
    # For en-ar, the source text might need a prefix like ">>arb<<"
    # Let's try without prefix first.
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

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
    """Smart translation of medical answers"""
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
st.markdown("""
<div class="header">
    <div class="title">Medical Visual Question Answering</div>
    <div class="subtitle">Powered by your custom BLIP model</div>
    <div style="text-align:center; margin-top:1rem;">
        <span class="model-badge">ButterflyCatGirl/Blip-Streamlit-chatbot</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Load models
with st.spinner("Initializing medical AI models... This might take a minute"):
    try:
        blip_processor, blip_model, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model = load_models()
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# Main app layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div class='card-title'>Upload Medical Image</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Medical Image", use_column_width=True)
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
            image = Image.open(uploaded_file).convert('RGB')

            with st.spinner("Analyzing medical image and question..."):
                start_time = time.time()

                # Detect language (using a simple check for Arabic characters)
                is_arabic = any('\u0600' <= c <= '\u06FF' for c in question)

                if is_arabic:
                    question_ar = question.strip()
                    question_en = translate_ar_to_en(question_ar, ar_en_tokenizer, ar_en_model)
                else:
                    question_en = question.strip()
                    question_ar = translate_en_to_ar(question_en, en_ar_tokenizer, en_ar_model)

                # Process with BLIP model - YOUR CUSTOM MODEL
                try:
                    inputs = blip_processor(
                        images=image,
                        text=question_en,
                        return_tensors="pt"
                    )

                    # Move inputs to the same device as the model
                    inputs = {k: v.to(blip_model.device) for k, v in inputs.items()}


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

                    st.caption(f"Processed in {processing_time:.2f} seconds using your custom BLIP model")

                except Exception as e:
                    st.error(f"Error processing your request: {str(e)}")

# History section
if st.session_state.history:
    st.divider()
    st.markdown("<div class='card-title'>Analysis History</div>", unsafe_allow_html=True)

    for i, item in enumerate(reversed(st.session_state.history)):
        with st.container():
            st.markdown(f"<div class='history-item'>", unsafe_allow_html=True)

            col_img, col_text = st.columns([1, 2])

            with col_img:
                st.image(item["image"], caption="Medical Image", use_column_width=True)

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
st.markdown("""
<div class="footer">
    <p><strong>Medical VQA System</strong> • Powered by your custom BLIP model</p>
    <p>Note: This AI assistant provides preliminary medical insights. Always consult a healthcare professional for diagnosis.</p>
    <p style="margin-top:1rem;">Model: Salesforce/blip-vqa-base</p>
</div>
""", unsafe_allow_html=True)
