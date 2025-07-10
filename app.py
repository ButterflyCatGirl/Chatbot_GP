# app.py - Complete Streamlit Medical VQA Chatbot with BLIP Model
import streamlit as st
from PIL import Image, ImageOps
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
import requests
from io import BytesIO
import base64
import json
from typing import Optional, Tuple, Dict, Any
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_IMAGE_SIZE = (512, 512)
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "bmp", "tiff"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

class MedicalVQASystem:
    """Medical Visual Question Answering System using BLIP"""
    def __init__(self):
        self.processor = None
        self.model = None
        self.ar_en_tokenizer = None
        self.ar_en_model = None
        self.en_ar_tokenizer = None
        self.en_ar_model = None
        self.device = self._get_device()

    def _get_device(self) -> str:
        """Determine the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _clear_memory(self):
        """Clear GPU/CPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def load_models(self) -> bool:
        """Load all required models with error handling"""
        try:
            self._clear_memory()

            # Load BLIP processor
            self.processor = BlipProcessor.from_pretrained("ButterflyCatGirl/Blip-Streamlit-chatbot")
            logger.info("BLIP processor loaded successfully")

            # Try to load custom model first, fallback to base model
            model_names = [
                "ButterflyCatGirl/Blip-Streamlit-chatbot",
                "Salesforce/blip-vqa-base"
            ]
            for model_name in model_names:
                try:
                    if self.device == "cpu":
                        self.model = BlipForQuestionAnswering.from_pretrained(
                            model_name, torch_dtype=torch.float32
                        )
                    else:
                        self.model = BlipForQuestionAnswering.from_pretrained(
                            model_name, torch_dtype=torch.float16
                        )
                    self.model = self.model.to(self.device)
                    logger.info(f"BLIP model ({model_name}) loaded successfully on {self.device}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {str(e)}")
                    continue
            if self.model is None:
                raise Exception("Failed to load any BLIP model")

            # Load translation models
            try:
                self.ar_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
                self.ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
                self.en_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
                self.en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
                logger.info("Translation models loaded successfully")
            except Exception as e:
                logger.warning(f"Translation models failed to load: {str(e)}")

            return True
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            return False

    def _detect_language(self, text: str) -> str:
        """Detect if text is Arabic or English"""
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return "ar" if arabic_chars > 0 else "en"

    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text between Arabic and English"""
        if source_lang == target_lang:
            return text
        try:
            if source_lang == "ar" and target_lang == "en" and self.ar_en_tokenizer:
                inputs = self.ar_en_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = self.ar_en_model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
                return self.ar_en_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            elif source_lang == "en" and target_lang == "ar" and self.en_ar_tokenizer:
                inputs = self.en_ar_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = self.en_ar_model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
                return self.en_ar_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception as e:
            logger.warning(f"Translation failed: {str(e)}")
        return text  # Return original if translation fails

    def _get_medical_translation(self, answer_en: str) -> str:
        """Get medical-specific translation for common terms"""
        medical_terms = {
            "chest x-ray": "أشعة سينية للصدر",
            "x-ray": "أشعة سينية",
            "x ray": "أشعة سينية",  # Added missing key
            "ct scan": "تصوير مقطعي محوسب",
            "mri": "تصوير بالرنين المغناطيسي",
            "ultrasound": "تصوير بالموجات فوق الصوتية",
            "normal": "طبيعي",
            "abnormal": "غير طبيعي",
            "brain": "الدماغ",
            "heart": "القلب",
            "lung": "الرئة",
            "fracture": "كسر",
            "pneumonia": "التهاب رئوي",
            "tumor": "ورم",
            "cancer": "سرطان",
            "infection": "عدوى",
            "liver": "الكبد",
            "kidney": "الكلى",
            "bone": "العظم",
            "blood": "دم",
            "artery": "شريان",
            "vein": "وريد",
            "benign": "حميد",
            "malignant": "خبيث",
            "healthy": "صحي",
            "disease": "مرض"
        }

        answer_lower = answer_en.lower()
        for term, translation in medical_terms.items():
            if term in answer_lower:
                answer_en = answer_en.replace(term, translation)
        return answer_en

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for optimal performance"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if image.size[0] > MAX_IMAGE_SIZE[0] or image.size[1] > MAX_IMAGE_SIZE[1]:
                image = ImageOps.fit(image, MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise

    def process_query(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Process medical VQA query"""
        try:
            image = self._preprocess_image(image)

            detected_lang = self._detect_language(question)
            if detected_lang == "ar":
                question_ar = question.strip()
                question_en = self._translate_text(question_ar, "ar", "en")
            else:
                question_en = question.strip()
                question_ar = self._translate_text(question_en, "en", "ar")

            inputs = self.processor(image, question_en, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            answer_en = self.processor.decode(outputs[0], skip_special_tokens=True).strip()

            if detected_lang == "ar":
                answer_ar = self._get_medical_translation(answer_en)
            else:
                answer_ar = self._translate_text(answer_en, "en", "ar")

            return {
                "question_en": question_en,
                "question_ar": question_ar,
                "answer_en": answer_en,
                "answer_ar": answer_ar,
                "detected_language": detected_lang,
                "success": True
            }
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {"error": str(e), "success": False}

# Initialize session state for VQA system
if 'vqa_system' not in st.session_state:
    st.session_state.vqa_system = MedicalVQASystem()
    logger.info("Initializing MedicalVQASystem...")

vqa_system = st.session_state.vqa_system

# Load models if not already loaded
if vqa_system.model is None:
    with st.spinner("🔄 Loading AI models... This may take a few minutes on first run..."):
        success = vqa_system.load_models()
        if success:
            st.success("✅ Medical AI models loaded successfully!")
        else:
            st.error("❌ Failed to load AI models. Please refresh the page and try again.")
            st.stop()

# Streamlit UI
st.set_page_config(page_title="Medical AI Assistant", layout="wide", page_icon="🩺", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .rtl { direction: rtl; text-align: right; }
    .main-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 10px; text-align: center; }
    .result-container { background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🩺 Medical AI Assistant</h1>
    <p>Advanced multilingual medical image analysis powered by AI</p>
</div>
""", unsafe_allow_html=True)

# Main UI
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Medical Image")
    uploaded_file = st.file_uploader("Choose a medical image...", type=SUPPORTED_FORMATS)
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")

with col2:
    st.markdown("### 💭 Ask Your Question")
    language = st.selectbox("Select Language / اختر اللغة:", options=["en", "ar"], format_func=lambda x: "English" if x == "en" else "العربية")
    question_placeholder = "اكتب سؤالك الطبي هنا..." if language == "ar" else "Type your medical question here..."
    question = st.text_area("Medical Question:", height=150, placeholder=question_placeholder)
    analyze_button = st.button("🔍 Analyze Medical Image", use_container_width=True)

    if analyze_button:
        if not uploaded_file:
            st.warning("⚠️ Please upload a medical image first.")
        elif not question.strip():
            st.warning("⚠️ Please enter a medical question.")
        else:
            with st.spinner("🧠 AI is analyzing the medical image..."):
                try:
                    start_time = time.time()
                    image = Image.open(uploaded_file)
                    result = vqa_system.process_query(image, question)
                    processing_time = time.time() - start_time

                    if result["success"]:
                        st.markdown("---")
                        st.markdown("### 📋 Analysis Results")
                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.markdown("**🇺🇸 English Results**")
                            st.markdown(f"**Question:** {result['question_en']}")
                            st.markdown(f"**Answer:** {result['answer_en']}")
                        with res_col2:
                            st.markdown("**🇪🇬 النتائج بالعربية**")
                            st.markdown(f"<div class='rtl'>**السؤال:** {result['question_ar']}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='rtl'>**الإجابة:** {result['answer_ar']}</div>", unsafe_allow_html=True)
                        st.markdown(f"**⏱️ Processing Time:** {processing_time:.2f} seconds")
                    else:
                        st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"❌ Processing error: {str(e)}")

# Sidebar
with st.sidebar:
    st.markdown("### ℹ️ Information")
    st.markdown("""
    **How to use:**
    1. Upload a medical image (X-ray, CT, MRI, etc.)
    2. Select your preferred language
    3. Ask a specific medical question
    4. Click 'Analyze' to get AI insights
    """)
    st.markdown("---")
    st.markdown("### 🔧 System Status")
    st.success("✅ AI Models: Loaded") if vqa_system.model else st.error("❌ AI Models: Not Loaded")
    st.info(f"🖥️ Device: {vqa_system.device.upper()}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Medical VQA System v2.0 | Powered by BLIP + Transformers</div>", unsafe_allow_html=True)
