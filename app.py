# app.py - Complete Streamlit Medical VQA Chatbot with BLIP2 Model
import streamlit as st
from PIL import Image, ImageOps
import torch
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_IMAGE_SIZE = (512, 512)
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "bmp", "tiff"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

class MedicalVQASystem:
    """Medical Visual Question Answering System using BLIP2"""

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

            # Load BLIP2 processor and model (FIXED)
            model_names = [
                "Salesforce/blip-vqa-base"
                "Salesforce/blip2-flan-t5-xl",
                "Salesforce/blip2-opt-2.7b",
                "Salesforce/blip2-flan-t5-base"
            ]

            for model_name in model_names:
                try:
                    self.processor = Blip2Processor.from_pretrained(model_name)
                    
                    if self.device == "cpu":
                        self.model = Blip2ForConditionalGeneration.from_pretrained(
                            model_name,
                            torch_dtype=torch.float32
                        )
                    else:
                        self.model = Blip2ForConditionalGeneration.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16
                        )

                    self.model = self.model.to(self.device)
                    logger.info(f"BLIP2 model ({model_name}) loaded successfully on {self.device}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {str(e)}")
                    continue

            if self.model is None:
                raise Exception("Failed to load any BLIP2 model")

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

        return text

    def _get_medical_translation(self, answer_en: str) -> str:
        """Get medical-specific translation with proper Arabic medical terminology"""
        
        # Comprehensive medical terms dictionary
        medical_terms = {
            # Medical imaging
            "chest x-ray": "أشعة سينية على الصدر",
            "x-ray": "أشعة سينية", 
            "ct scan": "تصوير مقطعي محوسب",
            "mri": "رنين مغناطيسي",
            "ultrasound": "موجات فوق صوتية",
            "radiograph": "صورة شعاعية",
            
            # Body parts
            "chest": "الصدر", "lung": "الرئة", "lungs": "الرئتين",
            "heart": "القلب", "brain": "الدماغ", "liver": "الكبد",
            "kidney": "الكلية", "bone": "العظم", "bones": "العظام",
            "spine": "العمود الفقري", "skull": "الجمجمة",
            
            # Medical conditions
            "normal": "طبيعي", "abnormal": "غير طبيعي",
            "pneumonia": "التهاب رئوي", "fracture": "كسر",
            "tumor": "ورم", "cancer": "سرطان",
            "infection": "التهاب", "inflammation": "التهاب",
            "fluid": "سوائل", "mass": "كتلة",
            
            # Medical assessments
            "appears": "يبدو", "shows": "يُظهر",
            "indicates": "يشير إلى", "suggests": "يوحي بـ",
            "consistent with": "متوافق مع",
            "compatible with": "متوافق مع",
            
            # Common words
            "the": "الـ", "a": "أ", "an": "أ",
            "is": "هو", "are": "هي", "in": "في",
            "of": "من", "with": "مع", "and": "و"
        }
        
        # First, try word-by-word replacement for medical terms
        words = answer_en.lower().split()
        translated_words = []
        
        for word in words:
            # Clean punctuation
            clean_word = word.strip('.,!?;:()')
            
            # Check for exact medical term match
            if clean_word in medical_terms:
                translated_words.append(medical_terms[clean_word])
            else:
                # Check for partial matches (e.g., "chest" in "chest pain")
                found_translation = False
                for term, arabic_term in medical_terms.items():
                    if term in clean_word:
                        translated_words.append(arabic_term)
                        found_translation = True
                        break
                
                if not found_translation:
                    # Use general translation for remaining words
                    translated_word = self._translate_text(clean_word, "en", "ar")
                    if translated_word and translated_word != clean_word:
                        translated_words.append(translated_word)
                    else:
                        translated_words.append(word)  # Keep original if translation fails
        
        # Join translated words
        arabic_response = " ".join(translated_words)
        
        # If translation is poor or failed, provide contextual medical response
        if (len(arabic_response) < 5 or 
            arabic_response == answer_en or 
            "nouriture" in arabic_response or  # Detect bad translation
            len([w for w in arabic_response.split() if '\u0600' <= w[0] <= '\u06FF']) < 2):
            
            # Provide contextual Arabic medical responses based on English content
            if "normal" in answer_en.lower():
                return "الصورة تبدو طبيعية ولا تظهر أي علامات غير طبيعية واضحة"
            elif "abnormal" in answer_en.lower():
                return "تظهر الصورة بعض النتائج غير الطبيعية التي تحتاج لتقييم طبي متخصص"
            elif "chest" in answer_en.lower():
                return "هذه صورة أشعة سينية للصدر تحتاج لتفسير من قبل طبيب مختص في الأشعة"
            elif "brain" in answer_en.lower():
                return "هذه صورة للدماغ تتطلب تحليل من قبل طبيب الأشعة أو طبيب الأعصاب"
            else:
                return "تحتاج هذه الصورة الطبية إلى تقييم وتشخيص من قبل طبيب مختص للحصول على رأي طبي دقيق"
        
        return arabic_response.strip()

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
        """Process medical VQA query with BLIP2"""
        try:
            # Preprocess image
            image = self._preprocess_image(image)

            # Detect language and prepare translations
            detected_lang = self._detect_language(question)

            if detected_lang == "ar":
                question_ar = question.strip()
                question_en = self._translate_text(question_ar, "ar", "en")
            else:
                question_en = question.strip()
                question_ar = self._translate_text(question_en, "en", "ar")

            # Process with BLIP2 model using English question
            inputs = self.processor(image, question_en, return_tensors="pt")

            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate answer using BLIP2
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=100,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )

            # Decode answer (FIXED for BLIP2)
            answer_en = self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()

            # Get proper Arabic translation
            answer_ar = self._get_medical_translation(answer_en)

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
            return {
                "error": str(e),
                "success": False
            }

# Initialize the VQA system
@st.cache_resource(show_spinner=False)
def get_vqa_system():
    """Get cached VQA system instance"""
    return MedicalVQASystem()

def init_streamlit_config():
    """Initialize Streamlit configuration"""
    st.set_page_config(
        page_title="Medical AI Assistant",
        layout="wide",
        page_icon="🩺",
        initial_sidebar_state="expanded"
    )

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .upload-section {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border: 2px dashed #dee2e6;
            margin-bottom: 1rem;
        }
        .result-container {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        .success-box {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        .info-box {
            background: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        .warning-box {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.7rem 2rem;
            font-weight: 600;
            width: 100%;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .rtl {
            direction: rtl;
            text-align: right;
        }
    </style>
    """, unsafe_allow_html=True)

def validate_uploaded_file(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded file"""
    if uploaded_file is None:
        return False, "No file uploaded"

    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"File size too large. Maximum size is {MAX_FILE_SIZE/1024/1024}MB"

    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in SUPPORTED_FORMATS:
        return False, f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_FORMATS)}"

    return True, "Valid file"

def main():
    """Main Streamlit application"""
    init_streamlit_config()
    apply_custom_css()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🩺 Medical AI Assistant</h1>
        <p>Advanced multilingual medical image analysis powered by BLIP2</p>
        <p><strong>Upload medical images and ask questions in Arabic or English</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize VQA system
    vqa_system = get_vqa_system()

    # Load models if not already loaded
    if vqa_system.model is None:
        with st.spinner("🔄 Loading BLIP2 models... This may take a few minutes on first run..."):
            success = vqa_system.load_models()
            if success:
                st.success("✅ Medical AI models loaded successfully!")
            else:
                st.error("❌ Failed to load AI models. Please refresh the page and try again.")
                st.stop()

    # Create main interface
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📤 Upload Medical Image")

        uploaded_file = st.file_uploader(
            "Choose a medical image...",
            type=SUPPORTED_FORMATS,
            help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)}. Max size: {MAX_FILE_SIZE/1024/1024}MB"
        )

        if uploaded_file:
            is_valid, message = validate_uploaded_file(uploaded_file)

            if is_valid:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
                    st.info(f"📊 Image size: {image.size[0]}×{image.size[1]} pixels | Format: {image.format}")

                except Exception as e:
                    st.error(f"❌ Error loading image: {str(e)}")
                    uploaded_file = None
            else:
                st.error(f"❌ {message}")
                uploaded_file = None

    with col2:
        st.markdown("### 💭 Ask Your Question")

        # Language selection
        language = st.selectbox(
            "Select Language / اختر اللغة:",
            options=["en", "ar"],
            format_func=lambda x: "English" if x == "en" else "العربية",
            help="Choose your preferred language for the question"
        )

        # Question input
        if language == "ar":
            question_placeholder = "اكتب سؤالك الطبي هنا... مثال: ما هو التشخيص المحتمل؟"
            question_label = "السؤال الطبي:"
        else:
            question_placeholder = "Type your medical question here... Example: What is the likely diagnosis?"
            question_label = "Medical Question:"

        question = st.text_area(
            question_label,
            height=150,
            placeholder=question_placeholder,
            help="Ask specific questions about the medical image"
        )

        # Analyze button
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
                                st.markdown(f"**السؤال:** {result['question_ar']}", unsafe_allow_html=True)
                                st.markdown(f"**الإجابة:** {result['answer_ar']}", unsafe_allow_html=True)

                            st.markdown(f"**⏱️ Processing Time:** {processing_time:.2f} seconds")
                            st.markdown(f"**🔍 Detected Language:** {'Arabic' if result['detected_language'] == 'ar' else 'English'}")

                        else:
                            st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

                    except Exception as e:
                        st.error(f"❌ Processing error: {str(e)}")

    # Sidebar with information
    with st.sidebar:
        st.markdown("### ℹ️ Information")
        st.markdown("""
        **How to use:**
        1. Upload a medical image (X-ray, CT, MRI, etc.)
        2. Select your preferred language
        3. Ask a specific medical question
        4. Click 'Analyze' to get AI insights

        **Supported Languages:**
        - English 🇺🇸
        - Arabic 🇪🇬

        **Supported Image Formats:**
        - JPG, JPEG, PNG, BMP, TIFF

        **Note:** This AI assistant provides preliminary insights for educational purposes. Always consult healthcare professionals for medical diagnosis and treatment decisions.
        """)

        st.markdown("---")
        st.markdown("### 🔧 System Status")

        if vqa_system.model is not None:
            st.success("✅ AI Models: Loaded")
            st.info(f"🖥️ Device: {vqa_system.device.upper()}")
        else:
            st.error("❌ AI Models: Not Loaded")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>Medical VQA System v3.0</strong> | Powered by BLIP2 + Enhanced Arabic Translation</p>
        <p>⚠️ <em>This system is for educational and research purposes. Not a substitute for professional medical advice.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
