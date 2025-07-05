# app.py - Medical Visual Question Answering System
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
import warnings
from io import BytesIO
from typing import Optional, Dict, Any
import re

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
MAX_IMAGE_SIZE = (512, 512)
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "bmp", "tiff"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MODEL_VARIANTS = [
    "Salesforce/blip2-opt-2.7b",
    "Salesforce/blip2-flan-t5-xl",
    "llava-hf/llava-1.5-7b-hf"
]

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalVQASystem:
    """Medical Visual Question Answering System using BLIP2"""
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = self._get_device()
        self.medical_terms = self._load_medical_terms()
        
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
    
    def _load_medical_terms(self) -> Dict[str, str]:
        """Load comprehensive medical terminology dictionary"""
        return {
            # Anatomical terms
            "left ventricle": "البطين الأيسر",
            "right atrium": "ال Auricle الأيمن",
            "aorta": "الشريان الأورطي",
            "pulmonary artery": "الشريان الرئوي",
            
            # Pathologies
            "pneumothorax": "انفجار الرئة",
            "cardiomegaly": "تكبر القلب",
            "atelectasis": "انخماص الرئة",
            "pleural effusion": "استسقاء جنبي",
            "fracture": "كسر",
            "osteoporosis": "هشاشة العظام",
            "metastasis": "انتشار النقائل",
            "hemorrhage": "نزيف",
            "edema": "ورم",
            "calcification": "تكلس",
            
            # Imaging terms
            "chest x-ray": "أشعة سينية للصدر",
            "ct scan": "تصوير مقطعي محوسب",
            "mri": "تصوير بالرنين المغناطيسي",
            "ultrasound": "تصوير بالموجات فوق الصوتية"
        }
    
    def _detect_language(self, text: str) -> str:
        """Detect if text is Arabic or English"""
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return "ar" if arabic_chars > 0 else "en"
    
    def _medical_translate(self, text: str, target_lang: str) -> str:
        """Medical-specific translation with fallback mechanism"""
        if target_lang == "en":
            return text  # English remains unchanged
            
        # First try dictionary-based translation
        translated = text
        for term, ar_term in self.medical_terms.items():
            if term.lower() in translated.lower():
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                translated = pattern.sub(ar_term, translated)
        
        # Then use general translation for remaining text
        try:
            # Implement translation model fallback here
            pass  # Placeholder for actual translation model
        except Exception as e:
            logger.warning(f"Translation fallback failed: {str(e)}")
            
        return translated

    def load_models(self) -> bool:
        """Load BLIP2 models with fallback mechanisms"""
        try:
            self._clear_memory()
            
            # Load BLIP2 processor
            self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            
            # Try to load models with fallback
            for model_name in MODEL_VARIANTS:
                try:
                    self.model = Blip2ForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                        device_map="auto" if self.device != "cpu" else None
                    )
                    logger.info(f"Loaded model: {model_name}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {str(e)}")
                    continue
                    
            return False
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            return False

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for optimal performance"""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Resize if too large
            if image.size[0] > MAX_IMAGE_SIZE[0] or image.size[1] > MAX_IMAGE_SIZE[1]:
                image = ImageOps.fit(image, MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
                
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise

    def process_query(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Process medical VQA query with enhanced error handling"""
        try:
            # Preprocess image
            image = self._preprocess_image(image)
            
            # Detect language
            detected_lang = self._detect_language(question)
            
            # Process with BLIP2 model
            prompt = question if detected_lang == "ar" else (
                "This is a medical image. Describe findings, abnormalities, and potential diagnosis."
            )
            
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
            
            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    num_beams=5,
                    early_stopping=True
                )
                
            answer_en = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Translate if needed
            answer_ar = self._medical_translate(answer_en, "ar") if detected_lang == "ar" else ""
            
            return {
                "question_en": question if detected_lang == "en" else self._medical_translate(question, "en"),
                "question_ar": question if detected_lang == "ar" else self._medical_translate(question, "ar"),
                "answer_en": answer_en,
                "answer_ar": answer_ar or self._medical_translate(answer_en, "ar"),
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
        page_title="AI Health Assistant",
        layout="wide",
        page_icon="🩺",
        initial_sidebar_state="expanded"
    )

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
        /* Main container styles */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        /* RTL support */
        .arabic-text {
            direction: rtl;
            text-align: right;
            font-family: 'Noto Sans Arabic', sans-serif;
        }
        
        /* Result containers */
        .result-container {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        /* Custom buttons */
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
    </style>
    """, unsafe_allow_html=True)

def validate_uploaded_file(uploaded_file) -> tuple:
    """Validate uploaded file"""
    if uploaded_file is None:
        return False, "No file uploaded"
        
    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"File size too large. Maximum size is {MAX_FILE_SIZE/1024/1024}MB"
        
    # Check file format
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
        <h1>🩺 AI Health Assistant</h1>
        <p>Multilingual Medical Image Analysis Powered by AI</p>
        <p><strong>Upload medical images and ask questions in Arabic or English</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize VQA system
    vqa_system = get_vqa_system()
    
    # Load models if not already loaded
    if vqa_system.model is None:
        with st.spinner("🔄 Loading AI models... This may take a few minutes on first run..."):
            success = vqa_system.load_models()
            if success:
                st.success("✅ Medical AI models loaded successfully!")
            else:
                st.error("❌ Failed to load AI models. Please refresh the page and try again.")
                st.stop()
    
    # Main interface columns
    col1, col2 = st.columns([1, 1], gap="large")
    
    # Image upload section
    with col1:
        st.markdown("### 📤 Upload Medical Image")
        uploaded_file = st.file_uploader(
            "Choose a medical image...",
            type=SUPPORTED_FORMATS,
            help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)}. Max size: {MAX_FILE_SIZE/1024/1024}MB"
        )
        
        if uploaded_file:
            # Validate file
            is_valid, message = validate_uploaded_file(uploaded_file)
            if is_valid:
                try:
                    # Display image
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
                    # Show image info
                    st.info(f"📊 Image size: {image.size[0]}×{image.size[1]} pixels | Format: {image.format}")
                except Exception as e:
                    st.error(f"❌ Error loading image: {str(e)}")
                    uploaded_file = None
            else:
                st.error(f"❌ {message}")
                uploaded_file = None
    
    # Question input section
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
                # Process the query
                with st.spinner("🧠 AI is analyzing the medical image..."):
                    try:
                        start_time = time.time()
                        image = Image.open(uploaded_file)
                        result = vqa_system.process_query(image, question)
                        processing_time = time.time() - start_time
                        
                        if result["success"]:
                            # Display results
                            st.markdown("---")
                            st.markdown("### 📋 Analysis Results")
                            
                            # Create result columns
                            res_col1, res_col2 = st.columns(2)
                            
                            with res_col1:
                                st.markdown("**🇺🇸 English Results**")
                                st.markdown(f"**Question:** {result['question_en']}")
                                st.markdown(f"**Answer:** {result['answer_en']}")
                            
                            with res_col2:
                                st.markdown("**🇸🇦 النتائج بالعربية**")
                                st.markdown(f"<div class='arabic-text'>**السؤال:** {result['question_ar']}</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='arabic-text'>**الإجابة:** {result['answer_ar']}</div>", unsafe_allow_html=True)
                            
                            # Processing info
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
        - Arabic 🇸🇦
        
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
        <p><strong>Medical VQA System v3.0</strong> | Powered by BLIP2 + Transformers</p>
        <p>⚠️ <em>This system is for educational and research purposes. Not a substitute for professional medical advice.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
