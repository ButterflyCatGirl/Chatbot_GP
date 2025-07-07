# Fixed Medical VQA Streamlit App with Fine-tuned Model
import streamlit as st
from PIL import Image, ImageOps
import torch
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
import logging
import time
import gc
from typing import Optional, Tuple, Dict, Any
import warnings
import re

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Optimized for speed
MAX_IMAGE_SIZE = (384, 384)  # Reduced for faster processing
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "bmp"]
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
FINE_TUNED_MODEL = "sharawy53/blip-vqa-medical-arabic"

class OptimizedMedicalVQA:
    """Optimized Medical VQA System using your fine-tuned model"""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.translator_tokenizer = None
        self.translator_model = None
        self.device = self._get_optimal_device()
        self.medical_knowledge = self._load_medical_knowledge()
        
    def _get_optimal_device(self) -> str:
        """Get the most optimal device configuration"""
        if torch.cuda.is_available():
            # Use CUDA with optimizations
            torch.backends.cudnn.benchmark = True
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_medical_knowledge(self) -> Dict[str, str]:
        """Load medical knowledge base for better responses"""
        return {
            # Medical imaging terms
            "chest x-ray": "أشعة سينية على الصدر",
            "x-ray": "الأشعة السينية",
            "ct scan": "الأشعة المقطعية", 
            "mri": "الرنين المغناطيسي",
            "ultrasound": "الموجات فوق الصوتية",
            
            # Anatomy
            "chest": "الصدر", "lung": "الرئة", "lungs": "الرئتان",
            "heart": "القلب", "brain": "المخ", "liver": "الكبد",
            "kidney": "الكلية", "spine": "العمود الفقري",
            "bone": "العظم", "bones": "العظام",
            
            # Conditions
            "normal": "طبيعي", "abnormal": "غير طبيعي",
            "pneumonia": "الالتهاب الرئوي", "fracture": "كسر",
            "tumor": "ورم", "mass": "كتلة", "nodule": "عقدة",
            "infection": "عدوى", "inflammation": "التهاب",
            "fluid": "سوائل", "air": "هواء",
            
            # Medical descriptions
            "shows": "يُظهر", "appears": "يبدو", "indicates": "يشير إلى",
            "suggests": "يوحي بـ", "consistent": "متوافق مع",
            "visible": "مرئي", "present": "موجود", "absent": "غائب",
            "enlarged": "متضخم", "decreased": "منخفض", "increased": "مرتفع",
            
            # Common medical responses
            "consult": "استشر طبيب مختص",
            "diagnosis": "التشخيص", "treatment": "العلاج",
            "follow up": "متابعة طبية", "urgent": "عاجل"
        }
    
    def _clear_memory(self):
        """Aggressive memory clearing for speed"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    @st.cache_resource(show_spinner=False)
    def load_models(_self):
        """Load your fine-tuned model with caching and optimizations"""
        try:
            _self._clear_memory()
            
            # Load your fine-tuned BLIP model (NOT BLIP2)
            logger.info(f"Loading fine-tuned model: {FINE_TUNED_MODEL}")
            
            # Load processor and model - FIXED to use BLIP architecture
            _self.processor = BlipProcessor.from_pretrained(FINE_TUNED_MODEL)
            
            # Load with optimizations
            if _self.device == "cpu":
                _self.model = BlipForConditionalGeneration.from_pretrained(
                    FINE_TUNED_MODEL,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            else:
                _self.model = BlipForConditionalGeneration.from_pretrained(
                    FINE_TUNED_MODEL,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            
            # Move to device with optimization
            _self.model = _self.model.to(_self.device)
            _self.model.eval()  # Set to evaluation mode
            
            # Enable optimizations
            if _self.device == "cuda":
                _self.model = torch.compile(_self.model, mode="reduce-overhead")
            
            logger.info(f"Fine-tuned model loaded successfully on {_self.device}")
            
            # Load lightweight Arabic translator
            try:
                _self.translator_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
                _self.translator_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
                _self.translator_model = _self.translator_model.to(_self.device)
                logger.info("Arabic translator loaded successfully")
            except Exception as e:
                logger.warning(f"Translator loading failed: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            return False
    
    def _detect_language(self, text: str) -> str:
        """Fast language detection"""
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return "ar" if arabic_chars > 0 else "en"
    
    def _translate_to_arabic(self, text_en: str) -> str:
        """Enhanced Arabic translation with medical knowledge"""
        if not text_en:
            return text_en
            
        # First, try medical term replacement
        text_lower = text_en.lower()
        translated_parts = []
        
        # Split into sentences for better translation
        sentences = re.split(r'[.!?]+', text_en)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Check for medical terms
            translated_sentence = sentence.strip()
            for en_term, ar_term in self.medical_knowledge.items():
                if en_term in translated_sentence.lower():
                    translated_sentence = re.sub(
                        re.escape(en_term),
                        ar_term,
                        translated_sentence,
                        flags=re.IGNORECASE
                    )
            
            # If still mostly English, use model translation
            if sum(1 for c in translated_sentence if '\u0600' <= c <= '\u06FF') < 3:
                if self.translator_model and self.translator_tokenizer:
                    try:
                        inputs = self.translator_tokenizer(
                            sentence.strip(),
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=128
                        ).to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.translator_model.generate(
                                **inputs,
                                max_length=128,
                                num_beams=3,
                                early_stopping=True,
                                do_sample=False
                            )
                        
                        translated_sentence = self.translator_tokenizer.decode(
                            outputs[0], skip_special_tokens=True
                        ).strip()
                        
                    except Exception as e:
                        logger.warning(f"Translation failed: {e}")
                        # Fallback to medical context response
                        translated_sentence = self._get_medical_context_response(sentence)
            
            translated_parts.append(translated_sentence)
        
        result = ". ".join(translated_parts).strip()
        
        # Final fallback if translation is poor
        if len(result) < 3 or result == text_en:
            result = self._get_medical_context_response(text_en)
        
        return result
    
    def _get_medical_context_response(self, text_en: str) -> str:
        """Provide contextual medical response in Arabic"""
        text_lower = text_en.lower()
        
        if any(term in text_lower for term in ["normal", "healthy", "fine"]):
            return "الصورة تبدو طبيعية وسليمة"
        elif any(term in text_lower for term in ["abnormal", "problem", "issue"]):
            return "تظهر الصورة بعض النتائج التي تحتاج لتقييم طبي"
        elif "chest" in text_lower:
            return "هذه صورة أشعة سينية للصدر تحتاج لفحص من طبيب الأشعة"
        elif any(term in text_lower for term in ["brain", "head", "skull"]):
            return "صورة للرأس أو المخ تتطلب تحليل من طبيب مختص"
        elif "bone" in text_lower or "fracture" in text_lower:
            return "صورة للعظام قد تظهر كسر أو مشكلة تحتاج لفحص طبي"
        else:
            return "الصورة الطبية تحتاج لتقييم من طبيب مختص للحصول على رأي طبي دقيق"
    
    def _preprocess_image_fast(self, image: Image.Image) -> Image.Image:
        """Fast image preprocessing"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for speed while maintaining quality
            if image.size[0] > MAX_IMAGE_SIZE[0] or image.size[1] > MAX_IMAGE_SIZE[1]:
                image = ImageOps.fit(image, MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise
    
    def process_medical_query(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Process medical query with your fine-tuned model"""
        try:
            start_time = time.time()
            
            # Fast image preprocessing
            image = self._preprocess_image_fast(image)
            
            # Detect language
            detected_lang = self._detect_language(question)
            
            # Prepare question (your model was trained on both languages)
            question_processed = question.strip()
            
            # Process with your fine-tuned model
            inputs = self.processor(
                image, 
                question_processed, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate answer with optimized parameters
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():  # Mixed precision for speed
                        generated_ids = self.model.generate(
                            **inputs,
                            max_length=64,  # Reduced for speed
                            num_beams=3,    # Reduced for speed
                            early_stopping=True,
                            do_sample=False,
                            pad_token_id=self.processor.tokenizer.pad_token_id
                        )
                else:
                    generated_ids = self.model.generate(
                        **inputs,
                        max_length=64,
                        num_beams=3,
                        early_stopping=True,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.pad_token_id
                    )
            
            # Decode answer
            answer = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            # Clean up the answer (remove input question if present)
            if question_processed.lower() in answer.lower():
                answer = answer.replace(question_processed, "").strip()
            
            # Prepare responses based on detected language
            if detected_lang == "ar":
                answer_ar = answer
                answer_en = answer  # Your model might already provide mixed responses
            else:
                answer_en = answer
                answer_ar = self._translate_to_arabic(answer)
            
            processing_time = time.time() - start_time
            
            return {
                "question_original": question,
                "answer_en": answer_en,
                "answer_ar": answer_ar,
                "detected_language": detected_lang,
                "processing_time": processing_time,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }

# Streamlit App Configuration
def init_app():
    """Initialize Streamlit app with optimized config"""
    st.set_page_config(
        page_title="Medical AI Assistant - Fine-tuned",
        layout="wide",
        page_icon="🩺",
        initial_sidebar_state="expanded"
    )

def apply_medical_theme():
    """Apply medical theme with Arabic support"""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&display=swap');
        
        .main-header {
            background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .upload-area {
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 10px;
            border: 2px dashed #4a90e2;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .result-success {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        .result-info {
            background: #e3f2fd;
            border: 1px solid #bbdefb;
            color: #0d47a1;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        .arabic-text {
            font-family: 'Amiri', serif;
            direction: rtl;
            text-align: right;
            line-height: 1.8;
        }
        
        .processing-time {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            font-size: 0.9em;
            margin: 0.5rem 0;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            width: 100%;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(74, 144, 226, 0.3);
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_vqa_system():
    """Get cached VQA system"""
    return OptimizedMedicalVQA()

def validate_file(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded file"""
    if not uploaded_file:
        return False, "لم يتم رفع ملف / No file uploaded"
    
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"حجم الملف كبير جداً / File too large (max {MAX_FILE_SIZE//1024//1024}MB)"
    
    extension = uploaded_file.name.split('.')[-1].lower()
    if extension not in SUPPORTED_FORMATS:
        return False, f"نوع ملف غير مدعوم / Unsupported format. Use: {', '.join(SUPPORTED_FORMATS)}"
    
    return True, "ملف صالح / Valid file"

def main():
    """Main application"""
    init_app()
    apply_medical_theme()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🩺 مساعد الذكاء الاصطناعي الطبي / Medical AI Assistant</h1>
        <p><strong>نظام متقدم لتحليل الصور الطبية باستخدام نموذج مدرب خصيصاً</strong></p>
        <p><em>Advanced Medical Image Analysis with Fine-tuned Model</em></p>
        <p>🚀 Model: <code>sharawy53/blip-vqa-medical-arabic</code></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    vqa_system = get_vqa_system()
    
    # Load models
    if vqa_system.model is None:
        with st.spinner("🔄 تحميل النموذج المدرب خصيصاً... Loading fine-tuned model..."):
            success = vqa_system.load_models()
            if success:
                st.success("✅ تم تحميل النموذج بنجاح! / Model loaded successfully!")
                st.balloons()
            else:
                st.error("❌ فشل في تحميل النموذج / Failed to load model")
                st.stop()
    
    # Main interface
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📤 رفع الصورة الطبية / Upload Medical Image")
        
        uploaded_file = st.file_uploader(
            "اختر صورة طبية... / Choose medical image...",
            type=SUPPORTED_FORMATS,
            help="الأنواع المدعومة / Supported: JPG, PNG, BMP | حد أقصى / Max: 5MB"
        )
        
        if uploaded_file:
            is_valid, message = validate_file(uploaded_file)
            
            if is_valid:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"📁 {uploaded_file.name}", use_container_width=True)
                    st.info(f"📊 الحجم / Size: {image.size[0]}×{image.size[1]} | النوع / Format: {image.format}")
                except Exception as e:
                    st.error(f"❌ خطأ في تحميل الصورة / Image loading error: {str(e)}")
                    uploaded_file = None
            else:
                st.error(f"❌ {message}")
                uploaded_file = None
    
    with col2:
        st.markdown("### 💭 اسأل سؤالك الطبي / Ask Medical Question")
        
        # Language selector
        language = st.selectbox(
            "اختر اللغة / Select Language:",
            options=["ar", "en"],
            format_func=lambda x: "🇪🇬 العربية" if x == "ar" else "🇺🇸 English"
        )
        
        # Question input
        if language == "ar":
            question_placeholder = "اكتب سؤالك الطبي هنا...\nمثال: ما هو التشخيص المحتمل لهذه الصورة؟"
            question_label = "🔍 السؤال الطبي:"
        else:
            question_placeholder = "Type your medical question here...\nExample: What is the likely diagnosis for this image?"
            question_label = "🔍 Medical Question:"
        
        question = st.text_area(
            question_label,
            height=120,
            placeholder=question_placeholder,
            help="اسأل أسئلة محددة حول الصورة الطبية / Ask specific questions about the medical image"
        )
        
        # Analyze button
        if st.button("🧠 تحليل الصورة الطبية / Analyze Medical Image"):
            if not uploaded_file:
                st.warning("⚠️ يرجى رفع صورة طبية أولاً / Please upload a medical image first")
            elif not question.strip():
                st.warning("⚠️ يرجى كتابة سؤال طبي / Please enter a medical question")
            else:
                with st.spinner("🔬 الذكاء الاصطناعي يحلل الصورة... AI analyzing image..."):
                    try:
                        image = Image.open(uploaded_file)
                        result = vqa_system.process_medical_query(image, question)
                        
                        if result["success"]:
                            # Success results
                            st.markdown("---")
                            st.markdown("### 📋 نتائج التحليل / Analysis Results")
                            
                            # Processing time
                            st.markdown(f"""
                            <div class="processing-time">
                                ⏱️ وقت المعالجة / Processing Time: <strong>{result['processing_time']:.2f} seconds</strong>
                                | 🔍 اللغة المكتشفة / Detected Language: <strong>{'العربية' if result['detected_language'] == 'ar' else 'English'}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Results in two columns
                            res_col1, res_col2 = st.columns(2)
                            
                            with res_col1:
                                st.markdown("""
                                <div class="result-info">
                                    <h4>🇺🇸 English Results</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown(f"**Question:** {result['question_original']}")
                                st.markdown(f"**Answer:** {result['answer_en']}")
                            
                            with res_col2:
                                st.markdown("""
                                <div class="result-success">
                                    <h4 class="arabic-text">🇪🇬 النتائج بالعربية</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown(f"""
                                <div class="arabic-text">
                                    <strong>السؤال:</strong> {result['question_original']}<br>
                                    <strong>الإجابة:</strong> {result['answer_ar']}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Medical disclaimer
                            st.warning("""
                            ⚠️ **تنبيه طبي مهم / Important Medical Disclaimer:**
                            هذا النظام للأغراض التعليمية والبحثية فقط. يجب استشارة طبيب مختص للحصول على تشخيص وعلاج دقيق.
                            
                            This system is for educational and research purposes only. Always consult healthcare professionals for accurate diagnosis and treatment.
                            """)
                            
                        else:
                            st.error(f"❌ فشل في التحليل / Analysis failed: {result.get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"❌ خطأ في المعالجة / Processing error: {str(e)}")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ℹ️ معلومات النظام / System Info")
        
        if vqa_system.model is not None:
            st.success("✅ النموذج: محمل / Model: Loaded")
            st.info(f"🖥️ الجهاز / Device: **{vqa_system.device.upper()}**")
            st.info(f"🤖 النموذج / Model: **{FINE_TUNED_MODEL}**")
        else:
            st.error("❌ النموذج غير محمل / Model not loaded")
        
        st.markdown("---")
        st.markdown("""
        **كيفية الاستخدام / How to use:**
        1. 📤 ارفع صورة طبية / Upload medical image
        2. 🌐 اختر لغتك المفضلة / Select language  
        3. ❓ اكتب سؤالاً طبياً محدداً / Ask specific medical question
        4. 🧠 اضغط تحليل / Click analyze
        
        **الصيغ المدعومة / Supported formats:**
        JPG, JPEG, PNG, BMP
        
        **اللغات المدعومة / Supported languages:**
        - 🇪🇬 العربية / Arabic
        - 🇺🇸 الإنجليزية / English
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>نظام الأسئلة والأجوبة الطبية المرئية v4.0</strong></p>
        <p><strong>Medical VQA System v4.0</strong></p>
        <p>مدعوم بنموذج مدرب خصيصاً | Powered by Fine-tuned Model: <code>sharawy53/blip-vqa-medical-arabic</code></p>
        <p><em>⚠️ للأغراض التعليمية والبحثية فقط - ليس بديلاً عن الاستشارة الطبية المهنية</em></p>
        <p><em>For educational and research purposes only - Not a substitute for professional medical advice</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
