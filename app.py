# app.py
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
import psutil
import os

# Setup logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Egyptian Medical Terminology Dictionary
EGYPTIAN_MEDICAL_TERMS = {
    "chest x-ray": "أشعة على الصدر",
    "x-ray": "أشعة سينية", 
    "ct scan": "أشعة مقطعية",
    "mri": "رنين مغناطيسي",
    "ultrasound": "سونار",
    "normal": "طبيعي",
    "abnormal": "مش طبيعي",
    "healthy": "سليم",
    "no abnormality detected": "مافيش حاجة غلط",
    "no issues found": "مافيش مشاكل",
    "everything looks fine": "كله تمام",
    "consultation needed": "محتاج استشارة دكتور"
}

# Enhanced Egyptian Arabic responses
EGYPTIAN_RESPONSES = {
    "general_health": ["الصحة العامة تبدو كويسة إن شاء الله", "بالنظر للصورة، الوضع يبدو طبيعي"],
    "chest_normal": ["الصدر سليم والحمد لله", "الرئتين شغالين كويس"],
    "needs_consultation": ["أنصحك تستشير دكتور متخصص", "الأفضل تعمل كشف عند دكتور"],
    "bone_health": ["العضام تبدو سليمة", "مافيش كسور واضحة"]
}

@st.cache_resource(show_spinner=False)
def load_models():
    """Load BLIP and translation models with enhanced error handling"""
    try:
        # Load BLIP processor and model
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        # Load translation models
        ar_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
        ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
        en_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
        en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
        logger.info("All models loaded successfully")
        return processor, model, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model
    except Exception as e:
        logger.exception("Critical model loading error")
        st.error(f"خطأ في تحميل النماذج: {str(e)}")
        st.stop()

def detect_language(text):
    """Detect if text is Arabic or English"""
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    return 'ar' if arabic_chars > len(text) * 0.3 else 'en'

def translate_medical_response(answer_en, en_ar_tokenizer, en_ar_model):
    """Enhanced medical translation with Egyptian terminology"""
    answer_lower = answer_en.lower().strip()
    # First check for exact matches in Egyptian medical terms
    if answer_lower in EGYPTIAN_MEDICAL_TERMS:
        return EGYPTIAN_MEDICAL_TERMS[answer_lower]
    # Fallback to machine translation
    try:
        inputs = en_ar_tokenizer(answer_en, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = en_ar_model.generate(**inputs, max_length=512, num_beams=2, early_stopping=True)
        return en_ar_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return answer_en

def resize_image(image, max_size=512):
    """Resize image to reduce memory usage"""
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image

def process_medical_query(image, question, processor, model, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model):
    """Process medical image and question with enhanced Egyptian Arabic support"""
    try:
        # Detect question language
        detected_lang = detect_language(question)
        # Prepare questions in both languages
        if detected_lang == 'ar':
            question_ar = question.strip()
            question_en = translate_ar_to_en(question_ar, ar_en_tokenizer, ar_en_model)
        else:
            question_en = question.strip()
            question_ar = translate_en_to_ar(question_en, en_ar_tokenizer, en_ar_model)
        # Resize image to save memory
        image_resized = resize_image(image)
        # Process with BLIP model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = processor(image_resized, question_en, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                num_beams=3,
                early_stopping=True,
                do_sample=False
            )
        # Decode answer
        answer_en = processor.decode(outputs[0], skip_special_tokens=True).strip()
        answer_ar = translate_medical_response(answer_en, en_ar_tokenizer, en_ar_model)
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return question_ar, question_en, answer_ar, answer_en
    except Exception as e:
        logger.exception("Medical query processing failed")
        error_msg = f"خطأ في المعالجة: {str(e)}"
        return error_msg, f"Processing error: {str(e)}", error_msg, f"Processing error: {str(e)}"

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def main():
    st.set_page_config(
        page_title="مساعد الصحة المصري - Egyptian Health Assistant",
        layout="wide",
        page_icon="🇪🇬",
        initial_sidebar_state="expanded"
    )

    # Header with Egyptian theme
    st.markdown("""
    <div class="main-header">
        <h1>🇪🇬 مساعد الصحة المصري 🇪🇬</h1>
        <h2>Egyptian Medical AI Assistant</h2>
        <p>مساعد ذكي للتشخيص الطبي باللغة العربية المصرية</p>
        <p>AI-powered medical diagnosis in Egyptian Arabic</p>
    </div>
    """, unsafe_allow_html=True)

    # Load models with progress indicator
    with st.spinner("🔄 جاري تحميل النماذج الطبية... Loading medical AI models..."):
        try:
            processor, model, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model = load_models()
            st.success("✅ تم تحميل النماذج بنجاح! Models loaded successfully!")
        except Exception as e:
            st.error(f"❌ خطأ في التحميل: {str(e)}")
            st.stop()

    # Main interface
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("""
        <div class="card">
            <h3 style="text-align: center; margin-bottom: 1.5rem;">
                📷 ارفع الصورة الطبية<br>Upload Medical Image
            </h3>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "اختر صورة طبية / Choose medical image", 
            type=["jpg", "png", "jpeg", "bmp", "tiff"],
            help="ارفع أشعة أو صور طبية للتشخيص / Upload X-rays or medical images for diagnosis"
        )
        if uploaded_file:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="الصورة المرفوعة / Uploaded Image", use_container_width=True)
            except Exception as e:
                st.error(f"خطأ في فتح الصورة: {str(e)}")

    with col2:
        st.markdown("""
        <div class="card">
            <h3 style="text-align: center; margin-bottom: 1.5rem;">
                💬 اسأل سؤالك الطبي<br>Ask Your Medical Question
            </h3>
        </div>
        """, unsafe_allow_html=True)
        st.info("💡 اكتب سؤالك بالعربية أو الإنجليزية / Write your question in Arabic or English")
        question = st.text_area(
            "سؤالك الطبي / Your medical question:",
            placeholder="مثال: إيه اللي باين في الأشعة دي؟\nExample: What do you see in this X-ray?",
            height=120,
            help="اكتب سؤالك بوضوح عن الصورة الطبية / Write your question clearly about the medical image"
        )
        # Analyze button
        if st.button("🔍 تحليل طبي / Medical Analysis", use_container_width=True):
            if not uploaded_file or not question.strip():
                st.warning("⚠️ يرجى رفع صورة وكتابة سؤال / Please upload an image and enter a question")
            else:
                with st.spinner("🔄 جاري التحليل الطبي... Analyzing medical image..."):
                    try:
                        start_time = time.time()
                        image_pil = Image.open(uploaded_file).convert("RGB")
                        q_ar, q_en, a_ar, a_en = process_medical_query(
                            image_pil, question, processor, model,
                            ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model
                        )
                        processing_time = time.time() - start_time

                        # Display results
                        st.markdown("""
                        <div class="result-container">
                            <h2 style="text-align: center; margin-bottom: 2rem;">
                                📋 نتائج التحليل الطبي / Medical Analysis Results
                            </h2>
                        </div>
                        """, unsafe_allow_html=True)
                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.markdown("### 🇪🇬 العربية / Arabic")
                            st.markdown(f'<div class="arabic-text">{q_ar}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="arabic-text" style="background: #e8f5e8; padding: 1rem; border-radius: 10px; color: #2d5a2d;">{a_ar}</div>', unsafe_allow_html=True)
                        with res_col2:
                            st.markdown("### 🇺🇸 English")
                            st.markdown(f'<div class="english-text">{q_en}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="english-text" style="background: #e8f5e8; padding: 1rem; border-radius: 10px; color: #2d5a2d;">{a_en}</div>', unsafe_allow_html=True)
                        memory_usage = get_memory_usage()
                        st.markdown(f"""
                        <div class="memory-info">
                            ⏱️ وقت المعالجة: {processing_time:.2f} ثانية / Processing time: {processing_time:.2f} seconds<br>
                            💾 استخدام الذاكرة: {memory_usage:.1f} MB / Memory usage: {memory_usage:.1f} MB
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ خطأ في المعالجة: {str(e)}")

    # Footer with medical disclaimer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 15px; margin-top: 2rem;">
        <h4>🏥 إخلاء مسؤولية طبية / Medical Disclaimer</h4>
        <p><strong>هذا المساعد الذكي يوفر معلومات أولية فقط ولا يحل محل الاستشارة الطبية المتخصصة</strong></p>
        <p><strong>This AI assistant provides preliminary information only and does not replace professional medical consultation</strong></p>
        <p>🔬 مدعوم بتقنية BLIP للذكاء الاصطناعي الطبي / Powered by BLIP Medical AI Technology</p>
        <p>🇪🇬 صُمم خصيصاً للمجتمع المصري / Designed specifically for the Egyptian community</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
