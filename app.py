import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
import base64
import requests
import json
import time
import gc
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    st.warning("psutil not available. Memory monitoring disabled.")

# Configure page
st.set_page_config(
    page_title="🏥 المساعد الطبي الذكي | AI Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Egyptian Medical Terms Dictionary
MEDICAL_TERMS_AR = {
    "headache": "صداع",
    "fever": "حمى",
    "cough": "كحة",
    "pain": "ألم",
    "stomach": "معدة",
    "heart": "قلب",
    "blood pressure": "ضغط الدم",
    "diabetes": "السكري",
    "infection": "عدوى",
    "medication": "دواء",
    "doctor": "دكتور",
    "hospital": "مستشفى",
    "treatment": "علاج",
    "symptoms": "أعراض",
    "diagnosis": "تشخيص"
}

# Enhanced Egyptian Arabic responses
ARABIC_RESPONSES = [
    "بناءً على الصورة اللي بعتهالي، أقدر أشوف {description}. انصحك تستشير دكتور متخصص عشان يقدر يساعدك أكتر.",
    "الصورة دي بتوضح {description}. ده مجرد تحليل أولي، والأفضل تروح لدكتور مختص عشان تاخد رأي طبي صحيح.",
    "من خلال الصورة، أقدر أقولك إن في {description}. بس خليني أذكرك إن ده مش بديل عن الكشف الطبي المباشر.",
    "الصورة بتبين {description}. نصيحتي ليك تروح لأقرب مستشفى أو عيادة عشان دكتور متخصص يشوفك.",
    "حسب اللي شايفه في الصورة، في {description}. مهم جداً تاخد رأي طبي متخصص قبل أي خطوة."
]

ENGLISH_RESPONSES = [
    "Based on the image you've shared, I can see {description}. I recommend consulting with a healthcare professional for proper medical advice.",
    "The image shows {description}. This is a preliminary analysis, and it's best to see a specialist doctor for accurate medical opinion.",
    "From the image, I can tell you that there is {description}. However, please remember this is not a substitute for direct medical examination.",
    "The image reveals {description}. My advice is to visit the nearest hospital or clinic for a specialist doctor to examine you.",
    "According to what I see in the image, there is {description}. It's very important to get specialized medical opinion before taking any steps."
]

class MemoryManager:
    @staticmethod
    def clear_cache():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_info():
        if not PSUTIL_AVAILABLE:
            return "Memory info unavailable"
        
        memory = psutil.virtual_memory()
        return f"Memory: {memory.percent}% used"

@st.cache_resource
def load_model():
    """Load and cache the BLIP model"""
    try:
        with st.spinner("🔄 Loading AI model... | جاري تحميل النموذج الذكي..."):
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            
            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            
            return processor, model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, "cpu"

def detect_language(text):
    """Detect if text contains Arabic characters"""
    arabic_pattern = r'[\u0600-\u06FF]'
    import re
    return 'ar' if re.search(arabic_pattern, text) else 'en'

def optimize_image(image, max_size=(800, 800), quality=85):
    """Optimize image for processing"""
    # Convert to RGB if necessary
    if image.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'P':
            image = image.convert('RGBA')
        background.paste(image, mask=image.split()[-1] if 'A' in image.mode else None)
        image = background
    
    # Resize if too large
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    return image

def analyze_image(image, user_question="", language="en"):
    """Analyze image using BLIP model"""
    processor, model, device = load_model()
    
    if model is None:
        return "❌ Model loading failed | فشل في تحميل النموذج"
    
    try:
        # Optimize image
        optimized_image = optimize_image(image)
        
        # Generate medical-focused prompt
        if language == 'ar':
            medical_prompt = "وصف طبي مفصل لهذة الصورة:"
        else:
            medical_prompt = "detailed medical description of this image:"
        
        # Process image
        inputs = processor(optimized_image, medical_prompt, return_tensors="pt").to(device)
        
        # Generate description
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=150,
                num_beams=5,
                temperature=0.7,
                do_sample=True,
                early_stopping=True
            )
        
        description = processor.decode(out[0], skip_special_tokens=True)
        
        # Remove the prompt from the description
        if medical_prompt in description:
            description = description.replace(medical_prompt, "").strip()
        
        # Clean up memory
        MemoryManager.clear_cache()
        
        # Generate contextual response
        if language == 'ar':
            import random
            response_template = random.choice(ARABIC_RESPONSES)
            return response_template.format(description=description)
        else:
            import random
            response_template = random.choice(ENGLISH_RESPONSES)
            return response_template.format(description=description)
            
    except Exception as e:
        error_msg = f"Error analyzing image: {str(e)}"
        if language == 'ar':
            return f"❌ حدث خطأ في تحليل الصورة: {error_msg}"
        return f"❌ {error_msg}"

def main():
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'language' not in st.session_state:
        st.session_state.language = 'en'

    # Sidebar
    with st.sidebar:
        st.markdown("### 🇪🇬 اللغة | Language")
        
        # Language selector
        language_options = {
            "🇪🇬 العربية (مصر)": "ar",
            "🇺🇸 English (US)": "en"
        }
        
        selected_lang = st.selectbox(
            "Choose Language:",
            options=list(language_options.keys()),
            index=0 if st.session_state.language == 'ar' else 1
        )
        st.session_state.language = language_options[selected_lang]
        
        st.markdown("---")
        
        # System info
        if PSUTIL_AVAILABLE:
            st.markdown(f"**System:** {MemoryManager.get_memory_info()}")
        
        device_info = "🔥 GPU" if torch.cuda.is_available() else "💻 CPU"
        st.markdown(f"**Device:** {device_info}")
        
        st.markdown("---")
        
        # Clear chat button
        if st.session_state.language == 'ar':
            if st.button("🗑️ مسح المحادثة", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        else:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    # Main content
    if st.session_state.language == 'ar':
        st.markdown("""
        <div style='text-align: right; direction: rtl;'>
            <h1>🏥 المساعد الطبي الذكي المصري</h1>
            <p>مساعد ذكي لتحليل الصور الطبية باستخدام تقنيات الذكاء الاصطناعي المتقدمة</p>
            <p><strong>تنبيه:</strong> هذا التطبيق للاستخدام التعليمي فقط وليس بديلاً عن الاستشارة الطبية المهنية</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        # 🏥 Egyptian AI Medical Assistant
        
        An intelligent assistant for medical image analysis using advanced AI technology.
        
        **Disclaimer:** This application is for educational purposes only and is not a substitute for professional medical consultation.
        """)

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user" and "image" in message:
                st.image(message["image"], width=300)
            st.markdown(message["content"])

    # File uploader
    if st.session_state.language == 'ar':
        uploaded_file = st.file_uploader(
            "📤 ارفع صورة طبية للتحليل",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="الصيغ المدعومة: PNG, JPG, JPEG, BMP, TIFF"
        )
    else:
        uploaded_file = st.file_uploader(
            "📤 Upload Medical Image for Analysis",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF"
        )

    # Text input for questions
    if st.session_state.language == 'ar':
        user_question = st.chat_input("اكتب سؤالك الطبي هنا...")
    else:
        user_question = st.chat_input("Type your medical question here...")

    # Process uploaded image
    if uploaded_file is not None:
        try:
            # Load and display image
            image = Image.open(uploaded_file)
            
            # Add user message with image
            user_msg = {
                "role": "user",
                "content": f"📸 Image uploaded: {uploaded_file.name}",
                "image": image
            }
            st.session_state.messages.append(user_msg)
            
            # Display user message
            with st.chat_message("user"):
                st.image(image, width=300)
                st.markdown(user_msg["content"])
            
            # Analyze image
            with st.chat_message("assistant"):
                with st.spinner("🔍 Analyzing image... | جاري تحليل الصورة..."):
                    analysis = analyze_image(image, "", st.session_state.language)
                    st.markdown(analysis)
                    
                    # Add assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": analysis
                    })
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    # Process text question
    if user_question:
        # Detect language of the question
        detected_lang = detect_language(user_question)
        
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": user_question
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Generate response
        with st.chat_message("assistant"):
            if detected_lang == 'ar':
                response = """
                شكراً لسؤالك. أنا مساعد طبي ذكي متخصص في تحليل الصور الطبية. 
                
                لأقدر أساعدك بشكل أفضل، ارفع صورة طبية وأنا هحللها ليك وأديك معلومات مفيدة.
                
                **تذكر:** هذا التحليل للمعلومات العامة فقط وليس بديل عن زيارة الطبيب.
                """
            else:
                response = """
                Thank you for your question. I'm an AI medical assistant specialized in medical image analysis.
                
                To help you better, please upload a medical image and I'll analyze it for you and provide useful information.
                
                **Remember:** This analysis is for general information only and is not a substitute for visiting a doctor.
                """
            
            st.markdown(response)
            
            # Add assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

if __name__ == "__main__":
    main()
