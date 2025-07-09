# app.py - Complete Streamlit Medical VQA Chatbot with BLIP Model
import streamlit as st
from PIL import Image, ImageOps
import torch
from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering
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
    """Medical Visual Question Answering System using BLIP"""

    def __init__(self):
        self.processor = None
        self.model = None
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
            try:
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
                logger.info("BLIP processor loaded successfully from Salesforce/blip-vqa-base")
            except Exception as e:
                logger.error(f"Failed to load BLIP processor from Salesforce/blip-vqa-base: {str(e)}")
                return False

            # Load model
            try:
                logger.info("Attempting to load model from Salesforce/blip-vqa-base")
                if self.device == "cpu":
                    self.model = BlipForQuestionAnswering.from_pretrained(
                        "Salesforce/blip-vqa-base",
                        torch_dtype=torch.float32
                    )
                else:
                    self.model = BlipForQuestionAnswering.from_pretrained(
                        "Salesforce/blip-vqa-base",
                        torch_dtype=torch.float16
                    )

                self.model = self.model.to(self.device)
                logger.info(f"BLIP model loaded successfully from Salesforce/blip-vqa-base on {self.device}")
                return True
            except Exception as e:
                logger.error(f"Failed to load model from Salesforce/blip-vqa-base: {str(e)}")
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
        """Process medical VQA query"""
        try:
            if not question or not question.strip():
                raise ValueError("No valid question provided")

            # Preprocess image
            image = self._preprocess_image(image)

            # Process with BLIP model using the original question
            inputs = self.processor(images=image, text=question, return_tensors="pt")

            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )

            # Decode answer
            answer = self.processor.decode(outputs[0], skip_special_tokens=True).strip()

            return {
                "question": question,
                "answer": answer,
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
        <h1>🩺 Medical AI Assistant</h1>
        <p>Advanced multilingual medical image analysis powered by AI</p>
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
                st.error("❌ Failed to load AI models. Please refresh the page and try again. Check the logs for details or ensure an internet connection and compatible dependencies (transformers, torch).")
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

                            st.markdown(f"**Question:** {result['question']}")
                            st.markdown(f"**Answer:** {result['answer']}")

                            # Processing info
                            st.markdown(f"**⏱️ Processing Time:** {processing_time:.2f} seconds")

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
        <p><strong>Medical VQA System v2.0</strong> | Powered by BLIP + Transformers</p>
        <p>⚠️ <em>This system is for educational and research purposes. Not a substitute for professional medical advice.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
