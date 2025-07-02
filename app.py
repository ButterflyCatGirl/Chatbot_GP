import streamlit as st
from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,  # Use AutoModel instead of specific class
    MarianTokenizer,
    MarianMTModel
)
import logging

# Setup logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Load models with enhanced error handling
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        # Load VQA model using AutoModel
        processor = AutoProcessor.from_pretrained("Mohamed264/llava-medical-VQA-lora-merged3")
        llava_model = AutoModelForCausalLM.from_pretrained(
            "Mohamed264/llava-medical-VQA-lora-merged3",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        logger.info("VQA model loaded successfully")
        
        # Load translation models
        ar_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
        ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
        en_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
        en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
        logger.info("Translation models loaded successfully")
        
        return processor, llava_model, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model
        
    except Exception as e:
        logger.exception("Model loading failed")
        st.error(f"Model loading error: {str(e)}")
        st.stop()

# Translation functions
def translate_ar_to_en(text, tokenizer, model):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs)
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except:
        return text  # Return original if translation fails

def translate_en_to_ar(text, tokenizer, model):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs)
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except:
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
    "infection": "عدوى"
}

def translate_answer_medical(answer_en, en_ar_tokenizer, en_ar_model):
    key = answer_en.lower().strip()
    if key in medical_terms:
        return medical_terms[key]
    else:
        return translate_en_to_ar(answer_en, en_ar_tokenizer, en_ar_model)

# Main function
def vqa_multilingual(image, question, processor, llava_model, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model):
    try:
        # Check if Arabic
        is_arabic = any('\u0600' <= c <= '\u06FF' for c in question)
        
        if is_arabic:
            question_ar = question.strip()
            question_en = translate_ar_to_en(question_ar, ar_en_tokenizer, ar_en_model)
        else:
            question_en = question.strip()
            question_ar = translate_en_to_ar(question_en, en_ar_tokenizer, en_ar_model)
        
        # Process with VQA model
        inputs = processor(text=question_en, images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = llava_model.generate(**inputs, max_new_tokens=200)
        answer_en = processor.decode(output[0], skip_special_tokens=True).strip()
        
        # Translate answer
        answer_ar = translate_answer_medical(answer_en, en_ar_tokenizer, en_ar_model)
        
        return question_ar, question_en, answer_ar, answer_en
        
    except Exception as e:
        logger.exception("VQA processing failed")
        return f"Error: {str(e)}", "", "", ""

# Streamlit UI
def main():
    st.set_page_config(page_title="Medical VQA Chatbot", layout="wide")
    st.title("🩺 Medical VQA Chatbot (عربي/English)")
    st.write("Upload a medical image and ask questions in Arabic or English")
    
    # Initialize models
    with st.spinner("Loading AI models... This may take a few minutes"):
        try:
            processor, llava_model, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model = load_models()
            st.success("Models loaded successfully!")
        except Exception as e:
            st.error(f"Critical error: {str(e)}")
            st.stop()
    
    # File upload
    uploaded_file = st.file_uploader("Upload Medical Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
    
    # Question input
    question = st.text_area("Your Question", placeholder="Type your question here...", height=100)
    
    if st.button("Get Answer", use_container_width=True):
        if not uploaded_file or not question.strip():
            st.warning("Please upload an image and enter a question")
        else:
            with st.spinner("Processing your question..."):
                try:
                    image_pil = Image.open(uploaded_file).convert("RGB")
                    q_ar, q_en, a_ar, a_en = vqa_multilingual(
                        image_pil, question, processor, llava_model, 
                        ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model
                    )
                    
                    # Display results
                    st.divider()
                    st.subheader("Results:")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Arabic Question:**")
                        st.info(q_ar)
                        st.markdown("**Arabic Answer:**")
                        st.success(a_ar)
                    
                    with col2:
                        st.markdown("**English Question:**")
                        st.info(q_en)
                        st.markdown("**English Answer:**")
                        st.success(a_en)
                    
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")
    
    # Footer
    st.divider()
    st.caption("Note: This is a preliminary medical assistant. Always consult a healthcare professional for diagnosis.")

if __name__ == "__main__":
    main()
