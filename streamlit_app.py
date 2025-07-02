import streamlit as st
from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    MarianTokenizer,
    MarianMTModel
)
import logging

# Disable unnecessary logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Load models (cached for performance)
@st.cache_resource(show_spinner=False)
def load_models():
    # Load VQA model
    processor = AutoProcessor.from_pretrained("Mohamed264/llava-medical-VQA-lora-merged3")
    llava_model = AutoModelForCausalLM.from_pretrained(
        "Mohamed264/llava-medical-VQA-lora-merged3",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load translation models
    ar_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    ar_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    
    en_ar_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    en_ar_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    
    return processor, llava_model, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model

# Translation functions
def translate_ar_to_en(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def translate_en_to_ar(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

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

# Streamlit UI
def main():
    st.set_page_config(page_title="Medical VQA Chatbot", layout="wide")
    st.title("🩺 نموذج طبي ثنائي اللغة (عربي - إنجليزي)")
    st.write("ارفع صورة طبية واسأل بالعربية أو الإنجليزية، وستحصل على الإجابة باللغتين")
    
    # Initialize models
    with st.spinner("جارٍ تحميل النماذج... قد يستغرق هذا بضع دقائق"):
        try:
            processor, llava_model, ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model = load_models()
            st.success("تم تحميل النماذج بنجاح!")
        except Exception as e:
            st.error(f"خطأ في تحميل النماذج: {str(e)}")
            st.stop()
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 ارفع صورة الأشعة")
        uploaded_file = st.file_uploader("Upload medical image", type=["jpg", "png", "jpeg"], label_visibility="visible")
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="الصورة المرفوعة", use_column_width=True)
            except Exception as e:
                st.error(f"خطأ في تحميل الصورة: {str(e)}")
    
    with col2:
        st.subheader("💬 أدخل سؤالك")
        question = st.text_area("Enter your question", placeholder="اكتب سؤالك هنا...", height=150, label_visibility="visible")
        
        if st.button("الحصول على الإجابة", use_container_width=True):
            if not uploaded_file or not question.strip():
                st.warning("يرجى رفع صورة وكتابة سؤال")
            else:
                with st.spinner("جاري معالجة طلبك..."):
                    try:
                        image_pil = Image.open(uploaded_file).convert("RGB")
                        q_ar, q_en, a_ar, a_en = vqa_multilingual(
                            image_pil, question, processor, llava_model, 
                            ar_en_tokenizer, ar_en_model, en_ar_tokenizer, en_ar_model
                        )
                        
                        st.divider()
                        st.subheader("النتائج:")
                        
                        st.markdown(f"**🟠 السؤال بالعربية:**\n{q_ar}")
                        st.markdown(f"**🟢 السؤال بالإنجليزية:**\n{q_en}")
                        st.markdown(f"**🟠 الإجابة بالعربية:**\n{a_ar}")
                        st.markdown(f"**🟢 الإجابة بالإنجليزية:**\n{a_en}")
                        
                    except Exception as e:
                        st.error(f"حدث خطأ أثناء المعالجة: {str(e)}")
    
    # Footer
    st.divider()
    st.caption("ملاحظة: هذا مساعد طبي أولي. استشر طبيبًا متخصصًا للتشخيص الدقيق.")

if __name__ == "__main__":
    main()
