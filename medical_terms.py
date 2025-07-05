# Medical terminology and response templates

MEDICAL_TERMS_AR = {
    # Body parts
    "heart": "القلب",
    "lung": "الرئة", 
    "lungs": "الرئتين",
    "brain": "المخ",
    "liver": "الكبد",
    "kidney": "الكلية",
    "stomach": "المعدة",
    "chest": "الصدر",
    "spine": "العمود الفقري",
    "bone": "العظم",
    "bones": "العظام",
    "rib": "الضلع",
    "ribs": "الأضلاع",
    
    # Medical conditions
    "fracture": "كسر",
    "pneumonia": "التهاب رئوي",
    "tumor": "ورم",
    "infection": "عدوى",
    "inflammation": "التهاب",
    "abnormal": "غير طبيعي",
    "normal": "طبيعي",
    "mass": "كتلة",
    "lesion": "آفة",
    "swelling": "تورم",
    
    # Medical imaging
    "x-ray": "أشعة سينية",
    "ct scan": "أشعة مقطعية",
    "mri": "رنين مغناطيسي",
    "ultrasound": "موجات فوق صوتية",
    "scan": "فحص",
    "image": "صورة",
    "medical image": "صورة طبية",
    
    # Medical terms
    "diagnosis": "تشخيص",
    "treatment": "علاج",
    "medication": "دواء",
    "doctor": "طبيب",
    "hospital": "مستشفى",
    "patient": "مريض",
    "symptoms": "أعراض",
    "pain": "ألم",
    "fever": "حمى"
}

def get_medical_translation(text, target_lang="ar"):
    """Translate medical terms in text"""
    if target_lang == "ar":
        translated_text = text
        for en_term, ar_term in MEDICAL_TERMS_AR.items():
            translated_text = translated_text.replace(en_term, f"{en_term} ({ar_term})")
        return translated_text
    return text

def get_medical_response_template(lang="en", user_input=""):
    """Get appropriate medical response template"""
    
    templates = {
        "en": {
            "general": """
Based on the medical image analysis, I can provide general observations. However, please remember:

🏥 **Important Medical Disclaimer:**
- This is an AI analysis for informational purposes only
- Always consult with qualified healthcare professionals
- Do not use this for medical diagnosis or treatment decisions
- Seek immediate medical attention for urgent health concerns

For a proper medical evaluation, please visit a licensed healthcare provider.
            """,
            "image_analysis": """
**Medical Image Analysis:**
The image shows various anatomical structures and findings. Key observations include medical features that should be evaluated by a qualified radiologist or physician.

⚠️ **Medical Disclaimer:** This AI analysis is not a substitute for professional medical interpretation. Please consult with healthcare professionals for accurate diagnosis.
            """
        },
        "ar": {
            "general": """
بناءً على تحليل الصورة الطبية، يمكنني تقديم ملاحظات عامة. ولكن يرجى تذكر:

🏥 **إخلاء مسؤولية طبية مهم:**
- هذا تحليل ذكي لأغراض إعلامية فقط
- استشر دائماً أخصائيي الرعاية الصحية المؤهلين
- لا تستخدم هذا للتشخيص الطبي أو قرارات العلاج
- اطلب العناية الطبية الفورية للمخاوف الصحية العاجلة

للحصول على تقييم طبي مناسب، يرجى زيارة مقدم رعاية صحية مرخص.
            """,
            "image_analysis": """
**تحليل الصورة الطبية:**
تُظهر الصورة هياكل تشريحية ونتائج مختلفة. تشمل الملاحظات الرئيسية الميزات الطبية التي يجب تقييمها من قبل أخصائي أشعة أو طبيب مؤهل.

⚠️ **إخلاء مسؤولية طبية:** هذا التحليل الذكي ليس بديلاً عن التفسير الطبي المهني. يرجى استشارة أخصائيي الرعاية الصحية للحصول على تشخيص دقيق.
            """
        }
    }
    
    # Determine response type
    if "image" in user_input.lower() or not user_input:
        return templates[lang]["image_analysis"]
    else:
        return templates[lang]["general"]

def get_medical_context_prompt(lang="en"):
    """Get medical context prompt for BLIP2"""
    prompts = {
        "en": "Analyze this medical image in detail. Describe anatomical structures, any visible abnormalities, and potential medical findings. Use proper medical terminology.",
        "ar": "حلل هذه الصورة الطبية بالتفصيل. صف الهياكل التشريحية وأي تشوهات مرئية ونتائج طبية محتملة. استخدم المصطلحات الطبية المناسبة."
    }
    return prompts.get(lang, prompts["en"])
