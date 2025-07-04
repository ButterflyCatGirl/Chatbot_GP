import re
import torch
import gc
from PIL import Image
import streamlit as st

def detect_arabic(text):
    """Detect if text contains Arabic characters"""
    arabic_pattern = r'[\u0600-\u06FF]'
    return bool(re.search(arabic_pattern, text))

def optimize_image_for_processing(image, max_size=(800, 800)):
    """Optimize image for BLIP processing"""
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

def clear_gpu_memory():
    """Clear GPU memory cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def format_medical_response(description, language="en"):
    """Format medical response based on language"""
    if language == "ar":
        return f"""
        **التحليل الطبي:**
        {description}
        
        **تنبيه مهم:** هذا التحليل أولي وللمعلومات العامة فقط. 
        يجب استشارة طبيب متخصص للحصول على تشخيص دقيق.
        """
    else:
        return f"""
        **Medical Analysis:**
        {description}
        
        **Important Notice:** This is a preliminary analysis for general information only. 
        Please consult a qualified medical professional for accurate diagnosis.
        """
