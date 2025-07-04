import os

# Model Configuration
MODEL_NAME = "Salesforce/blip-image-captioning-large"
MAX_IMAGE_SIZE = (800, 800)
IMAGE_QUALITY = 85

# API Configuration
API_TIMEOUT = 30
MAX_RETRIES = 3

# Language Configuration
DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = ["en", "ar"]

# Memory Management
ENABLE_GPU = True
CLEAR_CACHE_AFTER_ANALYSIS = True

# Logging
LOG_LEVEL = "INFO"
