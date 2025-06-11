# from deep_translator import GoogleTranslator
# from langdetect import detect

# def detect_language(text: str) -> str:
#     """Detect language of input text"""
#     try:
#         return detect(text)
#     except:
#         return 'en'  

# def translate_to_english(text: str, source_lang: str = None) -> str:
#     """Translate text to English"""
#     if not source_lang:
#         source_lang = detect_language(text)
    
#     if source_lang == 'en':
#         return text
    
#     try:
#         return GoogleTranslator(source=source_lang, target='en').translate(text)
#     except:
#         # Fallback: try auto-detect
#         return GoogleTranslator(source='auto', target='en').translate(text)

# def translate_from_english(text: str, target_lang: str) -> str:
#     """Translate from English to target language"""
#     if target_lang == 'en':
#         return text
    
#     try:
#         return GoogleTranslator(source='en', target=target_lang).translate(text)
#     except:
#         return text  


# KODE DUA
from deep_translator import GoogleTranslator
from langdetect import detect

def detect_language(text: str) -> str:
    """Detect language of input text"""
    try:
        detected = detect(text)
        # Map 'id' to 'indonesian' for consistency
        if detected == 'id':
            return 'indonesian'
        return detected
    except:
        return 'en'  

def is_indonesian(text: str) -> bool:
    """Check if text is in Indonesian language"""
    detected_lang = detect_language(text)
    return detected_lang in ['indonesian', 'id']

def translate_to_english(text: str, source_lang: str = None) -> str:
    """Translate text to English"""
    if not source_lang:
        source_lang = detect_language(text)
    
    # If it's already English, return as is
    if source_lang == 'en':
        return text
    
    try:
        # Handle Indonesian language code mapping
        source_code = 'id' if source_lang == 'indonesian' else source_lang
        return GoogleTranslator(source=source_code, target='en').translate(text)
    except:
        # Fallback: try auto-detect
        try:
            return GoogleTranslator(source='auto', target='en').translate(text)
        except:
            return text

def translate_from_english(text: str, target_lang: str) -> str:
    """Translate from English to target language"""
    if target_lang == 'en':
        return text
    
    try:
        # Handle Indonesian language code mapping
        target_code = 'id' if target_lang == 'indonesian' else target_lang
        return GoogleTranslator(source='en', target=target_code).translate(text)
    except:
        return text

def translate_to_target_language(text: str, target_lang: str, source_lang: str = 'en') -> str:
    """General translation function"""
    if source_lang == target_lang:
        return text
    
    try:
        # Handle Indonesian language code mapping
        source_code = 'id' if source_lang == 'indonesian' else source_lang
        target_code = 'id' if target_lang == 'indonesian' else target_lang
        
        return GoogleTranslator(source=source_code, target=target_code).translate(text)
    except:
        return text