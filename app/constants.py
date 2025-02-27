SAMPLE_RATE = 16000
BLOCK_SIZE = 1024
CHANNELS = 1
TRANSCRIPTION_CHUNK_DURATION = 5

AVAILABLE_MODELS = ['tiny', 'base', 'small', 'medium', 'large']
AVAILABLE_TRANSLATIONS = ['None', 'German', 'Mandarin', 'Spanish', 'French', 'Japanese']
TRANSLATION_LANGUAGE_CODES = {
    'None': None,
    'German': 'de',
    'Mandarin': 'zh-cn',
    'Spanish': 'es',
    'French': 'fr',
    'Japanese': 'ja'
}
AVAILABLE_SOURCE_LANGUAGES = [
    'Auto', 'English', 'German', 'Mandarin', 'Spanish',
    'French', 'Japanese', 'Korean', 'Italian', 'Russian',
    'Portuguese', 'Dutch', 'Arabic', 'Hindi', 'Bengali'
]
SOURCE_LANGUAGE_CODES = {
    'Auto': None,
    'English': 'en',
    'German': 'de',
    'Mandarin': 'zh',
    'Spanish': 'es',
    'French': 'fr',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Italian': 'it',
    'Russian': 'ru',
    'Portuguese': 'pt',
    'Dutch': 'nl',
    'Arabic': 'ar',
    'Hindi': 'hi',
    'Bengali': 'bn'
}