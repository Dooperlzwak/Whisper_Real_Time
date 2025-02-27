import json
import os
import logging
import customtkinter as ctk

logger = logging.getLogger("TranscriptionApp")

def load_preferences():
    if os.path.exists("preferences.json"):
        try:
            with open("preferences.json", "r") as f:
                prefs = json.load(f)
            logger.info("User preferences loaded.")
            return prefs
        except Exception as e:
            logger.error(f"Failed to load preferences: {e}", exc_info=True)
            return {}
    else:
        logger.info("No preferences file found. Using default settings.")
        return {}

def save_preferences(prefs):
    try:
        with open("preferences.json", "w") as f:
            json.dump(prefs, f)
        logger.info("User preferences saved.")
        return True
    except Exception as e:
        logger.error(f"Failed to save preferences: {e}", exc_info=True)
        return False

def apply_preferences(preferences):
    mode = preferences.get('appearance_mode')
    if mode:
        ctk.set_appearance_mode(mode)
        logger.info(f"Applied appearance mode: {mode}")
    scheme = preferences.get('color_scheme')
    if scheme and scheme in ['blue', 'green', 'dark-blue', 'purple', 'dark']:
        ctk.set_default_color_theme(scheme)
        logger.info(f"Applied color scheme: {scheme}")