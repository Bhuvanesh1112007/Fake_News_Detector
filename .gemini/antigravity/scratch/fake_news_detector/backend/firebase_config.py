"""
Firebase Configuration and Client Manager
Provides credentials loading and connection management for Firebase Firestore.
"""
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIREBASE_CONFIG_PATH = os.path.join(BASE_DIR, "firebase_config.json")

DEFAULT_FIREBASE_CONFIG = {
    "apiKey": os.getenv("FIREBASE_API_KEY", "AIzaSyFakeNewsDetectorDemoKey2026"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", "verishield-ai.firebaseapp.com"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID", "verishield-ai"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", "verishield-ai.appspot.com"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", "1029384756"),
    "appId": os.getenv("FIREBASE_APP_ID", "1:1029384756:web:abcdef123456"),
    "firestoreCollection": "predictions"
}

def get_firebase_config() -> dict:
    """Loads Firebase configuration from file or returns default."""
    if os.path.exists(FIREBASE_CONFIG_PATH):
        try:
            with open(FIREBASE_CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                return {**DEFAULT_FIREBASE_CONFIG, **config}
        except Exception:
            pass
    return DEFAULT_FIREBASE_CONFIG

def is_firebase_configured() -> bool:
    """Checks if external Firebase project credentials are provided."""
    return os.path.exists(FIREBASE_CONFIG_PATH) or "FIREBASE_PROJECT_ID" in os.environ
