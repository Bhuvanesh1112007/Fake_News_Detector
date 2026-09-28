"""
Text Preprocessor & Linguistic Cleaner
Provides text normalization, sanitization, and linguistic metric extraction.
"""
import re
import string
import unicodedata

class TextPreprocessor:
    def __init__(self):
        # Common sensationalist stop triggers and clickbait indicators
        self.clickbait_triggers = {
            "shocking", "unbelievable", "secret", "exposed", "banned",
            "miracle", "sheeple", "censored", "cure", "truth", "viral",
            "whistleblower", "conspiracy", "cabal", "underground", "leaked",
            "hate this", "big pharma", "mind-control", "5g", "blackout"
        }

    def clean_text(self, text: str) -> str:
        """Sanitizes raw article text for language modeling."""
        if not text or not isinstance(text, str):
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize("NFKD", text)
        
        # Remove URL patterns
        text = re.sub(r'http\S+|www\.\S+|ftp\S+', ' ', text)
        
        # Remove HTML tags if present
        text = re.sub(r'<.*?>', ' ', text)
        
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_linguistic_stats(self, raw_text: str) -> dict:
        """
        Extracts stylometric and linguistic features from raw text before aggressive cleaning.
        High uppercase ratio, excessive punctuation, and exclamation marks correlate with fake/sensational news.
        """
        if not raw_text or not isinstance(raw_text, str):
            return {
                "char_count": 0,
                "word_count": 0,
                "uppercase_ratio": 0.0,
                "exclamation_count": 0,
                "question_count": 0,
                "digit_count": 0,
                "sensational_word_count": 0,
                "lexical_diversity": 0.0
            }

        total_chars = len(raw_text)
        uppercase_chars = sum(1 for c in raw_text if c.isupper())
        exclamation_marks = raw_text.count("!")
        question_marks = raw_text.count("?")
        digits = sum(1 for c in raw_text if c.isdigit())
        
        # Word-level features
        words = re.findall(r'\b[A-Za-z0-9\'-]+\b', raw_text)
        word_count = len(words)
        unique_words = set(w.lower() for w in words)
        
        sensational_hits = sum(
            1 for trigger in self.clickbait_triggers 
            if trigger in raw_text.lower()
        )

        return {
            "char_count": total_chars,
            "word_count": word_count,
            "uppercase_ratio": uppercase_chars / max(1, total_chars),
            "exclamation_count": exclamation_marks,
            "question_count": question_marks,
            "digit_count": digits,
            "sensational_word_count": sensational_hits,
            "lexical_diversity": len(unique_words) / max(1, word_count)
        }
