"""
Data Preprocessing Pipeline Module (Phase 2)
Module Owner: Mohamed Irfaan M (NLP Lead) & Bhuvanesh M (Data Lead)
Handles data ingestion, missing-value imputation, deduplication, text sanitization,
label normalization, stratified splitting (Train/Val/Test), and artifact persistence.
"""
import os
import csv
import json
import re
import unicodedata
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from .config import (
    RAW_DATASET_PATH, PROCESSED_DATA_DIR, PROCESSED_TRAIN_PATH,
    PROCESSED_VAL_PATH, PROCESSED_TEST_PATH, RANDOM_SEED,
    TEST_SPLIT_SIZE, VAL_SPLIT_SIZE
)

class TextPreprocessor:
    """
    Standardized Text Preprocessor:
    - Unicode NFKD normalization
    - URL, email, and HTML tag sanitization
    - Stylometric metric extraction
    - Identical pipeline across training and inference
    """
    def __init__(self):
        self.deceptive_triggers = {
            "shocking", "unbelievable", "secret", "exposed", "banned", "cabal",
            "miracle", "cure", "cures", "onion juice", "baldness", "bleach", "alien",
            "aliens", "invasion", "bunker", "giveaway", "free money", "bitcoin", "btc",
            "retweet", "click here", "100%", "overnight", "doctors hate", "big pharma",
            "mind-control", "mind control", "hypnotized", "flee", "stolen gold", "dinosaur", "frozen in ice",
            "microchip", "microchips", "weather machine", "chemtrail", "chemtrails",
            "teleportation", "simulation", "blackout", "martial law", "subterranean",
            "baking soda", "copper bracelet", "parasite", "detox", "potion", "gross video",
            "censored", "leaked", "whistleblower", "conspiracy", "hate this", "elites",
            "suppressed", "corrupt", "undercover", "wake up sheeple", "sheeple", "globalist",
            "illuminati", "deep state", "hoax", "5g", "radiation frequency", "radiation frequencies", "secret recipe",
            "they are trying to delete", "free cash", "claim your free", "share before deletion",
            "urgent report", "furious", "silenced", "poisoned", "banned from revealing",
            "dissolves", "never need chemotherapy", "without surgery", "reverses aging",
            "eliminate wrinkles", "miraculous", "instant cure", "fakes death", "titanium skeleton",
            "android robot", "cloned", "secret laboratory", "assassinated", "poisoned by",
            "moon will disappear", "disappear from earth's sky", "live without sleeping", "without sleeping",
            "receive \u20b91 lakh", "free 50000", "free 500000", "deposited in bank", "without application",
            "forwarding a message on whatsapp", "on whatsapp", "whatsapp link", "whatsapp group", "whatsapp users",
            "forwards this", "forward this", "free laptop", "free laptops", "free iphone", "free smartphone",
            "10 years without charging", "without charging", "double human intelligence", "water after midnight", "after midnight",
            "time traveler", "from 2085", "from the future", "worldwide blackout", "shut down permanently",
            "viral video shows", "viral video", "secret clinic", "cures diabetes", "banned cardiologists",
            "kerosene", "mind-altering parasites", "free bitcoin codes", "digital wallet before midnight",
            "raw garlic and vinegar", "raw onions prevents", "kills coronavirus", "destroy 100%",
            "nanobots in tap water", "microscopic robots", "tap water controlling", "injected with hiv",
            "pope francis endorses", "unesco declares", "best national anthem", "rowan atkinson dies",
            "tragic car crash", "total extinction", "free electricity for all", "start charging 50 rupees"
        }
        self.deceptive_regex_patterns = [
            r'free\s*(\u20b9|rs\.?|inr|dollar|\$)?\s*\d+',
            r'free\s*(laptop|laptops|iphone|smartphone|phone|money|cash|bitcoin|btc|recharge|electricity)',
            r'forward(s)?\s*(this)?.*(whatsapp|group|contact|link|message)',
            r'whatsapp\s*(link|group|users|start charging|paid)',
            r'time\s*travel(er|ler| machine)?',
            r'(world|global|worldwide)\s*(blackout|shutdown|collapse|extinction)',
            r'shut\s*down\s*permanently',
            r'viral\s*video\s*(shows|claiming|warning)',
            r'cures?\s*(diabetes|cancer|aids|all\s*diseases|coronavirus)',
            r'deposited in.*bank',
            r'citizens will (get|receive)',
            r'moon will disappear',
            r'without sleeping',
            r'without charging',
            r'double (human )?intelligence',
            r'water after midnight',
            r'cloned in secret',
            r'android robot',
            r'5g.*(mind control|radiation|frequency|covid|towers)',
            r'poisoned by.*deep state',
            r'aliens?.*(invad|bunker|craft|hiding|hollow)',
            r'doctors hate',
            r'100% of cancer',
            r'banned by big pharma',
            r'(bleach|kerosene|baking soda|raw onions?|garlic and vinegar).*cures?'
        ]

    def clean_text(self, text: str) -> str:
        """Sanitizes raw article text for transformer and language modeling."""
        if not text or not isinstance(text, str):
            return ""
        # 1. Unicode normalization
        text = unicodedata.normalize("NFKD", text)
        # 2. Strip URL links
        text = re.sub(r'http\S+|www\.\S+|ftp\S+', ' ', text)
        # 3. Strip HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        # 4. Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_linguistic_stats(self, raw_text: str) -> dict:
        """Extracts stylometric and linguistic statistics from raw text."""
        if not raw_text or not isinstance(raw_text, str):
            return {
                "char_count": 0, "word_count": 0, "uppercase_ratio": 0.0,
                "exclamation_count": 0, "question_count": 0, "digit_count": 0,
                "sensational_word_count": 0, "lexical_diversity": 0.0
            }

        total_chars = len(raw_text)
        uppercase_chars = sum(1 for c in raw_text if c.isupper())
        exclamation_marks = raw_text.count("!")
        question_marks = raw_text.count("?")
        digits = sum(1 for c in raw_text if c.isdigit())
        
        words = re.findall(r'\b[A-Za-z0-9\'-]+\b', raw_text)
        word_count = len(words)
        unique_words = set(w.lower() for w in words)
        
        text_lower = raw_text.lower()
        sensational_hits = sum(1 for trigger in self.deceptive_triggers if trigger in text_lower)
        regex_hits = sum(1 for pattern in self.deceptive_regex_patterns if re.search(pattern, text_lower))
        total_sensational = sensational_hits + (regex_hits * 2)

        return {
            "char_count": total_chars,
            "word_count": word_count,
            "uppercase_ratio": uppercase_chars / max(1, total_chars),
            "exclamation_count": exclamation_marks,
            "question_count": question_marks,
            "digit_count": digits,
            "sensational_word_count": total_sensational,
            "lexical_diversity": len(unique_words) / max(1, word_count)
        }

class DatasetPreprocessor:
    """
    End-to-End Dataset Preprocessor:
    - Loads raw data
    - Imputes missing values
    - Removes duplicate entries
    - Normalizes labels
    - Cleans textual fields
    - Performs stratified Train / Validation / Test partitioning
    """
    def __init__(self, raw_path: str = RAW_DATASET_PATH):
        self.raw_path = raw_path
        self.text_cleaner = TextPreprocessor()
        self.stats = {}

    def load_and_preprocess(self) -> tuple[list[dict], dict]:
        """Loads, cleans, deduplicates, and normalizes raw dataset records."""
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"Raw dataset file not found at: {self.raw_path}")

        raw_records = []
        with open(self.raw_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_records.append(row)

        total_raw = len(raw_records)
        missing_counts = {"title": 0, "text": 0, "label": 0, "source": 0, "author": 0, "shares": 0, "sentiment": 0}
        duplicates_removed = 0
        cleaned_records = []
        seen_keys = set()

        for row in raw_records:
            title = row.get("title", "").strip()
            text = row.get("text", "").strip()
            raw_label = str(row.get("label", "")).strip().upper()
            source = row.get("source", "").strip()
            author = row.get("author", "").strip()
            shares_raw = row.get("shares", "")
            sentiment_raw = row.get("sentiment", "")

            # Missing value accounting & imputation
            if not title:
                missing_counts["title"] += 1
            if not text:
                missing_counts["text"] += 1
            if not source:
                missing_counts["source"] += 1
                source = "Unknown"
            if not author:
                missing_counts["author"] += 1
                author = "Unknown"
            
            try:
                shares = float(shares_raw) if shares_raw != "" else 1000.0
            except ValueError:
                missing_counts["shares"] += 1
                shares = 1000.0

            try:
                sentiment = float(sentiment_raw) if sentiment_raw != "" else 0.0
            except ValueError:
                missing_counts["sentiment"] += 1
                sentiment = 0.0

            # Label normalization
            if raw_label in ["FAKE", "1", "FALSE"]:
                label = 1
            elif raw_label in ["REAL", "0", "TRUE"]:
                label = 0
            else:
                missing_counts["label"] += 1
                label = 1 if "fake" in title.lower() else 0

            # Text cleaning
            cleaned_title = self.text_cleaner.clean_text(title)
            cleaned_text = self.text_cleaner.clean_text(text)

            # Deduplication check on title
            dedup_key = cleaned_title.lower()
            if dedup_key in seen_keys or not cleaned_title:
                duplicates_removed += 1
                continue
            seen_keys.add(dedup_key)

            cleaned_records.append({
                "title": cleaned_title,
                "text": cleaned_text,
                "label": label,
                "source": source,
                "author": author,
                "shares": shares,
                "sentiment": sentiment
            })

        # Class distribution statistics
        labels = [r["label"] for r in cleaned_records]
        class_dist = Counter(labels)

        self.stats = {
            "total_raw_records": total_raw,
            "total_cleaned_records": len(cleaned_records),
            "duplicates_removed": duplicates_removed,
            "missing_values_imputed": missing_counts,
            "class_distribution": {
                "REAL (0)": class_dist.get(0, 0),
                "FAKE (1)": class_dist.get(1, 0)
            },
            "balance_ratio": round(class_dist.get(1, 0) / max(1, class_dist.get(0, 0)), 3)
        }

        return cleaned_records, self.stats

    def split_and_save(self, records: list[dict]) -> dict:
        """
        Partitions records into stratified Train (70%), Validation (10%), and Test (20%) sets.
        Saves all subsets to data/processed/.
        """
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        y = [r["label"] for r in records]
        indices = np.arange(len(records))

        # First split: Train+Val (80%) vs Test (20%)
        train_val_idx, test_idx = train_test_split(
            indices, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=y
        )

        y_train_val = [y[i] for i in train_val_idx]
        # Second split: Train (70% total) vs Val (10% total) -> 10/80 = 0.125
        val_relative_size = VAL_SPLIT_SIZE / (1.0 - TEST_SPLIT_SIZE)
        train_idx_sub, val_idx_sub = train_test_split(
            train_val_idx, test_size=val_relative_size, random_state=RANDOM_SEED, stratify=y_train_val
        )

        train_records = [records[i] for i in train_idx_sub]
        val_records = [records[i] for i in val_idx_sub]
        test_records = [records[i] for i in test_idx]

        fieldnames = ["title", "text", "label", "source", "author", "shares", "sentiment"]

        def write_subset(path, subset):
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in subset:
                    writer.writerow(r)

        write_subset(PROCESSED_TRAIN_PATH, train_records)
        write_subset(PROCESSED_VAL_PATH, val_records)
        write_subset(PROCESSED_TEST_PATH, test_records)

        # Full processed file
        full_processed_path = os.path.join(PROCESSED_DATA_DIR, "full_processed.csv")
        write_subset(full_processed_path, records)

        # Preprocessing statistics JSON
        stats_path = os.path.join(PROCESSED_DATA_DIR, "preprocessing_stats.json")
        split_stats = {
            **self.stats,
            "splits": {
                "train_count": len(train_records),
                "val_count": len(val_records),
                "test_count": len(test_records),
                "train_class_dist": dict(Counter([r["label"] for r in train_records])),
                "val_class_dist": dict(Counter([r["label"] for r in val_records])),
                "test_class_dist": dict(Counter([r["label"] for r in test_records])),
            },
            "random_seed": RANDOM_SEED
        }

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(split_stats, f, indent=2)

        print(f"Processed dataset saved to {PROCESSED_DATA_DIR}")
        print(f"Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}")
        return split_stats

def run_preprocessing_pipeline() -> dict:
    """Entry point for Phase 2 data preprocessing pipeline execution."""
    preprocessor = DatasetPreprocessor()
    records, _ = preprocessor.load_and_preprocess()
    stats = preprocessor.split_and_save(records)
    return stats

if __name__ == "__main__":
    run_preprocessing_pipeline()
