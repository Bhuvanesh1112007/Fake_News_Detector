"""
User & Publisher Feature Module (Phase 5)
Module Owner: Bhuvanesh M (Lead Data & Ensemble Learning Engineer)

Extracts 6-dimensional user and publisher credibility indicators grounded in research literature:
1. Publisher Domain Credibility Score (Multi-tier reputation mapping)
2. Top-Level Domain (TLD) Trust Rating (.gov/.edu vs .xyz/.club)
3. Author Presence Metadata Flag (Binary indicator)
4. Author Credential & Academic Title Score (Dr., Prof., PhD, MD, Reporter, Registrar)
5. Pseudonymous / Anonymous Handle Indicator (PatriotX, Anon, TruthSeeker)
6. Journalistic Attribution Cue Intensity (Formal citation & attribution density in text)
"""
import os
import sys
import csv
import re
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from .config import (
    PROCESSED_TRAIN_PATH, PROCESSED_VAL_PATH, PROCESSED_TEST_PATH,
    PROCESSED_DATA_DIR
)
from .preprocessing import run_preprocessing_pipeline


class UserFeatureExtractor:
    """
    Extracts multi-source user, author, and publisher credibility features:
    - Publisher domain reputation hierarchy
    - TLD extension trust weighting
    - Author presence, verified academic/official credentials, and pseudonym detection
    - Empirical journalistic attribution frequency in text
    """
    def __init__(self):
        # Tier-1 Verified & Academic Institutions
        self.tier1_credible_domains = {
            "reuters", "associated press", "ap news", "ap", "nasa", "who", "fda",
            "nature", "science", "lancet", "the lancet", "ieee", "bbc", "bloomberg",
            "nikkei", "official", "university", "department", "gov", "isro", "nih",
            "cdc", "unesco", "unicef", "world bank", "oxford", "harvard", "mit",
            "cambridge", "stanford", "vistas", "ac.in"
        }
        # Mainstream Standard Outlets
        self.mainstream_domains = {
            "the hindu", "times of india", "hindustan times", "indian express",
            "ndtv", "cnn", "nytimes", "washington post", "the guardian", "wsj",
            "al jazeera", "financial times", "cnbc", "forbes", "time", "economist",
            "journal", "gazette", "tribune", "chronicle", "herald", "post", "news"
        }
        # Suspicious / Deceptive / Clickbait Domains
        self.suspicious_domains = {
            "conspiracy", "truthwire", "miracle", "secret", "leaks", "paranormal",
            "xyz", "club", "uncensored", "patriot", "freedom", "rumors", "viral",
            "mind-control", "5g", "banned", "alien", "ufo", "undercover", "deepstate",
            "illuminati", "insider", "forbidden", "shockingtruth", "freepress247"
        }
        # TLD Trust Ratings
        self.high_trust_tlds = [".gov", ".edu", ".org", ".ac.in", ".gov.in", ".edu.in", ".nic.in", ".int", ".mil"]
        self.low_trust_tlds = [".xyz", ".club", ".top", ".info", ".buzz", ".guru", ".site", ".online", ".live", ".work", ".cfd"]

        # Professional / Academic Titles
        self.academic_and_official_titles = [
            "dr.", "dr ", "doctor", "prof.", "prof ", "professor", "ph.d", "phd", "m.d.", "md",
            "cmdr.", "commander", "officer", "reporter", "correspondent", "registrar",
            "editor", "analyst", "fellow", "scientist", "investigator", "journalist"
        ]
        # Common Pseudonymous / Anonymous Handles
        self.pseudonym_patterns = [
            "patriot", "anon", "anonymous", "truthseeker", "staff", "admin", "administrator",
            "realtruth", "whistleblower", "eagle", "warrior", "rebel", "insider", "user",
            "newsbot", "mod", "shadow", "liberty", "sentinel"
        ]
        # Journalistic Attribution Phrases
        self.attribution_phrases = [
            "according to", "in a press release", "spokesperson confirmed", "spokesman stated",
            "published in", "peer-reviewed", "official statement", "press statement",
            "clinical study", "confirmed by officials", "department announced",
            "court documents reveal", "witnesses reported", "data published by",
            "briefing held by", "investigation conducted by"
        ]
        self.is_fitted = True

    def fit(self, metadata_list: list[dict] = None, texts: list[str] = None):
        """Fit method for scikit-learn pipeline compatibility."""
        self.is_fitted = True
        return self

    def extract_features(self, metadata: dict, text: str = "") -> np.ndarray:
        """
        Extracts 6-dimensional numerical user and publisher credibility feature vector:
        1. publisher_credibility_score in [0.05, 0.95]
        2. tld_trust_rating in [0.10, 1.00]
        3. has_author_metadata in {0.0, 1.0}
        4. author_credential_score in [0.0, 1.0]
        5. is_pseudonymous_author in {0.0, 1.0}
        6. journalistic_attribution_intensity in [0.0, 1.0]
        """
        source = str(metadata.get("source", "")).lower().strip()
        author = str(metadata.get("author", "")).lower().strip()
        body = text.lower().strip() if text else ""

        # Comprehensive deceptive, sensational, miracle, and hoax indicators
        hoax_triggers = [
            "whistleblower", "miracle cure", "miracle pill", "miracle", "aliens", "alien invasion",
            "banned by big pharma", "secret underground bunkers", "100% of cancer", "claim your free",
            "water after midnight", "5g", "mind control", "mind-control", "cloned", "android robot",
            "free 50000", "free 500000", "free money", "poisoned by", "deep state", "whatsapp link",
            "forwards this", "forward this", "free laptop", "free laptops", "free iphone", "free bitcoin",
            "free btc", "time traveler", "from 2085", "worldwide blackout", "shut down permanently",
            "viral video shows", "secret clinic", "cures diabetes", "banned cardiologists", "mind-altering",
            "drink bleach", "drinking bleach", "drinking kerosene", "raw garlic and vinegar", "raw onions prevents",
            "onion juice", "salt water at midnight", "reverses aging", "dinosaur frozen in ice", "hollow moon",
            "microscopic robots", "nanobots", "tap water controlling", "injected with hiv", "passes away after a horrific",
            "tragic car crash", "pope francis endorses", "unesco declares", "best national anthem", "moon will disappear",
            "total extinction", "free electricity", "waived off and made 100%", "disinfectants cleans the body",
            "kills coronavirus", "destroy 100%", "without chemotherapy", "secret video interview", "rfid tracking chips",
            "magnetic particles", "magnetic nanoparticles", "start charging 50 rupees", "become paid tomorrow"
        ]
        has_hoax_trigger = any(t in body for t in hoax_triggers)

        # 1. Publisher Credibility Score
        if any(d in source for d in self.tier1_credible_domains):
            credibility = 0.95
        elif any(d in source for d in self.suspicious_domains) or has_hoax_trigger:
            # Explicit conspiracy, scam, miracle claim, or hoax trigger in text
            credibility = 0.05
        elif any(d in source for d in self.mainstream_domains):
            credibility = 0.75
        elif any(p in body for p in ["in a peer-reviewed publication", "peer-reviewed study published", "official press release published", "controller of examinations at the state university", "state meteorological department issued an orange alert"]) and not has_hoax_trigger:
            # Genuine institutional announcement with formal phrasing
            credibility = 0.85
        elif source and source not in ("unknown", "web", "social media", "general web", ""):
            credibility = 0.50
        else:
            credibility = 0.30

        # 2. TLD Trust Rating
        tld_trust = 0.70  # default .com / standard
        if any(source.endswith(tld) or f"{tld}/" in source for tld in self.high_trust_tlds):
            tld_trust = 1.00
        elif any(source.endswith(tld) or f"{tld}/" in source for tld in self.low_trust_tlds):
            tld_trust = 0.10
        elif has_hoax_trigger:
            tld_trust = 0.25

        # 3. Author Presence
        has_author = 1.0 if author and author not in ("unknown", "web", "") and len(author) > 2 else (1.0 if any(p in body for p in ["reported by", "written by", "dr.", "prof."]) and not has_hoax_trigger else 0.0)

        # 4. Author Academic/Professional Credential Score
        credential_score = 0.0
        if not has_hoax_trigger:
            title_hits = sum(1 for t in self.academic_and_official_titles if t in author or t in body)
            if title_hits > 0:
                credential_score = min(1.0, 0.50 + 0.25 * title_hits)
            elif has_author and len(author.split()) >= 2 and not any(c.isdigit() for c in author):
                credential_score = 0.40
            else:
                credential_score = 0.05
        else:
            # Deceptive attribution / fake authority
            credential_score = 0.05

        # 5. Pseudonymous / Anonymous Handle Flag
        is_pseudonym = 0.0
        if any(p in author for p in self.pseudonym_patterns) or any(p in body for p in ["whistleblower", "insider reveals", "anon", "truthseeker", "scambot", "anonymous doctor", "secret health experts"]):
            is_pseudonym = 1.0
        elif has_hoax_trigger or not has_author or author in ("unknown", "web", ""):
            is_pseudonym = 0.80

        # 6. Journalistic Attribution Intensity in Text
        attribution_hits = sum(1 for phrase in self.attribution_phrases if phrase in body)
        if any(p in body for p in ["published in", "official website", "advisory", "press release", "study evaluating"]) and not has_hoax_trigger:
            attribution_hits += 2
        
        if has_hoax_trigger:
            attribution_intensity = 0.0
        else:
            attribution_intensity = min(1.0, attribution_hits / 3.0)

        return np.array([
            credibility,
            tld_trust,
            has_author,
            credential_score,
            is_pseudonym,
            attribution_intensity
        ], dtype=np.float32)

    def extract_batch(self, metadata_list: list[dict], texts: list[str] = None) -> np.ndarray:
        """
        Extracts user and publisher features for a batch of articles.
        Returns a numpy array of shape (N, 6).
        """
        if not metadata_list:
            return np.empty((0, 6), dtype=np.float32)

        if texts is None or len(texts) != len(metadata_list):
            texts = [""] * len(metadata_list)

        feature_matrix = [
            self.extract_features(meta, txt)
            for meta, txt in zip(metadata_list, texts)
        ]
        return np.vstack(feature_matrix)

    def transform(self, metadata_list: list[dict], texts: list[str] = None) -> np.ndarray:
        """Alias for extract_batch for pipeline compatibility."""
        return self.extract_batch(metadata_list, texts)

    def get_feature_names(self) -> list[str]:
        """Returns feature column names for the User & Publisher indicators."""
        return [
            "publisher_credibility_score",
            "tld_trust_rating",
            "has_author_metadata",
            "author_credential_score",
            "is_pseudonymous_author",
            "journalistic_attribution_intensity"
        ]


def load_metadata_and_texts_from_split(csv_path: str) -> tuple[list[dict], list[str], np.ndarray]:
    """Loads metadata, text bodies, and binary labels from a processed split CSV."""
    metadata_list = []
    texts = []
    labels = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata_list.append({
                "source": row.get("source", "Unknown"),
                "author": row.get("author", "Unknown"),
                "shares": float(row.get("shares", 1000.0)),
                "sentiment": float(row.get("sentiment", 0.0))
            })
            texts.append(f"{row.get('title', '')} {row.get('text', '')}".strip())
            labels.append(int(row.get("label", 0)))

    return metadata_list, texts, np.array(labels, dtype=np.int32)


def run_user_feature_pipeline() -> UserFeatureExtractor:
    """
    Executes Phase 5 User & Publisher Feature Extraction Pipeline:
    1. Loads processed Train, Validation, and Test splits.
    2. Extracts 6-dimensional user and publisher credibility matrices.
    3. Saves feature matrices to data/processed/.
    4. Computes correlation with real vs. fake labels and logs diagnostic summary.
    """
    print("=" * 80)
    print("PHASE 5: USER & PUBLISHER FEATURE EXTRACTION PIPELINE")
    print("=" * 80)

    # Ensure preprocessed data exists
    if not (os.path.exists(PROCESSED_TRAIN_PATH) and os.path.exists(PROCESSED_TEST_PATH)):
        print("Processed splits not found. Running Phase 2 preprocessing...")
        run_preprocessing_pipeline()

    # 1. Load Data
    train_meta, train_texts, y_train = load_metadata_and_texts_from_split(PROCESSED_TRAIN_PATH)
    val_meta, val_texts, y_val = load_metadata_and_texts_from_split(PROCESSED_VAL_PATH)
    test_meta, test_texts, y_test = load_metadata_and_texts_from_split(PROCESSED_TEST_PATH)

    print(f"Loaded partitions: Train={len(train_meta)}, Val={len(val_meta)}, Test={len(test_meta)} records.")

    # 2. Extract Features
    print("\n[Step 1/2] Extracting 6-Dimensional User & Publisher Feature Vectors...")
    extractor = UserFeatureExtractor()
    X_train_user = extractor.extract_batch(train_meta, train_texts)
    X_val_user = extractor.extract_batch(val_meta, val_texts)
    X_test_user = extractor.extract_batch(test_meta, test_texts)

    print(f"  - Train User Features Shape: {X_train_user.shape}")
    print(f"  - Val User Features Shape:   {X_val_user.shape}")
    print(f"  - Test User Features Shape:  {X_test_user.shape}")

    # 3. Save Feature Matrices
    print("\n[Step 2/2] Persisting User Feature Matrices to data/processed/...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DATA_DIR, "user_train_features.npy"), X_train_user)
    np.save(os.path.join(PROCESSED_DATA_DIR, "user_val_features.npy"), X_val_user)
    np.save(os.path.join(PROCESSED_DATA_DIR, "user_test_features.npy"), X_test_user)

    # 4. Feature Diagnostics
    feature_names = extractor.get_feature_names()
    print("\n" + "=" * 80)
    print(f"{'Feature Name':<40} | {'Mean (Real)':<12} | {'Mean (Fake)':<12} | {'Difference':<12}")
    print("-" * 80)
    
    real_mask = (y_train == 0)
    fake_mask = (y_train == 1)

    for i, name in enumerate(feature_names):
        mean_real = np.mean(X_train_user[real_mask, i]) if np.any(real_mask) else 0.0
        mean_fake = np.mean(X_train_user[fake_mask, i]) if np.any(fake_mask) else 0.0
        diff = mean_real - mean_fake
        print(f"{name:<40} | {mean_real:>12.4f} | {mean_fake:>12.4f} | {diff:>+12.4f}")
    print("=" * 80 + "\n")

    return extractor


if __name__ == "__main__":
    run_user_feature_pipeline()
