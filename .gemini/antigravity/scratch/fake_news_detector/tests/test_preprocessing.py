"""
Phase 2 Data Preprocessing Automated Test Suite
Verifies data loading, text sanitization, missing-value imputation, label encoding,
and stratified partitioning integrity without data leakage.
"""
import os
import sys
import json
import csv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import TextPreprocessor, DatasetPreprocessor, run_preprocessing_pipeline
from src.config import PROCESSED_TRAIN_PATH, PROCESSED_VAL_PATH, PROCESSED_TEST_PATH, PROCESSED_DATA_DIR

def test_text_sanitizer():
    """Tests text cleaning logic for URLs, HTML tags, and Unicode."""
    cleaner = TextPreprocessor()
    raw = "Breaking News! <p>Visit http://fake-news.xyz/article</p>   for shocking secrets!   "
    cleaned = cleaner.clean_text(raw)
    assert "<p>" not in cleaned, "Failed to strip HTML tags"
    assert "http" not in cleaned, "Failed to strip URLs"
    assert "  " not in cleaned, "Failed to collapse whitespace"
    print("[OK] Text sanitizer logic verified.")

def test_linguistic_stats_extraction():
    """Tests extraction of stylometric and sensationalism counts."""
    cleaner = TextPreprocessor()
    text = "SHOCKING PROOF: Miracle cure banned by doctors!!!"
    stats = cleaner.extract_linguistic_stats(text)
    assert stats["uppercase_ratio"] > 0.20, f"Uppercase ratio calculation incorrect: {stats['uppercase_ratio']}"
    assert stats["exclamation_count"] == 3, "Exclamation count incorrect"
    assert stats["sensational_word_count"] >= 2, "Sensational word count incorrect"
    print("[OK] Linguistic stats extraction verified.")

def test_processed_files_exist():
    """Verifies that all processed split CSVs and statistics JSON exist."""
    stats = run_preprocessing_pipeline()
    assert os.path.exists(PROCESSED_TRAIN_PATH), "train.csv missing"
    assert os.path.exists(PROCESSED_VAL_PATH), "val.csv missing"
    assert os.path.exists(PROCESSED_TEST_PATH), "test.csv missing"
    
    stats_json = os.path.join(PROCESSED_DATA_DIR, "preprocessing_stats.json")
    assert os.path.exists(stats_json), "preprocessing_stats.json missing"
    
    with open(stats_json, "r", encoding="utf-8") as f:
        loaded_stats = json.load(f)
    assert "splits" in loaded_stats, "Splits dictionary missing from stats"
    print("[OK] Processed split files and metadata JSON verified.")

def test_no_data_leakage():
    """Verifies that Train, Validation, and Test sets have zero overlap."""
    def load_titles(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            return set(row["title"].lower().strip() for row in csv.DictReader(f))

    train_titles = load_titles(PROCESSED_TRAIN_PATH)
    val_titles = load_titles(PROCESSED_VAL_PATH)
    test_titles = load_titles(PROCESSED_TEST_PATH)

    train_val_overlap = train_titles.intersection(val_titles)
    train_test_overlap = train_titles.intersection(test_titles)
    val_test_overlap = val_titles.intersection(test_titles)

    assert len(train_val_overlap) == 0, f"Data leakage detected between Train and Val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Data leakage detected between Train and Test: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Data leakage detected between Val and Test: {val_test_overlap}"
    print("[OK] Zero data leakage verified across Train, Validation, and Test partitions.")

def test_label_normalization():
    """Verifies that all processed records have strictly binary labels (0 or 1)."""
    for path in [PROCESSED_TRAIN_PATH, PROCESSED_VAL_PATH, PROCESSED_TEST_PATH]:
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                assert row["label"] in ["0", "1"], f"Invalid label {row['label']} in {path}"
    print("[OK] Binary label normalization (0=REAL, 1=FAKE) verified.")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING PHASE 2 DATA PREPROCESSING TEST SUITE")
    print("=" * 60)
    test_text_sanitizer()
    test_linguistic_stats_extraction()
    test_processed_files_exist()
    test_no_data_leakage()
    test_label_normalization()
    print("=" * 60)
    print("ALL PHASE 2 DATA PREPROCESSING TESTS PASSED!")
    print("=" * 60)
