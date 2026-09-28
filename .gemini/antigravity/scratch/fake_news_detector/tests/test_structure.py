"""
Phase 1 Structure & Import Verification Tests
Verifies that all required folders, configuration definitions, and modular packages exist and import cleanly.
"""
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def test_directory_structure():
    """Checks presence of all required project directories."""
    required_dirs = [
        "data/raw", "data/processed", "data/external",
        "notebooks", "src", "models",
        "results/metrics", "results/plots", "results/shap",
        "app/templates", "app/static", "tests", "docs"
    ]
    for d in required_dirs:
        full_path = os.path.join(BASE_DIR, d)
        assert os.path.exists(full_path), f"Missing required directory: {d}"
    print("[OK] All project directories verified successfully.")

def test_module_imports():
    """Checks that all src modules import without syntax or runtime error."""
    import src.config
    import src.preprocessing
    import src.bert_features
    import src.user_features
    import src.network_features
    import src.feature_fusion
    import src.train_baselines
    import src.train_random_forest
    import src.explainability
    import src.evaluation
    print("[OK] All src modules imported cleanly.")

def test_raw_data_file():
    """Checks that raw dataset file exists."""
    from src.config import RAW_DATASET_PATH
    assert os.path.exists(RAW_DATASET_PATH), f"Raw dataset missing at {RAW_DATASET_PATH}"
    print(f"[OK] Raw dataset verified: {RAW_DATASET_PATH}")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING PHASE 1 STRUCTURE & IMPORT TESTS")
    print("=" * 60)
    test_directory_structure()
    test_module_imports()
    test_raw_data_file()
    print("=" * 60)
    print("ALL PHASE 1 STRUCTURE VERIFICATION TESTS PASSED!")
    print("=" * 60)
