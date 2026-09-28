"""
Phase 5 User & Publisher Features Automated Test Suite
Module Owner: Bhuvanesh M (Lead Data & Ensemble Learning Engineer)

Verifies:
1. Publisher domain credibility hierarchy (Tier-1, Mainstream, Neutral, Suspicious)
2. Top-Level Domain (TLD) trust weighting
3. Author presence, academic/professional credential detection, and pseudonym identification
4. Journalistic attribution phrase extraction in text
5. Batch extraction dimensions (N, 6), array dtypes, and mathematical bounds [0.0, 1.0]
6. Feature naming schema completeness
7. Dataset split extraction and .npy artifact persistence
"""
import os
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.user_features import (
    UserFeatureExtractor, run_user_feature_pipeline
)
from src.config import PROCESSED_DATA_DIR


def test_user_extractor_initialization():
    """Verifies UserFeatureExtractor initialization and feature name definitions."""
    extractor = UserFeatureExtractor()
    assert extractor.is_fitted is True
    names = extractor.get_feature_names()
    assert len(names) == 6, f"Expected 6 user features, got {len(names)}"
    assert names[0] == "publisher_credibility_score"
    assert names[-1] == "journalistic_attribution_intensity"
    print("[OK] UserFeatureExtractor initialization and schema verified.")


def test_publisher_credibility_tiers():
    """Tests publisher credibility scores across different media tiers."""
    extractor = UserFeatureExtractor()

    # Tier-1 Credible
    tier1_meta = {"source": "NASA Press Release", "author": "Dr. Sarah Jenkins"}
    f_tier1 = extractor.extract_features(tier1_meta)
    assert f_tier1[0] == 0.95, f"Expected 0.95 for Tier-1, got {f_tier1[0]}"

    # Mainstream Standard
    mainstream_meta = {"source": "The Hindu Daily", "author": "Staff Reporter"}
    f_mainstream = extractor.extract_features(mainstream_meta)
    assert f_mainstream[0] == 0.75, f"Expected 0.75 for Mainstream, got {f_mainstream[0]}"

    # Suspicious / Clickbait
    suspicious_meta = {"source": "ConspiracyTruthWire247.net", "author": "PatriotX"}
    f_suspicious = extractor.extract_features(suspicious_meta)
    assert f_suspicious[0] == 0.05, f"Expected 0.05 for Suspicious, got {f_suspicious[0]}"

    # Neutral / Unknown
    neutral_meta = {"source": "TechDailyBlog", "author": "John"}
    f_neutral = extractor.extract_features(neutral_meta)
    assert f_neutral[0] == 0.50, f"Expected 0.50 for Neutral, got {f_neutral[0]}"
    print("[OK] Publisher credibility multi-tier scoring verified.")


def test_tld_trust_scoring():
    """Tests Top-Level Domain (TLD) trust ratings."""
    extractor = UserFeatureExtractor()

    # High-trust TLD (.gov, .edu, .ac.in)
    f_gov = extractor.extract_features({"source": "isro.gov.in", "author": "Dr. Rao"})
    assert f_gov[1] == 1.00, f"Expected 1.00 for .gov.in, got {f_gov[1]}"

    # Low-trust TLD (.xyz, .club, .top)
    f_xyz = extractor.extract_features({"source": "freemoneysecret.xyz", "author": "Admin"})
    assert f_xyz[1] == 0.10, f"Expected 0.10 for .xyz, got {f_xyz[1]}"

    # Standard TLD (.com, .net)
    f_com = extractor.extract_features({"source": "technews.com", "author": "John Doe"})
    assert f_com[1] == 0.70, f"Expected 0.70 for .com, got {f_com[1]}"
    print("[OK] TLD extension trust ratings verified.")


def test_author_presence_and_credentials():
    """Tests author metadata presence and academic/official title recognition."""
    extractor = UserFeatureExtractor()

    # Missing Author
    f_missing = extractor.extract_features({"source": "DailyNews", "author": "Unknown"})
    assert f_missing[2] == 0.0, "has_author must be 0.0 for unknown author"
    assert f_missing[3] == 0.0, "credential_score must be 0.0 for unknown author"

    # Academic Title (Dr.)
    f_doctor = extractor.extract_features({"source": "Lancet", "author": "Dr. Halima Traore"})
    assert f_doctor[2] == 1.0, "has_author must be 1.0"
    assert f_doctor[3] >= 0.75, f"Expected credential score >= 0.75 for Dr., got {f_doctor[3]}"

    # Professor (Prof.)
    f_prof = extractor.extract_features({"source": "MIT News", "author": "Prof. Robert Chang, PhD"})
    assert f_prof[3] >= 0.75, f"Expected high credential score for Prof./PhD, got {f_prof[3]}"

    # Standard Name (Sarah Jenkins)
    f_standard = extractor.extract_features({"source": "Herald", "author": "Sarah Jenkins"})
    assert f_standard[2] == 1.0
    assert 0.30 <= f_standard[3] <= 0.50
    print("[OK] Author presence and academic/professional credential detection verified.")


def test_pseudonym_detection():
    """Tests detection of pseudonymous, anonymous, and bot handles."""
    extractor = UserFeatureExtractor()

    # Pseudonym (PatriotX, AnonWarrior99)
    f_pseudo = extractor.extract_features({"source": "Blog", "author": "PatriotX"})
    assert f_pseudo[4] == 1.0, f"Expected is_pseudonym=1.0 for PatriotX, got {f_pseudo[4]}"

    f_bot = extractor.extract_features({"source": "Forum", "author": "TruthSeeker2024"})
    assert f_bot[4] == 1.0, f"Expected is_pseudonym=1.0 for TruthSeeker2024, got {f_bot[4]}"

    # Real human name (Dr. Sarah Jenkins)
    f_real = extractor.extract_features({"source": "Science", "author": "Dr. Sarah Jenkins"})
    assert f_real[4] == 0.0, f"Expected is_pseudonym=0.0 for real human name, got {f_real[4]}"
    print("[OK] Pseudonymous and anonymous author handle detection verified.")


def test_journalistic_attribution_extraction():
    """Tests journalistic attribution cue frequency in article text."""
    extractor = UserFeatureExtractor()

    # Article with strong formal attributions
    formal_text = (
        "According to a press release by the health department, the clinical study published in "
        "The Lancet was confirmed by officials during an official statement."
    )
    f_formal = extractor.extract_features({"source": "Reuters", "author": "Reporter"}, formal_text)
    assert f_formal[5] >= 0.66, f"Expected high attribution intensity, got {f_formal[5]}"

    # Sensational text with zero journalistic attributions
    clickbait_text = "SHOCKING SECRET! Banned miracle recipe cures everything! Share before deletion!"
    f_clickbait = extractor.extract_features({"source": "ViralWire", "author": "PatriotX"}, clickbait_text)
    assert f_clickbait[5] == 0.0, f"Expected 0.0 attribution intensity, got {f_clickbait[5]}"
    print("[OK] Journalistic citation & attribution cue extraction verified.")


def test_batch_extraction_and_bounds():
    """Tests batch extraction dimensions and mathematical bounds."""
    extractor = UserFeatureExtractor()
    meta_batch = [
        {"source": "NASA", "author": "Dr. Alan Stern"},
        {"source": "ConspiracyWire", "author": "Anon99"},
        {"source": "Reuters", "author": "David Shepardson"},
        {"source": "Unknown", "author": "Unknown"}
    ]
    texts = [
        "According to NASA scientists, the mission was confirmed by officials.",
        "Secret miracle cure Big Pharma hates.",
        "In a press release, the department announced new regulations.",
        "Breaking news update."
    ]

    features = extractor.extract_batch(meta_batch, texts)
    assert isinstance(features, np.ndarray), "Output must be a numpy array"
    assert features.shape == (4, 6), f"Expected shape (4, 6), got {features.shape}"
    assert features.dtype == np.float32, f"Expected dtype float32, got {features.dtype}"
    
    # Assert all values in [0.0, 1.0]
    assert np.all(features >= 0.0) and np.all(features <= 1.0), "Features out of [0.0, 1.0] bounds"
    print("[OK] Batch extraction shape (4, 6), float32 dtype, and [0, 1] bounds verified.")


def test_pipeline_execution_and_persistence():
    """Verifies running the Phase 5 user features extraction pipeline."""
    extractor = run_user_feature_pipeline()
    assert extractor.is_fitted is True

    train_path = os.path.join(PROCESSED_DATA_DIR, "user_train_features.npy")
    val_path = os.path.join(PROCESSED_DATA_DIR, "user_val_features.npy")
    test_path = os.path.join(PROCESSED_DATA_DIR, "user_test_features.npy")

    assert os.path.exists(train_path), "user_train_features.npy missing"
    assert os.path.exists(val_path), "user_val_features.npy missing"
    assert os.path.exists(test_path), "user_test_features.npy missing"

    train_mat = np.load(train_path)
    test_mat = np.load(test_path)
    assert train_mat.shape[1] == 6, f"Expected 6 features in train matrix, got {train_mat.shape[1]}"
    assert test_mat.shape[1] == 6, f"Expected 6 features in test matrix, got {test_mat.shape[1]}"
    print(f"[OK] Pipeline execution & persistence verified (Train: {train_mat.shape}, Test: {test_mat.shape}).")


if __name__ == "__main__":
    print("=" * 75)
    print("RUNNING PHASE 5 USER & PUBLISHER FEATURES AUTOMATED TEST SUITE")
    print("=" * 75)
    test_user_extractor_initialization()
    test_publisher_credibility_tiers()
    test_tld_trust_scoring()
    test_author_presence_and_credentials()
    test_pseudonym_detection()
    test_journalistic_attribution_extraction()
    test_batch_extraction_and_bounds()
    test_pipeline_execution_and_persistence()
    print("=" * 75)
    print("ALL PHASE 5 USER & PUBLISHER FEATURES TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)
