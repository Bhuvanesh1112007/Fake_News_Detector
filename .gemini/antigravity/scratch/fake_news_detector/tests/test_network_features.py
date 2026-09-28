"""
Phase 6 Network & Social Engagement Features Automated Test Suite
Module Owner: Bhuvanesh M (Lead Data & Ensemble Learning Engineer)

Verifies:
1. Log-transformed share scaling logic and bounds
2. Hyper-virality thresholding (>= 50,000 shares)
3. Share velocity proxy with forward cue amplification
4. Sentiment polarity normalization and emotional arousal intensity
5. Social engagement anomaly scoring (divergence between reputation and virality)
6. Batch extraction array dimensions (N, 6), float32 dtypes, and mathematical bounds [0.0, 1.0]
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

from src.network_features import (
    NetworkFeatureExtractor, run_network_feature_pipeline
)
from src.config import PROCESSED_DATA_DIR


def test_network_extractor_initialization():
    """Verifies NetworkFeatureExtractor initialization and feature name definitions."""
    extractor = NetworkFeatureExtractor()
    assert extractor.is_fitted is True
    names = extractor.get_feature_names()
    assert len(names) == 6, f"Expected 6 network features, got {len(names)}"
    assert names[0] == "log_social_shares"
    assert names[-1] == "engagement_anomaly_score"
    print("[OK] NetworkFeatureExtractor initialization and schema verified.")


def test_log_shares_scaling():
    """Tests log-transformed share scaling across various volume ranges."""
    extractor = NetworkFeatureExtractor()

    f_zero = extractor.extract_features({"shares": 0})
    assert f_zero[0] == 0.0, f"Expected 0.0 for 0 shares, got {f_zero[0]}"

    f_1k = extractor.extract_features({"shares": 1000})
    assert 0.40 <= f_1k[0] <= 0.50, f"Expected ~0.46 for 1,000 shares, got {f_1k[0]}"

    f_100k = extractor.extract_features({"shares": 100000})
    assert 0.70 <= f_100k[0] <= 0.85, f"Expected ~0.77 for 100,000 shares, got {f_100k[0]}"
    print("[OK] Log-transformed share scaling verified.")


def test_hyper_virality_indicator():
    """Tests binary hyper-virality thresholding at 50,000 shares."""
    extractor = NetworkFeatureExtractor()

    f_low = extractor.extract_features({"shares": 12000})
    assert f_low[1] == 0.0, "Expected hyper_virality=0.0 for 12k shares"

    f_threshold = extractor.extract_features({"shares": 50000})
    assert f_threshold[1] == 1.0, "Expected hyper_virality=1.0 for 50k shares"

    f_extreme = extractor.extract_features({"shares": 95000})
    assert f_extreme[1] == 1.0, "Expected hyper_virality=1.0 for 95k shares"
    print("[OK] Hyper-virality indicator thresholding verified.")


def test_share_velocity_proxy():
    """Tests velocity estimation and amplification from forward cues."""
    extractor = NetworkFeatureExtractor()

    normal_text = "NASA announced quarterly progress on astronomy instruments."
    f_normal = extractor.extract_features({"shares": 50000}, normal_text)

    viral_text = "Share before deletion! Circulating on WhatsApp! Viral post Big Pharma tries to hide!"
    f_viral = extractor.extract_features({"shares": 50000}, viral_text)

    assert f_viral[2] > f_normal[2], f"Expected higher velocity for viral cues ({f_viral[2]} vs {f_normal[2]})"
    print(f"[OK] Share velocity proxy verified (Normal: {f_normal[2]:.3f}, Viral: {f_viral[2]:.3f}).")


def test_sentiment_and_emotional_intensity():
    """Tests sentiment rescaling from [-1, 1] to [0, 1] and emotional arousal magnitude."""
    extractor = NetworkFeatureExtractor()

    # Negative Emotion (Fearmongering / Alarmist)
    f_neg = extractor.extract_features({"sentiment": -0.8})
    assert np.isclose(f_neg[3], 0.10, atol=1e-2), f"Expected normalized sentiment ~0.10, got {f_neg[3]}"
    assert np.isclose(f_neg[4], 0.80, atol=1e-2), f"Expected emotional intensity 0.80, got {f_neg[4]}"

    # Neutral Objective Sentiment
    f_neutral = extractor.extract_features({"sentiment": 0.0})
    assert np.isclose(f_neutral[3], 0.50, atol=1e-2), f"Expected normalized sentiment 0.50, got {f_neutral[3]}"
    assert np.isclose(f_neutral[4], 0.00, atol=1e-2), f"Expected emotional intensity 0.00, got {f_neutral[4]}"

    # Positive Sentiment
    f_pos = extractor.extract_features({"sentiment": +0.8})
    assert np.isclose(f_pos[3], 0.90, atol=1e-2), f"Expected normalized sentiment ~0.90, got {f_pos[3]}"
    assert np.isclose(f_pos[4], 0.80, atol=1e-2), f"Expected emotional intensity 0.80, got {f_pos[4]}"
    print("[OK] Sentiment normalization and emotional intensity verified.")


def test_engagement_anomaly_scoring():
    """Tests anomaly detection for unverified publishers with viral propagation."""
    extractor = NetworkFeatureExtractor()

    # Suspicious domain + extreme virality
    f_anomaly = extractor.extract_features({
        "source": "ConspiracyTruthWire247.net",
        "shares": 95000
    })
    assert f_anomaly[5] >= 0.85, f"Expected high anomaly score (>=0.85), got {f_anomaly[5]}"

    # Verified institution + moderate shares
    f_credible = extractor.extract_features({
        "source": "NASA Press Release",
        "shares": 14000
    })
    assert f_credible[5] == 0.0, f"Expected anomaly score 0.0 for credible outlet, got {f_credible[5]}"
    print(f"[OK] Engagement anomaly detection verified (Suspicious: {f_anomaly[5]}, Credible: {f_credible[5]}).")


def test_batch_extraction_and_bounds():
    """Tests batch extraction dimensions and mathematical bounds [0.0, 1.0]."""
    extractor = NetworkFeatureExtractor()
    meta_batch = [
        {"shares": 14000, "sentiment": 0.45, "source": "Nature"},
        {"shares": 95000, "sentiment": -0.80, "source": "ConspiracyWire"},
        {"shares": 500, "sentiment": 0.0, "source": "LocalNews"},
        {"shares": 65000, "sentiment": -0.50, "source": "Unknown"}
    ]
    texts = [
        "Peer-reviewed study published today.",
        "Share before deletion! Secret cure banned!",
        "City council meeting schedule.",
        "Circulating on WhatsApp viral warning."
    ]

    features = extractor.extract_batch(meta_batch, texts)
    assert isinstance(features, np.ndarray), "Output must be a numpy array"
    assert features.shape == (4, 6), f"Expected shape (4, 6), got {features.shape}"
    assert features.dtype == np.float32, f"Expected dtype float32, got {features.dtype}"
    assert np.all(features >= 0.0) and np.all(features <= 1.0), "Features out of [0.0, 1.0] bounds"
    print("[OK] Batch extraction shape (4, 6), float32 dtype, and [0, 1] bounds verified.")


def test_pipeline_execution_and_persistence():
    """Verifies running the Phase 6 network feature extraction pipeline."""
    extractor = run_network_feature_pipeline()
    assert extractor.is_fitted is True

    train_path = os.path.join(PROCESSED_DATA_DIR, "network_train_features.npy")
    val_path = os.path.join(PROCESSED_DATA_DIR, "network_val_features.npy")
    test_path = os.path.join(PROCESSED_DATA_DIR, "network_test_features.npy")

    assert os.path.exists(train_path), "network_train_features.npy missing"
    assert os.path.exists(val_path), "network_val_features.npy missing"
    assert os.path.exists(test_path), "network_test_features.npy missing"

    train_mat = np.load(train_path)
    test_mat = np.load(test_path)
    assert train_mat.shape[1] == 6, f"Expected 6 features in train matrix, got {train_mat.shape[1]}"
    assert test_mat.shape[1] == 6, f"Expected 6 features in test matrix, got {test_mat.shape[1]}"
    print(f"[OK] Pipeline execution & persistence verified (Train: {train_mat.shape}, Test: {test_mat.shape}).")


if __name__ == "__main__":
    print("=" * 75)
    print("RUNNING PHASE 6 NETWORK & ENGAGEMENT FEATURES AUTOMATED TEST SUITE")
    print("=" * 75)
    test_network_extractor_initialization()
    test_log_shares_scaling()
    test_hyper_virality_indicator()
    test_share_velocity_proxy()
    test_sentiment_and_emotional_intensity()
    test_engagement_anomaly_scoring()
    test_batch_extraction_and_bounds()
    test_pipeline_execution_and_persistence()
    print("=" * 75)
    print("ALL PHASE 6 NETWORK & ENGAGEMENT FEATURES TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)
