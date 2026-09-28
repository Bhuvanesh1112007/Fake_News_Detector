"""
Phase 7 Multi-Source Feature Fusion Automated Test Suite
Module Owner: Bhuvanesh M (Lead Data & Ensemble Learning Engineer) & Mohamed Irfaan M (Lead NLP Engineer)

Verifies:
1. MultiSourceFeatureFusion initialization & component architecture
2. Exact 79-dimensional feature naming schema and ordering
3. Batch transformation dimensions (N, 79) and float32 dtype
4. Single-sample encoding with Explainable AI token attention heatmap
5. Serialization, disk persistence, and deserialization consistency
6. Dataset split extraction and .npy artifact persistence
"""
import os
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.feature_fusion import (
    MultiSourceFeatureFusion, run_feature_fusion_pipeline
)
from src.config import FEATURE_FUSION_PATH, PROCESSED_DATA_DIR


def test_fusion_initialization():
    """Verifies MultiSourceFeatureFusion initialization and safety guards."""
    fusion = MultiSourceFeatureFusion()
    assert fusion.is_fitted is False, "Fusion layer should be un-fitted initially"
    assert fusion.embedding_dim == 64
    assert fusion.bert_extractor is not None
    assert fusion.user_extractor is not None
    assert fusion.network_extractor is not None

    try:
        fusion.transform(["Title"], ["Body"])
        assert False, "Should have raised ValueError on un-fitted transform"
    except ValueError:
        pass
    print("[OK] MultiSourceFeatureFusion initialization and safety guards verified.")


def test_feature_names_count_and_order():
    """Verifies that get_feature_names() returns exactly 79 feature names in correct grouping."""
    fusion = MultiSourceFeatureFusion()
    names = fusion.get_feature_names()
    assert len(names) == 79, f"Expected exactly 79 feature names, got {len(names)}"

    # BERT features (0..63)
    assert names[0] == "bert_dim_00"
    assert names[63] == "bert_dim_63"

    # User features (64..69)
    assert names[64] == "publisher_credibility_score"
    assert names[69] == "journalistic_attribution_intensity"

    # Network features (70..75)
    assert names[70] == "log_social_shares"
    assert names[75] == "engagement_anomaly_score"

    # Stylometric features (76..78)
    assert names[76] == "stylometric_uppercase_ratio"
    assert names[78] == "stylometric_lexical_diversity"
    print(f"[OK] Fused feature naming schema verified (Total: {len(names)} features).")


def test_fit_and_transform_dimensions():
    """Tests fit and transform producing (N, 79) float32 feature matrices."""
    fusion = MultiSourceFeatureFusion()
    titles = [
        "NASA James Webb Telescope Captures Distant Galaxy",
        "SHOCKING PROOF: Secret 5G mind control frequency leaked!",
        "WHO Releases Updated Global Pediatric Health Directives",
        "Alien spacecraft spotted over capital secret bunker revealed"
    ]
    texts = [
        "According to a press release, NASA published deep spectroscopic measurements.",
        "Whistleblower reveals secret frequency! Share before deletion!",
        "Official guidelines published in The Lancet following clinical trials.",
        "Terrifying classified video leaked by underground truth group."
    ]
    meta = [
        {"source": "NASA Press Release", "author": "Dr. Sarah Jenkins", "shares": 14000, "sentiment": 0.45},
        {"source": "ConspiracyTruthWire247.net", "author": "PatriotX", "shares": 95000, "sentiment": -0.80},
        {"source": "The Lancet", "author": "Dr. Halima Traore", "shares": 41000, "sentiment": 0.50},
        {"source": "Unknown", "author": "Anon", "shares": 80000, "sentiment": -0.60}
    ]

    fusion.fit([f"{t} {b}" for t, b in zip(titles, texts)])
    X_fused = fusion.transform(titles, texts, meta)

    assert isinstance(X_fused, np.ndarray), "Output must be a numpy array"
    assert X_fused.shape == (4, 79), f"Expected shape (4, 79), got {X_fused.shape}"
    assert X_fused.dtype == np.float32, f"Expected dtype float32, got {X_fused.dtype}"
    assert not np.isnan(X_fused).any(), "NaN detected in fused feature matrix"
    assert not np.isinf(X_fused).any(), "Inf detected in fused feature matrix"
    print("[OK] Multimodal feature fusion dimensions (4, 79) and numerical stability verified.")


def test_encode_single_with_attention():
    """Verifies single-article inference encoding and XAI token attention output."""
    fusion = MultiSourceFeatureFusion()
    fusion.fit(["NASA James Webb Space Telescope discovers distant galaxy cluster with spectroscopic data."])

    title = "NASA James Webb Space Telescope Captures Distant Galaxy"
    text = "According to an official press release, astronomers confirmed the discovery."
    meta = {"source": "NASA Press Release", "author": "Dr. Sarah Jenkins", "shares": 14000, "sentiment": 0.45}

    vec, tokens, attn = fusion.encode_single_with_attention(title, text, meta)

    assert isinstance(vec, np.ndarray), "Vector must be numpy array"
    assert vec.shape == (79,), f"Expected 1D vector of shape (79,), got {vec.shape}"
    assert isinstance(tokens, list) and len(tokens) > 0, "Tokens list empty"
    assert isinstance(attn, np.ndarray), "Attention must be numpy array"
    assert len(tokens) == len(attn), f"Tokens length ({len(tokens)}) != attention length ({len(attn)})"
    assert np.isclose(np.sum(attn), 1.0, atol=1e-3), "Attention weights must sum to 1.0"
    print(f"[OK] Single-sample inference encoding & XAI attention mapping verified ({len(tokens)} tokens, 79 features).")


def test_serialization_and_deserialization():
    """Verifies saving and loading of the MultiSourceFeatureFusion model artifact."""
    fusion = MultiSourceFeatureFusion()
    corpus = ["Sample news corpus for testing serialization and deserialization."]
    fusion.fit(corpus)

    test_path = os.path.join(PROCESSED_DATA_DIR, "test_fusion_temp.joblib")
    fusion.save(test_path)
    assert os.path.exists(test_path), "Failed to save fusion model"

    loaded = MultiSourceFeatureFusion.load(test_path)
    assert loaded.is_fitted is True
    assert len(loaded.get_feature_names()) == 79

    # Compare transform outputs
    vec_orig = fusion.transform(["Title"], ["Body"])
    vec_loaded = loaded.transform(["Title"], ["Body"])
    assert np.allclose(vec_orig, vec_loaded, atol=1e-5), "Reloaded fusion model produced different features"

    # Cleanup temp
    if os.path.exists(test_path):
        os.remove(test_path)
    print("[OK] MultiSourceFeatureFusion serialization and deserialization verified.")


def test_dataset_split_fusion_and_persistence():
    """Verifies running the full Phase 7 feature fusion pipeline."""
    fusion = run_feature_fusion_pipeline()
    assert fusion.is_fitted is True
    assert os.path.exists(FEATURE_FUSION_PATH), f"Missing model artifact at {FEATURE_FUSION_PATH}"

    train_path = os.path.join(PROCESSED_DATA_DIR, "fused_train_features.npy")
    val_path = os.path.join(PROCESSED_DATA_DIR, "fused_val_features.npy")
    test_path = os.path.join(PROCESSED_DATA_DIR, "fused_test_features.npy")

    assert os.path.exists(train_path), "fused_train_features.npy missing"
    assert os.path.exists(val_path), "fused_val_features.npy missing"
    assert os.path.exists(test_path), "fused_test_features.npy missing"

    train_arr = np.load(train_path)
    val_arr = np.load(val_path)
    test_arr = np.load(test_path)

    assert train_arr.shape == (24, 79), f"Expected (24, 79), got {train_arr.shape}"
    assert val_arr.shape == (4, 79), f"Expected (4, 79), got {val_arr.shape}"
    assert test_arr.shape == (7, 79), f"Expected (7, 79), got {test_arr.shape}"
    print(f"[OK] Full Phase 7 pipeline execution verified (Train: {train_arr.shape}, Val: {val_arr.shape}, Test: {test_arr.shape}).")


if __name__ == "__main__":
    print("=" * 75)
    print("RUNNING PHASE 7 MULTI-SOURCE FEATURE FUSION AUTOMATED TEST SUITE")
    print("=" * 75)
    test_fusion_initialization()
    test_feature_names_count_and_order()
    test_fit_and_transform_dimensions()
    test_encode_single_with_attention()
    test_serialization_and_deserialization()
    test_dataset_split_fusion_and_persistence()
    print("=" * 75)
    print("ALL PHASE 7 MULTI-SOURCE FEATURE FUSION TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)
