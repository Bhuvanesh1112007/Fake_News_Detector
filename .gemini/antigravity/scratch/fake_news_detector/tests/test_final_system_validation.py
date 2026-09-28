"""
Phase 12: Final Comprehensive System Validation Suite
Module Owner: Mohamed Irfaan M, Bhuvanesh M, Harish Hariharan MU (VISTAS)

Executes exhaustive end-to-end verification across:
1. File & Artifact Integrity (Models, Reports, Documentation, Presentation Deck)
2. 79-Dimensional Multimodal Feature Fusion Contract (64-6-6-3 Split)
3. 150-Tree Random Forest Classifier Inference & Calibration
4. Dual Explainable AI (Global & Local TreeSHAP + BERT Self-Attention Heatmaps)
5. Multi-Domain Generalization & False-Real Deficit Elimination (100% Accuracy)
6. FastAPI Full-Stack REST API & Database Synchronization
"""
import os
import sys
import json
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import backend.app as app
from src.feature_fusion import MultiSourceFeatureFusion
from src.train_random_forest import ProposedRandomForestClassifier
from src.explainability import ModelExplainer
from src.config import (
    BASE_DIR, MODELS_DIR, METRICS_DIR, RANDOM_FOREST_PATH,
    FEATURE_FUSION_PATH, BASELINES_MODEL_PATH, METRICS_JSON_PATH
)


def test_1_artifact_integrity():
    """Verifies that all persisted model weights, benchmark metrics, documents, and slides exist."""
    print("\n[Validation 1/6] Verifying Repository Artifact & File Integrity...")
    required_paths = [
        RANDOM_FOREST_PATH,
        FEATURE_FUSION_PATH,
        BASELINES_MODEL_PATH,
        METRICS_JSON_PATH,
        os.path.join(METRICS_DIR, "shap_global_importance.json"),
        os.path.join(METRICS_DIR, "shap_global_importance.csv"),
        os.path.join(METRICS_DIR, "model_comparison.csv"),
        os.path.join(METRICS_DIR, "ablation_study.csv"),
        os.path.join(BASE_DIR, "docs", "IEEE_Paper_Draft.md"),
        os.path.join(BASE_DIR, "docs", "Project_Review_and_Viva_Guide.md"),
        os.path.join(BASE_DIR, "docs", "Fake_News_Detection_Review_Presentation.pptx"),
        os.path.join(BASE_DIR, "README.md"),
        os.path.join(BASE_DIR, "frontend", "index.html"),
        os.path.join(BASE_DIR, "run.py")
    ]
    for path in required_paths:
        assert os.path.exists(path), f"Required artifact missing at: {path}"
        file_size = os.path.getsize(path)
        assert file_size > 0, f"Artifact is empty (0 bytes): {path}"
        print(f"  ✓ Found: {os.path.relpath(path, BASE_DIR)} ({file_size:,} bytes)")
    print("  --> Artifact Integrity Test: PASSED [100%]")


def test_2_multimodal_feature_contract():
    """Verifies the exact 79-dimensional multimodal tensor contract and modality indexing."""
    print("\n[Validation 2/6] Verifying 79-Dimensional Multimodal Feature Space...")
    fusion = MultiSourceFeatureFusion.load(FEATURE_FUSION_PATH)
    names = fusion.get_feature_names()
    assert len(names) == 79, f"Expected 79 features, found {len(names)}"

    # Check 64 BERT dimensions
    bert_names = names[0:64]
    assert len(bert_names) == 64
    assert bert_names[0] == "bert_dim_00"
    assert bert_names[63] == "bert_dim_63"

    # Check 6 User features
    user_names = names[64:70]
    assert len(user_names) == 6
    assert user_names[0] == "publisher_credibility_score"
    assert user_names[5] == "journalistic_attribution_intensity"

    # Check 6 Network features
    net_names = names[70:76]
    assert len(net_names) == 6
    assert net_names[0] == "log_social_shares"
    assert net_names[5] == "engagement_anomaly_score"

    # Check 3 Stylometric features
    stylo_names = names[76:79]
    assert len(stylo_names) == 3
    assert stylo_names[0] == "stylometric_uppercase_ratio"
    assert stylo_names[1] == "stylometric_exclamation_density"
    assert stylo_names[2] == "stylometric_lexical_diversity"

    # Test single-article live vector extraction
    vec, tokens, attn = fusion.encode_single_with_attention(
        title="Test Article Headline",
        text="This is a test article body evaluating multimodal feature encoding.",
        metadata={"source": "Reuters", "author": "Dr. Scientist", "shares": 10000.0, "sentiment": 0.2}
    )
    assert vec.shape == (79,), f"Expected shape (79,), got {vec.shape}"
    assert isinstance(tokens, list) and len(tokens) > 0
    assert isinstance(attn, np.ndarray) and len(attn) == len(tokens)

    print(f"  ✓ 79-Dimensional Contract Verified: 64 BERT + 6 User + 6 Network + 3 Stylometrics")
    print("  --> Multimodal Feature Fusion Contract: PASSED [100%]")


def test_3_random_forest_classification_and_latency():
    """Verifies 150-tree Random Forest inference, probability calibration, and sub-10ms latency."""
    print("\n[Validation 3/6] Verifying Random Forest Ensemble & Inference Speed...")
    rf_model = ProposedRandomForestClassifier.load(RANDOM_FOREST_PATH)
    assert len(rf_model.model.estimators_) == 150, f"Expected 150 trees, got {len(rf_model.model.estimators_)}"

    dummy_input = np.random.randn(10, 79).astype(np.float32)
    probs = rf_model.predict_proba(dummy_input)
    assert probs.shape == (10, 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(10), rtol=1e-5)

    import time
    t0 = time.perf_counter()
    _ = rf_model.predict(dummy_input)
    latency_ms = ((time.perf_counter() - t0) / 10.0) * 1000.0
    assert latency_ms < 10.0, f"Latency {latency_ms:.2f}ms exceeded 10ms threshold"

    print(f"  ✓ 150 Decision Trees loaded with OOB score: {rf_model.oob_accuracy_*100:.1f}%")
    print(f"  ✓ Average Inference Latency: {latency_ms:.3f} ms per sample (< 10ms target)")
    print("  --> Random Forest Classifier Test: PASSED [100%]")


def test_4_dual_explainable_ai():
    """Verifies Dual XAI (TreeSHAP directional factor attribution + BERT attention heatmaps)."""
    print("\n[Validation 4/6] Verifying Dual Explainable AI (SHAP & Attention)...")
    app.load_ai_engine()
    explainer = app.EXPLAINER

    # 1. Global SHAP Rankings
    global_shap = explainer.get_global_importance(top_k=79)
    assert len(global_shap) == 79
    modalities_present = set(f["modality"] for f in global_shap)
    assert modalities_present == {
        "BERT Contextual Sequence", "User & Publisher Credibility",
        "Network & Social Engagement", "Stylometric Linguistic Density"
    }

    # 2. Local Instance Explanation
    sample_vec = np.zeros(79, dtype=np.float32)
    sample_vec[64] = 0.05  # low credibility
    sample_vec[70] = 0.85  # high shares
    sample_vec[77] = 0.90  # high sensationalism

    exp = explainer.explain_prediction(
        feature_vector=sample_vec,
        title="BANNED SECRET CURE LEAKED!",
        text="Whistleblower exposes miraculous 100% cancer cure suppressed by big pharma!",
        prediction_label="FAKE",
        fake_prob=0.85,
        bert_tokens=["[CLS]", "banned", "secret", "cure", "leaked", "[SEP]"],
        bert_attention=np.eye(6, dtype=np.float32),
        top_k_features=6
    )

    assert "top_features" in exp and len(exp["top_features"]) > 0
    assert "token_attributions" in exp and len(exp["token_attributions"]) > 0
    assert "key_reasons" in exp and len(exp["key_reasons"]) > 0

    print(f"  ✓ Global SHAP ranked across all 4 modalities (Top feature: {global_shap[0]['feature']} - {global_shap[0]['importance_percent']:.2f}%)")
    print(f"  ✓ Local token heatmap generated with {len(exp['token_attributions'])} token attributions")
    print("  --> Dual Explainable AI (XAI) Test: PASSED [100%]")


def test_5_multi_domain_generalization_benchmarks():
    """Verifies 100% accuracy across 12 diverse real-world test cases in health, science, scams, and news."""
    print("\n[Validation 5/6] Verifying Multi-Domain Generalization & False-Real Resolution...")
    app.load_ai_engine()

    test_cases = [
        # Health & Medical Misinformation (FAKE)
        ("Drinking Kerosene Cures Diabetes Overnight", "Doctors in secret clinic revealed drinking kerosene eliminates diabetes in 48 hours.", "FAKE"),
        ("Raw Garlic and Vinegar Eliminates All Heart Blockages", "Banned cardiologists leak secret recipe of garlic and vinegar that dissolves arterial plaque instantly!", "FAKE"),
        ("Boiled Lemon Peels Cure 100% of Cancer in 48 Hours", "Big pharma is furious! A miraculous ancient home remedy dissolves all cancer cells in two days!", "FAKE"),
        # Scams & Giveaways (FAKE)
        ("Government Distributing Free Laptops to WhatsApp Users", "Every student who forwards this WhatsApp link to 5 groups will get a free high-end laptop delivered!", "FAKE"),
        ("Secret Telegram Channel Leaks Free Bitcoin Codes", "Claim free 1.5 BTC by clicking this link and verifying your digital wallet before midnight!", "FAKE"),
        # Conspiracy & Hoaxes (FAKE)
        ("Chemtrails Confirmed to Contain Mind-Altering Parasites", "Whistleblower pilots exposed commercial airlines are spraying subterranean mind-control chemicals on cities.", "FAKE"),
        ("Time Traveler from 2085 Predicts Worldwide Blackout Next Month", "Viral video shows man claiming to be from 2085 warning that global electrical grid will shut down permanently.", "FAKE"),
        # Authentic Scientific & Medical Research (REAL)
        ("Clinical Trial Demonstrates Efficacy of New Immunotherapy Drug", "Researchers published findings in The Lancet showing a 40 percent improvement in five-year survival rates.", "REAL"),
        ("Astronomers Detect Gravitational Waves from Binary Neutron Star Merger", "The international astrophysics collaboration announced the observation of gravitational wave signals confirmed by optical observatories.", "REAL"),
        # Authentic Governance & Public Infrastructure (REAL)
        ("Ministry of Finance Extends Income Tax Filing Deadline by Two Weeks", "The department of revenue notified taxpayers that the annual filing deadline has been extended to ensure smooth portal operations.", "REAL"),
        ("Public Works Department Inaugurates New Flyover to Ease Urban Traffic", "City municipal authorities formally opened the four-lane elevated corridor connecting the tech park with the highway.", "REAL"),
        ("State University Releases Semester Examination Timetable", "The controller of examinations released the schedule and guidelines for the upcoming semester examinations on the student portal.", "REAL")
    ]

    passed_count = 0
    for title, text, expected in test_cases:
        req = app.PredictRequest(title=title, text=text, source="Web")
        res = app.predict_news(req)
        assert res.prediction == expected, f"Failed on '{title}': Expected {expected}, got {res.prediction} (Fake Prob: {res.confidence_fake}%)"
        passed_count += 1
        print(f"  ✓ [{res.prediction:<4}] (Fake: {res.confidence_fake:>5.1f}%, Real: {res.confidence_real:>5.1f}%) | {title[:45]}")

    assert passed_count == len(test_cases)
    print(f"  --> Multi-Domain Benchmark Accuracy: {passed_count}/{len(test_cases)} (100.0%) [PASSED]")


def test_6_fastapi_rest_endpoints():
    """Verifies all FastAPI REST endpoints using TestClient."""
    print("\n[Validation 6/6] Verifying Full-Stack FastAPI REST API Endpoints...")
    client = TestClient(app.app)

    # 1. /api/health
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "online"
    assert r.json()["feature_dimensions"] == 79
    print("  ✓ GET /api/health: 200 OK")

    # 2. /api/predict (Single Inference)
    r = client.post("/api/predict", json={
        "title": "NASA Rover Discovers Ancient Riverbed on Mars",
        "text": "According to official data published by NASA, the Perseverance rover identified sedimentary rock structures indicating ancient water flow.",
        "source": "NASA Press Release",
        "author": "Dr. Sarah Jenkins",
        "shares": 18000.0,
        "sentiment": 0.25
    })
    assert r.status_code == 200
    res = r.json()
    assert res["prediction"] == "REAL"
    assert "token_attributions" in res
    assert "top_features" in res
    print(f"  ✓ POST /api/predict (Single): 200 OK (Verdict: {res['prediction']}, Latency: {res['inference_latency_ms']}ms)")

    # 3. /api/predict-batch (Batch Inference)
    r = client.post("/api/predict-batch", json={
        "items": [
            {"title": "Free Money Giveaway 100% Cash Bonus", "text": "Claim 50000 cash bonus now by forwarding to whatsapp contacts!", "source": "Web"},
            {"title": "Weather Alert: Heavy Rain Advisory", "text": "Meteorological department issued localized rainfall advisory for coastal areas.", "source": "State Weather Dept"}
        ]
    })
    assert r.status_code == 200
    assert r.json()["total_processed"] == 2
    print(f"  ✓ POST /api/predict-batch: 200 OK (Processed 2 articles in {r.json()['total_time_ms']}ms)")

    # 4. /api/metrics (Benchmark Metrics)
    r = client.get("/api/metrics")
    assert r.status_code == 200
    assert "proposed_model" in r.json()
    assert "ablation_study" in r.json()
    print("  ✓ GET /api/metrics: 200 OK")

    # 5. /api/shap/global (Global SHAP)
    r = client.get("/api/shap/global")
    assert r.status_code == 200
    assert r.json()["total_features"] == 79
    print("  ✓ GET /api/shap/global: 200 OK (79 features grouped by modality)")

    # 6. /api/history & /api/history/stats
    r = client.get("/api/history")
    assert r.status_code == 200
    r_stats = client.get("/api/history/stats")
    assert r_stats.status_code == 200
    assert r_stats.json()["total_predictions"] >= 1
    print(f"  ✓ GET /api/history & /api/history/stats: 200 OK (Total audited: {r_stats.json()['total_predictions']})")
    print("  --> FastAPI REST API Integration: PASSED [100%]")


def run_all_system_validations():
    """Master orchestrator for Phase 12 Final System Validation."""
    print("=" * 85)
    print("🛡️ VERISHIELD AI: PHASE 12 FINAL SYSTEM VALIDATION & PUBLICATION AUDIT")
    print("🏛️ Vels Institute of Science, Technology & Advanced Studies (VISTAS)")
    print("👨‍💻 Team: Mohamed Irfaan M | Bhuvanesh M | Harish Hariharan MU")
    print("=" * 85)

    test_1_artifact_integrity()
    test_2_multimodal_feature_contract()
    test_3_random_forest_classification_and_latency()
    test_4_dual_explainable_ai()
    test_5_multi_domain_generalization_benchmarks()
    test_6_fastapi_rest_endpoints()

    print("\n" + "=" * 85)
    print("🏆 ALL PHASE 12 SYSTEM VALIDATION TESTS COMPLETED SUCCESSFULLY WITH 100% PASS RATE!")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    run_all_system_validations()
