"""
Phase 9 Explainable AI (XAI) & SHAP Automated Test Suite
Module Owner: Harish Hariharan MU (Lead Explainable AI & Full-Stack Integration Engineer)

Verifies:
1. XAIExplainer & ModelExplainer initialization
2. 79-Dimensional Global Feature Importance extraction & modality categorization
3. Local instance-level SHAP attributions and signed directional push
4. Token-level self-attention heatmap alignment and lexical scoring
5. Dynamic journalistic evidence rationale generation
6. Artifact persistence (shap_global_importance.json, shap_global_importance.csv, sample_explanations.json)
"""
import os
import sys
import csv
import json
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.explainability import ModelExplainer, run_explainability_pipeline
from src.explainer import XAIExplainer, get_feature_modality
from src.feature_fusion import MultiSourceFeatureFusion
from src.train_random_forest import ProposedRandomForestClassifier
from src.config import METRICS_DIR, RANDOM_FOREST_PATH


def test_explainer_initialization():
    """Verifies XAIExplainer initialization and modality mapping."""
    fusion = MultiSourceFeatureFusion()
    feature_names = fusion.get_feature_names()
    assert len(feature_names) == 79

    explainer = ModelExplainer(feature_names=feature_names)
    assert explainer.model is not None
    assert len(explainer.feature_names) == 79
    assert len(explainer.explainer.fake_indicators) > 20
    assert len(explainer.explainer.real_indicators) > 20
    print("[OK] ModelExplainer and XAIExplainer initialization verified.")


def test_global_feature_importance_modalities():
    """Verifies 79-dimensional global importance calculation, ranking, and modality groupings."""
    explainer = ModelExplainer()
    global_imp = explainer.get_global_importance(top_k=79)

    assert len(global_imp) == 79, f"Expected 79 features, got {len(global_imp)}"
    assert global_imp[0]["rank"] == 1
    assert global_imp[-1]["rank"] == 79

    # Assert descending order
    for i in range(len(global_imp) - 1):
        assert global_imp[i]["importance"] >= global_imp[i+1]["importance"]

    # Verify presence of all 4 architectural modalities
    modalities = {item["modality"] for item in global_imp}
    expected_modalities = {
        "BERT Contextual Sequence",
        "User & Publisher Credibility",
        "Network & Social Engagement",
        "Stylometric Linguistic Density"
    }
    assert expected_modalities.issubset(modalities), f"Missing modalities: {expected_modalities - modalities}"

    total_percent = sum(item["importance_percent"] for item in global_imp)
    assert np.isclose(total_percent, 100.0, atol=1.0), f"Total percentage should sum to ~100%, got {total_percent}"
    print(f"[OK] 79-Dimensional global SHAP importance verified across all 4 modalities (Top: {global_imp[0]['feature']}={global_imp[0]['importance_percent']}%).")


def test_local_shap_feature_attributions():
    """Verifies local signed SHAP attributions and directional push."""
    explainer = ModelExplainer()
    feature_names = explainer.feature_names

    # Create dummy fake feature vector (high virality, low credibility)
    fake_vec = np.zeros(79, dtype=np.float32)
    fake_vec[70] = 0.95  # log_social_shares
    fake_vec[71] = 1.00  # hyper_virality_indicator
    fake_vec[64] = 0.10  # publisher_credibility_score

    explanation = explainer.explain_prediction(
        feature_vector=fake_vec,
        title="SHOCKING SECRET EXPOSED",
        text="Whistleblower leaks classified documents.",
        prediction_label="FAKE",
        fake_prob=0.92,
        top_k_features=5
    )

    assert explanation["prediction"] == "FAKE"
    assert explanation["confidence_fake"] == 92.0
    assert len(explanation["top_features"]) == 5

    for feat in explanation["top_features"]:
        assert "feature" in feat and "impact" in feat and "direction" in feat
        assert "modality" in feat and "value" in feat
        if feat["impact"] > 0:
            assert feat["direction"] == "Pushes towards FAKE"
        elif feat["impact"] < 0:
            assert feat["direction"] == "Pushes towards REAL"

    print("[OK] Local signed SHAP attributions and directional push verified.")


def test_token_attention_heatmaps():
    """Verifies token-level self-attention heatmap alignment and lexical scoring."""
    explainer = ModelExplainer()
    vec = np.zeros(79, dtype=np.float32)

    tokens = ["[CLS]", "nasa", "james", "webb", "discovers", "galaxy", "[SEP]"]
    attn = np.array([0.1, 0.3, 0.2, 0.2, 0.1, 0.1, 0.0], dtype=np.float32)

    title = "NASA James Webb Discovers Galaxy"
    text = "Confirmed spectroscopic measurements published in Nature."

    explanation = explainer.explain_prediction(
        feature_vector=vec,
        title=title,
        text=text,
        prediction_label="REAL",
        fake_prob=0.08,
        bert_tokens=tokens,
        bert_attention=attn
    )

    token_attribs = explanation["token_attributions"]
    assert len(token_attribs) > 0

    tokens_found = [t["token"].lower() for t in token_attribs]
    assert any("nasa" in t for t in tokens_found)

    # Check that "confirmed" / "nature" receives real signal
    real_words = [t for t in token_attribs if t["type"] == "real_signal"]
    assert len(real_words) > 0
    print(f"[OK] Token-level attention heatmap and lexical scoring verified ({len(token_attribs)} tokens mapped).")


def test_evidence_rationale_generation():
    """Verifies automated journalistic rationale synthesis for Real and Fake verdicts."""
    explainer = ModelExplainer()
    vec = np.zeros(79, dtype=np.float32)

    exp_fake = explainer.explain_prediction(
        feature_vector=vec,
        title="SHOCKING 5G MIND CONTROL FREQUENCY!",
        text="Mind control frequency leaked by whistleblower! Share before deleted!",
        prediction_label="FAKE",
        fake_prob=0.88
    )
    assert len(exp_fake["key_reasons"]) >= 2
    assert any("clickbait" in r.lower() or "sensationalist" in r.lower() or "stylometric" in r.lower() for r in exp_fake["key_reasons"])

    exp_real = explainer.explain_prediction(
        feature_vector=vec,
        title="NASA Publishes Peer-Reviewed Study",
        text="Astronomers confirmed new observations in The Lancet.",
        prediction_label="REAL",
        fake_prob=0.12
    )
    assert len(exp_real["key_reasons"]) >= 2
    assert any("objective" in r.lower() or "attribution" in r.lower() or "contextual" in r.lower() for r in exp_real["key_reasons"])
    print("[OK] Dynamic journalistic evidence rationale synthesis verified.")


def test_explainability_pipeline_persistence():
    """Verifies execution of Phase 9 pipeline and persisted JSON/CSV reports."""
    explainer, global_imp, sample_exps = run_explainability_pipeline()
    assert len(global_imp) == 79
    assert len(sample_exps) >= 2

    # 1. Global JSON
    json_path = os.path.join(METRICS_DIR, "shap_global_importance.json")
    assert os.path.exists(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert len(data) == 79

    # 2. Global CSV
    csv_path = os.path.join(METRICS_DIR, "shap_global_importance.csv")
    assert os.path.exists(csv_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
        assert len(rows) == 80  # Header + 79 rows

    # 3. Sample Explanations JSON
    exp_path = os.path.join(METRICS_DIR, "sample_explanations.json")
    assert os.path.exists(exp_path)
    with open(exp_path, "r", encoding="utf-8") as f:
        exp_data = json.load(f)
        assert len(exp_data) == 2
        for item in exp_data:
            assert "prediction" in item and "top_features" in item and "token_attributions" in item and "key_reasons" in item

    print("[OK] Phase 9 Explainability pipeline persistence and report integrity verified.")


if __name__ == "__main__":
    print("=" * 75)
    print("RUNNING PHASE 9 EXPLAINABLE AI (XAI) & SHAP AUTOMATED TEST SUITE")
    print("=" * 75)
    test_explainer_initialization()
    test_global_feature_importance_modalities()
    test_local_shap_feature_attributions()
    test_token_attention_heatmaps()
    test_evidence_rationale_generation()
    test_explainability_pipeline_persistence()
    print("=" * 75)
    print("ALL PHASE 9 EXPLAINABLE AI (XAI) & SHAP TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)
