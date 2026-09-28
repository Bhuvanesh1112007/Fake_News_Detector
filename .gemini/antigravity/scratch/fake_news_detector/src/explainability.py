"""
Explainable AI (XAI) & SHAP Pipeline Module (Phase 9)
Module Owner: Harish Hariharan MU (Lead Explainable AI & Full-Stack Integration Engineer)

Orchestrates:
1. Global SHAP Feature Importance Ranking across all 79 fused dimensions
2. Exporting global SHAP tables to results/metrics/shap_global_importance.json & .csv
3. Instance-level local explanations (Feature impact, WordPiece attention heatmaps, Evidence rationales)
4. Persisting benchmark instance explanations to results/metrics/sample_explanations.json
"""
import os
import sys
import csv
import json
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from .explainer import XAIExplainer, get_feature_modality
from .train_random_forest import ProposedRandomForestClassifier, run_random_forest_training_pipeline
from .feature_fusion import MultiSourceFeatureFusion, run_feature_fusion_pipeline
from .config import (
    RANDOM_FOREST_PATH, FEATURE_FUSION_PATH, METRICS_DIR,
    PROCESSED_TEST_PATH, PROCESSED_DATA_DIR
)


class ModelExplainer:
    """
    Unified Explainable AI Interface:
    Wraps ProposedRandomForestClassifier, MultiSourceFeatureFusion, and XAIExplainer.
    """
    def __init__(self, model=None, feature_names: list[str] = None):
        if model is None:
            if os.path.exists(RANDOM_FOREST_PATH):
                self.model = ProposedRandomForestClassifier.load(RANDOM_FOREST_PATH)
            else:
                self.model, _ = run_random_forest_training_pipeline()
        else:
            self.model = model

        if feature_names is None:
            fusion = MultiSourceFeatureFusion()
            self.feature_names = fusion.get_feature_names()
        else:
            self.feature_names = feature_names

        self.explainer = XAIExplainer(self.model, self.feature_names)

    def get_global_importance(self, top_k: int = 79) -> list[dict]:
        """Calculates global feature importance scores across all 79 fused dimensions."""
        return self.explainer.get_global_feature_importance(top_k=top_k)

    def explain_prediction(self, feature_vector: np.ndarray, title: str, text: str,
                           prediction_label: str, fake_prob: float,
                           bert_tokens: list[str] = None, bert_attention: np.ndarray = None,
                           top_k_features: int = 8) -> dict:
        """Generates detailed local explanation for a single article."""
        return self.explainer.explain_instance(
            feature_vector=feature_vector,
            title=title,
            text=text,
            prediction_label=prediction_label,
            fake_prob=fake_prob,
            bert_tokens=bert_tokens,
            bert_attention=bert_attention,
            top_k_features=top_k_features
        )


def run_explainability_pipeline() -> tuple[ModelExplainer, list[dict], list[dict]]:
    """
    Executes Phase 9 Explainability Pipeline:
    1. Loads trained Random Forest model and MultiSourceFeatureFusion.
    2. Computes global SHAP feature importances over 79 dimensions.
    3. Persists shap_global_importance.json and shap_global_importance.csv in results/metrics/.
    4. Evaluates local explanations for benchmark Real and Fake articles.
    5. Saves sample_explanations.json in results/metrics/.
    6. Displays terminal XAI summary report.
    """
    print("=" * 80)
    print("PHASE 9: EXPLAINABLE AI (XAI) & SHAP PIPELINE")
    print("=" * 80)

    # 1. Ensure Model and Fusion Artifacts
    if not os.path.exists(RANDOM_FOREST_PATH):
        print("Random Forest model not found. Running Phase 8 training...")
        run_random_forest_training_pipeline()

    fusion = MultiSourceFeatureFusion.load(FEATURE_FUSION_PATH) if os.path.exists(FEATURE_FUSION_PATH) else run_feature_fusion_pipeline()
    feature_names = fusion.get_feature_names()

    explainer = ModelExplainer(feature_names=feature_names)

    # 2. Global Feature Importance Calculation
    print("\n[Step 1/2] Computing Global Feature Attributions across 79 Dimensions...")
    global_importance = explainer.get_global_importance(top_k=79)

    os.makedirs(METRICS_DIR, exist_ok=True)
    # Save JSON
    json_path = os.path.join(METRICS_DIR, "shap_global_importance.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(global_importance, f, indent=2)

    # Save CSV
    csv_path = os.path.join(METRICS_DIR, "shap_global_importance.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Feature Name", "Raw Key", "SHAP Importance Weight", "Importance Share (%)", "Modality"])
        for item in global_importance:
            writer.writerow([
                item["rank"], item["feature"], item["raw_name"],
                f"{item['importance']:.6f}", f"{item['importance_percent']:.2f}%", item["modality"]
            ])

    print(f"  - Saved global SHAP importance JSON: {json_path}")
    print(f"  - Saved global SHAP importance CSV:  {csv_path}")

    # 3. Instance Explanations on Benchmark Articles
    print("\n[Step 2/2] Generating Local Explanations & Attention Heatmaps for Benchmarks...")
    benchmark_samples = [
        {
            "title": "NASA James Webb Space Telescope Captures Distant Galaxy",
            "text": "According to a press release, NASA published deep spectroscopic measurements confirming ancient galactic formations.",
            "metadata": {"source": "NASA Press Release", "author": "Dr. Sarah Jenkins", "shares": 14000.0, "sentiment": 0.45},
            "expected_label": "REAL"
        },
        {
            "title": "SHOCKING PROOF: Secret 5G Mind Control Frequency Leaked!",
            "text": "Whistleblower reveals classified 5G frequency designed to control population! Mind control towers exposed! Share immediately before this is banned!",
            "metadata": {"source": "ConspiracyTruthWire247.net", "author": "PatriotX", "shares": 95000.0, "sentiment": -0.80},
            "expected_label": "FAKE"
        }
    ]

    sample_explanations = []
    for sample in benchmark_samples:
        vec, tokens, attn = fusion.encode_single_with_attention(
            sample["title"], sample["text"], sample["metadata"]
        )
        pred_class = int(explainer.model.predict(vec.reshape(1, -1))[0])
        probs = explainer.model.predict_proba(vec.reshape(1, -1))[0]
        fake_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        pred_label = "FAKE" if fake_prob >= 0.5 else "REAL"

        explanation = explainer.explain_prediction(
            feature_vector=vec,
            title=sample["title"],
            text=sample["text"],
            prediction_label=pred_label,
            fake_prob=fake_prob,
            bert_tokens=tokens,
            bert_attention=attn,
            top_k_features=6
        )
        explanation["input_title"] = sample["title"]
        explanation["expected_label"] = sample["expected_label"]
        sample_explanations.append(explanation)

    explanations_path = os.path.join(METRICS_DIR, "sample_explanations.json")
    with open(explanations_path, "w", encoding="utf-8") as f:
        json.dump(sample_explanations, f, indent=2)

    print(f"  - Saved sample instance explanations: {explanations_path}")

    # 4. Display Diagnostic Summary
    print("\n" + "=" * 80)
    print("🔍 EXPLAINABLE AI (XAI) DIAGNOSTIC BENCHMARK")
    print("=" * 80)
    print("Top 6 Globally Influential Features across Multimodal Ensemble:")
    for item in global_importance[:6]:
        print(f"  #{item['rank']:<2} {item['feature']:<36} | {item['importance_percent']:>5.2f}% | {item['modality']}")
    print("-" * 80)

    for i, exp in enumerate(sample_explanations, 1):
        print(f"\nInstance #{i}: \"{exp['input_title']}\"")
        print(f"  Verdict: {exp['prediction']} (Fake: {exp['confidence_fake']}%, Real: {exp['confidence_real']}%)")
        print("  Key Reasons:")
        for r in exp["key_reasons"]:
            print(f"    • {r}")
        print("  Top Local Feature Attributions:")
        for feat in exp["top_features"][:3]:
            print(f"    - {feat['feature']:<32} | Impact: {feat['impact']:+.4f} ({feat['direction']})")

    print("\n" + "=" * 80 + "\n")
    return explainer, global_importance, sample_explanations


if __name__ == "__main__":
    run_explainability_pipeline()
