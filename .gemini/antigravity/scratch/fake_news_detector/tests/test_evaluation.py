"""
Phase 10 Model Evaluation & Benchmark Reporting Automated Test Suite
Module Owner: Bhuvanesh M (Lead Data & Ensemble Learning Engineer) & Mohamed Irfaan M (Lead NLP Engineer)

Verifies:
1. Classification metric calculations, specificity/sensitivity, and numerical bounds
2. Confusion matrix accounting (TP, FP, TN, FN)
3. Model comparison table generation and CSV formatting
4. Multimodal ablation study execution across 6 architectural settings
5. Multi-format artifact persistence (model_comparison.csv, comprehensive_benchmarks.csv, ablation_study.csv, metrics.json)
6. Full Phase 10 evaluation pipeline execution
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

from src.evaluation import (
    calculate_metrics, ModelEvaluator, run_evaluation_pipeline
)
from src.config import (
    METRICS_COMPARISON_PATH, METRICS_JSON_PATH, METRICS_DIR,
    PROCESSED_DATA_DIR
)


def test_calculate_metrics_bounds():
    """Verifies classification metric calculation, bounds, and confusion matrix consistency."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    y_prob = np.array([
        [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.4, 0.6],
        [0.2, 0.8], [0.1, 0.9], [0.3, 0.7], [0.6, 0.4]
    ])

    m = calculate_metrics(y_true, y_pred, y_prob, latency_sec=0.01)

    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["precision_weighted"] <= 1.0
    assert 0.0 <= m["recall_weighted"] <= 1.0
    assert 0.0 <= m["f1_score"] <= 1.0
    assert 0.0 <= m["specificity"] <= 1.0
    assert 0.0 <= m["sensitivity"] <= 1.0
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert m["inference_latency_ms"] > 0.0

    cm = m["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2
    assert (m["true_negatives"] + m["false_positives"] + m["false_negatives"] + m["true_positives"]) == 8
    print("[OK] Classification metrics and confusion matrix accounting verified.")


def test_ablation_study_execution():
    """Verifies systematic ablation study execution across all 6 configurations."""
    np.random.seed(42)
    N_tr, N_te, D = 30, 8, 79
    X_train = np.random.randn(N_tr, D)
    y_train = np.array([0] * 15 + [1] * 15)
    X_test = np.random.randn(N_te, D)
    y_test = np.array([0] * 4 + [1] * 4)

    evaluator = ModelEvaluator()
    ablations = evaluator.run_ablation_study(X_train, y_train, X_test, y_test)

    assert len(ablations) == 6, f"Expected 6 ablation settings, got {len(ablations)}"

    expected_dims = [79, 64, 6, 6, 3, 15]
    for item, d in zip(ablations, expected_dims):
        assert item["feature_dimensions"] == d
        assert 0.0 <= item["test_accuracy"] <= 1.0
        assert 0.0 <= item["test_f1"] <= 1.0
        assert 0.0 <= item["cv_accuracy_mean"] <= 1.0

    print(f"[OK] Multimodal ablation study verified across all 6 settings ({len(ablations)} configurations evaluated).")


def test_export_csv_and_json_integrity():
    """Verifies exporting and parsing of comparison CSVs and metrics JSON."""
    evaluator = ModelEvaluator()
    sample_rows = [
        {"Model": "Logistic Regression L2", "Accuracy": 85.71, "Precision": 0.8800, "Recall": 0.8571, "F1": 0.8540, "ROC-AUC": 0.9167},
        {"Model": "Proposed Multimodal RF (Ours)", "Accuracy": 100.0, "Precision": 1.0000, "Recall": 1.0000, "F1": 1.0000, "ROC-AUC": 1.0000}
    ]
    test_csv = os.path.join(PROCESSED_DATA_DIR, "test_comp_temp.csv")
    evaluator.export_model_comparison_table(sample_rows, test_csv)

    assert os.path.exists(test_csv)
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[1]["Model"] == "Proposed Multimodal RF (Ours)"
        assert reader[1]["Accuracy"] == "100.00%"

    if os.path.exists(test_csv):
        os.remove(test_csv)
    print("[OK] Comparison table CSV formatting and export integrity verified.")


def test_evaluation_pipeline_execution():
    """Verifies full execution of the Phase 10 evaluation and benchmark reporting pipeline."""
    payload = run_evaluation_pipeline()

    assert "proposed_model" in payload
    assert "baselines" in payload
    assert "ablation_study" in payload
    assert "model_comparison" in payload

    # 1. model_comparison.csv
    assert os.path.exists(METRICS_COMPARISON_PATH)
    with open(METRICS_COMPARISON_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
        assert len(rows) == 8  # Header + 6 baselines + 1 proposed = 8 rows

    # 2. comprehensive_benchmarks.csv
    comp_csv = os.path.join(METRICS_DIR, "comprehensive_benchmarks.csv")
    assert os.path.exists(comp_csv)
    with open(comp_csv, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
        assert len(rows) == 8

    # 3. ablation_study.csv & .json
    abl_csv = os.path.join(METRICS_DIR, "ablation_study.csv")
    abl_json = os.path.join(METRICS_DIR, "ablation_study.json")
    assert os.path.exists(abl_csv) and os.path.exists(abl_json)

    with open(abl_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert len(data) == 6

    # 4. metrics.json
    assert os.path.exists(METRICS_JSON_PATH)

    print(f"[OK] Full Phase 10 pipeline execution verified (7 Models compared, 6 Ablations exported).")


if __name__ == "__main__":
    print("=" * 75)
    print("RUNNING PHASE 10 MODEL EVALUATION & BENCHMARKS AUTOMATED TEST SUITE")
    print("=" * 75)
    test_calculate_metrics_bounds()
    test_ablation_study_execution()
    test_export_csv_and_json_integrity()
    test_evaluation_pipeline_execution()
    print("=" * 75)
    print("ALL PHASE 10 MODEL EVALUATION & BENCHMARKS TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)
