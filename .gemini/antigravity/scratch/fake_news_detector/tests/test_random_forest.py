"""
Phase 8 Random Forest Classifier Automated Test Suite
Module Owner: Bhuvanesh M (Lead Data & Ensemble Learning Engineer)

Verifies:
1. Proposed Random Forest ensemble initialization and hyperparameter configuration
2. Model training, convergence, and Out-of-Bag (OOB) accuracy calculation
3. Calibrated class probability distributions (sum to 1.0)
4. 79-Dimensional Mean Decrease in Impurity (MDI) feature importance extraction
5. Evaluation metrics calculation and confusion matrix structure
6. Model serialization, disk persistence (.pkl/.joblib), and deserialization consistency
7. End-to-end pipeline execution and benchmark artifact persistence
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

from src.train_random_forest import (
    ProposedRandomForestClassifier, run_random_forest_training_pipeline
)
from src.config import (
    RANDOM_FOREST_PATH, RANDOM_SEED, METRICS_DIR,
    PROCESSED_DATA_DIR
)


def test_rf_initialization():
    """Verifies ProposedRandomForestClassifier initialization and hyperparameter settings."""
    clf = ProposedRandomForestClassifier(random_state=RANDOM_SEED)
    assert clf.n_estimators == 150, f"Expected 150 trees, got {clf.n_estimators}"
    assert clf.max_depth == 12, f"Expected max_depth 12, got {clf.max_depth}"
    assert clf.class_weight == "balanced"
    assert clf.oob_score is True
    assert clf.is_fitted is False

    # Safety guard: predict on un-fitted model
    try:
        clf.predict(np.zeros((1, 79)))
        assert False, "Should have raised ValueError on un-fitted predict"
    except ValueError:
        pass
    print("[OK] ProposedRandomForestClassifier initialization and hyperparameter guards verified.")


def test_rf_training_and_oob_score():
    """Tests fitting on 79-dimensional features and Out-of-Bag (OOB) scoring."""
    np.random.seed(RANDOM_SEED)
    N, D = 40, 79
    X_train = np.random.randn(N, D)
    y_train = np.random.choice([0, 1], size=N)

    clf = ProposedRandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)

    assert clf.is_fitted is True
    assert 0.0 <= clf.oob_accuracy_ <= 1.0, f"Invalid OOB accuracy: {clf.oob_accuracy_}"
    print(f"[OK] Random Forest training and OOB scoring verified (OOB Accuracy: {clf.oob_accuracy_*100:.1f}%).")


def test_rf_prediction_and_probability_calibration():
    """Verifies prediction outputs and probability calibration."""
    np.random.seed(RANDOM_SEED)
    X_train = np.random.randn(30, 79)
    y_train = np.array([0]*15 + [1]*15)
    X_test = np.random.randn(8, 79)

    clf = ProposedRandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    assert len(preds) == 8
    assert set(preds).issubset({0, 1})
    assert probs.shape == (8, 2)
    
    # Probabilities must sum to 1.0
    prob_sums = np.sum(probs, axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-5), "Class probabilities do not sum to 1.0"
    print("[OK] Binary class predictions and probability calibration verified.")


def test_feature_importance_dimensions():
    """Verifies 79-dimensional MDI feature importance extraction and ranking."""
    np.random.seed(RANDOM_SEED)
    X_train = np.random.randn(30, 79)
    y_train = np.array([0]*15 + [1]*15)

    clf = ProposedRandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)

    feature_names = [f"feat_{i:02d}" for i in range(79)]
    ranked = clf.get_feature_importances(feature_names)

    assert len(ranked) == 79, f"Expected 79 ranked features, got {len(ranked)}"
    assert ranked[0]["rank"] == 1
    assert ranked[-1]["rank"] == 79
    
    # Assert descending order
    for i in range(len(ranked) - 1):
        assert ranked[i]["importance"] >= ranked[i+1]["importance"]

    total_imp = sum(item["importance"] for item in ranked)
    assert np.isclose(total_imp, 1.0, atol=1e-3), f"Total feature importance must sum to 1.0, got {total_imp}"
    print(f"[OK] 79-Dimensional MDI feature importance ranking verified (Top: {ranked[0]['feature']}={ranked[0]['importance']:.4f}).")


def test_evaluation_metric_bounds():
    """Tests comprehensive metric evaluation on test set."""
    np.random.seed(RANDOM_SEED)
    X_train = np.random.randn(30, 79)
    y_train = np.array([0]*15 + [1]*15)
    X_test = np.random.randn(10, 79)
    y_test = np.array([0]*5 + [1]*5)

    clf = ProposedRandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    metrics = clf.evaluate(X_test, y_test)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision_weighted"] <= 1.0
    assert 0.0 <= metrics["recall_weighted"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["inference_latency_ms"] >= 0.0

    cm = metrics["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2
    assert (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]) == len(y_test)
    print("[OK] Comprehensive evaluation metrics and confusion matrix verified.")


def test_serialization_and_deserialization():
    """Verifies model saving and reloading from disk."""
    np.random.seed(RANDOM_SEED)
    X_train = np.random.randn(20, 79)
    y_train = np.array([0]*10 + [1]*10)
    X_test = np.random.randn(5, 79)

    clf = ProposedRandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)

    test_path = os.path.join(PROCESSED_DATA_DIR, "test_rf_temp.pkl")
    clf.save(test_path)
    assert os.path.exists(test_path)

    loaded = ProposedRandomForestClassifier.load(test_path)
    assert loaded.is_fitted is True

    orig_probs = clf.predict_proba(X_test)
    loaded_probs = loaded.predict_proba(X_test)
    assert np.allclose(orig_probs, loaded_probs, atol=1e-5), "Reloaded model produced different probabilities"

    # Cleanup temp
    if os.path.exists(test_path):
        os.remove(test_path)
    alt_temp = test_path.replace(".pkl", ".joblib")
    if os.path.exists(alt_temp):
        os.remove(alt_temp)
    print("[OK] Model serialization, persistence, and deserialization verified.")


def test_pipeline_execution_and_artifacts():
    """Verifies execution of full Phase 8 training pipeline and generated artifacts."""
    rf_clf, metrics = run_random_forest_training_pipeline()
    assert rf_clf.is_fitted is True

    # 1. Model Artifacts
    assert os.path.exists(RANDOM_FOREST_PATH), f"Missing model at {RANDOM_FOREST_PATH}"

    # 2. Evaluation Metrics JSON
    metrics_path = os.path.join(METRICS_DIR, "rf_evaluation_metrics.json")
    assert os.path.exists(metrics_path), f"Missing metrics JSON at {metrics_path}"
    with open(metrics_path, "r", encoding="utf-8") as f:
        m_data = json.load(f)
        assert "accuracy" in m_data and "f1_score" in m_data and "roc_auc" in m_data

    # 3. Feature Importance Reports
    fi_json_path = os.path.join(METRICS_DIR, "rf_feature_importance.json")
    fi_csv_path = os.path.join(METRICS_DIR, "rf_feature_importance.csv")

    assert os.path.exists(fi_json_path), f"Missing feature importance JSON at {fi_json_path}"
    assert os.path.exists(fi_csv_path), f"Missing feature importance CSV at {fi_csv_path}"

    with open(fi_csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 80, f"Expected header + 79 rows, got {len(reader)}"
        assert reader[0] == ["Rank", "Feature Name", "MDI Importance Weight", "Modality"]

    print(f"[OK] Full Phase 8 pipeline execution verified (Accuracy: {metrics['accuracy']*100:.1f}%, F1: {metrics['f1_score']:.4f}).")


if __name__ == "__main__":
    print("=" * 75)
    print("RUNNING PHASE 8 RANDOM FOREST CLASSIFIER AUTOMATED TEST SUITE")
    print("=" * 75)
    test_rf_initialization()
    test_rf_training_and_oob_score()
    test_rf_prediction_and_probability_calibration()
    test_feature_importance_dimensions()
    test_evaluation_metric_bounds()
    test_serialization_and_deserialization()
    test_pipeline_execution_and_artifacts()
    print("=" * 75)
    print("ALL PHASE 8 RANDOM FOREST CLASSIFIER TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)
