"""
Phase 3 Baseline Models Automated Test Suite
Module Owner: Bhuvanesh M (Lead Data & Ensemble Learning Engineer) & Harish Hariharan MU (Evaluation Lead)

Verifies:
1. Baseline classifier instantiation & hyperparameter configuration
2. TF-IDF + metadata feature pipeline transformations
3. Classifier training, convergence, and predictions
4. Mathematical bounds of evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
5. Stratified 5-Fold Cross-Validation execution
6. Benchmark artifact generation (model_comparison.csv, metrics.json, baselines.joblib)
"""
import os
import sys
import csv
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    import joblib
except ImportError:
    import pickle as joblib

from src.train_baselines import (
    BaselineFeaturePipeline, get_baseline_classifiers,
    evaluate_classifier_model, train_and_evaluate_baselines,
    run_baseline_pipeline
)
from src.config import (
    METRICS_COMPARISON_PATH, METRICS_JSON_PATH,
    BASELINES_MODEL_PATH, RANDOM_SEED
)


def test_baseline_initialization():
    """Verifies that all 6 research baseline models initialize properly."""
    baselines = get_baseline_classifiers(random_state=RANDOM_SEED)
    expected_models = [
        "Majority Class (Dummy)",
        "Naïve Bayes",
        "Decision Tree",
        "Logistic Regression",
        "Support Vector Machine (RBF)",
        "Gradient Boosting (GBM)"
    ]
    for model_name in expected_models:
        assert model_name in baselines, f"Missing expected baseline model: {model_name}"
        clf = baselines[model_name]
        assert clf is not None, f"Model {model_name} initialized as None"
    print("[OK] All 6 baseline classifier architectures initialized successfully.")


def test_tfidf_feature_pipeline():
    """Tests TF-IDF n-gram vectorization and metadata concatenation."""
    pipeline = BaselineFeaturePipeline(max_features=100)
    titles = [
        "NASA Discovers Water on Mars",
        "SHOCKING PROOF: Miracle cure banned by doctors!!!",
        "WHO Releases Global Health Guidelines",
        "Alien spaceship spotted over London secret video leaked"
    ]
    texts = [
        "NASA scientists have confirmed the detection of liquid water beneath the Martian surface.",
        "BREAKING! Big Pharma does not want you to know about this miracle herb! Click here!",
        "The World Health Organization published comprehensive clinical guidelines today.",
        "Whistleblower releases terrifying classified footage of UFO invasion."
    ]
    meta = [
        {"shares": 5000, "sentiment": 0.2},
        {"shares": 95000, "sentiment": -0.5},
        {"shares": 12000, "sentiment": 0.1},
        {"shares": 80000, "sentiment": -0.8}
    ]

    X = pipeline.fit_transform(titles, texts, meta)
    assert isinstance(X, np.ndarray), "Output is not a numpy array"
    assert X.shape[0] == 4, f"Expected 4 rows, got {X.shape[0]}"
    assert X.shape[1] > 0, "Feature dimension is 0"
    
    # Test transform without metadata
    X_no_meta = pipeline.transform(titles, texts)
    assert X_no_meta.shape[0] == 4
    print(f"[OK] Baseline feature extraction pipeline verified (Feature Dim: {X.shape[1]}).")


def test_baseline_training_and_convergence():
    """Tests training, convergence, and evaluation on synthetic feature tensors."""
    np.random.seed(RANDOM_SEED)
    N_train, N_test, D = 30, 10, 20
    X_train = np.random.randn(N_train, D)
    y_train = np.random.choice([0, 1], size=N_train)
    X_test = np.random.randn(N_test, D)
    y_test = np.random.choice([0, 1], size=N_test)

    results = train_and_evaluate_baselines(X_train, y_train, X_test, y_test, run_cv=True)
    assert len(results) == 6, f"Expected 6 model evaluation results, got {len(results)}"
    
    for name, m in results.items():
        assert "accuracy" in m
        assert "f1_score" in m
        assert "roc_auc" in m
        assert "confusion_matrix" in m
    print("[OK] Baseline model training and convergence verified across all 6 architectures.")


def test_metric_bounds():
    """Verifies that all computed performance metrics satisfy theoretical mathematical bounds."""
    np.random.seed(RANDOM_SEED)
    X_train = np.random.randn(20, 10)
    y_train = np.array([0]*10 + [1]*10)
    X_test = np.random.randn(8, 10)
    y_test = np.array([0]*4 + [1]*4)

    results = train_and_evaluate_baselines(X_train, y_train, X_test, y_test, run_cv=False)
    for model_name, m in results.items():
        assert 0.0 <= m["accuracy"] <= 1.0, f"{model_name}: Accuracy out of bounds: {m['accuracy']}"
        assert 0.0 <= m["precision_weighted"] <= 1.0, f"{model_name}: Precision out of bounds: {m['precision_weighted']}"
        assert 0.0 <= m["recall_weighted"] <= 1.0, f"{model_name}: Recall out of bounds: {m['recall_weighted']}"
        assert 0.0 <= m["f1_score"] <= 1.0, f"{model_name}: F1 out of bounds: {m['f1_score']}"
        assert 0.0 <= m["roc_auc"] <= 1.0, f"{model_name}: ROC-AUC out of bounds: {m['roc_auc']}"
        
        cm = m["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2
        total_cm_samples = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
        assert total_cm_samples == len(y_test), f"{model_name}: Confusion matrix total mismatch"
    print("[OK] Mathematical metric bounds and confusion matrix integrity verified.")


def test_cross_validation_integrity():
    """Verifies that Stratified 5-Fold Cross-Validation computes valid mean and std scores."""
    np.random.seed(RANDOM_SEED)
    X_train = np.random.randn(30, 15)
    y_train = np.array([0]*15 + [1]*15)
    X_test = np.random.randn(10, 15)
    y_test = np.array([0]*5 + [1]*5)

    baselines = get_baseline_classifiers(random_state=RANDOM_SEED)
    lr = baselines["Logistic Regression"]
    res = evaluate_classifier_model(lr, X_train, y_train, X_test, y_test, run_cv=True, cv_folds=5)
    
    assert 0.0 <= res["cv_accuracy_mean"] <= 1.0, f"Invalid CV Accuracy: {res['cv_accuracy_mean']}"
    assert res["cv_accuracy_std"] >= 0.0, f"Invalid CV Std: {res['cv_accuracy_std']}"
    assert 0.0 <= res["cv_f1_mean"] <= 1.0, f"Invalid CV F1: {res['cv_f1_mean']}"
    print(f"[OK] Stratified Cross-Validation verified (CV Accuracy: {res['cv_accuracy_mean']*100:.1f}±{res['cv_accuracy_std']*100:.1f}%).")


def test_artifact_generation():
    """Verifies that running the pipeline generates and persists all expected benchmark artifacts."""
    results = run_baseline_pipeline()
    assert len(results) >= 6, "Expected at least 6 baseline results"
    
    # 1. Verify CSV Comparison File
    assert os.path.exists(METRICS_COMPARISON_PATH), f"Missing comparison CSV at {METRICS_COMPARISON_PATH}"
    with open(METRICS_COMPARISON_PATH, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) >= 7, f"Expected at least header + 6 rows, got {len(reader)}"
        assert reader[0][0] == "Model", "CSV header missing 'Model'"

    # 2. Verify JSON Metrics File
    assert os.path.exists(METRICS_JSON_PATH), f"Missing metrics JSON at {METRICS_JSON_PATH}"
    with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "baseline_metrics" in data, "Missing 'baseline_metrics' key in JSON"
        assert len(data["baseline_metrics"]) >= 6

    # 3. Verify Serialized Baseline Models Bundle
    assert os.path.exists(BASELINES_MODEL_PATH), f"Missing baselines model bundle at {BASELINES_MODEL_PATH}"
    bundle = joblib.load(BASELINES_MODEL_PATH)
    assert "models" in bundle and "feature_pipeline" in bundle and "results" in bundle
    assert len(bundle["models"]) >= 6
    print("[OK] Benchmark artifacts (CSV, JSON, and serialized model bundle) successfully validated.")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING PHASE 3 BASELINE MODELS AUTOMATED TEST SUITE")
    print("=" * 70)
    test_baseline_initialization()
    test_tfidf_feature_pipeline()
    test_baseline_training_and_convergence()
    test_metric_bounds()
    test_cross_validation_integrity()
    test_artifact_generation()
    print("=" * 70)
    print("ALL PHASE 3 BASELINE MODELS TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
