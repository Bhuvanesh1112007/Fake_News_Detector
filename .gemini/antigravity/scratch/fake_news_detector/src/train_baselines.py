"""
Baseline Models Training & Comparison Module (Phase 3)
Module Owner: Bhuvanesh M (Lead Data & Ensemble Learning Engineer) & Mohamed Irfaan M (Lead NLP Engineer)

Trains, cross-validates, and rigorously evaluates classical machine learning baselines:
1. Majority Class Dummy Baseline (Empirical chance rate)
2. Naïve Bayes (Multinomial / Gaussian probabilistic text classifier)
3. Decision Tree (CART non-linear decision tree)
4. Logistic Regression (L2-regularized linear model)
5. Support Vector Machine (RBF Kernel with Platt probability scaling)
6. Gradient Boosting Machine (Sequential boosted decision tree ensemble)

Evaluates standard academic metrics:
- Accuracy, Precision (Weighted & Macro), Recall (Weighted & Macro), F1-Score (Weighted & Macro),
- ROC-AUC, Log Loss, Confusion Matrix (TP, FP, TN, FN), and Inference Latency (ms/sample).
- Stratified 5-Fold Cross-Validation for generalization robustness.
- Persists comparative benchmarks to results/metrics/ and models/baselines.joblib.
"""
import os
import sys
import csv
import json
import time
import warnings
import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore", category=FutureWarning)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import joblib
except ImportError:
    import pickle as joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix
)

from .config import (
    RANDOM_SEED, PROCESSED_TRAIN_PATH, PROCESSED_VAL_PATH,
    PROCESSED_TEST_PATH, METRICS_COMPARISON_PATH, METRICS_JSON_PATH,
    BASELINES_MODEL_PATH, METRICS_DIR, MODELS_DIR
)
from .preprocessing import TextPreprocessor, run_preprocessing_pipeline


class BaselineFeaturePipeline:
    """
    Standard Baseline Feature Extractor:
    Combines sublinear TF-IDF word n-grams (1-2) with scaled stylometric/metadata features.
    """
    def __init__(self, max_features: int = 2500):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=max_features,
            strip_accents="unicode",
            lowercase=True
        )
        self.text_cleaner = TextPreprocessor()
        self.is_fitted = False

    def fit(self, texts: list[str]):
        """Fits the TF-IDF vectorizer on training text corpus."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, titles: list[str], texts: list[str], metadata: list[dict] = None) -> np.ndarray:
        """Transforms text and metadata into feature matrices."""
        if not self.is_fitted:
            raise ValueError("BaselineFeaturePipeline is not fitted. Call fit() first.")

        combined_texts = [f"{t} {b}".strip() for t, b in zip(titles, texts)]
        tfidf_sparse = self.vectorizer.transform(combined_texts)

        if metadata is not None and len(metadata) == len(titles):
            meta_features = []
            for i, meta in enumerate(metadata):
                raw_text = texts[i] if i < len(texts) else ""
                stats = self.text_cleaner.extract_linguistic_stats(raw_text)
                shares = float(meta.get("shares", 1000.0))
                sentiment = float(meta.get("sentiment", 0.0))
                
                # Scaled metadata row
                meta_row = [
                    stats["uppercase_ratio"],
                    min(1.0, stats["exclamation_count"] / 10.0),
                    min(1.0, stats["sensational_word_count"] / 5.0),
                    stats["lexical_diversity"],
                    np.log1p(max(0.0, shares)) / 15.0,
                    sentiment
                ]
                meta_features.append(meta_row)
            
            meta_sparse = sp.csr_matrix(np.array(meta_features, dtype=np.float32))
            fused_matrix = sp.hstack([tfidf_sparse, meta_sparse], format="csr")
            return fused_matrix.toarray()
        
        return tfidf_sparse.toarray()

    def fit_transform(self, titles: list[str], texts: list[str], metadata: list[dict] = None) -> np.ndarray:
        combined_texts = [f"{t} {b}".strip() for t, b in zip(titles, texts)]
        self.fit(combined_texts)
        return self.transform(titles, texts, metadata)


def get_baseline_classifiers(random_state: int = RANDOM_SEED) -> dict:
    """Returns the standardized suite of research baseline classifiers."""
    return {
        "Majority Class (Dummy)": DummyClassifier(strategy="most_frequent"),
        "Naïve Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=random_state),
        "Support Vector Machine (RBF)": SVC(probability=True, kernel="rbf", C=1.0, random_state=random_state),
        "Gradient Boosting (GBM)": GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=random_state)
    }


def evaluate_classifier_model(clf, X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              run_cv: bool = True, cv_folds: int = 5) -> dict:
    """
    Trains an individual classifier, measures latency, calculates test performance,
    and runs stratified cross-validation.
    """
    # 1. Measure Training Time
    t_start = time.perf_counter()
    clf.fit(X_train, y_train)
    t_train = time.perf_counter() - t_start

    # 2. Measure Inference Time
    t_infer_start = time.perf_counter()
    y_pred = clf.predict(X_test)
    t_infer = time.perf_counter() - t_infer_start
    latency_ms = (t_infer / max(1, len(y_test))) * 1000.0

    # 3. Probability estimation & ROC-AUC / Log-Loss
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)
        prob_pos = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]
        try:
            auc = float(roc_auc_score(y_test, prob_pos))
        except Exception:
            auc = 0.5
        try:
            loss = float(log_loss(y_test, y_prob, labels=[0, 1]))
        except Exception:
            loss = 0.693
    else:
        prob_pos = y_pred
        auc = float(roc_auc_score(y_test, prob_pos)) if len(np.unique(y_test)) > 1 else 0.5
        loss = 0.693

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

    # 4. Cross-Validation on Training Data
    cv_acc_mean, cv_acc_std = 0.0, 0.0
    cv_f1_mean, cv_f1_std = 0.0, 0.0

    if run_cv and len(y_train) >= cv_folds:
        # Determine appropriate number of folds
        min_class_count = min(np.sum(y_train == 0), np.sum(y_train == 1))
        effective_folds = min(cv_folds, max(2, int(min_class_count)))
        
        cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=RANDOM_SEED)
        scoring = {"acc": "accuracy", "f1": "f1_weighted"}
        try:
            cv_res = cross_validate(clf, X_train, y_train, cv=cv, scoring=scoring)
            cv_acc_mean = float(np.mean(cv_res["test_acc"]))
            cv_acc_std = float(np.std(cv_res["test_acc"]))
            cv_f1_mean = float(np.mean(cv_res["test_f1"]))
            cv_f1_std = float(np.std(cv_res["test_f1"]))
        except Exception:
            pass

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_weighted": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1_score_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "roc_auc": round(auc, 4),
        "log_loss": round(loss, 4),
        "confusion_matrix": cm,
        "training_time_sec": round(t_train, 4),
        "inference_latency_ms": round(latency_ms, 3),
        "cv_accuracy_mean": round(cv_acc_mean, 4),
        "cv_accuracy_std": round(cv_acc_std, 4),
        "cv_f1_mean": round(cv_f1_mean, 4),
        "cv_f1_std": round(cv_f1_std, 4)
    }


def train_and_evaluate_baselines(X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray,
                                 run_cv: bool = True) -> dict:
    """
    Trains and evaluates the complete suite of baseline classifiers on pre-extracted feature arrays.
    Returns structured results dictionary.
    """
    baselines = get_baseline_classifiers(random_state=RANDOM_SEED)
    results = {}

    for name, clf in baselines.items():
        res = evaluate_classifier_model(clf, X_train, y_train, X_test, y_test, run_cv=run_cv)
        res["architecture_type"] = "Baseline Classifier"
        results[name] = res

    return results


def save_baseline_artifacts(results: dict, models: dict = None, feature_pipeline=None):
    """Persists model comparison table, JSON metrics, and serialized model objects."""
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Save CSV Comparison Table
    headers = [
        "Model", "Architecture Type", "Accuracy", "Precision (Weighted)",
        "Recall (Weighted)", "F1-Score (Weighted)", "F1-Score (Macro)",
        "ROC-AUC", "Log Loss", "CV Accuracy (Mean±Std)", "CV F1 (Mean±Std)",
        "Latency (ms/sample)"
    ]

    with open(METRICS_COMPARISON_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for model_name, m in results.items():
            cv_acc_str = f"{m['cv_accuracy_mean']*100:.1f}±{m['cv_accuracy_std']*100:.1f}%" if m['cv_accuracy_mean'] > 0 else "N/A"
            cv_f1_str = f"{m['cv_f1_mean']:.3f}±{m['cv_f1_std']:.3f}" if m['cv_f1_mean'] > 0 else "N/A"
            writer.writerow([
                model_name,
                m.get("architecture_type", "Baseline"),
                f"{m['accuracy']*100:.1f}%",
                f"{m['precision_weighted']:.4f}",
                f"{m['recall_weighted']:.4f}",
                f"{m['f1_score']:.4f}",
                f"{m['f1_score_macro']:.4f}",
                f"{m['roc_auc']:.4f}",
                f"{m['log_loss']:.4f}",
                cv_acc_str,
                cv_f1_str,
                f"{m['inference_latency_ms']:.2f}"
            ])

    # 2. Save JSON Metrics
    metrics_payload = {
        "phase": "Phase 3: Baseline Models Training & Benchmark Comparison",
        "random_seed": RANDOM_SEED,
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_metrics": results
    }

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # 3. Save Serialized Baseline Models Bundle
    if models is not None:
        bundle = {
            "models": models,
            "feature_pipeline": feature_pipeline,
            "results": results
        }
        joblib.dump(bundle, BASELINES_MODEL_PATH)

    print(f"[OK] Baseline comparison table saved to: {METRICS_COMPARISON_PATH}")
    print(f"[OK] Baseline metrics JSON saved to: {METRICS_JSON_PATH}")
    if models is not None:
        print(f"[OK] Trained baseline model bundle saved to: {BASELINES_MODEL_PATH}")


def load_split_csv(file_path: str) -> tuple[list[str], list[str], np.ndarray, list[dict]]:
    """Loads titles, texts, labels, and metadata from processed CSV file."""
    titles, texts, labels, metadata = [], [], [], []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            titles.append(row["title"])
            texts.append(row["text"])
            labels.append(int(row["label"]))
            metadata.append({
                "source": row.get("source", "Unknown"),
                "author": row.get("author", "Unknown"),
                "shares": float(row.get("shares", 1000.0)),
                "sentiment": float(row.get("sentiment", 0.0))
            })
    return titles, texts, np.array(labels, dtype=np.int32), metadata


def run_baseline_pipeline() -> dict:
    """
    Complete execution pipeline for Phase 3:
    1. Loads or generates preprocessed splits.
    2. Builds TF-IDF + Metadata baseline feature representations.
    3. Fits all baseline classifiers and computes comprehensive metrics.
    4. Persists comparative CSV table, metrics JSON, and model artifacts.
    5. Displays structured console report.
    """
    print("=" * 80)
    print("PHASE 3: BASELINE MODELS TRAINING & BENCHMARK COMPARISON PIPELINE")
    print("=" * 80)

    # Ensure preprocessed splits exist
    if not (os.path.exists(PROCESSED_TRAIN_PATH) and os.path.exists(PROCESSED_TEST_PATH)):
        print("Processed splits not found. Triggering Phase 2 data preprocessing pipeline...")
        run_preprocessing_pipeline()

    # Load splits
    titles_train, texts_train, y_train, meta_train = load_split_csv(PROCESSED_TRAIN_PATH)
    titles_test, texts_test, y_test, meta_test = load_split_csv(PROCESSED_TEST_PATH)

    print(f"Loaded Training Set: {len(titles_train)} records (Real: {sum(y_train == 0)}, Fake: {sum(y_train == 1)})")
    print(f"Loaded Test Set:     {len(titles_test)} records (Real: {sum(y_test == 0)}, Fake: {sum(y_test == 1)})")

    # Fit feature extractor
    print("\n[Step 1/3] Extracting TF-IDF n-grams (1-2) and Stylometric Features...")
    feature_pipeline = BaselineFeaturePipeline(max_features=2500)
    X_train = feature_pipeline.fit_transform(titles_train, texts_train, meta_train)
    X_test = feature_pipeline.transform(titles_test, texts_test, meta_test)
    print(f"Constructed baseline feature matrix: {X_train.shape[1]} dimensions.")

    # Train and evaluate all baseline classifiers
    print("\n[Step 2/3] Training Baseline Classifiers & Running 5-Fold Stratified Cross-Validation...")
    baselines = get_baseline_classifiers(random_state=RANDOM_SEED)
    results = {}

    for name, clf in baselines.items():
        print(f"  --> Training & Evaluating: {name}...")
        res = evaluate_classifier_model(clf, X_train, y_train, X_test, y_test, run_cv=True, cv_folds=5)
        res["architecture_type"] = "Baseline Machine Learning"
        results[name] = res

    # Save artifacts
    print("\n[Step 3/3] Saving Benchmark Artifacts & Metrics Reports...")
    save_baseline_artifacts(results, models=baselines, feature_pipeline=feature_pipeline)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Baseline Model Architecture':<30} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'CV Accuracy':<15}")
    print("-" * 90)
    for model_name, m in results.items():
        cv_acc = f"{m['cv_accuracy_mean']*100:.1f}±{m['cv_accuracy_std']*100:.1f}%" if m['cv_accuracy_mean'] > 0 else "N/A"
        print(f"{model_name:<30} | {m['accuracy']*100:>8.1f}% | {m['f1_score']:>10.4f} | {m['roc_auc']:>10.4f} | {cv_acc:>15}")
    print("=" * 90 + "\n")

    return results


if __name__ == "__main__":
    run_baseline_pipeline()
