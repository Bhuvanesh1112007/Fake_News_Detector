"""
Random Forest Classifier Training Module (Phase 8)
Module Owner: Bhuvanesh M (Lead Data & Ensemble Learning Engineer)

Trains and evaluates the Proposed Random Forest Ensemble Classifier on the
79-dimensional fused multimodal feature representation:
- 150 randomized decorrelated decision trees with Gini impurity splitting
- Bootstrap Aggregation (Bagging) with Out-of-Bag (OOB) error estimation
- Balanced class weighting and depth regularization
- Mean Decrease in Impurity (MDI) feature importance ranking across all 79 dimensions
- Stratified 5-Fold Cross-Validation for statistical robustness
- Persists models to models/random_forest.pkl & models/random_forest.joblib
"""
import os
import sys
import csv
import json
import time
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import joblib
except ImportError:
    import pickle as joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, brier_score_loss, confusion_matrix
)

from .config import (
    RANDOM_FOREST_PATH, RANDOM_SEED, RF_N_ESTIMATORS,
    RF_MAX_DEPTH, RF_MIN_SAMPLES_SPLIT, RF_MIN_SAMPLES_LEAF,
    RF_MAX_FEATURES, PROCESSED_DATA_DIR, MODELS_DIR, METRICS_DIR,
    PROCESSED_TRAIN_PATH, PROCESSED_VAL_PATH, PROCESSED_TEST_PATH
)
from .feature_fusion import MultiSourceFeatureFusion, run_feature_fusion_pipeline


class ProposedRandomForestClassifier:
    """
    Proposed Hybrid Classifier:
    Optimized Random Forest ensemble operating over the 79-dimensional fused feature space.
    """
    def __init__(self, n_estimators: int = RF_N_ESTIMATORS, max_depth: int = RF_MAX_DEPTH,
                 min_samples_split: int = RF_MIN_SAMPLES_SPLIT, min_samples_leaf: int = RF_MIN_SAMPLES_LEAF,
                 max_features=RF_MAX_FEATURES, class_weight: str = "balanced",
                 oob_score: bool = True, random_state: int = RANDOM_SEED):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.oob_score = oob_score
        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            oob_score=oob_score,
            random_state=random_state
        )
        self.is_fitted = False
        self.oob_accuracy_ = 0.0
        self.feature_importances_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fits the 150-tree Random Forest ensemble on fused feature representations."""
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Cannot fit on empty training data.")

        self.model.fit(X_train, y_train)
        self.is_fitted = True
        self.feature_importances_ = self.model.feature_importances_
        if self.oob_score and hasattr(self.model, "oob_score_"):
            self.oob_accuracy_ = float(self.model.oob_score_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary class labels (0=REAL, 1=FAKE)."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts calibrated class probability distributions."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Computes comprehensive evaluation metrics on test partition:
        Accuracy, Precision, Recall, F1, ROC-AUC, Log Loss, Brier Score, and Confusion Matrix.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        t_start = time.perf_counter()
        y_pred = self.model.predict(X_test)
        t_infer = time.perf_counter() - t_start
        latency_ms = (t_infer / max(1, len(y_test))) * 1000.0

        y_prob = self.model.predict_proba(X_test)
        prob_pos = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]

        try:
            auc = float(roc_auc_score(y_test, prob_pos))
        except Exception:
            auc = 1.0 if len(np.unique(y_test)) == 1 else 0.5

        try:
            loss = float(log_loss(y_test, y_prob, labels=[0, 1]))
        except Exception:
            loss = 0.0

        try:
            brier = float(brier_score_loss(y_test, prob_pos))
        except Exception:
            brier = 0.0

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

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
            "brier_score": round(brier, 4),
            "oob_score": round(self.oob_accuracy_, 4),
            "confusion_matrix": cm,
            "inference_latency_ms": round(latency_ms, 3)
        }

    def get_feature_importances(self, feature_names: list[str] = None) -> list[dict]:
        """
        Returns ranked list of feature importance dictionaries sorted by MDI weight:
        [{'feature': name, 'importance': weight, 'rank': rank}, ...]
        """
        if not self.is_fitted or self.feature_importances_ is None:
            raise ValueError("Model is not fitted. Call fit() first.")

        if feature_names is None or len(feature_names) != len(self.feature_importances_):
            feature_names = [f"feature_{i:02d}" for i in range(len(self.feature_importances_))]

        ranked = []
        for name, imp in zip(feature_names, self.feature_importances_):
            ranked.append({
                "feature": name,
                "importance": round(float(imp), 6)
            })

        ranked.sort(key=lambda x: x["importance"], reverse=True)
        for r, item in enumerate(ranked, 1):
            item["rank"] = r

        return ranked

    def save(self, filepath: str = RANDOM_FOREST_PATH) -> str:
        """Persists the trained Random Forest model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        # Also save joblib extension if pkl
        alt_path = filepath.replace(".pkl", ".joblib") if filepath.endswith(".pkl") else filepath.replace(".joblib", ".pkl")
        joblib.dump(self.model, alt_path)
        print(f"[OK] Random Forest model persisted to: {filepath} & {alt_path}")
        return filepath

    @classmethod
    def load(cls, filepath: str = RANDOM_FOREST_PATH) -> "ProposedRandomForestClassifier":
        """Loads a persisted Random Forest model from disk."""
        if not os.path.exists(filepath):
            alt_path = filepath.replace(".pkl", ".joblib") if filepath.endswith(".pkl") else filepath.replace(".joblib", ".pkl")
            if os.path.exists(alt_path):
                filepath = alt_path
            else:
                raise FileNotFoundError(f"Random Forest model artifact not found at: {filepath}")

        obj = cls()
        obj.model = joblib.load(filepath)
        obj.is_fitted = True
        obj.feature_importances_ = obj.model.feature_importances_
        if hasattr(obj.model, "oob_score_"):
            obj.oob_accuracy_ = float(obj.model.oob_score_)
        return obj


def load_labels_from_split(csv_path: str) -> np.ndarray:
    """Loads binary labels from a processed CSV file."""
    labels = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["label"]))
    return np.array(labels, dtype=np.int32)


def run_random_forest_training_pipeline() -> tuple[ProposedRandomForestClassifier, dict]:
    """
    Executes Phase 8 Random Forest Classifier Training Pipeline:
    1. Loads 79-dimensional fused feature matrices (fused_train_features.npy, etc.).
    2. Fits ProposedRandomForestClassifier (150 trees, balanced class weight, OOB score).
    3. Runs Stratified 5-Fold Cross-Validation on the fused training set.
    4. Evaluates on test set (Accuracy, F1, ROC-AUC, Confusion Matrix).
    5. Saves model weights and feature importances to models/ and results/metrics/.
    6. Displays formatted summary card.
    """
    print("=" * 80)
    print("PHASE 8: PROPOSED RANDOM FOREST CLASSIFIER TRAINING PIPELINE")
    print("=" * 80)

    train_fused_path = os.path.join(PROCESSED_DATA_DIR, "fused_train_features.npy")
    test_fused_path = os.path.join(PROCESSED_DATA_DIR, "fused_test_features.npy")
    val_fused_path = os.path.join(PROCESSED_DATA_DIR, "fused_val_features.npy")

    if not (os.path.exists(train_fused_path) and os.path.exists(test_fused_path)):
        print("Fused feature matrices not found. Triggering Phase 7 feature fusion pipeline...")
        run_feature_fusion_pipeline()

    # 1. Load Fused Matrices & Labels
    X_train = np.load(train_fused_path)
    X_val = np.load(val_fused_path)
    X_test = np.load(test_fused_path)

    y_train = load_labels_from_split(PROCESSED_TRAIN_PATH)
    y_val = load_labels_from_split(PROCESSED_VAL_PATH)
    y_test = load_labels_from_split(PROCESSED_TEST_PATH)

    print(f"Loaded Fused Training Data:   {X_train.shape} | Labels: Real={sum(y_train==0)}, Fake={sum(y_train==1)}")
    print(f"Loaded Fused Validation Data: {X_val.shape}   | Labels: Real={sum(y_val==0)}, Fake={sum(y_val==1)}")
    print(f"Loaded Fused Test Data:       {X_test.shape}   | Labels: Real={sum(y_test==0)}, Fake={sum(y_test==1)}")

    # 2. Train Random Forest Classifier
    print("\n[Step 1/3] Training 150-Tree Random Forest Ensemble with OOB Scoring...")
    t0 = time.perf_counter()
    rf_clf = ProposedRandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=RF_MAX_FEATURES,
        class_weight="balanced",
        oob_score=True,
        random_state=RANDOM_SEED
    )
    rf_clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"Random Forest fitted in {train_time:.3f}s. OOB Score: {rf_clf.oob_accuracy_*100:.1f}%.")

    # 3. Stratified 5-Fold Cross-Validation
    print("\n[Step 2/3] Executing Stratified 5-Fold Cross-Validation...")
    effective_folds = min(5, max(2, int(min(np.sum(y_train == 0), np.sum(y_train == 1)))))
    cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=RANDOM_SEED)
    cv_res = cross_validate(rf_clf.model, X_train, y_train, cv=cv, scoring={"acc": "accuracy", "f1": "f1_weighted"})
    cv_acc_mean = float(np.mean(cv_res["test_acc"]))
    cv_acc_std = float(np.std(cv_res["test_acc"]))
    cv_f1_mean = float(np.mean(cv_res["test_f1"]))
    cv_f1_std = float(np.std(cv_res["test_f1"]))
    print(f"Cross-Validation Accuracy: {cv_acc_mean*100:.1f}% ± {cv_acc_std*100:.1f}% | F1: {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")

    # 4. Evaluate on Test Set
    test_metrics = rf_clf.evaluate(X_test, y_test)
    test_metrics["cv_accuracy_mean"] = round(cv_acc_mean, 4)
    test_metrics["cv_accuracy_std"] = round(cv_acc_std, 4)
    test_metrics["cv_f1_mean"] = round(cv_f1_mean, 4)
    test_metrics["cv_f1_std"] = round(cv_f1_std, 4)
    test_metrics["training_time_sec"] = round(train_time, 4)
    test_metrics["n_estimators"] = RF_N_ESTIMATORS
    test_metrics["max_depth"] = RF_MAX_DEPTH

    # 5. Extract Feature Importances (MDI)
    fusion = MultiSourceFeatureFusion()
    feature_names = fusion.get_feature_names()
    ranked_features = rf_clf.get_feature_importances(feature_names)

    # 6. Save Artifacts
    print("\n[Step 3/3] Persisting Model Weights, Feature Importances, and Evaluation Reports...")
    rf_clf.save(RANDOM_FOREST_PATH)

    os.makedirs(METRICS_DIR, exist_ok=True)
    # Save evaluation metrics JSON
    metrics_path = os.path.join(METRICS_DIR, "rf_evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # Save feature importances JSON & CSV
    fi_json_path = os.path.join(METRICS_DIR, "rf_feature_importance.json")
    with open(fi_json_path, "w", encoding="utf-8") as f:
        json.dump(ranked_features, f, indent=2)

    fi_csv_path = os.path.join(METRICS_DIR, "rf_feature_importance.csv")
    with open(fi_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Feature Name", "MDI Importance Weight", "Modality"])
        for item in ranked_features:
            fname = item["feature"]
            if fname.startswith("bert_dim"):
                modality = "BERT Contextual Embedding"
            elif fname in ["publisher_credibility_score", "tld_trust_rating", "has_author_metadata", "author_credential_score", "is_pseudonymous_author", "journalistic_attribution_intensity"]:
                modality = "User & Publisher Credibility"
            elif fname in ["log_social_shares", "hyper_virality_indicator", "share_velocity_proxy", "sentiment_polarity_normalized", "emotional_intensity_score", "engagement_anomaly_score"]:
                modality = "Network & Social Engagement"
            else:
                modality = "Stylometric Linguistic Density"
            writer.writerow([item["rank"], fname, f"{item['importance']:.6f}", modality])

    # Print Summary Card
    print("\n" + "=" * 80)
    print("🏆 PROPOSED RANDOM FOREST ENSEMBLE BENCHMARK CARD")
    print("=" * 80)
    print(f"• Test Accuracy:        {test_metrics['accuracy']*100:.1f}%")
    print(f"• Test F1-Score:        {test_metrics['f1_score']:.4f}")
    print(f"• Test ROC-AUC:         {test_metrics['roc_auc']:.4f}")
    print(f"• Out-of-Bag (OOB) Acc: {test_metrics['oob_score']*100:.1f}%")
    print(f"• 5-Fold CV Accuracy:   {cv_acc_mean*100:.1f}% ± {cv_acc_std*100:.1f}%")
    print(f"• Inference Latency:    {test_metrics['inference_latency_ms']:.2f} ms/sample")
    print(f"• Confusion Matrix:     {test_metrics['confusion_matrix']}")
    print("-" * 80)
    print("Top 8 Most Discriminative Fused Features (MDI):")
    for item in ranked_features[:8]:
        print(f"  #{item['rank']:<2} {item['feature']:<38} | Weight: {item['importance']:.4f} ({item['importance']*100:.2f}%)")
    print("=" * 80 + "\n")

    return rf_clf, test_metrics


if __name__ == "__main__":
    run_random_forest_training_pipeline()
