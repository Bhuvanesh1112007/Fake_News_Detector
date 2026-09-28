"""
Model Evaluation & Benchmark Reporting Module (Phase 10)
Module Owner: Bhuvanesh M (Lead Data & Ensemble Learning Engineer) & Mohamed Irfaan M (Lead NLP Engineer)

Provides:
1. Comprehensive IEEE-standard benchmark metrics calculation:
   - Accuracy, Precision (Weighted/Macro), Recall (Weighted/Macro), F1-Score (Weighted/Macro)
   - Specificity, Sensitivity, ROC-AUC, Log Loss, Brier Score
   - Confusion Matrix (TP, FP, TN, FN)
   - Out-of-Bag (OOB) Accuracy, 5-Fold Cross-Validation Accuracy & F1
   - Inference Latency (ms/sample) and Throughput (samples/sec)
2. Unified comparative benchmarking across 7 models (6 Baselines + Proposed Multimodal Random Forest)
3. Multimodal Ablation Study across 6 architectural settings:
   - Full Multimodal (79 dims)
   - Text & BERT Only (64 dims)
   - User & Publisher Signals Only (6 dims)
   - Network & Social Engagement Only (6 dims)
   - Stylometric Linguistic Density Only (3 dims)
   - Tabular Multi-Source Only (No BERT) (15 dims)
4. Multi-format artifact persistence:
   - results/metrics/model_comparison.csv
   - results/metrics/comprehensive_benchmarks.csv
   - results/metrics/ablation_study.csv & .json
   - results/metrics/metrics.json
"""
import os
import sys
import csv
import json
import time
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, brier_score_loss, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import (
    METRICS_COMPARISON_PATH, METRICS_JSON_PATH, METRICS_DIR,
    PROCESSED_DATA_DIR, RANDOM_SEED, RANDOM_FOREST_PATH
)
from .train_baselines import run_baseline_pipeline
from .train_random_forest import ProposedRandomForestClassifier, run_random_forest_training_pipeline, load_labels_from_split
from .feature_fusion import MultiSourceFeatureFusion, run_feature_fusion_pipeline


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None,
                      latency_sec: float = 0.0) -> dict:
    """
    Calculates comprehensive classification metrics with numerical safety checks.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    N = len(y_true)

    acc = float(accuracy_score(y_true, y_pred))
    p_wt = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    p_mac = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    r_wt = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    r_mac = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_wt = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # ROC-AUC, Log Loss, Brier Score
    auc = 0.5
    loss = 0.0
    brier = 0.0
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=np.float64)
        prob_pos = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] > 1 else y_prob.flatten()
        try:
            auc = float(roc_auc_score(y_true, prob_pos))
        except Exception:
            auc = 1.0 if acc == 1.0 else 0.5

        try:
            loss = float(log_loss(y_true, y_prob, labels=[0, 1]))
        except Exception:
            loss = 0.0

        try:
            brier = float(brier_score_loss(y_true, prob_pos))
        except Exception:
            brier = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    specificity = float(tn / max(1, tn + fp))
    sensitivity = float(tp / max(1, tp + fn))

    latency_ms = (latency_sec / max(1, N)) * 1000.0 if latency_sec > 0 else 0.0
    throughput = (N / latency_sec) if latency_sec > 0 else 1000.0

    return {
        "accuracy": round(acc, 4),
        "precision_weighted": round(p_wt, 4),
        "precision_macro": round(p_mac, 4),
        "recall_weighted": round(r_wt, 4),
        "recall_macro": round(r_mac, 4),
        "f1_score": round(f1_wt, 4),
        "f1_score_macro": round(f1_mac, 4),
        "specificity": round(specificity, 4),
        "sensitivity": round(sensitivity, 4),
        "roc_auc": round(auc, 4),
        "log_loss": round(loss, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": cm,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "inference_latency_ms": round(latency_ms, 3),
        "throughput_samples_per_sec": round(throughput, 1)
    }


class ModelEvaluator:
    """
    Comprehensive Benchmarking and Evaluation Engine:
    Handles multi-model comparison, modality ablation studies, and metrics persistence.
    """
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state

    def run_ablation_study(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> list[dict]:
        """
        Conducts systematic ablation study across 6 architectural configurations:
        1. Full Multimodal Model (79 dims)
        2. BERT Contextual Embeddings Only (64 dims, indices 0..63)
        3. User & Publisher Signals Only (6 dims, indices 64..69)
        4. Network & Social Diffusion Only (6 dims, indices 70..75)
        5. Stylometric Linguistic Density Only (3 dims, indices 76..78)
        6. Tabular Multi-Source Only without BERT (15 dims, indices 64..78)
        """
        configurations = [
            ("Full Multimodal (BERT + User + Network + Stylo)", slice(0, 79), 79, "Complete Architecture"),
            ("BERT Contextual Embeddings Only", slice(0, 64), 64, "Textual Sequence Only"),
            ("User & Publisher Signals Only", slice(64, 70), 6, "Source Credibility Only"),
            ("Network & Social Engagement Only", slice(70, 76), 6, "Diffusion Dynamics Only"),
            ("Stylometric Linguistic Density Only", slice(76, 79), 3, "Shallow Linguistics Only"),
            ("Tabular Multi-Source Only (No BERT)", slice(64, 79), 15, "Non-Textual Metadata Only")
        ]

        ablation_results = []
        for name, col_slice, n_dims, desc in configurations:
            X_tr_sub = X_train[:, col_slice]
            X_te_sub = X_test[:, col_slice]

            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=self.random_state
            )
            t0 = time.perf_counter()
            clf.fit(X_tr_sub, y_train)
            t_train = time.perf_counter() - t0

            t_infer_start = time.perf_counter()
            y_pred = clf.predict(X_te_sub)
            t_infer = time.perf_counter() - t_infer_start

            y_prob = clf.predict_proba(X_te_sub)
            m = calculate_metrics(y_test, y_pred, y_prob, latency_sec=t_infer)

            # 5-fold CV
            effective_folds = min(5, max(2, int(min(np.sum(y_train == 0), np.sum(y_train == 1)))))
            cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=self.random_state)
            cv_res = cross_validate(clf, X_tr_sub, y_train, cv=cv, scoring={"acc": "accuracy", "f1": "f1_weighted"})
            cv_acc_mean = float(np.mean(cv_res["test_acc"]))
            cv_acc_std = float(np.std(cv_res["test_acc"]))

            ablation_results.append({
                "configuration": name,
                "feature_dimensions": n_dims,
                "description": desc,
                "test_accuracy": m["accuracy"],
                "test_f1": m["f1_score"],
                "test_roc_auc": m["roc_auc"],
                "cv_accuracy_mean": round(cv_acc_mean, 4),
                "cv_accuracy_std": round(cv_acc_std, 4),
                "inference_latency_ms": m["inference_latency_ms"],
                "training_time_sec": round(t_train, 4)
            })

        return ablation_results

    def export_model_comparison_table(self, comparison_rows: list[dict], csv_path: str = METRICS_COMPARISON_PATH):
        """Exports standard model comparison table to CSV."""
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        fieldnames = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in comparison_rows:
                writer.writerow({
                    "Model": r["Model"],
                    "Accuracy": f"{r['Accuracy']:.2f}%" if isinstance(r['Accuracy'], (int, float)) else r['Accuracy'],
                    "Precision": f"{r['Precision']:.4f}" if isinstance(r['Precision'], (int, float)) else r['Precision'],
                    "Recall": f"{r['Recall']:.4f}" if isinstance(r['Recall'], (int, float)) else r['Recall'],
                    "F1": f"{r['F1']:.4f}" if isinstance(r['F1'], (int, float)) else r['F1'],
                    "ROC-AUC": f"{r['ROC-AUC']:.4f}" if isinstance(r['ROC-AUC'], (int, float)) else r['ROC-AUC']
                })
        print(f"[OK] Exported model comparison table to: {csv_path}")

    def export_comprehensive_benchmarks(self, benchmark_data: list[dict], csv_path: str):
        """Exports comprehensive benchmark table including CV, Latency, and Error rates."""
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not benchmark_data:
            return
        fieldnames = list(benchmark_data[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in benchmark_data:
                writer.writerow(row)
        print(f"[OK] Exported comprehensive benchmarks to: {csv_path}")

    def export_ablation_study(self, ablation_data: list[dict], csv_path: str, json_path: str):
        """Exports ablation study findings to CSV and JSON."""
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        # JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(ablation_data, f, indent=2)

        # CSV
        if ablation_data:
            fieldnames = list(ablation_data[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in ablation_data:
                    writer.writerow(row)
        print(f"[OK] Exported ablation study to: {csv_path} & {json_path}")

    def save_metrics_json(self, payload: dict, json_path: str = METRICS_JSON_PATH):
        """Persists consolidated metrics payload to JSON."""
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[OK] Consolidated metrics payload saved to: {json_path}")
        return json_path


def run_evaluation_pipeline() -> dict:
    """
    Executes Phase 10 Model Evaluation & Benchmark Reporting Pipeline:
    1. Loads / executes Baseline models and Proposed Random Forest classifier.
    2. Constructs unified 7-model comparative benchmark table.
    3. Runs 6-scenario Multimodal Modality Ablation Study.
    4. Persists all CSV and JSON benchmark artifacts.
    5. Displays formatted academic benchmark card in the terminal.
    """
    print("=" * 80)
    print("PHASE 10: MODEL EVALUATION & BENCHMARK REPORTING PIPELINE")
    print("=" * 80)

    evaluator = ModelEvaluator()

    # 1. Load Fused Features & Labels
    train_fused_path = os.path.join(PROCESSED_DATA_DIR, "fused_train_features.npy")
    test_fused_path = os.path.join(PROCESSED_DATA_DIR, "fused_test_features.npy")

    if not (os.path.exists(train_fused_path) and os.path.exists(test_fused_path)):
        print("Fused feature matrices missing. Running Phase 7...")
        run_feature_fusion_pipeline()

    X_train = np.load(train_fused_path)
    X_test = np.load(test_fused_path)

    y_train = load_labels_from_split(os.path.join(PROCESSED_DATA_DIR, "train.csv"))
    y_test = load_labels_from_split(os.path.join(PROCESSED_DATA_DIR, "test.csv"))

    # 2. Get Proposed Random Forest Metrics
    print("\n[Step 1/3] Benchmarking Proposed Multimodal Random Forest Classifier...")
    rf_clf, rf_test_metrics = run_random_forest_training_pipeline()

    # 3. Get Classical Baseline Metrics
    print("\n[Step 2/3] Benchmarking Classical Baseline Models...")
    baseline_metrics = run_baseline_pipeline()

    # 4. Construct Unified Comparison Table (7 Models)
    print("\n[Step 3/3] Compiling Comparative Benchmark Tables & Running Ablation Studies...")
    all_comparison_rows = []
    for model_name, bm in baseline_metrics.items():
        all_comparison_rows.append({
            "Model": model_name,
            "Accuracy": bm["accuracy"] * 100.0 if isinstance(bm["accuracy"], float) and bm["accuracy"] <= 1.0 else bm["accuracy"],
            "Precision": bm.get("precision_weighted", bm.get("precision", 0.0)),
            "Recall": bm.get("recall_weighted", bm.get("recall", 0.0)),
            "F1": bm["f1_score"],
            "ROC-AUC": bm["roc_auc"]
        })

    # Add Proposed Random Forest
    all_comparison_rows.append({
        "Model": "Proposed Multimodal RF (Ours)",
        "Accuracy": rf_test_metrics["accuracy"] * 100.0,
        "Precision": rf_test_metrics["precision_weighted"],
        "Recall": rf_test_metrics["recall_weighted"],
        "F1": rf_test_metrics["f1_score"],
        "ROC-AUC": rf_test_metrics["roc_auc"]
    })

    # Save standard comparison CSV
    evaluator.export_model_comparison_table(all_comparison_rows, METRICS_COMPARISON_PATH)

    # Comprehensive Benchmarks Table
    comprehensive_rows = []
    for model_name, bm in baseline_metrics.items():
        comprehensive_rows.append({
            "model_name": model_name,
            "category": "Baseline",
            "accuracy": bm["accuracy"],
            "f1_score": bm["f1_score"],
            "precision": bm.get("precision_weighted", bm.get("precision", 0.0)),
            "recall": bm.get("recall_weighted", bm.get("recall", 0.0)),
            "roc_auc": bm["roc_auc"],
            "cv_accuracy_mean": bm.get("cv_accuracy_mean", 0.0),
            "cv_accuracy_std": bm.get("cv_accuracy_std", 0.0),
            "inference_latency_ms": bm.get("inference_latency_ms", 0.0)
        })
    comprehensive_rows.append({
        "model_name": "Proposed Multimodal RF (Ours)",
        "category": "Proposed Method",
        "accuracy": rf_test_metrics["accuracy"],
        "f1_score": rf_test_metrics["f1_score"],
        "precision": rf_test_metrics["precision_weighted"],
        "recall": rf_test_metrics["recall_weighted"],
        "roc_auc": rf_test_metrics["roc_auc"],
        "cv_accuracy_mean": rf_test_metrics.get("cv_accuracy_mean", 0.0),
        "cv_accuracy_std": rf_test_metrics.get("cv_accuracy_std", 0.0),
        "inference_latency_ms": rf_test_metrics.get("inference_latency_ms", 0.0)
    })

    comp_csv_path = os.path.join(METRICS_DIR, "comprehensive_benchmarks.csv")
    evaluator.export_comprehensive_benchmarks(comprehensive_rows, comp_csv_path)

    # 5. Multimodal Ablation Study
    ablation_rows = evaluator.run_ablation_study(X_train, y_train, X_test, y_test)
    ablation_csv_path = os.path.join(METRICS_DIR, "ablation_study.csv")
    ablation_json_path = os.path.join(METRICS_DIR, "ablation_study.json")
    evaluator.export_ablation_study(ablation_rows, ablation_csv_path, ablation_json_path)

    # 6. Consolidated JSON Payload
    consolidated_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "proposed_model": rf_test_metrics,
        "baselines": baseline_metrics,
        "ablation_study": ablation_rows,
        "model_comparison": all_comparison_rows
    }
    evaluator.save_metrics_json(consolidated_payload, METRICS_JSON_PATH)

    # 7. Print Formatted Terminal Academic Benchmark Card
    print("\n" + "=" * 85)
    print("📊 IEEE COMPARATIVE MODEL BENCHMARK TABLE")
    print("=" * 85)
    print(f"{'Model Architecture':<32} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8} | {'ROC-AUC':<8}")
    print("-" * 85)
    for r in all_comparison_rows:
        is_proposed = "Proposed" in r["Model"]
        prefix = "⭐ " if is_proposed else "   "
        m_name = (prefix + r["Model"])[:32]
        acc_str = f"{r['Accuracy']:.1f}%" if isinstance(r['Accuracy'], (int, float)) else str(r['Accuracy'])
        print(f"{m_name:<32} | {acc_str:<10} | {r['Precision']:<10} | {r['Recall']:<8} | {r['F1']:<8} | {r['ROC-AUC']:<8}")
    print("=" * 85)

    print("\n" + "=" * 85)
    print("🔬 MULTIMODAL ARCHITECTURAL ABLATION STUDY")
    print("=" * 85)
    print(f"{'Modality Configuration':<46} | {'Dims':<5} | {'Test Acc':<9} | {'Test F1':<8} | {'5-Fold CV Acc':<12}")
    print("-" * 85)
    for a in ablation_rows:
        print(f"{a['configuration']:<46} | {a['feature_dimensions']:<5} | {a['test_accuracy']*100:>6.1f}%   | {a['test_f1']:>6.4f}   | {a['cv_accuracy_mean']*100:.1f}% ± {a['cv_accuracy_std']*100:.1f}%")
    print("=" * 85 + "\n")

    return consolidated_payload


if __name__ == "__main__":
    run_evaluation_pipeline()
