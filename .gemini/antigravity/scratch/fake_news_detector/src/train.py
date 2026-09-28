"""
End-to-End Training & Evaluation Pipeline
Trains BERT Transformer Contextual Embeddings, Deep Multimodal Neural Network,
and Hybrid Ensemble Classifier; runs comparative benchmarks and persists artifacts.
"""
import os
import sys
import csv
import json

# Ensure utf-8 output in Windows shell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import joblib
except ImportError:
    import pickle as joblib
import numpy as np
from sklearn.model_selection import train_test_split
from .preprocessor import TextPreprocessor
from .features import HybridFeatureExtractor
from .model import HybridFakeNewsClassifier
from .explainer import XAIExplainer
from dataset.generate_dataset import generate_csv, DATASET_PATH

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_PATH = os.path.join(MODELS_DIR, "evaluation_metrics.json")
MODEL_ARTIFACT_PATH = os.path.join(MODELS_DIR, "hybrid_model.joblib")
EXTRACTOR_ARTIFACT_PATH = os.path.join(MODELS_DIR, "feature_extractor.joblib")

def load_data():
    """Loads dataset from CSV, generating the latest benchmark dataset."""
    generate_csv()

    titles, texts, labels, metadata = [], [], [], []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            titles.append(row["title"])
            texts.append(row["text"])
            labels.append(1 if row["label"].strip().upper() == "FAKE" else 0)
            metadata.append({
                "source": row.get("source", "Unknown"),
                "author": row.get("author", "Unknown"),
                "shares": float(row.get("shares", 1000)),
                "sentiment": float(row.get("sentiment", 0.0))
            })
    return titles, texts, np.array(labels), metadata

def run_training_pipeline():
    """Executes BERT feature extraction, deep neural network training, benchmark evaluation, and artifact saving."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 65)
    print("🚀 STARTING TRAINING: BERT Deep Learning & Hybrid Detection Pipeline")
    print("=" * 65)

    # 1. Load Data
    titles, texts, y, metadata = load_data()
    print(f"Loaded {len(titles)} articles (Real: {sum(y == 0)}, Fake: {sum(y == 1)})")

    # 2. Train / Test Split
    idx_train, idx_test = train_test_split(
        np.arange(len(titles)), test_size=0.25, random_state=42, stratify=y
    )

    titles_train = [titles[i] for i in idx_train]
    texts_train = [texts[i] for i in idx_train]
    meta_train = [metadata[i] for i in idx_train]
    y_train = y[idx_train]

    titles_test = [titles[i] for i in idx_test]
    texts_test = [texts[i] for i in idx_test]
    meta_test = [metadata[i] for i in idx_test]
    y_test = y[idx_test]

    # 3. Fit BERT Transformer & Feature Extractor
    print("\n[Step 1/4] Building BERT WordPiece vocabulary & initializing Transformer encoder...")
    extractor = HybridFeatureExtractor(embedding_dim=64, num_layers=4, num_heads=4)
    extractor.fit(titles_train + texts_train)

    print("Extracting deep contextual embeddings & fusing with multimodal features...")
    X_train = extractor.transform(titles_train, texts_train, meta_train)
    X_test = extractor.transform(titles_test, texts_test, meta_test)
    print(f"Generated fused feature representation: {X_train.shape[1]} dimensions per article (BERT: 64-d, Multimodal: 14-d).")

    # 4. Train Hybrid Model, BERT Deep Neural Network & Comparative Baselines
    print("\n[Step 2/4] Training BERT Deep Neural Classifier & Research Baseline Models...")
    classifier = HybridFakeNewsClassifier(n_estimators=150, random_state=42)
    metrics = classifier.fit_and_evaluate_all(X_train, y_train, X_test, y_test)

    # 5. Initialize Explainable AI Engine
    print("\n[Step 3/4] Initializing Explainable AI (BERT Self-Attention + Tree SHAP) Engine...")
    explainer = XAIExplainer(classifier.model, extractor.get_feature_names())
    global_importance = explainer.get_global_feature_importance(top_k=15)

    # 6. Save Artifacts
    print("\n[Step 4/4] Persisting Model Weights and Evaluation Reports...")
    joblib.dump(classifier.model, MODEL_ARTIFACT_PATH)
    joblib.dump(extractor, EXTRACTOR_ARTIFACT_PATH)

    metrics_payload = {
        "benchmark_metrics": metrics,
        "global_feature_importance": global_importance,
        "feature_count": X_train.shape[1],
        "bert_embedding_dim": extractor.embedding_dim,
        "training_samples": len(titles_train),
        "test_samples": len(titles_test),
        "database": "Firebase Firestore"
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"Artifacts successfully saved to: {MODELS_DIR}")
    print("\n" + "=" * 65)
    print("📊 BENCHMARK EVALUATION SUMMARY")
    print("=" * 65)
    for model_name, res in metrics.items():
        print(f"• {model_name:<30} | Accuracy: {res['accuracy']*100:.1f}% | F1: {res['f1_score']:.4f} | ROC-AUC: {res['roc_auc']:.4f}")
    print("=" * 65)

    return metrics_payload

if __name__ == "__main__":
    run_training_pipeline()
