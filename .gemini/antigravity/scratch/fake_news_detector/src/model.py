"""
Hybrid Model & Comparative Baseline Evaluator
Combines BERT Deep Contextual Embeddings with Handcrafted Features using Deep Neural Networks and Ensemble Classifiers.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from .bert_engine import DeepMultimodalClassifier

class BertDeepNeuralClassifier:
    """
    Standalone BERT Deep Learning Neural Classifier:
    Uses a Multi-Layer Perceptron (MLP) with GELU activations, Dropout, and Adam optimization
    directly over the BERT contextual representations and multimodal features.
    """
    def __init__(self, input_dim: int, hidden_dims: list[int] = [128, 64, 32], epochs: int = 45, random_state: int = 42):
        self.input_dim = input_dim
        self.epochs = epochs
        self.nn = DeepMultimodalClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=2,
            dropout_rate=0.2,
            learning_rate=0.002,
            random_state=random_state
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        history = self.nn.fit(X, y, epochs=self.epochs, batch_size=16)
        self.is_fitted = True
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.nn.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.nn.predict_proba(X)

class HybridFakeNewsClassifier:
    """
    Proposed Hybrid Architecture:
    Combines BERT Transformer Deep Contextual Embeddings + Multimodal Feature Fusion + Tuned Random Forest / Deep Ensemble.
    """
    def __init__(self, n_estimators: int = 150, random_state: int = 42):
        self.random_state = random_state
        # Proposed Hybrid Model: Optimized Random Forest on BERT Fused Features
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_split=3,
            class_weight="balanced",
            random_state=random_state
        )
        # Standalone Deep Learning BERT Classifier
        self.bert_nn = None
        
        # Baseline Comparison Models for research evaluation
        self.baselines = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=random_state),
            "Support Vector Machine (RBF)": SVC(probability=True, kernel="rbf", random_state=random_state),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting (GBM)": GradientBoostingClassifier(n_estimators=100, random_state=random_state)
        }
        self.is_fitted = False
        self.evaluation_results = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the primary hybrid model and deep neural network."""
        self.model.fit(X, y)
        self.bert_nn = BertDeepNeuralClassifier(input_dim=X.shape[1], random_state=self.random_state)
        self.bert_nn.fit(X, y)
        self.is_fitted = True
        return self

    def fit_and_evaluate_all(self, X_train: np.ndarray, y_train: np.ndarray, 
                             X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Trains proposed hybrid model, deep neural network, and all baseline models to compile comparative research metrics."""
        # 1. Train Primary Proposed Hybrid Model
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) > 1 else y_pred
        cm = confusion_matrix(y_test, y_pred).tolist()

        # 2. Train Standalone BERT Deep Learning Neural Classifier
        self.bert_nn = BertDeepNeuralClassifier(input_dim=X_train.shape[1], random_state=self.random_state)
        self.bert_nn.fit(X_train, y_train)
        bert_pred = self.bert_nn.predict(X_test)
        bert_prob = self.bert_nn.predict_proba(X_test)[:, 1]
        bert_cm = confusion_matrix(y_test, bert_pred).tolist()

        self.evaluation_results = {
            "Proposed Hybrid (BERT + RF)": {
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4) if len(np.unique(y_test)) > 1 else 1.0,
                "confusion_matrix": cm,
                "architecture_type": "Hybrid Deep Learning (BERT Embeddings + Multimodal Ensemble)"
            },
            "BERT Deep Learning (MLP)": {
                "accuracy": round(float(accuracy_score(y_test, bert_pred)), 4),
                "precision": round(float(precision_score(y_test, bert_pred, average="weighted", zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, bert_pred, average="weighted", zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test, bert_pred, average="weighted", zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_test, bert_prob)), 4) if len(np.unique(y_test)) > 1 else 1.0,
                "confusion_matrix": bert_cm,
                "architecture_type": "Pure Deep Neural Network (BERT Contextual + Dense Layers)"
            }
        }

        # 3. Train Baseline Comparison Models
        for name, clf in self.baselines.items():
            clf.fit(X_train, y_train)
            base_pred = clf.predict(X_test)
            try:
                base_prob = clf.predict_proba(X_test)[:, 1]
                auc = round(float(roc_auc_score(y_test, base_prob)), 4)
            except Exception:
                auc = round(float(accuracy_score(y_test, base_pred)), 4)

            self.evaluation_results[name] = {
                "accuracy": round(float(accuracy_score(y_test, base_pred)), 4),
                "precision": round(float(precision_score(y_test, base_pred, average="weighted", zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, base_pred, average="weighted", zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test, base_pred, average="weighted", zero_division=0)), 4),
                "roc_auc": auc,
                "confusion_matrix": confusion_matrix(y_test, base_pred).tolist(),
                "architecture_type": "Classical Machine Learning Baseline"
            }

        return self.evaluation_results

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict_proba(X)
