"""
Multi-Source Feature Fusion Module (Phase 7)
Module Owner: Bhuvanesh M (Lead Data & Ensemble Learning Engineer) & Mohamed Irfaan M (Lead NLP Engineer)

Constructs the unified 79-Dimensional Multimodal Feature Fusion Tensor:
- BERT Deep Contextual Sequence Representations (64 dimensions)
- User & Publisher Credibility Features (6 dimensions)
- Network & Social Engagement Features (6 dimensions)
- Stylometric & Linguistic Density Indicators (3 dimensions)

Formula:
x_fused = [z_BERT || f_user || f_network || f_stylometrics] in R^79
"""
import os
import sys
import csv
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import joblib
except ImportError:
    import pickle as joblib

from .bert_features import BertFeatureExtractor
from .user_features import UserFeatureExtractor
from .network_features import NetworkFeatureExtractor
from .preprocessing import TextPreprocessor, run_preprocessing_pipeline
from .config import (
    BERT_EMBEDDING_DIM, BERT_NUM_LAYERS, BERT_NUM_HEADS,
    BERT_MAX_SEQ_LENGTH, BERT_VOCAB_SIZE, FEATURE_FUSION_PATH,
    PROCESSED_TRAIN_PATH, PROCESSED_VAL_PATH, PROCESSED_TEST_PATH,
    PROCESSED_DATA_DIR, RANDOM_SEED
)


class MultiSourceFeatureFusion:
    """
    Multi-Source Feature Fusion Layer:
    Fuses 64-dimensional BERT contextual embeddings with 15 multi-source features:
    - 64-d BERT Transformer representations
    - 6-d User & Publisher credibility features
    - 6-d Network & Social engagement features
    - 3-d Stylometric linguistic metrics
    Total Dimension: 79 features per article.
    """
    def __init__(self, embedding_dim: int = BERT_EMBEDDING_DIM, num_layers: int = BERT_NUM_LAYERS,
                 num_heads: int = BERT_NUM_HEADS, max_seq_length: int = BERT_MAX_SEQ_LENGTH,
                 vocab_size: int = BERT_VOCAB_SIZE, random_state: int = RANDOM_SEED):
        self.embedding_dim = embedding_dim
        self.bert_extractor = BertFeatureExtractor(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_length=max_seq_length,
            vocab_size=vocab_size,
            random_state=random_state
        )
        self.user_extractor = UserFeatureExtractor()
        self.network_extractor = NetworkFeatureExtractor()
        self.text_preprocessor = TextPreprocessor()
        self.is_fitted = False

    def fit(self, texts: list[str]):
        """Fits the underlying BERT WordPiece tokenizer on text corpus."""
        if not texts:
            raise ValueError("Cannot fit MultiSourceFeatureFusion on empty text corpus.")
        self.bert_extractor.fit(texts)
        self.is_fitted = True
        return self

    def _extract_stylometrics_batch(self, texts: list[str]) -> np.ndarray:
        """Extracts 3-dimensional stylometric feature vectors for a list of texts."""
        stylometrics = []
        for txt in texts:
            stats = self.text_preprocessor.extract_linguistic_stats(txt)
            sensational_intensity = min(1.0, (stats["exclamation_count"] / 3.0) + (stats["sensational_word_count"] / 2.0))
            stylometrics.append([
                stats["uppercase_ratio"],
                sensational_intensity,
                stats["lexical_diversity"]
            ])
        return np.array(stylometrics, dtype=np.float32)

    def transform(self, titles: list[str], texts: list[str], metadata_list: list[dict] = None) -> np.ndarray:
        """
        Transforms text and metadata into the unified 79-dimensional fused feature matrix.
        Returns a numpy array of shape (N, 79).
        """
        if not self.is_fitted:
            raise ValueError("MultiSourceFeatureFusion is not fitted. Call fit() first.")
        if not titles and not texts:
            return np.empty((0, 79), dtype=np.float32)

        N = len(titles) if titles else len(texts)
        if titles and not texts:
            texts = [""] * N
        elif texts and not titles:
            titles = [""] * N

        combined_texts = [f"{t} {b}".strip() for t, b in zip(titles, texts)]

        if metadata_list is None or len(metadata_list) != N:
            metadata_list = [{"source": "Unknown", "author": "Unknown", "shares": 1000.0, "sentiment": 0.0}] * N

        # 1. BERT Contextual Embeddings (64 Dimensions)
        z_bert = self.bert_extractor.extract_embeddings(combined_texts)

        # 2. User & Publisher Credibility Features (6 Dimensions)
        f_user = self.user_extractor.extract_batch(metadata_list, combined_texts)

        # 3. Network & Social Engagement Features (6 Dimensions)
        f_network = self.network_extractor.extract_batch(metadata_list, combined_texts)

        # 4. Stylometric Linguistic Density Features (3 Dimensions)
        f_stylo = self._extract_stylometrics_batch(combined_texts)

        # 5. Multimodal Concatenation: 64 + 6 + 6 + 3 = 79 Dimensions
        x_fused = np.hstack([z_bert, f_user, f_network, f_stylo])
        return x_fused.astype(np.float32)

    def fit_transform(self, titles: list[str], texts: list[str], metadata_list: list[dict] = None) -> np.ndarray:
        combined_texts = [f"{t} {b}".strip() for t, b in zip(titles, texts)]
        self.fit(combined_texts)
        return self.transform(titles, texts, metadata_list)

    def encode_single_with_attention(self, title: str, text: str, metadata: dict = None) -> tuple[np.ndarray, list[str], np.ndarray]:
        """
        Extracts 79-dimensional fused feature vector, WordPiece tokens, and self-attention weights
        for single article live inference and Explainable AI heatmap visualization.
        """
        if not self.is_fitted:
            raise ValueError("MultiSourceFeatureFusion is not fitted. Call fit() first.")

        combined_text = f"{title} {text}".strip()
        if metadata is None:
            metadata = {"source": "Unknown", "author": "Unknown", "shares": 1000.0, "sentiment": 0.0}

        # 1. BERT embedding + attention heatmap
        z_bert, tokens, attn_weights = self.bert_extractor.extract_single_with_attention(combined_text)

        # 2. User features (6-dim)
        f_user = self.user_extractor.extract_features(metadata, combined_text)

        # 3. Network features (6-dim)
        f_network = self.network_extractor.extract_features(metadata, combined_text)

        # 4. Stylometrics (3-dim)
        stats = self.text_preprocessor.extract_linguistic_stats(combined_text)
        sensational_intensity = min(1.0, (stats["exclamation_count"] / 3.0) + (stats["sensational_word_count"] / 2.0))
        f_stylo = np.array([
            stats["uppercase_ratio"],
            sensational_intensity,
            stats["lexical_diversity"]
        ], dtype=np.float32)

        # 5. Unified 79-dimensional vector
        fused_vector = np.concatenate([z_bert, f_user, f_network, f_stylo]).astype(np.float32)
        return fused_vector, tokens, attn_weights

    def get_feature_names(self) -> list[str]:
        """Returns the 79 feature names in precise concatenation order."""
        bert_names = self.bert_extractor.get_feature_names()  # 64
        user_names = self.user_extractor.get_feature_names()  # 6
        net_names = self.network_extractor.get_feature_names()  # 6
        stylo_names = [
            "stylometric_uppercase_ratio",
            "stylometric_exclamation_density",
            "stylometric_lexical_diversity"
        ]  # 3
        return bert_names + user_names + net_names + stylo_names

    def save(self, path: str = FEATURE_FUSION_PATH):
        """Persists the fitted feature fusion extractor to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[OK] MultiSourceFeatureFusion persisted to: {path}")

    @classmethod
    def load(cls, path: str = FEATURE_FUSION_PATH) -> "MultiSourceFeatureFusion":
        """Loads a persisted feature fusion extractor from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature fusion artifact not found at: {path}")
        return joblib.load(path)


def load_dataset_split(csv_path: str) -> tuple[list[str], list[str], list[dict], np.ndarray]:
    """Loads titles, texts, metadata dictionaries, and labels from a split CSV."""
    titles, texts, metadata, labels = [], [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            titles.append(row.get("title", ""))
            texts.append(row.get("text", ""))
            metadata.append({
                "source": row.get("source", "Unknown"),
                "author": row.get("author", "Unknown"),
                "shares": float(row.get("shares", 1000.0)),
                "sentiment": float(row.get("sentiment", 0.0))
            })
            labels.append(int(row.get("label", 0)))
    return titles, texts, metadata, np.array(labels, dtype=np.int32)


def run_feature_fusion_pipeline() -> MultiSourceFeatureFusion:
    """
    Executes Phase 7 Multi-Source Feature Fusion Pipeline:
    1. Loads preprocessed Train, Validation, and Test splits.
    2. Fits BERT and multi-source encoders on training corpus.
    3. Transforms all splits into 79-dimensional fused feature matrices.
    4. Persists fused numpy arrays and the fitted MultiSourceFeatureFusion model artifact.
    5. Displays multimodal feature breakdown diagnostics.
    """
    print("=" * 80)
    print("PHASE 7: MULTI-SOURCE FEATURE FUSION PIPELINE")
    print("=" * 80)

    # Ensure preprocessed splits exist
    if not (os.path.exists(PROCESSED_TRAIN_PATH) and os.path.exists(PROCESSED_TEST_PATH)):
        print("Processed data splits not found. Running Phase 2 preprocessing...")
        run_preprocessing_pipeline()

    # 1. Load Data
    titles_train, texts_train, meta_train, y_train = load_dataset_split(PROCESSED_TRAIN_PATH)
    titles_val, texts_val, meta_val, y_val = load_dataset_split(PROCESSED_VAL_PATH)
    titles_test, texts_test, meta_test, y_test = load_dataset_split(PROCESSED_TEST_PATH)

    print(f"Loaded partitions: Train={len(titles_train)}, Val={len(titles_val)}, Test={len(titles_test)} records.")

    # 2. Fit and Transform
    print("\n[Step 1/2] Fitting Multi-Source Fusion Layer & Transforming Partitions...")
    fusion = MultiSourceFeatureFusion()
    combined_train = [f"{t} {b}".strip() for t, b in zip(titles_train, texts_train)]
    fusion.fit(combined_train)

    X_train_fused = fusion.transform(titles_train, texts_train, meta_train)
    X_val_fused = fusion.transform(titles_val, texts_val, meta_val)
    X_test_fused = fusion.transform(titles_test, texts_test, meta_test)

    print(f"  - Train Fused Matrix Shape: {X_train_fused.shape} (Expected: (24, 79))")
    print(f"  - Val Fused Matrix Shape:   {X_val_fused.shape} (Expected: (4, 79))")
    print(f"  - Test Fused Matrix Shape:  {X_test_fused.shape} (Expected: (7, 79))")

    # 3. Save Fused Features and Model Artifact
    print("\n[Step 2/2] Persisting Fused Feature Matrices and Fusion Extractor...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DATA_DIR, "fused_train_features.npy"), X_train_fused)
    np.save(os.path.join(PROCESSED_DATA_DIR, "fused_val_features.npy"), X_val_fused)
    np.save(os.path.join(PROCESSED_DATA_DIR, "fused_test_features.npy"), X_test_fused)

    fusion.save(FEATURE_FUSION_PATH)

    # 4. Modality Breakdown Diagnostics
    names = fusion.get_feature_names()
    print("\n" + "=" * 80)
    print("🔬 MULTIMODAL FEATURE FUSION MODALITY BREAKDOWN")
    print("=" * 80)
    print(f"Total Fused Feature Dimensions: {len(names)}")
    print(f"  • BERT Contextual Embeddings:  64 Dimensions ({names[0]} to {names[63]})")
    print(f"  • User & Publisher Signals:    6 Dimensions ({names[64]} to {names[69]})")
    print(f"  • Network & Engagement Cues:   6 Dimensions ({names[70]} to {names[75]})")
    print(f"  • Stylometric Linguistic Stats:3 Dimensions ({names[76]} to {names[78]})")
    print("=" * 80 + "\n")

    return fusion


if __name__ == "__main__":
    run_feature_fusion_pipeline()
