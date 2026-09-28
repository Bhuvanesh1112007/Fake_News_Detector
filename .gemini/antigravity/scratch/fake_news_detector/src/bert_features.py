"""
BERT Contextual Feature Extraction Module (Phase 4)
Module Owner: Mohamed Irfaan M (Lead NLP & Deep Learning Engineer)

Encapsulates WordPiece tokenization, BERT Transformer multi-head self-attention encoder,
dense contextual representation pooling (64-dim vectors), and attention heatmap extraction.
"""
import os
import sys
import csv
import time
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import joblib
except ImportError:
    import pickle as joblib

from .bert_engine import BertModel
from .config import (
    BERT_EMBEDDING_DIM, BERT_NUM_LAYERS, BERT_NUM_HEADS,
    BERT_MAX_SEQ_LENGTH, BERT_VOCAB_SIZE, BERT_MODEL_PATH,
    PROCESSED_TRAIN_PATH, PROCESSED_VAL_PATH, PROCESSED_TEST_PATH,
    MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED
)
from .preprocessing import run_preprocessing_pipeline


class BertFeatureExtractor:
    """
    BERT Contextual Feature Extractor:
    Extracts 64-dimensional semantic sequence representations from text using
    bidirectional multi-head self-attention transformer layers.
    """
    def __init__(self, embedding_dim: int = BERT_EMBEDDING_DIM, num_layers: int = BERT_NUM_LAYERS,
                 num_heads: int = BERT_NUM_HEADS, max_seq_length: int = BERT_MAX_SEQ_LENGTH,
                 vocab_size: int = BERT_VOCAB_SIZE, random_state: int = RANDOM_SEED):
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.random_state = random_state
        
        self.bert = BertModel(
            vocab_size=vocab_size,
            hidden_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_dim=embedding_dim * 2,
            max_seq_length=max_seq_length
        )
        self.is_fitted = False

    def fit(self, texts: list[str]):
        """Builds WordPiece subword vocabulary and initializes semantic transformer weights."""
        if not texts:
            raise ValueError("Cannot fit BertFeatureExtractor on empty text corpus.")
        self.bert.fit_tokenizer(texts, max_vocab_size=self.vocab_size)
        self.is_fitted = True
        return self

    def extract_embeddings(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Extracts pooled [CLS] embeddings for a list of texts in batches.
        Returns a numpy array of shape (N, embedding_dim).
        """
        if not self.is_fitted:
            raise ValueError("BertFeatureExtractor is not fitted. Call fit() first.")
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded_batch = self.bert.tokenizer.encode_batch(batch_texts)
            res = self.bert.forward(
                input_ids=encoded_batch["input_ids"],
                attention_mask=encoded_batch["attention_mask"],
                token_type_ids=encoded_batch["token_type_ids"]
            )
            embeddings.append(res["pooled_output"])

        return np.vstack(embeddings)

    def extract_single_with_attention(self, text: str) -> tuple[np.ndarray, list[str], np.ndarray]:
        """
        Extracts single text embedding, WordPiece tokens, and [CLS] self-attention weights.
        Returns:
            - embedding: 1D numpy array of shape (embedding_dim,)
            - tokens: list of token strings (e.g. ['[CLS]', 'nasa', 'discovers', ...])
            - cls_attn: 1D numpy array of attention scores from [CLS] to each token in the sequence.
        """
        if not self.is_fitted:
            raise ValueError("BertFeatureExtractor is not fitted. Call fit() first.")

        enc = self.bert.tokenizer.encode(text)
        inp = enc["input_ids"][np.newaxis, :]
        mask = enc["attention_mask"][np.newaxis, :]
        token_types = enc["token_type_ids"][np.newaxis, :]
        
        bert_res = self.bert.forward(inp, mask, token_types)
        embedding = bert_res["pooled_output"][0]
        
        # Last-layer attention map: shape (seq_len, seq_len)
        last_attn = bert_res["attention_maps"][-1][0]
        # Attention from [CLS] token (row 0) to all active sequence tokens
        cls_attn = last_attn[0, :len(enc["tokens"])].copy()
        
        # Normalize attention weights to sum to 1.0 for valid visualization
        attn_sum = np.sum(cls_attn)
        if attn_sum > 0:
            cls_attn = cls_attn / attn_sum

        return embedding, enc["tokens"], cls_attn

    def get_feature_names(self) -> list[str]:
        """Returns feature column names for the BERT embedding dimensions."""
        return [f"bert_dim_{i:02d}" for i in range(self.embedding_dim)]

    def save(self, path: str = BERT_MODEL_PATH):
        """Persists the fitted BERT feature extractor to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[OK] BERT Feature Extractor persisted to: {path}")

    @classmethod
    def load(cls, path: str = BERT_MODEL_PATH) -> "BertFeatureExtractor":
        """Loads a persisted BERT feature extractor from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"BERT model artifact not found at: {path}")
        obj = joblib.load(path)
        return obj


def load_texts_from_split(csv_path: str) -> list[str]:
    """Loads combined title and article text from a processed CSV file."""
    combined_texts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("title", "").strip()
            body = row.get("text", "").strip()
            combined_texts.append(f"{title} {body}".strip())
    return combined_texts


def run_bert_extraction_pipeline() -> BertFeatureExtractor:
    """
    Executes Phase 4 BERT Contextual Feature Extraction Pipeline:
    1. Loads preprocessed Train, Validation, and Test partitions.
    2. Builds WordPiece vocabulary and initializes Transformer weights.
    3. Extracts 64-dimensional dense contextual embeddings for all splits.
    4. Persists the fitted BERT extractor and embedding matrices.
    5. Displays diagnostic summary.
    """
    print("=" * 80)
    print("PHASE 4: BERT CONTEXTUAL FEATURE EXTRACTION PIPELINE")
    print("=" * 80)

    # Ensure preprocessed splits exist
    if not (os.path.exists(PROCESSED_TRAIN_PATH) and os.path.exists(PROCESSED_TEST_PATH)):
        print("Processed data splits not found. Triggering Phase 2 data preprocessing...")
        run_preprocessing_pipeline()

    # 1. Load Texts
    train_texts = load_texts_from_split(PROCESSED_TRAIN_PATH)
    val_texts = load_texts_from_split(PROCESSED_VAL_PATH)
    test_texts = load_texts_from_split(PROCESSED_TEST_PATH)

    print(f"Loaded Corpus: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)} articles.")

    # 2. Fit BERT Feature Extractor
    print("\n[Step 1/3] Building WordPiece Vocabulary & Initializing 4-Layer Transformer Encoder...")
    t0 = time.perf_counter()
    extractor = BertFeatureExtractor(
        embedding_dim=BERT_EMBEDDING_DIM,
        num_layers=BERT_NUM_LAYERS,
        num_heads=BERT_NUM_HEADS,
        max_seq_length=BERT_MAX_SEQ_LENGTH,
        vocab_size=BERT_VOCAB_SIZE,
        random_state=RANDOM_SEED
    )
    extractor.fit(train_texts)
    fit_time = time.perf_counter() - t0
    print(f"Tokenizer fitted in {fit_time:.3f}s. Vocabulary size: {len(extractor.bert.tokenizer.vocab)} tokens.")

    # 3. Extract Embeddings
    print("\n[Step 2/3] Extracting 64-Dimensional Pooled [CLS] Embeddings across partitions...")
    t1 = time.perf_counter()
    X_train_bert = extractor.extract_embeddings(train_texts)
    X_val_bert = extractor.extract_embeddings(val_texts)
    X_test_bert = extractor.extract_embeddings(test_texts)
    extract_time = time.perf_counter() - t1

    print(f"Extraction completed in {extract_time:.3f}s ({(extract_time / (len(train_texts)+len(val_texts)+len(test_texts))) * 1000:.2f} ms/sample).")
    print(f"  - Train Embeddings Shape: {X_train_bert.shape}")
    print(f"  - Val Embeddings Shape:   {X_val_bert.shape}")
    print(f"  - Test Embeddings Shape:  {X_test_bert.shape}")

    # 4. Save Embeddings and Model Artifacts
    print("\n[Step 3/3] Persisting BERT Embeddings and Model Weights...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DATA_DIR, "bert_train_embeddings.npy"), X_train_bert)
    np.save(os.path.join(PROCESSED_DATA_DIR, "bert_val_embeddings.npy"), X_val_bert)
    np.save(os.path.join(PROCESSED_DATA_DIR, "bert_test_embeddings.npy"), X_test_bert)

    extractor.save(BERT_MODEL_PATH)

    # 5. Diagnostic Sample Extraction
    sample_news = "NASA James Webb Space Telescope discovers ancient galaxy cluster with deep spectroscopic data."
    emb, tokens, attn = extractor.extract_single_with_attention(sample_news)
    print("\n" + "-" * 70)
    print("🔬 DIAGNOSTIC SAMPLE: BERT SELF-ATTENTION TOKEN MAPPING")
    print("-" * 70)
    print(f"Sample Headline: \"{sample_news}\"")
    print(f"Extracted Embedding Norm: {np.linalg.norm(emb):.4f} (Dim: {len(emb)})")
    print("Token Self-Attention Breakdown (Top 8 Tokens):")
    top_indices = np.argsort(attn)[::-1][:min(8, len(tokens))]
    for idx in top_indices:
        print(f"  • Token: {tokens[idx]:<15} | Attention Score: {attn[idx]:.4f} ({attn[idx]*100:.1f}%)")
    print("-" * 70 + "\n")

    return extractor


if __name__ == "__main__":
    run_bert_extraction_pipeline()
