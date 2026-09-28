"""
Phase 4 BERT Contextual Feature Extraction Automated Test Suite
Module Owner: Mohamed Irfaan M (Lead NLP & Deep Learning Engineer)

Verifies:
1. BERT transformer initialization & hyperparameter configuration
2. WordPiece subword tokenization, special token placement ([CLS], [SEP], [PAD]), and attention masks
3. Embedding layers (Token + Sinusoidal Positional + Segment Embeddings + LayerNorm)
4. Multi-Head Scaled Dot-Product Self-Attention computation and masking
5. Dense 64-dimensional [CLS] pooled contextual representation extraction
6. Single-text token attention heatmap generation
7. Model serialization, disk persistence, and deserialization
8. End-to-end integration with preprocessed dataset partitions
"""
import os
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.bert_features import (
    BertFeatureExtractor, run_bert_extraction_pipeline
)
from src.config import (
    BERT_EMBEDDING_DIM, BERT_NUM_LAYERS, BERT_NUM_HEADS,
    BERT_MAX_SEQ_LENGTH, BERT_VOCAB_SIZE, BERT_MODEL_PATH,
    PROCESSED_DATA_DIR, RANDOM_SEED
)


def test_bert_initialization():
    """Verifies that BertFeatureExtractor initializes with expected architecture settings."""
    extractor = BertFeatureExtractor(
        embedding_dim=BERT_EMBEDDING_DIM,
        num_layers=BERT_NUM_LAYERS,
        num_heads=BERT_NUM_HEADS,
        max_seq_length=BERT_MAX_SEQ_LENGTH,
        vocab_size=BERT_VOCAB_SIZE,
        random_state=RANDOM_SEED
    )
    assert extractor.embedding_dim == 64, f"Expected embedding_dim 64, got {extractor.embedding_dim}"
    assert extractor.num_layers == 4, f"Expected 4 layers, got {extractor.num_layers}"
    assert extractor.num_heads == 4, f"Expected 4 heads, got {extractor.num_heads}"
    assert extractor.is_fitted is False, "Extractor should be un-fitted upon initialization"
    
    # Assert exception raised when un-fitted
    try:
        extractor.extract_embeddings(["Test sentence."])
        assert False, "Should have raised ValueError for un-fitted extractor"
    except ValueError:
        pass
    print("[OK] BERT feature extractor initialization and safety guards verified.")


def test_wordpiece_tokenization():
    """Verifies WordPiece subword splitting, special tokens, and mask alignment."""
    extractor = BertFeatureExtractor(max_seq_length=16, vocab_size=500)
    corpus = [
        "NASA James Webb Space Telescope discovers distant galaxy.",
        "Clinical trials show high efficacy for pediatric malaria vaccine."
    ]
    extractor.fit(corpus)
    
    enc = extractor.bert.tokenizer.encode("NASA telescope discovery")
    assert enc["tokens"][0] == "[CLS]", "First token must be [CLS]"
    assert enc["tokens"][-1] == "[SEP]", "Last meaningful token must be [SEP]"
    assert len(enc["input_ids"]) == 16, f"Expected sequence length 16, got {len(enc['input_ids'])}"
    assert len(enc["attention_mask"]) == 16
    
    # Non-padded tokens should have mask=1.0, padded tokens mask=0.0
    num_tokens = len(enc["tokens"])
    assert np.all(enc["attention_mask"][:num_tokens] == 1.0), "Active tokens must have attention_mask = 1.0"
    assert np.all(enc["attention_mask"][num_tokens:] == 0.0), "Padding tokens must have attention_mask = 0.0"
    print("[OK] WordPiece subword tokenization and attention masking verified.")


def test_embedding_and_attention_layers():
    """Verifies token + sinusoidal positional embeddings and multi-head self-attention mechanics."""
    extractor = BertFeatureExtractor(embedding_dim=64, num_layers=4, num_heads=4, max_seq_length=32)
    corpus = ["Artificial intelligence systems detect synthetic misinformation accurately."]
    extractor.fit(corpus)

    enc = extractor.bert.tokenizer.encode(corpus[0])
    inp = enc["input_ids"][np.newaxis, :]
    mask = enc["attention_mask"][np.newaxis, :]
    token_types = enc["token_type_ids"][np.newaxis, :]

    # 1. Test Embeddings Layer
    embeddings = extractor.bert.embeddings.forward(inp, token_types)
    assert embeddings.shape == (1, 32, 64), f"Unexpected embedding shape: {embeddings.shape}"
    assert not np.isnan(embeddings).any(), "NaN found in BERT embeddings"

    # 2. Test Transformer Encoder Forward
    bert_out = extractor.bert.forward(inp, mask, token_types)
    assert "sequence_output" in bert_out
    assert "pooled_output" in bert_out
    assert "attention_maps" in bert_out
    assert len(bert_out["attention_maps"]) == 4, f"Expected 4 attention layers, got {len(bert_out['attention_maps'])}"
    
    # Check attention map shape: (batch_size, seq_len, seq_len)
    last_attn = bert_out["attention_maps"][-1]
    assert last_attn.shape == (1, 32, 32), f"Expected shape (1, 32, 32), got {last_attn.shape}"
    
    # Attention probabilities along sequence axis must be valid probabilities
    attn_sums = np.sum(last_attn, axis=-1)
    assert np.allclose(attn_sums, 1.0, atol=1e-4), f"Self-attention scores do not sum to 1.0: {attn_sums}"
    print("[OK] Embedding layers and multi-head self-attention forward pass verified.")


def test_pooled_cls_extraction():
    """Verifies dense 64-dimensional [CLS] representation extraction and batch handling."""
    extractor = BertFeatureExtractor(embedding_dim=64, num_layers=4, num_heads=4, max_seq_length=64)
    texts = [
        "NASA announces new astrophysics discovery.",
        "Breaking news: Secret miracle potion banned by doctors!",
        "Quarterly earnings report published by international bank.",
        "Shocking proof of ancient alien spacecraft under ice sheet!"
    ]
    extractor.fit(texts)

    embeddings = extractor.extract_embeddings(texts, batch_size=2)
    assert isinstance(embeddings, np.ndarray), "Output must be a numpy array"
    assert embeddings.shape == (4, 64), f"Expected shape (4, 64), got {embeddings.shape}"
    assert not np.isnan(embeddings).any(), "NaN detected in pooled representations"
    
    # Verify semantic differentiation (embeddings for different texts should not be identical)
    norm_0 = np.linalg.norm(embeddings[0])
    norm_1 = np.linalg.norm(embeddings[1])
    assert norm_0 > 0.01 and norm_1 > 0.01, "Embeddings have zero magnitude"
    
    dot_prod = np.dot(embeddings[0], embeddings[1]) / (norm_0 * norm_1)
    assert dot_prod < 0.999, f"Embeddings for different texts are nearly identical: cosine={dot_prod}"
    print("[OK] Dense 64-dimensional [CLS] pooled representation extraction verified.")


def test_single_attention_extraction():
    """Verifies word-level attention heatmap extraction for Explainable AI."""
    extractor = BertFeatureExtractor(embedding_dim=64, num_layers=4, num_heads=4, max_seq_length=64)
    extractor.fit(["NASA James Webb Space Telescope captures distant galaxy cluster."])

    sample = "NASA telescope captures deep field galaxies"
    emb, tokens, attn = extractor.extract_single_with_attention(sample)

    assert emb.shape == (64,), f"Expected 1D embedding of size 64, got {emb.shape}"
    assert isinstance(tokens, list) and len(tokens) > 0, "Tokens list empty"
    assert isinstance(attn, np.ndarray), "Attention weights must be a numpy array"
    assert len(tokens) == len(attn), f"Tokens length ({len(tokens)}) != attention weights length ({len(attn)})"
    assert np.isclose(np.sum(attn), 1.0, atol=1e-4), f"Attention weights must sum to 1.0, got {np.sum(attn)}"
    print(f"[OK] Explainable AI attention heatmap extraction verified ({len(tokens)} tokens mapped).")


def test_feature_naming():
    """Verifies get_feature_names() returns 64 consistent feature names."""
    extractor = BertFeatureExtractor(embedding_dim=64)
    names = extractor.get_feature_names()
    assert len(names) == 64, f"Expected 64 feature names, got {len(names)}"
    assert names[0] == "bert_dim_00"
    assert names[-1] == "bert_dim_63"
    print("[OK] BERT feature naming schema verified.")


def test_serialization_and_deserialization():
    """Verifies that BertFeatureExtractor saves to and reloads from disk cleanly."""
    extractor = BertFeatureExtractor(embedding_dim=64)
    corpus = ["Test document for persistence check."]
    extractor.fit(corpus)

    test_path = os.path.join(PROCESSED_DATA_DIR, "test_bert_temp.joblib")
    extractor.save(test_path)
    assert os.path.exists(test_path), "Failed to persist BERT extractor to disk"

    loaded = BertFeatureExtractor.load(test_path)
    assert loaded.is_fitted is True
    assert loaded.embedding_dim == 64

    # Compare embedding outputs
    emb_orig = extractor.extract_embeddings(corpus)
    emb_loaded = loaded.extract_embeddings(corpus)
    assert np.allclose(emb_orig, emb_loaded, atol=1e-5), "Reloaded model produced different embeddings"

    # Cleanup temp file
    if os.path.exists(test_path):
        os.remove(test_path)
    print("[OK] BERT Feature Extractor serialization and deserialization verified.")


def test_end_to_end_pipeline():
    """Verifies running the full Phase 4 BERT extraction pipeline."""
    extractor = run_bert_extraction_pipeline()
    assert extractor.is_fitted is True
    assert os.path.exists(BERT_MODEL_PATH), f"Missing model artifact at {BERT_MODEL_PATH}"
    
    # Check generated embedding numpy arrays
    train_emb_path = os.path.join(PROCESSED_DATA_DIR, "bert_train_embeddings.npy")
    val_emb_path = os.path.join(PROCESSED_DATA_DIR, "bert_val_embeddings.npy")
    test_emb_path = os.path.join(PROCESSED_DATA_DIR, "bert_test_embeddings.npy")

    assert os.path.exists(train_emb_path), "bert_train_embeddings.npy missing"
    assert os.path.exists(val_emb_path), "bert_val_embeddings.npy missing"
    assert os.path.exists(test_emb_path), "bert_test_embeddings.npy missing"

    train_arr = np.load(train_emb_path)
    test_arr = np.load(test_emb_path)
    assert train_arr.shape[1] == 64, f"Expected 64-dim embeddings, got {train_arr.shape[1]}"
    assert test_arr.shape[1] == 64, f"Expected 64-dim embeddings, got {test_arr.shape[1]}"
    print(f"[OK] Full Phase 4 pipeline execution verified (Train: {train_arr.shape}, Test: {test_arr.shape}).")


if __name__ == "__main__":
    print("=" * 75)
    print("RUNNING PHASE 4 BERT FEATURE EXTRACTION AUTOMATED TEST SUITE")
    print("=" * 75)
    test_bert_initialization()
    test_wordpiece_tokenization()
    test_embedding_and_attention_layers()
    test_pooled_cls_extraction()
    test_single_attention_extraction()
    test_feature_naming()
    test_serialization_and_deserialization()
    test_end_to_end_pipeline()
    print("=" * 75)
    print("ALL PHASE 4 BERT FEATURE EXTRACTION TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)
