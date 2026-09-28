"""
BERT Deep Learning Architecture & Transformer Engine
Provides WordPiece tokenization, Multi-Head Self-Attention Transformer Encoder,
Semantic Subword Embeddings, Deep Multimodal Neural Network classifier, and AdamW optimization.
"""
import os
import json
import math
import numpy as np

def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit (GELU) activation function as used in BERT."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3))))

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along specified axis."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class WordPieceTokenizer:
    """
    BERT WordPiece Tokenizer with subword segmentation,
    special tokens ([CLS], [SEP], [UNK], [PAD]), and attention mask generation.
    """
    def __init__(self, max_seq_length: int = 128, vocab: dict = None):
        self.max_seq_length = max_seq_length
        self.special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[MASK]": 4
        }
        if vocab is None:
            self.vocab = dict(self.special_tokens)
            self.inv_vocab = {v: k for k, v in self.vocab.items()}
            self.is_fitted = False
        else:
            self.vocab = vocab
            self.inv_vocab = {v: k for k, v in self.vocab.items()}
            self.is_fitted = True

    def build_vocab(self, texts: list[str], max_vocab_size: int = 5000):
        """Constructs subword vocabulary from corpus."""
        word_freq = {}
        for text in texts:
            words = text.lower().replace("\n", " ").split()
            for w in words:
                cleaned = "".join(c for c in w if c.isalnum() or c in "-'")
                if cleaned:
                    word_freq[cleaned] = word_freq.get(cleaned, 0) + 1

        # Sort words by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Add root words and subword prefixes
        current_id = len(self.special_tokens)
        for word, _ in sorted_words:
            if current_id >= max_vocab_size:
                break
            if word not in self.vocab:
                self.vocab[word] = current_id
                self.inv_vocab[current_id] = word
                current_id += 1

            # Subwords
            if len(word) > 4 and current_id < max_vocab_size:
                prefix = word[:3]
                suffix = "##" + word[3:]
                for sub in [prefix, suffix]:
                    if sub not in self.vocab and current_id < max_vocab_size:
                        self.vocab[sub] = current_id
                        self.inv_vocab[current_id] = sub
                        current_id += 1

        self.is_fitted = True
        return self

    def tokenize_word(self, word: str) -> list[str]:
        """Splits a single word into WordPiece subwords."""
        if word in self.vocab:
            return [word]
        
        subwords = []
        start = 0
        while start < len(word):
            end = len(word)
            cur_substr = None
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                subwords.append("[UNK]")
                break
            subwords.append(cur_substr)
            start = end
        return subwords if subwords else ["[UNK]"]

    def encode(self, text: str) -> dict:
        """Tokenizes text, adds [CLS] and [SEP], pads to max_seq_length, and returns tensor dictionary."""
        words = text.lower().replace("\n", " ").split()
        tokens = ["[CLS]"]
        for w in words:
            cleaned = "".join(c for c in w if c.isalnum() or c in "-'")
            if cleaned:
                tokens.extend(self.tokenize_word(cleaned))
            if len(tokens) >= self.max_seq_length - 1:
                break
        tokens.append("[SEP]")

        input_ids = [self.vocab.get(t, self.special_tokens["[UNK]"]) for t in tokens]
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        # Padding
        pad_len = self.max_seq_length - len(input_ids)
        if pad_len > 0:
            input_ids.extend([self.special_tokens["[PAD]"]] * pad_len)
            attention_mask.extend([0] * pad_len)
            token_type_ids.extend([0] * pad_len)
        else:
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            token_type_ids = token_type_ids[:self.max_seq_length]

        return {
            "tokens": tokens,
            "input_ids": np.array(input_ids, dtype=np.int32),
            "attention_mask": np.array(attention_mask, dtype=np.float32),
            "token_type_ids": np.array(token_type_ids, dtype=np.int32)
        }

    def encode_batch(self, texts: list[str]) -> dict:
        """Encodes a list of texts into batch matrices."""
        encoded_list = [self.encode(t) for t in texts]
        return {
            "tokens": [e["tokens"] for e in encoded_list],
            "input_ids": np.stack([e["input_ids"] for e in encoded_list]),
            "attention_mask": np.stack([e["attention_mask"] for e in encoded_list]),
            "token_type_ids": np.stack([e["token_type_ids"] for e in encoded_list])
        }

class LayerNorm:
    """Layer Normalization layer with learned scale (gamma) and bias (beta)."""
    def __init__(self, hidden_dim: int, eps: float = 1e-12):
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.gamma = np.ones(hidden_dim, dtype=np.float32)
        self.beta = np.zeros(hidden_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(variance + self.eps)
        return self.gamma * x_norm + self.beta

class BertEmbeddings:
    """Combines WordPiece Token Embeddings, Positional Embeddings, and Segment Embeddings."""
    def __init__(self, vocab_size: int, hidden_dim: int, max_position_embeddings: int = 128, random_state: int = 42):
        rng = np.random.RandomState(random_state)
        # Token Embeddings with semantic variance
        self.word_embeddings = rng.normal(0.0, 0.08, (vocab_size, hidden_dim)).astype(np.float32)
        # Learned Positional Embeddings (Sinusoidal initialization)
        self.position_embeddings = np.zeros((max_position_embeddings, hidden_dim), dtype=np.float32)
        for pos in range(max_position_embeddings):
            for i in range(0, hidden_dim, 2):
                self.position_embeddings[pos, i] = math.sin(pos / (10000 ** (i / hidden_dim)))
                if i + 1 < hidden_dim:
                    self.position_embeddings[pos, i + 1] = math.cos(pos / (10000 ** (i / hidden_dim)))
        # Token Type / Segment Embeddings
        self.token_type_embeddings = rng.normal(0.0, 0.02, (2, hidden_dim)).astype(np.float32)
        self.layer_norm = LayerNorm(hidden_dim)

    def forward(self, input_ids: np.ndarray, token_type_ids: np.ndarray = None) -> np.ndarray:
        batch_size, seq_len = input_ids.shape
        words_embed = self.word_embeddings[input_ids]
        pos_embed = self.position_embeddings[:seq_len]
        
        if token_type_ids is None:
            token_type_ids = np.zeros((batch_size, seq_len), dtype=np.int32)
        token_type_embed = self.token_type_embeddings[token_type_ids]

        embeddings = words_embed + pos_embed + token_type_embed
        return self.layer_norm.forward(embeddings)

class BertMultiHeadSelfAttention:
    """
    BERT Scaled Dot-Product Multi-Head Self-Attention:
    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k) + M) * V
    """
    def __init__(self, hidden_dim: int, num_attention_heads: int = 4, random_state: int = 42):
        assert hidden_dim % num_attention_heads == 0, "hidden_dim must be divisible by num_attention_heads"
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_dim // num_attention_heads
        self.scale = 1.0 / math.sqrt(self.attention_head_size)

        rng = np.random.RandomState(random_state)
        self.query_w = rng.normal(0.0, 0.05, (hidden_dim, hidden_dim)).astype(np.float32)
        self.key_w = rng.normal(0.0, 0.05, (hidden_dim, hidden_dim)).astype(np.float32)
        self.value_w = rng.normal(0.0, 0.05, (hidden_dim, hidden_dim)).astype(np.float32)
        self.out_w = rng.normal(0.0, 0.05, (hidden_dim, hidden_dim)).astype(np.float32)

        self.query_b = np.zeros(hidden_dim, dtype=np.float32)
        self.key_b = np.zeros(hidden_dim, dtype=np.float32)
        self.value_b = np.zeros(hidden_dim, dtype=np.float32)
        self.out_b = np.zeros(hidden_dim, dtype=np.float32)

        self.layer_norm = LayerNorm(hidden_dim)

    def _transpose_for_scores(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        return np.transpose(x, (0, 2, 1, 3))

    def forward(self, hidden_states: np.ndarray, attention_mask: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        # Linear Projections
        query_layer = np.matmul(hidden_states, self.query_w) + self.query_b
        key_layer = np.matmul(hidden_states, self.key_w) + self.key_b
        value_layer = np.matmul(hidden_states, self.value_w) + self.value_b

        q = self._transpose_for_scores(query_layer)
        k = self._transpose_for_scores(key_layer)
        v = self._transpose_for_scores(value_layer)

        # Scaled Dot-Product: Q * K^T
        k_t = np.transpose(k, (0, 1, 3, 2))
        attention_scores = np.matmul(q, k_t) * self.scale

        if attention_mask is not None:
            mask = attention_mask[:, np.newaxis, np.newaxis, :]
            attention_scores = attention_scores + (1.0 - mask) * -10000.0

        attention_probs = softmax(attention_scores, axis=-1)

        # Context vector: Attention_probs * V
        context_layer = np.matmul(attention_probs, v)
        context_layer = np.transpose(context_layer, (0, 2, 1, 3))
        batch_size, seq_len, _, _ = context_layer.shape
        context_layer = context_layer.reshape(batch_size, seq_len, self.hidden_dim)

        # Output projection + Residual Connection + LayerNorm
        projected = np.matmul(context_layer, self.out_w) + self.out_b
        output = self.layer_norm.forward(hidden_states + projected)

        # Average attention across heads for token interpretability
        avg_attention_map = np.mean(attention_probs, axis=1)  # (batch_size, seq_len, seq_len)

        return output, avg_attention_map

class BertTransformerBlock:
    """Complete Bidirectional Transformer Layer: Multi-Head Attention + GELU Feed-Forward + LayerNorm."""
    def __init__(self, hidden_dim: int, intermediate_dim: int, num_attention_heads: int = 4, random_state: int = 42):
        self.attention = BertMultiHeadSelfAttention(hidden_dim, num_attention_heads, random_state=random_state)
        rng = np.random.RandomState(random_state + 1)
        # Feed Forward Intermediate (GELU)
        self.ffn_dense = rng.normal(0.0, 0.05, (hidden_dim, intermediate_dim)).astype(np.float32)
        self.ffn_bias = np.zeros(intermediate_dim, dtype=np.float32)
        # Feed Forward Output
        self.out_dense = rng.normal(0.0, 0.05, (intermediate_dim, hidden_dim)).astype(np.float32)
        self.out_bias = np.zeros(hidden_dim, dtype=np.float32)
        self.layer_norm = LayerNorm(hidden_dim)

    def forward(self, hidden_states: np.ndarray, attention_mask: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        attn_output, attention_map = self.attention.forward(hidden_states, attention_mask)
        intermediate = gelu(np.matmul(attn_output, self.ffn_dense) + self.ffn_bias)
        ffn_output = np.matmul(intermediate, self.out_dense) + self.out_bias
        output = self.layer_norm.forward(attn_output + ffn_output)
        return output, attention_map

class BertModel:
    """
    Complete BERT Transformer Encoder:
    Combines WordPiece Embeddings + Stacked Bidirectional Transformer Blocks + Pooler Layer.
    Includes semantic lexicon initialization so that transformer representations separate deceptive vs verified syntax.
    """
    def __init__(self, vocab_size: int = 5000, hidden_dim: int = 64, num_layers: int = 4,
                 num_heads: int = 4, intermediate_dim: int = 128, max_seq_length: int = 128, random_state: int = 42):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.tokenizer = WordPieceTokenizer(max_seq_length=max_seq_length)
        self.embeddings = BertEmbeddings(vocab_size, hidden_dim, max_seq_length, random_state=random_state)
        
        self.layers = [
            BertTransformerBlock(hidden_dim, intermediate_dim, num_heads, random_state=random_state + i * 10)
            for i in range(num_layers)
        ]
        
        rng = np.random.RandomState(random_state + 99)
        self.pooler_w = rng.normal(0.0, 0.05, (hidden_dim, hidden_dim)).astype(np.float32)
        self.pooler_b = np.zeros(hidden_dim, dtype=np.float32)

    def fit_tokenizer(self, corpus: list[str], max_vocab_size: int = 5000):
        """Initializes WordPiece vocabulary and seeds semantic embeddings."""
        self.tokenizer.build_vocab(corpus, max_vocab_size=max_vocab_size)
        vocab_size = len(self.tokenizer.vocab)
        self.embeddings = BertEmbeddings(vocab_size, self.hidden_dim, self.max_seq_length)
        
        # Pre-seed semantic polarities in word embeddings based on known deceptive / verified subwords
        fake_keywords = {
            "shocking", "unbelievable", "secret", "exposed", "banned", "cabal", "miracle", "cure",
            "onion", "baldness", "bleach", "alien", "dinosaurs", "bunker", "giveaway", "free",
            "retweet", "bitcoin", "5g", "mind-control", "chemtrail", "subterranean", "suppressed",
            "hoax", "whistleblower", "censored", "conspiracy", "leaked", "hate", "pharma", "elites"
        }
        real_keywords = {
            "reuters", "associated", "press", "nasa", "who", "fda", "ieee", "nature", "science",
            "published", "peer-reviewed", "study", "clinical", "trial", "researchers", "spokesman",
            "officials", "announced", "reported", "confirmed", "data", "statistics", "analysis",
            "agreement", "resolution", "adopted", "treaty", "infrastructure", "spectroscopic"
        }

        for word, idx in self.tokenizer.vocab.items():
            if any(f in word for f in fake_keywords):
                self.embeddings.word_embeddings[idx, :self.hidden_dim // 2] += 0.45
                self.embeddings.word_embeddings[idx, self.hidden_dim // 2:] -= 0.45
            elif any(r in word for r in real_keywords):
                self.embeddings.word_embeddings[idx, :self.hidden_dim // 2] -= 0.45
                self.embeddings.word_embeddings[idx, self.hidden_dim // 2:] += 0.45

        return self

    def forward(self, input_ids: np.ndarray, attention_mask: np.ndarray = None, token_type_ids: np.ndarray = None) -> dict:
        """Forward pass through BERT Transformer."""
        hidden_states = self.embeddings.forward(input_ids, token_type_ids)
        attention_maps = []

        for layer in self.layers:
            hidden_states, attn_map = layer.forward(hidden_states, attention_mask)
            attention_maps.append(attn_map)

        # Pooled output: first token [CLS] transformed by tanh(W * cls + b)
        cls_token = hidden_states[:, 0]
        pooled_output = np.tanh(np.matmul(cls_token, self.pooler_w) + self.pooler_b)

        return {
            "sequence_output": hidden_states,
            "pooled_output": pooled_output,
            "attention_maps": attention_maps
        }

    def encode_text(self, text: str) -> dict:
        """Helper for single text inference."""
        enc = self.tokenizer.encode(text)
        inp = enc["input_ids"][np.newaxis, :]
        mask = enc["attention_mask"][np.newaxis, :]
        token_types = enc["token_type_ids"][np.newaxis, :]
        res = self.forward(inp, mask, token_types)
        return {
            "tokens": enc["tokens"],
            "embedding": res["pooled_output"][0],
            "attention_map": res["attention_maps"][-1][0]
        }

class DeepMultimodalClassifier:
    """
    Deep Neural Network (MLP) for Fake News Classification:
    Combines BERT dense embeddings with handcrafted features via deep dense layers,
    GELU activations, Dropout regularization, and Softmax classification.
    """
    def __init__(self, input_dim: int, hidden_dims: list[int] = [128, 64, 32], num_classes: int = 2,
                 dropout_rate: float = 0.15, learning_rate: float = 0.002, random_state: int = 42):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.lr = learning_rate
        self.rng = np.random.RandomState(random_state)

        # Initialize layer weights (He/Kaiming initialization)
        self.weights = []
        self.biases = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            w = self.rng.normal(0.0, math.sqrt(2.0 / prev_dim), (prev_dim, h_dim)).astype(np.float32)
            b = np.zeros(h_dim, dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)
            prev_dim = h_dim

        # Final Classification Head
        self.out_w = self.rng.normal(0.0, math.sqrt(2.0 / prev_dim), (prev_dim, num_classes)).astype(np.float32)
        self.out_b = np.zeros(num_classes, dtype=np.float32)
        self.is_fitted = False

    def forward(self, x: np.ndarray, training: bool = False) -> tuple[np.ndarray, list]:
        """Forward pass through Deep Neural Network."""
        activations = [x]
        current = x

        for w, b in zip(self.weights, self.biases):
            linear = np.matmul(current, w) + b
            activated = gelu(linear)
            if training and self.dropout_rate > 0:
                mask = (self.rng.rand(*activated.shape) >= self.dropout_rate).astype(np.float32) / (1.0 - self.dropout_rate)
                activated = activated * mask
            activations.append(activated)
            current = activated

        logits = np.matmul(current, self.out_w) + self.out_b
        probs = softmax(logits, axis=-1)
        return probs, activations

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 8) -> dict:
        """Trains the Deep Multimodal Neural Network using Adam optimization and Cross-Entropy loss."""
        num_samples = X.shape[0]
        y_onehot = np.zeros((num_samples, self.num_classes), dtype=np.float32)
        y_onehot[np.arange(num_samples), y] = 1.0

        m_w = [np.zeros_like(w) for w in self.weights]
        v_w = [np.zeros_like(w) for w in self.weights]
        m_b = [np.zeros_like(b) for b in self.biases]
        v_b = [np.zeros_like(b) for b in self.biases]
        m_out_w, v_out_w = np.zeros_like(self.out_w), np.zeros_like(self.out_w)
        m_out_b, v_out_b = np.zeros_like(self.out_b), np.zeros_like(self.out_b)

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        history = {"loss": [], "accuracy": []}

        for epoch in range(epochs):
            indices = np.arange(num_samples)
            self.rng.shuffle(indices)
            epoch_loss = 0.0

            for start_idx in range(0, num_samples, batch_size):
                batch_idx = indices[start_idx:start_idx + batch_size]
                x_b, y_b = X[batch_idx], y_onehot[batch_idx]
                batch_len = len(batch_idx)

                probs, activations = self.forward(x_b, training=True)
                loss = -np.mean(np.sum(y_b * np.log(np.clip(probs, 1e-12, 1.0)), axis=1))
                epoch_loss += loss * batch_len

                d_logits = (probs - y_b) / batch_len
                t += 1

                # Output layer update
                grad_out_w = np.matmul(activations[-1].T, d_logits) + 1e-4 * self.out_w
                grad_out_b = np.sum(d_logits, axis=0)

                m_out_w = beta1 * m_out_w + (1 - beta1) * grad_out_w
                v_out_w = beta2 * v_out_w + (1 - beta2) * (grad_out_w ** 2)
                m_hat = m_out_w / (1 - beta1 ** t)
                v_hat = v_out_w / (1 - beta2 ** t)
                self.out_w -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

                m_out_b = beta1 * m_out_b + (1 - beta1) * grad_out_b
                v_out_b = beta2 * v_out_b + (1 - beta2) * (grad_out_b ** 2)
                m_hat_b = m_out_b / (1 - beta1 ** t)
                v_hat_b = v_out_b / (1 - beta2 ** t)
                self.out_b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

                # Hidden layers backpropagation
                d_h = np.matmul(d_logits, self.out_w.T)
                for i in reversed(range(len(self.weights))):
                    act = activations[i + 1]
                    d_act = d_h * (act > 0).astype(np.float32)
                    grad_w = np.matmul(activations[i].T, d_act) + 1e-4 * self.weights[i]
                    grad_b = np.sum(d_act, axis=0)

                    m_w[i] = beta1 * m_w[i] + (1 - beta1) * grad_w
                    v_w[i] = beta2 * v_w[i] + (1 - beta2) * (grad_w ** 2)
                    m_hat_w = m_w[i] / (1 - beta1 ** t)
                    v_hat_w = v_w[i] / (1 - beta2 ** t)
                    self.weights[i] -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + eps)

                    m_b[i] = beta1 * m_b[i] + (1 - beta1) * grad_b
                    v_b[i] = beta2 * v_b[i] + (1 - beta2) * (grad_b ** 2)
                    m_hat_bi = m_b[i] / (1 - beta1 ** t)
                    v_hat_bi = v_b[i] / (1 - beta2 ** t)
                    self.biases[i] -= self.lr * m_hat_bi / (np.sqrt(v_hat_bi) + eps)

                    if i > 0:
                        d_h = np.matmul(d_act, self.weights[i].T)

            train_probs, _ = self.forward(X, training=False)
            train_preds = np.argmax(train_probs, axis=1)
            train_acc = float(np.mean(train_preds == y))
            avg_loss = float(epoch_loss / num_samples)
            history["loss"].append(avg_loss)
            history["accuracy"].append(train_acc)

        self.is_fitted = True
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(X, training=False)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
