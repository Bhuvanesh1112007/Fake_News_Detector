# System Architecture Document (Phase 0)

## Project Title
**Hybrid Deep Learning Model for Fake News Detection Using Multi-Source Features and Explainable AI**

---

## 1. High-Level System Architecture

```
                      INPUT NEWS ARTICLE / TEXT
                                  │
                                  ▼
                        DATA PREPROCESSING
               (Sanitization, Tokenization, Normalization)
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
           TEXT MODULE                    MULTI-SOURCE MODULE
  (WordPiece Subword Tokenizer)        (Stylometrics & Sensationalism)
                 │                                 │
                 ▼                                 ▼
     BERT TRANSFORMER ENCODER             LINGUISTIC & METADATA
 (Multi-Head Self-Attention Layers)   (Attribution, Shares, Polarity)
                 │                                 │
                 ▼                                 ▼
       BERT DENSE EMBEDDINGS             MULTI-SOURCE VECTORS
            (64-dim)                           (15-dim)
                 │                                 │
                 └────────────────┬────────────────┘
                                  ▼
                         FEATURE FUSION LAYER
                    (Concatenation Vector: 79-dim)
                                  │
                                  ▼
                       RANDOM FOREST CLASSIFIER
                        (150 Decision Trees)
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
        PREDICTION VERDICT                 SHAP EXPLAINABILITY
       (FAKE / REAL + Conf %)           (Global SHAP + Token Attention)
                 │                                 │
                 └────────────────┬────────────────┘
                                  ▼
                       INTERACTIVE WEB DASHBOARD
              (Real-Time Analysis, Metrics, Audit Log)
```

---

## 2. Component Specifications

### A. Data Preprocessing Pipeline
* **Text Normalization:** Unicode NFKD decomposition, URL/HTML tag stripping, whitespace collapse.
* **Linguistic Metric Extraction:** Uppercase character ratio, exclamation mark frequency, question mark frequency, lexical diversity, sensational clickbait trigger counts.
* **Label Encoding:** Binary target vector ($0 = \text{REAL}, 1 = \text{FAKE}$).

### B. BERT Contextual Language Module
* **WordPiece Tokenizer:** 5,000 subword vocabulary with special tokens (`[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`).
* **Embeddings:** Token Embeddings + Sinusoidal Positional Embeddings + Segment Embeddings with LayerNorm.
* **Encoder:** 4 Bidirectional Transformer Layers with 4-Head Scaled Dot-Product Attention:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
* **Pooling:** $\tanh(W_{\text{pool}} \cdot [CLS] + b)$ generating 64-dimensional dense contextual representations.

### C. Multi-Source Feature Engineering Module
* **Deception & Hoax Lexical Density:** Quantifies frequency of miracle claims, conspiracies, WhatsApp forward scams, and impossible assertions.
* **Journalistic & Institutional Density:** Quantifies peer-reviewed citations, official spokespersons, municipal notices, and empirical reporting markers.
* **Stylometric & Emotion Signals:** Capitalization intensity, punctuation density, sentiment polarity, and emotional arousal scores.
* **Publisher Credibility:** Source domain reputation weighting.

### D. Feature Fusion Layer
* Concatenates the 64-dimensional BERT vector with the 15-dimensional multi-source feature vector into a unified 79-dimensional feature representation:
  $$\mathbf{x}_{\text{fused}} = \left[\mathbf{z}_{\text{BERT}} \,\|\, \mathbf{f}_{\text{multi-source}}\right] \in \mathbb{R}^{79}$$

### E. Random Forest Classification Engine
* Ensemble of 150 decorrelated decision trees with Gini impurity splitting criteria.
* Evaluates out-of-bag error and outputs class probabilities $P(\text{FAKE})$ and $P(\text{REAL})$.

### F. Dual Explainable AI (XAI) Engine
* **SHAP Tree Explainer:** Computes exact Shapley feature attributions:
  $$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} (v(S \cup \{i\}) - v(S))$$
* **BERT Self-Attention Token Heatmap:** Combines last-layer $[CLS]$ attention weights with local word attribution scores for interactive, color-coded visual transparency.

---

## 3. Academic Distinction: Existing vs. Proposed Framework

| Dimension | Existing System (Park & Chai, 2023) | Proposed Framework (Our Implementation) |
| :--- | :--- | :--- |
| **Language Modeling** | Traditional Bag-of-Words / TF-IDF / Metadata only | **BERT Contextual Subword Transformer Embeddings** |
| **Feature Set** | Social metadata (user followers, retweet counts) | **Multi-Source Fusion (BERT Text + Stylometrics + Sensationalism + Metadata)** |
| **Classifier** | XGBoost, SVM, Decision Tree, Logistic Regression | **Tuned Random Forest on Deeply Fused Contextual Vectors** |
| **Explainability** | Feature importance ranking only (Black-box) | **Dual XAI: Global Tree SHAP + Word-Level BERT Attention Heatmaps** |
| **Deployment** | Offline empirical study | **Live Interactive Web Application + REST API** |
