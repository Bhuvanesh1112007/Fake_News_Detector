# VeriShield AI: 79-Dimensional Multimodal Fake News Detection System with BERT & Dual Explainable AI (SHAP)

An interpretable hybrid deep learning framework for fake news detection combining **BERT (Bidirectional Encoder Representations from Transformers)** contextual representations, **79-Dimensional Multimodal Feature Fusion**, persistent database audit logging (**SQLite / Firebase Firestore**), and **Dual Explainable AI (TreeSHAP & BERT Multi-Head Self-Attention)**.

---

## 🏛️ Project & Research Team
* **Institution:** Department of Computer Science and Engineering, Vels Institute of Science, Technology & Advanced Studies (VISTAS), Chennai, India
* **Team Members:**
  * **Mohamed Irfaan M** (`24619122@vistas.ac.in`) — *Lead NLP & Deep Learning Engineer*
  * **Bhuvanesh M** (`24619104@vistas.ac.in`) — *Lead Data & Ensemble Learning Engineer*
  * **Harish Hariharan MU** (`24619115@vistas.ac.in`) — *Lead Explainable AI & Full-Stack Integration Engineer*

---

## 🌟 Key Features & Technical Innovations

1. **BERT Deep Contextual Language Representations (64 Dimensions):**
   - WordPiece subword tokenization with a $5,000$ token vocabulary.
   - 4-layer stacked bidirectional Multi-Head Self-Attention transformer encoder ($h=4, d_k=16$).
   - Scaled dot-product attention: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
   - Mean-pooled sequence representations capturing nuanced polysemy, syntactic dependencies, and semantic context.

2. **Multi-Source Feature Engineering (15 Dimensions):**
   - **User & Publisher Credibility (6-d):** Domain reputation hierarchy, TLD trust rating, author presence, academic/professional credentials, pseudonym detection, and journalistic attribution intensity.
   - **Network Diffusion Dynamics (6-d):** Log social shares, hyper-virality indicators, share velocity acceleration proxies, normalized sentiment polarity, emotional intensity, and engagement anomaly scoring.
   - **Stylometric Linguistic Density (3-d):** Uppercase casing ratio, composite exclamation & sensational trigger density, and Type-Token lexical diversity.

3. **79-Dimensional Multimodal Feature Fusion Tensor:**
   $$\mathbf{x}_{\text{fused}} = [\mathbf{z}_{\text{BERT (64)}} \parallel \mathbf{f}_{\text{user (6)}} \parallel \mathbf{f}_{\text{network (6)}} \parallel \mathbf{f}_{\text{stylo (3)}}] \in \mathbb{R}^{79}$$

4. **150-Tree Random Forest Ensemble Classifier:**
   - Bagging with Gini impurity splitting, balanced class weighting, and Out-of-Bag (OOB) error monitoring ($95.8\%$).
   - Multimodal subspace randomization (`max_features = 0.45`, evaluating $\approx 35$ features per split).
   - Fast inference latency: **$1.42\text{ ms}$ per sample**.

5. **Dual Explainable AI (XAI):**
   - **Global & Local TreeSHAP:** Game-theoretic feature attributions ($\phi_i$) ranking the exact directional impact (+/-) of all 79 feature dimensions.
   - **BERT Token-Level Attention Heatmaps:** Visual highlight badges identifying deceptive keywords in rose and factual institutional citations in emerald green.

6. **FastAPI Server & Glassmorphism Web Dashboard:**
   - Real-time single (`/api/predict`) and bulk batch (`/api/predict-batch`) inference endpoints.
   - Global SHAP (`/api/shap/global`) and comparative benchmark (`/api/metrics`) endpoints.
   - Responsive UI with interactive credibility gauges, Chart.js analytics, and persistent SQLite/Firestore audit history.

---

## 📁 Repository Structure

```
fake_news_detector/
├── backend/
│   ├── app.py                     # FastAPI server & REST endpoints
│   ├── database.py                # SQLite & Firebase persistent audit storage
│   ├── firebase_config.py         # Client Firebase Firestore credentials manager
│   └── schemas.py                 # Pydantic v2 data models & request/response schemas
├── data/
│   ├── raw/
│   │   └── fake_or_real_news.csv  # Curated benchmark dataset of real and fake articles
│   └── processed/
│       ├── train.csv, val.csv, test.csv
│       ├── fused_train_features.npy, fused_val_features.npy, fused_test_features.npy
│       └── preprocessing_stats.json
├── docs/
│   ├── IEEE_Paper_Draft.md        # Publication-ready IEEE conference manuscript
│   ├── Project_Review_and_Viva_Guide.md # Comprehensive 25+ question viva defense guide
│   ├── 5_IEEE_Base_Papers.md      # IEEE literature survey (2021–2025)
│   ├── Fake_News_Detection_Review_Presentation.pptx # 18-slide PowerPoint deck
│   ├── generate_pptx.py           # Presentation deck generator script
│   └── architecture.md            # Detailed system architecture document
├── frontend/
│   ├── index.html                 # Modern glassmorphism single-page UI
│   ├── style.css                  # Responsive styling, gauges & token heatmap badges
│   └── app.js                     # API client, Chart.js visualizer & live controller
├── models/
│   ├── random_forest.pkl          # Trained 150-tree Random Forest model
│   ├── feature_fusion.joblib      # Fitted 79-dimensional multimodal extractor
│   ├── baselines.joblib           # Trained baseline models bundle
│   └── bert_model.joblib          # Persisted WordPiece transformer weights
├── results/
│   └── metrics/
│       ├── metrics.json           # Consolidated benchmark metrics payload
│       ├── model_comparison.csv   # 7-Model comparative evaluation table
│       ├── ablation_study.csv     # 6-Scenario architectural ablation findings
│       └── shap_global_importance.csv # 79-Dimensional global SHAP rankings
├── src/
│   ├── bert_features.py           # WordPiece tokenizer & 4-layer BERT transformer
│   ├── user_features.py           # 6-d User & Publisher credibility extractor
│   ├── network_features.py        # 6-d Network engagement & diffusion extractor
│   ├── feature_fusion.py          # Unified 79-dimensional multimodal fusion tensor
│   ├── train_random_forest.py     # Random Forest training & OOB evaluation pipeline
│   ├── explainability.py          # Unified XAI ModelExplainer interface
│   ├── explainer.py               # TreeSHAP & token self-attention engine
│   ├── evaluation.py              # Baseline benchmarking & ablation study runner
│   ├── preprocessing.py           # Text cleaning, normalization & stat extraction
│   └── config.py                  # Global hyperparameters & path definitions
├── tests/
│   ├── test_webapp_integration.py # Full-stack REST API automated test suite
│   ├── test_extended_benchmarks.py# 10-Case multi-domain real-world test suite
│   ├── diagnose_model.py          # Core diagnostic verification suite
│   ├── test_final_system_validation.py # Phase 12 end-to-end system verification
│   └── test_pipeline.py           # End-to-end pipeline test
├── requirements.txt               # Dependencies
├── run.py                         # One-click startup runner
└── README.md                      # Project documentation
```

---

## 📊 Empirical Evaluation & Benchmark Summary

### 1. Comparative Model Performance (7 Architectures)

| Model Architecture | Feature Space | Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Majority Class (Dummy) | Constant Label | 57.1% | 0.3265 | 0.5714 | 0.4156 | 0.5000 | 58.0% $\pm$ 4.0% |
| Gaussian Naïve Bayes | TF-IDF (2011-d) | 85.7% | 0.8929 | 0.8571 | 0.8571 | 0.8750 | 82.0% $\pm$ 22.3% |
| Decision Tree (CART) | TF-IDF (2011-d) | 100.0% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 96.0% $\pm$ 8.0% |
| Logistic Regression (L2) | TF-IDF (2011-d) | 85.7% | 0.8857 | 0.8571 | 0.8508 | 1.0000 | 92.0% $\pm$ 16.0% |
| Support Vector Machine (RBF) | TF-IDF (2011-d) | 85.7% | 0.8857 | 0.8571 | 0.8508 | 1.0000 | 92.0% $\pm$ 16.0% |
| Gradient Boosting (GBM) | TF-IDF (2011-d) | 100.0% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.0% $\pm$ 0.0% |
| **⭐ Proposed Multimodal RF (Ours)** | **79-Dim Multimodal** | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **96.0% $\pm$ 8.0%** |

---

### 2. Multimodal Architectural Ablation Study (6 Scenarios)

| Modality Configuration | Active Dimensions | Test Accuracy | Test F1-Score | 5-Fold CV Accuracy |
|:---|:---:|:---:|:---:|:---:|
| **Full Multimodal (BERT + User + Network + Stylo)** | **79** | **100.0%** | **1.0000** | **96.0% $\pm$ 8.0%** |
| BERT Contextual Embeddings Only | 64 | 71.4% | 0.7024 | 88.0% $\pm$ 9.8% |
| User & Publisher Signals Only | 6 | 100.0% | 1.0000 | 96.0% $\pm$ 8.0% |
| Network & Social Engagement Only | 6 | 100.0% | 1.0000 | 96.0% $\pm$ 8.0% |
| Stylometric Linguistic Density Only | 3 | 100.0% | 1.0000 | 100.0% $\pm$ 0.0% |
| Tabular Multi-Source Only (No BERT) | 15 | 100.0% | 1.0000 | 100.0% $\pm$ 0.0% |

---

## 🚀 Quickstart & Execution Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline Retraining & Synchronization
```bash
python -c "from src.preprocessing import run_preprocessing_pipeline; run_preprocessing_pipeline(); from src.user_features import run_user_feature_pipeline; run_user_feature_pipeline(); from src.network_features import run_network_feature_pipeline; run_network_feature_pipeline(); from src.feature_fusion import run_feature_fusion_pipeline; run_feature_fusion_pipeline(); from src.train_random_forest import run_random_forest_training_pipeline; run_random_forest_training_pipeline(); from src.evaluation import run_evaluation_pipeline; run_evaluation_pipeline(); from src.explainability import run_explainability_pipeline; run_explainability_pipeline()"
```

### 3. Launch Web Application & REST API
```bash
python run.py
```
* **Interactive Web Dashboard:** [http://localhost:8000](http://localhost:8000)
* **Interactive OpenAPI Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Run Automated Test Suites
```bash
# Phase 12 Final System Verification Test
python -m tests.test_final_system_validation

# Full-Stack REST API Integration Test
python -m tests.test_webapp_integration

# Multi-Domain Real-World Test Benchmarks
python -m tests.test_extended_benchmarks
```
