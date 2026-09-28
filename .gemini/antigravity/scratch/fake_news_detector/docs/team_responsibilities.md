# Team Roles & Responsibilities Matrix

## Project Title
**Hybrid Deep Learning Model for Fake News Detection Using Multi-Source Features and Explainable AI**

**Institution:** Vels Institute of Science, Technology & Advanced Studies (VISTAS), Chennai  
**Department:** Department of Computer Science and Engineering  

---

## 👥 Individual Module Ownership

### 1. Mohamed Irfaan M (Lead NLP & Deep Learning Engineer)
* **NLP Preprocessing:** Text normalization, Unicode sanitization, URL/HTML stripping, and tokenization.
* **BERT Implementation:** WordPiece tokenizer, Sinusoidal Positional Embeddings, and Multi-Head Self-Attention Transformer Layers.
* **Text Feature Extraction:** Extracting 64-dimensional dense contextual embeddings from BERT $[CLS]$ pooler representations.
* **BERT Experiments:** Transformer fine-tuning, loss monitoring, attention map generation, and BERT-only baseline evaluations.

---

### 2. Bhuvanesh M (Lead Data & Ensemble Learning Engineer)
* **Dataset Analysis:** Data ingestion, quality audit, missing-value handling, deduplication, and class balance analysis.
* **User Feature Extraction:** Modeling author presence, institutional credentials, and publisher credibility heuristics.
* **Network Feature Extraction:** Modeling social diffusion metrics, viral share intensities, and engagement anomaly indicators.
* **Feature Fusion:** High-dimensional concatenation of BERT contextual embeddings with multi-source metadata features (79-dim fused space).
* **Random Forest Training:** Hyperparameter optimization (`n_estimators`, `max_depth`, `min_samples_split`, `max_features`) and ensemble classification.

---

### 3. Harish Hariharan MU (Lead Explainable AI & Full-Stack Integration Engineer)
* **SHAP Explainability:** Implementing TreeSHAP feature attributions and global/local game-theoretic importance scoring.
* **Model Evaluation:** Computing Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix generation across all baselines.
* **Web Application:** Developing the backend application endpoints and REST API.
* **Frontend Integration:** Designing responsive UI, real-time prediction forms, color-coded attention heatmaps, and audit logs.
* **Final System Testing:** End-to-end integration testing, user benchmark verification, and API latency validation.

---

## 🤝 Collaborative Responsibilities (All Team Members)
* **Literature Survey & Base Paper Analysis:** Comprehensive review of 5 IEEE Base Papers (2021–2025) and anchor paper by Park & Chai (2023).
* **Automated & Manual Testing:** Test suite creation, pipeline verification, edge case testing.
* **Documentation & Technical Writing:** IEEE conference paper manuscript drafting, viva review guide, and system manuals.
* **Presentation & Review Defense:** Canva slide deck preparation, viva question preparation, and oral project defense.
