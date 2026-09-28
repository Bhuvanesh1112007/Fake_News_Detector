# Requirements Specification Document (Phase 0)

## Project Title
**Hybrid Deep Learning Model for Fake News Detection Using Multi-Source Features and Explainable AI**

---

## 1. Project Overview & Objective
This project is an academic final-year investigation conducted by a 3-person student team at Vels Institute of Science, Technology & Advanced Studies (VISTAS). The objective is to design, implement, and rigorously evaluate an end-to-end fake news detection system that:
1. Accepts raw news text and associated article metadata.
2. Cleans and normalizes text and tabular features using an identical pipeline across training and inference.
3. Uses BERT (Bidirectional Encoder Representations from Transformers) to obtain deep contextual semantic embeddings.
4. Extracts available user-related features (author credibility, publication source).
5. Extracts available social/network engagement metrics (e.g., shares, engagement counts, sentiment).
6. Combines the textual and multi-source tabular features into a unified fused feature representation.
7. Classifies articles using a tuned **Random Forest** ensemble classifier.
8. Explains predictions transparently using **SHAP (Shapley Additive exPlanations)** with global feature importance and individual token-level attributions.
9. Evaluates baseline models against the proposed hybrid approach using standard academic metrics: **Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrices**.
10. Deploys an interactive web interface allowing users to input text, receive Real/Fake predictions with confidence scores, and view explainable AI rationale.

---

## 2. Academic Foundation & Literature Survey
* **Primary Anchor Paper:**  
  *Park, S.-H., & Chai, J. (2023). "Constructing a User-Centered Fake News Detection Model by Using Classification Algorithms in Machine Learning Techniques." IEEE Access, Vol. 11, pp. 24890–24902.*
  * *Original Paper Methodology:* Evaluated content, user, and network features using XGBoost for feature selection and compared Logistic Regression, Random Forest, SVM, CART (Decision Trees), and Neural Networks on social metadata.
* **Secondary Literature (2021–2025):**
  * *Sheng et al. (IEEE TKDE, 2024):* Multi-domain and fused contextual representations.
  * *Choudhary & Arora (IEEE TCSS, 2023):* Linguistic indicators and multi-head attention.
  * *Akhtar et al. (IEEE TNNLS, 2022):* Game-theoretic post-hoc explainability with SHAP/LIME.
  * *Kaliyar et al. (IEEE TCSS, 2021):* Deep contextual transformer representations (BERT).

---

## 3. Functional Requirements
* **FR-1: Data Ingestion & Preprocessing:** Load, clean, normalize, and split dataset into stratified Train, Validation, and Test sets without data leakage.
* **FR-2: BERT Contextual Feature Extraction:** Extract fixed-length dense embeddings from BERT transformer encoders.
* **FR-3: Multi-Source Feature Extraction:** Extract stylometric, deception density, journalistic attribution, sentiment polarity, and metadata statistics.
* **FR-4: Feature Fusion:** Concatenate BERT dense contextual embeddings with scaled multi-source features into a consistent feature tensor.
* **FR-5: Model Training & Tuning:** Train baseline models (Logistic Regression, Decision Tree, SVM, Naive Bayes, Gradient Boosting, BERT-only) and the proposed Hybrid Random Forest model on the fused representations.
* **FR-6: Explainable AI Engine:** Compute global and local feature importance using SHAP algorithms aligned with BERT self-attention token heatmaps.
* **FR-7: Interactive Web Dashboard:** Provide an accessible, responsive web interface for live analysis, benchmark metrics visualization, and prediction history.

---

## 4. Non-Functional & Reproducibility Requirements
* **NFR-1 (Academic Integrity):** Zero fabricated metrics. All performance metrics must be derived directly from real test-set experiments with recorded random seeds.
* **NFR-2 (Latency):** Sub-100ms inference time per news article.
* **NFR-3 (Consistency):** Strict alignment between training feature dimensions and runtime inference feature dimensions.
* **NFR-4 (Reproducibility):** Documented random seeds (`random_state=42`), fixed stratified splits, and pinned library versions.

---

## 5. Technology Stack & Environment
* **Language:** Python 3.10+ (Current runtime: Python 3.14.6 AMD64 on Windows 11)
* **Libraries:** `numpy`, `scipy`, `scikit-learn`, `joblib`, `fastapi`/`flask`, `uvicorn`, `pydantic`
* **Explainability:** SHAP-based Tree Explainer / Game-theoretic feature attribution
* **Storage:** SQLite (prediction audit history)
* **Frontend:** HTML5, CSS3, Vanilla JavaScript
