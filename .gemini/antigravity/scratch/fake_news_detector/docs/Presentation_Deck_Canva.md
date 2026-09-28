# Presentation Deck Guide (Canva / PowerPoint Ready)

**Project Title:** Hybrid Deep Learning Model for Fake News Detection Using Multimodal Features and Explainable AI  
**Institution:** Vels Institute of Science, Technology & Advanced Studies (VISTAS)  
**Presented by:** Mohamed Irfaan M, Bhuvanesh M, Harish Hariharan MU  

---

## 🎨 Recommended Canva Design Theme & Style
* **Color Palette:** Deep Navy / Slate (`#0B0F19`), Electric Indigo (`#4F46E5`), Neon Pink/Rose (`#F43F5E`), Emerald (`#10B981`)
* **Typography:** Headers: *Montserrat* / *Plus Jakarta Sans* (Bold); Body: *Inter* / *Roboto*
* **Visual Elements:** Glassmorphism cards, glowing gradient badges, clean comparison tables, and architecture flowcharts.

---

## 🎞️ Slide-by-Slide Content & Speaker Notes

### **Slide 1: Title Slide**
* **Header:** Hybrid Deep Learning Model for Fake News Detection Using Multimodal Features and Explainable AI
* **Sub-header:** A Robust, Interpretable Framework Combining BERT, Feature Fusion & SHAP
* **Presented by:**
  * Mohamed Irfaan M (Lead & AI Model Development)
  * Bhuvanesh M (Data Collection & Backend Integration)
  * Harish Hariharan MU (Frontend UI & Explainable AI)
* **Department:** Department of Computer Science and Engineering, VISTAS
* **Speaker Note:** *"Good morning respected faculty and jury members. Today, our team presents an interpretable hybrid deep learning framework to combat online misinformation using modern transformer embeddings and Explainable AI."*

---

### **Slide 2: Abstract & Key Objectives**
* **Key Points:**
  * Massive surge in viral digital misinformation across social networks.
  * Integration of **BERT** contextual representations + **stylometric/sentiment metadata**.
  * High-accuracy ensemble classification with **Random Forest**.
  * **Explainable AI (SHAP)** providing full transparency and token-level heatmaps for predictions.
* **Speaker Note:** *"Our objective is not just building a high-accuracy classifier, but solving the black-box problem by explaining exactly why an article is flagged."*

---

### **Slide 3: Problem Statement**
* **Key Challenges:**
  1. Traditional detection models rely solely on simplistic keyword frequencies, which fail against modern nuanced misinformation.
  2. Deep learning systems lack interpretability, making it hard for moderators and users to trust model outputs.
  3. High false positive rates due to ignoring metadata such as sentiment polarity and publisher credibility.
* **Speaker Note:** *"Existing systems either analyze text or user metadata in isolation without explaining their decisions to users."*

---

### **Slide 4: Literature Survey & Research Gap (5 IEEE Base Papers: 2021–2024)**
* **Literature Survey Table:**
  1. *[IEEE TKDE 2024]* **Sheng et al.:** Multimodal Feature Fusion (~94.6%) ➔ *Gap: High inference complexity; lacks token-level explainability.*
  2. *[IEEE Access 2023]* **Park & Chai:** XGBoost & ML on Twitter Metadata (~94.1%) ➔ *Gap: No deep contextual language modeling (BERT).*
  3. *[IEEE TCSS 2023]* **Choudhary & Arora:** Linguistic Cues & Attention (~92.4%) ➔ *Gap: Evaluated only offline; lacks live web dashboard.*
  4. *[IEEE TNNLS 2022]* **Akhtar et al.:** Explainable AI via SHAP & LIME (~91.2%) ➔ *Gap: Standalone scripts without real-time REST API deployment.*
  5. *[IEEE TCSS 2021]* **Kaliyar et al.:** BERT Contextual Transformers (~93.8%) ➔ *Gap: Black-box model; ignores auxiliary metadata.*
* **Speaker Note:** *"We surveyed 5 prestigious IEEE papers published strictly between 2021 and 2024, identifying the research gap between deep contextual representation and transparent explainability."*

---

### **Slide 5: Existing System vs. Proposed System**
* **Existing System:**
  * Classical Bag-of-Words / TF-IDF.
  * Opaque decision-making.
  * Restricted to single platform heuristics.
* **Proposed System:**
  * Multi-dimensional feature fusion (BERT Embeddings + Stylometrics + Sentiment).
  * Tuned Random Forest ensemble classifier.
  * Dual-level Explainable AI (Global feature importance + Local token heatmap).
  * Full-stack interactive web verification platform with real-time API.

---

### **Slide 6: Proposed System Architecture & Flow**
* **Pipeline Flowchart:**
  `Input Article -> Text Normalization -> [BERT Embeddings + Metadata Extractors] -> Feature Fusion -> Random Forest Ensemble -> [Prediction Verdict & SHAP Explanations] -> Web Dashboard`
* **Speaker Note:** *"Here is our modular architectural pipeline, showing how feature fusion and SHAP work synchronously in real-time."*

---

### **Slide 7: Experimental Results & Benchmarks**
* **Benchmark Metrics:**
  * Proposed Hybrid Model: **95.8% Accuracy | 0.958 F1-Score | 0.985 ROC-AUC**
  * Gradient Boosting / XGBoost: 94.2% Accuracy
  * Support Vector Machine: 93.4% Accuracy
  * Logistic Regression: 91.8% Accuracy
  * Decision Tree: 89.5% Accuracy
* **Speaker Note:** *"Our hybrid architecture outperforms standard baselines across all statistical benchmarks while maintaining sub-50ms latency."*

---

### **Slide 8: Conclusion & Thank You**
* **Summary of Contributions:**
  * Successful implementation of a transparent, high-precision fake news detector.
  * Demonstrated live on our full-stack web application.
  * Complete research paper prepared for IEEE publication.
* **Footer:** Q&A Session · Thank You!
* **Speaker Note:** *"Thank you for your time and guidance. We now welcome questions and are ready for the live system demonstration."*
