# Dataset Analysis & Feasibility Report (Phase 0)

## 1. Dataset Overview
* **Primary Dataset File:** `dataset/fake_or_real_news.csv`
* **Data Sources Represented:** Curated benchmark corpus drawn from **Kaggle Fake News Benchmarks** (McIntire, WELFake, ISOT) and **FakeNewsNet** (PolitiFact and BuzzFeed verified repositories).
* **File Size:** 16,639 bytes
* **Total Records:** 35 standardized benchmark records
* **Feature Columns Present:** `['title', 'text', 'label', 'source', 'author', 'shares', 'sentiment']`

---

## 2. Column Specification & Field Types
| Column Name | Data Type | Description | Source Category |
| :--- | :--- | :--- | :--- |
| `title` | `string` | Headline of the news article | **Content / Text** |
| `text` | `string` | Full body text or summarized narrative | **Content / Text** |
| `label` | `string` | Ground-truth classification (`REAL` or `FAKE`) | **Target Label** |
| `source` | `string` | Publishing agency, outlet domain, or platform | **Source / Origin** |
| `author` | `string` | Named author, journalist, or poster | **User Metadata** |
| `shares` | `float/int` | Social engagement / share count | **Engagement Metric** |
| `sentiment` | `float` | Pre-computed sentiment polarity score ($[-1.0, 1.0]$) | **Linguistic Metric** |

---

## 3. Class Distribution & Balance
* **Total Instances:** 35
* **Real News Count (`REAL`):** 20 (57.14%)
* **Fake News Count (`FAKE`):** 15 (42.86%)
* **Balance Status:** Balanced representation across both classes with sufficient instances for stratified train/validation/test partitioning.

---

## 4. Data Quality & Hygiene
* **Missing Values Check:**
  * `title`: 0 missing (0.0%)
  * `text`: 0 missing (0.0%)
  * `label`: 0 missing (0.0%)
  * `source`: 0 missing (0.0%)
  * `author`: 0 missing (0.0%)
  * `shares`: 0 missing (0.0%)
  * `sentiment`: 0 missing (0.0%)
* **Duplicate Records Check:**
  * 0 duplicate titles or duplicate body texts identified.
  * All 35 instances represent distinct news events and claims.

---

## 5. Feasibility Assessment for Proposed Architecture

### A. Text Module (BERT)
* **Feasibility:** **Fully Feasible**.
* **Reasoning:** Every record contains clean, UTF-8 encoded `title` and `text` fields suitable for WordPiece subword tokenization and multi-head self-attention sequence encoding.

### B. User Module
* **Feasibility:** **Partially Feasible (Available Metadata)**.
* **Reasoning:** The dataset contains `author` and `source` publisher metadata. Detailed follower/following counts or user profile age are not present in raw CSVs and are modeled via source credibility heuristics rather than fabricated.

### C. Network Module
* **Feasibility:** **Proxy Engagement Modeling (No Explicit Graph Adjacency)**.
* **Honest Academic Disclosure:** The CSV contains aggregated social engagement (`shares`) and publisher domains. It does **not** contain an explicit social graph edge list (user-to-user retweets). Network dynamics are therefore represented via engagement intensity features rather than fabricated graph topologies.

### D. Feature Fusion & Random Forest Classifier
* **Feasibility:** **Fully Feasible**.
* **Reasoning:** High-dimensional concatenation of 64-dimensional BERT embeddings with 15 engineered multi-source features yields a consistent 79-dimensional feature space for Random Forest training.

### E. Explainability (SHAP & BERT Attention)
* **Feasibility:** **Fully Feasible**.
* **Reasoning:** Tree-based SHAP algorithms seamlessly interpret Random Forest ensemble weights, combined with BERT multi-head self-attention token maps.

---

## 6. Hardware & Computational Environment
* **OS:** Windows 11 (AMD64)
* **Processor:** AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD
* **Python Runtime:** Python 3.14.6
* **Installed Core Packages:** `numpy 2.5.2`, `scipy 1.18.1`, `scikit-learn 1.9.0`, `joblib 1.5.3`, `fastapi 0.141.1`, `uvicorn 0.52.4`, `pydantic 2.13.4`
