"""
Project Configuration & Path Definitions
Manages global hyperparameters, random seeds, file paths, and environment settings.
"""
import os

# Root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, "external")

RAW_DATASET_PATH = os.path.join(RAW_DATA_DIR, "fake_or_real_news.csv")
PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train.csv")
PROCESSED_VAL_PATH = os.path.join(PROCESSED_DATA_DIR, "val.csv")
PROCESSED_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "test.csv")

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
RANDOM_FOREST_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
BERT_MODEL_PATH = os.path.join(MODELS_DIR, "bert_model.joblib")
FEATURE_FUSION_PATH = os.path.join(MODELS_DIR, "feature_fusion.joblib")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
BASELINES_MODEL_PATH = os.path.join(MODELS_DIR, "baselines.joblib")

# Results & Metrics paths
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
SHAP_DIR = os.path.join(RESULTS_DIR, "shap")

METRICS_COMPARISON_PATH = os.path.join(METRICS_DIR, "model_comparison.csv")
METRICS_JSON_PATH = os.path.join(METRICS_DIR, "metrics.json")
CONFUSION_MATRIX_PATH = os.path.join(PLOTS_DIR, "confusion_matrix.png")
ROC_CURVE_PATH = os.path.join(PLOTS_DIR, "roc_curve.png")
MODEL_COMPARISON_PLOT_PATH = os.path.join(PLOTS_DIR, "model_comparison.png")
SHAP_SUMMARY_PLOT_PATH = os.path.join(SHAP_DIR, "shap_summary.png")

# Hyperparameters & Constants
RANDOM_SEED = 42
TEST_SPLIT_SIZE = 0.20
VAL_SPLIT_SIZE = 0.10

# BERT Architecture Settings
BERT_EMBEDDING_DIM = 64
BERT_NUM_LAYERS = 4
BERT_NUM_HEADS = 4
BERT_MAX_SEQ_LENGTH = 128
BERT_VOCAB_SIZE = 5000

# Random Forest Hyperparameters
RF_N_ESTIMATORS = 150
RF_MAX_DEPTH = 12
RF_MIN_SAMPLES_SPLIT = 2
RF_MIN_SAMPLES_LEAF = 1
RF_MAX_FEATURES = 0.45

def ensure_directories():
    """Ensures all required project directories exist."""
    directories = [
        RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
        MODELS_DIR, METRICS_DIR, PLOTS_DIR, SHAP_DIR,
        os.path.join(BASE_DIR, "notebooks"),
        os.path.join(BASE_DIR, "app", "templates"),
        os.path.join(BASE_DIR, "app", "static"),
        os.path.join(BASE_DIR, "tests"),
        os.path.join(BASE_DIR, "docs")
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)
