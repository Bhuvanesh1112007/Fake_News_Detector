"""
Validation & Statistical Profiling Script for Linguistic Features (Phase 8)
"""
import pandas as pd
import spacy

def run_validation():
    df = pd.read_csv("data/processed/linguistic_features.csv")
    num_cols = [c for c in df.columns if c not in ["sample_id", "title", "text", "label"]]

    print("=" * 80)
    print("1. FIRST 5 ROWS OF EXTRACTED LINGUISTIC FEATURES")
    print("=" * 80)
    subset_cols = [
        "sample_id", "label", "noun_count", "verb_count", "adjective_count",
        "total_word_count", "sentence_count", "average_sentence_length",
        "exclamation_count", "uppercase_ratio", "anomaly_score"
    ]
    print(df[subset_cols].head(5).to_string(index=False))

    print("\n" + "=" * 80)
    print(f"2 & 3. GENERATED FEATURE NAMES ({len(num_cols)} Numerical Features)")
    print("=" * 80)
    for i, name in enumerate(num_cols, 1):
        print(f"  [{i:02d}] {name}")

    print("\n" + "=" * 80)
    print("4. MISSING VALUES AUDIT")
    print("=" * 80)
    missing = df[num_cols].isnull().sum()
    print(f"Total NaNs across all columns: {missing.sum()}")
    print(f"All columns 100% complete: {missing.sum() == 0}")

    print("\n" + "=" * 80)
    print("5. BASIC DESCRIPTIVE STATISTICS (df.describe())")
    print("=" * 80)
    desc = df[num_cols].describe().round(3).T
    print(desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].to_string())

    print("\n" + "=" * 80)
    print("6. ACTUAL SPACY POS ANALYSIS FOR 3 NEWS SAMPLES")
    print("=" * 80)
    nlp = spacy.load("en_core_web_sm")
    sample_indices = [0, 1, 20]

    for s_idx in sample_indices:
        title = df.iloc[s_idx]["title"]
        lbl = df.iloc[s_idx]["label"]
        print(f"\n[Sample ID: {df.iloc[s_idx]['sample_id']} | Label: {lbl}]")
        print(f"Text: \"{title}\"")
        print("POS Output:")
        doc = nlp(title)
        for token in doc:
            if not token.is_space and not token.is_punct:
                print(f"  {token.text:<20} -> {token.pos_:<6} ({token.tag_})")

if __name__ == "__main__":
    run_validation()
