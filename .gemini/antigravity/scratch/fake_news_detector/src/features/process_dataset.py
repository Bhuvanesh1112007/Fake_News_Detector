"""
Dataset Processing Pipeline for Linguistic Feature Extraction (Phase 7)
Reads the real project dataset (data/raw/fake_or_real_news.csv), executes grammar/POS
and linguistic anomaly extraction, and persists the tabular feature matrix to
data/processed/linguistic_features.csv while preserving raw data integrity and sample identifiers.
"""

import os
import pandas as pd
from typing import Optional

from .linguistic_features import extract_linguistic_features


def process_and_save_dataset_linguistic_features(
    input_path: str = "data/raw/fake_or_real_news.csv",
    output_path: str = "data/processed/linguistic_features.csv"
) -> pd.DataFrame:
    """
    Extracts linguistic features from the raw dataset and saves them to a CSV file.

    Args:
        input_path: Path to the raw dataset CSV.
        output_path: Destination path for the extracted feature CSV.

    Returns:
        pd.DataFrame containing metadata (sample_id, title, label, text) and all 31 linguistic features.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input dataset not found at: {input_path}")

    # Load raw dataset
    df_raw = pd.read_csv(input_path)
    print(f"[Phase 7] Ingested raw dataset from: {input_path} (Shape: {df_raw.shape})")

    # Detect text column
    text_col = "text" if "text" in df_raw.columns else df_raw.columns[1]
    title_col = "title" if "title" in df_raw.columns else None
    label_col = "label" if "label" in df_raw.columns else None

    # Combine title and body text for comprehensive linguistic extraction
    full_texts = []
    for _, row in df_raw.iterrows():
        t_val = str(row[title_col]) if title_col and pd.notnull(row[title_col]) else ""
        b_val = str(row[text_col]) if text_col and pd.notnull(row[text_col]) else ""
        combined = f"{t_val}. {b_val}".strip() if t_val and b_val else (b_val or t_val)
        full_texts.append(combined)

    print(f"[Phase 7] Extracting 31 POS and anomaly features across {len(full_texts)} records...")
    linguistic_df = extract_linguistic_features(full_texts, include_text_col=False)

    # Construct clean, unified result DataFrame with stable sample_id
    result_df = pd.DataFrame()
    result_df["sample_id"] = [f"sample_{i:03d}" for i in range(len(df_raw))]
    if title_col:
        result_df["title"] = df_raw[title_col]
    if text_col:
        result_df["text"] = df_raw[text_col]
    if label_col:
        result_df["label"] = df_raw[label_col]

    # Concatenate extracted linguistic features
    result_df = pd.concat([result_df, linguistic_df], axis=1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"[Phase 7] Successfully persisted feature matrix to: {output_path} (Shape: {result_df.shape})")

    return result_df


if __name__ == "__main__":
    process_and_save_dataset_linguistic_features()
