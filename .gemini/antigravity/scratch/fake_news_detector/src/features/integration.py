"""
Linguistic Features Integration & Downstream Alignment Module (Phase 12)
Provides standardized merge interfaces and schema contracts for fusing Grammar/POS
and Linguistic Anomaly features with future BERT, User Credibility, and Network Diffusion modalities.
"""

import os
from typing import List, Optional
import pandas as pd
import numpy as np


def get_linguistic_feature_names() -> List[str]:
    """
    Returns the exact ordered list of all 31 linguistic feature column names.
    Used by downstream Feature Fusion and SHAP Explainability layers.
    """
    return [
        # 16 Grammar & POS features
        "noun_count", "pronoun_count", "verb_count", "adverb_count", "adjective_count",
        "proper_noun_count", "conjunction_count", "preposition_count", "determiner_count",
        "auxiliary_count", "modal_count", "numeral_count", "interjection_count",
        "total_word_count", "sentence_count", "average_sentence_length",
        # 15 Anomaly features
        "exclamation_count", "question_mark_count", "uppercase_word_count", "uppercase_ratio",
        "repeated_word_count", "special_character_count", "special_character_ratio",
        "url_count", "hashtag_count", "mention_count", "long_sentence_count",
        "short_sentence_count", "whitespace_anomaly", "fragmented_sentence_count",
        "anomaly_score"
    ]


def load_linguistic_features(
    filepath: str = "data/processed/linguistic_features.csv",
    return_meta: bool = True
) -> pd.DataFrame:
    """
    Loads persisted linguistic features indexed by stable 'sample_id'.

    Args:
        filepath: Path to linguistic_features.csv.
        return_meta: If True, includes metadata columns ('sample_id', 'title', 'text', 'label').
                     If False, returns only the 31 numerical feature columns.

    Returns:
        pd.DataFrame containing the requested features.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Linguistic features file not found at: {filepath}")

    df = pd.read_csv(filepath)
    if "sample_id" not in df.columns:
        df["sample_id"] = [f"sample_{i:03d}" for i in range(len(df))]

    if return_meta:
        return df
    else:
        feat_cols = get_linguistic_feature_names()
        return df[["sample_id"] + [c for c in feat_cols if c in df.columns]]


def merge_multimodal_feature_frames(
    linguistic_df: pd.DataFrame,
    other_feature_df: pd.DataFrame,
    on_key: str = "sample_id",
    how: str = "inner"
) -> pd.DataFrame:
    """
    Horizontally fuses linguistic features with an external modality DataFrame
    (e.g., BERT embeddings, User features, Network features) on a shared sample identifier.

    Args:
        linguistic_df: DataFrame containing linguistic features and the merge key.
        other_feature_df: DataFrame containing another modality (e.g., BERT/User/Network) and the merge key.
        on_key: Column identifier used for join alignment (default: 'sample_id').
        how: Type of merge ('inner', 'left', 'outer').

    Returns:
        Merged pd.DataFrame containing combined feature spaces with verified row integrity.
    """
    if on_key not in linguistic_df.columns:
        raise KeyError(f"Key '{on_key}' missing from linguistic_df columns: {linguistic_df.columns.tolist()}")
    if on_key not in other_feature_df.columns:
        raise KeyError(f"Key '{on_key}' missing from other_feature_df columns: {other_feature_df.columns.tolist()}")

    merged_df = pd.merge(linguistic_df, other_feature_df, on=on_key, how=how)

    # Validate merge integrity
    if len(merged_df) == 0:
        raise ValueError(f"Feature merge produced 0 records on key '{on_key}'. Check ID alignment.")

    return merged_df
