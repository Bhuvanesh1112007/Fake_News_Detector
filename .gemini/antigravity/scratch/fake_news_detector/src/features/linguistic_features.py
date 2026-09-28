"""
Unified Linguistic Feature Engineering Pipeline (Phase 5)
Combines spaCy-based Grammar/POS Analysis and Rule-Based Linguistic Anomaly Detection
into a unified, high-performance feature extraction interface.
"""

from typing import Union, List, Optional, Any
import numpy as np
import pandas as pd

from .grammar_features import (
    extract_grammar_features,
    extract_single_text_grammar,
    get_spacy_model,
    get_pos_tags_for_tokens
)
from .anomaly_features import (
    extract_anomaly_features,
    extract_single_text_anomaly,
    calculate_anomaly_score
)


def extract_single_linguistic_record(text: Any, nlp=None) -> dict:
    """
    Extracts all 31 grammar and anomaly features for a single text instance.

    Args:
        text: Input raw text string.
        nlp: Optional preloaded spaCy model.

    Returns:
        Unified dictionary with 31 linguistic feature keys.
    """
    grammar_dict = extract_single_text_grammar(text, nlp=nlp)
    anomaly_dict = extract_single_text_anomaly(text)

    # Merge dictionaries cleanly
    combined_record = {}
    combined_record.update(grammar_dict)
    combined_record.update(anomaly_dict)
    return combined_record


def extract_linguistic_features(
    text: Union[str, List[str], pd.Series, pd.DataFrame],
    text_column: str = "text",
    include_text_col: bool = False
) -> pd.DataFrame:
    """
    Extracts unified Part-of-Speech and Linguistic Anomaly features.

    Args:
        text: Single text string, list of strings, pd.Series, or pd.DataFrame.
        text_column: Column name to use if input is a pd.DataFrame (default: 'text').
        include_text_col: Whether to include the original text string as the first column.

    Returns:
        pd.DataFrame containing 31 numerical linguistic feature columns (plus 'text' if requested).
    """
    nlp = get_spacy_model()

    # Case 1: Input is a pandas DataFrame
    if isinstance(text, pd.DataFrame):
        col_to_use = text_column if text_column in text.columns else text.columns[0]
        text_series = text[col_to_use].fillna("")
        records = [extract_single_linguistic_record(t, nlp=nlp) for t in text_series]
        features_df = pd.DataFrame(records, index=text.index)
        if include_text_col:
            features_df.insert(0, "text", text_series)
        return features_df

    # Case 2: Input is a pandas Series
    elif isinstance(text, pd.Series):
        text_series = text.fillna("")
        records = [extract_single_linguistic_record(t, nlp=nlp) for t in text_series]
        features_df = pd.DataFrame(records, index=text.index)
        if include_text_col:
            features_df.insert(0, "text", text_series)
        return features_df

    # Case 3: Input is a list or tuple of strings
    elif isinstance(text, (list, tuple, np.ndarray)):
        records = [extract_single_linguistic_record(t, nlp=nlp) for t in text]
        features_df = pd.DataFrame(records)
        if include_text_col:
            features_df.insert(0, "text", [str(t) if t is not None else "" for t in text])
        return features_df

    # Case 4: Single string or scalar
    else:
        raw_str = "" if (text is None or (isinstance(text, float) and np.isnan(text))) else str(text)
        record = extract_single_linguistic_record(raw_str, nlp=nlp)
        features_df = pd.DataFrame([record])
        if include_text_col:
            features_df.insert(0, "text", [raw_str])
        return features_df
