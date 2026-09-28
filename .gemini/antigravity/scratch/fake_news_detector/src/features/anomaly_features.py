"""
Linguistic Anomaly Detection Module (Phase 4)
Extracts rule-based syntactic, stylistic, and formatting anomaly indicators from news text.

IMPORTANT:
An anomaly does NOT imply that a news item is fake. Anomaly features serve as
supporting linguistic indicators measuring stylistic informality, sensational formatting,
and textual irregularities.
"""

from typing import Union, Dict, List, Any
import re
import numpy as np
import pandas as pd


def get_empty_anomaly_features() -> Dict[str, Union[int, float]]:
    """Returns a zeroed dictionary representing default anomaly features for empty/invalid text."""
    return {
        "exclamation_count": 0,
        "question_mark_count": 0,
        "uppercase_word_count": 0,
        "uppercase_ratio": 0.0,
        "repeated_word_count": 0,
        "special_character_count": 0,
        "special_character_ratio": 0.0,
        "url_count": 0,
        "hashtag_count": 0,
        "mention_count": 0,
        "long_sentence_count": 0,
        "short_sentence_count": 0,
        "whitespace_anomaly": 0,
        "fragmented_sentence_count": 0,
        "anomaly_score": 0.0
    }


def calculate_anomaly_score(
    exclamation_cnt: int,
    question_cnt: int,
    upper_word_cnt: int,
    upper_ratio: float,
    repeated_cnt: int,
    special_cnt: int,
    special_ratio: float,
    url_cnt: int,
    hashtag_cnt: int,
    mention_cnt: int,
    long_sent_cnt: int,
    short_sent_cnt: int,
    whitespace_cnt: int,
    fragment_cnt: int
) -> float:
    """
    Calculates a transparent, deterministic rule-based anomaly score in range [0.0, 1.0].
    
    Formula Breakdown (5 Bounded Sub-Penalties, max 0.20 each):
    ----------------------------------------------------------
    1. Punctuation Intensity (p1): min(0.20, (exclamations + questions) * 0.04)
    2. Capitalization Intensity (p2): min(0.20, uppercase_ratio * 0.50 + uppercase_words * 0.02)
    3. Repetition & Fragmentation (p3): min(0.20, repeated_words * 0.05 + fragments * 0.05)
    4. Special Chars & Hyperlinks (p4): min(0.20, special_ratio * 1.0 + (urls + hashtags + mentions) * 0.03)
    5. Structural Extremes (p5): min(0.20, long_sents * 0.05 + short_sents * 0.03 + whitespace_anomalies * 0.02)

    Total Anomaly Score = min(1.0, p1 + p2 + p3 + p4 + p5)
    """
    p1 = min(0.20, (exclamation_cnt + question_cnt) * 0.04)
    p2 = min(0.20, (upper_ratio * 0.50) + (upper_word_cnt * 0.02))
    p3 = min(0.20, (repeated_cnt * 0.05) + (fragment_cnt * 0.05))
    p4 = min(0.20, (special_ratio * 1.00) + ((url_cnt + hashtag_cnt + mention_cnt) * 0.03))
    p5 = min(0.20, (long_sent_cnt * 0.05) + (short_sent_cnt * 0.03) + (whitespace_cnt * 0.02))

    total_score = p1 + p2 + p3 + p4 + p5
    return float(round(min(1.0, max(0.0, total_score)), 4))


def extract_single_text_anomaly(text: Any) -> Dict[str, Union[int, float]]:
    """
    Extracts 14 numerical anomaly features and calculates a rule-based anomaly score for a single text.

    Args:
        text: Input news article text.

    Returns:
        Dictionary containing 15 anomaly metrics.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return get_empty_anomaly_features()

    raw_text = str(text)
    trimmed_text = raw_text.strip()
    if not trimmed_text:
        return get_empty_anomaly_features()

    total_chars = len(trimmed_text)

    # 1. Punctuation metrics
    exclamation_cnt = trimmed_text.count("!")
    question_cnt = trimmed_text.count("?")

    # 2. Capitalization metrics
    # Identify words with length >= 2 written entirely in uppercase (excluding purely numbers)
    raw_words = re.findall(r'\b[A-Za-z0-9\'-]+\b', trimmed_text)
    alpha_words = [w for w in raw_words if any(c.isalpha() for c in w)]
    uppercase_words = [w for w in alpha_words if len(w) >= 2 and w.isupper()]
    upper_word_cnt = len(uppercase_words)

    total_alpha_chars = sum(1 for c in trimmed_text if c.isalpha())
    total_upper_chars = sum(1 for c in trimmed_text if c.isupper())
    upper_ratio = round(total_upper_chars / max(1, total_alpha_chars), 4) if total_alpha_chars > 0 else 0.0

    # 3. Repeated consecutive words (e.g. "urgent urgent", "shocking shocking")
    repeated_cnt = 0
    for i in range(len(alpha_words) - 1):
        if alpha_words[i].lower() == alpha_words[i + 1].lower():
            repeated_cnt += 1

    # 4. Special characters & Symbols (excluding alphanumeric, spaces, and standard sentence punctuation . , ; : ' " -)
    special_chars = re.findall(r'[^A-Za-z0-9\s.,;:?!\'\"()\-–—]', trimmed_text)
    special_cnt = len(special_chars)
    special_ratio = round(special_cnt / max(1, total_chars), 4)

    # 5. Web & Social Indicators (URLs, Hashtags, Mentions)
    url_matches = re.findall(r'https?://\S+|www\.\S+|bit\.ly/\S+', trimmed_text, re.IGNORECASE)
    url_cnt = len(url_matches)

    hashtag_matches = re.findall(r'#\w+', trimmed_text)
    hashtag_cnt = len(hashtag_matches)

    mention_matches = re.findall(r'@\w+', trimmed_text)
    mention_cnt = len(mention_matches)

    # 6. Sentence structural splits
    raw_sentences = [s.strip() for s in re.split(r'[.!?]+', trimmed_text) if s.strip()]
    long_sent_cnt = 0
    short_sent_cnt = 0
    fragment_cnt = 0

    for sent in raw_sentences:
        sent_words = re.findall(r'\b[A-Za-z0-9\'-]+\b', sent)
        word_len = len(sent_words)
        if word_len > 35:
            long_sent_cnt += 1
        elif 0 < word_len < 5:
            short_sent_cnt += 1

        # Fragment check: very short (< 3 words) or lacking verbs
        if 0 < word_len < 3:
            fragment_cnt += 1

    # 7. Whitespace anomalies (e.g., 3+ consecutive spaces, multiple blank lines)
    whitespace_matches = re.findall(r'[ \t]{3,}|\n{3,}', raw_text)
    whitespace_anomaly = len(whitespace_matches)

    # 8. Compute rule-based anomaly score
    anomaly_score = calculate_anomaly_score(
        exclamation_cnt=exclamation_cnt,
        question_cnt=question_cnt,
        upper_word_cnt=upper_word_cnt,
        upper_ratio=upper_ratio,
        repeated_cnt=repeated_cnt,
        special_cnt=special_cnt,
        special_ratio=special_ratio,
        url_cnt=url_cnt,
        hashtag_cnt=hashtag_cnt,
        mention_cnt=mention_cnt,
        long_sent_cnt=long_sent_cnt,
        short_sent_cnt=short_sent_cnt,
        whitespace_cnt=whitespace_anomaly,
        fragment_cnt=fragment_cnt
    )

    return {
        "exclamation_count": int(exclamation_cnt),
        "question_mark_count": int(question_cnt),
        "uppercase_word_count": int(upper_word_cnt),
        "uppercase_ratio": float(upper_ratio),
        "repeated_word_count": int(repeated_cnt),
        "special_character_count": int(special_cnt),
        "special_character_ratio": float(special_ratio),
        "url_count": int(url_cnt),
        "hashtag_count": int(hashtag_cnt),
        "mention_count": int(mention_cnt),
        "long_sentence_count": int(long_sent_cnt),
        "short_sentence_count": int(short_sent_cnt),
        "whitespace_anomaly": int(whitespace_anomaly),
        "fragmented_sentence_count": int(fragment_cnt),
        "anomaly_score": float(anomaly_score)
    }


def extract_anomaly_features(text: Union[str, List[str], pd.Series]) -> pd.DataFrame:
    """
    Extracts 15 numerical anomaly indicators from raw input text.

    Args:
        text: Single text string, list of strings, or pandas Series.

    Returns:
        pd.DataFrame with 15 numerical anomaly columns.
    """
    if isinstance(text, str):
        feature_dict = extract_single_text_anomaly(text)
        return pd.DataFrame([feature_dict])

    elif isinstance(text, (list, tuple, np.ndarray, pd.Series)):
        records = [extract_single_text_anomaly(t) for t in text]
        return pd.DataFrame(records)

    else:
        feature_dict = extract_single_text_anomaly(str(text))
        return pd.DataFrame([feature_dict])
