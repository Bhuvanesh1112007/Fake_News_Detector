"""
Grammar and Part-of-Speech (POS) Analysis Module (Phase 3)
Extracts syntactic structures, Part-of-Speech tag distributions, and sentence complexity metrics
using spaCy's pre-trained 'en_core_web_sm' English language model.
"""

from typing import Union, Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import spacy

# Module-level cached spaCy model instance
_NLP_MODEL = None


def get_spacy_model():
    """
    Retrieves or lazily initializes the spaCy English pipeline ('en_core_web_sm').
    Uses singleton caching to avoid reloading the model on every function call.
    """
    global _NLP_MODEL
    if _NLP_MODEL is None:
        try:
            _NLP_MODEL = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback download if not already cached
            from spacy.cli import download
            download("en_core_web_sm")
            _NLP_MODEL = spacy.load("en_core_web_sm")
    return _NLP_MODEL


def get_empty_grammar_features() -> Dict[str, Union[int, float]]:
    """Returns a zeroed dictionary representing default grammar features for empty/invalid text."""
    return {
        "noun_count": 0,
        "pronoun_count": 0,
        "verb_count": 0,
        "adverb_count": 0,
        "adjective_count": 0,
        "proper_noun_count": 0,
        "conjunction_count": 0,
        "preposition_count": 0,
        "determiner_count": 0,
        "auxiliary_count": 0,
        "modal_count": 0,
        "numeral_count": 0,
        "interjection_count": 0,
        "total_word_count": 0,
        "sentence_count": 0,
        "average_sentence_length": 0.0
    }


def extract_single_text_grammar(text: Any, nlp=None) -> Dict[str, Union[int, float]]:
    """
    Performs POS tagging and syntactic extraction on a single raw text string.

    Args:
        text: Input text string (can be str, None, or NaN).
        nlp: Optional pre-loaded spaCy Language model.

    Returns:
        Dictionary containing 16 grammar and syntactic features.
    """
    # Edge case handling for None, NaN, non-string, or empty whitespace
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return get_empty_grammar_features()

    text_str = str(text).strip()
    if not text_str:
        return get_empty_grammar_features()

    if nlp is None:
        nlp = get_spacy_model()

    # Process document with spaCy NLP pipeline
    doc = nlp(text_str)

    # Count tokens based on Universal Dependencies POS tags & PTB tag morphology
    noun_cnt = 0
    pronoun_cnt = 0
    verb_cnt = 0
    adverb_cnt = 0
    adj_cnt = 0
    propn_cnt = 0
    conj_cnt = 0
    prep_cnt = 0
    det_cnt = 0
    aux_cnt = 0
    modal_cnt = 0
    num_cnt = 0
    intj_cnt = 0
    word_cnt = 0

    for token in doc:
        # Ignore purely punctuation or whitespace tokens from word count
        if not token.is_punct and not token.is_space:
            word_cnt += 1

        pos = token.pos_
        tag = token.tag_

        if pos == "NOUN":
            noun_cnt += 1
        elif pos == "PRON":
            pronoun_cnt += 1
        elif pos == "VERB":
            verb_cnt += 1
        elif pos == "ADV":
            adverb_cnt += 1
        elif pos == "ADJ":
            adj_cnt += 1
        elif pos == "PROPN":
            propn_cnt += 1
        elif pos in ("CCONJ", "SCONJ", "CONJ"):
            conj_cnt += 1
        elif pos == "ADP":
            prep_cnt += 1
        elif pos == "DET":
            det_cnt += 1
        elif pos == "AUX":
            aux_cnt += 1
        elif pos == "NUM":
            num_cnt += 1
        elif pos == "INTJ":
            intj_cnt += 1

        # Check for Modal verbs (e.g., could, should, will, might) via PTB tag 'MD'
        if tag == "MD" or (pos == "AUX" and "Modal" in token.morph.get("VerbType", [])):
            modal_cnt += 1

    # Sentence boundary detection
    sentences = [sent for sent in doc.sents if sent.text.strip()]
    sent_cnt = len(sentences)

    # Average sentence length (words per sentence)
    if sent_cnt > 0 and word_cnt > 0:
        avg_sent_len = round(float(word_cnt) / float(sent_cnt), 4)
    else:
        avg_sent_len = 0.0

    return {
        "noun_count": int(noun_cnt),
        "pronoun_count": int(pronoun_cnt),
        "verb_count": int(verb_cnt),
        "adverb_count": int(adverb_cnt),
        "adjective_count": int(adj_cnt),
        "proper_noun_count": int(propn_cnt),
        "conjunction_count": int(conj_cnt),
        "preposition_count": int(prep_cnt),
        "determiner_count": int(det_cnt),
        "auxiliary_count": int(aux_cnt),
        "modal_count": int(modal_cnt),
        "numeral_count": int(num_cnt),
        "interjection_count": int(intj_cnt),
        "total_word_count": int(word_cnt),
        "sentence_count": int(sent_cnt),
        "average_sentence_length": float(avg_sent_len)
    }


def extract_grammar_features(text: Union[str, List[str], pd.Series]) -> pd.DataFrame:
    """
    Extracts 16 grammar and POS features from raw input text.

    Args:
        text: A single text string, list of strings, or pandas Series.

    Returns:
        pd.DataFrame containing 16 numerical grammar feature columns.
    """
    nlp = get_spacy_model()

    if isinstance(text, str):
        feature_dict = extract_single_text_grammar(text, nlp=nlp)
        return pd.DataFrame([feature_dict])

    elif isinstance(text, (list, tuple, np.ndarray, pd.Series)):
        records = [extract_single_text_grammar(t, nlp=nlp) for t in text]
        return pd.DataFrame(records)

    else:
        # Fallback for unexpected types
        feature_dict = extract_single_text_grammar(str(text), nlp=nlp)
        return pd.DataFrame([feature_dict])


def get_pos_tags_for_tokens(text: str) -> List[Tuple[str, str]]:
    """
    Utility function to return token-level POS tags for demonstration and explainability.

    Example Output:
        [('Scientists', 'NOUN'), ('discovered', 'VERB'), ('a', 'DET'), ('new', 'ADJ'), ('method', 'NOUN')]
    """
    if not text or not isinstance(text, str):
        return []
    nlp = get_spacy_model()
    doc = nlp(text.strip())
    return [(token.text, token.pos_) for token in doc if not token.is_space]
