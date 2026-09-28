"""
Comprehensive Unit Test Suite for Grammar/POS & Linguistic Anomaly Detection (Phase 10)
Validates all required scenarios:
1. Empty text
2. Normal sentence
3. Text with excessive !!!
4. TEXT WITH EXCESSIVE CAPITALIZATION
5. Repeated words
6. URL-heavy text
7. Text with multiple questions
8. Long sentence
9. Normal news text
"""

import unittest
import numpy as np
import pandas as pd

from src.features.grammar_features import (
    extract_grammar_features,
    extract_single_text_grammar,
    get_pos_tags_for_tokens
)
from src.features.anomaly_features import (
    extract_anomaly_features,
    extract_single_text_anomaly,
    calculate_anomaly_score
)
from src.features.linguistic_features import (
    extract_linguistic_features,
    extract_single_linguistic_record
)


class TestGrammarAndAnomalyModule(unittest.TestCase):

    def test_01_empty_text(self):
        """1. Empty text scenario: Must return zeroed metrics without crashing."""
        df = extract_linguistic_features("")
        self.assertEqual(df.shape, (1, 31))
        self.assertEqual(df["total_word_count"].iloc[0], 0)
        self.assertEqual(df["noun_count"].iloc[0], 0)
        self.assertEqual(df["anomaly_score"].iloc[0], 0.0)
        self.assertFalse(df.isnull().any().any())

    def test_02_normal_sentence(self):
        """2. Normal sentence: Standard syntax with proper POS tags."""
        text = "The government announced new environmental policies."
        df = extract_linguistic_features(text)
        self.assertEqual(df.shape, (1, 31))
        self.assertGreaterEqual(df["noun_count"].iloc[0], 2)
        self.assertGreaterEqual(df["verb_count"].iloc[0], 1)
        self.assertGreaterEqual(df["adjective_count"].iloc[0], 1)
        self.assertLess(df["anomaly_score"].iloc[0], 0.05)
        self.assertFalse(df.isnull().any().any())

    def test_03_excessive_exclamations(self):
        """3. Text with excessive !!!: Must detect exclamation marks & trigger punctuation penalty."""
        text = "ALERT!!! THIS IS SHOCKING NEWS!!!! MUST WATCH NOW!!!!"
        df = extract_linguistic_features(text)
        self.assertEqual(df.shape, (1, 31))
        self.assertGreaterEqual(df["exclamation_count"].iloc[0], 10)
        self.assertGreater(df["anomaly_score"].iloc[0], 0.20)
        self.assertFalse(df.isnull().any().any())

    def test_04_excessive_capitalization(self):
        """4. TEXT WITH EXCESSIVE CAPITALIZATION: Must detect uppercase ratio & word counts."""
        text = "BREAKING NEWS SCIENTISTS DISCOVER REVOLUTIONARY ENERGY SOURCE"
        df = extract_linguistic_features(text)
        self.assertEqual(df.shape, (1, 31))
        self.assertGreater(df["uppercase_word_count"].iloc[0], 5)
        self.assertGreater(df["uppercase_ratio"].iloc[0], 0.80)
        self.assertGreater(df["anomaly_score"].iloc[0], 0.15)
        self.assertFalse(df.isnull().any().any())

    def test_05_repeated_words(self):
        """5. Repeated words: Detects consecutive duplicate terms."""
        text = "Urgent urgent shocking shocking please please share this message immediately!"
        df = extract_linguistic_features(text)
        self.assertEqual(df.shape, (1, 31))
        self.assertGreaterEqual(df["repeated_word_count"].iloc[0], 3)
        self.assertFalse(df.isnull().any().any())

    def test_06_url_heavy_text(self):
        """6. URL-heavy text: Counts links accurately without regex failures."""
        text = "Read the report at https://news.org/item?id=123 and mirror at http://backup-site.com/docs"
        df = extract_linguistic_features(text)
        self.assertEqual(df.shape, (1, 31))
        self.assertGreaterEqual(df["url_count"].iloc[0], 2)
        self.assertFalse(df.isnull().any().any())

    def test_07_multiple_questions(self):
        """7. Text with multiple questions: Detects question mark counts & short sentence fragments."""
        text = "Is this real? What happened? Who is responsible? Why now?"
        df = extract_linguistic_features(text)
        self.assertEqual(df.shape, (1, 31))
        self.assertEqual(df["question_mark_count"].iloc[0], 4)
        self.assertGreaterEqual(df["short_sentence_count"].iloc[0], 3)
        self.assertFalse(df.isnull().any().any())

    def test_08_long_sentence(self):
        """8. Long sentence: Identifies sentences containing more than 35 words."""
        text = (
            "The international consortium of aerospace engineers, university professors, and meteorological researchers "
            "convened in Geneva to formally present their findings regarding atmospheric perturbations, climatological "
            "variability, and statistical anomaly detection over the preceding quarter century across eleven distinct geographic regions."
        )
        df = extract_linguistic_features(text)
        self.assertEqual(df.shape, (1, 31))
        self.assertGreaterEqual(df["long_sentence_count"].iloc[0], 1)
        self.assertGreater(df["total_word_count"].iloc[0], 35)
        self.assertFalse(df.isnull().any().any())

    def test_09_normal_news_text(self):
        """9. Normal news text: Validates full multi-sentence news sample."""
        text = (
            "The university announced the timetable for the upcoming semester examinations through its official website. "
            "Students are advised to verify their registration details and download hall tickets before the deadline."
        )
        df = extract_linguistic_features(text)
        self.assertEqual(df.shape, (1, 31))
        self.assertEqual(df["sentence_count"].iloc[0], 2)
        self.assertGreaterEqual(df["noun_count"].iloc[0], 5)
        self.assertGreaterEqual(df["verb_count"].iloc[0], 2)
        self.assertLess(df["anomaly_score"].iloc[0], 0.05)
        self.assertFalse(df.isnull().any().any())

    def test_10_missing_and_scalar_inputs(self):
        """10. Robustness for None, NaN, and numeric scalar inputs."""
        for val in [None, np.nan, 12345]:
            df = extract_linguistic_features(val)
            self.assertEqual(df.shape, (1, 31))
            self.assertFalse(df.isnull().any().any())


if __name__ == "__main__":
    unittest.main()
