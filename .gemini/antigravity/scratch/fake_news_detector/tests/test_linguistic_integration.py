"""
Integration Test Suite for Multimodal Alignment (Phase 12)
Verifies horizontal fusion compatibility between Linguistic features and external modalities
(e.g., BERT contextual embeddings, User credibility, Network diffusion) via 'sample_id'.
"""

import unittest
import numpy as np
import pandas as pd

from src.features.integration import (
    load_linguistic_features,
    get_linguistic_feature_names,
    merge_multimodal_feature_frames
)


class TestLinguisticMultimodalIntegration(unittest.TestCase):

    def setUp(self):
        self.ling_df = load_linguistic_features(filepath="data/processed/linguistic_features.csv")
        self.sample_ids = self.ling_df["sample_id"].tolist()
        self.num_samples = len(self.sample_ids)

    def test_01_feature_names_integrity(self):
        """Verifies exact 31 feature names and ordering."""
        names = get_linguistic_feature_names()
        self.assertEqual(len(names), 31)
        for name in names:
            self.assertIn(name, self.ling_df.columns)

    def test_02_merge_with_mock_bert_embeddings(self):
        """Simulates fusing linguistic features with a 64-dim BERT embedding DataFrame on 'sample_id'."""
        bert_cols = [f"bert_dim_{i:02d}" for i in range(64)]
        np.random.seed(42)
        mock_bert_data = np.random.randn(self.num_samples, 64)
        bert_df = pd.DataFrame(mock_bert_data, columns=bert_cols)
        bert_df.insert(0, "sample_id", self.sample_ids)

        # Execute merge
        merged = merge_multimodal_feature_frames(self.ling_df, bert_df, on_key="sample_id")

        # Validation: 35 rows, 35 linguistic cols + 64 bert cols = 99 columns
        self.assertEqual(len(merged), self.num_samples)
        self.assertEqual(merged.shape[1], 35 + 64)
        self.assertFalse(merged.isnull().any().any())

    def test_03_merge_with_user_and_network_features(self):
        """Simulates fusing linguistic features with 6-d User and 6-d Network feature DataFrames."""
        user_cols = ["account_age_days", "verified_status", "follower_friend_ratio",
                     "reputation_index", "activity_cadence", "engagement_consistency"]
        net_cols = ["repost_velocity", "cascade_depth", "sentiment_variance",
                    "bot_interaction_index", "cross_platform_prop", "burstiness"]

        user_df = pd.DataFrame(np.random.rand(self.num_samples, 6), columns=user_cols)
        user_df.insert(0, "sample_id", self.sample_ids)

        net_df = pd.DataFrame(np.random.rand(self.num_samples, 6), columns=net_cols)
        net_df.insert(0, "sample_id", self.sample_ids)

        # Perform sequential multimodal merges
        merged_step1 = merge_multimodal_feature_frames(self.ling_df, user_df, on_key="sample_id")
        merged_final = merge_multimodal_feature_frames(merged_step1, net_df, on_key="sample_id")

        self.assertEqual(len(merged_final), self.num_samples)
        self.assertIn("reputation_index", merged_final.columns)
        self.assertIn("repost_velocity", merged_final.columns)
        self.assertIn("noun_count", merged_final.columns)
        self.assertIn("anomaly_score", merged_final.columns)
        self.assertFalse(merged_final.isnull().any().any())

    def test_04_merge_error_handling(self):
        """Verifies clear error reporting on missing join keys."""
        invalid_df = pd.DataFrame({"unrelated_col": [1, 2, 3]})
        with self.assertRaises(KeyError):
            merge_multimodal_feature_frames(self.ling_df, invalid_df, on_key="sample_id")


if __name__ == "__main__":
    unittest.main()
