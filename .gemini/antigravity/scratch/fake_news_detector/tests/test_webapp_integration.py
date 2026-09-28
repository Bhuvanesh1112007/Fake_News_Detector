"""
Phase 11 Web Application & REST API Automated Integration Test Suite
Module Owner: Harish Hariharan MU (Lead Explainable AI & Full-Stack Integration Engineer)

Verifies:
1. Health check endpoint handler (health_check) and 79-dim model status
2. Single-sample prediction endpoint handler (predict_news) on Real and Fake news
3. High-throughput batch prediction endpoint handler (predict_batch_news)
4. 79-Dimensional global SHAP feature attribution endpoint handler (get_global_shap_importance)
5. Comprehensive metrics and ablation study endpoint handler (get_metrics)
6. Audit history (get_history), aggregate statistics (get_history_statistics), and history reset (clear_history_endpoint)
7. Error handling and input validation (HTTPException 400 Bad Request)
"""
import os
import sys
import json
from fastapi import HTTPException

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app import (
    startup_event, health_check, predict_news, predict_batch_news,
    get_global_shap_importance, get_metrics, get_history,
    get_history_statistics, clear_history_endpoint, get_db_status_endpoint
)
from backend.schemas import (
    PredictRequest, BatchPredictRequest, PredictResponse,
    BatchPredictResponse, GlobalSHAPResponse, MetricsResponse,
    HistoryStatsResponse
)

# Initialize server state
startup_event()


def test_health_endpoint():
    """Verifies health_check endpoint handler status and model metadata."""
    data = health_check()
    assert data["status"] == "online"
    assert data["model_loaded"] is True
    assert data["feature_dimensions"] == 79
    assert "BERT Transformer" in data["model_architecture"]
    print("[OK] Health check endpoint (/api/health) verified (79-dim Multimodal Engine Online).")


def test_single_prediction_real_and_fake():
    """Tests single prediction for Real and Fake articles with XAI attributions."""
    # 1. Real Article
    req_real = PredictRequest(
        title="NASA James Webb Space Telescope Captures Distant Galaxy Cluster",
        text="According to an official press release, astronomers confirmed spectroscopic measurements published in Nature.",
        source="NASA Press Release",
        author="Dr. Sarah Jenkins",
        shares=14000.0,
        sentiment=0.45
    )
    res_real = predict_news(req_real)

    assert isinstance(res_real, PredictResponse)
    assert res_real.prediction == "REAL"
    assert res_real.confidence_real >= 50.0
    assert len(res_real.top_features) >= 5
    assert len(res_real.token_attributions) > 0
    assert len(res_real.key_reasons) >= 2
    assert res_real.inference_latency_ms is not None

    # 2. Fake Article
    req_fake = PredictRequest(
        title="SHOCKING PROOF: Secret 5G Mind Control Frequency Leaked!",
        text="Whistleblower reveals classified 5G mind-control frequency designed to control populations! Share immediately before deleted!",
        source="ConspiracyTruthWire247.net",
        author="PatriotX",
        shares=95000.0,
        sentiment=-0.80
    )
    res_fake = predict_news(req_fake)

    assert isinstance(res_fake, PredictResponse)
    assert res_fake.prediction == "FAKE"
    assert res_fake.confidence_fake >= 50.0
    assert len(res_fake.top_features) >= 5
    assert any(feat.direction == "Pushes towards FAKE" for feat in res_fake.top_features)

    print(f"[OK] Single-article predictions & XAI verified (Real Conf: {res_real.confidence_real}%, Fake Conf: {res_fake.confidence_fake}%).")


def test_batch_prediction_endpoint():
    """Tests batch prediction endpoint (predict_batch_news) and latency tracking."""
    batch_req = BatchPredictRequest(
        items=[
            PredictRequest(
                title="WHO Announces Global Pediatric Immunization Directives",
                text="Clinical trial results published in The Lancet support updated directives.",
                source="The Lancet",
                author="Dr. Halima Traore",
                shares=41000.0,
                sentiment=0.50
            ),
            PredictRequest(
                title="SECRET CURE EXPOSED: Doctors Furious Over Miracle Mineral!",
                text="Big pharma hates this! Drink baking soda potion to cure all illnesses overnight!",
                source="MiracleHealthTruth.org",
                author="Unknown",
                shares=88000.0,
                sentiment=-0.70
            )
        ]
    )
    res_batch = predict_batch_news(batch_req)

    assert isinstance(res_batch, BatchPredictResponse)
    assert res_batch.total_processed == 2
    assert len(res_batch.predictions) == 2
    assert res_batch.total_time_ms > 0
    assert res_batch.avg_latency_ms > 0
    assert res_batch.predictions[0].prediction == "REAL"
    assert res_batch.predictions[1].prediction == "FAKE"
    print(f"[OK] Batch prediction endpoint verified ({res_batch.total_processed} articles in {res_batch.total_time_ms}ms, Avg: {res_batch.avg_latency_ms}ms).")


def test_global_shap_endpoint():
    """Tests 79-dimensional global SHAP endpoint (get_global_shap_importance)."""
    res_shap = get_global_shap_importance()

    assert isinstance(res_shap, GlobalSHAPResponse)
    assert res_shap.total_features == 79
    assert len(res_shap.features) == 79
    assert len(res_shap.modalities) == 4

    expected_modalities = {
        "BERT Contextual Sequence",
        "User & Publisher Credibility",
        "Network & Social Engagement",
        "Stylometric Linguistic Density"
    }
    assert set(res_shap.modalities.keys()) == expected_modalities
    print(f"[OK] Global SHAP endpoint verified (79 features categorized across 4 modalities).")


def test_metrics_endpoint():
    """Tests get_metrics endpoint returning IEEE benchmarks and ablation studies."""
    res_m = get_metrics()

    assert isinstance(res_m, MetricsResponse)
    assert res_m.feature_count == 79
    assert res_m.bert_embedding_dim == 64
    assert res_m.proposed_model is not None
    assert res_m.baselines is not None
    assert res_m.ablation_study is not None
    assert len(res_m.ablation_study) == 6
    assert res_m.model_comparison is not None
    print(f"[OK] Metrics endpoint (/api/metrics) verified (7 Models compared, 6 Ablations returned).")


def test_history_and_stats_endpoints():
    """Tests audit history, statistics calculation, and history clearing."""
    # 1. Fetch history
    hist = get_history()
    assert isinstance(hist, list)

    # 2. Fetch stats
    stats = get_history_statistics()
    assert isinstance(stats, HistoryStatsResponse)
    assert stats.total_predictions >= 0

    # 3. Clear history
    del_data = clear_history_endpoint()
    assert "deleted_count" in del_data

    # 4. Verify empty history
    after_hist = get_history()
    assert len(after_hist) == 0
    print("[OK] Audit history, stats, and deletion endpoints verified.")


def test_invalid_inputs_and_error_handling():
    """Tests API input validation and HTTPException 400 Bad Request responses."""
    # Empty title
    try:
        predict_news(PredictRequest(title="", text="Valid text body."))
        assert False, "Should raise HTTPException for empty title"
    except HTTPException as e:
        assert e.status_code == 400

    # Empty body text
    try:
        predict_news(PredictRequest(title="Valid title", text="   "))
        assert False, "Should raise HTTPException for empty body"
    except HTTPException as e:
        assert e.status_code == 400

    # Empty batch list
    try:
        predict_batch_news(BatchPredictRequest(items=[]))
        assert False, "Should raise HTTPException for empty batch"
    except HTTPException as e:
        assert e.status_code == 400

    print("[OK] Input validation and error handling (HTTPException 400) verified.")


if __name__ == "__main__":
    print("=" * 75)
    print("RUNNING PHASE 11 FULL-STACK REST API INTEGRATION AUTOMATED TEST SUITE")
    print("=" * 75)
    test_health_endpoint()
    test_single_prediction_real_and_fake()
    test_batch_prediction_endpoint()
    test_global_shap_endpoint()
    test_metrics_endpoint()
    test_history_and_stats_endpoints()
    test_invalid_inputs_and_error_handling()
    print("=" * 75)
    print("ALL PHASE 11 FULL-STACK REST API TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)
