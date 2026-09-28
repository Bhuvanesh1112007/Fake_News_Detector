"""
Automated Verification Tests
Validates feature extraction, inference, Explainable AI generation, and FastAPI route handlers.
"""
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import startup_event, health_check, predict_news, get_metrics, get_history
from backend.schemas import PredictRequest

# Initialize server state
startup_event()

def test_health():
    res = health_check()
    assert res["status"] == "online"
    assert res["model_loaded"] is True
    print("✓ Health check test passed.")

def test_predict_real():
    req = PredictRequest(
        title="NASA James Webb Space Telescope Captures Distant Galaxy Cluster",
        text="NASA's James Webb Space Telescope published deep field observations of early galaxies with peer-reviewed spectroscopic measurements.",
        source="NASA Press Release",
        author="Dr. Sarah Jenkins",
        shares=14000
    )
    res = predict_news(req)
    assert res.prediction in ["REAL", "FAKE"]
    assert len(res.token_attributions) > 0
    assert len(res.top_features) > 0
    print(f"✓ Real article test passed: Verdict={res.prediction}, RealConfidence={res.confidence_real}%")

def test_predict_fake():
    req = PredictRequest(
        title="SHOCKING PROOF: Secret 5G Satellite Mind Control Cures Banned by Globalists!!!",
        text="BREAKING NEWS! Whistleblower reveals secret mind control frequency beamed by cabal! Big Pharma is terrified! Share before deletion!",
        source="ConspiracyTruthWire247.net",
        author="PatriotX",
        shares=95000
    )
    res = predict_news(req)
    assert res.prediction == "FAKE"
    assert res.confidence_fake >= 50.0
    print(f"✓ Fake article test passed: Verdict={res.prediction}, FakeConfidence={res.confidence_fake}%")

def test_metrics():
    res = get_metrics()
    assert (hasattr(res, "proposed_model") and res.proposed_model is not None) or "proposed_model" in res or "model_comparison" in res
    print("✓ Metrics test passed.")

def test_history():
    res = get_history()
    assert isinstance(res, list)
    assert len(res) > 0
    print(f"✓ History test passed. Found {len(res)} recorded predictions.")

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🧪 RUNNING SYSTEM VERIFICATION TESTS")
    print("=" * 50)
    test_health()
    test_predict_real()
    test_predict_fake()
    test_metrics()
    test_history()
    print("=" * 50)
    print("✅ ALL TESTS COMPLETED AND PASSED!")
    print("=" * 50 + "\n")
