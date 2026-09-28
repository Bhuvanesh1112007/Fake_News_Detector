"""
Phase 14: Automated Benchmark Reporting & Threat Intelligence Test Suite
Module Owner: Mohamed Irfaan M, Bhuvanesh M, Harish Hariharan MU (VISTAS)

Validates:
1. Automated Forensic PDF/HTML Audit Report Generator (backend/report_generator.py)
2. Live Threat Intelligence Feed & RSS Misinformation Monitor (backend/threat_feed.py)
3. Active Learning Annotation System & Concept Drift Analytics (backend/active_learning.py)
4. Phase 14 FastAPI REST API Endpoints (/api/export-report, /api/threat-stream, /api/feedback, /api/active-learning/stats)
"""

import os
import sys
import json
from unittest.mock import patch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import backend.app as app
from backend.report_generator import generate_audit_html
from backend.threat_feed import fetch_live_threat_feed
from backend.active_learning import submit_fact_checker_feedback, get_active_learning_analytics


def test_1_report_generator():
    """Verifies that generate_audit_html outputs a valid, publication-grade HTML certificate."""
    print("\n[Phase 14 Test 1/4] Verifying Automated Forensic Audit Report Generator...")
    mock_prediction = {
        "id": 101,
        "firebase_id": "fb_doc_101",
        "title": "NASA Confirms Discovery of Water Vapor in Exoplanet Atmosphere",
        "source": "NASA Press",
        "author": "Dr. Sarah Jenkins",
        "prediction": "REAL",
        "confidence_fake": 6.0,
        "confidence_real": 94.0,
        "risk_level": "Low Risk",
        "inference_latency_ms": 11.2,
        "timestamp": "2026-09-27 20:00:00 UTC",
        "top_features": [
            {"feature": "Share Velocity Proxy", "category": "Network", "value": 0.12, "shap_value": 0.24, "direction": "DECREASES_FAKE"},
            {"feature": "BERT Semantic Dim 14", "category": "Linguistic", "value": 0.85, "shap_value": 0.18, "direction": "DECREASES_FAKE"}
        ],
        "token_attributions": [
            {"token": "NASA", "weight": 0.82, "is_suspicious": False},
            {"token": "discovery", "weight": 0.45, "is_suspicious": False}
        ],
        "key_reasons": ["Linguistic markers match verified institutional style."]
    }

    report_html = generate_audit_html(mock_prediction)
    assert "<!DOCTYPE html>" in report_html
    assert "VeriShield AI Forensic Audit Report" in report_html
    assert "NASA Confirms Discovery of Water Vapor" in report_html
    assert "94.0%" in report_html
    assert "Share Velocity Proxy" in report_html
    assert "window.print()" in report_html

    print("  ✓ Forensic HTML report generated with metadata, SHAP table, and print styling")
    print("  --> Report Generator Test: PASSED [100%]")


def test_2_threat_intelligence_feed():
    """Verifies threat feed ingestion, AI evaluation, and Misinformation Threat Index calculation."""
    print("\n[Phase 14 Test 2/4] Verifying Live Threat Intelligence Stream & MTI Calculation...")
    app.init_db()
    app.load_ai_engine()

    feed_result = fetch_live_threat_feed(force_refresh=True)
    assert "threat_index" in feed_result
    assert "threat_status" in feed_result
    assert "feed_items" in feed_result
    assert len(feed_result["feed_items"]) >= 4

    threat_idx = feed_result["threat_index"]
    assert 0.0 <= threat_idx <= 100.0
    print(f"  ✓ Processed {feed_result['total_monitored']} live wire items (Global Threat Index: {threat_idx}/100, Status: {feed_result['threat_status']})")
    print("  --> Threat Intelligence Feed Test: PASSED [100%]")


def test_3_active_learning_and_drift():
    """Verifies human annotation logging and Population Stability Index (PSI) drift analytics."""
    print("\n[Phase 14 Test 3/4] Verifying Active Learning & Concept Drift Analytics...")
    
    # 1. Submit sample annotation
    rec = submit_fact_checker_feedback(
        article_id="test_art_001",
        article_title="Drinking Lemon Water Cures Cancer",
        model_prediction="REAL",
        corrected_label="FAKE",
        notes="Medical consensus explicitly disproves this claim.",
        analyst_name="Senior Fact-Checker (VISTAS)"
    )
    assert rec["article_id"] == "test_art_001"
    assert rec["corrected_label"] == "FAKE"
    print("  ✓ Human fact-checker annotation logged to SQLite & JSON successfully")

    # 2. Compute drift & retrain metrics
    analytics = get_active_learning_analytics()
    assert analytics["total_human_feedback_samples"] >= 1
    assert "concept_drift" in analytics
    assert "population_stability_index" in analytics["concept_drift"]
    assert analytics["concept_drift"]["drift_status"] in ["STABLE", "MODERATE_DRIFT", "HIGH_DRIFT"]
    assert "retrain_readiness_pct" in analytics["active_learning_pipeline"]

    print(f"  ✓ Concept drift computed: PSI = {analytics['concept_drift']['population_stability_index']} ({analytics['concept_drift']['drift_status']})")
    print("  --> Active Learning & Drift Test: PASSED [100%]")


def test_4_fastapi_phase14_endpoints():
    """Verifies all Phase 14 REST API endpoints via TestClient."""
    print("\n[Phase 14 Test 4/4] Verifying Phase 14 FastAPI REST API Endpoints...")
    app.init_db()
    app.load_ai_engine()
    client = TestClient(app.app)

    # 1. POST /api/export-report
    mock_payload = {
        "prediction_data": {
            "id": 999,
            "title": "Astronomers Detect New Pulsar System",
            "prediction": "REAL",
            "confidence_fake": 5.0,
            "confidence_real": 95.0,
            "source": "Astronomy Today",
            "author": "Dr. Miller",
            "top_features": [],
            "token_attributions": [],
            "key_reasons": []
        }
    }
    r_rep = client.post("/api/export-report", json=mock_payload)
    assert r_rep.status_code == 200
    assert "<!DOCTYPE html>" in r_rep.text
    print("  ✓ POST /api/export-report: 200 OK (Rendered Forensic Report HTML)")

    # 2. GET /api/threat-stream
    r_threat = client.get("/api/threat-stream")
    assert r_threat.status_code == 200
    res_threat = r_threat.json()
    assert "threat_index" in res_threat
    assert len(res_threat["feed_items"]) > 0
    print(f"  ✓ GET /api/threat-stream: 200 OK (Threat Index: {res_threat['threat_index']})")

    # 3. POST /api/feedback
    feed_payload = {
        "article_id": "art_test_123",
        "article_title": "Emergency Wealth Transfer Announced",
        "model_prediction": "REAL",
        "corrected_label": "FAKE",
        "notes": "Phishing financial scam.",
        "analyst_name": "Lead Examiner"
    }
    r_feed = client.post("/api/feedback", json=feed_payload)
    assert r_feed.status_code == 200
    assert r_feed.json()["status"] == "success"
    print("  ✓ POST /api/feedback: 200 OK (Active Learning Annotation Recorded)")

    # 4. GET /api/active-learning/stats
    r_stats = client.get("/api/active-learning/stats")
    assert r_stats.status_code == 200
    res_stats = r_stats.json()
    assert "concept_drift" in res_stats
    print(f"  ✓ GET /api/active-learning/stats: 200 OK (Total Feedback: {res_stats['total_human_feedback_samples']})")

    print("  --> Phase 14 REST Endpoints Test: PASSED [100%]")


def run_all_phase14_tests():
    """Master runner for Phase 14."""
    print("=" * 85)
    print("🚀 VERISHIELD AI: PHASE 14 THREAT INTEL, FORENSIC REPORTS & DRIFT TEST SUITE")
    print("🏛️ Vels Institute of Science, Technology & Advanced Studies (VISTAS)")
    print("👨‍💻 Team: Mohamed Irfaan M | Bhuvanesh M | Harish Hariharan MU")
    print("=" * 85)

    test_1_report_generator()
    test_2_threat_intelligence_feed()
    test_3_active_learning_and_drift()
    test_4_fastapi_phase14_endpoints()

    print("\n" + "=" * 85)
    print("🏆 ALL PHASE 14 THREAT INTEL & FORENSIC REPORT TESTS PASSED SUCCESSFULLY [100%]!")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    run_all_phase14_tests()
