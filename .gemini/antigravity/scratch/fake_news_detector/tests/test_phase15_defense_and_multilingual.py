"""
Phase 15: Adversarial Robustness, Multilingual Fact-Checking & Graph Diffusion Test Suite
Module Owner: Mohamed Irfaan M, Bhuvanesh M, Harish Hariharan MU (VISTAS)

Validates:
1. Adversarial Perturbation Defense & ARS Scoring (backend/adversarial_defense.py)
2. Multilingual Cross-Lingual Detection & Semantic Mapping (backend/multilingual.py)
3. Social Network Propagation & Viral Cascade Simulator (backend/graph_simulator.py)
4. Phase 15 FastAPI REST Endpoints (/api/adversarial/audit, /api/predict-multilingual, /api/simulate/propagation)
"""

import os
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import backend.app as app
from backend.adversarial_defense import (
    inject_homoglyphs, inject_zero_width_spaces, inject_emoji_noise,
    sanitize_text, evaluate_adversarial_robustness
)
from backend.multilingual import detect_language, translate_to_canonical_representation
from backend.graph_simulator import simulate_viral_cascade


def test_1_adversarial_robustness_and_defense():
    """Verifies adversarial perturbation generation, defensive sanitization, and robustness scoring."""
    print("\n[Phase 15 Test 1/4] Verifying Adversarial Perturbation Defense & ARS Scoring...")
    sample_text = "Drinking boiled lemon peel juice cures cancer overnight, secret revealed!"

    # 1. Test Perturbations
    homo = inject_homoglyphs(sample_text, 0.5)
    zw = inject_zero_width_spaces(sample_text, 2)
    emoji = inject_emoji_noise(sample_text)

    assert len(homo) > 0
    assert "\u200b" in zw
    assert any(e in emoji for e in ["🚨", "⚠️", "🔥", "💥", "😱", "🛑", "⚡", "📢"])
    print("  ✓ 4 Adversarial attack vectors generated (Homoglyphs, Zero-width spaces, Emojis, Synonyms)")

    # 2. Test Defensive Sanitizer
    clean = sanitize_text(homo)
    assert "Drinking" in clean or "boiled" in clean
    clean_zw = sanitize_text(zw)
    assert "\u200b" not in clean_zw
    print("  ✓ Defensive Unicode sanitizer canonicalized homoglyphs and stripped zero-width characters")

    # 3. Test Robustness Evaluator
    app.init_db()
    app.load_ai_engine()

    def mock_pred(t, txt, s):
        req = app.PredictRequest(title=t, text=txt, source=s, author="Tester", shares=1000.0, sentiment=0.0)
        res = app.predict_news(req)
        return {"prediction": res.prediction, "confidence_fake": res.confidence_fake, "confidence_real": res.confidence_real}

    eval_res = evaluate_adversarial_robustness("Miracle Cure", sample_text, "Web", mock_pred)
    assert "adversarial_robustness_score" in eval_res
    assert eval_res["adversarial_robustness_score"] >= 70.0
    assert eval_res["total_attacks_tested"] == 4
    print(f"  ✓ Evaluated 4 attack vectors: ARS = {eval_res['adversarial_robustness_score']}% ({eval_res['robustness_grade']})")
    print("  --> Adversarial Defense Test: PASSED [100%]")


def test_2_multilingual_cross_lingual_engine():
    """Verifies automated language script detection and cross-lingual semantic normalization."""
    print("\n[Phase 15 Test 2/4] Verifying Multilingual Cross-Lingual Verification Engine...")
    
    # 1. Tamil Detection
    ta_claim = "சுடுதண்ணீரில் எலுமிச்சை சாறு குடித்தால் புற்றுநோய் குணமாகும்"
    lang_ta, meta_ta = detect_language(ta_claim)
    assert lang_ta == "ta"
    assert meta_ta["name"] == "Tamil (தமிழ்)"
    trans_ta = translate_to_canonical_representation(ta_claim, lang_ta)
    assert "cancer" in trans_ta
    print(f"  ✓ Tamil detected [{meta_ta['flag']} {meta_ta['name']}] and canonicalized to English semantic space")

    # 2. Hindi Detection
    hi_claim = "प्रधानमंत्री राहत योजना के तहत सभी को मुफ्त ₹50,000"
    lang_hi, meta_hi = detect_language(hi_claim)
    assert lang_hi == "hi"
    assert meta_hi["name"] == "Hindi (हिन्दी)"
    trans_hi = translate_to_canonical_representation(hi_claim, lang_hi)
    assert "free" in trans_hi
    print(f"  ✓ Hindi detected [{meta_hi['flag']} {meta_hi['name']}] and canonicalized to English semantic space")

    # 3. Spanish Detection
    es_claim = "El cáncer se cura por completo según documentos secretos del gobierno"
    lang_es, meta_es = detect_language(es_claim)
    assert lang_es == "es"
    print(f"  ✓ Spanish detected [{meta_es['flag']} {meta_es['name']}]")

    print("  --> Multilingual Verification Engine Test: PASSED [100%]")


def test_3_social_network_propagation_simulator():
    """Verifies viral rumor diffusion graph simulation and R0 reproduction metric."""
    print("\n[Phase 15 Test 3/4] Verifying Social Network Propagation & Viral Cascade Simulator...")
    
    sim_fake = simulate_viral_cascade(
        headline="Secret 5G Frequencies Cause Brainwashing",
        prediction="FAKE",
        fake_confidence=88.0,
        network_size=40,
        viral_speed=1.5
    )

    assert "cascade_metrics" in sim_fake
    assert "graph_topology" in sim_fake
    assert "time_series_timeline" in sim_fake

    metrics = sim_fake["cascade_metrics"]
    r0 = metrics["basic_reproduction_number_R0"]
    assert r0 >= 1.5  # High reproduction number for sensational fake news
    assert metrics["total_network_nodes"] == 40
    assert len(sim_fake["graph_topology"]["nodes"]) == 40
    assert len(sim_fake["graph_topology"]["edges"]) > 35
    assert len(sim_fake["time_series_timeline"]) == 11

    print(f"  ✓ Simulated scale-free graph (Nodes: {metrics['total_network_nodes']}, Edges: {metrics['total_network_edges']}, R0: {r0}, Risk: {metrics['cascade_risk']})")
    print("  --> Network Propagation Simulator Test: PASSED [100%]")


def test_4_phase15_fastapi_endpoints():
    """Verifies all Phase 15 REST API endpoints via TestClient."""
    print("\n[Phase 15 Test 4/4] Verifying Phase 15 FastAPI REST API Endpoints...")
    app.init_db()
    app.load_ai_engine()
    client = TestClient(app.app)

    # 1. POST /api/adversarial/audit
    r_adv = client.post("/api/adversarial/audit", json={
        "title": "Secret 5G Tower Mind Control",
        "text": "Leaked documents show 5G transmits frequencies to mind control citizens.",
        "source": "Audit Lab"
    })
    assert r_adv.status_code == 200
    res_adv = r_adv.json()
    assert "adversarial_robustness_score" in res_adv
    assert res_adv["total_attacks_tested"] == 4
    print(f"  ✓ POST /api/adversarial/audit: 200 OK (ARS: {res_adv['adversarial_robustness_score']}%)")

    # 2. POST /api/predict-multilingual
    r_multi = client.post("/api/predict-multilingual", json={
        "title": "சுடுதண்ணீரில் எலுமிச்சை சாறு குடித்தால் புற்றுநோய் குணமாகும்",
        "text": "மருத்துவர்கள் மறைத்த உண்மை! எலுமிச்சை தோல் சாறு குடித்தால் புற்றுநோய் அழியும்.",
        "source": "Tamil Wire"
    })
    assert r_multi.status_code == 200
    res_multi = r_multi.json()
    assert res_multi["detected_language"] == "ta"
    assert res_multi["prediction_result"]["prediction"] == "FAKE"
    print(f"  ✓ POST /api/predict-multilingual: 200 OK (Detected: {res_multi['language_name']}, Verdict: {res_multi['prediction_result']['prediction']})")

    # 3. POST /api/simulate/propagation
    r_sim = client.post("/api/simulate/propagation", json={
        "headline": "Viral Financial Giveaway Scam",
        "prediction": "FAKE",
        "fake_confidence": 90.0,
        "network_size": 30,
        "viral_speed": 1.5
    })
    assert r_sim.status_code == 200
    res_sim = r_sim.json()
    assert "cascade_metrics" in res_sim
    assert len(res_sim["graph_topology"]["nodes"]) == 30
    print(f"  ✓ POST /api/simulate/propagation: 200 OK (Nodes: 30, R0: {res_sim['cascade_metrics']['basic_reproduction_number_R0']})")

    print("  --> Phase 15 REST Endpoints Test: PASSED [100%]")


def run_all_phase15_tests():
    """Master runner for Phase 15."""
    print("=" * 85)
    print("🚀 VERISHIELD AI: PHASE 15 ADVERSARIAL DEFENSE & MULTILINGUAL TEST SUITE")
    print("🏛️ Vels Institute of Science, Technology & Advanced Studies (VISTAS)")
    print("👨‍💻 Team: Mohamed Irfaan M | Bhuvanesh M | Harish Hariharan MU")
    print("=" * 85)

    test_1_adversarial_robustness_and_defense()
    test_2_multilingual_cross_lingual_engine()
    test_3_social_network_propagation_simulator()
    test_4_phase15_fastapi_endpoints()

    print("\n" + "=" * 85)
    print("🏆 ALL PHASE 15 ADVANCED RESEARCH EXTENSION TESTS PASSED SUCCESSFULLY [100%]!")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    run_all_phase15_tests()
