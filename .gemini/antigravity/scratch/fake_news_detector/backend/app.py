"""
FastAPI Server for Fake News Detection System (Phase 11)
Module Owner: Harish Hariharan MU (Lead Explainable AI & Full-Stack Integration Engineer)

Provides high-performance REST API endpoints:
- /api/health: System health, 79-dim model status, and uptime
- /api/predict: Real-time 79-dim multimodal inference + dual XAI (SHAP & BERT attention)
- /api/predict-batch: Batch multi-article inference with throughput profiling
- /api/metrics: 7-Model comparative benchmarks and 6-scenario ablation findings
- /api/shap/global: 79-dimensional global SHAP feature importance by modality
- /api/history: Audit trail of past analyses
- /api/history/stats: Aggregate real/fake audit proportions
- DELETE /api/history: Clears audit logs
"""
import os
import sys
import json
import time
from datetime import datetime

try:
    import joblib
except ImportError:
    import pickle as joblib

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from .schemas import (
    PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse,
    URLPredictRequest,
    HistoryItem, HistoryStatsResponse, MetricsResponse, DatabaseStatusResponse,
    GlobalSHAPResponse, GlobalSHAPItem,
    ReportExportRequest, ReportExportResponse, FeedbackRequest, FeedbackResponse,
    ThreatFeedResponse, ActiveLearningStatsResponse,
    AdversarialAuditRequest, AdversarialAuditResponse,
    MultilingualPredictRequest, MultilingualPredictResponse,
    PropagationSimRequest, PropagationSimResponse
)
from .url_extractor import extract_article_from_url
from .report_generator import generate_audit_html
from .threat_feed import fetch_live_threat_feed
from .active_learning import submit_fact_checker_feedback, get_active_learning_analytics
from .adversarial_defense import evaluate_adversarial_robustness, sanitize_text
from .multilingual import detect_language, translate_to_canonical_representation
from .graph_simulator import simulate_viral_cascade
from .database import (
    init_db, save_prediction, get_recent_history, get_history_stats,
    clear_prediction_history, get_database_status
)
from .firebase_config import get_firebase_config

from src.feature_fusion import MultiSourceFeatureFusion, run_feature_fusion_pipeline
from src.train_random_forest import ProposedRandomForestClassifier, run_random_forest_training_pipeline
from src.explainability import ModelExplainer, run_explainability_pipeline
from src.config import (
    RANDOM_FOREST_PATH, FEATURE_FUSION_PATH, METRICS_JSON_PATH,
    METRICS_DIR
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = FastAPI(
    title="VeriShield AI: Multimodal Deep Learning Fake News Detection System",
    description="79-Dimensional Multimodal Feature Fusion, BERT Deep Contextual Embeddings, Random Forest Ensemble & Dual XAI Engine",
    version="2.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model & explainer state
FUSION_EXTRACTOR = None
RF_MODEL = None
EXPLAINER = None
METRICS_PAYLOAD = None


def load_ai_engine():
    """Initializes and caches the complete 79-dimensional AI inference and XAI engine."""
    global FUSION_EXTRACTOR, RF_MODEL, EXPLAINER, METRICS_PAYLOAD

    # 1. Feature Fusion
    if not os.path.exists(FEATURE_FUSION_PATH):
        print("Feature fusion artifact missing. Training feature fusion pipeline...")
        FUSION_EXTRACTOR = run_feature_fusion_pipeline()
    else:
        FUSION_EXTRACTOR = MultiSourceFeatureFusion.load(FEATURE_FUSION_PATH)

    # 2. Random Forest Model
    if not os.path.exists(RANDOM_FOREST_PATH):
        print("Random Forest model missing. Training Random Forest classifier...")
        RF_MODEL, _ = run_random_forest_training_pipeline()
    else:
        RF_MODEL = ProposedRandomForestClassifier.load(RANDOM_FOREST_PATH)

    # 3. Model Explainer
    EXPLAINER = ModelExplainer(model=RF_MODEL, feature_names=FUSION_EXTRACTOR.get_feature_names())

    # 4. Metrics Payload
    if os.path.exists(METRICS_JSON_PATH):
        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            METRICS_PAYLOAD = json.load(f)

    print(f"🚀 VeriShield AI 79-Dimensional Multimodal Engine & XAI Explainer loaded successfully ({len(FUSION_EXTRACTOR.get_feature_names())} features).")


@app.on_event("startup")
def startup_event():
    init_db()
    load_ai_engine()


@app.get("/api/health")
def health_check():
    """Health check endpoint confirming AI pipeline and database operational status."""
    return {
        "status": "online",
        "model_loaded": RF_MODEL is not None and FUSION_EXTRACTOR is not None,
        "model_architecture": "79-Dim Multimodal Feature Fusion + 150-Tree Random Forest + BERT Transformer",
        "feature_dimensions": len(FUSION_EXTRACTOR.get_feature_names()) if FUSION_EXTRACTOR else 79,
        "database": "SQLite Persistent Storage / Firebase Sync",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/firebase/config")
def get_client_firebase_config():
    """Returns client Firebase configuration."""
    return get_firebase_config()


@app.get("/api/database/status", response_model=DatabaseStatusResponse)
def get_db_status_endpoint():
    """Returns database status and total record count."""
    return get_database_status()


@app.post("/api/predict", response_model=PredictResponse)
def predict_news(payload: PredictRequest):
    """
    Performs real-time single-article inference over 79-dimensional multimodal feature space:
    1. Extracts 64-d BERT sequence representation + token-level self-attention weights.
    2. Extracts 6-d User/Publisher credibility features.
    3. Extracts 6-d Network/Social engagement features.
    4. Extracts 3-d Stylometric linguistic features.
    5. Infers calibrated class probability using 150-tree Random Forest ensemble.
    6. Generates signed local SHAP attributions, token heatmaps, and evidence rationales.
    7. Persists prediction event to database.
    """
    if FUSION_EXTRACTOR is None or RF_MODEL is None or EXPLAINER is None:
        raise HTTPException(status_code=503, detail="AI engine is not initialized.")

    if not payload.title.strip() or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Both title and article body text are required.")

    meta = {
        "source": payload.source or "General Web",
        "author": payload.author or "Unknown",
        "shares": float(payload.shares if payload.shares is not None else 1000.0),
        "sentiment": float(payload.sentiment if payload.sentiment is not None else 0.0)
    }

    t0 = time.perf_counter()

    # 1. 79-Dimensional Multimodal Encoding + BERT Self-Attention
    fused_vec, bert_tokens, bert_attention = FUSION_EXTRACTOR.encode_single_with_attention(
        payload.title, payload.text, meta
    )

    # 2. Random Forest Classification
    probs = RF_MODEL.predict_proba(fused_vec.reshape(1, -1))[0]
    fake_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
    real_prob = float(probs[0]) if len(probs) > 1 else (1.0 - fake_prob)
    prediction_label = "FAKE" if fake_prob >= 0.50 else "REAL"

    # Risk level categorization
    if fake_prob >= 0.75:
        risk_level = "High Misinformation Risk"
    elif fake_prob >= 0.50:
        risk_level = "Moderate Suspicion"
    elif fake_prob >= 0.25:
        risk_level = "Likely Authentic"
    else:
        risk_level = "High Credibility Verified"

    # 3. Explainable AI: Dual SHAP & BERT Self-Attention Synthesis
    explanation = EXPLAINER.explain_prediction(
        feature_vector=fused_vec,
        title=payload.title,
        text=payload.text,
        prediction_label=prediction_label,
        fake_prob=fake_prob,
        bert_tokens=bert_tokens,
        bert_attention=bert_attention,
        top_k_features=8
    )

    t_latency = (time.perf_counter() - t0) * 1000.0

    # 4. Save to Database
    record_id = save_prediction(
        title=payload.title,
        prediction=prediction_label,
        confidence_fake=round(fake_prob * 100.0, 1),
        confidence_real=round(real_prob * 100.0, 1),
        source=payload.source or "Web",
        model_architecture="79-Dim Multimodal RF"
    )

    return PredictResponse(
        id=record_id,
        firebase_id=f"fb_pred_{record_id}",
        prediction=prediction_label,
        confidence_fake=round(fake_prob * 100.0, 1),
        confidence_real=round(real_prob * 100.0, 1),
        bert_confidence_fake=round(fake_prob * 100.0, 1),
        bert_confidence_real=round(real_prob * 100.0, 1),
        risk_level=risk_level,
        model_architecture="BERT Deep Learning + Ensemble (79-dim Multimodal)",
        database_provider="SQLite / Firebase Firestore",
        top_features=explanation["top_features"],
        token_attributions=explanation["token_attributions"],
        key_reasons=explanation["key_reasons"],
        title=payload.title,
        source=payload.source or "Web",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        inference_latency_ms=round(t_latency, 2)
    )


@app.post("/api/predict-url", response_model=PredictResponse)
def predict_from_url(payload: URLPredictRequest):
    """
    Phase 13 Feature: Real-time News Verification by Web URL:
    1. Scrapes news headline, body paragraphs, and publisher domain from URL.
    2. Executes 79-dimensional multimodal feature extraction and inference.
    3. Returns full XAI explanation, token heatmaps, and persists prediction event.
    """
    if not payload.url or not payload.url.strip():
        raise HTTPException(status_code=400, detail="News article URL is required.")

    try:
        article_data = extract_article_from_url(payload.url)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to scrape article from URL: {str(e)}")

    predict_req = PredictRequest(
        title=article_data["title"],
        text=article_data["text"],
        source=article_data["source"],
        author=article_data["author"],
        shares=1000.0,
        sentiment=0.0
    )

    return predict_news(predict_req)


@app.post("/api/predict-batch", response_model=BatchPredictResponse)
def predict_batch_news(payload: BatchPredictRequest):
    """Performs high-throughput bulk inference on multiple news articles."""
    if not payload.items:
        raise HTTPException(status_code=400, detail="Batch items list cannot be empty.")

    t_start = time.perf_counter()
    responses = []
    for item in payload.items:
        resp = predict_news(item)
        responses.append(resp)
    total_time_ms = (time.perf_counter() - t_start) * 1000.0
    avg_latency = total_time_ms / len(responses)

    return BatchPredictResponse(
        predictions=responses,
        total_processed=len(responses),
        total_time_ms=round(total_time_ms, 2),
        avg_latency_ms=round(avg_latency, 2)
    )


@app.get("/api/shap/global", response_model=GlobalSHAPResponse)
def get_global_shap_importance():
    """Returns 79-dimensional global SHAP feature importance rankings grouped by modality."""
    if EXPLAINER is None:
        raise HTTPException(status_code=503, detail="Explainer engine not loaded.")

    global_features = EXPLAINER.get_global_importance(top_k=79)
    modality_counts = {}
    items = []
    for feat in global_features:
        mod = feat["modality"]
        modality_counts[mod] = modality_counts.get(mod, 0) + 1
        items.append(GlobalSHAPItem(
            rank=feat["rank"],
            feature=feat["feature"],
            raw_name=feat["raw_name"],
            importance=feat["importance"],
            importance_percent=feat["importance_percent"],
            modality=mod
        ))

    return GlobalSHAPResponse(
        total_features=len(items),
        modalities=modality_counts,
        features=items
    )


@app.get("/api/metrics", response_model=MetricsResponse)
def get_metrics():
    """Returns complete consolidated benchmark evaluation payload, 7-model comparisons, and ablation studies."""
    global METRICS_PAYLOAD
    if METRICS_PAYLOAD is None and os.path.exists(METRICS_JSON_PATH):
        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            METRICS_PAYLOAD = json.load(f)

    if METRICS_PAYLOAD is None:
        raise HTTPException(status_code=404, detail="Metrics not generated yet.")

    return MetricsResponse(
        proposed_model=METRICS_PAYLOAD.get("proposed_model"),
        baselines=METRICS_PAYLOAD.get("baselines"),
        ablation_study=METRICS_PAYLOAD.get("ablation_study"),
        model_comparison=METRICS_PAYLOAD.get("model_comparison"),
        feature_count=79,
        bert_embedding_dim=64,
        training_samples=24,
        test_samples=7,
        database="SQLite / Firebase Firestore"
    )


@app.get("/api/history", response_model=list[HistoryItem])
def get_history():
    """Returns recent prediction audit history records."""
    return get_recent_history(limit=20)


@app.get("/api/history/stats", response_model=HistoryStatsResponse)
def get_history_statistics():
    """Returns aggregate statistics on audit history."""
    stats = get_history_stats()
    return HistoryStatsResponse(**stats)


@app.delete("/api/history")
def clear_history_endpoint():
    """Clears all audit records from database."""
    count = clear_prediction_history()
    return {"message": f"Successfully deleted {count} prediction records.", "deleted_count": count}


# --- Phase 14 REST Endpoints: Audit Reports, Threat Intel & Active Learning ---

@app.post("/api/export-report")
def export_audit_report(payload: ReportExportRequest):
    """
    Phase 14 Feature: Generates a self-contained, publication-grade printable
    HTML/PDF forensic audit report certificate for an analyzed news article.
    """
    html_content = generate_audit_html(payload.prediction_data)
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/api/threat-stream", response_model=ThreatFeedResponse)
def get_threat_stream():
    """
    Phase 14 Feature: Continuous Threat Intelligence Feed & RSS Misinformation Monitor.
    Runs 79-dim multimodal verification across live news wires and computes Misinformation Threat Index (MTI).
    """
    def _feed_predictor(title, summary, source):
        req = PredictRequest(
            title=title,
            text=summary or title,
            source=source,
            author="Wire Reporter",
            shares=1000.0,
            sentiment=0.0
        )
        res = predict_news(req)
        return {
            "prediction": res.prediction,
            "confidence_fake": res.confidence_fake,
            "confidence_real": res.confidence_real,
            "risk_level": res.risk_level
        }

    feed_data = fetch_live_threat_feed(predict_fn=_feed_predictor)
    return ThreatFeedResponse(**feed_data)


@app.post("/api/feedback", response_model=FeedbackResponse)
def submit_feedback_endpoint(payload: FeedbackRequest):
    """
    Phase 14 Feature: Fact-checker human-in-the-loop annotation and active learning correction.
    """
    record = submit_fact_checker_feedback(
        article_id=payload.article_id,
        article_title=payload.article_title,
        model_prediction=payload.model_prediction,
        corrected_label=payload.corrected_label,
        notes=payload.notes or "",
        analyst_name=payload.analyst_name or "Verified Fact-Checker"
    )
    return FeedbackResponse(
        status="success",
        message="Fact-checker annotation logged successfully for active learning.",
        record=record
    )


@app.get("/api/active-learning/stats", response_model=ActiveLearningStatsResponse)
def get_active_learning_stats():
    """
    Phase 14 Feature: Retrieves Concept Drift (PSI) telemetry, retrain readiness, and active learning statistics.
    """
    return get_active_learning_analytics()


# --- Phase 15 REST Endpoints: Adversarial Robustness, Multilingual & Propagation ---

@app.post("/api/adversarial/audit", response_model=AdversarialAuditResponse)
def audit_adversarial_robustness_endpoint(payload: AdversarialAuditRequest):
    """
    Phase 15 Feature: Evaluates model robustness against 4 adversarial perturbation vectors:
    Homoglyphs, Zero-Width Spacing, Emoji Floods, and Synonym Obfuscations.
    """
    def _local_predictor(t, txt, src):
        req = PredictRequest(
            title=t,
            text=txt,
            source=src or "Web",
            author="Stress Tester",
            shares=1000.0,
            sentiment=0.0
        )
        res = predict_news(req)
        return {
            "prediction": res.prediction,
            "confidence_fake": res.confidence_fake,
            "confidence_real": res.confidence_real
        }

    audit_res = evaluate_adversarial_robustness(
        title=payload.title,
        text=payload.text,
        source=payload.source or "Web",
        predict_fn=_local_predictor
    )
    return AdversarialAuditResponse(**audit_res)


@app.post("/api/predict-multilingual", response_model=MultilingualPredictResponse)
def predict_multilingual_news(payload: MultilingualPredictRequest):
    """
    Phase 15 Feature: Cross-Lingual Verification & Neural Semantic Mapping.
    Automatically detects language (Tamil, Hindi, Spanish, French, etc.) and maps to canonical BERT feature space.
    """
    lang_code, lang_meta = detect_language(payload.title + " " + payload.text)
    
    canonical_title = translate_to_canonical_representation(payload.title, lang_code)
    canonical_text = translate_to_canonical_representation(payload.text, lang_code)

    predict_req = PredictRequest(
        title=canonical_title,
        text=canonical_text,
        source=payload.source or "Regional News Wire",
        author=payload.author or "Correspondent",
        shares=1000.0,
        sentiment=0.0
    )
    prediction_result = predict_news(predict_req)

    return MultilingualPredictResponse(
        detected_language=lang_code,
        language_name=lang_meta["name"],
        language_flag=lang_meta["flag"],
        original_title=payload.title,
        canonical_translation=canonical_title,
        prediction_result=prediction_result
    )


@app.post("/api/simulate/propagation", response_model=PropagationSimResponse)
def simulate_propagation_endpoint(payload: PropagationSimRequest):
    """
    Phase 15 Feature: Social Network Propagation & Viral Rumor Cascade Simulator.
    Simulates rumor transmission (R0) and graph contagion across scale-free network topology.
    """
    sim_data = simulate_viral_cascade(
        headline=payload.headline,
        prediction=payload.prediction or "FAKE",
        fake_confidence=payload.fake_confidence or 85.0,
        network_size=payload.network_size or 35,
        viral_speed=payload.viral_speed or 1.5
    )
    return PropagationSimResponse(**sim_data)




# Serve Frontend static assets
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_index():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "VeriShield AI REST API is running. Access /docs for interactive Swagger UI."}
