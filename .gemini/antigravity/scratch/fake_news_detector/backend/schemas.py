"""
Pydantic Data Schemas for REST API (Phase 11)
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    title: str = Field(..., description="Headline or title of the news article")
    text: str = Field(..., description="Main text / body content of the article")
    source: Optional[str] = Field("General Web", description="Publisher or domain source")
    author: Optional[str] = Field("Unknown", description="Author or byline")
    shares: Optional[float] = Field(1000.0, description="Approximate social shares / engagement")
    sentiment: Optional[float] = Field(0.0, description="Sentiment score [-1.0, 1.0]")


class URLPredictRequest(BaseModel):
    url: str = Field(..., description="Public web news article URL to scrape and classify")


class BatchPredictRequest(BaseModel):
    items: List[PredictRequest] = Field(..., description="List of news articles to classify in batch")


class FeatureImpact(BaseModel):
    feature: str
    raw_name: str
    impact: float
    relative_influence_percent: Optional[float] = 0.0
    direction: str
    value: float
    modality: Optional[str] = "Multimodal Tabular Signal"
    is_bert: Optional[bool] = False


class TokenAttribution(BaseModel):
    token: str
    score: float
    bert_attention_weight: Optional[float] = 0.0
    type: str


class PredictResponse(BaseModel):
    id: Optional[int] = None
    firebase_id: Optional[str] = None
    prediction: str
    confidence_fake: float
    confidence_real: float
    bert_confidence_fake: Optional[float] = None
    bert_confidence_real: Optional[float] = None
    risk_level: str
    model_architecture: str = "BERT Deep Learning + Ensemble (79-dim Multimodal)"
    database_provider: str = "SQLite / Firebase Firestore"
    top_features: List[FeatureImpact]
    token_attributions: List[TokenAttribution]
    key_reasons: List[str]
    title: str
    source: str
    timestamp: str
    inference_latency_ms: Optional[float] = None


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]
    total_processed: int
    total_time_ms: float
    avg_latency_ms: float


class GlobalSHAPItem(BaseModel):
    rank: int
    feature: str
    raw_name: str
    importance: float
    importance_percent: float
    modality: str


class GlobalSHAPResponse(BaseModel):
    total_features: int = 79
    modalities: Dict[str, int]
    features: List[GlobalSHAPItem]


class HistoryItem(BaseModel):
    id: int
    firebase_id: Optional[str] = None
    title: str
    prediction: str
    confidence_fake: float
    confidence_real: float
    source: Optional[str] = "Web"
    model_architecture: Optional[str] = "BERT Deep Learning + Ensemble"
    timestamp: str


class HistoryStatsResponse(BaseModel):
    total_predictions: int
    real_count: int
    fake_count: int
    real_percent: float
    fake_percent: float


class DatabaseStatusResponse(BaseModel):
    provider: str
    project_id: str
    collection: str
    status: str
    total_records: int


class MetricsResponse(BaseModel):
    proposed_model: Optional[dict] = None
    baselines: Optional[dict] = None
    ablation_study: Optional[list] = None
    model_comparison: Optional[list] = None
    benchmark_metrics: Optional[dict] = None
    global_feature_importance: Optional[list] = None
    feature_count: Optional[int] = 79
    bert_embedding_dim: Optional[int] = 64
    training_samples: Optional[int] = 24
    test_samples: Optional[int] = 7
    database: Optional[str] = "SQLite / Firebase Firestore"


# --- Phase 14 Schemas: Audit Reports, Threat Intel & Active Learning ---

class ReportExportRequest(BaseModel):
    prediction_data: dict


class ReportExportResponse(BaseModel):
    html_content: str
    audit_id: str
    timestamp: str


class ThreatFeedItem(BaseModel):
    id: str
    title: str
    summary: str
    source: str
    category: str
    pubDate: str
    link: str
    prediction: str
    confidence_fake: float
    confidence_real: float
    risk_level: str
    timestamp: str


class ThreatFeedResponse(BaseModel):
    status: str
    threat_index: float
    threat_status: str
    total_monitored: int
    flagged_threats: int
    verified_authentic: int
    last_updated: str
    feed_items: List[ThreatFeedItem]


class FeedbackRequest(BaseModel):
    article_id: str
    article_title: str
    model_prediction: str
    corrected_label: str
    notes: Optional[str] = ""
    analyst_name: Optional[str] = "Verified Fact-Checker"


class FeedbackResponse(BaseModel):
    status: str = "success"
    message: str
    record: dict


class ActiveLearningStatsResponse(BaseModel):
    total_human_feedback_samples: int
    label_breakdown: Dict[str, int]
    concept_drift: dict
    active_learning_pipeline: dict
    recent_corrections: List[dict]


# --- Phase 15 Schemas: Adversarial Robustness, Multilingual & Propagation ---

class AdversarialAuditRequest(BaseModel):
    title: str
    text: str
    source: Optional[str] = "Web"


class AdversarialAuditResponse(BaseModel):
    baseline_prediction: str
    baseline_confidence_fake: float
    adversarial_robustness_score: float
    robustness_grade: str
    total_attacks_tested: int
    evasions_prevented: int
    max_confidence_deviation: float
    attack_breakdown: List[dict]
    sanitizer_verification: dict


class MultilingualPredictRequest(BaseModel):
    title: str
    text: str
    source: Optional[str] = "Regional News Wire"
    author: Optional[str] = "Correspondent"


class MultilingualPredictResponse(BaseModel):
    detected_language: str
    language_name: str
    language_flag: str
    original_title: str
    canonical_translation: str
    prediction_result: PredictResponse


class PropagationSimRequest(BaseModel):
    headline: str
    prediction: Optional[str] = "FAKE"
    fake_confidence: Optional[float] = 85.0
    network_size: Optional[int] = 35
    viral_speed: Optional[float] = 1.5


class PropagationSimResponse(BaseModel):
    headline: str
    prediction: str
    cascade_metrics: dict
    graph_topology: dict
    time_series_timeline: List[dict]


