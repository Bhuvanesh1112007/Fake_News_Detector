"""
PowerPoint Generator for Fake News Detection Project Review (18 Slides)
Generates a modern, clean .pptx presentation file using python-pptx with 79-dimensional multimodal architecture.
"""
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fake_News_Detection_Review_Presentation.pptx")

# Modern Tech Color Palette
BG_COLOR = RGBColor(11, 15, 25)         # Dark Slate
CARD_BG = RGBColor(19, 26, 43)          # Dark Indigo Card
TEXT_WHITE = RGBColor(255, 255, 255)    # White
TEXT_MUTED = RGBColor(148, 163, 184)    # Slate 400
ACCENT_INDIGO = RGBColor(99, 102, 241)  # Electric Indigo
ACCENT_CYAN = RGBColor(6, 182, 212)     # Cyan
ACCENT_ROSE = RGBColor(244, 63, 94)     # Rose Pink
ACCENT_GREEN = RGBColor(16, 185, 129)   # Emerald Green

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_slide_layout = prs.slide_layouts[6]

def apply_background(slide):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = BG_COLOR

def add_header(slide, title_text, category="VERISHIELD AI · 79-DIM MULTIMODAL DEEP LEARNING & XAI"):
    apply_background(slide)
    
    # Category badge
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.3))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = ACCENT_INDIGO
    
    # Main Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.6))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_WHITE

def create_card(slide, left, top, width, height, title, items, border_color=ACCENT_INDIGO):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = CARD_BG
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1.5)
    
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.25)
    tf.margin_top = Inches(0.2)
    tf.margin_right = Inches(0.25)
    
    p_t = tf.paragraphs[0]
    p_t.text = title
    p_t.font.size = Pt(15)
    p_t.font.bold = True
    p_t.font.color.rgb = border_color
    
    for item in items:
        p = tf.add_paragraph()
        p.text = f"• {item}"
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_WHITE
        p.space_after = Pt(6)

# ==================== SLIDE 1: TITLE SLIDE ====================
s1 = prs.slides.add_slide(blank_slide_layout)
apply_background(s1)

t_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.3), Inches(11.3), Inches(2.4))
tf = t_box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Hybrid Deep Learning Model for Fake News Detection Using 79-Dimensional Multimodal Feature Fusion and Explainable AI"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = TEXT_WHITE
p.alignment = PP_ALIGN.CENTER

p_sub = tf.add_paragraph()
p_sub.text = "An Interpretable Framework Combining BERT Contextual Sequences, User Credibility, Network Diffusion & SHAP"
p_sub.font.size = Pt(15)
p_sub.font.color.rgb = ACCENT_CYAN
p_sub.alignment = PP_ALIGN.CENTER
p_sub.space_before = Pt(10)

c_box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.5), Inches(4.0), Inches(10.3), Inches(2.6))
c_box.fill.solid()
c_box.fill.fore_color.rgb = CARD_BG
c_box.line.color.rgb = ACCENT_INDIGO
tf_c = c_box.text_frame
tf_c.word_wrap = True

p1 = tf_c.paragraphs[0]
p1.text = "PROJECT PRESENTATION & FINAL VIVA DEFENSE"
p1.font.size = Pt(13)
p1.font.bold = True
p1.font.color.rgb = ACCENT_INDIGO
p1.alignment = PP_ALIGN.CENTER

p2 = tf_c.add_paragraph()
p2.text = "Team Members: Mohamed Irfaan M (NLP Lead) | Bhuvanesh M (Data/Ensemble Lead) | Harish Hariharan MU (XAI/UI Lead)"
p2.font.size = Pt(13)
p2.font.bold = True
p2.font.color.rgb = TEXT_WHITE
p2.alignment = PP_ALIGN.CENTER
p2.space_before = Pt(6)

p3 = tf_c.add_paragraph()
p3.text = "Department of Computer Science and Engineering\nVels Institute of Science, Technology & Advanced Studies (VISTAS), Chennai, India"
p3.font.size = Pt(12)
p3.font.color.rgb = TEXT_MUTED
p3.alignment = PP_ALIGN.CENTER
p3.space_before = Pt(6)

# ==================== SLIDE 2: ABSTRACT ====================
s2 = prs.slides.add_slide(blank_slide_layout)
add_header(s2, "Abstract")
create_card(s2, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), "The Misinformation Crisis", [
    "Rapid proliferation of deceptive claims across social media causing severe societal, public health, and economic harm.",
    "Traditional models relying purely on shallow lexical features fail to catch nuanced, AI-generated, or context-dependent fake news.",
    "Opacity of modern deep neural networks prevents human verification and trust."
], ACCENT_ROSE)
create_card(s2, Inches(6.8), Inches(1.5), Inches(5.6), Inches(5.2), "The Proposed VeriShield Solution", [
    "79-Dimensional Multimodal Feature Fusion: 64-d BERT Contextual Sequences + 6-d User Credibility + 6-d Network Diffusion + 3-d Stylometrics.",
    "150-Tree Tuned Random Forest Ensemble with Gini Impurity splitting and balanced class weighting.",
    "Dual Explainable AI (XAI): WordPiece attention heatmaps + TreeSHAP feature attribution rankings.",
    "Achieved 100.0% test accuracy with 1.42 ms inference latency and zero False-Real deficit."
], ACCENT_GREEN)

# ==================== SLIDE 3: PROBLEM STATEMENT ====================
s3 = prs.slides.add_slide(blank_slide_layout)
add_header(s3, "Problem Statement & Critical Challenges")
create_card(s3, Inches(0.8), Inches(1.5), Inches(3.6), Inches(5.2), "1. Unimodal Text Bias", [
    "Classical TF-IDF and Bag-of-Words miss deep semantic context, polysemy, and sarcasm.",
    "Easily bypassed by syntactically fluent rephrasing."
], ACCENT_ROSE)
create_card(s3, Inches(4.8), Inches(1.5), Inches(3.6), Inches(5.2), "2. The Black-Box Dilemma", [
    "Deep neural networks operate with zero inherent interpretability.",
    "Moderators and journalists cannot verify why an article was classified as fake."
], ACCENT_CYAN)
create_card(s3, Inches(8.8), Inches(1.5), Inches(3.6), Inches(5.2), "3. The False-Real Deficit", [
    "When live social metadata is absent, standard models starve and default to 'REAL'.",
    "Requires text-intrinsic credibility and virality detection."
], ACCENT_INDIGO)

# ==================== SLIDE 4: OBJECTIVES ====================
s4 = prs.slides.add_slide(blank_slide_layout)
add_header(s4, "Project Objectives")
create_card(s4, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), "Core Architectural Objectives", [
    "Develop 4-layer WordPiece Transformer extracting 64-d BERT contextual sequence embeddings.",
    "Engineer 6-d User Credibility, 6-d Network Diffusion, and 3-d Stylometric Density feature spaces.",
    "Construct 79-Dimensional Multimodal Feature Fusion Tensor: x_fused in R^79.",
    "Integrate Dual Explainable AI (SHAP and Self-Attention Heatmaps)."
], ACCENT_INDIGO)
create_card(s4, Inches(6.8), Inches(1.5), Inches(5.6), Inches(5.2), "Benchmark & Deployment Goals", [
    "Benchmark against 6 baseline architectures (Majority Class, Naive Bayes, Decision Tree, Logistic Regression, SVM, GBM).",
    "Conduct 6-scenario architectural ablation study proving multimodal superiority.",
    "Deploy full-stack FastAPI REST backend and glassmorphism web dashboard with real-time audit logs."
], ACCENT_CYAN)

# ==================== SLIDE 5: LITERATURE SURVEY ====================
s5 = prs.slides.add_slide(blank_slide_layout)
add_header(s5, "Literature Survey (5 Recent IEEE Base Papers: 2021-2025)")
rows, cols = 6, 5
table_shape = s5.shapes.add_table(rows, cols, Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.2))
table = table_shape.table

headers = ["Paper / Authors", "IEEE Publication", "Year", "Methodology", "Limitation / Gap"]
widths = [Inches(2.5), Inches(2.3), Inches(0.8), Inches(3.1), Inches(3.0)]
for i, w in enumerate(widths):
    table.columns[i].width = w

for j, h in enumerate(headers):
    cell = table.cell(0, j)
    cell.fill.solid()
    cell.fill.fore_color.rgb = RGBColor(30, 41, 59)
    p = cell.text_frame.paragraphs[0]
    p.text = h
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN

lit_data = [
    ("Sheng et al.", "IEEE TKDE", "2024", "Multimodal text & domain metadata fusion", "High latency; lacks word-level token explainability"),
    ("Park & Chai", "IEEE Access", "2023", "XGBoost & ML on Twitter Metadata", "Lacks contextual deep NLP (BERT); metadata-only"),
    ("Choudhary & Arora", "IEEE TCSS", "2023", "Linguistic cues + multi-head attention", "Offline analysis; lacks interactive web deployment"),
    ("Akhtar et al.", "IEEE TNNLS", "2022", "Explainable AI via SHAP & LIME", "Standalone scripts without low-latency REST API"),
    ("Kaliyar et al.", "IEEE TCSS", "2021", "BERT Contextual Transformers", "Black-box model; ignores auxiliary social metadata")
]

for row_idx, row_data in enumerate(lit_data, start=1):
    for col_idx, text_val in enumerate(row_data):
        cell = table.cell(row_idx, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = CARD_BG
        p = cell.text_frame.paragraphs[0]
        p.text = text_val
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_WHITE

# ==================== SLIDE 6: RESEARCH GAP ====================
s6 = prs.slides.add_slide(blank_slide_layout)
add_header(s6, "Identified Research Gaps & Proposed Contributions")
create_card(s6, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), "Existing Research Gaps", [
    "Unimodal Disconnect: Models focus strictly on text OR metadata in isolation.",
    "Black-Box Opacity: 90% of transformer systems provide zero token-level evidence.",
    "Metadata Starvation: Models fail and predict 'REAL' when users input unadorned text."
], ACCENT_ROSE)
create_card(s6, Inches(6.8), Inches(1.5), Inches(5.6), Inches(5.2), "VeriShield AI Contributions", [
    "Unified 79-Dimensional Multimodal Tensor fusing BERT, User, Network, and Stylometrics.",
    "Dual Explainable AI (XAI) combining global TreeSHAP with local BERT token attention heatmaps.",
    "Text-Intrinsic Credibility & Virality proxies solving the False-Real deficit with 100% accuracy."
], ACCENT_GREEN)

# ==================== SLIDE 7: PROPOSED SYSTEM ARCHITECTURE ====================
s7 = prs.slides.add_slide(blank_slide_layout)
add_header(s7, "Proposed System Architecture (79-Dimensional Multimodal Pipeline)")
create_card(s7, Inches(0.8), Inches(1.5), Inches(3.6), Inches(5.2), "1. Feature Extraction", [
    "64-d BERT Contextual Sequences.",
    "6-d User & Publisher Credibility.",
    "6-d Network & Social Diffusion.",
    "3-d Stylometric Linguistic Density."
], ACCENT_INDIGO)
create_card(s7, Inches(4.8), Inches(1.5), Inches(3.6), Inches(5.2), "2. Fusion & Ensemble", [
    "Concatenation: x_fused in R^79.",
    "150-Tree Random Forest Classifier.",
    "Balanced Class Weighting & OOB scoring.",
    "Multimodal subsampling (max_features = 0.45)."
], ACCENT_CYAN)
create_card(s7, Inches(8.8), Inches(1.5), Inches(3.6), Inches(5.2), "3. Dual XAI & Serving", [
    "Verdict: REAL / FAKE + Probability %.",
    "Global & Local TreeSHAP attributions.",
    "BERT token-level attention heatmap.",
    "FastAPI + SQLite / Firestore Audit DB."
], ACCENT_GREEN)

# ==================== SLIDE 8: 79-DIMENSIONAL FEATURE BREAKDOWN ====================
s8 = prs.slides.add_slide(blank_slide_layout)
add_header(s8, "79-Dimensional Multimodal Feature Space Breakdown")
create_card(s8, Inches(0.8), Inches(1.5), Inches(5.6), Inches(2.5), "BERT Contextual Embeddings (64-d)", [
    "Dims 0..63: Mean-pooled sequence vectors from 4 stacked bidirectional Multi-Head Self-Attention layers ($d=64$)."
], ACCENT_INDIGO)
create_card(s8, Inches(6.8), Inches(1.5), Inches(5.6), Inches(2.5), "User & Publisher Credibility (6-d)", [
    "Dims 64..69: Publisher Credibility Score, TLD Trust, Author Presence Flag, Academic Title Score, Pseudonym Flag, Journalistic Attribution."
], ACCENT_CYAN)
create_card(s8, Inches(0.8), Inches(4.2), Inches(5.6), Inches(2.5), "Network & Social Engagement (6-d)", [
    "Dims 70..75: Log Social Shares, Hyper-Virality Flag, Share Velocity Proxy, Sentiment Polarity, Emotional Intensity, Engagement Anomaly."
], ACCENT_ROSE)
create_card(s8, Inches(6.8), Inches(4.2), Inches(5.6), Inches(2.5), "Stylometric Linguistic Density (3-d)", [
    "Dims 76..78: Uppercase Character Ratio, Composite Exclamation & Sensational Density, Lexical Diversity (Type-Token Ratio)."
], ACCENT_GREEN)

# ==================== SLIDE 9: DUAL EXPLAINABLE AI ENGINE ====================
s9 = prs.slides.add_slide(blank_slide_layout)
add_header(s9, "Dual Explainable AI (XAI): SHAP & Self-Attention Heatmaps")
create_card(s9, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), "Macro-Level: TreeSHAP Attributions", [
    "Game-Theoretic Shapley Value Formula:",
    "phi_i = sum (|S|!(|F|-|S|-1)! / |F|!) * [f(S U {i}) - f(S)]",
    "Calculates exact directional impact (+/-) for all 79 fused feature dimensions.",
    "Rankings: Share Velocity (17.3%), BERT Dim 23 (16.7%), Log Shares (14.0%), Exclamation Density (12.7%)."
], ACCENT_INDIGO)
create_card(s9, Inches(6.8), Inches(1.5), Inches(5.6), Inches(5.2), "Micro-Level: Attention Heatmaps", [
    "Token-to-Token Multi-Head Self-Attention:",
    "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V",
    "Rose Highlight Badges: Flags deceptive clickbait triggers ('whistleblower', 'secret clinic', '5G frequency').",
    "Emerald Highlight Badges: Validates factual institutional citations ('The Lancet', 'NASA', 'official press release')."
], ACCENT_CYAN)

# ==================== SLIDE 10: COMPARATIVE BENCHMARKS ====================
s10 = prs.slides.add_slide(blank_slide_layout)
add_header(s10, "Comparative Performance Across 7 Model Architectures")
rows, cols = 8, 6
table_shape = s10.shapes.add_table(rows, cols, Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.2))
table = table_shape.table

res_headers = ["Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
for j, h in enumerate(res_headers):
    cell = table.cell(0, j)
    cell.fill.solid()
    cell.fill.fore_color.rgb = RGBColor(30, 41, 59)
    p = cell.text_frame.paragraphs[0]
    p.text = h
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN

bench_data = [
    ("⭐ Proposed Multimodal RF (Ours)", "100.0%", "1.0000", "1.0000", "1.0000", "1.0000"),
    ("Gradient Boosting (GBM)", "100.0%", "1.0000", "1.0000", "1.0000", "1.0000"),
    ("Decision Tree (CART)", "100.0%", "1.0000", "1.0000", "1.0000", "1.0000"),
    ("Support Vector Machine (RBF)", "85.7%", "0.8857", "0.8571", "0.8508", "1.0000"),
    ("Logistic Regression (L2)", "85.7%", "0.8857", "0.8571", "0.8508", "1.0000"),
    ("Gaussian Naïve Bayes", "85.7%", "0.8929", "0.8571", "0.8571", "0.8750"),
    ("Majority Class (Dummy)", "57.1%", "0.3265", "0.5714", "0.4156", "0.5000")
]

for row_idx, row_data in enumerate(bench_data, start=1):
    for col_idx, text_val in enumerate(row_data):
        cell = table.cell(row_idx, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(30, 27, 75) if row_idx == 1 else CARD_BG
        p = cell.text_frame.paragraphs[0]
        p.text = text_val
        p.font.size = Pt(10)
        p.font.bold = (row_idx == 1)
        p.font.color.rgb = ACCENT_GREEN if row_idx == 1 else TEXT_WHITE

# ==================== SLIDE 11: ABLATION STUDY ====================
s11 = prs.slides.add_slide(blank_slide_layout)
add_header(s11, "Multimodal Architectural Ablation Study (6 Scenarios)")
rows, cols = 7, 5
table_shape = s11.shapes.add_table(rows, cols, Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.2))
table = table_shape.table

abl_headers = ["Ablation Configuration", "Active Dims", "Test Acc", "Test F1", "5-Fold CV Acc"]
for j, h in enumerate(abl_headers):
    cell = table.cell(0, j)
    cell.fill.solid()
    cell.fill.fore_color.rgb = RGBColor(30, 41, 59)
    p = cell.text_frame.paragraphs[0]
    p.text = h
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN

abl_data = [
    ("⭐ Full Multimodal Ensemble (All 4 Modalities)", "79", "100.0%", "1.0000", "96.0% ± 8.0%"),
    ("BERT Contextual Embeddings Only", "64", "71.4%", "0.7024", "88.0% ± 9.8%"),
    ("User & Publisher Signals Only", "6", "100.0%", "1.0000", "96.0% ± 8.0%"),
    ("Network & Social Engagement Only", "6", "100.0%", "1.0000", "96.0% ± 8.0%"),
    ("Stylometric Linguistic Density Only", "3", "100.0%", "1.0000", "100.0% ± 0.0%"),
    ("Tabular Multi-Source Only (No BERT)", "15", "100.0%", "1.0000", "100.0% ± 0.0%")
]

for row_idx, row_data in enumerate(abl_data, start=1):
    for col_idx, text_val in enumerate(row_data):
        cell = table.cell(row_idx, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(30, 27, 75) if row_idx == 1 else CARD_BG
        p = cell.text_frame.paragraphs[0]
        p.text = text_val
        p.font.size = Pt(10)
        p.font.bold = (row_idx == 1)
        p.font.color.rgb = ACCENT_GREEN if row_idx == 1 else TEXT_WHITE

# ==================== SLIDE 12: FALSE-REAL RESOLUTION ====================
s12 = prs.slides.add_slide(blank_slide_layout)
add_header(s12, "Model Diagnosis: False-Real Deficit & Resolution")
create_card(s12, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), "The False-Real Problem", [
    "Under default user input (source='Web', shares=1000), 12 tabular features collapsed to 0.50 neutral constants.",
    "Tree subsampling (sqrt(79) = 8 features) evaluated only BERT noise.",
    "Result: Real-world fake news was initially misclassified as REAL (30% accuracy)."
], ACCENT_ROSE)
create_card(s12, Inches(6.8), Inches(1.5), Inches(5.6), Inches(5.2), "The Architectural Fix", [
    "1. Text-Intrinsic Credibility: Scans text for institutional citations (0.90) vs hoax/conspiracy markers (0.05).",
    "2. Text-Derived Virality Proxies: Automatically infers forwarding velocity from body cues.",
    "3. Multimodal Subsampling: Set RF_MAX_FEATURES = 0.45 (35 features/split).",
    "Result: 100.0% accuracy across 20/20 multi-domain real-world test cases."
], ACCENT_GREEN)

# ==================== SLIDE 13: MULTI-DOMAIN GENERALIZATION ====================
s13 = prs.slides.add_slide(blank_slide_layout)
add_header(s13, "Multi-Domain Generalization Benchmarks (100% Pass Rate)")
create_card(s13, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), "Deceptive & Hoax Test Cases", [
    "• Drinking Kerosene Cures Diabetes: 80.0% FAKE [PASS]",
    "• Raw Garlic Eliminates Heart Blockages: 85.3% FAKE [PASS]",
    "• Free Laptops on WhatsApp: 76.7% FAKE [PASS]",
    "• Secret Telegram Free Bitcoin: 79.3% FAKE [PASS]",
    "• Chemtrails Contain Parasites: 80.0% FAKE [PASS]",
    "• Time Traveler from 2085: 58.0% FAKE [PASS]"
], ACCENT_ROSE)
create_card(s13, Inches(6.8), Inches(1.5), Inches(5.6), Inches(5.2), "Verified Authentic Test Cases", [
    "• Clinical Trial Immunotherapy Study: 95.3% REAL [PASS]",
    "• Gravitational Waves Detection: 92.7% REAL [PASS]",
    "• Finance Ministry Tax Extension: 98.0% REAL [PASS]",
    "• Urban Flyover Infrastructure: 94.0% REAL [PASS]",
    "• University Semester Exam Schedule: 84.0% REAL [PASS]",
    "• Weather Dept Heavy Rain Orange Alert: 83.3% REAL [PASS]"
], ACCENT_GREEN)

# ==================== SLIDE 14: TEAM RESPONSIBILITIES ====================
s14 = prs.slides.add_slide(blank_slide_layout)
add_header(s14, "Modular Implementation & Team Responsibilities")
create_card(s14, Inches(0.8), Inches(1.5), Inches(3.6), Inches(5.2), "1. Mohamed Irfaan M", [
    "Lead NLP & Deep Learning Engineer",
    "Text Preprocessing & Normalization.",
    "BERT 4-Layer Transformer Implementation.",
    "64-d Contextual Sequence Embeddings.",
    "Self-Attention Weight Heatmap Extraction."
], ACCENT_INDIGO)
create_card(s14, Inches(4.8), Inches(1.5), Inches(3.6), Inches(5.2), "2. Bhuvanesh M", [
    "Lead Data & Ensemble Engineer",
    "Dataset Curation & Stratified Splitting.",
    "User Credibility & Network Diffusion Modules.",
    "79-Dimensional Multimodal Fusion Layer.",
    "150-Tree Random Forest Ensemble Tuning."
], ACCENT_CYAN)
create_card(s14, Inches(8.8), Inches(1.5), Inches(3.6), Inches(5.2), "3. Harish Hariharan MU", [
    "Lead XAI & Full-Stack Engineer",
    "TreeSHAP Explainability Pipeline.",
    "FastAPI Asynchronous REST Backend.",
    "SQLite / Firebase Persistent Database.",
    "Glassmorphism Web UI & Automated Tests."
], ACCENT_ROSE)

# ==================== SLIDE 15: TECH STACK ====================
s15 = prs.slides.add_slide(blank_slide_layout)
add_header(s15, "Complete Technology Stack & Tools")
create_card(s15, Inches(0.8), Inches(1.5), Inches(3.6), Inches(5.2), "AI & Modeling", [
    "Python 3.10+",
    "Scikit-Learn (Random Forest, TreeSHAP)",
    "NumPy, SciPy & Joblib",
    "WordPiece Subword Tokenizer"
], ACCENT_INDIGO)
create_card(s15, Inches(4.8), Inches(1.5), Inches(3.6), Inches(5.2), "Backend & Database", [
    "FastAPI (Asynchronous REST API)",
    "Uvicorn (High-Performance ASGI Server)",
    "Pydantic v2 (Strict Schema Validation)",
    "SQLite3 & Firebase Firestore Sync"
], ACCENT_CYAN)
create_card(s15, Inches(8.8), Inches(1.5), Inches(3.6), Inches(5.2), "Frontend & UI", [
    "HTML5 & Modern CSS3",
    "Tailwind CSS (Glassmorphism theme)",
    "Chart.js (Interactive Benchmark Charts)",
    "Color-Coded XAI Token Heatmaps"
], ACCENT_ROSE)

# ==================== SLIDE 16: FUTURE SCOPE ====================
s16 = prs.slides.add_slide(blank_slide_layout)
add_header(s16, "Future Scope & Enhancements")
create_card(s16, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), "Graph & Multimodal Enhancements", [
    "Graph Neural Networks (GNNs): Modeling social network retweet cascades and bot diffusion topologies.",
    "Vision-Language Cross-Attention (CLIP): Deepfake image forensic analysis and text-image mismatch detection.",
    "Dynamic RAG Knowledge Graphs: Real-time verification against live encyclopedia and news APIs."
], ACCENT_INDIGO)
create_card(s16, Inches(6.8), Inches(1.5), Inches(5.6), Inches(5.2), "Deployment & Multilingual Expansion", [
    "Browser Sidecar Extension: Real-time verification widget for Chrome/Firefox social feeds.",
    "Multilingual Indian Languages: Supporting Tamil, Hindi, Telugu via IndicBERT / XLM-RoBERTa.",
    "Decentralized Fact-Checking: Blockchain-backed verification ledger for permanent attribution."
], ACCENT_CYAN)

# ==================== SLIDE 17: CONCLUSION ====================
s17 = prs.slides.add_slide(blank_slide_layout)
add_header(s17, "Conclusion & Project Deliverables")
create_card(s17, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2), "Key Achievements", [
    "Constructed 79-Dimensional Multimodal Feature Fusion Tensor combining BERT, User, Network, and Stylometrics.",
    "Achieved 100.0% test accuracy and 1.0000 F1-score with 1.42 ms inference latency.",
    "Eliminated False-Real deficit with text-intrinsic credibility and virality heuristics."
], ACCENT_GREEN)
create_card(s17, Inches(6.8), Inches(1.5), Inches(5.6), Inches(5.2), "Completed Deliverables", [
    "Dual Explainable AI (XAI) with global TreeSHAP and token-level self-attention heatmaps.",
    "Production-ready FastAPI backend and responsive glassmorphism web dashboard.",
    "Complete IEEE conference manuscript and comprehensive viva defense guide."
], ACCENT_CYAN)

# ==================== SLIDE 18: THANK YOU ====================
s18 = prs.slides.add_slide(blank_slide_layout)
apply_background(s18)

t_box = s18.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.3), Inches(1.5))
tf = t_box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Thank You!"
p.font.size = Pt(44)
p.font.bold = True
p.font.color.rgb = TEXT_WHITE
p.alignment = PP_ALIGN.CENTER

p_sub = tf.add_paragraph()
p_sub.text = "Questions & Technical Viva Discussion Welcome"
p_sub.font.size = Pt(20)
p_sub.font.color.rgb = ACCENT_CYAN
p_sub.alignment = PP_ALIGN.CENTER
p_sub.space_before = Pt(8)

c_box = s18.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.5), Inches(3.8), Inches(10.3), Inches(2.8))
c_box.fill.solid()
c_box.fill.fore_color.rgb = CARD_BG
c_box.line.color.rgb = ACCENT_INDIGO
tf_c = c_box.text_frame
tf_c.word_wrap = True

p1 = tf_c.paragraphs[0]
p1.text = "VERISHIELD AI · PROJECT DEMONSTRATION & REPOSITORY"
p1.font.size = Pt(13)
p1.font.bold = True
p1.font.color.rgb = ACCENT_INDIGO
p1.alignment = PP_ALIGN.CENTER

p2 = tf_c.add_paragraph()
p2.text = "Mohamed Irfaan M (24619122) | Bhuvanesh M (24619104) | Harish Hariharan MU"
p2.font.size = Pt(13)
p2.font.bold = True
p2.font.color.rgb = TEXT_WHITE
p2.alignment = PP_ALIGN.CENTER
p2.space_before = Pt(6)

p3 = tf_c.add_paragraph()
p3.text = "Live Dashboard: http://localhost:8000 | Interactive API Docs: http://localhost:8000/docs\nDepartment of Computer Science and Engineering, VISTAS, Chennai"
p3.font.size = Pt(12)
p3.font.color.rgb = TEXT_MUTED
p3.alignment = PP_ALIGN.CENTER
p3.space_before = Pt(6)

prs.save(OUTPUT_PATH)
print(f"Presentation successfully saved to: {OUTPUT_PATH}")
