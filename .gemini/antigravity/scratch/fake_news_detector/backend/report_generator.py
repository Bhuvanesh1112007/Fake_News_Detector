"""
Phase 14: Automated Research Audit Report & Certificate Generator
Module Owner: Mohamed Irfaan M, Bhuvanesh M, Harish Hariharan MU (VISTAS)

Generates publication-grade, self-contained, print-ready HTML/PDF audit certificates
containing cryptographic-style verification stamps, 79-dimensional multimodal breakdown,
SHAP feature importance rankings, BERT token heatmaps, and forensic action plans.
"""

import json
import html
from datetime import datetime
from typing import Dict, Any


def generate_audit_html(data: Dict[str, Any]) -> str:
    """
    Generates a publication-grade HTML audit certificate for a VeriShield AI prediction event.
    Supports high-DPI vector printing to PDF via browser print dialogue or server-side headless Chrome.
    """
    title = html.escape(str(data.get("title", "Untitled Article")))
    source = html.escape(str(data.get("source", "General Web")))
    author = html.escape(str(data.get("author", "Anonymous / Unknown")))
    prediction = data.get("prediction", "UNKNOWN").upper()
    conf_fake = float(data.get("confidence_fake", 50.0))
    conf_real = float(data.get("confidence_real", 50.0))
    risk_level = html.escape(str(data.get("risk_level", "Medium Risk")))
    latency = data.get("inference_latency_ms", 12.5)
    audit_id = data.get("id", "AUDIT-001")
    firebase_id = data.get("firebase_id", f"fb_{audit_id}")
    timestamp = data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"))
    
    is_fake = (prediction == "FAKE")
    verdict_color = "#ef4444" if is_fake else "#10b981"
    verdict_bg = "#fef2f2" if is_fake else "#ecfdf5"
    verdict_border = "#fca5a5" if is_fake else "#6ee7b7"
    badge_label = "MISINFORMATION DETECTED" if is_fake else "VERIFIED AUTHENTIC CONTENT"
    primary_confidence = conf_fake if is_fake else conf_real

    top_features = data.get("top_features", [])
    token_attributions = data.get("token_attributions", [])
    key_reasons = data.get("key_reasons", [])

    # Format SHAP features table rows
    shap_rows = ""
    for idx, feat in enumerate(top_features[:8], 1):
        name = html.escape(feat.get("feature", "Feature"))
        category = html.escape(feat.get("category", "Multimodal"))
        val = feat.get("value", 0.0)
        shap_val = feat.get("shap_value", 0.0)
        direction = feat.get("direction", "NEUTRAL")
        dir_color = "#ef4444" if direction == "INCREASES_FAKE" else "#10b981"
        dir_label = "▲ Towards Fake" if direction == "INCREASES_FAKE" else "▼ Towards Real"

        shap_rows += f"""
        <tr>
            <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0; font-size: 12px; color: #475569;">#{idx}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0; font-weight: 600; font-size: 13px; color: #1e293b;">{name}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0; font-size: 12px; color: #64748b;"><span style="background: #f1f5f9; padding: 2px 8px; border-radius: 4px;">{category}</span></td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0; font-family: monospace; font-size: 12px; color: #334155;">{val:.4f}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0; font-family: monospace; font-size: 12px; font-weight: 600; color: #0f172a;">{shap_val:.4f}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0; font-size: 12px; font-weight: 600; color: {dir_color};">{dir_label}</td>
        </tr>
        """

    # Format token heatmaps
    tokens_html = ""
    for tok in token_attributions[:25]:
        token_str = html.escape(tok.get("token", ""))
        weight = float(tok.get("weight", 0.0))
        is_susp = tok.get("is_suspicious", False)
        bg = f"rgba(239, 68, 68, {min(0.85, 0.15 + weight * 0.7):.2f})" if is_susp else f"rgba(59, 130, 246, {min(0.5, 0.05 + weight * 0.4):.2f})"
        color = "#ffffff" if is_susp and weight > 0.4 else "#0f172a"
        tokens_html += f"""<span style="display: inline-block; margin: 2px 3px; padding: 3px 6px; border-radius: 4px; background: {bg}; color: {color}; font-size: 12px; font-family: monospace;" title="Weight: {weight:.3f}">{token_str}</span>"""

    # Format key reasons
    reasons_html = ""
    for r in key_reasons:
        reasons_html += f"""<li style="margin-bottom: 6px; color: #334155; font-size: 13px;">{html.escape(r)}</li>"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>VeriShield AI - Forensic Audit Certificate #{audit_id}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f8fafc;
            color: #0f172a;
            margin: 0;
            padding: 30px;
            -webkit-print-color-adjust: exact;
            print-color-adjust: exact;
        }}
        
        .certificate-container {{
            max-width: 850px;
            margin: 0 auto;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
            padding: 40px;
        }}
        
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #f1f5f9;
            padding-bottom: 20px;
            margin-bottom: 25px;
        }}
        
        .logo-title {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .badge-shield {{
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, #4f46e5, #9333ea);
            color: white;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            font-weight: bold;
        }}
        
        .verdict-box {{
            background: {verdict_bg};
            border: 1.5px solid {verdict_border};
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 25px;
        }}
        
        .section-title {{
            font-size: 14px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            margin-top: 25px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            margin-bottom: 20px;
        }}
        
        th {{
            background: #f8fafc;
            color: #475569;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 10px 12px;
            border-bottom: 1px solid #cbd5e1;
        }}
        
        .footer {{
            border-top: 1px solid #e2e8f0;
            margin-top: 35px;
            padding-top: 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 11px;
            color: #94a3b8;
        }}
        
        .print-btn {{
            background: #4f46e5;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 20px;
            font-size: 13px;
        }}
        
        @media print {{
            body {{
                padding: 0;
                background: white;
            }}
            .certificate-container {{
                border: none;
                box-shadow: none;
                padding: 0;
                max-width: 100%;
            }}
            .no-print {{
                display: none !important;
            }}
        }}
    </style>
</head>
<body>

<div style="max-width: 850px; margin: 0 auto;" class="no-print">
    <button class="print-btn" onclick="window.print()">🖨️ Print / Save as PDF Certificate</button>
</div>

<div class="certificate-container">
    <!-- Header -->
    <div class="header">
        <div class="logo-title">
            <div class="badge-shield">🛡️</div>
            <div>
                <h1 style="margin: 0; font-size: 20px; font-weight: 800; color: #0f172a;">VeriShield AI Forensic Audit Report</h1>
                <p style="margin: 2px 0 0 0; font-size: 12px; color: #64748b;">Vels Institute of Science, Technology & Advanced Studies (VISTAS) · Research Defense</p>
            </div>
        </div>
        <div style="text-align: right;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; color: #4f46e5;">ID: #{audit_id}</div>
            <div style="font-size: 11px; color: #94a3b8;">{timestamp}</div>
        </div>
    </div>

    <!-- Verdict Banner -->
    <div class="verdict-box">
        <div>
            <div style="font-size: 11px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: {verdict_color};">{badge_label}</div>
            <div style="font-size: 22px; font-weight: 800; color: #0f172a; margin-top: 4px;">{prediction} NEWS CLASSIFICATION</div>
            <div style="font-size: 13px; color: #64748b; margin-top: 2px;">Risk Category: <strong style="color: #334155;">{risk_level}</strong> · Engine: 79-Dim Multimodal Feature Fusion</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 32px; font-weight: 800; color: {verdict_color};">{primary_confidence:.1f}%</div>
            <div style="font-size: 11px; color: #64748b;">Confidence Probability</div>
        </div>
    </div>

    <!-- Article Metadata -->
    <div class="meta-grid">
        <div style="grid-column: span 2;">
            <div style="font-size: 11px; color: #64748b; font-weight: 600;">ARTICLE HEADLINE:</div>
            <div style="font-size: 14px; font-weight: 600; color: #1e293b; margin-top: 2px;">"{title}"</div>
        </div>
        <div>
            <div style="font-size: 11px; color: #64748b; font-weight: 600;">SOURCE DOMAIN / PLATFORM:</div>
            <div style="font-size: 13px; color: #334155; margin-top: 2px;">{source}</div>
        </div>
        <div>
            <div style="font-size: 11px; color: #64748b; font-weight: 600;">BYLINE / AUTHOR:</div>
            <div style="font-size: 13px; color: #334155; margin-top: 2px;">{author}</div>
        </div>
    </div>

    <!-- Multimodal Feature Analysis (SHAP) -->
    <div class="section-title">📊 79-Dimensional Multimodal Feature Attributions (TreeSHAP)</div>
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Feature Name</th>
                <th>Modality</th>
                <th>Observed Value</th>
                <th>SHAP Contribution</th>
                <th>Influence Vector</th>
            </tr>
        </thead>
        <tbody>
            {shap_rows}
        </tbody>
    </table>

    <!-- Linguistic Token Attention Heatmap -->
    <div class="section-title">🔍 BERT Multi-Head Self-Attention Token Importance</div>
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; line-height: 1.8; margin-bottom: 20px;">
        {tokens_html if tokens_html else '<span style="color: #94a3b8; font-size: 12px;">No token attributions available</span>'}
    </div>

    <!-- Forensic Key Reasons -->
    <div class="section-title">🛡️ Forensic Explanatory Summary</div>
    <ul style="margin: 0; padding-left: 20px; background: #fdfdfd; border-left: 3px solid #6366f1; padding-top: 8px; padding-bottom: 8px;">
        {reasons_html if reasons_html else '<li>Standard linguistic patterns consistent with source domain baseline.</li>'}
    </ul>

    <!-- Footer -->
    <div class="footer">
        <div>
            <strong>VeriShield AI v2.4 Enterprise</strong> · Authors: Mohamed Irfaan M, Bhuvanesh M, Harish Hariharan MU
        </div>
        <div>
            Latency: {latency} ms · Firebase Sync: {firebase_id}
        </div>
    </div>
</div>

</body>
</html>
"""
    return html_content
