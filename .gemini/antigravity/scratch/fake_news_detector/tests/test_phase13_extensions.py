"""
Phase 13: Advanced Production Extensions & Deployment Automated Test Suite
Module Owner: Mohamed Irfaan M, Bhuvanesh M, Harish Hariharan MU (VISTAS)

Validates:
1. Web Article URL Extractor & HTML Parsing
2. FastAPI POST /api/predict-url Endpoint
3. Manifest V3 Chrome Extension Files & Configuration Integrity
4. Dockerfile & docker-compose.yml Container Orchestration Integrity
5. Interactive Research Jupyter Notebook Structure & Syntax
"""
import os
import sys
import json
from unittest.mock import patch, MagicMock

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import backend.app as app
from backend.url_extractor import ArticleHTMLParser, extract_article_from_url
from src.config import BASE_DIR


def test_1_url_extractor_parsing():
    """Verifies that ArticleHTMLParser properly extracts OpenGraph titles, body paragraphs, and authors."""
    print("\n[Phase 13 Test 1/5] Verifying Web Article HTML Parser & Scraper...")
    mock_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ignored Standard Title - BBC News</title>
        <meta property="og:title" content="Clinical Trial Shows 90% Efficacy for New Cancer Vaccine">
        <meta name="author" content="Dr. Sarah Jenkins">
    </head>
    <body>
        <nav><p>Nav link to ignore</p></nav>
        <header><h1>Header to ignore</h1></header>
        <main>
            <h1>Clinical Trial Shows 90% Efficacy for New Cancer Vaccine</h1>
            <p>Researchers at the medical institute announced breakthrough results from a phase 3 clinical trial evaluating an oncology vaccine.</p>
            <p>The study was published in a leading peer-reviewed medical journal following a five-year randomized multi-center evaluation.</p>
            <script>var x = "ignore this script";</script>
        </main>
        <footer><p>Copyright 2026</p></footer>
    </body>
    </html>
    """
    parser = ArticleHTMLParser()
    parser.feed(mock_html)

    assert parser.meta_og_title == "Clinical Trial Shows 90% Efficacy for New Cancer Vaccine"
    assert parser.meta_author == "Dr. Sarah Jenkins"
    assert len(parser.paragraphs) == 2
    assert "Researchers at the medical institute" in parser.paragraphs[0]
    assert "The study was published in a leading" in parser.paragraphs[1]
    print("  ✓ ArticleHTMLParser extracted title, author, and paragraphs while ignoring nav/footer/scripts")
    print("  --> Web Article Parser Test: PASSED [100%]")


def test_2_predict_url_endpoint():
    """Verifies the POST /api/predict-url FastAPI endpoint with mocked HTTP response."""
    print("\n[Phase 13 Test 2/5] Verifying POST /api/predict-url Endpoint...")
    app.init_db()
    app.load_ai_engine()
    client = TestClient(app.app)

    # 1. Test empty URL returns 400
    r_empty = client.post("/api/predict-url", json={"url": ""})
    assert r_empty.status_code == 400
    print("  ✓ Empty URL validation returned 400 Bad Request")

    # 2. Test mock article scraping & inference
    mock_article = {
        "title": "NASA James Webb Telescope Captures Distant Star Merger",
        "text": "According to official data released by NASA, deep spectroscopic readings confirmed gravitational wave signatures.",
        "source": "nasa.gov",
        "author": "Dr. Sarah Jenkins",
        "url": "https://www.nasa.gov/news/article-123"
    }

    with patch("backend.app.extract_article_from_url", return_value=mock_article):
        r = client.post("/api/predict-url", json={"url": "https://www.nasa.gov/news/article-123"})
        assert r.status_code == 200
        res = r.json()
        assert res["prediction"] == "REAL"
        assert res["confidence_real"] >= 70.0
        assert "top_features" in res
        assert "token_attributions" in res
        print(f"  ✓ POST /api/predict-url: 200 OK (Verdict: {res['prediction']}, Real: {res['confidence_real']}%)")

    print("  --> Live URL Inference Endpoint Test: PASSED [100%]")


def test_3_chrome_extension_integrity():
    """Verifies Manifest V3 Chrome Extension configuration and asset presence."""
    print("\n[Phase 13 Test 3/5] Verifying Manifest V3 Chrome Browser Extension...")
    ext_dir = os.path.join(BASE_DIR, "extension")
    manifest_path = os.path.join(ext_dir, "manifest.json")
    popup_html_path = os.path.join(ext_dir, "popup.html")
    popup_js_path = os.path.join(ext_dir, "popup.js")
    popup_css_path = os.path.join(ext_dir, "popup.css")
    bg_js_path = os.path.join(ext_dir, "background.js")
    content_js_path = os.path.join(ext_dir, "content.js")

    assert os.path.exists(manifest_path), "manifest.json missing"
    assert os.path.exists(popup_html_path), "popup.html missing"
    assert os.path.exists(popup_js_path), "popup.js missing"
    assert os.path.exists(popup_css_path), "popup.css missing"
    assert os.path.exists(bg_js_path), "background.js missing"
    assert os.path.exists(content_js_path), "content.js missing"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["manifest_version"] == 3
    assert "activeTab" in manifest["permissions"]
    assert "contextMenus" in manifest["permissions"]
    assert manifest["action"]["default_popup"] == "popup.html"
    assert manifest["background"]["service_worker"] == "background.js"

    print(f"  ✓ Manifest V3 validated with permissions: {manifest['permissions']}")
    print(f"  ✓ Popup UI, Background Worker, and In-Page Content Script verified")
    print("  --> Chrome Extension Integrity Test: PASSED [100%]")


def test_4_docker_and_cicd_configuration():
    """Verifies Dockerfile, docker-compose.yml, and GitHub Actions CI workflow files."""
    print("\n[Phase 13 Test 4/5] Verifying Docker & CI/CD Deployment Files...")
    dockerfile_path = os.path.join(BASE_DIR, "Dockerfile")
    compose_path = os.path.join(BASE_DIR, "docker-compose.yml")
    ci_path = os.path.join(BASE_DIR, ".github", "workflows", "ci.yml")

    assert os.path.exists(dockerfile_path), "Dockerfile missing"
    assert os.path.exists(compose_path), "docker-compose.yml missing"
    assert os.path.exists(ci_path), "ci.yml missing"

    with open(dockerfile_path, "r", encoding="utf-8") as f:
        df_content = f.read()
    assert "FROM python:3.11-slim" in df_content
    assert "EXPOSE 8000" in df_content
    assert "CMD [\"python\", \"run.py\"]" in df_content

    with open(compose_path, "r", encoding="utf-8") as f:
        dc_content = f.read()
    assert "8000:8000" in dc_content
    assert "verishield-ai" in dc_content

    with open(ci_path, "r", encoding="utf-8") as f:
        ci_content = f.read()
    assert "actions/checkout" in ci_content
    assert "test_final_system_validation" in ci_content

    print("  ✓ Multi-stage Dockerfile, docker-compose.yml, and GitHub Actions CI verified")
    print("  --> Docker & CI/CD Configuration Test: PASSED [100%]")


def test_5_jupyter_notebook_structure():
    """Verifies that the interactive research Jupyter notebook is valid JSON and contains all walkthrough cells."""
    print("\n[Phase 13 Test 5/5] Verifying Research Jupyter Notebook...")
    nb_path = os.path.join(BASE_DIR, "notebooks", "01_verishield_full_pipeline_walkthrough.ipynb")
    assert os.path.exists(nb_path), "Jupyter notebook missing"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb
    assert len(nb["cells"]) >= 10, f"Expected at least 10 cells, found {len(nb['cells'])}"

    cell_types = [c["cell_type"] for c in nb["cells"]]
    assert "markdown" in cell_types
    assert "code" in cell_types

    print(f"  ✓ Notebook 01_verishield_full_pipeline_walkthrough.ipynb validated ({len(nb['cells'])} cells)")
    print("  --> Jupyter Research Notebook Test: PASSED [100%]")


def run_all_phase13_tests():
    """Master runner for Phase 13 Extensions."""
    print("=" * 85)
    print("🚀 VERISHIELD AI: PHASE 13 PRODUCTION EXTENSIONS & DEPLOYMENT TEST SUITE")
    print("🏛️ Vels Institute of Science, Technology & Advanced Studies (VISTAS)")
    print("👨‍💻 Team: Mohamed Irfaan M | Bhuvanesh M | Harish Hariharan MU")
    print("=" * 85)

    test_1_url_extractor_parsing()
    test_2_predict_url_endpoint()
    test_3_chrome_extension_integrity()
    test_4_docker_and_cicd_configuration()
    test_5_jupyter_notebook_structure()

    print("\n" + "=" * 85)
    print("🏆 ALL PHASE 13 ADVANCED PRODUCTION EXTENSION TESTS PASSED SUCCESSFULLY [100%]!")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    run_all_phase13_tests()
