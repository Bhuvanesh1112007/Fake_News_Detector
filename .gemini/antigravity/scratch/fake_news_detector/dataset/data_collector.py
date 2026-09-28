"""
Data Collection & Ingestion Pipeline: Kaggle & FakeNewsNet Integration
Provides loaders, parsers, and scrapers for Kaggle Fake News datasets and FakeNewsNet (PolitiFact & BuzzFeed).
"""
import os
import csv
import json
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
BENCHMARK_PATH = os.path.join(DATASET_DIR, "fake_or_real_news.csv")

class KaggleFakeNewsCollector:
    """
    Ingests and standardizes Kaggle Fake News datasets (e.g. McIntire dataset, WELFake, ISOT).
    Maps column schemas: title, text, label (REAL/FAKE), source, author, shares, sentiment.
    """
    def __init__(self, raw_csv_path: str = None):
        self.raw_csv_path = raw_csv_path

    def load_and_standardize(self, filepath: str = None) -> List[Dict]:
        target_path = filepath or self.raw_csv_path
        if not target_path or not os.path.exists(target_path):
            return []
        
        standardized = []
        with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = row.get("title") or row.get("headline") or ""
                text = row.get("text") or row.get("body") or row.get("content") or ""
                raw_label = row.get("label") or row.get("category") or "0"
                
                # Normalize label to REAL / FAKE
                if str(raw_label).strip().upper() in ["FAKE", "1", "FALSE"]:
                    label = "FAKE"
                else:
                    label = "REAL"
                    
                if title.strip() and text.strip():
                    standardized.append({
                        "title": title.strip(),
                        "text": text.strip(),
                        "label": label,
                        "source": row.get("source", "Kaggle Dataset"),
                        "author": row.get("author", "Kaggle Contributor"),
                        "shares": float(row.get("shares", 2500)),
                        "sentiment": float(row.get("sentiment", 0.0))
                    })
        return standardized

class FakeNewsNetCollector:
    """
    Ingests and standardizes FakeNewsNet (PolitiFact & BuzzFeed news repositories).
    Ground truth fact-checks from professional fact-checking organizations.
    """
    def __init__(self, data_root: str = None):
        self.data_root = data_root

    def load_from_json(self, json_path: str) -> List[Dict]:
        if not os.path.exists(json_path):
            return []
        
        with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)

        standardized = []
        for item in data:
            title = item.get("title", "")
            text = item.get("text", "") or item.get("body", "")
            label = "FAKE" if "fake" in str(item.get("label", "")).lower() else "REAL"
            
            if title.strip() and text.strip():
                standardized.append({
                    "title": title.strip(),
                    "text": text.strip(),
                    "label": label,
                    "source": item.get("news_source", "FakeNewsNet (PolitiFact/BuzzFeed)"),
                    "author": item.get("author", "FactCheck Reporter"),
                    "shares": float(item.get("twitter_shares", 5000)),
                    "sentiment": float(item.get("sentiment", 0.0))
                })
        return standardized

def merge_and_export_dataset(kaggle_data: List[Dict], fakenewsnet_data: List[Dict], output_path: str = BENCHMARK_PATH):
    """Combines datasets, cleans duplicates, and exports standardized CSV."""
    combined = kaggle_data + fakenewsnet_data
    seen_titles = set()
    unique_data = []

    for item in combined:
        title_key = item["title"].lower().strip()
        if title_key not in seen_titles and len(item["text"]) > 20:
            seen_titles.add(title_key)
            unique_data.append(item)

    fieldnames = ["title", "text", "label", "source", "author", "shares", "sentiment"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in unique_data:
            writer.writerow(row)

    print(f"Exported merged Kaggle & FakeNewsNet dataset with {len(unique_data)} articles to {output_path}")
    return unique_data
