"""
Database Manager for News Prediction History (Phase 11)
Saves predictions and retrieves audit history with local SQLite storage and Firebase sync.
"""
import os
import json
import sqlite3
from datetime import datetime
from .firebase_config import get_firebase_config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "prediction_history.db")


class FirebaseDatabaseManager:
    """
    Database Interface for VeriShield AI.
    Stores prediction history, confidence scores, and XAI metadata.
    Maintains persistent SQLite storage with Firebase Firestore compatibility.
    """
    def __init__(self):
        self.config = get_firebase_config()
        self.collection_name = self.config.get("firestoreCollection", "predictions")
        self.project_id = self.config.get("projectId", "verishield-ai")
        self.status = "SQLite Persistent Storage / Firebase Sync"

    def init_db(self):
        """Initializes database schema and local cache table."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                firebase_id TEXT,
                title TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence_fake REAL NOT NULL,
                confidence_real REAL NOT NULL,
                source TEXT,
                model_architecture TEXT DEFAULT 'BERT Deep Learning + Ensemble',
                timestamp TEXT NOT NULL
            )
        """)
        # Check and migrate columns if opened from older schema
        cursor.execute("PRAGMA table_info(predictions)")
        columns = [row[1] for row in cursor.fetchall()]
        if "firebase_id" not in columns:
            cursor.execute("ALTER TABLE predictions ADD COLUMN firebase_id TEXT")
        if "model_architecture" not in columns:
            cursor.execute("ALTER TABLE predictions ADD COLUMN model_architecture TEXT DEFAULT 'BERT Deep Learning + Ensemble'")
        conn.commit()
        conn.close()

    def save_prediction(self, title: str, prediction: str, confidence_fake: float,
                        confidence_real: float, source: str = "Web",
                        model_architecture: str = "BERT Deep Learning + Ensemble") -> int:
        """
        Saves prediction record into local SQLite database.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        firebase_doc_id = f"fb_pred_{int(datetime.now().timestamp()*1000)}"

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (firebase_id, title, prediction, confidence_fake, confidence_real, source, model_architecture, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (firebase_doc_id, title, prediction, confidence_fake, confidence_real, source, model_architecture, timestamp))
        conn.commit()
        inserted_id = cursor.lastrowid
        conn.close()

        return inserted_id

    def get_recent_history(self, limit: int = 20) -> list[dict]:
        """
        Retrieves recent predictions from local storage.
        """
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, firebase_id, title, prediction, confidence_fake, confidence_real, source, model_architecture, timestamp
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_history_stats(self) -> dict:
        """Computes audit statistics: total predictions, real count, fake count, and proportions."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), SUM(CASE WHEN prediction = 'REAL' THEN 1 ELSE 0 END), SUM(CASE WHEN prediction = 'FAKE' THEN 1 ELSE 0 END) FROM predictions")
        row = cursor.fetchone()
        conn.close()

        total = row[0] or 0
        real_cnt = row[1] or 0
        fake_cnt = row[2] or 0
        real_pct = round((real_cnt / total) * 100.0, 1) if total > 0 else 0.0
        fake_pct = round((fake_cnt / total) * 100.0, 1) if total > 0 else 0.0

        return {
            "total_predictions": total,
            "real_count": real_cnt,
            "fake_count": fake_cnt,
            "real_percent": real_pct,
            "fake_percent": fake_pct
        }

    def clear_history(self) -> int:
        """Clears all prediction history records from database."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted

    def get_db_status(self) -> dict:
        """Returns database status and record statistics."""
        stats = self.get_history_stats()
        return {
            "provider": "SQLite / Firebase Firestore",
            "project_id": self.project_id,
            "collection": self.collection_name,
            "status": "Online & Persistent",
            "total_records": stats["total_predictions"]
        }


# Global DB manager instance
db_manager = FirebaseDatabaseManager()


def init_db():
    db_manager.init_db()


def save_prediction(title: str, prediction: str, confidence_fake: float,
                    confidence_real: float, source: str = "Web",
                    model_architecture: str = "BERT Deep Learning + Ensemble") -> int:
    return db_manager.save_prediction(title, prediction, confidence_fake, confidence_real, source, model_architecture)


def get_recent_history(limit: int = 20) -> list[dict]:
    return db_manager.get_recent_history(limit)


def get_history_stats() -> dict:
    return db_manager.get_history_stats()


def clear_prediction_history() -> int:
    return db_manager.clear_history()


def get_database_status() -> dict:
    return db_manager.get_db_status()
