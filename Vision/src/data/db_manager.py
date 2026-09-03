import sqlite3
import datetime
from pathlib import Path
from typing import Any, Dict, List

# Use project root for SQLite DB
DB_PATH: Path = Path(__file__).parent.parent.parent / "agrivision.db"

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            crop_name TEXT NOT NULL,
            disease TEXT,
            health_index REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def insert_scan(crop_name: str, disease: str, health_index: float) -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO scans (timestamp, crop_name, disease, health_index)
        VALUES (?, ?, ?, ?)
    ''', (now, crop_name, disease, health_index))
    conn.commit()
    conn.close()

def get_recent_scans(limit: int = 10) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT timestamp, crop_name, disease, health_index 
        FROM scans 
        ORDER BY id DESC 
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    # We cast to float and str to ensure Pylance is happy
    results: List[Dict[str, Any]] = []
    for row in rows:
        results.append({
            "Time": str(row[0]),
            "Crop": str(row[1]),
            "Disease": str(row[2]) if row[2] else "None",
            "Health": round(float(row[3]), 1) if row[3] is not None else 0.0
        })
    return results

# Initialize DB when module is imported
init_db()
