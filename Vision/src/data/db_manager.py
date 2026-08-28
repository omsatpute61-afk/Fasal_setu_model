import sqlite3
import datetime
from pathlib import Path

# Use project root for SQLite DB
DB_PATH = Path(__file__).parent.parent.parent / "agrivision.db"

def init_db():
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

def insert_scan(crop_name: str, disease: str, health_index: float):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO scans (timestamp, crop_name, disease, health_index)
        VALUES (?, ?, ?, ?)
    ''', (now, crop_name, disease, health_index))
    conn.commit()
    conn.close()

def get_recent_scans(limit: int = 10):
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
    
    # Convert to list of dicts for easy Pandas DataFrame rendering in Streamlit
    return [
        {
            "Time": row[0],
            "Crop": row[1],
            "Disease": row[2] if row[2] else "None",
            "Health": round(row[3], 1) if row[3] is not None else 0.0
        } for row in rows
    ]

# Initialize DB when module is imported
init_db()
