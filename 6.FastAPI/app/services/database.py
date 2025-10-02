import os
import sqlite3
from datetime import datetime
from google.cloud import storage

storage_client = storage.Client()

def store_result_in_db_in_bucket(bucket_name: str, db_blob_path: str,
                                 video: str, model_type: str, model_filename: str,
                                 avg_stress: float, state_message: str):
    local_db_path = "/tmp/results.db"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(db_blob_path)
    
    if blob.exists():
        blob.download_to_filename(local_db_path)
    else:
        os.makedirs(os.path.dirname(local_db_path), exist_ok=True)
        conn = sqlite3.connect(local_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video TEXT,
                model_type TEXT,
                model_filename TEXT,
                avg_stress REAL,
                state TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    conn = sqlite3.connect(local_db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO results (video, model_type, model_filename, avg_stress, state, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (video, model_type, model_filename, avg_stress, state_message, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    
    blob.upload_from_filename(local_db_path)
