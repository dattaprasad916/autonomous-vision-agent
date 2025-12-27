import sqlite3

# Create / connect to database
conn = sqlite3.connect("detections.db", check_same_thread=False)
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT,
    label TEXT,
    confidence REAL,
    inference_time INTEGER
)
""")

conn.commit()


def insert_detection(model, label, confidence, inference_time):
    cursor.execute(
        "INSERT INTO detections (model, label, confidence, inference_time) VALUES (?, ?, ?, ?)",
        (model, label, confidence, inference_time)
    )
    conn.commit()


def fetch_all():
    cursor.execute(
        "SELECT model, label, confidence, inference_time FROM detections"
    )
    return cursor.fetchall()
