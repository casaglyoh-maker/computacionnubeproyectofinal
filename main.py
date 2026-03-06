from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import mysql.connector
import joblib
import numpy as np
import os

app = FastAPI(title="ML API", version="1.0.0")

# ── DB config ────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "mysql"),
    "user": os.getenv("DB_USER", "apiuser"),
    "password": os.getenv("DB_PASSWORD", "apipassword"),
    "database": os.getenv("DB_NAME", "apidb"),
}

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

# ── Load ML model ─────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

# ── Schemas ───────────────────────────────────────────────────────────────────
class IrisRecord(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: Optional[str] = None

class IrisRecordOut(IrisRecord):
    id: int

class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# ── Test endpoint ─────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "ok", "message": "API is running"}

# ── CREATE ────────────────────────────────────────────────────────────────────
@app.post("/records/", response_model=IrisRecordOut, status_code=201)
def create_record(record: IrisRecord):
    conn = get_connection()
    cursor = conn.cursor()
    sql = """INSERT INTO iris (sepal_length, sepal_width, petal_length, petal_width, species)
             VALUES (%s, %s, %s, %s, %s)"""
    cursor.execute(sql, (record.sepal_length, record.sepal_width,
                         record.petal_length, record.petal_width, record.species))
    conn.commit()
    new_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return IrisRecordOut(id=new_id, **record.dict())

# ── READ ALL ──────────────────────────────────────────────────────────────────
@app.get("/records/", response_model=List[IrisRecordOut])
def list_records():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM iris")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [IrisRecordOut(**row) for row in rows]

# ── READ ONE ──────────────────────────────────────────────────────────────────
@app.get("/records/{record_id}", response_model=IrisRecordOut)
def get_record(record_id: int):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM iris WHERE id = %s", (record_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Record not found")
    return IrisRecordOut(**row)

# ── UPDATE ────────────────────────────────────────────────────────────────────
@app.put("/records/{record_id}", response_model=IrisRecordOut)
def update_record(record_id: int, record: IrisRecord):
    conn = get_connection()
    cursor = conn.cursor()
    sql = """UPDATE iris SET sepal_length=%s, sepal_width=%s,
             petal_length=%s, petal_width=%s, species=%s WHERE id=%s"""
    cursor.execute(sql, (record.sepal_length, record.sepal_width,
                         record.petal_length, record.petal_width,
                         record.species, record_id))
    if cursor.rowcount == 0:
        cursor.close(); conn.close()
        raise HTTPException(status_code=404, detail="Record not found")
    conn.commit()
    cursor.close()
    conn.close()
    return IrisRecordOut(id=record_id, **record.dict())

# ── DELETE ────────────────────────────────────────────────────────────────────
@app.delete("/records/{record_id}")
def delete_record(record_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM iris WHERE id = %s", (record_id,))
    if cursor.rowcount == 0:
        cursor.close(); conn.close()
        raise HTTPException(status_code=404, detail="Record not found")
    conn.commit()
    cursor.close()
    conn.close()
    return {"message": f"Record {record_id} deleted"}

# ── PREDICT ───────────────────────────────────────────────────────────────────
@app.post("/predict/")
def predict(data: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")
    features = np.array([[data.sepal_length, data.sepal_width,
                          data.petal_length, data.petal_width]])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0].tolist()
    return {"prediction": prediction, "probabilities": proba}
