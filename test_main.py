"""
test/test_main.py — Automated tests for the FastAPI endpoints.
Run with: pytest test/test_main.py -v
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Mock DB and model before importing app ────────────────────────────────────
mock_model = MagicMock()
mock_model.predict.return_value = np.array(["setosa"])
mock_model.predict_proba.return_value = np.array([[0.9, 0.05, 0.05]])

with patch("mysql.connector.connect"), patch("joblib.load", return_value=mock_model), \
     patch("os.path.exists", return_value=True):
    from main import app

client = TestClient(app)

SAMPLE_RECORD = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
    "species": "setosa",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. TEST ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
class TestHealthEndpoint:
    def test_health_check_status_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_health_check_body(self):
        response = client.get("/")
        data = response.json()
        assert data["status"] == "ok"
        assert "message" in data


# ─────────────────────────────────────────────────────────────────────────────
# 2. CRUD ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
class TestCRUDEndpoints:

    def _mock_cursor(self, rows=None, lastrowid=1, rowcount=1):
        cursor = MagicMock()
        cursor.lastrowid = lastrowid
        cursor.rowcount = rowcount
        cursor.fetchall.return_value = rows or []
        cursor.fetchone.return_value = rows[0] if rows else None
        return cursor

    # CREATE
    def test_create_record_returns_201(self):
        cursor = self._mock_cursor()
        conn = MagicMock()
        conn.cursor.return_value = cursor
        with patch("main.get_connection", return_value=conn):
            response = client.post("/records/", json=SAMPLE_RECORD)
        assert response.status_code == 201

    def test_create_record_returns_id(self):
        cursor = self._mock_cursor(lastrowid=42)
        conn = MagicMock()
        conn.cursor.return_value = cursor
        with patch("main.get_connection", return_value=conn):
            response = client.post("/records/", json=SAMPLE_RECORD)
        assert response.json()["id"] == 42

    # READ ALL
    def test_list_records_returns_200(self):
        rows = [{**SAMPLE_RECORD, "id": 1}]
        cursor = self._mock_cursor(rows=rows)
        conn = MagicMock()
        conn.cursor.return_value = cursor
        with patch("main.get_connection", return_value=conn):
            response = client.get("/records/")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_records_empty(self):
        cursor = self._mock_cursor(rows=[])
        conn = MagicMock()
        conn.cursor.return_value = cursor
        with patch("main.get_connection", return_value=conn):
            response = client.get("/records/")
        assert response.json() == []

    # READ ONE
    def test_get_record_found(self):
        row = {**SAMPLE_RECORD, "id": 1}
        cursor = self._mock_cursor(rows=[row])
        conn = MagicMock()
        conn.cursor.return_value = cursor
        with patch("main.get_connection", return_value=conn):
            response = client.get("/records/1")
        assert response.status_code == 200
        assert response.json()["id"] == 1

    def test_get_record_not_found(self):
        cursor = self._mock_cursor(rows=[])
        conn = MagicMock()
        conn.cursor.return_value = cursor
        with patch("main.get_connection", return_value=conn):
            response = client.get("/records/999")
        assert response.status_code == 404

    # UPDATE
    def test_update_record_success(self):
        cursor = self._mock_cursor(rowcount=1)
        conn = MagicMock()
        conn.cursor.return_value = cursor
        with patch("main.get_connection", return_value=conn):
            response = client.put("/records/1", json=SAMPLE_RECORD)
        assert response.status_code == 200
        assert response.json()["id"] == 1

    def test_update_record_not_found(self):
        cursor = self._mock_cursor(rowcount=0)
        conn = MagicMock()
        conn.cursor.return_value = cursor
        with patch("main.get_connection", return_value=conn):
            response = client.put("/records/999", json=SAMPLE_RECORD)
        assert response.status_code == 404

    # DELETE
    def test_delete_record_success(self):
        cursor = self._mock_cursor(rowcount=1)
        conn = MagicMock()
        conn.cursor.return_value = cursor
        with patch("main.get_connection", return_value=conn):
            response = client.delete("/records/1")
        assert response.status_code == 200

    def test_delete_record_not_found(self):
        cursor = self._mock_cursor(rowcount=0)
        conn = MagicMock()
        conn.cursor.return_value = cursor
        with patch("main.get_connection", return_value=conn):
            response = client.delete("/records/999")
        assert response.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREDICT ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
class TestPredictEndpoint:
    PREDICT_PAYLOAD = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    def test_predict_returns_200(self):
        with patch("main.model", mock_model):
            response = client.post("/predict/", json=self.PREDICT_PAYLOAD)
        assert response.status_code == 200

    def test_predict_contains_prediction(self):
        with patch("main.model", mock_model):
            response = client.post("/predict/", json=self.PREDICT_PAYLOAD)
        data = response.json()
        assert "prediction" in data
        assert "probabilities" in data

    def test_predict_returns_valid_species(self):
        with patch("main.model", mock_model):
            response = client.post("/predict/", json=self.PREDICT_PAYLOAD)
        prediction = response.json()["prediction"]
        assert prediction in ["setosa", "versicolor", "virginica"]

    def test_predict_no_model_returns_503(self):
        with patch("main.model", None):
            response = client.post("/predict/", json=self.PREDICT_PAYLOAD)
        assert response.status_code == 503
