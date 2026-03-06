"""
train.py — Fetches data from MySQL, trains a RandomForest classifier
on the Iris dataset and saves the model as model.pkl.
"""
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import mysql.connector

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "mysql"),
    "user": os.getenv("DB_USER", "apiuser"),
    "password": os.getenv("DB_PASSWORD", "apipassword"),
    "database": os.getenv("DB_NAME", "apidb"),
}

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")


def fetch_data() -> pd.DataFrame:
    """Retrieve all records from the iris table."""
    conn = mysql.connector.connect(**DB_CONFIG)
    df = pd.read_sql("SELECT sepal_length, sepal_width, petal_length, petal_width, species FROM iris", conn)
    conn.close()
    return df


def build_dataset(df: pd.DataFrame):
    """Split DataFrame into features and target."""
    df = df.dropna(subset=["species"])
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
    y = df["species"].values
    return X, y


def train_model(X, y):
    """Train a RandomForestClassifier and print evaluation metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    return clf


def save_model(clf, path: str):
    joblib.dump(clf, path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    print("Fetching data from MySQL...")
    df = fetch_data()
    print(f"Loaded {len(df)} records.")

    X, y = build_dataset(df)
    print(f"Dataset built: {X.shape[0]} samples, {X.shape[1]} features.")

    clf = train_model(X, y)
    save_model(clf, MODEL_PATH)
