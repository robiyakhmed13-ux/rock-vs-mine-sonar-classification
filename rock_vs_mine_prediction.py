# ============================================================
#  Rock vs Mine Prediction using Logistic Regression
#  Dataset : UCI Sonar Dataset (208 samples, 60 features)
#  Author  : (your name here)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ----------------------------------------------------------
# 1. DATA COLLECTION
# ----------------------------------------------------------
def load_data(filepath: str) -> pd.DataFrame:
    """Load the Sonar CSV dataset (no header row)."""
    sonar_data = pd.read_csv(filepath, header=None)
    # Encode label column: R (Rock) -> 0, M (Mine) -> 1
    sonar_data[60] = sonar_data[60].map({"M": 1, "R": 0})
    return sonar_data


# ----------------------------------------------------------
# 2. DATA EXPLORATION  (optional — comment out in production)
# ----------------------------------------------------------
def explore_data(df: pd.DataFrame) -> None:
    print("=== Dataset Shape ===")
    print(df.shape)                         # (208, 61)

    print("\n=== First 5 Rows ===")
    print(df.head())

    print("\n=== Statistical Summary ===")
    print(df.describe())

    print("\n=== Class Distribution ===")
    print(df[60].value_counts())            # 0 = Rock, 1 = Mine

    print("\n=== Mean values per class ===")
    print(df.groupby(60).mean(numeric_only=True))


# ----------------------------------------------------------
# 3. DATA PREPROCESSING
# ----------------------------------------------------------
def preprocess(df: pd.DataFrame):
    """Split features (X) and target (y)."""
    X = df.iloc[:, :60]   # 60 sonar frequency features
    y = df.iloc[:, 60]    # binary label: 0 = Rock, 1 = Mine
    return X, y


# ----------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ----------------------------------------------------------
def split_data(X, y, test_size: float = 0.1, random_state: int = 1):
    """
    Stratified split to preserve class balance.
    Default: 90% train, 10% test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    print(f"Total: {X.shape[0]} | Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------
# 5. MODEL TRAINING
# ----------------------------------------------------------
def train_model(X_train, y_train) -> LogisticRegression:
    """Train a Logistic Regression classifier."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


# ----------------------------------------------------------
# 6. MODEL EVALUATION
# ----------------------------------------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test) -> None:
    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc  = accuracy_score(y_test,  test_preds)

    print(f"\n=== Model Accuracy ===")
    print(f"Training Accuracy : {train_acc:.4f}  ({train_acc*100:.2f}%)")
    print(f"Test Accuracy     : {test_acc:.4f}  ({test_acc*100:.2f}%)")


# ----------------------------------------------------------
# 7. PREDICTION SYSTEM
# ----------------------------------------------------------
def predict(model, input_data: tuple) -> str:
    """
    Predict whether a sonar reading is a Rock or a Mine.

    Parameters
    ----------
    model      : trained LogisticRegression model
    input_data : tuple of 60 float values (sonar frequency readings)

    Returns
    -------
    str : "Rock" or "Mine"
    """
    arr = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(arr)
    return "Rock" if prediction[0] == 0 else "Mine"


# ----------------------------------------------------------
# 8. MAIN PIPELINE
# ----------------------------------------------------------
if __name__ == "__main__":

    # ── Path to the Sonar dataset ──────────────────────────
    DATA_PATH = "Sonar.csv"   # update path if needed

    # ── Load ───────────────────────────────────────────────
    df = load_data(DATA_PATH)

    # ── Explore (optional) ─────────────────────────────────
    explore_data(df)

    # ── Preprocess ─────────────────────────────────────────
    X, y = preprocess(df)

    # ── Split ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ── Train ──────────────────────────────────────────────
    model = train_model(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────
    evaluate_model(model, X_train, y_train, X_test, y_test)

    # ── Example Prediction ─────────────────────────────────
    sample_input = (
        0.0388, 0.0324, 0.0688, 0.0898, 0.1267, 0.1515, 0.2134, 0.2613,
        0.2832, 0.2718, 0.3645, 0.3934, 0.3843, 0.4677, 0.5364, 0.4823,
        0.4835, 0.5862, 0.7579, 0.6997, 0.6918, 0.8633, 0.9107, 0.9346,
        0.7884, 0.8585, 0.9261, 0.7080, 0.5779, 0.5215, 0.4505, 0.3129,
        0.1448, 0.1046, 0.1820, 0.1519, 0.1017, 0.1438, 0.1986, 0.2039,
        0.2778, 0.2879, 0.1331, 0.1140, 0.1310, 0.1433, 0.0624, 0.0100,
        0.0098, 0.0131, 0.0152, 0.0255, 0.0071, 0.0263, 0.0079, 0.0111,
        0.0107, 0.0068, 0.0097, 0.0067,
    )

    result = predict(model, sample_input)
    print(f"\n=== Prediction Result ===")
    print(f"The object is a : {result}")
