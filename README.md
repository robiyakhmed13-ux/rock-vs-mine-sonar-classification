# 🪨 Rock vs Mine Prediction — Sonar Signal Classification

> **25-Day ML Challenge | Day X — Project 1**
> A binary classification project using Logistic Regression to distinguish underwater rocks from naval mines based on sonar frequency readings.

---

## 📌 Project Overview

This project uses the **UCI Sonar Dataset** to train a Logistic Regression model that classifies sonar signals as either a **Rock (R)** or a **Mine (M)**. It is part of my 25-day Machine Learning challenge, where I build 2 projects per day, culminating in a large FinTech deployment.

---

## 🔄 Workflow

```
Sonar Data  →  Data Preprocessing  →  Train/Test Split  →  Logistic Regression Model
                                                                       ↓
                                              New Data  →  Trained Model  →  Rock / Mine
```

The workflow diagram below describes the full pipeline:

![Workflow](workflow.png)

| Step | Description |
|------|-------------|
| **Sonar Data** | 208 samples, each with 60 sonar frequency energy readings |
| **Data Preprocessing** | Label encoding — `R → 0` (Rock), `M → 1` (Mine) |
| **Train/Test Split** | 90% training / 10% test with stratified sampling |
| **Logistic Regression** | Fits a binary classifier on the 60-feature vectors |
| **Prediction** | Takes new sonar readings → outputs Rock or Mine |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | [UCI Machine Learning Repository — Sonar Dataset](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)) |
| Samples | 208 |
| Features | 60 (sonar frequency energy bands) |
| Classes | Rock (R) = 0, Mine (M) = 1 |
| Class Balance | 97 Rocks / 111 Mines |

Each of the 60 features represents the energy within a specific sonar frequency band, measured by bouncing sonar signals off either underwater rocks or metal cylinders (simulated mines).

---

## 🧠 Model

**Algorithm:** Logistic Regression (sklearn)

Logistic Regression is a linear classifier that models the probability of a binary outcome. It applies the sigmoid function to a linear combination of input features:

```
P(Mine) = σ(w₁x₁ + w₂x₂ + ... + w₆₀x₆₀ + b)
```

Despite being a simple algorithm, it achieves solid accuracy on this dataset because the sonar frequency patterns of rocks and mines are meaningfully separable in the feature space.

---

## 📈 Results

| Dataset | Accuracy |
|---------|----------|
| Training | ~83.4% |
| Test | ~81.0% |

---

## 🗂️ Project Structure

```
rock-vs-mine-prediction/
│
├── rock_vs_mine_prediction.py   # Clean, modular Python script
├── Rock_vs_Mine_prediction.ipynb # Original Jupyter Notebook (exploration)
├── Sonar.csv                    # Dataset (download separately)
├── workflow.png                 # Pipeline diagram
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/rock-vs-mine-prediction.git
cd rock-vs-mine-prediction
```

### 2. Install dependencies
```bash
pip install numpy pandas scikit-learn
```

### 3. Download the dataset
Get `Sonar.csv` from the [UCI Repository](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)) and place it in the project root.

### 4. Run the script
```bash
python rock_vs_mine_prediction.py
```

### 5. Make a custom prediction
Edit the `sample_input` tuple at the bottom of the script with your own 60 sonar values, or call the `predict()` function directly:

```python
from rock_vs_mine_prediction import load_data, preprocess, split_data, train_model, predict

df = load_data("Sonar.csv")
X, y = preprocess(df)
X_train, X_test, y_train, y_test = split_data(X, y)
model = train_model(X_train, y_train)

my_reading = (0.03, 0.05, ...)  # 60 values
result = predict(model, my_reading)
print(result)  # "Rock" or "Mine"
```

---

## 🛠️ Tech Stack

- Python 3.12
- NumPy
- Pandas
- Scikit-learn

---

## 📅 25-Day ML Challenge

This is part of my personal challenge to build **2 ML projects per day for 25 days**. Each project targets a different algorithm or domain, building toward a final FinTech application that combines multiple models.

| Day | Projects |
|-----|----------|
| 1 | Diabetes Prediction, **Rock vs Mine Prediction** ← you are here |
| ... | ... |
| 25 | FinTech Model Deployment |

---

## 📬 Connect

Feel free to star ⭐ the repo if you found it useful, and follow along as I complete the challenge!
