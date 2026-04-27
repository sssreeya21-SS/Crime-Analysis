import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ── Load Data ──────────────────────────────────────────────────────────────────
train = pd.read_csv("../data/train.csv")
test  = pd.read_csv("../data/test.csv")

FEATURES = [
    "hour_of_day", "day_of_week", "month", "district",
    "population_density", "unemployment_rate", "poverty_rate",
    "temperature", "is_weekend", "street_lights", "prior_incidents"
]
TARGET = "crime_type"

X_train = train[FEATURES]
y_train = train[TARGET]
X_test  = test[FEATURES]

# ── Encode Labels ──────────────────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y_train)

# ── Train Baseline Model ───────────────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_encoded)

# ── Save Model ─────────────────────────────────────────────────────────────────
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/baseline_crime_model.pkl")
print("✅ Baseline model saved to models/baseline_crime_model.pkl")

# ── Predict & Save Submission ──────────────────────────────────────────────────
preds        = model.predict(X_test)
pred_labels  = le.inverse_transform(preds)

submission = pd.DataFrame({"id": test["id"], "crime_type": pred_labels})
os.makedirs("../submission", exist_ok=True)
submission.to_csv("../submission/submission.csv", index=False)
print("✅ Baseline submission saved to submission/submission.csv")
print(submission)
