import pandas as pd
import os
import json
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime

SUBMISSION_DIR = "../submission"
GROUND_TRUTH   = "../data/test_with_labels.csv"   # hidden on server
DOCS_DIR       = "../docs"
LEADERBOARD    = os.path.join(DOCS_DIR, "leaderboard.json")

def load_ground_truth():
    df = pd.read_csv(GROUND_TRUTH)
    return df.set_index("id")["crime_type"]

def evaluate_submission(filepath, ground_truth):
    df = pd.read_csv(filepath)
    if "id" not in df.columns or "crime_type" not in df.columns:
        return None, "Missing required columns: id, crime_type"

    df = df.set_index("id")
    aligned = df.reindex(ground_truth.index)

    if aligned["crime_type"].isnull().any():
        return None, "Submission is missing some IDs"

    acc = accuracy_score(ground_truth, aligned["crime_type"])
    f1  = f1_score(ground_truth, aligned["crime_type"], average="weighted")
    return {"accuracy": round(acc, 4), "f1_score": round(f1, 4)}, None

def run_grader():
    if not os.path.exists(GROUND_TRUTH):
        print("⚠️  Ground truth file not found. Skipping grading.")
        return

    ground_truth = load_ground_truth()
    results = []

    for fname in os.listdir(SUBMISSION_DIR):
        if not fname.endswith(".csv") or fname in ("submission.csv", "results.csv"):
            continue
        fpath  = os.path.join(SUBMISSION_DIR, fname)
        name   = fname.replace(".csv", "")
        scores, error = evaluate_submission(fpath, ground_truth)
        if error:
            print(f"❌ {name}: {error}")
            continue
        scores["name"] = name
        scores["timestamp"] = datetime.utcnow().isoformat()
        results.append(scores)
        print(f"✅ {name} | Accuracy: {scores['accuracy']} | F1: {scores['f1_score']}")

    if not results:
        print("No valid submissions found.")
        return

    results.sort(key=lambda x: x["accuracy"], reverse=True)

    # Save leaderboard JSON for docs/index.html
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(LEADERBOARD, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Leaderboard saved to {LEADERBOARD}")

    # Save results CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(SUBMISSION_DIR, "results.csv"), index=False)
    print("✅ results.csv saved")

if __name__ == "__main__":
    run_grader()
