# 🚔 Crime Analysis ML Competition

## 📌 Project Description

Predict the **type of crime** based on time, location, and socio-economic features.  
Submit your predictions as a CSV and get automatically ranked on the leaderboard!

---

## 🗂️ Repository Structure

```
crime-analysis-ml/
├── .github/workflows/grade.yml   ← Auto-grades on every submission push
├── data/
│   ├── train.csv                 ← Training data with labels
│   ├── test.csv                  ← Test data (no labels)
│   └── test.txt                  ← Test set description
├── docs/
│   ├── index.html                ← Live leaderboard page
│   └── leaderboard.json          ← Auto-updated scores
├── grader/
│   └── grader.py                 ← Auto-grading script
├── models/
│   └── baseline_crime_model.pkl  ← Pretrained baseline model
├── starter_code/
│   ├── baseline.py               ← Starter code to get you going
│   └── requirements.txt
├── submission/
│   ├── .gitkeep
│   ├── submission.csv            ← Your submission goes here
│   └── results.csv               ← Auto-generated results
└── README.md
```

---

## 📊 Dataset

### Features (`train.csv` / `test.csv`)

| Column | Description |
|---|---|
| `hour_of_day` | Hour of incident (0–23) |
| `day_of_week` | Day (0=Monday, 6=Sunday) |
| `month` | Month (1–12) |
| `district` | Police district (1–10) |
| `population_density` | People per sq km |
| `unemployment_rate` | Local unemployment % |
| `poverty_rate` | Local poverty % |
| `temperature` | Temperature in °C |
| `is_weekend` | 1 if weekend, else 0 |
| `street_lights` | 1 if lights present, else 0 |
| `prior_incidents` | Prior incidents in area |

### Target

`crime_type` → one of: **Theft, Assault, Burglary, Vandalism, Robbery**

---

## 🚀 How to Participate

1. **Clone the repo**
   ```bash
   git clone https://github.com/YOUR_USERNAME/crime-analysis-ml.git
   cd crime-analysis-ml
   ```

2. **Install dependencies**
   ```bash
   pip install -r starter_code/requirements.txt
   ```

3. **Run the baseline**
   ```bash
   cd starter_code
   python baseline.py
   ```

4. **Submit your predictions**
   - Your submission must be a CSV with columns: `id`, `crime_type`
   - Save it as `submission/YourName.csv`
   - Push to GitHub — leaderboard updates automatically!

---

## 🏆 Leaderboard

Live leaderboard → [GitHub Pages](https://YOUR_USERNAME.github.io/crime-analysis-ml/)

Ranked by **Accuracy** on the hidden test set.

---

## 📌 Rules

- Only use `train.csv` for training
- Do **not** hardcode test labels
- One submission file per participant (`YourName.csv`)
- Submissions are graded automatically on push via GitHub Actions

---

## 👨‍💻 Author

Your Name  
Your College / Team
