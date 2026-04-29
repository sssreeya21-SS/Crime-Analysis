print("STARTED")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
print("IMPORT DONE")

data = pd.read_csv("data/crime.csv")
print("DATA LOADED")

X = data.drop("target", axis=1)
y = data["target"]
print("SPLITTING")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("MODEL IS TRAINED")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

with open("results.txt", "w") as f:
    f.write(f"accuracy:{accuracy}\n")
    f.write(f"f1_score:{f1}\n")

print("DONE")
print("Accuracy:", accuracy)
print("F1:", f1)
