"""
Iris Flower Classification – end-to-end
Run: python iris_ml.py
This script:
  1) Loads & explores the Iris dataset
  2) Trains multiple ML models with cross-validation
  3) Tunes the best model with GridSearchCV
  4) Evaluates on a test set
  5) Saves the best model to disk (best_iris_model.joblib)
  6) Shows simple predictions
"""

import warnings
warnings.filterwarnings("ignore")

# ---------- 1. Imports ----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Candidate models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import joblib
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------- 2. Load dataset ----------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

target_names = iris.target_names
feature_names = iris.feature_names

print("\n=== Basic Info ===")
print(f"Features: {feature_names}")
print(f"Targets: {list(target_names)}")
print(f"Shape: X={X.shape}, y={y.shape}\n")

# ---------- 3. Quick EDA ----------
print("=== Head ===")
print(X.head(), "\n")

print("=== Describe ===")
print(X.describe(), "\n")

print("=== Class balance ===")
print(y.value_counts().rename(index=lambda i: target_names[i]), "\n")

# Optional: visualizations (will pop up windows)
plt.figure()
sns.pairplot(pd.concat([X, y.map(dict(enumerate(target_names)))], axis=1),
             hue='species', diag_kind='hist')
plt.suptitle("Pairplot: Iris features by species", y=1.02)
plt.show()

plt.figure()
sns.heatmap(X.corr(numeric_only=True), annot=True, fmt=".2f")
plt.title("Feature correlation")
plt.show()

# ---------- 4. Train/Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ---------- 5. Build Pipelines ----------
# Note: trees/forests don't need scaling; others benefit from it.
pipelines = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ]),
    "DecisionTree": Pipeline([
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))
    ]),
    "RandomForest": Pipeline([
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
    ]),
}

# ---------- 6. Cross-Validation ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("=== 5-fold CV Accuracy (mean ± std) ===")
cv_results = {}
for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    cv_results[name] = (scores.mean(), scores.std())
    print(f"{name:>15}: {scores.mean():.4f} ± {scores.std():.4f}")

# Pick top model by CV mean
best_name = max(cv_results, key=lambda k: cv_results[k][0])
print(f"\nTop model after CV: {best_name}")

# ---------- 7. Hyperparameter Tuning (example grids) ----------
param_grids = {
    "LogisticRegression": {
        "clf__C": [0.1, 1, 5, 10],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"]
    },
    "SVM": {
        "clf__C": [0.1, 1, 5, 10],
        "clf__gamma": ["scale", 0.1, 0.01, 0.001],
        "clf__kernel": ["rbf"]
    },
    "KNN": {
        "clf__n_neighbors": [3, 5, 7, 9],
        "clf__weights": ["uniform", "distance"]
    },
    "DecisionTree": {
        "clf__max_depth": [None, 2, 3, 4, 5],
        "clf__criterion": ["gini", "entropy"]
    },
    "RandomForest": {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [None, 2, 3, 4, 5]
    },
}

grid = GridSearchCV(
    estimator=pipelines[best_name],
    param_grid=param_grids[best_name],
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)
grid.fit(X_train, y_train)

print(f"\nBest params for {best_name}: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_:.4f}")

best_model = grid.best_estimator_

# ---------- 8. Final Evaluation ----------
y_pred = best_model.predict(X_test)
print("\n=== Test Set Classification Report ===")
print(classification_report(y_test, y_pred, target_names=target_names))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(values_format="d")
plt.title("Confusion Matrix (Test Set)")
plt.show()

# ---------- 9. Save the trained model ----------
joblib.dump(best_model, "best_iris_model.joblib")
print("\nSaved model -> best_iris_model.joblib")

# ---------- 10. Try a quick prediction ----------
# Order must match feature_names
example = np.array([[5.1, 3.5, 1.4, 0.2]])  # likely Setosa
pred_idx = best_model.predict(example)[0]
print(f"\nExample features: {feature_names}")
print(f"Prediction for {example.tolist()[0]} -> {target_names[pred_idx]}")
