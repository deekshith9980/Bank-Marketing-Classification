
import pandas as pd
import numpy as np
import joblib
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

NUMERICAL_FEATURES = [
    "age", "balance", "day",
    "campaign", "pdays", "previous"
]

CATEGORICAL_FEATURES = [
    "job", "marital", "education",
    "default", "housing", "loan",
    "contact", "month", "poutcome"
]

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

TARGET = "y"

MODEL_DIR = "model"
RESULT_DIR = "results"
CM_DIR = os.path.join(RESULT_DIR, "confusion_matrices")

os.makedirs(CM_DIR, exist_ok=True)


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("Loading test dataset...")

test_data = pd.read_csv("test_data.csv")

X_test = test_data[FEATURES]
y_test = test_data[TARGET]

print("Test dataset shape:", test_data.shape)


# ============================================================
# 3. LOAD TRAINED MODELS
# ============================================================

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}


# ============================================================
# 4. EVALUATE MODELS
# ============================================================

results = []

for model_name, filename in model_files.items():

    print("\nEvaluating:", model_name)

    model_path = os.path.join(MODEL_DIR, filename)

    # Load trained model
    model = joblib.load(model_path)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate prediction scores for AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(y_test, y_score)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc
    })

    # Display classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["No", "Yes"],
            zero_division=0
        )
    )

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"]
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.tight_layout()

    image_name = (
        model_name.lower()
        .replace(" ", "_")
        + ".png"
    )

    plt.savefig(
        os.path.join(CM_DIR, image_name),
        dpi=150
    )

    plt.close()


# ============================================================
# 5. CREATE MODEL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(results)

# Round metrics for readability
metric_columns = [
    "Accuracy", "AUC", "Precision",
    "Recall", "F1", "MCC"
]

comparison[metric_columns] = (
    comparison[metric_columns].round(4)
)

print("\n========================================")
print("MODEL PERFORMANCE COMPARISON")
print("========================================")

print(comparison.to_string(index=False))


# ============================================================
# 6. IDENTIFY WINNING MODEL
# ============================================================

# Use F1 score as the primary selection metric
winner = comparison.loc[
    comparison["F1"].idxmax()
]

print("\nOverall Winner (based on F1 Score):")
print(winner["Model"])

print("F1 Score:", winner["F1"])


# ============================================================
# 7. SAVE RESULTS
# ============================================================

os.makedirs(RESULT_DIR, exist_ok=True)

comparison.to_csv(
    os.path.join(RESULT_DIR, "model_comparison.csv"),
    index=False
)

print("\nResults saved successfully!")
print("Comparison: results/model_comparison.csv")
print("Confusion matrices:", CM_DIR)
