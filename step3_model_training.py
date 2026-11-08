
import pandas as pd
import joblib
import os
import time

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

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

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

print("Loading training dataset...")

train_data = pd.read_csv("train_data.csv")

X_train = train_data[FEATURES]
y_train = train_data[TARGET]

print("Training data shape:", X_train.shape)
print("Target shape:", y_train.shape)


# ============================================================
# 3. CREATE PREPROCESSING PIPELINE
# ============================================================

numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, NUMERICAL_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ]
)


# ============================================================
# 4. DEFINE CLASSIFICATION MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Naive Bayes": GaussianNB(),

    "Support Vector Machine": LinearSVC(
        max_iter=10000,
        random_state=RANDOM_STATE
    )
}


# ============================================================
# 5. TRAIN AND SAVE MODELS
# ============================================================

trained_models = {}

print("\nStarting model training...\n")

for model_name, classifier in models.items():

    print(f"Training {model_name}...")

    start_time = time.time()

    # Combine preprocessing and classifier
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    # Train model
    pipeline.fit(X_train, y_train)

    # Calculate training time
    training_time = time.time() - start_time

    # Generate filename
    filename = (
        model_name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    filepath = os.path.join(MODEL_DIR, filename)

    # Save complete pipeline
    joblib.dump(pipeline, filepath)

    trained_models[model_name] = pipeline

    print(f"{model_name} trained successfully!")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Saved at: {filepath}\n")


# ============================================================
# 6. TRAINING SUMMARY
# ============================================================

print("=" * 50)
print("ALL MODELS TRAINED SUCCESSFULLY")
print("=" * 50)

print("\nSaved Models:")

for model_name in trained_models:
    print("-", model_name)

print("\nModel directory:", MODEL_DIR)
