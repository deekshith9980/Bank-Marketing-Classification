
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

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


# ============================================================
# 2. PREPROCESS DATASET
# ============================================================

def preprocess_dataset(df):

    print("\nStarting preprocessing...")

    # Remove duplicate records
    duplicates = df.duplicated().sum()

    print("Duplicate records:", duplicates)

    df = df.drop_duplicates().copy()

    # Select required features
    # Exclude duration to prevent data leakage
    df = df[FEATURES + [TARGET]].copy()

    # Encode target
    df[TARGET] = df[TARGET].map({
        "no": 0,
        "yes": 1
    })

    # Separate features and target
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)

    return X, y


# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

def split_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\nTraining target distribution:")
    print(y_train.value_counts(normalize=True) * 100)

    print("\nTesting target distribution:")
    print(y_test.value_counts(normalize=True) * 100)

    return X_train, X_test, y_train, y_test


# ============================================================
# 4. EXPORT TEST DATA AND TRAIN DATA
# ============================================================

def export_test_data(X_test, y_test):

    test_data = X_test.copy()
    test_data[TARGET] = y_test

    test_data.to_csv("test_data.csv", index=False)

    print("\ntest_data.csv exported successfully!")
    print("Test data shape:", test_data.shape)


def export_train_data(X_train, y_train):

    train_data = X_train.copy()
    train_data[TARGET] = y_train

    train_data.to_csv("train_data.csv", index=False)

    print("train_data.csv exported successfully!")


# ============================================================
# 5. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    # Import dataset loader from Step 1
    from step1_dataset import load_dataset

    # Load dataset
    df = load_dataset()

    # Preprocessing
    X, y = preprocess_dataset(df)

    # Train-test split
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Export test data and train data
    export_test_data(X_test, y_test)
    export_train_data(X_train, y_train)

    print("\n===================================")
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("===================================")

    print("Input features:", len(FEATURES))
    print("Numerical features:", len(NUMERICAL_FEATURES))
    print("Categorical features:", len(CATEGORICAL_FEATURES))
