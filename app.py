import hashlib
import io
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Analytics | ML Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. CUSTOM CSS - BANKING THEME
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #F4F7FB;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        section[data-testid="stSidebar"] {
            background-color: #14243A;
            border-right: 1px solid #263B52;
        }

        section[data-testid="stSidebar"] * {
            color: #E8EEF5;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 10px;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            background-color: #1C3049;
            padding: 12px 14px;
            border-radius: 8px;
            border: 1px solid #30445C;
            transition: all 0.2s ease;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background-color: #28445D;
        }

        h1, h2, h3 {
            color: #14243A;
        }

        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E5EAF0;
            border-radius: 12px;
            padding: 18px;
            box-shadow: 0 2px 7px rgba(20, 36, 58, 0.04);
        }

        div[data-testid="stMetricLabel"] {
            color: #64748B;
            font-size: 13px;
        }

        div[data-testid="stMetricValue"] {
            color: #14243A;
            font-weight: 700;
        }

        .dashboard-card {
            background-color: #FFFFFF;
            border: 1px solid #E5EAF0;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 12px;
            box-shadow: 0 2px 7px rgba(20, 36, 58, 0.04);
        }

        .page-subtitle {
            color: #64748B;
            font-size: 14px;
            margin-top: -12px;
            margin-bottom: 25px;
        }

        .stButton > button[kind="primary"] {
            background-color: #168F83;
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: 600;
        }

        .stButton > button[kind="primary"]:hover {
            background-color: #11776D;
            color: white;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #E5EAF0;
            border-radius: 10px;
        }

        [data-testid="stFileUploader"] {
            background-color: #FFFFFF;
            border: 1px dashed #9AAFC1;
            border-radius: 10px;
            padding: 10px;
        }

        hr {
            border-color: #E2E8F0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. PROJECT CONFIGURATION
# ============================================================

NUMERICAL_FEATURES = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = "y"
MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


# ============================================================
# 4. SESSION STATE
# ============================================================

# Keep a copy of the uploaded file bytes separately from the uploader
# widget. This preserves the dataset when the user changes pages.
if "uploaded_dataset_bytes" not in st.session_state:
    st.session_state["uploaded_dataset_bytes"] = None

if "uploaded_dataset_name" not in st.session_state:
    st.session_state["uploaded_dataset_name"] = None

if "comparison_results" not in st.session_state:
    st.session_state["comparison_results"] = None

if "comparison_dataset_hash" not in st.session_state:
    st.session_state["comparison_dataset_hash"] = None


def save_uploaded_dataset():
    """Copy the uploaded file into session state before page navigation."""
    uploaded = st.session_state.get("dataset_uploader")

    if uploaded is None:
        st.session_state["uploaded_dataset_bytes"] = None
        st.session_state["uploaded_dataset_name"] = None
        st.session_state["comparison_results"] = None
        st.session_state["comparison_dataset_hash"] = None
    else:
        st.session_state["uploaded_dataset_bytes"] = uploaded.getvalue()
        st.session_state["uploaded_dataset_name"] = uploaded.name
        # Results from a previous dataset must not be reused.
        st.session_state["comparison_results"] = None
        st.session_state["comparison_dataset_hash"] = None


# ============================================================
# 5. MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():
    loaded_models = {}

    for model_name, filename in MODEL_FILES.items():
        filepath = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Model file not found: {filepath}. "
                "Make sure the model folder is present beside app.py."
            )

        loaded_models[model_name] = joblib.load(filepath)

    return loaded_models


try:
    models = load_models()
except Exception as error:
    st.error(f"Unable to load the trained models: {error}")
    st.stop()


# ============================================================
# 6. EVALUATION FUNCTIONS
# ============================================================

def evaluate_model(model, features, target):
    predictions = model.predict(features)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(features)
    else:
        raise ValueError(
            "This model does not provide probabilities or decision scores "
            "required to calculate AUC."
        )

    metrics = {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": roc_auc_score(target, scores),
        "Precision": precision_score(
            target, predictions, zero_division=0
        ),
        "Recall": recall_score(
            target, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            target, predictions, zero_division=0
        ),
        "MCC": matthews_corrcoef(target, predictions),
    }

    cm = confusion_matrix(target, predictions, labels=[0, 1])

    report = classification_report(
        target,
        predictions,
        labels=[0, 1],
        target_names=["No", "Yes"],
        output_dict=True,
        zero_division=0,
    )

    return metrics, cm, report


@st.cache_data(show_spinner=False)
def evaluate_selected_model(model_name, csv_bytes):
    data = pd.read_csv(io.BytesIO(csv_bytes))
    features = data[FEATURES]
    target = data[TARGET].astype(int)

    metrics, cm, report = evaluate_model(
        load_models()[model_name],
        features,
        target,
    )

    return metrics, cm, report


@st.cache_data(show_spinner=False)
def compare_all_models(csv_bytes):
    data = pd.read_csv(io.BytesIO(csv_bytes))
    features = data[FEATURES]
    target = data[TARGET].astype(int)

    results = []

    for model_name, model in load_models().items():
        metrics, _, _ = evaluate_model(model, features, target)
        metrics["Model"] = model_name
        results.append(metrics)

    comparison = pd.DataFrame(results)
    return comparison.set_index("Model")


# ============================================================
# 7. DATASET VALIDATION
# ============================================================

def validate_dataset(csv_bytes):
    """
    Validate a Bank Marketing test CSV.

    Required:
      - The 15 model input features
      - Target column y with both classes (yes/no or 0/1)

    Extra columns are ignored. Missing feature values are allowed because
    the saved preprocessing pipelines contain imputers.
    """
    data = pd.read_csv(io.BytesIO(csv_bytes))

    if data.empty:
        raise ValueError("The uploaded dataset is empty.")

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The uploaded CSV is missing these required columns: "
            + ", ".join(missing_columns)
        )

    # Keep only the required columns. Additional columns are not used.
    data = data[required_columns].copy()

    # Convert numerical values. Distinguish genuinely missing values
    # from non-numeric strings that cannot be converted.
    for column in NUMERICAL_FEATURES:
        original_values = data[column]
        converted_values = pd.to_numeric(
            original_values,
            errors="coerce",
        )

        invalid_mask = original_values.notna() & converted_values.isna()

        if invalid_mask.any():
            raise ValueError(
                f"Column '{column}' contains non-numeric values. "
                "Please correct those values and upload the CSV again."
            )

        data[column] = converted_values

    # Normalize target labels.
    if data[TARGET].dtype == object:
        normalized_target = (
            data[TARGET]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        data[TARGET] = normalized_target.map(
            {"no": 0, "yes": 1, "0": 0, "1": 1}
        )
    else:
        data[TARGET] = pd.to_numeric(
            data[TARGET],
            errors="coerce",
        )

    if data[TARGET].isna().any():
        raise ValueError(
            "Target column 'y' contains invalid or missing values. "
            "Use only 'yes'/'no' or 1/0."
        )

    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError(
            "Target column 'y' must contain only 'yes'/'no' or 1/0."
        )

    if data[TARGET].nunique() != 2:
        raise ValueError(
            "The test dataset must contain both target classes "
            "(yes and no) to calculate AUC and the full evaluation metrics."
        )

    data[TARGET] = data[TARGET].astype(int)

    return data


# Read the dataset from the session-state copy. This works on every page,
# even when the uploader itself is only displayed on Dashboard Overview.
uploaded_bytes = st.session_state["uploaded_dataset_bytes"]
uploaded_filename = st.session_state["uploaded_dataset_name"]

test_data = None
X_test = None
y_test = None
dataset_error = None

if uploaded_bytes is not None:
    try:
        test_data = validate_dataset(uploaded_bytes)
        X_test = test_data[FEATURES]
        y_test = test_data[TARGET]
    except Exception as error:
        dataset_error = str(error)
        test_data = None


# A fingerprint is used to prevent comparison results from one upload
# being displayed for a different uploaded dataset.
current_dataset_hash = (
    hashlib.sha256(uploaded_bytes).hexdigest()
    if uploaded_bytes is not None
    else None
)

if (
    st.session_state["comparison_dataset_hash"] != current_dataset_hash
):
    st.session_state["comparison_results"] = None
    st.session_state["comparison_dataset_hash"] = current_dataset_hash

if (
    st.session_state.get("selected_evaluation") is not None
    and st.session_state["selected_evaluation"].get("dataset_hash")
    != current_dataset_hash
):
    st.session_state["selected_evaluation"] = None


# ============================================================
# 8. REUSABLE PAGE HELPERS
# ============================================================

def page_header(title, subtitle):
    st.title(title)
    st.markdown(
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def require_dataset():
    if test_data is None:
        if dataset_error:
            st.error(f"Uploaded dataset could not be validated: {dataset_error}")
        else:
            st.warning(
                "No valid test dataset is currently loaded. "
                "Please upload a CSV from the Dashboard Overview page."
            )
        return False

    return True


def show_dataset_status():
    if test_data is not None:
        st.caption(
            f"Loaded dataset: **{uploaded_filename or 'Uploaded CSV'}** "
            f"· {len(test_data):,} records"
        )


# ============================================================
# 9. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 20px 0;">
            <div style="display:flex;align-items:center;gap:12px;">
                <div style="
                    background:#18A999;
                    padding:10px;
                    border-radius:9px;
                    font-size:22px;">
                    🏦
                </div>
                <div>
                    <div style="
                        font-size:19px;
                        font-weight:700;
                        color:#FFFFFF;">
                        BANK ANALYTICS
                    </div>
                    <div style="
                        font-size:11px;
                        color:#A9BDCE;">
                        Classification Intelligence
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### NAVIGATION")

    page = st.radio(
        "Navigation",
        [
            "Dashboard Overview",
            "Model Comparison",
            "About Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### PROJECT INFORMATION")

    st.markdown(
        """
        <div style="font-size:12px;color:#A9BDCE;">
            <b>PROJECT DETAILS</b><br><br>
            Dataset: Bank Marketing<br>
            Algorithms: 5<br>
            Input Features: 15<br>
            Classification: Binary<br><br>
            <span style="color:#18A999;">
                ● Models loaded successfully
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if test_data is not None:
        st.markdown("---")
        st.markdown("### DATASET STATUS")
        st.markdown(
            f"""
            <div style="font-size:12px;color:#A9BDCE;">
                <span style="color:#18A999;">● Dataset loaded</span><br><br>
                File: {uploaded_filename or "Uploaded CSV"}<br>
                Records: {len(test_data):,}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# 10. DASHBOARD OVERVIEW
# ============================================================

if page == "Dashboard Overview":
    page_header(
        "Dashboard Overview",
        "Bank Marketing | Machine Learning Analytics",
    )

    st.markdown(
        """
        <div style="text-align:center; margin-top:8px; margin-bottom:10px;">
            <h2 style="margin-bottom:5px;">Upload Your Test Dataset</h2>
            <p style="color:#64748B; font-size:14px;">
                Upload a compatible Bank Marketing test CSV to evaluate
                our five trained classification models.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload_left, upload_center, upload_right = st.columns([1, 2, 1])

    with upload_center:
        st.file_uploader(
            "Choose your test CSV file",
            type=["csv"],
            key="dataset_uploader",
            on_change=save_uploaded_dataset,
            help=(
                "The CSV must contain the 15 Bank Marketing input features "
                "and the target column 'y'."
            ),
        )

        if test_data is not None:
            st.success(
                f"Dataset loaded: {uploaded_filename} "
                f"({len(test_data):,} records)"
            )
        elif dataset_error:
            st.error(f"Dataset validation failed: {dataset_error}")
        else:
            st.caption(
                "Required target column: y (yes/no or 1/0). "
                "Extra columns are ignored."
            )

    st.markdown("---")

    # Central, guided evaluation workflow.
    st.markdown(
        """
        <div style="text-align:center; margin-top:5px; margin-bottom:12px;">
            <h2 style="margin-bottom:5px;">Evaluate a Classification Model</h2>
            <p style="color:#64748B; font-size:14px;">
                Step 1: Upload your test CSV above.
                Step 2: Select a model below.
                Step 3: Run the evaluation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_left, model_center, model_right = st.columns([1, 2, 1])

    with model_center:
        selected_model_name = st.selectbox(
            "Step 2 — Choose a classification model",
            list(MODEL_FILES.keys()),
            key="dashboard_model_selection",
            help="Choose one of the five models trained on the Bank Marketing dataset.",
            disabled=(test_data is None),
        )

        evaluate_clicked = st.button(
            "Evaluate Selected Model",
            type="primary",
            use_container_width=True,
            disabled=(test_data is None),
            key="dashboard_evaluate_button",
        )

    if test_data is None:
        st.info(
            "Upload a valid Bank Marketing test CSV above to enable model selection and evaluation."
        )
    elif evaluate_clicked:
        try:
            with st.spinner(f"Evaluating {selected_model_name}..."):
                metrics, cm, report = evaluate_selected_model(
                    selected_model_name,
                    uploaded_bytes,
                )

            st.session_state["selected_evaluation"] = {
                "dataset_hash": current_dataset_hash,
                "model_name": selected_model_name,
                "metrics": metrics,
                "confusion_matrix": cm,
                "report": report,
            }
            st.success(f"{selected_model_name} evaluation completed successfully!")

        except Exception as error:
            st.error(f"Evaluation failed: {error}")

    # Retain the last evaluation for the current dataset and selected model.
    evaluation = st.session_state.get("selected_evaluation")
    if (
        test_data is not None
        and evaluation
        and evaluation.get("dataset_hash") == current_dataset_hash
        and evaluation.get("model_name") == selected_model_name
    ):
        st.markdown("---")
        st.markdown(f"### Results — {selected_model_name}")

        metrics = evaluation["metrics"]
        cm = evaluation["confusion_matrix"]
        report = evaluation["report"]

        metric_cols = st.columns(3)
        for index, (metric_name, metric_value) in enumerate(metrics.items()):
            metric_cols[index % 3].metric(metric_name, f"{metric_value:.4f}")

        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="GnBu",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
            linewidths=0.5,
            cbar=False,
            annot_kws={"size": 11},
            ax=ax,
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title(selected_model_name)
        plt.tight_layout()

        matrix_left, matrix_center, matrix_right = st.columns([1, 2, 1])
        with matrix_center:
            st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        st.markdown("#### Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4), use_container_width=True)

    st.markdown("---")
    st.markdown("### Welcome to Bank Analytics")
    st.write(
        "An interactive dashboard for evaluating five machine learning "
        "classification algorithms on the Bank Marketing dataset."
    )

    if test_data is None:
        st.info(
            "Upload your test dataset above to view dataset statistics "
            "and model performance."
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Classification Models", "05")
        col2.metric("Input Features", "15")
        col3.metric("Target Classes", "02")

    else:
        st.success("Test dataset successfully uploaded and validated.")

        # Summary cards
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Test Records", f"{len(test_data):,}")
        col2.metric("Input Features", len(FEATURES))
        col3.metric("Positive Cases", f"{int((y_test == 1).sum()):,}")
        col4.metric("Negative Cases", f"{int((y_test == 0).sum()):,}")

        st.markdown("### Dataset Overview")

        left, right = st.columns([1.2, 1])

        with left:
            st.markdown("#### Target Distribution")

            target_counts = (
                y_test.map({0: "No Subscription", 1: "Subscribed"})
                .value_counts()
                .reindex(["No Subscription", "Subscribed"], fill_value=0)
            )

            st.bar_chart(
                target_counts,
                color="#18A999",
            )

        with right:
            st.markdown("#### Dataset Information")

            info_df = pd.DataFrame(
                {
                    "Property": [
                        "Total Records",
                        "Numerical Features",
                        "Categorical Features",
                        "Target Variable",
                        "Classification Type",
                    ],
                    "Value": [
                        f"{len(test_data):,}",
                        "6",
                        "9",
                        "y",
                        "Binary",
                    ],
                }
            )

            st.dataframe(
                info_df,
                hide_index=True,
                use_container_width=True,
            )

        st.markdown("### Available Classification Models")

        model_cols = st.columns(5)
        model_icons = ["📈", "🌳", "📍", "🔬", "⚖️"]

        for index, (name, _) in enumerate(MODEL_FILES.items()):
            with model_cols[index]:
                st.markdown(
                    f"""
                    <div class="dashboard-card"
                         style="text-align:center;min-height:115px;">
                        <div style="font-size:25px;">
                            {model_icons[index]}
                        </div>
                        <div style="
                            color:#14243A;
                            font-weight:600;
                            font-size:13px;
                            margin-top:8px;">
                            {name}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ============================================================
# 11. MODEL EVALUATION PAGE
# ============================================================

# ============================================================
# 11. MODEL COMPARISON PAGE
# ============================================================

elif page == "Model Comparison":
    page_header(
        "Model Comparison",
        "Compare the performance of all five classification algorithms.",
    )

    show_dataset_status()

    if require_dataset():
        st.info(
            "All five models will be evaluated using the same "
            "uploaded test dataset."
        )

        if st.button(
            "Run Model Comparison",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Evaluating all five models. Please wait..."
                ):
                    comparison = compare_all_models(uploaded_bytes)

                st.session_state["comparison_results"] = comparison
                st.session_state["comparison_dataset_hash"] = (
                    current_dataset_hash
                )

                st.success("Model comparison completed successfully!")

            except Exception as error:
                st.error(f"Comparison failed: {error}")

        comparison = st.session_state.get("comparison_results")

        if (
            comparison is not None
            and st.session_state.get("comparison_dataset_hash")
            == current_dataset_hash
        ):
            st.markdown("### Performance Comparison Table")

            st.dataframe(
                comparison.round(4),
                use_container_width=True,
            )

            st.markdown("### Metric Visualizations")

            metric_options = [
                "Accuracy",
                "AUC",
                "Precision",
                "Recall",
                "F1 Score",
                "MCC",
            ]

            selected_metric = st.selectbox(
                "Select metric to visualize",
                metric_options,
            )

            chart_data = comparison[[selected_metric]].sort_values(
                by=selected_metric,
                ascending=False,
            )

            st.bar_chart(
                chart_data,
                color="#18A999",
            )

            # Identify the best model using F1 score.
            best_model = comparison["F1 Score"].idxmax()
            best_f1 = comparison.loc[best_model, "F1 Score"]

            st.markdown("### Best Performing Model")
            st.success(
                f"{best_model} achieved the highest F1 Score "
                f"of {best_f1:.4f} on this uploaded dataset."
            )

            csv_output = comparison.to_csv().encode("utf-8")

            st.download_button(
                "Download Comparison Results",
                csv_output,
                "model_comparison.csv",
                "text/csv",
                use_container_width=True,
            )


# ============================================================
# 12. ABOUT PROJECT PAGE
# ============================================================

elif page == "About Project":
    page_header(
        "About Project",
        "Project information, dataset details and machine learning methodology.",
    )

    st.markdown("### Problem Statement")

    st.write(
        """
        The objective of this project is to develop machine learning
        classification models that predict whether a bank customer
        will subscribe to a term deposit based on demographic,
        financial and previous marketing campaign information.

        Five classification algorithms are implemented and compared
        using standard classification evaluation metrics.
        """
    )

    st.markdown("### Dataset Description")

    st.write(
        """
        The Bank Marketing dataset is obtained from the UCI Machine
        Learning Repository. It contains 45,211 customer records
        and 16 original input features.

        For this project, the duration feature is excluded to avoid
        target leakage. Therefore, 15 input features are used.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Original Records", "45,211")
    col2.metric("Selected Features", "15")
    col3.metric("Target Classes", "2")

    st.markdown("### Classification Algorithms")

    algorithms = {
        "Logistic Regression": (
            "Linear classification algorithm that estimates class probabilities."
        ),
        "Decision Tree": (
            "Tree-based model that uses feature-based decision rules."
        ),
        "KNN": (
            "Classifies observations based on their nearest neighbors."
        ),
        "Gaussian Naive Bayes": (
            "Probabilistic classifier based on Bayes' theorem."
        ),
        "Support Vector Machine": (
            "Finds a decision boundary that separates the classes."
        ),
    }

    for name, description in algorithms.items():
        with st.expander(name):
            st.write(description)

    st.markdown("### Evaluation Metrics")

    metric_info = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "AUC",
                "Precision",
                "Recall",
                "F1 Score",
                "MCC",
            ],
            "Description": [
                "Overall prediction correctness.",
                "Ability to distinguish between classes.",
                "Correctness of positive predictions.",
                "Ability to identify actual positive cases.",
                "Harmonic mean of precision and recall.",
                "Balanced measure of classification quality.",
            ],
        }
    )

    st.dataframe(
        metric_info,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### Technology Stack")

    st.write(
        """
        - Python 3.13
        - Pandas and NumPy
        - Scikit-learn
        - Matplotlib and Seaborn
        - Streamlit
        - Joblib
        """
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#64748B;font-size:12px;">
        Bank Marketing Classification Dashboard<br>
        Machine Learning Assignment | Five Classification Algorithms
    </div>
    """,
    unsafe_allow_html=True,
)
