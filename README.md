Bank Marketing Classification Dashboard
An interactive machine learning application that evaluates five classification algorithms on the UCI Bank Marketing dataset. The application is built with Python and Streamlit.
Project Overview
The objective is to predict whether a bank customer will subscribe to a term deposit based on demographic, financial, and previous marketing campaign information.

Five classification algorithms are trained and evaluated:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (LinearSVC)
The Streamlit application allows users to upload a compatible test CSV, select a trained model, view evaluation metrics, inspect a confusion matrix and classification report, and compare all five models.
Dataset
- Dataset: Bank Marketing (bank-full.csv)
- Source: UCI Machine Learning Repository – Bank Marketing
- Original records: 45,211
- Original input features: 16
- Target: y (yes / no)
- Features used for modeling: 15
The duration feature is excluded because it is only known after a call has taken place and could introduce target leakage.
Features used
Numerical features
- age
- balance
- day
- campaign
- pdays
- previous
Categorical features
- job
- marital
- education
- default
- housing
- loan
- contact
- month
- poutcome
The target variable is y, where no is encoded as 0 and yes as 1.
Model Evaluation
The models are evaluated using the following six metrics:
- Accuracy: Overall proportion of correct predictions.
- AUC: Ability to distinguish between the two classes.
- Precision: Proportion of predicted positive cases that are actually positive.
- Recall: Proportion of actual positive cases correctly identified.
- F1 Score: Harmonic mean of precision and recall.
- MCC: Matthews Correlation Coefficient, a balanced measure of binary classification quality.
Test-set results
The following results were obtained from the held-out test set during model evaluation.
Model	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.8933	0.7717	0.6632	0.1786	0.2815	0.3065
Decision Tree	0.8309	0.6132	0.2982	0.3289	0.3128	0.2170
KNN	0.8874	0.7027	0.5529	0.1975	0.2911	0.2833
Gaussian Naive Bayes	0.8452	0.7514	0.3708	0.4641	0.4123	0.3271
Support Vector Machine	0.8932	0.7682	0.6513	0.1871	0.2907	0.3101


Results summary
- Logistic Regression achieved the highest accuracy (0.8933) and AUC (0.7717).
- Gaussian Naive Bayes achieved the highest F1 Score (0.4123), recall (0.4641), and MCC (0.3271).
- The dataset is imbalanced, with substantially more no than yes outcomes. Accuracy alone is therefore not sufficient to judge model performance.
- Gaussian Naive Bayes had the strongest F1 Score on this test set, while its precision was lower than that of Logistic Regression and SVM.
The best model can depend on the evaluation metric and the relative cost of false positives and false negatives.
Streamlit Application
The dashboard provides:
Dashboard Overview
- Upload a Bank Marketing test CSV in the main page.
- View the number of test records, input features, positive cases, and negative cases.
- View the target distribution and available classification models.
Model Evaluation
- Select one of the five trained models.
- Evaluate the selected model on the uploaded test dataset.
- View Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- Inspect the confusion matrix and classification report.
Model Comparison
- Evaluate all five models on the same uploaded test dataset.
- View a comparison table and metric visualizations.
- Download the comparison results as a CSV file.
About Project
- Review the dataset, algorithms, evaluation metrics, and technology stack.
Important: Uploaded CSV Format
The application evaluates uploaded data using the five models already trained on the Bank Marketing dataset. It does not train new models for arbitrary datasets.

An uploaded CSV must contain:
- All 15 input feature columns listed above.
- The target column y, containing both classes.
- Target values represented as yes/`no` or 1/`0`.
Additional columns are ignored. Missing feature values can be handled by the preprocessing pipelines saved with the models. The uploaded data must use the same feature meanings and compatible values as the Bank Marketing dataset.
Project Structure
Bank-Marketing-Classification/
├── app.py
├── step1_dataset.py
├── step2_preprocessing.py
├── step3_model_training.py
├── step4_model_evaluation.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── support_vector_machine.joblib
└── results/
    ├── model_comparison.csv
    └── confusion_matrices/
The exact files included in the GitHub repository may vary. The trained model files are required by app.py.
Installation and Local Setup
1. Clone the repository
git clone https://github.com/deekshith9980/Bank-Marketing-Classification.git
cd Bank-Marketing-Classification
2. Create and activate a virtual environment (recommended)
Windows
python -m venv venv
venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Run the Streamlit application
python -m streamlit run app.py
The application will open in your browser. Upload a compatible Bank Marketing test CSV from the Dashboard Overview page to begin.
Requirements
The project uses Python and the following main libraries:
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
Install the exact versions and dependencies listed in requirements.txt.
Deployment
The application is intended to be deployed using Streamlit Community Cloud.
- GitHub repository: https://github.com/deekshith9980/Bank-Marketing-Classification.git
- Live Streamlit application: https://bank-marketing-classification-eevw2rotu8amqwhfgzhj3i.streamlit.app/

Limitations
- The saved models are specific to the Bank Marketing feature set and target.
- The application evaluates labeled test data; it is not designed to accept unrelated datasets.
- AUC and the full binary classification evaluation require both target classes to be present in the uploaded test CSV.
- Model performance may vary on a different compatible test sample.
Technologies Used
Python · Pandas · NumPy · Scikit-learn · Streamlit · Matplotlib · Seaborn · Joblib