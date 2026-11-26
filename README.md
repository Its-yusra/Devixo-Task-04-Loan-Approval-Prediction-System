# 🏦 Loan Approval Prediction System

An end-to-end Machine Learning project that predicts loan approval status based on applicant information. The project includes data analysis, preprocessing, feature engineering, model comparison, hyperparameter tuning, and an interactive Streamlit web application.

## 📌 Project Overview

The Loan Approval Prediction System uses machine learning classification algorithms to predict whether a loan application is likely to be approved or rejected.

A Random Forest Classifier was selected and optimized using GridSearchCV. The trained model is integrated into a Streamlit application where users can enter applicant details and receive predictions with probability scores.

## 🎯 Project Objectives

- Perform Exploratory Data Analysis (EDA).
- Handle missing values and numerical outliers.
- Apply categorical encoding and numerical scaling.
- Perform feature engineering.
- Train and compare multiple classification models.
- Optimize the Random Forest model using GridSearchCV.
- Evaluate model performance using classification metrics.
- Develop an interactive Streamlit application.
- Save the trained model using Joblib.

## 📂 Dataset

The project uses a Loan Prediction dataset containing **614 records and 13 columns**.

**Dataset source:** [Loan Prediction Dataset](https://raw.githubusercontent.com/dsrscientist/DSData/master/loan_prediction.csv)

### Target Distribution

| Loan Status | Records |
|---|---:|
| Approved (Y) | 422 |
| Not Approved (N) | 192 |

### Features

- Gender
- Married
- Dependents
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area

## ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning and preprocessing |
| Joblib | Model serialization |
| Streamlit | Web application |

## 🔍 Project Workflow

### 1. Data Collection and EDA

- Loaded and inspected the loan prediction dataset.
- Analyzed missing values and duplicate records.
- Examined numerical and categorical distributions.
- Created correlation heatmaps and outlier visualizations.
- Analyzed income and loan amount relationships.

### 2. Data Preprocessing

- Removed the Loan_ID column.
- Identified numerical outliers using the IQR method.
- Implemented a custom `OutlierCapper` transformer.
- Handled numerical missing values using median imputation.
- Handled categorical missing values using most frequent imputation.
- Applied One-Hot Encoding to categorical variables.
- Applied StandardScaler to numerical variables.

### 3. Feature Engineering

Created two additional features:

**Total Income**

`Total_Income = ApplicantIncome + CoapplicantIncome`

**Loan-to-Income Ratio**

`Loan_to_Income_Ratio = LoanAmount / Total_Income`

### 4. Train-Test Split

The dataset was divided into training and testing sets using an 80:20 stratified split.

| Dataset | Records |
|---|---:|
| Training | 491 |
| Testing | 123 |

### 5. Model Training

Five classification algorithms were trained:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)

## 📊 Model Comparison

The following results were obtained on the test dataset:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6992 | 0.7069 | 0.9647 | 0.8159 | 0.6043 |
| Decision Tree | 0.6098 | 0.7229 | 0.7059 | 0.7143 | 0.5503 |
| Random Forest | **0.7398** | **0.7524** | 0.9294 | **0.8316** | **0.6347** |
| SVM | 0.6992 | 0.6967 | 1.0000 | 0.8213 | 0.6074 |
| KNN | 0.6748 | 0.7103 | 0.8941 | 0.7917 | 0.6067 |

Random Forest achieved the highest initial test accuracy and F1-score.

## 🛠️ Hyperparameter Tuning

GridSearchCV was applied to optimize the Random Forest Classifier.

**Best Parameters:**

```python
{
    'classifier__class_weight': None,
    'classifier__max_depth': 5,
    'classifier__min_samples_leaf': 1,
    'classifier__min_samples_split': 5,
    'classifier__n_estimators': 200
}
```

- Cross-validation folds: 5
- Parameter combinations: 48
- Total fits: 240
- Best Cross-Validation F1-Score: 0.8131
- Tuning Time: 92.75 seconds

### Before vs After Tuning

| Metric | Before | After |
|---|---:|---:|
| Accuracy | 0.7398 | 0.7154 |
| Precision | 0.7524 | 0.7119 |
| Recall | 0.9294 | 0.9882 |
| F1-Score | 0.8316 | 0.8276 |
| ROC-AUC | 0.6347 | 0.6254 |

Tuning improved Recall but slightly reduced the other reported test metrics. These results are documented without assuming that tuning improved overall performance.

## 📈 Final Model Evaluation

The tuned Random Forest model achieved approximately **71.54% test accuracy**.

| Metric | Not Approved | Approved |
|---|---:|---:|
| Precision | 0.80 | 0.71 |
| Recall | 0.11 | 0.99 |
| F1-Score | 0.19 | 0.83 |

The model performs better at identifying approved applications than rejected applications. Its low recall for the Not Approved class is an important limitation.

## ⭐ Feature Importance

The top five features according to Random Forest feature importance were:

| Feature | Importance |
|---|---:|
| Loan_to_Income_Ratio | 0.165669 |
| ApplicantIncome | 0.153738 |
| Total_Income | 0.148595 |
| LoanAmount | 0.134460 |
| CoapplicantIncome | 0.096202 |

Loan-to-Income Ratio was the most important feature in the trained model.

## 🌐 Streamlit Application

The project includes an interactive Streamlit application.

### Application Features

- Applicant information input form.
- Income and loan amount inputs.
- Credit history selection.
- Education and employment details.
- Property area selection.
- Automatic feature engineering.
- Loan approval prediction.
- Approved and Not Approved probability display.

### Run the Application Locally

Clone the repository:

```bash
git clone https://github.com/Its-yusra/Loan-Approval-Prediction-System.git
```

Navigate to the project directory:

```bash
cd Loan-Approval-Prediction-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

**Live Application:** [Add your Streamlit deployment URL here after successful deployment]

## 📁 Project Structure

```text
Loan-Approval-Prediction-System/
│
├── app.py
├── outlier_copper.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── loan_prediction.csv
│   └── processed/
│
├── models/
│   └── best_loan_model.joblib
│
└── outputs/
    ├── model_comparison.csv
    ├── hyperparameter_tuning_comparison.csv
    ├── feature_importance.csv
    │
    └── plots/
        ├── loan_status_distribution.png
        ├── numerical_distributions.png
        ├── categorical_distributions.png
        ├── correlation_heatmap.png
        ├── outlier_detection.png
        ├── income_vs_loan.png
        ├── hyperparameter_tuning_comparison.png
        ├── final_confusion_matrix.png
        ├── final_roc_curve.png
        └── feature_importance.png
```

## 🚀 Future Improvements

- Improve detection of rejected loan applications.
- Address class imbalance and evaluate alternative classification thresholds.
- Perform additional model optimization.
- Improve prediction reliability on unseen data.
- Enhance the Streamlit interface with additional visualizations.

## ⚠️ Disclaimer

This project is developed for educational and learning purposes only. Predictions are based on historical dataset patterns and should not be considered actual financial advice or official bank loan decisions.

## 👩‍💻 Author

**Yusra Komal**

- GitHub: [Its-yusra](https://github.com/Its-yusra)

---

⭐ If you find this project useful, feel free to explore the repository.
