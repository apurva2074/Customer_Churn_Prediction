# Customer Churn Prediction

## Project Overview
This project explores customer churn in a telecommunications company and applies machine learning to predict whether a customer is likely to leave the service.

## Dataset
- **Source:** IBM Telco Customer Churn Dataset
- **Records:** 7,043 customers
- **Features:** 21 columns, including customer demographics, subscribed services, contract details, and billing information.
- **Target variable:** `Churn`

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib / Seaborn (if used in the notebook)
- Scikit-learn
- Jupyter Notebook

## Project Workflow
1. Load and inspect the dataset.
2. Perform exploratory data analysis.
3. Check missing values and duplicate records.
4. Encode categorical variables.
5. Split the data into training and testing sets.
6. Train a Logistic Regression model.

## Model
Logistic Regression is used as the initial classification model to predict customer churn.

## Current Status
The initial data exploration, preprocessing, and model training workflow is implemented.

**Next improvements:**
- Handle `TotalCharges` as a numeric feature.
- Review feature selection and encoding.
- Evaluate the model using precision, recall, F1-score, confusion matrix, and ROC-AUC.
- Document the final model results.

## How to Run
1. Clone this repository.
2. Install the required dependencies.
3. Open `load_dataset.ipynb` in Jupyter Notebook.
4. Run the notebook cells in sequence.

## Disclaimer
This is a learning project for practicing data analysis and machine learning classification.
