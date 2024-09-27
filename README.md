Credit Card Fraud Detection

Overview

This project focuses on detecting fraudulent transactions in credit card data using various machine learning models. The dataset used is the Credit Card Fraud Detection dataset from Kaggle.

Problem Statement: Fraudulent transactions in credit card data are rare events. Therefore, detecting them requires techniques that can handle imbalanced datasets and minimize false positives and false negatives.

Dataset
The dataset contains transactions made by credit cards in September 2013 by European cardholders. It is highly imbalanced, with only 0.172% of transactions being fraudulent. The dataset has 31 columns:

V1, V2, ..., V28: These are the result of a PCA transformation.
Time: The number of seconds elapsed between this transaction and the first transaction in the dataset.
Amount: The transaction amount.
Class: The target variable (1 for fraud, 0 for non-fraud).
Project Structure
r
Copy code

├── README.md                 <- The project README file

├── credit_card_fraud.ipynb    <- Jupyter notebook for the project

├── data/                    

│   └── creditcard.csv         <- The dataset

└── models/                   

    └── trained_models.pkl     <- Trained models stored for later use
    
Technologies Used
Python: Main programming language.
Pandas: For data manipulation and analysis.
NumPy: For numerical computations.
Scikit-learn: For building machine learning models.
XGBoost: Gradient boosting for classification.
Matplotlib/Seaborn: For data visualization.
Jupyter Notebook: To interactively run the project.
Models Used
We experimented with the following machine learning techniques:

Logistic Regression
Random Forest Classifier
Support Vector Machine (SVM)
AdaBoost Classifier
XGBoost Classifier
Each model was evaluated using metrics such as:

Confusion Matrix
F1 Score
Precision and Recall
AUC-ROC Curve
Installation and Setup
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/credit-card-fraud-detection.git
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Download the dataset: Download the dataset from Kaggle:

Credit Card Fraud Detection Dataset
Place the downloaded CSV file in the data/ folder.
Run the notebook: Open the Jupyter notebook and run the cells to reproduce the results:

bash
Copy code
jupyter notebook credit_card_fraud.ipynb
Exploratory Data Analysis (EDA)
We conducted EDA to understand the data and highlight important insights. The data was found to be highly imbalanced, with only 0.172% fraudulent transactions. A correlation matrix was used to explore relationships between features.

Key visualizations:

Distribution of fraudulent vs non-fraudulent transactions.
Correlation heatmap to detect feature relationships.
Feature importance using Random Forest and XGBoost.
Model Evaluation
After training the models, we evaluated their performance based on the following metrics:

F1 Score: Balances precision and recall, especially useful for imbalanced datasets.
Confusion Matrix: Shows True Positives, True Negatives, False Positives, and False Negatives.
AUC-ROC Curve: Plots true positive rate (sensitivity) against false positive rate.
Results:

Random Forest: High accuracy but slightly lower F1 score.
XGBoost: Best performance in terms of F1 score and ROC-AUC.
AdaBoost: Decent performance but slower due to boosting iterations.
Logistic Regression: Simple but effective baseline.
SVM: Effective but computationally expensive.
Feature Importance
We also plotted feature importance to identify which variables contributed most to predicting fraudulent transactions. The Amount, V17, and V12 features were found to be among the most important.

python
# Feature Importance Plot Example
tmp = pd.DataFrame({'Feature': X_train.columns, 'Feature importance': xgb.feature_importances_})
tmp = tmp.sort_values(by='Feature importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Feature', y='Feature importance', data=tmp)
plt.title('Feature Importance - XGBoost')
plt.xticks(rotation=90)
plt.show()
Conclusion
XGBoost outperformed other models in detecting fraudulent transactions based on its F1 score and AUC-ROC curve. The dataset's imbalance made it necessary to carefully tune models to reduce false positives and false negatives.

Future Work
Implement deep learning models (e.g., LSTM) to enhance performance.
Fine-tune hyperparameters further using techniques like grid search.
Test models on more recent data to check for model robustness.
