🧠 Fraud Detection Using Machine Learning

This project applies Machine Learning models to detect fraudulent financial transactions using a public transaction dataset. It compares the performance of Random Forest and Logistic Regression classifiers, incorporating SMOTE to handle class imbalance.

📂 Dataset

The dataset used is the Fraud Detection Dataset (PS_20174392719_1491204439457_log.csv), containing records of financial transactions with labels indicating fraudulent behavior.

After random sampling (n=10,000), the dataset includes the following columns:

Feature	Description
step	Time step (1 unit = 1 hour)
type	Type of transaction (e.g., CASH_IN, TRANSFER)
amount	Amount of the transaction
nameOrig	Customer initiating the transaction
oldbalanceOrg	Initial balance of origin account
newbalanceOrig	New balance of origin account after transaction
nameDest	Recipient account
oldbalanceDest	Initial balance of destination account
newbalanceDest	New balance of destination account
isFraud	Fraud indicator (1 = Fraud, 0 = Legitimate)
isFlaggedFraud	Flagged as suspicious by system
⚙️ Project Workflow

Data Loading & Exploration

Load dataset using pandas

Display info, data types, and sample rows

Preprocessing

Encode categorical variable type using LabelEncoder

Drop irrelevant columns: nameOrig and nameDest

Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)

Train-Test Split

80% training and 20% testing with stratification to preserve class ratio

Feature Scaling

Apply StandardScaler to standardize features for Logistic Regression

Model Training

Random Forest Classifier

Logistic Regression (with class weights to handle imbalance)

Model Evaluation

Accuracy, Precision, Recall, F1-Score

Confusion Matrix Visualization

ROC-AUC Score comparison

Feature Importance Plot (for Random Forest)

Correlation Matrix of Features

📊 Results
Model	Accuracy	ROC-AUC	F1-Score
Random Forest	0.999	0.999	1.00
Logistic Regression	0.969	0.969	0.97

Confusion Matrices and Feature Importance plots show Random Forest performing exceptionally well after SMOTE balancing.

📈 Visualizations

🔹 Feature Importance (Random Forest)
Displays which features most influence fraud prediction.

🔹 Feature Correlation Matrix
Highlights relationships between numerical features.

🔹 Confusion Matrices
Separate heatmaps for Random Forest and Logistic Regression predictions.

🧩 Libraries Used
pandas
numpy
matplotlib
seaborn
scikit-learn
imblearn

🚀 How to Run

Clone this repository:

git clone https://github.com/<your-username>/fraud-detection-ml.git
cd fraud-detection-ml


Install dependencies:

pip install -r requirements.txt


Run the notebook or Python script:

jupyter notebook fraud_detection.ipynb
# or
python fraud_detection.py

💡 Insights

SMOTE improved model performance significantly by balancing minority fraud cases.

Random Forest outperformed Logistic Regression in all metrics.

Feature importance revealed transaction amount and type as strong predictors of fraud.

🧬 Author

👤 Talha Rehman
Bioinformatics Student | AI & Data Science Enthusiast
🔗 LinkedIn

💻 GitHub

🏁 Future Work

Implement XGBoost and Neural Networks

Apply Explainable AI (SHAP/LIME) to interpret model predictions

Deploy as a web dashboard for real-time fraud detection
