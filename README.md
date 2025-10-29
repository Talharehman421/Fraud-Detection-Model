# 🧠 Fraud Detection Using Machine Learning

This project applies **Machine Learning** models to detect **fraudulent financial transactions** using a public transaction dataset. It compares the performance of **Random Forest** and **Logistic Regression** classifiers, incorporating **SMOTE** to handle class imbalance.

---

## 📂 Dataset

The dataset used is the **Fraud Detection Dataset** (`fraud_dataset.csv`), containing records of financial transactions with labels indicating fraudulent behavior.

After random sampling (`n=10,000`), the dataset includes the following columns:

| Feature | Description |
|----------|--------------|
| `step` | Time step (1 unit = 1 hour) |
| `type` | Type of transaction (e.g., CASH_IN, TRANSFER) |
| `amount` | Amount of the transaction |
| `nameOrig` | Customer initiating the transaction |
| `oldbalanceOrg` | Initial balance of origin account |
| `newbalanceOrig` | New balance of origin account after transaction |
| `nameDest` | Recipient account |
| `oldbalanceDest` | Initial balance of destination account |
| `newbalanceDest` | New balance of destination account |
| `isFraud` | Fraud indicator (1 = Fraud, 0 = Legitimate) |
| `isFlaggedFraud` | Flagged as suspicious by system |

---

## ⚙️ Project Workflow

1. **Data Loading & Exploration**
   - Load dataset using `pandas`
   - Display info, data types, and sample rows

2. **Preprocessing**
   - Encode categorical variable `type` using `LabelEncoder`
   - Drop irrelevant columns: `nameOrig` and `nameDest`
   - Handle class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**

3. **Train-Test Split**
   - 80% training and 20% testing with stratification to preserve class ratio

4. **Feature Scaling**
   - Apply `StandardScaler` to standardize features for Logistic Regression

5. **Model Training**
   - **Random Forest Classifier**
   - **Logistic Regression** (with class weights to handle imbalance)

6. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix Visualization
   - ROC-AUC Score comparison
   - Feature Importance Plot (for Random Forest)
   - Correlation Matrix of Features

---

## 📊 Results

| Model | Accuracy | ROC-AUC | F1-Score |
|--------|-----------|----------|----------|
| Random Forest | **0.999** | **0.999** | **1.00** |
| Logistic Regression | 0.969 | 0.969 | 0.97 |

**Confusion Matrices** and **Feature Importance** plots show Random Forest performing exceptionally well after SMOTE balancing.

---

## 📈 Visualizations

- 🔹 **Feature Importance (Random Forest)**  
  Displays which features most influence fraud prediction.

- 🔹 **Feature Correlation Matrix**  
  Highlights relationships between numerical features.

- 🔹 **Confusion Matrices**  
  Separate heatmaps for Random Forest and Logistic Regression predictions.

---

## 🧩 Libraries Used

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
imblearn
