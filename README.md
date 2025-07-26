# 🧠 Insurance Claim Prediction (Logistic Regression)

A hands-on data science project to predict whether a customer is likely to make an insurance claim, using demographic and policy-related information. This project showcases custom feature engineering, statistical modeling, and performance evaluation — all built from scratch without pre-labeled target data.

---

## 📌 Problem Statement

The original dataset does **not contain a claim label**, so I engineered a binary target (`will_claim`) using domain logic, assuming that individuals with insurance charges **above the mean** are likely to have made a claim.

---

## 🎯 Goal

Predict whether a customer is likely to make an insurance claim, based on personal and policy-related attributes.

---

## 📊 Dataset Overview

- Source: `insurance.csv`
- Rows: 1338
- Features:
  - `age`, `bmi`, `children`
  - `sex`, `smoker`, `region` (categorical)
  - `charges` (used for custom label creation)

---

## 🛠 Feature Engineering

- Converted categorical features using `pd.get_dummies()` with `drop_first=True`
- Engineered a binary classification label `will_claim`:
  
```python
df['will_claim'] = (df['charges'] > df['charges'].mean()).astype(int)
```
---

## 🧠 Key Takeaways

- ✅ **Created a custom target label** (`will_claim`) using domain-driven logic by applying a threshold on medical `charges` — a key example of real-world **feature engineering**.

- ✅ **Chose logistic regression** instead of linear regression due to the **binary classification** nature of the problem, and to obtain **interpretable probabilities**.

- ✅ **Interpreted model outputs** using **odds ratios**, helping quantify how much each feature (like smoking status or age) increases or decreases the likelihood of a claim.

- ✅ **Evaluated model performance** with **accuracy, precision, and recall**, demonstrating not just how accurate the model was, but how reliable and sensitive it was to claim predictions.

- ✅ **Visualized model behavior** through scatter plots of predicted probabilities by features like age and smoker status, making insights easier to understand and communicate.

---

