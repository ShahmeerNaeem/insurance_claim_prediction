# 🧠 Insurance Claim Prediction (Logistic Regression)

A hands-on data science project to predict whether a customer is likely to make an insurance claim, using demographic and policy-related information. This project showcases custom feature engineering, statistical modeling, and performance evaluation — all built from scratch without pre-labeled target data.

---

## 📌 Problem Statement

The original dataset does **not contain a claim label**, so I engineered a binary target (`will_claim`) using domain logic, assuming that individuals with insurance charges **above the mean** are likely to have made a claim.

---

## 🎯 Goal

> Predict whether a customer is likely to make an insurance claim, based on personal and policy-related attributes.

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
