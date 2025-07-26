import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt

# 📌 Project Goal:
# Predict whether a customer will make an insurance claim based on personal and policy-related info.

# Step 1: Load the dataset
df = pd.read_csv('insurance.csv')

# Step 2: One-hot encode categorical variables for modeling
df = pd.get_dummies(df, columns=['smoker', 'region', 'sex'], drop_first=True)

# Step 3: Define a threshold for high charges (mean of charges column), because its not given in the dataset
threshold = df['charges'].mean()

# Step 4: Create a new binary target column 'will_claim' based on the threshold
def will_claim(r):
    if r['charges'] > threshold:
        return 1
    else:
        return 0
df['will_claim'] = df.apply(will_claim, axis=1)

# Step 5: Show the distribution of the new target variable
print(df['will_claim'].value_counts())

# Step 6: Fit a logistic regression model to predict 'will_claim'
import statsmodels.formula.api as smf
model_logit = smf.logit(
    'will_claim ~ age + bmi + children + smoker_yes + region_northwest + region_southeast + region_southwest',
    data=df
).fit()

# Step 7: Display model coefficients and their exponentiated values (odds ratios)
print(model_logit.params)
print(np.exp(model_logit.params))

# Step 8: Add predicted probabilities to the DataFrame
df['predicted_prob'] = model_logit.predict()

# Step 9: Visualize predicted probabilities by age and smoking status
sns.scatterplot(x='age', y='predicted_prob', hue='smoker_yes', data=df)
plt.title('Predicted Probability of Claim by Age and Smoking Status')
plt.xlabel('Age')
plt.ylabel('Predicted Probability of Claim')
plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
plt.axvline(x=df['age'].mean(), color='green', linestyle='--',label='Mean Age')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(title='Smoker Status')
plt.show()

# Step 10: Predict class labels using a 0.5 probability threshold
df['predicted_class'] = (df['predicted_prob'] >= 0.5).astype(int)

# Step 11: Evaluate model performance using accuracy, recall, and precision
accuracy = accuracy_score(df['will_claim'], df['predicted_class'])
recall = recall_score(df['will_claim'], df['predicted_class'])
precision = precision_score(df['will_claim'], df['predicted_class'])

print(f"Accuracy: {accuracy:.3f}")
print(f"Recall (Sensitivity): {recall:.3f}")
print(f"Precision: {precision:.3f}")