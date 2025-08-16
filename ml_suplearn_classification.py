# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 13:34:12 2025

@author: MEAL ASSISTANT 1
"""

### Data Loading and Initial Inspection

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Read the CSV file into a pandas DataFrame
filepath = r"C:\Users\MEAL ASSISTANT 1\Documents\IBM ML Professional Certificate\3-Supervised ML Classification\BsFHefkXTv2BR3n5F2792A_77e34164a837477889a347b01e1656f1_Data-and-Python-Assets\Telco_customer_churn_status.csv"
df = pd.read_csv(filepath)

# Display the first 5 rows to understand the data
print("Initial Data Head:")
print(df.head())

# Display information about the DataFrame, including data types and non-null values
print("\nInitial Data Info:")
print(df.info())

### Data Preprocessing and Feature Engineering

# Drop all columns that could lead to data leakage or are duplicates
# These are columns that provide information about the churn event that would not be available beforehand.
columns_to_drop = [
    'Customer ID', 'Count', 'Quarter', 'Satisfaction Score', 'Customer Status',
    'Churn Label', 'Churn Score', 'Churn Category', 'Churn Reason'
]
df_cleaned = df.drop(columns=columns_to_drop)

# Separate features (X) and target (y)
X = df_cleaned.drop(columns=['Churn Value'])
y = df_cleaned['Churn Value']

# Scale the numerical feature (CLTV)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

### Model Training and Evaluation

# Initialize and train the models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

results = {}
confusion_matrices = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}
    confusion_matrices[name] = cm

# Print the results in a clear format
print("\nModel Evaluation Results:")
for name, metrics in results.items():
    print(f"\n--- {name} ---")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.2f}")
        
print("\nConfusion Matrices:")
for name, cm in confusion_matrices.items():
    print(f"\n--- {name} ---")
    print(cm)