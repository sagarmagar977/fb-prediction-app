import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# --- Configuration (Based on your fc.ipynb snippet) ---
FILE_NAME = 'SP1.csv'
MODEL_FILE = 'football_model.joblib'
# Using Pinnacle Sports (PS) odds as input features
FEATURES = ['PSH', 'PSD', 'PSA']  
TARGET = 'FTR'

print(f"Loading data from {FILE_NAME}...")

try:
    df = pd.read_csv(FILE_NAME)
except FileNotFoundError:
    print(f"Error: {FILE_NAME} not found. Please ensure it is in the same directory.")
    exit()

# --- Preprocessing & Feature Selection ---
# Map the FTR (Full Time Result) to a numeric value: H -> 2, D -> 1, A -> 0
result_mapping = {'H': 2, 'D': 1, 'A': 0}
df['Target'] = df[TARGET].map(result_mapping)

# Drop rows with missing feature/target data
df_model = df[[*FEATURES, 'Target']].dropna()

if df_model.empty:
    print("Error: No valid data found for the selected features/target. Cannot train model.")
    exit()

# --- Model Training ---
X = df_model[FEATURES]
y = df_model['Target']

# Split data (using only train for final model in this simple prototype)
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("Training complete.")

# --- Save Model ---
joblib.dump(model, MODEL_FILE)
print(f"Model successfully trained and saved as {MODEL_FILE}")