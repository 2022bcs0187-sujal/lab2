import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Paths
DATA_PATH = "Dataset/winequality-red.csv"
MODEL_PATH = "Output/model/model.joblib"
RESULTS_PATH = "Output/results/results.json"

# Create output directories
os.makedirs("Output/model", exist_ok=True)
os.makedirs("Output/results", exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH, sep=';')

# Features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics (IMPORTANT)
print(f"MSE : {mse}")
print(f"R2 Score : {r2}")

# Save model
joblib.dump(model, MODEL_PATH)

# Save metrics
results = {
    "mse": mse,
    "r2_score": r2
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)
