import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, auc

# Step 1: Create synthetic dataset
rng = np.random.default_rng(42)
n_samples = 8000
fraud_ratio = 0.025
n_fraud = int(n_samples * fraud_ratio)
n_normal = n_samples - n_fraud

normal = rng.normal(0, 1, (n_normal, 20))
fraud = rng.normal(2.5, 1.2, (n_fraud, 20))
X = np.vstack([normal, fraud])
y = np.array([0]*n_normal + [1]*n_fraud)

# Step 2: Split & scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Oversample minority class
fraud_indices = np.where(y_train == 1)[0]
oversampled_indices = np.hstack([np.where(y_train == 0)[0], 
                                 np.tile(fraud_indices, int((len(y_train) // len(fraud_indices))))[:len(y_train)]])
X_train_balanced = X_train_scaled[oversampled_indices]
y_train_balanced = y_train[oversampled_indices]

# Step 4: Train models
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=150),
    "GradientBoosting": GradientBoostingClassifier()
}

print("Training models...")
for name, model in models.items():
    model.fit(X_train_balanced, y_train_balanced)
    print(f"{name} trained.")

# Step 5: Evaluate
for name, model in models.items():
    preds = model.predict_proba(X_test_scaled)[:,1]
    precision, recall, _ = precision_recall_curve(y_test, preds)
    pr_auc = auc(recall, precision)
    print(f"{name} PR-AUC: {pr_auc:.4f}")

print("Analysis complete.")
