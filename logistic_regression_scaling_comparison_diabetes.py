import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Load dataset
data = pd.read_csv("pima_indians_diabetes.csv")

#print(data.columns)
# 2. Separate features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------------
# WITHOUT FEATURE SCALING
# --------------------------------------------------

# 4. Create logistic regression model
model_without_scaling = LogisticRegression(max_iter=1000)

# 5. Train model
model_without_scaling.fit(X_train, y_train)

# 6. Make predictions
y_pred_without = model_without_scaling.predict(X_test)

# 7. Calculate metrics
accuracy_without = accuracy_score(y_test, y_pred_without)
precision_without = precision_score(y_test, y_pred_without)
recall_without = recall_score(y_test, y_pred_without)
f1_without = f1_score(y_test, y_pred_without)

print("WITHOUT FEATURE SCALING")
print("Accuracy :", accuracy_without)
print("Precision:", precision_without)
print("Recall   :", recall_without)
print("F1-score :", f1_without)


# --------------------------------------------------
# WITH FEATURE SCALING
# --------------------------------------------------

# 8. Create scaler
scaler = StandardScaler()

# 9. Scale training data
X_train_scaled = scaler.fit_transform(X_train)

# 10. Scale testing data
X_test_scaled = scaler.transform(X_test)

# 11. Create logistic regression model
model_with_scaling = LogisticRegression(max_iter=1000)

# 12. Train model
model_with_scaling.fit(X_train_scaled, y_train)

# 13. Make predictions
y_pred_with = model_with_scaling.predict(X_test_scaled)

# 14. Calculate metrics
accuracy_with = accuracy_score(y_test, y_pred_with)
precision_with = precision_score(y_test, y_pred_with)
recall_with = recall_score(y_test, y_pred_with)
f1_with = f1_score(y_test, y_pred_with)

print("\nWITH FEATURE SCALING")
print("Accuracy :", accuracy_with)
print("Precision:", precision_with)
print("Recall   :", recall_with)
print("F1-score :", f1_with)

