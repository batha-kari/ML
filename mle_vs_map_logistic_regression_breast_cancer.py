
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load CSV dataset
df = pd.read_csv("breast_cancer.csv")

# Separate features and target
df = df.drop(columns=['id'], errors='ignore')
X = df.drop('diagnosis', axis=1)
y = df['diagnosis'].map({'M': 1, 'B': 0})

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MLE - No regularization
mle = LogisticRegression(penalty=None, max_iter=10000)
mle.fit(X_train, y_train)

# MAP - L1 and L2 regularization
C = [0.01, 0.1, 1, 10, 100]

l1 = GridSearchCV(
    LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000),
    {'C': C}, cv=5
)

l2 = GridSearchCV(
    LogisticRegression(penalty='l2', max_iter=10000),
    {'C': C}, cv=5
)

l1.fit(X_train, y_train)
l2.fit(X_train, y_train)

# Predictions
models = [mle, l1, l2]
names = ['MLE', 'MAP-L1', 'MAP-L2']
predictions = [model.predict(X_test) for model in models]

# Evaluation
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
metrics = [accuracy_score, precision_score, recall_score, f1_score]

scores = []

print("MODEL PERFORMANCE")

for name, pred in zip(names, predictions):
    values = [metric(y_test, pred) for metric in metrics]
    scores.append(values)

    print("\n", name)
    for metric_name, value in zip(metric_names, values):
        print(metric_name, ":", round(value, 4))

# Parameter comparison
coef_mle = mle.coef_[0]
coef_l1 = l1.best_estimator_.coef_[0]
coef_l2 = l2.best_estimator_.coef_[0]

parameters = pd.DataFrame({
    'Feature': X.columns,
    'MLE': coef_mle,
    'MAP-L1': coef_l1,
    'MAP-L2': coef_l2
})

print("\nPARAMETER ESTIMATES")
print(parameters)

print("\nBEST C VALUES")
print("L1:", l1.best_params_['C'])
print("L2:", l2.best_params_['C'])

print("\nCOEFFICIENT NORMS")
print("MLE:", np.linalg.norm(coef_mle))
print("L1:", np.linalg.norm(coef_l1))
print("L2:", np.linalg.norm(coef_l2))
print("Non-zero L1 coefficients:", np.count_nonzero(coef_l1))

# Graph 1: Performance comparison
scores = np.array(scores)
x = np.arange(len(metric_names))
width = 0.25

plt.figure(figsize=(9, 5))

for i in range(3):
    plt.bar(x + (i-1)*width, scores[i], width, label=names[i])

plt.xticks(x, metric_names)
plt.ylabel("Score")
plt.title("MLE vs MAP (L1 and L2) Performance")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(axis='y')
plt.tight_layout()

# Graph 2: Coefficient comparison
plt.figure(figsize=(10, 5))

plt.plot(np.abs(coef_mle), marker='o', label='MLE')
plt.plot(np.abs(coef_l1), marker='o', label='MAP-L1')
plt.plot(np.abs(coef_l2), marker='o', label='MAP-L2')

plt.xlabel("Feature Number")
plt.ylabel("Absolute Coefficient Value")
plt.title("Comparison of Estimated Parameters")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
