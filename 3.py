
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("diabetes.csv")

X = df.drop('target', axis=1)
y = df['target']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
linear = LinearRegression()
linear.fit(X_train, y_train)
pred_linear = linear.predict(X_test)

# Ridge Regression
alphas = [0.001, 0.01, 0.1, 1, 10, 100]

ridge_grid = GridSearchCV(
    Ridge(), {'alpha': alphas}, cv=5,
    scoring='neg_mean_squared_error'
)
ridge_grid.fit(X_train, y_train)
pred_ridge = ridge_grid.predict(X_test)

# Lasso Regression
lasso_grid = GridSearchCV(
    Lasso(max_iter=10000), {'alpha': alphas}, cv=5,
    scoring='neg_mean_squared_error'
)
lasso_grid.fit(X_train, y_train)
pred_lasso = lasso_grid.predict(X_test)

# Store models and predictions
models = ['Linear', 'Ridge', 'Lasso']
predictions = [pred_linear, pred_ridge, pred_lasso]

# Print model performance
print("MODEL PERFORMANCE")
print("Model       MSE       R2")

for name, pred in zip(models, predictions):
    print(name, round(mean_squared_error(y_test, pred), 4),
          round(r2_score(y_test, pred), 4))

# Plot 1: Actual vs Predicted
plt.figure(figsize=(8, 5))

for name, pred in zip(models, predictions):
    plt.scatter(y_test, pred, label=name, alpha=0.7)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Plot 2: Coefficient Comparison
coefficients = [
    linear.coef_,
    ridge_grid.best_estimator_.coef_,
    lasso_grid.best_estimator_.coef_
]

x = np.arange(len(X.columns))
width = 0.25

plt.figure(figsize=(10, 5))

for i in range(3):
    plt.bar(x + i*width, coefficients[i],
            width, label=models[i])

plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Coefficient Comparison (Shrinkage Effect)")
plt.xticks(x + width, X.columns, rotation=45)
plt.legend()
plt.grid(axis='y')
plt.tight_layout()


# Print best alpha values
print("\nBEST ALPHA VALUES")
print("Ridge Alpha:", ridge_grid.best_params_['alpha'])
print("Lasso Alpha:", lasso_grid.best_params_['alpha'])

plt.show()
