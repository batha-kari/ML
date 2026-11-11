
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("auto-mpg.csv")

# Remove missing values
df = df.dropna(subset=['mpg', 'displacement'])

# Select feature and target
X = df[['displacement']].values
y = df['mpg'].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Different polynomial degrees
degrees = [1, 2, 3, 4]


# Generate smooth X values
X_curve = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

plt.figure(figsize=(10, 6))

# Plot actual data points
plt.scatter(X_test, y_test, color='black',
            alpha=0.5, label='Actual Data')

# Plot different polynomial regression curves
for d in degrees:

    poly = PolynomialFeatures(degree=d)

    X_train_poly = poly.fit_transform(X_train)
    X_curve_poly = poly.transform(X_curve)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_curve = model.predict(X_curve_poly)

    plt.plot(X_curve, y_curve, label=f'Degree {d}')

    # Evaluation
    X_test_poly = poly.transform(X_test)
    y_pred = model.predict(X_test_poly)

    print("\nDegree:", d)
    print("MSE:", round(mean_squared_error(y_test, y_pred), 4))
    print("R2:", round(r2_score(y_test, y_pred), 4))

# Graph formatting
plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.title("Polynomial Regression for Different Degrees")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
