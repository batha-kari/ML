## QUESTION

Implement linear regression with one variable on the **California Housing dataset** to predict
housing prices based on a single feature (e.g., the average number of rooms per dwelling).
Tasks:
- Load and preprocess the datase.
- Implement linear regression using both gradient descent and the normal equation.
- Evaluate the model performance using metrics such as Mean Squared Error
(MSE) and R-squared.
- Visualize the fitted line along with the data points.

## ALGORITHM

1. Start.

2. Import the required libraries: NumPy, Pandas, Matplotlib, `train_test_split`, `mean_squared_error` and `r2_score`.

3. Load the California Housing dataset from the CSV file using Pandas.

4. Select `AveRooms` as the independent feature (X) and `MedHouseVal` as the dependent target (y).

5. Split the dataset into training and testing sets using an 80:20 ratio.

6. Perform feature scaling by calculating the mean and standard deviation of the training data. Standardize both training and testing features using these values.

7. Add a column of ones to the feature matrices to include the intercept term.

8. Implement Gradient Descent:

- Initialize the parameters (θ) to zero.
- Set the learning rate (α), number of iterations and training sample count.
- Calculate predicted values using matrix multiplication.
- Calculate the gradients using the prediction errors.
- Update the parameters using the gradient descent rule.
- Repeat the process for the specified number of iterations.
- Predict housing prices using the optimized parameters.

9. Implement Normal Equation:

- Calculate the optimal parameters directly using the pseudoinverse of the feature matrix.
- Predict housing prices using the calculated parameters.

10. Evaluate both models using Mean Squared Error (MSE) and R-squared (R²) score.

11. Visualize the actual data points using a scatter plot and plot the fitted regression lines obtained from both methods.

12. Display the graph with appropriate labels, title and legend.

13. Stop.

## CODE

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_squared_error,r2_score)

data = pd.read_csv("california_housing.csv")
data = data.dropna()[:1500]

X = data[['AveRooms']].values
y = data['MedHouseVal'].values

# Standardization
mean = X.mean()
std = X.std()
X_norm = ((X - mean) / std).flatten()

# Gradient Descent
b0 = 0
b1 = 0
lr = 0.01
ini = 1000
n = len(X_norm)

for i in range(ini):
    yp = b0 + b1 * X_norm

    db0 = (2/n) * np.sum(yp - y)
    db1 = (2/n) * np.sum(X_norm * (yp - y))

    b0 = b0 - db0 * lr
    b1 = b1 - db1 * lr

yg = b0 + b1 * X_norm

print("Gradient Descent MSE:", mean_squared_error(y, yg))
print("Gradient Descent R2",r2_score(y,yg))

# Normal Equation
Xn = np.column_stack((np.ones(len(X_norm)), X_norm))
t = np.linalg.inv(Xn.T@Xn)@Xn.T@y
b0n = t[0]
b1n = t[1]

yn = b0n + b1n * X_norm

print("Normal Equation MSE:", mean_squared_error(y, yn))
print("Normal Equation R2",r2_score(y,yn))

# Plot
#idx = np.argsort(X.flatten())

plt.scatter(X, y, label = "Actual Data points")
plt.plot(X,yg, label = "GD")
plt.plot(X,yn, '--',label = "NE")
plt.legend()
plt.show()

```
