
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load downloaded dataset
df = pd.read_csv("Iris.csv")

# Remove unnecessary ID column if present
df = df.drop('Id',axis=1)

# Select features and target
X = df[['SepalLengthCm', 'SepalWidthCm']].values
y = (df['Species'] == 'Iris-setosa').astype(int).values

svm = SVC(kernel="linear", C=1)
svm.fit(X, y)

# 5. PREDICT
y_pred = svm.predict(X)
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)

# 6. GET DECISION BOUNDARY
w = svm.coef_[0]
b = svm.intercept_[0]

x_values = np.array([X[:, 0].min(), X[:, 0].max()])

# Decision boundary: w0*x + w1*y + b = 0
y_values = -(w[0] * x_values + b) / w[1]

# 7. GET MARGIN LINES
# Upper margin: w0*x + w1*y + b = 1
y_upper = -(w[0] * x_values + b - 1) / w[1]

# Lower margin: w0*x + w1*y + b = -1
y_lower = -(w[0] * x_values + b + 1) / w[1]

# 8. VISUALIZE
#plt.scatter(svm.support_vectors_[:, 0],
            #svm.support_vectors_[:, 1],
            #s=100, facecolors='none', edgecolors='green',
            #label="Support Vectors")
plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Setosa")
plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Non-Setosa")

plt.plot(x_values, y_values, label="Decision Boundary")
plt.plot(x_values, y_upper, "--", label="Margin")
plt.plot(x_values, y_lower, "--")

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Linear SVM - Iris Dataset")
plt.legend()
plt.show()
