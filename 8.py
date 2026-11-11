import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Load datasets
train_df = pd.read_csv("fashion-mnist_train.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")

# 2. Separate features and labels
X_train = train_df.drop('label', axis=1).values
y_train = train_df['label'].values

X_test = test_df.drop('label', axis=1).values
y_test = test_df['label'].values

# 3. Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Use a smaller subset to reduce execution time
X_train = X_train[:10000]
y_train = y_train[:10000]

X_test = X_test[:2000]
y_test = y_test[:2000]

# 4. Experiment with different K values
k_values = [1, 3, 5, 7, 9]

accuracies = []

for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    accuracies.append(acc)

    print("K =", k)
    print("Accuracy:", acc)
    print()

# 5. Find best K
best_index = np.argmax(accuracies)

print("Best K:", k_values[best_index])
print("Best Accuracy:", accuracies[best_index])

# 6. Visualization
plt.plot(k_values, accuracies, marker='o')

plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("KNN Performance for Different K Values")
plt.xticks(k_values)
plt.grid()
plt.show()
