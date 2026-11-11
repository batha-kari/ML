
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("fashion-mnist_train.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")

# Separate features and labels
X_train = train_df.drop('label', axis=1).values
y_train = train_df['label'].values

X_test = test_df.drop('label', axis=1).values
y_test = test_df['label'].values

# Select samples
X_train = X_train[:5000]
y_train = y_train[:5000]

X_test = X_test[:1000]
y_test = y_test[:1000]

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0


# Define SVM kernels
models = {
    'Linear': SVC(kernel='linear', C=1),
    'Polynomial': SVC(kernel='poly', degree=3, C=1, gamma='scale'),
    'RBF': SVC(kernel='rbf', C=1, gamma='scale')
}

accuracies = []
training_times = []

# Train and evaluate each model
for name, model in models.items():

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    elapsed_time = end - start

    accuracies.append(accuracy)
    training_times.append(elapsed_time)

    print("\n", name, "Kernel")
    print("Accuracy:", accuracy)
    print("Training Time:", round(elapsed_time, 2), "seconds")

# Compare accuracy
plt.figure(figsize=(8, 5))
plt.bar(models.keys(), accuracies)
plt.xlabel("SVM Kernel")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of SVM Kernels")
plt.ylim(0, 1)

# Compare training time
plt.figure(figsize=(8, 5))
plt.bar(models.keys(), training_times)
plt.xlabel("SVM Kernel")
plt.ylabel("Training Time (seconds)")
plt.title("Training Time Comparison of SVM Kernels")
plt.show()
