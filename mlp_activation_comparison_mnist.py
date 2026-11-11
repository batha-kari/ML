
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load single MNIST CSV file
df = pd.read_csv("mnist_784.csv")

# Separate features and target
X = df.drop('class', axis=1).values
y = df['class'].values

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Select smaller subsets
X_train = X_train[:10000]
y_train = y_train[:10000]

X_test = X_test[:2000]
y_test = y_test[:2000]

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define activation functions
activations = {
    'Sigmoid': 'logistic',
    'ReLU': 'relu',
    'Tanh': 'tanh'
}

accuracies = []
training_times = []
iterations = []

# Train and evaluate each network
for name, activation in activations.items():

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation=activation,
        solver='adam',
        max_iter=50,
        random_state=42
    )

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    elapsed = end - start

    accuracies.append(accuracy)
    training_times.append(elapsed)
    iterations.append(model.n_iter_)

    print("\nActivation:", name)
    print("Accuracy:", round(accuracy, 4))
    print("Training Time:", round(elapsed, 2), "seconds")
    print("Iterations:", model.n_iter_)

# Compare accuracy
plt.figure(figsize=(7, 5))
plt.bar(activations.keys(), accuracies)
plt.xlabel("Activation Function")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")

# Compare training time
plt.figure(figsize=(7, 5))
plt.bar(activations.keys(), training_times)
plt.xlabel("Activation Function")
plt.ylabel("Training Time (seconds)")
plt.title("Training Time Comparison")

# Compare convergence
plt.figure(figsize=(7, 5))
plt.bar(activations.keys(), iterations)
plt.xlabel("Activation Function")
plt.ylabel("Iterations")
plt.title("Convergence Comparison")
plt.show()
