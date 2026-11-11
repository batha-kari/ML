
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load Fashion MNIST dataset
train_df = pd.read_csv("fashion-mnist_train.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")

# Separate features and labels
X_train = train_df.drop('label', axis=1).values[:5000]
y_train = train_df['label'].values[:5000]

X_test = test_df.drop('label', axis=1).values[:1000]
y_test = test_df['label'].values[:1000]

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Hyperparameter combinations
learning_rates = [0.001, 0.01]
batch_sizes = [32, 64]
epochs = [20, 40]

results = []

# Experiment with different hyperparameters
for lr in learning_rates:
    for batch in batch_sizes:
        for epoch in epochs:

            model = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                solver='adam',
                learning_rate_init=lr,
                batch_size=batch,
                max_iter=epoch,
                random_state=42
            )

            start = time.time()
            model.fit(X_train, y_train)
            end = time.time()

            # Predict test labels
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            elapsed = end - start

            results.append([
                lr, batch, epoch, accuracy, elapsed
            ])

            print("\nLearning Rate:", lr)
            print("Batch Size:", batch)
            print("Epochs:", epoch)
            print("Accuracy:", round(accuracy, 4))
            print("Training Time:", round(elapsed, 2), "seconds")

# Convert results into DataFrame
result_df = pd.DataFrame(
    results,
    columns=['Learning Rate', 'Batch Size',
             'Epochs', 'Accuracy', 'Training Time']
)

print("\nHyperparameter Comparison:")
print(result_df)

# Plot accuracy comparison
plt.figure(figsize=(10, 5))
plt.bar(range(len(results)), result_df['Accuracy'])
plt.xlabel("Hyperparameter Combination")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.xticks(range(len(results)), range(1, len(results)+1))

# Plot training time comparison
plt.figure(figsize=(10, 5))
plt.bar(range(len(results)), result_df['Training Time'])
plt.xlabel("Hyperparameter Combination")
plt.ylabel("Training Time (seconds)")
plt.title("Training Time Comparison")
plt.xticks(range(1, len(results)+1))
plt.show()

# Display best combination
best = result_df.loc[result_df['Accuracy'].idxmax()]
print("\nBest Hyperparameters:")
print(best)
