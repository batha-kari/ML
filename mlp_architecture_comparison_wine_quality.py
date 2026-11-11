import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load Wine Quality dataset
df = pd.read_csv("winequality-red.csv")

# Separate features and target
X = df.drop('quality', axis=1)
y = df['quality']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define different MLP architectures
architectures = {
    '1 Layer (32)': (32,),
    '1 Layer (64)': (64,),
    '2 Layers (64,32)': (64, 32),
    '2 Layers (128,64)': (128, 64),
    '3 Layers (128,64,32)': (128, 64, 32)
}

rmse_scores = []
mae_scores = []
r2_scores = []

# Train and evaluate each architecture
for name, layers in architectures.items():

    model = MLPRegressor(
        hidden_layer_sizes=layers,
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    rmse_scores.append(rmse)
    mae_scores.append(mae)
    r2_scores.append(r2)

    print("\n", name)
    print("RMSE:", round(rmse, 4))
    print("MAE:", round(mae, 4))
    print("R2 Score:", round(r2, 4))
    print("Training Time:", round(end-start, 2), "seconds")


# Plot 1: RMSE and MAE comparison
plt.figure(figsize=(10, 5))

plt.plot(architectures.keys(), rmse_scores,
         marker='o', label='RMSE')

plt.plot(architectures.keys(), mae_scores,
         marker='s', label='MAE')

plt.xlabel("MLP Architecture")
plt.ylabel("Error")
plt.title("Performance Comparison of MLP Architectures")
plt.legend()
plt.grid(True)
plt.tight_layout()


# Plot 2: R2 Score comparison
plt.figure(figsize=(10, 5))

plt.bar(architectures.keys(), r2_scores)

plt.xlabel("MLP Architecture")
plt.ylabel("R2 Score")
plt.title("R2 Score for Different MLP Architectures")
plt.tight_layout()
plt.show()
