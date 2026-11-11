
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load Digits CSV dataset
df = pd.read_csv("digits.csv")

# Separate features and labels
X = df.drop('label', axis=1).values
y = df['label'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Different numbers of clusters
k_values = [5, 8, 10, 12, 15]

inertias = []
silhouette_scores = []

# Apply K-Means for different K values
for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    # Calculate evaluation metrics
    inertia = model.inertia_
    score = silhouette_score(X_scaled, labels)

    inertias.append(inertia)
    silhouette_scores.append(score)

    print("\nNumber of Clusters:", k)
    print("Inertia:", round(inertia, 2))
    print("Silhouette Score:", round(score, 4))

# Plot Elbow Method
plt.figure(figsize=(7, 5))
plt.plot(k_values, inertias, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid()

# Plot Silhouette Scores
plt.figure(figsize=(7, 5))
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Comparison")
plt.grid()

# Visualize sample digit images
images = X[:5].reshape(-1, 8, 8)

fig, axes = plt.subplots(1, 5, figsize=(10, 3))

for i, ax in enumerate(axes):
    ax.imshow(images[i], cmap='gray')
    ax.axis('off')

plt.suptitle("Sample Digits")
plt.show()
