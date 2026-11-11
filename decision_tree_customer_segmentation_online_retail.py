import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("online_retail.csv", encoding='latin1')

# 2. Preprocess dataset
df = df.dropna(subset=['CustomerID', 'InvoiceDate'])

# Remove cancelled transactions
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# Remove invalid quantities and prices
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Convert date column
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Calculate transaction amount
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# 3. Customer-wise aggregation
customer = df.groupby('CustomerID').agg({
    'TotalAmount': 'sum',
    'InvoiceNo': 'nunique',
    'Quantity': 'sum',
    'StockCode': 'nunique',
    'InvoiceDate': 'max',
    'UnitPrice': 'mean'
}).reset_index()

# Rename columns
customer.columns = [
    'CustomerID', 'TotalSpend', 'Frequency',
    'TotalQuantity', 'UniqueProducts',
    'LastPurchase', 'AvgUnitPrice'
]

# Calculate recency
latest_date = df['InvoiceDate'].max()

customer['Recency'] = (
    latest_date - customer['LastPurchase']
).dt.days

# 4. Create customer segments
customer['Segment'] = pd.qcut(
    customer['TotalSpend'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

# 5. Select features and target
X = customer[
    ['Frequency', 'TotalQuantity', 'UniqueProducts',
     'Recency', 'AvgUnitPrice']
]

y = customer['Segment']

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Implement ID3-style Decision Tree
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# 8. Predict customer segments
y_pred = model.predict(X_test)

print("DECISION TREE CLASSIFICATION")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. Display tree structure
print("\nDECISION TREE STRUCTURE:")
print(export_text(model, feature_names=list(X.columns)))

# 10. Visualize decision tree
plt.figure(figsize=(18, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True
)

plt.title("ID3 Decision Tree - Customer Segmentation")
plt.show()

# 11. Feature importance
importance = model.feature_importances_

plt.figure(figsize=(8, 5))

plt.bar(X.columns, importance)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance in Customer Segmentation")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 12. Print feature importance
print("\nFEATURE IMPORTANCE")

for feature, value in zip(X.columns, importance):
    print(feature, ":", round(value, 4))
