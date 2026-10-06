import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("dataset/data.csv")
print(df.head())
df.info()
print(df.isnull().sum())
df = df.drop(columns=["id", "Unnamed: 32"])
print(df.shape)
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]
print(X.shape)
print(y.shape)
y = y.map({"B": 0, "M": 1})

print(y.head())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train[:5])
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model trained successfully")
y_pred = model.predict(X_test)

print("Predictions:")
print(y_pred[:10])
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
y_prob = model.predict_proba(X_test)[:, 1]

print("First 10 probabilities:")
print(y_prob[:10])
from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC:", roc_auc)
threshold = 0.3

y_pred_threshold = (y_prob >= threshold).astype(int)

print("Predictions with threshold 0.3:")
print(y_pred_threshold[:10])
precision_threshold = precision_score(y_test, y_pred_threshold)
recall_threshold = recall_score(y_test, y_pred_threshold)

print("Threshold:", threshold)
print("Precision:", precision_threshold)
print("Recall:", recall_threshold)
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-10, 10, 100)

sigmoid = 1 / (1 + np.exp(-z))

plt.plot(z, sigmoid)
plt.xlabel("Input (z)")
plt.ylabel("Sigmoid output")
plt.title("Sigmoid Function")
plt.grid()

plt.savefig("sigmoid_curve.png")
plt.show()