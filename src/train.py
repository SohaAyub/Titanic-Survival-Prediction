# ==========================================
# Titanic Survival Prediction - train.py
# ==========================================

# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load Dataset
df = pd.read_csv("../data/Titanic-Dataset.csv")

# ===============================
# Data Cleaning
# ===============================

# Fill missing Age values
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Remove unnecessary columns
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

# ===============================
# Encoding
# ===============================

encoder = LabelEncoder()

df["Sex"] = encoder.fit_transform(df["Sex"])
df["Embarked"] = encoder.fit_transform(df["Embarked"])

# ===============================
# Feature Selection
# ===============================

X = df.drop("Survived", axis=1)
y = df["Survived"]

# ===============================
# Train Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===============================
# Model Training
# ===============================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# Prediction
# ===============================

y_pred = model.predict(X_test)

# ===============================
# Evaluation
# ===============================

print("="*40)
print("Accuracy Score")
print("="*40)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print("\n")

print("="*40)
print("Confusion Matrix")
print("="*40)

print(confusion_matrix(y_test, y_pred))

print("\n")

print("="*40)
print("Classification Report")
print("="*40)

print(classification_report(y_test, y_pred))

# ===============================
# Save Model
# ===============================

joblib.dump(model, "../models/Titanic_Model.pkl")

print("\nModel Saved Successfully!")
