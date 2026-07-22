# ==========================================
# Titanic Survival Prediction - predict.py
# ==========================================

import joblib
import pandas as pd

# Load Saved Model
model = joblib.load("../models/Titanic_Model.pkl")

print("Titanic Survival Prediction")
print("-" * 40)

# ===============================
# User Input
# ===============================

pclass = int(input("Passenger Class (1,2,3): "))
sex = input("Sex (male/female): ").lower()
age = float(input("Age: "))
sibsp = int(input("Siblings/Spouse: "))
parch = int(input("Parents/Children: "))
fare = float(input("Fare: "))
embarked = input("Embarked (S/C/Q): ").upper()

# ===============================
# Encoding
# ===============================

if sex == "male":
    sex = 1
else:
    sex = 0

if embarked == "C":
    embarked = 0
elif embarked == "Q":
    embarked = 1
else:
    embarked = 2

# ===============================
# Create DataFrame
# ===============================

new_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked]
})

# ===============================
# Prediction
# ===============================

prediction = model.predict(new_data)

print("\nPrediction Result")
print("-" * 40)

if prediction[0] == 1:
    print("✅ Passenger Survived")
else:
    print("❌ Passenger Did Not Survive")
