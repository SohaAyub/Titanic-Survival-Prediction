Titanic Survival Prediction using Machine Learning

Project Overview

This project predicts whether a passenger survived the Titanic disaster using Machine Learning. The model is trained on passenger information such as age, gender, ticket class, fare, and family details.

The project demonstrates the complete Machine Learning workflow including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and performance evaluation.

---

 Dataset

Dataset: Titanic-Dataset.csv

The dataset contains passenger information including:

- PassengerId
- Survived
- Pclass
- Name
- Sex
- Age
- SibSp
- Parch
- Ticket
- Fare
- Cabin
- Embarked

Target Variable:

- Survived
  - 1 = Survived
  - 0 = Did Not Survive

---

Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib
- Google Colab

---

Machine Learning Algorithm

Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple Decision Trees and uses majority voting to make accurate predictions.

---

Exploratory Data Analysis

The following visualizations were created:

- Survival Count
- Gender Distribution
- Survival by Gender
- Passenger Class Distribution
- Survival by Passenger Class
- Age Distribution
- Fare Distribution
- Correlation Heatmap

---
 Data Preprocessing

The following preprocessing steps were performed:

- Filled missing Age values using Median
- Filled missing Embarked values using Mode
- Removed Cabin column
- Removed PassengerId, Name, and Ticket columns
- Encoded categorical features using LabelEncoder

---

 Model Training

Model Used:

Random Forest Classifier

Dataset Split:

- Training Data: 80%
- Testing Data: 20%

---

Model Evaluation

Evaluation Metrics:

- Accuracy Score
- Confusion Matrix
- Classification Report
- Feature Importance

---

Project Structure

```text
Titanic-Survival-Prediction/
│
├── data/
├── images/
├── models/
├── notebook/
├── src/
├── README.md
├── requirements.txt
├── LICENSE
```

---

Future Improvements

- Hyperparameter Tuning
- Web Application using Streamlit
- Deploy using Hugging Face Spaces
- Try XGBoost and LightGBM
- Feature Engineering

---
Author

Soha Ayub

Software Engineering Student

Machine Learning & Data Science Enthusiast

---

If you found this project helpful, don't forget to star this repository.
