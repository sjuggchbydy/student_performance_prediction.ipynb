# ---------------------------------------------
# Student Performance Prediction (ML Mini Project)
# ---------------------------------------------

# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------
# Sample Dataset (You can replace it with CSV)
# ---------------------------------------------

data = {
    'StudyHours': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Attendance': [60, 65, 70, 72, 75, 78, 80, 85, 90, 95],
    'InternalMarks': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    'FinalScore': [50, 55, 60, 62, 65, 70, 72, 78, 82, 88]
}

df = pd.DataFrame(data)

# ---------------------------------------------
# Exploratory Data Analysis
# ---------------------------------------------

print("\nDataset Preview:")
print(df.head())

print("\nDataset Summary:")
print(df.describe())

# Visualizing correlations
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

# ---------------------------------------------
# Splitting Data
# ---------------------------------------------

X = df[['StudyHours', 'Attendance', 'InternalMarks']]
y = df['FinalScore']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ---------------------------------------------
# Model 1: Linear Regression
# ---------------------------------------------

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

print("\n--- Linear Regression Results ---")
print("MSE:", mean_squared_error(y_test, lr_predictions))
print("R² Score:", r2_score(y_test, lr_predictions))

# ---------------------------------------------
# Model 2: Decision Tree Regressor
# ---------------------------------------------

dt_model = DecisionTreeRegressor(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)

print("\n--- Decision Tree Results ---")
print("MSE:", mean_squared_error(y_test, dt_predictions))
print("R² Score:", r2_score(y_test, dt_predictions))

# ---------------------------------------------
# Predicting new student performance
# ---------------------------------------------

new_student = [[8, 85, 21]]  # StudyHours, Attendance, InternalMarks
predicted_score = lr_model.predict(new_student)

print("\nPredicted Final Score for New Student:", predicted_score[0])
