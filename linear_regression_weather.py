import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('weather.csv')

# Select the features (independent variables) and target variable
x = df[["MaxTemp", "Rainfall", "Evaporation"]]
y = df["MinTemp"]

# Instantiate the model
model_linReg = LinearRegression()

# Train the model
model_linReg.fit(x, y)

# Make predictions
prediction = model_linReg.predict(x)

# Visualize actual vs predicted values with regression line
plt.scatter(y, prediction, label="Actual vs Predicted")
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', linewidth=2, label="Regression Line")
plt.xlabel("Actual MinTemp")
plt.ylabel("Predicted MinTemp")
plt.title("Actual vs Predicted MinTemp with Regression Line")
plt.legend()
plt.show()
