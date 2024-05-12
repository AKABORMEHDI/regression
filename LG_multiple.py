import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Visualize actual vs predicted values in 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df["MaxTemp"], df["Rainfall"], y, label='Actual MinTemp', c='blue', marker='o')
ax.scatter(df["MaxTemp"], df["Rainfall"], prediction, label='Predicted MinTemp', c='red', marker='^')

ax.set_xlabel('MaxTemp')
ax.set_ylabel('Rainfall')
ax.set_zlabel('MinTemp')
ax.set_title('Actual vs Predicted MinTemp in 3D')
ax.legend()

plt.show()
