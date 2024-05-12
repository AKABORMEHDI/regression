import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_csv('weather.csv')

# Select the features (independent variables) and target variable
x = df[['MaxTemp', 'Rainfall', 'Evaporation']]
y = df['MinTemp']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Fit a polynomial regression model
degree = 2  # Adjust the degree as needed
poly_features = PolynomialFeatures(degree=degree)
x_train_poly = poly_features.fit_transform(x_train)
x_test_poly = poly_features.transform(x_test)

model_poly = LinearRegression()
model_poly.fit(x_train_poly, y_train)

# Make predictions on the test set
y_pred_poly = model_poly.predict(x_test_poly)

# Evaluate the model
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f'Mean Squared Error (Polynomial Regression): {mse_poly:.2f}')
print(f'R-squared (Polynomial Regression): {r2_poly:.2f}')

# Visualize the results
# Sort the values for a smooth curve
sort_idx = np.argsort(x_test['MaxTemp'])
plt.scatter(x_test['MaxTemp'], y_test, color='blue', label='Actual MinTemp')
plt.plot(x_test['MaxTemp'].iloc[sort_idx], y_pred_poly[sort_idx], color='red', label='Predicted MinTemp (Polynomial)')
plt.xlabel('MaxTemp')
plt.ylabel('MinTemp')
plt.title('Polynomial Regression')
plt.legend()
plt.show()
