import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor

np.random.seed(0)

# Generate synthetic data
x, y = make_regression(n_samples=100, n_features=1, noise=10)

# Scatter plot of the data
plt.scatter(x, y)

# Fit the model with SGD and plot the regression line
model = SGDRegressor(max_iter=100, eta0=0.0001)
model.fit(x, y)
print('Coeff R2 =', model.score(x, y))
plt.scatter(x, y)
plt.plot(x, model.predict(x), c='red', lw=3)

# Fit the model with more iterations and plot the regression line
model = SGDRegressor(max_iter=1000, eta0=0.001)
model.fit(x, y)
print('Coeff R2 =', model.score(x, y))
plt.scatter(x, y)
plt.plot(x, model.predict(x), c='red', lw=3)

# Fit the model with cost history tracking
model = SGDRegressor(max_iter=1000, eta0=0.001, warm_start=True)
cost_history = []
for _ in range(1000):
    model.partial_fit(x, y)
    cost_history.append(model.score(x, y))  # Using score as a measure of cost

# Plot cost history
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_ylabel('R^2 (coefficient of determination)')
ax.set_xlabel('Iterations')
_ = ax.plot(range(1, len(cost_history) + 1), cost_history)
plt.show()
