import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.metrics import mean_squared_error
import itertools

# Sample coordinates
coordinates = [
    (0, 0),
    (0.1, 0.37),
    (1, 1),
    (2, 2),
    (3, 1),
    (4, 0),
    (5, 1),
    (6, 2),
]

# Separate the coordinates into x and y arrays
x = np.array([coord[0] for coord in coordinates])
y = np.array([coord[1] for coord in coordinates])

# Function to calculate BIC
def bic(y, y_pred, num_params):
    mse = mean_squared_error(y, y_pred)
    n = len(y)
    return n * np.log(mse) + num_params * np.log(n)

# Function to fit piecewise cubic splines with given knots
def fit_splines(x, y, knots):
    splines = []
    for i in range(len(knots) - 1):
        x_sub = x[(x >= knots[i]) & (x <= knots[i+1])]
        y_sub = y[(x >= knots[i]) & (x <= knots[i+1])]
        try:
            spline = CubicSpline(x_sub, y_sub)
            splines.append(spline)
        except:
            pass
    return splines

# Function to predict y values with fitted piecewise cubic splines
def predict_splines(x, splines, knots):
    y_pred = np.empty_like(x)
    for i, spline in enumerate(splines):
        idx = (x >= knots[i]) & (x <= knots[i+1])
        y_pred[idx] = spline(x[idx])
    return y_pred

# Function to find the best breakpoints using BIC
def find_best_breakpoints(x, y, max_breakpoints):
    best_bic = np.inf
    best_knots = None
    best_splines = None
    n = len(x)

    for num_breakpoints in range(1, max_breakpoints + 1):
        for knots in itertools.combinations(range(1, n), num_breakpoints):
            knots = (0,) + knots + (n - 1,)
            splines = fit_splines(x, y, knots)
            y_pred = predict_splines(x, splines, knots)
            cur_bic = bic(y, y_pred, (num_breakpoints + 1) * 4)

            if cur_bic < best_bic:
                best_bic = cur_bic
                best_knots = knots
                best_splines = splines

    return best_knots, best_splines

# Find the best breakpoints and fit the piecewise cubic splines
max_breakpoints = 5
best_knots, best_splines = find_best_breakpoints(x, y, max_breakpoints)

# Plot the coordinates and the fitted piecewise cubic splines
plt.scatter(x, y, color='blue', label='Coordinates')

x_line = np.linspace(x.min(), x.max(), 100)
for i, spline in enumerate(best_splines):
    x_spline = x_line[(x_line >= best_knots[i]) & (x_line <= best_knots[i+1])]
    y_spline = spline(x_spline)
    plt.plot(x_spline, y_spline, label=f'Spline {i + 1}')

plt.legend()
plt.show()


