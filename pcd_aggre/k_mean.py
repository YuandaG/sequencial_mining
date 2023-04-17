import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [10, 20],
    [11, 22],
    [12, 24]
])

kmeans = KMeans(n_clusters=2)
labels = kmeans.fit_predict(data)

models = []
for label in np.unique(labels):
    cluster_points = data[labels == label]
    x = cluster_points[:, 0].reshape(-1, 1)
    y = cluster_points[:, 1].reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    models.append(model)

# Plot the original points
for label, color in zip(np.unique(labels), ['red', 'blue']):
    plt.scatter(data[labels == label, 0], data[labels == label, 1], c=color)

# Plot the two lines
x_range = np.linspace(data[:, 0].min(), data[:, 0].max(), 100).reshape(-1, 1)
for model, color in zip(models, ['red', 'blue']):
    plt.plot(x_range, model.predict(x_range), c=color)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Point Cloud with Two Lines")
plt.show()
