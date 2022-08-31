import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from methods import *
import sys

# read data from csv files
raw_centroids = pd.read_csv('data/Centroids.csv',
                            usecols=["Visual_Appeal", "Value_for_Money", "Healthiness", "Taste", "Branding", "Ethics"])
raw_widths = pd.read_csv('data/Widths.csv',
                         usecols=["Visual_Appeal", "Value_for_Money", "Healthiness", "Taste", "Branding", "Ethics"])
raw_overall_centroids = pd.read_csv('data/Overall_Centroids.csv', usecols=["Overall_Centroids"])
raw_overall_widths = pd.read_csv('data/Overall_Widths.csv', usecols=["Overall_Widths"])

# transfer raw data into ndarray
centroids = raw_centroids.to_numpy()
widths = raw_widths.to_numpy()
overall_centroids = raw_overall_centroids.to_numpy()
overall_widths = raw_overall_widths.to_numpy()

n, p = centroids.shape

x_lower = centroids - widths / 2
x_upper = centroids + widths / 2
x_lu = np.ones(n).reshape(n, 1)

for j in range(p):
    x_lu = np.insert(x_lu, x_lu.shape[1], x_lower[:, j], axis=1)
    x_lu = np.insert(x_lu, x_lu.shape[1], x_upper[:, j], axis=1)

# Parametrized method
beta_l, beta_u = PM_Method(centroids, overall_centroids, widths, overall_widths)

j = int(sys.argv[1])

colors = ["red", "green", "blue", "purple", "cyan", "darkorange"]
x_names = ["Visual Appeal", "Value for Money", "Healthiness", "Taste", "Branding", "Ethics"]

fig, ax = plt.subplots()

for i in range(n):
    # x coordinate of lower left corner
    x0 = x_lower[i, j]
    # y coordinate of lower left corner
    y0 = beta_l[0, 0] + beta_l[2 * j + 1, 0] * x_lower[i, j] + beta_l[2 * j + 2, 0] * x_upper[i, j]
    # y coordinate of top right corner
    y1 = beta_u[0, 0] + beta_u[2 * j + 1, 0] * x_lower[i, j] + beta_u[2 * j + 2, 0] * x_upper[i, j]

    ax.add_patch(Rectangle((x0, y0), widths[i, j], y1 - y0, lw=0.5, ec=colors[j], fc="None"))

ax.plot()
ax.set_xlabel(x_names[j])
ax.set_ylabel("Estimated Overall Intention")
ax.set_title("PM Linear Regression Method")

plt.show()
