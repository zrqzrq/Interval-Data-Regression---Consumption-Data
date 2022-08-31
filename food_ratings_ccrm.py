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

# Constrained center and range method
beta_c, beta_r = CCRM_Method(centroids, overall_centroids, widths, overall_widths)
print(beta_r)
print(beta_c)

j = int(sys.argv[1])

colors = ["red", "green", "blue", "purple", "cyan", "darkorange"]
x_names = ["Visual Appeal", "Value for Money", "Healthiness", "Taste", "Branding", "Ethics"]

fig, ax = plt.subplots()

for i in range(n):
    # x coordinate of lower left corner
    x0 = centroids[i, j] - widths[i, j] / 2
    # y coordinate of lower left corner
    C_y0 = beta_c[0, 0] + beta_c[j + 1, 0] * centroids[i, j]
    delta_y0 = beta_r[0, 0] + beta_r[j + 1, 0] * widths[i, j]
    y0 = C_y0 - delta_y0 / 2

    ax.add_patch(Rectangle((x0, y0), widths[i, j], delta_y0, lw=0.5, ec=colors[j], fc="None"))

ax.plot()
ax.set_xlabel(x_names[j])
ax.set_ylabel("Estimated Overall Intention")
ax.set_title("CCRM Linear Regression Method")

plt.show()
