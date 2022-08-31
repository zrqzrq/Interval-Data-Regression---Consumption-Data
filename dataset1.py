import numpy as np
from scipy.stats import uniform
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from methods import *

np.random.seed(9)

# number of intervals
n = 1000

C_x = uniform.rvs(size=n, loc=-10, scale=20)
delta_x = uniform.rvs(size=n, loc=0, scale=5)

(beta0, beta1) = uniform.rvs(size=2, loc=-5, scale=10)
(beta2, beta3) = uniform.rvs(size=2, loc=0, scale=5)

eps_C = norm.rvs(size=n, loc=0, scale=1)
eps_delta = norm.rvs(size=n, loc=0, scale=1)

C_y = beta0 + beta1 * C_x + eps_C
delta_y = beta2 + beta3 * delta_x + abs(eps_delta)

fig, ax = plt.subplots()
for i in range(n):
    # coordinate of the lower left corner
    x0 = C_x[i] - delta_x[i] / 2
    y0 = C_y[i] - delta_y[i] / 2
    # add rectangle to the plot
    ax.add_patch(Rectangle((x0, y0), delta_x[i], delta_y[i], lw=0.5, ec="black", fc="None"))

ax.plot()
ax.set_xlabel("x")
ax.set_ylabel("y")

plt.show()
