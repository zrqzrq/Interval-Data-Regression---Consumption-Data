import numpy as np
from scipy.stats import uniform
from scipy.stats import norm
from methods import *

np.random.seed(9)
# number of intervals
n = 1000

C_x = uniform.rvs(size=n, loc=-10, scale=20)
delta_x = uniform.rvs(size=n, loc=0, scale=5)

x_lower = C_x - delta_x / 2
x_upper = C_x + delta_x / 2

(beta0,) = uniform.rvs(size=1, loc=-67.5, scale=5)
(beta1, beta3) = uniform.rvs(size=2, loc=-5, scale=10)
(beta2,) = uniform.rvs(size=1, loc=62.5, scale=5)
(lambda_p, lambda_q) = uniform.rvs(size=2, loc=0, scale=1)

eps = norm.rvs(size=n, loc=0, scale=1)

p_x = (1 - lambda_p) * x_lower + lambda_p * x_upper
q_x = (1 - lambda_q) * x_lower + lambda_q * x_upper

y_lower = beta0 + beta1 * p_x + eps
y_upper = beta2 + beta3 * q_x + eps

C_y = (y_lower + y_upper) / 2
delta_y = y_upper - y_lower

# CCRM Method
beta_c, beta_r = CCRM_Method(C_x.reshape(n, 1), C_y.reshape(n, 1), delta_x.reshape(n, 1), delta_y.reshape(n, 1))

est_C_y = beta_c[0, 0] + np.dot(C_x.reshape(n, 1), beta_c[1:, 0])
est_delta_y = beta_r[0, 0] + np.dot(delta_x.reshape(n, 1), beta_r[1:, 0])

mmer = MMER(C_y, est_C_y, delta_y, est_delta_y)
rmse_l, rmse_u = RMSE(C_y, est_C_y, delta_y, est_delta_y)

print("Real beta0 ", beta0)
print("Real beta1 ", beta1)
print("Real beta2 ", beta2)
print("Real beta3 ", beta3, "\n")

print("CCRM method, estimated beta0 ", beta_c[0, 0] - beta_r[0, 0] / 2)
print("CCRM method, estimated beta2 ", beta_c[0, 0] + beta_r[0, 0] / 2)
print("CCRM method, MMER ", mmer)
print("CCRM method, RMSE_L and RMSE_U ", rmse_l, rmse_u, "\n")

# PM Method
beta_l, beta_u = PM_Method(C_x.reshape(n, 1), C_y.reshape(n, 1), delta_x.reshape(n, 1), delta_y.reshape(n, 1))

x_lower = C_x - delta_x / 2
x_upper = C_x + delta_x / 2
x_lu = np.ones(n).reshape(n, 1)
x_lu = np.insert(x_lu, x_lu.shape[1], x_lower, axis=1)
x_lu = np.insert(x_lu, x_lu.shape[1], x_upper, axis=1)

est_y_lower = np.dot(x_lu, beta_l)
est_y_upper = np.dot(x_lu, beta_u)
est_C_y = (est_y_lower + est_y_upper) / 2
est_delta_y = est_y_upper - est_y_lower

mmer = MMER(C_y, est_C_y, delta_y, est_delta_y)
rmse_l, rmse_u = RMSE(C_y, est_C_y, delta_y, est_delta_y)

print("PM method, estimated beta0 ", beta_l[0, 0])
print("PM method, estimated beta1 ", (beta_l[1, 0] / (1 - lambda_p) + beta_l[2, 0] / lambda_p) / 2)
print("PM method, estimated beta2 ", beta_u[0, 0])
print("PM method, estimated beta3 ", (beta_u[1, 0] / (1 - lambda_q) + beta_u[2, 0] / lambda_q) / 2)
print("PM method, MMER ", mmer)
print("PM method, RMSE_L and RMSE_U ", rmse_l, rmse_u, "\n")
