import numpy as np
import pandas as pd
from methods import *

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

# testing sample
test_centroid = np.array([[1, 61.16330029, 53.26621592, 49.49883705, 47.19952614, 56.49864461, 59.01523447]])
test_width = np.array([[1, 12.46012455, 11.46095993, 18.1832342, 10.4336089, 10.1292637, 43.86395145]])
test_x_lower = test_centroid[:, 1:7] - test_width[:, 1:7] / 2
test_x_upper = test_centroid[:, 1:7] + test_width[:, 1:7] / 2

n, p = centroids.shape

test_x_lu = np.ones(1).reshape(1, 1)
for j in range(p):
    test_x_lu = np.insert(test_x_lu, test_x_lu.shape[1], test_x_lower[:, j], axis=1)
    test_x_lu = np.insert(test_x_lu, test_x_lu.shape[1], test_x_upper[:, j], axis=1)

# Constrained center and range method
beta_c, beta_r = CCRM_Method(centroids, overall_centroids, widths, overall_widths)

est_test_centroid = np.dot(test_centroid, beta_c)
est_test_width = np.dot(test_width, beta_r)

print("CCRM method, estimated overall centroid ", est_test_centroid[0, 0])
print("CCRM method, estimated overall width ", est_test_width[0, 0])

est_centroids = beta_c[0, 0] + np.dot(centroids, beta_c[1:, 0])
est_widths = beta_r[0, 0] + np.dot(widths, beta_r[1:, 0])
rmse_l, rmsl_u = RMSE(overall_centroids, est_centroids, overall_widths, est_widths)
mmer_CCRM = MMER(overall_centroids, est_centroids, overall_widths, est_widths)
print("CCRM method, RMSE_L and RMSE_U ", rmse_l, rmsl_u)
print("CCRM method, MMER ", mmer_CCRM, "\n")

# Parametrized method
beta_l, beta_u = PM_Method(centroids, overall_centroids, widths, overall_widths)

est_test_y_lower = np.dot(test_x_lu, beta_l)
est_test_y_upper = np.dot(test_x_lu, beta_u)
est_test_centroid = (est_test_y_upper + est_test_y_lower) / 2
est_test_width = est_test_y_upper - est_test_y_lower

print("PM method, estimated overall centroid ", est_test_centroid[0, 0])
print("PM method, estimated overall width ", est_test_width[0, 0])

x_lower = centroids - widths / 2
x_upper = centroids + widths / 2
x_lu = np.ones(n).reshape(n, 1)
for j in range(p):
    x_lu = np.insert(x_lu, x_lu.shape[1], x_lower[:, j], axis=1)
    x_lu = np.insert(x_lu, x_lu.shape[1], x_upper[:, j], axis=1)

est_y_lower = np.dot(x_lu, beta_l)
est_y_upper = np.dot(x_lu, beta_u)
est_centroids = (est_y_lower + est_y_upper) / 2
est_widths = est_y_upper - est_y_lower

rmse_l, rmsl_u = RMSE(overall_centroids, est_centroids, overall_widths, est_widths)
mmer_PM = MMER(overall_centroids, est_centroids, overall_widths, est_widths)
print("PM method, RMSE_L and RMSE_U ", rmse_l, rmsl_u)
print("PM method, MMER ", mmer_PM, "\n")
