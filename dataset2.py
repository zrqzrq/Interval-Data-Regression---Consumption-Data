import numpy as np
from scipy.stats import uniform
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

np.random.seed(9)

#number of intervals
n=1000


C_x=uniform.rvs(size=n,loc=-10,scale=20)
delta_x=uniform.rvs(size=n,loc=0,scale=5)

x_lower=C_x-delta_x/2
x_upper=C_x+delta_x/2


(beta0,)=uniform.rvs(size=1,loc=-67.5,scale=5)
(beta1,beta3)=uniform.rvs(size=2,loc=-5,scale=10)
(beta2,)=uniform.rvs(size=1,loc=62.5,scale=5)
(lambda_p,lambda_q)=uniform.rvs(size=2,loc=0,scale=1)

eps=norm.rvs(size=n,loc=0,scale=1)

p_x=(1-lambda_p)*x_lower+lambda_p*x_upper
q_x=(1-lambda_q)*x_lower+lambda_q*x_upper

y_lower=beta0+beta1*p_x+eps
y_upper=beta2+beta3*q_x+eps


fig, ax = plt.subplots()
for i in range(n):
	#coordinate of the lower left corner
	x0=x_lower[i]
	y0=y_lower[i]
	#width and height of the rectangle
	r_width=delta_x[i]
	r_height=y_upper[i]-y_lower[i]
	#add rectangle to the plot
	ax.add_patch(Rectangle((x0, y0), r_width, r_height,lw=0.5,ec="black",fc="None"))


ax.plot()
ax.set_xlabel("x")
ax.set_ylabel("y")

plt.show()
