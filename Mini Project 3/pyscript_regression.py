# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # MSE Module
# %% [markdown]
# ### loading data frame from csv file

# %%
import pandas as pd
data = pd.read_csv('GOOGL.csv', index_col=0, header=0, parse_dates=True)
train = len(data) - 10
head = data.head(train)['Open'].to_numpy()


# %%
from matplotlib import pyplot as plt
plt.plot(head, color="blue", label="training records")
plt.legend()
plt.show()

# %% [markdown]
# ### calculating coeffitients of 3 regression among head(training) records 
# #### regression 1 linear model: b0 + b1x
# #### regression 2 quadratic model: b0 + b1x + b2x^2
# #### regression 3 linear-sinusoidial model: b0 + b1x + b2 sin(wx)
# #### regression 4 exponential model: a0(a1^x) or exp(b0 + b1x) - b0=log(a0), b1=log(a1)

# %%
import numpy as np
import math as ma
from numpy import matrix as mt
from numpy import linalg as la
##
# domain = np.arange(1., train + 1, 1)
# head = 62+0.24*domain+100*np.sin(2 * ma.pi / 500 * domain)
reg_detail = {
    0:"linear(simple) regression",
    1:"quadratic regression",
    2:"linear-sinusoidal regression",
    3:"exponential regression"
}
## 
x_t = np.arange(1, train + 1, 1)
x_t = [x_t, np.copy(x_t), np.copy(x_t), np.copy(x_t)]
##
x_t[0] = np.array([np.ones(train), x_t[0]])
x_t[1] = np.array([np.ones(train), x_t[1], x_t[1]**2])
x_t[2] = np.array([np.ones(train), x_t[2], np.sin(2 * ma.pi / 500 * x_t[2])])
x_t[3] = np.array([np.ones(train), x_t[3]])
## calculation
product = [np.ones(train), np.ones(train), np.ones(train), np.ones(train)]
for i in range(4):
    product[i] = np.copy(x_t[i])
    product[i] = np.matmul(product[i], mt.transpose(product[i]))
    product[i] = la.inv(product[i])
    product[i] = np.matmul(product[i], x_t[i])
    product[i] = np.matmul(product[i],(np.log(mt.transpose(head)) if (i == 3)  else mt.transpose(head))) # exponential must fit the logarithm of records


# %%
product

# %% [markdown]
# ### plotting whole records and regression points through whole domain

# %%
domain = np.arange(1., train + 11, 1)
regression = np.array([np.copy(domain), np.copy(domain), np.copy(domain), np.copy(domain)])
regression[0] = product[0][0] + product[0][1] * (regression[0])
regression[1] = product[1][0] + product[1][1] * (regression[1]) + product[1][2] * (regression[1]**2) 
regression[2] = product[2][0] + product[2][1] * (regression[2]) + product[2][2] * np.sin(2 * ma.pi / 500 * (regression[2]))
regression[3] = np.exp(product[3][0] + product[3][1] * (regression[3]))


# %%
plt.plot(regression[0], color='red', linewidth=3, label=reg_detail[0])
plt.plot(regression[1], color='orange', linewidth=3, label=reg_detail[1])
plt.plot(regression[2], color='black', label=reg_detail[2], linestyle='-.')
plt.plot(regression[3], color='blue', label=reg_detail[3], linestyle='--')
plt.scatter(domain, data.head(train + 10)['Open'], color='green', s=1, label="total records")
# plt.scatter(domain, 62+0.24*domain+100*np.sin(2 * ma.pi / 500 * domain), color='green', s=2, label="total records")
plt.legend()
plt.show()

# %% [markdown]
# ### calculating error of 10 tail records

# %%
reg_flag = -1
min_er = 10**10
##
error = np.array([0, 0, 0, 0])
tail = data.tail(10)["Open"].to_list()
for k in range(4):
    print("############ " + reg_detail[k] + " ############")
    for i in range(10):
        predicted = regression[k][train + i]
        e = predicted - tail[i]
        print("actual:" + str(tail[i]) + ", predicted:" + str(predicted) + ", error:" + str(e) + "\n")
        error[k] += e
        ##
        if abs(min_er) >= abs(error[k]):
            min_er = error[k]
            reg_flag = k;
    print("total predication error:" + str(error[k]) + "\n" + "mean error:" + str(error[k]/10) + "\n")
###
print("best prediction by " +  reg_detail[reg_flag])
plt.plot(regression[reg_flag], color='red', linestyle="--"  , label=reg_detail[reg_flag])
plt.scatter(domain, data.head(train + 10)['Open'], color='green', s=1, label="total records")
plt.legend()
plt.show()


# %%




