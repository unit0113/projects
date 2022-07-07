import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

factor = 'TV'
ads = pd.read_csv('NNs\ISL\data\Advertising.csv')
sales_mean = ads['sales'].mean()
tv_mean = ads[factor].mean()

# Calc least squares
slope_top = slope_bottom = 0
for index in range(len(ads)):
    slope_top += (ads[factor][index] - tv_mean) * (ads['sales'][index] - sales_mean)
    slope_bottom += (ads[factor][index] - tv_mean) ** 2
slope = slope_top / slope_bottom

# Fit line
intercept = sales_mean - slope * tv_mean
regression_x = np.linspace(0,300, 600)
regression_y = intercept + slope * regression_x

figure = plt.figure()
figure.set_figwidth(16)
figure.set_figheight(12)

plt.scatter(ads[factor], ads['sales'], color='red', s=6)
plt.plot(regression_x, regression_y)
plt.title('TV Advertising Expendature vs Sales')
plt.xlabel('TV Advertising ($1000\'s)')
plt.ylabel('Sales')

# Residuals
RSS = TSS = 0
for index in range(len(ads)):
    x = [ads[factor][index]] * 2
    y = [intercept + slope * ads[factor][index], ads['sales'][index]]
    plt.plot(x, y, color='gray', linewidth=1)

    # Also calc RSS/TSS
    RSS += (ads['sales'][index] - intercept - slope * ads[factor][index]) ** 2
    TSS += (ads['sales'][index] - sales_mean) ** 2

# Other stats
RSE = math.sqrt(RSS / (len(ads) - 2))
sd = np.std(ads['sales'])
SE = sd ** 2 / len(ads)
var_intercept = (RSE ** 2) * ((1 / len(ads)) + (tv_mean ** 2) / slope_bottom)
var_slope = (RSE ** 2) / slope_bottom
con_int_intercept = [intercept - 2 * math.sqrt(var_intercept), intercept + 2 * math.sqrt(var_intercept)]
con_int_slope = [slope - 2 * math.sqrt(var_slope), slope + 2 * math.sqrt(var_slope)]
t_intercept = intercept / math.sqrt(var_intercept)
t_slope = slope / math.sqrt(var_slope)
r_sqrd = 1 - RSS / TSS
cor = slope_top / (slope_bottom * math.sqrt(TSS))

plt.show()