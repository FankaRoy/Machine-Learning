# -*- coding: utf-8 -*-

#%%
#========================================#
# LINEAR REGRESSION AND MACHINE LEARNING #
#========================================#
#%%
import seaborn as sns
sns.set()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gas = pd.read_csv('./data/gas_consumption.csv', names=['tax', 'income', 'highway', 'drivers', 'gas'])
print(gas.head())

gas.plot(x='drivers', y='gas', kind='scatter', color='b')
plt.xlabel('% of population driving')
plt.ylabel('Gas consumption (millions of gallons)');

#%%
# Draw a line describing the trend in the data, but which is the best one?
gas.plot(x='drivers', y='gas', kind='scatter', color='b')
plt.xlabel('% of population driving')
plt.ylabel('Gas consumption (millions gallons)')

plt.plot([.4, .8], [300, 1000], 'r-')
plt.plot([.4, .8], [200, 1100], 'g-');

#%%
# Using the cost function or the loss function or the mean square erroy (MSE)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression(fit_intercept=True)
linreg.fit(gas[['drivers']], gas['gas'])

gas.plot(x='drivers', y='gas', kind='scatter', color='b')
plt.xlabel('% of population driving')
plt.ylabel('Gas consumption (millions gallons)')

x = np.linspace(.4, .8).reshape(-1, 1)
print(x.shape)
plt.plot(x, linreg.predict(x), 'k-')
plt.plot([.4, .8], [300, 1000], 'r-')
plt.plot([.4, .8], [200, 1100], 'g-');

print((linreg.intercept_, linreg.coef_[0]))

#%%
#=================
# GRADIENT DESCENT
#=================

beta0 = linreg.intercept_
beta1 = np.linspace(1300, 1500)

MSE = [((gas['gas'] - (beta0 + m * gas['drivers']))**2).sum() for m in beta1]

plt.plot(beta1, MSE);

#%%
#=========================================
# COMPARE THE MSE WITH THE HUBER LOSS FUNCTION
#=========================================

from sklearn.linear_model import HuberRegressor

huber = HuberRegressor(fit_intercept=True, alpha=0)
huber.fit(gas[['drivers']], gas['gas'])
gas.plot(x='drivers', y='gas', kind='scatter', color='b')
plt.xlabel('% of population driving')
plt.ylabel('Gas consumption (millions gallons)')

x = np.linspace(.4, .8).reshape(-1, 1)
plt.plot(x, linreg.predict(x), 'k-')
plt.plot(x, huber.predict(x), 'm-')
plt.legend(['Simple linear regression (MSE)', 'Huber regression']);

#%%
#=========================
# MULTIVARIATE REGRESSION
#=========================

from ipywidgets import widgets

feature_desc = {'tax': 'Gas tax', 'drivers': '% of population driving', 'income': 'Average income (USD)', 'highway': 'Miles of paved highway'}
def plot_feature(column):
    plt.plot(gas[column], gas['gas'], '.')
    plt.xlabel(feature_desc[column])
    plt.ylabel('Gas consumption (millions gallons)')

dropdown_menu = {v: k for k, v in feature_desc.items()}

widgets.interact(plot_feature, column=dropdown_menu);

from mpl_toolkits.mplot3d import Axes3D

plt3d = plt.figure().gca(projection='3d')
plt3d.scatter(gas['tax'], gas['drivers'], gas['gas']);

print(linreg.fit(gas[['tax', 'drivers']], gas['gas']))

plt3d = plt.figure().gca(projection='3d')

xx, yy = np.meshgrid(np.linspace(5, 11), np.linspace(.4, .8))
z = linreg.intercept_ + linreg.coef_[0] * xx + linreg.coef_[1] * yy
plt3d.plot_surface(xx, yy, z, alpha=0.2)
plt3d.scatter(gas['tax'], gas['drivers'], gas['gas']);

from ipywidgets import interact

def plot_cross(tax=7):
    x = np.linspace(.4, .8)
    plt.plot(x, linreg.intercept_ + linreg.coef_[0]*tax + linreg.coef_[1]*x)
    alpha = 1 - abs(gas['tax'] - tax) / abs(gas['tax'] - tax).max()
    colors = np.zeros((len(gas), 4))
    colors[:, 3] = alpha
    plt.scatter(gas['drivers'], gas['gas'], color=colors)

# Check the best fit using different tax values
for t in range(5,12):
    interact(plot_cross, tax=(t,11,1));

