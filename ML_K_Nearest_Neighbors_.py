# -*- coding: utf-8 -*-
#%%
#=======================
# K NEAREAST NEIGHBORS #
#=======================

import seaborn as sns
sns.set()

import matplotlib.pyplot as plt
import numpy as np

#%%
#=======================
# FINDING THE NEIGHBORS
#=======================
from ipywidgets import interact, IntSlider
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

data = load_iris()
X = data['data']
y = data['target']

Xt = StandardScaler().fit_transform(X)

def plot_neighbors(n_neighbors=1):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(Xt[:, 2:], y)
    
    X1, X2 = np.meshgrid(np.linspace(-2, 2), np.linspace(-2, 2))
    y_pred = knn.predict(np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1))))
    plt.contourf(X1, X2, y_pred.reshape(50, 50)/2, cmap='viridis', alpha=0.25, vmin=0, vmax=0.95)

    plt.scatter(*Xt[:, 2:].T, c=y, cmap='viridis');

interact(plot_neighbors, n_neighbors=IntSlider(min=1, max=36, step=2, description='neighbors'));

#%%
#==================
# VARIANCE VS BIAS
#==================

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
knn = KNeighborsClassifier(n_neighbors=2)

pipe = Pipeline([('scaler', StandardScaler()), ('clf', knn)])
pipe.fit(X_train, y_train)

param_grid = {'clf__n_neighbors': range(1, 36, 1)}
grid_search = GridSearchCV(pipe, param_grid, cv=20, n_jobs=2)
grid_search.fit(X, y)

plt.plot(grid_search.cv_results_['mean_test_score'])
plt.xlabel('Neighbors')
plt.ylabel('CV Accuray');

#%%




























