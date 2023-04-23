# -*- coding: utf-8 -*-

#%%
#==========================
# SUPPORT VECTOR MACHINES #
#==========================

import seaborn as sns
sns.set()

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

#%%
#========================
# HARD MARGIN CLASSIFIER
#========================

from sklearn.datasets import make_blobs

X, y = make_blobs(centers=[[1, 1], [-1, -1]], cluster_std=0.4, random_state=0)
x = np.linspace(-2, 2, 100)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
plt.plot(x, -x+0.25, '--k')
plt.plot(x, -0.25*x-0.3, 'r--')
plt.plot(x, -1.5*x+1, 'b--')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$');

#%%
#================================
# DETERMINING THE MAXIMUM MARGIN
#================================

from ipywidgets import interact, IntSlider, FloatSlider, fixed
from sklearn import svm

def plot_svc_interact(X, y):
    def plotter(log_C=1):
        clf = svm.SVC(C=10**log_C, kernel='linear')
        clf.fit(X, y)
    
        beta = clf.coef_[0]
        beta_0 = clf.intercept_
        slope = -beta[0]/beta[1]
        intercept = -beta_0/beta[1]
       
        x_max = np.ceil(np.abs(X).max())
        x = np.linspace(-x_max, x_max, 100)
        margin_bound_1 = 1/beta[1] + slope*x + intercept
        margin_bound_2 = -1/beta[1] + slope*x + intercept

        plt.plot(x, slope*x + intercept, 'k')
        plt.fill_between(x, margin_bound_1, margin_bound_2, color='k', alpha=0.25, linewidth=0)
        plt.scatter(*clf.support_vectors_.T, s=100, c='y')
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
        plt.axis([-x_max, x_max, -x_max, x_max])

    return plotter

plot_svc_interact(X, y)(log_C=2)

#%%
#========================
# SOFT MARGIN CLASSIFIER
#========================


x = np.linspace(-1, 2, 100)
hinge_loss = lambda x: -(x-1) if x-1 < 0 else 0

plt.plot(x, list(map(hinge_loss, x)))
plt.xlabel("$y(x\cdot\\beta + \\beta_0$)")
plt.ylabel('loss');

#%%
# Training the soft margin classifier on a data set that is not completely linear separable

from sklearn.datasets import make_blobs

X, y = make_blobs(centers=[[1, 1], [-1, -1]], cluster_std=1.5, random_state=0, n_samples=200)

log_C_slider = FloatSlider(min=-4, max=2, step=0.25, value=0, description='$\log(C)$')
interact(plot_svc_interact(X, y), log_C=log_C_slider);

#%%
#======================================
# KERNELS FOR NONLINEAR CLASSIFICATION
#======================================

from sklearn.datasets import make_circles

X, y = make_circles(n_samples=200, noise=0.2, factor=0.25, random_state=0)
plt.scatter(*X.T, c=y, cmap=plt.cm.bwr)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$');

#%%
# we can create a new feature, 𝑥3, the distance from the origin.
# With the new feature, we are projecting our data onto a higher dimensional space.
# 
from mpl_toolkits.mplot3d import Axes3D

def plot_projection(X, y):
    XX, YY = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
    ZZ = 0.6*np.ones((20, 20))
    x_3 = (X[:, 0]**2 + X[:, 1]**2)**0.5
    X_new = np.hstack((X, x_3.reshape(-1, 1)))

    def plotter(elev=30, azim=30):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(*X_new.T, c=y, cmap=plt.cm.bwr)
        ax.plot_surface(XX, YY, ZZ, alpha=0.2);
        ax.view_init(elev, azim)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$x_3$')

    return plotter

interact(plot_projection(X, y), elev=(0, 360), azim=(0, 360));

#%%
#====================
# CHOICES OF KERNELS
#====================

def plot_decision_boundary(X_train, X_test, y_train, y_test):
    def plotter(kernel='linear', log_gamma=1, log_C=1, deg=1, coef0=1):
        clf = svm.SVC(C=10**log_C, kernel=kernel, gamma=10**log_gamma, coef0=coef0, probability=True)
        clf.fit(X_train, y_train)
        
        X1, X2 = np.meshgrid(np.linspace(-2, 3), np.linspace(-2, 2))
        y_proba = clf.predict_proba(np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1))))[:, 1]
        plt.contourf(X1, X2, y_proba.reshape(50, 50), 16, cmap=plt.cm.bwr, alpha=0.75)
        plt.colorbar()

        accuracy = clf.score(X_test, y_test)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='white', cmap=plt.cm.bwr)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('test set accuracy: {}'.format(accuracy));

    return plotter


from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(400, noise=0.25, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

log_C_slider = FloatSlider(min=-4, max=4, step=0.25, value=0, description='$\log(C)$')
log_gamma_slider = FloatSlider(min=-3, max=2, step=0.01, value=0, description='$\log(\gamma$)')
deg_slider = IntSlider(min=1, max=4, step=1, value=2, description='$d$')
coef0_slider = FloatSlider(min=-100, max=100, step=0.1, value=0, description='$r$')

interact(plot_decision_boundary(X_train, X_test, y_train, y_test),
         log_C=log_C_slider,
         log_gamma=log_gamma_slider, 
         kernel=['rbf', 'linear', 'sigmoid', 'poly'],
         deg=deg_slider,
         coef0=coef0_slider);

#%%
#=====================================
# COMPARISON WITH LOGISTIC REGRESSION
#=====================================

x = np.linspace(-6, 4, 100)
hinge_loss = lambda x: -(x-1) if x < 1 else 0
log_loss = np.log(1+np.exp(-x))

plt.plot(x, list(map(hinge_loss, x)))
plt.plot(x, log_loss, '--r')
plt.xlabel("$y(x \cdot \\beta + \\beta_0)$")
plt.ylabel('loss');

#%%




from sklearn.linear_model import LogisticRegression

def plot_svc_vs_lr(cluster_std=0.8, log_C=1, model='logistic regression', outlier=False):
    X, y = make_blobs(centers=[[1, 1], [-1, -1]], cluster_std=cluster_std, random_state=0)

    if outlier:
        X = np.vstack((X, [-1.5, 0.]))
        y = np.hstack((y, [0]))

    name_to_clf = {'logistic regression': LogisticRegression(C=10**log_C, solver='lbfgs'),
              'SVM': svm.SVC(C=10**log_C, kernel='linear')}
    
    clf = name_to_clf[model]
    clf.fit(X, y)
    
    beta = clf.coef_[0]
    beta_0 = clf.intercept_
    slope = -beta[0]/beta[1]
    intercept = -beta_0/beta[1]
       
    x_max = np.ceil(np.abs(X).max())
    x = np.linspace(-x_max, x_max, 100)

    plt.plot(x, slope*x + intercept, 'k')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.axis([-x_max, x_max, -x_max, x_max])
    
#%%   
log_C_slider = FloatSlider(min=-4, max=4, step=0.25, value=1, description='$\log(C)$')
cluster_std_slider = FloatSlider(min=0.2, max=1.0, step=0.05, value=0.8, description='cluster $\sigma$')

interact(plot_svc_vs_lr,
         cluster_std=cluster_std_slider,
         log_C=log_C_slider,
         model=['logistic regression', 'SVM']);

#%%
#====================
# SVM FOR REGRESSION
#====================

eps = 0.25
x = np.linspace(-1, 1, 100)
well_loss = list(map(lambda x: abs(x)-eps if abs(x) > eps else 0, x))
square_loss = x**2

plt.plot(x, well_loss)
plt.plot(x, square_loss)
plt.xlabel('distance from the center')
plt.ylabel('loss')
plt.legend(['well loss', 'square loss']);

#%%

def plot_svr_interact(X, y):
    def plotter(epsilon=0.5, log_C=2):
        rgr = svm.SVR(kernel='linear', epsilon=epsilon, C=10**log_C)
        rgr.fit(X, y)
    
        y_pred = rgr.predict(X)
        ind = np.abs(y - y_pred) >= epsilon

        plt.scatter(X[ind], y[ind], s=100, color='y')
        plt.scatter(X, y)
        plt.fill_between(X.reshape(-1,), y_pred - epsilon,  y_pred + epsilon, alpha=0.25, color='k', linewidth=0)
        plt.plot(X, y_pred, '-k')
        plt.xlabel('$x$')
        plt.ylabel('$y$')

    return plotter

#%%

np.random.seed(0)
x = np.linspace(-1, 1, 100)
y = 2*x + 1 + 0.5*np.random.randn(100)

log_C_slider = FloatSlider(min=-3, max=1, step=0.05, value=-1, description='$\log(C)$')
epsilon_slider = FloatSlider(min=0.05, max=2, step=0.05, value=0.5, description='$\epsilon$')
interact(plot_svr_interact(x.reshape(-1, 1), y), log_C=log_C_slider, epsilon=epsilon_slider);














