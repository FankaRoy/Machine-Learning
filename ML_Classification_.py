# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 09:32:10 2023

@author: FANKA
"""
import seaborn as sns
sns.set()

import matplotlib.pyplot as plt
import numpy as np

#%%
from sklearn import metrics

# generate our results
y_pred = np.zeros(100, dtype=np.int32)
y_pred[:12] = 1
y = np.zeros(100)
y[:8] = 1
y[-2:] = 1

print("precision: {:g}".format(metrics.precision_score(y, y_pred)))
print("recall: {:g}".format(metrics.recall_score(y, y_pred)))
print(metrics.classification_report(y, y_pred))

#%%
#====================================
# PROBABILISTIC CLASSIFICATION MODELS
#====================================

# generate data
np.random.seed(0)
y_proba = np.linspace(0, 1, 1000)
y_pred = (y_proba > 0.5).astype(np.int32)
y = np.random.binomial(1, y_proba)

print("accuracy: {}".format(metrics.accuracy_score(y, y_pred)))

precision, recall, threshold = metrics.precision_recall_curve(y, y_proba)
f1_score = 2*precision*recall/(precision + recall)
threshold = np.hstack((0, threshold))

plt.plot(threshold, precision)
plt.plot(threshold, recall)
plt.plot(threshold, f1_score)
plt.xlabel('threshold')
plt.legend(['precision', 'recall', '$F_1$']);

#%%
#============================
# AREA UNDER THE CURVE (AUC)
#============================
plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.xlim([0, 1])
plt.ylim([0, 1]);

print("precision-recall AUC: {}".format(metrics.auc(recall, precision)))
print("receiver-operator AUC: {}".format(metrics.roc_auc_score(y, y_proba)))

#%%
#==========================
# LOG LOSS (CROSS ENTROPY)
#==========================

p = np.linspace(1E-6, 1-1E-6, 1000)
y = 1
log_loss = -(y*np.log(p) + (1 - y)*np.log(1 - p))

plt.plot(p, log_loss)
plt.xlabel('probability')
plt.ylabel('log loss')
plt.legend(['$y$ = 1']);

#%%
#=====================
# LOGISTIC REGRESSION
#=====================

x = np.linspace(-10, 10, 100)
s = 1/(1 + np.exp(-x))

plt.plot(x, s)
plt.xlabel('$x$')
plt.ylabel('$S(x)$');

#%
# train a model and plot it's decision boundary

from sklearn.datasets import make_blobs
X, y = make_blobs(centers=[[1, 1], [-1, -1]], cluster_std=1.5, random_state=0)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$');

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='lbfgs')
clf.fit(X, y)
y_pred = clf.predict(X)

print("accuracy: {}".format(metrics.accuracy_score(y, y_pred)))

X1, X2 = np.meshgrid(np.linspace(-5, 5), np.linspace(-5, 5))
y_proba = clf.predict_proba(np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1))))[:, 1]
plt.contourf(X1, X2, y_proba.reshape(50, 50), cmap=plt.cm.bwr, alpha=0.75, vmin=0, vmax=0.95)
plt.colorbar()

plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='white', cmap=plt.cm.bwr)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$');

#%%
#=================================================
# MULTICLASS CLASSIFICATION FOR BINARY CLASSIFIER
#=================================================


