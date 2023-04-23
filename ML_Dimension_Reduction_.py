# -*- coding: utf-8 -*-
#%%
#==========================================
# DIMENSION REDUCTION IN MACHINE LEARNING #
#==========================================

#%%

import seaborn as sns
sns.set()

import matplotlib.pyplot as plt
import numpy as np

#%%
#====================================
# PRINCIPAL COMPONENT ANALYSIS (PCA)
#====================================

# Using 2 dimension x1, x2
np.random.seed(0)
x1 = np.linspace(0, 1, 500)
x2 = 2*x1 + 1 + 0.2*np.random.randn(500)
X = np.vstack((x1, x2)).T

plt.scatter(*X.T, alpha=0.25)
plt.plot(x1, 2*x1 + 1, '--k', linewidth=2)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$');

#%%
# Using principal components instead of x1, x2

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
Xt = pca.fit_transform(X)

xi_1_max, xi_2_max = Xt.max(axis=0)
xi_1_min, xi_2_min = Xt.min(axis=0)

plt.hlines(0, xi_1_min, xi_1_max, linestyles='--')
plt.vlines(0, xi_2_min, xi_2_max, linestyles='--')

plt.scatter(*Xt.T, alpha=0.25)
plt.xlim([-1.75, 1.75])
plt.ylim([-1.75, 1.75])
plt.xlabel('$\\xi _1$')
plt.ylabel('$\\xi _2$');

#%%
# Represent the data points in the space of either one (reduced) or two 
# principal components or project back onto the original space after reducing 
# the dimension.

from ipywidgets import interact, fixed

np.random.seed(0)
ind = np.random.choice(Xt.shape[0], 50)

def reduce_dim(X, Xt, step='one PC'):
    if step == 'original space':   
        pca = PCA(n_components=1)
        X_t = pca.fit_transform(X)
        plt.scatter(*pca.inverse_transform(X_t[ind, :]).T)
        plt.scatter(*X[ind, :].T, c='b', alpha=0.1)

        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$');
        
        return 
    
    elif step == 'two PC':
        plt.scatter(*Xt[ind, :].T)

        for x in Xt[ind, :]:
            plt.vlines(x[0], 0, x[1], linestyles='--')    
    else:
        plt.scatter(Xt[ind, 0], np.zeros(50))
        plt.scatter(*Xt[ind, :].T, alpha=0.1, c='b')

    plt.xlim([-1.75, 1.75])
    plt.ylim([-0.5, 0.5])
    plt.xlabel('$\\xi _1$')
    plt.ylabel('$\\xi _2$')
       
# interact(reduce_dim, X=fixed(X), Xt=fixed(Xt), step=['two PC', 'one PC', 'original space']);

steps = ['two PC', 'one PC', 'original space']
for s in steps:
    interact(reduce_dim, X=fixed(X), Xt=fixed(Xt), step=[s]);

#%%
#=====================
# PCA IN SCIKIT-LEARN
#=====================
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
X = data['data']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=4)
Xt = pca.fit_transform(X_scaled)

print("number of dimension before reduction: {}".format(X_scaled.shape[-1]))
print("number of dimension after reduction: {}".format(Xt.shape[-1]))

#%%
# How much original info have we retained?

print("explained variance ratio: {}".format(pca.explained_variance_ratio_))
print("cumulative explained variance ratio: {}".format(np.cumsum(pca.explained_variance_ratio_)[-1]))

#%%
# Calculate the total explained variance by using the inverse_transform method

print("retained variance: {}".format(pca.inverse_transform(Xt).var()))

#%%
#===================================
# CHOOSING THE NUMBER OF COMPONENTS
#===================================

X_scaled = scaler.fit_transform(X)
p = X_scaled.shape[-1]
pca = PCA(n_components=p)
pca.fit(X_scaled)
cumulative_explained_var = np.cumsum(pca.explained_variance_ratio_)

plt.plot(range(1, p + 1), cumulative_explained_var)
plt.hlines(0.9, 1, p+1, linestyles='--')
plt.hlines(0.99, 1, p+1, linestyles='--')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

#%%
#===================================
# NON-NEGATIVE MATRIX FACTORIZATION
#===================================

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

data = fetch_20newsgroups(shuffle=True, remove=('headers', 'footers', 'quotes'))
X = data['data']

n_topics = 10
n_top_words = 20

tfidf = TfidfVectorizer(stop_words='english')
nmf = NMF(n_components=n_topics, random_state=0)
pipe = Pipeline([('vectorizer', tfidf), ('dim-red', nmf)])
pipe.fit(X)

feature_names = tfidf.get_feature_names()

for i, topic in enumerate(nmf.components_):
    print("Topic: {}".format(i))
    indices = topic.argsort()[-n_top_words-1:-1]
    top_words = [feature_names[ind] for ind in indices]
    print(" ".join(top_words), "\n")

#%%
#===================================
# USING PCA WITH A SUPERVISED MODEL
#===================================

from shutil import rmtree
from tempfile import mkdtemp
import time

from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=10000, n_features=100, n_informative=10, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
pca = PCA(n_components=10)
tree = DecisionTreeClassifier()

cache = mkdtemp()
pipe = Pipeline([('scaler', scaler), ('dim-red', pca), ('clf', tree)], memory=cache)
param_grid = {'clf__max_depth': range(2, 20)}
grid_search = GridSearchCV(pipe, param_grid, cv=3, n_jobs=2)

t_0 = time.time()
grid_search.fit(X_train, y_train)
t_elapsed = time.time() - t_0

print("training time: {:g} seconds".format(t_elapsed))
print("test accuracy: {}".format(grid_search.score(X_test, y_test)))
rmtree(cache)

#%%
pipe = Pipeline([('scaler', scaler), ('clf', tree)], memory=cache)
param_grid = {'clf__max_depth': range(2, 20)}
grid_search = GridSearchCV(pipe, param_grid, cv=3, n_jobs=2)

t_0 = time.time()
grid_search.fit(X_train, y_train)
t_elapsed = time.time() - t_0

print("training time {:g} seconds".format(t_elapsed))
print("test accuracy {}".format(grid_search.score(X_test, y_test)))
rmtree(cache)

#%%
#======================================
# DIMENSION REDUCTION IN VISUALIZATION
#======================================

from sklearn.datasets import load_iris

data = load_iris()
X = data['data']
y = data['target']

pca = PCA(n_components=2)
pipe = Pipeline([('scaler', StandardScaler()), ('dim-red', pca)])
Xt = pipe.fit_transform(X)

plt.scatter(*Xt.T, c=y, cmap='viridis')
plt.xlabel('$\\xi_1$')
plt.ylabel('$\\xi_2$');

#%%

explained_var = np.cumsum(pca.explained_variance_ratio_)
print('explained variance with two dimensions: {}'.format(explained_var[-1]))

#%%
