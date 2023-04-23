# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 05:59:21 2023

@author: FANKA
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# load data set
data = load_wine()
X = data['data']

# truncate to two variables
pipe = Pipeline([('scaler', StandardScaler()), ('dim_red', PCA(n_components=2))])
Xt = pipe.fit_transform(X)

# generate novel/outlier points
np.random.seed(1)
theta = 2*np.pi*np.random.random(10)
X_test = np.vstack((4*np.cos(theta) + np.random.random(10), 4*np.sin(theta) + np.random.random(10)))

plt.scatter(*Xt.T)
plt.scatter(*X_test, c='red')
plt.xlabel('$\\xi_1$')
plt.ylabel('$\\xi_2$');
plt.legend(["training set", "novel/outliers"]);

#%%
from sklearn.svm import OneClassSVM
from ipywidgets import interact, FloatSlider

def plot_one_class_svm(X, X_test):
    def plotter(nu=0.95):
        clf = OneClassSVM(nu=nu, gamma='auto')
        clf.fit(X)
        y_pred = clf.predict(X)
        fp_rate = (y_pred == -1).sum()/len(X)
        
        X1, X2 = np.meshgrid(np.linspace(-5, 5), np.linspace(-5, 5))
        y_proba = clf.decision_function(np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1))))
        Z = y_proba.reshape(50, 50)
        
        fig = plt.figure(figsize=(8, 5), facecolor='w', edgecolor='k')
        plt.contourf(X1, X2, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
        plt.colorbar()
        a = plt.contour(X1, X2, Z, levels=[0], linewidths=2, colors='black')            
        b1 = plt.scatter(*X.T, c='blue')
        b2 = plt.scatter(*X_test, c='red')
        plt.title("false positive rate: {:g}".format(fp_rate))
        plt.legend([a.collections[0], b1, b2], ["boundary", " true inliers", "true outliers"], frameon=True, 
                   loc="lower left")
    return plotter

nu_slider = FloatSlider(min=0.01, max=0.99, step=0.01, value=0.5, description='$\\nu$')
interact(plot_one_class_svm(Xt, X_test), nu=nu_slider);

#%%
from sklearn.ensemble import IsolationForest

def plot_isolation_forest(X, X_test):
    def plotter(contamination=0.2):
        clf = IsolationForest(n_estimators=100, contamination=contamination)
        clf.fit(X)
    
        y_pred = clf.predict(X)
        outlier_rate = (y_pred == -1).sum()/len(X)
        
        X1, X2 = np.meshgrid(np.linspace(-5, 5), np.linspace(-5, 5))
        y_proba = clf.decision_function(np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1))))
        Z = y_proba.reshape(50, 50)
        
        fig = plt.figure(figsize=(8, 5), facecolor='w', edgecolor='k')
        plt.contourf(X1, X2, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
        plt.colorbar()
        a = plt.contour(X1, X2, Z, levels=[0], linewidths=2, colors='black')            
        b1 = plt.scatter(*X.T, c='blue')
        b2 = plt.scatter(*X_test, c='red')
        plt.title("outlier fraction: {:g}".format(outlier_rate))
        plt.legend([a.collections[0], b1, b2], ["boundary", " true inliers", "true outliers"], frameon=True, 
                   loc="lower left")    
    return plotter

cont_slider = FloatSlider(min=0.01, max=0.5, value=0.1, step=0.01, description="fraction")
interact(plot_isolation_forest(Xt, X_test), contamination=cont_slider);

#%%
df = pd.read_csv("data/energy_data.csv", parse_dates=["date"])              
df = df.set_index('date')
df_hourly = df.resample("H").mean() # resample hourly

energy = df_hourly['Appliances']
energy.plot()
plt.ylabel("energy (Wh)");

#%%
#===================
# FOURIER ANALYSIS
#===================
from scipy import fftpack

sampling_rate = (energy.index[1] - energy.index[0]).total_seconds()
sampling_rate = sampling_rate / (60 * 60 * 24) # day

Y = fftpack.fft(energy.values - energy.mean())
freq = np.linspace(0, 1/sampling_rate, len(Y))

plt.plot(freq[:len(freq)//2], np.abs(Y[:len(Y)//2]))
plt.xlabel("cycles per day")
plt.ylabel("Fourier transform");

#%%
#=======
# INCORPORATING DAY OF THE WEEK
#=========
df_day_of_week = pd.DataFrame({'day': energy.index.dayofweek, 'count': energy.values})
grouped_by_day = df_day_of_week.groupby('day')

grouped_by_day.mean().plot(kind='bar')
plt.xticks(range(7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']);

#%%
#=======================
# INITIAL BASELINE MODEL
#=======================
from sklearn.base import BaseEstimator, TransformerMixin

class IndexSelector(BaseEstimator, TransformerMixin):

    def __init__(self):
        """Return indices of a data frame for use in other estimators."""
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        return df.index

class FourierComponents(BaseEstimator, TransformerMixin):

    def __init__(self, freqs):
        """Create features based on sin(2*pi*f*t) and cos(2*pi*f*t)."""
        self.freqs = freqs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xt = np.zeros((X.shape[0], 2 * len(self.freqs)))
        t_0 = X[0]
        for i, f in enumerate(self.freqs):
            Xt[:, 2 * i] = np.cos(2 * np.pi * f * (X)).reshape(-1)
            Xt[:, 2 * i + 1] = np.sin(2 * np.pi * f * (X)).reshape(-1)

        return Xt

class EpochTime(BaseEstimator, TransformerMixin):

    def __init__(self, unit):
        """Transform datetime object to some unit of time since the start of the epoch."""
        self.unit = unit

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        epoch_time = np.array([x.value for x in X])

        if self.unit == "seconds":
            return epoch_time / (1000000000)
        elif self.unit == "minutes":
            return epoch_time / (1000000000) / 60
        elif self.unit == "hours":
            return epoch_time / (1000000000) / 60 / 60
        else:
            return epoch_time
        
class DayOfWeek(BaseEstimator, TransformerMixin):

    def __init__(self):
        """Determine the day of the week for datetime objects."""
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([x.dayofweek for x in X]).reshape(-1, 1)
    
# ADDITIONAL USEFUL FUNCTIONS
def ts_train_test_split(df, cutoff, target):
    """Perform a train/test split on a data frame based on a cutoff date."""
    
    ind = df.index < cutoff
    
    df_train = df.loc[ind]
    df_test = df.loc[~ind]
    y_train = df.loc[ind, target]
    y_test = df.loc[~ind, target]
    
    return df_train, df_test, y_train, y_test

def plot_results(df, y_pred):
    """Plot predicted results and residuals."""
    
    plt.plot(df.index, y_pred, '-r')
    energy.plot()
    plt.ylabel('energy (Wh)')
    plt.legend(['true', 'predicted'])
    plt.show();

    plt.plot(resd)
    plt.ylabel('residual');

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder

# perform train/test split
cutoff = "Mar-2016" # roughly corresponding to 10% of the data
df_train, df_test, y_train, y_test = ts_train_test_split(df_hourly, cutoff, 'Appliances')

# construct and train model
freqs = np.array([1, 2, 3]) / 24 / 60 # 24, 12, and 8 hour periods
selector = IndexSelector()
epoch_time = EpochTime("minutes")
fourier_components = FourierComponents(freqs)
one_hot = OneHotEncoder(sparse=False, categories='auto')
lr = LinearRegression()

fourier = Pipeline([("time", epoch_time),
                    ("sine_cosine", fourier_components)])
day_of_week = Pipeline([("day", DayOfWeek()),
                        ("encoder", one_hot)])
union = FeatureUnion([("fourier", fourier),
                      ("day_of_week", day_of_week)])

pipe = Pipeline([("indices", selector),
                 ("union", union),
                 ("regressor", lr)])
pipe.fit(df_train, y_train)

# make predictions
y_pred = pipe.predict(df_hourly)
resd = energy - y_pred
print("Test set R^2: {:g}".format(pipe.score(df_test, y_test)))
plot_results(df_hourly, y_pred)

#%%
#======================
# NOISE-BASED FEATURES
#======================
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(resd)
plt.xlabel('Lag (hour)')
plt.xlim([0, 50]);

from sklearn.base import RegressorMixin

class ResidualFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, window=100):
        """Generate features based on window statistics of past noise/residuals."""
        self.window = window
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = pd.DataFrame()
        df['residual'] = X
        df['prior'] = df['residual'].shift(1)
        df['mean'] = df['residual'].rolling(window=self.window).mean()
        df['diff'] = df['residual'].diff().rolling(window=self.window).mean()
        df = df.fillna(method='bfill')
        
        return df
    
class FullModel(BaseEstimator, RegressorMixin):
    def __init__(self, baseline, residual_model, steps=20):
        """Combine a baseline and residual model to predict any number of steps in the future."""
        
        self.baseline = baseline
        self.residual_model = residual_model
        self.steps = steps
        
    def fit(self, X, y):
        self.baseline.fit(X, y)
        resd = y - self.baseline.predict(X)
        self.residual_model.fit(resd.iloc[:-self.steps], resd.shift(-self.steps).dropna())
                
        return self
    
    def predict(self, X):
        y_b = pd.Series(self.baseline.predict(X), index=X.index)
        resd = X['Appliances'] - y_b
        resd_pred = pd.Series(self.residual_model.predict(resd), index=X.index)
        resd_pred = resd_pred.shift(self.steps)
        y_pred = y_b + resd_pred
        
        return y_pred

from sklearn.metrics import r2_score

# construct residual model
resd_train = y_train - pipe.predict(df_train)
residual_feats = ResidualFeatures(window=20)
residual_model = Pipeline([('residual_features', residual_feats), ('regressor', LinearRegression())])
    
# construct and train full model
full_model = FullModel(pipe, residual_model, steps=1)
full_model.fit(df_train, y_train)

# make predictions
y_pred = full_model.predict(df_hourly)
resd = energy - y_pred
ind = resd.index > cutoff
print("Test set R^2: {:g}".format(r2_score(energy.loc[ind], y_pred.loc[ind])))
plot_results(df_hourly, y_pred)

#%%
# FINAL RESIDUALS
resd.hist(bins=50, density=True);
plt.show()

autocorrelation_plot(resd.dropna())
plt.xlabel("Lag (hours)")
plt.xlim([0, 100]);

#%%
#========
# Z-SCORE
#========
z = (resd - resd.mean())/ resd.std()
z.plot()
pd.Series(3, index=resd.index).plot(color="r")
pd.Series(-2, index=resd.index).plot(color="r")
plt.ylabel("z-score")
plt.legend(["residual", "z-score cutoff"]);

#%%
#===============
# FIND ANOMALIES
#===============
def find_anomalies(z, cutoff_lower=-2, cutoff_upper=2):
    ind_lower = z < cutoff_lower
    ind_upper = z > cutoff_upper
    
    return z[ind_lower | ind_upper]

find_anomalies(z, cutoff_lower=-2, cutoff_upper=3)

#%%
#=================
# ROLLING Z-SCORE
#=================
def rolling_z_score(x, window=20):
    rolling = x.rolling(window=window)
    mean_roll = rolling.mean().shift(1) # shift to not include current value
    std_roll = rolling.std().shift(1)
    
    return (x - mean_roll) / std_roll

z_roll = rolling_z_score(resd, window=20)
z_roll.plot()
pd.Series(3, index=resd.index).plot(color="r")
pd.Series(-2, index=resd.index).plot(color="r")
plt.ylabel("z-score")
plt.legend(["residual", "z-score cutoff"]);

find_anomalies(z_roll, cutoff_lower=-2, cutoff_upper=3)

#%%










