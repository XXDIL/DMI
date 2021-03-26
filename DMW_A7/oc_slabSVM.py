#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 17:37:36 2021

@author: manavagrawal
"""

import pandas as pd
import numpy as np
import scipy as spy
import sklearn as skl
import seaborn as sns
import matplotlib.pyplot as plt
import math,random
import random as rnd

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import chi2_kernel as chi2
from cvxopt import solvers, lapack, matrix, spmatrix
from scipy.io import loadmat

class OCC:
    
    rand = 'ann'
    def __init__(self, X, step = 0.05, eps = 0):
        import matplotlib.pyplot as plt
        import numpy as np
        
        self.X = X
        self.n = X.shape[0]
        self.figsize = (10,5)
        
        # plot parameters
        self.step = step
        self.eps = eps
        self.norm_colors = mpl.colors.Normalize(vmin=0,vmax=100)
        self.set_grid()
        
        
        pass
    
    def plot(self):
        plt.scatter(self.X[:,0], self.X[:,1], edgecolors  = 'black')
        plt.xlim((min(self.X[:,0])-self.step,max(self.X[:,0])+self.step))
        plt.ylim((min(self.X[:,1])-self.step,max(self.X[:,1])+self.step))
        pass
    
    def set_grid(self):
        self.x_axis = np.arange(min(self.X[:,0])-self.step,max(self.X[:,0])+2*self.step,self.step)
        self.y_axis = np.arange(min(self.X[:,1])-self.step,max(self.X[:,1])+2*self.step,self.step)

        self.my_grid = []
        for i in self.x_axis:
            for j in self.y_axis:
                self.my_grid.append([i,j])
        self.my_grid = np.array(self.my_grid)

        pass

    def Normalize_Pred(self):    
        self.pred_100 = self.pred -np.min(self.pred)
        self.pred_100 = (self.pred_100 / np.max(self.pred_100)) * 100
        pass

    def set_delim(self):
        ix_delim = np.where((self.pred  < self.eps) & (self.pred  > -self.eps))[0]
        self.x_delim = []
        self.y_delim = []
        for i in range(len(ix_delim)):
            a, b = np.divmod(ix_delim[i],len(self.y_axis))
            self.x_delim.append(self.x_axis[a])
            self.y_delim.append(self.y_axis[b])
        pass

    def plot_pred(self, pred):
        
        X_axis, Y_axis = np.meshgrid(self.x_axis, self.y_axis)
        C = np.transpose(pred.reshape((len(self.x_axis), len(self.y_axis))))
        plt.pcolor(X_axis, Y_axis, C, norm = self.norm_colors, cmap = 'YlOrRd')
        plt.scatter(self.x_delim, self.y_delim, c = 'black', s = 10)
        self.plot()
        pass

    def plot_pred_plan(self):
        self.pred = self.predict(self.my_grid)
        self.pred_bin = np.sign(self.pred)
        self.set_delim()
        self.Normalize_Pred()
        
        plt.figure(figsize=self.figsize)
        plt.subplot(1,2,1)
        self.plot_pred(self.pred_100)
        plt.subplot(1,2,2)
        self.plot_pred((self.pred_bin+1)*100)
        pass
    
    def fit(self):
        pass
    
    def predict(self):
        pass
    
    
    def slabFitFunction(X_train,y_train,ker):
        if ker == 'hellinger':
            ker = 'poly'
        clf = svm.OneClassSVM(nu = 0.01, kernel=ker, gamma=0.0001)
        clf.fit(X_train,y_train)
        OCC.rand = ker
        return clf
    
    def slabPredictFunction(clf,X_test):
        y_pred = clf.predict(X_test)
        return y_pred
    
    def getAcc(y_pred,y_test):
        s = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                s += 1
        if OCC.rand == 'poly' :
            return (s/len(y_test))*100
        return (1 - (s/len(y_test)))*100
        
    
def RBF_Kernel(X, sigma2, Y = None):
    " Compute the RBF kernel matrix of X"
    from sklearn.metrics.pairwise import euclidean_distances
    
    if type(Y)==type(None):
        Y = X
    
    K = euclidean_distances(X,Y, squared=True)
    K *= -1./sigma2
    K = np.exp(K)
    return K

def Poly_Kernel(X, d, c, Y=None):
    if type(Y)==type(None):
        Y = X.copy()
    K = ( np.dot(X,Y.T) +c)**d
    return(K)

def fit(X, nu, kernel):  
    n = X.shape[0]
    if kernel[0] == 'RBF':
        sigma2 = kernel[1]
        K = RBF_Kernel(X, sigma2)
    else:
        d, c = kernel[1:]
        K = Poly_Kernel(X,d,c)

    P = matrix(K.astype(np.double), tc = 'd')
    q = matrix([0]*n, tc = 'd')
    G = matrix(np.concatenate([np.eye(n),-np.eye(n)], axis = 0), tc = 'd')
    h_value = [1./(n * nu)]*n
    h_value.extend([0]*n)
    h = matrix(h_value, tc = 'd')
    A = matrix(np.reshape([1]*n, (1,n)), tc = 'd')
    b = matrix(1, tc = 'd')
    
    sol = solvers.qp(P,q,G,h,A,b)
    alpha = np.array(sol['x'])
    ix_in = np.where((alpha > 1e-5) & (alpha < 1./(n*nu)))[0][0]
    rho = np.transpose(np.dot(np.reshape(alpha, (1,n)), K[:,ix_in]))
    
    return alpha, rho

def predict(X, newData, alpha, rho, kernel):
    n = X.shape[0]
    if kernel[0] == 'RBF':
        sigma2 = kernel[1]
        K = RBF_Kernel(X, sigma2, newData)
    else:
        d, c = kernel[1:]
        K = Poly_Kernel(X,d,c,Y=newData)
    return np.transpose(np.dot(np.reshape(alpha, (1,n)), K) - rho)

def get_grid(X, step = 0.05):
    x_axis = np.arange(min(X[:,0])-step,max(X[:,0])+2*step,step)
    y_axis = np.arange(min(X[:,1])-step,max(X[:,1])+2*step,step)

    my_grid = []
    for i in x_axis:
        for j in y_axis:
            my_grid.append([i,j])#,1])
    my_grid = np.array(my_grid)
    
    return x_axis, y_axis, my_grid

def plot(X,step):
    plt.scatter(X[:,0], X[:,1], edgecolors  = 'black')
    plt.xlim((min(X[:,0])-step,max(X[:,0])+step))
    plt.ylim((min(X[:,1])-step,max(X[:,1])+step))
    pass

def Normalize_Pred(pred):    
    pred_100 = pred -np.min(pred)
    pred_100 = (pred_100 / np.max(pred_100)) * 100
    return pred_100

def get_delim(pred_100, x_axis, y_axis, eps = 0):
    ix_delim = np.where((pred_100  < eps) & (pred_100  > -eps))[0]
    x_delim = []
    y_delim = []
    for i in range(len(ix_delim)):
        a, b = np.divmod(ix_delim[i],len(y_axis))
        x_delim.append(x_axis[a])
        y_delim.append(y_axis[b])
    return x_delim, y_delim

def plot_pred(X, x_axis, y_axis, pred_100, x_delim, y_delim, step = 0.05):
    norm_colors = mpl.colors.Normalize(vmin=0,vmax=100)
    X_axis, Y_axis = np.meshgrid(x_axis, y_axis)
    C = np.transpose(pred_100.reshape((len(x_axis), len(y_axis))))
    plt.pcolor(X_axis, Y_axis, C, norm= norm_colors, cmap = 'YlOrRd')
    plt.scatter(x_delim, y_delim, c = 'blue', s = 10)
    plot(X,step)
    plt.xlim((min(X[:,0])-step,max(X[:,0])+step))
    plt.ylim((min(X[:,1])-step,max(X[:,1])+step))
    pass

def plot_pred_plan(X, alpha, rho, kernel, step = 0.05, eps = 0):
    x_axis, y_axis, my_grid = get_grid(X, step)
    pred = predict(X, my_grid, alpha, rho, kernel)
    x_delim, y_delim = get_delim(pred, x_axis, y_axis, eps)
    pred_100 = Normalize_Pred(pred)
    plot_pred(X, x_axis, y_axis, pred_100, x_delim, y_delim, step)
    return pred