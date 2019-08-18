# -*- coding: utf-8 -*-
"""
Gradient Boosting Machine implementation
@author: nguqu781
"""
#Import library
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import urllib.request as request
import csv
import d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from IPython.display import display
from sklearn import metrics
import seaborn as sns
from random import seed
from random import randrange
from scipy.stats.stats import pearsonr
import math

#Define decision tree
class DecisionTree():
	def __init__(self,x,y,index=None,min_leaf=2):
		if index is None:
			index =np.arange(len(y))
		self.index =index
		self.x=x
		self.y=y
		self.min_leaf = min_leaf
		self.val = np.mean(y[self.index])
		self.score = np.float('inf')
		self.obs = len(self.index)
		self.col_count = x.shape[1]
		self.find_varsplit()

	def find_varsplit(self):
		for c in range(self.col_count):	self.find_better_split(c)
		if self.score == np.float('inf'):return
		x = self.split_col
		left = np.nonzero(x<=self.split)[0]
		right = np.nonzero(x>self.split)[0]
		self.left = DecisionTree(self.x,self.y,self.index[left])
		self.right= DecisionTree(self.x,self.y,self.index[right])

	def find_better_split(self,var_index):
		x,y =self.x.values[self.index,var_index], self.y[self.index]
		sort_index = np.argsort(x)
		sort_y, sort_x = y[sort_index], x[sort_index]
		right_obs, right_sum, right_sum2 = self.obs, sort_y.sum(), (sort_y**2).sum()
		left_obs, left_sum, left_sum2 = 0, 0., 0.

		for i in range(0,self.obs-self.min_leaf-1):
			xi, yi = sort_x[i], sort_y[i]
			left_obs +=1; right_obs -=1
			left_sum += yi; right_sum -= yi
			left_sum2 +=yi**2; right_sum2 -=yi**2
			if i < self.min_leaf or xi==sort_x[i+1]:continue
			left_std = std_agg(left_obs, left_sum, left_sum2)
			right_std = std_agg(right_obs, right_sum, right_sum2)
			curr_score = left_std * left_obs + right_std * right_obs
			if curr_score < self.score:self.var_index, self.score, self.split = var_index, curr_score, xi

	@property
	def split_name(self):
		return self.x.columns[self.var_index]

	@property
	def split_col(self):
		return self.x.values[self.index,self.var_index]

	@property
	def is_leaf(self):
		return self.score == np.float('inf')

	def __repr__(self):
		s = f'n: {self.obs}; val: {self.val}'
		if not self.is_leaf:
			s+= f'; score: {self.score}; split: {self.split}; var: {self.split_name}'
		return s

	def predict(self,x):
		return np.array([self.predict_row(xi) for xi in x])

	def predict_row(self,xi):
		if self.is_leaf:
			return self.val
		t= self.left if xi[self.var_index] <= self.split else self.right
		return t.predict_row(xi)

def std_agg(obs,s1,s2):
	return math.sqrt((s2/obs) - (s1/obs)**2)


####TEst dataset
dataset = pd.DataFrame([[2.771244718,1.784783929,11],
	[1.728571309,1.169761413,10],
	[3.678319846,2.81281357,3],
	[3.961043357,2.61995032,-2],
	[2.999208922,2.209014212,10],
	[7.497545867,3.162953546,15],
	[9.00220326,3.339047188,-17],
	[7.444542326,0.476683375,-8],
	[10.12493903,3.234550982,9],
	[6.642287351,3.319983761,-10]])
dataset.columns = ["X1","X2","y"]
print(dataset)
x= dataset[["X1"]]
y= np.array(dataset["y"])
y = y[:,None]
print(y)
print(x)
#row the tree
plt.plot(x,y,'o')

tree =DecisionTree(x,y)
tree.find_better_split(0)
predict = tree.predict_row(np.array(x))

print(predict)

#Gradient Boostining
#intialize the input
xi=x
yi=y
ei=0 #numer of observation
n= len(yi)#number of rows
predf = 0 #inital prediction

for i in range(30):
	tree =DecisionTree(xi,yi)
	tree.find_better_split(0)
	row = np.where(xi==tree.split)[0][0]
	left_index = np.where(xi<= tree.split)[0]
	right_index =np.where(xi>= tree.split)[0]
	predi = np.zeros(n)
	np.put(predi, left_index, np.repeat(np.mean(yi[left_index]), row)) #replace left size with mean y
	np.put(predi, right_index, np.repeat(np.mean(yi[right_index]),row)) #replace right size with mean y
	predi = predi[:, None]  # make long vector (n*1) in compatible with y
	predf = predf + predi # final prediction will be previous prediction value + new prediction of residual
	ei = y - predf
	yi = ei #update yi as residual

	# plot after prediction
	xa = np.array(x.T)[0] #column name of x is x
	order = np.argsort(xa)
	xs = np.array(xa)[order]
	ys = np.array(predf)[order]

	f, (ax1, ax2) = plt.subplots(1,2, sharey = True, figsize=(13,2.5))
	ax1.plot(x,y,'o')
	ax1.plot(xs,ys, 'r')
	ax1.set_title(f'Prediction (Iteration {i+1})')
	ax1.set_xlabel('x')
	ax1.set_ylabel('y/y_pred')

	ax2.plot(x, ei, 'go')
	ax2.set_title(f'REsidual vs. x (Iteration {i+1})')
	ax2.set_xlabel('x')
	ax2.set_ylabel('Residuals')

