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


#Build decision tree
class DecisionTreeRegressor:
	def fit(self,X,y,min_leaf = 5, max_depth =1 ):
		self.dtree = Node(X,y,np.array(np.arange(len(y))), min_leaf, max_depth)
		return self
	def predict (self,X):
		return self.dtree.predict(X.values)

#Define Nodes
class Node:
	"""Represnt one decision point in our model. Each deciision within each model
	has 2 possible outcomes, left or right.
	Idxs stores the indexes of the subset of the data that Node is working with
	The decision is determined on the value of the node holds. It is the avarage
	of the data of the dependent variable for this Node.
	Find_varsplit fidn where we should split data """
	def __init__(self,x,y, indices, min_leaf=5, max_depth=1):
		self.x = x
		self.y = y
		self.indices = indices
		self.min_leaf = min_leaf
		self.max_depth= max_depth
		self.row_count = len(indices)
		self.col_count = x.shape[1]
		self.val = np.mean(y[indices])
		self.score = float('inf')
		self.find_varsplit()

	def find_varsplit(self):
		for c in range(self.col_count):	self.find_better_split(c)
		if self.is_leaf: return
		x = self.split_col
		left = np.nonzero(x<= self.split)[0]
		right = np.nonzero(x>=self.split)[0]
		self.left = Node(self.x,self.y,self.indices[left], self.min_leaf,self.max_depth)
		self.right= Node(self.x,self.y,self.indices[right], self.min_leaf, self.max_depth)

	def find_better_split(self, var_index):
		x = self.x.values[self.indices, var_index]

		for row in range(self.row_count):
			left = x <= x[row]
			right = x > x[row]
			if right.sum() < self.min_leaf or left.sum() < self.min_leaf: continue
			curr_score = self.find_score(left, right)
			if curr_score < self.score:
				self.var_index = var_index
				self.score = curr_score
				self.split = x[row]

	def find_score(self,left,right):
		y = self.y[self.indices]
		left_std = y[left].std()
		right_std = y[right].std()
		return left_std*left.sum()+ right_std*right.sum()

	@property
	def split_col(self): return self.x.values[self.indices, self.var_index]

	@property
	def is_leaf(self): return self.score == float('inf')

	def predict(self,x):
		return np.array([self.predict_row(xi) for xi in x])

	def predict_row(self,xi):
		if self.is_leaf: return self.val
		node = self.left if xi[self.var_index] <= self.split else self.right
		return node.predict_row(xi)



####TEst dataset
#Create a dataset
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
sns.scatterplot(x="X1",y="X2", hue="y", data=dataset)
print(dataset)
X= dataset[["X1","X2"]]
y= dataset["y"]

#Test the tree
regressor = DecisionTreeRegressor().fit(X,y,min_leaf = 10, max_depth =10)
preds = regressor.predict(X)

print(pearsonr(preds,y))

#Import data
#Get data
boston = datasets.load_boston()
boston_data = pd.DataFrame(data= np.c_[boston['data'], boston['target']],                     columns= np.append (boston['feature_names'].astype(object), ['target']))
#Get dummy
boston_data = pd.concat([boston_data, pd.get_dummies(boston_data['RAD'],prefix = 'RAD_')], axis=1)
print(boston_data.head(2))
print(boston_data.columns)
#Get X and y
y = boston_data['target']
X = boston_data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX',\
				 'PTRATIO', 'B', 'LSTAT', 'target', 'RAD__2.0', 'RAD__3.0',\
				 'RAD__4.0', 'RAD__5.0', 'RAD__6.0', 'RAD__7.0', 'RAD__8.0','RAD__24.0']]
X.shape, y.shape

#Test the tree
regressor = DecisionTreeRegressor().fit(X,y,min_leaf = 10, max_depth =10)
preds = regressor.predict(X)
print(pearsonr(preds,y))

