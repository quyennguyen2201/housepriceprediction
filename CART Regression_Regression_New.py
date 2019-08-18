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

#Remove array
def removearray(L,arr):
	ind = 0
	size = len(L)
	while ind != size and not np.array_equal(L[ind],arr):
		ind += 1
	if ind != size:
		L.pop(ind)
	else:
		raise ValueError('array not found in list.')
	return L
#Create cost function
#Gini index
def score_index(groups):
	#sum weighted gini index for each group
	score = 0.0
	for group in groups:
		size = float(len(group))
		#avoid divide by zero
		if size ==0:
			continue
		score =0.0
		#score the group based on the score for each glass
		std_group = group[-1].std()* size
		#weight the group score by its relative size
		score += std_group
	return score


#Split the dataset
def test_split(index,value,dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

#Select the best split point for a dataset
def get_split(dataset):
	"""get the class value from the latest column in the dataset
	set the initial split index, value, score to be 999, and set class to be None
	loop through all columns in the datapoint
	loop through all observations in the datapoint
	split the dataset based on the value of the observations
	if gini score is smaller, update the split values """
	b_index, b_value, b_score, b_groups = 999,999,999, np.float('inf')
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			score = score_index(groups)
#			print('Split: [X%d < %.3f] Score: %.3f' % ((index+1), row[index], score))
			if score < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], score, groups
	return {'index': b_index, 'value': b_value, 'groups': b_groups}


#Create a terminal node value -
def to_terminal(group):
	"""select a class value for a group of rows.
	retrun the most common class """
	return np.mean(group[-1])


#Create schild scplit
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left+right)
		return
	#check for depth max
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	#process left chidl
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	#process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)

#Build a decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size,1)
	return root

#Print the tree
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))

#Make a prediction
def predict(node,row):
	if row[node['index']]<node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'],row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'],row)
		else:
			return node['right']

#Decision tree
def decision_tree(train,test,max_depth, min_size):
	tree = build_tree(train,max_depth,min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return predictions

#Calculate accuracy percentage
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset)/n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

#Calculate accuracy score
def accuracy_metric(actual, predicted):
	correct =0
	for i in range(len(actual)):
		values =abs(predicted[i]-actual[i])/actual[i]
		correct +=values
	return correct / float(len(actual))

#Evaluate an algorithm
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set = removearray(train_set, fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores




############################################################################
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
dataset =np.array(dataset)

#Test the tree
tree =build_tree(dataset,10,1)
print_tree(tree)

############################################################################
#Test on Iris Dataset
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

boston_data_array=np.array(pd.concat([X,y],axis=1))
print(boston_data_array.shape)

#Test the tree
tree = build_tree(boston_data_array,10,1)
print_tree(tree)

#Test the tree
n_folds = 5
max_depth = 5
min_size =10
scores = evaluate_algorithm(boston_data_array, decision_tree, n_folds, max_depth, min_size)
print('Scores: %s' %scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))




