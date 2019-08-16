# -*- coding: utf-8 -*-
"""
Neural Net implementation
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
from sklearn.model_selection import ShuffleSplit

#Define activation function
def tanh(x):
    """It is a hyperbolic tangent function-Tanh.
    The function: f(x) = (1-exp(-2x))/(1+exp(-2x)).
    Function range from -1-1, with zero centered
    Vanishing gradient problem - saturate and kill gradient """
    return (np.exp(2*x) - 1)/(np.exp(2*x) +1)
def relu(x):
    """Rectified Linear unit.
    The function: f(x) = max(0,x)
    No vaninish gradient problem
    Should be used for hidden layers only.
    Should not be used for output layers"""
    return np.argmax(0,x)
def softmax(x):
    """The function: f(x)= exp(x)/sum(exp(x)).
    It turns logits into probabilitis that sum to 1.
    Compute softmax values for each sets of scores in x"""
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
def sigmoid(x):
    """It is an activation function of form: f(x) = 1/(1+exp(-x))
    It ranges from 0-1, with a shape S-curve
    Vanishing gradient problem - saturate and kill gradient
    Slow convergence
    Output is not zero centered. """
    return 1/(1+exp(-x))

#Define loss derivatives
def loss_derivative(y,y_hat):
    """Get the loss derivative of (y_hat-y)"""
    return (y_hat-y)

def tanh_derivative(x):
    return (1 - np.power(x, 2))

def softmax_loss(y, y_hat):
    min_val = 0.000000000001
    n = y.shape[0]
    loss = -1/n*np.sum(y * np.log(y_hat.clip(min=min_val)))
    return loss

#Forward Propagation
def forward_propagation(model,a0):
    #load parameters
    w1,b1,w2,b2,w3,b3 = model["w1"], model["b1"], model["w2"], model["b2"], model["w3"], model["b3"]
    #Calculate the linear combination
    z1 = a0.dot(w1) + b1
    #Put it through the first activation
    a1 = np.tanh(z1)
    #Calculate the second linear combination -1st hidden
    z2 = a1.dot(w2) + b2
    #Put it through the second activation - 2nd hidden layer
    a2= np.tanh(z2)
    #Calculate the third linear step
    z3 = a2.dot(w3) + b3
    #Put it throught the third activation -
    a3 = z3
    #Store all results in these values
    cache = {'a0': a0, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'a3': a3, 'z3': z3}
    return cache

#Backward Propagation
def backward_propagation(model,cache,y):
    #Load parameter
    w1,b1,w2,b2,w3,b3 = model["w1"], model["b1"], model["w2"], model["b2"], model["w3"], model["b3"]
    #Load forward propagation results
    a0, a1, a2, a3 = cache['a0'], cache['a1'], cache['a2'], cache['a3']
    #Get the number of samples
    n = y.shape[0]
    #Calculate the loss derivatives with respect to outputs
    dz3 = loss_derivative(y=y,y_hat =a3)
    #Calculate the loss derivatives with respective to the second layer weight
    dw3 = 1/n*(a2.T).dot(dz3)
     #Calculate the loss derivatives with respective to the second layer bias
    db3 = 1/n*np.sum(dz3, axis=0)
    #Calculate the loss derivatives with respective to the second layer
    dz2 = np.multiply(dz3.dot(w3.T), tanh_derivative(a2))
    #Calculate the loss derivatives with respective to the first layer weight
    dw2 = 1/n *np.dot(a1.T, dz2)
    #Calculate the loss derivatives with respective to the first layer bias
    db2 = 1/n*np.sum(dz2, axis=0)
    #Calculate the loss derivatives with respective to the linear
    dz1= np.multiply(dz2.dot(w2.T), tanh_derivative(a1))
    #Calculate the loss derivatiives with respective to the linear weight
    dw1 = 1/n*np.dot(a0.T, dz1)
    #Calculate the loss derivatives with respective to the linear bias
    db1 = 1/n*np.sum(dz1,axis=0)
    #Store the graditent
    grads = {'dw3':dw3, 'db3': db3, 'dw2': dw2, 'db2': db2, 'dw1': dw1, 'db1': db1}
    return grads

 #Training Phase
def initialize_parameters(nn_input_dim, nn_hdim, nn_output_dim):
    #First layer weights
    w1 = 2 * np.random.randn(nn_input_dim, nn_hdim) -1
    #First layer bias
    b1 = np.zeros((1,nn_hdim))
    #Second layer weights
    w2 = 2 * np.random.randn(nn_hdim, nn_hdim)-1
    #Second layer bias
    b2 = np.zeros((1,nn_hdim))
    #Output layer weights
    w3= 2 * np.random.randn(nn_hdim, nn_output_dim) -1
    #Output layer bias
    b3 = np.zeros((1,nn_output_dim))
    #Package and return model
    model = { 'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2,'w3':w3,'b3':b3}
    return model
#Update parameter
def update_parameters(model,grads,learning_rate):
    #Load parameter
    w1,b1, w2, b2, w3, b3=model['w1'], model['b1'], model['w2'], model['b2'], model['w3'], model['b3']
    #Update parameter
    w1  -= learning_rate * grads['dw1']
    b1  -= learning_rate * grads['db1']
    w2  -= learning_rate * grads['dw2']
    b2  -= learning_rate * grads['db2']
    w3  -= learning_rate * grads['dw3']
    b3  -= learning_rate * grads['db3']
    #Store and return parameters
    model = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2, 'w3': w3, 'b3': b3}
    return model

#Predict model
def predict(model,x):
    #Do forward propagation
    c = forward_propagation(model,x)
    #Get prediction y_hat
    y_hat = c['a3']
    return y_hat

#Calculate prediction accuracy
def calc_accuracy(model,x,y):
    #Get the total number of samples
    n = y.shapes[0]
    #Do prediction
    pred = predict(model,x)
    # Transform the shape of prediction
    pred = pred.reshape(y.shape)
    #Calculate the number of wrong examples
    error = np.sum(np.abs(pred-y))
    #Calculate accuracy
    accuracy = (n-error)/n * 100
    return accuracy

#Implement the model -training
def train(model,X_,y_, learning_rate, epochs = 100, print_loss = False):
    losses = []
    #Gradient descent. Loop over epochs
    for i in range(0, epochs):
        #Forward propagation
        cache = forward_propagation(model,X_)
        #Backward propagation
        grads = backward_propagation(model, cache, y_)
        #Update gradient descent parameter
        model = update_parameters(model=model, grads=grads, learning_rate = learning_rate)
        #Print loss and accuracy for each of 10 interation
        if print_loss and i % 100 == 0:
            a3 = cache['a3']
            print('Loss after iteration ', i, ':', mean_absolute_error(y_,a3))
            y_hat = predict(model,X_)
            y_true = y_
            print('Accuracy after iteration ', i, ':',\
				   mean_absolute_error(y_pred=y_hat, y_true = y_true), )
        losses.append(mean_absolute_error(y_pred=y_hat,y_true=y_true))
    return model, losses

#Get data
boston = datasets.load_boston()
boston_data = pd.DataFrame(data= np.c_[boston['data'], boston['target']], \
						   columns= np.append (boston['feature_names'].astype(object), ['target']))
#Get dummy
boston_data = pd.concat([boston_data, pd.get_dummies(boston_data['RAD'],prefix = 'RAD_')], axis=1)
print(boston_data.head(2))
print(boston_data.columns)

# Train Base Models for CF12
YChoice = ['target']
XChoice =  boston_data.columns.tolist()
XChoice.remove('RAD')
XChoice.remove('RAD__1.0')
XChoice.remove('target')
print(len(XChoice))
DataChoice = boston_data 
Data = DataChoice[XChoice +YChoice].dropna()
print("Dataset has {} entries and {} features".format(*Data[XChoice].shape))
#Train XGBoosting
print("Dataset has {} entries and {} features".format(*Data.shape))
X, y = Data[XChoice], Data[YChoice]
train_inds, test_inds = next(ShuffleSplit(test_size = .1, random_state = 42).split(X,y))
X_train, X_test, y_train, y_test, = X.iloc[train_inds], \
													X.iloc[test_inds],\
													y.iloc[train_inds],\
													y.iloc[test_inds]

print("Dataset Train has {} entries and {} features".format(*X_train.shape))
print("Dataset Test has {} entries and {} features".format(*X_test .shape))

#Preprocessing
numeric_features =  ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',\
					  'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
all_features = pd.concat((X_train,X_test))
all_features[numeric_features] = all_features[numeric_features].apply(\
			lambda x: (x - x.mean()) / (x.std()))
X_train_scaled = all_features.iloc[train_inds]
X_test_scaled = all_features.iloc[test_inds]
print(X_train.shape)
#test
model = initialize_parameters(nn_input_dim = len(XChoice), nn_hdim = 10, nn_output_dim =len(YChoice))
model, losses = train(model,X_train_scaled.values,y_train.values, learning_rate=0.01, epochs=4000, print_loss=True)
Figure = plt.figure(figsize=(10,10))
print(model.get('w1'))
plt.plot(losses)

y_hat = predict(model,X_test_scaled.values)
mean_absolute_error(y_test,y_hat)
