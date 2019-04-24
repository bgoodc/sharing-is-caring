import sic
from sic.testing.data_generators import MultiModalData
from sic.local_model import LocalCNNModel
from sic.meta_model import MetaModel
from sic.utils import change_image_data

import seaborn as sns

import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.model_selection import train_test_split

import matplotlib as mpl
import matplotlib.patches as mpatches
import heapq
from scipy.linalg import eigh

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
"""This file has 3 parts -
1. show an encrypted image
2. Add noise to training data, and then train on meta-models w/ or w/o DP
3. Train experts with DP, , and then train on meta-models w/ or w/o DP
"""
"""Show how images look like after adding Laplace noise"""

#MNIST
# part of the code is from: 
# https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
image_index = 7777 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8
plt.imshow(x_train[image_index], cmap='Greys')
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

x_train+=np.random.laplace(0, 0.5, x_train.shape)
x_train*=255
x_train=x_train[:,:,:,0]
x_train.reshape(60000,28,28)

print(x_train.shape)
plt.imshow(x_train[image_index], cmap='Greys')

"""Training on private data. i.e. Add laplace noise"""

# training experts 
dim=28
out_dim=10
input_shape=(28,28,1)
n_experts=3
n_one_record=5000
n_records=n_experts*n_one_record

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
train_X,train_y=change_image_data(x_train, y_train,0.5,n_experts)

test_X = x_test.reshape(x_test.shape[0], 28, 28, 1)
test_X = test_X.astype('float32')
test_X /= 255
test_y=y_test

# Define centralized meta model
flags = dict()
flags['dpsgd'] = False
flags['learning_rate'] = 0.15
flags['noise_multiplier'] = 1.1
flags['l2_norm_clip'] = 1.0
flags['batch_size'] = 250
flags['epochs'] = 300
flags['microbatches'] = 1
flags['model_dir'] = None

# Train local experts
models=[] #experts
for i in range(n_experts):
  expert1 = LocalCNNModel(dim,out_dim,(28,28,1),flags)
  expert1.fit(train_X[i], train_y[i],epochs=5)
  models.append(expert1)

# preprocess data to 1D
input_shape=(784,)
meta_X_train, meta_y_train = train_X,train_y

# Meta model training data
X = np.array(meta_X_train)
Y = np.array(meta_y_train)
plot_X = test_X
plot_y = test_y
for i in range(n_experts):
  print("Expert ",i, " ", models[i].accuracy(plot_X, plot_y))

# Joint training phase
krange=10
trange=1
acc_cnn1=[]
acc_cnn2=[]
# 
flags['dpsgd'] = False
meta_model_CNN = MetaModel(input_shape=input_shape,  models=models, flags=flags)
for k in range(krange):
    meta_model_CNN.update_gradient([meta_model_CNN.compute_gradient(X[i][:11262], Y[i][:11262], update=False) for i in range(n_experts)])
    if k%1==0:
        print("num=",k)
        val=meta_model_CNN.accuracy(plot_X, plot_y)
        acc_cnn1.append((k,val))
        print(f"meta without DP : {val}")

flags['dpsgd'] = True
meta_model_CNN2 = MetaModel(input_shape=input_shape, models=models, flags=flags)
for k in range(krange):
    meta_model_CNN2.update_gradient([meta_model_CNN2.compute_gradient(X[i][:11262], Y[i][:11262], update=False) for i in range(n_experts)])
    if k%1==0:
        print("num=",k)
        val=meta_model_CNN2.accuracy(plot_X, plot_y)
        acc_cnn2.append((k,val))
        print(f"meta with DP : {val}")


plt.plot(list(zip(*acc_cnn1))[0], list(zip(*acc_cnn1))[1], 'ro',label='Without DP')
plt.plot(list(zip(*acc_cnn2))[0], list(zip(*acc_cnn2))[1], 'bo',label='With DP')
for i in range(n_experts):
    e=models[i].accuracy(plot_X, plot_y)
    plt.plot([e for i in range(acc_cnn1[-1][0])],label=f'expert{i}')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('accuracy')
plt.xlabel('epochs')

"""Training with tensorflow privacy"""
# Add DP to experts
# training experts 
dim=28
out_dim=10
input_shape=(28,28,1)
n_experts=3
n_one_record=5000
n_records=n_experts*n_one_record

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


train_X,train_y=change_image_data(x_train, y_train,0)

test_X = x_test.reshape(x_test.shape[0], 28, 28, 1)
test_X = test_X.astype('float32')
test_X /= 255
test_y=y_test

# Define centralized meta model
flags = dict()
flags['dpsgd'] = True
flags['learning_rate'] = 0.15
flags['noise_multiplier'] = 1.1
flags['l2_norm_clip'] = 1.0
flags['batch_size'] = 250
flags['epochs'] = 300
flags['microbatches'] = 1
flags['model_dir'] = None

# Train local experts
models=[] #experts
for i in range(n_experts):
  expert1 = LocalCNNModel(dim,out_dim,(28,28,1),flags)
  expert1.fit(train_X[i], train_y[i],epochs=5)
  models.append(expert1)

# preprocess data to 1D
input_shape=(784,)
meta_X_train, meta_y_train = train_X,train_y

# Meta model training data
X = np.array(meta_X_train)
Y = np.array(meta_y_train)

plot_X = test_X
plot_y = test_y

for i in range(n_experts):
    print("Expert ",i, " ", models[i].accuracy(plot_X, plot_y))

# Joint training phase
krange=10
trange=1

acc_cnn1=[]
acc_cnn2=[]
# 
flags['dpsgd'] = False
meta_model_CNN = MetaModel(input_shape=input_shape,  models=models, flags=flags)
for k in range(krange):
    meta_model_CNN.update_gradient([meta_model_CNN.compute_gradient(X[i][:11262], Y[i][:11262], update=False) for i in range(n_experts)])
    if k%1==0:
        print("num=",k)
        val=meta_model_CNN.accuracy(plot_X, plot_y)
        acc_cnn1.append((k,val))
        print(f"meta without DP : {val}")

flags['dpsgd'] = True
meta_model_CNN2 = MetaModel(input_shape=input_shape, models=models, flags=flags)
for k in range(krange):
    meta_model_CNN2.update_gradient([meta_model_CNN2.compute_gradient(X[i][:11262], Y[i][:11262], update=False) for i in range(n_experts)])
    if k%1==0:
        print("num=",k)
        val=meta_model_CNN2.accuracy(plot_X, plot_y)
        acc_cnn2.append((k,val))
        print(f"meta with DP : {val}")

plt.plot(list(zip(*acc_cnn1))[0], list(zip(*acc_cnn1))[1], 'ro',label='Without DP')
plt.plot(list(zip(*acc_cnn2))[0], list(zip(*acc_cnn2))[1], 'bo',label='With DP')
for i in range(n_experts):
    e=models[i].accuracy(plot_X, plot_y)
    plt.plot([e for i in range(acc_cnn1[-1][0])],label=f'expert{i}')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('accuracy')
plt.xlabel('epochs')