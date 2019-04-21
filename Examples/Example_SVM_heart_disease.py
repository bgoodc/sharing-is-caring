import sic
from sic.testing.data_generators import MultiModalData
from sic.local_model import LocalLinearModel, LocalSVM
from sic.meta_model import MetaModel

import seaborn as sns

import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.model_selection import train_test_split

import matplotlib as mpl
import heapq
from scipy.linalg import eigh

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('heart.csv')
df = df.sample(frac=1)

x = df.values[:,:-1] #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
y = df.values[:,-1]

xtrain, xtest, ytrain, ytest = train_test_split(x_scaled, y)

model = SVC(kernel='rbf')
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

proba = 0.5
d1, d2 = [], []
for row in df.itertuples():
    rand = random.random()
    if row.chol > 246:
        if rand > 0.1:
            d2.append(row)
        else:
            d1.append(row)
    elif row.chol < 246:
        if rand > 0.1:
            d1.append(row)
        else:
            d2.append(row)
    else:
        if rand > 0.5:
            d1.append(row)
        else:
            d2.append(row)

df1 = pd.DataFrame(d1)
df1.set_index('Index', inplace=True)
df2 = pd.DataFrame(d2)
df2.set_index('Index', inplace=True)

x1 = df1.values[:,:-1][:147] #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x1_scaled = min_max_scaler.fit_transform(x1)
y1 = df1.values[:,-1][:147]
x1train, x1test, y1train, y1test = train_test_split(x1_scaled, y1, test_size=0.5)

x2 = df2.values[:,:-1] #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x2_scaled = min_max_scaler.fit_transform(x2)
y2 = df2.values[:,-1]
x2train, x2test, y2train, y2test = train_test_split(x2_scaled, y2, test_size=0.5)

# Generate local experts
expert1 = LocalSVM()
expert2 = LocalSVM()
expert1.fit(x1train, y1train)
expert2.fit(x2train, y2train)


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

models = [expert1, expert2]
krange=200
X=[x1train,x2train]
Y=[y1train,y2train]
plot_X=np.concatenate((x1test,x2test),axis=0)
plot_y=np.concatenate((y1test,y2test),axis=0)

print(f"Expert 1: {expert1.accuracy(plot_X, plot_y)}")
print(f"Expert 2: {expert2.accuracy(plot_X, plot_y)}")

flags['dpsgd'] = False
acc_svm1=[]
meta_model_svm = MetaModel(input_shape=(13,),  models=models, flags=flags)
for k in range(krange):
  meta_model_svm.update_gradient([meta_model_svm.compute_gradient(X[i], Y[i], update=False) for i in range(2)])
  if k%5==0:
    print("num= ",k)
    val=meta_model_svm.accuracy(plot_X, plot_y)
    print(f"meta without DP : {val}")
    acc_svm1.append((k,val))


flags['dpsgd'] = True
acc_svm2=[]
meta_model_svm2 = MetaModel(input_shape=(13,), models=models, flags=flags)
for k in range(krange):
  meta_model_svm2.update_gradient([meta_model_svm2.compute_gradient(X[i], Y[i], update=False) for i in range(2)])
  if k%5==0:
    print("num= ",k)
    val=meta_model_svm2.accuracy(plot_X, plot_y)
    print(f"meta with DP : {val}")
    acc_svm2.append((k,val))
# First time, need to run visualization code below

e1=expert1.accuracy(plot_X, plot_y)
e2=expert2.accuracy(plot_X, plot_y)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.plot(list(zip(*acc_svm1))[0], list(zip(*acc_svm1))[1], 'ro',label='Without DP')
plt.plot(list(zip(*acc_svm2))[0], list(zip(*acc_svm2))[1], 'bo',label='With DP')
plt.plot([e1 for i in range(acc_svm1[-1][0])],label='expert1')
plt.plot([e2 for i in range(acc_svm2[-1][0])],label='expert2')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('accuracy')
plt.xlabel('epochs')