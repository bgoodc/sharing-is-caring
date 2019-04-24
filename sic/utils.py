import random
import numpy as np
import matplotlib.pyplot as plt

# Visualization function
def plot_decision_boundary(model, X, y):
    """Plot decision boundary
    note: taken from stackexchange
    """
    color1 = [0.38,0.659,1,0.5]
    color2 = [1,0.412,0.38,0.5]
    
    # Set min and max values and give it some padding
    x_min, x_max = X['0'].min() - .5, X['0'].max() + .5
    y_min, y_max = X['1'].min() - .5, X['1'].max() + .5
    h = 0.1
    
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict the function value for the whole gid
    def pred_func(x):
      return model.predict_classes(x)
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
   
    Z = Z.reshape(xx.shape)
    
    # Generate colors
    colors = list(map(lambda v: color1 if v==0 else color2, y))
    
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z)#, cmap=plt.cm.Pastel1)
    plt.scatter(X['0'], X['1'], c=colors)

#plot_decision_boundary(meta_model_svm, plot_X, plot_y)


#sample a random distribution
def random_distribution(dim):
  left=1
  ids=[i for i in range(dim)]
  random.shuffle(ids)
  result=[0 for i in range(dim)]
  for i in range(dim):
    tmp=random.uniform(0,left)
    result[ids[i]]=tmp
    left-=tmp


def change_image_data(x_train,y_train,noise=0.5,n_experts=3):
  train_X=[]
  train_y=[]
  for i in range(n_experts):
    train_filter = np.where((y_train == i*2 ) | (y_train == i*2+1))
    xtrain, ytrain = x_train[train_filter], y_train[train_filter]
    xtrain = xtrain.reshape(xtrain.shape[0], 28, 28, 1)
    xtrain = xtrain.astype('float32')
    xtrain /= 255
    if noise!=0:
      xtrain+=np.random.laplace(0, noise, xtrain.shape)
    train_X.append(xtrain)
    train_y.append(ytrain)
  return train_X,train_y