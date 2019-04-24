import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers
from keras.utils.np_utils import to_categorical
from tensorflow.train import GradientDescentOptimizer
from privacy.dp_query.gaussian_query import GaussianAverageQuery
from privacy.optimizers.dp_optimizer import DPGradientDescentOptimizer

from sklearn import svm


class LocalModel(object):
    """Wrapper around model.
    """
    def predict_classes(self, X):
        pass

    def accuracy(self, X, y):
        return np.mean(np.equal(self.predict_classes(X), y))
    def default_flags(self):
        opts = dict()
        opts['dpsgd'] = True
        opts['learning_rate'] = 0.15
        opts['noise_multiplier'] = 1.1
        opts['l2_norm_clip'] = 1.0
        opts['batch_size'] = 250
        opts['epochs'] = 60
        opts['microbatches'] = 250
        opts['model_dir'] = None
        return opts

class LocalLinearModel(LocalModel):
    def __init__(self, input_dim=2, output_dim=2,flags=None):
        # Build model
        model = Sequential()
        model.add(Dense(input_dim=input_dim, output_dim=output_dim,
                        activation="softmax", init="normal"))
        

       
        self.input_dim = input_dim
        self.output_dim = output_dim
        if flags is None:
            flags = self.default_flags()
        self.flags = flags
        if flags['dpsgd']:
            dp_average_query = GaussianAverageQuery(
                flags['l2_norm_clip'],
                flags['l2_norm_clip'] * flags['noise_multiplier'],
                flags['microbatches'])
            optimizer = DPGradientDescentOptimizer(
                dp_average_query,
                flags['microbatches'],
                learning_rate=flags['learning_rate'],
                unroll_microbatches=True)
            
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer, metrics=['accuracy'])
        self.model = model

    def fit(self, X, y, epochs=30):
        y_categorical = to_categorical(y, num_classes=self.output_dim)
        self.model.fit(X, y_categorical, nb_epoch=epochs)

    def predict_classes(self, X):
        return self.model.predict_classes(X)
        
class LocalCNNModel(LocalModel):
    def __init__(self, input_dim=28, output_dim=10,input_shape=(28,28,1),flags=None,optimizer='adam'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 8,
                             strides=2,
                             padding='same',
                             activation='relu',
                             input_shape=input_shape),
        tf.keras.layers.MaxPool2D(2, 1),
        tf.keras.layers.Conv2D(32, 4,
                             strides=2,
                             padding='valid',
                             activation='relu'),
        tf.keras.layers.MaxPool2D(2, 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(output_dim)])
        
        if flags is None:
            flags = self.default_flags()
        self.flags = flags
        if flags['dpsgd']:
            dp_average_query = GaussianAverageQuery(
                flags['l2_norm_clip'],
                flags['l2_norm_clip'] * flags['noise_multiplier'],
                flags['microbatches'])
            optimizer = DPGradientDescentOptimizer(
                dp_average_query,
                flags['microbatches'],
                learning_rate=flags['learning_rate'],
                unroll_microbatches=True)
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
        else:
            optimizer = GradientDescentOptimizer(learning_rate=flags['learning_rate'])
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        
        self.model=model
        
    def fit(self, X, y, epochs=10):
        y_categorical = to_categorical(y, num_classes=self.output_dim)
        self.model.fit(X, y,
            epochs=epochs,
            batch_size=self.flags['batch_size'])
        #self.model.fit(x=X,y=y_categorical, epochs=epochs)
        
    def predict_classes(self, X):
        return self.model.predict_classes(X)    
        
class LocalSVM(LocalModel):
    def __init__(self, input_dim=2, output_dim=2, kernel='rbf'):
        self.model = svm.SVC(kernel=kernel)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def fit(self, X, y):
        X=self.encrypt(X)
        self.model.fit(X,y)
        
    def encrypt(self,X,noise=0.5):
        return X+np.random.laplace(0, noise, X.shape)
      
    def predict_classes(self, X):
        return self.model.predict(X)