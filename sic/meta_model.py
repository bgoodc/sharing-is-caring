import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Flatten
from keras import optimizers
import tensorflow as tf
from tensorflow.train import GradientDescentOptimizer
from privacy.dp_query.gaussian_query import GaussianAverageQuery
from privacy.optimizers.dp_optimizer import DPGradientDescentOptimizer, make_optimizer_class

from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# META MODEL
class MetaModel:
    def __init__(self,  input_shape, models, X=None, y=None,flags=None):
        """
        Implements a linear meta-model

        Args:
        X: train data input
        y: train data labels
        models: list of models
        flags: tensorflow flags
        """
        self.num_peers = num_peers = len(models)
        self.input_shape = input_shape
        self.X = X
        self.ground_truth = y
        self.models = models

        # compute expert_correctness matrix
        if X is not None:
            self.Y = self.zero_one_loss(X, y, models)

        # initialize meta-model

        model = Sequential()
        model.add(Dense(2*num_peers + 2,input_shape=input_shape))
        model.add(Activation('relu'))
        if len(input_shape)!=1:
          model.add(Flatten())
        model.add(Dense(num_peers))
        model.add(Activation('softmax'))

        # set flags
        if flags is None:
            flags = MetaModel.default_flags()
        self.flags = flags

        # optimizer (see code from mnist example in tensoflow/privacy)
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
        else:
            optimizer = GradientDescentOptimizer(learning_rate=flags['learning_rate'])

        model.compile(optimizer=optimizer, loss='mean_squared_error')
        self.model = model

    def predict_classes(self, X, batch_size=None):
        """Given a list of input data, return the output prediction.
        """
        which_expert = self.get_experts(X)
        predictions = np.transpose([model.predict_classes(X)
            for model in self.models])
        expert_prediction = [pair[1][pair[0]]
                for pair in zip(which_expert, predictions)]
        return np.array(expert_prediction)

    def accuracy(self, X_test, y_test):
        return np.mean(np.equal(self.predict_classes(X_test), y_test))

    def fit(self, x=None, y=None):
        """Run the fit function on model.
        """
        if x is not None:
          py = self.zero_one_loss(x,y,self.models)
          if len(x.shape) > 2:
            x=x.reshape(x.shape[0], -1)
          self.model.fit(x, py, batch_size=1)
        else:
            self.model.fit(self.X, self.Y,batch_size=1)

    def get_experts(self, X):
        """Given a list of input data, return a list of experts for each data.
        """
        if len(X.shape)>2:
          tmp=X.reshape(X.shape[0],-1)
          return self.model.predict_classes(tmp)
        return self.model.predict_classes(X)

    def get_weights(self):
        """Return list of weights for each layer in the model
        """
        return self.model.get_weights()

    def set_weights(self, weights):
        """Sets the weights for each layer in the model
        """
        self.model.set_weights(weights)

    def compute_gradient(self, X=None, Y=None, update=True):
        """
        Computes gradients, possibly updating the meta-model

        Args:
        X: data matrix with training instances
        Y: for each instance, a vector describing correctness of each peer
           e.g. Y_i = [ 1 0 0 1 0 1 1 ] means that on the ith training example
           the 0th, 3rd, 5th, and 6th peers were correct.
        update: if True, compute gradient and take a step
                if False, compute_gradient will have no side-effect on MetaModel
        """
        if X is None:
            X = self.X
            Y = self.Y
        else:
          Y=self.zero_one_loss(X,Y,self.models)
        if len(X.shape) > 2:
          X = X.reshape(X.shape[0], -1)

        saved_weights = self.get_weights()
        self.model.train_on_batch(X, Y)
        new_weights = self.get_weights()

        gradient = [new - old for (new, old) in zip(new_weights, saved_weights)]

        if not update:
            self.set_weights(saved_weights)

        return gradient

    def update_gradient(self, gradients):
        """Averages array of gradients and add to model weights
        """
        weights = self.get_weights()
        gradient = [0 for _ in weights]
        num_gradients = len(gradients)
        for steps in gradients:
            gradient = [weight + (step / num_gradients)
                    for (weight, step) in zip(gradient, steps)]

            updates = [weight + step for (weight, step) in zip(weights, gradient)]
        self.model.set_weights(updates)


    @staticmethod
    def zero_one_loss(X, y, models):
        """Computes expert-instance correctness matrix.
        Args
        X: training instances
        y: training labels
        models: list of models


        Return:
        output_{ij} is a matrix showing that on the ith instance, the jth expert
        was correct (1) or incorrect (0)
        """
        num_instances = X.shape[0]
        num_experts = len(models)
        output = np.empty([num_experts, num_instances])

        for i, model in enumerate(models):
            predictions = model.predict_classes(X)
            y=y.reshape(-1)
            output[i] = np.equal(predictions, y)

        return np.transpose(output)


    @staticmethod
    def default_flags():
        """Returns default tensorflow DP flags settings.
        """
        opts = dict()
        opts['dpsgd'] = True
        opts['learning_rate'] = 0.15
        opts['noise_multiplier'] = 1.1
        opts['l2_norm_clip'] = 1.0
        opts['batch_size'] = 250
        opts['epochs'] = 300
        opts['microbatches'] = 1
        opts['model_dir'] = None
        return opts

class MetaModelCIFAR(MetaModel):

    def __init__(self, input_shape, models, X=None, y=None, flags=None):
        """
        Implements a CNN meta-model for use in image classification.

        Args:
        X: train data input
        y: train data labels
        models: list of models
        flags: tensorflow flags
        """
        self.num_peers = num_peers = len(models)
        self.X = X
        self.ground_truth = y
        self.models = models

        # compute expert_correctness matrix
        if X is not None:
            self.Y = self.zero_one_loss(X, y, models)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
            input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_peers))
        model.add(Activation('softmax'))


        # set flags
        if flags is None:
            flags = MetaModel.default_flags()
        self.flags = flags

        # optimizer (see code from mnist example in tensoflow/privacy)
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
        else:
            optimizer = GradientDescentOptimizer(learning_rate=flags['learning_rate'])

        model.compile(optimizer=optimizer, loss='mean_squared_error')

        self.model = model
