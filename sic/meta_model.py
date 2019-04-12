import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import tensorflow as tf
from tensorflow.train import GradientDescentOptimizer
from privacy.dp_query.gaussian_query import GaussianAverageQuery
from privacy.optimizers.dp_optimizer import DPGradientDescentOptimizer

# META MODEL
class MetaModel:
    def __init__(self, num_peers, input_shape, X, y, models, flags=None):
	"""
        Implements meta-model

        Args:
        num_peers: the number of peers with models
        input_shape: tuple describing the shape of an instance
                     e.g. for MNIST, input_shape = (784,)
        X: train data input
        y: train data labels
        models: list of models
        flags: tensorflow flags
        """
        self.num_peers = num_peers
        self.input_shape = input_shape
        self.X = X
        self.ground_truth = y
        self.models = models

        assert len(models) == num_peers, "num_peers != len(models)"

        # compute expert_correctness matrix
        self.Y = self.zero_one_loss(X, y, models)

        # initialize meta-model
        self.model = model = Sequential([
	    Dense(2*num_peers + 2, input_shape=input_shape),
	    Activation('relu'),
	    Dense(num_peers),
	    Activation('softmax'),
	])

        # set flags
        if flags is None:
	    flags = self.default_flags()
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

        self.model.compile(optimizer=optimizer, loss='mean_squared_error',
	                   metrics=['accuracy'])

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

    def fit(self):
        """Run the fit function on model.
        """
        self.model.fit(self.X, self.Y)

    def get_experts(self, X):
        """Given a list of input data, return a list of experts for each data.
        """
        val=self.model.predict(X)
        return np.argmax(val, axis=1)

    def get_model_weights(self):
        """Return list of weights for each layer in the model
        """
        return [layer.get_weights() for layer in self.model.layers]

    def set_model_weights(self, weights):
        """Sets the weights for each layer in the model
        """
        for i, layer in enumerate(self.model.layers):
	    layer.set_weights(weights[i])
            
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
        if Y is None:
	    Y = self.Y

        saved_weights = self.get_model_weights()
        self.model.train_on_batch(X, Y)

        gradient = []
        for i, layer in enumerate(self.model.layers):
	    gradient.append(layer.get_weights() - saved_weights[i])

        if not update:
	    self.set_model_weights(saved_weights)

        return gradient

    def update_gradient(self, gradients):
        """Averages array of gradients and add to model weights
        """
        weights = self.get_model_weights()
        for i, layer in enumerate(self.model.layers):
            avg_grad = np.mean(np.array([gradient[i] for gradient in gradients]))
        layer.set_weights(weights[i] + avg_grad)


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
            y_p=model.predict(X)
        if len(y_p.shape)>1 and y_p.shape[1]>1:
	    predictions = np.argmax(model.predict(X), axis=1)
        else:
	    predictions = model.predict(X)
            output[i] = np.equal(predictions, y)

        return np.transpose(output)


    @staticmethod
    def default_flags():
        """Returns default tensorflow DP flags settings.
        """
        opts = dict()
        opts['dpsgd'] =True
        opts['learning_rate'] = 0.15
        opts['noise_multiplier'] = 1.1
        opts['l2_norm_clip'] = 1.0
        opts['batch_size'] = 250
        opts['epochs'] = 300
        opts['microbatches'] = 1
        opts['model_dir'] = None
        return opts
