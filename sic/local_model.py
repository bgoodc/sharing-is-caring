import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers
from keras.utils.np_utils import to_categorical

from sklearn import svm


class LocalModel(object):
    """Wrapper around model.
    """
    def predict_classes(self, X):
        pass

    def accuracy(self, X, y):
        return np.mean(np.equal(self.predict_classes(X), y))


class LocalLinearModel(LocalModel):
    def __init__(self, input_dim=2, output_dim=2):
        # Build model
        model = Sequential()
        model.add(Dense(input_dim=input_dim, output_dim=output_dim,
                        activation="softmax", init="normal"))
        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd', metrics=['accuracy'])

        self.model = model
        self.input_dim = input_dim
        self.output_dim = output_dim

    def fit(self, X, y, epochs=30):
        y_categorical = to_categorical(y, num_classes=self.output_dim)
        self.model.fit(X, y_categorical, nb_epoch=epochs)

    def predict_classes(self, X):
        return self.model.predict_classes(X)
        
        
class LocalSVM(LocalModel):
    def __init__(self, input_dim=2, output_dim=2, kernel='rbf'):
        self.model = svm.SVC(kernel=kernel)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def fit(self, X, y):
        self.model.fit(X,y)

    def predict_classes(self, X):
        return self.model.predict(X)
