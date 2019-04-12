import random
import math
import numpy as np
import pandas as pd
from numpy.linalg import norm
from numpy.random import normal, multivariate_normal
from sklearn.preprocessing import normalize

# BASIC DATASET GENERATOR
class MultiModalData:
    """A class that generates data according to a k-modal mixture of Gaussians,
    and for each mode, there is a corresponding linear separator that splits
    points into one class or the other, with some amount of margin-based noise.

    Args:
    modes: number of modes in the distribution
    dimension: dimension of generated data
    separation: parameter to control how much overlap between modes
    noise: paramter to control how much margin-based noise
    """
    def __init__(self, modes=3, dimension=2, separation=3, noise=16, params=None):
        if params is not None:
            self.modes = params['modes']
            self.dimension = params['dimension']
            self.separation = params['separation']
            self.centers = params['centers']
            self.covariances = params['covariances']
            self.directions = params['directions']
        else:
            self.modes = modes
            self.dimension = dimension
            self.separation = separation
            self.centers = (np.random.rand(modes, dimension) - 0.5) * 10

            self.covariances = np.empty((modes, dimension, dimension))

            for i in range(modes):
                a = np.random.normal(loc=0.0, scale=1.0/separation,
                                     size=(dimension, dimension))
                self.covariances[i] = a.dot(a.T)


            self.directions = normalize(normal(0.0, size=(modes, dimension)), axis=1)
            self.noise = noise

    def generate(self, num_samples, modal_distribution=None, randomize=True):
        """Generates num_samples of data points, where the probability
        of selecting the ith mode is given by modal_distribution. From there,
        a datapoint is generated from a Gaussian, N(center_i, covariance_i)
        The label will be labeled according to hyperplane defined by direction_i
        through center_i. Noise will be generated with respect to margin.

        Args:
        num_samples: number of samples to generate
        modal_distribution: probability that a sample is drawn from each mode
        randomize: if no randomization, data from each mode will be adjacent

        Return:
        df: a dataframe with columns 0, 1, and y
        """
        X = []
        y = []

        if modal_distribution is None:
            modal_distribution = [1,1,1]
            
        modal_distribution = np.absolute(np.array(modal_distribution))
        modal_distribution = modal_distribution / np.linalg.norm(modal_distribution, ord=1)

        # number of times each mode will be used
        per_mode = np.random.multinomial(num_samples, modal_distribution)

        for i, num_mode_i in enumerate(per_mode):
            center = self.centers[i]
            direction = self.directions[i]
            instances = np.random.multivariate_normal(mean=center,
                                                      cov=self.covariances[i],
                                                      size=num_mode_i)
            for instance in instances:
                X.append(instance)
                threshold = (instance - center).dot(direction)
                if threshold < 0:
                    label = 0
                else:
                    label = 1
                if np.random.binomial(1, 2**(-self.noise*abs(threshold))):
                    label = 1 - label
                y.append(label)

            X = np.array(X)
            y = np.array(y)
            df = pd.DataFrame()
            for i in range(self.dimension):
                df[f'{i}'] = X[:,i]
            df['y'] = y
            if not randomize:
                return df
            return df.sample(frac=1).reset_index(drop=True)
