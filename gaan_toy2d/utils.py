import math
import numpy as np
import tensorflow as tf
from model_mlp import theta_def, mlp

def sample_z(m, n):
    return np.random.uniform(-1.0, 1.0, size=[m, n])
    
def gradient_penalty(inter_data, theta):
    _, inter = mlp(inter_data, theta)
    gradients = tf.gradients([inter], [inter_data])[0]
    slopes = tf.sqrt(tf.reduce_mean(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)
    return gradient_penalty


def random_mini_batches(X, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batches.append(mini_batch_X)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batches.append(mini_batch_X)
    
    return mini_batches
