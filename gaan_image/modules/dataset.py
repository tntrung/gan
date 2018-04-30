import math
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

class Dataset(object):

	def __init__(self, name='mnist', source='./data/mnist/', one_hot=True, batch_size = 64, seed = 0):


		self.name    		 = name
		self.source  		 = source
		self.one_hot 		 = one_hot
		self.batch_size      = batch_size
		self.seed            = seed

		self.count           = 0

		tf.set_random_seed(self.seed)  # Fix the random seed for randomized tensorflow operations.

		if name == 'mnist':
			self.mnist = input_data.read_data_sets(source)
			self.data  = self.mnist.train.images
			print('data shape: {}'.format(np.shape(self.data)))
			
		self.minibatches = self.random_mini_batches(self.data.T, self.batch_size, self.seed)


	def db_name(self):
		return self.name

	def data_dim(self):
		if self.name == 'mnist':
			return 784
		else:
			print('data_dim is unknown.\n')

	def data_shape(self):
		if self.name == 'mnist':
			return [28, 28]
		else:
			print('data_dim is unknown.\n')
			
	def mb_size(self):
		return self.batch_size

	def next_batch(self):

		if self.count == len(self.minibatches):
			self.count = 0
			self.minibatches = self.random_mini_batches(self.data.T, self.batch_size, self.seed)

		batch = self.minibatches[self.count]
		self.count = self.count + 1
		#if self.count % 10 == 0:
		#    print('Minibatch count = {} / {}'.format(self.count, len(self.minibatches)))
		return batch.T

	# Random minibatches for training
	def random_mini_batches(self, X, mini_batch_size = 64, seed = 0):
	    """
	    Creates a list of random minibatches from (X)
	    
	    Arguments:
	    X -- input data, of shape (input size, number of examples)
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
	    num_complete_minibatches = int(math.floor(m/self.batch_size)) # number of mini batches of size mini_batch_size in your partitionning
	    for k in range(0, num_complete_minibatches):
	        mini_batch_X = shuffled_X[:, k * self.batch_size : (k+1) * self.batch_size]
	        mini_batches.append(mini_batch_X)
	    
	    # Handling the end case (last mini-batch < mini_batch_size)
	    if m % mini_batch_size != 0:
	        mini_batch_X = shuffled_X[:, num_complete_minibatches * self.batch_size : m]
	        mini_batches.append(mini_batch_X)
	    
	    return mini_batches
