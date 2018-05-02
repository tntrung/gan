import math
import numpy as np
import tensorflow as tf

# for mnist
from tensorflow.examples.tutorials.mnist import input_data

# for cifar10
import cPickle as pickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict['data'], dict['labels']
    
def read_cifar10(data_dir, filenames):
    all_data = []
    all_labels = []
    for filename in filenames:        
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    #normalize into [0,1]
    images = images.astype(float)/255.0
    images = np.reshape(images,(-1, 3, 32, 32))
    images = np.transpose(images,(0,2,3,1)) #tranpose to standard order of channels
    images = np.reshape(images,(-1, 32*32*3))
    
    print('cifar10 data: {}'.format(np.shape(images)))
    return images, labels

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
		elif name == 'cifar10':
			# download data files from: 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz' 
			# extract into the correct folder
			data_files = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
			self.data, _ = read_cifar10(source, data_files)
			
		self.minibatches = self.random_mini_batches(self.data.T, self.batch_size, self.seed)

	def db_name(self):
		return self.name

	def data_dim(self):
		if self.name == 'mnist':
			return 784  #28x28
		elif self.name == 'cifar10':
			return 3072 #32x32x3
		else:
			print('data_dim is unknown.\n')

	def data_shape(self):
		if self.name == 'mnist':
			return [28, 28, 1]
		elif self.name == 'cifar10':
			return [32, 32, 3]
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
