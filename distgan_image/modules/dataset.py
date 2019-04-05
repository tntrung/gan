import math
import glob
import os

# for image io
from skimage import io
from skimage.transform import resize

import numpy as np
import tensorflow as tf


# for mnist
from tensorflow.examples.tutorials.mnist import input_data

# for cifar10
import cPickle as pickle

# List all dir with specific name 
def list_dir(folder_dir, ext="png"):
    all_dir = sorted(glob.glob(folder_dir+"*."+ext), key=os.path.getmtime)
    return all_dir

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict['data'], dict['labels']

def imread(path, is_grayscale=False):
    if (is_grayscale):
        img = io.imread(path, is_grayscale=True).astype(np.float)
    else:
        img = io.imread(path).astype(np.float)
    return np.array(img)  
        
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

def read_stl10(data_dir):
    with open(data_dir, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))

        images = images.astype(float) / 255.0

        nb_imgs = np.shape(images)[0]
        new_imgs = np.zeros([nb_imgs, 48, 48, 3])
        for ii in range(nb_imgs):
            print(ii, nb_imgs)
            new_imgs[ii,:,:,:] = resize(images[ii,:,:,:], [48, 48])
        new_imgs = np.reshape(new_imgs, (-1, 48*48*3))
    return new_imgs

# processing for celeba dataset
def preprocess(img):
    crop_size = 108
    re_size   = 64
    top_left  = [(218 - crop_size)//2, (178 - crop_size)//2]
    img       = img[top_left[0]:top_left[0]+crop_size, top_left[1]:top_left[1]+crop_size, :]
    img       = resize(img, [re_size, re_size])
    return img
    
class Dataset(object):

    def __init__(self, name='mnist', source='./data/mnist/', one_hot=True, batch_size = 64, seed = 0):


        self.name            = name
        self.source          = source
        self.one_hot         = one_hot
        self.batch_size      = batch_size
        self.seed            = seed
        np.random.seed(seed) # To make your "random" minibatches the same as ours

        self.count           = 0

        tf.set_random_seed(self.seed)  # Fix the random seed for randomized tensorflow operations.

        if name == 'mnist':
            self.mnist = input_data.read_data_sets(source)
            self.data  = self.mnist.train.images
            print('data shape: {}'.format(np.shape(self.data)))
            self.minibatches = self.random_mini_batches(self.data.T, self.batch_size, self.seed)
        elif name == 'cifar10':
            # download data files from: 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz' 
            # extract into the correct folder
            data_files = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
            self.data, _ = read_cifar10(source, data_files)
            self.minibatches = self.random_mini_batches(self.data.T, self.batch_size, self.seed)
        elif name == 'celeba':
            # Count number of data images
            self.im_list  = list_dir(source, 'jpg')
            self.nb_imgs  = len(self.im_list)
            self.nb_compl_batches  = int(math.floor(self.nb_imgs/self.batch_size))
            self.nb_total_batches     = self.nb_compl_batches
            if self.nb_imgs % batch_size != 0:
               self.num_total_batches = self.nb_compl_batches + 1
            self.count = 0
            self.color_space = 'RGB'
        #elif name == 'stl10':
        #    self.data = read_stl10(source)
        #    self.minibatches = self.random_mini_batches(self.data.T, self.batch_size, self.seed)
        elif name == 'stl10':
            # Count number of data images
            self.im_list  = list_dir(source, 'png')
            self.nb_imgs  = len(self.im_list)
            self.nb_compl_batches  = int(math.floor(self.nb_imgs/self.batch_size))
            self.nb_total_batches     = self.nb_compl_batches
            if self.nb_imgs % batch_size != 0:
               self.num_total_batches = self.nb_compl_batches + 1
            self.count = 0
            self.color_space = 'RGB'

    def db_name(self):
        return self.name

    def data_dim(self):
        if self.name == 'mnist':
            return 784  #28x28
        elif self.name == 'cifar10':
            return 3072 #32x32x3
        elif self.name == 'celeba':
            return 12288 #64x64x3
        elif self.name == 'stl10':
            return 6912 # 48x48x3
        else:
            print('data_dim is unknown.\n')

    def data_shape(self):
        if self.name == 'mnist':
            return [28, 28, 1]
        elif self.name == 'cifar10':
            return [32, 32, 3]
        elif self.name == 'celeba':
            return [64, 64, 3]
        elif self.name == 'stl10':
            return [48, 48, 3]
        else:
            print('data_shape is unknown.\n')
            
    def mb_size(self):
        return self.batch_size

    def next_batch(self):

        if self.name == 'mnist' or self.name == 'cifar10': #or self.name == 'stl10':
            if self.count == len(self.minibatches):
                self.count = 0
                self.minibatches = self.random_mini_batches(self.data.T, self.batch_size, self.seed)
            batch = self.minibatches[self.count]
            self.count = self.count + 1
            return batch.T
        elif self.name in ['celeba', 'stl10']:
            batch = self.random_mini_batches([], self.batch_size, self.seed)
            return batch

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
        
        if self.name == 'mnist' or self.name == 'cifar10': #or self.name == 'stl10':
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
            #if m % mini_batch_size != 0:
            #    mini_batch_X = shuffled_X[:, num_complete_minibatches * self.batch_size : m]
            #    mini_batches.append(mini_batch_X)
            
            return mini_batches
            
        elif self.name in ['celeba', 'stl10']:
            
            if self.count == 0:
                self.permutation = list(np.random.permutation(self.nb_imgs))
                cur_batch = self.permutation[self.count * self.batch_size : (self.count + 1) * self.batch_size]  
            elif self.count > 0 and self.count < self.nb_compl_batches:
                cur_batch = self.permutation[self.count * self.batch_size : (self.count + 1) * self.batch_size]                           
            elif self.count == self.nb_compl_batches and self.num_total_batches > self.nb_compl_batches:
            #    cur_batch = self.permutation[self.nb_compl_batches * self.batch_size : self.nb_imgs]
            #elif self.count >= self.num_total_batches:
                self.count = 0
                self.permutation = list(np.random.permutation(self.nb_imgs))
                cur_batch = self.permutation[self.count * self.batch_size : (self.count + 1) * self.batch_size]                
            else:
                print('something is wrong with mini-batches')
            
            mini_batches = []

            # handle complete cases
            for k in cur_batch:
                img = imread(self.im_list[k])
                #print('loading: {}'.format(self.im_list[k]))
                if self.name == 'celeba':
                    img = preprocess(img)
                if self.color_space == 'YUV':
                    img = RGB2YUV(img)
                img = img / 255.0
                #print('img shape: {}'.format(np.shape(img)))
                mini_batches.append(np.reshape(img,(1,np.shape(img)[0] * np.shape(img)[1] * np.shape(img)[2])))
            
            mini_batches = np.concatenate(mini_batches, axis=0)
            self.count = self.count + 1
                    
            return mini_batches
