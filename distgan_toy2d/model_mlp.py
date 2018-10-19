import numpy as np 
import tensorflow as tf

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)
    
def theta_def(dim):
    theta = []
    for i in range(len(dim) - 1):
        D_W = tf.Variable(xavier_init([dim[i], dim[i + 1]]))
        D_b = tf.Variable(tf.zeros(shape=[dim[i + 1]]))
        theta.extend((D_W, D_b))
    return theta

def mlp(input, theta):
    h = tf.nn.relu(tf.matmul(input, theta[0]) + theta[1])
    if (len(theta) > 2):
        for i in range(1, int(len(theta) / 2) - 1):
            h = tf.nn.relu(tf.matmul(h, theta[2 * i]) + theta[2 * i + 1])
        logit = tf.matmul(h, theta[-2]) + theta[-1]
        prob = tf.nn.sigmoid(logit)
        return prob, logit
    elif len(theta) == 2:
        logit = h
        prob = tf.nn.sigmoid(logit)
        return prob, logit
    else:
        print("Wrong Input")   
        
def mlp_feat(input, theta):
    h = tf.nn.relu(tf.matmul(input, theta[0]) + theta[1])
    if (len(theta) > 2):
        for i in range(1, int(len(theta) / 2) - 2):
            h = tf.nn.relu(tf.matmul(h, theta[2 * i]) + theta[2 * i + 1])
        f = tf.matmul(h, theta[-4]) + theta[-3]
        h = tf.nn.relu(f)
        logit = tf.matmul(h, theta[-2]) + theta[-1]
        prob = tf.nn.sigmoid(logit)
        return prob, logit, f
    elif len(theta) == 2:
        logit = h
        prob = tf.nn.sigmoid(logit)
        return prob, logit, []
    else:
        print("Wrong Input")         
