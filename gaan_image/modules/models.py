from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from modules import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim

from functools import partial

conv    = partial(slim.conv2d, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
dconv = partial(slim.conv2d_transpose, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
fc = partial(ops.flatten_fully_connected, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
relu = tf.nn.relu
lrelu = partial(ops.leak_relu, leak=0.2)
ln = slim.layer_norm
max_pooling = partial(slim.max_pool2d, kernel_size=2, stride=2, padding='VALID')
batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)

'''
Common encoder/generator/discriminator
'''
def encoder(img, z_dim=128, dim=64, kernel_size=5, stride=2, name = 'encoder', reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(img, dim, kernel_size, stride))
        y = conv_bn_lrelu(y, dim * 2, kernel_size, stride)
        y = conv_bn_lrelu(y, dim * 4, kernel_size, stride)
        y = conv_bn_lrelu(y, dim * 8, kernel_size, stride)
        logit = fc(y, z_dim)
        return logit

def generator(z, dim=64, kernel_size=5, stride=2, name = 'generator', reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 4 * 4 * dim * 8)
        y = tf.reshape(y, [-1, 4, 4, dim * 8])
        y = dconv_bn_relu(y, dim * 4, kernel_size, stride)
        y = dconv_bn_relu(y, dim * 2, kernel_size, stride)
        y = dconv_bn_relu(y, dim * 1, kernel_size, stride)
        y = dconv(y, 3, 5, stride)
        return tf.tanh(y)
               
def discriminator(img, dim=64, kernel_size=5, stride=2, name='discriminator', reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope(dname, reuse=reuse):
        y = lrelu(conv(img, dim, kernel_size, 2))
        y = conv_bn_lrelu(y, dim * 2, kernel_size, stride)
        y = conv_bn_lrelu(y, dim * 4, kernel_size, stride)
        feature = y
        y = conv_bn_lrelu(y, dim * 8, kernel_size, stride)
        logit = fc(y, 1)
        return tf.nn.sigmoid(logit), logit, tf.reshape(feature,[tf.shape(img)[0], -1])

'''
The encoder/generator/discriminator for MNIST (smaller network for 32x32 images)
'''

def encoder_mnist(img, z_dim=128, dim=64, kernel_size=5, stride=2, name = 'encoder', reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_relu = partial(conv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    
    print("encoder---")
    y = tf.reshape(img,[-1, 28, 28, 1])
    print(y.get_shape())
    with tf.variable_scope(name, reuse=reuse):
        y = relu(conv(y, dim, kernel_size, stride))
        print(y.get_shape())
        y = conv_bn_relu(y, dim * 2, kernel_size, stride)
        print(y.get_shape())
        y = conv_bn_relu(y, dim * 4, kernel_size, stride)
        print(y.get_shape())
        logit = fc(y, z_dim)
        print(logit.get_shape())
        return logit

def generator_mnist(z, x_dim, dim=64, kernel_size=5, stride=2, name = 'generator', reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    print("generator---")
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 4 * 4 * dim * 4)
        y = tf.reshape(y, [-1, 4, 4, dim * 4])
        print(y.get_shape())
        y = dconv_bn_relu(y, dim * 2, kernel_size, stride)
        print(y.get_shape())
        y = tf.reshape(y, [-1, 8 * 8 * 2 * dim]) # process the feature map 8x8 compare to 7x7 of mirror layer of encoder
        y = relu(fc(y, 7 * 7 * 2 * dim))
        y = tf.reshape(y, [-1, 7, 7, 2 * dim])
        print(y.get_shape())
        y = dconv_bn_relu(y, dim * 1, kernel_size, stride)
        print(y.get_shape())
        y = dconv(y, 1, kernel_size, stride)
        print(y.get_shape())
        y = tf.reshape(y, [-1, x_dim])
        return tf.sigmoid(y)
               
def discriminator_mnist(img, dim=64, kernel_size=5, stride=2, name='discriminator', reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    print("discriminator---")
    y = tf.reshape(img,[-1, 28, 28, 1])
    print(y.get_shape())
    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(y, dim, kernel_size, stride))
        print(y.get_shape())
        y = conv_bn_lrelu(y, dim * 2, kernel_size, stride)
        print(y.get_shape())
        y = conv_bn_lrelu(y, dim * 4, kernel_size, stride)
        print(y.get_shape())
        feature = y
        logit = fc(y, 1)
        print(logit.get_shape())
        return tf.nn.sigmoid(logit), logit, tf.reshape(feature,[tf.shape(img)[0], -1])
