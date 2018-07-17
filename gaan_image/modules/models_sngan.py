from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from   modules import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim
from   functools import partial

conv    = partial(slim.conv2d, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
dconv = partial(slim.conv2d_transpose, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
fc = partial(ops.flatten_fully_connected, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
relu = tf.nn.relu
lrelu = partial(ops.leak_relu, leak=0.1) # 0.2 as in GAAN, 0.1 as in SNGAN
batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)

# ======================================================================
# Common encoder/generator/discriminator architecture for CIFAR-10 (32x32)
# like SN-GAN.
# ======================================================================
def encoder_sngan_cifar(img, x_shape, z_dim=128, dim=64, kernel_size=5, stride=2, name = 'encoder', reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)
    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    with tf.variable_scope(name, reuse=reuse):
        y = relu(conv(y, 64 * 2, 3, 1))
        y = conv_bn_lrelu(y, 128 * 2, 4, 2)
        y = conv_bn_lrelu(y, 256 * 2, 4, 2)
        y = conv_bn_lrelu(y, 512 * 2, 4, 2)
        logit = fc(y, z_dim)
        return logit

def generator_sngan_cifar(z, x_shape, dim=64, kernel_size=5, stride=2, name = 'generator', reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)   
    x_dim = x_shape[0] * x_shape[1] * x_shape[2]
    z_dim = z.get_shape()[1].value
    assert(z_dim == 128)
    with tf.variable_scope(name, reuse=reuse):
        y = relu(fc(z, 4 * 4 * 512 * 2))
        y = tf.reshape(y, [-1, 4, 4, 512 * 2])
        y = dconv_bn_relu(y, 256 * 2, 4, 2)
        y = dconv_bn_relu(y, 128 * 2, 4, 2)
        y = dconv_bn_relu(y, 64 * 2, 4, 2)
        y = conv(y, x_shape[2], 3, 1)
        y = tf.reshape(y, [-1, x_dim])
        return tf.sigmoid(y)

def discriminator_sngan_cifar(img, x_shape, dim=64, kernel_size=5, stride=2, name='discriminator', reuse=True, training=True):        
    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    assert(x_shape[0]==32) # check input size
    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(y, 64, 3, 1))
        y = lrelu(conv(y, 64, 4, 2))
        y = lrelu(conv(y, 128, 3, 1))
        y = lrelu(conv(y, 128, 4, 2))
        y = lrelu(conv(y, 256, 3, 1))
        y = lrelu(conv(y, 256, 4, 2))
        y = lrelu(conv(y, 512, 3, 1))
        feature = y
        logit = fc(y, 1)
        return tf.nn.sigmoid(logit), logit, tf.reshape(feature,[tf.shape(img)[0], -1])
                
# ======================================================================
# Common encoder/generator/discriminator architecture for STL-10 (48x48)
# like SN-GAN.
# ======================================================================
             
def encoder_sngan_stl10(img, x_shape, z_dim=128, dim=64, kernel_size=5, stride=2, name = 'encoder', reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)   
    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    with tf.variable_scope(name, reuse=reuse):
        y = relu(conv(y, 64 * 2, 3, 1))
        y = conv_bn_lrelu(y, 128 * 2, 4, 2)
        y = conv_bn_lrelu(y, 256 * 2, 4, 2)
        y = conv_bn_lrelu(y, 512 * 2, 4, 2)
        logit = fc(y, z_dim)
        return logit

def generator_sngan_stl10(z, x_shape, dim=64, kernel_size=5, stride=2, name = 'generator', reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)   
    x_dim = x_shape[0] * x_shape[1] * x_shape[2]
    z_dim = z.get_shape()[1].value
    assert(z_dim == 128)
    with tf.variable_scope(name, reuse=reuse):
        y = relu(fc(z, 6 * 6 * 512 * 2))
        y = tf.reshape(y, [-1, 6, 6, 512 * 2])
        y = dconv_bn_relu(y, 256 * 2, 4, 2)
        y = dconv_bn_relu(y, 128 * 2, 4, 2)
        y = dconv_bn_relu(y, 64 * 2, 4, 2)
        y = conv(y, x_shape[2], 3, 1)
        y = tf.reshape(y, [-1, x_dim])
        return tf.sigmoid(y)
        
def discriminator_sngan_stl10(img, x_shape, dim=64, kernel_size=5, stride=2, name='discriminator', reuse=True, training=True):        
    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    assert(x_shape[0]==48) # check input size
    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(y, 64, 3, 1))
        y = lrelu(conv(y, 64, 4, 2))
        y = lrelu(conv(y, 128, 3, 1))
        y = lrelu(conv(y, 128, 4, 2))
        y = lrelu(conv(y, 256, 3, 1))
        y = lrelu(conv(y, 256, 4, 2))
        y = lrelu(conv(y, 512, 3, 1))
        feature = y
        logit = fc(y, 1)
        return tf.nn.sigmoid(logit), logit, tf.reshape(feature,[tf.shape(img)[0], -1])
