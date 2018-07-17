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
The encoder/generator/discriminator for MNIST (smaller network for 28x28 images)
'''

def encoder_mnist(img, x_shape, z_dim=128, dim=64, kernel_size=5, stride=2, name = 'encoder', reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_relu = partial(conv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    
    print("encoder---")
    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
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

def generator_mnist(z, x_shape, dim=64, kernel_size=5, stride=2, name = 'generator', reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    x_dim = x_shape[0] * x_shape[1] * x_shape[2]  
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
        y = dconv(y, x_shape[2], kernel_size, stride)
        print(y.get_shape())
        y = tf.reshape(y, [-1, x_dim])
        return tf.sigmoid(y)
               
def discriminator_mnist(img, x_shape, dim=64, kernel_size=5, stride=2, name='discriminator', reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    print("discriminator---")
    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
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


'''
Common encoder/generator/discriminator for cifar-10 (32x32)
'''
def encoder_cifar(img, x_shape, z_dim=128, dim=64, kernel_size=5, stride=2, name = 'encoder', reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(y, dim, kernel_size, stride))
        y = conv_bn_lrelu(y, dim * 2, kernel_size, stride)
        y = conv_bn_lrelu(y, dim * 4, kernel_size, stride)
        y = conv_bn_lrelu(y, dim * 8, kernel_size, stride)
        logit = fc(y, z_dim)
        return logit

def generator_cifar(z, x_shape, dim=64, kernel_size=5, stride=2, name = 'generator', reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    x_dim = x_shape[0] * x_shape[1] * x_shape[2]
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 2 * 2 * dim * 8)
        y = tf.reshape(y, [-1, 2, 2, dim * 8])
        y = dconv_bn_relu(y, dim * 4, kernel_size, stride)
        y = dconv_bn_relu(y, dim * 2, kernel_size, stride)
        y = dconv_bn_relu(y, dim * 1, kernel_size, stride)
        y = dconv(y, x_shape[2], kernel_size, stride)
        y = tf.reshape(y, [-1, x_dim])
        return tf.sigmoid(y)
               
def discriminator_cifar(img, x_shape, dim=64, kernel_size=5, stride=2, name='discriminator', reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(y, dim, kernel_size, 2))
        y = conv_bn_lrelu(y, dim * 2, kernel_size, stride)
        y = conv_bn_lrelu(y, dim * 4, kernel_size, stride)
        y = conv_bn_lrelu(y, dim * 8, kernel_size, stride)
        feature = y
        logit = fc(y, 1)
        return tf.nn.sigmoid(logit), logit, tf.reshape(feature,[tf.shape(img)[0], -1])


'''
Common encoder/generator/discriminator for celeba (64x64)
'''
def encoder_celeba(img, x_shape, z_dim=128, dim=64, kernel_size=5, stride=2, name = 'encoder', reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(y, dim, kernel_size, stride))        #[32 x 32 x dim]
        y = conv_bn_lrelu(y, dim * 2, kernel_size, stride)  #[16 x 16 x 2 x dim]
        y = conv_bn_lrelu(y, dim * 4, kernel_size, stride)  #[8 x 8 x 4 x dim]
        y = conv_bn_lrelu(y, dim * 8, kernel_size, stride)  #[4 x 4 x 8 x dim]
        logit = fc(y, z_dim)                                #[z_dim]
        return logit

def generator_celeba(z, x_shape, dim=64, kernel_size=5, stride=2, name = 'generator', reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    x_dim = x_shape[0] * x_shape[1] * x_shape[2]
    with tf.variable_scope(name, reuse=reuse):
        y = fc_bn_relu(z, 4 * 4 * dim * 8)
        y = tf.reshape(y, [-1, 2, 2, dim * 8])
        y = dconv_bn_relu(y, dim * 4, kernel_size, stride)
        y = dconv_bn_relu(y, dim * 2, kernel_size, stride)
        y = dconv_bn_relu(y, dim * 1, kernel_size, stride)
        y = dconv(y, x_shape[2], kernel_size, stride)
        y = tf.reshape(y, [-1, x_dim])
        return tf.sigmoid(y)
               
def discriminator_celeba(img, x_shape, dim=64, kernel_size=5, stride=2, name='discriminator', reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    y = tf.reshape(img,[-1, x_shape[0], x_shape[1], x_shape[2]])
    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(conv(y, dim, kernel_size, 2))
        y = conv_bn_lrelu(y, dim * 2, kernel_size, stride)
        y = conv_bn_lrelu(y, dim * 4, kernel_size, stride)
        y = conv_bn_lrelu(y, dim * 8, kernel_size, stride)
        feature = y
        logit = fc(y, 1)
        return tf.nn.sigmoid(logit), logit, tf.reshape(feature,[tf.shape(img)[0], -1])
