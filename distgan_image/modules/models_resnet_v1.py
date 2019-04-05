"""WGAN-GP ResNet for CIFAR-10"""

import os, sys
sys.path.append(os.getcwd())

import modules.tflib_ops as lib
import modules.tflib_ops.linear
import modules.tflib_ops.cond_batchnorm
import modules.tflib_ops.conv2d
import modules.tflib_ops.deconv2d
import modules.tflib_ops.batchnorm
import modules.tflib_ops.layernorm

import numpy as np
import tensorflow as tf

from functools import partial
import functools

def leak_relu(x, leak, scope=None):
    with tf.name_scope(scope, 'leak_relu', [x, leak]):
        if leak < 1:
            y = tf.maximum(x, leak * x)
        else:
            y = tf.minimum(x, leak * x)
        return y

lrelu = partial(leak_relu, leak=0.2) # 0.2 as in GAAN, 0.1 as in SNGAN

def nonlinearity(name, x):
    if 'encoder' in name:
       return lrelu(x)
    else:
		return tf.nn.relu(x)

def Normalize(name, inputs, num_classes = None, labels = None):

    if ('generator' in name or 'encoder' in name):
        if labels is not None:
            labels = tf.squeeze(labels)
            return lib.cond_batchnorm.Batchnorm(name,[0,2,3],inputs,labels=labels,n_labels=num_classes)
        else:
            return lib.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)
    else:
        return inputs

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output   
    
def ResidualBlock1(name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False, num_classes = None, labels = None):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_1        = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim, stride=2)
        conv_shortcut = ConvMeanPool
    elif resample=='up':
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2        = functools.partial(lib.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.conv2d.Conv2D
        conv_1        = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.N1', output, num_classes=num_classes, labels=labels)
    output = nonlinearity(name, output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output)    
    output = Normalize(name+'.N2', output, num_classes=num_classes, labels=labels)
    output = nonlinearity(name, output)            
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output)

    return shortcut + output 

   
def OptimizedResBlockDisc1(inputs, dim):
    conv_1      = functools.partial(lib.conv2d.Conv2D, input_dim=3, output_dim=dim)
    conv_2      = functools.partial(lib.conv2d.Conv2D, input_dim=dim, output_dim=dim, stride=2)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut('discriminator.1.Shortcut', input_dim=3, output_dim=dim, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = conv_1('discriminator.1.Conv1', filter_size=3, inputs=output)    
    output = nonlinearity('discriminator', output)            
    output = conv_2('discriminator.1.Conv2', filter_size=3, inputs=output)
    return shortcut + output
    
def encoder_resnet_cifar(inputs, x_shape, z_dim=128, dim=128, num_classes = None, labels = None, name = 'encoder', reuse=False):
    #transform to NCHW to use tflib
    dim = dim * 2 #256
    output = tf.transpose(tf.reshape(inputs, [-1, x_shape[0], x_shape[1], x_shape[2]]), perm=[0, 3, 1, 2])
    output = lib.conv2d.Conv2D('encoder.Input', 3, dim, 3, output, he_init=False) 										 #32x32
    output = ResidualBlock1('encoder.1', dim, dim, 3, output, resample='down', num_classes=num_classes, labels=labels)   #16x16
    output = ResidualBlock1('encoder.2', dim, dim, 3, output, resample='down', num_classes=num_classes, labels=labels)   #8x8
    output = ResidualBlock1('encoder.3', dim, dim, 3, output, resample='down', num_classes=num_classes, labels=labels)   #4x4
    output = Normalize('encoder.OutputN', output, num_classes=num_classes, labels=labels)
    output = nonlinearity('encoder', output)
    output = tf.reshape(output, [-1, dim * 4 * 4])
    output = lib.linear.Linear('encoder.Output', dim * 4 * 4, z_dim, output)
    return output

def encoder_resnet_stl10(inputs, x_shape, z_dim=128, dim=128, num_classes = None, labels = None, name = 'encoder', reuse=False):
    #transform to NCHW to use tflib
    output = tf.transpose(tf.reshape(inputs, [-1, x_shape[0], x_shape[1], x_shape[2]]), perm=[0, 3, 1, 2])
    output = lib.conv2d.Conv2D('encoder.Input', 3, dim, 3, output, he_init=False)  #32x32
    output = ResidualBlock1('encoder.1', dim, dim*2, 3, output, resample='down', num_classes=num_classes, labels=labels)   #16x16
    output = ResidualBlock1('encoder.2', dim*2, dim*4, 3, output, resample='down', num_classes=num_classes, labels=labels) #8x8
    output = ResidualBlock1('encoder.3', dim*4, dim*8, 3, output, resample='down', num_classes=num_classes, labels=labels) #4x4
    output = Normalize('encoder.OutputN', output, num_classes=num_classes, labels=labels)
    output = nonlinearity('encoder', output)
    output = tf.reshape(output, [-1, dim*8 * 6 * 6])
    output = lib.linear.Linear('encoder.Output', dim*8 * 6 * 6, z_dim, output)
    return output
    
def generator_resnet_cifar(noise, x_shape, dim=128, num_classes = None, labels = None, name = 'generator', reuse=False):
    dim = dim * 2 #256
    x_dim = x_shape[0] * x_shape[1] * x_shape[2]
    output = lib.linear.Linear('generator.Input', 128, 4*4*dim, noise)
    output = tf.reshape(output, [-1, dim, 4, 4])                                 								         #4x4x256
    output = ResidualBlock1('generator.1', dim, dim, 3, output, resample='up', num_classes=num_classes, labels=labels)   #8x8x256
    output = ResidualBlock1('generator.2', dim, dim, 3, output, resample='up', num_classes=num_classes, labels=labels)   #16x16x256
    output = ResidualBlock1('generator.3', dim, dim, 3, output, resample='up', num_classes=num_classes, labels=labels)   #32x32x256
    output = Normalize('generator.OutputN', output, num_classes=num_classes, labels=labels)
    output = nonlinearity('generator', output)
    output = lib.conv2d.Conv2D('generator.Output', dim, 3, 3, output, he_init=False)									 #32x32x3
    #transform to NHWC
    output = tf.transpose(tf.reshape(output, [-1, x_shape[2], x_shape[0], x_shape[1]]), perm=[0, 2, 3, 1])
    output = tf.sigmoid(output)
    return tf.reshape(output, [-1, x_dim])

def generator_resnet_stl10(noise, x_shape, dim=64, num_classes = None, labels = None, name = 'generator', reuse=False):
    x_dim = x_shape[0] * x_shape[1] * x_shape[2]
    output = lib.linear.Linear('generator.Input', 128, 6*6*dim*8, noise)
    output = tf.reshape(output, [-1, dim*8, 6, 6])                                  										#6x6x512
    output = ResidualBlock1('generator.1', dim*8, dim*4, 3, output, resample='up', num_classes=num_classes, labels=labels)  #12x12x256
    output = ResidualBlock1('generator.2', dim*4, dim*2, 3, output, resample='up', num_classes=num_classes, labels=labels)  #24x24x128
    output = ResidualBlock1('generator.3', dim*2, dim, 3, output,   resample='up', num_classes=num_classes, labels=labels)  #48x48x64
    output = Normalize('generator.OutputN', output, num_classes=num_classes, labels=labels)
    output = nonlinearity('generator', output)
    output = lib.conv2d.Conv2D('generator.Output', dim, 3, 3, output, he_init=False)										#48x48x3
    #transform to NHWC
    output = tf.transpose(tf.reshape(output, [-1, x_shape[2], x_shape[0], x_shape[1]]), perm=[0, 2, 3, 1])
    output = tf.sigmoid(output)
    return tf.reshape(output, [-1, x_dim])
    
def discriminator_resnet_cifar(inputs, x_shape, dim=64, num_classes = None, labels = None, name = 'discriminator', reuse=False):
    #transform to NCHW to use tflib
    output = tf.transpose(tf.reshape(inputs, [-1, x_shape[0], x_shape[1], x_shape[2]]), perm=[0, 3, 1, 2])
    output = OptimizedResBlockDisc1(output, dim)
    output = ResidualBlock1('discriminator.2', dim, dim, 3, output, resample='down', num_classes=num_classes, labels=labels)
    output = ResidualBlock1('discriminator.3', dim, dim, 3, output, resample=None, num_classes=num_classes, labels=labels)
    output = ResidualBlock1('discriminator.4', dim, dim, 3, output, resample=None, num_classes=num_classes, labels=labels)
    output = nonlinearity('discriminator', output)
    feature = output #save feature before nonlinearity layer
    output = tf.reduce_mean(output, axis=[2,3])
    y = lib.linear.Linear('discriminator.Output', dim, 1, output)
    output_wgan = tf.reshape(y, [-1])
    return tf.sigmoid(output_wgan), output_wgan, tf.reshape(feature, [-1, 512 * 4 * 4])

def discriminator_resnet_stl10(inputs, x_shape, dim=64, num_classes = None, labels = None, name = 'discriminator', reuse=False):
    #transform to NCHW to use tflib
    output = tf.transpose(tf.reshape(inputs, [-1, x_shape[0], x_shape[1], x_shape[2]]), perm=[0, 3, 1, 2])
    output = OptimizedResBlockDisc1(output, dim)
    output = ResidualBlock1('discriminator.2', dim, dim*2, 3, output, resample='down', num_classes=num_classes, labels=labels)
    output = ResidualBlock1('discriminator.3', dim*2, dim*4, 3, output, resample='down', num_classes=num_classes, labels=labels)
    output = ResidualBlock1('discriminator.4', dim*4, dim*8, 3, output, resample='down', num_classes=num_classes, labels=labels)
    output = ResidualBlock1('discriminator.5', dim*8, dim*16, 3, output, resample=None, num_classes=num_classes, labels=labels)
    output = nonlinearity('discriminator', output)
    feature = output
    output = tf.reduce_mean(output, axis=[2,3])
    output = lib.linear.Linear('discriminator.Output', dim*16, 1, output)
    output = tf.reshape(output, [-1])
    print('feature shape',feature.get_shape().as_list())
    return tf.sigmoid(output), output, tf.reshape(feature, [-1, 1024*3*3])
