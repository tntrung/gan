"""Modified from SAGAN: https://github.com/brain-research/self-attention-gan"""

import numpy as np
import tensorflow as tf

from functools import partial
import functools
from sagan_ops import ops

'''
========================================================================
Utils functions
========================================================================
'''
def leak_relu(x, leak, scope=None):
    with tf.name_scope(scope, 'leak_relu', [x, leak]):
        if leak < 1:
            y = tf.maximum(x, leak * x)
        else:
            y = tf.minimum(x, leak * x)
        return y

lrelu = partial(leak_relu, leak=0.2)

def upscale(x, n):
  """Builds box upscaling (also called nearest neighbors).

  Args:
    x: 4D image tensor in B01C format.
    n: integer scale (must be a power of 2).

  Returns:
    4D tensor of images up scaled by a factor n.
  """
  if n == 1:
    return x
  return tf.batch_to_space(tf.tile(x, [n**2, 1, 1, 1]), [[0, 0], [0, 0]], n)


def usample_tpu(x):
  """Upscales the width and height of the input vector by a factor of 2."""
  x = upscale(x, 2)
  return x

def usample(x):
  _, nh, nw, nx = x.get_shape().as_list()
  x = tf.image.resize_nearest_neighbor(x, [nh * 2, nw * 2])
  return x

def dsample_pool(x, name = 'dsample'): #original of SAGAN
  """Downsamples the image by a factor of 2."""
  xd = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
  return xd

def dsample_conv(x, name = 'dsample'):
  """Downsamples the image by a factor of 2."""
  xd = ops.conv2d(x, x.get_shape().as_list()[-1], 1, 1, 2, 2, name=name)
  return xd

def hw_flatten(x):
    return tf.reshape(x, shape=[tf.shape(x)[0], -1, tf.shape(x)[-1]])
        
'''
========================================================================
Resnet blocks
========================================================================
'''
def g_block(x, out_channels, is_training, name):
  """Builds the residual blocks used in the generator.

  Compared with block, optimized_block always downsamples the spatial
  resolution of the input vector by a factor of 4.

  Args:
    x: The 4D input vector.
    out_channels: Number of features in the output layer.
    name: The variable scope name for the block.
  Returns:
    A `Tensor` representing the output of the operation.
  """
  with tf.variable_scope(name):
    
    bn0 = ops.batch_norm(name='bn0')
    bn1 = ops.batch_norm(name='bn1')
    
    x_0 = x
    x = tf.nn.relu(bn0(x, train = is_training))
    x = usample(x)
    x = ops.conv2d(x, out_channels, 3, 3, 1, 1, name='conv1')
    x = tf.nn.relu(bn1(x, train = is_training))
    x = ops.conv2d(x, out_channels, 3, 3, 1, 1, name='conv2')

    x_0 = usample(x_0)
    x_0 = ops.conv2d(x_0, out_channels, 1, 1, 1, 1, name='conv3')

    return x_0 + x

def e_block(x, out_channels, is_training, name, downsample=True, \
                                                        act=tf.nn.relu):
  """Builds the residual blocks used in the discriminator in SNGAN.

  Args:
    x: The 4D input vector.
    out_channels: Number of features in the output layer.
    name: The variable scope name for the block.
    update_collection: The update collections used in the
                       spectral_normed_weight.
    downsample: If True, downsample the spatial size the input tensor by
                a factor of 4. If False, the spatial size of the input 
                tensor is unchanged.
    act: The activation function used in the block.
  Returns:
    A `Tensor` representing the output of the operation.
  """
  with tf.variable_scope(name):
      
    bn0 = ops.batch_norm(name='bn0')
    bn1 = ops.batch_norm(name='bn1')
    
    input_channels = x.get_shape().as_list()[-1]
    x_0 = x
    x = act(bn0(x, train = is_training))
    x = ops.conv2d(x, out_channels, 3, 3, 1, 1, name='conv1')
    x = act(bn1(x, train = is_training))
    x = ops.conv2d(x, out_channels, 3, 3, 1, 1, name='conv2')
    if downsample:
      x = dsample_pool(x, "e_dsample_1")
    if downsample or input_channels != out_channels:
      x_0 = ops.conv2d(x_0, out_channels, 1, 1, 1, 1, name='conv3')
      if downsample:
        x_0 = dsample_pool(x_0, "e_dsample_2")
    return x_0 + x
    
def g_block_cond(x, out_channels, num_classes, labels, is_training, name):
  """Builds the residual blocks used in the generator.

  Compared with block, optimized_block always downsamples the spatial 
  resolution of the input vector by a factor of 4.

  Args:
    x: The 4D input vector.
    labels: The conditional labels in the generation.
    out_channels: Number of features in the output layer.
    num_classes: Number of classes in the labels.
    name: The variable scope name for the block.
  Returns:
    A `Tensor` representing the output of the operation.
  """
  with tf.variable_scope(name):
    bn0 = ops.ConditionalBatchNorm(num_classes, name='cbn0')
    bn1 = ops.ConditionalBatchNorm(num_classes, name='cbn1')
    x_0 = x
    x = tf.nn.relu(bn0(x, labels, is_training))
    x = usample(x)
    x = ops.conv2d(x, out_channels, 3, 3, 1, 1, name='conv1')
    x = tf.nn.relu(bn1(x, labels, is_training))
    x = ops.conv2d(x, out_channels, 3, 3, 1, 1, name='conv2')

    x_0 = usample(x_0)
    x_0 = ops.conv2d(x_0, out_channels, 1, 1, 1, 1, name='conv3')

    return x_0 + x


def e_block_cond(x, out_channels, num_classes, labels, \
                    is_training, name, downsample=True, act=tf.nn.relu):
  """Builds the residual blocks used in the discriminator in SNGAN.

  Args:
    x: The 4D input vector.
    out_channels: Number of features in the output layer.
    name: The variable scope name for the block.
    update_collection: The update collections used in the
                       spectral_normed_weight.
    downsample: If True, downsample the spatial size the input tensor by
                a factor of 4. If False, the spatial size of the input 
                tensor is unchanged.
    act: The activation function used in the block.
  Returns:
    A `Tensor` representing the output of the operation.
  """
  with tf.variable_scope(name):
      
    bn0 = ops.ConditionalBatchNorm(num_classes, name='cbn0')
    bn1 = ops.ConditionalBatchNorm(num_classes, name='cbn1')
    
    input_channels = x.get_shape().as_list()[-1]
    x_0 = x
    x = act(bn0(x, labels, is_training = is_training))
    x = ops.conv2d(x, out_channels, 3, 3, 1, 1, name='conv1')
    x = act(bn1(x, labels, is_training = is_training))
    x = ops.conv2d(x, out_channels, 3, 3, 1, 1, name='conv2')
    if downsample:
      x = dsample_pool(x, "e_dsample_1")
    if downsample or input_channels != out_channels:
      x_0 = ops.conv2d(x_0, out_channels, 1, 1, 1, 1, name='conv3')
      if downsample:
        x_0 = dsample_pool(x_0, "e_dsample_2")
    return x_0 + x
        
def d_block(x, out_channels, name, downsample=True, act=tf.nn.relu):
  """Builds the residual blocks used in the discriminator in SNGAN.

  Args:
    x: The 4D input vector.
    out_channels: Number of features in the output layer.
    name: The variable scope name for the block.
    update_collection: The update collections used in the
                       spectral_normed_weight.
    downsample: If True, downsample the spatial size the input tensor by
                a factor of 4. If False, the spatial size of the input 
                tensor is unchanged.
    act: The activation function used in the block.
  Returns:
    A `Tensor` representing the output of the operation.
  """
  with tf.variable_scope(name):
    input_channels = x.get_shape().as_list()[-1]
    x_0 = x
    x = act(x)
    x = ops.conv2d(x, out_channels, 3, 3, 1, 1, name='conv1')
    x = act(x)
    x = ops.conv2d(x, out_channels, 3, 3, 1, 1, name='conv2')
    if downsample:
      x = dsample_conv(x, "d_dsample_1")
    if downsample or input_channels != out_channels:
      x_0 = ops.conv2d(x_0, out_channels, 1, 1, 1, 1, name='conv3')
      if downsample:
        x_0 = dsample_conv(x_0, "d_dsample_2")
    return x_0 + x


def optimized_block(x, out_channels, name, act=tf.nn.relu):
  """Builds the simplified residual blocks for downsampling.

  Compared with block, optimized_block always downsamples the spatial
  resolution of the input vector by a factor of 4.

  Args:
    x: The 4D input vector.
    out_channels: Number of features in the output layer.
    name: The variable scope name for the block.
    update_collection: The update collections used in the
                       spectral_normed_weight.
    act: The activation function used in the block.
  Returns:
    A `Tensor` representing the output of the operation.
  """
  with tf.variable_scope(name):
    x_0 = x
    x = ops.conv2d(x, out_channels, 3, 3, 1, 1, name='conv1')
    x = act(x)
    x = ops.conv2d(x, out_channels, 3, 3, 1, 1, name='conv2')
    x = dsample_conv(x, "o_dsample_1")
    x_0 = dsample_conv(x_0, "o_dsample_2")
    x_0 = ops.conv2d(x_0, out_channels, 1, 1, 1, 1, name='conv3')
    return x + x_0
      
'''
========================================================================
ResNet for CIFAR-10 (32x32)
========================================================================
'''

def encoder_resnet_cifar(x, x_shape, z_dim=128, dim=128, \
                         num_classes = None, labels = None, \
                         name = 'encoder', \
                         update_collection=None, \
                                         reuse=False, is_training=True):
                                             
    global count_reuse
                                             
    if labels is not None:
        labels = tf.squeeze(labels)
    dim = dim * 2 # 256 like sn-gan paper
    act = lrelu
    is_conditional = num_classes is not None and labels is not None
    with tf.variable_scope(name, reuse=reuse):
        image = tf.reshape(x, [-1, x_shape[0], x_shape[1], x_shape[2]])
        image = ops.conv2d(image, dim, 3, 3, 1, 1, \
                                               name='e_conv0') # 32 * 32
        if is_conditional:
            act0  = e_block_cond(image, dim,\
                                    num_classes = num_classes, \
                                    labels = labels,\
                                    is_training = is_training,\
                                    name = 'e_block1', act=act)# 16 * 16            
        else:
            act0  = e_block(image, dim, is_training = is_training,\
                                    name = 'e_block1', act=act)# 16 * 16
                                    
        if is_conditional:
            act1 = e_block_cond(act0, dim, \
                                      num_classes, labels,\
                                      is_training = is_training,\
                                      name = 'e_block2', act=act)# 8 * 8            
        else:
            act1 = e_block(act0, dim, is_training, \
                                      name = 'e_block2', act=act)# 8 * 8
                                     
        if is_conditional:
            act2 = e_block_cond(act1, dim, \
                                 num_classes, labels, \
                                 is_training = is_training,\
                                 name =  'e_block3', act=act)    # 4 * 4            
        else:                              
            act2 = e_block(act1, dim, is_training, \
                                 name =  'e_block3', act=act)    # 4 * 4
                                 
        if is_conditional:
            bn   = ops.batch_norm(num_classes, name='e_bn')
        else:                     
            bn   = ops.batch_norm(name='e_bn')
        act2 = act(bn(act2, is_training))                 
        act2 = tf.reshape(act2, [-1, 4 * 4 * dim])                      
        out  = ops.linear(act2, z_dim)
        return out

def generator_resnet_cifar(z, x_shape, dim=128, \
                           num_classes = None, labels = None, \
                           name = 'generator', reuse=False, \
                           is_training=True):

    if labels is not None:
        labels = tf.squeeze(labels)
    dim = dim * 2 # 256 like sn-gan paper
    x_dim = x_shape[0] * x_shape[1] * x_shape[2]
    is_conditional = num_classes is not None and labels is not None
    with tf.variable_scope(name, reuse=reuse):
        act0 = ops.linear(z, dim * 4 * 4, scope='g_linear0')
        act0 = tf.reshape(act0, [-1, 4, 4, dim])
        if is_conditional:
            act1 = g_block_cond(act0, dim, num_classes, labels, \
                                       is_training, 'g_block1')  # 8 * 8
        else:
            act1 = g_block(act0, dim, is_training, 'g_block1')   # 8 * 8
            
        if is_conditional:
            act2 = g_block_cond(act1, dim, num_classes, labels, \
                                      is_training, 'g_block2') # 16 * 16
        else:   
            act2 = g_block(act1, dim, is_training, 'g_block2') # 16 * 16
        if is_conditional:
            act3 = g_block_cond(act2, dim, num_classes, labels, \
                                      is_training, 'g_block3') # 32 * 32
        else:
            act3 = g_block(act2, dim, is_training, 'g_block3') # 32 * 32
        if is_conditional:
            bn   = ops.batch_norm(num_classes, name='g_bn')
        else:
            bn   = ops.batch_norm(name='g_bn')
        act3 = tf.nn.relu(bn(act3, is_training))
        act4 = ops.conv2d(act3, 3, 3, 3, 1, 1, name='g_conv_last')
        out  = tf.nn.sigmoid(act4)
        return tf.reshape(out, [-1, x_dim])
        

def discriminator_resnet_cifar(x, x_shape, dim=128, \
                            num_classes = None, labels = None, \
                            name = 'discriminator',\
                            update_collection=None, reuse=False):

  """Builds the discriminator graph.

  Args:
    x: The current batch of images to classify as fake or real.
    x_shape: the shape [h x w x c] of image
    dim: The d dimension.
    update_collection: The update collections used in the
                       spectral_normed_weight.
  Returns:
    A `Tensor` representing the logits of the discriminator.
  """
  if labels is not None:
     labels = tf.squeeze(labels)
  relu=tf.nn.relu
  is_conditional = num_classes is not None and labels is not None
  with tf.variable_scope(name, reuse=reuse):
    
    image = tf.reshape(x, [-1, x_shape[0], x_shape[1], x_shape[2]])
    h0 = optimized_block(image, dim, 'd_optimized_block1', \
                                             act=relu) # 16 * 16
    h1 = d_block(h0, dim, 'd_block2', act=relu)        # 8 * 8
    h2 = d_block(h1, dim, 'd_block3', None, act=relu)  # 8 * 8
    h3 = d_block(h2, dim, 'd_block4', None, act=relu)  # 8 * 8
    h3_act = relu(h3)
    feat   = tf.reshape(h3_act,[-1, 8 * 8 * dim])
    h4     = tf.reduce_sum(h3_act, [1, 2])
    out    = ops.linear(h4, 1, scope = 'linear_out')
    
    if is_conditional:
        h_labels = ops.sn_embedding(labels, num_classes, dim,
                                update_collection=update_collection,
                                name='d_embedding')
        out += tf.reduce_sum(h4 * h_labels, axis=1, keep_dims=True)
    
    return tf.sigmoid(out), out, tf.reshape(feat, [-1, 512 * 4 * 4])
   


'''
========================================================================
ResNet for STL-10 (48x48)
========================================================================
'''
    
def encoder_resnet_stl10(x, x_shape, z_dim=128, dim=64, \
                         num_classes = None, labels = None, \
                         name = 'encoder', \
                         update_collection=None, \
                                         reuse=False, is_training=True):
    if labels is not None:
        labels = tf.squeeze(labels)
    act = lrelu
    is_conditional = num_classes is not None and labels is not None
    with tf.variable_scope(name, reuse=reuse):
        image = tf.reshape(x, [-1, x_shape[0], x_shape[1], x_shape[2]])
        image = ops.conv2d(image, dim, 3, 3, 1, 1, \
                                           name='e_conv0') # 48 * 48 * dim
                                                    
        if is_conditional:
            act0  = e_block_cond(image, dim * 2,\
                                    num_classes = num_classes, \
                                    labels = labels,\
                                    is_training = is_training,\
                                    name = 'e_block1', act=act) # 24 * 24 * dim * 2                                 
            act1 = e_block_cond(act0, dim * 4, \
                                    num_classes, labels,\
                                    is_training = is_training,\
                                    name = 'e_block2', act=act) # 12 * 12 * dim * 4
                                      
            act2 = e_block_cond(act1, dim * 8, \
                                 num_classes, labels, \
                                 is_training = is_training,\
                                 name =  'e_block3', act=act)   # 6 * 6 * dim * 8
            bn   = ops.batch_norm(num_classes, name='e_bn')    
        
        else:
            
            act0  = e_block(image, dim * 2, is_training = is_training,\
                                    name = 'e_block1', act=act) # 24 * 24 * dim * 2    
            act1 = e_block(act0, dim * 4, is_training, \
                                      name = 'e_block2', act=act) # 12 * 12 * dim * 4 
            act2 = e_block(act1, dim * 8, is_training, \
                                 name =  'e_block3', act=act)     # 6 * 6 * dim * 8                                             
            bn   = ops.batch_norm(name='e_bn')
            
        act2 = act(bn(act2, is_training))                 
        act2 = tf.reshape(act2, [-1, 6 * 6 * dim * 8])                      
        out  = ops.linear(act2, z_dim)
        return out


def generator_resnet_stl10(z, x_shape, dim=64, \
                           num_classes = None, labels = None, \
                           name = 'generator', reuse=False, \
                           is_training=True):

    if labels is not None:
        labels = tf.squeeze(labels)
    x_dim = x_shape[0] * x_shape[1] * x_shape[2]
    is_conditional = num_classes is not None and labels is not None
    with tf.variable_scope(name, reuse=reuse):
        act0 = ops.linear(z, dim * 8 * 6 * 6, scope='g_linear0')
        act0 = tf.reshape(act0, [-1, 6, 6, dim * 8]) # 6 * 6 * dim * 8
        
        if is_conditional:
            act1 = g_block_cond(act0, dim * 4, num_classes, labels, \
                           is_training, 'g_block1')  # 12 * 12 * dim * 4
            act2 = g_block_cond(act1, dim * 2, num_classes, labels, \
                           is_training, 'g_block2')  # 24 * 24 * dim * 2 
            act3 = g_block_cond(act2, dim * 1, num_classes, labels, \
                           is_training, 'g_block3')  # 48 * 48 * dim * 1
            bn   = ops.batch_norm(num_classes, name='g_bn')                                                       
        else:
            act1 = g_block(act0, dim * 4, is_training, 'g_block1') # 12 * 12 * dim * 4
            act2 = g_block(act1, dim * 2, is_training, 'g_block2') # 24 * 24 * dim * 2
            act3 = g_block(act2, dim * 1, is_training, 'g_block3') # 48 * 48 * dim * 1
            bn   = ops.batch_norm(name='g_bn')
                        
        act3 = tf.nn.relu(bn(act3, is_training))
        act4 = ops.conv2d(act3, 3, 3, 3, 1, 1, name='g_conv_last')
        out  = tf.nn.sigmoid(act4)
        return tf.reshape(out, [-1, x_dim])



def discriminator_resnet_stl10(x, x_shape, dim=64, \
                            num_classes = None, labels = None, \
                            name = 'discriminator',\
                            update_collection=None, reuse=False):

  """Builds the discriminator graph.

  Args:
    x: The current batch of images to classify as fake or real.
    x_shape: the shape [h x w x c] of image
    dim: The d dimension.
    update_collection: The update collections used in the
                       spectral_normed_weight.
  Returns:
    A `Tensor` representing the logits of the discriminator.
  """
  if labels is not None:
     labels = tf.squeeze(labels)
  relu=tf.nn.relu
  is_conditional = num_classes is not None and labels is not None
  with tf.variable_scope(name, reuse=reuse):
    
    image = tf.reshape(x, [-1, x_shape[0], x_shape[1], x_shape[2]]) #48 * 48 * 3
    h0 = optimized_block(image, dim, 'd_optimized_block1', \
                                             act=relu) # 24 * 24 * dim
    h1 = d_block(h0, dim * 2, 'd_block2', act=relu)  # 12 * 12 * dim * 2
    h2 = d_block(h1, dim * 4, 'd_block3', act=relu)  # 6 * 6 * dim * 4
    h3 = d_block(h2, dim * 8, 'd_block4', act=relu)  # 3 * 3 * dim * 8
    h4 = d_block(h3, dim * 16, 'd_block5', None, act=relu)  # 3 * 3 * dim * 16
    h4_act = relu(h4)
    feat   = h4_act
    h5     = tf.reduce_sum(h4_act, [1, 2])
    out    = ops.linear(h5, 1, scope = 'linear_out')
    
    if is_conditional:
        h_labels = ops.sn_embedding(labels, num_classes, dim,
                                update_collection=update_collection,
                                name='d_embedding')
        out += tf.reduce_sum(h5 * h_labels, axis=1, keep_dims=True)
    
    feat = tf.reshape(feat, [-1, dim * 16 * 3 * 3])
    return tf.sigmoid(out), out, feat
