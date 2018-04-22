import os, sys
sys.path.append(os.getcwd())

import time
from time import gmtime, strftime

import numpy as np
import tensorflow as tf
import sklearn.datasets

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot


x_dim       = 784      # Number of pixels in MNIST (28*28)
z_dim       = 100      # Noise dim
batch_size  = 50       # Batch size

lambda_p    = 1.0      # discriminator penalty term
lambda_r    = 1.0      # autoencoder regularization term
lambda_w    = 0.1562   # scalar factor between delta x and delta z: = sqrt(d/D)

nb_iters    = 200000   # How many iterations to train.
dim_net     = 64       # For network size
lr          = 2e-4     # Learning rate
beta1       = 0.5      # Beta1
beta2       = 0.9      # Beta2
nb_critics  = 1        # Number of critics

train = True

if train == True:

    curr_time = time.strftime("%Y%m%d_%H%M")
    outdir = 'outputs/experiment_' + curr_time + '/'

else:
    outdir = 'outputs/expriment_20180215_0040/'
    
if not os.path.exists(outdir):
    os.makedirs(outdir)
    
def sample_z(m,n):
    return np.random.uniform(-1.0,1.0,size=[m,n])    
    
lib.print_model_settings(locals().copy())

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def gradient_penalty(inter_X):
    _,inter,_ = Discriminator(inter_X)
    gradients = tf.gradients([inter], [inter_X])[0]
    slopes = tf.sqrt(tf.reduce_mean(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)
    return gradient_penalty

def Encoder(inputs):
    output = tf.reshape(inputs, [-1, 1, 28, 28])
    output = lib.ops.conv2d.Conv2D('Encoder.1',1, dim_net,5,output,stride=2)
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Encoder.2', dim_net, 2*dim_net, 5, output, stride=2)
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Encoder.3', 2*dim_net, 4*dim_net, 5, output, stride=2)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*4*4*dim_net])
    output = lib.ops.linear.Linear('Encoder.Output', 4*4*4*dim_net, z_dim, output)
    output = tf.reshape(output, [-1, z_dim])
    return output

def Generator(input_z):
    output = lib.ops.linear.Linear('Generator.Input', z_dim, 8*4*4*dim_net, input_z)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 8*dim_net, 4, 4])
    output = lib.ops.deconv2d.Deconv2D('Generator.1', 8*dim_net, 4*dim_net, 5, output)
    output = tf.nn.relu(output)
    output = output[:,:,:7,:7]
    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*dim_net, 2*dim_net, 5, output)
    output = tf.nn.relu(output)
    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2 * dim_net, 1, 5, output)
    output = tf.reshape(output, [-1, x_dim])
    return tf.nn.sigmoid(output), output

def Discriminator(inputs):
    output = tf.reshape(inputs, [batch_size, 1, 28, 28])
    output = lib.ops.conv2d.Conv2D('Discriminator.1',1,dim_net,5,output,stride=2)
    output = LeakyReLU(output)
    output = lib.ops.conv2d.Conv2D('Discriminator.2', dim_net, 2*dim_net, 5, output, stride=2)
    output = LeakyReLU(output)
    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim_net, 4*dim_net, 5, output, stride=2)
    output = LeakyReLU(output)
    f_output = tf.reshape(output, [batch_size, 4*4*4*dim_net])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*dim_net, 1, f_output)
    output = tf.reshape(output, [batch_size, 1])
    return tf.nn.sigmoid(output), output, f_output

X     = tf.placeholder(tf.float32, shape=[None, x_dim])
z     = tf.placeholder(tf.float32, shape=[None, z_dim])

# generate and encode fake data
fake_X, fake_X_logit = Generator(z)
encode_fake_X        = Encoder(fake_X)

# encode and decode real data
encode_X = Encoder(X)
decode_X, decode_X_logit = Generator(encode_X)

d_real,  d_real_logit,  f_real  = Discriminator(X)
d_recon, d_recon_logit, f_recon = Discriminator(decode_X)
d_fake,  d_fake_logit,  f_fake  = Discriminator(fake_X)

# debug info
f_size = tf.cast(f_real.get_shape()[1],tf.float32) # = 4096
print('feature size: {}'.format(f_size))

# auto-encoder regularization
rec_loss     = tf.reduce_mean(tf.square(f_real - f_recon))
rec_loss_x   = tf.reduce_mean((f_real - f_fake))
rec_loss_z   = tf.reduce_mean((encode_X - z)) * lambda_w
rec_reg      = tf.square(rec_loss_x - rec_loss_z)

# gradient penalty
beta      = tf.random_uniform(shape=[tf.shape(encode_X)[0], 1], minval=0, maxval=1)
inter_X   = (1. - beta) * X + beta * fake_X
grad_X    = gradient_penalty(inter_X)

# Cost on data
d_cost_real  =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logit,  labels=tf.ones_like(d_real_logit)))
d_cost_recon =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_recon_logit, labels=tf.ones_like(d_recon_logit))) #as real
d_cost_fake  =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logit,  labels=tf.zeros_like(d_fake_logit)))
d_cost       =  (d_cost_real + d_cost_recon) * 0.5 + d_cost_fake 

# final costs
g_cost =  tf.abs(tf.reduce_mean(d_real) - tf.reduce_mean(d_fake))
d_cost =  d_cost   + lambda_p * grad_X
r_cost =  rec_loss + lambda_r * rec_reg

enc_params    = lib.params_with_name('Encoder')
gen_params    = lib.params_with_name('Generator')
disc_params   = lib.params_with_name('Discriminator')

# Setup for weight decay
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = lr 
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 1.0, staircase=True) #no decay

# Optimizer
gen_train_op  = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2).minimize(g_cost, var_list=gen_params, global_step=global_step)
disc_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2).minimize(d_cost, var_list=disc_params, global_step=global_step)
rec_train_op  = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2).minimize(r_cost, var_list=enc_params  + gen_params, global_step=global_step)
    
# For saving samples
_fixed_z = sample_z(128, z_dim)
def generate_image(frame):
    samples = session.run(fake_X, feed_dict={z: _fixed_z})
    lib.save_images.save_images(
        samples.reshape((128, 28, 28)), 
        outdir + '/samples128_{}.png'.format(frame)
    )

def save_image(frame,data,datatype): #1: real, #2, reconstructed, #3 generated samples
    samples = data
    if datatype == 1:
        strtype = 'real'
    elif datatype == 2:
        strtype = 'reconstructed'
    elif datatype == 3:
        strtype = 'generated'
    lib.save_images.save_images(
        samples.reshape((batch_size, 28, 28)), 
        outdir + '/samples_minibatch_{}_{}.png'.format(frame,strtype)
    )    

saver            = tf.train.Saver()
saved_model_path = outdir + '/save_model/'

if not os.path.exists(saved_model_path):
    os.makedirs(saved_model_path)

def load_checkpoint(checkpoint_dir, session, var_list=None):
    print(' [*] Loading checkpoint...: {}'.format(checkpoint_dir))
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    try:
        restorer = tf.train.Saver(var_list)
        restorer.restore(session, ckpt_path)
        print(' [*] Loading successful! Copy variables from % s' % ckpt_path)
        return True
    except:
        print(' [*] No suitable checkpoint!')
        return False

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True

# Dataset iterator
train_gen, dev_gen, test_gen = lib.mnist.load(batch_size, batch_size)
def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images

if train == True:
    # Train loop
    with tf.Session(config=run_config) as session:
        session.run(tf.global_variables_initializer())
        gen = inf_train_gen()

        for iteration in range(nb_iters):

            _X = gen.next()
            _z = sample_z(batch_size,z_dim)
            _r_cost, _ = session.run([r_cost, rec_train_op],feed_dict={X: _X, z: _z})
            
            for k in range(nb_critics):
                _z = sample_z(batch_size,z_dim)
                _X = gen.next()
                _d_cost, _ = session.run([d_cost, disc_train_op],feed_dict={X: _X, z: _z})
            
            _X = gen.next() 
            _z = sample_z(batch_size,z_dim)
            _g_cost, _ = session.run([g_cost, gen_train_op],feed_dict={X: _X, z: _z})
            
            # save model
            if iteration % 10000 == 9999:
                saver.save(session, save_path=saved_model_path, global_step=iteration)

            if iteration % 100 == 99:
                print('Iteration: %d, r_cost: %f, d_cost: %f, g_cost: %f' % (iteration, _r_cost, _d_cost, _g_cost))
                _decode_X, _fake_X = session.run([decode_X, fake_X], feed_dict={X: _X, z: _z})
                
                # save images to debug
                save_image(iteration, _X, 1)        # save real images
                save_image(iteration, _decode_X, 2) # save reconstruction
                save_image(iteration, _fake_X, 3)   # save generated samples            
               
                # Generate fixed samples
                generate_image(iteration)
else:
    
    with tf.Session(config=run_config) as session:
                        
        load_checkpoint(saved_model_path,session)
        
        print('Start!')
                                         
        # latent interpolation
        latent_begin  = sample_z(1,z_dim)
        sample_begin  = session.run(fake_X, feed_dict={z: latent_begin})
        latent_end    = sample_z(1,z_dim)
        sample_end    = session.run(fake_X, feed_dict={z: latent_end})
        
        N = 128 # change if you want
        inter_zs = []
        for i in range(N):
            latent_curr = latent_begin * (1 - i * 1/N) + latent_end * i * 1/N
            inter_zs.append(latent_curr)
        inter_zs = np.concatenate(inter_zs, axis=0)
        inter_samples = session.run(fake_X, feed_dict={z: inter_zs})
        lib.save_images.save_images(inter_samples.reshape((N, 28, 28)), outdir + '/interpolated_samples.png')
        
        # latent arithemtic, just for interests
        zs  = sample_z(3,z_dim)
        samples = session.run(fake_X, feed_dict={z: zs})
        inter_z  = np.reshape(zs[0,:] - zs[1,:] + zs[2,:],(1,z_dim))
        inter_samples = session.run(fake_X, feed_dict={z: inter_z})
        inter_samples = np.concatenate([samples, inter_samples], axis=0)
        lib.save_images.save_images(inter_samples.reshape((4, 28, 28)), outdir + '/interpolated_samples_ari.png')
        
        print('End!')
