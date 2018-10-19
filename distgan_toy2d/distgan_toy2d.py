# Implementation of Dist-GAN with 2d synthetic data
# by Ngoc-Trung Tran and Tuan-Anh Bui 2018
# Contact: tntrung@gmail.com
# If you use our code, please cite our paper.

import tensorflow as tf
import numpy as np
import random
import os
import os.path
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
from   time import gmtime, strftime

from utils import sample_z, gradient_penalty, random_mini_batches
from utils_toy2d import read_toydata, normalize_toydata, plot_data, evaluate_toydata, callback_centroids, random_batches
from model_mlp import theta_def, mlp, mlp_feat

	
#tf.reset_default_graph()
"""
Test case
"""
testcase      = 'SQUARE'  # 'SINE' 'PLUS' 'SQUARE'
OUTDIR        = 'outputs/'

# ------------------ Parameters ----------------------------------------
num_epoch     = 500     # number of epochs
mb_size       = 128     # mini-batch size
lr            = 1e-3    # learning rate
beta1         = 0.8     # beta1 of Adam optimizer
beta2         = 0.999   # beta2 of Adam optimizer
decay_niter   = 10000   # decay step
decay_base    = 0.9     # decay rate
X_dim         = 2       # data dimension
z_dim         = 2       # latent dimension
lamda_p       = 0.1     # regularization term of gradient penalty
lamda_r       = 0.1     # autoencoders regularization term
n_critics     = 5       # number critics for training discriminators

# ------------------ Load toydata --------------------------------------
var = 0.1
toydata = []

if testcase == 'SQUARE':
	toydata = read_toydata('toy_data/toydatav2.txt')
elif testcase == 'SINE':
	toydata = read_toydata('toy_data/sinedata.txt')
elif testcase == 'PLUS':
	toydata = read_toydata('toy_data/plusdatav2.txt')
# Normalize toydata
[toydata, toydata_size, centroids, var] = normalize_toydata(toydata, testcase, var)

# ------------------ Network architecture ------------------------------
X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

""" Generator: G(z) """
theta_G = theta_def([z_dim, 128, 128, 128, X_dim])
""" Encoder: E(X) """
theta_E = theta_def([X_dim, 128, 128, 128, z_dim])
""" Discriminator: D(x) """
theta_D = theta_def([X_dim, 128, 128, 128, 1])

# ------------------ Setup criteria/loss functions ---------------------

# Encode and decode data
_, _ze              = mlp(X, theta_E)
_Xr_prob, _Xr_logit = mlp(_ze, theta_G)

# Latent interpolation
alpha = 0.5
zi    = (1. - 1. * alpha) * z + 1. * alpha * _ze

# Sample from random z
_X_prob, _X_logit   = mlp(z, theta_G)
_Xi_prob, _Xi_logit   = mlp(zi, theta_G)

D_real , D_real_logit , f_real    = mlp_feat(X       , theta_D)
D_recon, D_recon_logit, f_recon   = mlp_feat(_Xr_prob, theta_D)
D_fake , D_fake_logit , f_fake    = mlp_feat(_X_prob , theta_D)
D_inter, D_inter_logit, f_inter   = mlp_feat(_Xi_prob , theta_D)

# auto-encoders and its regularization
R_loss = tf.reduce_mean(
				 tf.nn.sigmoid_cross_entropy_with_logits(logits=_Xr_logit, labels=X)) #X needed to normalized to the range [0,1]
#R_loss = tf.reduce_mean(tf.square(_Xr_prob - X)) # another option

f            = tf.reduce_mean(_Xr_prob - _X_prob)
g            = tf.reduce_mean(_ze - z)
R_reg        = tf.square(f - g)

# Gradient penalty 
beta = tf.random_uniform(shape=[tf.shape(_ze)[0], 1], minval=0, maxval=1)
_X_i = (1. - beta) * X + beta * _X_prob
gp = gradient_penalty(_X_i, theta_D)

# Discriminator loss on data
d_loss_real = tf.reduce_mean(
	tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logit, labels=tf.ones_like(D_real)))
d_loss_recon = tf.reduce_mean(
	tf.nn.sigmoid_cross_entropy_with_logits(logits=D_recon_logit, labels=tf.zeros_like(D_recon)))
d_loss_fake = tf.reduce_mean(
	tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logit, labels=tf.zeros_like(D_fake)))
d_loss_inter = tf.reduce_mean(
	tf.nn.sigmoid_cross_entropy_with_logits(logits=D_inter_logit, labels=tf.zeros_like(D_inter))) 

D_loss = (d_loss_real + d_loss_recon)*0.5 + d_loss_fake

# final loss
G_loss = tf.abs(tf.reduce_mean(D_real) - tf.reduce_mean(D_fake))
D_loss = D_loss + lamda_p * gp
R_loss = R_loss + lamda_r * R_reg

# ------------------ Solvers -------------------------------------------

# Setup for weight decay
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = lr
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_niter, decay_base, staircase=True)

# Build solver
R_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2).minimize(R_loss, var_list=theta_E + theta_G,global_step=global_step)
D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2).minimize(D_loss, var_list=theta_D, global_step=global_step)
G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2).minimize(G_loss, var_list=theta_G, global_step=global_step)

# Session and CPU configuration
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True
sess = tf.Session(config=run_config)
sess.run(tf.global_variables_initializer())

# ------------------ Output --------------------------------------------

currtime   = strftime("%Y%m%d_%H%M", gmtime()) # output based on current time
output_dir = OUTDIR + '/' + currtime + '/'
if not os.path.exists(output_dir): # create folder it not exist
	os.makedirs(output_dir)

# Plot groundtruth
plotcnt         = 0

# Log files
logfile      = os.path.join(output_dir, "loghistory.txt")
fid          = open(logfile,"w")

nb_samples = 2000
seed = 0
count = 0

for i in range(num_epoch):

	seed        = seed + 1
	mini        = 0

	minibatches = random_mini_batches(toydata.T, mini_batch_size=mb_size, seed=seed)
			
	for minibatch in minibatches:
		mini = mini + 1
		X_mb = minibatch.T
		count = count + 1

		# Train auto-encoders
		z_mb = sample_z(X_mb.shape[0], z_dim)
		sess.run([R_solver], feed_dict={X: X_mb, z: z_mb})
						
		# Train disciminator
		for k in range(n_critics):
			z_mb_critics = sample_z(X_mb.shape[0], z_dim)
			X_mb_critics = random_batches(toydata, X_mb.shape[0])
			sess.run([D_solver], feed_dict={X: X_mb_critics, z: z_mb_critics})
			
		# Train generator
		z_mb = sample_z(X_mb.shape[0], z_dim)
		sess.run([G_solver], feed_dict={X: X_mb, z: z_mb})

		# Display loss results at final mini-batch of each epoch
		if mini == len(minibatches):

			global_step_curr = sess.run([global_step])
			learning_rate_curr = sess.run([learning_rate])
			D_loss_curr, G_loss_curr, R_loss_curr, R_reg_curr, f_curr, g_curr = sess.run([D_loss, G_loss, R_loss, R_reg, f, g],
																					feed_dict={X: X_mb, z: z_mb})
																					
			_X_prob_plot   = sess.run(_X_prob, feed_dict={z: sample_z(nb_samples,z_dim)})
																					
			# Console display                                                                                                                       
			print('Epoch: {}'.format(i))
			#print("Global step: ", global_step_curr, "learning_rate", learning_rate_curr)
			print('D_loss: {:.4}; G_loss: {:.4}; R_loss: {:.4}; R_reg: {:.4}; f: {:.4}; g: {:.4};'.format(
				   D_loss_curr, G_loss_curr, R_loss_curr, R_reg_curr, f_curr, g_curr))

			plotcnt    = plotcnt + 1
			prename = 'epoch_' + str(i) + '_mini_' + str(mini) + '_'
						
			# Output images
			plot_data(_X_prob_plot, output_dir + prename + 'X_sample_' , plotcnt, title='X_sample')         
			
			# Evaluate based on ground-truth
			evaluate = evaluate_toydata(_X_prob_plot,centroids,var)
			# Print [number of samples of modes, total registered points and number of covered modes]
			print(evaluate,np.sum(evaluate[1:]),np.sum(np.asarray(evaluate[1:])<20))
			fid.write(str(evaluate) + '\t\t'+ str(np.sum(evaluate[1:]))+'\n')
			fid.flush()
