# Ngoc-Trung Tran, 2018
# Tensorflow implementation of GAN models

import os
import numpy as np
import tensorflow as tf
import time

from modules.imutils import *
from modules.mdutils import *
from modules.models_dcgan  import  *
from modules.models_sngan  import  *
from modules import ops

class DISTGAN(object):

    """
    Implementation of GAN methods.
    """

    def __init__(self, model='distgan', \
                 lambda_p = 1.0, lambda_r = 1.0, lambda_w = 0.15625, \
                 ncritics = 1, \
                 lr=2e-4, beta1 = 0.5, beta2 = 0.9, \
                 noise_dim = 100, \
                 nnet_type='dcgan', \
                 loss_type='log',\
                 df_dim = 64, gf_dim = 64, ef_dim = 64, \
                 dataset = None, batch_size = 64, \
                 n_steps = 400000, \
                 decay_step = 10000, decay_rate = 1.0,\
                 log_interval=10, \
                 out_dir = './output/', \
                 verbose = True):
        """
        Initializing GAAN

        :param model: the model name
        :param lr:    learning rate 
        :param nnet_type: the network architecture type of generator, discrimintor, ...
        :dataset:     the dataset pointer
        :db_name:     database name obtained from dataset
        """

        # dataset
        self.dataset   = dataset
        self.db_name   = self.dataset.db_name()

        # training parameters
        self.model      = model
        self.lr         = lr
        self.beta1      = beta1
        self.beta2      = beta2
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.n_steps    = n_steps
        self.batch_size = self.dataset.mb_size()

        # architecture
        self.nnet_type = nnet_type
        self.loss_type = loss_type
        self.ef_dim    = ef_dim
        self.gf_dim    = gf_dim
        self.df_dim    = df_dim

        # dimensions
        self.data_dim   = dataset.data_dim()
        self.data_shape = dataset.data_shape()
        self.noise_dim  = noise_dim

        # pamraeters
        self.lambda_p  = lambda_p
        self.lambda_r  = lambda_r
        self.lambda_w  = lambda_w
        
        self.nb_test_real = 10000
        self.nb_test_fake = 5000

        # others
        self.out_dir      = out_dir
        self.ckpt_dir     = out_dir + '/model/'
        self.log_interval = log_interval
        self.verbose      = verbose 

        self.create_model()

    def sample_z(self, N):
        return np.random.uniform(-1.0,1.0,size=[N, self.noise_dim])

    def create_discriminator(self):
        if self.nnet_type == 'dcgan' and self.db_name == 'mnist':
            return discriminator_dcgan_mnist
        elif self.nnet_type == 'dcgan' and self.db_name == 'celeba':
            return discriminator_dcgan_celeba
        elif self.nnet_type == 'dcgan' and self.db_name == 'cifar10':
            return discriminator_dcgan_cifar
        elif self.nnet_type == 'sngan' and self.db_name == 'cifar10':
            return discriminator_sngan_cifar            
        elif self.nnet_type == 'sngan' and self.db_name == 'stl10':
            return discriminator_sngan_stl10
        else:
            print('The dataset are not supported by the network');

    def create_generator(self):
        if self.nnet_type == 'dcgan' and self.db_name == 'mnist':
            return generator_dcgan_mnist
        elif self.nnet_type == 'dcgan' and self.db_name == 'celeba':
            return generator_dcgan_celeba    
        elif self.nnet_type == 'dcgan' and self.db_name == 'cifar10':
            return generator_dcgan_cifar
        elif self.nnet_type == 'sngan' and self.db_name == 'cifar10':
            return generator_sngan_cifar            
        elif self.nnet_type == 'sngan' and self.db_name == 'stl10':
            return generator_sngan_stl10
        else:
            print('The dataset are not supported by the network');
            
    def create_encoder(self):
        if self.nnet_type == 'dcgan' and self.db_name == 'mnist':
            return encoder_dcgan_mnist
        elif self.nnet_type == 'dcgan' and self.db_name == 'celeba':
            return encoder_dcgan_celeba
        elif self.nnet_type == 'dcgan' and self.db_name == 'cifar10':
            return encoder_dcgan_cifar
        elif self.nnet_type == 'sngan' and self.db_name == 'cifar10':
            return encoder_dcgan_cifar           
        elif self.nnet_type == 'sngan' and self.db_name == 'stl10':
            return encoder_sngan_stl10
        else:
            print('The dataset are not supported by the network');            

    def create_optimizer(self, loss, var_list, learning_rate, beta1, beta2):
        """Create the optimizer operation.

        :param loss: The loss to minimize.
        :param var_list: The variables to update.
        :param learning_rate: The learning rate.
        :param beta1: First moment hyperparameter of ADAM.
        :param beta2: Second moment hyperparameter of ADAM.
        :return: Optimizer operation.
        """
        return tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2).minimize(loss, var_list=var_list)    


    def create_model(self):

        self.X = tf.placeholder(tf.float32, shape=[None, self.data_dim])
        self.z = tf.placeholder(tf.float32, shape=[None, self.noise_dim])

        # create encoder
        with tf.variable_scope('encoder'):
            self.E   = self.create_encoder()
            self.z_e = self.E(self.X, self.data_shape, self.noise_dim, dim = self.ef_dim, reuse=False)

        # create generator
        with tf.variable_scope('generator'):
            self.G   = self.create_generator()
            self.X_f = self.G(self.z,   self.data_shape, dim = self.gf_dim, reuse=False) #generate fake samples
            self.X_r = self.G(self.z_e, self.data_shape, dim = self.gf_dim, reuse=True)  #generate reconstruction samples

        # create discriminator
        with tf.variable_scope('discriminator'):
            self.D   = self.create_discriminator()
            self.d_real_sigmoid,  self.d_real_logit,  self.f_real   = self.D(self.X,   self.data_shape, dim = self.df_dim, reuse=False)
            self.d_fake_sigmoid,  self.d_fake_logit,  self.f_fake   = self.D(self.X_f, self.data_shape, dim = self.df_dim, reuse=True)
            self.d_recon_sigmoid, self.d_recon_logit, self.f_recon  = self.D(self.X_r, self.data_shape, dim = self.df_dim, reuse=True)
            
            # Compute gradient penalty
            epsilon = tf.random_uniform(shape=[tf.shape(self.X)[0],1], minval=0., maxval=1.)
            interpolation = epsilon * self.X + (1 - epsilon) * self.X_f
            _,d_inter,_ = self.D(interpolation, self.data_shape, reuse=True)
            gradients = tf.gradients([d_inter], [interpolation])[0]
            slopes = tf.sqrt(tf.reduce_mean(tf.square(gradients), reduction_indices=[1]))
            penalty = tf.reduce_mean((slopes - 1) ** 2)

        # Reconstruction and regularization
        self.ae_loss    = tf.reduce_mean(tf.square(self.f_real - self.f_recon))
        self.md_x       = tf.reduce_mean(self.f_recon - self.f_fake)
        self.md_z       = tf.reduce_mean(self.z_e - self.z) * self.lambda_w
        self.ae_reg     = tf.square(self.md_x - self.md_z)

        if self.loss_type == 'log':
            # Loss
            self.d_real  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_logit, labels=tf.ones_like(self.d_real_sigmoid)))
            self.d_fake  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_logit, labels=tf.zeros_like(self.d_fake_sigmoid)))
            self.d_recon = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_recon_logit,labels=tf.ones_like(self.d_recon_sigmoid)))
            
            # lower weights for d_recon to achieve sharper generated images, slightly improved from the original paper
            self.d_cost  = 0.95 * self.d_real + 0.05 * self.d_recon + self.d_fake + self.lambda_p * penalty
        elif self.loss_type == 'hinge':
            #self.d_cost = -(0.95 * tf.reduce_mean(tf.minimum(0.,-1 + self.d_real_sigmoid))  + \
            #                0.05 * tf.reduce_mean(tf.minimum(0.,-1 + self.d_recon_sigmoid)) + \
            #                tf.reduce_mean(tf.minimum(0.,-1 - self.d_fake_sigmoid)) + self.lambda_p * self.penalty)

            self.d_cost = -(0.95 * tf.reduce_mean(tf.minimum(0.,-1 + self.d_real_logit))  + \
                            0.05 * tf.reduce_mean(tf.minimum(0.,-1 + self.d_recon_logit)) + \
                            tf.reduce_mean(tf.minimum(0.,-1 - self.d_fake_logit)) + self.lambda_p * self.penalty)  
                                  
        self.r_cost  = self.ae_loss  + self.lambda_r * self.ae_reg
        self.g_cost  = tf.abs(tf.reduce_mean(self.d_real_sigmoid) - tf.reduce_mean(self.d_fake_sigmoid))

        # Create optimizers
        self.vars_e = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        self.vars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.vars_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        
        print('[encoder parameters]')
        print(self.vars_e)
        print('[generator parameters]')
        print(self.vars_g)
        print('[discriminator parameters]')
        print(self.vars_d)

        # Setup for weight decay
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.lr, self.global_step, self.decay_step, self.decay_rate, staircase=True)

        if self.db_name in ['mnist', 'celeba']:
            self.opt_r = self.create_optimizer(self.r_cost, self.vars_e + self.vars_g, self.learning_rate, self.beta1, self.beta2)
        else:
            self.opt_r = self.create_optimizer(self.r_cost, self.vars_e, self.learning_rate, self.beta1, self.beta2)
        self.opt_g = self.create_optimizer(self.g_cost, self.vars_g, self.learning_rate, self.beta1, self.beta2)
        self.opt_d = self.create_optimizer(self.d_cost, self.vars_d, self.learning_rate, self.beta1, self.beta2)
        
        self.init = tf.global_variables_initializer()

    def train(self):
        """
        Training the model
        """
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        
        saver = tf.train.Saver(max_to_keep=2)

        with tf.Session(config=run_config) as sess:
            
            start = time.time()
            sess.run(self.init)

            for step in range(self.n_steps + 1):

                # train auto-encoder
                mb_X = self.dataset.next_batch()
                mb_z = self.sample_z(np.shape(mb_X)[0])
                
                if step == 0:
                    # check f_feature size of discriminator
                    f_real = sess.run(self.f_real,feed_dict={self.X: mb_X, self.z: mb_z})
                    print('=== Important!!!: Put this feature size: {} feature_dim of the main function ==='.format(np.shape(f_real)))
                
                X, X_f, X_r = sess.run([self.X, self.X_f, self.X_r],feed_dict={self.X: mb_X, self.z: mb_z})
                loss_r, _ = sess.run([self.r_cost, self.opt_r],feed_dict={self.X: mb_X, self.z: mb_z})

                # train discriminator
                mb_X = self.dataset.next_batch()
                mb_z = self.sample_z(np.shape(mb_X)[0])
                loss_d, _ = sess.run([self.d_cost, self.opt_d],feed_dict={self.X: mb_X, self.z: mb_z})

                # train generator
                mb_X = self.dataset.next_batch()
                mb_z = self.sample_z(np.shape(mb_X)[0])
                loss_g, _ = sess.run([self.g_cost, self.opt_g],feed_dict={self.X: mb_X, self.z: mb_z})

                if step % self.log_interval == 0:
                    if self.verbose:
                            elapsed = int(time.time() - start)
                            print('step: {:4d}, D loss: {:8.4f}, G loss: {:8.4f}, R loss: {:8.4f}, time: {:3d} s'.format(step, loss_d, loss_g, loss_r, elapsed)) 

                if step % 1000 == 0:
                    im_real_save = np.reshape(mb_X,(-1, self.data_shape[0], self.data_shape[1],self.data_shape[2]))
                    ncols, nrows = immerge_row_col(np.shape(im_real_save)[0])
                    im_save_path = os.path.join(self.out_dir,'image_%d_real.jpg' % (step))
                    im_merge = immerge(im_real_save, ncols, nrows)
                    imwrite(im_merge, im_save_path)
                    

                    # save generated images
                    im_fake_save = sess.run(self.X_f,feed_dict={self.z: mb_z})
                    im_fake_save = np.reshape(im_fake_save,(-1, self.data_shape[0], self.data_shape[1], self.data_shape[2]))
                    ncols, nrows = immerge_row_col(np.shape(im_fake_save)[0])
                    im_save_path = os.path.join(self.out_dir,'image_%d_fake.jpg' % (step))
                    im_merge = immerge(im_fake_save, ncols, nrows)
                    imwrite(im_merge, im_save_path)
                    
                if step % (self.log_interval*1000) == 0:
                                     
                    if step == 0:
                        real_dir = self.out_dir + '/real/'
                        if not os.path.exists(real_dir):
                            os.makedirs(real_dir)
                            
                    fake_dir = self.out_dir + '/fake_%d/'%(step)
                    if not os.path.exists(fake_dir):
                        os.makedirs(fake_dir)
                        
                    #generate reals
                    if step == 0:
                        for v in range(self.nb_test_real // self.batch_size + 1):
                            #print(v, self.nb_test_real)
                            # train auto-encoder
                            mb_X = self.dataset.next_batch()
                            im_real_save = np.reshape(mb_X,(-1, self.data_shape[0], self.data_shape[1],self.data_shape[2]))
                            
                            for ii in range(np.shape(mb_X)[0]):
                                real_path = real_dir + '/image_%05d.jpg' % (np.min([v*self.batch_size + ii, self.nb_test_real]))
                                imwrite(im_real_save[ii,:,:,:], real_path)
                    elif step > 0:
                        #generate fake
                        for v in range(self.nb_test_fake // self.batch_size + 1):
                            #print(v, self.nb_test_fake)
                            mb_z = self.sample_z(np.shape(mb_X)[0])
                            im_fake_save = sess.run(self.X_f,feed_dict={self.z: mb_z})
                            im_fake_save = np.reshape(im_fake_save,(-1, self.data_shape[0], self.data_shape[1], self.data_shape[2]))

                            for ii in range(np.shape(mb_z)[0]):
                                fake_path = fake_dir + '/image_%05d.jpg' % (np.min([v*self.batch_size + ii, self.nb_test_fake]))
                                imwrite(im_fake_save[ii,:,:,:], fake_path)


                if step % (self.log_interval*2000) == 0:
                    if not os.path.exists(self.ckpt_dir):
                        os.makedirs(self.ckpt_dir)
                    save_path = saver.save(sess, '%s/epoch_%d.ckpt' % (self.ckpt_dir, step))
                    print('Model saved in file: % s' % save_path)

    def generate(self):
        
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
                
        with tf.Session(config=run_config) as sess:

            flag = load_checkpoint(self.ckpt_dir, sess)
                
            if flag == True:
                
                fake_dir = self.out_dir + '/fake/'
                
                if not os.path.exists(fake_dir):
                    os.makedirs(fake_dir)               
                    
                for v in range(self.nb_test_fake // self.batch_size + 1):
                    mb_z = self.sample_z(np.shape(mb_X)[0])
                    im_fake_save = sess.run(self.X_f,feed_dict={self.z: mb_z})
                    im_fake_save = np.reshape(im_fake_save,(-1, self.data_shape[0], self.data_shape[1], self.data_shape[2]))

                    for ii in range(np.shape(mb_z)[0]):
                        fake_path = fake_dir + '/image_%05d.jpg' % (np.min([v*self.batch_size + ii, self.nb_test_fake]))
                        imwrite(im_fake_save[ii,:,:,:], fake_path)
