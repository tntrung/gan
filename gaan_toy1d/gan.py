# Dist-GAN,
# by Ngoc-Trung Tran and Tuan-Anh Bui, 2018
# Our tensorflow code is based on:
# 1) (WGAN-GP) Jan Kremer, 2017: https://github.com/kremerj/gan
# 2) (GAN) Eric Jang, http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html
# We re-use code of (1) as main visualization part.
# In addition, we implement also other GAN methods: VAEGAN, MDGAN.

import numpy as np
import tensorflow as tf
import time

from tensorflow.contrib.keras import layers


class GAN(object):
    """Implementation of the Dist-GAN and other GAN algorithm.

    The models for critic and generator are relatively simple and can be modified for anything more complicated than
    the 1D toy example.
    """

    def __init__(self, model='gaan', n_step=2000, n_critic=5, n_batch=64, n_hidden=4, n_sample=10000, learning_rate=1e-3,
                 lambda_reg=0.5, log_interval=10, seed=0, beta1=0.5, beta2=0.9, verbose=True, callback=None):
        """Initialize the GAN.

        :param n_step: Number of optimization steps.
        :param n_critic: Number of critic optimization step per generator optimization step.
        :param n_batch: Mini-batch size.
        :param n_hidden: Number of hidden neurons in critic and generator.
        :param n_sample: Number of samples to draw from the model.
        :param learning_rate: The learning rate of the optimizer.
        :param lambda_reg: The regularization parameter lambda that controls the gradient regularization when training.
        :param log_interval: The number of steps between logging the training process.
        :param seed: The seed to control random number generation during training.
        :param beta1: Hyperparameter to control the first moment decay of the ADAM optimizer.
        :param beta2: Hyperparameter to control the second moment decay of the ADAM optimizer.
        :param verbose: Whether to print log messages during training or not.
        :param callback: Callback method to call after each training step with signature (model, session, data).
        """
        self.model    = model
        self.n_step   = n_step
        self.n_critic = n_critic
        self.n_batch  = n_batch
        self.n_hidden = n_hidden
        self.n_sample = n_sample
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.log_interval = log_interval
        self.seed = seed
        self.beta1 = beta1
        self.beta2 = beta2
        self.verbose = verbose
        self.callback = callback
        self.loss_d_curve = []
        self.loss_g_curve = []
        self.loss_r_curve = []
        self.loss_e_curve = []
        self.graph = self._create_graph()

    def kl_loss(self, mu, log_var):
        return -0.5 * tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var))

    def nll_normal(self, pred, target):
 
        c = -0.5 * tf.log(2 * np.pi)
        multiplier = 1.0 / (2.0 * 1)
        tmp = tf.square(pred - target)
        tmp *= -multiplier
        tmp += c
 
        return tmp   
        
    def _create_encoder(self, activation=tf.nn.relu):
        """Create the computational graph of the encoder and return it as a functional of its input.

        :param activation: The activation function to use.
        :return: Functional to create the tensorflow operation given its input.
        """
        h = layers.Dense(self.n_hidden, activation=activation)
        output = layers.Dense(1)
        return lambda x: output(h(x))

    def _create_encoder_vaegan(self, activation=tf.nn.relu):
        """Create the computational graph of the encoder and return it as a functional of its input.

        :param activation: The activation function to use.
        :return: Functional to create the tensorflow operation given its input.
        """
        h      = layers.Dense(self.n_hidden, activation=activation)
        z      = layers.Dense(1)
        output_m    = layers.Dense(1)
        output_s    = layers.Dense(1)
        return lambda x: output_m(z(h(x))), lambda x: output_s(z(h(x)))

    def _create_generator(self, activation=tf.nn.relu):
        """Create the computational graph of the generator and return it as a functional of its input.

        :param activation: The activation function to use.
        :return: Functional to create the tensorflow operation given its input.
        """
        h = layers.Dense(self.n_hidden, activation=activation)
        output = layers.Dense(1)
        return lambda x: output(h(x))

    def _create_discriminator(self, activation=tf.nn.relu):
        """Create the computational graph of the critic and return it as a functional of its input.

        :param activation: The activation function to use.
        :return: Functional to create the tensorflow operation given its input.
        """
        h = layers.Dense(self.n_hidden, activation=activation)
        output = layers.Dense(1)
        return lambda x: output(h(x))

    def _create_optimizer(self, loss, var_list, learning_rate, beta1, beta2):
        """Create the optimizer operation.

        :param loss: The loss to minimize.
        :param var_list: The variables to update.
        :param learning_rate: The learning rate.
        :param beta1: First moment hyperparameter of ADAM.
        :param beta2: Second moment hyperparameter of ADAM.
        :return: Optimizer operation.
        """
        return tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2).minimize(loss, var_list=var_list)
        
    def get_weights(self, wname='generator'):
        return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith(wname)]       

    def _create_graph(self):
        """Creates the computational graph.

        :return: The computational graph.
        """
        
        eps = 1e-5
        with tf.Graph().as_default() as graph:
            tf.set_random_seed(self.seed)  # Fix the random seed for randomized tensorflow operations.
            
            self.x = tf.placeholder(tf.float32, shape=(None, 1))
            self.z = tf.placeholder(tf.float32, shape=(None, 1))

            with tf.variable_scope('generator'):  # Create generator operations.
                self.G  = self._create_generator()
                self.xg = self.G(self.z)
                           
            if self.model == 'gan': 
                
                with tf.variable_scope('discriminator'):  # Create critic operations.
                    D = self._create_discriminator() # D prob and D logit
                    self.D_real = tf.sigmoid(D(self.x))   # discriminate real
                    self.D_fake = tf.sigmoid(D(self.xg))  # discriminate fake
                    
                    epsilon = tf.random_uniform(shape=tf.shape(self.x), minval=0., maxval=1.)
                    interpolation = epsilon * self.x + (1 - epsilon) * self.xg
                    penalty = (tf.norm(tf.gradients(D(interpolation), interpolation), axis=1) - 1) ** 2.0    
                    
                #GAN   
                self.loss_d = tf.reduce_mean(-tf.log(self.D_real + eps) - tf.log(1 - self.D_fake + eps)) * 0.5
                self.loss_g = tf.reduce_mean(-tf.log(self.D_fake + eps))
                
            elif self.model == 'rgan':
                with tf.variable_scope('encoder'):  # Create encoder operations.
                    self.E  = self._create_encoder()
                    self.ze = self.E(self.x)
                    self.xr = self.G(self.ze)
                with tf.variable_scope('discriminator'):  # Create critic operations.
                    D = self._create_discriminator() # D prob and D logit
                    self.D_real  = tf.sigmoid(D(self.x))   # discriminate real
                    self.D_fake  = tf.sigmoid(D(self.xg))  # discriminate fake
                    self.D_recon = tf.sigmoid(D(self.xr))  # discriminate recon
                    self.mse     = tf.reduce_sum(tf.square(self.x - self.xr),1)
                    
                    lambda1 = 1e-2
                    lambda2 = 1e-2
                    self.loss_d = tf.reduce_mean(-tf.log(self.D_real + eps) - tf.log(1 - self.D_fake + eps))
                    self.loss_e = tf.reduce_mean(lambda1*self.mse + lambda2*self.D_recon)  
                    self.loss_g = tf.reduce_mean(-tf.log(self.D_fake + eps) + self.loss_e)
                    
            elif self.model == 'mdgan':
                with tf.variable_scope('encoder'):  # Create encoder operations.
                    self.E  = self._create_encoder()
                    self.ze = self.E(self.x)
                    self.xr = self.G(self.ze)
                with tf.variable_scope('discriminator1'):  # Create critic operations.
                    D1 = self._create_discriminator() # D prob and D logit
                    self.D1_real  = tf.sigmoid(D1(self.x))   # discriminate real
                    self.D1_recon = tf.sigmoid(D1(self.xr))  # discriminate recon
                    self.mse      = tf.reduce_sum(tf.square(self.x - self.xr),1)
                    
                    lambda1 = 1e-2
                    lambda2 = 1e-2
                    self.loss_d1 = tf.reduce_mean(-tf.log(self.D1_real + eps) - tf.log(1 - self.D1_recon + eps))
                    self.loss_g1 = tf.reduce_mean(self.mse - lambda1*self.D1_recon)
                    
                with tf.variable_scope('discriminator'):  # Create critic operations.
                    D = self._create_discriminator() # D prob and D logit
                    self.D_fake = tf.sigmoid(D(self.xg))  # discriminate fake
                    self.D_real = tf.sigmoid(D(self.xr))  # discriminate recon
                    
                    self.loss_d = tf.reduce_mean(-tf.log(self.D_real + eps) - tf.log(1 - self.D_fake + eps))
                    self.loss_g = tf.reduce_mean(-tf.log(self.D_fake + eps))   

            elif self.model == 'vaegan':

                self.ep = tf.random_normal(shape=[tf.shape(self.x)[0], 1])

                with tf.variable_scope('encoder'):  # Create encoder operations.
                    self.Em, self.Es = self._create_encoder_vaegan()
                    self.ze_m        = self.Em(self.x)
                    self.ze_s        = self.Es(self.x)
                    self.ze_x        = tf.add(self.ze_m, tf.sqrt(tf.exp(self.ze_s)) * self.ep)
                    self.xr          = self.G(self.ze_x)

                with tf.variable_scope('discriminator'):
                    D = self._create_discriminator() # D prob and D logit.
                    self.D_real  = tf.sigmoid(D(self.x))
                    self.D_recon = tf.sigmoid(D(self.xr))
                    self.D_fake  = tf.sigmoid(D(self.xg))

					# if want to use gradient penalty?
                    # epsilon = tf.random_uniform(shape=tf.shape(self.x), minval=0., maxval=1.)
                    # interpolation = epsilon * self.x + (1 - epsilon) * self.xg
                    # penalty = (tf.norm(tf.gradients(D(interpolation), interpolation), axis=1) - 1) ** 2.0

                # kl loss
                self.kl = self.kl_loss(self.ze_m, self.ze_s)
                # gan loss
                self.ld = tf.reduce_mean(-tf.log(self.D_real + eps) - tf.log(1 - self.D_recon + eps) - tf.log(1 - self.D_fake + eps))
                # preceptual loss (feature loss or reconstruction loss)
                self.lr = tf.reduce_mean(self.nll_normal(self.xr, self.x))
                # encoder
                self.le = -self.lr + self.kl/(self.n_batch) 
                # generator
                self.lg = tf.reduce_mean(-tf.log(self.D_fake + eps) - tf.log(self.D_recon + eps)) - 1e-6*self.lr
                
                self.loss_d = self.ld #+ self.lambda_reg * penalty #for use penalty as gaan
                self.loss_g = self.lg
                self.loss_e = self.le       

                self.loss_d   = tf.reshape(self.loss_d, []) #convert to scalar                    
                                     
            elif self.model == 'wgangp':
                with tf.variable_scope('discriminator'):  # Create critic operations.
                    D = self._create_discriminator() # D prob and D logit.
                    self.D_real = D(self.x)  # Criticize real data.
                    self.D_fake = D(self.xg)  # Criticize generated data.
                    diff = tf.abs(tf.reduce_mean(self.D_real - self.D_fake))
                    # Create the gradient penalty operations.
                    epsilon = tf.random_uniform(shape=tf.shape(self.x), minval=0., maxval=1.)
                    interpolation = epsilon * self.x + (1 - epsilon) * self.xg
                    penalty = (tf.norm(tf.gradients(D(interpolation), interpolation), axis=1) - 1) ** 2.0
                    
                self.loss_d = tf.reduce_mean(self.D_fake - self.D_real + self.lambda_reg * penalty) 
                self.loss_g = -tf.reduce_mean(self.D_fake)
                                                
            elif self.model == 'gaan': 
                with tf.variable_scope('encoder'):  # Create generator operations.
                    self.E  = self._create_encoder()
                    self.ze = self.E(self.x)
                    self.xr = self.G(self.ze)
                    
                    #encoder xg
                    self.zg = self.E(self.xg)
                    
                with tf.variable_scope('discriminator'):  # Create critic operations.
                    D = self._create_discriminator() # D prob and D logit
                    self.D_real_logit  = D(self.x)          # discriminate real
                    self.D_fake_logit  = D(self.xg)         # discriminate fake
                    self.D_recon_logit = D(self.xr)         # discriminate fake                    
                    self.D_real  = tf.sigmoid(self.D_real_logit)          # discriminate real
                    self.D_fake  = tf.sigmoid(self.D_fake_logit)          # discriminate fake
                    self.D_recon = tf.sigmoid(self.D_recon_logit)         # discriminate fake
                    diff = tf.abs(tf.reduce_mean(self.D_real - self.D_fake))
                    # Create the gradient penalty operations.
                    epsilon = tf.random_uniform(shape=tf.shape(self.x), minval=0., maxval=1.)
                    interpolation = epsilon * self.x + (1 - epsilon) * self.xg
                    penalty = (tf.norm(tf.gradients(D(interpolation), interpolation), axis=1) - 1) ** 2.0
                    #penalty = (tf.norm(tf.gradients(D(interpolation), interpolation), axis=1) - tf.minimum(diff,1.0)) ** 2.0
                
                self.recon   = tf.reduce_mean(tf.square(self.x - self.xr)) #reconstruction
                self.loss_x  = tf.reduce_mean(self.x - self.xg)
                self.loss_z  = tf.reduce_mean(self.ze - self.z)
                self.reg     = tf.square(self.loss_x - self.loss_z)
                                
                self.ld      = tf.reduce_mean(- 0.5 * tf.log(self.D_real + eps) - 0.5 * tf.log(self.D_recon + eps) - tf.log(1 - self.D_fake + eps))
                self.lg      = tf.abs(tf.reduce_mean(self.D_real - self.D_fake))
                self.loss_d  = self.ld      + self.lambda_reg * penalty
                self.loss_r  = self.recon   + 0.1 * self.reg
                self.loss_g  = self.lg
                
                self.loss_d   = tf.reshape(self.loss_d, []) #convert to scalar
                self.loss_g   = tf.reshape(self.loss_g, []) #//
                self.loss_r   = tf.reshape(self.loss_r, []) #//
                    
            # Store the variables of the critic and the generator.
            self.vars_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            self.vars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

            # Create optimizer operations for critic and generator.
            self.opt_d = self._create_optimizer(self.loss_d, self.vars_d, self.learning_rate, self.beta1, self.beta2)
            self.opt_g = self._create_optimizer(self.loss_g, self.vars_g, self.learning_rate, self.beta1, self.beta2)

            if  self.model == 'rgan':
                self.vars_e = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
                self.opt_e  = self._create_optimizer(self.loss_e, self.vars_e, self.learning_rate, self.beta1, self.beta2)
            elif self.model == 'mdgan':
                self.vars_e  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
                self.vars_d1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator1')
                self.opt_d1  = self._create_optimizer(self.loss_d1, self.vars_d1, self.learning_rate, self.beta1, self.beta2)
                self.opt_g1  = self._create_optimizer(self.loss_g1, self.vars_g + self.vars_e, self.learning_rate, self.beta1, self.beta2)
            elif self.model == 'vaegan':

                self.vars_e = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
                
                #for D
                self.trainer_D = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                self.gradients_D = self.trainer_D.compute_gradients(self.loss_d, var_list=self.vars_d)
                self.clipped_gradients_D = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in self.gradients_D]
                self.opti_D = self.trainer_D.apply_gradients(self.clipped_gradients_D)

                #for G
                self.trainer_G = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                self.gradients_G = self.trainer_G.compute_gradients(self.loss_g, var_list=self.vars_g)
                self.clipped_gradients_G = [(tf.clip_by_value(_[0], -1, 1.), _[1]) for _ in self.gradients_G]
                self.opti_G = self.trainer_G.apply_gradients(self.clipped_gradients_G)
         
                #for E
                self.trainer_E = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                self.gradients_E = self.trainer_E.compute_gradients(self.loss_e, var_list=self.vars_e)
                self.clipped_gradients_E = [(tf.clip_by_value(_[0], -1, 1.), _[1]) for _ in self.gradients_E]
                self.opti_E = self.trainer_E.apply_gradients(self.clipped_gradients_E)
                
            elif self.model == 'gaan':
                self.vars_e    = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
                self.opt_r     = self._create_optimizer(self.loss_r, self.vars_e + self.vars_g, self.learning_rate, self.beta1, self.beta2)
                self.opt_recon = self._create_optimizer(self.recon, self.vars_e + self.vars_g, self.learning_rate, self.beta1, self.beta2)
                
            summary_enc_1_params = self.get_weights('encoder/dense/kernel') 
            summary_enc_2_params = self.get_weights('encoder/dense_1/kernel') 
            tf.summary.histogram('enc_1_params',summary_enc_1_params)
            tf.summary.histogram('enc_2_params',summary_enc_2_params)
            
            summary_gen_1_params = self.get_weights('generator/dense/kernel')
            summary_gen_2_params = self.get_weights('generator/dense_1/kernel')
            tf.summary.histogram('gen_1_params',summary_gen_1_params)
            tf.summary.histogram('gen_2_params',summary_gen_2_params)
            
            summary_disc_1_params = self.get_weights('discriminator/dense/kernel')
            summary_disc_2_params = self.get_weights('discriminator/dense_1/kernel')
            tf.summary.histogram('disc_1_params',summary_disc_1_params)
            tf.summary.histogram('disc_2_params',summary_disc_2_params)

            # Create variable initialization operation.
            self.init = tf.global_variables_initializer()
            #graph.finalize()
        return graph

    def _sample_latent(self, n_sample):
        """Sample the input data to generate synthetic samples from.

        :param n_sample: Sample size.
        :return: Sample of input noise.
        """
        return np.random.randn(n_sample, 1)      

    def fit(self, X):
        """Fit the GAN model.

        :param X: Training data.
        :return: The fit model.
        """
        np.random.seed(self.seed)  # Fix the seed for random data generation in numpy.
        
        logs_dir = 'logs/'
        
        with tf.Session(graph=self.graph) as session:
            start = time.time()
            session.run(self.init)

            summary_writer = tf.summary.FileWriter(logs_dir, session.graph)
            summary_op     = tf.summary.merge_all()
                        
            for step in range(self.n_step + 1):
                if self.model == 'gan':
                    
                    x, _ = X.next_batch(self.n_batch)
                    z = self._sample_latent(self.n_batch)
                    loss_d, _ = session.run([self.loss_d, self.opt_d], {self.x: x, self.z: z})
                    
                    # Sample noise and optimize the generator.
                    x, _ = X.next_batch(self.n_batch)
                    z = self._sample_latent(self.n_batch)
                    loss_g, _ = session.run([self.loss_g, self.opt_g], {self.x: x, self.z: z})
                        
                elif self.model == 'rgan':     
                    
                    # Training discriminator
                    x, _      = X.next_batch(self.n_batch)
                    z         = self._sample_latent(self.n_batch)
                    loss_d, _ = session.run([self.loss_d, self.opt_d], {self.x: x, self.z: z})                  

                    # Sample noise and optimize the generator.
                    x, _ = X.next_batch(self.n_batch)
                    z = self._sample_latent(self.n_batch)
                    loss_g, _ = session.run([self.loss_g, self.opt_g], {self.x: x, self.z: z}) 
                                        
                    # Traing encoder
                    x, _ = X.next_batch(self.n_batch)
                    z = self._sample_latent(self.n_batch)
                    loss_e, _ = session.run([self.loss_e, self.opt_e], {self.x: x, self.z: z})
                    
                elif self.model == 'mdgan':     
                    
                    # Training discriminator 1
                    x, _      = X.next_batch(self.n_batch)
                    z         = self._sample_latent(self.n_batch)
                    loss_d1, _ = session.run([self.loss_d1, self.opt_d1], {self.x: x, self.z: z})    
                    
                    # Traing encoder and generator
                    #x, _ = X.next_batch(self.n_batch)
                    #z = self._sample_latent(self.n_batch)
                    loss_g1, _ = session.run([self.loss_g1, self.opt_g1], {self.x: x, self.z: z})  
                    
                    # Training discriminator 2
                    x, _      = X.next_batch(self.n_batch)
                    z         = self._sample_latent(self.n_batch)
                    loss_d, _ = session.run([self.loss_d, self.opt_d], {self.x: x, self.z: z})
                    
                    # Training generator
                    #x, _ = X.next_batch(self.n_batch)
                    #z = self._sample_latent(self.n_batch)
                    loss_g, _ = session.run([self.loss_g, self.opt_g], {self.x: x, self.z: z})

                elif self.model == 'vaegan':

                    x, _      = X.next_batch(self.n_batch)
                    z         = self._sample_latent(self.n_batch)    
                    
                    #optimizaiton E
                    session.run(self.opti_E, feed_dict={self.x: x, self.z: z})
                    #optimizaiton G
                    session.run(self.opti_G, feed_dict={self.x: x, self.z: z})
                    # optimization D
                    session.run(self.opti_D, feed_dict={self.x: x, self.z: z})

                    loss_d, loss_g, loss_e = session.run([self.loss_d, self.loss_g, self.loss_e], {self.x: x, self.z: z})
             
                elif self.model == 'wgangp':
                    # Optimize the critic for several rounds.
                    for _ in range(self.n_critic):
                        x, _ = X.next_batch(self.n_batch)
                        z = self._sample_latent(self.n_batch)
                        loss_d, _ = session.run([self.loss_d, self.opt_d], {self.x: x, self.z: z})

                    # Sample noise and optimize the generator.
                    x, _ = X.next_batch(self.n_batch)
                    z = self._sample_latent(self.n_batch)
                    loss_g, _ = session.run([self.loss_g, self.opt_g], {self.x: x, self.z: z})

                elif self.model == 'gaan':
                                        
                    # Traing encoder and generator
                    x, _ = X.next_batch(self.n_batch)
                    z = self._sample_latent(self.n_batch)
                    loss_r, _ = session.run([self.loss_r, self.opt_r], {self.x: x, self.z: z})

                    x, _ = X.next_batch(self.n_batch)
                    z = self._sample_latent(self.n_batch)
                    # Optimize the critic for several rounds.
                    for _ in range(self.n_critic):
                        x, _      = X.next_batch(self.n_batch)
                        z         = self._sample_latent(self.n_batch)
                        loss_d, _ = session.run([self.loss_d, self.opt_d], {self.x: x, self.z: z})
                                            
                    # Sample noise and optimize the generator.
                    z = self._sample_latent(self.n_batch)
                    loss_g, _ = session.run([self.loss_g, self.opt_g], {self.x: x, self.z: z})     
                    #loss_g, _ = session.run([self.lg2, self.opt_g], {self.x: x, self.z: z})     
                
                
                if step % 10 == 9:    
                    summary_str =  session.run(summary_op,feed_dict={self.x: x, self.z: z})
                    summary_writer.add_summary(summary_str,step)
                    summary_writer.flush()

                # Log the training procedure and call callback method for actions like plotting.
                if step % self.log_interval == 0:
                    if self.model == 'gan':
                        self.loss_d_curve += [loss_d]
                        self.loss_g_curve += [loss_g]                        
                        if self.verbose:
                            elapsed = int(time.time() - start)
                            print('step: {:4d}, D loss: {:8.4f}, G loss: {:8.4f}, time: {:3d} s'.format(step, loss_d, loss_g, elapsed)) 
                    elif self.model == 'rgan':
                        self.loss_d_curve += [loss_d]
                        self.loss_g_curve += [loss_g]
                        self.loss_e_curve += [loss_e]
                        if self.verbose:
                            elapsed = int(time.time() - start)
                            print('step: {:4d}, D loss: {:8.4f}, G loss: {:8.4f}, E loss: {:8.4f}, time: {:3d} s'.format(step, loss_d, loss_g, loss_e, elapsed)) 
                    elif self.model == 'mdgan':
                        self.loss_d_curve += [loss_d]
                        self.loss_g_curve += [loss_g]
                        if self.verbose:
                            elapsed = int(time.time() - start)
                            print('step: {:4d}, D loss: {:8.4f}, G loss: {:8.4f}, D1 loss: {:8.4f}, G1 loss: {:8.4f}, time: {:3d} s'.format(step, loss_d, loss_g, loss_d1, loss_g1, elapsed))       
                    elif self.model == 'vaegan':
                        self.loss_d_curve += [loss_d]
                        self.loss_g_curve += [loss_g]
                        if self.verbose:
                            elapsed = int(time.time() - start)
                            print('step: {:4d}, D loss: {:8.4f}, G loss: {:8.4f}, time: {:3d} s'.format(step, loss_d, loss_g, elapsed))                                                    
                    elif self.model == 'wgangp':
                        self.loss_d_curve += [-loss_d]
                        self.loss_g_curve += [-loss_g]
                        if self.verbose:
                            elapsed = int(time.time() - start)
                            print('step: {:4d}, negative critic loss: {:8.4f}, , negative G loss: {:8.4f}, time: {:3d} s'.format(step, -loss_d, -loss_g, elapsed))

                    elif self.model == 'gaan':
                        self.loss_d_curve += [loss_d]
                        self.loss_g_curve += [loss_g]
                        self.loss_r_curve += [loss_r/10.] #to better draw                          
                        if self.verbose:
                            elapsed = int(time.time() - start)
                            print('step: {:4d}, D loss: {:8.4f}, G loss: {:8.4f}, R loss: {:8.4f}, time: {:3d} s'.format(step, loss_d, loss_g, loss_r, elapsed))                  

                    if self.callback is not None:
                        self.callback(self, session, X)
        return self

    def sample(self, session):
        """Sample generated data.

        :param session: The current tensorflow session holding the trained graph.
        :return: A sample of generated data.
        """
        z = self._sample_latent(self.n_sample)
        return np.array(session.run(self.xg, {self.z: z}))

    def reconstruct(self, session, x):
        """Reconstruct data.

        :param session: The current tensorflow session holding the trained graph.
        :return: A sample of generated data.
        """
        return np.array(session.run(self.xr, {self.x: x}))     
        
    def encode(self, session, x):
        """Reconstruct data.

        :param session: The current tensorflow session holding the trained graph.
        :return: A sample of generated data.
        """
        return np.array(session.run(self.ze, {self.x: x}))            

    def dreal(self, session, x):
        """Returns the critic function.

        :param session: Tensorflow session.
        :param x: Input data to criticize.
        :return: The current critic function.
        """
        return np.array(session.run(self.D_real, {self.x: x}))
