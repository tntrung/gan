import os
import numpy as np
from modules.dataset import Dataset
from distgan import DISTGAN

if __name__ == '__main__':

    out_dir = 'output/'
    
    # downloading cifar-10 from 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    # and extracting it into the correct folder
    db_name     = 'cifar10'
    #data_source = './data/cifar10/'
    data_source  = '/home/mangroup/Documents/Code/Generative_Adversarial_Network/gaan/gaan_image/data/cifar10/'
       
    is_train = True
    
    model     = 'distgan' 
    nnet_type = 'resnet' #'dcgan', 'sngan', 'resnet'
    loss_type = 'hinge'  #'log' or 'hinge'
    '''
    model parameters
    '''
    noise_dim = 128       #latent dim
    '''
    Feture dim, set your self as in the paper:
    dcgan: 2048
    sngan: 8192
    resnet 8192
    '''
    if nnet_type == 'sngan' or \
       nnet_type == 'resnet':
		feature_dim = 8192
    elif nnet_type == 'dcgan':
		feature_dim = 2048
    
    if nnet_type == 'resnet':
        df_dim = 128
        gf_dim = 128
        ef_dim = 128
        beta1  = 0.0
        beta2  = 0.9
    else:
        df_dim = 64
        gf_dim = 64
        ef_dim = 64
        beta1  = 0.5
        beta2  = 0.9
            
    n_steps      = 300000 #number of iterations
        
    lambda_p  = 1.0
    lambda_r  = 1.0
    '''
    [Impotant]
    lambda_w = sqrt(d/D) as in the paper, if you change the network 
    architecture: (d: data noise dim, D: feature dim)    
    '''
    lambda_w  = np.sqrt(noise_dim * 1.0/feature_dim)
    
    #output dir
    out_dir = os.path.join(out_dir, model + '_' + nnet_type, db_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # setup dataset
    dataset = Dataset(name=db_name, source=data_source)
    
    # setup gan model and train
    distgan = DISTGAN(model=model, \
                              loss_type = loss_type, \
                              lambda_p=lambda_p, lambda_r=lambda_r, \
                              lambda_w=lambda_w, \
                              noise_dim = noise_dim, \
                              beta1 = beta1, \
                              beta2 = beta2, \
                              nnet_type = nnet_type, \
                              df_dim = df_dim, \
                              gf_dim = gf_dim, \
                              ef_dim = ef_dim, \
                              dataset=dataset, \
                              n_steps = n_steps, \
                              out_dir=out_dir)
    if is_train == True:
        distgan.train()
    else:
        distgan.generate()
