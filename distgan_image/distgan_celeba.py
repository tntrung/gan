import os
import numpy as np
from modules.dataset import Dataset
from distgan import DISTGAN

if __name__ == '__main__':

    out_dir = 'output/'
    
    # downloading celeba from [https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAB06FXaQRUNtjW9ntaoPGvCa?dl=0]
    # and extracting it into the correct folder (should be prepared by yourself)
    db_name     = 'celeba'
    #data_source = './data/celeba/'
    data_source = '/home/mangroup/Documents/Code/Generative_Adversarial_Network/gaan/gaan_image/data/celeba/'
    
    model     = 'distgan'
    nnet_type = 'dcgan'
    loss_type = 'log' #'log' or 'hinge'
    
    is_train = True 
    
    '''
    model parameters
    '''
    noise_dim    = 100    #latent dim
    '''
    Feture dim, set your self as in the paper:
    dcgan: 8192
    sngan: 18432
    '''
    feature_dim  = 8192
    
    n_steps      = 200000 #number of iterations
        
    lambda_p  = 1.0
    lambda_r  = 1.0    
    '''
    [Impotant]
    lambda_w = sqrt(d/D) as in the paper, if you change the network 
    architecture: (d: data noise dim, D: feature dim)    
    '''
    lambda_w  = np.sqrt(noise_dim/feature_dim) 
    
    #output dir
    out_dir = os.path.join(out_dir, model + '_' + nnet_type, db_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # setup dataset
    dataset = Dataset(name=db_name, source=data_source)
    
    # setup gan model and train
    # setup gan model and train
    distgan = DISTGAN(model=model, \
                              loss_type = loss_type, \
                              lambda_p=lambda_p, lambda_r=lambda_r, \
                              lambda_w=lambda_w, \
                              noise_dim = noise_dim, \
                              nnet_type = nnet_type, \
                              dataset=dataset, \
                              n_steps = n_steps, \
                              out_dir=out_dir)
    if is_train == True:
        distgan.train()
    else:
        distgan.generate()
