import os
import numpy as np
from modules.dataset import Dataset
from distgan import DISTGAN

if __name__ == '__main__':

    out_dir = 'output/'
    db_name = 'mnist'
    # mnist data is automatically downloaded
    data_source = './data/mnist/'
    
    is_train = True
    
    model       = 'distgan'
    loss_type   = 'log' #'log' or 'hinge'
    
    '''
    model parameters
    '''
    noise_dim    = 100    #latent dim
    feature_dim  = 4096   #feture dim, set your self as in the paper
    
    n_steps      = 100000 #number of iterations
    
    lambda_p  = 1.0
    lambda_r  = 1.0
    
    # [Impotant]
    # lambda_w = sqrt(d/D) as in the paper, if you change the network 
    #  architecture: (d: data noise dim, D: feature dim)
    lambda_w  = np.sqrt(noise_dim * 1.0/feature_dim)
    
    # output dir
    out_dir = os.path.join(out_dir, model, db_name)
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
                              dataset=dataset, \
                              n_steps = n_steps, \
                              out_dir=out_dir)
    if is_train == True:
        distgan.train()
    else:
        distgan.generate()
        

    
    


