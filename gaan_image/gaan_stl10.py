import os
import numpy as np

from modules.dataset import Dataset
from gaan import GAAN

if __name__ == '__main__':

    out_dir     = 'output/'
    # downloading stl-10 and extracting it into the correct folder
    db_name     = 'stl10'
    data_source = '/home/mangroup/Documents/Data/stl10_binary/images/'

    model       = 'gaan'    #'gan', 'wgangp', 'gaan'
    noise_dim   = 128
    nnet_type   = 'sngan'

    '''
    training parameters
    '''
    n_steps   = 200000
    
    # set f_dim the correct size (output from last layer of discriminator)
    if nnet_type == 'sngan': 
        df_dim = 64
        gf_dim = 64
        ef_dim = 64  
        lr     = 2e-4
        beta1  = 0.5
        beta2  = 0.9
        f_dim  = 18432.
    elif nnet_type == 'dcgan':
        lr    = 2e-4
        beta2 = 0.9
        beta1 = 0.5
        f_dim = 8192.

    '''
    model parameters
    '''
    ncritics = 1
    lambda_p = 1.0
    lambda_r = 1.0
    # or you set by yourself = sqrt(d/D) as in the paper, if you change
    # the network architecture
    lambda_w  = np.sqrt(noise_dim/f_dim)
    
    batch_size = 64
    
    model_dir   = model + '_' + nnet_type + '_' + db_name + '_z_%d' % (noise_dim)  
    
    #output dir
    out_dir = os.path.join(out_dir, model_dir, db_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # setup dataset
    dataset = Dataset(name=db_name, source=data_source, batch_size=batch_size)

    # setup gan model and train
    gaan     = GAAN(model=model, \
                    lambda_p=lambda_p, lambda_r=lambda_r, \
                    lambda_w=lambda_w, \
                    ncritics = ncritics, \
                    lr=lr, beta1 = beta1, beta2 = beta2, \
                    noise_dim = noise_dim, \
                    nnet_type = nnet_type, \
                    df_dim = df_dim, gf_dim = gf_dim, ef_dim = ef_dim, \
                    n_steps = n_steps, \
                    dataset=dataset, out_dir=out_dir)
    gaan.train()
