import os
import numpy as np
from modules.dataset import Dataset
from gaan import GAAN

if __name__ == '__main__':

    out_dir = 'output/'
    
    # automatically download mnist into the correct folder
    db_name = 'mnist'
    data_source = './data/mnist/'
    
    model   = 'gaan'
    '''
    model parameters
    '''
    lambda_p  = 1.0
    lambda_r  = 1.0
    # or you set by yourself = sqrt(d/D) as in the paper, if you change
    # the network architecture
    lambda_w  = 0.15625
    
    #output dir
    out_dir = os.path.join(out_dir, model, db_name)
    if not os.path.exists(out_dir):
		os.makedirs(out_dir)

    # setup dataset
    dataset = Dataset(name=db_name, source=data_source)
    
    # setup gan model and train
    gaan     = GAAN(model=model, lambda_p=lambda_p, lambda_r=lambda_r, \
                    lambda_w=lambda_w, dataset=dataset, out_dir=out_dir)
    gaan.train()


