import os
import numpy as np

from modules.dataset import Dataset
from gaan import GAAN

if __name__ == '__main__':

    out_dir = 'output/'
    
    # downloading celeba from [https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAB06FXaQRUNtjW9ntaoPGvCa?dl=0]
    # and extracting it into the correct folder (should be prepared by yourself)
    db_name     = 'celeba'
    data_source = './data/celeba/'
    
    model   = 'gaan'
    '''
    model parameters
    '''
    lambda_p  = 100.0
    lambda_r  = 100.0 # bigger D, bigger lambda_r
    # or you set by yourself = sqrt(d/D) as in the paper, if you change
    # the network architecture
    lambda_w  = 0.11049
    
    batch_size = 32 #smaller batch size than mnist and cifar-10
    out_step   = 100 #save images every out_step
    
    #output dir
    out_dir = os.path.join(out_dir, model, db_name)
    if not os.path.exists(out_dir):
		os.makedirs(out_dir)

    # setup dataset
    dataset = Dataset(name=db_name, source=data_source)
    
    # setup gan model and train
    gaan     = GAAN(model=model, lambda_p=lambda_p, lambda_r=lambda_r, \
                    lambda_w=lambda_w, \
                    batch_size = batch_size, dataset=dataset, \
                    out_dir=out_dir, out_step=out_step)
    gaan.train()


