import os
import numpy as np

from modules.dataset import Dataset
from gaan import GAN

if __name__ == '__main__':

    model   = 'gaan'
    db_name = 'mnist'
    out_dir = 'output/'
    data_source = './data/mnist/'
    
    #output dir
    out_dir = os.path.join(out_dir, model, db_name)
    if not os.path.exists(out_dir):
		os.makedirs(out_dir)

    # setup dataset
    dataset = Dataset(name=db_name, source=data_source)
    
    # setup gan model and train
    gan     = GAN(model=model, dataset=dataset, out_dir=out_dir)
    gan.train()


