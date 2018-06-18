import os
import numpy as np

from skimage import io
from skimage.transform import resize

# for image io
from imutils import *

def create_stl10(source = 'unlabeled_X.bin', outdir = 'slt10'):
    '''
    Generate SLT-10 images from matlab files.
    '''
    with open(source, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        images = images.astype(float) / 255.0
        
        if not os.path.exists(outdir):
            os.mkdirs(outdir)

        nb_imgs = np.shape(images)[0]
        for ii in range(nb_imgs):
            print(ii, nb_imgs)
            img = resize(images[ii,:,:,:], [48, 48])
            imwrite(img, os.path.join(outdir, 'image_%06d.png' %(ii)))
            
            
if __name__ == '__main__':
    source = '/home/mangroup/Documents/Data/stl10_binary/unlabeled_X.bin'
    outdir = '/home/mangroup/Documents/Data/stl10_binary/images/'
    create_stl10(source, outdir)        
    
    
