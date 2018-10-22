import numpy as np
from skimage import io

def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        path_dir, _ = os.path.split(path)
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir)

def imwrite(image, path):
    """ save an [-1.0, 1.0] image """
    if image.ndim == 3 and image.shape[2] == 1:  # for gray image
        image = np.array(image, copy=True)
        image.shape = image.shape[0:2]
    return io.imsave(path, image)
     
def immerge_row_col(N):
	c = int(np.floor(np.sqrt(N)))
	for v in range(c,N):
		if N % v == 0:
			c = v
			break
	r = N / c
	return r, c
	
def immerge(images, row, col):
    """
    merge images into an image with (row * h) * (col * w)
    @images: is in shape of N * H * W(* C=1 or 3)
    """
    row = int(row)
    col = int(col)
    h, w = images.shape[1], images.shape[2]
    if images.ndim == 4:
        img = np.zeros((h * row, w * col, images.shape[3]))
    elif images.ndim == 3:
        img = np.zeros((h * row, w * col))
    for idx, image in enumerate(images):
        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = image
    return img
