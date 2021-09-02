import random
import numpy as np

def HorizontalFlip(batch_of_images, p=0.5):
    """
    Reverse the entire rows and columns of the video pixels, with a probability p (default 0.5)
    """
    # Input size : (T, H, W, C)
    if random.random() > p:
        batch_of_images = batch_of_images[:,:,::-1,...]
    return batch_of_images

def RandomDeleting(batch_of_images, p=0.05):
    """
    Delete frames with probability p
    """
    # Input size : (T, H, W, C)
    for idx, image in enumerate(batch_of_images):
        if random.random() < p:
            batch_of_images[idx] = np.zeros(((128, 64, 3)))
    return batch_of_images