import numpy as np
import math

def _trasnfrom_vector(vec, transformation):
    """Transform a vector through a transformation

    :param vec: The vector should be n*3
    :type vec: np.ndarray
    :param transformation: The transformation should be a 4*4 matrix
    :type transformation: np.ndarray
    """    
    vec = np.concatenate((vec, np.ones((np.shape(vec)[0], 1))), axis=1)
    new_vec = np.transpose(np.dot(transformation, np.transpose(vec)))
    return new_vec[:, :3]