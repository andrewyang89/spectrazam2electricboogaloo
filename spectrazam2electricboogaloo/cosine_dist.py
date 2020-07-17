import numpy as np


def cos_dist(d1, d2):
    """
    Returns the cosine distance between two description vectors.
    
    Parameters
    ----------
    d1: np.array
        First description vector
    d2: np.array
        Second description vector
        
    Returns
    -------
    cos_dist: float
        The distance between the two input vectors on range [0, 2]
    """
    d2 = d2.reshape(512, 1)
    cos = d1 @ d2 / (np.linalg.norm(d1) * np.linalg.norm(d2))
    return 1 - cos
