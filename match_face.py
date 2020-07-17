import numpy as np
from cosine_dist import cos_dist
import pickle


def match(new_descriptor, cutoff_value, filename):
    """
    Matches an unknown descriptor value with a name from the database.
    Parameters
    ----------
    new_descriptor: np.array
        Descriptor of the unknown person
    cutoff_value: float
        A threshold value to identify how "close" descriptors must be
        (experimentally determined)
    
    Returns
    -------
    name: String
        Name of identified person, None if no person is identified
    """
    # Loading the mean vectors into a list
    with open(filename, mode="rb") as opened_file:
        database = pickle.load(opened_file)

    database_means = []  
    for profile in database.values():
        database_means.append(profile.d_mean)
    
    # Loading the cos distances into a list
    distances = []  
    for d in database_means:
        distances.append(cos_dist(new_descriptor, d))
    
    # Calculating the minimum distance
    min_distance = np.min(distances)
    index = np.argmin(distances)
    names = list(database.keys())
    
    # Verifying that the minimum distance does not exceed threshold value,
    # returning corresponding name or None
    if min_distance < cutoff_value:
        return names[index]
    else:
        return None
