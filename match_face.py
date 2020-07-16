import numpy as np

def match(new_descriptor, cutoff_value):
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
    database_means = np.array(list(database.values()))
    distances = []  
    for d in database_means:
        distances.append(cos_dist(new_descriptor, d))
    min_distance = np.min(distances)
    index = np.argmin(distances)
    names = list(database.keys())
    if min_distance < cutoff_value:
        return names[index]
    else:
        return None

    

