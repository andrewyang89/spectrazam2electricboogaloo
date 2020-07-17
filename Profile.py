import numpy as np

class Profile:
    """
    Defines a profile for a face, including the respective face label and the
    mean descriptor vector of all associated descriptor vectors
    """

    def __init__(self, name: str, d_vectors_in):
        """Creates an instance of the Profile class

            Parameters
            ----------
            name: str
                The name of the owner of the input face and associated
                descriptor vectors

            d_vectors_in: list[np.ndarray]
                A list of descriptor vectors for the given person derived from
                images of their face

            Returns
            -------
            None

        """
        self.d_vectors = np.empty((0, 512))
        self.d_mean = np.empty(512,)
        self.name = name
        self.addVector(d_vectors_in)


    def addVector(self, d_vectors_in):
        """Appends the supplied descriptor vector to the associated profile's
            collection of descriptor vectors and computes the new mean
            descriptor vector behind the scenes

            Parameters
            ----------
            d_vectors_in: np.ndarray
                A 2D array of descriptor vectors of shape (M, 512) to be
                appended to the given Profile instance's descriptor vector
                collection, where M is the number of added vectors

            Returns
            -------
            None

        """

        self.d_vectors = np.append(self.d_vectors, d_vectors_in, axis=0)
        self.d_mean = np.sum(self.d_vectors, axis=0) / len(self.d_vectors)
