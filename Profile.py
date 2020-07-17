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
        self.d_mean = np.sum(d_vectors, axis=0) / len(d_vectors)


    @property
    def mean_vector(self):
        """Supplies the up-to-date Profile instance mean descriptor vector

            Returns
            -------
            d_mean: np.ndarray, shape-(512,)
                The mean vector of all Profile instance-associated descriptor
                vectors
        """
        return self.d_mean


    @property
    def name(self):
        """Supplies the up-to-date Profile instance name

            Returns
            -------
            name: str
                The associated name of the Profile instance
        """
        return self.name


    @property
    def vectors(self):
        """Supplies the up-to-date Profile instance descriptor vectors
            (all of them)

            Returns
            -------
            d_vectors: np.ndarray, shape-(N, 512)
                An array containing all input descriptor vectors for a given Profile
                instance, where N is the number of input descriptor vectors
        """
        return self.d_vectors
