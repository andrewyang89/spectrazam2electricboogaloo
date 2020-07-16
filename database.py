import numpy as np
import os
import pickle
from face_descriptors import run_loaded_image

def create_database(model, images_dir, filename):
    """
    Create and save a database of profiles
    
    Parameters
    ----------
    model : FacenetModel
        pretrained model of facial recognition algorithm
        
    images_dir : str
        contains directory with all folders of peoples name and associated images
    
    filename : str
        path to store database
    
    """
    images = {}
    for name in os.listdir(images_dir):
        images[name] = os.listdir(images_dir + name + '/')
    descriptors = {name : np.hstack(run_loaded_image(image)[0] for image in images[name]) for name in images}  # Temporary function name for getting descriptor from image
        
    profiles = [Profile(name) for name in descriptors]
    for p in profiles:
        p.add_descriptors(descriptors[p.name])  # Temp class attributes and function names
        
    database = {}
    for p in profiles:
        database[p.name] = p
    
    with open(filename, mode="wb") as path:
        pickle.dump(database, path) 

def add_profile(model, name, image, filename):
    """
    Add profile of name and image to database
    
    Parameters
    ----------
    model : FacenetModel
        pretrained facial recognition model
    name : str
        name of person in image
    image : 
        picture of person
    filename : str
        path where database is located
    
    """
    with open(filename, mode='rb') as path:
        database = pickle.load(path)
    descriptor = run_loaded_image(model, image)
    
    if name in database:
        database[name].add_descriptors(descriptor)
    else:
        new_profile = Profile(name)
        new_profile.add_descriptors(descriptor)
        database[name] = new_profile

def remove_profile(name, filename):
    """
    Remove specified profile by name from database
    
    Parameters
    ----------
    name : str
        name of person to remove from database
    filename : str
        path where database is located
    """
    with open(filename, mode='rb') as path:
        database = pickle.load(path)
    del database[name]
    with open(filename, mode='wb') as path:
        pickle.dump(database, path)