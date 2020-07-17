import numpy as np
import os
import pickle
from face_descriptors import run_loaded_image, show_boxes
from Profile import Profile


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
        if not name[0] == '.':
            images[name] = [x for x in os.listdir(images_dir + name + '/') if not x.startswith('.')]
    print (images)
    
    descriptors = {name : np.concatenate([run_loaded_image(model, images_dir + name + '/' + image)[0] for image in images[name]]) for name in images}  # Temporary function name for getting descriptor from image                        
    
    print (descriptors)
    for name in descriptors:
        print (descriptors[name].shape)
    
    
    profiles = [Profile(name, descriptors[name]) for name in descriptors]
        
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
        database[name].addVector(descriptor)
    else:
        new_profile = Profile(name, descriptor)
        database[name] = new_profile
    with open(filename, mode='wb') as path:
        pickle.dump(database, path)


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