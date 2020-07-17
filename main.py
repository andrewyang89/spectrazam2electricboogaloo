import face_descriptors
from facenet_models import FacenetModel
import unknown


def run(pic_path, database):
    """
    Runs the model on an image and plots the names of the people
    :param pic_path: path for the image
    :param database: path for the database
    :return: None
    """
    model = FacenetModel()
    path = pic_path
    database = database

    cutoff = 0.3
    # Run line below for images downloaded to the computer
    # descriptors, shape = face_descriptors.run_loaded_image(model, path)
    unknown_count, unk_desc = face_descriptors.show_boxes_labels(model, path, cutoff, database)
    unknown.prompt_unknown(unknown_count, unk_desc, database)

    # Run line below to take an image using the computer camera
    # descriptors, shape = face_descriptors.run_new_image(model)
