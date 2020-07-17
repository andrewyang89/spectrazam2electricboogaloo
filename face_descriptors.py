# get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
from camera import take_picture
from facenet_models import FacenetModel
from matplotlib.patches import Rectangle
import match_face
import database as db

def run_loaded_image(model, path):
    """
    Loads a picture and then processes the image using MTCNN and FaceNet
    :param model: the FaceNet model
    :param path: the path for the image file
    :return: the descriptor array, the shape of the array
    """
    pic = plt.imread(path)
    if pic.shape[-1] > 4:
        pic = pic[:, :, :3]
    boxes = show_boxes(model, pic)
    descriptors = model.compute_descriptors(pic, boxes)
    return descriptors, descriptors.shape


def run_new_image(model, path):
    """
    Takes an image and then processes the image using MTCNN and FaceNet
    :param model: the FaceNet model
    :return: the descriptor array, the shape of the array
    """
    pic = plt.imread(path)
    if pic.shape[-1] > 4:
        pic = pic[:, :, :3]
    boxes = show_boxes(model, pic)
    descriptors = model.compute_descriptors(pic, boxes)
    return descriptors, descriptors.shape


def show_boxes(model, pic):
    """
    Plots the image with the boxes applied and returns the boxes needed to compute the descriptors
    :param model: the FaceNet model
    :param pic: the picture to be processed
    :return: the boxes (detected faces)
    """
    fig, ax = plt.subplots()
    ax.imshow(pic)

    boxes, probabilities, landmarks = model.detect(pic)

    for box, prob, landmark in zip(boxes, probabilities, landmarks):
        # draw the box on the screen
        ax.add_patch(Rectangle(box[:2], *(box[2:] - box[:2]), fill=None, lw=2, color="green"))
        # Get the landmarks/parts for the face in box d.
        # Draw the face landmarks on the screen.
        for i in range(len(landmark)):
            ax.plot(landmark[i, 0], landmark[i, 1], '+', color="blue")

    plt.show()
    return boxes


def show_boxes_labels(model, pic, cutoff):
    """
    Plots the image with the boxes and labels applied and returns the boxes needed to compute the descriptors
    :param model: the FaceNet model
    :param pic: the picture to be processed
    :param cutoff: the cutoff value for a face to be labeled
    :return the number of unknowns and the array of descriptors for each unknown
    """
    fig, ax = plt.subplots()
    ax.imshow(pic)

    boxes, probabilities, landmarks = model.detect(pic)
    descriptors = model.compute_descriptors(pic, boxes)

    labels = match_face.match(descriptors, cutoff)

    unknown_counter = 0
    unk_descriptors = []

    counter = 0
    for box, prob, landmark, label in zip(boxes, probabilities, landmarks, labels):
        # draw the box on the screen
        ax.add_patch(Rectangle(box[:2], *(box[2:] - box[:2]), fill=None, lw=2, color="green"))
        if label is None:
            ax.annotate(f"Unknown{unknown_counter}", box[:2], color="red")
            unk_descriptors = unk_descriptors.append(descriptors[counter])
            unknown_counter += 1
        else:
            ax.annotate(label, box[:2], color="blue")
        # Get the landmarks/parts for the face in box d.
        # Draw the face landmarks on the screen.
        for i in range(len(landmark)):
            ax.plot(landmark[i, 0], landmark[i, 1], '+', color="blue")

        counter += 1
    plt.show()
    return unknown_counter, unk_descriptors


def prompt_unknown(num_unknown, unk_descriptors, database_name):
    """
    Prompts the user to name unknown faces
    :param num_unknown: the number of unknown faces
    :param unk_descriptors: the list of unknown descriptors
    :param database_name: the path of the database
    :return: nothing
    """
    for num in range(num_unknown):
        print(f"Enter the name of Unknown{num}:")
        name = str(input())
        db.add_profile(name, unk_descriptors[num], database_name)




