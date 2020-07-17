# get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import match_face
import camera


def run_loaded_image(model, path):
    """
    Loads a picture and then processes the image using MTCNN and FaceNet
    :param model: the FaceNet model
    :param path: the path for the image file
    :return: the descriptor array, the shape of the array
    """
    pic = plt.imread(path)
    if pic.shape[-1] > 3:
        pic = pic[:, :, :3]
    boxes, probabilities, landmarks = model.detect(pic)
    descriptors = model.compute_descriptors(pic, boxes)
    return descriptors, descriptors.shape


def run_new_image(model):
    """
    Takes an image and then processes the image using MTCNN and FaceNet
    :param model: the FaceNet model
    :return: the descriptor array, the shape of the array
    """
    pic = camera.take_picture()
    if pic.shape[-1] > 3:
        pic = pic[:, :, :3]
    boxes, probabilities, landmarks = model.detect(pic)
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


def show_boxes_labels(model, path, cutoff, database):
    """
    Plots the image with the boxes and labels applied and returns the boxes needed to compute the descriptors
    :param path: the path of the image
    :param database: database
    :param model: the FaceNet model
    :param cutoff: the cutoff value for a face to be labeled
    :return the number of unknowns and the array of descriptors for each unknown
    """
    fig, ax = plt.subplots()
    pic = plt.imread(path)
    ax.imshow(pic)

    if pic.shape[-1] > 3:
        pic = pic[:, :, :3]

    boxes, probabilities, landmarks = model.detect(pic)
    descriptors = model.compute_descriptors(pic, boxes)

    unknown_counter = 0
    unk_descriptors = []

    counter = 0
    zipped = list(zip(boxes, probabilities, landmarks))
    for box, prob, landmark in zip(boxes, probabilities, landmarks):
        # draw the box on the screen
        ax.add_patch(Rectangle(box[:2], *(box[2:] - box[:2]), fill=None, lw=2, color="green"))
        label = match_face.match(descriptors[counter], cutoff, database)
        if label is None:
            ax.annotate(f"Unknown{unknown_counter}", box[:2], color="red")
            unk_descriptors.append(descriptors[counter])
            unknown_counter += 1
        else:
            ax.annotate(label, box[:2], color="blue")

        counter += 1
    plt.show()
    return unknown_counter, unk_descriptors





