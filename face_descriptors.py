# get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
from camera import take_picture
from facenet_models import FacenetModel
from matplotlib.patches import Rectangle


def run_new_image(model):
    pic = take_picture()
    boxes = show_boxes(model, pic)
    descriptors = model.compute_descriptors(pic, boxes)
    return descriptors, descriptors.shape


def run_loaded_image(model, path):
    pic = plt.imread(path)
    boxes = show_boxes(model, pic)
    descriptors = model.compute_descriptors(pic, boxes)
    return descriptors, descriptors.shape


def show_boxes(model, pic):
    fig, ax = plt.subplots()
    ax.imshow(pic)

    boxes, probabilities, landmarks = model.detect(pic)

    for box, prob, landmark in zip(boxes, probabilities, landmarks):
        # draw the box on the screen
        ax.add_patch(Rectangle(box[:2], *(box[2:] - box[:2]), fill=None, lw=2, color="red"))

        # Get the landmarks/parts for the face in box d.
        # Draw the face landmarks on the screen.
        for i in range(len(landmark)):
            ax.plot(landmark[i, 0], landmark[i, 1], '+', color="blue")

    return boxes
