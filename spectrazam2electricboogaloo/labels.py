from matplotlib.patches import Rectangle

from . import database as db
import matplotlib.pyplot as plt

from spectrazam2electricboogaloo import match_face


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

    unknown_counter = 0
    unk_descriptors = []
    labeled_descriptors = []
    names = []

    boxes, probabilities, landmarks = model.detect(pic)
    if boxes is not None:
        descriptors = model.compute_descriptors(pic, boxes)

        counter = 0
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
                labeled_descriptors.append(descriptors[counter])
                names.append(label)

            counter += 1
    plt.show()
    if len(labeled_descriptors) > 0:
        print("Are all of the labeled faces correct? Type 'Y' or 'N'.")
        answer = str(input())
        if answer == "Y":
            for i in range(len(descriptors)):
                db.add_profile(names[i], descriptors[i].reshape(1, 512), database)

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
        print(f"Enter the name of Unknown{num}. Type 'exit' if you do not want to add a label.")
        name = str(input())
        if name != "exit":
            db.add_profile(name, unk_descriptors[num].reshape(1, 512), database_name)
