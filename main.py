import face_descriptors
from facenet_models import FacenetModel

model = FacenetModel()
pic = "g20.jpg"
database = "database.pkl"
cutoff = 0.01
# Run line below for images downloaded to the computer
descriptors, shape = face_descriptors.run_loaded_image(model, pic)
unknown_count, unk_desc = face_descriptors.show_boxes_labels(model, pic, cutoff)
face_descriptors.prompt_unknown(unknown_count, unk_desc, database)


# Run line below to take an image using the computer camera
# descriptors, shape = face_descriptors.run_new_image(model)

print(descriptors)
print(shape)
