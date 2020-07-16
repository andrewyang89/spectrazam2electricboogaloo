import face_descriptors
from facenet_models import FacenetModel

model = FacenetModel()

# Run line below for images downloaded to the computer
descriptors, shape = face_descriptors.run_loaded_image(model, "g20.jpg")

# Run line below to take an image using the computer camera
# descriptors, shape = face_descriptors.run_new_image(model)

print(descriptors)
print(shape)
