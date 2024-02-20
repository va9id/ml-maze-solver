import cv2
import numpy as np
from keras.models import load_model

# Load the model
model = load_model("keras_maze_classifier/keras_model.h5", compile=False)

# Load the labels
class_names = open("keras_maze_classifier/labels.txt", "r").readlines()

image = cv2.imread('Mazes/circle_maze2.png')
resized_image = cv2.resize(image, (224, 224))

# Convert image to numpy array
image_array = np.asarray(resized_image, dtype=np.float32)

# Expand dimensions to match the model input shape
image_array = np.expand_dims(image_array, axis=0)

prediction = model.predict(image_array)

index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print the result
print(f"Class: {class_name}, Confidence: {confidence_score}")
