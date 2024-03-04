import numpy as np
import cv2
from keras.models import load_model

class RectangleMazeClassifier():
    def __init__(self):
        # The threshold confidence for what is considered a maze (i.e. 70% confidence)
        self.threshold = 0.7

        # Load the model and the labels
        self.model = load_model("keras_maze_classifier/keras_model.h5", compile=False)
        self.class_names = open("keras_maze_classifier/labels.txt", "r").readlines()

        # The index of the maze label
        self.maze_label_index = 0


    def is_maze(self, image: cv2.typing.MatLike) -> bool:
        # Resize image and convert it to numpy array for prediction
        resized_image = cv2.resize(image, (224, 224))
        image_array = np.asarray(resized_image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        prediction = self.model.predict(image_array)
        index = np.argmax(prediction)
        confidence_score = prediction[0][index]

        return (index == self.maze_label_index) and (confidence_score >= self.threshold)
    