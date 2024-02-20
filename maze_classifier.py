import numpy as np
import cv2
from keras.models import load_model

class MazeHelper():
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
    
    def get_binary_image(self, image: cv2.typing.MatLike):
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(grayscale_image, 1, 255, cv2.THRESH_BINARY)
        return binary_image

    def find_start_and_end(self, image: cv2.typing.MatLike):
        binary_image = self.get_binary_image(image)
        cropped_image = self.crop_image(binary_image)
        rows, cols = cropped_image.shape
        top_row, bottom_row = cropped_image[0, :], cropped_image[rows - 1, :]
        left_col, right_col = cropped_image[:, 0], cropped_image[:, cols - 1]
        
        top_white = [j for j, pixel_value in enumerate(top_row) if pixel_value == 255]
        bottom_white = [j for j, pixel_value in enumerate(bottom_row) if pixel_value == 255]
        left_white = [i for i, pixel_value in enumerate(left_col) if pixel_value == 255]
        right_white = [i for i, pixel_value in enumerate(right_col) if pixel_value == 255]       

        
        arr = [top_length, bottom_length, left_length, right_length] = [len(top_white), len(bottom_white), len(left_white), len(right_white)]
        arr.sort()
        opening1, opening2 = arr[-1], arr[-2]
        if opening1 == 0 or opening2 == 0:
            raise Exception("There must be an entrance and exit in the maze!")
        
        result = []
        if opening1 == top_length or opening2 == top_length:
            result.append(((0, top_white[0]), (0, top_white[-1])))
        if opening1 == bottom_length or opening2 == bottom_length:
            result.append(((rows - 1, bottom_white[0]), (rows - 1, bottom_white[-1])))
        if opening1 == left_length or opening2 == left_length:
            result.append(((left_white[0], 0), (left_white[-1], 0)))
        if opening1 == right_length or opening2 == right_length:
            result.append(((right_white[0], cols - 1), (right_white[-1], cols - 1)))

        # cropped_image[result[0][0][0]:result[0][1][0], result[0][0][1]:result[0][1][1]+10] = 0
        # cropped_image[result[1][0][0]:result[1][1][0], result[1][0][1]-10:result[1][1][1]] = 0 
        # cv2.imwrite("TEST.jpg", cropped_image)
        
        return result
        
    def find_offset(self, image: cv2.typing.MatLike):
        # image = self.get_binary_image(image)
        rows, cols = image.shape
        top_left, top_right, bottom_right = None, None, None
        
        for i in range(rows):
            for j in range(cols):
                if (image[i,j] == 0) and (top_left is None):
                    top_left = i, j
                    top_right = i, np.max(np.where(image[i] == 0)[0])
                    bottom_right = np.max(np.where(image[j] == 0)[0]), top_right[1]
        
        return top_left, top_right, bottom_right

    def crop_image(self, image):
        top_left, top_right, bottom_right = self.find_offset(image)
        y_start, y_end = top_left[0], bottom_right[0]
        x_start, x_end = top_left[1], top_right[1]
        return image[y_start:y_end, x_start:x_end]
        
        

helper = MazeHelper()
img = cv2.imread('Mazes/maze6.jpg')
# result = helper.find_start_and_end(img)
# print(result)