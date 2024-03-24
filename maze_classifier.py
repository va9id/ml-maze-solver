import cv2
import os
from constants import *
from model_creator import ConvNet
from output_images import output_maze_images, output_non_maze_images
import torch
import torch.nn.functional as F
from torchvision import transforms

class MazeClassifier():
    def __init__(self):
        if not os.path.exists(os.path.join(MODEL_FOLDER_NAME, f'{MODEL_FILE_NAME}')):
            raise Exception("You must train the model before classifying an image (run model_creator.py)!")
        model_path = os.path.join(MODEL_FOLDER_NAME, f'{MODEL_FILE_NAME}')
        model_state_dict = torch.load(model_path)
        self.model = ConvNet()
        self.model.load_state_dict(model_state_dict)
        self.model.eval()  # Set model to evaluation mode
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((64, 64)), transforms.ToTensor()])

    def predict_image_class(self, image: cv2.typing.MatLike) -> int:
        '''
        Predicts class (maze of non-maze) of the image
        '''
        image_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        return predicted_class

    def is_maze(self, image: cv2.typing.MatLike) -> bool:
        '''
        Checks if the image is a maze
        '''
        return self.predict_image_class(image) == MAZE_LABEL

def test_classifier(num_images):
    '''
    Tests the maze classifier
    '''
    output_maze_images(num_images)
    output_non_maze_images(num_images, False)
    classifier = MazeClassifier()
    print("------Testing With Mazes------")
    num_mazes_failed = 0
    for filename in os.listdir(MAZE_FOLDER_NAME):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image = cv2.imread(os.path.join(MAZE_FOLDER_NAME, filename))
            is_maze = classifier.is_maze(image)
            if not is_maze: num_mazes_failed += 1
            print(f"{filename} is maze: {is_maze}")

    print(f"Number of mazes incorrectly classified: {num_mazes_failed}/{num_images}")
    print("------Testing With Non-Mazes------")
    num_non_mazes_failed = 0
    for filename in os.listdir(NON_MAZE_FOLDER_NAME):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image = cv2.imread(os.path.join(NON_MAZE_FOLDER_NAME, filename))
            is_maze = classifier.is_maze(image)
            if is_maze: num_non_mazes_failed += 1
            print(f"{filename} is maze: {is_maze}")
    
    print(f"Number of non-mazes incorrectly classified: {num_non_mazes_failed}/{num_images}")

if __name__ == "__main__":
    test_classifier(100)