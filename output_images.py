import os
from constants import *
from maze_generator import generate_random_maze_image
from torchvision import datasets, transforms
import cv2

def output_maze_images(num_images):
    if not os.path.isdir(MAZE_FOLDER_NAME): 
        os.mkdir(MAZE_FOLDER_NAME)
    for i in range(num_images):
       cv2.imwrite(os.path.join(MAZE_FOLDER_NAME, f'maze_{i+1}.jpg'), generate_random_maze_image()) 


def output_non_maze_images(num_images, train):
    if not os.path.isdir(NON_MAZE_FOLDER_NAME): 
        os.mkdir(NON_MAZE_FOLDER_NAME)

    # Download dataset to use to train non-mazes
    cifar_dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transforms.ToTensor())

    count = 0
    for (image, label) in cifar_dataset:
        if count >= num_images:
            break

        image_path = os.path.join(NON_MAZE_FOLDER_NAME, f'non_maze_{count+1}.jpg')
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype('uint8')
        cv2.imwrite(image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        count += 1