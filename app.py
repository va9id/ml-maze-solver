import cv2
import tkinter as tk
from tkinter import filedialog
from maze_solver import find_path
from maze_classifier import MazeClassifier

def open_file_dialog():
    file_path = filedialog.askopenfilename()  
    return file_path

def main():
    mazeClassifier = MazeClassifier()
    root = tk.Tk()
    root.withdraw() 

    while True:
        image_path = open_file_dialog()
        if not image_path:
            print("No image selected, exiting!")
            return
        try:
            img = cv2.imread(image_path)
            if mazeClassifier.is_maze(img):
                find_path(img)
            else:
                print("The uploaded image does not look like a maze!")
        except Exception as error:
            print(error)
            print("The uploaded maze could not be solved!")


if __name__ == '__main__':
    main()