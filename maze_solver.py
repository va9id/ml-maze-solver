from typing import List
import numpy as np
import cv2
from pathfinding import astar
from constants import PATHWAY, WALL, START, END, BLACK_PIXEL 



def get_binary_image(image: cv2.typing.MatLike):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(grayscale_image, 1, 255, cv2.THRESH_BINARY)
    return binary_image

def find_start_and_end(image: cv2.typing.MatLike):
    binary_image = get_binary_image(image)
    cropped_image = crop_image(binary_image)
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
        
def find_offset(image: cv2.typing.MatLike):
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

def crop_image(image):
    top_left, top_right, bottom_right = find_offset(image)
    y_start, y_end = top_left[0], bottom_right[0]
    x_start, x_end = top_left[1], top_right[1]
    return image[y_start:y_end, x_start:x_end]
        

def find_maze_size(binary_image: cv2.typing.MatLike):
    rows, cols = binary_image.shape
    top_row, bottom_row = binary_image[0, :], binary_image[rows - 1, :]
    left_col, right_col = binary_image[:, 0], binary_image[:, cols - 1]
    
    top_white = [j for j, pixel_value in enumerate(top_row) if pixel_value == PATHWAY]
    bottom_white = [j for j, pixel_value in enumerate(bottom_row) if pixel_value == PATHWAY]
    left_white = [i for i, pixel_value in enumerate(left_col) if pixel_value == PATHWAY]
    right_white = [i for i, pixel_value in enumerate(right_col) if pixel_value == PATHWAY]  

    return max([len(top_white), len(bottom_white), len(left_white), len(right_white)])

def add_start_end_to_maze(maze: cv2.typing.MatLike) -> List[tuple]:
    rows, cols = maze.shape
    top_row, bottom_row = maze[0, :], maze[rows - 1, :]
    left_col, right_col = maze[:, 0], maze[:, cols - 1]
    
    top_white = [j for j, pixel_value in enumerate(top_row) if pixel_value == PATHWAY]
    bottom_white = [j for j, pixel_value in enumerate(bottom_row) if pixel_value == PATHWAY]
    left_white = [i for i, pixel_value in enumerate(left_col) if pixel_value == PATHWAY]
    right_white = [i for i, pixel_value in enumerate(right_col) if pixel_value == PATHWAY]       

    
    arr = [top_length, bottom_length, left_length, right_length] = [len(top_white), len(bottom_white), len(left_white), len(right_white)]
    arr.sort()
    opening1, opening2 = arr[-1], arr[-2]
    if opening1 == 0 or opening2 == 0:
        raise Exception("There must be an entrance and exit in the maze!")
    
    result = []
    added_start = False
    if opening1 == top_length or opening2 == top_length:
        starting_col = (top_white[0] + top_white[-1]) // 2
        maze[0, starting_col] = START
        added_start = True
        result.append((0, starting_col))
    if opening1 == left_length or opening2 == left_length:
        starting_row = (left_white[0] + left_white[-1]) // 2
        maze[starting_row, 0] = END if added_start else START
        added_start = True
        result.append((starting_row, 0))
    if opening1 == bottom_length or opening2 == bottom_length:
        starting_col = (bottom_white[0] + bottom_white[-1]) // 2
        maze[rows-1, starting_col] = END if added_start else START
        added_start = True
        result.append((rows-1, starting_col))
    if opening1 == right_length or opening2 == right_length:
        starting_row = (right_white[0] + right_white[-1]) // 2
        maze[starting_row, cols-1] = END if added_start else START
        added_start = True    
        result.append((starting_row, cols-1))

    return result

def find_path(image: cv2.typing.MatLike):
    gray_img = crop_image(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    
    #convert image to black and white
    gray_img[gray_img > 100] = 255
    gray_img[gray_img <= 100] = 0

    
    rows, cols = gray_img.shape

    maze = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            maze[i,j] = WALL if gray_img[i,j] == BLACK_PIXEL else PATHWAY
    
    start, end = add_start_end_to_maze(maze)
    path = set(astar(maze, start, end))

    #Output result onto image
    output_image = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    PATH_THICKNESS = 5 # 5 pixel path thickness
    for i in range(rows):
        for j in range(cols):
            for k in range(PATH_THICKNESS):
                if maze[i,j] != WALL and ((i, j) in path or (i+k, j) in path or (i-k, j) in path or (i, j+k) in path or (i, j-k) in path):
                    output_image[i, j] = [255, 0, 0]
                    break
    
    return output_image


img = cv2.imread('Mazes/maze6.jpg')
output = find_path(img)
cv2.imwrite("output.jpg", output)