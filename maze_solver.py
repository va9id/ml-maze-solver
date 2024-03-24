import numpy as np
import cv2
from pathfinding import astar
from constants import PATHWAY, WALL, START, END, BLACK_PIXEL, WHITE_PIXEL, MAZE_WINDOW_NAME
from typing import List, Tuple

def display_image_with_delay(image: cv2.typing.MatLike, delay=5) -> None:
    '''
    Displays the maze being solved
    '''
    cv2.imshow(MAZE_WINDOW_NAME, image)
    cv2.waitKey(delay)

def get_binary_image(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    '''
    Converts image into a binary image

    Parameters:
    image: the image to convert a binary image

    Returns:
    image: binary image with pixel
    '''
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(grayscale_image, 1, 255, cv2.THRESH_BINARY)
    return binary_image

def find_inner_contours(gray_image: cv2.typing.MatLike) -> List:
    '''
    Finds the inner contours of an image
    
    Parameters:
    gray_image: the gray scale image to find inner contours for

    Returns:
    List: the inner contours of the gray scale image
    '''
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    inner_contours = []
    for i, h in enumerate(hierarchy[0]):
        if h[3] == -1:  # Contour has no parent, it is outermost
            continue
        if hierarchy[0][h[3]][3] == -1:  # Parent has no parent, contour is innermost
            inner_contours.append(contours[i])
    
    return inner_contours

def crop_image_using_contours(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    '''
    Crops the image based on the contours of the image

    Parameters:
    image: the image to crop

    Returns:
    image: the cropped image from its contours
    '''
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = gray_img.shape
    inner_contours = find_inner_contours(gray_img)

    min_i, min_j, max_i, max_j = np.inf, np.inf, -1, -1
    for contour in inner_contours:
        for point in contour[:, 0]:  # Iterate over all points in the contour
            i, j = point
            min_i = min(min_i, i)
            min_j = min(min_j, j) 
            max_i = max(max_i, i)
            max_j = max(max_j, j)        

    # If image does not need to be cropped
    if min_j == np.inf: min_j = -1
    if min_i == np.inf: min_i = -1
    if max_j == -1: max_j = rows
    if max_i == -1: max_i = cols

    cropped_image = image[min_j+1:max_j-1, min_i+1:max_i-1]
    return cropped_image

def find_offset(image: cv2.typing.MatLike) -> Tuple:
    '''
    Finds the offset values from a border to the actual maze

    Parameters:
    image: the image to crop

    Returns:
    Tuple: the offsets for the top left, right and bottom right
    '''
    rows, cols = image.shape
    top_left, top_right, bottom_right = None, None, None
    
    for i in range(rows):
        for j in range(cols):
            if (image[i,j] == BLACK_PIXEL) and (top_left is None):
                top_left = i, j
                top_right = i, np.max(np.where(image[i] == BLACK_PIXEL))
                bottom_right = np.max(np.where(image[:, j] == BLACK_PIXEL)), top_right[1]
    
    if top_left is None or top_right is None or bottom_right is None: raise Exception("Error cropping image!")
    return top_left, top_right, bottom_right

def crop_image(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    '''
    Crops the image based on the the value of pixels. Note that the image must already 
    be a black and white image and that this method will not work if the image has a 
    black border

    Parameters:
    image: the image to crop

    Returns:
    image: the cropped image from offsets
    '''
    top_left, top_right, bottom_right = find_offset(image)
    y_start, y_end = top_left[0], bottom_right[0]
    x_start, x_end = top_left[1], top_right[1]
    return image[y_start:y_end, x_start:x_end]

def find_maze_size(binary_image: cv2.typing.MatLike) -> int:
    '''
    Finds the size (thickness) between the walls of the maze. Note that this method
    only works when the maze has exactly one entrance, one exit, and has a uniform 
    size throughout the maze.

    Parameters:
    binary_image: maze image to find thickness of opening for

    Returns:
    int: the thickness of opening/path for the maze
    '''
    rows, cols = binary_image.shape
    top_row, bottom_row = binary_image[0, :], binary_image[rows - 1, :]
    left_col, right_col = binary_image[:, 0], binary_image[:, cols - 1]
    
    top_white = [j for j, pixel_value in enumerate(top_row) if pixel_value == PATHWAY]
    bottom_white = [j for j, pixel_value in enumerate(bottom_row) if pixel_value == PATHWAY]
    left_white = [i for i, pixel_value in enumerate(left_col) if pixel_value == PATHWAY]
    right_white = [i for i, pixel_value in enumerate(right_col) if pixel_value == PATHWAY]  

    arr = [len(top_white), len(bottom_white), len(left_white), len(right_white)]
    arr.sort()
    opening1, opening2 = arr[-1], arr[-2]
    if opening2 == 0:
        return opening1 // 2 # Both entrance and exit are on same side
    
    return opening1

def add_start_end_to_maze(maze: cv2.typing.MatLike) -> List[tuple]:
    '''
    Adds the start and end locations to the maze image using their respective constant values
    and returns their values

    Parameters:
    maze: the maze image to add the start and end to

    Returns:
    List: the start and end pixel locations
    '''
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
    
    # Check for special case where both opening and exit are on same side of maze
    if opening1 != 0 and opening2 == 0:
        if opening1 == top_length:
            for k in range(1, len(top_white)):
                # This means there is a gap between white pixels on the top
                if top_white[k] != top_white[k-1] + 1:
                    starting_col = (top_white[0] + top_white[k-1]) // 2
                    ending_col = (top_white[k] + top_white[-1]) // 2
                    maze[0, starting_col] = START
                    maze[0, ending_col] = END
                    return [(0, starting_col), (0, ending_col)] 
        elif opening1 == bottom_length:
            for k in range(1, len(bottom_white)):
                # This means there is a gap between white pixels on the bottom
                if bottom_white[k] != bottom_white[k-1] + 1:
                    starting_col = (bottom_white[0] + bottom_white[k-1]) // 2
                    ending_col = (bottom_white[k] + bottom_white[-1]) // 2
                    maze[rows-1, starting_col] = START
                    maze[rows-1, ending_col] = END
                    return [(rows-1, starting_col), (rows-1, ending_col)] 
        elif opening1 == left_length:
            for k in range(1, len(left_white)):
                # This means there is a gap between white pixels on the left
                if left_white[k] != left_white[k-1] + 1:
                    starting_row = (left_white[0] + left_white[k-1]) // 2
                    ending_row = (left_white[k] + left_white[-1]) // 2
                    maze[starting_row, 0] = START
                    maze[ending_row, 0] = END
                    return [(starting_row, 0), (ending_row, 0)]
        elif opening1 == right_length:
            for k in range(1, len(right_white)):
                # This means there is a gap between white pixels on the right
                if right_white[k] != right_white[k-1] + 1:
                    starting_row = (right_white[0] + right_white[k-1]) // 2
                    ending_row = (right_white[k] + right_white[-1]) // 2
                    maze[starting_row, cols-1] = START
                    maze[ending_row, cols-1] = END
                    return [(starting_row, cols-1), (ending_row, cols-1)]

    if opening1 == 0 or opening2 == 0:
        raise Exception("There must be an entrance and exit in the maze!")
    if arr[-3] != 0:
        raise Exception("There should only be a single entrance and exit in the maze!")
    
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

def convert_to_black_and_white(gray_img: cv2.typing.MatLike) -> None:
    '''
    Converts a grayscale image to black and white based on its contours
    '''
    pass

def resize_image(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    '''
    Resizes maze image 

    Parameters:
    image: the maze image to resize

    Returns:
    image: the resized image
    '''
    resized_width = 600
    resized_height = 400
    original_height, original_width = image.shape[:2]

    # Calculate the scaling factor for both dimensions
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height
    scale = min(scale_x, scale_y)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    return cv2.resize(image, (new_width, new_height))

def find_path(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    '''
    Finds the path from the start to end of the maze using the A* 
    search algorithm and displays the path on the output image

    Parameters:
    image: the maze image to find the path for

    Returns:
    image: an output image with sovled maze path
    '''
    image = resize_image(image)
    image = crop_image_using_contours(image)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #convert image to black and white
    gray_img[gray_img > 100] = WHITE_PIXEL
    gray_img[gray_img <= 100] = BLACK_PIXEL
    
    rows, cols = gray_img.shape

    maze = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            maze[i,j] = WALL if gray_img[i,j] == BLACK_PIXEL else PATHWAY
    
    PATH_THICKNESS = int(find_maze_size(maze) * 0.5) # Using 50% of path width
    start, end = add_start_end_to_maze(maze)
    path = astar(maze, start, end)

    if path is None:
        raise Exception("No path has been found!")

    #Output result onto image
    output_image = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    for i, j in path:
        # Set a range of pixels around each path point
        for x in range(i - PATH_THICKNESS//2, i + PATH_THICKNESS//2 + 1):
            for y in range(j - PATH_THICKNESS//2, j + PATH_THICKNESS//2 + 1):
                # Check if the coordinates are within the image boundaries
                if 0 <= x < rows and 0 <= y < cols and maze[x,y] != WALL:
                    output_image[x, y] = [255, 0, 0]

        display_image_with_delay(output_image)
        
    return output_image
