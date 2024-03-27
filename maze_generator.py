from constants import *
import numpy as np
import cv2
import random
from typing import Tuple, List

class Cell:
    def __init__(self, x, y, top_wall=True, bottom_wall=True, left_wall=True, right_wall=True, visited=False):
        self.x = x
        self.y = y
        self.top_wall = top_wall
        self.bottom_wall = bottom_wall
        self.left_wall = left_wall
        self.right_wall = right_wall
        self.visited = False

def remove_random_wall(cells: Cell, excluded_cell=None) -> Tuple:
    '''
    Removes a random wall from the maze cells.

    Parameters:
    cells: The 2D list representing the maze cells.
    excluded_cell: A tuple representing the coordinates of a cell to exclude

    Returns:
    Tuple: a tuple of the cell from which the wall was removed
    '''
    rows, cols = len(cells), len(cells[0])
    removed_cell = None
    random_number = random.randint(1, 4)
    
    # pick start on top cells
    if random_number == 1:
        random_col = random.randint(0, cols-1)
        removed_cell = (0, random_col)
        if removed_cell != excluded_cell: cells[0 , random_col].top_wall = False
    # pick start on bottom cells
    elif random_number == 2:
        random_col = random.randint(0, cols-1)
        removed_cell = (rows-1, random_col)
        if removed_cell != excluded_cell: cells[rows-1 , random_col].bottom_wall = False
    # pick start on right cells
    elif random_number == 3:
        random_row = random.randint(0, rows-1)
        removed_cell = (random_row, cols-1)
        if removed_cell != excluded_cell: cells[random_row , cols-1].right_wall = False
    # pick start on left cells
    else:
        random_row = random.randint(0, rows-1)
        removed_cell = (random_row, 0)
        if removed_cell != excluded_cell: cells[random_row , 0].left_wall = False

    return removed_cell

def pick_start_and_end(cells):
    start, end = remove_random_wall(cells), None
    while (end is None) or (end == start):
        end = remove_random_wall(cells, start)

    return start, end

def generate_pathways(cells: List, start: Tuple) -> None:
    '''
    Generates pathways using randomized DFS algorithm
    
    Parameters:
    cells: a 2D list representing the maze cells.
    start: a tuple representing the starting coordinates in the maze
    '''
    rows, cols = len(cells), len(cells[0])
    current_cell = cells[start[0], start[1]]
    current_cell.visited = True
    stack = [current_cell]

    while len(stack) != 0:
        x, y = current_cell.x, current_cell.y
        unvisited_neighbors = []

        if (x > 0) and (not cells[y, x-1].visited):
            unvisited_neighbors.append(cells[y, x-1])
        if (x < cols-1) and (not cells[y, x+1].visited):
            unvisited_neighbors.append(cells[y, x+1])
        if (y > 0) and (not cells[y-1, x].visited):
            unvisited_neighbors.append(cells[y-1, x])
        if (y < rows-1) and (not cells[y+1, x].visited):
            unvisited_neighbors.append(cells[y+1, x])

        if len(unvisited_neighbors) != 0:
            random.shuffle(unvisited_neighbors) # randomize order
            next_cell = unvisited_neighbors[0]
            if current_cell.x + 1 == next_cell.x: 
                current_cell.right_wall = False
                next_cell.left_wall = False
            elif current_cell.x - 1 == next_cell.x: 
                current_cell.left_wall = False
                next_cell.right_wall = False
            elif current_cell.y + 1 == next_cell.y: 
                current_cell.bottom_wall = False
                next_cell.top_wall = False
            else:
                current_cell.top_wall = False
                next_cell.bottom_wall = False
        
            next_cell.visited = True
            stack.append(next_cell)
            current_cell = next_cell

        else:
            stack.pop() # current cell has no unvisited neighbors so remove from stack
            current_cell = None if len(stack) == 0 else stack[-1] # Go back to previous cell

def generate_maze_image(cols, rows, cell_size, line_thickness, padding_size=0) -> cv2.typing.MatLike:
    '''
    Generates a maze image based on the specified parameters.

    Parameters:
    cols: number of columns in the maze.
    rows: number of rows in the maze.
    cell_size: size of each maze cell in pixels.
    line_thickness: thickness of maze walls in pixels.
    padding_size: size of padding to add around the maze image. Defaults to 0.

    Returns:
    image: the maze image generated
    '''
    # Create numpy array of Cell objects
    cells = np.array([[Cell(x, y) for x in range(cols)] for y in range(rows)])
    start, _ = pick_start_and_end(cells)
    generate_pathways(cells, start)

    image_size = (rows * cell_size + line_thickness, cols * cell_size + line_thickness, 1) # Define grayscale image based on image and cell size
    image = np.full(image_size, WHITE_PIXEL)

    # Draw pathways and walls on the image
    for cell_row in cells:
        for cell in cell_row:
            x1, y1 = cell.x * cell_size + (line_thickness//2), cell.y * cell_size + (line_thickness//2)
            x2, y2 = x1 + cell_size, y1 + cell_size

            if cell.top_wall:
                cv2.line(image, (x1, y1), (x2, y1), BLACK_PIXEL, line_thickness)
            if cell.bottom_wall:
                cv2.line(image, (x1, y2), (x2, y2), BLACK_PIXEL, line_thickness)
            if cell.left_wall:
                cv2.line(image, (x1, y1), (x1, y2), BLACK_PIXEL, line_thickness)
            if cell.right_wall:
                cv2.line(image, (x2, y1), (x2, y2), BLACK_PIXEL, line_thickness)

    image = cv2.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=WHITE_PIXEL)
    return image

def generate_random_maze_image() -> cv2.typing.MatLike:
    '''
    Generates a random maze image.

    Returns:
    image: the random generated maze image
    '''
    cols = random.randint(6, 20)
    rows = random.randint(6, 20)
    cell_size = random.randint(10, 25)
    line_thickness = random.randint(3, 5)
    padding = random.randint(0, 10)
    return generate_maze_image(cols, rows, cell_size, line_thickness, padding)
