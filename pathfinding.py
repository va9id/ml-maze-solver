import numpy as np
import heapq
from constants import WALL    

def astar(grid, start, end):
    '''
    Finds the shortest path from the start to the end using the A* algorithm

    Parameters:
    grid: a list representing the grid/map where each cell is either passable (0) or impassable (1)
    start: a tuple representing the starting coordinates (row, column) in the grid
    end: a tuple representing the ending coordinates (row, column) in the grid

    Returns:
    List: a list of tuples representing the path from the start to the end in the grid (if found)
    '''
    open_set = []  # Priority queue to store nodes to be explored
    heapq.heappush(open_set, (0, start))  # Initial node with cost 0
    came_from = {}  # Dictionary to store parent of each node
    cost_so_far = {start: 0}  # Dictionary to store cost to reach each node

    while open_set:
        _, current_node = heapq.heappop(open_set)

        if current_node == end:
            path = reconstruct_path(came_from, start, end)
            return path

        for neighbor in neighbors(grid, current_node):
            new_cost = cost_so_far[current_node] + 1  # Assuming all moves have a cost of 1

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(end, neighbor, grid)
                heapq.heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current_node

    return None  # If no path is found

def reconstruct_path(came_from, start, end):
    '''
    Reconstructs the path from the start to the end based on the parent pointers.

    Parameters:
    came_from: a dictionary mapping each node to its parent node.
    start: a tuple representing the starting coordinates (row, column) in the grid.
    end: a tuple representing the ending coordinates (row, column) in the grid.

    Returns:
    List: a list of tuples representing the path from the start to the end in the grid.
    '''
    path = []
    current_node = end
    while current_node != start:
        path.append(current_node)
        current_node = came_from[current_node]
    path.append(start)
    path.reverse()
    return path

def heuristic(a, b, grid):
    '''
    Calculates the heuristic (estimated cost) from node 'a' to node 'b' 
    using the Manhattan distance

    Parameters:
    a: a tuple representing the coordinates (row, column) of node 'a'
    b: a tuple representing the coordinates (row, column) of node 'b'
    grid: a list representing the grid/map where each cell is either passable (0) or impassable (1)

    Returns:
    float: the estimated cost (heuristic) from node 'a' to node 'b'
    '''
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) #+ 3 * distance_to_nearest_wall(b, grid)

def distance_to_nearest_wall(node, grid):
    '''
    Finds the distance to the nearest wall from the given node in the grid

    Parameters:
    node: a tuple representing the pixel coordinate (row, column) in the grid
    grid: a 2D numpy array representing the grid of pixels around the node

    Returns:
    int: the distance to the nearest wall from the given node
    '''
    rows, cols = grid.shape
    row, col = node

    # Calculate distance to the nearest wall in each direction
    distances = [
        col,             # Left
        rows - 1 - col,  # Right
        row,             # Up
        cols - 1 - row   # Down
    ]

    return min(distances)

def neighbors(grid, node):
    '''
    Finds the valid neighboring nodes of the given node within the grid

    Parameters:
    grid: a 2D numpy array representing the grid/map
    node: a tuple representing the pixel coordinate (row, column) of the node

    Returns:
    List: a list of tuples representing the valid neighboring nodes of the given node
    '''
    row, col = node
    potential_neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
    valid_neighbors = [neighbor for neighbor in potential_neighbors if is_valid(grid, neighbor)]
    return valid_neighbors

def is_valid(grid, node):
    '''
    Checks if the given node is within the bounds of the grid and is not a wall

    Parameters:
    grid: a 2D numpy array representing the grid/map
    node: a tuple representing the pixel coordinate (row, column) of the node

    Returns:
    bool: True if the node is within bounds and not a wall, False otherwise
    '''
    rows, cols = grid.shape
    row, col = node
    return 0 <= row < rows and 0 <= col < cols and grid[row, col] != WALL  # Check if the node is within bounds and not a wall

