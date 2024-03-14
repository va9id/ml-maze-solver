import numpy as np
import heapq
from constants import WALL    

def astar(grid, start, end):
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
    path = []
    current_node = end
    while current_node != start:
        path.append(current_node)
        current_node = came_from[current_node]
    path.append(start)
    path.reverse()
    return path

def heuristic(a, b, grid):
    # Manhattan distance heuristic while accounting for distance to nearest wall
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + 0.9 * distance_to_nearest_wall(b, grid)


def distance_to_nearest_wall(node, grid):
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
    row, col = node
    potential_neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
    valid_neighbors = [neighbor for neighbor in potential_neighbors if is_valid(grid, neighbor)]
    return valid_neighbors

def is_valid(grid, node):
    rows, cols = grid.shape
    row, col = node
    return 0 <= row < rows and 0 <= col < cols and grid[row, col] != WALL  # Check if the node is within bounds and not a wall

