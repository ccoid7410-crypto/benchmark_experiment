"""Maze carving algorithm for generated maps."""

import random
import sys


class MapMazeMixin:
    def _make_maze(self):
        """
        Maze generation algorithm using Depth-First Search (DFS) / Recursive Backtracker.
        """
        # Starting point for the maze generation
        sx, sy = 1, 1
        self.grid[sy][sx] = 0
        
        queue = [(sx, sy)]
        # Possible directions to carve paths: (dx, dy) jumping by 2 cells
        d = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        
        # Start carving paths
        while queue:
            cx, cy = queue[-1]
            
            # Shuffle directions to ensure a random maze structure
            random.shuffle(d)
            tmp = False
            for dx, dy in d:
                nx, ny = cx + dx, cy + dy
                
                # Check boundaries to prevent IndexErrors
                if 0 < nx < self.width - 1 and 0 < ny < self.height - 1:
                    # If the target cell is a wall, carve a path through to it
                    if self.grid[ny][nx] == 1:
                        # Carve the middle cell
                        self.grid[cy + dy // 2][cx + dx // 2] = 0
                        # Carve the destination cell
                        self.grid[ny][nx] = 0
                        
                        queue.append((nx, ny))
                        tmp = True
                        break
            # If no valid directions are found, backtrack
            if not tmp:
                queue.pop()
