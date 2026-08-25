"""Grid map generation for MACI simulations."""

import random
from .map_grid_mixin import MapGridMixin
from .map_maze_mixin import MapMazeMixin
from .map_room_mixin import MapRoomMixin
from .map_render_mixin import MapRenderMixin


class MapGenerator(
    MapGridMixin,
    MapMazeMixin,
    MapRoomMixin,
    MapRenderMixin,
):
    """
    A class dedicated to generating various types of grid-based maps (mazes, rooms).
    0 represents empty space (paths), 1 represents walls.
    """
    def __init__(self, height: int, width: int):
        # Set default dimensions for the map
        self.height = height
        self.width = width
        self.grid = []
        self.rooms = [] # List of (rx, ry, rw, rh)
  
    def generate_map(self, seed: str):
        """
        Generates a map based on a provided seed string.
        Seed format example: '12345H'
        - index 0: Type (1 for Maze, 2 for Room)
        - index 1-4: Random Seed (e.g., 2345)
        - index 5: Difficulty (E, M, H)
        """
        self.type = seed[0]
        self.seed = int(seed[1:5])
        self.diff = seed[5]
        
        # Fix the random seed for reproducibility
        random.seed(self.seed)
        
        # Initialize the map with solid walls
        self._fill_map()
        
        # Generate according to the specified map type
        if self.type == '1':
            self._make_maze()
        if self.type == '2':
            self._make_room()
        
        return self.grid
  
  

  
  

  
  
