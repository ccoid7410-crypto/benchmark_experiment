"""Basic grid setup and queries for map generation."""

import random
import sys


class MapGridMixin:
    def _fill_map(self):
        """Fills the entire grid with walls (1)."""
        self.grid = [[1] * self.width for _ in range(self.height)]

    def get_empty_spaces(self):
        """Returns a list of all coordinates (x, y) that are not walls (value == 0)."""
        tmp = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 0:
                    tmp.append((x, y))
        return tmp

    def get_random_map(self, difficulty=None):
        """Generates a map with a completely random seed and optional difficulty."""
        t = random.randint(1, 2)
        s = random.randint(1000, 9999)
        d = difficulty if difficulty in ['E', 'M', 'H'] else random.choice(['E', 'M', 'H'])
        self.generate_map(f'{t}{s}{d}')
