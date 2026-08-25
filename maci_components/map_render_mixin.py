"""Terminal rendering helpers for generated maps."""

import random
import sys


class MapRenderMixin:
    def _get_wall_char(self, x: int, y: int) -> str:
        """
        Determines the appropriate Unicode box-drawing character for a wall cell 
        based on its adjacent walls (Up, Down, Left, Right).
        """
        up = y > 0 and self.grid[y - 1][x] == 1
        down = y < self.height - 1 and self.grid[y + 1][x] == 1
        left = x > 0 and self.grid[y][x - 1] == 1
        right = x < self.width - 1 and self.grid[y][x + 1] == 1

        neighbors = (up, down, left, right)
        wall_chars = {
            (True, True, True, True): '┼',
            (True, True, True, False): '┤',
            (True, True, False, True): '├',
            (True, False, True, True): '┴',
            (False, True, True, True): '┬',
            (True, True, False, False): '│',
            (False, False, True, True): '─',
            (False, True, False, True): '┌',
            (False, True, True, False): '┐',
            (True, False, False, True): '└',
            (True, False, True, False): '┘',
            (True, False, False, False): '╵',
            (False, True, False, False): '╷',
            (False, False, True, False): '╴',
            (False, False, False, True): '╶',
            (False, False, False, False): '┼',
        }
        return wall_chars[neighbors]

    def display(self):
        """Prints the generated map to the terminal."""
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')

        print(f"\n[Map Type: {self.type}] | [Diff: {self.diff}] | [Seed: {self.seed}]")
        for y, row in enumerate(self.grid):
            rendered_row = []
            for x, cell in enumerate(row):
                if cell == 1:
                    rendered_row.append(self._get_wall_char(x, y))
                else:
                    rendered_row.append(' ')
            print("".join(rendered_row))
