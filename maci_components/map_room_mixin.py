"""Room and tunnel generation for generated maps."""

import random
import sys


class MapRoomMixin:
    def _make_room(self):
        """
        Room generation algorithm for a dungeon-like map.
        Creates random rooms and connects them with tunnels.
        """
        # Determine the number of rooms based on difficulty
        count_map = {'E': 2, 'M': random.randint(3, 4), 'H': random.randint(5, 7)}
        count = count_map[self.diff]
        
        # Subdivide the map into sections to distribute rooms evenly
        if count <= 2:
            cols, rows = random.choice([(2, 1), (1, 2)])
        elif count <= 4:
            cols, rows = (2, 2)
        else:
            cols, rows = (3, 3)

        cell_w, cell_h = self.width // cols, self.height // rows
        all_cells = [(c, r) for c in range(cols) for r in range(rows)]
        selected_cells = random.sample(all_cells, count)
        
        self.rooms = [] # Stores room coordinates: (rx, ry, rw, rh)

        for c, r in selected_cells:
            cell_x, cell_y = c * cell_w, r * cell_h
            
            # Attempt to create a room within the section (max 15 attempts)
            for _ in range(15):
                # Apply variance to room dimensions
                if random.random() < 0.3:
                    rw = random.randint(int(cell_w * 0.7), int(cell_w * 1.1))
                    rh = random.randint(int(cell_h * 0.7), int(cell_h * 1.1))
                else:
                    rw = random.randint(max(3, int(cell_w * 0.5)), int(cell_w * 0.8))
                    rh = random.randint(max(3, int(cell_h * 0.5)), int(cell_h * 0.8))

                rx = cell_x + random.randint(1, max(1, cell_w - rw - 1))
                ry = cell_y + random.randint(1, max(1, cell_h - rh - 1))

                # Check if the new room overlaps with existing ones (with a +1 buffer)
                is_ok = True
                for (ox, oy, ow, oh) in self.rooms:
                    if not (rx + rw + 1 < ox or rx > ox + ow + 1 or 
                            ry + rh + 1 < oy or ry > oy + oh + 1):
                        is_ok = False
                        break
                
                if is_ok:
                    # Carve out the room (set cells to 0)
                    for y in range(ry, ry + rh):
                        for x in range(rx, rx + rw):
                            if 0 < x < self.width - 1 and 0 < y < self.height - 1:
                                self.grid[y][x] = 0

                    # Connect the new room to the previous one via a tunnel
                    curr_center = (rx + rw // 2, ry + rh // 2)
                    if self.rooms:
                        prev_rx, prev_ry, prev_rw, prev_rh = self.rooms[-1]
                        prev_center = (prev_rx + prev_rw // 2, prev_ry + prev_rh // 2)
                        self._create_tunnel(*prev_center, *curr_center)
                        
                    self.rooms.append((rx, ry, rw, rh))
                    break

    def _create_tunnel(self, x1, y1, x2, y2):
        """Creates an L-shaped tunnel connecting two coordinates."""
        for x in range(min(x1, x2), max(x1, x2) + 1):
            self.grid[y1][x] = 0
        for y in range(min(y1, y2), max(y1, y2) + 1):
            self.grid[y][x2] = 0
