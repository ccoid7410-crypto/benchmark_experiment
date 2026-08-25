"""Surrounding and memory-map helpers for MACI agents."""

from .agent_support import *


class AgentPerceptionMixin:
    def get_surroundings(self):
        """
        Inspects reachable surroundings up to vision_range by Manhattan distance.
        Records static terrain in self.known_map.
        Returns a descriptive text and a list of visible agents.
        """
        from collections import deque

        x, y = self.pos
        grid = self.model.map_data
        target = self.model.target_pos
        
        self.known_map[(x, y)] = '.'
        
        descriptions = []
        visible_agents = []
        visible_open = {(x, y)}
        visible_walls = set()
        queue = deque([((x, y), 0)])
        visited = {(x, y)}
        directions = [("North", 0, -1), ("South", 0, 1), ("West", -1, 0), ("East", 1, 0)]

        while queue:
            (cx, cy), dist = queue.popleft()
            if dist >= self.vision_range:
                continue

            for _, dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if nx < 0 or nx >= self.model.width or ny < 0 or ny >= self.model.height:
                    continue

                tile_symbol = self._tile_symbol_at((nx, ny))
                if grid[ny][nx] == 1 and tile_symbol != 'G':
                    self.known_map[(nx, ny)] = '#'
                    visible_walls.add((nx, ny))
                    continue
                if grid[ny][nx] == 1 and tile_symbol == 'G':
                    self.known_map[(nx, ny)] = 'G'
                    visible_walls.add((nx, ny))
                    dist = abs(nx - x) + abs(ny - y)
                    descriptions.append(f"Symbol 'G' visible at ({nx}, {ny}), Manhattan distance {dist}, currently blocking movement")
                    continue

                if (nx, ny) in visited:
                    continue

                visited.add((nx, ny))
                visible_open.add((nx, ny))
                self.known_map[(nx, ny)] = tile_symbol or '.'
                queue.append(((nx, ny), dist + 1))

        for nx, ny in visible_open:
            dist = abs(nx - x) + abs(ny - y)
            sym = self._tile_symbol_at((nx, ny))
            if sym:
                self.known_map[(nx, ny)] = sym
                descriptions.append(f"Symbol '{sym}' visible at ({nx}, {ny}), Manhattan distance {dist}")

            cellmates = self.model.grid.get_cell_list_contents([(nx, ny)])
            other_agents = [a for a in cellmates if a != self]
            for oa in other_agents:
                visible_agents.append((oa, (nx, ny)))
                descriptions.append(f"Agent '{oa.unique_id}' visible at ({nx}, {ny}), Manhattan distance {dist}")

        open_count = len(visible_open) - 1
        wall_count = len(visible_walls)
        descriptions.insert(
            0,
            f"Visible area: {open_count} open tile(s) and {wall_count} wall tile(s) within Manhattan range {self.vision_range}. Walls block visibility expansion."
        )

        return "\n".join(descriptions), visible_agents

    def get_memory_map(self, visible_agents):
        """
        Generates a text grid representation of the currently known area, centered on the agent.
        Top is North, Bottom is South, Left is West, Right is East.
        """
        cx, cy = self.pos
        agent_positions = {pos: str(a.unique_id)[0] for a, pos in visible_agents}
        
        if not self.known_map:
            # Fallback if known_map is somehow empty
            v = self.vision_range
            min_x, max_x = cx - v, cx + v
            min_y, max_y = cy - v, cy + v
        else:
            min_x = min(x for x, y in self.known_map.keys())
            max_x = max(x for x, y in self.known_map.keys())
            min_y = min(y for x, y in self.known_map.keys())
            max_y = max(y for x, y in self.known_map.keys())
            
            # Ensure the agent's current position and a small buffer is always visible
            min_x = min(min_x, cx - 3)
            max_x = max(max_x, cx + 3)
            min_y = min(min_y, cy - 3)
            max_y = max(max_y, cy + 3)

        map_lines = []
        for y in range(min_y, max_y + 1):
            row = []
            for x in range(min_x, max_x + 1):
                if (x, y) == (cx, cy):
                    row.append('@')
                elif (x, y) in agent_positions:
                    row.append(agent_positions[(x, y)])
                elif (x, y) in self.known_map:
                    row.append(self.known_map[(x, y)])
                else:
                    row.append(' ') # Use space for unexplored to make the map cleaner
            map_lines.append("".join(row)) # No spaces between characters makes the map more compact and readable
            
        return "\n".join(map_lines)
