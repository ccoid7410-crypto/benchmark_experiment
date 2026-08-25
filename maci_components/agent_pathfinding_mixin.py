"""Pathfinding tool helpers for MACI agents."""

from .agent_support import *


class AgentPathfindingMixin:
    def _run_dijkstra_tool(self, target_coord):
        from collections import deque
        
        queue = deque([(self.pos, [])])
        visited = {self.pos}
        
        allowed_moves = [
            ("UP", 0, -1),
            ("DOWN", 0, 1),
            ("LEFT", -1, 0),
            ("RIGHT", 1, 0)
        ]

        
        while queue:
            (cx, cy), path = queue.popleft()
            
            if (cx, cy) == target_coord and (cx, cy) != self.pos:
                if len(path) > 0:
                    first_moves = " -> ".join(path[:3])
                    return f"Found target {target_coord} at distance {len(path)}. Shortest path starts with: {first_moves}"
                    
            for action, dx, dy in allowed_moves:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited:
                    if (nx, ny) in self.known_map and self.known_map[(nx, ny)] != '#':
                        visited.add((nx, ny))
                        queue.append(((nx, ny), path + [action]))
                    elif (nx, ny) == target_coord:
                        visited.add((nx, ny))
                        queue.append(((nx, ny), path + [action]))
                        
        return f"No path is available to {target_coord} using your known map and allowed directions."
