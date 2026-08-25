"""Agent placement and reachability helpers for MACI models."""

from .agent_support import *
from .maci_agent import MACI_Agent


class ModelPlacementMixin:
    def _place_agents(self, agent_configs, empty_spaces):
        # Determine rooms if available
        available_rooms = list(self.map_generator.rooms) if hasattr(self.map_generator, 'rooms') and self.map_generator.rooms else []
        random.shuffle(available_rooms)
        
        # Filter out rooms that contain the target 'F'
        target_rooms = []
        tx, ty = self.target_pos
        for r in list(available_rooms):
            rx, ry, rw, rh = r
            if rx <= tx < rx + rw and ry <= ty < ry + rh:
                target_rooms.append(r)
                available_rooms.remove(r)

        for i in range(self.num_agents):
            # ... (config and agent initialization) ...
            if hasattr(self, 'agents') and len(self.agents) > i:
                agent = self.agents[i]
                config = {
                    "vision_range": agent.vision_range,
                    "speed_limit": agent.speed_limit
                }
            else:
                config = agent_configs[i] if i < len(agent_configs) else {
                    "model_name": "gpt-4o-mini",
                    "vision_range": 5,
                    "speed_limit": 1,
                    "byte_limit": 500,
                    "map_share_radius": 0
                }
                agent = MACI_Agent(self, config, self.thinking_effort)


            start_pos = None
            
            # Try to pick from a unique room that DOES NOT have the target 'F'
            if available_rooms:
                room = available_rooms.pop()
                rx, ry, rw, rh = room
                room_spaces = [(x, y) for y in range(ry, ry + rh) for x in range(rx, rx + rw) if (x, y) in empty_spaces]
                if room_spaces:
                    start_pos = random.choice(room_spaces)
            elif target_rooms:
                # If only target rooms are left, pick a spot furthest from 'F' in one of them
                room = random.choice(target_rooms)
                rx, ry, rw, rh = room
                room_spaces = [(x, y) for y in range(ry, ry + rh) for x in range(rx, rx + rw) if (x, y) in empty_spaces]
                if room_spaces:
                    room_spaces.sort(key=lambda p: abs(p[0]-tx) + abs(p[1]-ty), reverse=True)
                    start_pos = random.choice(room_spaces[:max(1, len(room_spaces)//3)])

            
            # Fallback to distance-based spawning if no rooms left or not Room map
            if not start_pos:
                reachable_nodes = self.get_reachable_starts(self.target_pos)
                valid_starts = {pos: dist for pos, dist in reachable_nodes.items() if pos in empty_spaces}

                
                if valid_starts:
                    max_dist = max(valid_starts.values())
                    threshold = max(15, int(max_dist * 0.7))
                    far_starts = [pos for pos, dist in valid_starts.items() if dist >= threshold]
                    if not far_starts: far_starts = list(valid_starts.keys())
                    
                    if self.placed_agent_positions:
                        far_starts.sort(key=lambda p: min(abs(p[0]-pa[0]) + abs(p[1]-pa[1]) for pa in self.placed_agent_positions), reverse=True)
                        start_pos = random.choice(far_starts[:max(1, len(far_starts)//4)])
                    else:
                        start_pos = random.choice(far_starts)
                else:
                    if empty_spaces:
                        empty_spaces.sort(key=lambda p: abs(p[0]-self.target_pos[0]) + abs(p[1]-self.target_pos[1]), reverse=True)
                        start_pos = random.choice(empty_spaces[:max(1, len(empty_spaces)//10)])

            if start_pos:
                self.grid.place_agent(agent, start_pos)
                if start_pos in empty_spaces:
                    empty_spaces.remove(start_pos)
                self.placed_agent_positions.append(start_pos)

    def get_reachable_starts(self, target):
        from collections import deque
        
        # All directions are now allowed
        reverse_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]


        queue = deque([(target[0], target[1], 0)])
        visited = {target}
        reachable_nodes = {}
        
        while queue:
            cx, cy, dist = queue.popleft()
            reachable_nodes[(cx, cy)] = dist
            
            for dx, dy in reverse_dirs:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) not in visited and self.map_data[ny][nx] == 0:
                        visited.add((nx, ny))
                        queue.append((nx, ny, dist + 1))
                        
        return reachable_nodes
