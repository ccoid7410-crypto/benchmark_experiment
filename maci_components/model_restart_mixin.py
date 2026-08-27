"""Agent reset and reflection restart orchestration for MACI models."""

from .agent_support import *


class ModelRestartMixin:
    def reset_agents_for_next_round(self, restart_positions, agents_to_reset):
        self.success_debrief_done = False
        self.success_debrief = {
            "discussion": [],
            "keywords": [],
            "communication_improvements": list(getattr(self, "communication_improvements", [])),
            "current_communication_guideline": getattr(self, "current_communication_guideline", ""),
            "token_usage": self.token_summary()
        }
        self.communication_log = []

        self._restore_static_layout()
        self.placed_agent_positions = []
        for agent in agents_to_reset:
            self.grid.remove_agent(agent)
            agent.messages = []
            agent.action_history = []
            agent.is_done = False
            agent.known_map = {}
            agent.inbox = []
            agent.total_tokens = 0
            agent.prompt_tokens = 0
            agent.completion_tokens = 0
            agent.reasoning_tokens = 0
            agent.cached_tokens = 0
            agent.last_token_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "cached_tokens": 0,
                "total_tokens": 0
            }
            agent.token_history = []
            agent.turns = 0
            agent.landmarks = {}
            agent.frontier_memory = []
            agent.blocked_positions = []
            agent.inventory = []
            agent.turns_since_broadcast = 0
            agent.consecutive_blocked_turns = 0
            agent.active_auto_move = None

        reusable_positions = (
            len(restart_positions) >= len(agents_to_reset)
            and len(set(restart_positions[:len(agents_to_reset)])) == len(agents_to_reset)
        )
        if reusable_positions:
            for agent, start_pos in zip(agents_to_reset, restart_positions):
                x, y = start_pos
                if not (0 <= x < self.width and 0 <= y < self.height):
                    reusable_positions = False
                    break
                if self.map_data[y][x] == 1 and start_pos not in getattr(self, "gates", []):
                    reusable_positions = False
                    break

        if reusable_positions:
            for agent, start_pos in zip(agents_to_reset, restart_positions):
                self.grid.place_agent(agent, start_pos)
                self.placed_agent_positions.append(start_pos)
            self.update_gates()
        else:
            empty_spaces = self.map_generator.get_empty_spaces()
            if self.target_pos in empty_spaces:
                empty_spaces.remove(self.target_pos)
            self._place_agents(None, empty_spaces)

    def reflect_and_restart_session(self):
        reflection_context = self.build_reflection_context()
        print("> [SYSTEM] Reflection consultation starting...")
        discussion = self.run_reflection_consultation(reflection_context, max_rounds=2)
        print("> [SYSTEM] Reflection synthesis starting...")
        synthesis = self.synthesize_reflection_prompt_patch(reflection_context, discussion)
        self.apply_reflection_prompt_patch(synthesis)
        consultation_record = {
            "session": len(getattr(self, "reflection_consultations", [])) + 1,
            "context": reflection_context,
            "discussion": discussion,
            "synthesis": synthesis,
            "token_usage": self.token_summary()
        }
        self.reflection_consultations.append(consultation_record)
        self.strategy_improvements.append({
            "session": len(self.strategy_improvements) + 1,
            "type": "reflection_synthesis",
            "summary": synthesis.get("summary", ""),
            "global_system_patch": synthesis.get("global_system_patch", ""),
            "agent_patches": synthesis.get("agent_patches", {}),
            "communication_guideline": synthesis.get("communication_guideline", "")
        })
        self.log("\n--- REFLECTION CONSULTATION ---\n" + json.dumps(consultation_record, ensure_ascii=False, indent=2))

        restart_positions = list(getattr(self, "placed_agent_positions", []))
        agents_to_reset = list(self.agents)
        self.reset_agents_for_next_round(restart_positions, agents_to_reset)
