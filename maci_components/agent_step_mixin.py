"""Turn execution logic for MACI agents."""

from .agent_support import *


class AgentStepMixin:
    def step(self):
        """
        The core action loop executed every simulation step.
        The agent perceives the environment, asks the LLM, and moves.
        """
        if self.is_done:
            return

        self.turns += 1
        decision_start_pos = self.pos
        
        target = getattr(self.model, 'target_pos', (-1,-1))
        
        # Check if already on target before making a move
        if self.pos == target:
            print(f"\n[FOUND] [Agent {self.unique_id} - {self.model_name}] has found the Target 'F'!!!")
            self.is_done = True
            return
            
        # Determine what the agent is currently standing on
        standing_on = self._tile_symbol_at(self.pos) or '.'
        pre_move_events = self._apply_tile_interactions()
        if pre_move_events:
            standing_on = self._tile_symbol_at(self.pos) or '.'

        # --- Map Sharing Processing ---
        for other in self.model.agents:
            if other != self and other.map_share_radius > 0:
                ox, oy = other.pos
                for (kx, ky), tile in other.known_map.items():
                    dist = max(abs(kx - ox), abs(ky - oy))
                    if dist <= other.map_share_radius:
                        self.known_map[(kx, ky)] = tile

        if not self.messages:
            system_prompt = self._build_system_prompt()
            if self.optimization_mode:
                self._remember_optimization_base_prompt()
                system_prompt += f"\n[Optimization Mode Active]\nGuidelines: {self.prompt_addition}\n"
            
            self.messages.append({"role": "system", "content": system_prompt})

        surroundings, visible_agents = self.get_surroundings()
        
        # --- Automated Structured Memory Update ---
        self.structured_memory["path"] = self.action_history[-20:]
        x, y = self.pos
        
        # Update landmarks from known_map
        for (kx, ky), val in self.known_map.items():
            if val not in ['.', '#', '@', ' '] and not val.isdigit():
                self.structured_memory["landmarks"][val] = [kx, ky]

        # Track exploration frontiers: known open/symbol tiles adjacent to unknown space.
        frontier_candidates = []
        for (kx, ky), val in self.known_map.items():
            if val == '#':
                continue
            if (kx, ky) in self.structured_memory["blocked_pos"]:
                continue
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                neighbor = (kx + dx, ky + dy)
                if neighbor not in self.known_map:
                    dist = abs(kx - x) + abs(ky - y)
                    frontier_candidates.append((dist, kx, ky))
                    break
        frontier_candidates.sort()
        self.structured_memory["frontier_memory"] = [
            [kx, ky] for _, kx, ky in frontier_candidates[:12]
        ]

        # Get immediate neighbors
        neighbor_data = []
        for name, dx, dy in [("UP (North)", 0, -1), ("DOWN (South)", 0, 1), ("LEFT (West)", -1, 0), ("RIGHT (East)", 1, 0)]:
            nx, ny = x + dx, y + dy
            block_reason = self._movement_block_reason((nx, ny))
            if 0 <= nx < self.model.width and 0 <= ny < self.model.height:
                tile = self.known_map.get((nx, ny), " ")
                status = "OPEN" if not block_reason else block_reason
                neighbor_data.append(f"- {name}: '{tile}' -> {status}")
            else:
                neighbor_data.append(f"- {name}: 'OUT OF BOUNDS' -> {block_reason}")
        neighbor_str = "\n".join(neighbor_data)


        ascii_map = self.get_memory_map(visible_agents)

        actions_str = " -> ".join(self.action_history[-10:]) if self.action_history else "Just started"
        last_move = self.action_history[-1] if self.action_history else "None (Just started)"

        coded_communication = getattr(self.model, "coded_communication", False)
        allowed_code_class = "".join(re.escape(code) for code in getattr(self.model, "allowed_broadcast_codes", DEFAULT_BROADCAST_CODES))
        allowed_code_list = "/".join(getattr(self.model, "allowed_broadcast_codes", DEFAULT_BROADCAST_CODES))
        symbol_space_notes = self.symbol_space_prompt.strip() or "No extra experimenter-defined symbol-space notes."
        communication_space_str = (
            json.dumps(self.structured_memory.get("communication_space", {}), ensure_ascii=False)
            if coded_communication
            else "Disabled for this run; use natural-language broadcasts."
        )

        if coded_communication:
            for inbox_item in self.inbox:
                for msg_code in re.findall(rf"\b([{allowed_code_class}])(?:10|[0-9])?\b", inbox_item.upper()):
                    self.structured_memory["communication_space"].setdefault(
                        msg_code,
                        f"Received from partner; infer meaning from context."
                    )
        inbox_str = "\n".join(self.inbox) if self.inbox else "No messages."
        self.inbox.clear()

        # Prepare memory history for prompt
        mem_hist_str = "\n".join([f"- {m}" for m in self.memory_history[-10:]]) if self.memory_history else "No previous memory notes."
        struct_mem_str = json.dumps(self.structured_memory, indent=2)


        # Highlight visible agents in text

        other_agent_info = ""
        if visible_agents:
            other_agent_info = "VISIBLE AGENTS NEARBY: " + ", ".join([f"Agent {a.unique_id} at {p}" for a, p in visible_agents])
        else:
            other_agent_info = "No other agents currently visible."

        interaction_rules_str = json.dumps(getattr(self.model, "interaction_rules", {}))

        if coded_communication:
            communication_instruction = (
                f"Use broadcast_message only for confirmed allowed-code events ({allowed_code_list}). "
                f"broadcast_message is REQUIRED every turn and MUST be one base code only. Valid: N, S, G, H, F, K, D, X. Invalid: empty messages, codes with digits, or natural-language sentences. "
                f"If no cooperative event is active, send N to report navigation/frontier status. "
                f"Numeric suffixes are disabled; do not append numbers to any code. "
                f"Prefer S when holding switch support, G when staged at a gate, H when requesting partner help, and F when the finish is reachable but support must continue. "
                f"Always include structured_memory.communication_space. If you send or interpret any base code, add or update that exact key with a short local meaning. "
            )
        else:
            communication_instruction = (
                "Coded communication is disabled. Do not use numbered compact codes. "
                "broadcast_message may be an empty string or one short natural-language status for partners. "
                "Use plain language for switch/gate/exit/help status when it matters. "
                "structured_memory.communication_space is optional and should not be used for code slots. "
            )

        user_prompt = (
            f"=== CURRENT STATUS ===\n"
            f"Position: {self.pos}\n"
            f"Standing on: '{standing_on}'\n"
            f"MOVEMENT RULES:\n"
            f" - SPEED LIMIT: Up to {self.speed_limit} blocks per turn.\n\n"
            f"=== IMMEDIATE NEIGHBORS (1-step) ===\n{neighbor_str}\n\n"
            f"=== DATA & MEMORY ===\n"
            f"Structured Memory: {json.dumps(self.structured_memory)}\n"
            f"Frontier Memory: {json.dumps(self.structured_memory['frontier_memory'])}\n"
            f"Allowed Broadcast Codes: {allowed_code_list}\n"
            f"Known Communication Space: {communication_space_str}\n"
            f"Experimenter Symbol Space Notes:\n{symbol_space_notes}\n"
            f"Interaction Rules: {interaction_rules_str}\n"
            f"Previous Strategy Notes: {self.memory}\n\n"
            f"=== PERCEPTION ===\n"
            f"Partner/Other Info: {other_agent_info}\n"
            f"Recent Path: {actions_str}\n"
            f"Inbox: {inbox_str}\n"
            f"Memory Map (@=YOU):\n{ascii_map}\n\n"
            f"--- FINAL INSTRUCTION ---\n"
            f"Choose the best next move now. Return only JSON with keys: "
            f"'reason', 'action', 'blocks', 'broadcast_message', 'notes', and 'structured_memory'. "
            f"Do not choose any direction marked BLOCKED in IMMEDIATE NEIGHBORS. Treat BLOCKED directions as unavailable until map/gate/key state changes. "
            f"Use frontier_memory for exploration when no target/interactive tile is urgent. "
            f"{communication_instruction}"
            f"Keep 'reason' concise and tactical; do not include hidden chain-of-thought."
        )






        
        turn_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0
        }

        while True:
            MAX_TURNS = 5
            pairs = self.messages[1:]

            if len(pairs) > MAX_TURNS * 2:
                pairs = pairs[-(MAX_TURNS * 2):]

            current_call = [self.messages[0]] + pairs + [{"role": "user", "content": user_prompt}]

            raw_answer = ""
            raw_answer_full = ""
            reasoning_content = ""
            for attempt in range(3):
                try:
                    kwargs = {
                        "model": self.model_name,
                        "messages": current_call,
                    }
                    if getattr(self.model, "json_response_format_supported", True):
                        kwargs["response_format"] = {"type": "json_object"}
                    if getattr(self.model, "reasoning_effort_supported", True) and self.thinking_effort in ["low", "medium", "high"]:
                        kwargs["reasoning_effort"] = self.thinking_effort

                    response = self.llm_client.chat.completions.create(**kwargs)
                    message = response.choices[0].message
                    raw_answer_full = (message.content or "").strip()
                    provider_reasoning = str(getattr(message, "reasoning_content", "") or getattr(message, "reasoning", "") or "").strip()

                    usage_record = {}
                    if hasattr(response, 'usage') and response.usage:
                        usage_record = self._record_token_usage(response.usage, source="turn")
                        merge_token_usage(turn_token_usage, usage_record)

                    if raw_answer_full:
                        inline_reasoning, cleaned_answer = extract_reasoning_and_answer(raw_answer_full)
                        reasoning_content = "\n\n".join(part for part in [provider_reasoning, inline_reasoning] if part)
                        raw_answer = cleaned_answer.replace("```json", "").replace("```", "").strip()

                        if hasattr(self.model, "log_llm_io"):
                            self.model.log_llm_io({
                                "agent": self.model._agent_label(self) if hasattr(self.model, "_agent_label") else str(self.unique_id),
                                "agent_id": str(self.unique_id),
                                "model": self.model_name,
                                "provider": getattr(self.model, "provider", ""),
                                "turn": self.turns,
                                "attempt": attempt + 1,
                                "messages_sent": current_call,
                                "raw_response": raw_answer_full,
                                "reasoning_content": reasoning_content,
                                "cleaned_response": raw_answer,
                                "token_usage": usage_record,
                            })

                        if raw_answer:
                            break
                        else:
                            print(f"> [Agent {self.unique_id}] [WARNING] Empty response after removing reasoning tags. Retrying... ({attempt + 1}/3)")
                    else:
                        print(f"> [Agent {self.unique_id}] [WARNING] Empty response. Retrying... ({attempt + 1}/3)")

                except Exception as e:
                    print(f"> [Agent {self.unique_id}] Communication Error: {e}")
                    if attempt == 2:
                        raw_answer = '{"reason": "API Error", "action": "UP", "memory": "Error"}'
                        raw_answer_full = raw_answer_full or raw_answer

            if not raw_answer:
                return

            action = "UNKNOWN"
            reason = "No reason provided"
            new_notes = ""
            new_msg = ""
            message_meaning = ""
            parsed_data = {}

            try:
                # Find outermost braces to handle nested JSON
                first_brace = raw_answer.find('{')
                last_brace = raw_answer.rfind('}')
                if first_brace != -1 and last_brace != -1:
                    clean_answer = raw_answer[first_brace:last_brace+1]
                else:
                    clean_answer = raw_answer
                
                parsed_data = json.loads(clean_answer)


                if "tool_call" in parsed_data:
                    tool_call = parsed_data["tool_call"]
                    if tool_call.get("name") == "dijkstra":
                        target_x = tool_call.get("target_x")
                        target_y = tool_call.get("target_y")
                        if target_x is not None and target_y is not None:
                            try:
                                target_coord = (int(target_x), int(target_y))
                            except ValueError:
                                target_coord = self.pos # Fallback if not integers
                                
                            tool_result = self._run_dijkstra_tool(target_coord)
                            
                            print(f"> [Agent {self.unique_id} ({self.model_name})] [TOOL] Used Tool Dijkstra to search for {target_coord}. Result: {tool_result}")
                            
                            # Add interaction to messages
                            self.messages.append({"role": "user", "content": user_prompt})
                            self.messages.append({"role": "assistant", "content": clean_answer})
                            
                            # Set new prompt for the second pass
                            user_prompt = (
                                f"Tool Result for Dijkstra {target_coord}: {tool_result}\n"
                                f"Now provide the final JSON with keys: 'reason', 'action', "
                                f"'blocks', 'broadcast_message', 'notes', and 'structured_memory'. "
                                f"{communication_instruction}"
                                f"Keep 'reason' concise."
                            )
                            continue # Loop again to get action

                action = parsed_data.get("action", "UNKNOWN").upper()
                reason = parsed_data.get("reason", "No reason provided")
                new_notes = parsed_data.get("notes", "")
                new_msg = self._normalize_broadcast_message(parsed_data.get("broadcast_message", ""))
                memory_update = parsed_data.get("structured_memory", {})
                if isinstance(memory_update, dict):
                    private_codebook = memory_update.get("private_codebook")
                    if isinstance(private_codebook, dict):
                        reserved_codes = set(getattr(self.model, "allowed_broadcast_codes", DEFAULT_BROADCAST_CODES))
                        for code, meaning in private_codebook.items():
                            code = str(code).strip()[:1]
                            if code and code not in reserved_codes:
                                self.structured_memory["private_codebook"][code] = str(meaning)[:80]
                    communication_space = memory_update.get("communication_space")
                    if coded_communication and isinstance(communication_space, dict):
                        for code, meaning in communication_space.items():
                            normalized_code = self._normalize_broadcast_message(code)
                            if normalized_code:
                                self.structured_memory["communication_space"][normalized_code] = str(meaning)[:100]

                self.memory = new_notes
                self.memory_history.append(new_notes)
                # Keep all history, do not pop.




                log_msg = f"\n--- [Agent {self.unique_id} Turn] ---\n"
                log_msg += f"Model: {self.model_name}\n"
                log_msg += f"Position: {self.pos}\n"
                log_msg += f"Reasoning: {reason}\n"
                log_msg += f"Action: {action}\n"
                log_msg += f"Token Usage: {json.dumps(turn_token_usage)}\n"
                log_msg += f"Memory (Notes): {new_notes}\n"
                log_msg += f"Structured Memory: {json.dumps(self.structured_memory)}\n"
                if new_msg:
                    log_msg += f"Broadcast: {new_msg}\n"
                self.model.log(log_msg)

                if new_msg:
                    if coded_communication:
                        # Symbolic communication: standard code plus optional numeric slot.
                        if re.match(r"^[A-Z]$", new_msg) and new_msg not in self.structured_memory["communication_space"]:
                            self.structured_memory["communication_space"][new_msg] = (
                                f"Base symbol used from context: {new_notes or reason}"
                            )
                        self.structured_memory["communication_space"].setdefault(
                            new_msg,
                            f"Sent from context: {new_notes or reason}"
                        )
                        message_meaning = self.structured_memory["communication_space"].get(new_msg, "")
                    else:
                        message_meaning = new_msg
                    sender_label = self.model._agent_label(self) if hasattr(self.model, "_agent_label") else str(self.unique_id)
                    recipients = []
                    for other in self.model.agents:
                        if other != self:
                            other_label = self.model._agent_label(other) if hasattr(self.model, "_agent_label") else str(other.unique_id)
                            recipients.append(other_label)
                            other.inbox.append(f"Agent {sender_label}: {new_msg}")
                    if hasattr(self.model, "communication_log"):
                        self.model.communication_log.append({
                            "model_step": len(self.model.communication_log) + 1,
                            "agent_turn": self.turns,
                            "from": sender_label,
                            "to": recipients,
                            "message": new_msg,
                            "meaning": message_meaning,
                            "position": list(self.pos),
                            "reason": reason,
                            "notes": new_notes
                        })
                else:
                    message_meaning = ""


            except json.JSONDecodeError:
                found_actions = re.findall(r"\b(UP|DOWN|LEFT|RIGHT|NORTH|SOUTH|EAST|WEST)\b", raw_answer.upper())
                found_reasons = re.findall(r'"reason":\s*"(.*?)"', raw_answer)
                if found_actions:
                    action = found_actions[-1]
                    reason = found_reasons[-1] if found_reasons else "Extracted via Regex Fallback"
                else:
                    action = "UP"
                    reason = "Fallback"
                    
            if action == "NORTH": action = "UP"
            elif action == "SOUTH": action = "DOWN"
            elif action == "WEST": action = "LEFT"
            elif action == "EAST": action = "RIGHT"

            break # Exit loop when valid action is parsed

        print(f"\n>>> Model Step: Agent {self.unique_id} ({self.model_name})")
        if new_msg:
            print(f"[BROADCAST]: \"{new_msg}\"")
        print(f"[POS] Position: {self.pos}")
        print(f"[THINK] Reasoning: {reason}")
        print(f"[ACTION] Action: {action}")
        print(f"[MEMO] Notes: {new_notes}")

        print("-" * 30)

        self.messages.append({"role": "user", "content": user_prompt})


        self.messages.append({"role": "assistant", "content": raw_answer})

        x, y = self.pos
        new_pos = self.pos
        # Multi-block movement logic
        try:
            blocks = int(parsed_data.get("blocks", 1))
        except:
            blocks = 1
        blocks = max(1, min(blocks, self.speed_limit))

        dx, dy = 0, 0
        if action == "UP": dy = -1
        elif action == "DOWN": dy = 1
        elif action == "LEFT": dx = -1
        elif action == "RIGHT": dx = 1

        blocked_feedback = ""
        blocked_coords = None
        blocked_reason = ""

        if dx != 0 or dy != 0:
            for _ in range(blocks):
                nx, ny = new_pos[0] + dx, new_pos[1] + dy
                block_reason = self._movement_block_reason((nx, ny))
                if not block_reason:
                    new_pos = (nx, ny)
                else:
                    blocked_coords = (nx, ny)
                    blocked_reason = block_reason
                    break

        if new_pos != self.pos:
            self.model.grid.move_agent(self, new_pos)
            self.action_history.append(f"{action} x{blocks}")
            
            interaction_events = self._apply_tile_interactions()

            # Clear blocked list if we successfully moved
            if self.pos in self.structured_memory["blocked_pos"]:
                self.structured_memory["blocked_pos"].remove(self.pos)
        else:
            interaction_events = []
            if blocked_coords is None:
                if action == "UP": blocked_coords = (x, y - 1)
                elif action == "DOWN": blocked_coords = (x, y + 1)
                elif action == "LEFT": blocked_coords = (x - 1, y)
                elif action == "RIGHT": blocked_coords = (x + 1, y)
                if blocked_coords is not None:
                    blocked_reason = self._movement_block_reason(blocked_coords)
            blocked_reason = blocked_reason or "BLOCKED: unavailable move"
            self.action_history.append(f"Tried {action}(Blocked)")
            block_coords = None
            if action == "UP": block_coords = (x, y - 1)
            elif action == "DOWN": block_coords = (x, y + 1)
            elif action == "LEFT": block_coords = (x - 1, y)
            elif action == "RIGHT": block_coords = (x + 1, y)
            
            if block_coords and block_coords not in self.structured_memory["blocked_pos"]:
                self.structured_memory["blocked_pos"].append(block_coords)
                # Keep only last 10 blocked spots to save space
                if len(self.structured_memory["blocked_pos"]) > 10:
                    self.structured_memory["blocked_pos"].pop(0)
            if blocked_coords is not None:
                blocked_feedback = (
                    f"BLOCKED FEEDBACK: {action} is unavailable from {decision_start_pos} "
                    f"toward {blocked_coords}: {blocked_reason}. Choose another OPEN direction next turn."
                )
                self.memory = blocked_feedback
                self.memory_history.append(blocked_feedback)
                self.model.log(f"> [Agent {self.unique_id}] {blocked_feedback}")

        self.last_decision = {
            "turn": self.turns,
            "reason": reason,
            "action": action,
            "blocks": blocks,
            "notes": new_notes,
            "broadcast_message": new_msg,
            "broadcast_meaning": message_meaning if new_msg else "",
            "raw_response": raw_answer_full or raw_answer,
            "reasoning_content": reasoning_content,
            "token_usage": turn_token_usage,
            "cumulative_token_usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "cached_tokens": self.cached_tokens,
                "total_tokens": self.total_tokens
            },
            "start_position": list(decision_start_pos),
            "end_position": list(self.pos),
            "blocked": new_pos == decision_start_pos,
            "blocked_reason": blocked_feedback,
            "interactions": pre_move_events + interaction_events,
            "inventory": list(self.structured_memory["inventory"])
        }
