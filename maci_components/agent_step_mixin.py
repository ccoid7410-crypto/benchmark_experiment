"""Turn execution logic for MACI agents (Code as Policies).

Each turn, the agent LLM writes a short Python snippet (a policy) that is
executed in a locked-down subprocess (see policy_executor.py) to produce a
move/broadcast decision, instead of returning a fixed JSON action. There is
no explicit memory API - the agent's only "memory" is the accumulating
conversation history (self.messages) it can see each turn. Before
committing, the model may call the test_policy_code tool to dry-run drafts
and see errors/results, up to MAX_TOOL_ROUNDS times.

The policy calls auto_move(direction, blocks=1, until=None) instead of a
one-shot move(): with no `until`, it behaves like a single-turn action (the
LLM is asked again next turn) - but with an `until` condition (a boolean
expression string over `state`), the same direction keeps getting applied
automatically on later turns, with no LLM call, until that condition
becomes true or the agent gets blocked - see _run_active_auto_move below.
"""

from .agent_support import *
from .policy_executor import run_policy_code, check_condition

POLICY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "test_policy_code",
        "description": (
            "Dry-run a draft of your policy code without committing to it, so you "
            "can debug before submitting your final answer. Returns the resulting "
            "action/blocks/broadcast/until, or an error if the code raised an exception."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The Python policy code to test."},
            },
            "required": ["code"],
        },
    },
}

SUBMIT_POLICY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_policy_code",
        "description": (
            "Submit your final policy code for this turn. This is the ONLY way to "
            "act - plain text answers are ignored. The code is checked for valid "
            "Python syntax before it is accepted: if it fails to compile, you get "
            "the error back and must call this again with a fix (it is NOT run for "
            "real and does NOT cost you the turn until it compiles)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The final Python policy code for this turn."},
            },
            "required": ["code"],
        },
    },
}

MAX_TOOL_ROUNDS = 4
MAX_MESSAGE_HISTORY_TURNS = 5
DIRECTION_DELTAS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
ZERO_TOKEN_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0, "total_tokens": 0}


def _extract_policy_code(content_text):
    """Pulls just the policy code out of a model's final answer, even when it
    ignores the "just code, no prose" instruction - some models wrap the
    answer in an invented <answer>/<final>/<code> tag, add an explanatory
    paragraph before it, or use a ```python fence; naively exec()-ing the
    whole raw text in those cases is a SyntaxError."""
    text = str(content_text or "").strip()

    tag_match = re.search(
        r"<(?:answer|final_answer|final|code)>(.*?)</(?:answer|final_answer|final|code)>",
        text, flags=re.IGNORECASE | re.DOTALL,
    )
    if tag_match:
        text = tag_match.group(1).strip()

    fence_match = re.search(r"```(?:python)?\s*\n?(.*?)```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    return text.replace("```python", "").replace("```", "").strip()


def _check_compiles(code):
    """Returns None if `code` is valid Python, else a short error string.
    Catching this before ever running the code means leftover stray prose
    (that _extract_policy_code couldn't identify as a wrapper) gets fed back
    to the model as a concrete compile error and retried, instead of being
    silently discarded as a fallback UP move."""
    try:
        compile(code, "<policy>", "exec")
        return None
    except SyntaxError as e:
        return f"{type(e).__name__}: {e}"


class AgentStepMixin:
    def _build_perception_state(self, standing_on, visible_agents, coordination_hint):
        x, y = self.pos
        neighbor_status = {}
        for name, (dx, dy) in DIRECTION_DELTAS.items():
            nx, ny = x + dx, y + dy
            block_reason = self._movement_block_reason((nx, ny))
            if 0 <= nx < self.model.width and 0 <= ny < self.model.height:
                neighbor_status[name] = block_reason or "OPEN"
            else:
                neighbor_status[name] = "OUT OF BOUNDS"

        recent_path = self.action_history[-10:] if self.action_history else []
        other_agent_info = [f"Agent {a.unique_id} at {p}" for a, p in visible_agents]

        return {
            "pos": list(self.pos),
            "standing_on": standing_on,
            "speed_limit": self.speed_limit,
            "neighbors": neighbor_status,
            "frontier_memory": self.frontier_memory,
            "landmarks": self.landmarks,
            "blocked_pos": [list(p) for p in self.blocked_positions],
            "visible_agents": other_agent_info,
            "inbox": list(self.inbox),
            "inventory": list(self.inventory),
            "interaction_rules": getattr(self.model, "interaction_rules", {}),
            "recent_path": recent_path,
            "coordination_hint": coordination_hint,
        }

    def _apply_movement(self, action, blocks, decision_start_pos):
        """Applies one move (already decided, by the LLM or an active
        auto-move plan) and updates blocked-position tracking. Returns
        (interaction_events, blocked_feedback)."""
        x, y = self.pos
        new_pos = self.pos

        if action not in DIRECTION_DELTAS:
            # No usable decision this turn (invalid/failed policy code) -
            # stay put. Distinct from a real BLOCKED direction: nothing was
            # actually chosen, so it shouldn't count against the "stuck at
            # a wall" nudge tracking.
            self.action_history.append("No-op (no valid decision this turn)")
            return [], ""

        dx, dy = DIRECTION_DELTAS[action]

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
            self.consecutive_blocked_turns = 0
            if self.pos in self.blocked_positions:
                self.blocked_positions.remove(self.pos)
        else:
            interaction_events = []
            self.consecutive_blocked_turns += 1
            if blocked_coords is None and (dx != 0 or dy != 0):
                blocked_coords = (x + dx, y + dy)
                blocked_reason = self._movement_block_reason(blocked_coords)
            blocked_reason = blocked_reason or "BLOCKED: unavailable move"
            self.action_history.append(f"Tried {action}(Blocked)")

            if blocked_coords and blocked_coords not in self.blocked_positions:
                self.blocked_positions.append(blocked_coords)
                if len(self.blocked_positions) > 10:
                    self.blocked_positions.pop(0)
            if blocked_coords is not None:
                blocked_feedback = (
                    f"BLOCKED FEEDBACK: {action} is unavailable from {decision_start_pos} "
                    f"toward {blocked_coords}: {blocked_reason}. Choose another OPEN direction next turn."
                )
                self.model.log(f"> [Agent {self.unique_id}] {blocked_feedback}")

        return interaction_events, blocked_feedback

    def _run_active_auto_move(self, agent_label, decision_start_pos, standing_on, visible_agents, pre_move_events):
        """Tries to keep an already-submitted auto_move() plan running without
        calling the LLM. Returns True if it handled this turn (caller should
        return), or False if the plan ended (blocked, condition true/errored,
        or no plan) and a normal LLM turn should run instead."""
        plan = self.active_auto_move
        if not plan:
            return False

        direction = plan["direction"]
        blocks = plan["blocks"]
        until_expr = plan.get("until")
        dx, dy = DIRECTION_DELTAS.get(direction, (0, 0))
        target_pos = (self.pos[0] + dx, self.pos[1] + dy)
        block_reason = self._movement_block_reason(target_pos)

        condition_triggered = False
        if not block_reason and until_expr:
            condition_state = self._build_perception_state(standing_on, visible_agents, "")
            cond = check_condition(until_expr, condition_state, timeout=5)
            if cond.get("error"):
                print(f"> [Agent {agent_label}] [WARNING] auto-move condition errored ({cond['error']}); calling LLM back.")
                condition_triggered = True
            elif cond.get("result"):
                condition_triggered = True

        if block_reason or condition_triggered:
            why = block_reason or "condition met"
            print(f"> [Agent {agent_label}] auto-move plan ended ({why}); calling the LLM back.")
            self.model.log(f"> [Agent {self.unique_id}] auto-move plan ended ({why}); calling the LLM back.")
            self.active_auto_move = None
            return False

        interaction_events, blocked_feedback = self._apply_movement(direction, blocks, decision_start_pos)
        print(f"> [Agent {agent_label}] auto-move {direction} x{blocks} (plan continues, no LLM call)")
        self.turns_since_broadcast += 1

        self.last_decision = {
            "turn": self.turns,
            "reason": f"(auto-move: continuing {direction} until '{until_expr}')",
            "action": direction,
            "blocks": blocks,
            "notes": "",
            "broadcast_message": "",
            "broadcast_meaning": "",
            "raw_response": "",
            "reasoning_content": "",
            "policy_code": f"# auto-move plan active - no LLM call this turn\nauto_move({direction!r}, {blocks}, until={until_expr!r})",
            "policy_error": None,
            "auto_move": True,
            "token_usage": dict(ZERO_TOKEN_USAGE),
            "cumulative_token_usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "cached_tokens": self.cached_tokens,
                "total_tokens": self.total_tokens,
            },
            "start_position": list(decision_start_pos),
            "end_position": list(self.pos),
            "blocked": self.pos == decision_start_pos,
            "blocked_reason": blocked_feedback,
            "interactions": pre_move_events + interaction_events,
            "inventory": list(self.inventory),
        }
        return True

    def _run_policy_plain_loop(self, working_messages, state, turn_token_usage, agent_label):
        """Fallback for providers that don't reliably support OpenAI-style tool
        calling (RELAXED_API_PROVIDERS - typically local llama.cpp/ollama
        setups): there's no tool call to gate on, so the plain-text reply is
        treated as the code directly, compile-checked, and - on failure - fed
        back with the exact error for a retry, within the same round budget
        used by the tool-calling path. Returns (final_code, raw_answer_full,
        reasoning_content, usage_record, messages_sent)."""
        combined_usage = {}
        last_content_text = ""
        last_reasoning = ""

        for round_idx in range(MAX_TOOL_ROUNDS + 1):
            kwargs = {"model": self.model_name, "messages": working_messages}
            if getattr(self.model, "reasoning_effort_supported", True) and self.thinking_effort in ["low", "medium", "high"]:
                kwargs["reasoning_effort"] = self.thinking_effort

            response = self.llm_client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            usage_record = {}
            if hasattr(response, "usage") and response.usage:
                usage_record = self._record_token_usage(response.usage, source="turn")
                merge_token_usage(turn_token_usage, usage_record)
                merge_token_usage(combined_usage, usage_record)

            provider_reasoning = extract_provider_reasoning(message)
            content_text = (message.content or "").strip()
            inline_reasoning, cleaned_content = extract_reasoning_and_answer(content_text)
            round_reasoning = "\n\n".join(part for part in [provider_reasoning, inline_reasoning] if part)
            last_content_text = content_text
            last_reasoning = round_reasoning

            code = _extract_policy_code(cleaned_content)
            compile_error = _check_compiles(code) if code.strip() else "empty response"
            print(f"> [Agent {agent_label} answer]\n{code}")
            if round_reasoning:
                print(f"> [Agent {agent_label} reasoning] {round_reasoning}")

            if compile_error is None:
                return code, last_content_text, last_reasoning, combined_usage, working_messages

            print(f"> [Agent {agent_label}] [round {round_idx + 1}] not valid Python ({compile_error}); asking again with that fed back...")
            working_messages = working_messages + [
                {"role": "assistant", "content": message.content},
                {"role": "user", "content": (
                    f"Your previous answer for this turn was NOT valid, directly-executable Python - "
                    f"it failed to compile: {compile_error}\n"
                    f"Your entire reply must be nothing but Python code (plus an optional single leading "
                    f"'# ' comment) - no prose, no tags, no markdown fences. Try again for the same turn below."
                )},
            ]

        return "", last_content_text, last_reasoning, combined_usage, working_messages

    def _run_policy_debug_loop(self, working_messages, state, turn_token_usage, agent_label):
        """Runs the write-code -> optionally dry-run via test_policy_code ->
        commit via submit_policy_code flow for one attempt. The model is never
        trusted to hand back a direction directly in plain text: the only way
        to act is the submit_policy_code tool call, and it's only accepted
        once its code actually compiles - a syntax error comes back as a tool
        result and the model must call it again. Returns (final_code,
        raw_answer_full, reasoning_content, usage_record, messages_sent).

        Falls back to _run_policy_plain_loop for providers that don't
        reliably support tool calling."""
        if not getattr(self.model, "json_response_format_supported", True):
            return self._run_policy_plain_loop(working_messages, state, turn_token_usage, agent_label)

        combined_usage = {}
        last_content_text = ""
        last_reasoning = ""

        for round_idx in range(MAX_TOOL_ROUNDS + 1):
            kwargs = {"model": self.model_name, "messages": working_messages, "tools": [POLICY_TOOL_SCHEMA, SUBMIT_POLICY_TOOL_SCHEMA]}
            if getattr(self.model, "reasoning_effort_supported", True) and self.thinking_effort in ["low", "medium", "high"]:
                kwargs["reasoning_effort"] = self.thinking_effort

            response = self.llm_client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            usage_record = {}
            if hasattr(response, "usage") and response.usage:
                usage_record = self._record_token_usage(response.usage, source="turn")
                merge_token_usage(turn_token_usage, usage_record)
                merge_token_usage(combined_usage, usage_record)

            provider_reasoning = extract_provider_reasoning(message)
            content_text = (message.content or "").strip()
            inline_reasoning, _cleaned_content = extract_reasoning_and_answer(content_text)
            round_reasoning = "\n\n".join(part for part in [provider_reasoning, inline_reasoning] if part)
            last_content_text = content_text
            last_reasoning = round_reasoning

            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                # No tool call at all - plain text is never accepted as the
                # decision, so nudge the model to actually submit and use up
                # a round.
                print(f"> [Agent {agent_label}] [round {round_idx + 1}] no tool call (ignoring plain text); reminding model to submit_policy_code...")
                if round_reasoning:
                    print(f"> [Agent {agent_label} reasoning] {round_reasoning}")
                working_messages = working_messages + [
                    {"role": "assistant", "content": message.content},
                    {"role": "user", "content": (
                        "That was plain text, which is ignored - you must call the "
                        "submit_policy_code tool with your final code to act this turn "
                        "(or test_policy_code first if you want to dry-run a draft)."
                    )},
                ]
                continue

            print(f"> [Agent {agent_label}] [round {round_idx + 1}] {len(tool_calls)} tool call(s)...")
            if round_reasoning:
                print(f"> [Agent {agent_label} reasoning] {round_reasoning}")
            working_messages = working_messages + [{
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            }]

            accepted_code = None
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                code = _extract_policy_code(str(args.get("code", "")))

                if tc.function.name == "submit_policy_code":
                    compile_error = _check_compiles(code)
                    if compile_error is None:
                        result = {"accepted": True}
                        if accepted_code is None:
                            accepted_code = code
                    else:
                        result = {"accepted": False, "error": f"NOT accepted - {compile_error}. Fix it and call submit_policy_code again."}
                    print(f"> [Agent {agent_label}]   submit -> {result}")
                else:
                    result = run_policy_code(code, state, timeout=10)
                    print(f"> [Agent {agent_label}]   test -> {result}")

                working_messages = working_messages + [{
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }]

            if accepted_code is not None:
                print(f"> [Agent {agent_label} answer]\n{accepted_code}")
                return accepted_code, last_content_text, last_reasoning, combined_usage, working_messages

        return "", last_content_text, last_reasoning, combined_usage, working_messages

    def step(self):
        """
        The core action loop executed every simulation step.
        The agent perceives the environment, writes policy code, and moves -
        or, if an auto_move() plan from an earlier turn is still active,
        keeps running it without calling the LLM (see _run_active_auto_move).
        """
        if self.is_done:
            return

        self.turns += 1
        decision_start_pos = self.pos

        target = getattr(self.model, 'target_pos', (-1, -1))

        if self.pos == target:
            print(f"\n[FOUND] [Agent {self.unique_id} - {self.model_name}] has found the Target 'F'!!!")
            self.is_done = True
            return

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

        surroundings, visible_agents = self.get_surroundings()

        x, y = self.pos

        # Update landmarks from known_map
        for (kx, ky), val in self.known_map.items():
            if val not in ['.', '#', '@', ' '] and not val.isdigit():
                self.landmarks[val] = [kx, ky]

        # Track exploration frontiers: known open/symbol tiles adjacent to unknown space.
        frontier_candidates = []
        for (kx, ky), val in self.known_map.items():
            if val == '#':
                continue
            if (kx, ky) in self.blocked_positions:
                continue
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                neighbor = (kx + dx, ky + dy)
                if neighbor not in self.known_map:
                    dist = abs(kx - x) + abs(ky - y)
                    frontier_candidates.append((dist, kx, ky))
                    break
        frontier_candidates.sort()
        self.frontier_memory = [[kx, ky] for _, kx, ky in frontier_candidates[:12]]

        agent_label = self.model._agent_label(self) if hasattr(self.model, "_agent_label") else str(self.unique_id)

        # --- Auto-move fast path: skip the LLM entirely if a plan is active
        # and still valid this turn. ---
        if self._run_active_auto_move(agent_label, decision_start_pos, standing_on, visible_agents, pre_move_events):
            return

        ascii_map = self.get_memory_map(visible_agents)
        inbox_snapshot = list(self.inbox)
        self.inbox.clear()

        if not self.messages:
            system_prompt = self._build_system_prompt()
            if self.optimization_mode:
                self._remember_optimization_base_prompt()
                system_prompt += f"\n[Optimization Mode Active]\nGuidelines: {self.prompt_addition}\n"
            self.messages.append({"role": "system", "content": system_prompt})

        # --- Coordination nudge: communication is optional every turn, but a
        # persistently silent agent that is also stuck or being messaged gets
        # a nudge instead of being allowed to just never communicate.
        coordination_hint = ""
        if self.consecutive_blocked_turns >= 2 and self.turns_since_broadcast >= 3:
            coordination_hint = (
                f"You have been blocked for {self.consecutive_blocked_turns} turns in a row and haven't "
                f"broadcast anything in {self.turns_since_broadcast} turns - consider whether telling your "
                f"partner would help them unblock you."
            )
        elif inbox_snapshot and self.turns_since_broadcast >= 3:
            coordination_hint = (
                f"Your partner has messaged you but you haven't broadcast anything in "
                f"{self.turns_since_broadcast} turns - consider whether a reply would help coordination."
            )

        state = self._build_perception_state(standing_on, visible_agents, coordination_hint)
        state["inbox"] = inbox_snapshot

        user_prompt = (
            f"=== TURN {self.turns} ===\n"
            f"Memory map (@ = you):\n{ascii_map}\n\n"
            f"state = {json.dumps(state, ensure_ascii=False)}\n\n"
            f"Call submit_policy_code with your policy code now (or call test_policy_code first to debug)."
        )

        turn_token_usage = dict(ZERO_TOKEN_USAGE)

        pairs = self.messages[1:]
        if len(pairs) > MAX_MESSAGE_HISTORY_TURNS * 2:
            pairs = pairs[-(MAX_MESSAGE_HISTORY_TURNS * 2):]
        # Shrinks (dropping the oldest turns first) only if the model's context
        # window turns out to be too small for the full history.
        history_limit = len(pairs)

        final_code = ""
        raw_answer_full = ""
        reasoning_content = ""
        give_up_reason = ""

        for attempt in range(3):
            history_slice = pairs[-history_limit:] if history_limit > 0 else []
            working_messages = [self.messages[0]] + history_slice + [{"role": "user", "content": user_prompt}]
            try:
                final_code, raw_answer_full, reasoning_content, usage_record, messages_sent = self._run_policy_debug_loop(
                    working_messages, state, turn_token_usage, agent_label
                )

                if hasattr(self.model, "log_llm_io") and (final_code or raw_answer_full):
                    self.model.log_llm_io({
                        "agent": agent_label,
                        "agent_id": str(self.unique_id),
                        "model": self.model_name,
                        "provider": getattr(self.model, "provider", ""),
                        "turn": self.turns,
                        "attempt": attempt + 1,
                        "messages_sent": messages_sent,
                        "raw_response": raw_answer_full,
                        "reasoning_content": reasoning_content,
                        "cleaned_response": final_code,
                        "token_usage": usage_record,
                    })

                # _run_policy_debug_loop only ever returns code that already
                # compiled (accepted via submit_policy_code) or "" (gave up
                # after MAX_TOOL_ROUNDS without an accepted submission) - no
                # separate compile retry needed here.
                if final_code.strip():
                    break
                print(f"> [Agent {self.unique_id}] [WARNING] No accepted policy code this attempt. Retrying... ({attempt + 1}/3)")
            except APIStatusError as e:
                if is_context_overflow_error(e) and history_limit > 0 and attempt < 2:
                    history_limit = max(0, history_limit - 2)  # drop the oldest user/assistant turn
                    print(f"> [Agent {self.unique_id}] [WARNING] Prompt too long for the model's context window; dropping older turns (keeping last {history_limit}) and retrying...")
                    continue
                print(f"> [Agent {self.unique_id}] Communication Error: {e}")
                if attempt == 2:
                    give_up_reason = f"Communication error after 3 attempts: {e}"
            except Exception as e:
                print(f"> [Agent {self.unique_id}] Communication Error: {e}")
                if attempt == 2:
                    give_up_reason = f"Communication error after 3 attempts: {e}"

        if not final_code.strip():
            # No usable policy after 3 attempts. Rather than defaulting to an
            # arbitrary direction the model never chose, the agent just
            # doesn't move this turn - but the failure is still recorded
            # (not a silent return) so it's visible in the GUI/logs.
            give_up_reason = give_up_reason or "LLM did not produce valid policy code after 3 attempts."
            print(f"> [Agent {self.unique_id}] [WARNING] {give_up_reason} Agent will not move this turn.")
            policy_result = {"action": None, "blocks": 1, "broadcast": None, "until": None, "error": give_up_reason}
        else:
            policy_result = run_policy_code(final_code, state, timeout=15)
            if policy_result.get("error"):
                print(f"> [Agent {self.unique_id}] [WARNING] policy code error on final submission: {policy_result['error']}; agent will not move this turn.")

        reason = ""
        code_lines = final_code.splitlines()
        if code_lines and code_lines[0].strip().startswith("#"):
            reason = code_lines[0].strip().lstrip("#").strip()

        action = str(policy_result.get("action") or "").upper()
        new_msg = str(policy_result.get("broadcast") or "").strip()
        until_expr = policy_result.get("until") if not policy_result.get("error") else None

        message_meaning = ""
        if new_msg:
            self.turns_since_broadcast = 0
            sender_label = agent_label
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
                    "position": list(self.pos),
                    "reason": reason,
                })
        else:
            self.turns_since_broadcast += 1

        log_msg = f"\n--- [Agent {self.unique_id} Turn] ---\n"
        log_msg += f"Model: {self.model_name}\n"
        log_msg += f"Position: {self.pos}\n"
        log_msg += f"Reasoning: {reason}\n"
        log_msg += f"Action: {action}\n"
        log_msg += f"Token Usage: {json.dumps(turn_token_usage)}\n"
        if new_msg:
            log_msg += f"Broadcast: {new_msg}\n"
        if until_expr:
            log_msg += f"Auto-move plan: {action} until '{until_expr}'\n"
        if policy_result.get("error"):
            log_msg += f"Policy Error: {policy_result['error']}\n"
        self.model.log(log_msg)

        print(f"\n>>> Model Step: Agent {self.unique_id} ({self.model_name})")
        if new_msg:
            print(f"[BROADCAST]: \"{new_msg}\"")
        print(f"[POS] Position: {self.pos}")
        print(f"[THINK] Reasoning: {reason}")
        print(f"[ACTION] Action: {action}" + (f" (auto-move until '{until_expr}')" if until_expr else ""))
        print("-" * 30)

        self.messages.append({"role": "user", "content": user_prompt})
        self.messages.append({"role": "assistant", "content": final_code})

        try:
            blocks = int(policy_result.get("blocks", 1))
        except (TypeError, ValueError):
            blocks = 1
        blocks = max(1, min(blocks, self.speed_limit))

        interaction_events, blocked_feedback = self._apply_movement(action, blocks, decision_start_pos)

        # A plan only survives into future turns if this move actually
        # succeeded and there was no error - a blocked or failed first move
        # means the condition never even gets a chance to matter.
        if until_expr and self.pos != decision_start_pos:
            self.active_auto_move = {"direction": action, "blocks": blocks, "until": until_expr}
        else:
            self.active_auto_move = None

        no_action_reason = policy_result.get("error") if action not in DIRECTION_DELTAS else ""

        self.last_decision = {
            "turn": self.turns,
            "reason": reason,
            "action": action,
            "blocks": blocks,
            "notes": "",
            "broadcast_message": new_msg,
            "broadcast_meaning": message_meaning,
            "raw_response": raw_answer_full or final_code,
            "reasoning_content": reasoning_content,
            "policy_code": final_code,
            "policy_error": policy_result.get("error"),
            "auto_move": False,
            "auto_move_until": until_expr,
            "no_action_reason": no_action_reason,
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
            "blocked": self.pos == decision_start_pos,
            "blocked_reason": blocked_feedback or no_action_reason,
            "interactions": pre_move_events + interaction_events,
            "inventory": list(self.inventory),
        }
