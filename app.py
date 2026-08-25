import streamlit as st
import os
from map_gen import MapGenerator
from agents import MACI_Model

# ==========================================
# 페이지 기본 설정
# ==========================================
st.set_page_config(page_title="Project MACI", layout="wide")

st.title("Project MACI: LLM Agent Maze Runner")

import json

CONFIG_FILE = "last_config.json"
EXPERIMENT_RUNS_DIR = "experiment_runs"
SAVED_MAPS_DIR = "saved_maps"
NO_NUMERIC_SYMBOL_SPACE_PROMPT = """Communication mode:
- Coded communication is enabled, but numeric suffixes are disabled.
- Use only base codes: F, S, G, K, D, H, X, N.
- Do not append digits to broadcast codes.
- If no cooperative event is active, use N.

Experimenter notes:
- Add temporary meanings or hypotheses here when needed."""

PROMPT_TEMPLATES = {
    "gpt": """<role>
You are Agent {agent_id}, a cooperative maze-running agent in Project MACI.
Your goal is to reach the exit symbol 'F' in as few turns and tokens as possible while coordinating through compact symbolic broadcasts.
</role>

<output_contract>
Reply ONLY with one valid JSON object. No markdown, no prose outside JSON.
Required keys:
- "reason": concise tactical rationale, 1-3 sentences.
- "action": one of "UP", "DOWN", "LEFT", "RIGHT".
- "blocks": integer from 1 to {speed_limit}.
- "broadcast_message": one base code only, no numeric suffix. Valid: F/S/G/K/D/H/X/N.
- "notes": short memory update.
- "structured_memory": compact memory update; communication_space may define base-code meanings.
</output_contract>""",
    "gemini": """<role>
You are a cooperative maze-running agent in Project MACI.
You control exactly one agent: Agent {agent_id}.
Your goal is to reach the exit F efficiently while coordinating with partner agents under limited communication.
</role>

<task>
At each turn, inspect the provided map, memory, interaction rules, inbox messages, and recent path.
Choose exactly one movement action and broadcast a compact base code.
</task>

<output_contract>
Return ONLY one valid JSON object. No markdown. No extra prose.
Required keys:
{{
  "reason": "1-3 concise tactical sentences",
  "action": "UP | DOWN | LEFT | RIGHT",
  "blocks": integer from 1 to {speed_limit},
  "broadcast_message": "one of F/S/G/K/D/H/X/N, with no numeric suffix",
  "notes": "short memory update for next turn",
  "structured_memory": {{"communication_space": {{}}}}
}}
</output_contract>""",
    "kimi": """You are Agent {agent_id}, a cooperative maze-running agent in Project MACI.

You must act as one grid-world agent. This is tactical navigation with compact communication.
Be decisive. Keep reasoning short. Preserve and reuse compact communication conventions.
Do not output anything except the requested JSON object.

Output exactly one JSON object:
{{
  "reason": "concise tactical rationale, 1-3 sentences",
  "action": "UP | DOWN | LEFT | RIGHT",
  "blocks": 1,
  "broadcast_message": "N",
  "notes": "short next-turn memory",
  "structured_memory": {{"communication_space": {{}}}}
}}

broadcast_message must be one of F/S/G/K/D/H/X/N with no numeric suffix.
Valid examples: "N", "S", "H", "G".
Invalid examples: "", any code with digits, "Switch here".""" 
}

DEFAULT_SYMBOL_SPACE_PROMPT = """Communication mode:
- Coded communication is enabled, but numeric suffixes are disabled.
- Use only base codes: F, S, G, K, D, H, X, N.
- Do not append digits to broadcast codes.
- If no cooperative event is active, use N.

Experimenter notes:
- Add temporary meanings or hypotheses here when needed."""

def strip_numeric_suffix_prompt_text(value, include_default=True):
    import re
    text = str(value or "")
    if not text.strip():
        return ""
    cleaned_lines = []
    for line in text.splitlines():
        lowered = line.lower()
        mentions_numeric = any(token in lowered for token in [
            "numeric suffix space",
            "integer suffix",
            "optional integer suffix",
            "optionally followed",
            "code+number",
            "0-10",
            "n-slot",
            "s3",
            "s7",
            "h10",
            "g0",
            "s0",
            "n0",
        ])
        is_no_numeric_rule = (
            "disabled" in lowered
            or "do not use" in lowered
            or "invalid examples" in lowered
            or "no numeric suffix" in lowered
            or "with no numeric suffix" in lowered
        )
        if mentions_numeric and not is_no_numeric_rule:
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if include_default and "Communication mode:" not in text:
        text = f"{NO_NUMERIC_SYMBOL_SPACE_PROMPT}\n\n{text}".strip()
    return text

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_config(config):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except:
        pass

def get_saved_prompt_templates():
    templates = PROMPT_TEMPLATES.copy()
    saved_templates = last_cfg.get("prompt_templates", {})
    if isinstance(saved_templates, dict):
        for key in ["gpt", "gemini", "kimi"]:
            if saved_templates.get(key):
                templates[key] = strip_numeric_suffix_prompt_text(saved_templates[key], include_default=False)
    return templates

def infer_prompt_profile(model_name, default="gpt"):
    lowered = str(model_name or "").lower()
    if "gemini" in lowered:
        return "gemini"
    if "kimi" in lowered or "moonshot" in lowered or "qwen" in lowered or "llama" in lowered or "gemma" in lowered or "mistral" in lowered:
        return "kimi"
    return default

def get_saved_symbol_space_prompt():
    return strip_numeric_suffix_prompt_text(last_cfg.get("symbol_space_prompt", DEFAULT_SYMBOL_SPACE_PROMPT), include_default=True)

def get_saved_broadcast_codes():
    saved_codes = str(last_cfg.get("allowed_broadcast_codes", "F,S,G,K,D,H,X,N"))
    return saved_codes if "N" in saved_codes.upper() else f"{saved_codes},N"

def create_experiment_run_dir(timestamp):
    run_dir = os.path.join(EXPERIMENT_RUNS_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def create_experiment_round_dir(run_dir, round_number, group_name="Experimental"):
    safe_group = str(group_name).lower().replace(" ", "_")
    round_dir = os.path.join(run_dir, f"round_{int(round_number):04d}_{safe_group}")
    os.makedirs(round_dir, exist_ok=True)
    return round_dir

def safe_filename(value, fallback="map"):
    text = str(value or fallback).strip()
    cleaned = "".join(ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in text)
    return cleaned.strip("_") or fallback

def save_persistent_map_state(label, state):
    os.makedirs(SAVED_MAPS_DIR, exist_ok=True)
    import datetime
    saved_at = datetime.datetime.now().isoformat(timespec="seconds")
    filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_filename(label)}.json"
    path = os.path.join(SAVED_MAPS_DIR, filename)
    payload = dict(state)
    payload["label"] = label or state.get("label", "saved_map")
    payload["saved_at"] = saved_at
    payload["schema"] = "maci_saved_map_v1"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

def load_persistent_map_records():
    if not os.path.isdir(SAVED_MAPS_DIR):
        return []
    records = []
    for filename in sorted(os.listdir(SAVED_MAPS_DIR), reverse=True):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(SAVED_MAPS_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            records.append({
                "path": path,
                "filename": filename,
                "label": state.get("label", filename),
                "saved_at": state.get("saved_at", ""),
                "state": state
            })
        except Exception:
            continue
    return records

def apply_map_state_to_model(model, mg, state):
    mg.grid = [row[:] for row in state["grid"]]
    mg.height = len(mg.grid)
    mg.width = len(mg.grid[0]) if mg.grid else getattr(mg, "width", 25)
    mg.type = state.get("map_type", getattr(mg, "type", "2"))
    mg.seed = state.get("map_seed", getattr(mg, "seed", 0))
    mg.diff = state.get("map_diff", getattr(mg, "diff", "M"))
    model.map_data = mg.grid
    model.width = mg.width
    model.height = mg.height
    model.target_pos = tuple(state["target_pos"])
    model.fake_symbols = {tuple(item["pos"]): item["symbol"] for item in state.get("fake_symbols", [])}
    model.switches = [tuple(pos) for pos in state.get("switches", [])]
    model.gates = [tuple(pos) for pos in state.get("gates", [])]
    model.open_gates = set()
    model.gates_open = False
    sanitize_interaction_rules(model, state.get("interaction_rules", {}))
    if hasattr(model, "_capture_static_layout"):
        model._capture_static_layout()
    model.update_gates()

def reset_agent_positions_for_map(model, mg, agent_configs):
    for agent in list(model.agents):
        try:
            model.grid.remove_agent(agent)
        except Exception:
            pass
    symbol_positions = {tuple(getattr(model, "target_pos", (-1, -1)))}
    symbol_positions.update(tuple(pos) for pos in getattr(model, "switches", []))
    symbol_positions.update(tuple(pos) for pos in getattr(model, "gates", []))
    symbol_positions.update(tuple(pos) for pos in getattr(model, "fake_symbols", {}).keys())
    empty_spaces = [pos for pos in mg.get_empty_spaces() if pos not in symbol_positions]
    model.placed_agent_positions = []
    model._place_agents(agent_configs, empty_spaces)

def pos_key(pos):
    return f"{pos[0]},{pos[1]}"

def parse_agent_spec(value, valid_labels):
    text = str(value or "*").strip().upper()
    if text in ["", "*", "ALL"]:
        return ["*"]
    labels = []
    for part in text.replace(";", ",").split(","):
        label = part.strip().upper()
        if label in valid_labels and label not in labels:
            labels.append(label)
    return labels or ["*"]

def sanitize_interaction_rules(model, rules=None):
    rules = rules or getattr(model, "interaction_rules", {}) or {}
    switch_keys = {pos_key(pos) for pos in getattr(model, "switches", [])}
    gate_keys = {pos_key(pos) for pos in getattr(model, "gates", [])}

    def clean_agents(rule_name, keys):
        source = rules.get(rule_name, {})
        return {key: source.get(key, ["*"]) for key in keys}

    switch_links = {}
    source_links = rules.get("switch_links", {})
    for key in switch_keys:
        links = source_links.get(key, ["*"])
        if isinstance(links, str):
            links = [item.strip() for item in links.split(";") if item.strip()]
        if not links or "*" in links:
            switch_links[key] = ["*"]
        else:
            valid_links = [link for link in links if link in gate_keys]
            switch_links[key] = valid_links or ["*"]

    sanitized = {
        "switch_agents": clean_agents("switch_agents", switch_keys),
        "gate_agents": clean_agents("gate_agents", gate_keys),
        "switch_links": switch_links
    }
    model.interaction_rules = sanitized
    return sanitized

def apply_existing_interaction_rules(new_model, old_model=None):
    old_rules = getattr(old_model, "interaction_rules", None) if old_model else None
    return sanitize_interaction_rules(new_model, old_rules)

def parse_gate_link_spec(value, gate_label_to_key, valid_gate_keys):
    text = str(value or "*").strip().upper()
    if text in ["", "*", "ALL"]:
        return ["*"]
    links = []
    for part in text.replace(";", ",").split(","):
        item = part.strip().upper()
        if item in gate_label_to_key:
            key = gate_label_to_key[item]
        else:
            key = item
        if key in valid_gate_keys and key not in links:
            links.append(key)
    return links or ["*"]

def rule_value_text(value):
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value or "*")

def broadcast_display(message, meaning="", max_meaning=48):
    if not message:
        return ""
    meaning = str(meaning or "").strip()
    if not meaning:
        return str(message)
    if len(meaning) > max_meaning:
        meaning = meaning[:max_meaning - 3] + "..."
    return f"{message} ({meaning})"

def get_model_token_summary(model):
    if hasattr(model, "token_summary"):
        return model.token_summary()
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 0
    }
    agents = []
    for i, agent in enumerate(getattr(model, "agents", [])):
        item = {
            "agent": chr(65 + i),
            "model": getattr(agent, "model_name", ""),
            "turns": getattr(agent, "turns", 0),
            "prompt_tokens": getattr(agent, "prompt_tokens", 0),
            "completion_tokens": getattr(agent, "completion_tokens", 0),
            "reasoning_tokens": getattr(agent, "reasoning_tokens", 0),
            "cached_tokens": getattr(agent, "cached_tokens", 0),
            "total_tokens": getattr(agent, "total_tokens", 0)
        }
        agents.append(item)
        for key in totals:
            totals[key] += int(item.get(key, 0))
    return {"total": totals, "agents": agents}

def get_current_agent_configs():
    if st.session_state.get("current_group") == "Control":
        return st.session_state.control_configs
    return st.session_state.experimental_configs

def bump_map_editor_version():
    st.session_state.map_editor_version = st.session_state.get("map_editor_version", 0) + 1

def capture_map_state(label=""):
    model = st.session_state.model
    mg = st.session_state.mg
    return {
        "label": label,
        "grid": [row[:] for row in mg.grid],
        "target_pos": list(model.target_pos),
        "fake_symbols": [{"pos": list(pos), "symbol": sym} for pos, sym in getattr(model, "fake_symbols", {}).items()],
        "switches": [list(pos) for pos in getattr(model, "switches", [])],
        "gates": [list(pos) for pos in getattr(model, "gates", [])],
        "interaction_rules": json.loads(json.dumps(getattr(model, "interaction_rules", {}))),
        "map_type": getattr(mg, "type", "2"),
        "map_seed": getattr(mg, "seed", 0),
        "map_diff": getattr(mg, "diff", "M"),
    }

def remember_map_state(label):
    history = st.session_state.setdefault("map_history", [])
    history.append(capture_map_state(label))
    st.session_state.map_history = history[-12:]

def restore_map_state(state):
    model = st.session_state.model
    mg = st.session_state.mg
    apply_map_state_to_model(model, mg, state)
    bump_map_editor_version()

def reset_current_simulation_round(label):
    current_map_state = capture_map_state(label)
    configs = get_current_agent_configs()
    old_model = st.session_state.model
    new_model = MACI_Model(
        num_agents=len(configs),
        map_generator=st.session_state.mg,
        agent_configs=configs,
        thinking_effort=old_model.thinking_effort,
        provider=old_model.provider,
        api_key=old_model.api_key,
        base_url=old_model.base_url,
        optimization_mode=getattr(old_model, "optimization_mode", False),
        log_file=st.session_state.log_file
    )
    apply_existing_interaction_rules(new_model, old_model)
    st.session_state.model = new_model
    restore_map_state(current_map_state)
    st.session_state.step_count = 0
    st.session_state.experiment_round += 1
    st.session_state.experiment_round_dir = create_experiment_round_dir(
        st.session_state.experiment_run_dir,
        st.session_state.experiment_round,
        st.session_state.current_group
    )
    save_turn_snapshot(
        st.session_state.model,
        st.session_state.mg,
        st.session_state.step_count,
        st.session_state.experiment_round_dir,
        st.session_state.current_group
    )
    bump_map_editor_version()

def apply_manual_map_edit(edited_df):
    remember_map_state("before_manual_edit")
    new_grid = []
    new_fakes = {}
    new_target = st.session_state.model.target_pos
    new_switches = []
    new_gates = []

    for y, row in edited_df.iterrows():
        grid_row = []
        for x, val in enumerate(row):
            val = str(val).upper().strip()
            if val == "#":
                grid_row.append(1)
            elif val == "F":
                grid_row.append(0)
                new_target = (x, y)
            elif val == "S":
                grid_row.append(0)
                new_switches.append((x, y))
            elif val == "G":
                grid_row.append(1)
                new_gates.append((x, y))
            elif val in [".", "", " "]:
                grid_row.append(0)
            else:
                grid_row.append(0)
                new_fakes[(x, y)] = val[:2] if val else "."
        new_grid.append(grid_row)

    st.session_state.mg.grid = new_grid
    st.session_state.model.map_data = new_grid
    st.session_state.model.target_pos = new_target
    st.session_state.model.fake_symbols = new_fakes
    st.session_state.model.switches = new_switches
    st.session_state.model.gates = new_gates
    st.session_state.model.open_gates = set()
    st.session_state.model.gates_open = False
    sanitize_interaction_rules(st.session_state.model)
    if hasattr(st.session_state.model, "_capture_static_layout"):
        st.session_state.model._capture_static_layout()
    st.session_state.model.update_gates()
    bump_map_editor_version()

def reroll_current_map(map_type=None, difficulty=None):
    import random
    remember_map_state("before_reroll")
    mg = st.session_state.mg
    current_type = str(map_type) if map_type is not None else str(random.randint(1, 2))
    current_diff = str(difficulty) if difficulty is not None else random.choice(["E", "M", "H"])
    if current_type not in ["1", "2"]:
        current_type = str(random.randint(1, 2))
    if current_diff not in ["E", "M", "H"]:
        current_diff = random.choice(["E", "M", "H"])
    new_seed = random.randint(1000, 9999)
    mg.generate_map(f"{current_type}{new_seed}{current_diff}")

    configs = get_current_agent_configs()
    old_model = st.session_state.model
    new_model = MACI_Model(
        num_agents=len(configs),
        map_generator=mg,
        agent_configs=configs,
        thinking_effort=old_model.thinking_effort,
        provider=old_model.provider,
        api_key=old_model.api_key,
        base_url=old_model.base_url,
        optimization_mode=getattr(old_model, "optimization_mode", False),
        log_file=st.session_state.log_file
    )
    sanitize_interaction_rules(new_model)
    st.session_state.model = new_model
    st.session_state.step_count = 0
    st.session_state.experiment_round += 1
    st.session_state.experiment_round_dir = create_experiment_round_dir(
        st.session_state.experiment_run_dir,
        st.session_state.experiment_round,
        st.session_state.current_group
    )
    save_turn_snapshot(
        st.session_state.model,
        st.session_state.mg,
        st.session_state.step_count,
        st.session_state.experiment_round_dir,
        st.session_state.current_group
    )
    bump_map_editor_version()

def save_turn_snapshot(model, map_generator, step_count, run_dir, group_name="Experimental"):
    try:
        from PIL import Image, ImageDraw, ImageFont
        import csv
        import datetime

        os.makedirs(run_dir, exist_ok=True)
        safe_group = str(group_name).lower().replace(" ", "_")
        round_label = os.path.basename(run_dir)
        image_name = f"{safe_group}_turn_{int(step_count):04d}.png"
        image_path = os.path.join(run_dir, image_name)

        cell = 32
        header_h = 52
        legend_h = 220
        width = map_generator.width * cell
        height = header_h + map_generator.height * cell + legend_h
        img = Image.new("RGB", (width, height), "#202124")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 14)
            font_bold = ImageFont.truetype("arialbd.ttf", 16)
        except:
            font = ImageFont.load_default()
            font_bold = ImageFont.load_default()

        draw.rectangle([0, 0, width, header_h], fill="#111827")
        draw.text((12, 8), f"{group_name} | {round_label} | Turn {step_count}", fill="#ffffff", font=font_bold)
        draw.text((12, 30), f"Saved: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fill="#cbd5e1", font=font)

        agent_colors = ["#ef4444", "#2563eb", "#16a34a", "#9333ea"]
        agent_positions = {}
        for i, agent in enumerate(model.agents):
            label = chr(65 + i)
            agent_positions[agent.pos] = (label, agent_colors[i % len(agent_colors)])

        for y, row in enumerate(map_generator.grid):
            for x, cell_value in enumerate(row):
                left = x * cell
                top = header_h + y * cell
                right = left + cell
                bottom = top + cell

                fill = "#2f3437"
                text = ""
                text_fill = "#ffffff"

                if cell_value == 1:
                    fill = "#050505"

                if hasattr(model, "target_pos") and (x, y) == model.target_pos:
                    fill = "#3f3f46"
                    text = "F"
                    text_fill = "#facc15"

                if hasattr(model, "switches") and (x, y) in model.switches:
                    is_pressed = any(a.pos == (x, y) for a in model.agents)
                    fill = "#f97316" if is_pressed else "#7c2d12"
                    text = "S"
                    text_fill = "#ffffff"

                if hasattr(model, "gates") and (x, y) in model.gates:
                    fill = "#0f766e" if (x, y) in getattr(model, "open_gates", set()) else "#52525b"
                    text = "G"
                    text_fill = "#ffffff"

                if hasattr(model, "fake_symbols") and (x, y) in model.fake_symbols:
                    text = str(model.fake_symbols[(x, y)])[:2]
                    if text == "K":
                        fill = "#713f12"
                        text_fill = "#fde047"
                    elif text == "D":
                        fill = "#7f1d1d"
                        text_fill = "#fecaca"
                    else:
                        fill = "#3b3b3b"
                        text_fill = "#fbbf24"

                draw.rectangle([left, top, right, bottom], fill=fill, outline="#4b5563")

                if (x, y) in agent_positions:
                    label, color = agent_positions[(x, y)]
                    pad = 5
                    draw.ellipse([left + pad, top + pad, right - pad, bottom - pad], fill=color, outline="#ffffff")
                    bbox = draw.textbbox((0, 0), label, font=font_bold)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    draw.text((left + (cell - tw) / 2, top + (cell - th) / 2 - 1), label, fill="#ffffff", font=font_bold)
                elif text:
                    bbox = draw.textbbox((0, 0), text, font=font_bold)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    draw.text((left + (cell - tw) / 2, top + (cell - th) / 2 - 1), text, fill=text_fill, font=font_bold)

        legend_y = header_h + map_generator.height * cell + 10
        legend = "A/B/C/D=Agents  F=Exit  S=Switch  G=Gate  K=Key  D=Door  black=Wall"
        draw.text((12, legend_y), legend, fill="#e5e7eb", font=font)
        decision_y = legend_y + 22
        for i, agent in enumerate(model.agents):
            decision = getattr(agent, "last_decision", {}) or {}
            label = chr(65 + i)
            action = decision.get("action", "")
            blocks = decision.get("blocks", "")
            broadcast = decision.get("broadcast_message", "")
            broadcast_meaning = decision.get("broadcast_meaning", "")
            notes = str(decision.get("notes", ""))[:70]
            summary = f"{label}: {action} x{blocks}"
            if broadcast:
                summary += f" | msg={broadcast_display(broadcast, broadcast_meaning)}"
            if notes:
                summary += f" | {notes}"
            draw.text((12, decision_y), summary, fill="#d1d5db", font=font)
            decision_y += 18
        recent_comms = getattr(model, "communication_log", [])[-4:]
        if recent_comms:
            draw.text((12, decision_y), "Communication:", fill="#e5e7eb", font=font)
            decision_y += 18
            for item in recent_comms:
                targets = ",".join(item.get("to", []))
                line = f"{item.get('from', '?')}->{targets}: {broadcast_display(item.get('message', ''), item.get('meaning', ''))}"
                draw.text((12, decision_y), line[:95], fill="#bfdbfe", font=font)
                decision_y += 18
        img.save(image_path)

        agent_state = [
            {
                "agent": chr(65 + i),
                "id": str(agent.unique_id),
                "position": list(agent.pos),
                "model": getattr(agent, "model_name", ""),
                "done": bool(getattr(agent, "is_done", False)),
                "last_decision": getattr(agent, "last_decision", {}),
                "token_usage": {
                    "prompt_tokens": getattr(agent, "prompt_tokens", 0),
                    "completion_tokens": getattr(agent, "completion_tokens", 0),
                    "reasoning_tokens": getattr(agent, "reasoning_tokens", 0),
                    "cached_tokens": getattr(agent, "cached_tokens", 0),
                    "total_tokens": getattr(agent, "total_tokens", 0),
                    "last": getattr(agent, "last_token_usage", {})
                },
            }
            for i, agent in enumerate(model.agents)
        ]
        token_summary = get_model_token_summary(model)
        record = {
            "step": int(step_count),
            "round": round_label,
            "round_dir": run_dir,
            "group": group_name,
            "image": image_path,
            "target": list(getattr(model, "target_pos", [])),
            "agents": agent_state,
            "gates_open": bool(getattr(model, "gates_open", False)),
            "open_gates": [list(pos) for pos in sorted(getattr(model, "open_gates", set()))],
            "interaction_rules": getattr(model, "interaction_rules", {}),
            "communication_log": getattr(model, "communication_log", []),
            "recent_communication": getattr(model, "communication_log", [])[-10:],
            "communication_improvements": getattr(model, "communication_improvements", []),
            "current_communication_guideline": getattr(model, "current_communication_guideline", ""),
            "strategy_improvements": getattr(model, "strategy_improvements", []),
            "reflection_consultations": getattr(model, "reflection_consultations", []),
            "active_prompt_additions": [
                {
                    "agent": chr(65 + i),
                    "prompt_addition": getattr(agent, "prompt_addition", ""),
                    "prompt_addition_source": getattr(agent, "prompt_addition_source", ""),
                    "optimization_base_prompt": getattr(agent, "optimization_base_prompt", "")
                }
                for i, agent in enumerate(getattr(model, "agents", []))
            ],
            "token_usage": token_summary,
            "allowed_broadcast_codes": getattr(model, "allowed_broadcast_codes", []),
            "symbol_space": [
                {
                    "agent": chr(65 + i),
                    "communication_space": getattr(agent, "structured_memory", {}).get("communication_space", {}),
                    "symbol_space_notes": getattr(agent, "structured_memory", {}).get("symbol_space_notes", "")
                }
                for i, agent in enumerate(getattr(model, "agents", []))
            ],
            "success_debrief": getattr(model, "success_debrief", {}),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        }

        manifest_path = os.path.join(run_dir, "manifest.jsonl")
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        index_path = os.path.join(run_dir, "index.csv")
        write_header = not os.path.exists(index_path)
        with open(index_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "step", "group", "image", "target", "agents", "decisions", "communication", "token_usage", "gates_open", "timestamp"])
            if write_header:
                writer.writeheader()
            writer.writerow({
                "round": record["round"],
                "step": record["step"],
                "group": record["group"],
                "image": record["image"],
                "target": record["target"],
                "agents": agent_state,
                "decisions": [
                    {
                        "agent": agent["agent"],
                        **agent.get("last_decision", {})
                    }
                    for agent in agent_state
                ],
                "communication": record["recent_communication"],
                "token_usage": record["token_usage"],
                "gates_open": record["gates_open"],
                "timestamp": record["timestamp"],
            })

        return image_path
    except Exception as e:
        st.warning(f"스냅샷 저장 실패: {e}")
        return None

last_cfg = load_config()

# ==========================================
# 1. 시뮬레이션 설정 화면 (Setup Phase)
# ==========================================
if "simulation_initialized" not in st.session_state:
    st.header("[Setup] Simulation Setup")
    
    with st.form("setup_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            providers = ["openai", "gemini", "ollama"]
            p_idx = providers.index(last_cfg["provider"]) if "provider" in last_cfg and last_cfg["provider"] in providers else 0
            provider = st.selectbox("LLM Provider", providers, index=p_idx)
        with col2:
            api_key = st.text_input("API Key (leave blank for env var)", value=last_cfg.get("api_key", ""), type="password")
        with col3:
            types = ["Room (Dungeon)", "Maze", "Random"]
            t_idx = types.index(last_cfg["map_type"]) if "map_type" in last_cfg and last_cfg["map_type"] in types else 0
            if "experiment_mode" in st.session_state and st.session_state.experiment_mode:
                map_type = st.selectbox("Map Type", ["Room (Dungeon)"], disabled=True)
            else:
                map_type = st.selectbox("Map Type", types, index=t_idx)
        with col4:
            complexities = ["H", "M", "E", "Random"]
            c_idx = complexities.index(last_cfg["map_complexity"]) if "map_complexity" in last_cfg and last_cfg["map_complexity"] in complexities else 0
            map_complexity = st.selectbox("Map Complexity", complexities, index=c_idx)
            
        default_base_url = last_cfg.get("custom_base_url", "")
        if provider == "ollama" and not default_base_url:
            default_base_url = "http://localhost:11434/v1"
        custom_base_url = st.text_input("Custom Base URL (Optional)", value=default_base_url)
            
        num_agents = st.number_input("Number of Agents", min_value=1, max_value=4, value=last_cfg.get("num_agents", 2))
        efforts = ["low", "medium", "high"]
        e_idx = efforts.index(last_cfg["thinking_effort"]) if "thinking_effort" in last_cfg and last_cfg["thinking_effort"] in efforts else 1
        thinking_effort = st.selectbox("Reasoning Effort (for supported models)", efforts, index=e_idx)
        
        st.subheader("Agent Specific Constraints")
        optimization_mode = st.checkbox("Enable Optimization Mode", value=last_cfg.get("optimization_mode", False))
        experiment_mode = st.checkbox("Enable Experiment Mode", value=last_cfg.get("experiment_mode", False))
        
        if experiment_mode:
            st.info("Dungeon Room only. Control Group will have full vision and no communication limits.")

        prompt_templates = get_saved_prompt_templates()
        with st.expander("Prompt Profiles (GPT / Gemini / Kimi)", expanded=False):
            st.caption("여기서 모델별 기본 system prompt를 수정할 수 있습니다. 에이전트별 선택값은 아래 탭에서 정합니다.")
            prompt_templates["gpt"] = st.text_area(
                "GPT prompt",
                value=prompt_templates["gpt"],
                height=180,
                key="prompt_template_gpt"
            )
            prompt_templates["gemini"] = st.text_area(
                "Gemini prompt",
                value=prompt_templates["gemini"],
                height=180,
                key="prompt_template_gemini"
            )
            prompt_templates["kimi"] = st.text_area(
                "Kimi prompt",
                value=prompt_templates["kimi"],
                height=180,
                key="prompt_template_kimi"
            )

        with st.expander("부호 추가 공간 / Communication Symbol Space", expanded=True):
            st.caption("허용할 기본 부호와 0~10 숫자 suffix의 실험자 정의를 적습니다. 이 내용은 모든 에이전트 prompt에 추가됩니다.")
            allowed_broadcast_codes = st.text_input(
                "Allowed broadcast base codes",
                value=get_saved_broadcast_codes(),
                help="예: F,S,G,K,D,H,X 또는 F,S,G,K,D,H,X,A,B"
            )
            symbol_space_prompt = st.text_area(
                "Symbol space notes",
                value=get_saved_symbol_space_prompt(),
                height=180,
                key="symbol_space_prompt_setup"
            )

        saved_map_records = load_persistent_map_records()
        saved_map_options = ["Generate new map"] + [
            f"{record['label']} | {record['saved_at']} | {record['filename']}"
            for record in saved_map_records
        ]
        with st.expander("Saved Map Loader", expanded=bool(saved_map_records)):
            st.caption("여기서 저장된 맵을 고르면 새 실험을 시작해도 같은 맵 상태를 불러옵니다.")
            selected_saved_map_label = st.selectbox(
                "Start from saved map",
                saved_map_options,
                index=0,
                key="setup_saved_map"
            )
            if saved_map_records:
                st.caption(f"저장된 맵 {len(saved_map_records)}개를 찾았습니다: {SAVED_MAPS_DIR}/")
            else:
                st.caption("아직 디스크에 저장된 맵이 없습니다.")
        
        agent_configs = []
        
        tabs = st.tabs([f"Agent {chr(65+i)}" for i in range(int(num_agents))])
        for i, tab in enumerate(tabs):
            with tab:
                # Try to get previous agent settings
                prev_a = last_cfg.get("agent_configs", [{}])[i] if "agent_configs" in last_cfg and i < len(last_cfg["agent_configs"]) else {}
                
                c1, c2 = st.columns(2)
                with c1:
                    default_model = "llama3.1:8b" if provider == "ollama" else "gpt-4o-mini"
                    m_name = st.text_input(f"Model Name", value=prev_a.get("model_name", default_model), key=f"m_{i}")
                    v_range = st.slider(f"Vision Range", min_value=1, max_value=10, value=prev_a.get("vision_range", 5), key=f"v_{i}")
                    s_limit = st.slider(f"Speed (Blocks/Turn)", min_value=1, max_value=5, value=prev_a.get("speed_limit", 1), key=f"speed_{i}")
                with c2:
                    b_limit = st.slider(f"Communication Byte Limit", min_value=0, max_value=1000, value=prev_a.get("byte_limit", 500), step=50, key=f"b_{i}")
                    m_share = st.slider(f"Map Sharing Radius (0 = OFF)", min_value=0, max_value=10, value=prev_a.get("map_share_radius", 0), key=f"s_{i}")

                prompt_options = ["gpt", "gemini", "kimi", "custom"]
                default_profile = prev_a.get("prompt_profile", infer_prompt_profile(m_name, provider))
                if default_profile not in prompt_options:
                    default_profile = "gpt"
                prompt_profile = st.selectbox(
                    "Prompt Profile",
                    prompt_options,
                    index=prompt_options.index(default_profile),
                    key=f"prompt_profile_{i}"
                )

                if prompt_profile == "custom":
                    custom_default = strip_numeric_suffix_prompt_text(prev_a.get("custom_system_prompt", "").strip(), include_default=False) or prompt_templates.get("gpt", "")
                    selected_prompt = st.text_area(
                        "Custom system prompt",
                        value=custom_default,
                        height=220,
                        key=f"custom_prompt_{i}"
                    )
                else:
                    selected_prompt = prompt_templates.get(prompt_profile, "")
                    st.caption(f"이 에이전트는 `{prompt_profile}` 프로필 prompt를 사용합니다.")
                
                agent_configs.append({
                    "model_name": m_name,
                    "vision_range": v_range,
                    "speed_limit": s_limit,
                    "byte_limit": b_limit,
                    "map_share_radius": m_share,
                    "prompt_profile": prompt_profile,
                    "custom_system_prompt": strip_numeric_suffix_prompt_text(selected_prompt, include_default=False),
                    "allowed_broadcast_codes": allowed_broadcast_codes,
                    "symbol_space_prompt": strip_numeric_suffix_prompt_text(symbol_space_prompt, include_default=True),
                    "optimization_mode": optimization_mode,
                    "coded_communication": True
                })

        submitted = st.form_submit_button("RUN SIMULATION", use_container_width=True)
        
        if submitted:
            selected_saved_map_state = None
            if selected_saved_map_label != "Generate new map":
                selected_index = saved_map_options.index(selected_saved_map_label) - 1
                if 0 <= selected_index < len(saved_map_records):
                    selected_saved_map_state = saved_map_records[selected_index]["state"]

            # Save current config
            save_config({
                "provider": provider,
                "api_key": api_key,
                "map_type": map_type,
                "map_complexity": map_complexity,
                "custom_base_url": custom_base_url,
                "num_agents": num_agents,
                "thinking_effort": thinking_effort,
                "optimization_mode": optimization_mode,
                "experiment_mode": experiment_mode,
                "prompt_templates": prompt_templates,
                "allowed_broadcast_codes": allowed_broadcast_codes,
                "symbol_space_prompt": symbol_space_prompt,
                "agent_configs": agent_configs
            })
            
            with st.spinner("Generating Map and Deploying Agents... (BFS Reachability Check)"):
                mg = MapGenerator(25, 25)
                comp = None if map_complexity == "Random" else map_complexity
                
                # Determine map type 't'
                if selected_saved_map_state:
                    t = selected_saved_map_state.get("map_type", "2")
                    d = selected_saved_map_state.get("map_diff", "M")
                    mg.grid = [row[:] for row in selected_saved_map_state["grid"]]
                    mg.height = len(mg.grid)
                    mg.width = len(mg.grid[0]) if mg.grid else 25
                    mg.type = str(t)
                    mg.seed = selected_saved_map_state.get("map_seed", 0)
                    mg.diff = str(d)
                elif experiment_mode:
                    t = 2 # Forced Room type for experiments
                elif map_type == "Room (Dungeon)":
                    t = 2
                elif map_type == "Maze":
                    t = 1
                else:
                    import random
                    t = random.randint(1, 2)
                    
                import random
                if not selected_saved_map_state:
                    s = random.randint(1000, 9999)
                    d = comp if comp in ['E', 'M', 'H'] else random.choice(['E', 'M', 'H'])
                    mg.generate_map(f'{t}{s}{d}')
                
                # Store original configs for switching
                st.session_state.experimental_configs = agent_configs
                st.session_state.prompt_templates = prompt_templates
                st.session_state.allowed_broadcast_codes = allowed_broadcast_codes
                st.session_state.symbol_space_prompt = symbol_space_prompt
                # Generate Control configs (Full vision, no comm limit)
                control_configs = []
                for cfg in agent_configs:
                    c_cfg = cfg.copy()
                    c_cfg["vision_range"] = 25
                    c_cfg["byte_limit"] = 10000
                    control_configs.append(c_cfg)
                st.session_state.control_configs = control_configs
                st.session_state.current_group = "Experimental" # Default

                
                print("\n" + "="*50)
                print(f"NEW SIMULATION STARTED (Map Type: {t}, Diff: {d})")
                print("="*50 + "\n")
                
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = f"log_{timestamp}.txt"
                experiment_run_dir = create_experiment_run_dir(timestamp)
                
                maci_world = MACI_Model(
                    num_agents=int(num_agents),
                    map_generator=mg,
                    agent_configs=agent_configs,
                    thinking_effort=thinking_effort,
                    provider=provider,
                    api_key=api_key,
                    base_url=custom_base_url.strip() if custom_base_url.strip() else None,
                    optimization_mode=optimization_mode,
                    log_file=log_file
                )
                sanitize_interaction_rules(maci_world)
                if selected_saved_map_state:
                    apply_map_state_to_model(maci_world, mg, selected_saved_map_state)
                    reset_agent_positions_for_map(maci_world, mg, agent_configs)
                
                st.session_state.mg = mg
                st.session_state.model = maci_world
                st.session_state.step_count = 0
                st.session_state.simulation_initialized = True
                st.session_state.experiment_mode = experiment_mode
                st.session_state.log_file = log_file
                st.session_state.experiment_run_dir = experiment_run_dir
                st.session_state.experiment_round = 1
                st.session_state.map_editor_version = 0
                initial_label = f"loaded_{selected_saved_map_state.get('label', 'saved_map')}" if selected_saved_map_state else "initial_map"
                st.session_state.map_history = [capture_map_state(initial_label)]
                st.session_state.experiment_round_dir = create_experiment_round_dir(
                    st.session_state.experiment_run_dir,
                    st.session_state.experiment_round,
                    st.session_state.current_group
                )
                save_turn_snapshot(
                    st.session_state.model,
                    st.session_state.mg,
                    st.session_state.step_count,
                    st.session_state.experiment_round_dir,
                    st.session_state.current_group
                )
            st.rerun()

# ==========================================
# 2. 맵 시각화 및 실행 화면 (Simulation Phase)
# ==========================================
else:
    def render_map_to_html(model, map_generator):
        agent_markers = {}
        symbols = [
            "<div style='background-color: #FF4B4B; color: white; border-radius: 50%; width: 90%; height: 90%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 13px; box-shadow: 0 0 5px #FF4B4B;'>A</div>", 
            "<div style='background-color: #1E88E5; color: white; border-radius: 50%; width: 90%; height: 90%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 13px; box-shadow: 0 0 5px #1E88E5;'>B</div>",
            "<div style='background-color: #4CAF50; color: white; border-radius: 50%; width: 90%; height: 90%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 13px; box-shadow: 0 0 5px #4CAF50;'>C</div>",
            "<div style='background-color: #AB47BC; color: white; border-radius: 50%; width: 90%; height: 90%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 13px; box-shadow: 0 0 5px #AB47BC;'>D</div>"
        ]
        
        for i, agent in enumerate(model.agents):
            agent_markers[agent.pos] = symbols[i % len(symbols)]

        cols = map_generator.width
        map_html = f"<div style='display: grid; grid-template-columns: repeat({cols}, 24px); gap: 0px; background-color: #1e1e1e; padding: 20px; border-radius: 10px; width: max-content;'>"
        
        for y, row in enumerate(map_generator.grid):
            for x, cell in enumerate(row):
                content = ""
                bg_color = "#2b2b2b"
                
                # Priority: Agents > Target > Switch > Gate > Fake > Wall/Path
                if (x, y) in agent_markers:
                    content = agent_markers[(x, y)]
                elif hasattr(model, 'target_pos') and (x, y) == model.target_pos:
                    content = "<div style='color: #FFD700; font-weight: bold; font-size: 16px;'>[F]</div>"
                    bg_color = "#444"
                elif hasattr(model, 'switches') and (x, y) in model.switches:
                    # If an agent is on switch, highlight it
                    is_pressed = any(a.pos == (x, y) for a in model.agents)
                    content = "<div style='font-size: 14px;'>[S]</div>"
                    bg_color = "#FF9800" if is_pressed else "#6D4C41"
                elif hasattr(model, 'gates') and (x, y) in model.gates:
                    content = "<div style='font-size: 14px;'>[G:O]</div>" if (x, y) in getattr(model, "open_gates", set()) else "<div style='font-size: 14px;'>[G:L]</div>"
                    bg_color = "#424242"
                elif hasattr(model, 'fake_symbols') and (x, y) in model.fake_symbols:
                    sym = model.fake_symbols[(x, y)]
                    if sym == 'K':
                        content = "<div style='color: #FFD700; font-weight: bold;'>[K]</div>"
                    elif sym == 'D':
                        content = "<div style='color: #FF5252; font-weight: bold;'>[D]</div>"
                    else:
                        content = f"<div style='color: #FFC107; font-weight: bold;'>{sym}</div>"
                    bg_color = "#333"
                elif cell == 1:
                    bg_color = "#000"
                
                map_html += f"<div style='width: 24px; height: 24px; background-color: {bg_color}; border: 0.5px solid #444; display: flex; align-items: center; justify-content: center;'>{content}</div>"

            
        map_html += "</div>"
        return map_html

    if "current_group" not in st.session_state:
        st.session_state.current_group = "Experimental"
    if "experiment_run_dir" not in st.session_state:
        import datetime
        fallback_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.experiment_run_dir = create_experiment_run_dir(fallback_timestamp)
    if "experiment_round" not in st.session_state:
        st.session_state.experiment_round = 1
    if "map_history" not in st.session_state:
        st.session_state.map_history = []
    if "map_editor_version" not in st.session_state:
        st.session_state.map_editor_version = 0
    if "experiment_round_dir" not in st.session_state:
        st.session_state.experiment_round_dir = create_experiment_round_dir(
            st.session_state.experiment_run_dir,
            st.session_state.experiment_round,
            st.session_state.current_group
        )
        save_turn_snapshot(
            st.session_state.model,
            st.session_state.mg,
            st.session_state.step_count,
            st.session_state.experiment_round_dir,
            st.session_state.current_group
        )

    with st.sidebar:
        st.header("Control Panel")
        st.write(f"**현재 스텝:** {st.session_state.step_count}")
        
        if st.session_state.get("experiment_mode", False):
            st.divider()
            st.subheader("Experiment Setup")
            selected_group = st.radio("실행 그룹 선택", ["Experimental", "Control"], index=0 if st.session_state.current_group == "Experimental" else 1)
            
            if selected_group != st.session_state.current_group:
                # Reset simulation for the other group
                st.session_state.current_group = selected_group
                configs = st.session_state.experimental_configs if selected_group == "Experimental" else st.session_state.control_configs
                
                with st.spinner(f"Switching to {selected_group} Group..."):
                    # Use same map, same seed
                    new_model = MACI_Model(
                        num_agents=len(configs),
                        map_generator=st.session_state.mg,
                        agent_configs=configs,
                        thinking_effort=st.session_state.model.thinking_effort,
                        provider=st.session_state.model.provider,
                        api_key=st.session_state.model.api_key,
                        base_url=st.session_state.model.base_url,
                        log_file=st.session_state.log_file
                    )
                    apply_existing_interaction_rules(new_model, st.session_state.model)
                    st.session_state.model = new_model
                    st.session_state.step_count = 0
                    st.session_state.experiment_round += 1
                    st.session_state.experiment_round_dir = create_experiment_round_dir(
                        st.session_state.experiment_run_dir,
                        st.session_state.experiment_round,
                        st.session_state.current_group
                    )
                    save_turn_snapshot(
                        st.session_state.model,
                        st.session_state.mg,
                        st.session_state.step_count,
                        st.session_state.experiment_round_dir,
                        st.session_state.current_group
                    )
                st.rerun()
        
        if st.session_state.get("optimization_mode", False):
            st.divider()
            with st.expander("Difficulty Adjustment (Optimization Mode)"):
                st.write("환경의 복잡도를 변경하여 에이전트를 테스트하세요.")
                current_diff = st.session_state.mg.diff
                new_diff = st.selectbox("새 난이도", ["E", "M", "H"], index=["E", "M", "H"].index(current_diff))
                
                if st.button("Apply & Restart", use_container_width=True):
                    # Re-generate map with new difficulty
                    seed_str = f"{st.session_state.mg.type}{st.session_state.mg.seed}{new_diff}"
                    st.session_state.mg.generate_map(seed_str)
                    
                    # Re-initialize model
                    new_model = MACI_Model(
                        num_agents=st.session_state.model.num_agents,
                        map_generator=st.session_state.mg,
                        agent_configs=st.session_state.experimental_configs,
                        thinking_effort=st.session_state.model.thinking_effort,
                        provider=st.session_state.model.provider,
                        api_key=st.session_state.model.api_key,
                        base_url=st.session_state.model.base_url,
                        optimization_mode=True,
                        log_file=st.session_state.log_file
                    )
                    apply_existing_interaction_rules(new_model, st.session_state.model)
                    st.session_state.model = new_model
                    st.session_state.step_count = 0
                    st.session_state.experiment_round += 1
                    st.session_state.experiment_round_dir = create_experiment_round_dir(
                        st.session_state.experiment_run_dir,
                        st.session_state.experiment_round,
                        st.session_state.current_group
                    )
                    save_turn_snapshot(
                        st.session_state.model,
                        st.session_state.mg,
                        st.session_state.step_count,
                        st.session_state.experiment_round_dir,
                        st.session_state.current_group
                    )
                    st.rerun()
        
        st.divider()

        with st.expander("Agent Prompt Config", expanded=False):
            st.write("모델별 prompt, 허용 부호, 숫자 suffix 공간을 수정하고 각 에이전트가 어떤 prompt를 쓸지 지정합니다.")
            runtime_templates = st.session_state.get("prompt_templates", get_saved_prompt_templates()).copy()
            runtime_templates["gpt"] = st.text_area("GPT prompt", value=runtime_templates.get("gpt", PROMPT_TEMPLATES["gpt"]), height=120, key="runtime_prompt_gpt")
            runtime_templates["gemini"] = st.text_area("Gemini prompt", value=runtime_templates.get("gemini", PROMPT_TEMPLATES["gemini"]), height=120, key="runtime_prompt_gemini")
            runtime_templates["kimi"] = st.text_area("Kimi prompt", value=runtime_templates.get("kimi", PROMPT_TEMPLATES["kimi"]), height=120, key="runtime_prompt_kimi")
            st.session_state.prompt_templates = runtime_templates

            runtime_allowed_codes = st.text_input(
                "Allowed broadcast base codes",
                value=st.session_state.get("allowed_broadcast_codes", get_saved_broadcast_codes()),
                key="runtime_allowed_broadcast_codes",
                help="예: F,S,G,K,D,H,X 또는 새 부호를 추가해 F,S,G,K,D,H,X,A,B"
            )
            runtime_symbol_space_prompt = st.text_area(
                "부호 추가 공간 / Symbol space notes",
                value=strip_numeric_suffix_prompt_text(st.session_state.get("symbol_space_prompt", get_saved_symbol_space_prompt()), include_default=True),
                height=140,
                key="runtime_symbol_space_prompt"
            )
            st.session_state.allowed_broadcast_codes = runtime_allowed_codes
            st.session_state.symbol_space_prompt = runtime_symbol_space_prompt

            prompt_options = ["gpt", "gemini", "kimi", "custom"]
            active_configs = get_current_agent_configs()
            edited_prompt_configs = []
            for i, cfg in enumerate(active_configs):
                label = chr(65 + i)
                current_profile = cfg.get("prompt_profile", infer_prompt_profile(cfg.get("model_name", ""), "gpt"))
                if current_profile not in prompt_options:
                    current_profile = "gpt"
                profile = st.selectbox(
                    f"Agent {label} prompt profile",
                    prompt_options,
                    index=prompt_options.index(current_profile),
                    key=f"runtime_prompt_profile_{i}"
                )
                updated_cfg = cfg.copy()
                updated_cfg["prompt_profile"] = profile
                if profile == "custom":
                    updated_cfg["custom_system_prompt"] = st.text_area(
                        f"Agent {label} custom prompt",
                        value=strip_numeric_suffix_prompt_text(cfg.get("custom_system_prompt", "").strip(), include_default=False) or runtime_templates["gpt"],
                        height=140,
                        key=f"runtime_custom_prompt_{i}"
                    )
                else:
                    updated_cfg["custom_system_prompt"] = runtime_templates.get(profile, "")
                updated_cfg["allowed_broadcast_codes"] = runtime_allowed_codes
                updated_cfg["custom_system_prompt"] = strip_numeric_suffix_prompt_text(updated_cfg["custom_system_prompt"], include_default=False)
                updated_cfg["symbol_space_prompt"] = strip_numeric_suffix_prompt_text(runtime_symbol_space_prompt, include_default=True)
                edited_prompt_configs.append(updated_cfg)

            if st.button("Apply Prompt Config & Restart Round", use_container_width=True):
                if st.session_state.current_group == "Control":
                    st.session_state.control_configs = edited_prompt_configs
                    for idx, cfg in enumerate(edited_prompt_configs):
                        if idx < len(st.session_state.experimental_configs):
                            st.session_state.experimental_configs[idx]["prompt_profile"] = cfg["prompt_profile"]
                            st.session_state.experimental_configs[idx]["custom_system_prompt"] = cfg["custom_system_prompt"]
                            st.session_state.experimental_configs[idx]["allowed_broadcast_codes"] = runtime_allowed_codes
                            st.session_state.experimental_configs[idx]["symbol_space_prompt"] = runtime_symbol_space_prompt
                else:
                    st.session_state.experimental_configs = edited_prompt_configs
                    for idx, cfg in enumerate(edited_prompt_configs):
                        if idx < len(st.session_state.control_configs):
                            st.session_state.control_configs[idx]["prompt_profile"] = cfg["prompt_profile"]
                            st.session_state.control_configs[idx]["custom_system_prompt"] = cfg["custom_system_prompt"]
                            st.session_state.control_configs[idx]["allowed_broadcast_codes"] = runtime_allowed_codes
                            st.session_state.control_configs[idx]["symbol_space_prompt"] = runtime_symbol_space_prompt

                saved = last_cfg.copy()
                saved.update({
                    "provider": st.session_state.model.provider,
                    "api_key": st.session_state.model.api_key,
                    "custom_base_url": st.session_state.model.base_url or "",
                    "num_agents": len(st.session_state.experimental_configs),
                    "thinking_effort": st.session_state.model.thinking_effort,
                    "optimization_mode": st.session_state.get("optimization_mode", False),
                    "experiment_mode": st.session_state.get("experiment_mode", False),
                    "prompt_templates": runtime_templates,
                    "allowed_broadcast_codes": runtime_allowed_codes,
                    "symbol_space_prompt": runtime_symbol_space_prompt,
                    "agent_configs": st.session_state.experimental_configs
                })
                save_config(saved)
                reset_current_simulation_round("prompt_config_update")
                st.rerun()

        with st.expander("Map State / Reroll", expanded=True):
            st.write("현재 맵을 디스크에 저장하거나, 바로 새 맵으로 리롤할 수 있습니다.")
            c1, c2 = st.columns(2)
            with c1:
                reroll_type_label = st.selectbox(
                    "Map Type",
                    ["Keep Current", "Room (Dungeon)", "Maze", "Random"],
                    key="reroll_map_type"
                )
            with c2:
                reroll_diff_label = st.selectbox(
                    "Difficulty",
                    ["Keep Current", "E", "M", "H", "Random"],
                    key="reroll_map_diff"
                )

            persistent_map_label = st.text_input(
                "Persistent map name",
                value=f"map_round_{st.session_state.experiment_round}_step_{st.session_state.step_count}",
                key="persistent_map_label"
            )

            if st.button("SAVE CURRENT MAP STATE", use_container_width=True):
                remember_map_state(f"manual_save_step_{st.session_state.step_count}")
                path = save_persistent_map_state(persistent_map_label, capture_map_state(persistent_map_label))
                st.success(f"현재 맵 상태를 디스크에 저장했습니다: {path}")

            if st.button("REROLL MAP & RESTART ROUND", use_container_width=True):
                type_map = {
                    "Keep Current": getattr(st.session_state.mg, "type", "2"),
                    "Room (Dungeon)": "2",
                    "Maze": "1",
                    "Random": None
                }
                diff_value = None if reroll_diff_label == "Random" else (
                    getattr(st.session_state.mg, "diff", "M") if reroll_diff_label == "Keep Current" else reroll_diff_label
                )
                reroll_current_map(type_map[reroll_type_label], diff_value)
                st.rerun()

            history = st.session_state.get("map_history", [])
            if history:
                labels = [f"{i + 1}: {item.get('label', 'saved')} ({item.get('map_type', '?')}{item.get('map_seed', '?')}{item.get('map_diff', '?')})" for i, item in enumerate(history)]
                selected_history = st.selectbox("Restore Session Map History", labels, index=len(labels) - 1)
                if st.button("RESTORE SESSION MAP & RESTART ROUND", use_container_width=True):
                    idx = labels.index(selected_history)
                    restore_map_state(history[idx])
                    reset_current_simulation_round("restore_saved_map")
                    st.rerun()
            else:
                st.caption("현재 세션 안에서 복원할 임시 맵 히스토리가 아직 없습니다.")

        if st.button("NEXT STEP", use_container_width=True):
            with st.spinner('LLM 에이전트들이 생각 중입니다... (터미널 로그 확인)'):
                st.session_state.model.step()
                st.session_state.step_count += 1
                save_turn_snapshot(
                    st.session_state.model,
                    st.session_state.mg,
                    st.session_state.step_count,
                    st.session_state.experiment_round_dir,
                    st.session_state.current_group
                )
                
        st.divider()
        st.divider()
        with st.expander("GUI Map Editor (Click & Type)"):
            st.info("""
            **How to Edit:**
            1. Click a cell in the table below.
            2. Type the symbol: **#** (Wall), **.** (Path), **F** (Exit), **K** (Key), **D** (Door), **S** (Switch), **G** (Gate).
            3. Press **Enter** to confirm.
            4. Click **SAVE CHANGES** below to apply.
            """)


            
            # Create a DataFrame for editing
            import pandas as pd
            grid_data = []
            for y in range(st.session_state.mg.height):
                row = []
                for x in range(st.session_state.mg.width):
                    if (x, y) == st.session_state.model.target_pos:
                        row.append("F")
                    elif hasattr(st.session_state.model, 'switches') and (x, y) in st.session_state.model.switches:
                        row.append("S")
                    elif hasattr(st.session_state.model, 'gates') and (x, y) in st.session_state.model.gates:
                        row.append("G")
                    elif hasattr(st.session_state.model, 'fake_symbols') and (x, y) in st.session_state.model.fake_symbols:
                        row.append(st.session_state.model.fake_symbols[(x, y)])
                    elif st.session_state.mg.grid[y][x] == 1:
                        row.append("#")
                    else:
                        row.append(".")
                grid_data.append(row)
            
            df = pd.DataFrame(grid_data)

            st.write("Quick Paint")
            p1, p2, p3 = st.columns(3)
            with p1:
                paint_symbol = st.selectbox("Tile", [".", "#", "F", "S", "G", "K", "D"], key="paint_symbol")
            with p2:
                paint_x = st.number_input("X", min_value=0, max_value=st.session_state.mg.width - 1, value=0, key="paint_x")
            with p3:
                paint_y = st.number_input("Y", min_value=0, max_value=st.session_state.mg.height - 1, value=0, key="paint_y")

            if st.button("APPLY SINGLE CELL", use_container_width=True):
                df.iat[int(paint_y), int(paint_x)] = paint_symbol
                apply_manual_map_edit(df)
                st.success(f"({int(paint_x)}, {int(paint_y)}) 위치를 {paint_symbol}로 바꿨습니다.")
                st.rerun()

            if st.button("MAKE HORIZONTALLY SYMMETRIC", use_container_width=True):
                mid = st.session_state.mg.width // 2
                for y in range(st.session_state.mg.height):
                    for x in range(mid):
                        df.iat[y, st.session_state.mg.width - 1 - x] = df.iat[y, x]
                apply_manual_map_edit(df)
                st.success("맵의 왼쪽 절반을 기준으로 좌우 대칭이 적용되었습니다.")
                st.rerun()
            
            # Display the editor
            edited_df = st.data_editor(df, 
                                     hide_index=True, 
                                     column_config={i: st.column_config.TextColumn(width="small") for i in range(st.session_state.mg.width)},
                                     key=f"map_editor_grid_{st.session_state.get('map_editor_version', 0)}")
            
            if st.button("SAVE CHANGES", use_container_width=True):
                apply_manual_map_edit(edited_df)
                st.success("맵 수정 사항이 저장되었습니다!")
                st.rerun()

            st.divider()
            st.subheader("Agent Position Editor")
            if hasattr(st.session_state.model, 'agents') and len(st.session_state.model.agents) > 0:
                agent_cols = st.columns(len(st.session_state.model.agents))
                for idx, agent in enumerate(st.session_state.model.agents):
                    with agent_cols[idx]:
                        st.markdown(f"**Agent {agent.unique_id}**")
                        ax = st.number_input(f"X (Agent {agent.unique_id})", min_value=0, max_value=st.session_state.mg.width - 1, value=agent.pos[0] if agent.pos else 0, key=f"agent_x_{agent.unique_id}")
                        ay = st.number_input(f"Y (Agent {agent.unique_id})", min_value=0, max_value=st.session_state.mg.height - 1, value=agent.pos[1] if agent.pos else 0, key=f"agent_y_{agent.unique_id}")
                        if st.button(f"MOVE", use_container_width=True, key=f"move_btn_{agent.unique_id}"):
                            if agent.pos:
                                st.session_state.model.grid.move_agent(agent, (int(ax), int(ay)))
                            else:
                                st.session_state.model.grid.place_agent(agent, (int(ax), int(ay)))
                            st.success(f"Agent {agent.unique_id}가 ({ax}, {ay})로 이동했습니다.")
                            st.rerun()

            st.divider()
            st.subheader("Interaction Rules")
            st.caption("Allowed Agents: A,B 또는 * 입력. Linked Gates: G1,G2 또는 * 입력.")

            sanitize_interaction_rules(st.session_state.model)
            agent_labels = [chr(65 + i) for i, _ in enumerate(st.session_state.model.agents)]
            valid_agent_text = ", ".join(agent_labels) if agent_labels else "A"
            gate_positions = list(getattr(st.session_state.model, "gates", []))
            switch_positions = list(getattr(st.session_state.model, "switches", []))
            gate_label_to_key = {f"G{i + 1}": pos_key(pos) for i, pos in enumerate(gate_positions)}
            valid_gate_keys = set(gate_label_to_key.values())
            rules = st.session_state.model.interaction_rules

            if not switch_positions and not gate_positions:
                st.info("맵에 S 또는 G를 먼저 추가하고 SAVE CHANGES를 누르면 규칙을 편집할 수 있습니다.")
            else:
                if switch_positions:
                    switch_rows = []
                    for i, pos in enumerate(switch_positions):
                        key = pos_key(pos)
                        linked = rules["switch_links"].get(key, ["*"])
                        if linked == ["*"]:
                            linked_text = "*"
                        else:
                            reverse_gate_labels = {v: k for k, v in gate_label_to_key.items()}
                            linked_text = ",".join(reverse_gate_labels.get(item, item) for item in linked)
                        switch_rows.append({
                            "Switch": f"S{i + 1}",
                            "Coord": key,
                            "Allowed Agents": rule_value_text(rules["switch_agents"].get(key, ["*"])),
                            "Linked Gates": linked_text
                        })

                    switch_df = pd.DataFrame(switch_rows)
                    edited_switch_rules = st.data_editor(
                        switch_df,
                        hide_index=True,
                        disabled=["Switch", "Coord"],
                        key="switch_interaction_rules"
                    )
                else:
                    edited_switch_rules = pd.DataFrame(columns=["Switch", "Coord", "Allowed Agents", "Linked Gates"])

                if gate_positions:
                    gate_rows = []
                    for i, pos in enumerate(gate_positions):
                        key = pos_key(pos)
                        gate_rows.append({
                            "Gate": f"G{i + 1}",
                            "Coord": key,
                            "Allowed Agents": rule_value_text(rules["gate_agents"].get(key, ["*"]))
                        })

                    gate_df = pd.DataFrame(gate_rows)
                    edited_gate_rules = st.data_editor(
                        gate_df,
                        hide_index=True,
                        disabled=["Gate", "Coord"],
                        key="gate_interaction_rules"
                    )
                else:
                    edited_gate_rules = pd.DataFrame(columns=["Gate", "Coord", "Allowed Agents"])

                if st.button("SAVE INTERACTION RULES", use_container_width=True):
                    new_rules = {"switch_agents": {}, "gate_agents": {}, "switch_links": {}}

                    for _, row in edited_switch_rules.iterrows():
                        key = str(row["Coord"]).strip()
                        if key:
                            new_rules["switch_agents"][key] = parse_agent_spec(row.get("Allowed Agents", "*"), agent_labels)
                            new_rules["switch_links"][key] = parse_gate_link_spec(row.get("Linked Gates", "*"), gate_label_to_key, valid_gate_keys)

                    for _, row in edited_gate_rules.iterrows():
                        key = str(row["Coord"]).strip()
                        if key:
                            new_rules["gate_agents"][key] = parse_agent_spec(row.get("Allowed Agents", "*"), agent_labels)

                    sanitize_interaction_rules(st.session_state.model, new_rules)
                    st.session_state.model.update_gates()
                    st.success(f"상호작용 규칙이 저장되었습니다. 사용 가능한 에이전트: {valid_agent_text}")
                    st.rerun()

        st.divider()

        if st.button("AUTO RUN", use_container_width=True):
            auto_run_placeholder = st.empty()
            while not any(agent.is_done for agent in st.session_state.model.agents):
                with st.spinner(f"자동 진행 중... (Step: {st.session_state.step_count})"):
                    st.session_state.model.step()
                    st.session_state.step_count += 1
                    save_turn_snapshot(
                        st.session_state.model,
                        st.session_state.mg,
                        st.session_state.step_count,
                        st.session_state.experiment_round_dir,
                        st.session_state.current_group
                    )
                    # Update Map in placeholder
                    with auto_run_placeholder.container():
                        st.subheader(f"Map Situation (Step: {st.session_state.step_count})")
                        map_str = render_map_to_html(st.session_state.model, st.session_state.mg)
                        st.markdown(map_str, unsafe_allow_html=True)
                    import time
                    time.sleep(0.5)
            st.rerun()
                
        st.divider()
        
        if st.button("RESET TO SETUP", use_container_width=True):
            del st.session_state.simulation_initialized
            st.rerun()

    token_summary = get_model_token_summary(st.session_state.model)
    token_total = token_summary.get("total", {})
    title_col, token_col = st.columns([3, 1])
    with title_col:
        st.subheader(f"Map Situation (Step: {st.session_state.step_count})")
    with token_col:
        st.metric(
            "Tokens",
            int(token_total.get("total_tokens", 0)),
            help=f"prompt={token_total.get('prompt_tokens', 0)}, completion={token_total.get('completion_tokens', 0)}, reasoning={token_total.get('reasoning_tokens', 0)}"
        )
        st.caption(f"reasoning: {int(token_total.get('reasoning_tokens', 0))}")

    
    if any(agent.is_done for agent in st.session_state.model.agents):
        st.success("Target reached! Agent found 'F'!")
        if not getattr(st.session_state.model, "success_debrief_done", False):
            with st.spinner("성공 후 에이전트들이 필요한 만큼 상의하고 핵심 키워드를 정리 중입니다..."):
                st.session_state.model.run_success_debrief()
                save_turn_snapshot(
                    st.session_state.model,
                    st.session_state.mg,
                    st.session_state.step_count,
                    st.session_state.experiment_round_dir,
                    st.session_state.current_group
                )

        success_debrief = getattr(st.session_state.model, "success_debrief", {})
        keywords = success_debrief.get("keywords", [])
        discussion = success_debrief.get("discussion", [])
        debrief_tokens = success_debrief.get("token_usage", get_model_token_summary(st.session_state.model))
        communication_improvements = getattr(st.session_state.model, "communication_improvements", [])
        current_comm_guideline = getattr(st.session_state.model, "current_communication_guideline", "")
        if discussion:
            with st.expander("Success Debrief Discussion"):
                st.dataframe(discussion, use_container_width=True, hide_index=True)
        if keywords:
            st.subheader("Final Keywords")
            st.dataframe(keywords, use_container_width=True, hide_index=True)
        if current_comm_guideline or communication_improvements:
            st.subheader("Communication Improvements")
            if current_comm_guideline:
                st.info(current_comm_guideline)
            if communication_improvements:
                st.json(communication_improvements[-1])
        if debrief_tokens:
            st.subheader("Token Settlement")
            st.json(debrief_tokens)

        if getattr(st.session_state.model, 'optimization_mode', False):
            if st.button("Next Session (Reflection & Restart)"):
                with st.spinner("에이전트들이 이전 세션을 반성하고 새로운 행동 지침을 생성 중입니다..."):
                    st.session_state.model.reflect_and_restart_session()
                    st.session_state.step_count = 0
                    st.session_state.experiment_round += 1
                    st.session_state.experiment_round_dir = create_experiment_round_dir(
                        st.session_state.experiment_run_dir,
                        st.session_state.experiment_round,
                        st.session_state.current_group
                    )
                    save_turn_snapshot(
                        st.session_state.model,
                        st.session_state.mg,
                        st.session_state.step_count,
                        st.session_state.experiment_round_dir,
                        st.session_state.current_group
                    )
                st.rerun()

    map_str = render_map_to_html(st.session_state.model, st.session_state.mg)
    st.markdown(map_str, unsafe_allow_html=True)
    if "experiment_run_dir" in st.session_state:
        st.caption(f"Experiment snapshots: {st.session_state.experiment_run_dir}")
    if "experiment_round_dir" in st.session_state:
        st.caption(f"Current round snapshots: {st.session_state.experiment_round_dir}")
    if getattr(st.session_state.model, "current_communication_guideline", ""):
        with st.expander("Current Self-Improved Communication Protocol", expanded=False):
            st.write(st.session_state.model.current_communication_guideline)
            improvements = getattr(st.session_state.model, "communication_improvements", [])
            if improvements:
                st.json(improvements[-1])
    recent_comms = getattr(st.session_state.model, "communication_log", [])[-10:]
    if recent_comms:
        st.subheader("Communication Log")
        comm_rows = []
        for item in recent_comms:
            row = dict(item)
            row["broadcast"] = broadcast_display(item.get("message", ""), item.get("meaning", ""), max_meaning=90)
            comm_rows.append(row)
        st.dataframe(comm_rows, use_container_width=True, hide_index=True)
    st.info("Check the Python terminal for real-time agent thoughts and communication logs.")
