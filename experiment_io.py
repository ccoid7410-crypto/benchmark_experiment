"""Framework-agnostic experiment persistence helpers for Project MACI.

Extracted out of the old Streamlit app so both the CLI (maci.py) and the
desktop GUI (gui_app.py) can share the same config/map/snapshot logic
without depending on Streamlit.
"""
from __future__ import annotations

import csv
import datetime
import json
import os

from maci_components.agent_support import (
    DEFAULT_PROMPT_TEMPLATES as PROMPT_TEMPLATES,
    NO_NUMERIC_SYMBOL_SPACE_PROMPT,
    PROVIDER_DEFAULT_BASE_URLS,
    strip_numeric_suffix_prompt_text,
)

CONFIG_FILE = "last_config.json"
EXPERIMENT_RUNS_DIR = "experiment_runs"
SAVED_MAPS_DIR = "saved_maps"

DEFAULT_SYMBOL_SPACE_PROMPT = NO_NUMERIC_SYMBOL_SPACE_PROMPT

PROVIDERS = ["openai", "gemini", "ollama", "llamacpp", "openrouter", "custom"]


def default_base_url_for_provider(provider):
    return PROVIDER_DEFAULT_BASE_URLS.get(str(provider or "").lower(), "")


def infer_prompt_profile(model_name, default="gpt"):
    lowered = str(model_name or "").lower()
    if "gemini" in lowered:
        return "gemini"
    if "kimi" in lowered or "moonshot" in lowered or "qwen" in lowered or "llama" in lowered or "gemma" in lowered or "mistral" in lowered or "deepseek" in lowered:
        return "kimi"
    return default


# ---------------------------------------------------------------------------
# Config persistence (last_config.json)
# ---------------------------------------------------------------------------

def load_config(path=CONFIG_FILE):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_config(config, path=CONFIG_FILE):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_saved_prompt_templates(last_cfg):
    templates = PROMPT_TEMPLATES.copy()
    saved_templates = last_cfg.get("prompt_templates", {})
    if isinstance(saved_templates, dict):
        for key in ["gpt", "gemini", "kimi"]:
            if saved_templates.get(key):
                templates[key] = strip_numeric_suffix_prompt_text(saved_templates[key], include_default=False)
    return templates


def get_saved_symbol_space_prompt(last_cfg):
    return strip_numeric_suffix_prompt_text(last_cfg.get("symbol_space_prompt", DEFAULT_SYMBOL_SPACE_PROMPT), include_default=True)


def get_saved_broadcast_codes(last_cfg):
    saved_codes = str(last_cfg.get("allowed_broadcast_codes", "F,S,G,K,D,H,X,N"))
    return saved_codes if "N" in saved_codes.upper() else f"{saved_codes},N"


# ---------------------------------------------------------------------------
# Run / round directories
# ---------------------------------------------------------------------------

def create_experiment_run_dir(timestamp, base_dir=EXPERIMENT_RUNS_DIR):
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
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


# ---------------------------------------------------------------------------
# Saved maps
# ---------------------------------------------------------------------------

def capture_map_state(model, mg, label=""):
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


def save_persistent_map_state(label, state, directory=SAVED_MAPS_DIR):
    os.makedirs(directory, exist_ok=True)
    saved_at = datetime.datetime.now().isoformat(timespec="seconds")
    filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_filename(label)}.json"
    path = os.path.join(directory, filename)
    payload = dict(state)
    payload["label"] = label or state.get("label", "saved_map")
    payload["saved_at"] = saved_at
    payload["schema"] = "maci_saved_map_v1"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_persistent_map_records(directory=SAVED_MAPS_DIR):
    if not os.path.isdir(directory):
        return []
    records = []
    for filename in sorted(os.listdir(directory), reverse=True):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(directory, filename)
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


def pos_key(pos):
    return f"{pos[0]},{pos[1]}"


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


# ---------------------------------------------------------------------------
# Token / communication display helpers
# ---------------------------------------------------------------------------

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
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0, "total_tokens": 0}
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


# ---------------------------------------------------------------------------
# Turn snapshot (image + manifest.jsonl + index.csv)
# ---------------------------------------------------------------------------

def save_turn_snapshot(model, map_generator, step_count, run_dir, group_name="Experimental", on_error=None):
    """Renders the current turn to a PNG and appends a full record (including
    every agent's raw LLM response / reasoning_content) to manifest.jsonl and
    index.csv inside run_dir. Returns the saved image path, or None on failure."""
    try:
        from PIL import Image, ImageDraw, ImageFont

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
        except Exception:
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
            "success_debrief": getattr(model, "success_debrief", {}),
            "llm_io_log_path": getattr(model, "llm_io_log_path", ""),
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
                    {"agent": agent["agent"], **agent.get("last_decision", {})}
                    for agent in agent_state
                ],
                "communication": record["recent_communication"],
                "token_usage": record["token_usage"],
                "gates_open": record["gates_open"],
                "timestamp": record["timestamp"],
            })

        return image_path
    except Exception as e:
        if on_error:
            on_error(e)
        else:
            print(f"[WARN] Snapshot save failed: {e}")
        return None
