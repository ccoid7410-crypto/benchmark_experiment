"""Project MACI desktop GUI (Tkinter) - replaces the old Streamlit app.

Streamlit's full-page-rerun model made long LLM-driven simulation runs
fragile (state loss, reruns interrupting in-flight requests). This is a
plain Tkinter/ttk desktop app instead: a single persistent window, a
background worker thread for the blocking LLM calls, and no framework
managing app state behind your back.

Run with:  python gui_app.py
"""
from __future__ import annotations

import os
import queue
import random
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import experiment_io as eio
from map_gen import MapGenerator
from maci_components.maci_model import MACI_Model

AGENT_COLORS = ["#ef4444", "#2563eb", "#16a34a", "#9333ea"]
CELL = 22
MAX_AGENTS = 4


class ScrollableFrame(ttk.Frame):
    """A vertically scrollable container (Tkinter has no built-in one)."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.body = ttk.Frame(canvas)

        self.body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=self.body, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(canvas_window, width=e.width))
        canvas.configure(yscrollcommand=vscroll.set)

        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Scope the wheel binding to hover time only, so switching screens
        # (which destroys this canvas) never leaves a stale bind_all behind.
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))


class MaciGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project MACI — Cooperative LLM Benchmark")
        self.root.geometry("1500x950")

        self.last_cfg = eio.load_config()
        self.saved_map_records = eio.load_persistent_map_records()

        self.model = None
        self.mg = None
        self.step_count = 0
        self.experiment_run_dir = None
        self.experiment_round_dir = None
        self.experiment_round = 1
        self.current_group = "Experimental"
        self.experimental_configs = []
        self.control_configs = []
        self.experiment_mode = False
        self.optimization_mode = False
        self.log_file = None

        self.op_mode = None  # None | "step" | "auto" | "debrief"
        self.stop_auto = threading.Event()
        self.render_ack = threading.Event()
        self.worker_queue: "queue.Queue" = queue.Queue()

        self.container = ttk.Frame(root)
        self.container.pack(fill="both", expand=True)

        self._build_setup_screen()
        self.root.after(150, self._poll_worker_queue)

    # ------------------------------------------------------------------
    # Setup screen
    # ------------------------------------------------------------------

    def _clear_container(self):
        for child in self.container.winfo_children():
            child.destroy()

    def _build_setup_screen(self):
        self._clear_container()
        cfg = self.last_cfg

        scroller = ScrollableFrame(self.container)
        scroller.pack(fill="both", expand=True)
        body = scroller.body

        ttk.Label(body, text="Project MACI: LLM Cooperative Benchmark Setup", font=("Segoe UI", 16, "bold")).pack(
            anchor="w", padx=16, pady=(14, 4)
        )

        conn = ttk.LabelFrame(body, text="LLM Connection")
        conn.pack(fill="x", padx=16, pady=8)

        self.provider_var = tk.StringVar(value=cfg.get("provider", "openai") if cfg.get("provider") in eio.PROVIDERS else "openai")
        self.base_url_var = tk.StringVar(value=cfg.get("custom_base_url", ""))
        self.api_key_var = tk.StringVar(value=cfg.get("api_key", ""))

        row = ttk.Frame(conn)
        row.pack(fill="x", padx=10, pady=6)
        ttk.Label(row, text="Provider", width=16).pack(side="left")
        provider_combo = ttk.Combobox(row, textvariable=self.provider_var, values=eio.PROVIDERS, state="readonly", width=16)
        provider_combo.pack(side="left", padx=(0, 20))
        provider_combo.bind("<<ComboboxSelected>>", self._on_provider_changed)

        ttk.Label(row, text="Model names are set per agent below; llama.cpp/Ollama/any").pack(side="left")

        row2 = ttk.Frame(conn)
        row2.pack(fill="x", padx=10, pady=6)
        ttk.Label(row2, text="Base URL", width=16).pack(side="left")
        ttk.Entry(row2, textvariable=self.base_url_var, width=50).pack(side="left", fill="x", expand=True)
        ttk.Label(
            row2,
            text="  (llama.cpp server default: http://localhost:8080/v1)",
            foreground="#666",
        ).pack(side="left")

        row3 = ttk.Frame(conn)
        row3.pack(fill="x", padx=10, pady=6)
        ttk.Label(row3, text="API Key", width=16).pack(side="left")
        ttk.Entry(row3, textvariable=self.api_key_var, width=50, show="*").pack(side="left", fill="x", expand=True)
        ttk.Label(row3, text="  (blank = <PROVIDER>_API_KEY / OPENAI_API_KEY env var)", foreground="#666").pack(side="left")

        if not self.base_url_var.get():
            self.base_url_var.set(eio.default_base_url_for_provider(self.provider_var.get()))

        # Map / run settings -------------------------------------------------
        sim = ttk.LabelFrame(body, text="Simulation Settings")
        sim.pack(fill="x", padx=16, pady=8)

        grid_row = ttk.Frame(sim)
        grid_row.pack(fill="x", padx=10, pady=6)

        ttk.Label(grid_row, text="Map Type").grid(row=0, column=0, sticky="w")
        self.map_type_var = tk.StringVar(value=cfg.get("map_type", "Room (Dungeon)"))
        ttk.Combobox(
            grid_row, textvariable=self.map_type_var, state="readonly", width=16,
            values=["Room (Dungeon)", "Maze", "Random"],
        ).grid(row=1, column=0, padx=(0, 16), sticky="w")

        ttk.Label(grid_row, text="Complexity").grid(row=0, column=1, sticky="w")
        self.map_complexity_var = tk.StringVar(value=cfg.get("map_complexity", "H"))
        ttk.Combobox(
            grid_row, textvariable=self.map_complexity_var, state="readonly", width=10,
            values=["H", "M", "E", "Random"],
        ).grid(row=1, column=1, padx=(0, 16), sticky="w")

        ttk.Label(grid_row, text="# Agents").grid(row=0, column=2, sticky="w")
        self.num_agents_var = tk.IntVar(value=cfg.get("num_agents", 2))
        num_agents_spin = ttk.Spinbox(
            grid_row, from_=1, to=MAX_AGENTS, textvariable=self.num_agents_var, width=5,
            command=self._rebuild_agent_tabs,
        )
        num_agents_spin.grid(row=1, column=2, padx=(0, 16), sticky="w")

        ttk.Label(grid_row, text="Reasoning Effort").grid(row=0, column=3, sticky="w")
        self.thinking_effort_var = tk.StringVar(value=cfg.get("thinking_effort", "medium"))
        ttk.Combobox(
            grid_row, textvariable=self.thinking_effort_var, state="readonly", width=10,
            values=["low", "medium", "high"],
        ).grid(row=1, column=3, sticky="w")

        mode_row = ttk.Frame(sim)
        mode_row.pack(fill="x", padx=10, pady=6)
        self.optimization_mode_var = tk.BooleanVar(value=cfg.get("optimization_mode", False))
        self.experiment_mode_var = tk.BooleanVar(value=cfg.get("experiment_mode", False))
        ttk.Checkbutton(mode_row, text="Optimization Mode (self-improvement across sessions)", variable=self.optimization_mode_var).pack(side="left", padx=(0, 20))
        ttk.Checkbutton(mode_row, text="Experiment Mode (Experimental vs Control group)", variable=self.experiment_mode_var).pack(side="left")

        # Communication settings ---------------------------------------------
        comm = ttk.LabelFrame(body, text="Communication Symbol Space")
        comm.pack(fill="x", padx=16, pady=8)
        code_row = ttk.Frame(comm)
        code_row.pack(fill="x", padx=10, pady=6)
        ttk.Label(code_row, text="Allowed broadcast codes").pack(side="left")
        self.broadcast_codes_var = tk.StringVar(value=eio.get_saved_broadcast_codes(cfg))
        ttk.Entry(code_row, textvariable=self.broadcast_codes_var, width=40).pack(side="left", padx=8)

        ttk.Label(comm, text="Symbol space notes (added to every agent's prompt)").pack(anchor="w", padx=10)
        self.symbol_space_text = tk.Text(comm, height=5, wrap="word")
        self.symbol_space_text.pack(fill="x", padx=10, pady=(0, 8))
        self.symbol_space_text.insert("1.0", eio.get_saved_symbol_space_prompt(cfg))

        # Saved maps -----------------------------------------------------
        maps = ttk.LabelFrame(body, text="Saved Map Loader")
        maps.pack(fill="x", padx=16, pady=8)
        map_row = ttk.Frame(maps)
        map_row.pack(fill="x", padx=10, pady=6)
        ttk.Label(map_row, text="Start from").pack(side="left")
        options = ["Generate new map"] + [f"{r['label']} | {r['saved_at']} | {r['filename']}" for r in self.saved_map_records]
        self.saved_map_var = tk.StringVar(value=options[0])
        ttk.Combobox(map_row, textvariable=self.saved_map_var, values=options, state="readonly", width=70).pack(side="left", padx=8)

        # Agent tabs -------------------------------------------------------
        agents_frame = ttk.LabelFrame(body, text="Agent Configuration")
        agents_frame.pack(fill="x", padx=16, pady=8)
        self.agent_notebook = ttk.Notebook(agents_frame)
        self.agent_notebook.pack(fill="both", expand=True, padx=8, pady=8)
        self.agent_widgets = []
        self._rebuild_agent_tabs()

        ttk.Button(body, text="RUN SIMULATION", command=self._on_run_clicked).pack(pady=18, ipadx=20, ipady=6)

    def _on_provider_changed(self, _event=None):
        default_url = eio.default_base_url_for_provider(self.provider_var.get())
        if default_url:
            self.base_url_var.set(default_url)
        elif self.provider_var.get() in ("openai", "custom"):
            self.base_url_var.set("")

    def _rebuild_agent_tabs(self):
        for tab in list(self.agent_notebook.tabs()):
            self.agent_notebook.forget(tab)
        self.agent_widgets = []

        n = max(1, min(MAX_AGENTS, int(self.num_agents_var.get() or 1)))
        prev_agent_configs = self.last_cfg.get("agent_configs", [])

        for i in range(n):
            label = chr(65 + i)
            prev = prev_agent_configs[i] if i < len(prev_agent_configs) else {}
            tab = ttk.Frame(self.agent_notebook)
            self.agent_notebook.add(tab, text=f"Agent {label}")

            default_model = "llama3.1:8b" if self.provider_var.get() in ("ollama", "llamacpp") else "gpt-4o-mini"
            model_var = tk.StringVar(value=prev.get("model_name", default_model))
            vision_var = tk.IntVar(value=prev.get("vision_range", 5))
            speed_var = tk.IntVar(value=prev.get("speed_limit", 1))
            byte_var = tk.IntVar(value=prev.get("byte_limit", 500))
            share_var = tk.IntVar(value=prev.get("map_share_radius", 0))
            default_profile = prev.get("prompt_profile", eio.infer_prompt_profile(model_var.get(), self.provider_var.get()))
            if default_profile not in ("gpt", "gemini", "kimi", "custom"):
                default_profile = "gpt"
            profile_var = tk.StringVar(value=default_profile)

            row1 = ttk.Frame(tab)
            row1.pack(fill="x", padx=8, pady=6)
            ttk.Label(row1, text="Model name", width=14).pack(side="left")
            ttk.Entry(row1, textvariable=model_var, width=30).pack(side="left")
            ttk.Label(row1, text="Prompt profile", width=14).pack(side="left", padx=(20, 0))
            profile_combo = ttk.Combobox(row1, textvariable=profile_var, state="readonly", width=10, values=["gpt", "gemini", "kimi", "custom"])
            profile_combo.pack(side="left")

            sliders = ttk.Frame(tab)
            sliders.pack(fill="x", padx=8, pady=6)

            def add_slider(parent, text, var, frm, to):
                f = ttk.Frame(parent)
                f.pack(side="left", padx=10)
                ttk.Label(f, text=text).pack()
                ttk.Scale(f, from_=frm, to=to, orient="horizontal", variable=var, length=140).pack()
                ttk.Label(f, textvariable=var).pack()

            add_slider(sliders, "Vision Range", vision_var, 1, 10)
            add_slider(sliders, "Speed (blocks/turn)", speed_var, 1, 5)
            add_slider(sliders, "Byte Limit", byte_var, 0, 1000)
            add_slider(sliders, "Map Share Radius", share_var, 0, 10)

            ttk.Label(tab, text="Custom system prompt (used when profile = custom)").pack(anchor="w", padx=8)
            custom_prompt_text = tk.Text(tab, height=8, wrap="word")
            custom_prompt_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            custom_prompt_text.insert("1.0", eio.strip_numeric_suffix_prompt_text(prev.get("custom_system_prompt", ""), include_default=False))

            self.agent_widgets.append({
                "model_var": model_var,
                "vision_var": vision_var,
                "speed_var": speed_var,
                "byte_var": byte_var,
                "share_var": share_var,
                "profile_var": profile_var,
                "custom_prompt_text": custom_prompt_text,
            })

    def _collect_agent_configs(self):
        allowed_codes = self.broadcast_codes_var.get()
        symbol_space_prompt = eio.strip_numeric_suffix_prompt_text(self.symbol_space_text.get("1.0", "end"), include_default=True)
        configs = []
        for w in self.agent_widgets:
            profile = w["profile_var"].get()
            custom_prompt = eio.strip_numeric_suffix_prompt_text(w["custom_prompt_text"].get("1.0", "end"), include_default=False)
            configs.append({
                "model_name": w["model_var"].get().strip() or "gpt-4o-mini",
                "vision_range": int(w["vision_var"].get()),
                "speed_limit": int(w["speed_var"].get()),
                "byte_limit": int(w["byte_var"].get()),
                "map_share_radius": int(w["share_var"].get()),
                "prompt_profile": profile,
                "custom_system_prompt": custom_prompt,
                "allowed_broadcast_codes": allowed_codes,
                "symbol_space_prompt": symbol_space_prompt,
                "optimization_mode": self.optimization_mode_var.get(),
                "coded_communication": True,
            })
        return configs

    def _on_run_clicked(self):
        if self.op_mode:
            return
        try:
            agent_configs = self._collect_agent_configs()
        except Exception as e:
            messagebox.showerror("Config error", str(e))
            return

        provider = self.provider_var.get()
        api_key = self.api_key_var.get().strip()
        base_url = self.base_url_var.get().strip() or None
        thinking_effort = self.thinking_effort_var.get()
        optimization_mode = self.optimization_mode_var.get()
        experiment_mode = self.experiment_mode_var.get()

        eio.save_config({
            "provider": provider,
            "api_key": api_key,
            "map_type": self.map_type_var.get(),
            "map_complexity": self.map_complexity_var.get(),
            "custom_base_url": base_url or "",
            "num_agents": len(agent_configs),
            "thinking_effort": thinking_effort,
            "optimization_mode": optimization_mode,
            "experiment_mode": experiment_mode,
            "allowed_broadcast_codes": self.broadcast_codes_var.get(),
            "symbol_space_prompt": self.symbol_space_text.get("1.0", "end"),
            "agent_configs": agent_configs,
        })

        selected_saved_map_state = None
        saved_label = self.saved_map_var.get()
        if saved_label and saved_label != "Generate new map":
            for record in self.saved_map_records:
                if f"{record['label']} | {record['saved_at']} | {record['filename']}" == saved_label:
                    selected_saved_map_state = record["state"]
                    break

        mg = MapGenerator(25, 25)
        complexity = self.map_complexity_var.get()
        comp = None if complexity == "Random" else complexity

        if selected_saved_map_state:
            t = selected_saved_map_state.get("map_type", "2")
            d = selected_saved_map_state.get("map_diff", "M")
            mg.grid = [row[:] for row in selected_saved_map_state["grid"]]
            mg.height = len(mg.grid)
            mg.width = len(mg.grid[0]) if mg.grid else 25
            mg.type = str(t)
            mg.seed = selected_saved_map_state.get("map_seed", 0)
            mg.diff = str(d)
        else:
            if experiment_mode:
                t = 2
            elif self.map_type_var.get() == "Room (Dungeon)":
                t = 2
            elif self.map_type_var.get() == "Maze":
                t = 1
            else:
                t = random.randint(1, 2)
            s = random.randint(1000, 9999)
            d = comp if comp in ["E", "M", "H"] else random.choice(["E", "M", "H"])
            mg.generate_map(f"{t}{s}{d}")

        self.experimental_configs = agent_configs
        control_configs = []
        for c in agent_configs:
            cc = c.copy()
            cc["vision_range"] = 25
            cc["byte_limit"] = 10000
            control_configs.append(cc)
        self.control_configs = control_configs
        self.current_group = "Experimental"
        self.experiment_mode = experiment_mode
        self.optimization_mode = optimization_mode

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join("logs", f"log_{timestamp}.txt")
        os.makedirs("logs", exist_ok=True)
        self.experiment_run_dir = eio.create_experiment_run_dir(timestamp)

        try:
            model = MACI_Model(
                num_agents=len(agent_configs),
                map_generator=mg,
                agent_configs=agent_configs,
                thinking_effort=thinking_effort,
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                optimization_mode=optimization_mode,
                log_file=self.log_file,
                llm_io_log_path=os.path.join(self.experiment_run_dir, "llm_io.jsonl"),
            )
        except Exception as e:
            messagebox.showerror("Failed to start simulation", str(e))
            return

        eio.sanitize_interaction_rules(model)
        if selected_saved_map_state:
            eio.apply_map_state_to_model(model, mg, selected_saved_map_state)
            eio.reset_agent_positions_for_map(model, mg, agent_configs)

        self.mg = mg
        self.model = model
        self.step_count = 0
        self.experiment_round = 1
        self.experiment_round_dir = eio.create_experiment_round_dir(self.experiment_run_dir, self.experiment_round, self.current_group)
        eio.save_turn_snapshot(self.model, self.mg, self.step_count, self.experiment_round_dir, self.current_group)

        self._build_run_screen()

    # ------------------------------------------------------------------
    # Run screen
    # ------------------------------------------------------------------

    def _build_run_screen(self):
        self._clear_container()

        top = ttk.Frame(self.container)
        top.pack(fill="x", padx=10, pady=6)
        self.step_label_var = tk.StringVar()
        ttk.Label(top, textvariable=self.step_label_var, font=("Segoe UI", 12, "bold")).pack(side="left")
        self.token_label_var = tk.StringVar()
        ttk.Label(top, textvariable=self.token_label_var, foreground="#333").pack(side="right")

        main = ttk.Frame(self.container)
        main.pack(fill="both", expand=True, padx=10, pady=6)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, background="#1e1e1e")
        self.canvas.pack(fill="both", expand=True)
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(left, textvariable=self.status_var, foreground="#555").pack(anchor="w", pady=(4, 0))

        right = ScrollableFrame(main, width=440)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        panel = right.body

        controls = ttk.LabelFrame(panel, text="Controls")
        controls.pack(fill="x", padx=8, pady=8)
        btn_row = ttk.Frame(controls)
        btn_row.pack(fill="x", padx=6, pady=6)
        self.step_btn = ttk.Button(btn_row, text="Step", command=self._on_step_clicked)
        self.step_btn.pack(side="left", padx=4)
        self.auto_btn = ttk.Button(btn_row, text="Auto-Run", command=self._on_auto_run_toggle)
        self.auto_btn.pack(side="left", padx=4)
        ttk.Button(btn_row, text="Reset to Setup", command=self._on_reset_clicked).pack(side="left", padx=4)

        if self.experiment_mode:
            group_frame = ttk.LabelFrame(panel, text="Experiment Group")
            group_frame.pack(fill="x", padx=8, pady=8)
            self.group_var = tk.StringVar(value=self.current_group)
            ttk.Radiobutton(group_frame, text="Experimental", value="Experimental", variable=self.group_var, command=self._on_group_switch).pack(side="left", padx=8)
            ttk.Radiobutton(group_frame, text="Control", value="Control", variable=self.group_var, command=self._on_group_switch).pack(side="left", padx=8)

        if self.optimization_mode:
            diff_frame = ttk.LabelFrame(panel, text="Difficulty Adjustment")
            diff_frame.pack(fill="x", padx=8, pady=8)
            self.diff_var = tk.StringVar(value=getattr(self.mg, "diff", "M"))
            ttk.Combobox(diff_frame, textvariable=self.diff_var, state="readonly", values=["E", "M", "H"], width=6).pack(side="left", padx=6)
            ttk.Button(diff_frame, text="Apply & Restart", command=self._on_apply_difficulty).pack(side="left", padx=6)

        map_tools = ttk.LabelFrame(panel, text="Map Tools")
        map_tools.pack(fill="x", padx=8, pady=8)
        ttk.Button(map_tools, text="Save Current Map", command=self._on_save_map).pack(side="left", padx=6, pady=6)
        ttk.Button(map_tools, text="Reroll Map & Restart Round", command=self._on_reroll_map).pack(side="left", padx=6, pady=6)

        agents_frame = ttk.LabelFrame(panel, text="Agent Decisions (latest turn)")
        agents_frame.pack(fill="x", padx=8, pady=8)
        self.agent_status_vars = []
        for i, agent in enumerate(self.model.agents):
            label = chr(65 + i)
            f = ttk.LabelFrame(agents_frame, text=f"Agent {label} ({agent.model_name})")
            f.pack(fill="x", padx=4, pady=4)
            v = tk.StringVar(value="(no turn yet)")
            ttk.Label(f, textvariable=v, wraplength=380, justify="left").pack(anchor="w", padx=6, pady=4)
            self.agent_status_vars.append(v)

        comm_frame = ttk.LabelFrame(panel, text="Communication Log")
        comm_frame.pack(fill="both", padx=8, pady=8)
        self.comm_list = tk.Listbox(comm_frame, height=8)
        self.comm_list.pack(fill="both", expand=True, padx=6, pady=6)

        token_frame = ttk.LabelFrame(panel, text="Token Usage")
        token_frame.pack(fill="x", padx=8, pady=8)
        columns = ("agent", "model", "turns", "prompt", "completion", "reasoning", "cached", "total")
        self.token_tree = ttk.Treeview(token_frame, columns=columns, show="headings", height=len(self.model.agents) + 1)
        for c in columns:
            self.token_tree.heading(c, text=c)
            self.token_tree.column(c, width=60, anchor="center")
        self.token_tree.pack(fill="x", padx=6, pady=6)

        cot_frame = ttk.LabelFrame(panel, text="LLM Output / Chain-of-Thought")
        cot_frame.pack(fill="both", expand=True, padx=8, pady=8)
        self.cot_agent_var = tk.StringVar(value="A")
        cot_row = ttk.Frame(cot_frame)
        cot_row.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(cot_row, text="Agent").pack(side="left")
        ttk.Combobox(
            cot_row, textvariable=self.cot_agent_var, state="readonly", width=6,
            values=[chr(65 + i) for i in range(len(self.model.agents))],
            postcommand=lambda: self._refresh_cot_view(),
        ).pack(side="left", padx=6)
        ttk.Button(cot_row, text="Refresh", command=self._refresh_cot_view).pack(side="left")
        self.cot_text = tk.Text(cot_frame, height=14, wrap="word")
        self.cot_text.pack(fill="both", expand=True, padx=6, pady=6)
        ttk.Label(
            cot_frame,
            text=f"Full history for every agent/turn is saved to: {self.model.llm_io_log_path}",
            foreground="#666", wraplength=400, justify="left",
        ).pack(anchor="w", padx=6, pady=(0, 6))

        self._redraw_all()
        self._set_controls_state()

    def _agent_index_from_label(self, label):
        idx = ord(label.upper()) - 65
        if 0 <= idx < len(self.model.agents):
            return idx
        return 0

    def _refresh_cot_view(self):
        idx = self._agent_index_from_label(self.cot_agent_var.get())
        agent = self.model.agents[idx]
        decision = getattr(agent, "last_decision", {}) or {}
        text = (
            f"Turn: {decision.get('turn', 0)}\n"
            f"Action: {decision.get('action', '')}  Blocks: {decision.get('blocks', '')}\n"
            f"Broadcast: {decision.get('broadcast_message', '')} ({decision.get('broadcast_meaning', '')})\n"
            f"Notes: {decision.get('notes', '')}\n"
            f"{'-'*60}\n"
            f"Reasoning / chain-of-thought:\n{decision.get('reasoning_content', '') or '(none captured for this turn)'}\n"
            f"{'-'*60}\n"
            f"Full raw model response:\n{decision.get('raw_response', '')}\n"
        )
        self.cot_text.delete("1.0", "end")
        self.cot_text.insert("1.0", text)

    # ------------------------------------------------------------------
    # Canvas rendering
    # ------------------------------------------------------------------

    def _redraw_all(self):
        self._redraw_map()
        self._update_step_label()
        self._update_agent_panels()
        self._update_comm_log()
        self._update_token_table()
        self._refresh_cot_view()

    def _update_step_label(self):
        self.step_label_var.set(f"{self.current_group} | Round {self.experiment_round} | Step {self.step_count}")

    def _update_token_table(self):
        for row in self.token_tree.get_children():
            self.token_tree.delete(row)
        summary = eio.get_model_token_summary(self.model)
        for item in summary["agents"]:
            self.token_tree.insert("", "end", values=(
                item["agent"], item["model"], item["turns"], item["prompt_tokens"],
                item["completion_tokens"], item["reasoning_tokens"], item["cached_tokens"], item["total_tokens"],
            ))
        total = summary["total"]
        self.token_label_var.set(
            f"Tokens: {total['total_tokens']} (prompt={total['prompt_tokens']}, "
            f"completion={total['completion_tokens']}, reasoning={total['reasoning_tokens']})"
        )

    def _update_agent_panels(self):
        for i, agent in enumerate(self.model.agents):
            decision = getattr(agent, "last_decision", {}) or {}
            if not decision.get("turn"):
                self.agent_status_vars[i].set("(no turn yet)")
                continue
            text = (
                f"Turn {decision.get('turn')} @ {agent.pos}\n"
                f"Action: {decision.get('action')} x{decision.get('blocks')}"
                f"{'  [BLOCKED]' if decision.get('blocked') else ''}\n"
                f"Reason: {decision.get('reason', '')}\n"
                f"Broadcast: {decision.get('broadcast_message', '')}\n"
                f"Notes: {decision.get('notes', '')}"
            )
            self.agent_status_vars[i].set(text)

    def _update_comm_log(self):
        self.comm_list.delete(0, "end")
        for item in getattr(self.model, "communication_log", [])[-30:]:
            targets = ",".join(item.get("to", []))
            line = f"[{item.get('agent_turn', '?')}] {item.get('from', '?')} -> {targets}: " + eio.broadcast_display(
                item.get("message", ""), item.get("meaning", ""), max_meaning=60
            )
            self.comm_list.insert("end", line)
        self.comm_list.yview_moveto(1.0)

    def _redraw_map(self):
        self.canvas.delete("all")
        mg = self.mg
        model = self.model
        agent_positions = {}
        for i, agent in enumerate(model.agents):
            agent_positions[agent.pos] = (chr(65 + i), AGENT_COLORS[i % len(AGENT_COLORS)])

        for y, row in enumerate(mg.grid):
            for x, cell_value in enumerate(row):
                left, top = x * CELL, y * CELL
                right, bottom = left + CELL, top + CELL
                fill = "#2f3437"
                text, text_fill = "", "#ffffff"

                if cell_value == 1:
                    fill = "#050505"
                if (x, y) == model.target_pos:
                    fill, text, text_fill = "#3f3f46", "F", "#facc15"
                if (x, y) in model.switches:
                    pressed = any(a.pos == (x, y) for a in model.agents)
                    fill = "#f97316" if pressed else "#7c2d12"
                    text = "S"
                if (x, y) in model.gates:
                    fill = "#0f766e" if (x, y) in getattr(model, "open_gates", set()) else "#52525b"
                    text = "G"
                if (x, y) in getattr(model, "fake_symbols", {}):
                    sym = str(model.fake_symbols[(x, y)])[:1]
                    text = sym
                    if sym == "K":
                        fill, text_fill = "#713f12", "#fde047"
                    elif sym == "D":
                        fill, text_fill = "#7f1d1d", "#fecaca"
                    else:
                        fill, text_fill = "#3b3b3b", "#fbbf24"

                self.canvas.create_rectangle(left, top, right, bottom, fill=fill, outline="#4b5563")

                if (x, y) in agent_positions:
                    label, color = agent_positions[(x, y)]
                    pad = 3
                    self.canvas.create_oval(left + pad, top + pad, right - pad, bottom - pad, fill=color, outline="#ffffff")
                    self.canvas.create_text((left + right) / 2, (top + bottom) / 2, text=label, fill="#ffffff", font=("Segoe UI", 10, "bold"))
                elif text:
                    self.canvas.create_text((left + right) / 2, (top + bottom) / 2, text=text, fill=text_fill, font=("Segoe UI", 10, "bold"))

        self.canvas.configure(scrollregion=(0, 0, mg.width * CELL, mg.height * CELL))

    # ------------------------------------------------------------------
    # Worker thread orchestration
    # ------------------------------------------------------------------

    def _set_controls_state(self):
        state = "disabled" if self.op_mode else "normal"
        if hasattr(self, "step_btn"):
            self.step_btn.configure(state=state)
        if hasattr(self, "auto_btn"):
            self.auto_btn.configure(state="normal" if self.op_mode in (None, "auto") else "disabled")
            self.auto_btn.configure(text="Stop" if self.op_mode == "auto" else "Auto-Run")

    def _on_step_clicked(self):
        if self.op_mode:
            return
        self.op_mode = "step"
        self._set_controls_state()
        self.status_var.set("LLM agents are thinking... (see terminal for live logs)")

        def work():
            try:
                self.model.step()
                self.worker_queue.put(("step_done", None))
            except Exception as e:
                self.worker_queue.put(("error", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _on_auto_run_toggle(self):
        if self.op_mode == "auto":
            self.stop_auto.set()
            self.status_var.set("Stopping after current step...")
            return
        if self.op_mode:
            return

        self.stop_auto.clear()
        self.op_mode = "auto"
        self._set_controls_state()
        self.status_var.set("Auto-running... (see terminal for live logs)")

        def work():
            try:
                while not self.stop_auto.is_set():
                    if any(agent.is_done for agent in self.model.agents):
                        break
                    self.model.step()
                    # Wait for the main thread to finish reading/rendering this
                    # turn's state before mutating the model again, so the UI
                    # never reads agent/model state mid-mutation.
                    self.render_ack.clear()
                    self.worker_queue.put(("step_done", None))
                    self.render_ack.wait(timeout=5)
                if any(agent.is_done for agent in self.model.agents) and not getattr(self.model, "success_debrief_done", False):
                    self.model.run_success_debrief()
                    eio.save_turn_snapshot(self.model, self.mg, self.step_count, self.experiment_round_dir, self.current_group)
                self.worker_queue.put(("auto_done", None))
            except Exception as e:
                self.worker_queue.put(("error", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _on_group_switch(self):
        if self.op_mode:
            self.group_var.set(self.current_group)
            return
        selected = self.group_var.get()
        if selected == self.current_group:
            return
        configs = self.experimental_configs if selected == "Experimental" else self.control_configs
        self._rebuild_model(configs, selected, reason=f"switch_to_{selected.lower()}")

    def _on_apply_difficulty(self):
        if self.op_mode:
            return
        new_diff = self.diff_var.get()
        seed_str = f"{self.mg.type}{self.mg.seed}{new_diff}"
        self.mg.generate_map(seed_str)
        self._rebuild_model(self.experimental_configs if self.current_group == "Experimental" else self.control_configs, self.current_group, keep_map=True, reason="difficulty_change")

    def _on_reroll_map(self):
        if self.op_mode:
            return
        mtype = str(random.randint(1, 2))
        mdiff = random.choice(["E", "M", "H"])
        seed = random.randint(1000, 9999)
        self.mg.generate_map(f"{mtype}{seed}{mdiff}")
        self._rebuild_model(self.experimental_configs if self.current_group == "Experimental" else self.control_configs, self.current_group, keep_map=True, reason="reroll_map")

    def _on_save_map(self):
        state = eio.capture_map_state(self.model, self.mg, label=f"map_round_{self.experiment_round}_step_{self.step_count}")
        path = eio.save_persistent_map_state(state["label"], state)
        messagebox.showinfo("Map saved", f"Saved to:\n{path}")

    def _rebuild_model(self, configs, group_name, keep_map=False, reason="restart"):
        old_model = self.model
        model = MACI_Model(
            num_agents=len(configs),
            map_generator=self.mg,
            agent_configs=configs,
            thinking_effort=old_model.thinking_effort,
            provider=old_model.provider,
            api_key=old_model.api_key,
            base_url=old_model.base_url,
            optimization_mode=getattr(old_model, "optimization_mode", False),
            log_file=self.log_file,
            llm_io_log_path=old_model.llm_io_log_path,
        )
        eio.apply_existing_interaction_rules(model, old_model)
        self.model = model
        self.current_group = group_name
        self.step_count = 0
        self.experiment_round += 1
        self.experiment_round_dir = eio.create_experiment_round_dir(self.experiment_run_dir, self.experiment_round, self.current_group)
        eio.save_turn_snapshot(self.model, self.mg, self.step_count, self.experiment_round_dir, self.current_group)
        self._build_run_screen()

    def _on_reset_clicked(self):
        if self.op_mode:
            return
        if not messagebox.askyesno("Reset", "Discard the current run and go back to setup?"):
            return
        self.model = None
        self._build_setup_screen()

    def _poll_worker_queue(self):
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                if kind == "step_done":
                    self.step_count += 1
                    eio.save_turn_snapshot(self.model, self.mg, self.step_count, self.experiment_round_dir, self.current_group)
                    self._redraw_all()
                    self.render_ack.set()
                    if self.op_mode == "step":
                        if any(agent.is_done for agent in self.model.agents) and not getattr(self.model, "success_debrief_done", False):
                            self.op_mode = "debrief"
                            self._set_controls_state()
                            self.status_var.set("Target reached! Running success debrief...")
                            self._start_debrief_thread()
                        else:
                            self.op_mode = None
                            self._set_controls_state()
                            self.status_var.set("Ready.")
                    # while op_mode == "auto" the worker thread keeps looping on its own
                elif kind == "auto_done":
                    self.op_mode = None
                    self._set_controls_state()
                    self._redraw_all()
                    self.status_var.set("Auto-run stopped.")
                elif kind == "debrief_done":
                    self.op_mode = None
                    self._set_controls_state()
                    self._redraw_all()
                    self.status_var.set("Success debrief complete.")
                elif kind == "error":
                    self.op_mode = None
                    self.stop_auto.set()
                    self._set_controls_state()
                    self.status_var.set("Error - see popup.")
                    messagebox.showerror("Simulation error", str(payload))
        except queue.Empty:
            pass

        self.root.after(150, self._poll_worker_queue)

    def _start_debrief_thread(self):
        def work():
            try:
                self.model.run_success_debrief()
                eio.save_turn_snapshot(self.model, self.mg, self.step_count, self.experiment_round_dir, self.current_group)
                self.worker_queue.put(("debrief_done", None))
            except Exception as e:
                self.worker_queue.put(("error", str(e)))

        threading.Thread(target=work, daemon=True).start()


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass
    app = MaciGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
