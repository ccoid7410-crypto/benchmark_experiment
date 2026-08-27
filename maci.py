import time
import argparse
from map_gen import MapGenerator
from agents import MACI_Model
from rich.console import Console
from rich.text import Text

console = Console()

def render_map(model, map_generator, step):
    """
    Renders the current state of the map to the terminal using the 'rich' library.
    Displays walls, open paths, agents (A, B, C...), and the target (F).
    """
    console.print(f"\n[bold yellow]============= Step {step} Map =============[/bold yellow]")
    
    # Optional: Comment this out if you don't want to see the target coordinates in terminal
    if hasattr(model, 'target_pos'):
        console.print(f"[bold yellow]Target (F) coordinates: {model.target_pos}[/bold yellow]")
    
    # Map agents to alphabetical symbols (A, B, C...) and assign colors
    agent_markers = {}
    colors = ["bold red", "bold blue", "bold green", "bold magenta", "bold cyan", "bold yellow"]
    for i, agent in enumerate(model.agents):
        symbol = chr(65 + i) # 65 is 'A' in ASCII
        color = colors[i % len(colors)]
        agent_markers[agent.pos] = (symbol, color)

    # Build and print the map row by row
    for y, row in enumerate(map_generator.grid):
        line = Text()
        for x, cell in enumerate(row):
            if (x, y) in agent_markers:
                # Render Agent
                symbol, color = agent_markers[(x, y)]
                line.append(symbol, style=color)
            elif hasattr(model, 'target_pos') and (x, y) == model.target_pos:
                # Render Target 'F'
                line.append("F", style="bold yellow on grey37")
            elif hasattr(model, 'fake_symbols') and (x, y) in model.fake_symbols:
                line.append(model.fake_symbols[(x, y)], style="bold yellow on grey37")
            elif cell == 1:
                # Render Wall
                line.append(map_generator._get_wall_char(x, y), style="white")
            else:
                # Render Empty Space
                line.append(" ", style="white")
        console.print(line)
    print("\n")

def get_simulation_setup(args):
    """
    Interactive prompt to collect setup configuration from the user.
    Allows dynamic customization of agent count and their specific LLM models.
    """
    console.print("\n[bold cyan]--- Simulation Configuration ---[/bold cyan]")
    
    # 0. Get Provider and API Key
    provider = input("Select Provider (openai / gemini / ollama / llamacpp / openrouter / custom) [Default: openai]: ").strip().lower()
    if provider not in ["openai", "gemini", "ollama", "llamacpp", "openrouter", "custom"]:
        provider = "openai"

    api_key = input(f"Enter {provider.upper()} API Key (Press Enter to use <PROVIDER>_API_KEY / OPENAI_API_KEY env var): ").strip()

    default_base_url = {
        "ollama": "http://localhost:11434/v1",
        "llamacpp": "http://localhost:8080/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }.get(provider)
    base_url_hint = default_base_url or ("required for 'custom'" if provider == "custom" else "skip")
    base_url = input(f"Enter Custom Base URL for OpenAI compatible endpoint (Default: {base_url_hint}): ").strip()
    base_url = base_url if base_url else default_base_url
    
    # 1. Get number of agents
    num_agents = 0
    while True:
        try:
            user_input = input("Enter the number of agents to spawn (e.g., 2): ").strip()
            num_agents = int(user_input)
            if num_agents > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    # 2. Get the model name for each agent
    agent_models_list = []
    for i in range(num_agents):
        symbol = chr(65 + i)
        print(f"\n[Agent {symbol}] Provide the Ollama/API model name.")
        print("Examples: 'llama3.1:8b', 'qwen2.5:7b', 'gemma2:9b'")
        model_name = input(f"Model for Agent {symbol} (Press Enter for default 'llama3.1:8b'): ").strip()
        
        # Use default if user leaves it blank
        if not model_name:
            model_name = "llama3.1:8b"
            
        agent_models_list.append(model_name)
        console.print(f" -> Agent {symbol} assigned model: [bold green]{model_name}[/bold green]")
        
    # 3. Get thinking effort
    thinking_effort = input("\nEnter the reasoning effort (low, medium, high) [Default: medium]: ").strip().lower()
    if thinking_effort not in ["low", "medium", "high"]:
        thinking_effort = "medium"
    console.print(f" -> Reasoning effort set to: [bold green]{thinking_effort}[/bold green]")
        
    # 4. Get Map Complexity
    map_complexity = input("\nEnter Map Complexity (E for Easy, M for Medium, H for Hard) [Default: Random]: ").strip().upper()
    if map_complexity not in ['E', 'M', 'H']:
        map_complexity = None
        console.print(" -> Map Complexity set to: [bold green]Random[/bold green]\n")
    else:
        console.print(f" -> Map Complexity set to: [bold green]{map_complexity}[/bold green]\n")
        
    # 5. Get Optimization Mode
    if args.optimize:
        optimization_mode = True
        console.print(" -> Optimization Mode: [bold green]ON (via CLI flag)[/bold green]\n")
    else:
        opt_mode_input = input("\nEnable Optimization Mode (Multi-Session Self-Improvement)? (y/n) [Default: n]: ").strip().lower()
        optimization_mode = True if opt_mode_input == 'y' else False
        if optimization_mode:
            console.print(" -> Optimization Mode: [bold green]ON[/bold green]\n")
        
    return num_agents, agent_models_list, thinking_effort, map_complexity, provider, api_key, base_url, optimization_mode

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MACI Blind Maze Exploration")
    parser.add_argument("--optimize", action="store_true", help="Enable Optimization Mode")
    args = parser.parse_args()

    console.print("[bold green] ------ Project MACI: Blind Maze Exploration ------ [/bold green]")

    # Phase 1: User Configuration
    num_agents, agent_models_list, thinking_effort, map_complexity, provider, api_key, base_url, optimization_mode = get_simulation_setup(args)

    # Phase 2: Map Generation
    console.print("\n[bold cyan]Generating Map...[/bold cyan]")
    mg = MapGenerator(25, 25)
    mg.get_random_map(difficulty=map_complexity)

    # Phase 3: Model Initialization
    console.print("[bold cyan]Deploying Agents...[/bold cyan]")
    # Pass the list of configured models to the environment
    agent_configs = []
    for model_name in agent_models_list:
        agent_configs.append({
            "model_name": model_name,
            "vision_range": 5,
            "speed_limit": 1,
            "byte_limit": 500,
            "map_share_radius": 0,
            "optimization_mode": optimization_mode,
        })

    maci_world = MACI_Model(
        num_agents=num_agents, 
        map_generator=mg, 
        agent_configs=agent_configs, 
        thinking_effort=thinking_effort, 
        provider=provider, 
        api_key=api_key,
        base_url=base_url,
        optimization_mode=optimization_mode
    )

    # Phase 4: Initial Render (Step 0)
    render_map(maci_world, mg, 0)

    # Phase 5: Infinite Deathmatch Loop
    # Continues until at least one agent steps on 'F'
    step_count = 0
    while True:
        step_count += 1
        
        # All agents take their turn
        maci_world.step()
        
        # Re-render the map after everyone moved
        render_map(maci_world, mg, step_count)
        
        # Check Win Condition
        if any(agent.is_done for agent in maci_world.agents):
            console.print(f"\n[bold yellow]SUCCESS: An agent found the Target 'F' at Step {step_count}![/bold yellow]")
            if getattr(maci_world, 'optimization_mode', False):
                console.print("\n[bold cyan]--- Optimization Mode: Reflection & Restart ---[/bold cyan]")
                maci_world.reflect_and_restart_session()
                step_count = 0
                console.print("\n[bold green] ------ New Session Started ------ [/bold green]")
                render_map(maci_world, mg, 0)
                continue
            break
            
        # Pause briefly to make terminal output readable
        time.sleep(1) 

    console.print("[bold green] ------ Simulation Terminated ------ [/bold green]")
