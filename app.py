import base64
import csv
import io
import json
import logging
import os
import secrets
import sys
import threading
import time
from pathlib import Path

if sys.platform == "linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "osmesa")

import gymnasium
import gymnasium_robotics
import numpy as np
import streamlit as st
import torch
from PIL import Image

import train_compare


ROOT = Path(__file__).resolve().parent
DEFAULT_MAX_STEPS = 1000
DEFAULT_SEED = 42
MODEL_CLASSES = {
    "DeMa": ("dema", train_compare.DeMa),
    "Decision Transformer": ("decision_transformer", train_compare.DecisionTransformer),
    "Multi-Grained SSM": ("multi-grained_ssm", train_compare.MultiGrainedSSM),
    "DeMa + SER": ("dema_ser", train_compare.DeMa),
}
MAZE = [[1, 1, 1, 1, 1], [1, "g", 0, 0, 1], [1, 1, 1, 0, 1], [1, "r", 0, 0, 1], [1, 1, 1, 1, 1]]
TASKS = {
    "HalfCheetah": {
        "env_id": "HalfCheetah-v5",
        "kwargs": {"xml_file": str(ROOT / "assets" / "half_cheetah_large_floor.xml")},
        "checkpoint_dir": ROOT / "assets" / "models",
        "dataset_mean": 12089.21,
        "random_return": -295.60,
        "target": 12000,
        "mission": "Travel as far forward as you can, without wasting motor effort.",
        "rule": "Every decision scores forward speed minus motor effort. There is no finish line: maximize the total before time runs out. Falling has no separate penalty here; it hurts the score when forward progress suffers.",
    },
    "AntMaze": {
        "env_id": "AntMaze_UMaze-v4",
        "kwargs": {"maze_map": MAZE, "reward_type": "sparse", "continuing_task": True, "reset_target": False},
        "checkpoint_dir": ROOT / "assets" / "models" / "antmaze",
        "dataset_mean": 438.2091,
        "random_return": 0.0,
        "target": 538,
        "mission": "Find the green goal by going around the wall. Then stay near it.",
        "rule": "A decision earns +1 only inside the goal radius (0.45 m), otherwise 0. Getting closer earns no points. This task has no motor-effort or collision penalty in its returned reward.",
    },
}


def bfs_distances(maze_map, goal_cell):
    from collections import deque

    distances = {tuple(goal_cell): 0}
    queue = deque([tuple(goal_cell)])
    while queue:
        row, col = queue.popleft()
        for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            cell = (next_row, next_col)
            if 0 <= next_row < len(maze_map) and 0 <= next_col < len(maze_map[0]) and maze_map[next_row][next_col] != 1 and cell not in distances:
                distances[cell] = distances[(row, col)] + 1
                queue.append(cell)
    return distances


def find_checkpoints(checkpoint_dir):
    return {label: str(checkpoint_dir / (key + ".pth")) for label, (key, _) in MODEL_CLASSES.items() if (checkpoint_dir / (key + ".pth")).is_file()}


@st.cache_resource
def simulation_lock():
    return threading.RLock()


@st.cache_resource
def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def configure_checkpoint(checkpoint):
    for name in ("d_model", "num_layers", "norm_first", "dema_norm_first", "token_layout"):
        setattr(train_compare, name, checkpoint[name])
    has_timestep = checkpoint.get("uses_timestep_embedding", any("timestep_embed" in key for key in checkpoint["model"]))
    train_compare.dt_uses_timestep_embedding = has_timestep
    train_compare.dema_uses_timestep_embedding = has_timestep
    train_compare.dropout = 0.0


@st.cache_resource
def cached_model(path, label):
    checkpoint = load_checkpoint(path)
    configure_checkpoint(checkpoint)
    model = MODEL_CLASSES[label][1](checkpoint["obs_dim"], checkpoint["act_dim"])
    model.load_state_dict(checkpoint["model"])
    model.to(train_compare.device)
    model.eval()
    return model


def load_model(path, label):
    checkpoint = load_checkpoint(path)
    configure_checkpoint(checkpoint)
    model = cached_model(path, label)
    model.eval()
    return model, checkpoint


def episode_steps(task):
    limit = gymnasium.spec(TASKS[task]["env_id"]).max_episode_steps
    return min(DEFAULT_MAX_STEPS, limit) if limit is not None else DEFAULT_MAX_STEPS


def new_run_setup(task, previous=None):
    if task != "AntMaze":
        return DEFAULT_SEED, None
    seed = secrets.randbelow(1_000_000)
    if previous and seed == previous.get("seed"):
        seed = (seed + 1) % 1_000_000
    return seed, {"reset_cell": [3, 1], "goal_cell": [1, 1]}


class LocalEnvironment:
    def __init__(self, task):
        self.task = task

    def recover_environment(self, render_mode=None):
        config = TASKS[self.task]
        kwargs = dict(config["kwargs"])
        if render_mode:
            kwargs.update(width=640, height=360)
            if self.task == "HalfCheetah":
                kwargs["camera_id"] = 0
        env = gymnasium.make(config["env_id"], render_mode=render_mode, **kwargs)
        if render_mode and self.task == "AntMaze":
            env.unwrapped.model.site_rgba[env.unwrapped.target_site_id] = [0.03, 0.75, 0.45, 0.85]
            env.unwrapped.model.geom_rgba[env.unwrapped.model.geom_bodyid > 0] = [0.85, 0.06, 0.06, 1.0]
            env.unwrapped.ant_env.mujoco_renderer.default_cam_config = {"distance": 18, "azimuth": 90, "elevation": -90, "lookat": [0, 0, 0.5]}
        return env


class RandomPolicy(torch.nn.Module):
    def __init__(self, act_dim, seed):
        super().__init__()
        self.act_dim = act_dim
        self.rng = np.random.default_rng(seed)

    def forward(self, states, actions, returns, timesteps):
        values = self.rng.uniform(-1.0, 1.0, (states.shape[0], states.shape[1], self.act_dim)).astype(np.float32)
        return torch.as_tensor(values, device=states.device)


def encode_frame(frame):
    output = io.BytesIO()
    Image.fromarray(frame).save(output, format="JPEG", quality=66)
    return base64.b64encode(output.getvalue()).decode("ascii")


@st.cache_data(show_spinner=False)
def initial_episode(task, seed, reset_options):
    env = LocalEnvironment(task).recover_environment(render_mode="rgb_array")
    try:
        observation, info = env.reset(seed=seed * 1000, options=reset_options)
        distances = bfs_distances(MAZE, reset_options["goal_cell"]) if task == "AntMaze" else None
        action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        detail = train_compare.rollout_snapshot(env, observation, action, action, info, 0, 0.0, 0.0, TASKS[task]["target"], distances, False)
        return {"name": "Ready to explore", "kind": "preview", "details": [detail], "frames": [encode_frame(env.render())], "frame_steps": [0]}
    finally:
        env.close()


def run_episode(task, label, path, seed, target, max_steps, reset_options, intervention, progress):
    source_label = label if label in MODEL_CLASSES else next(name for name, (key, _) in MODEL_CLASSES.items() if Path(path).stem == key)
    if label == "Random actions":
        checkpoint = load_checkpoint(path)
        configure_checkpoint(checkpoint)
        model = RandomPolicy(checkpoint["act_dim"], seed)
    else:
        model, checkpoint = load_model(path, source_label)
    data = {
        "dataset": LocalEnvironment(task),
        "state_mean": checkpoint["state_mean"],
        "state_std": checkpoint["state_std"],
        "actions": torch.zeros(1, 1, checkpoint["act_dim"]),
        "used_mean_return": TASKS[task]["dataset_mean"],
    }
    train_compare.seed = seed
    train_compare.measured_random_return = TASKS[task]["random_return"]
    scores = train_compare.evaluate(
        model, data, float(target), 1, max_steps=max_steps,
        capture_frames=True, capture_details=True, capture_stride=3,
        reset_options=reset_options,
        cell_distances=bfs_distances(MAZE, reset_options["goal_cell"]) if task == "AntMaze" else None,
        brake_window=(400, 440) if intervention else None,
        on_step=progress, frame_encoder=encode_frame, fast_inference=True,
    )
    return {
        "name": label,
        "kind": "random" if label == "Random actions" else "trained",
        "details": scores["frame_details"],
        "frames": scores["frames"],
        "frame_steps": scores["frame_steps"],
    }


def telemetry_csv(episodes):
    output = io.StringIO()
    fields = list(episodes[0]["details"][0])
    writer = csv.DictWriter(output, fieldnames=["agent"] + fields)
    writer.writeheader()
    for episode in episodes:
        for detail in episode["details"]:
            writer.writerow({"agent": episode["name"], **{key: json.dumps(value) if isinstance(value, list) else value for key, value in detail.items()}})
    return output.getvalue()


def build_player_html(payload):
    template = (ROOT / "assets" / "simulation.html").read_text(encoding="utf-8")
    if not payload.get("preview"):
        payload = {**payload, "csv": telemetry_csv(payload["episodes"])}
    serialized = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    template = template.replace("__SIMULATION_STYLE__", (ROOT / "assets" / "simulation.css").read_text(encoding="utf-8-sig"))
    template = template.replace("__SIMULATION_SCRIPT__", (ROOT / "assets" / "simulation.js").read_text(encoding="utf-8-sig"))
    return template.replace("__SIMULATION_DATA__", serialized)


def show_player(payload):
    st.iframe(build_player_html(payload), height="content", alt="Interactive robot decision replay with rewards, actions, and comparison charts")


def main():
    torch.set_num_threads(1)
    st.set_page_config(page_title="DeMa - Reinforcement Learning Simulator", layout="wide", initial_sidebar_state="expanded")
    st.html("""<style>
    .stApp{background:#faf9f6;color:#292c29;font-family:Arial,Helvetica,sans-serif}
    .stApp h1,.stApp h2,.stApp h3,.stApp p,.stApp label,.stApp button,.stApp input,.stApp select,.stApp summary,.stApp [data-testid="stMarkdownContainer"],.stApp [data-baseweb="select"]{font-family:Arial,Helvetica,sans-serif!important;letter-spacing:normal}
    .block-container{max-width:1580px;padding-top:4rem;padding-bottom:0.7rem}
    [data-testid="stSidebar"]{background:#f0f0ea;border-right:1px solid #d8d9d2}
    [data-testid="stSidebar"] h2{font-size:14px;font-weight:600;margin-top:8px}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"]{display:none}
    .project-heading{margin:0 0 6px;width:100%;min-width:0;container-type:inline-size}.project-heading h1{font:700 min(34px,3cqw)/1.3 Arial,Helvetica,sans-serif;margin:0;padding:2px 0;white-space:nowrap;overflow:visible}.project-heading div{font:600 13px/1.4 Arial,Helvetica,sans-serif;color:#000;margin-bottom:4px}
    @media(max-width:700px){.project-heading h1{font-size:22px;white-space:normal;text-wrap:balance}}
    div.stButton>button[kind="primary"]{background:#292c29;border-color:#292c29;border-radius:2px;min-height:40px;font-weight:500}
    [data-testid="stExpander"]{border-radius:2px;border-color:#d8d9d2}
    [data-baseweb="select"]>div{border-radius:2px}
    </style>""")

    with st.sidebar:
        st.header("Simulation")
        task = st.radio("Simulation", list(TASKS), label_visibility="collapsed")
        config = TASKS[task]
        checkpoints = {name: path for name, path in find_checkpoints(config["checkpoint_dir"]).items() if name in ("DeMa", "Decision Transformer")}
        if not checkpoints:
            st.error("Missing checkpoints.")
            st.stop()
        st.header("Models")
        comparisons = {
            "DeMa vs DT": ["DeMa", "Decision Transformer"],
            "DeMa vs Random Actions": ["DeMa", "Random actions"],
            "All Three Approaches": ["DeMa", "Decision Transformer", "Random actions"],
            "DeMa only": ["DeMa"],
        }
        comparison = st.selectbox("Comparison", list(comparisons), filter_mode=None, label_visibility="collapsed")
        selected = comparisons[comparison]
        intervention = False
        go = st.button("Run", type="primary", use_container_width=True)

    st.html('<header class="project-heading"><div>DeMa</div><h1>Linear-Time Sequence Modeling for Offline Reinforcement Learning</h1></header>')
    target = config["target"]
    max_steps = episode_steps(task)
    signature = (task, tuple(selected), intervention)
    setup_key = f"setup_{task}_trained_areas"
    if setup_key not in st.session_state:
        st.session_state[setup_key] = new_run_setup(task)
    completed = False
    if go:
        seed, reset_options = st.session_state[setup_key]
        episodes = []
        progress_bar = st.progress(0, text="Loading...")
        started = time.perf_counter()
        try:
            with simulation_lock():
                for index, agent in enumerate(selected):
                    model_label = "DeMa" if agent == "Random actions" else agent
                    def update_progress(step):
                        if step % 25 == 0 or step == max_steps:
                            fraction = min((index + step / max_steps) / len(selected), 1.0)
                            progress_bar.progress(fraction, text=f"Simulating {agent} - {fraction:.0%}")
                    episodes.append(run_episode(task, agent, checkpoints[model_label], seed, target, max_steps, reset_options, intervention, update_progress))
            st.session_state["experiment"] = {
                "signature": signature, "task": task, "target": target, "seed": seed,
                "budget": max_steps, "intervention": intervention, "reset_options": reset_options,
                "goal_mode": "Random positions in trained areas" if task == "AntMaze" else None,
                "runtime_seconds": time.perf_counter() - started, "episodes": episodes,
            }
            st.session_state[setup_key] = new_run_setup(task, st.session_state["experiment"])
            completed = True
        except Exception:
            logging.getLogger(__name__).exception("Simulation failed")
            st.error("Could not load this simulation. Please try again.")
        finally:
            progress_bar.empty()

    experiment = st.session_state.get("experiment")
    if experiment and any(episode["name"] not in ("DeMa", "Decision Transformer", "Random actions") for episode in experiment["episodes"]):
        st.session_state.pop("experiment")
        experiment = None
    if experiment and experiment["task"] == "AntMaze" and experiment.get("goal_mode") != "Random positions in trained areas":
        st.session_state.pop("experiment")
        experiment = None
    if experiment and experiment["task"] == task:
        if experiment["signature"] != signature:
            st.info("Selection changed. Run to update.")
        payload = {key: value for key, value in experiment.items() if key != "signature"}
        payload["preview"] = False
        payload["autoplay"] = completed
        show_player(payload)
    else:
        try:
            preview_seed, options = st.session_state[setup_key]
            preview = initial_episode(task, preview_seed, options)
            previews = [{**preview, "name": name} for name in selected]
            show_player({"task": task, "target": target, "seed": preview_seed, "budget": max_steps, "intervention": intervention, "preview": True, "episodes": previews})
        except Exception:
            logging.getLogger(__name__).exception("Simulation failed")
            st.error("Could not load this simulation. Please try again.")


if __name__ == "__main__":
    main()
