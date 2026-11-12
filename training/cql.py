import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import time
import numpy as np
import torch
import minari
from train_compare import dataset_name, measured_random_return, device, flatten_obs

num_episodes_used = 150
seed = int(os.environ.get("SEED", "42"))

def load_transitions():
    dataset = minari.load_dataset(dataset_name, download=True)
    episodes = list(dataset.iterate_episodes())[:num_episodes_used]

    states, next_states, actions, rewards, dones, timeouts = [], [], [], [], [], []
    episode_returns = []

    for episode in episodes:
        states.append(flatten_obs(episode.observations)[:-1])
        next_states.append(flatten_obs(episode.observations)[1:])
        actions.append(episode.actions)
        rewards.append(episode.rewards)
        dones.append(episode.terminations)


        episode_timeouts = np.zeros(len(episode.terminations), dtype=np.float32)
        episode_timeouts[-1] = 1.0
        timeouts.append(episode_timeouts)
        episode_returns.append(float(episode.rewards.sum()))

    states = np.concatenate(states).astype(np.float32)
    next_states = np.concatenate(next_states).astype(np.float32)
    actions = np.concatenate(actions).astype(np.float32)
    rewards = np.concatenate(rewards).astype(np.float32)
    dones = np.concatenate(dones).astype(np.float32)
    timeouts = np.concatenate(timeouts).astype(np.float32)

    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0) + 1e-6
    states = (states - state_mean) / state_std
    next_states = (next_states - state_mean) / state_std

    used_mean_return = float(np.mean(episode_returns))
    print(f"{len(episodes)} episodes, {len(states)} transitions")
    print(f"mean return of the episodes used {used_mean_return:.2f}")

    return {
        "states": torch.tensor(states),
        "next_states": torch.tensor(next_states),
        "actions": torch.tensor(actions),
        "rewards": torch.tensor(rewards),
        "dones": torch.tensor(dones),
        "timeouts": torch.tensor(timeouts),
        "state_mean": state_mean,
        "state_std": state_std,
        "dataset": dataset,
        "used_mean_return": used_mean_return,
    }


def evaluate(policy, data, rollouts):
    env = data["dataset"].recover_environment()
    state_mean = data["state_mean"]
    state_std = data["state_std"]

    policy.eval()
    scores = []
    lengths = []
    forward_velocities = []
    decision_times = []

    for index in range(rollouts):
        state, _ = env.reset(seed=seed * 1000 + index)
        total_reward = 0.0
        steps = 0
        velocities = []
        done = False

        while not done:
            state = flatten_obs(state)
            normalized = (state - state_mean) / state_std
            state_input = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(device)

            start_time = time.time()
            with torch.no_grad():
                action = policy.deterministic(state_input)[0].cpu().numpy()
            decision_times.append(time.time() - start_time)

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1
            velocities.append(float(info.get("x_velocity", float("nan"))))

        scores.append(total_reward)
        lengths.append(steps)
        forward_velocities.append(float(np.mean(velocities)))

    env.close()
    policy.train()

    used_mean_return = data["used_mean_return"]
    normalized_return = 100.0 * (np.mean(scores) - measured_random_return) / (used_mean_return - measured_random_return)

    return {
        "rollouts": rollouts,
        "return_mean": float(np.mean(scores)),
        "return_std": float(np.std(scores)),
        "return_min": float(np.min(scores)),
        "return_median": float(np.median(scores)),
        "return_max": float(np.max(scores)),
        "return_scores": [float(s) for s in scores],
        "normalized_return": float(normalized_return),
        "episode_length_mean": float(np.mean(lengths)),
        "forward_velocity_mean": float(np.mean(forward_velocities)),
        "parameters": sum(p.numel() for p in policy.parameters()),
        "decisions_per_second": float(1.0 / np.mean(decision_times)),
    }
