import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))












import json
import os

import d3rlpy
import numpy as np
import torch

from train_compare import dataset_name, measured_random_return, device, results_dir, checkpoint_dir, flatten_obs
from cql import load_transitions, evaluate

seed = int(os.environ.get("SEED", "42"))
total_steps = int(os.environ.get("CQL_STEPS", "100000"))
num_rollouts = int(os.environ.get("CQL_ROLLOUTS", "100"))
is_antmaze = "antmaze" in dataset_name.lower()


reward_shift = -1.0 if is_antmaze else 0.0


is_medium_v0 = "medium-v0" in dataset_name.lower()
mujoco_conservative_weight = 10.0 if is_medium_v0 else 5.0
mujoco_encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])

torch.manual_seed(seed)
np.random.seed(seed)


class D3rlpyPolicyAdapter:

    def __init__(self, algo):
        self.algo = algo

    def eval(self):
        pass

    def train(self):
        pass

    def deterministic(self, state_input):
        action = self.algo.predict(state_input.cpu().numpy())
        return torch.from_numpy(action).to(state_input.device)

    def parameters(self):
        try:
            for module in self.algo.impl.modules.get_torch_modules().values():
                yield from module.parameters()
        except Exception:
            return iter([])


def main():
    data = load_transitions()
    obs_dim = data["states"].shape[-1]
    act_dim = data["actions"].shape[-1]

    rewards = data["rewards"].numpy() + reward_shift
    terminals = data["dones"].numpy()
    timeouts = data["timeouts"].numpy()
    mdp_dataset = d3rlpy.dataset.MDPDataset(
        observations=data["states"].numpy(),
        actions=data["actions"].numpy(),
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )

    config = d3rlpy.algos.CQLConfig(
        batch_size=256,
        conservative_weight=mujoco_conservative_weight,
        alpha_learning_rate=0.0,
        actor_encoder_factory=mujoco_encoder,
        critic_encoder_factory=mujoco_encoder,
    )
    algo = config.create(device="cuda:0" if torch.cuda.is_available() else None)
    algo.build_with_dataset(mdp_dataset)


    fallback_name = f"cql_{dataset_name.split('/')[-2] if '/' in dataset_name else dataset_name}_seed{seed}"
    fit_kwargs = {"experiment_name": os.environ.get("RUN_TAG", fallback_name)}
    if os.environ.get("WANDB_API_KEY"):
        fit_kwargs["logger_adapter"] = d3rlpy.logging.WanDBAdapterFactory(project="dema-vs-dt")

    algo.fit(mdp_dataset, n_steps=total_steps, n_steps_per_epoch=max(total_steps // 20, 1000), **fit_kwargs)

    policy = D3rlpyPolicyAdapter(algo)
    scores = evaluate(policy, data, num_rollouts)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "cql"), exist_ok=True)
    record = {
        "dataset_name": dataset_name,
        "seed": seed,
        "total_steps": total_steps,
        "num_rollouts": num_rollouts,
        "reward_shift": reward_shift,
        "alpha_learning_rate": 0.0,
        "conservative_weight": mujoco_conservative_weight,
        "encoder": "VectorEncoderFactory([256, 256, 256])",
        "recipe_source": "d3rlpy official reproduction script (reproductions/offline/cql.py), dual-off fix",
        "implementation": "d3rlpy",
        "d3rlpy_version": d3rlpy.__version__,
        "measured_random_return": measured_random_return,
        "used_mean_return": data["used_mean_return"],
        "evaluation": scores,
    }
    with open(os.path.join(results_dir, "cql", f"run_seed{seed}_cql.json"), "w") as handle:
        json.dump(record, handle)
    print(
        f"CQL(d3rlpy) return {scores['return_mean']:.1f} +/- {scores['return_std']:.1f} "
        f"(n={scores['rollouts']}), normalized {scores['normalized_return']:.1f}, "
        f"velocity {scores['forward_velocity_mean']:.2f}, {scores['parameters']} parameters, "
        f"{scores['decisions_per_second']:.1f} decisions/s"
    )


if __name__ == "__main__":
    main()
