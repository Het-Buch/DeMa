import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset_name = os.environ.get("DATASET_NAME", "mujoco/halfcheetah/medium-v0")
measured_random_return = float(os.environ.get("MEASURED_RANDOM_RETURN", "-295.60"))
dataset_mean_return = float(os.environ.get("DATASET_MEAN_RETURN", "12089.21"))

context_len = 20
num_episodes_used = 150
d_model = 128
num_layers = 3
num_heads = 1
batch_size = 64
total_steps = int(os.environ.get("TOTAL_STEPS", "100000"))
warmup_steps = int(os.environ.get("WARMUP_STEPS", "10000"))
learning_rate = float(os.environ.get("LEARNING_RATE", "1e-4"))
weight_decay = 1e-4
return_scale = 1000.0
eval_targets = [float(x) for x in os.environ.get("EVAL_TARGETS", "12000.0,6000.0").split(",")]
num_rollouts = int(os.environ.get("NUM_ROLLOUTS", "100"))
checkpoint_every = 5000
seed = int(os.environ.get("SEED", "42"))
checkpoint_dir = "checkpoints"
results_dir = "results"
models_to_run = os.environ.get("MODELS", "Decision Transformer,DeMa").split(",")
dema_uses_timestep_embedding = os.environ.get("DEMA_TIMESTEP", "0") == "1"
dt_uses_timestep_embedding = os.environ.get("DT_TIMESTEP", "0") == "1"
timestep_embed_std = float(os.environ.get("TIMESTEP_EMBED_STD", "1.0"))
token_layout = os.environ.get("TOKEN_LAYOUT", "interleaved")
assert token_layout in ("interleaved", "grouped")
use_ser = os.environ.get("SER", "0") == "1"
ser_beta_min = float(os.environ.get("SER_BETA_MIN", "0.5"))
ser_beta_k = float(os.environ.get("SER_BETA_K", "0.95"))
ser_snapshot_every = int(os.environ.get("SER_SNAPSHOT_EVERY", "1000"))
dropout = float(os.environ.get("DROPOUT", "0.0"))
norm_first = os.environ.get("NORM_FIRST", "0") == "1"
dema_norm_first = os.environ.get("DEMA_NORM_FIRST", "0") == "1"
validation_episodes = 50
max_episode_len = 1000

torch.manual_seed(seed)
np.random.seed(seed)


def flatten_obs(obs):
    if isinstance(obs, dict):
        return np.concatenate([obs["observation"], obs["achieved_goal"], obs["desired_goal"]], axis=-1)
    return obs


def load_data():
    import minari

    dataset = minari.load_dataset(dataset_name, download=True)
    episodes = list(dataset.iterate_episodes())[:num_episodes_used]

    states = []
    actions = []
    returns_to_go = []
    timesteps = []
    valid_starts = []
    episode_returns = []
    offset = 0

    for episode in episodes:
        episode_actions = episode.actions
        rewards = episode.rewards
        length = len(episode_actions)

        states.append(flatten_obs(episode.observations)[:-1])
        actions.append(episode_actions)
        returns_to_go.append(np.cumsum(rewards[::-1])[::-1].copy() / return_scale)
        timesteps.append(np.minimum(np.arange(length), max_episode_len - 1))
        episode_returns.append(float(rewards.sum()))

        valid_starts.append(np.arange(offset, offset + length - context_len + 1))
        offset += length

    states = np.concatenate(states).astype(np.float32)
    actions = np.concatenate(actions).astype(np.float32)
    returns_to_go = np.concatenate(returns_to_go).astype(np.float32)[:, None]
    timesteps = np.concatenate(timesteps).astype(np.int64)
    valid_starts = np.concatenate(valid_starts)

    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0) + 1e-6
    states = (states - state_mean) / state_std

    validation = {}
    if len(episodes) > validation_episodes:
        held_out = list(dataset.iterate_episodes())[num_episodes_used:num_episodes_used + validation_episodes]
        if held_out:
            validation = build_validation(held_out, state_mean, state_std)

    used_mean_return = float(np.mean(episode_returns))
    print(f"{len(episodes)} episodes, {len(states)} transitions, {len(valid_starts)} windows")
    print(f"mean return of the episodes used {used_mean_return:.2f}")

    return {
        "states": torch.tensor(states),
        "actions": torch.tensor(actions),
        "returns_to_go": torch.tensor(returns_to_go),
        "timesteps": torch.tensor(timesteps),
        "valid_starts": torch.tensor(valid_starts),
        "state_mean": state_mean,
        "state_std": state_std,
        "dataset": dataset,
        "used_mean_return": used_mean_return,
        "validation": validation,
    }


def build_validation(episodes, state_mean, state_std):
    states = []
    actions = []
    returns_to_go = []
    timesteps = []
    valid_starts = []
    offset = 0

    for episode in episodes:
        episode_actions = episode.actions
        rewards = episode.rewards
        length = len(episode_actions)

        states.append(flatten_obs(episode.observations)[:-1])
        actions.append(episode_actions)
        returns_to_go.append(np.cumsum(rewards[::-1])[::-1].copy() / return_scale)
        timesteps.append(np.minimum(np.arange(length), max_episode_len - 1))

        valid_starts.append(np.arange(offset, offset + length - context_len + 1, context_len))
        offset += length

    states = (np.concatenate(states).astype(np.float32) - state_mean) / state_std

    return {
        "states": torch.tensor(states),
        "actions": torch.tensor(np.concatenate(actions).astype(np.float32)),
        "returns_to_go": torch.tensor(np.concatenate(returns_to_go).astype(np.float32)[:, None]),
        "timesteps": torch.tensor(np.concatenate(timesteps).astype(np.int64)),
        "valid_starts": torch.tensor(np.concatenate(valid_starts)),
    }


def validation_loss(model, validation):
    if not validation:
        return None

    model.eval()
    starts = validation["valid_starts"]
    rows = starts[:, None] + torch.arange(context_len)[None, :]
    total = 0.0
    batches = 0

    with torch.no_grad():
        for begin in range(0, len(rows), 256):
            chunk = rows[begin:begin + 256]
            predicted = model(
                validation["states"][chunk].to(device),
                validation["actions"][chunk].to(device),
                validation["returns_to_go"][chunk].to(device),
                validation["timesteps"][chunk].to(device),
            )
            total += float(((predicted - validation["actions"][chunk].to(device)) ** 2).mean())
            batches += 1

    model.train()
    return total / max(batches, 1)


def sample_batch(data, generator):
    choice = torch.randint(len(data["valid_starts"]), (batch_size,), generator=generator)
    starts = data["valid_starts"][choice]
    rows = starts[:, None] + torch.arange(context_len)[None, :]

    return (
        data["states"][rows].to(device),
        data["actions"][rows].to(device),
        data["returns_to_go"][rows].to(device),
        data["timesteps"][rows].to(device),
    )


class TokenEmbed(nn.Module):
    def __init__(self, obs_dim, act_dim, use_timestep_embedding):
        super().__init__()
        self.state_embed = nn.Linear(obs_dim, d_model)
        self.action_embed = nn.Linear(act_dim, d_model)
        self.return_embed = nn.Linear(1, d_model)
        self.use_timestep_embedding = use_timestep_embedding
        if use_timestep_embedding:
            self.timestep_embed = nn.Embedding(max_episode_len, d_model)
            # Default std=1.0 reproduces nn.Embedding's own default init exactly (same
            # as calling nothing) - existing checkpoints and the un-rescaled PE ablation
            # are unaffected. timestep_embed_std=0.02 is the rescaled-init PE variant:
            # 0.02 is standard Transformer/GPT-2 embedding init, not tuned for this
            # project's result (see findings.md, PE rescaled-init caveat).
            nn.init.normal_(self.timestep_embed.weight, std=timestep_embed_std)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, states, actions, returns, timesteps):
        batch, length = states.shape[0], states.shape[1]

        if token_layout == "grouped":
            # BL3D: one token per timestep, built from (R_t, s_t, a_{t-1}) - the
            # *previous* action, not a_t, or the current action being predicted
            # would leak into its own prediction (see findings.md, axis 2).
            prev_actions = torch.cat(
                [torch.zeros_like(actions[:, :1]), actions[:, :-1]], dim=1
            )
            state_tokens = self.state_embed(states)
            action_tokens = self.action_embed(prev_actions)
            return_tokens = self.return_embed(returns)

            if self.use_timestep_embedding:
                time_tokens = self.timestep_embed(timesteps)
                state_tokens = state_tokens + time_tokens
                action_tokens = action_tokens + time_tokens
                return_tokens = return_tokens + time_tokens

            return self.norm(return_tokens + state_tokens + action_tokens)

        state_tokens = self.state_embed(states)
        action_tokens = self.action_embed(actions)
        return_tokens = self.return_embed(returns)

        if self.use_timestep_embedding:
            time_tokens = self.timestep_embed(timesteps)
            state_tokens = state_tokens + time_tokens
            action_tokens = action_tokens + time_tokens
            return_tokens = return_tokens + time_tokens

        stacked = torch.stack((return_tokens, state_tokens, action_tokens), dim=2)
        return self.norm(stacked.reshape(batch, 3 * length, d_model))

    def single_tokens(self, state, target_return, timestep):
        return_token = self.return_embed(target_return)
        state_token = self.state_embed(state)

        if self.use_timestep_embedding:
            time_token = self.timestep_embed(timestep)
            return_token = return_token + time_token
            state_token = state_token + time_token

        return self.norm(torch.stack((return_token, state_token), dim=1))

    def single_action_token(self, action, timestep):
        action_token = self.action_embed(action)
        if self.use_timestep_embedding:
            action_token = action_token + self.timestep_embed(timestep)
        return self.norm(action_token.unsqueeze(1))


class DecisionTransformer(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.embed = TokenEmbed(obs_dim, act_dim, use_timestep_embedding=dt_uses_timestep_embedding)
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.action_head = nn.Linear(d_model, act_dim)
        mask_len = context_len if token_layout == "grouped" else 3 * context_len
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(mask_len))

    def forward(self, states, actions, returns, timesteps):
        batch, length = states.shape[0], states.shape[1]
        tokens = self.embed(states, actions, returns, timesteps)

        if token_layout == "grouped":
            mask = self.mask[:length, :length]
            hidden = self.backbone(tokens, mask=mask, is_causal=True)
            return torch.tanh(self.action_head(hidden))

        mask = self.mask[:3 * length, :3 * length]
        hidden = self.backbone(tokens, mask=mask, is_causal=True)
        state_positions = hidden.reshape(batch, length, 3, d_model)[:, :, 1]
        return torch.tanh(self.action_head(state_positions))


class S6Block(nn.Module):
    def __init__(self, d_state=16, d_conv=4, expand=2, norm_first=False, use_conv=True):
        super().__init__()
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)
        self.norm_first = norm_first
        # use_conv=False: skips the Conv1D token-mixing step (used by
        # MultiGrainedBlock's coarse-grained branch, lv2024multigrained Eq. 10, which
        # has no conv - only the fine-grained branch, Eq. 11, does). Default True
        # preserves this class's normal standalone (DT/DeMa) behavior unchanged.
        self.use_conv = use_conv

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        if use_conv:
            self.conv1d = nn.Conv1d(
                self.d_inner, self.d_inner, kernel_size=d_conv,
                groups=self.d_inner, padding=d_conv - 1,
            )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.a_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).unsqueeze(0).repeat(self.d_inner, 1)
        )
        self.d_param = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model)
        )
        self.feedforward_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def delta_b_c(self, inner):
        projected = self.x_proj(inner)
        delta_raw, b, c = projected.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta_raw))
        return delta, b, c

    def scan_sequential(self, x):
        batch, length, _ = x.shape
        projected = self.in_proj(x)
        inner_raw, gate = projected.chunk(2, dim=-1)

        padded = F.pad(inner_raw.transpose(1, 2), (self.d_conv - 1, 0))
        conv_weight = self.conv1d.weight.squeeze(1)
        inner = torch.zeros_like(inner_raw)
        for step in range(length):
            window = padded[:, :, step:step + self.d_conv]
            inner[:, step] = (window * conv_weight).sum(-1) + self.conv1d.bias
        inner = F.silu(inner)

        delta, b, c = self.delta_b_c(inner)
        a = -torch.exp(self.a_log)

        state = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for step in range(length):
            decay = torch.exp(delta[:, step].unsqueeze(-1) * a)
            update = delta[:, step].unsqueeze(-1) * b[:, step].unsqueeze(1) * inner[:, step].unsqueeze(-1)
            state = state * decay + update
            y = (state * c[:, step].unsqueeze(1)).sum(-1) + inner[:, step] * self.d_param
            outputs.append(y)

        y = torch.stack(outputs, dim=1) * F.silu(gate)
        return self.out_proj(y)

    def scan_parallel(self, x):
        length = x.shape[1]
        projected = self.in_proj(x)
        inner_raw, gate = projected.chunk(2, dim=-1)

        if self.use_conv:
            convolved = self.conv1d(inner_raw.transpose(1, 2))[:, :, :length]
            inner = F.silu(convolved.transpose(1, 2))
        else:
            inner = F.silu(inner_raw)

        delta, b, c = self.delta_b_c(inner)
        a = -torch.exp(self.a_log)

        decay = torch.exp(delta.unsqueeze(-1) * a)
        update = delta.unsqueeze(-1) * b.unsqueeze(2) * inner.unsqueeze(-1)

        offset = 1
        while offset < length:
            shifted_decay = torch.zeros_like(decay)
            shifted_update = torch.zeros_like(update)
            shifted_decay[:, offset:] = decay[:, :length - offset]
            shifted_update[:, offset:] = update[:, :length - offset]

            update = update + decay * shifted_update
            decay = decay * shifted_decay
            offset *= 2

        y = (update * c.unsqueeze(2)).sum(-1) + inner * self.d_param
        y = y * F.silu(gate)
        return self.out_proj(y)

    def step(self, token, carried):
        ssm_state, conv_state = carried
        source = self.norm(token) if self.norm_first else token

        projected = self.in_proj(source)
        inner_raw, gate = projected.chunk(2, dim=-1)

        history = torch.cat([conv_state, inner_raw.unsqueeze(-1)], dim=-1)
        conv_weight = self.conv1d.weight.squeeze(1)
        conv_out = (history * conv_weight).sum(-1) + self.conv1d.bias
        inner = F.silu(conv_out)
        conv_state = history[:, :, 1:]

        delta, b, c = self.delta_b_c(inner)
        a = -torch.exp(self.a_log)
        decay = torch.exp(delta.unsqueeze(-1) * a)
        update = delta.unsqueeze(-1) * b.unsqueeze(1) * inner.unsqueeze(-1)
        ssm_state = ssm_state * decay + update

        y = (ssm_state * c.unsqueeze(1)).sum(-1) + inner * self.d_param
        y = y * F.silu(gate)
        scanned = self.out_proj(y)
        carried = (ssm_state, conv_state)

        if self.norm_first:
            mixed = token + scanned
            return mixed + self.feedforward_dropout(self.feedforward(self.norm2(mixed))), carried

        mixed = self.norm(token + scanned)
        return self.norm2(mixed + self.feedforward_dropout(self.feedforward(mixed))), carried

    def initial_state(self, batch, dtype, target_device):
        ssm_state = torch.zeros(batch, self.d_inner, self.d_state, device=target_device, dtype=dtype)
        conv_state = torch.zeros(batch, self.d_inner, self.d_conv - 1, device=target_device, dtype=dtype) if self.use_conv else None
        return (ssm_state, conv_state)

    def raw_step(self, source, carried):
        """Same recurrence as step(), without the norm/feedforward wrapper - lets a
        caller (MultiGrainedBlock) combine two independent raw SSM branches before
        applying its own single wrapper, instead of each branch wrapping itself."""
        ssm_state, conv_state = carried
        projected = self.in_proj(source)
        inner_raw, gate = projected.chunk(2, dim=-1)

        if self.use_conv:
            history = torch.cat([conv_state, inner_raw.unsqueeze(-1)], dim=-1)
            conv_weight = self.conv1d.weight.squeeze(1)
            conv_out = (history * conv_weight).sum(-1) + self.conv1d.bias
            inner = F.silu(conv_out)
            conv_state = history[:, :, 1:]
        else:
            inner = F.silu(inner_raw)

        delta, b, c = self.delta_b_c(inner)
        a = -torch.exp(self.a_log)
        decay = torch.exp(delta.unsqueeze(-1) * a)
        update = delta.unsqueeze(-1) * b.unsqueeze(1) * inner.unsqueeze(-1)
        ssm_state = ssm_state * decay + update

        y = (ssm_state * c.unsqueeze(1)).sum(-1) + inner * self.d_param
        y = y * F.silu(gate)
        return self.out_proj(y), (ssm_state, conv_state)

    def forward(self, x):
        if self.norm_first:
            scanned = self.scan_parallel(self.norm(x))
            x = x + scanned
            return x + self.feedforward_dropout(self.feedforward(self.norm2(x)))

        scanned = self.scan_parallel(x)
        x = self.norm(x + scanned)
        return self.norm2(x + self.feedforward_dropout(self.feedforward(x)))


class DeMa(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.embed = TokenEmbed(obs_dim, act_dim, use_timestep_embedding=dema_uses_timestep_embedding)
        self.blocks = nn.ModuleList([S6Block(norm_first=dema_norm_first) for _ in range(num_layers)])
        self.action_head = nn.Linear(d_model, act_dim)

    def forward(self, states, actions, returns, timesteps):
        batch, length = states.shape[0], states.shape[1]
        x = self.embed(states, actions, returns, timesteps)
        for block in self.blocks:
            x = block(x)

        if token_layout == "grouped":
            return torch.tanh(self.action_head(x))

        state_positions = x.reshape(batch, length, 3, d_model)[:, :, 1]
        return torch.tanh(self.action_head(state_positions))

    def initial_states(self, batch, dtype, target_device):
        return [block.initial_state(batch, dtype, target_device) for block in self.blocks]

    def step_tokens(self, tokens, carried):
        for index in range(tokens.shape[1]):
            token = tokens[:, index]
            for depth, block in enumerate(self.blocks):
                token, carried[depth] = block.step(token, carried[depth])
        return token, carried

    def step_decision(self, state, target_return, timestep, carried):
        tokens = self.embed.single_tokens(state, target_return, timestep)
        _, carried = self.step_tokens(tokens[:, :1], carried)
        hidden, carried = self.step_tokens(tokens[:, 1:2], carried)
        return torch.tanh(self.action_head(hidden)), carried

    def absorb_action(self, action, timestep, carried):
        token = self.embed.single_action_token(action, timestep)
        _, carried = self.step_tokens(token, carried)
        return carried


class MultiGrainedBlock(nn.Module):
    """lv2024multigrained (arXiv:2406.05427 v3, "Decision Mamba: A Multi-Grained SSM
    with Self-Evolution Regularization"), Eq. 10-15: a coarse-grained S6 scan (Eq. 10,
    NO conv) and a fine-grained S6 scan (Eq. 11-13, WITH a Conv1D the coarse branch
    doesn't have) run in parallel from the same layer input - the only architectural
    difference the paper's equations give between the two branches (confirmed via the
    paper's HTML full text; Algorithm 1 is internally self-contradictory about where the
    conv goes - line 7 reuses the coarse branch's own conv module to build the fine
    branch's input, then line 12 applies a conv to both branches again - so the cleaner,
    internally-consistent Eq. 10-13 is used instead of the inconsistent algorithm here,
    the opposite tie-break from the scan-axis ambiguity elsewhere in this file, which
    favored the algorithm over the prose; see findings.md for both). Combined via
    LayerNorm(coarse_out + fine_out) (Eq. 14, verbatim: h_i^MG = LN(h_i^CG + h_i^FG)),
    then h_i = Proj(h_i^MG + h_{i-1}) (Eq. 15, verbatim). Eq. 15's h_i IS the paper's
    complete per-layer output (no further FFN/norm sublayer exists in the paper's own
    Eqs. 9-15) - so its "+ h_{i-1}" residual is NOT duplicated by an extra residual
    here; the surrounding feedforward/norm wrapper this project's own S6Block already
    uses for DT/DeMa (reused here for a fair, same-FFN-capacity comparison, a declared
    choice since the paper has no equivalent sublayer) supplies its OWN single residual
    around that already-complete output, exactly as it does for DeMa's plain S6Block."""
    def __init__(self, norm_first=False):
        super().__init__()
        self.norm_first = norm_first
        self.coarse = S6Block(norm_first=norm_first, use_conv=False)
        self.fine = S6Block(norm_first=norm_first, use_conv=True)
        # coarse/fine are only ever driven via scan_parallel()/raw_step() (the raw SSM
        # mixing), never forward()/step() - their own norm/feedforward/norm2 sublayers
        # (inherited from S6Block's standalone-use design) are dead weight here, counted
        # in parameter totals but never exercised. Deleted, not just unused, so they
        # don't inflate this model's reported parameter count.
        for branch in (self.coarse, self.fine):
            del branch.norm, branch.feedforward, branch.feedforward_dropout, branch.norm2
        self.combine_norm = nn.LayerNorm(d_model)
        self.combine_proj = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model)
        )
        self.feedforward_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def _combine(self, source):
        coarse_out = self.coarse.scan_parallel(source)
        fine_out = self.fine.scan_parallel(source)
        combined = self.combine_norm(coarse_out + fine_out)
        return self.combine_proj(combined + source)

    def forward(self, x):
        # Eq. 15's h_i is already a complete residual-connected output (w.r.t. its own
        # input to _combine), unlike plain S6Block's scan_parallel() output (which has
        # no residual baked in) - so the wrapper below applies only ONE further norm/
        # FFN sublayer on top of it, never re-adding the same residual a second time.
        if self.norm_first:
            x = self._combine(self.norm(x))
            return x + self.feedforward_dropout(self.feedforward(self.norm2(x)))

        x = self.norm(self._combine(x))
        return self.norm2(x + self.feedforward_dropout(self.feedforward(x)))

    def step(self, token, carried):
        coarse_carried, fine_carried = carried
        source = self.norm(token) if self.norm_first else token

        coarse_out, coarse_carried = self.coarse.raw_step(source, coarse_carried)
        fine_out, fine_carried = self.fine.raw_step(source, fine_carried)
        combined = self.combine_norm(coarse_out + fine_out)
        scanned = self.combine_proj(combined + source)
        carried = (coarse_carried, fine_carried)

        if self.norm_first:
            return scanned + self.feedforward_dropout(self.feedforward(self.norm2(scanned))), carried

        mixed = self.norm(scanned)
        return self.norm2(mixed + self.feedforward_dropout(self.feedforward(mixed))), carried

    def initial_state(self, batch, dtype, target_device):
        return (
            self.coarse.initial_state(batch, dtype, target_device),
            self.fine.initial_state(batch, dtype, target_device),
        )


class MultiGrainedSSM(nn.Module):
    """DeMa's architecture (TokenEmbed + stacked blocks + linear action head) with
    S6Block replaced by MultiGrainedBlock - isolates the fine-grained-SSM addition
    from DeMa's own baseline, matching the project's fairness rule: same embed, same
    token layout, same hyperparameters as DeMa gets (see findings.md)."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.embed = TokenEmbed(obs_dim, act_dim, use_timestep_embedding=dema_uses_timestep_embedding)
        self.blocks = nn.ModuleList([MultiGrainedBlock(norm_first=dema_norm_first) for _ in range(num_layers)])
        self.action_head = nn.Linear(d_model, act_dim)

    def forward(self, states, actions, returns, timesteps):
        batch, length = states.shape[0], states.shape[1]
        x = self.embed(states, actions, returns, timesteps)
        for block in self.blocks:
            x = block(x)

        if token_layout == "grouped":
            return torch.tanh(self.action_head(x))

        state_positions = x.reshape(batch, length, 3, d_model)[:, :, 1]
        return torch.tanh(self.action_head(state_positions))

    def initial_states(self, batch, dtype, target_device):
        return [block.initial_state(batch, dtype, target_device) for block in self.blocks]

    def step_tokens(self, tokens, carried):
        for index in range(tokens.shape[1]):
            token = tokens[:, index]
            for depth, block in enumerate(self.blocks):
                token, carried[depth] = block.step(token, carried[depth])
        return token, carried

    def step_decision(self, state, target_return, timestep, carried):
        tokens = self.embed.single_tokens(state, target_return, timestep)
        _, carried = self.step_tokens(tokens[:, :1], carried)
        hidden, carried = self.step_tokens(tokens[:, 1:2], carried)
        return torch.tanh(self.action_head(hidden)), carried

    def absorb_action(self, action, timestep, carried):
        token = self.embed.single_action_token(action, timestep)
        _, carried = self.step_tokens(token, carried)
        return carried


def train(model, name, data):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, name + ".pth")

    # Direct W&B logging during training, not a separate post-hoc upload step (that
    # older pattern - saving results locally, then running wandb_upload.py afterward -
    # is superseded by this; see findings.md). Gated on WANDB_API_KEY being present,
    # imported lazily so importing this module (e.g. from app.py or a notebook) never
    # triggers a wandb session.
    wandb_enabled = bool(os.environ.get("WANDB_API_KEY"))
    if wandb_enabled:
        import wandb
        wandb.init(project="dema-vs-dt", name=os.environ.get("RUN_TAG", f"{name}_seed{seed}"), config={
            "dataset_name": dataset_name, "seed": seed, "model": name,
            "d_model": d_model, "num_layers": num_layers, "num_heads": num_heads,
            "batch_size": batch_size, "total_steps": total_steps,
            "warmup_steps": warmup_steps, "learning_rate": learning_rate,
            "weight_decay": weight_decay, "dropout": dropout,
            "norm_first": norm_first, "dema_norm_first": dema_norm_first,
            "token_layout": token_layout, "use_ser": use_ser,
            "parameters": sum(p.numel() for p in model.parameters()),
        })

    model.to(device)
    no_decay_names = {"a_log", "d_param"}
    decay_params = [p for n, p in model.named_parameters() if n.rsplit(".", 1)[-1] not in no_decay_names]
    no_decay_params = [p for n, p in model.named_parameters() if n.rsplit(".", 1)[-1] in no_decay_names]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / warmup_steps, 1.0))
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)

    start_step = 0
    loss_history = []

    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = checkpoint["step"]
        loss_history = checkpoint["loss_history"]
        print(f"resumed {name} from step {start_step}")

    model.train()
    running_loss = 0.0
    started = time.time()

    # Progressive Self-Evolution Regularization (Lv et al. 2024, arXiv:2406.05427,
    # eq. 16/20/22): the regression target is blended with the model's OWN prediction
    # from a periodically-refreshed earlier snapshot, so training partially trusts its
    # own past knowledge over the (possibly noisy/suboptimal) dataset label. beta_k
    # ramps linearly from 0 (pure supervised) up to ser_beta_k, floored at ser_beta_min
    # once the first snapshot exists.
    ser_prev_model = None

    for step in range(start_step, total_steps):
        batch_states, batch_actions, batch_returns, batch_timesteps = sample_batch(data, generator)

        predicted = model(batch_states, batch_actions, batch_returns, batch_timesteps)

        if use_ser and ser_prev_model is not None:
            with torch.no_grad():
                prev_prediction = ser_prev_model(batch_states, batch_actions, batch_returns, batch_timesteps)
            beta = max(ser_beta_min, ser_beta_k * (step + 1) / total_steps)
            target = (1 - beta) * batch_actions + beta * prev_prediction
        else:
            target = batch_actions

        loss = loss_fn(predicted, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        scheduler.step()

        if use_ser and (step + 1) % ser_snapshot_every == 0:
            ser_prev_model = copy.deepcopy(model).eval()
            for p in ser_prev_model.parameters():
                p.requires_grad_(False)

        running_loss += loss.item()

        if (step + 1) % 1000 == 0:
            average = running_loss / 1000
            entry = {"step": step + 1, "loss": average}
            elapsed = time.time() - started

            if (step + 1) % 5000 == 0:
                held_out = validation_loss(model, data["validation"])
                if held_out is not None:
                    entry["validation_loss"] = held_out
                    print(f"{name} step {step + 1}/{total_steps} loss {average:.5f} validation {held_out:.5f} ({elapsed:.0f}s)")
                else:
                    print(f"{name} step {step + 1}/{total_steps} loss {average:.5f} ({elapsed:.0f}s)")
            else:
                print(f"{name} step {step + 1}/{total_steps} loss {average:.5f} ({elapsed:.0f}s)")

            loss_history.append(entry)
            if wandb_enabled:
                wandb.log(entry, step=step + 1)
            running_loss = 0.0

        if (step + 1) % checkpoint_every == 0 or step + 1 == total_steps:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step + 1,
                "loss_history": loss_history,
                "state_mean": data["state_mean"],
                "state_std": data["state_std"],
                "obs_dim": data["states"].shape[-1],
                "act_dim": data["actions"].shape[-1],
                "d_model": d_model,
                "num_layers": num_layers,
                "dropout": dropout,
                "norm_first": norm_first,
                "dema_norm_first": dema_norm_first,
                "token_layout": token_layout,
                "uses_timestep_embedding": model.embed.use_timestep_embedding,
            }, path)

    return model, loss_history


def rollout_snapshot(env, observation, action, proposed_action, info, step, reward, total_reward, target_return, cell_distances, intervened):
    maze_task = isinstance(observation, dict)
    simulator = getattr(env.unwrapped, "ant_env", env.unwrapped)
    position = env.unwrapped.data.qpos
    goal = observation["desired_goal"] if maze_task else None
    xy = observation["achieved_goal"] if maze_task else np.array([position[0], 0.0])
    cell = tuple(int(c) for c in env.unwrapped.maze.cell_xy_to_rowcol(xy)) if maze_task else None
    return {
        "step": step,
        "time": float(step * simulator.dt),
        "state": flatten_obs(observation).tolist(),
        "action": np.asarray(action).tolist(),
        "proposed_action": np.asarray(proposed_action).tolist(),
        "reward": float(reward),
        "total_reward": float(total_reward),
        "remaining_return": float(target_return - total_reward),
        "forward_reward": float(info.get("reward_forward", 0.0)) if not maze_task else 0.0,
        "control_penalty": float(info.get("reward_ctrl", 0.0)) if not maze_task else 0.0,
        "velocity": float(info.get("x_velocity", 0.0)),
        "xy": np.asarray(xy).tolist(),
        "height": float(simulator.data.body("torso").xpos[2]),
        "body_velocity": float(simulator.data.qvel[0]),
        "pitch": float(position[2]) if not maze_task else None,
        "goal": np.asarray(goal).tolist() if maze_task else None,
        "goal_distance": float(np.linalg.norm(xy - goal)) if maze_task else None,
        "route_cells": cell_distances.get(cell) if cell_distances is not None else None,
        "success": bool(info.get("success", False)),
        "intervened": bool(intervened),
    }


def evaluate(model, data, target_return, rollouts, max_steps=None, capture_frames=False, on_frame=None, reset_options=None, cell_distances=None, brake_window=None, capture_details=False, capture_stride=1, on_step=None, frame_encoder=None, fast_inference=False):
    if capture_stride < 1:
        raise ValueError("capture_stride must be at least 1")
    env = data["dataset"].recover_environment(render_mode="rgb_array") if capture_frames else data["dataset"].recover_environment()
    frames = []
    frame_rewards = []
    frame_positions = []
    frame_forward_rewards = []
    frame_penalties = []
    frame_step_rewards = []
    frame_velocities = []
    frame_goal_distances = []
    frame_steps = []
    frame_details = []
    state_mean = data["state_mean"]
    state_std = data["state_std"]
    act_dim = data["actions"].shape[-1]

    model.eval()
    inference_context = torch.inference_mode if fast_inference else torch.no_grad
    scores = []
    lengths = []
    forward_velocities = []
    saturation = []
    decision_times = []

    for index in range(rollouts):
        observation, reset_info = env.reset(seed=seed * 1000 + index, options=reset_options)
        state = observation
        state = flatten_obs(state)
        states = [state]
        actions = []
        returns = [target_return / return_scale]
        total_reward = 0.0
        steps = 0
        velocities = []
        saturated = []
        done = False

        if capture_details and index == 0:
            initial_action = np.zeros(act_dim, dtype=np.float32)
            frame_details.append(rollout_snapshot(env, observation, initial_action, initial_action, reset_info, 0, 0.0, 0.0, target_return, cell_distances, False))

        if capture_frames and index == 0:
            frame = env.render()
            frames.append(frame_encoder(frame) if frame_encoder else frame)
            frame_steps.append(0)
            frame_rewards.append(0.0)
            frame_positions.append(float(env.unwrapped.data.qpos[0]))
            frame_forward_rewards.append(0.0)
            frame_penalties.append(0.0)
            frame_step_rewards.append(0.0)
            frame_velocities.append(0.0)
            if cell_distances is not None:
                cell = tuple(int(c) for c in env.unwrapped.maze.cell_xy_to_rowcol(env.unwrapped.data.qpos[:2]))
                frame_goal_distances.append(cell_distances.get(cell))
            else:
                frame_goal_distances.append(None)
            if on_frame:
                on_frame(frame, frame_rewards[-1], frame_positions[-1], 0.0, 0.0, 0.0, 0.0, frame_goal_distances[-1])

        while not done and (max_steps is None or steps < max_steps):
            window = min(len(states), context_len)
            recent_states = (np.array(states[-window:], dtype=np.float32) - state_mean) / state_std
            recent_actions = np.zeros((window, act_dim), dtype=np.float32)
            if actions:
                history = np.array(actions[-(window - 1):], dtype=np.float32) if window > 1 else np.zeros((0, act_dim), dtype=np.float32)
                recent_actions[:len(history)] = history
            recent_returns = np.array(returns[-window:], dtype=np.float32)
            recent_timesteps = np.arange(steps - window + 1, steps + 1).clip(0, max_episode_len - 1)

            state_input = torch.tensor(recent_states).unsqueeze(0).to(device)
            action_input = torch.tensor(recent_actions).unsqueeze(0).to(device)
            return_input = torch.tensor(recent_returns).unsqueeze(0).unsqueeze(-1).to(device)
            timestep_input = torch.tensor(recent_timesteps).unsqueeze(0).to(device)

            start_time = time.time()
            with inference_context():
                action = model(state_input, action_input, return_input, timestep_input)[0, -1].cpu().numpy()
            decision_times.append(time.time() - start_time)

            proposed_action = action.copy() if capture_details else action
            intervened = bool(brake_window and brake_window[0] <= steps < brake_window[1])
            if intervened:
                action = np.zeros_like(action)

            observation, reward, terminated, truncated, info = env.step(action)
            state = observation
            state = flatten_obs(state)
            done = terminated or truncated

            total_reward += reward
            steps += 1
            velocities.append(float(info.get("x_velocity", float("nan"))))
            saturated.append(float(np.mean(np.abs(action) > 0.99)))

            states.append(state)
            actions.append(action)
            returns.append(returns[-1] - reward / return_scale)

            if capture_details and index == 0:
                frame_details.append(rollout_snapshot(env, observation, action, proposed_action, info, steps, reward, total_reward, target_return, cell_distances, intervened))
            if on_step:
                on_step(steps)

            capture_step = steps % capture_stride == 0 or done or (max_steps is not None and steps >= max_steps)
            if capture_frames and index == 0 and capture_step:
                frame = env.render()
                frames.append(frame_encoder(frame) if frame_encoder else frame)
                frame_steps.append(steps)
                frame_rewards.append(total_reward)
                frame_positions.append(float(info.get("x_position", env.unwrapped.data.qpos[0])))
                frame_forward_rewards.append(float(info.get("reward_forward", 0.0)))
                frame_penalties.append(float(info.get("reward_ctrl", 0.0)))
                frame_step_rewards.append(float(reward))
                frame_velocities.append(float(info.get("x_velocity", 0.0)))
                if cell_distances is not None:
                    cell = tuple(int(c) for c in env.unwrapped.maze.cell_xy_to_rowcol(env.unwrapped.data.qpos[:2]))
                    frame_goal_distances.append(cell_distances.get(cell))
                else:
                    frame_goal_distances.append(None)
                if on_frame:
                    on_frame(frame, frame_rewards[-1], frame_positions[-1], frame_forward_rewards[-1], frame_penalties[-1], frame_step_rewards[-1], frame_velocities[-1], frame_goal_distances[-1])

        scores.append(total_reward)
        lengths.append(steps)
        forward_velocities.append(float(np.mean(velocities)))
        saturation.append(float(np.mean(saturated)))

    env.close()
    model.train()

    used_mean_return = data["used_mean_return"]
    normalized = 100.0 * (np.mean(scores) - measured_random_return) / (used_mean_return - measured_random_return)

    return {
        "target_return": target_return,
        "rollouts": rollouts,
        "return_mean": float(np.mean(scores)),
        "return_std": float(np.std(scores)),
        "return_min": float(np.min(scores)),
        "return_median": float(np.median(scores)),
        "return_max": float(np.max(scores)),
        "return_scores": [float(s) for s in scores],
        "normalized_return": float(normalized),
        "success_rate": float(np.mean([s > 0 for s in scores])),
        "episode_length_mean": float(np.mean(lengths)),
        "forward_velocity_mean": float(np.mean(forward_velocities)),
        "saturated_fraction": float(np.mean(saturation)),
        "parameters": sum(p.numel() for p in model.parameters()),
        "decisions_per_second": float(1.0 / np.mean(decision_times)),
        **({
            "frames": frames, "frame_rewards": frame_rewards, "frame_positions": frame_positions,
            "frame_forward_rewards": frame_forward_rewards, "frame_penalties": frame_penalties,
            "frame_step_rewards": frame_step_rewards, "frame_velocities": frame_velocities,
            "frame_goal_distances": frame_goal_distances,
            "frame_steps": frame_steps,
        } if capture_frames else {}),
        **({"frame_details": frame_details} if capture_details else {}),
    }


def evaluate_recurrent(model, data, target_return, rollouts):
    env = data["dataset"].recover_environment()
    state_mean = data["state_mean"]
    state_std = data["state_std"]

    model.eval()
    scores = []
    lengths = []
    forward_velocities = []
    saturation = []
    decision_times = []

    for index in range(rollouts):
        state, _ = env.reset(seed=seed * 1000 + index)
        state = flatten_obs(state)
        carried = model.initial_states(1, torch.float32, device)
        remaining = target_return / return_scale
        total_reward = 0.0
        steps = 0
        velocities = []
        saturated = []
        done = False

        while not done:
            normalized = (state - state_mean) / state_std
            state_input = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(device)
            return_input = torch.tensor([[remaining]], dtype=torch.float32).to(device)
            timestep_input = torch.tensor([min(steps, max_episode_len - 1)]).to(device)

            start_time = time.time()
            with torch.no_grad():
                predicted, carried = model.step_decision(state_input, return_input, timestep_input, carried)
            decision_times.append(time.time() - start_time)

            action = predicted[0].cpu().numpy()
            state, reward, terminated, truncated, info = env.step(action)
            state = flatten_obs(state)
            done = terminated or truncated

            with torch.no_grad():
                carried = model.absorb_action(
                    torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device),
                    timestep_input,
                    carried,
                )

            total_reward += reward
            steps += 1
            remaining -= reward / return_scale
            velocities.append(float(info.get("x_velocity", float("nan"))))
            saturated.append(float(np.mean(np.abs(action) > 0.99)))

        scores.append(total_reward)
        lengths.append(steps)
        forward_velocities.append(float(np.mean(velocities)))
        saturation.append(float(np.mean(saturated)))

    env.close()
    model.train()

    used_mean_return = data["used_mean_return"]
    normalized_return = 100.0 * (np.mean(scores) - measured_random_return) / (used_mean_return - measured_random_return)

    return {
        "mode": "recurrent_unlimited_history",
        "target_return": target_return,
        "rollouts": rollouts,
        "return_mean": float(np.mean(scores)),
        "return_std": float(np.std(scores)),
        "return_min": float(np.min(scores)),
        "return_median": float(np.median(scores)),
        "return_max": float(np.max(scores)),
        "return_scores": [float(s) for s in scores],
        "normalized_return": float(normalized_return),
        "success_rate": float(np.mean([s > 0 for s in scores])),
        "episode_length_mean": float(np.mean(lengths)),
        "forward_velocity_mean": float(np.mean(forward_velocities)),
        "saturated_fraction": float(np.mean(saturation)),
        "parameters": sum(p.numel() for p in model.parameters()),
        "decisions_per_second": float(1.0 / np.mean(decision_times)),
    }


def plot(results, used_mean_return):
    import matplotlib.pyplot as plt

    names = list(results.keys())
    primary = eval_targets[0]
    colors = ["gray", "steelblue"][:len(names)]

    returns = [results[n][primary]["return_mean"] for n in names]
    errors = [results[n][primary]["return_std"] for n in names]
    parameters = [results[n][primary]["parameters"] for n in names]
    speeds = [results[n][primary]["decisions_per_second"] for n in names]

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].bar(names, returns, yerr=errors, capsize=5, color=colors)
    axes[0].set_title(f"episode return (target {primary:.0f})")
    axes[0].axhline(used_mean_return, color="green", linestyle="--", linewidth=1)
    axes[0].axhline(measured_random_return, color="red", linestyle="--", linewidth=1)
    axes[0].text(0.02, used_mean_return, "mean of data used", color="green", fontsize=8, va="bottom")
    axes[0].text(0.02, measured_random_return, "random policy", color="red", fontsize=8, va="bottom")

    axes[1].bar(names, parameters, color=colors)
    axes[1].set_title("parameters")

    axes[2].bar(names, speeds, color=colors)
    axes[2].set_title("decisions per second")

    for axis in axes:
        axis.grid(axis="y", linestyle=":", alpha=0.6)

    figure.suptitle(f"{dataset_name}: Decision Transformer vs DeMa ({num_rollouts} rollouts, seed {seed})")
    figure.tight_layout()
    figure.savefig("comparison.png", dpi=150)


def main():
    data = load_data()
    obs_dim = data["states"].shape[-1]
    act_dim = data["actions"].shape[-1]

    os.makedirs(results_dir, exist_ok=True)
    results = {}
    record = {
        "dataset_name": dataset_name,
        "seed": seed,
        "context_len": context_len,
        "num_episodes_used": num_episodes_used,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "batch_size": batch_size,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "eval_targets": eval_targets,
        "num_rollouts": num_rollouts,
        "dropout": dropout,
        "norm_first": norm_first,
        "dema_norm_first": dema_norm_first,
        "token_layout": token_layout,
        "use_ser": use_ser,
        "ser_beta_min": ser_beta_min,
        "ser_beta_k": ser_beta_k,
        "dt_uses_timestep_embedding": dt_uses_timestep_embedding,
        "dema_uses_timestep_embedding": dema_uses_timestep_embedding,
        "timestep_embed_std": timestep_embed_std,
        "used_mean_return": data["used_mean_return"],
        "dataset_mean_return": dataset_mean_return,
        "measured_random_return": measured_random_return,
        "models": {},
    }

    builders = {
        "Decision Transformer": DecisionTransformer,
        "DeMa": DeMa,
        "Multi-Grained SSM": MultiGrainedSSM,
    }

    wandb_enabled = bool(os.environ.get("WANDB_API_KEY"))
    for name in models_to_run:
        try:
            model = builders[name](obs_dim, act_dim)
            trained, loss_history = train(model, name.replace(" ", "_").lower(), data)

            results[name] = {}
            record["models"][name] = {"loss_history": loss_history, "evaluations": {}, "recurrent": {}}

            for target in eval_targets:
                scores = evaluate(trained, data, target, num_rollouts)
                results[name][target] = scores
                record["models"][name]["evaluations"][str(target)] = scores
                print(
                    f"{name} target {target:.0f}: return {scores['return_mean']:.1f} +/- {scores['return_std']:.1f} "
                    f"(n={scores['rollouts']}), normalized {scores['normalized_return']:.1f}, "
                    f"velocity {scores['forward_velocity_mean']:.2f}, "
                    f"{scores['parameters']} parameters, {scores['decisions_per_second']:.1f} decisions/s"
                )
                if wandb_enabled:
                    import wandb
                    wandb.log({
                        f"eval/target_{target:.0f}/normalized_return": scores["normalized_return"],
                        f"eval/target_{target:.0f}/return_mean": scores["return_mean"],
                        f"eval/target_{target:.0f}/forward_velocity": scores["forward_velocity_mean"],
                    })

            if hasattr(trained, "step_decision") and token_layout == "interleaved":
                for target in eval_targets:
                    scores = evaluate_recurrent(trained, data, target, num_rollouts)
                    record["models"][name]["recurrent"][str(target)] = scores
                    print(
                        f"{name} target {target:.0f} RECURRENT (unlimited history): "
                        f"return {scores['return_mean']:.1f} +/- {scores['return_std']:.1f} "
                        f"(n={scores['rollouts']}), normalized {scores['normalized_return']:.1f}, "
                        f"velocity {scores['forward_velocity_mean']:.2f}, "
                        f"{scores['decisions_per_second']:.1f} decisions/s"
                    )
        finally:
            if wandb_enabled:
                import wandb
                if wandb.run is not None:
                    wandb.finish()

        short_names = {"Decision Transformer": "dt", "DeMa": "dema", "Multi-Grained SSM": "mgssm"}
        slug = "_".join(short_names[n] for n in models_to_run)
        with open(os.path.join(results_dir, f"run_seed{seed}_{slug}.json"), "w") as handle:
            json.dump(record, handle, indent=2)

    plot(results, data["used_mean_return"])


if __name__ == "__main__":
    main()
