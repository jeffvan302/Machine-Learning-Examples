from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


WORLD_X_MIN = -1.25
WORLD_X_MAX = 1.25
GROUND_Y = 0.0
WORLD_Y_MAX = 1.35
PAD_X = 0.0
ROCKET_HEIGHT = 0.11
ROCKET_WIDTH = 0.060
INPUT_DIM = 13


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


ACTIVATIONS = {
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": np.tanh,
}


@dataclass(frozen=True)
class DemoConfig:
    generations: int
    trainer: str
    population: int
    batch_episodes: int
    rollouts: int
    hidden_layers: tuple[int, ...]
    activation: str
    episode_steps: int
    seed: int | None
    mutation_scale: float
    elite_fraction: float
    inject_fraction: float
    learning_rate: float
    weight_decay: float
    gamma: float
    gae_lambda: float
    entropy_coef: float
    value_coef: float
    ppo_clip: float
    ppo_epochs: int
    minibatch_size: int
    gravity: float
    main_thrust: float
    turn_power: float
    fuel_burn: float
    pad_width: float
    landing_vx: float
    landing_vy: float
    landing_angle: float
    reward_progress_scale: float
    reward_step_penalty: float
    reward_throttle_penalty: float
    reward_turn_penalty: float
    reward_out_of_bounds_penalty: float
    reward_landing_bonus: float
    reward_landing_fuel_bonus: float
    reward_landing_precision_bonus: float
    reward_landing_precision_scale: float
    reward_crash_penalty: float
    reward_crash_impact_penalty: float
    reward_timeout_penalty: float
    spawn_mode: str
    spawn_randomness: str
    fps: int
    no_gui: bool


@dataclass(slots=True)
class RocketState:
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    ang_vel: float
    fuel: float


@dataclass(frozen=True)
class StepResult:
    reward: float
    next_cost: float
    throttle: float
    done: bool
    outcome: str | None
    landed: bool


def parse_hidden_layers(value: str | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    stripped = value.strip()
    if not stripped or stripped == "0":
        return ()
    layers = []
    for part in stripped.split(","):
        item = part.strip()
        if not item:
            continue
        size = int(item)
        if size < 0:
            raise argparse.ArgumentTypeError("hidden layer sizes must be zero or greater")
        if size > 0:
            layers.append(size)
    return tuple(layers)


def format_hidden_layers(hidden_layers: tuple[int, ...]) -> str:
    return "0" if not hidden_layers else ",".join(str(value) for value in hidden_layers)


def _activation_module(name: str):
    if nn is None:
        raise RuntimeError("PyTorch is required for the rocket policy.")
    if name == "relu":
        return nn.ReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def default_config() -> DemoConfig:
    return DemoConfig(
        generations=500,
        trainer="ppo",
        population=16,
        batch_episodes=20,
        rollouts=20,
        hidden_layers=(10, 40, 6),
        activation="relu",
        episode_steps=200,
        seed=None,
        mutation_scale=0.14,
        elite_fraction=0.24,
        inject_fraction=0.10,
        learning_rate=0.0005,
        weight_decay=0.00005,
        gamma=0.985,
        gae_lambda=0.95,
        entropy_coef=0.0015,
        value_coef=0.35,
        ppo_clip=0.20,
        ppo_epochs=6,
        minibatch_size=128,
        gravity=0.0098,
        main_thrust=0.0215,
        turn_power=0.0850,
        fuel_burn=0.0090,
        pad_width=0.38,
        landing_vx=0.055,
        landing_vy=0.060,
        landing_angle=0.28,
        reward_progress_scale=6.0,
        reward_step_penalty=0.020,
        reward_throttle_penalty=0.012,
        reward_turn_penalty=0.004,
        reward_out_of_bounds_penalty=12.0,
        reward_landing_bonus=25.0,
        reward_landing_fuel_bonus=6.0,
        reward_landing_precision_bonus=1.5,
        reward_landing_precision_scale=3.0,
        reward_crash_penalty=16.0,
        reward_crash_impact_penalty=6.0,
        reward_timeout_penalty=1.5,
        spawn_mode="random",
        spawn_randomness="dramatic",
        fps=30,
        no_gui=False,
    )


def validate_config(config: DemoConfig) -> None:
    if config.trainer not in {"ppo", "reinforce"}:
        raise ValueError("trainer must be ppo or reinforce")
    if config.generations < 1:
        raise ValueError("generations must be at least 1")
    if config.batch_episodes < 1:
        raise ValueError("batch_episodes must be at least 1")
    if config.rollouts < 1:
        raise ValueError("rollouts must be at least 1")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be greater than 0")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be zero or greater")
    if not 0.0 < config.gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if not 0.0 < config.gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in (0, 1]")
    if config.ppo_clip <= 0.0:
        raise ValueError("ppo_clip must be greater than 0")
    if config.ppo_epochs < 1:
        raise ValueError("ppo_epochs must be at least 1")
    if config.minibatch_size < 1:
        raise ValueError("minibatch_size must be at least 1")
    if config.gravity <= 0.0 or config.main_thrust <= 0.0 or config.turn_power <= 0.0:
        raise ValueError("physics settings must be greater than 0")
    if config.fuel_burn <= 0.0:
        raise ValueError("fuel_burn must be greater than 0")
    if config.pad_width <= 0.0 or config.landing_vx <= 0.0 or config.landing_vy <= 0.0 or config.landing_angle <= 0.0:
        raise ValueError("landing settings must be greater than 0")
    if config.reward_progress_scale < 0.0:
        raise ValueError("reward_progress_scale must be zero or greater")
    if config.reward_step_penalty < 0.0 or config.reward_throttle_penalty < 0.0 or config.reward_turn_penalty < 0.0:
        raise ValueError("step, throttle, and turn reward penalties must be zero or greater")
    if config.reward_out_of_bounds_penalty < 0.0:
        raise ValueError("reward_out_of_bounds_penalty must be zero or greater")
    if config.reward_landing_bonus < 0.0 or config.reward_landing_fuel_bonus < 0.0:
        raise ValueError("landing reward terms must be zero or greater")
    if config.reward_landing_precision_bonus < 0.0 or config.reward_landing_precision_scale < 0.0:
        raise ValueError("landing precision reward terms must be zero or greater")
    if config.reward_crash_penalty < 0.0 or config.reward_crash_impact_penalty < 0.0:
        raise ValueError("crash reward penalties must be zero or greater")
    if config.reward_timeout_penalty < 0.0:
        raise ValueError("reward_timeout_penalty must be zero or greater")
    if config.spawn_mode not in {"random", "side", "centered"}:
        raise ValueError("spawn_mode must be random, side, or centered")
    if config.spawn_randomness not in {"standard", "dramatic"}:
        raise ValueError("spawn_randomness must be standard or dramatic")
    if config.activation not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}")


def _randomness_profile(level: str) -> dict[str, tuple[float, float]]:
    if level == "dramatic":
        return {
            "altitude": (0.78, 1.16),
            "random_x": (-1.02, 1.02),
            "center_x": (-0.40, 0.40),
            "side_x": (0.58, 1.02),
            "vx": (-0.040, 0.040),
            "vy": (-0.020, 0.010),
            "angle": (-0.55, 0.55),
            "ang_vel": (-0.030, 0.030),
        }
    return {
        "altitude": (0.84, 1.02),
        "random_x": (-0.76, 0.76),
        "center_x": (-0.22, 0.22),
        "side_x": (0.42, 0.82),
        "vx": (-0.024, 0.024),
        "vy": (-0.012, 0.006),
        "angle": (-0.28, 0.28),
        "ang_vel": (-0.015, 0.015),
    }


def spawn_state(rng: np.random.Generator, config: DemoConfig) -> RocketState:
    profile = _randomness_profile(config.spawn_randomness)
    if config.spawn_mode == "centered":
        x = float(rng.uniform(*profile["center_x"]))
    elif config.spawn_mode == "side":
        side = -1.0 if rng.random() < 0.5 else 1.0
        x = float(side * rng.uniform(*profile["side_x"]))
    else:
        x = float(rng.uniform(*profile["random_x"]))

    y = float(rng.uniform(*profile["altitude"]))
    inward_drift = -0.020 * np.sign(x) if abs(x) > 0.10 else 0.0
    vx = float(rng.uniform(*profile["vx"]) + inward_drift)
    vy = float(rng.uniform(*profile["vy"]))
    angle_bias = -0.18 * np.sign(x) if abs(x) > 0.12 else 0.0
    angle = float(np.clip(angle_bias + rng.uniform(*profile["angle"]), -0.95, 0.95))
    ang_vel = float(rng.uniform(*profile["ang_vel"]))
    return RocketState(x=x, y=y, vx=vx, vy=vy, angle=angle, ang_vel=ang_vel, fuel=1.0)


def landing_cost(state: RocketState, config: DemoConfig) -> float:
    dx = abs(state.x - PAD_X) / (WORLD_X_MAX - WORLD_X_MIN)
    dy = max(0.0, state.y - GROUND_Y) / max(WORLD_Y_MAX - GROUND_Y, 1e-6)
    speed = np.hypot(state.vx, state.vy)
    cost = 1.65 * dx + 1.10 * dy + 4.25 * speed + 1.50 * abs(state.angle) + 0.80 * abs(state.ang_vel)
    if abs(state.x - PAD_X) <= config.pad_width * 0.5:
        cost *= 0.88
    if state.fuel <= 0.0:
        cost += 0.15
    return float(cost)


def build_inputs(state: RocketState, config: DemoConfig, step_fraction: float) -> np.ndarray:
    dx_to_pad = PAD_X - state.x
    speed = np.hypot(state.vx, state.vy)
    on_pad = 1.0 if abs(dx_to_pad) <= config.pad_width * 0.5 else 0.0
    return np.array(
        [
            np.clip(state.x / max(abs(WORLD_X_MIN), abs(WORLD_X_MAX)), -1.2, 1.2),
            np.clip(state.y / WORLD_Y_MAX, 0.0, 1.4),
            np.clip(dx_to_pad / (WORLD_X_MAX - WORLD_X_MIN), -1.0, 1.0),
            np.clip(state.vx / 0.12, -1.5, 1.5),
            np.clip(state.vy / 0.12, -1.5, 1.5),
            np.sin(state.angle),
            np.cos(state.angle),
            np.clip(state.angle / 1.2, -1.0, 1.0),
            np.clip(state.ang_vel / 0.20, -1.5, 1.5),
            np.clip(state.fuel, 0.0, 1.0),
            np.clip(speed / 0.14, 0.0, 2.2),
            on_pad,
            np.clip(step_fraction, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def step_environment(
    state: RocketState,
    turn_command: float,
    throttle_request: float,
    previous_cost: float,
    config: DemoConfig,
) -> StepResult:
    turn = float(np.clip(turn_command, -1.0, 1.0))
    requested_throttle = float(np.clip(throttle_request, 0.0, 1.0))
    if config.fuel_burn > 0.0:
        max_throttle = float(np.clip(state.fuel / config.fuel_burn, 0.0, 1.0))
    else:
        max_throttle = 0.0
    throttle = min(requested_throttle, max_throttle)

    state.fuel = max(0.0, state.fuel - throttle * config.fuel_burn)
    state.ang_vel = 0.82 * state.ang_vel + config.turn_power * turn
    state.angle = float(np.clip(state.angle + state.ang_vel, -1.35, 1.35))

    thrust = config.main_thrust * throttle
    state.vx = state.vx * 0.996 + np.sin(state.angle) * thrust
    state.vy = state.vy * 0.998 + np.cos(state.angle) * thrust - config.gravity
    state.x += state.vx
    state.y += state.vy

    next_cost = landing_cost(state, config)
    reward = (
        (previous_cost - next_cost) * config.reward_progress_scale
        - config.reward_step_penalty
        - config.reward_throttle_penalty * throttle
        - config.reward_turn_penalty * abs(turn)
    )
    landed = False
    done = False
    outcome: str | None = None

    if state.x < WORLD_X_MIN - 0.18 or state.x > WORLD_X_MAX + 0.18 or state.y > WORLD_Y_MAX + 0.25:
        reward -= config.reward_out_of_bounds_penalty
        done = True
        outcome = "out_of_bounds"
    elif state.y <= GROUND_Y:
        state.y = GROUND_Y
        on_pad = abs(state.x - PAD_X) <= config.pad_width * 0.5
        safe_touchdown = (
            on_pad
            and abs(state.vx) <= config.landing_vx
            and abs(state.vy) <= config.landing_vy
            and abs(state.angle) <= config.landing_angle
        )
        done = True
        state.ang_vel = 0.0
        state.vx = 0.0
        state.vy = 0.0
        if safe_touchdown:
            landed = True
            outcome = "landed"
            reward = (
                config.reward_landing_bonus
                + config.reward_landing_fuel_bonus * state.fuel
                + max(0.0, config.reward_landing_precision_bonus - next_cost * config.reward_landing_precision_scale)
            )
        else:
            outcome = "crashed" if on_pad else "missed_pad"
            impact = min(1.5, abs(state.vx) + abs(state.vy) + abs(state.angle))
            reward = -config.reward_crash_penalty - config.reward_crash_impact_penalty * impact

    return StepResult(
        reward=float(reward),
        next_cost=float(next_cost),
        throttle=float(throttle),
        done=done,
        outcome=outcome,
        landed=landed,
    )


def rocket_vertices(position: np.ndarray, angle: float) -> np.ndarray:
    body = np.array(
        [
            [0.0, ROCKET_HEIGHT * 0.5],
            [-ROCKET_WIDTH * 0.5, -ROCKET_HEIGHT * 0.45],
            [0.0, -ROCKET_HEIGHT * 0.20],
            [ROCKET_WIDTH * 0.5, -ROCKET_HEIGHT * 0.45],
        ],
        dtype=np.float64,
    )
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return body @ rotation.T + position.reshape(1, 2)


class TorchPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...], output_dim: int, activation_name: str) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is required for the rocket policy.")
        super().__init__()
        self.hidden_layers = hidden_layers
        self.activation_name = activation_name

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(_activation_module(activation_name))
            previous_dim = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.mean_head = nn.Linear(previous_dim, output_dim)
        self.value_head = nn.Linear(previous_dim, 1)
        self.log_std = nn.Parameter(torch.full((output_dim,), -0.60))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        assert nn is not None
        for module in self.modules():
            if isinstance(module, nn.Linear):
                gain = np.sqrt(2.0)
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.10)
        nn.init.orthogonal_(self.value_head.weight, gain=1.00)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(inputs)
        mean = self.mean_head(features)
        std = torch.exp(self.log_std).clamp(0.08, 1.20).expand_as(mean)
        value = self.value_head(features).squeeze(-1)
        return mean, std, value


def run_policy_episode(
    policy: TorchPolicy,
    config: DemoConfig,
    rng: np.random.Generator,
    record_episode: bool = False,
    deterministic: bool = False,
    requires_grad: bool = True,
) -> dict[str, object]:
    if torch is None:
        raise RuntimeError("PyTorch is required for policy rollouts.")

    state = spawn_state(rng, config)
    previous_cost = landing_cost(state, config)
    total_reward = 0.0
    landed = False
    outcome = "timed_out"

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    old_log_probs: list[float] = []
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    value_predictions: list[float] = []
    step_rewards: list[float] = []
    dones: list[bool] = []

    for step in range(config.episode_steps):
        step_fraction = step / max(config.episode_steps - 1, 1)
        observation = build_inputs(state, config, step_fraction)
        tensor = torch.from_numpy(observation).unsqueeze(0)
        if requires_grad:
            mean, std, value = policy(tensor)
            distribution = torch.distributions.Normal(mean, std)
            if deterministic:
                raw_action = mean.squeeze(0)
                log_prob = distribution.log_prob(raw_action.unsqueeze(0)).sum(dim=-1).squeeze(0)
            else:
                raw_action = distribution.rsample().squeeze(0)
                log_prob = distribution.log_prob(raw_action.unsqueeze(0)).sum(dim=-1).squeeze(0)
            entropy = distribution.entropy().sum(dim=-1).squeeze(0)
        else:
            with torch.inference_mode():
                mean, std, value = policy(tensor)
                distribution = torch.distributions.Normal(mean, std)
                if deterministic:
                    raw_action = mean.squeeze(0)
                    log_prob = distribution.log_prob(raw_action.unsqueeze(0)).sum(dim=-1).squeeze(0)
                else:
                    raw_action = distribution.sample().squeeze(0)
                    log_prob = distribution.log_prob(raw_action.unsqueeze(0)).sum(dim=-1).squeeze(0)
                entropy = distribution.entropy().sum(dim=-1).squeeze(0)

        turn_command = float(torch.tanh(raw_action[0]).item())
        throttle_request = float(torch.sigmoid(raw_action[1]).item())
        step_result = step_environment(state, turn_command, throttle_request, previous_cost, config)
        previous_cost = step_result.next_cost
        total_reward += step_result.reward

        observations.append(observation.astype(np.float32))
        actions.append(raw_action.detach().cpu().numpy().astype(np.float32))
        old_log_probs.append(float(log_prob.item()))
        if requires_grad:
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value.squeeze(0))
        value_predictions.append(float(value.item()))
        step_rewards.append(float(step_result.reward))
        dones.append(bool(step_result.done))

        if step_result.done:
            landed = step_result.landed
            outcome = step_result.outcome or outcome
            break
    else:
        total_reward -= config.reward_timeout_penalty
        if step_rewards:
            step_rewards[-1] -= config.reward_timeout_penalty
            dones[-1] = True

    return {
        "reward": float(total_reward),
        "landed": bool(landed),
        "outcome": outcome,
        "observations": observations,
        "actions": actions,
        "old_log_probs": old_log_probs,
        "log_probs": log_probs,
        "entropies": entropies,
        "values": values,
        "value_predictions": value_predictions,
        "step_rewards": step_rewards,
        "dones": dones,
        "record_episode": record_episode,
    }


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    returns = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        returns[index] = running
    return returns


def generalized_advantages(
    rewards: list[float],
    value_predictions: list[float],
    dones: list[bool],
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    count = len(rewards)
    advantages = [0.0] * count
    returns = [0.0] * count
    last_advantage = 0.0
    next_value = 0.0
    for index in range(count - 1, -1, -1):
        mask = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * next_value * mask - value_predictions[index]
        last_advantage = delta + gamma * gae_lambda * mask * last_advantage
        advantages[index] = last_advantage
        returns[index] = last_advantage + value_predictions[index]
        next_value = value_predictions[index]
    return advantages, returns


class GradientTrainer:
    def __init__(self, config: DemoConfig) -> None:
        if torch is None or nn is None or F is None:
            raise RuntimeError("PyTorch is required for training.")
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        if config.seed is not None:
            torch.manual_seed(config.seed)
        self.policy = TorchPolicy(INPUT_DIM, config.hidden_layers, 2, config.activation)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.last_loss: float | None = None

    def _train_reinforce_generation(self) -> None:
        seed_values = [int(value) for value in self.rng.integers(0, 2_000_000_000, size=self.config.batch_episodes)]
        episodes = []
        total_steps = 0
        for seed_value in seed_values:
            episode = run_policy_episode(
                self.policy,
                self.config,
                np.random.default_rng(seed_value),
                record_episode=False,
                deterministic=False,
                requires_grad=True,
            )
            if episode["log_probs"]:
                episodes.append(episode)
                total_steps += len(episode["log_probs"])

        self.last_loss = None
        if not episodes:
            return

        returns = [
            torch.tensor(discounted_returns(episode["step_rewards"], self.config.gamma), dtype=torch.float32)
            for episode in episodes
        ]
        flattened_returns = torch.cat(returns)
        if flattened_returns.numel() > 1:
            flattened_returns = (flattened_returns - flattened_returns.mean()) / (
                flattened_returns.std(unbiased=False) + 1e-6
            )

        self.optimizer.zero_grad()
        policy_loss = torch.tensor(0.0)
        value_loss = torch.tensor(0.0)
        entropy_bonus = torch.tensor(0.0)
        cursor = 0
        for episode_index, episode in enumerate(episodes):
            episode_returns = flattened_returns[cursor : cursor + len(returns[episode_index])]
            cursor += len(returns[episode_index])
            log_prob_tensor = torch.stack(episode["log_probs"])
            entropy_tensor = torch.stack(episode["entropies"])
            value_tensor = torch.stack(episode["values"])
            advantages = episode_returns - value_tensor.detach()
            policy_loss = policy_loss - (log_prob_tensor * advantages).sum()
            value_loss = value_loss + F.mse_loss(value_tensor, episode_returns, reduction="sum")
            entropy_bonus = entropy_bonus + entropy_tensor.sum()

        loss = (
            policy_loss / max(total_steps, 1)
            + self.config.value_coef * value_loss / max(total_steps, 1)
            - self.config.entropy_coef * entropy_bonus / max(total_steps, 1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.last_loss = float(loss.item())

    def _train_ppo_generation(self) -> None:
        seed_values = [int(value) for value in self.rng.integers(0, 2_000_000_000, size=self.config.batch_episodes)]
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        old_log_probs: list[float] = []
        returns: list[float] = []
        advantages: list[float] = []

        for seed_value in seed_values:
            episode = run_policy_episode(
                self.policy,
                self.config,
                np.random.default_rng(seed_value),
                record_episode=False,
                deterministic=False,
                requires_grad=False,
            )
            if not episode["observations"]:
                continue
            episode_advantages, episode_returns = generalized_advantages(
                episode["step_rewards"],
                episode["value_predictions"],
                episode["dones"],
                self.config.gamma,
                self.config.gae_lambda,
            )
            observations.extend(episode["observations"])
            actions.extend(episode["actions"])
            old_log_probs.extend(episode["old_log_probs"])
            returns.extend(episode_returns)
            advantages.extend(episode_advantages)

        self.last_loss = None
        if not observations:
            return

        obs_tensor = torch.tensor(np.array(observations, dtype=np.float32), dtype=torch.float32)
        action_tensor = torch.tensor(np.array(actions, dtype=np.float32), dtype=torch.float32)
        old_log_prob_tensor = torch.tensor(np.array(old_log_probs, dtype=np.float32), dtype=torch.float32)
        return_tensor = torch.tensor(np.array(returns, dtype=np.float32), dtype=torch.float32)
        advantage_tensor = torch.tensor(np.array(advantages, dtype=np.float32), dtype=torch.float32)
        if advantage_tensor.numel() > 1:
            advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
                advantage_tensor.std(unbiased=False) + 1e-6
            )

        sample_count = obs_tensor.shape[0]
        minibatch_size = min(self.config.minibatch_size, sample_count)
        losses: list[float] = []

        for _ in range(self.config.ppo_epochs):
            permutation = torch.randperm(sample_count)
            for start in range(0, sample_count, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                batch_obs = obs_tensor[indices]
                batch_actions = action_tensor[indices]
                batch_old_log_probs = old_log_prob_tensor[indices]
                batch_returns = return_tensor[indices]
                batch_advantages = advantage_tensor[indices]

                mean, std, values = self.policy(batch_obs)
                distribution = torch.distributions.Normal(mean, std)
                new_log_probs = distribution.log_prob(batch_actions).sum(dim=-1)
                entropy = distribution.entropy().sum(dim=-1)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(ratio, 1.0 - self.config.ppo_clip, 1.0 + self.config.ppo_clip) * batch_advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, batch_returns)
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.optimizer.step()
                losses.append(float(loss.item()))

        if losses:
            self.last_loss = float(np.mean(losses))

    def evaluate(self) -> tuple[float, float, float]:
        reward_values: list[float] = []
        landings = 0.0
        seed_values = [int(value) for value in self.rng.integers(0, 2_000_000_000, size=self.config.rollouts)]
        for seed_value in seed_values:
            episode = run_policy_episode(
                self.policy,
                self.config,
                np.random.default_rng(seed_value),
                record_episode=False,
                deterministic=True,
                requires_grad=False,
            )
            reward_values.append(float(episode["reward"]))
            landings += float(episode["landed"])
        return (
            float(np.max(reward_values)),
            float(np.mean(reward_values)),
            float(landings / max(len(reward_values), 1)),
        )

    def train_generation(self) -> tuple[float, float, float]:
        if self.config.trainer == "reinforce":
            self._train_reinforce_generation()
        else:
            self._train_ppo_generation()
        return self.evaluate()


def build_parser() -> argparse.ArgumentParser:
    defaults = default_config()
    parser = argparse.ArgumentParser(
        description="Rocket landing RL backend. Use rocket_landing_drl_gui.py for the interactive visualization."
    )
    parser.add_argument("--generations", type=int, default=defaults.generations)
    parser.add_argument("--trainer", choices=["ppo", "reinforce"], default=defaults.trainer)
    parser.add_argument("--population", type=int, default=defaults.population)
    parser.add_argument("--batch-episodes", type=int, default=defaults.batch_episodes)
    parser.add_argument("--rollouts", type=int, default=defaults.rollouts)
    parser.add_argument("--hidden", type=parse_hidden_layers, default=defaults.hidden_layers)
    parser.add_argument("--activation", choices=tuple(sorted(ACTIVATIONS)), default=defaults.activation)
    parser.add_argument("--episode-steps", type=int, default=defaults.episode_steps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--gae-lambda", type=float, default=defaults.gae_lambda)
    parser.add_argument("--entropy-coef", type=float, default=defaults.entropy_coef)
    parser.add_argument("--value-coef", type=float, default=defaults.value_coef)
    parser.add_argument("--ppo-clip", type=float, default=defaults.ppo_clip)
    parser.add_argument("--ppo-epochs", type=int, default=defaults.ppo_epochs)
    parser.add_argument("--minibatch-size", type=int, default=defaults.minibatch_size)
    parser.add_argument("--gravity", type=float, default=defaults.gravity)
    parser.add_argument("--main-thrust", type=float, default=defaults.main_thrust)
    parser.add_argument("--turn-power", type=float, default=defaults.turn_power)
    parser.add_argument("--fuel-burn", type=float, default=defaults.fuel_burn)
    parser.add_argument("--pad-width", type=float, default=defaults.pad_width)
    parser.add_argument("--landing-vx", type=float, default=defaults.landing_vx)
    parser.add_argument("--landing-vy", type=float, default=defaults.landing_vy)
    parser.add_argument("--landing-angle", type=float, default=defaults.landing_angle)
    parser.add_argument("--reward-progress-scale", type=float, default=defaults.reward_progress_scale)
    parser.add_argument("--reward-step-penalty", type=float, default=defaults.reward_step_penalty)
    parser.add_argument("--reward-throttle-penalty", type=float, default=defaults.reward_throttle_penalty)
    parser.add_argument("--reward-turn-penalty", type=float, default=defaults.reward_turn_penalty)
    parser.add_argument("--reward-out-of-bounds-penalty", type=float, default=defaults.reward_out_of_bounds_penalty)
    parser.add_argument("--reward-landing-bonus", type=float, default=defaults.reward_landing_bonus)
    parser.add_argument("--reward-landing-fuel-bonus", type=float, default=defaults.reward_landing_fuel_bonus)
    parser.add_argument("--reward-landing-precision-bonus", type=float, default=defaults.reward_landing_precision_bonus)
    parser.add_argument("--reward-landing-precision-scale", type=float, default=defaults.reward_landing_precision_scale)
    parser.add_argument("--reward-crash-penalty", type=float, default=defaults.reward_crash_penalty)
    parser.add_argument("--reward-crash-impact-penalty", type=float, default=defaults.reward_crash_impact_penalty)
    parser.add_argument("--reward-timeout-penalty", type=float, default=defaults.reward_timeout_penalty)
    parser.add_argument("--spawn-mode", choices=["random", "side", "centered"], default=defaults.spawn_mode)
    parser.add_argument("--spawn-randomness", choices=["standard", "dramatic"], default=defaults.spawn_randomness)
    parser.add_argument("--fps", type=int, default=defaults.fps)
    parser.add_argument("--no-gui", action="store_true", help="Run a headless training loop in the terminal.")
    return parser


def config_from_args(args: argparse.Namespace) -> DemoConfig:
    config = DemoConfig(
        generations=args.generations,
        trainer=args.trainer,
        population=args.population,
        batch_episodes=args.batch_episodes,
        rollouts=args.rollouts,
        hidden_layers=args.hidden,
        activation=args.activation,
        episode_steps=args.episode_steps,
        seed=args.seed,
        mutation_scale=0.14,
        elite_fraction=0.24,
        inject_fraction=0.10,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        ppo_clip=args.ppo_clip,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        gravity=args.gravity,
        main_thrust=args.main_thrust,
        turn_power=args.turn_power,
        fuel_burn=args.fuel_burn,
        pad_width=args.pad_width,
        landing_vx=args.landing_vx,
        landing_vy=args.landing_vy,
        landing_angle=args.landing_angle,
        reward_progress_scale=args.reward_progress_scale,
        reward_step_penalty=args.reward_step_penalty,
        reward_throttle_penalty=args.reward_throttle_penalty,
        reward_turn_penalty=args.reward_turn_penalty,
        reward_out_of_bounds_penalty=args.reward_out_of_bounds_penalty,
        reward_landing_bonus=args.reward_landing_bonus,
        reward_landing_fuel_bonus=args.reward_landing_fuel_bonus,
        reward_landing_precision_bonus=args.reward_landing_precision_bonus,
        reward_landing_precision_scale=args.reward_landing_precision_scale,
        reward_crash_penalty=args.reward_crash_penalty,
        reward_crash_impact_penalty=args.reward_crash_impact_penalty,
        reward_timeout_penalty=args.reward_timeout_penalty,
        spawn_mode=args.spawn_mode,
        spawn_randomness=args.spawn_randomness,
        fps=args.fps,
        no_gui=args.no_gui,
    )
    validate_config(config)
    return config


def run_headless(config: DemoConfig) -> None:
    trainer = GradientTrainer(config)
    for generation in range(1, config.generations + 1):
        best_reward, mean_reward, landing_rate = trainer.train_generation()
        if generation == 1 or generation == config.generations or generation % max(1, min(25, config.generations // 10 or 1)) == 0:
            loss_text = "n/a" if trainer.last_loss is None else f"{trainer.last_loss:.4f}"
            print(
                f"generation={generation} "
                f"best_reward={best_reward:.2f} "
                f"mean_reward={mean_reward:.2f} "
                f"landing_rate={landing_rate * 100.0:.1f}% "
                f"loss={loss_text}"
            )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = config_from_args(args)
    except Exception as exc:
        parser.error(str(exc))
        return

    if not config.no_gui:
        print("rocket_landing_rl_demo.py has been restored as the backend for rocket_landing_drl_gui.py.")
        print("For the interactive presentation GUI, run:")
        print("  python rocket_landing_drl_gui.py")
        print("")
        print("To run the backend directly in headless mode, rerun this script with --no-gui.")
        return

    run_headless(config)


if __name__ == "__main__":
    main()
