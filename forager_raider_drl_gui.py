from __future__ import annotations

import argparse
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from tkinter import messagebox, ttk

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - handled at runtime
    torch = None
    nn = None
    F = None


ACTIONS = ("Up", "Down", "Left", "Right", "Stay")
ACTION_DELTAS = (
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0),
    (0, 0),
)
OBSERVATION_NAMES = (
    "self_x",
    "self_y",
    "opp_dx",
    "opp_dy",
    "base_dx",
    "base_dy",
    "food_dx",
    "food_dy",
    "self_carry",
    "opp_carry",
    "food_frac",
    "self_score",
    "opp_score",
    "opp_dist",
    "target_dist",
)
ACTIVATIONS = ("relu", "sigmoid", "tanh")
INPUT_DIM = len(OBSERVATION_NAMES)
OUTPUT_DIM = len(ACTIONS)


def parse_hidden_layers(value: str) -> tuple[int, ...]:
    cleaned = value.strip()
    if not cleaned:
        raise argparse.ArgumentTypeError("hidden layout cannot be empty")
    parts = [part.strip() for part in cleaned.split(",")]
    try:
        layers = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("hidden layout must look like 32 or 32,16") from exc
    if any(layer < 0 for layer in layers):
        raise argparse.ArgumentTypeError("hidden layers must be zero or greater")
    if 0 in layers and layers != (0,):
        raise argparse.ArgumentTypeError("0 can only be used by itself to mean no hidden layers")
    return () if layers == (0,) else layers


def format_hidden_layers(hidden_layers: tuple[int, ...]) -> str:
    return "linear" if not hidden_layers else ",".join(str(layer) for layer in hidden_layers)


def torch_activation_factory(name: str):
    if nn is None:
        raise RuntimeError("PyTorch is required for the deep reinforcement learning demo.")
    mapping = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
    }
    return mapping[name]


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def normalize_delta(delta: int, grid_size: int) -> float:
    return float(delta) / max(1, grid_size - 1)


def normalize_coord(coord: int, grid_size: int) -> float:
    if grid_size <= 1:
        return 0.0
    return (2.0 * coord / (grid_size - 1)) - 1.0


def positive_ratio(value: int, scale: int) -> float:
    return float(value) / max(1, scale)


def color_for_value(value: float) -> str:
    clipped = float(np.tanh(value))
    if clipped >= 0.0:
        blue = int(160 + 70 * clipped)
        red = int(255 - 110 * clipped)
        green = int(240 - 80 * clipped)
    else:
        amount = abs(clipped)
        red = int(215 + 35 * amount)
        green = int(220 - 95 * amount)
        blue = int(240 - 140 * amount)
    return f"#{red:02x}{green:02x}{blue:02x}"


def edge_color(weight: float) -> str:
    clipped = max(-1.0, min(1.0, weight))
    if clipped >= 0.0:
        red = int(208 - 88 * clipped)
        green = int(220 - 36 * clipped)
        blue = int(230 + 12 * clipped)
    else:
        amount = abs(clipped)
        red = int(214 + 22 * amount)
        green = int(205 - 85 * amount)
        blue = int(220 - 105 * amount)
    return f"#{red:02x}{green:02x}{blue:02x}"


@dataclass(slots=True)
class AppConfig:
    trainer: str
    train_side: str
    grid_size: int
    food_count: int
    episode_steps: int
    hidden_layers: tuple[int, ...]
    activation: str
    learning_rate: float
    weight_decay: float
    gamma: float
    batch_size: int
    replay_size: int
    min_replay_size: int
    target_update_steps: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    steps_per_tick: int
    tick_ms: int
    seed: int | None
    smoke_test_steps: int = 0


@dataclass(slots=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.storage: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, transition: Transition) -> None:
        self.storage.append(transition)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Transition]:
        indices = rng.integers(0, len(self.storage), size=batch_size)
        return [self.storage[int(index)] for index in indices]


class QNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: tuple[int, ...],
        output_dim: int,
        activation_name: str,
    ) -> None:
        super().__init__()
        activation_cls = torch_activation_factory(activation_name)
        self.hidden_layers = hidden_layers
        self.activation_name = activation_name
        self.hidden_linears = nn.ModuleList()
        previous_dim = input_dim
        for hidden_dim in hidden_layers:
            linear = nn.Linear(previous_dim, hidden_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.hidden_linears.append(linear)
            previous_dim = hidden_dim
        self.output_linear = nn.Linear(previous_dim, output_dim)
        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)
        self.activation = activation_cls()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for linear in self.hidden_linears:
            values = self.activation(linear(values))
        return self.output_linear(values)

    def forward_with_activations(self, values: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_values: list[torch.Tensor] = []
        for linear in self.hidden_linears:
            values = self.activation(linear(values))
            hidden_values.append(values)
        outputs = self.output_linear(values)
        return outputs, hidden_values

    def weight_matrices(self) -> list[np.ndarray]:
        weights = [layer.weight.detach().cpu().numpy() for layer in self.hidden_linears]
        weights.append(self.output_linear.weight.detach().cpu().numpy())
        return weights


class ForagerRaiderEnv:
    def __init__(self, config: AppConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        mid = config.grid_size // 2
        self.forager_base = (1, mid)
        self.raider_base = (config.grid_size - 2, mid)
        self.step_index = 0
        self.last_event = ""
        self.done = False
        self.food_positions: set[tuple[int, int]] = set()
        self.forager_position = self.forager_base
        self.raider_position = self.raider_base
        self.forager_carry = False
        self.raider_carry = False
        self.forager_score = 0
        self.raider_score = 0
        self.reset()

    def reset(self) -> np.ndarray:
        self.step_index = 0
        self.done = False
        self.last_event = "New episode"
        self.forager_position = self.forager_base
        self.raider_position = self.raider_base
        self.forager_carry = False
        self.raider_carry = False
        self.forager_score = 0
        self.raider_score = 0
        self.food_positions = self._spawn_food()
        return self.observation_for(self.config.train_side)

    def _spawn_food(self) -> set[tuple[int, int]]:
        positions: set[tuple[int, int]] = set()
        candidates = [
            (x, y)
            for x in range(1, self.config.grid_size - 1)
            for y in range(0, self.config.grid_size)
            if (x, y) not in {self.forager_base, self.raider_base}
        ]
        self.rng.shuffle(candidates)
        for position in candidates[: self.config.food_count]:
            positions.add(position)
        return positions

    def _move(self, position: tuple[int, int], action_index: int) -> tuple[int, int]:
        dx, dy = ACTION_DELTAS[action_index]
        return (
            clamp(position[0] + dx, 0, self.config.grid_size - 1),
            clamp(position[1] + dy, 0, self.config.grid_size - 1),
        )

    def _nearest_food(self, position: tuple[int, int]) -> tuple[int, int]:
        if not self.food_positions:
            return self.forager_base if position[0] < self.config.grid_size // 2 else self.raider_base
        return min(self.food_positions, key=lambda food: (manhattan(position, food), food[0], food[1]))

    def _progress_target(self, side: str) -> tuple[int, int]:
        if side == "forager":
            if self.forager_carry:
                return self.forager_base
            if self.food_positions:
                return self._nearest_food(self.forager_position)
            return self.forager_base
        if self.raider_carry:
            return self.raider_base
        if self.forager_carry:
            return self.forager_position
        if self.food_positions:
            return self._nearest_food(self.raider_position)
        return self.forager_position

    def _side_state(self, side: str) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], bool, bool, int, int]:
        if side == "forager":
            return (
                self.forager_position,
                self.raider_position,
                self.forager_base,
                self.forager_carry,
                self.raider_carry,
                self.forager_score,
                self.raider_score,
            )
        return (
            self.raider_position,
            self.forager_position,
            self.raider_base,
            self.raider_carry,
            self.forager_carry,
            self.raider_score,
            self.forager_score,
        )

    def observation_for(self, side: str) -> np.ndarray:
        self_position, other_position, base_position, self_carry, other_carry, self_score, other_score = self._side_state(side)
        nearest_food = self._nearest_food(self_position)
        target = self._progress_target(side)
        scale = 2 * max(1, self.config.grid_size - 1)
        values = np.array(
            [
                normalize_coord(self_position[0], self.config.grid_size),
                normalize_coord(self_position[1], self.config.grid_size),
                normalize_delta(other_position[0] - self_position[0], self.config.grid_size),
                normalize_delta(other_position[1] - self_position[1], self.config.grid_size),
                normalize_delta(base_position[0] - self_position[0], self.config.grid_size),
                normalize_delta(base_position[1] - self_position[1], self.config.grid_size),
                normalize_delta(nearest_food[0] - self_position[0], self.config.grid_size),
                normalize_delta(nearest_food[1] - self_position[1], self.config.grid_size),
                float(self_carry),
                float(other_carry),
                positive_ratio(len(self.food_positions), self.config.food_count),
                positive_ratio(self_score, self.config.food_count),
                positive_ratio(other_score, self.config.food_count),
                positive_ratio(manhattan(self_position, other_position), scale),
                positive_ratio(manhattan(self_position, target), scale),
            ],
            dtype=np.float32,
        )
        return values

    def _heuristic_action(self, side: str) -> int:
        if side == "forager":
            current = self.forager_position
            score_fn = self._forager_heuristic_score
        else:
            current = self.raider_position
            score_fn = self._raider_heuristic_score
        best_index = 0
        best_score = -1e9
        for index in range(OUTPUT_DIM):
            next_position = self._move(current, index)
            score = score_fn(next_position)
            score += float(self.rng.normal(scale=0.025))
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def _forager_heuristic_score(self, next_position: tuple[int, int]) -> float:
        score = 0.0
        target = self.forager_base if self.forager_carry else self._nearest_food(next_position)
        score -= 1.15 * manhattan(next_position, target)
        score += 0.35 * manhattan(next_position, self.raider_position)
        if self.forager_carry and next_position == self.raider_position:
            score -= 4.5
        if next_position == self.forager_base and self.forager_carry:
            score += 4.0
        if next_position in self.food_positions and not self.forager_carry:
            score += 1.2
        return score

    def _raider_heuristic_score(self, next_position: tuple[int, int]) -> float:
        score = 0.0
        if self.raider_carry:
            score -= 1.20 * manhattan(next_position, self.raider_base)
            if next_position == self.raider_base:
                score += 4.0
        elif self.forager_carry:
            score -= 1.35 * manhattan(next_position, self.forager_position)
            if next_position == self.forager_position:
                score += 4.4
        elif self.food_positions:
            score -= 0.90 * manhattan(next_position, self._nearest_food(next_position))
            score -= 0.25 * manhattan(next_position, self.forager_position)
        else:
            score -= manhattan(next_position, self.forager_position)
        return score

    def _pickup_food(self, position: tuple[int, int], side: str, events: list[str]) -> bool:
        if position not in self.food_positions:
            return False
        if side == "forager":
            if self.forager_carry:
                return False
            self.forager_carry = True
            self.food_positions.remove(position)
            events.append("Forager picked up food")
            return True
        if self.raider_carry:
            return False
        self.raider_carry = True
        self.food_positions.remove(position)
        events.append("Raider picked up food")
        return True

    def _resolve_collisions(self, old_forager: tuple[int, int], old_raider: tuple[int, int], events: list[str]) -> tuple[bool, bool]:
        encounter = self.forager_position == self.raider_position
        crossed = self.forager_position == old_raider and self.raider_position == old_forager
        stole = False
        recovered = False
        if encounter or crossed:
            if self.forager_carry and not self.raider_carry:
                self.forager_carry = False
                self.raider_carry = True
                events.append("Raider stole the cargo")
                stole = True
            elif self.raider_carry and not self.forager_carry:
                self.raider_carry = False
                self.forager_carry = True
                events.append("Forager recovered the cargo")
                recovered = True
        return stole, recovered

    def _resolve_deliveries(self, events: list[str]) -> tuple[bool, bool]:
        forager_delivered = False
        raider_delivered = False
        if self.forager_carry and self.forager_position == self.forager_base:
            self.forager_carry = False
            self.forager_score += 1
            events.append("Forager scored")
            forager_delivered = True
        if self.raider_carry and self.raider_position == self.raider_base:
            self.raider_carry = False
            self.raider_score += 1
            events.append("Raider scored")
            raider_delivered = True
        return forager_delivered, raider_delivered

    def _apply_reward_shaping(
        self,
        side: str,
        before_position: tuple[int, int],
        before_target: tuple[int, int],
        events: dict[str, bool],
    ) -> float:
        reward = -0.015
        if side == "forager":
            after_position = self.forager_position
            progress = manhattan(before_position, before_target) - manhattan(after_position, before_target)
            reward += 0.055 * progress
            if events["forager_pickup"]:
                reward += 0.55
            if events["forager_score"]:
                reward += 1.80
            if events["raider_steal"]:
                reward -= 1.05
            if events["forager_recover"]:
                reward += 0.45
            reward -= 0.02 * (1.0 - positive_ratio(manhattan(self.forager_position, self.raider_position), self.config.grid_size))
        else:
            after_position = self.raider_position
            progress = manhattan(before_position, before_target) - manhattan(after_position, before_target)
            reward += 0.055 * progress
            if events["raider_pickup"]:
                reward += 0.35
            if events["raider_score"]:
                reward += 1.65
            if events["raider_steal"]:
                reward += 0.80
            if events["forager_score"]:
                reward -= 0.95
            if events["forager_recover"]:
                reward -= 0.60
        return reward

    def step(self, action_index: int) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.done:
            return self.reset(), 0.0, False, {"event": "Episode reset"}
        old_forager = self.forager_position
        old_raider = self.raider_position
        side = self.config.train_side
        before_position = self.forager_position if side == "forager" else self.raider_position
        before_target = self._progress_target(side)
        forager_action = action_index if side == "forager" else self._heuristic_action("forager")
        raider_action = action_index if side == "raider" else self._heuristic_action("raider")
        self.forager_position = self._move(self.forager_position, forager_action)
        self.raider_position = self._move(self.raider_position, raider_action)
        events_text: list[str] = []
        if (
            self.forager_position == self.raider_position
            and self.forager_position in self.food_positions
            and not self.forager_carry
            and not self.raider_carry
        ):
            if float(self.rng.random()) < 0.5:
                forager_pickup = self._pickup_food(self.forager_position, "forager", events_text)
                raider_pickup = False
            else:
                raider_pickup = self._pickup_food(self.raider_position, "raider", events_text)
                forager_pickup = False
        else:
            forager_pickup = self._pickup_food(self.forager_position, "forager", events_text)
            raider_pickup = self._pickup_food(self.raider_position, "raider", events_text)
        raider_steal, forager_recover = self._resolve_collisions(old_forager, old_raider, events_text)
        forager_score, raider_score = self._resolve_deliveries(events_text)
        self.step_index += 1
        if not events_text:
            events_text.append("Agents repositioned")
        self.last_event = " | ".join(events_text)
        self.done = self.step_index >= self.config.episode_steps or (
            not self.food_positions and not self.forager_carry and not self.raider_carry
        )
        events = {
            "forager_pickup": forager_pickup,
            "raider_pickup": raider_pickup,
            "raider_steal": raider_steal,
            "forager_recover": forager_recover,
            "forager_score": forager_score,
            "raider_score": raider_score,
        }
        reward = self._apply_reward_shaping(side, before_position, before_target, events)
        if self.done:
            score_delta = self.forager_score - self.raider_score
            if side == "raider":
                score_delta = -score_delta
            reward += 0.65 * score_delta
        info = {
            "event": self.last_event,
            "forager_action": ACTIONS[forager_action],
            "raider_action": ACTIONS[raider_action],
            "forager_score": self.forager_score,
            "raider_score": self.raider_score,
        }
        return self.observation_for(side), reward, self.done, info

    def snapshot(self) -> dict[str, object]:
        return {
            "grid_size": self.config.grid_size,
            "food_positions": sorted(self.food_positions),
            "forager_position": self.forager_position,
            "raider_position": self.raider_position,
            "forager_base": self.forager_base,
            "raider_base": self.raider_base,
            "forager_carry": self.forager_carry,
            "raider_carry": self.raider_carry,
            "forager_score": self.forager_score,
            "raider_score": self.raider_score,
            "step_index": self.step_index,
            "episode_steps": self.config.episode_steps,
            "event": self.last_event,
            "done": self.done,
        }


class DQNTrainer:
    def __init__(self, config: AppConfig) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for the deep reinforcement learning demo.")
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        if config.seed is not None:
            torch.manual_seed(config.seed)
        self.device = torch.device("cpu")
        self.online_net = QNetwork(INPUT_DIM, config.hidden_layers, OUTPUT_DIM, config.activation).to(self.device)
        self.target_net = QNetwork(INPUT_DIM, config.hidden_layers, OUTPUT_DIM, config.activation).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.AdamW(
            self.online_net.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.replay = ReplayBuffer(config.replay_size)
        self.env = ForagerRaiderEnv(config, self.rng)
        self.current_state = self.env.reset()
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0
        self.global_step = 0
        self.episode_index = 1
        self.update_count = 0
        self.last_loss: float | None = None
        self.last_action_index = OUTPUT_DIM - 1
        self.last_action_mode = "inspect"
        self.last_q_values = np.zeros(OUTPUT_DIM, dtype=np.float32)
        self.last_hidden_values: list[np.ndarray] = [np.zeros(layer, dtype=np.float32) for layer in config.hidden_layers]
        self.last_reward = 0.0
        self.last_info: dict[str, object] = {"event": self.env.last_event}
        self.recent_rewards: deque[float] = deque(maxlen=60)
        self.recent_lengths: deque[int] = deque(maxlen=60)
        self.recent_scores: deque[float] = deque(maxlen=60)
        self.reward_history: list[float] = []
        self.score_history: list[float] = []
        self.loss_history: list[float] = []
        self.inspect(self.current_state)

    def epsilon(self) -> float:
        progress = min(1.0, self.global_step / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + (self.config.epsilon_end - self.config.epsilon_start) * progress

    def inspect(self, state: np.ndarray) -> dict[str, object]:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values, hidden_values = self.online_net.forward_with_activations(state_tensor)
        self.last_q_values = q_values.squeeze(0).cpu().numpy().astype(np.float32)
        self.last_hidden_values = [layer.squeeze(0).cpu().numpy().astype(np.float32) for layer in hidden_values]
        greedy_action = int(np.argmax(self.last_q_values))
        return {
            "input_names": OBSERVATION_NAMES,
            "input_values": state.copy(),
            "hidden_values": [values.copy() for values in self.last_hidden_values],
            "output_names": ACTIONS,
            "output_values": self.last_q_values.copy(),
            "selected_action": greedy_action,
            "weight_matrices": self.online_net.weight_matrices(),
        }

    def _select_action(self) -> tuple[int, str]:
        epsilon = self.epsilon()
        greedy_action = int(np.argmax(self.last_q_values))
        if float(self.rng.random()) < epsilon:
            return int(self.rng.integers(0, OUTPUT_DIM)), "explore"
        return greedy_action, "greedy"

    def _optimize(self) -> None:
        if len(self.replay) < self.config.min_replay_size:
            return
        batch = self.replay.sample(self.config.batch_size, self.rng)
        states = torch.tensor(np.stack([transition.state for transition in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([transition.action for transition in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor([transition.reward for transition in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(
            np.stack([transition.next_state for transition in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor([transition.done for transition in batch], dtype=torch.float32, device=self.device)
        current_q = self.online_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            if self.config.trainer == "double_dqn":
                next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(dim=1).values
            targets = rewards + self.config.gamma * (1.0 - dones) * next_q
        loss = F.smooth_l1_loss(current_q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.last_loss = float(loss.item())
        self.loss_history.append(self.last_loss)
        self.update_count += 1
        if self.global_step % self.config.target_update_steps == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def training_step(self) -> None:
        self.inspect(self.current_state)
        action, action_mode = self._select_action()
        next_state, reward, done, info = self.env.step(action)
        self.replay.add(
            Transition(
                state=self.current_state.copy(),
                action=action,
                reward=float(reward),
                next_state=next_state.copy(),
                done=bool(done),
            )
        )
        self.last_action_index = action
        self.last_action_mode = action_mode
        self.last_reward = float(reward)
        self.last_info = info
        self.current_episode_reward += float(reward)
        self.current_episode_steps += 1
        self.global_step += 1
        self._optimize()
        self.current_state = next_state
        if done:
            score = self.env.forager_score if self.config.train_side == "forager" else self.env.raider_score
            self.recent_rewards.append(self.current_episode_reward)
            self.recent_lengths.append(self.current_episode_steps)
            self.recent_scores.append(float(score))
            self.reward_history.append(float(np.mean(self.recent_rewards)))
            self.score_history.append(float(np.mean(self.recent_scores)))
            self.current_state = self.env.reset()
            self.current_episode_reward = 0.0
            self.current_episode_steps = 0
            self.episode_index += 1
            self.inspect(self.current_state)

    def stat_lines(self) -> list[str]:
        avg_reward = float(np.mean(self.recent_rewards)) if self.recent_rewards else 0.0
        avg_length = float(np.mean(self.recent_lengths)) if self.recent_lengths else 0.0
        avg_score = float(np.mean(self.recent_scores)) if self.recent_scores else 0.0
        loss_text = "n/a" if self.last_loss is None else f"{self.last_loss:.4f}"
        return [
            f"trainer={self.config.trainer} | side={self.config.train_side} | hidden={format_hidden_layers(self.config.hidden_layers)} | activation={self.config.activation}",
            f"episode={self.episode_index} | global_step={self.global_step} | epsilon={self.epsilon():.3f} | replay={len(self.replay)}/{self.config.replay_size}",
            f"last_reward={self.last_reward:.3f} | avg_reward={avg_reward:.3f} | avg_score={avg_score:.2f} | avg_len={avg_length:.1f}",
            f"last_loss={loss_text} | last_action={ACTIONS[self.last_action_index]} ({self.last_action_mode})",
            f"event={self.last_info.get('event', self.env.last_event)}",
        ]

    def snapshot(self) -> dict[str, object]:
        return {
            "env": self.env.snapshot(),
            "network": self.inspect(self.current_state),
            "stats": self.stat_lines(),
        }


class ForagerRaiderApp:
    def __init__(self, root: tk.Tk, initial_config: AppConfig) -> None:
        self.root = root
        self.root.title("Forager vs Raider Deep RL")
        self.root.geometry("1760x960")
        self.root.minsize(1440, 800)
        self.root.configure(bg="#edf1f3")
        self.running = False
        self._tick_job: str | None = None
        self.trainer: DQNTrainer | None = None
        self.status_var = tk.StringVar(value="Ready")
        self.stats_var = tk.StringVar(value="")
        self.network_var = tk.StringVar(value="")
        self.vars: dict[str, tk.Variable] = {}
        self._build_layout()
        self._set_vars_from_config(initial_config)
        self._reset_trainer(run=False)

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        panes = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        panes.grid(row=0, column=0, sticky="nsew")
        left = ttk.Frame(panes, padding=(14, 14, 10, 14), width=330)
        center = ttk.Frame(panes, padding=(8, 14, 8, 14), width=820)
        right = ttk.Frame(panes, padding=(10, 14, 14, 14), width=560)
        panes.add(left, weight=1)
        panes.add(center, weight=4)
        panes.add(right, weight=3)
        center.columnconfigure(0, weight=1)
        center.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        self._build_settings_panel(left)
        ttk.Label(center, text="Arena", font=("Segoe UI Semibold", 16)).grid(row=0, column=0, sticky="w")
        self.arena_canvas = tk.Canvas(center, bg="#f6fafb", highlightthickness=0, width=760, height=760)
        self.arena_canvas.grid(row=1, column=0, sticky="nsew", pady=(8, 8))
        ttk.Label(center, textvariable=self.stats_var, justify="left", wraplength=760).grid(row=2, column=0, sticky="ew")
        ttk.Label(right, text="Network Panel", font=("Segoe UI Semibold", 16)).grid(row=0, column=0, sticky="w")
        self.network_canvas = tk.Canvas(right, bg="#f8faf9", highlightthickness=0, width=560, height=760)
        self.network_canvas.grid(row=1, column=0, sticky="nsew", pady=(8, 8))
        ttk.Label(right, textvariable=self.network_var, justify="left", wraplength=540).grid(row=2, column=0, sticky="ew")
        ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
            padding=(10, 5),
        ).grid(row=1, column=0, sticky="ew")
        self.arena_canvas.bind("<Configure>", lambda _event: self._draw_world())
        self.network_canvas.bind("<Configure>", lambda _event: self._draw_network())

    def _build_settings_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Controls", font=("Segoe UI Semibold", 16)).pack(anchor="w")
        scenario = ttk.LabelFrame(parent, text="Scenario", padding=10)
        brain = ttk.LabelFrame(parent, text="Brain", padding=10)
        learning = ttk.LabelFrame(parent, text="Learning", padding=10)
        display = ttk.LabelFrame(parent, text="Display", padding=10)
        for frame in (scenario, brain, learning, display):
            frame.pack(fill="x", pady=(8, 0))
        self._add_combo(scenario, "trainer", "Algorithm", ("double_dqn", "dqn"))
        self._add_combo(scenario, "train_side", "Train side", ("forager", "raider"))
        self._add_entry(scenario, "grid_size", "Grid size")
        self._add_entry(scenario, "food_count", "Food items")
        self._add_entry(scenario, "episode_steps", "Episode steps")
        self._add_entry(scenario, "seed", "Seed (blank=random)")
        self._add_entry(brain, "hidden_layers", "Hidden layers")
        self._add_combo(brain, "activation", "Activation", ACTIVATIONS)
        self._add_entry(learning, "learning_rate", "Learning rate")
        self._add_entry(learning, "weight_decay", "Weight decay")
        self._add_entry(learning, "gamma", "Gamma")
        self._add_entry(learning, "batch_size", "Batch size")
        self._add_entry(learning, "replay_size", "Replay size")
        self._add_entry(learning, "min_replay_size", "Warmup replay")
        self._add_entry(learning, "target_update_steps", "Target update")
        self._add_entry(learning, "epsilon_start", "Epsilon start")
        self._add_entry(learning, "epsilon_end", "Epsilon end")
        self._add_entry(learning, "epsilon_decay_steps", "Epsilon decay")
        self._add_entry(display, "steps_per_tick", "Steps / tick")
        self._add_entry(display, "tick_ms", "Tick ms")
        button_row = ttk.Frame(parent)
        button_row.pack(fill="x", pady=(12, 0))
        ttk.Button(button_row, text="Start / Restart", command=self._start).pack(fill="x")
        ttk.Button(button_row, text="Pause", command=self._pause).pack(fill="x", pady=(6, 0))
        ttk.Button(button_row, text="Resume", command=self._resume).pack(fill="x", pady=(6, 0))
        ttk.Button(button_row, text="Reset", command=lambda: self._reset_trainer(run=False)).pack(fill="x", pady=(6, 0))
        ttk.Label(
            parent,
            text=(
                "Adjust settings, then Start / Restart.\n"
                "The network panel shows current inputs, hidden activations, and Q-values."
            ),
            justify="left",
            wraplength=300,
        ).pack(fill="x", pady=(10, 0))

    def _add_entry(self, parent: ttk.LabelFrame, key: str, label: str) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=18).pack(side="left")
        var = tk.StringVar()
        self.vars[key] = var
        ttk.Entry(row, textvariable=var, width=16).pack(side="right")

    def _add_combo(self, parent: ttk.LabelFrame, key: str, label: str, values: tuple[str, ...]) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=18).pack(side="left")
        var = tk.StringVar()
        self.vars[key] = var
        ttk.Combobox(row, textvariable=var, values=values, state="readonly", width=13).pack(side="right")

    def _set_vars_from_config(self, config: AppConfig) -> None:
        values = {
            "trainer": config.trainer,
            "train_side": config.train_side,
            "grid_size": str(config.grid_size),
            "food_count": str(config.food_count),
            "episode_steps": str(config.episode_steps),
            "seed": "" if config.seed is None else str(config.seed),
            "hidden_layers": format_hidden_layers(config.hidden_layers),
            "activation": config.activation,
            "learning_rate": str(config.learning_rate),
            "weight_decay": str(config.weight_decay),
            "gamma": str(config.gamma),
            "batch_size": str(config.batch_size),
            "replay_size": str(config.replay_size),
            "min_replay_size": str(config.min_replay_size),
            "target_update_steps": str(config.target_update_steps),
            "epsilon_start": str(config.epsilon_start),
            "epsilon_end": str(config.epsilon_end),
            "epsilon_decay_steps": str(config.epsilon_decay_steps),
            "steps_per_tick": str(config.steps_per_tick),
            "tick_ms": str(config.tick_ms),
        }
        for key, value in values.items():
            self.vars[key].set(value)

    def _config_from_vars(self) -> AppConfig:
        seed_text = self.vars["seed"].get().strip()
        config = AppConfig(
            trainer=self.vars["trainer"].get().strip() or "double_dqn",
            train_side=self.vars["train_side"].get().strip() or "forager",
            grid_size=int(self.vars["grid_size"].get().strip()),
            food_count=int(self.vars["food_count"].get().strip()),
            episode_steps=int(self.vars["episode_steps"].get().strip()),
            hidden_layers=parse_hidden_layers(self.vars["hidden_layers"].get().strip()),
            activation=self.vars["activation"].get().strip() or "relu",
            learning_rate=float(self.vars["learning_rate"].get().strip()),
            weight_decay=float(self.vars["weight_decay"].get().strip()),
            gamma=float(self.vars["gamma"].get().strip()),
            batch_size=int(self.vars["batch_size"].get().strip()),
            replay_size=int(self.vars["replay_size"].get().strip()),
            min_replay_size=int(self.vars["min_replay_size"].get().strip()),
            target_update_steps=int(self.vars["target_update_steps"].get().strip()),
            epsilon_start=float(self.vars["epsilon_start"].get().strip()),
            epsilon_end=float(self.vars["epsilon_end"].get().strip()),
            epsilon_decay_steps=int(self.vars["epsilon_decay_steps"].get().strip()),
            steps_per_tick=int(self.vars["steps_per_tick"].get().strip()),
            tick_ms=int(self.vars["tick_ms"].get().strip()),
            seed=None if not seed_text else int(seed_text),
        )
        validate_config(config)
        return config

    def _start(self) -> None:
        self._reset_trainer(run=True)

    def _pause(self) -> None:
        self.running = False
        self.status_var.set("Paused")
        if self._tick_job is not None:
            self.root.after_cancel(self._tick_job)
            self._tick_job = None

    def _resume(self) -> None:
        if self.trainer is None:
            self._start()
            return
        if self.running:
            return
        self.running = True
        self.status_var.set("Running")
        self._schedule_tick()

    def _reset_trainer(self, run: bool) -> None:
        try:
            config = self._config_from_vars()
            self.trainer = DQNTrainer(config)
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            self.status_var.set("Settings error")
            return
        self.running = run
        if self._tick_job is not None:
            self.root.after_cancel(self._tick_job)
            self._tick_job = None
        self.status_var.set("Running" if run else "Ready")
        self._refresh_views()
        if run:
            self._schedule_tick()

    def _schedule_tick(self) -> None:
        if not self.running or self.trainer is None:
            return
        self._run_training_tick()
        self._tick_job = self.root.after(self.trainer.config.tick_ms, self._schedule_tick)

    def _run_training_tick(self) -> None:
        if self.trainer is None:
            return
        for _ in range(self.trainer.config.steps_per_tick):
            self.trainer.training_step()
        self._refresh_views()

    def _refresh_views(self) -> None:
        if self.trainer is None:
            return
        snapshot = self.trainer.snapshot()
        self.stats_var.set("\n".join(snapshot["stats"]))
        network = snapshot["network"]
        selected = int(network["selected_action"])
        self.network_var.set(
            "\n".join(
                [
                    f"Inputs: {len(network['input_values'])} | Hidden: {format_hidden_layers(self.trainer.config.hidden_layers)}",
                    f"Chosen action: {ACTIONS[self.trainer.last_action_index]} ({self.trainer.last_action_mode})",
                    f"Greedy action: {ACTIONS[selected]}",
                    f"Q-range: {float(np.min(network['output_values'])):.3f} to {float(np.max(network['output_values'])):.3f}",
                ]
            )
        )
        self._draw_world(snapshot["env"])
        self._draw_network(network)

    def _draw_world(self, env_snapshot: dict[str, object] | None = None) -> None:
        if self.trainer is None:
            return
        snapshot = env_snapshot or self.trainer.env.snapshot()
        canvas = self.arena_canvas
        canvas.delete("all")
        width = max(200, canvas.winfo_width())
        height = max(200, canvas.winfo_height())
        grid_size = int(snapshot["grid_size"])
        margin = 44
        usable = min(width - 2 * margin, height - 2 * margin)
        cell = usable / grid_size
        left = (width - cell * grid_size) / 2
        top = (height - cell * grid_size) / 2
        canvas.create_rectangle(left, top, left + cell * grid_size, top + cell * grid_size, fill="#ffffff", outline="#c9d4d8")
        for index in range(grid_size + 1):
            x = left + index * cell
            y = top + index * cell
            canvas.create_line(x, top, x, top + cell * grid_size, fill="#d8e3e6")
            canvas.create_line(left, y, left + cell * grid_size, y, fill="#d8e3e6")
        self._draw_cell(canvas, snapshot["forager_base"], left, top, cell, "#dcefff", "F Base")
        self._draw_cell(canvas, snapshot["raider_base"], left, top, cell, "#ffe3df", "R Base")
        for food in snapshot["food_positions"]:
            x0, y0, x1, y1 = self._cell_bounds(food, left, top, cell)
            canvas.create_rectangle(x0 + 10, y0 + 10, x1 - 10, y1 - 10, fill="#f2c84b", outline="#c4961d", width=2)
        self._draw_agent(
            canvas,
            snapshot["forager_position"],
            left,
            top,
            cell,
            fill="#4c91ff",
            label="Forager",
            carrying=bool(snapshot["forager_carry"]),
        )
        self._draw_agent(
            canvas,
            snapshot["raider_position"],
            left,
            top,
            cell,
            fill="#ff7859",
            label="Raider",
            carrying=bool(snapshot["raider_carry"]),
        )
        canvas.create_text(
            width / 2,
            20,
            text=(
                f"Forager {snapshot['forager_score']}  |  Raider {snapshot['raider_score']}  |  "
                f"Step {snapshot['step_index']}/{snapshot['episode_steps']}"
            ),
            font=("Segoe UI Semibold", 14),
            fill="#294045",
        )
        canvas.create_text(
            width / 2,
            height - 18,
            text=snapshot["event"],
            font=("Segoe UI", 11),
            fill="#4c5d62",
            width=width - 40,
        )

    def _cell_bounds(self, position: tuple[int, int], left: float, top: float, cell: float) -> tuple[float, float, float, float]:
        x0 = left + position[0] * cell
        y0 = top + position[1] * cell
        return x0, y0, x0 + cell, y0 + cell

    def _draw_cell(
        self,
        canvas: tk.Canvas,
        position: tuple[int, int],
        left: float,
        top: float,
        cell: float,
        fill: str,
        label: str,
    ) -> None:
        x0, y0, x1, y1 = self._cell_bounds(position, left, top, cell)
        canvas.create_rectangle(x0 + 4, y0 + 4, x1 - 4, y1 - 4, fill=fill, outline="#869aa0", width=2)
        canvas.create_text((x0 + x1) / 2, y0 + 12, text=label, font=("Segoe UI", 9), fill="#405055")

    def _draw_agent(
        self,
        canvas: tk.Canvas,
        position: tuple[int, int],
        left: float,
        top: float,
        cell: float,
        fill: str,
        label: str,
        carrying: bool,
    ) -> None:
        x0, y0, x1, y1 = self._cell_bounds(position, left, top, cell)
        canvas.create_oval(x0 + 8, y0 + 8, x1 - 8, y1 - 8, fill=fill, outline="#20343a", width=2)
        canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=label[0], font=("Segoe UI Semibold", 16), fill="white")
        if carrying:
            canvas.create_oval(x1 - 18, y0 + 6, x1 - 6, y0 + 18, fill="#fff7d6", outline="#b28b00", width=2)

    def _draw_network(self, network_snapshot: dict[str, object] | None = None) -> None:
        if self.trainer is None:
            return
        network = network_snapshot or self.trainer.snapshot()["network"]
        canvas = self.network_canvas
        canvas.delete("all")
        width = max(260, canvas.winfo_width())
        height = max(260, canvas.winfo_height())
        input_values = list(network["input_values"])
        hidden_values = [list(layer) for layer in network["hidden_values"]]
        output_values = list(network["output_values"])
        layers = [input_values] + hidden_values + [output_values]
        layer_names = ["Inputs"] + [f"H{i + 1}" for i in range(len(hidden_values))] + ["Q"]
        weights = network["weight_matrices"]
        x_positions = np.linspace(54, width - 54, num=len(layers))
        node_positions: list[list[tuple[float, float]]] = []
        for layer_index, layer_values in enumerate(layers):
            layer_height = max(1, len(layer_values))
            top_margin = 70
            bottom_margin = 54
            if layer_height == 1:
                ys = [height / 2]
            else:
                ys = np.linspace(top_margin, height - bottom_margin, num=layer_height)
            radius = 12 if len(layer_values) <= 16 else 9 if len(layer_values) <= 24 else 7
            positions: list[tuple[float, float]] = []
            for y in ys:
                positions.append((float(x_positions[layer_index]), float(y)))
            node_positions.append(positions)
            canvas.create_text(
                x_positions[layer_index],
                24,
                text=f"{layer_names[layer_index]} ({len(layer_values)})",
                font=("Segoe UI Semibold", 11),
                fill="#2f464d",
            )
            for node_index, value in enumerate(layer_values):
                x, y = positions[node_index]
                fill = color_for_value(float(value))
                canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline="#304247", width=1)
                if layer_index == 0:
                    label = OBSERVATION_NAMES[node_index]
                    canvas.create_text(x - 18, y, text=label, anchor="e", font=("Consolas", 9), fill="#4a5f65")
                    canvas.create_text(x + 18, y, text=f"{value:+.2f}", anchor="w", font=("Consolas", 9), fill="#22383e")
                elif layer_index == len(layers) - 1:
                    label = ACTIONS[node_index]
                    mark = " *" if node_index == self.trainer.last_action_index else ""
                    canvas.create_text(x + 18, y - 7, text=label + mark, anchor="w", font=("Segoe UI", 9), fill="#4a5f65")
                    canvas.create_text(x + 18, y + 7, text=f"{value:+.2f}", anchor="w", font=("Consolas", 10), fill="#1f3338")
                elif len(layer_values) <= 20:
                    canvas.create_text(x + 16, y, text=f"{value:+.2f}", anchor="w", font=("Consolas", 8), fill="#21383e")
        total_edges = sum(len(layers[index]) * len(layers[index + 1]) for index in range(len(layers) - 1))
        edge_budget = 240
        edge_specs: list[tuple[float, float, float, float, float, float]] = []
        total_targets = sum(len(layer) for layer in layers[1:])
        per_target_budget = max(2, min(6, edge_budget // max(1, total_targets)))
        for layer_index, matrix in enumerate(weights):
            scaled = np.tanh(matrix)
            source_positions = node_positions[layer_index]
            target_positions = node_positions[layer_index + 1]
            for target_index, target_xy in enumerate(target_positions):
                row = scaled[target_index]
                if total_edges <= edge_budget:
                    selected_indices = range(len(source_positions))
                else:
                    ranked = np.argsort(np.abs(row))
                    selected_indices = ranked[-per_target_budget:]
                for source_index in selected_indices:
                    weight = float(row[source_index])
                    if total_edges > edge_budget and abs(weight) < 0.08:
                        continue
                    source_xy = source_positions[int(source_index)]
                    edge_specs.append(
                        (
                            abs(weight),
                            source_xy[0] + 10,
                            source_xy[1],
                            target_xy[0] - 10,
                            target_xy[1],
                            weight,
                        )
                    )
        if len(edge_specs) > edge_budget:
            edge_specs.sort(key=lambda item: item[0], reverse=True)
            edge_specs = edge_specs[:edge_budget]
        for _strength, x0, y0, x1, y1, weight in edge_specs:
            canvas.create_line(
                x0,
                y0,
                x1,
                y1,
                fill=edge_color(weight),
                width=1.0 + 1.2 * abs(weight),
                tags=("edge",),
            )
        canvas.tag_lower("edge")
        canvas.create_text(
            width / 2,
            height - 18,
            text=f"activation={self.trainer.config.activation} | greedy={ACTIONS[int(network['selected_action'])]}",
            font=("Segoe UI", 10),
            fill="#42575d",
        )


def validate_config(config: AppConfig) -> None:
    if config.trainer not in {"dqn", "double_dqn"}:
        raise ValueError("trainer must be dqn or double_dqn")
    if config.train_side not in {"forager", "raider"}:
        raise ValueError("train_side must be forager or raider")
    if config.grid_size < 5:
        raise ValueError("grid_size must be at least 5")
    if config.food_count < 1:
        raise ValueError("food_count must be at least 1")
    capacity = (config.grid_size - 2) * config.grid_size - 2
    if config.food_count > capacity:
        raise ValueError("food_count is too large for this grid")
    if config.episode_steps < 10:
        raise ValueError("episode_steps must be at least 10")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be greater than 0")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay cannot be negative")
    if not 0.0 < config.gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if config.replay_size < config.batch_size:
        raise ValueError("replay_size must be at least batch_size")
    if config.min_replay_size < config.batch_size:
        raise ValueError("min_replay_size must be at least batch_size")
    if config.min_replay_size > config.replay_size:
        raise ValueError("min_replay_size cannot exceed replay_size")
    if config.target_update_steps < 1:
        raise ValueError("target_update_steps must be at least 1")
    if not 0.0 <= config.epsilon_end <= config.epsilon_start <= 1.0:
        raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
    if config.epsilon_decay_steps < 1:
        raise ValueError("epsilon_decay_steps must be at least 1")
    if config.steps_per_tick < 1:
        raise ValueError("steps_per_tick must be at least 1")
    if config.tick_ms < 1:
        raise ValueError("tick_ms must be at least 1")


def default_config() -> AppConfig:
    return AppConfig(
        trainer="double_dqn",
        train_side="forager",
        grid_size=9,
        food_count=6,
        episode_steps=90,
        hidden_layers=(32, 16),
        activation="relu",
        learning_rate=0.0012,
        weight_decay=0.0001,
        gamma=0.985,
        batch_size=64,
        replay_size=12000,
        min_replay_size=640,
        target_update_steps=120,
        epsilon_start=1.0,
        epsilon_end=0.08,
        epsilon_decay_steps=7000,
        steps_per_tick=24,
        tick_ms=45,
        seed=None,
    )


def config_from_args(args: argparse.Namespace) -> AppConfig:
    config = AppConfig(
        trainer=args.trainer,
        train_side=args.train_side,
        grid_size=args.grid_size,
        food_count=args.food_count,
        episode_steps=args.episode_steps,
        hidden_layers=args.hidden,
        activation=args.activation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        min_replay_size=args.min_replay_size,
        target_update_steps=args.target_update_steps,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        steps_per_tick=args.steps_per_tick,
        tick_ms=args.tick_ms,
        seed=args.seed,
        smoke_test_steps=args.smoke_test_steps,
    )
    validate_config(config)
    return config


def run_smoke_test(config: AppConfig) -> None:
    trainer = DQNTrainer(config)
    total_steps = max(1, config.smoke_test_steps)
    while trainer.global_step < total_steps:
        trainer.training_step()
    summary = trainer.snapshot()
    print("Smoke test complete")
    print(f"steps={trainer.global_step}")
    print(f"episodes={trainer.episode_index}")
    print(f"epsilon={trainer.epsilon():.3f}")
    print(f"replay={len(trainer.replay)}")
    print(f"last_event={summary['env']['event']}")
    print(f"last_action={ACTIONS[trainer.last_action_index]}")
    print(f"q_values={np.array2string(trainer.last_q_values, precision=3)}")


def build_parser() -> argparse.ArgumentParser:
    defaults = default_config()
    parser = argparse.ArgumentParser(description="Deep RL Forager vs Raider demo with a Tkinter control panel.")
    parser.add_argument("--trainer", choices=["dqn", "double_dqn"], default=defaults.trainer)
    parser.add_argument("--train-side", choices=["forager", "raider"], default=defaults.train_side)
    parser.add_argument("--grid-size", type=int, default=defaults.grid_size)
    parser.add_argument("--food-count", type=int, default=defaults.food_count)
    parser.add_argument("--episode-steps", type=int, default=defaults.episode_steps)
    parser.add_argument("--hidden", type=parse_hidden_layers, default=defaults.hidden_layers)
    parser.add_argument("--activation", choices=ACTIVATIONS, default=defaults.activation)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--replay-size", type=int, default=defaults.replay_size)
    parser.add_argument("--min-replay-size", type=int, default=defaults.min_replay_size)
    parser.add_argument("--target-update-steps", type=int, default=defaults.target_update_steps)
    parser.add_argument("--epsilon-start", type=float, default=defaults.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=defaults.epsilon_end)
    parser.add_argument("--epsilon-decay-steps", type=int, default=defaults.epsilon_decay_steps)
    parser.add_argument("--steps-per-tick", type=int, default=defaults.steps_per_tick)
    parser.add_argument("--tick-ms", type=int, default=defaults.tick_ms)
    parser.add_argument("--seed", type=int, default=defaults.seed, help="Leave unset for a fresh random run.")
    parser.add_argument("--smoke-test-steps", type=int, default=0, help="Run headless training steps instead of the GUI.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = config_from_args(args)
    except Exception as exc:
        parser.error(str(exc))
        return
    if config.smoke_test_steps > 0:
        run_smoke_test(config)
        return
    root = tk.Tk()
    app = ForagerRaiderApp(root, config)
    root.protocol("WM_DELETE_WINDOW", lambda: (app._pause(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
