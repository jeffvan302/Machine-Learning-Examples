from __future__ import annotations

import argparse
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np

import rocket_landing_rl_demo as rocket

try:
    from PIL import Image, ImageFilter, ImageTk
except ImportError:  # pragma: no cover - optional runtime dependency
    Image = None
    ImageFilter = None
    ImageTk = None


torch = rocket.torch
nn = rocket.nn

INPUT_LABELS = (
    "x",
    "y",
    "dx",
    "vx",
    "vy",
    "sin(a)",
    "cos(a)",
    "angle",
    "ang_vel",
    "fuel",
    "speed",
    "on_pad",
    "step",
)
OUTPUT_LABELS = ("turn_mu", "throttle_mu", "value")

SETTING_TOOLTIPS = {
    "generations": (
        "How many training generations to run before stopping.\n\n"
        "One generation means: collect a training batch, update the brain, then evaluate it.\n"
        "Higher values give the policy more time to improve, but the full run takes longer."
    ),
    "trainer": (
        "Which deep-RL algorithm to use.\n\n"
        "PPO is the recommended default because it is usually more stable.\n"
        "REINFORCE is simpler and easier to explain, but it is noisier and often learns more slowly."
    ),
    "batch_episodes": (
        "How many training episodes are collected before each optimizer update.\n\n"
        "Higher values make the gradient estimate smoother and less noisy, but each generation becomes slower."
    ),
    "rollouts": (
        "How many evaluation episodes are used to compute the generation metrics and landing rate.\n\n"
        "Higher values make the reported success rate more trustworthy, but each generation takes longer to score."
    ),
    "episode_steps": (
        "Maximum simulation steps in a single landing attempt.\n\n"
        "Higher values give the rocket more time to recover from a bad start, but also make each episode longer."
    ),
    "spawn_mode": (
        "Where the rocket starts relative to the landing pad.\n\n"
        "random: anywhere in the sky.\n"
        "side: mostly to the left or right.\n"
        "centered: closer to the pad and easier to learn from."
    ),
    "spawn_randomness": (
        "How dramatic the starting conditions are.\n\n"
        "dramatic uses wider variation in altitude, drift, and angle.\n"
        "standard is narrower and usually easier for training and presentations."
    ),
    "seed": (
        "Random seed for reproducibility.\n\n"
        "Leave this blank for a fresh run every time.\n"
        "Set a number when you want the same training sequence and the same random starts again."
    ),
    "hidden_layers": (
        "Hidden-layer layout for the rocket brain.\n\n"
        "Examples: 10,40,6 or 32,16.\n"
        "More neurons can represent richer behavior, but also make training slower and sometimes harder to tune.\n"
        "Use 0 for a linear policy with no hidden layer."
    ),
    "activation": (
        "Activation function used in the hidden layers.\n\n"
        "relu is a strong default.\n"
        "tanh is smoother and often easier to interpret.\n"
        "sigmoid compresses heavily and can make the network feel more conservative."
    ),
    "learning_rate": (
        "AdamW learning rate.\n\n"
        "This controls how big each optimizer step is.\n"
        "Too high can make training unstable. Too low can make progress very slow."
    ),
    "weight_decay": (
        "AdamW weight decay regularization.\n\n"
        "This gently discourages very large weights, which can help keep the network smoother and less erratic."
    ),
    "gamma": (
        "Discount factor for future rewards.\n\n"
        "Values closer to 1.0 make the rocket care more about long-term outcomes instead of only immediate reward."
    ),
    "gae_lambda": (
        "Generalized Advantage Estimation smoothing for PPO.\n\n"
        "Higher values usually reduce bias and use more long-range information, but can add variance.\n"
        "0.95 is a very common compromise."
    ),
    "entropy_coef": (
        "Exploration bonus strength.\n\n"
        "Higher values encourage the policy to stay more random for longer.\n"
        "Too much can prevent the rocket from settling into precise landing behavior."
    ),
    "value_coef": (
        "Weight for the value-head loss.\n\n"
        "This controls how strongly the critic is trained compared with the policy itself."
    ),
    "ppo_clip": (
        "PPO clipping range.\n\n"
        "Smaller values make updates more conservative.\n"
        "Larger values allow bigger policy changes, which can be faster but less stable."
    ),
    "ppo_epochs": (
        "How many times PPO reuses the same collected batch.\n\n"
        "More epochs squeeze more learning out of each batch, but can also overfit that batch."
    ),
    "minibatch_size": (
        "How many sampled states are used per optimizer step inside PPO.\n\n"
        "Larger minibatches give smoother updates. Smaller minibatches add more noise but may explore more."
    ),
    "gravity": (
        "Downward acceleration applied every step.\n\n"
        "Higher gravity makes the rocket fall faster and increases the difficulty of recovery."
    ),
    "main_thrust": (
        "Strength of the main engine.\n\n"
        "Higher thrust makes it easier to fight gravity, but can also make throttle control more sensitive."
    ),
    "turn_power": (
        "How strongly the turn command changes the rocket angle each step.\n\n"
        "Higher values make steering quicker but also easier to overcorrect."
    ),
    "fuel_burn": (
        "Fuel consumed by full throttle for one step.\n\n"
        "Higher values make fuel more expensive and encourage more efficient landing behavior."
    ),
    "pad_width": (
        "Width of the landing pad.\n\n"
        "A wider pad makes the task easier because the rocket has more horizontal room for a safe touchdown."
    ),
    "landing_vx": (
        "Maximum safe horizontal landing speed.\n\n"
        "Higher values make sideways motion more forgiving at touchdown."
    ),
    "landing_vy": (
        "Maximum safe vertical landing speed.\n\n"
        "Higher values make hard downward impacts more forgiving."
    ),
    "landing_angle": (
        "Maximum safe tilt angle at touchdown, in radians.\n\n"
        "Higher values allow the rocket to land at a larger tilt instead of needing to be nearly upright."
    ),
    "reward_progress_scale": (
        "Multiplier on the improvement in landing cost from one step to the next.\n\n"
        "Higher values reward progress toward a safe landing more aggressively."
    ),
    "reward_step_penalty": (
        "Small base penalty paid every simulation step.\n\n"
        "Higher values encourage faster landings and punish drifting around for too long."
    ),
    "reward_throttle_penalty": (
        "Extra penalty applied in proportion to throttle use.\n\n"
        "Higher values encourage fuel-efficient behavior and lighter engine use."
    ),
    "reward_turn_penalty": (
        "Extra penalty applied in proportion to steering effort.\n\n"
        "Higher values discourage twitchy turning and favor smoother control."
    ),
    "reward_out_of_bounds_penalty": (
        "Penalty when the rocket leaves the playable area.\n\n"
        "Higher values strongly punish flying off-screen or escaping upward."
    ),
    "reward_landing_bonus": (
        "Base reward for a successful landing.\n\n"
        "Higher values make touchdown itself much more important relative to shaping rewards."
    ),
    "reward_landing_fuel_bonus": (
        "Extra reward multiplier for fuel remaining after a successful landing.\n\n"
        "Higher values encourage efficient descents that conserve propellant."
    ),
    "reward_landing_precision_bonus": (
        "Maximum extra reward for a very clean, precise landing.\n\n"
        "This bonus is reduced by the landing cost at touchdown."
    ),
    "reward_landing_precision_scale": (
        "How strongly touchdown cost reduces the precision bonus.\n\n"
        "Higher values make the bonus fall off faster when the landing is less tidy."
    ),
    "reward_crash_penalty": (
        "Base penalty for a failed touchdown.\n\n"
        "Higher values punish crashes and missed-pad landings more severely."
    ),
    "reward_crash_impact_penalty": (
        "Extra crash penalty based on impact severity.\n\n"
        "Higher values punish fast or badly angled impacts more strongly."
    ),
    "reward_timeout_penalty": (
        "Penalty applied when an episode runs out of steps before landing.\n\n"
        "Higher values discourage hovering or wandering until the timer expires."
    ),
    "fps": (
        "Replay speed in frames per second.\n\n"
        "This only changes how fast the visualization plays. It does not change training."
    ),
}


class HoverTooltip:
    def __init__(self, widget: tk.Widget, text: str, wraplength: int = 460) -> None:
        self.widget = widget
        self.text = text
        self.wraplength = wraplength
        self.tipwindow: tk.Toplevel | None = None
        self.after_id: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Motion>", self._move, add="+")

    def _schedule(self, _event=None) -> None:
        self._unschedule()
        self.after_id = self.widget.after(260, self._show)

    def _unschedule(self) -> None:
        if self.after_id is not None:
            self.widget.after_cancel(self.after_id)
            self.after_id = None

    def _show(self) -> None:
        if self.tipwindow is not None:
            return
        self.tipwindow = tk.Toplevel(self.widget)
        self.tipwindow.wm_overrideredirect(True)
        try:
            self.tipwindow.wm_attributes("-topmost", True)
        except tk.TclError:
            pass
        frame = tk.Frame(self.tipwindow, background="#fff8dd", borderwidth=1, relief="solid")
        frame.pack(fill="both", expand=True)
        label = tk.Label(
            frame,
            text=self.text,
            justify="left",
            background="#fff8dd",
            foreground="#20343a",
            font=("Segoe UI", 11),
            wraplength=self.wraplength,
            padx=14,
            pady=12,
        )
        label.pack(fill="both", expand=True)
        self._position()

    def _position(self) -> None:
        if self.tipwindow is None:
            return
        x = self.widget.winfo_pointerx() + 18
        y = self.widget.winfo_pointery() + 18
        self.tipwindow.wm_geometry(f"+{x}+{y}")

    def _move(self, _event=None) -> None:
        if self.tipwindow is not None:
            self._position()

    def _hide(self, _event=None) -> None:
        self._unschedule()
        if self.tipwindow is not None:
            self.tipwindow.destroy()
            self.tipwindow = None


class RocketSpriteRenderer:
    def __init__(self, asset_path: Path) -> None:
        self.asset_path = asset_path
        self.available = Image is not None and ImageTk is not None and asset_path.exists()
        self.base_image = None
        self.cache: dict[tuple[int, int, str], object] = {}
        self.current_photo = None
        if self.available:
            self._load_image()

    def _load_image(self) -> None:
        assert Image is not None
        image = Image.open(self.asset_path).convert("RGBA")
        cleaned = self._remove_purple_background(image)
        bounds = cleaned.getbbox()
        if bounds is None:
            self.available = False
            return
        cropped = cleaned.crop(bounds)
        padding = max(36, int(max(cropped.size) * 0.18))
        padded = Image.new("RGBA", (cropped.width + padding * 2, cropped.height + padding * 2), (0, 0, 0, 0))
        padded.paste(cropped, (padding, padding), cropped)
        self.base_image = padded

    def _remove_purple_background(self, image):
        rgba = np.array(image.convert("RGBA"), dtype=np.uint8)
        corner_rgb = np.array(
            [
                rgba[0, 0, :3],
                rgba[0, -1, :3],
                rgba[-1, 0, :3],
                rgba[-1, -1, :3],
            ],
            dtype=np.float32,
        )
        key_color = np.median(corner_rgb, axis=0)
        rgb = rgba[:, :, :3].astype(np.float32)
        color_distance = np.linalg.norm(rgb - key_color, axis=2)
        purple_bias = (rgb[:, :, 2] > rgb[:, :, 1] + 26.0) & (rgb[:, :, 0] > rgb[:, :, 1] + 12.0)
        background = (color_distance < 58.0) | ((color_distance < 86.0) & purple_bias)
        feather = (~background) & (color_distance < 120.0) & purple_bias
        alpha = rgba[:, :, 3].astype(np.float32)
        alpha[background] = 0.0
        feather_strength = np.clip((color_distance[feather] - 58.0) / 62.0, 0.0, 1.0)
        alpha[feather] = np.minimum(alpha[feather], feather_strength * 255.0)
        rgba[:, :, 3] = alpha.astype(np.uint8)
        return Image.fromarray(rgba, "RGBA")

    def _apply_glow(self, image, glow_kind: str):
        assert Image is not None and ImageFilter is not None
        color_map = {
            "success": (110, 255, 140),
            "failure": (255, 92, 92),
            "neutral": (214, 233, 255),
        }
        alpha_scale = {
            "success": 0.92,
            "failure": 0.92,
            "neutral": 0.40,
        }
        alpha = image.getchannel("A")
        blur_radius = max(4, image.size[1] // 11)
        glow_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        glow_alpha = glow_alpha.point(lambda value: min(255, int(value * alpha_scale[glow_kind])))
        glow_layer = Image.new("RGBA", image.size, color_map[glow_kind] + (0,))
        glow_layer.putalpha(glow_alpha)
        return Image.alpha_composite(glow_layer, image)

    def render(self, pixel_height: int, angle_radians: float, glow_kind: str):
        if not self.available or self.base_image is None or Image is None or ImageTk is None:
            return None
        resampling = getattr(Image, "Resampling", Image)
        height_key = max(40, int(round(pixel_height / 4.0) * 4))
        angle_key = int(round(np.degrees(angle_radians) / 3.0) * 3)
        cache_key = (height_key, angle_key, glow_kind)
        if cache_key in self.cache:
            self.current_photo = self.cache[cache_key]
            return self.current_photo

        width = max(26, int(height_key * self.base_image.width / self.base_image.height))
        resized = self.base_image.resize((width, height_key), resample=resampling.LANCZOS)
        composited = self._apply_glow(resized, glow_kind)
        rotated = composited.rotate(-angle_key, resample=resampling.BICUBIC, expand=True)
        photo = ImageTk.PhotoImage(rotated)
        if len(self.cache) > 220:
            self.cache.clear()
        self.cache[cache_key] = photo
        self.current_photo = photo
        return photo


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


def format_hidden_layers(hidden_layers: tuple[int, ...]) -> str:
    return rocket.format_hidden_layers(hidden_layers)


@dataclass(slots=True)
class GuiConfig:
    generations: int
    trainer: str
    batch_episodes: int
    rollouts: int
    hidden_layers: tuple[int, ...]
    activation: str
    episode_steps: int
    seed: int | None
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
    smoke_test_generations: int = 0

    def to_demo_config(self) -> rocket.DemoConfig:
        return rocket.DemoConfig(
            generations=self.generations,
            trainer=self.trainer,
            population=16,
            batch_episodes=self.batch_episodes,
            rollouts=self.rollouts,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            episode_steps=self.episode_steps,
            seed=self.seed,
            mutation_scale=0.14,
            elite_fraction=0.24,
            inject_fraction=0.10,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef,
            ppo_clip=self.ppo_clip,
            ppo_epochs=self.ppo_epochs,
            minibatch_size=self.minibatch_size,
            gravity=self.gravity,
            main_thrust=self.main_thrust,
            turn_power=self.turn_power,
            fuel_burn=self.fuel_burn,
            pad_width=self.pad_width,
            landing_vx=self.landing_vx,
            landing_vy=self.landing_vy,
            landing_angle=self.landing_angle,
            reward_progress_scale=self.reward_progress_scale,
            reward_step_penalty=self.reward_step_penalty,
            reward_throttle_penalty=self.reward_throttle_penalty,
            reward_turn_penalty=self.reward_turn_penalty,
            reward_out_of_bounds_penalty=self.reward_out_of_bounds_penalty,
            reward_landing_bonus=self.reward_landing_bonus,
            reward_landing_fuel_bonus=self.reward_landing_fuel_bonus,
            reward_landing_precision_bonus=self.reward_landing_precision_bonus,
            reward_landing_precision_scale=self.reward_landing_precision_scale,
            reward_crash_penalty=self.reward_crash_penalty,
            reward_crash_impact_penalty=self.reward_crash_impact_penalty,
            reward_timeout_penalty=self.reward_timeout_penalty,
            spawn_mode=self.spawn_mode,
            spawn_randomness=self.spawn_randomness,
            fps=self.fps,
            no_gui=True,
        )


@dataclass(slots=True)
class VisualReplay:
    positions: np.ndarray
    angles: np.ndarray
    throttles: np.ndarray
    turn_commands: np.ndarray
    fuels: np.ndarray
    inputs: list[np.ndarray]
    hidden_values: list[list[np.ndarray]]
    outputs: list[np.ndarray]
    reward: float
    outcome: str
    landed: bool


@dataclass(slots=True)
class GenerationUpdate:
    generation: int
    best_reward: float
    mean_reward: float
    success_rate: float
    replay: VisualReplay
    elapsed_seconds: float
    last_loss: float | None


def policy_forward_with_activations(policy, inputs_tensor):
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for the rocket landing GUI.")

    features = inputs_tensor
    hidden_outputs: list[torch.Tensor] = []
    if not isinstance(policy.backbone, nn.Identity):
        for module in policy.backbone:
            features = module(features)
            if isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
                hidden_outputs.append(features)
    mean = policy.mean_head(features)
    value = policy.value_head(features).squeeze(-1)
    std = torch.exp(policy.log_std).clamp(0.08, 1.2).expand_as(mean)
    return mean, std, value, hidden_outputs


def policy_weight_matrices(policy) -> list[np.ndarray]:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for the rocket landing GUI.")

    matrices: list[np.ndarray] = []
    if not isinstance(policy.backbone, nn.Identity):
        for module in policy.backbone:
            if isinstance(module, nn.Linear):
                matrices.append(module.weight.detach().cpu().numpy())
    combined_output = np.vstack(
        [
            policy.mean_head.weight.detach().cpu().numpy(),
            policy.value_head.weight.detach().cpu().numpy(),
        ]
    )
    matrices.append(combined_output)
    return matrices


def run_visual_episode(policy, config: rocket.DemoConfig, rng: np.random.Generator) -> VisualReplay:
    if torch is None:
        raise RuntimeError("PyTorch is required for the rocket landing GUI.")

    state = rocket.spawn_state(rng, config)
    reward = 0.0
    outcome = "timed_out"
    landed = False
    previous_cost = rocket.landing_cost(state, config)

    positions: list[np.ndarray] = []
    angles: list[float] = []
    throttles: list[float] = []
    turn_commands: list[float] = []
    fuels: list[float] = []
    inputs: list[np.ndarray] = []
    hidden_values: list[list[np.ndarray]] = []
    outputs: list[np.ndarray] = []

    for step in range(config.episode_steps):
        step_fraction = step / max(config.episode_steps - 1, 1)
        observation = rocket.build_inputs(state, config, step_fraction)
        tensor = torch.from_numpy(observation).unsqueeze(0)
        with torch.inference_mode():
            mean, _std, value, hidden_layers = policy_forward_with_activations(policy, tensor)
        raw_action = mean.squeeze(0)
        turn_command = float(torch.tanh(raw_action[0]).item())
        throttle_request = float(torch.sigmoid(raw_action[1]).item())

        step_result = rocket.step_environment(state, turn_command, throttle_request, previous_cost, config)
        reward += step_result.reward
        previous_cost = step_result.next_cost

        positions.append(np.array([state.x, state.y], dtype=np.float64))
        angles.append(state.angle)
        throttles.append(step_result.throttle)
        turn_commands.append(turn_command)
        fuels.append(state.fuel)
        inputs.append(observation.astype(np.float32))
        hidden_values.append(
            [layer.squeeze(0).detach().cpu().numpy().astype(np.float32) for layer in hidden_layers]
        )
        outputs.append(
            np.array(
                [
                    float(mean[0, 0].item()),
                    float(mean[0, 1].item()),
                    float(value[0].item()),
                ],
                dtype=np.float32,
            )
        )

        if step_result.done:
            outcome = step_result.outcome or outcome
            landed = step_result.landed
            break
    else:
        reward -= config.reward_timeout_penalty

    return VisualReplay(
        positions=np.array(positions, dtype=np.float64),
        angles=np.array(angles, dtype=np.float64),
        throttles=np.array(throttles, dtype=np.float64),
        turn_commands=np.array(turn_commands, dtype=np.float64),
        fuels=np.array(fuels, dtype=np.float64),
        inputs=inputs,
        hidden_values=hidden_values,
        outputs=outputs,
        reward=reward,
        outcome=outcome,
        landed=landed,
    )


class GradientRocketTrainer:
    def __init__(self, gui_config: GuiConfig) -> None:
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for the rocket landing GUI.")

        self.gui_config = gui_config
        self.config = gui_config.to_demo_config()
        self.rng = np.random.default_rng(self.config.seed)
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
        self.policy = rocket.TorchPolicy(
            rocket.INPUT_DIM,
            self.config.hidden_layers,
            2,
            self.config.activation,
        )
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.history: list[GenerationUpdate] = []
        self.started = time.perf_counter()
        self.last_loss: float | None = None
        self.preview = self._build_preview()

    def _build_preview(self) -> GenerationUpdate:
        replay = self.sample_replay()
        return GenerationUpdate(
            generation=0,
            best_reward=replay.reward,
            mean_reward=replay.reward,
            success_rate=float(replay.landed),
            replay=replay,
            elapsed_seconds=0.0,
            last_loss=None,
        )

    def _train_reinforce_generation(self) -> None:
        batch_seed_values = [int(value) for value in self.rng.integers(0, 2_000_000_000, size=self.config.batch_episodes)]
        batch_episodes = []
        total_steps = 0

        for episode_seed in batch_seed_values:
            episode_rng = np.random.default_rng(episode_seed)
            episode = rocket.run_policy_episode(
                self.policy,
                self.config,
                episode_rng,
                record_episode=False,
                deterministic=False,
                requires_grad=True,
            )
            if episode["log_probs"]:
                batch_episodes.append(episode)
                total_steps += len(episode["log_probs"])

        self.last_loss = None
        if not batch_episodes:
            return

        return_tensors = [
            torch.tensor(rocket.discounted_returns(episode["step_rewards"], self.config.gamma), dtype=torch.float32)
            for episode in batch_episodes
        ]
        flattened_returns = torch.cat(return_tensors)
        if flattened_returns.numel() > 1:
            flattened_returns = (flattened_returns - flattened_returns.mean()) / (
                flattened_returns.std(unbiased=False) + 1e-6
            )

        self.optimizer.zero_grad()
        policy_loss = torch.tensor(0.0)
        value_loss = torch.tensor(0.0)
        entropy_bonus = torch.tensor(0.0)
        cursor = 0
        for episode_index, episode in enumerate(batch_episodes):
            episode_returns = flattened_returns[cursor : cursor + len(return_tensors[episode_index])]
            cursor += len(return_tensors[episode_index])
            log_prob_tensor = torch.stack(episode["log_probs"])
            entropy_tensor = torch.stack(episode["entropies"])
            value_tensor = torch.stack(episode["values"])
            advantages = episode_returns - value_tensor.detach()
            policy_loss = policy_loss - (log_prob_tensor * advantages).sum()
            value_loss = value_loss + torch.nn.functional.mse_loss(value_tensor, episode_returns, reduction="sum")
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
        batch_seed_values = [int(value) for value in self.rng.integers(0, 2_000_000_000, size=self.config.batch_episodes)]
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        old_log_probs: list[float] = []
        returns: list[float] = []
        advantages: list[float] = []

        for episode_seed in batch_seed_values:
            episode_rng = np.random.default_rng(episode_seed)
            episode = rocket.run_policy_episode(
                self.policy,
                self.config,
                episode_rng,
                record_episode=False,
                deterministic=False,
                requires_grad=False,
            )
            if not episode["observations"]:
                continue

            episode_advantages, episode_returns = rocket.generalized_advantages(
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

        for _epoch in range(self.config.ppo_epochs):
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
                value_loss = torch.nn.functional.mse_loss(values, batch_returns)
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.optimizer.step()
                losses.append(float(loss.item()))

        if losses:
            self.last_loss = float(np.mean(losses))

    def _evaluate_visual(self) -> tuple[float, float, float, VisualReplay]:
        eval_seed_values = [int(value) for value in self.rng.integers(0, 2_000_000_000, size=self.config.rollouts)]
        eval_rewards: list[float] = []
        eval_landings = 0.0
        showcase_seed = eval_seed_values[int(self.rng.integers(0, len(eval_seed_values)))]
        champion_record: VisualReplay | None = None

        for eval_seed in eval_seed_values:
            eval_rng = np.random.default_rng(eval_seed)
            if eval_seed == showcase_seed:
                evaluation = run_visual_episode(self.policy, self.config, eval_rng)
                champion_record = evaluation
                eval_rewards.append(float(evaluation.reward))
                eval_landings += float(evaluation.landed)
            else:
                evaluation = rocket.run_policy_episode(
                    self.policy,
                    self.config,
                    eval_rng,
                    record_episode=False,
                    deterministic=True,
                    requires_grad=False,
                )
                eval_rewards.append(float(evaluation["reward"]))
                eval_landings += float(evaluation["landed"])

        if champion_record is None:
            champion_record = run_visual_episode(self.policy, self.config, np.random.default_rng(showcase_seed))
        return (
            float(np.max(eval_rewards)),
            float(np.mean(eval_rewards)),
            float(eval_landings / max(len(eval_rewards), 1)),
            champion_record,
        )

    def train_generation(self) -> GenerationUpdate:
        if self.config.trainer == "reinforce":
            self._train_reinforce_generation()
        else:
            self._train_ppo_generation()

        best_reward, mean_reward, success_rate, replay = self._evaluate_visual()
        update = GenerationUpdate(
            generation=len(self.history) + 1,
            best_reward=best_reward,
            mean_reward=mean_reward,
            success_rate=success_rate,
            replay=replay,
            elapsed_seconds=time.perf_counter() - self.started,
            last_loss=self.last_loss,
        )
        self.history.append(update)
        return update

    def sample_replay(self) -> VisualReplay:
        replay_seed = int(self.rng.integers(0, 2_000_000_000))
        return run_visual_episode(self.policy, self.config, np.random.default_rng(replay_seed))


class RocketLandingGuiApp:
    def __init__(self, root: tk.Tk, initial_config: GuiConfig) -> None:
        self.root = root
        self.root.title("Rocket Landing Deep RL GUI")
        self.root.geometry("1820x980")
        self.root.minsize(1480, 820)
        self.root.configure(bg="#edf1f3")
        self.training_running = False
        self.replay_playing = True
        self.training_job: str | None = None
        self.replay_job: str | None = None
        self.trainer: GradientRocketTrainer | None = None
        self.current_update: GenerationUpdate | None = None
        self.live_replay: VisualReplay | None = None
        self.replay_step = 0
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.StringVar(value="Generation 0/0 | Landing rate 0.0%")
        self.stats_var = tk.StringVar(value="")
        self.network_var = tk.StringVar(value="")
        self.vars: dict[str, tk.Variable] = {}
        self.tooltips: list[HoverTooltip] = []
        self.sprite_renderer = RocketSpriteRenderer(Path(__file__).resolve().with_name("rocket.png"))
        self._build_layout()
        self._set_vars_from_config(initial_config)
        self._reset_trainer(run=False)
        self._schedule_replay()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        panes = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        panes.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(panes, padding=(10, 14, 6, 14), width=370)
        center = ttk.Frame(panes, padding=(8, 14, 8, 14), width=840)
        right = ttk.Frame(panes, padding=(10, 14, 14, 14), width=620)
        panes.add(left, weight=1)
        panes.add(center, weight=4)
        panes.add(right, weight=3)

        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)
        center.rowconfigure(1, weight=4)
        center.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        settings_canvas = tk.Canvas(left, bg="#edf1f3", highlightthickness=0, width=360)
        settings_canvas.grid(row=0, column=0, sticky="nsew")
        settings_scrollbar = ttk.Scrollbar(left, orient="vertical", command=settings_canvas.yview)
        settings_scrollbar.grid(row=0, column=1, sticky="ns")
        settings_canvas.configure(yscrollcommand=settings_scrollbar.set)

        settings_inner = ttk.Frame(settings_canvas, padding=(4, 0, 8, 0))
        settings_window = settings_canvas.create_window((0, 0), window=settings_inner, anchor="nw")

        def sync_settings_scroll(_event=None) -> None:
            settings_canvas.configure(scrollregion=settings_canvas.bbox("all"))

        def sync_settings_width(event) -> None:
            settings_canvas.itemconfigure(settings_window, width=event.width)

        def scroll_settings(event) -> str:
            delta = 0
            if hasattr(event, "delta") and event.delta:
                delta = -1 * int(event.delta / 120) if abs(event.delta) >= 120 else (-1 if event.delta > 0 else 1)
            elif getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            if delta:
                settings_canvas.yview_scroll(delta, "units")
            return "break"

        def bind_scroll_events(widget: tk.Widget) -> None:
            widget.bind("<MouseWheel>", scroll_settings, add="+")
            widget.bind("<Button-4>", scroll_settings, add="+")
            widget.bind("<Button-5>", scroll_settings, add="+")
            for child in widget.winfo_children():
                bind_scroll_events(child)

        settings_inner.bind("<Configure>", sync_settings_scroll)
        settings_canvas.bind("<Configure>", sync_settings_width)
        settings_canvas.bind("<MouseWheel>", scroll_settings)
        settings_inner.bind("<MouseWheel>", scroll_settings)

        self._build_settings_panel(settings_inner)
        bind_scroll_events(settings_inner)
        self.root.after_idle(sync_settings_scroll)

        header = ttk.Frame(center)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Rocket Replay", font=("Segoe UI Semibold", 18)).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            textvariable=self.progress_var,
            font=("Segoe UI Semibold", 18),
            foreground="#18363f",
        ).grid(row=0, column=1, sticky="e")
        self.world_canvas = tk.Canvas(center, bg="#f7faf9", highlightthickness=0, width=840, height=760)
        self.world_canvas.grid(row=1, column=0, sticky="nsew", pady=(8, 8))
        self.metrics_canvas = tk.Canvas(center, bg="#fbfcfb", highlightthickness=0, width=840, height=230)
        self.metrics_canvas.grid(row=2, column=0, sticky="nsew", pady=(0, 8))
        ttk.Label(center, textvariable=self.stats_var, justify="left", wraplength=820).grid(row=3, column=0, sticky="ew")

        ttk.Label(right, text="Network Panel", font=("Segoe UI Semibold", 16)).grid(row=0, column=0, sticky="w")
        self.network_canvas = tk.Canvas(right, bg="#f8faf9", highlightthickness=0, width=620, height=760)
        self.network_canvas.grid(row=1, column=0, sticky="nsew", pady=(8, 8))
        ttk.Label(right, textvariable=self.network_var, justify="left", wraplength=600).grid(row=2, column=0, sticky="ew")

        ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
            padding=(10, 5),
        ).grid(row=1, column=0, sticky="ew")

        self.world_canvas.bind("<Configure>", lambda _event: self._draw_world())
        self.metrics_canvas.bind("<Configure>", lambda _event: self._draw_metrics())
        self.network_canvas.bind("<Configure>", lambda _event: self._draw_network())

    def _build_settings_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Controls", font=("Segoe UI Semibold", 16)).pack(anchor="w")
        training = ttk.LabelFrame(parent, text="Training", padding=10)
        brain = ttk.LabelFrame(parent, text="Brain", padding=10)
        optimizer = ttk.LabelFrame(parent, text="Optimizer", padding=10)
        physics = ttk.LabelFrame(parent, text="Physics", padding=10)
        rewards = ttk.LabelFrame(parent, text="Rewards", padding=10)
        penalties = ttk.LabelFrame(parent, text="Penalties", padding=10)
        display = ttk.LabelFrame(parent, text="Display", padding=10)
        for frame in (training, brain, optimizer, physics, rewards, penalties, display):
            frame.pack(fill="x", pady=(8, 0))

        self._add_entry(training, "generations", "Generations")
        self._add_combo(training, "trainer", "Trainer", ("ppo", "reinforce"))
        self._add_entry(training, "batch_episodes", "Batch episodes")
        self._add_entry(training, "rollouts", "Eval rollouts")
        self._add_entry(training, "episode_steps", "Episode steps")
        self._add_combo(training, "spawn_mode", "Spawn mode", ("random", "side", "centered"))
        self._add_combo(training, "spawn_randomness", "Spawn randomness", ("dramatic", "standard"))
        self._add_entry(training, "seed", "Seed (blank=random)")

        self._add_entry(brain, "hidden_layers", "Hidden layers")
        self._add_combo(brain, "activation", "Activation", tuple(sorted(rocket.ACTIVATIONS)))

        self._add_entry(optimizer, "learning_rate", "Learning rate")
        self._add_entry(optimizer, "weight_decay", "Weight decay")
        self._add_entry(optimizer, "gamma", "Gamma")
        self._add_entry(optimizer, "gae_lambda", "GAE lambda")
        self._add_entry(optimizer, "entropy_coef", "Entropy coef")
        self._add_entry(optimizer, "value_coef", "Value coef")
        self._add_entry(optimizer, "ppo_clip", "PPO clip")
        self._add_entry(optimizer, "ppo_epochs", "PPO epochs")
        self._add_entry(optimizer, "minibatch_size", "Minibatch size")

        self._add_entry(physics, "gravity", "Gravity")
        self._add_entry(physics, "main_thrust", "Main thrust")
        self._add_entry(physics, "turn_power", "Turn power")
        self._add_entry(physics, "fuel_burn", "Fuel burn")
        self._add_entry(physics, "pad_width", "Pad width")
        self._add_entry(physics, "landing_vx", "Landing vx")
        self._add_entry(physics, "landing_vy", "Landing vy")
        self._add_entry(physics, "landing_angle", "Landing angle")

        self._add_entry(rewards, "reward_progress_scale", "Progress scale")
        self._add_entry(rewards, "reward_landing_bonus", "Landing bonus")
        self._add_entry(rewards, "reward_landing_fuel_bonus", "Fuel bonus")
        self._add_entry(rewards, "reward_landing_precision_bonus", "Precision bonus")
        self._add_entry(rewards, "reward_landing_precision_scale", "Precision scale")

        self._add_entry(penalties, "reward_step_penalty", "Step penalty")
        self._add_entry(penalties, "reward_throttle_penalty", "Throttle penalty")
        self._add_entry(penalties, "reward_turn_penalty", "Turn penalty")
        self._add_entry(penalties, "reward_out_of_bounds_penalty", "Out-of-bounds penalty")
        self._add_entry(penalties, "reward_crash_penalty", "Crash penalty")
        self._add_entry(penalties, "reward_crash_impact_penalty", "Impact penalty")
        self._add_entry(penalties, "reward_timeout_penalty", "Timeout penalty")

        self._add_entry(display, "fps", "Replay fps")
        button_row = ttk.Frame(parent)
        button_row.pack(fill="x", pady=(12, 0))
        ttk.Button(button_row, text="Start / Restart", command=self._start).pack(fill="x")
        ttk.Button(button_row, text="Pause Training", command=self._pause_training).pack(fill="x", pady=(6, 0))
        ttk.Button(button_row, text="Resume Training", command=self._resume_training).pack(fill="x", pady=(6, 0))
        ttk.Button(button_row, text="Pause Replay", command=self._toggle_replay).pack(fill="x", pady=(6, 0))
        ttk.Button(button_row, text="Reset", command=lambda: self._reset_trainer(run=False)).pack(fill="x", pady=(6, 0))
        ttk.Label(
            parent,
            text=(
                "Hover any setting label or value box for a larger detailed tooltip.\n"
                "This GUI focuses on the deep-RL rocket trainers.\n"
                "It starts with your PPO defaults and shows live network activations during replay.\n"
                "Use the Rewards and Penalties sections to tune what the rocket is trying to achieve and what mistakes cost it."
            ),
            justify="left",
            wraplength=320,
        ).pack(fill="x", pady=(10, 0))

    def _add_entry(self, parent: ttk.LabelFrame, key: str, label: str) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        label_widget = ttk.Label(row, text=label, width=18)
        label_widget.pack(side="left")
        var = tk.StringVar()
        self.vars[key] = var
        entry_widget = ttk.Entry(row, textvariable=var, width=16)
        entry_widget.pack(side="right")
        self._attach_tooltip_widgets(key, label_widget, entry_widget)

    def _add_combo(self, parent: ttk.LabelFrame, key: str, label: str, values: tuple[str, ...]) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        label_widget = ttk.Label(row, text=label, width=18)
        label_widget.pack(side="left")
        var = tk.StringVar()
        self.vars[key] = var
        combo_widget = ttk.Combobox(row, textvariable=var, values=values, state="readonly", width=13)
        combo_widget.pack(side="right")
        self._attach_tooltip_widgets(key, label_widget, combo_widget)

    def _attach_tooltip_widgets(self, key: str, *widgets: tk.Widget) -> None:
        tooltip_text = SETTING_TOOLTIPS.get(key)
        if not tooltip_text:
            return
        for widget in widgets:
            self.tooltips.append(HoverTooltip(widget, tooltip_text))

    def _set_vars_from_config(self, config: GuiConfig) -> None:
        values = {
            "generations": str(config.generations),
            "trainer": config.trainer,
            "batch_episodes": str(config.batch_episodes),
            "rollouts": str(config.rollouts),
            "episode_steps": str(config.episode_steps),
            "spawn_mode": config.spawn_mode,
            "spawn_randomness": config.spawn_randomness,
            "seed": "" if config.seed is None else str(config.seed),
            "hidden_layers": format_hidden_layers(config.hidden_layers),
            "activation": config.activation,
            "learning_rate": str(config.learning_rate),
            "weight_decay": str(config.weight_decay),
            "gamma": str(config.gamma),
            "gae_lambda": str(config.gae_lambda),
            "entropy_coef": str(config.entropy_coef),
            "value_coef": str(config.value_coef),
            "ppo_clip": str(config.ppo_clip),
            "ppo_epochs": str(config.ppo_epochs),
            "minibatch_size": str(config.minibatch_size),
            "gravity": str(config.gravity),
            "main_thrust": str(config.main_thrust),
            "turn_power": str(config.turn_power),
            "fuel_burn": str(config.fuel_burn),
            "pad_width": str(config.pad_width),
            "landing_vx": str(config.landing_vx),
            "landing_vy": str(config.landing_vy),
            "landing_angle": str(config.landing_angle),
            "reward_progress_scale": str(config.reward_progress_scale),
            "reward_step_penalty": str(config.reward_step_penalty),
            "reward_throttle_penalty": str(config.reward_throttle_penalty),
            "reward_turn_penalty": str(config.reward_turn_penalty),
            "reward_out_of_bounds_penalty": str(config.reward_out_of_bounds_penalty),
            "reward_landing_bonus": str(config.reward_landing_bonus),
            "reward_landing_fuel_bonus": str(config.reward_landing_fuel_bonus),
            "reward_landing_precision_bonus": str(config.reward_landing_precision_bonus),
            "reward_landing_precision_scale": str(config.reward_landing_precision_scale),
            "reward_crash_penalty": str(config.reward_crash_penalty),
            "reward_crash_impact_penalty": str(config.reward_crash_impact_penalty),
            "reward_timeout_penalty": str(config.reward_timeout_penalty),
            "fps": str(config.fps),
        }
        for key, value in values.items():
            self.vars[key].set(value)

    def _config_from_vars(self) -> GuiConfig:
        seed_text = self.vars["seed"].get().strip()
        config = GuiConfig(
            generations=int(self.vars["generations"].get().strip()),
            trainer=self.vars["trainer"].get().strip() or "ppo",
            batch_episodes=int(self.vars["batch_episodes"].get().strip()),
            rollouts=int(self.vars["rollouts"].get().strip()),
            hidden_layers=rocket.parse_hidden_layers(self.vars["hidden_layers"].get().strip()),
            activation=self.vars["activation"].get().strip() or "relu",
            episode_steps=int(self.vars["episode_steps"].get().strip()),
            seed=None if not seed_text else int(seed_text),
            learning_rate=float(self.vars["learning_rate"].get().strip()),
            weight_decay=float(self.vars["weight_decay"].get().strip()),
            gamma=float(self.vars["gamma"].get().strip()),
            gae_lambda=float(self.vars["gae_lambda"].get().strip()),
            entropy_coef=float(self.vars["entropy_coef"].get().strip()),
            value_coef=float(self.vars["value_coef"].get().strip()),
            ppo_clip=float(self.vars["ppo_clip"].get().strip()),
            ppo_epochs=int(self.vars["ppo_epochs"].get().strip()),
            minibatch_size=int(self.vars["minibatch_size"].get().strip()),
            gravity=float(self.vars["gravity"].get().strip()),
            main_thrust=float(self.vars["main_thrust"].get().strip()),
            turn_power=float(self.vars["turn_power"].get().strip()),
            fuel_burn=float(self.vars["fuel_burn"].get().strip()),
            pad_width=float(self.vars["pad_width"].get().strip()),
            landing_vx=float(self.vars["landing_vx"].get().strip()),
            landing_vy=float(self.vars["landing_vy"].get().strip()),
            landing_angle=float(self.vars["landing_angle"].get().strip()),
            reward_progress_scale=float(self.vars["reward_progress_scale"].get().strip()),
            reward_step_penalty=float(self.vars["reward_step_penalty"].get().strip()),
            reward_throttle_penalty=float(self.vars["reward_throttle_penalty"].get().strip()),
            reward_turn_penalty=float(self.vars["reward_turn_penalty"].get().strip()),
            reward_out_of_bounds_penalty=float(self.vars["reward_out_of_bounds_penalty"].get().strip()),
            reward_landing_bonus=float(self.vars["reward_landing_bonus"].get().strip()),
            reward_landing_fuel_bonus=float(self.vars["reward_landing_fuel_bonus"].get().strip()),
            reward_landing_precision_bonus=float(self.vars["reward_landing_precision_bonus"].get().strip()),
            reward_landing_precision_scale=float(self.vars["reward_landing_precision_scale"].get().strip()),
            reward_crash_penalty=float(self.vars["reward_crash_penalty"].get().strip()),
            reward_crash_impact_penalty=float(self.vars["reward_crash_impact_penalty"].get().strip()),
            reward_timeout_penalty=float(self.vars["reward_timeout_penalty"].get().strip()),
            spawn_mode=self.vars["spawn_mode"].get().strip() or "random",
            spawn_randomness=self.vars["spawn_randomness"].get().strip() or "dramatic",
            fps=int(self.vars["fps"].get().strip()),
        )
        validate_gui_config(config)
        return config

    def _start(self) -> None:
        self._reset_trainer(run=True)

    def _pause_training(self) -> None:
        self.training_running = False
        if self.training_job is not None:
            self.root.after_cancel(self.training_job)
            self.training_job = None
        self._start_live_replay(force_new=True)
        self._sync_progress_indicators(prefix="Training paused")

    def _resume_training(self) -> None:
        if self.trainer is None:
            self._start()
            return
        if self.training_running:
            return
        if len(self.trainer.history) >= self.trainer.config.generations:
            self._sync_progress_indicators(prefix="Training already complete")
            return
        self.training_running = True
        self.live_replay = None
        self._sync_progress_indicators(prefix="Training running")
        self._schedule_training()

    def _toggle_replay(self) -> None:
        self.replay_playing = not self.replay_playing
        self._sync_progress_indicators(prefix="Replay playing" if self.replay_playing else "Replay paused")

    def _reset_trainer(self, run: bool) -> None:
        try:
            config = self._config_from_vars()
            self.trainer = GradientRocketTrainer(config)
            self.current_update = self.trainer.preview
            self.live_replay = None
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            self.status_var.set("Settings error")
            return
        self.replay_step = 0
        self.training_running = run
        if self.training_job is not None:
            self.root.after_cancel(self.training_job)
            self.training_job = None
        self._refresh_views()
        self._sync_progress_indicators(prefix="Training running" if run else "Ready")
        if run:
            self._schedule_training()

    def _schedule_training(self) -> None:
        if not self.training_running or self.trainer is None:
            return
        if len(self.trainer.history) >= self.trainer.config.generations:
            self.training_running = False
            self._start_live_replay(force_new=True)
            self._sync_progress_indicators(prefix="Training complete")
            return
        self._run_training_generation()
        if self.training_running:
            self.training_job = self.root.after(1, self._schedule_training)

    def _run_training_generation(self) -> None:
        if self.trainer is None:
            return
        update = self.trainer.train_generation()
        self.current_update = update
        self.live_replay = None
        self.replay_step = 0
        self._refresh_views()
        if update.generation >= self.trainer.config.generations:
            self.training_running = False
            self._start_live_replay(force_new=True)
            self._sync_progress_indicators(prefix="Training complete")
        else:
            self._sync_progress_indicators(prefix="Training running")
        self.root.update_idletasks()

    def _schedule_replay(self) -> None:
        delay = 33
        if self.trainer is not None:
            delay = max(18, int(1000 / max(self.trainer.config.fps, 1)))
        replay = self._active_replay()
        if self.replay_playing and replay is not None and len(replay.positions) > 0:
            if self.replay_step + 1 >= len(replay.positions):
                if self._is_live_replay_mode():
                    self._start_live_replay(force_new=True)
                else:
                    self.replay_step = 0
            else:
                self.replay_step += 1
            self._draw_world()
            self._draw_network()
            self._refresh_network_text()
        self.replay_job = self.root.after(delay, self._schedule_replay)

    def _stats_lines(self) -> list[str]:
        if self.trainer is None or self.current_update is None:
            return ["No training data yet."]
        update = self.current_update
        replay = self._active_replay()
        loss_text = "n/a" if update.last_loss is None else f"{update.last_loss:.4f}"
        replay_mode = "live current-brain" if self._is_live_replay_mode() else "generation replay"
        replay_outcome = replay.outcome if replay is not None else "n/a"
        replay_reward = replay.reward if replay is not None else 0.0
        return [
            f"trainer={self.trainer.config.trainer} | hidden={format_hidden_layers(self.trainer.config.hidden_layers)} | activation={self.trainer.config.activation}",
            f"generation={update.generation}/{self.trainer.config.generations} | best_reward={update.best_reward:.2f} | mean_reward={update.mean_reward:.2f}",
            f"landing_rate={update.success_rate * 100:.1f}% | last_loss={loss_text} | elapsed={update.elapsed_seconds:.1f}s",
            f"spawn={self.trainer.config.spawn_mode}/{self.trainer.config.spawn_randomness} | batch_episodes={self.trainer.config.batch_episodes} | rollouts={self.trainer.config.rollouts}",
            f"replay_mode={replay_mode} | replay_outcome={replay_outcome} | replay_reward={replay_reward:.2f}",
        ]

    def _refresh_views(self) -> None:
        self.stats_var.set("\n".join(self._stats_lines()))
        self._refresh_network_text()
        self._sync_progress_indicators()
        self._draw_world()
        self._draw_metrics()
        self._draw_network()

    def _sync_progress_indicators(self, prefix: str | None = None) -> None:
        if self.trainer is None or self.current_update is None:
            progress_text = "Generation 0/0 | Landing rate 0.0%"
        else:
            progress_text = (
                f"Generation {self.current_update.generation}/{self.trainer.config.generations} | "
                f"Landing rate {self.current_update.success_rate * 100:.1f}% | "
                f"Best {self.current_update.best_reward:.2f} | Mean {self.current_update.mean_reward:.2f}"
            )
        self.progress_var.set(progress_text)
        status_bits = [prefix] if prefix else []
        status_bits.append(progress_text)
        if self._is_live_replay_mode():
            status_bits.append("Live replay")
        if not self.replay_playing:
            status_bits.append("Replay paused")
        self.status_var.set(" | ".join(status_bits))

    def _refresh_network_text(self) -> None:
        replay = self._active_replay()
        if self.trainer is None or self.current_update is None or replay is None or not replay.outputs:
            self.network_var.set("No replay available yet.")
            return
        frame = min(self.replay_step, len(replay.outputs) - 1)
        outputs = replay.outputs[frame]
        turn_cmd = float(replay.turn_commands[frame])
        throttle = float(replay.throttles[frame])
        fuel = float(replay.fuels[frame])
        replay_label = "Live replay" if self._is_live_replay_mode() else "Generation replay"
        self.network_var.set(
            "\n".join(
                [
                    f"{replay_label} | Generation: {self.current_update.generation} | Step: {frame + 1}/{len(replay.outputs)}",
                    f"Hidden layout: {format_hidden_layers(self.trainer.config.hidden_layers)}",
                    f"Turn mean={outputs[0]:+.3f} | Throttle mean={outputs[1]:+.3f} | Value={outputs[2]:+.3f}",
                    f"Applied turn={turn_cmd:+.3f} | Applied throttle={throttle:.3f} | Fuel={fuel:.3f}",
                ]
            )
        )

    def _is_live_replay_mode(self) -> bool:
        return self.live_replay is not None and not self.training_running

    def _active_replay(self) -> VisualReplay | None:
        if self._is_live_replay_mode():
            return self.live_replay
        if self.current_update is None:
            return None
        return self.current_update.replay

    def _start_live_replay(self, force_new: bool) -> None:
        if self.trainer is None:
            return
        if force_new or self.live_replay is None:
            self.live_replay = self.trainer.sample_replay()
            self.replay_step = 0
        self._refresh_views()

    def _world_to_canvas(self, x: float, y: float, width: float, height: float) -> tuple[float, float]:
        margin_x = 54.0
        margin_y = 42.0
        scale_x = (width - 2 * margin_x) / (rocket.WORLD_X_MAX - rocket.WORLD_X_MIN)
        scale_y = (height - 2 * margin_y) / (rocket.WORLD_Y_MAX - rocket.GROUND_Y + 0.08)
        px = margin_x + (x - rocket.WORLD_X_MIN) * scale_x
        py = height - margin_y - (y - rocket.GROUND_Y) * scale_y
        return px, py

    def _rocket_glow_kind(self, outcome: str) -> str:
        if outcome == "landed":
            return "success"
        if outcome in {"crashed", "out_of_bounds"}:
            return "failure"
        return "neutral"

    def _draw_world(self) -> None:
        canvas = self.world_canvas
        canvas.delete("all")
        width = max(320, canvas.winfo_width())
        height = max(320, canvas.winfo_height())
        sky_top = "#f9fcff"
        sky_bottom = "#d7edf7"
        canvas.create_rectangle(0, 0, width, height, fill=sky_top, outline="")
        canvas.create_rectangle(0, height * 0.60, width, height, fill=sky_bottom, outline="")

        ground_left = self._world_to_canvas(rocket.WORLD_X_MIN, rocket.GROUND_Y, width, height)
        ground_right = self._world_to_canvas(rocket.WORLD_X_MAX, rocket.GROUND_Y, width, height)
        canvas.create_rectangle(0, ground_left[1], width, height, fill="#d9d0bf", outline="")
        canvas.create_line(ground_left[0], ground_left[1], ground_right[0], ground_right[1], fill="#6f5d44", width=3)

        replay = self._active_replay()
        if self.trainer is None or self.current_update is None or replay is None or len(replay.positions) == 0:
            canvas.create_text(
                width / 2,
                height / 2,
                text="Start training to see a rocket replay.",
                font=("Segoe UI", 18),
                fill="#4c6269",
            )
            return

        config = self.trainer.config
        frame = min(self.replay_step, len(replay.positions) - 1)
        path = replay.positions[: frame + 1]
        position = path[-1]
        angle = float(replay.angles[frame])
        throttle = float(replay.throttles[frame])

        pad_left = rocket.PAD_X - config.pad_width * 0.5
        pad_right = rocket.PAD_X + config.pad_width * 0.5
        pad_y = rocket.GROUND_Y
        pad_a = self._world_to_canvas(pad_left, pad_y, width, height)
        pad_b = self._world_to_canvas(pad_right, pad_y, width, height)
        canvas.create_rectangle(pad_a[0], pad_a[1] - 8, pad_b[0], pad_b[1] + 8, fill="#f2c14d", outline="#2b2520", width=2)

        path_points: list[float] = []
        for point in path:
            px, py = self._world_to_canvas(float(point[0]), float(point[1]), width, height)
            path_points.extend((px, py))
        if len(path_points) >= 4:
            canvas.create_line(*path_points, fill="#4f87d9", width=3, smooth=True)

        canvas.create_line(
            *self._world_to_canvas(position[0], position[1], width, height),
            *self._world_to_canvas(rocket.PAD_X, rocket.GROUND_Y, width, height),
            fill="#adb9be",
            dash=(4, 3),
            width=1,
        )

        if throttle > 0.05:
            tail_world = position + np.array([0.0, -rocket.ROCKET_HEIGHT * 0.18], dtype=np.float64)
            flame_direction = np.array([np.sin(angle), np.cos(angle)], dtype=np.float64)
            flame_end = tail_world - flame_direction * (0.10 + throttle * 0.18)
            tail_px = self._world_to_canvas(float(tail_world[0]), float(tail_world[1]), width, height)
            flame_px = self._world_to_canvas(float(flame_end[0]), float(flame_end[1]), width, height)
            canvas.create_line(tail_px[0], tail_px[1], flame_px[0], flame_px[1], fill="#ff8b2d", width=4)

        center_px = self._world_to_canvas(float(position[0]), float(position[1]), width, height)
        world_zero = self._world_to_canvas(0.0, 0.0, width, height)
        world_rocket = self._world_to_canvas(0.0, rocket.ROCKET_HEIGHT, width, height)
        rocket_pixel_height = int(max(48.0, abs(world_rocket[1] - world_zero[1]) * 1.55))
        sprite = self.sprite_renderer.render(
            rocket_pixel_height,
            angle,
            self._rocket_glow_kind(replay.outcome),
        )
        if sprite is not None:
            canvas.create_image(center_px[0], center_px[1], image=sprite)
        else:
            vertices = rocket.rocket_vertices(position, angle)
            polygon_points: list[float] = []
            for vertex in vertices:
                px, py = self._world_to_canvas(float(vertex[0]), float(vertex[1]), width, height)
                polygon_points.extend((px, py))

            fill = "#ffffff"
            if replay.outcome == "landed":
                fill = "#c8f0c5"
            elif replay.outcome == "crashed":
                fill = "#f6b3a8"
            canvas.create_polygon(*polygon_points, fill=fill, outline="#20343a", width=2)

        observation = replay.inputs[frame]
        speed = float(observation[10])
        on_pad = "yes" if observation[11] > 0.5 else "no"
        canvas.create_text(
            width / 2,
            24,
            text=(
                f"Generation {self.current_update.generation}/{config.generations} | "
                f"{'Live replay' if self._is_live_replay_mode() else 'Replay'} step {frame + 1}/{len(replay.positions)} | "
                f"Outcome: {replay.outcome.replace('_', ' ')}"
            ),
            font=("Segoe UI Semibold", 14),
            fill="#294045",
        )
        canvas.create_text(
            width / 2,
            height - 24,
            text=(
                f"x={observation[0]:+.3f} y={observation[1]:+.3f} vx={observation[3]:+.3f} vy={observation[4]:+.3f} "
                f"speed={speed:.3f} on_pad={on_pad}"
            ),
            font=("Consolas", 11),
            fill="#41575d",
        )

    def _draw_metrics(self) -> None:
        canvas = self.metrics_canvas
        canvas.delete("all")
        width = max(360, canvas.winfo_width())
        height = max(180, canvas.winfo_height())
        canvas.create_rectangle(0, 0, width, height, fill="#fbfcfb", outline="")

        title_y = 22
        canvas.create_text(
            18,
            title_y,
            text="Training Progress",
            anchor="w",
            font=("Segoe UI Semibold", 18),
            fill="#234048",
        )

        if self.trainer is None or not self.trainer.history:
            canvas.create_text(
                width / 2,
                height / 2,
                text="Training curves will appear here as generations finish.",
                font=("Segoe UI", 15),
                fill="#60757b",
            )
            return

        history = self.trainer.history
        left_margin = 62
        right_margin = 64
        top_margin = 58
        bottom_margin = 34
        plot_left = left_margin
        plot_top = top_margin
        plot_right = width - right_margin
        plot_bottom = height - bottom_margin
        plot_width = max(10.0, plot_right - plot_left)
        plot_height = max(10.0, plot_bottom - plot_top)

        best_rewards = np.array([item.best_reward for item in history], dtype=np.float64)
        mean_rewards = np.array([item.mean_reward for item in history], dtype=np.float64)
        landing_rates = np.array([item.success_rate * 100.0 for item in history], dtype=np.float64)
        reward_min = float(min(np.min(best_rewards), np.min(mean_rewards)))
        reward_max = float(max(np.max(best_rewards), np.max(mean_rewards)))
        if abs(reward_max - reward_min) < 1e-6:
            reward_min -= 1.0
            reward_max += 1.0
        padding = max(0.8, (reward_max - reward_min) * 0.14)
        reward_min -= padding
        reward_max += padding

        canvas.create_rectangle(plot_left, plot_top, plot_right, plot_bottom, outline="#cfd8da", width=1)
        for tick in range(5):
            frac = tick / 4.0
            y = plot_bottom - frac * plot_height
            canvas.create_line(plot_left, y, plot_right, y, fill="#e5ecee")
            reward_value = reward_min + frac * (reward_max - reward_min)
            landing_value = frac * 100.0
            canvas.create_text(plot_left - 12, y, text=f"{reward_value:.1f}", anchor="e", font=("Consolas", 11), fill="#50666c")
            canvas.create_text(plot_right + 12, y, text=f"{landing_value:.0f}%", anchor="w", font=("Consolas", 11), fill="#50666c")

        generation_max = max(1, self.trainer.config.generations)
        for tick in range(5):
            frac = tick / 4.0
            x = plot_left + frac * plot_width
            canvas.create_line(x, plot_top, x, plot_bottom, fill="#eef3f4")
            generation_value = int(round(1 + frac * max(generation_max - 1, 0)))
            canvas.create_text(x, plot_bottom + 16, text=str(generation_value), anchor="n", font=("Consolas", 11), fill="#50666c")

        def map_generation(generation: int) -> float:
            if generation_max <= 1:
                return plot_left
            return plot_left + ((generation - 1) / (generation_max - 1)) * plot_width

        def map_reward(value: float) -> float:
            return plot_bottom - ((value - reward_min) / (reward_max - reward_min)) * plot_height

        def map_landing(value: float) -> float:
            return plot_bottom - (value / 100.0) * plot_height

        def draw_series(values: np.ndarray, mapper, color: str, width_px: int, dash: tuple[int, int] | None = None) -> None:
            points: list[float] = []
            for index, value in enumerate(values, start=1):
                points.extend((map_generation(index), mapper(float(value))))
            if len(points) >= 4:
                canvas.create_line(*points, fill=color, width=width_px, smooth=True, dash=dash if dash else ())
            elif len(points) == 2:
                canvas.create_oval(points[0] - 3, points[1] - 3, points[0] + 3, points[1] + 3, fill=color, outline=color)

        draw_series(best_rewards, map_reward, "#336fd1", 3)
        draw_series(mean_rewards, map_reward, "#7aa6ea", 2, dash=(6, 4))
        draw_series(landing_rates, map_landing, "#2da45a", 3)

        current_generation = self.current_update.generation if self.current_update is not None else len(history)
        marker_x = map_generation(current_generation)
        canvas.create_line(marker_x, plot_top, marker_x, plot_bottom, fill="#1f2b30", width=1, dash=(4, 4))

        legend_items = (
            ("Best reward", "#336fd1"),
            ("Mean reward", "#7aa6ea"),
            ("Landing rate", "#2da45a"),
        )
        legend_x = plot_left
        for label, color in legend_items:
            canvas.create_line(legend_x, 44, legend_x + 24, 44, fill=color, width=4)
            canvas.create_text(legend_x + 34, 44, text=label, anchor="w", font=("Segoe UI Semibold", 12), fill="#486066")
            legend_x += 156

        canvas.create_text(plot_left - 42, plot_top - 12, text="Reward", anchor="w", font=("Segoe UI Semibold", 12), fill="#486066")
        canvas.create_text(plot_right - 18, plot_top - 12, text="Landing %", anchor="w", font=("Segoe UI Semibold", 12), fill="#486066")
        canvas.create_text(
            plot_right,
            plot_top - 10,
            text=f"Now: G{current_generation} | best {best_rewards[-1]:.2f} | mean {mean_rewards[-1]:.2f} | land {landing_rates[-1]:.1f}%",
            anchor="e",
            font=("Segoe UI Semibold", 12),
            fill="#27444b",
        )

    def _draw_network(self) -> None:
        canvas = self.network_canvas
        canvas.delete("all")
        width = max(300, canvas.winfo_width())
        height = max(320, canvas.winfo_height())
        replay = self._active_replay()
        if self.trainer is None or self.current_update is None or replay is None or not replay.outputs:
            canvas.create_text(
                width / 2,
                height / 2,
                text="Network activations will appear here during replay.",
                font=("Segoe UI", 16),
                fill="#50656b",
            )
            return

        frame = min(self.replay_step, len(replay.outputs) - 1)
        input_values = list(replay.inputs[frame])
        hidden_values = [list(layer) for layer in replay.hidden_values[frame]]
        output_values = list(replay.outputs[frame])
        layers = [input_values] + hidden_values + [output_values]
        layer_names = ["Inputs"] + [f"H{i + 1}" for i in range(len(hidden_values))] + ["Outputs"]
        weights = policy_weight_matrices(self.trainer.policy)
        x_positions = np.linspace(56, width - 56, num=len(layers))
        node_positions: list[list[tuple[float, float]]] = []

        for layer_index, layer_values in enumerate(layers):
            if len(layer_values) == 1:
                ys = [height / 2]
            else:
                ys = np.linspace(70, height - 58, num=len(layer_values))
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
                canvas.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill=color_for_value(float(value)),
                    outline="#304247",
                    width=1,
                )
                if layer_index == 0:
                    canvas.create_text(x - 18, y, text=INPUT_LABELS[node_index], anchor="e", font=("Consolas", 9), fill="#4a5f65")
                    canvas.create_text(x + 18, y, text=f"{value:+.2f}", anchor="w", font=("Consolas", 9), fill="#22383e")
                elif layer_index == len(layers) - 1:
                    canvas.create_text(x + 18, y - 7, text=OUTPUT_LABELS[node_index], anchor="w", font=("Segoe UI", 9), fill="#4a5f65")
                    canvas.create_text(x + 18, y + 7, text=f"{value:+.2f}", anchor="w", font=("Consolas", 10), fill="#1f3338")
                elif len(layer_values) <= 20:
                    canvas.create_text(x + 16, y, text=f"{value:+.2f}", anchor="w", font=("Consolas", 8), fill="#21383e")

        total_edges = sum(len(layers[index]) * len(layers[index + 1]) for index in range(len(layers) - 1))
        edge_budget = 260
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
            height - 20,
            text=f"activation={self.trainer.config.activation} | raw outputs -> tanh(turn), sigmoid(throttle)",
            font=("Segoe UI", 10),
            fill="#42575d",
        )


def validate_gui_config(config: GuiConfig) -> None:
    if config.generations < 1:
        raise ValueError("generations must be at least 1")
    if config.trainer not in {"ppo", "reinforce"}:
        raise ValueError("trainer must be ppo or reinforce")
    if config.batch_episodes < 1:
        raise ValueError("batch_episodes must be at least 1")
    if config.rollouts < 1:
        raise ValueError("rollouts must be at least 1")
    if config.episode_steps < 20:
        raise ValueError("episode_steps must be at least 20")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be greater than 0")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be zero or greater")
    if not 0.0 < config.gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if not 0.0 < config.gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in (0, 1]")
    if config.entropy_coef < 0.0:
        raise ValueError("entropy_coef must be zero or greater")
    if config.value_coef < 0.0:
        raise ValueError("value_coef must be zero or greater")
    if config.ppo_clip <= 0.0:
        raise ValueError("ppo_clip must be greater than 0")
    if config.ppo_epochs < 1:
        raise ValueError("ppo_epochs must be at least 1")
    if config.minibatch_size < 1:
        raise ValueError("minibatch_size must be at least 1")
    if config.gravity <= 0.0:
        raise ValueError("gravity must be greater than 0")
    if config.main_thrust <= 0.0:
        raise ValueError("main_thrust must be greater than 0")
    if config.turn_power <= 0.0:
        raise ValueError("turn_power must be greater than 0")
    if config.fuel_burn <= 0.0:
        raise ValueError("fuel_burn must be greater than 0")
    if config.pad_width <= 0.0:
        raise ValueError("pad_width must be greater than 0")
    if config.landing_vx <= 0.0 or config.landing_vy <= 0.0 or config.landing_angle <= 0.0:
        raise ValueError("landing thresholds must be greater than 0")
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
    if config.fps < 1:
        raise ValueError("fps must be at least 1")


def default_config() -> GuiConfig:
    backend_defaults = rocket.default_config()
    return GuiConfig(
        generations=backend_defaults.generations,
        trainer=backend_defaults.trainer,
        batch_episodes=backend_defaults.batch_episodes,
        rollouts=backend_defaults.rollouts,
        hidden_layers=backend_defaults.hidden_layers,
        activation=backend_defaults.activation,
        episode_steps=backend_defaults.episode_steps,
        seed=backend_defaults.seed,
        learning_rate=backend_defaults.learning_rate,
        weight_decay=backend_defaults.weight_decay,
        gamma=backend_defaults.gamma,
        gae_lambda=backend_defaults.gae_lambda,
        entropy_coef=backend_defaults.entropy_coef,
        value_coef=backend_defaults.value_coef,
        ppo_clip=backend_defaults.ppo_clip,
        ppo_epochs=backend_defaults.ppo_epochs,
        minibatch_size=backend_defaults.minibatch_size,
        gravity=backend_defaults.gravity,
        main_thrust=backend_defaults.main_thrust,
        turn_power=backend_defaults.turn_power,
        fuel_burn=backend_defaults.fuel_burn,
        pad_width=backend_defaults.pad_width,
        landing_vx=backend_defaults.landing_vx,
        landing_vy=backend_defaults.landing_vy,
        landing_angle=backend_defaults.landing_angle,
        reward_progress_scale=backend_defaults.reward_progress_scale,
        reward_step_penalty=backend_defaults.reward_step_penalty,
        reward_throttle_penalty=backend_defaults.reward_throttle_penalty,
        reward_turn_penalty=backend_defaults.reward_turn_penalty,
        reward_out_of_bounds_penalty=backend_defaults.reward_out_of_bounds_penalty,
        reward_landing_bonus=backend_defaults.reward_landing_bonus,
        reward_landing_fuel_bonus=backend_defaults.reward_landing_fuel_bonus,
        reward_landing_precision_bonus=backend_defaults.reward_landing_precision_bonus,
        reward_landing_precision_scale=backend_defaults.reward_landing_precision_scale,
        reward_crash_penalty=backend_defaults.reward_crash_penalty,
        reward_crash_impact_penalty=backend_defaults.reward_crash_impact_penalty,
        reward_timeout_penalty=backend_defaults.reward_timeout_penalty,
        spawn_mode=backend_defaults.spawn_mode,
        spawn_randomness=backend_defaults.spawn_randomness,
        fps=backend_defaults.fps,
    )


def config_from_args(args: argparse.Namespace) -> GuiConfig:
    config = GuiConfig(
        generations=args.generations,
        trainer=args.trainer,
        batch_episodes=args.batch_episodes,
        rollouts=args.rollouts,
        hidden_layers=args.hidden,
        activation=args.activation,
        episode_steps=args.episode_steps,
        seed=args.seed,
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
        smoke_test_generations=args.smoke_test_generations,
    )
    validate_gui_config(config)
    return config


def run_smoke_test(config: GuiConfig) -> None:
    trainer = GradientRocketTrainer(config)
    total_generations = max(1, config.smoke_test_generations)
    latest = trainer.preview
    for _ in range(total_generations):
        latest = trainer.train_generation()
    print("Smoke test complete")
    print(f"generations={latest.generation}")
    print(f"best_reward={latest.best_reward:.2f}")
    print(f"mean_reward={latest.mean_reward:.2f}")
    print(f"landing_rate={latest.success_rate * 100:.1f}%")
    print(f"replay_outcome={latest.replay.outcome}")
    print(f"replay_steps={len(latest.replay.positions)}")
    print(f"last_loss={'n/a' if latest.last_loss is None else f'{latest.last_loss:.4f}'}")


def build_parser() -> argparse.ArgumentParser:
    defaults = default_config()
    parser = argparse.ArgumentParser(description="Rocket landing deep-RL GUI with PPO defaults and network visualization.")
    parser.add_argument("--generations", type=int, default=defaults.generations)
    parser.add_argument("--trainer", choices=["ppo", "reinforce"], default=defaults.trainer)
    parser.add_argument("--batch-episodes", type=int, default=defaults.batch_episodes)
    parser.add_argument("--rollouts", type=int, default=defaults.rollouts)
    parser.add_argument("--hidden", type=rocket.parse_hidden_layers, default=defaults.hidden_layers)
    parser.add_argument("--activation", choices=tuple(sorted(rocket.ACTIVATIONS)), default=defaults.activation)
    parser.add_argument("--episode-steps", type=int, default=defaults.episode_steps)
    parser.add_argument("--seed", type=int, default=defaults.seed, help="Leave unset for a fresh random run.")
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
    parser.add_argument(
        "--smoke-test-generations",
        type=int,
        default=0,
        help="Run a short headless training check instead of opening the GUI.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = config_from_args(args)
    except Exception as exc:
        parser.error(str(exc))
        return
    if config.smoke_test_generations > 0:
        run_smoke_test(config)
        return
    root = tk.Tk()
    app = RocketLandingGuiApp(root, config)
    root.protocol(
        "WM_DELETE_WINDOW",
        lambda: (
            app._pause_training(),
            root.after_cancel(app.replay_job) if app.replay_job is not None else None,
            root.destroy(),
        ),
    )
    root.mainloop()


if __name__ == "__main__":
    main()
