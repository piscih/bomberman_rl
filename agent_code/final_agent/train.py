import csv
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .callbacks import (
    ACTIONS,
    DIRECTIONS,
    in_bounds,
    prepare_state,
    calculate_blast_cells,
)

from .dqn_model import DQNResNet

# ---- Hyperparameters ----
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4
MEMORY_SIZE = 100_000
TARGET_UPDATE_FREQ = 2500

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_ROUNDS = 6000  

# ---- Reward shaping ----
REWARD_KILL = 50.0
REWARD_COIN = 8.0
REWARD_CRATE = 0.6
REWARD_WAIT_PENALTY = -0.1
REWARD_DEATH = -100.0

COIN_POTENTIAL_WEIGHT = 5.0
OPPONENT_POTENTIAL_WEIGHT = 1.5
REWARD_BOMB_USEFUL = 0.1  
REWARD_BOMB_WASTED = 0.05 

STATE_SHAPE = (12, 17, 17)
ACTION_DIM = 6

_THIS_DIR = os.path.dirname(__file__)
CSV_LOG_PATH = os.path.join(_THIS_DIR, "training_log.csv")
STATE_META_PATH = os.path.join(_THIS_DIR, "training_state.json")
MODEL_PATH = os.path.join(_THIS_DIR, "dqn_model.pt")

CSV_FIELDNAMES = [
    "round", "steps", "total_steps_done", "epsilon",
    "total_reward", "avg_reward_per_step",
    "coins_collected", "crates_destroyed", "kills",
    "died", "suicide", "survived_round",
    "opponents_remaining", "buffer_size", "avg_loss",
]


def _proximity_potential(distance):
    if distance is None:
        return 0.0
    return 1.0 / (1.0 + float(distance))


def _bomb_targets_something(game_state):
    field = game_state["field"]
    sx, sy = game_state["self"][3]

    for dx, dy in DIRECTIONS:
        nx, ny = sx + dx, sy + dy
        if in_bounds(nx, ny) and field[nx, ny] == 1:
            return True

    blast = calculate_blast_cells(field, (sx, sy))
    opponent_positions = {tuple(o[3]) for o in game_state.get("others", [])}
    return bool(blast & opponent_positions)


def get_current_epsilon(rounds_done: int) -> float:
    decay_progress = min(1.0, rounds_done / EPS_DECAY_ROUNDS)
    return EPS_START - (EPS_START - EPS_END) * decay_progress


class ReplayBuffer:
    def __init__(self, capacity: int, state_shape=STATE_SHAPE, action_dim=ACTION_DIM):
        self.capacity = capacity
        self.states = np.zeros((capacity + 1,) + state_shape, dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_masks = np.zeros((capacity, action_dim), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done, mask):
        idx = self.pos

        self.states[idx] = state
        self.states[idx + 1] = next_state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.next_masks[idx] = mask

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)
        next_indices = indices + 1 

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.states[next_indices],
            self.dones[indices],
            self.next_masks[indices],
        )

    def __len__(self):
        return self.size


def setup_training(self):
    self.replay_buffer = ReplayBuffer(MEMORY_SIZE)
    self.target_net = DQNResNet(
        input_channels=12,
        num_actions=6
    ).to(self.device)

    self.optimizer = optim.Adam(
        self.policy_net.parameters(),
        lr=LR
    )

    self.steps_done = 0
    self.rounds_done = 0
    self.epsilon = EPS_START
    if os.path.isfile(STATE_META_PATH):
        try:
            with open(STATE_META_PATH, "r") as f:
                state = json.load(f)
            self.steps_done = state.get("steps_done", 0)
            self.rounds_done = state.get("rounds_done", 0)
            self.epsilon = get_current_epsilon(self.rounds_done)
            self.logger.info(
                f"Resumed training progress: round {self.rounds_done}, "
                f"step {self.steps_done}, epsilon {self.epsilon:.4f}"
            )
        except (json.JSONDecodeError, OSError) as e:
            self.logger.warning(f"Could not read {STATE_META_PATH}: {e}. Starting counters from zero.")

    self.target_net.load_state_dict(
        self.policy_net.state_dict()
    )

    self.target_net.eval()
    self._round_reward = 0.0
    self._round_steps = 0
    self._round_coins = 0
    self._round_crates = 0
    self._round_kills = 0
    self._round_died = False
    self._round_suicide = False
    self._round_losses = []

    write_header = not os.path.isfile(CSV_LOG_PATH)
    self._csv_file = open(CSV_LOG_PATH, "a", newline="")
    self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDNAMES)
    if write_header:
        self._csv_writer.writeheader()
        self._csv_file.flush()

    self.logger.info(
        f"Training initialized (round {self.rounds_done}). "
        f"Initial Epsilon = {self.epsilon:.4f}"
    )


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None:
        return
    reward = 0.0

    n_kills = events.count("KILLED_OPPONENT")
    reward += REWARD_KILL * n_kills

    n_coins = events.count("COIN_COLLECTED")
    reward += REWARD_COIN * n_coins

    n_crates = events.count("CRATE_DESTROYED")
    reward += REWARD_CRATE * n_crates

    if self_action == "WAIT":
        reward += REWARD_WAIT_PENALTY

    died = "KILLED_SELF" in events or "GOT_KILLED" in events
    suicide = "KILLED_SELF" in events
    if died:
        reward += REWARD_DEATH

    if (
        new_game_state is not None
        and old_game_state.get("others")
        and new_game_state.get("others")
    ):
        old_sx, old_sy = old_game_state["self"][3]
        new_sx, new_sy = new_game_state["self"][3]

        old_min_dist = min(
            abs(old_sx - ox) + abs(old_sy - oy)
            for _, _, _, (ox, oy) in old_game_state["others"]
        )
        new_min_dist = min(
            abs(new_sx - ox) + abs(new_sy - oy)
            for _, _, _, (ox, oy) in new_game_state["others"]
        )

        reward += OPPONENT_POTENTIAL_WEIGHT * (
            GAMMA * _proximity_potential(new_min_dist)
            - _proximity_potential(old_min_dist)
        )

    if self_action == "BOMB":
        if _bomb_targets_something(old_game_state):
            reward += REWARD_BOMB_USEFUL
        else:
            reward -= REWARD_BOMB_WASTED

    if getattr(self, "cached_features", None) is not None:
        old_features = self.cached_features
    else:
        old_features, _, _, _ = prepare_state(old_game_state)

    old_coin_distance = getattr(self, "cached_coin_distance", None)

    if new_game_state is not None:
        new_features, next_mask, _, new_coin_distance = prepare_state(new_game_state)
        is_terminal = False
    else:
        new_features = np.zeros((12, 17, 17), dtype=np.float32)
        next_mask = np.zeros(6, dtype=np.float32)
        new_coin_distance = None
        is_terminal = True
    reward += COIN_POTENTIAL_WEIGHT * (
        GAMMA * _proximity_potential(new_coin_distance)
        - _proximity_potential(old_coin_distance)
    )

    action_idx = (
        ACTIONS.index(self_action)
        if (self_action is not None and self_action in ACTIONS)
        else 4
    )

    self.replay_buffer.push(
        old_features, action_idx, reward, new_features, is_terminal, next_mask
    )

    self._round_reward += reward
    self._round_steps += 1
    self._round_coins += n_coins
    self._round_crates += n_crates
    self._round_kills += n_kills
    self._round_died = self._round_died or died
    self._round_suicide = self._round_suicide or suicide
    self.steps_done += 1

    if self.steps_done % 4 == 0:
        loss = _optimize_model(self)
        if loss is not None:
            self._round_losses.append(loss)


def end_of_round(self, last_game_state, last_action, events):
    game_events_occurred(self, last_game_state, last_action, None, events)

    self.rounds_done += 1
    self.epsilon = get_current_epsilon(self.rounds_done)

    round_number = last_game_state.get("round", self.rounds_done)
    avg_reward = self._round_reward / self._round_steps if self._round_steps else 0.0
    avg_loss = float(np.mean(self._round_losses)) if self._round_losses else float("nan")
    opponents_remaining = len(last_game_state.get("others", []))

    self._csv_writer.writerow({
        "round": round_number,
        "steps": self._round_steps,
        "total_steps_done": self.steps_done,
        "epsilon": round(self.epsilon, 4),
        "total_reward": round(self._round_reward, 4),
        "avg_reward_per_step": round(avg_reward, 4),
        "coins_collected": self._round_coins,
        "crates_destroyed": self._round_crates,
        "kills": self._round_kills,
        "died": int(self._round_died),
        "suicide": int(self._round_suicide),
        "survived_round": int(not self._round_died),
        "opponents_remaining": opponents_remaining,
        "buffer_size": len(self.replay_buffer),
        "avg_loss": avg_loss,
    })
    self._csv_file.flush()
    self._round_reward = 0.0
    self._round_steps = 0
    self._round_coins = 0
    self._round_crates = 0
    self._round_kills = 0
    self._round_died = False
    self._round_suicide = False
    self._round_losses = []
    torch.save(self.policy_net.state_dict(), MODEL_PATH)
    with open(STATE_META_PATH, "w") as f:
        json.dump({"rounds_done": self.rounds_done, "steps_done": self.steps_done}, f)


def _optimize_model(self):
    if len(self.replay_buffer) < BATCH_SIZE:
        return None

    states, actions, rewards, next_states, dones, next_masks = (
        self.replay_buffer.sample(BATCH_SIZE)
    )

    states_t = torch.from_numpy(states).float().to(self.device)
    actions_t = torch.from_numpy(actions).long().to(self.device).unsqueeze(1)
    rewards_t = torch.from_numpy(rewards).float().to(self.device).unsqueeze(1)
    next_states_t = torch.from_numpy(next_states).float().to(self.device)
    dones_t = torch.from_numpy(dones).float().to(self.device).unsqueeze(1)
    next_masks_t = torch.from_numpy(next_masks).float().to(self.device)

    q_values = self.policy_net(states_t).gather(1, actions_t)

    with torch.no_grad():
        next_q_policy = self.policy_net(next_states_t)

        masked_next_q = next_q_policy.clone()
        masked_next_q[next_masks_t == 0.0] = -1e9

        best_actions = masked_next_q.argmax(dim=1, keepdim=True)

        next_q_target = self.target_net(next_states_t).gather(1, best_actions)
        expected_q = rewards_t + (1.0 - dones_t) * GAMMA * next_q_target

    loss = nn.SmoothL1Loss()(q_values, expected_q)

    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
    self.optimizer.step()

    if self.steps_done % TARGET_UPDATE_FREQ == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    return loss.item()