import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .callbacks import ACTIONS, DIRECTIONS, calculate_blast_cells, in_bounds, prepare_state
from .dqn_model import DQNResNet

BATCH_SIZE = 128
GAMMA = 0.99
LR = 1e-4
MEMORY_SIZE = 100_000
TARGET_UPDATE_FREQ = 2500

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 100_000
OPTIMIZE_EVERY = 16

STAGE_CONFIG = {
    "coin-heaven": {
        "start_round": 0,
        "end_round": 2000,
        "epsilon_start": 1.00,
        "epsilon_decay_steps": 100_000,
    },
    "loot-crate": {
        "start_round": 2000,
        "end_round": 5000,
        "epsilon_start": 0.30,
        "epsilon_decay_steps": 150_000,
    },
    "classic": {
        "start_round": 5000,
        "end_round": 10000,
        "epsilon_start": 0.15,
        "epsilon_decay_steps": 100_000,
    },
}

REWARD_KILL = 50.0
REWARD_COIN = 8.0
REWARD_CRATE = 0.6
REWARD_WAIT_PENALTY = -0.1
REWARD_DEATH = -100.0
COIN_POTENTIAL_WEIGHT = 7.0
BOMB_TARGETS_CRATE_BONUS = 0.5
BOMB_TARGETS_OPPONENT_BONUS = 2.0
STATE_SHAPE = (12, 17, 17)

ACTION_DIM = 6
_THIS_DIR = os.path.dirname(__file__)

CSV_LOG_PATH = os.path.join(_THIS_DIR, "training_log.csv")
STATE_META_PATH = os.path.join(_THIS_DIR, "training_state.json")
MODEL_PATH = os.path.join(_THIS_DIR, "dqn_model.pt")

CSV_FIELDNAMES = [
    "round",
    "steps",
    "total_steps_done",
    "epsilon",
    "total_reward",
    "avg_reward_per_step",
    "coins_collected",
    "crates_destroyed",
    "kills",
    "died",
    "suicide",
    "survived_round",
    "opponents_remaining",
    "buffer_size",
    "optimizer_updates",
    "avg_loss",
]


def get_stage_from_round(rounds_done):
    for stage_name, config in STAGE_CONFIG.items():
        if rounds_done < config["end_round"]:
            return stage_name
    return "classic"


def get_stage_config(stage):
    return STAGE_CONFIG.get(stage, STAGE_CONFIG["classic"])


def get_stage_epsilon(stage_steps_done, stage):
    config = get_stage_config(stage)
    eps_start = float(config["epsilon_start"])
    decay_steps = float(config["epsilon_decay_steps"])
    decay_progress = min(1.0, stage_steps_done / decay_steps)
    epsilon = eps_start - (eps_start - EPS_END) * decay_progress
    return max(EPS_END, epsilon)


def _proximity_potential(distance):
    if distance is None:
        return 0.0
    return 1.0 / (1.0 + float(distance))


def _bomb_target_bonus(old_game_state):
    field = old_game_state['field']
    sx, sy = old_game_state['self'][3]
    own_blast = calculate_blast_cells(field, (sx, sy))
    targets_crate = any(field[x, y] == 1 for x, y in own_blast)
    other_positions = {tuple(o[3]) for o in old_game_state.get('others', [])}
    targets_opponent = any((x, y) in other_positions for x, y in own_blast)

    bonus = 0.0
    if targets_crate:
        bonus += BOMB_TARGETS_CRATE_BONUS
    if targets_opponent:
        bonus += BOMB_TARGETS_OPPONENT_BONUS

    return bonus, targets_crate, targets_opponent


class ReplayBuffer:
    def __init__(self, capacity: int, state_shape=STATE_SHAPE, action_dim=ACTION_DIM):
        self.capacity = capacity
        self.states = np.zeros((capacity,) + state_shape, dtype=np.float32)
        self.next_states = np.zeros((capacity,) + state_shape, dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_masks = np.zeros((capacity, action_dim), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done, mask):
        idx = self.pos
        self.states[idx] = state
        self.next_states[idx] = next_state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.next_masks[idx] = mask
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.next_masks[indices],
        )

    def __len__(self):
        return self.size


def setup_training(self):
    self.replay_buffer = ReplayBuffer(MEMORY_SIZE)
    self.target_net = DQNResNet(input_channels=12, num_actions=6).to(self.device)
    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)

    self.steps_done = 0
    self.stage_steps_done = 0
    self.rounds_done = 0
    self.optimizer_updates = 0
    self.current_stage = "coin-heaven"
    self.epsilon = 1.0

    if os.path.isfile(STATE_META_PATH):
        try:
            with open(STATE_META_PATH, "r") as f:
                state = json.load(f)

            self.steps_done = int(state.get("steps_done", 0))
            self.rounds_done = int(state.get("rounds_done", 0))
            self.optimizer_updates = int(state.get("optimizer_updates", 0))
            saved_stage = state.get("stage", None)
            saved_stage_steps = int(state.get("stage_steps_done", 0))

            self.current_stage = get_stage_from_round(self.rounds_done)

            if saved_stage == self.current_stage:
                self.stage_steps_done = saved_stage_steps
            else:
                self.stage_steps_done = 0

            self.epsilon = get_stage_epsilon(self.stage_steps_done, self.current_stage)
            self.logger.info(
                f"Resumed training: round={self.rounds_done}, total_steps={self.steps_done}, "
                f"stage={self.current_stage}, stage_steps={self.stage_steps_done}, "
                f"epsilon={self.epsilon:.4f}, updates={self.optimizer_updates}"
            )
        except (json.JSONDecodeError, OSError, ValueError) as e:
            self.logger.warning(
                f"Could not read {STATE_META_PATH}: {e}. Starting training counters from zero."
            )
            self.steps_done = 0
            self.stage_steps_done = 0
            self.rounds_done = 0
            self.optimizer_updates = 0
            self.current_stage = "coin-heaven"
            self.epsilon = 1.0
    else:
        self.current_stage = get_stage_from_round(self.rounds_done)
        self.stage_steps_done = 0
        self.epsilon = get_stage_epsilon(self.stage_steps_done, self.current_stage)

    self.target_net.load_state_dict(self.policy_net.state_dict())
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
        f"Training initialized: round={self.rounds_done}, stage={self.current_stage}, "
        f"stage_steps={self.stage_steps_done}, total_steps={self.steps_done}, epsilon={self.epsilon:.4f}"
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

    if "BOMB_DROPPED" in events:
        bonus, _, _ = _bomb_target_bonus(old_game_state)
        reward += bonus

    if self_action == "WAIT":
        reward += REWARD_WAIT_PENALTY

    died = "KILLED_SELF" in events or "GOT_KILLED" in events
    suicide = "KILLED_SELF" in events

    if died:
        reward += REWARD_DEATH

    if getattr(self, "cached_features", None) is not None:
        old_features = self.cached_features
    else:
        old_features, _, _, _ = prepare_state(old_game_state)

    old_coin_distance = getattr(self, "cached_coin_distance", None)

    if new_game_state is not None:
        new_features, next_mask, _, new_coin_distance = prepare_state(new_game_state)
        is_terminal = False
        if np.sum(next_mask) == 0:
            next_mask = np.zeros(ACTION_DIM, dtype=np.float32)
            next_mask[4] = 1.0
    else:
        new_features = np.zeros(STATE_SHAPE, dtype=np.float32)
        next_mask = np.zeros(ACTION_DIM, dtype=np.float32)
        new_coin_distance = None
        is_terminal = True

    if not is_terminal:
        old_potential = _proximity_potential(old_coin_distance)
        new_potential = _proximity_potential(new_coin_distance)
        reward += COIN_POTENTIAL_WEIGHT * (GAMMA * new_potential - old_potential)

    if self_action is not None and self_action in ACTIONS:
        action_idx = ACTIONS.index(self_action)
    else:
        action_idx = 4

    self.replay_buffer.push(old_features, action_idx, reward, new_features, is_terminal, next_mask)

    self._round_reward += reward
    self._round_steps += 1
    self._round_coins += n_coins
    self._round_crates += n_crates
    self._round_kills += n_kills
    self._round_died = self._round_died or died
    self._round_suicide = self._round_suicide or suicide

    self.steps_done += 1
    self.stage_steps_done += 1

    self.epsilon = get_stage_epsilon(self.stage_steps_done, self.current_stage)

    if self.steps_done % OPTIMIZE_EVERY == 0:
        loss = _optimize_model(self)
        if loss is not None:
            self._round_losses.append(loss)


def end_of_round(self, last_game_state, last_action, events):
    game_events_occurred(self, last_game_state, last_action, None, events)
    self.rounds_done += 1

    new_stage = get_stage_from_round(self.rounds_done)
    if new_stage != self.current_stage:
        self.current_stage = new_stage
        self.stage_steps_done = 0
        self.epsilon = get_stage_epsilon(0, self.current_stage)
        self.logger.info("====================================")
        self.logger.info(f"NEW TRAINING STAGE: {self.current_stage}")
        self.logger.info(f"epsilon reset to {self.epsilon:.4f}")
        self.logger.info("====================================")
    else:
        self.epsilon = get_stage_epsilon(self.stage_steps_done, self.current_stage)

    if last_game_state is not None:
        round_number = last_game_state.get("round", self.rounds_done)
        opponents_remaining = len(last_game_state.get("others", []))
    else:
        round_number = self.rounds_done
        opponents_remaining = 0

    avg_reward = self._round_reward / self._round_steps if self._round_steps else 0.0
    avg_loss = float(np.mean(self._round_losses)) if self._round_losses else float("nan")

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
        "optimizer_updates": self.optimizer_updates,
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

    checkpoint = {
        "policy_net": self.policy_net.state_dict(),
        "target_net": self.target_net.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "steps_done": self.steps_done,
        "rounds_done": self.rounds_done,
        "optimizer_updates": self.optimizer_updates,
        "stage": self.current_stage,
        "stage_steps_done": self.stage_steps_done,
    }

    TEMP_MODEL_PATH = MODEL_PATH + ".tmp"
    torch.save(checkpoint, TEMP_MODEL_PATH)
    os.replace(TEMP_MODEL_PATH, MODEL_PATH)

    with open(STATE_META_PATH, "w") as f:
        json.dump(
            {
                "rounds_done": self.rounds_done,
                "steps_done": self.steps_done,
                "optimizer_updates": self.optimizer_updates,
                "stage": self.current_stage,
                "stage_steps_done": self.stage_steps_done,
            },
            f,
        )


def _optimize_model(self):
    if len(self.replay_buffer) < BATCH_SIZE:
        return None

    states, actions, rewards, next_states, dones, next_masks = self.replay_buffer.sample(BATCH_SIZE)

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

    self.optimizer_updates += 1

    if self.optimizer_updates % TARGET_UPDATE_FREQ == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    return loss.item()