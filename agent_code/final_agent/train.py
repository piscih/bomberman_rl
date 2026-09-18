import collections
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .callbacks import (
    ACTIONS,
    prepare_state,
)

from .dqn_model import DQNResNet

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4
MEMORY_SIZE = 100_000
TARGET_UPDATE_FREQ = 2500

EPS_START = 0.5
EPS_END = 0.05
EPSILON_DECAY = 20_000

# Reward Shaping
REWARD_KILL = 50.0
REWARD_COIN = 8.0
REWARD_CRATE = 0.6
REWARD_WAIT_PENALTY = -0.1
REWARD_CLOSER_TO_OPPONENT = 0.0
COIN_POTENTIAL_WEIGHT = 5.0


def _coin_potential(distance):
    """Phi(s) = 1/(1+distance) -- higher when closer to the nearest coin.
    0.0 when no coin exists or none is reachable (distance is None)."""
    if distance is None:
        return 0.0
    return 1.0 / (1.0 + float(distance))


def get_current_epsilon(steps_done: int) -> float:
    """Calculates linear decay epsilon based on total steps taken."""
    decay_progress = min(1.0, steps_done / EPSILON_DECAY)
    return EPS_END + (EPS_START - EPS_END) * (1.0 - decay_progress)


STATE_SHAPE = (12, 17, 17)
ACTION_DIM = 6


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
        next_indices = indices + 1  # always in bounds: indices in [0, capacity-1]

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
    self.epsilon = EPS_START

    self.target_net.load_state_dict(
        self.policy_net.state_dict()
    )

    self.target_net.eval()

    self.logger.info(
        f"Fresh training initialized. "
        f"Initial Epsilon = {self.epsilon:.4f}"
    )


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None:
        return

    # 1. Compute event-based reward
    reward = 0.0
    if "KILLED_OPPONENT" in events:
        reward += REWARD_KILL
    if "COIN_COLLECTED" in events:
        reward += REWARD_COIN
    if "CRATE_DESTROYED" in events:
        reward += REWARD_CRATE
    if self_action == "WAIT":
        reward += REWARD_WAIT_PENALTY
    if "KILLED_SELF" in events or "GOT_KILLED" in events:
        reward -= 100.0

    if (
        new_game_state is not None
        and old_game_state.get("others")
        and new_game_state.get("others")
    ):
        old_sx, old_sy = old_game_state["self"][3]
        new_sx, new_sy = new_game_state["self"][3]

        old_min_dist = min(
            [
                abs(old_sx - ox) + abs(old_sy - oy)
                for _, _, _, (ox, oy) in old_game_state["others"]
            ]
        )
        new_min_dist = min(
            [
                abs(new_sx - ox) + abs(new_sy - oy)
                for _, _, _, (ox, oy) in new_game_state["others"]
            ]
        )

        if new_min_dist < old_min_dist:
            reward += REWARD_CLOSER_TO_OPPONENT
        elif new_min_dist > old_min_dist:
            reward -= REWARD_CLOSER_TO_OPPONENT

    # 2. Features & Next Action Masking
    # Bug fix: previously called prepare_state(new_game_state) and threw
    # away its coin_distance with `_, _`, then never used
    # self.cached_coin_distance (the old state's value) either -- so the
    # BFS coin-distance computed every step never reached the reward.
    # Now captured once here and used for potential-based shaping below,
    # without calling prepare_state a second time.

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

    # Potential-based coin shaping: F(s,s') = gamma*Phi(s') - Phi(s)
    # (Ng, Harada & Russell 1999). Provably leaves the optimal policy
    # unchanged while giving dense per-step signal toward the nearest
    # coin, instead of relying purely on the sparse +8 pickup event.
    reward += COIN_POTENTIAL_WEIGHT * (
        GAMMA * _coin_potential(new_coin_distance)
        - _coin_potential(old_coin_distance)
    )

    action_idx = (
        ACTIONS.index(self_action)
        if (self_action is not None and self_action in ACTIONS)
        else 4
    )

    # 3. Store transition in replay buffer
    self.replay_buffer.push(
        old_features, action_idx, reward, new_features, is_terminal, next_mask
    )

    # 4. Update step counter & decay epsilon
    self.steps_done += 1
    self.epsilon = get_current_epsilon(self.steps_done)

    # 5. Optimize step
    if self.steps_done % 4 == 0:
        _optimize_model(self)


def end_of_round(self, last_game_state, last_action, events):
    game_events_occurred(self, last_game_state, last_action, None, events)

    # Save only the deployed evaluation weights (dqn_model.pt)
    model_path = os.path.join(os.path.dirname(__file__), "dqn_model.pt")
    torch.save(self.policy_net.state_dict(), model_path)


def _optimize_model(self):
    if len(self.replay_buffer) < BATCH_SIZE:
        return

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