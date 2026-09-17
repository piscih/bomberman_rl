import csv
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .callbacks import (
    ACTIONS,
    prepare_state,
)
from .dqn_model import DQNResNet


REWARD_KILL = 50.0
REWARD_COIN = 3.0
REWARD_CRATE = 1.2
REWARD_STEP = -0.05
REWARD_DEATH = -80.0
REWARD_BOMB_NEAR_CRATE = 0.5
COIN_POTENTIAL_SCALE = 0.05
REWARD_ESCAPE_DANGER = 0.30

GAMMA = 0.99
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
BUFFER_SIZE = 50000
TARGET_UPDATE_FREQ = 2000 

EPS_START = 0.50
EPS_END = 0.10
EPS_DECAY_STEPS = 200000


LOG_INTERVAL = 100
BEST_WINDOW = 100
SAVE_INTERVAL = 100


FINAL_TRAINING_ROUNDS = 10000


def setup_training(self):

    self.replay_buffer = deque(
        maxlen=BUFFER_SIZE
    )

    self.target_net = DQNResNet(
        input_channels=12,
        num_actions=6
    ).to(self.device)

    self.target_net.load_state_dict(
        self.policy_net.state_dict()
    )

    self.target_net.eval()

    self.optimizer = optim.Adam(
        self.policy_net.parameters(),
        lr=LEARNING_RATE
    )

    self.steps_done = 0
    self.epsilon = EPS_START
    self.training_round = 0

    self.episode_reward = 0.0
    self.episode_coins = 0
    self.episode_kills = 0
    self.episode_deaths = 0
    self.episode_crates = 0
    self.episode_invalid = 0
    self.episode_steps = 0

    self.recent_rewards = deque(
        maxlen=BEST_WINDOW
    )

    self.recent_coins = deque(
        maxlen=BEST_WINDOW
    )

    self.loss_history = deque(
        maxlen=1000
    )

    self.cached_features = None
    self.cached_mask = None
    self.cached_danger = None
    self.cached_coin_distance = None

    self.log_path = os.path.join(
        os.path.dirname(__file__),
        "training_log.csv"
    )

    if not os.path.isfile(self.log_path):
        with open(
            self.log_path,
            "w",
            newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow([
                "round",
                "steps",
                "reward",
                "avg_reward",
                "coins",
                "avg_coins",
                "kills",
                "deaths",
                "crates",
                "invalid",
                "epsilon",
                "loss"
            ])

    self.logger.info(
        "Training initialized from scratch. "
        "No checkpoint restoration."
    )


def episode_trigger(self):

    self.cached_features = None
    self.cached_mask = None
    self.cached_danger = None
    self.cached_coin_distance = None

    self.episode_reward = 0.0
    self.episode_coins = 0
    self.episode_kills = 0
    self.episode_deaths = 0
    self.episode_crates = 0
    self.episode_invalid = 0
    self.episode_steps = 0

def update_epsilon(self):

    progress = min(
        self.steps_done / EPS_DECAY_STEPS,
        1.0
    )

    self.epsilon = (
        EPS_START
        - progress * (EPS_START - EPS_END)
    )

def calculate_reward(
    self,
    old_game_state,
    new_game_state,
    self_action,
    events,
    old_coin_distance=None,
    new_coin_distance=None,
    old_danger=None,
    new_danger=None
):

    reward = REWARD_STEP

    if "KILLED_OPPONENT" in events:
        reward += REWARD_KILL
        self.episode_kills += 1

    if "COIN_COLLECTED" in events:
        reward += REWARD_COIN
        self.episode_coins += 1

    if "CRATE_DESTROYED" in events:
        reward += REWARD_CRATE
        self.episode_crates += 1

    if (
        "KILLED_SELF" in events
        or "GOT_KILLED" in events
    ):
        reward += REWARD_DEATH
        self.episode_deaths += 1

    if "INVALID_ACTION" in events:
        self.episode_invalid += 1

    if (
        self_action == "BOMB"
        and old_game_state is not None
    ):

        sx, sy = old_game_state["self"][3]
        field = old_game_state["field"]

        adjacent_crates = 0

        for dx, dy in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]:
            nx = sx + dx
            ny = sy + dy

            if (
                0 <= nx < 17
                and 0 <= ny < 17
                and field[nx, ny] == 1
            ):
                adjacent_crates += 1

        reward += (
            REWARD_BOMB_NEAR_CRATE
            * adjacent_crates
        )

    if (
        old_coin_distance is not None
        and new_coin_distance is not None
    ):

        old_phi = (
            -COIN_POTENTIAL_SCALE
            * old_coin_distance
        )

        new_phi = (
            -COIN_POTENTIAL_SCALE
            * new_coin_distance
        )

        reward += (
            GAMMA * new_phi
            - old_phi
        )

    if (
        old_danger is not None
        and new_danger is not None
        and old_game_state is not None
        and new_game_state is not None
    ):

        ox, oy = old_game_state["self"][3]
        nx, ny = new_game_state["self"][3]

        old_danger_val = old_danger[ox, oy]
        new_danger_val = new_danger[nx, ny]

        if (
            old_danger_val <= 1
            and new_danger_val > 1
        ):
            reward += REWARD_ESCAPE_DANGER

    return reward

def game_events_occurred(
    self,
    old_game_state,
    self_action,
    new_game_state,
    events
):

    if old_game_state is None or new_game_state is None:
        return

    if (
        self.cached_features is not None
        and self.cached_mask is not None
        and self.cached_danger is not None
        and self.cached_coin_distance is not None
    ):

        old_features = self.cached_features
        old_mask = self.cached_mask
        old_danger = self.cached_danger
        old_coin_distance = self.cached_coin_distance

    else:

        (
            old_features,
            old_mask,
            old_danger,
            old_coin_distance
        ) = prepare_state(old_game_state)


    (
        new_features,
        new_mask,
        new_danger,
        new_coin_distance
    ) = prepare_state(new_game_state)


    self.cached_features = new_features
    self.cached_mask = new_mask
    self.cached_danger = new_danger
    self.cached_coin_distance = new_coin_distance


    reward = calculate_reward(
        self,
        old_game_state,
        new_game_state,
        self_action,
        events,
        old_coin_distance=old_coin_distance,
        new_coin_distance=new_coin_distance,
        old_danger=old_danger,
        new_danger=new_danger
    )

    action_idx = ACTIONS.index(self_action)

    self.replay_buffer.append((
        old_features,
        action_idx,
        reward,
        new_features,
        new_mask,
        False
    ))

    self.episode_reward += reward
    self.episode_steps += 1
    self.steps_done += 1

    update_epsilon(self)


    loss = optimize_model(self)

    if loss is not None:
        self.loss_history.append(loss)


def optimize_model(self):

    if len(self.replay_buffer) < BATCH_SIZE:
        return None

    batch = random.sample(
        self.replay_buffer,
        BATCH_SIZE
    )

    (
        states,
        actions,
        rewards,
        next_states,
        next_masks,
        dones
    ) = zip(*batch)

    states_t = torch.as_tensor(
        np.asarray(states),
        dtype=torch.float32,
        device=self.device
    )

    actions_t = torch.as_tensor(
        actions,
        dtype=torch.long,
        device=self.device
    ).unsqueeze(1)

    rewards_t = torch.as_tensor(
        rewards,
        dtype=torch.float32,
        device=self.device
    )

    next_states_t = torch.as_tensor(
        np.asarray(next_states),
        dtype=torch.float32,
        device=self.device
    )

    next_masks_t = torch.as_tensor(
        np.asarray(next_masks),
        dtype=torch.bool,
        device=self.device
    )

    dones_t = torch.as_tensor(
        dones,
        dtype=torch.float32,
        device=self.device
    )

    current_q = (
        self.policy_net(states_t)
        .gather(1, actions_t)
        .squeeze(1)
    )


    with torch.no_grad():

        next_policy_q = self.policy_net(
            next_states_t
        )

        valid_action_exists = (
            next_masks_t.any(
                dim=1,
                keepdim=True
            )
        )

        min_val = torch.finfo(next_policy_q.dtype).min
        next_policy_q = next_policy_q.masked_fill(
            ~next_masks_t,
            min_val
        )

        best_next_actions = (
            next_policy_q.argmax(
                dim=1,
                keepdim=True
            )
        )

        next_target_q = (
            self.target_net(next_states_t)
            .gather(
                1,
                best_next_actions
            )
            .squeeze(1)
        )

        next_target_q = torch.where(
            valid_action_exists.squeeze(1),
            next_target_q,
            torch.zeros_like(next_target_q)
        )

        bootstrap_valid = (
            (1.0 - dones_t)
            * valid_action_exists.squeeze(1).float()
        )

        target_q = (
            rewards_t
            +
            bootstrap_valid
            * GAMMA
            * next_target_q
        )

    loss = F.smooth_l1_loss(
        current_q,
        target_q
    )

    self.optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        self.policy_net.parameters(),
        max_norm=1.0
    )

    self.optimizer.step()

    if self.steps_done % TARGET_UPDATE_FREQ == 0:
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    return float(loss.item())

    return float(loss.item())

def end_of_round(
    self,
    last_game_state,
    last_action,
    events
):

    if last_game_state is None or last_action is None:
        return

    self.training_round += 1

    if (
        self.cached_features is not None
        and self.cached_danger is not None
        and self.cached_coin_distance is not None
    ):

        old_features = self.cached_features
        old_danger = self.cached_danger
        old_coin_distance = self.cached_coin_distance

    else:

        (
            old_features,
            _,
            old_danger,
            old_coin_distance
        ) = prepare_state(last_game_state)

    reward = calculate_reward(
        self,
        last_game_state,
        None,
        last_action,
        events,
        old_coin_distance=old_coin_distance,
        new_coin_distance=None,
        old_danger=old_danger,
        new_danger=None
    )

    action_idx = ACTIONS.index(last_action)

    terminal_state = np.zeros_like(
        old_features
    )

    terminal_mask = np.zeros(
        6,
        dtype=np.bool_
    )

    self.replay_buffer.append((
        old_features,
        action_idx,
        reward,
        terminal_state,
        terminal_mask,
        True
    ))

    self.episode_reward += reward
    self.episode_steps += 1
    self.steps_done += 1

    update_epsilon(self)

    loss = optimize_model(self)

    if loss is not None:
        self.loss_history.append(loss)

    self.recent_rewards.append(
        self.episode_reward
    )

    self.recent_coins.append(
        self.episode_coins
    )

    average_reward = float(
        np.mean(self.recent_rewards)
    )

    average_coins = float(
        np.mean(self.recent_coins)
    )

    average_loss = (
        float(np.mean(self.loss_history))
        if self.loss_history
        else 0.0
    )

    model_dir = os.path.dirname(__file__)

    final_model_path = os.path.join(
        model_dir,
        "dqn_model.pt"
    )

    should_save = (
        self.training_round % SAVE_INTERVAL == 0
        or self.training_round >= FINAL_TRAINING_ROUNDS
    )

    if should_save:

        torch.save(
            self.policy_net.state_dict(),
            final_model_path
        )

        self.logger.info(
            "MODEL SAVED | "
            f"round={self.training_round} | "
            f"path={final_model_path}"
        )

    with open(
        self.log_path,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            self.training_round,
            self.episode_steps,
            round(self.episode_reward, 4),
            round(average_reward, 4),
            self.episode_coins,
            round(average_coins, 4),
            self.episode_kills,
            self.episode_deaths,
            self.episode_crates,
            self.episode_invalid,
            round(self.epsilon, 6),
            round(average_loss, 6)
        ])

    if self.training_round % LOG_INTERVAL == 0:

        self.logger.info(
            "TRAIN | "
            f"round={self.training_round} | "
            f"reward={self.episode_reward:.2f} | "
            f"avg_reward={average_reward:.2f} | "
            f"coins={self.episode_coins} | "
            f"avg_coins={average_coins:.2f} | "
            f"kills={self.episode_kills} | "
            f"deaths={self.episode_deaths} | "
            f"crates={self.episode_crates} | "
            f"invalid={self.episode_invalid} | "
            f"epsilon={self.epsilon:.3f} | "
            f"buffer={len(self.replay_buffer)} | "
            f"loss={average_loss:.4f}"
        )

    self.cached_features = None
    self.cached_mask = None
    self.cached_danger = None
    self.cached_coin_distance = None

    self.episode_reward = 0.0
    self.episode_coins = 0
    self.episode_kills = 0
    self.episode_deaths = 0
    self.episode_crates = 0
    self.episode_invalid = 0
    self.episode_steps = 0