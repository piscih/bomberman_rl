from collections import deque, namedtuple
import os
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import events as e

from .callbacks import (
    ACTIONS,
    DQN,
    game_state_to_vector,
    get_safe_action_indices,
    state_to_features,
    get_coin_target_and_distance,
)

USEFUL_BOMB = "USEFUL_BOMB"
ATTACK_BOMB = "ATTACK_BOMB"
USELESS_BOMB = "USELESS_BOMB"

MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"


# DQN hyperparameters

GAMMA = 0.9

LEARNING_RATE = 1e-3

BATCH_SIZE = 64

REPLAY_BUFFER_SIZE = 20_000
MIN_REPLAY_SIZE = 500

TARGET_UPDATE_INTERVAL = 500

EPSILON_START = 0.15
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.997



# Replay transition

Transition = namedtuple(
    "Transition",
    (
        "state",
        "action",
        "reward",
        "next_state",
        "next_legal_mask",
        "done",
    ),
)




def setup_training(self):
    """
    Initialise DQN training.
    """

    self.logger.info("Setting up DQN training.")

    self.epsilon = EPSILON_START

    self.replay_buffer = deque(
        maxlen=REPLAY_BUFFER_SIZE
    )

    # Target network

    self.target_net = DQN().to(self.device)

    self.target_net.load_state_dict(
        self.policy_net.state_dict()
    )

    self.target_net.eval()

    # Optimizer

    self.optimizer = optim.Adam(
        self.policy_net.parameters(),
        lr=LEARNING_RATE,
    )

    # Huber loss 
    self.loss_function = nn.SmoothL1Loss()

    # Number of gradient updates performed.
    self.training_steps = 0

    # Number of completed episodes.
    self.episode = 0

def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    if old_game_state is None:
        return

    state = game_state_to_vector(old_game_state)
    next_state = game_state_to_vector(new_game_state)

    action = ACTIONS.index(self_action)

    # Coin progress shaping

    _, old_coin_distance = get_coin_target_and_distance(
        old_game_state
    )

    _, new_coin_distance = get_coin_target_and_distance(
        new_game_state
    )

    if e.COIN_COLLECTED not in events:

        if (
            old_coin_distance is not None
            and new_coin_distance is not None
        ):

            if new_coin_distance < old_coin_distance:
                events.append(MOVED_TOWARDS_COIN)

            elif new_coin_distance > old_coin_distance:
                events.append(MOVED_AWAY_FROM_COIN)

    if e.BOMB_DROPPED in events:

        features = state_to_features(old_game_state)

        (
            local_tiles,
            target_type,
            target_direction,
            opponent_direction,
            danger_level,
            escape_direction,
            bomb_available,
            bomb_value,
            opponent_in_blast,
            bomb_escape_possible,
            normalized_coin_distance,
        ) = features

        if opponent_in_blast:
            events.append(ATTACK_BOMB)

        elif bomb_value > 0:
            events.append(USEFUL_BOMB)

        else:
            events.append(USELESS_BOMB)
    reward = reward_from_events(self, events)

    next_legal_mask = np.zeros(
        len(ACTIONS),
        dtype=np.bool_,
    )

    if new_game_state is not None:
        legal_actions = get_safe_action_indices(
            new_game_state
        )

        next_legal_mask[legal_actions] = True

    self.replay_buffer.append(
        Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            next_legal_mask=next_legal_mask,
            done=False,
        )
    )

    optimize_model(self)

def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: List[str],
):
    """
    Store the terminal transition, perform one final training update,
    decay epsilon, and save the model.
    """

    if last_game_state is not None:

        state = game_state_to_vector(last_game_state)

        action = ACTIONS.index(last_action)

        reward = reward_from_events(
            self,
            events,
        )

        # Terminal states have no legal next actions.
        terminal_mask = np.zeros(
            len(ACTIONS),
            dtype=np.bool_,
        )

        self.replay_buffer.append(
            Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=None,
                next_legal_mask=terminal_mask,
                done=True,
            )
        )

        optimize_model(self)

    # Episode bookkeeping

    self.episode += 1

    self.epsilon = max(
        EPSILON_MIN,
        self.epsilon * EPSILON_DECAY,
    )

    # Save current policy network

    model_path = os.path.join(
        os.path.dirname(__file__),
        "dqn_model.pt",
    )

    torch.save(
        {
            "policy_net": self.policy_net.state_dict(),
            "epsilon": self.epsilon,
            "episode": self.episode,
            "training_steps": self.training_steps,
            "replay_size": len(self.replay_buffer),
        },
        model_path,
    )

    

def optimize_model(self):
    """
    Sample a random minibatch from replay memory and perform
    one DQN gradient update.
    """

    # Wait until enough experience has been collected.
    if len(self.replay_buffer) < MIN_REPLAY_SIZE:
        return

    batch = random.sample(
        self.replay_buffer,
        BATCH_SIZE,
    )

    # Convert current states to tensors

    states = torch.tensor(
        np.stack([
            transition.state
            for transition in batch
        ]),
        dtype=torch.float32,
        device=self.device,
    )

    actions = torch.tensor(
        [
            transition.action
            for transition in batch
        ],
        dtype=torch.long,
        device=self.device,
    )

    rewards = torch.tensor(
        [
            transition.reward
            for transition in batch
        ],
        dtype=torch.float32,
        device=self.device,
    )

    dones = torch.tensor(
        [
            transition.done
            for transition in batch
        ],
        dtype=torch.bool,
        device=self.device,
    )

    # Current Q(s, a)

    all_q_values = self.policy_net(states)

    current_q_values = all_q_values.gather(
        1,
        actions.unsqueeze(1),
    ).squeeze(1)

    # Q-values for next states

    next_q_values = torch.zeros(
        BATCH_SIZE,
        dtype=torch.float32,
        device=self.device,
    )

    non_terminal_indices = [
        i
        for i, transition in enumerate(batch)
        if not transition.done
    ]

    if non_terminal_indices:

        next_states = torch.tensor(
            np.stack([
                batch[i].next_state
                for i in non_terminal_indices
            ]),
            dtype=torch.float32,
            device=self.device,
        )

        next_legal_masks = torch.tensor(
            np.stack([
                batch[i].next_legal_mask
                for i in non_terminal_indices
            ]),
            dtype=torch.bool,
            device=self.device,
        )

        with torch.no_grad():

            # Vanilla DQN:
            # target network both selects and evaluates
            # the maximum next-state action.
            target_q_values = self.target_net(
                next_states
            )

            # Illegal actions must not participate in the maximum.
            target_q_values = target_q_values.masked_fill(
                ~next_legal_masks,
                float("-inf"),
            )

            max_next_q_values = target_q_values.max(
                dim=1
            ).values

        index_tensor = torch.tensor(
            non_terminal_indices,
            dtype=torch.long,
            device=self.device,
        )

        next_q_values[index_tensor] = max_next_q_values

    # Bellman target

    targets = (
        rewards
        + GAMMA
        * next_q_values
        * (~dones).float()
    )

    # Loss

    loss = self.loss_function(
        current_q_values,
        targets,
    )

    # Backpropagation

    self.optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        self.policy_net.parameters(),
        max_norm=10.0,
    )

    self.optimizer.step()

    self.training_steps += 1

    # Periodically synchronize target network

    if (
        self.training_steps
        % TARGET_UPDATE_INTERVAL
        == 0
    ):
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

def reward_from_events(
    self,
    events: List[str],
) -> float:
    

    game_rewards = {

        e.COIN_COLLECTED: 1.0,

        e.COIN_FOUND: 0.5,

        MOVED_TOWARDS_COIN: 0.05,
        MOVED_AWAY_FROM_COIN: -0.02,

        e.CRATE_DESTROYED: 0.2,

        USEFUL_BOMB: 0.1,
        ATTACK_BOMB: 0.15,
        USELESS_BOMB: -0.1,

        e.KILLED_OPPONENT: 5.0,

        e.SURVIVED_ROUND: 1.0,

        e.INVALID_ACTION: -1.0,

        e.KILLED_SELF: -5.0,

        e.GOT_KILLED: -5.0,
    }

    return sum(
        game_rewards.get(event, 0.0)
        for event in events
    )