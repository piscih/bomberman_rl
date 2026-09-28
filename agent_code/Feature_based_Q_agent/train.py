from collections import namedtuple, deque

import os
import pickle
from typing import List
import numpy as np

import events as e
from .callbacks import (
    ACTIONS,
    EPSILON_MIN,
    EPSILON_DECAY,
    state_to_features,
    get_q_values,
    get_legal_action_indices,
)




def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    # self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    """
    Initialise training.
    """

    self.logger.info("Setting up Q-learning training.")
    self.transitions = []

    # Count completed training episodes.
    self.episode = 0

    # Save a checkpoint every N episodes.
    self.checkpoint_interval = 100


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):

    if old_game_state is None:
        return

    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)


    reward = reward_from_events(self, events)

    old_q_values = get_q_values(self, old_state)
    action_index = ACTIONS.index(self_action)

    if new_state is None:
        target = reward

    else:
        new_q_values = get_q_values(self, new_state)

        legal_next_actions = get_legal_action_indices(
            new_game_state
        )

        max_next_q = max(
            new_q_values[i]
            for i in legal_next_actions
        )

        target = (
            reward
            + self.gamma * max_next_q
        )
        
    old_q_values[action_index] += (
        self.alpha
        * (target - old_q_values[action_index])
    )




def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    if last_game_state is not None:

        state = state_to_features(last_game_state)

        reward = reward_from_events(self, events)

        q_values = get_q_values(self, state)

        action_index = ACTIONS.index(last_action)

        # Terminal-state update.
        q_values[action_index] += (
            self.alpha
            * (reward - q_values[action_index])
        )

    # Decay exploration.
    self.epsilon = max(
        EPSILON_MIN,
        self.epsilon * EPSILON_DECAY,
    )

    self.episode += 1

    # Save latest model

    model_path = os.path.join(
        os.path.dirname(__file__),
        "my-saved-model.pkl",
    )

    with open(model_path, "wb") as file:
        pickle.dump(
            {
                "q_table": self.model,
                "epsilon": self.epsilon,
                "episode": self.episode,
            },
            file,
        )

    
    # Periodic checkpoint

    if self.episode % self.checkpoint_interval == 0:

        checkpoint_dir = os.path.join(
            os.path.dirname(__file__),
            "checkpoints",
        )

        os.makedirs(
            checkpoint_dir,
            exist_ok=True,
        )

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_{self.episode}.pkl",
        )

        with open(checkpoint_path, "wb") as file:
            pickle.dump(
                {
                    "q_table": self.model,
                    "epsilon": self.epsilon,
                    "episode": self.episode,
                },
                file,
            )



def reward_from_events(self, events: List[str]) -> int:
    

    game_rewards = {

        # Main objective
        e.COIN_COLLECTED: 1.0,

        # Intermediate progress toward hidden coins
        e.COIN_FOUND: 0.5,

        # Useful consequence of bombing
        e.CRATE_DESTROYED: 0.2,

        # Small cost for using a bomb.
        e.BOMB_DROPPED: -0.05,

        e.KILLED_OPPONENT: 5.0,

        # Survival
        e.SURVIVED_ROUND: 1.0,

        # Bad behavior
        e.INVALID_ACTION: -1.0,

        # Death
        e.KILLED_SELF: -5.0,
        e.GOT_KILLED: -5.0,


    }

    return sum(
        game_rewards.get(event, 0.0)
        for event in events
    )
