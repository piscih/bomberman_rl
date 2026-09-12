import pickle
from typing import List

import events as e

from .callbacks import (
    state_to_features,
    get_valid_actions,
    get_safe_actions,
    ACTIONS,
)


# Custom events
MOVED_TOWARDS_TARGET = "MOVED_TOWARDS_TARGET"
ESCAPED_DANGER = "ESCAPED_DANGER"
PLACED_HIGH_VALUE_BOMB = "PLACED_HIGH_VALUE_BOMB"
PLACED_LOW_VALUE_BOMB = "PLACED_LOW_VALUE_BOMB"
PLACED_KILL_BOMB = "PLACED_KILL_BOMB"

OSCILLATION_PENALTY = "OSCILLATION_PENALTY"
USELESS_WAIT = "USELESS_WAIT"


OPPOSITES = {
    'UP': 'DOWN',
    'DOWN': 'UP',
    'LEFT': 'RIGHT',
    'RIGHT': 'LEFT',
}


# setup training parameters for the Q-learning agent, including learning rate, discount factor, and exploration parameters:
def setup_training(self):

    self.alpha = 0.15
    self.gamma = 0.95

    self.epsilon_decay = 0.9997
    self.epsilon_min = 0.1

    if not hasattr(self, 'epsilon'):
        self.epsilon = 1.0

    self.previous_features = None
    self.previous_action = None



# Reward function for the Q-learning agent, calculating the total reward based on the events that occurred during the game:
def reward_from_events(
    self,
    events: List[str]
):

    game_rewards = {

    e.COIN_COLLECTED: 150.0,

    e.CRATE_DESTROYED: 30.0,

    e.KILLED_OPPONENT: 200.0,

    e.KILLED_SELF: -1000.0,

    e.GOT_KILLED: -600.0,

    e.INVALID_ACTION: -10.0,

    ESCAPED_DANGER: 20.0,

    MOVED_TOWARDS_TARGET: 5.0,

    PLACED_KILL_BOMB: 40.0,

    PLACED_HIGH_VALUE_BOMB: 10.0,

    PLACED_LOW_VALUE_BOMB: -10.0,

    USELESS_WAIT: -2.0,

    OSCILLATION_PENALTY: -10.0,
}

    return sum(
        game_rewards.get(event, 0.0)
        for event in events
    )


# Game events
def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):

    if old_game_state is None:
        return

 
    old_features = getattr(
        self,
        'previous_features',
        None
    )

    previous_action = getattr(
        self,
        'previous_action',
        None
    )

    if old_features is None:

        previous_action = getattr(
            self,
            'last_action',
            'WAIT'
        )

        old_features = state_to_features(
            old_game_state,
            last_action=previous_action
        )


    new_features = state_to_features(
        new_game_state,
        last_action=self_action
    )

    if (
        old_features is None
        or
        new_features is None
    ):
        return

    (
        old_danger,
        old_danger_direction,
        old_target_direction,
        old_bomb_quality,
        old_blocked_up,
        old_blocked_down,
        old_blocked_left,
        old_blocked_right,
        old_last_action,
    ) = old_features

    if (
        old_danger == 'SAFE'
        and
        previous_action in OPPOSITES
        and
        self_action
        == OPPOSITES[previous_action]
    ):

        events.append(
            OSCILLATION_PENALTY
        )

    if (
        self_action == 'WAIT'
        and
        old_danger == 'SAFE'
    ):

        events.append(
            USELESS_WAIT
        )

    if self_action == 'BOMB':

        if old_bomb_quality == 3:

            events.append(
                PLACED_KILL_BOMB
            )

        elif old_bomb_quality == 2:

            events.append(
                PLACED_HIGH_VALUE_BOMB
            )

        elif old_bomb_quality == 1:

            events.append(
                PLACED_LOW_VALUE_BOMB
            )

    if (
        old_danger != 'SAFE'
        and
        old_danger_direction
        in [
            'UP',
            'RIGHT',
            'DOWN',
            'LEFT'
        ]
        and
        self_action
        == old_danger_direction
    ):

        events.append(
            ESCAPED_DANGER
        )


    if (
        old_danger == 'SAFE'
        and
        old_target_direction != 'NONE'
        and
        self_action
        == old_target_direction
    ):

        events.append(
            MOVED_TOWARDS_TARGET
        )

    reward = reward_from_events(
        self,
        events
    )

    if old_features not in self.q_table:

        self.q_table[old_features] = [
            0.0
            for _ in ACTIONS
        ]

    if new_features not in self.q_table:

        self.q_table[new_features] = [
            0.0
            for _ in ACTIONS
        ]

    valid_actions = get_valid_actions(
        new_game_state
    )

    safe_actions = get_safe_actions(
        new_game_state,
        valid_actions
    )


    if not safe_actions:

        safe_actions = valid_actions


    next_max = max(
        (
            self.q_table[
                new_features
            ][
                ACTIONS.index(action)
            ]

            for action in safe_actions
        ),
        default=0.0
    )

    action_index = ACTIONS.index(
        self_action
    )

    old_q = self.q_table[
        old_features
    ][action_index]

    target = (
        reward
        +
        self.gamma * next_max
    )


    self.q_table[
        old_features
    ][action_index] = (
        old_q
        +
        self.alpha
        *
        (
            target
            -
            old_q
        )
    )


    self.previous_features = new_features

    self.previous_action = self_action


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: List[str]
):

    if (
        last_game_state is None
        or
        last_action is None
    ):
        return

    last_features = state_to_features(
        last_game_state,
        last_action=last_action
    )

    if last_features is None:
        return

    reward = reward_from_events(
        self,
        events
    )

    if last_features not in self.q_table:

        self.q_table[last_features] = [
            0.0
            for _ in ACTIONS
        ]

    action_index = ACTIONS.index(
        last_action
    )

    old_q = self.q_table[
        last_features
    ][action_index]

    target = reward

    self.q_table[
        last_features
    ][action_index] = (
        old_q
        +
        self.alpha
        *
        (
            target
            -
            old_q
        )
    )


    self.epsilon = max(
        self.epsilon_min,
        self.epsilon * self.epsilon_decay
    )

    self.rounds_trained = (
        getattr(
            self,
            'rounds_trained',
            0
        )
        +
        1
    )


    with open(
        self.model_file,
        "wb"
    ) as file:

        pickle.dump(
            {
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'rounds_trained': (
                    self.rounds_trained
                ),
            },
            file,
        )

    if self.rounds_trained % 200 == 0:

        print(
            f"round "
            f"{self.rounds_trained}: "
            f"epsilon="
            f"{self.epsilon:.3f}, "
            f"states="
            f"{len(self.q_table)}"
        )

    self.previous_features = None
    self.previous_action = None