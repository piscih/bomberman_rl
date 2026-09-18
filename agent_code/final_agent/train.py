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


# ============================================================
# HYPERPARAMETERS
# ============================================================

BATCH_SIZE = 64

GAMMA = 0.99

LR = 1e-4

MEMORY_SIZE = 100_000

TARGET_UPDATE_FREQ = 2500

# ------------------------------------------------------------
# Epsilon is now based on ENVIRONMENT STEPS rather than rounds.
# ------------------------------------------------------------

EPS_START = 1.0
EPS_END = 0.05

EPS_DECAY_STEPS = 100_000

# Optimize every environment step.
OPTIMIZE_EVERY = 1


# ============================================================
# REWARDS
# ============================================================

REWARD_KILL = 50.0

REWARD_COIN = 8.0

REWARD_CRATE = 0.6

REWARD_WAIT_PENALTY = -0.1

REWARD_DEATH = -100.0

# Coin-directed potential.
COIN_POTENTIAL_WEIGHT = 5.0


# ============================================================
# STATE / MODEL
# ============================================================

STATE_SHAPE = (
    12,
    17,
    17
)

ACTION_DIM = 6


# ============================================================
# FILE PATHS
# ============================================================

_THIS_DIR = os.path.dirname(
    __file__
)

CSV_LOG_PATH = os.path.join(
    _THIS_DIR,
    "training_log.csv"
)

STATE_META_PATH = os.path.join(
    _THIS_DIR,
    "training_state.json"
)

MODEL_PATH = os.path.join(
    _THIS_DIR,
    "dqn_model.pt"
)


# ============================================================
# CSV
# ============================================================

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


# ============================================================
# COIN POTENTIAL
# ============================================================

def _proximity_potential(distance):

    if distance is None:
        return 0.0

    return 1.0 / (
        1.0 + float(distance)
    )


# ============================================================
# EPSILON
# ============================================================

def get_current_epsilon(
    steps_done: int
) -> float:

    decay_progress = min(
        1.0,
        steps_done / EPS_DECAY_STEPS
    )

    return (
        EPS_START
        - (
            EPS_START - EPS_END
        ) * decay_progress
    )


# ============================================================
# REPLAY BUFFER
# ============================================================

class ReplayBuffer:

    def __init__(
        self,
        capacity: int,
        state_shape=STATE_SHAPE,
        action_dim=ACTION_DIM
    ):

        self.capacity = capacity

        self.states = np.zeros(
            (capacity,) + state_shape,
            dtype=np.float32
        )

        self.next_states = np.zeros(
            (capacity,) + state_shape,
            dtype=np.float32
        )

        self.actions = np.zeros(
            capacity,
            dtype=np.int64
        )

        self.rewards = np.zeros(
            capacity,
            dtype=np.float32
        )

        self.dones = np.zeros(
            capacity,
            dtype=np.float32
        )

        self.next_masks = np.zeros(
            (capacity, action_dim),
            dtype=np.float32
        )

        self.pos = 0
        self.size = 0

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        mask
    ):

        idx = self.pos

        self.states[idx] = state

        self.next_states[idx] = next_state

        self.actions[idx] = action

        self.rewards[idx] = reward

        self.dones[idx] = float(done)

        self.next_masks[idx] = mask

        self.pos = (
            self.pos + 1
        ) % self.capacity

        self.size = min(
            self.size + 1,
            self.capacity
        )

    def sample(
        self,
        batch_size: int
    ):

        indices = np.random.randint(
            0,
            self.size,
            size=batch_size
        )

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


# ============================================================
# SETUP TRAINING
# ============================================================

def setup_training(self):

    self.replay_buffer = ReplayBuffer(
        MEMORY_SIZE
    )

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

    self.optimizer_updates = 0

    # --------------------------------------------------------
    # Resume counters
    # --------------------------------------------------------

    if os.path.isfile(
        STATE_META_PATH
    ):

        try:

            with open(
                STATE_META_PATH,
                "r"
            ) as f:

                state = json.load(f)

            self.steps_done = int(
                state.get(
                    "steps_done",
                    0
                )
            )

            self.rounds_done = int(
                state.get(
                    "rounds_done",
                    0
                )
            )

            self.optimizer_updates = int(
                state.get(
                    "optimizer_updates",
                    0
                )
            )

            self.epsilon = get_current_epsilon(
                self.steps_done
            )

            self.logger.info(
                f"Resumed training progress: "
                f"round {self.rounds_done}, "
                f"step {self.steps_done}, "
                f"epsilon {self.epsilon:.4f}, "
                f"updates {self.optimizer_updates}"
            )

        except (
            json.JSONDecodeError,
            OSError,
            ValueError
        ) as e:

            self.logger.warning(
                f"Could not read "
                f"{STATE_META_PATH}: {e}. "
                f"Starting counters from zero."
            )

            self.steps_done = 0
            self.rounds_done = 0
            self.optimizer_updates = 0
            self.epsilon = EPS_START

    else:

        self.epsilon = EPS_START

    # --------------------------------------------------------
    # Target network
    # --------------------------------------------------------

    self.target_net.load_state_dict(
        self.policy_net.state_dict()
    )

    self.target_net.eval()

    # --------------------------------------------------------
    # Round statistics
    # --------------------------------------------------------

    self._round_reward = 0.0

    self._round_steps = 0

    self._round_coins = 0

    self._round_crates = 0

    self._round_kills = 0

    self._round_died = False

    self._round_suicide = False

    self._round_losses = []

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    write_header = not os.path.isfile(
        CSV_LOG_PATH
    )

    self._csv_file = open(
        CSV_LOG_PATH,
        "a",
        newline=""
    )

    self._csv_writer = csv.DictWriter(
        self._csv_file,
        fieldnames=CSV_FIELDNAMES
    )

    if write_header:

        self._csv_writer.writeheader()

        self._csv_file.flush()

    self.logger.info(
        f"Training initialized. "
        f"Round={self.rounds_done}, "
        f"Steps={self.steps_done}, "
        f"Epsilon={self.epsilon:.4f}"
    )


# ============================================================
# EVENTS
# ============================================================

def game_events_occurred(
    self,
    old_game_state,
    self_action,
    new_game_state,
    events
):

    if old_game_state is None:
        return

    reward = 0.0

    # --------------------------------------------------------
    # EVENT REWARDS
    # --------------------------------------------------------

    n_kills = events.count(
        "KILLED_OPPONENT"
    )

    reward += (
        REWARD_KILL
        * n_kills
    )

    n_coins = events.count(
        "COIN_COLLECTED"
    )

    reward += (
        REWARD_COIN
        * n_coins
    )

    n_crates = events.count(
        "CRATE_DESTROYED"
    )

    reward += (
        REWARD_CRATE
        * n_crates
    )

    # --------------------------------------------------------
    # WAIT PENALTY
    # --------------------------------------------------------

    if self_action == "WAIT":

        reward += REWARD_WAIT_PENALTY

    # --------------------------------------------------------
    # DEATH
    # --------------------------------------------------------

    died = (
        "KILLED_SELF" in events
        or
        "GOT_KILLED" in events
    )

    suicide = (
        "KILLED_SELF" in events
    )

    if died:

        reward += REWARD_DEATH

    # --------------------------------------------------------
    # OLD STATE
    # --------------------------------------------------------

    if getattr(
        self,
        "cached_features",
        None
    ) is not None:

        old_features = (
            self.cached_features
        )

    else:

        (
            old_features,
            _,
            _,
            _
        ) = prepare_state(
            old_game_state
        )

    old_coin_distance = getattr(
        self,
        "cached_coin_distance",
        None
    )

    # --------------------------------------------------------
    # NEW STATE
    # --------------------------------------------------------

    if new_game_state is not None:

        (
            new_features,
            next_mask,
            _,
            new_coin_distance
        ) = prepare_state(
            new_game_state
        )

        is_terminal = False

        # Safety guarantee:
        # every nonterminal next state must have at least
        # one legal action.
        if np.sum(next_mask) == 0:

            next_mask = np.zeros(
                ACTION_DIM,
                dtype=np.float32
            )

            next_mask[4] = 1.0

    else:

        # Terminal state does not bootstrap.
        new_features = np.zeros(
            STATE_SHAPE,
            dtype=np.float32
        )

        next_mask = np.zeros(
            ACTION_DIM,
            dtype=np.float32
        )

        new_coin_distance = None

        is_terminal = True

    # --------------------------------------------------------
    # COIN POTENTIAL
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # We DO NOT apply a terminal potential.
    #
    # Previously:
    #
    #     new potential = 0 on death
    #
    # which could create an additional artificial negative
    # reward on top of REWARD_DEATH.
    #
    # Now potential shaping only occurs between two real
    # game states.
    # --------------------------------------------------------

    if not is_terminal:

        old_potential = (
            _proximity_potential(
                old_coin_distance
            )
        )

        new_potential = (
            _proximity_potential(
                new_coin_distance
            )
        )

        reward += (
            COIN_POTENTIAL_WEIGHT
            * (
                GAMMA * new_potential
                - old_potential
            )
        )

    # --------------------------------------------------------
    # ACTION INDEX
    # --------------------------------------------------------

    if (
        self_action is not None
        and self_action in ACTIONS
    ):

        action_idx = ACTIONS.index(
            self_action
        )

    else:

        action_idx = 4

    # --------------------------------------------------------
    # REPLAY
    # --------------------------------------------------------

    self.replay_buffer.push(
        old_features,
        action_idx,
        reward,
        new_features,
        is_terminal,
        next_mask
    )

    # --------------------------------------------------------
    # ROUND METRICS
    # --------------------------------------------------------

    self._round_reward += reward

    self._round_steps += 1

    self._round_coins += n_coins

    self._round_crates += n_crates

    self._round_kills += n_kills

    self._round_died = (
        self._round_died
        or died
    )

    self._round_suicide = (
        self._round_suicide
        or suicide
    )

    # --------------------------------------------------------
    # GLOBAL STEP
    # --------------------------------------------------------

    self.steps_done += 1

    self.epsilon = get_current_epsilon(
        self.steps_done
    )

    # --------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------

    if (
        self.steps_done % OPTIMIZE_EVERY
        == 0
    ):

        loss = _optimize_model(
            self
        )

        if loss is not None:

            self._round_losses.append(
                loss
            )


# ============================================================
# END OF ROUND
# ============================================================

def end_of_round(
    self,
    last_game_state,
    last_action,
    events
):

    game_events_occurred(
        self,
        last_game_state,
        last_action,
        None,
        events
    )

    self.rounds_done += 1

    self.epsilon = get_current_epsilon(
        self.steps_done
    )

    if last_game_state is not None:

        round_number = last_game_state.get(
            "round",
            self.rounds_done
        )

        opponents_remaining = len(
            last_game_state.get(
                "others",
                []
            )
        )

    else:

        round_number = self.rounds_done

        opponents_remaining = 0

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    avg_reward = (
        self._round_reward
        / self._round_steps
        if self._round_steps
        else 0.0
    )

    avg_loss = (
        float(
            np.mean(
                self._round_losses
            )
        )
        if self._round_losses
        else float("nan")
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    self._csv_writer.writerow({

        "round":
            round_number,

        "steps":
            self._round_steps,

        "total_steps_done":
            self.steps_done,

        "epsilon":
            round(
                self.epsilon,
                4
            ),

        "total_reward":
            round(
                self._round_reward,
                4
            ),

        "avg_reward_per_step":
            round(
                avg_reward,
                4
            ),

        "coins_collected":
            self._round_coins,

        "crates_destroyed":
            self._round_crates,

        "kills":
            self._round_kills,

        "died":
            int(
                self._round_died
            ),

        "suicide":
            int(
                self._round_suicide
            ),

        "survived_round":
            int(
                not self._round_died
            ),

        "opponents_remaining":
            opponents_remaining,

        "buffer_size":
            len(
                self.replay_buffer
            ),

        "optimizer_updates":
            self.optimizer_updates,

        "avg_loss":
            avg_loss,
    })

    self._csv_file.flush()

    # --------------------------------------------------------
    # Reset round statistics
    # --------------------------------------------------------

    self._round_reward = 0.0

    self._round_steps = 0

    self._round_coins = 0

    self._round_crates = 0

    self._round_kills = 0

    self._round_died = False

    self._round_suicide = False

    self._round_losses = []

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    checkpoint = {
        "policy_net":
            self.policy_net.state_dict(),

        "target_net":
            self.target_net.state_dict(),

        "optimizer":
            self.optimizer.state_dict(),

        "steps_done":
            self.steps_done,

        "rounds_done":
            self.rounds_done,

        "optimizer_updates":
            self.optimizer_updates,
    }

    torch.save(
        checkpoint,
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Save training metadata
    # --------------------------------------------------------

    with open(
        STATE_META_PATH,
        "w"
    ) as f:

        json.dump(
            {
                "rounds_done":
                    self.rounds_done,

                "steps_done":
                    self.steps_done,

                "optimizer_updates":
                    self.optimizer_updates,
            },
            f
        )


# ============================================================
# OPTIMIZATION
# ============================================================

def _optimize_model(self):

    if len(
        self.replay_buffer
    ) < BATCH_SIZE:

        return None

    (
        states,
        actions,
        rewards,
        next_states,
        dones,
        next_masks
    ) = self.replay_buffer.sample(
        BATCH_SIZE
    )

    states_t = torch.from_numpy(
        states
    ).float().to(
        self.device
    )

    actions_t = torch.from_numpy(
        actions
    ).long().to(
        self.device
    ).unsqueeze(1)

    rewards_t = torch.from_numpy(
        rewards
    ).float().to(
        self.device
    ).unsqueeze(1)

    next_states_t = torch.from_numpy(
        next_states
    ).float().to(
        self.device
    )

    dones_t = torch.from_numpy(
        dones
    ).float().to(
        self.device
    ).unsqueeze(1)

    next_masks_t = torch.from_numpy(
        next_masks
    ).float().to(
        self.device
    )

    # --------------------------------------------------------
    # Current Q
    # --------------------------------------------------------

    q_values = (
        self.policy_net(states_t)
        .gather(
            1,
            actions_t
        )
    )

    # --------------------------------------------------------
    # Double DQN target
    # --------------------------------------------------------

    with torch.no_grad():

        next_q_policy = (
            self.policy_net(
                next_states_t
            )
        )

        # Apply the legal-action mask to the policy network.
        masked_next_q = (
            next_q_policy.clone()
        )

        masked_next_q[
            next_masks_t == 0.0
        ] = -1e9

        best_actions = (
            masked_next_q.argmax(
                dim=1,
                keepdim=True
            )
        )

        # Evaluate selected action using target network.
        next_q_target = (
            self.target_net(
                next_states_t
            )
            .gather(
                1,
                best_actions
            )
        )

        expected_q = (
            rewards_t
            + (
                1.0 - dones_t
            )
            * GAMMA
            * next_q_target
        )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss = nn.SmoothL1Loss()(
        q_values,
        expected_q
    )

    # --------------------------------------------------------
    # Gradient update
    # --------------------------------------------------------

    self.optimizer.zero_grad()

    loss.backward()

    nn.utils.clip_grad_norm_(
        self.policy_net.parameters(),
        max_norm=1.0
    )

    self.optimizer.step()

    self.optimizer_updates += 1

    # --------------------------------------------------------
    # Target network
    # --------------------------------------------------------

    if (
        self.optimizer_updates
        % TARGET_UPDATE_FREQ
        == 0
    ):

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    return loss.item()