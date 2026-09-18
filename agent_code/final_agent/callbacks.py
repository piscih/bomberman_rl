import os
from collections import deque

import numpy as np
import torch

from .dqn_model import DQNResNet


# ============================================================
# ACTIONS / CONSTANTS
# ============================================================

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
N_ACTIONS = 6

ROWS = 17
COLS = 17

BLAST_RADIUS = 3
BOMB_TIMER = 4

DIRECTIONS = [
    (0, -1),   # UP
    (0, 1),    # DOWN
    (-1, 0),   # LEFT
    (1, 0),    # RIGHT
]

FALLBACK_EPS_START = 0.5


# ============================================================
# BASIC HELPERS
# ============================================================

def in_bounds(x, y):
    return 0 <= x < COLS and 0 <= y < ROWS


def get_bomb_positions(bombs):
    return {tuple(position) for position, _ in bombs}


def get_other_positions(others):
    return {tuple(player[3]) for player in others}


# ============================================================
# BOMB / DANGER MAP
# ============================================================

def calculate_blast_cells(field, bomb_position, radius=BLAST_RADIUS):
    bx, by = bomb_position

    blast = {(bx, by)}

    for dx, dy in DIRECTIONS:
        for step in range(1, radius + 1):
            nx = bx + dx * step
            ny = by + dy * step

            if not in_bounds(nx, ny):
                break

            if field[nx, ny] == -1:
                break

            blast.add((nx, ny))

            # Crates stop the explosion.
            if field[nx, ny] == 1:
                break

    return blast


def build_danger_map(game_state):
    """
    Computes the earliest known explosion time for every tile.

    Smaller value = more dangerous.

    inf = no known explosion.
    """

    field = game_state['field']
    bombs = game_state['bombs']

    danger_time = np.full(
        (COLS, ROWS),
        np.inf,
        dtype=np.float32
    )

    explosion_map = game_state.get(
        'explosion_map',
        np.zeros((COLS, ROWS))
    )

    # Currently exploding tiles are immediately dangerous.
    danger_time[explosion_map > 0] = 0.0

    bomb_data = {
        tuple(position): {
            'timer': float(timer),
            'blast': calculate_blast_cells(
                field,
                tuple(position)
            )
        }
        for position, timer in bombs
    }

    # --------------------------------------------------------
    # Chain reactions
    # --------------------------------------------------------

    changed = True

    while changed:
        changed = False

        for position, data in bomb_data.items():

            timer = data['timer']
            blast = data['blast']

            for other_position, other_data in bomb_data.items():

                if other_position == position:
                    continue

                if other_position in blast:

                    new_timer = min(
                        other_data['timer'],
                        timer
                    )

                    if new_timer < other_data['timer']:
                        other_data['timer'] = new_timer
                        changed = True

    # --------------------------------------------------------
    # Write danger times
    # --------------------------------------------------------

    for data in bomb_data.values():

        timer = max(0.0, data['timer'])

        for x, y in data['blast']:
            danger_time[x, y] = min(
                danger_time[x, y],
                timer
            )

    return danger_time


# ============================================================
# SAFE REACHABILITY
# ============================================================

def compute_safe_reachable(
    game_state,
    start_position=None,
    max_depth=20,
    danger_time=None
):
    field = game_state['field']

    if start_position is None:
        start_position = tuple(
            game_state['self'][3]
        )

    if danger_time is None:
        danger_time = build_danger_map(game_state)

    bomb_positions = get_bomb_positions(
        game_state['bombs']
    )

    other_positions = get_other_positions(
        game_state['others']
    )

    sx, sy = start_position

    queue = deque([
        (sx, sy, 0)
    ])

    visited = {
        (sx, sy)
    }

    safe_tiles = set()

    while queue:

        x, y, distance = queue.popleft()

        # Tile must remain safe until arrival.
        if danger_time[x, y] > distance:
            safe_tiles.add((x, y))

        if distance >= max_depth:
            continue

        for dx, dy in DIRECTIONS:

            nx = x + dx
            ny = y + dy

            if not in_bounds(nx, ny):
                continue

            if (nx, ny) in visited:
                continue

            if field[nx, ny] != 0:
                continue

            if (nx, ny) in bomb_positions:
                continue

            if (nx, ny) in other_positions:
                continue

            arrival_time = distance + 1

            if danger_time[nx, ny] <= arrival_time:
                continue

            visited.add((nx, ny))

            queue.append(
                (nx, ny, arrival_time)
            )

    return safe_tiles


def has_safe_escape(game_state):
    """
    Simulates placing a bomb at the player's current position
    and checks whether a safe escape path exists.

    This is only used to determine whether BOMB is legally
    available. It does NOT select a direction for the DQN.
    """

    field = game_state['field']

    sx, sy = game_state['self'][3]

    simulated_bombs = list(
        game_state['bombs']
    ) + [
        ((sx, sy), BOMB_TIMER)
    ]

    simulated_state = dict(
        game_state,
        bombs=simulated_bombs
    )

    danger_time = build_danger_map(
        simulated_state
    )

    bomb_positions = get_bomb_positions(
        simulated_bombs
    )

    other_positions = get_other_positions(
        game_state['others']
    )

    queue = deque([
        (sx, sy, 0)
    ])

    visited = {
        (sx, sy)
    }

    while queue:

        x, y, distance = queue.popleft()

        if danger_time[x, y] <= distance:
            continue

        # We have escaped the bomb location.
        if distance > 0:

            continuation_exists = False

            for dx, dy in DIRECTIONS:

                cx = x + dx
                cy = y + dy

                if not in_bounds(cx, cy):
                    continue

                if field[cx, cy] != 0:
                    continue

                if (cx, cy) in bomb_positions:
                    continue

                if (cx, cy) in other_positions:
                    continue

                if danger_time[cx, cy] > distance + 1:
                    continuation_exists = True
                    break

            if continuation_exists:
                return True

            if danger_time[x, y] > BOMB_TIMER:
                return True

        if distance >= BOMB_TIMER - 1:
            continue

        for dx, dy in DIRECTIONS:

            nx = x + dx
            ny = y + dy

            if not in_bounds(nx, ny):
                continue

            if (nx, ny) in visited:
                continue

            if field[nx, ny] != 0:
                continue

            if (nx, ny) in bomb_positions:
                continue

            if (nx, ny) in other_positions:
                continue

            arrival = distance + 1

            if danger_time[nx, ny] <= arrival:
                continue

            visited.add((nx, ny))

            queue.append(
                (nx, ny, arrival)
            )

    return False


# ============================================================
# COIN DISTANCE MAP
# ============================================================

def nearest_coin_distance(
    game_state,
    start_position=None
):
    """
    Builds a TRUE distance-to-nearest-reachable-coin map.

    Unlike the previous implementation, this is not a map of
    distance from the player.

    Each reachable tile contains:

        1 - distance_to_nearest_coin / 17

    Therefore:

        close to coin -> high value
        far from coin  -> lower value
        unreachable    -> -1

    The DQN receives this only as an input feature.
    It does not directly choose a movement direction.
    """

    field = game_state['field']

    coins = {
        tuple(c)
        for c in game_state['coins']
    }

    # No coins.
    if not coins:

        return (
            None,
            np.full(
                (COLS, ROWS),
                -1.0,
                dtype=np.float32
            )
        )

    if start_position is None:
        start_position = tuple(
            game_state['self'][3]
        )

    bomb_positions = get_bomb_positions(
        game_state['bombs']
    )

    other_positions = get_other_positions(
        game_state['others']
    )

    # --------------------------------------------------------
    # Multi-source BFS starting from all coins.
    # --------------------------------------------------------

    distance_map = np.full(
        (COLS, ROWS),
        np.inf,
        dtype=np.float32
    )

    queue = deque()

    for cx, cy in coins:

        if not in_bounds(cx, cy):
            continue

        # Coins are normally on walkable tiles.
        # They are still valid targets if they are currently
        # occupied by the player.
        if field[cx, cy] == 1:
            continue

        distance_map[cx, cy] = 0.0
        queue.append((cx, cy))

    while queue:

        x, y = queue.popleft()

        current_distance = distance_map[x, y]

        for dx, dy in DIRECTIONS:

            nx = x + dx
            ny = y + dy

            if not in_bounds(nx, ny):
                continue

            if distance_map[nx, ny] != np.inf:
                continue

            if field[nx, ny] != 0:
                continue

            # Existing bombs and opponents are treated as
            # temporarily non-walkable.
            if (nx, ny) in bomb_positions:
                continue

            if (nx, ny) in other_positions:
                continue

            distance_map[nx, ny] = (
                current_distance + 1
            )

            queue.append((nx, ny))

    sx, sy = start_position

    if np.isfinite(distance_map[sx, sy]):
        nearest_distance = int(
            distance_map[sx, sy]
        )
    else:
        nearest_distance = None

    # --------------------------------------------------------
    # Normalize map for the neural network.
    # --------------------------------------------------------

    normalized = np.full(
        (COLS, ROWS),
        -1.0,
        dtype=np.float32
    )

    reachable = np.isfinite(distance_map)

    normalized[reachable] = (
        1.0
        - np.minimum(
            distance_map[reachable] / 17.0,
            1.0
        )
    )

    return nearest_distance, normalized


# ============================================================
# STATE PREPARATION
# ============================================================

def prepare_state(game_state):

    danger_time = build_danger_map(
        game_state
    )

    self_position = tuple(
        game_state['self'][3]
    )

    safe_tiles = compute_safe_reachable(
        game_state,
        self_position,
        max_depth=12,
        danger_time=danger_time
    )

    coin_distance, coin_map = nearest_coin_distance(
        game_state
    )

    features, mask = extract_features_and_mask(
        game_state,
        precomputed_danger=danger_time,
        precomputed_safe_tiles=safe_tiles,
        precomputed_coin_map=coin_map
    )

    return (
        features,
        mask,
        danger_time,
        coin_distance
    )


# ============================================================
# FEATURES + ACTION MASK
# ============================================================

def extract_features_and_mask(
    game_state,
    precomputed_danger=None,
    precomputed_safe_tiles=None,
    precomputed_coin_map=None
):

    field = game_state['field']

    sx, sy = game_state['self'][3]

    bombs = game_state['bombs']
    others = game_state['others']
    coins = game_state['coins']

    bomb_positions = get_bomb_positions(
        bombs
    )

    other_positions = get_other_positions(
        others
    )

    danger_time = (
        build_danger_map(game_state)
        if precomputed_danger is None
        else precomputed_danger
    )

    safe_tiles = (
        compute_safe_reachable(
            game_state,
            (sx, sy),
            max_depth=12,
            danger_time=danger_time
        )
        if precomputed_safe_tiles is None
        else precomputed_safe_tiles
    )

    coin_distance_map = (
        nearest_coin_distance(game_state)[1]
        if precomputed_coin_map is None
        else precomputed_coin_map
    )

    # --------------------------------------------------------
    # ACTION MASK
    # --------------------------------------------------------

    mask = np.zeros(
        N_ACTIONS,
        dtype=np.float32
    )

    # Movement actions.
    for idx, (dx, dy) in enumerate(DIRECTIONS):

        nx = sx + dx
        ny = sy + dy

        if not in_bounds(nx, ny):
            continue

        if field[nx, ny] != 0:
            continue

        if (nx, ny) in bomb_positions:
            continue

        if (nx, ny) in other_positions:
            continue

        # Hard safety constraint.
        #
        # The agent needs to survive until it arrives on the
        # destination tile.
        if danger_time[nx, ny] <= 1:
            continue

        mask[idx] = 1.0

    # --------------------------------------------------------
    # WAIT
    # --------------------------------------------------------

    # WAIT is allowed when the current tile is not immediately
    # exploding and there is no movement option.
    #
    # This prevents an all-zero nonterminal mask.
    if danger_time[sx, sy] > 1:
        if len(safe_tiles) > 1:
            mask[4] = 1.0

    # If no movement action is available, WAIT becomes the
    # fallback action so that the network always has a valid
    # nonterminal action.
    if np.sum(mask[:4]) == 0:
        mask[4] = 1.0

    # --------------------------------------------------------
    # BOMB
    # --------------------------------------------------------

    has_bomb = bool(
        game_state['self'][2]
    )

    # Bombing is only useful for the current coin-learning
    # stage when crates actually exist.
    #
    # On empty / coin-heaven boards there is no reason to
    # learn bombing behavior.
    has_crates = bool(
        np.any(field == 1)
    )

    if (
        has_bomb
        and has_crates
        and (sx, sy) not in bomb_positions
        and has_safe_escape(game_state)
    ):
        mask[5] = 1.0

    # --------------------------------------------------------
    # 12-channel state
    # --------------------------------------------------------

    features = np.zeros(
        (12, COLS, ROWS),
        dtype=np.float32
    )

    # Channel 0: walls
    features[0] = (
        field == -1
    ).astype(np.float32)

    # Channel 1: crates
    features[1] = (
        field == 1
    ).astype(np.float32)

    # Channel 2: coins
    for cx, cy in coins:

        if in_bounds(cx, cy):
            features[2, cx, cy] = 1.0

    # Channel 3: player
    features[3, sx, sy] = 1.0

    # Channel 4: opponents
    for ox, oy in other_positions:

        if in_bounds(ox, oy):
            features[4, ox, oy] = 1.0

    # Channel 5: bombs
    for (bx, by), timer in bombs:

        if in_bounds(bx, by):
            features[5, bx, by] = 1.0

    # Channel 6: bomb timer
    for (bx, by), timer in bombs:

        if in_bounds(bx, by):
            features[6, bx, by] = np.clip(
                float(timer) / BOMB_TIMER,
                0.0,
                1.0
            )

    # Channel 7: active explosions
    explosion_map = game_state.get(
        'explosion_map',
        np.zeros((COLS, ROWS))
    )

    features[7] = (
        explosion_map > 0
    ).astype(np.float32)

    # Channel 8: danger intensity
    finite = np.isfinite(
        danger_time
    )

    danger_feature = np.zeros(
        (COLS, ROWS),
        dtype=np.float32
    )

    danger_feature[finite] = np.clip(
        1.0 - (
            danger_time[finite] / BOMB_TIMER
        ),
        0.0,
        1.0
    )

    features[8] = danger_feature

    # Channel 9: bomb availability
    if has_bomb:
        features[9].fill(1.0)

    # Channel 10: safe reachable tiles
    for x, y in safe_tiles:
        features[10, x, y] = 1.0

    # Channel 11: distance-to-nearest-coin
    features[11] = coin_distance_map

    return features, mask


# ============================================================
# SETUP
# ============================================================

def setup(self):

    self.device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    self.policy_net = DQNResNet(
        input_channels=12,
        num_actions=6
    ).to(self.device)

    self.steps_done = 0

    self.epsilon = (
        FALLBACK_EPS_START
        if getattr(self, "train", False)
        else 0.0
    )

    model_path = os.path.join(
        os.path.dirname(__file__),
        "dqn_model.pt"
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    if not getattr(self, "train", False):

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Missing trained model for evaluation: "
                f"{model_path}"
            )

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        # Support both the new checkpoint format and an older
        # raw state_dict checkpoint.
        if (
            isinstance(checkpoint, dict)
            and "policy_net" in checkpoint
        ):
            self.policy_net.load_state_dict(
                checkpoint["policy_net"]
            )
        else:
            self.policy_net.load_state_dict(
                checkpoint
            )

        self.logger.info(
            "Loaded dqn_model.pt for evaluation."
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    else:

        if os.path.isfile(model_path):

            checkpoint = torch.load(
                model_path,
                map_location=self.device
            )

            if (
                isinstance(checkpoint, dict)
                and "policy_net" in checkpoint
            ):
                self.policy_net.load_state_dict(
                    checkpoint["policy_net"]
                )
            else:
                self.policy_net.load_state_dict(
                    checkpoint
                )

            self.logger.info(
                "Resumed training from existing dqn_model.pt."
            )

        else:

            self.logger.info(
                "No existing checkpoint found -- "
                "starting training from scratch."
            )

    self.policy_net.eval()

    # --------------------------------------------------------
    # Cached transition information
    # --------------------------------------------------------

    self.cached_features = None
    self.cached_mask = None
    self.cached_danger = None
    self.cached_coin_distance = None


# ============================================================
# ACTION
# ============================================================

def act(self, game_state: dict) -> str:

    features, mask, danger_time, coin_distance = (
        prepare_state(game_state)
    )

    # Cache the exact state used to choose the action.
    self.cached_features = features
    self.cached_mask = mask
    self.cached_danger = danger_time
    self.cached_coin_distance = coin_distance

    # --------------------------------------------------------
    # Epsilon-greedy exploration
    # --------------------------------------------------------

    if getattr(self, "train", False):

        epsilon = getattr(
            self,
            "epsilon",
            FALLBACK_EPS_START
        )

        if np.random.random() < epsilon:

            valid_indices = np.where(
                mask == 1.0
            )[0]

            if len(valid_indices) > 0:

                return ACTIONS[
                    np.random.choice(
                        valid_indices
                    )
                ]

            return 'WAIT'

    # --------------------------------------------------------
    # DQN action
    # --------------------------------------------------------

    with torch.no_grad():

        state_t = torch.tensor(
            features,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        q_values = (
            self.policy_net(state_t)
            .squeeze(0)
            .cpu()
            .numpy()
        )

    # Hard safety / legality mask.
    q_values[
        mask == 0.0
    ] = -1e9

    action_idx = int(
        np.argmax(q_values)
    )

    return ACTIONS[action_idx]