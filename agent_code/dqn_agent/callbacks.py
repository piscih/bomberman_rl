from collections import deque
import os
import random

import numpy as np
import torch
import torch.nn as nn

import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

INPUT_DIM = 51
HIDDEN_DIM = 64
OUTPUT_DIM = len(ACTIONS)


class DQN(nn.Module):
    """
    Small fully connected Q-network.

    Input:
        51-dimensional engineered state vector

    Output:
        Q-value for each of the 6 actions
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )

    def forward(self, x):
        return self.network(x)


def setup(self):
    """
    Set up the DQN agent.

    If a trained model exists, load its policy weights during both
    training and evaluation. This allows curriculum stages to continue
    from the previously learned network.
    """

    torch.set_num_threads(1)

    self.device = torch.device("cpu")

    self.policy_net = DQN().to(self.device)

    self.safety_fallback_count = 0
    self.danger_action_count = 0

    model_path = os.path.join(
        os.path.dirname(__file__),
        "dqn_model.pt",
    )

    if os.path.isfile(model_path):
        self.logger.info("Loading existing DQN model.")

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
        )

        self.policy_net.load_state_dict(
            checkpoint["policy_net"]
        )

    else:
        self.logger.info(
            "No existing DQN model found. Starting from scratch."
        )

    if self.train:
        self.policy_net.train()
    else:
        self.policy_net.eval()

def get_occupied_positions(game_state):
    bomb_positions = {
        tuple(position)
        for position, timer in game_state["bombs"]
    }

    opponent_positions = {
        tuple(other[3])
        for other in game_state["others"]
    }

    return bomb_positions, opponent_positions


def is_walkable(game_state, x, y):
    field = game_state["field"]

    if x < 0 or y < 0:
        return False

    if x >= field.shape[0] or y >= field.shape[1]:
        return False

    if field[x, y] != 0:
        return False

    bomb_positions, opponent_positions = get_occupied_positions(game_state)

    if (x, y) in bomb_positions:
        return False

    if (x, y) in opponent_positions:
        return False

    return True

def get_coin_target_and_distance(game_state):
    """
    Return:
        first_direction: first move on a shortest path to a reachable coin
        distance: shortest-path distance to that coin

    Returns:
        (None, None) if no visible coin is reachable.
    """

    field = game_state["field"]
    coins = game_state["coins"]

    if not coins:
        return None, None

    _, _, _, (start_x, start_y) = game_state["self"]

    coin_positions = set(coins)

    directions = [
        ((0, -1), "UP"),
        ((1, 0), "RIGHT"),
        ((0, 1), "DOWN"),
        ((-1, 0), "LEFT"),
    ]

    queue = deque([
        (start_x, start_y, None, 0)
    ])

    visited = {
        (start_x, start_y)
    }

    while queue:

        cx, cy, first_direction, distance = queue.popleft()

        if (cx, cy) in coin_positions:
            return first_direction, distance

        for (dx, dy), direction_name in directions:

            nx = cx + dx
            ny = cy + dy

            if (nx, ny) in visited:
                continue

            if not is_walkable(
                game_state,
                nx,
                ny,
            ):
                continue

            visited.add((nx, ny))

            next_first_direction = (
                direction_name
                if first_direction is None
                else first_direction
            )

            queue.append(
                (
                    nx,
                    ny,
                    next_first_direction,
                    distance + 1,
                )
            )

    return None, None
    
def get_legal_action_indices(game_state):
    """
    Return indices of actions that are currently legal.
    """
    _, _, bombs_left, (x, y) = game_state["self"]

    legal_actions = []

    movement_actions = [
        ("UP", x, y - 1),
        ("RIGHT", x + 1, y),
        ("DOWN", x, y + 1),
        ("LEFT", x - 1, y),
    ]

    for action, nx, ny in movement_actions:
        if is_walkable(game_state, nx, ny):
            legal_actions.append(ACTIONS.index(action))

    # WAIT is always legal.
    legal_actions.append(ACTIONS.index("WAIT"))

    # BOMB only if available.
    if bombs_left:
        legal_actions.append(ACTIONS.index("BOMB"))

    return legal_actions

def get_safe_action_indices(game_state, return_info=False):
    """
    Return legal actions after removing unsafe choices.

    When the agent is currently threatened by a bomb, prefer actions
    that preserve a valid time-aware escape route.

    Outside danger, this remains only a safety filter; the DQN still
    chooses the action.
    """

    legal_actions = get_legal_action_indices(game_state)

    _, _, _, (x, y) = game_state["self"]

    danger_map = build_danger_map(game_state)

    current_danger_time = danger_map[x, y]

    movement_targets = {
        "UP": (x, y - 1),
        "RIGHT": (x + 1, y),
        "DOWN": (x, y + 1),
        "LEFT": (x - 1, y),
        "WAIT": (x, y),
    }

    safe_actions = []

    for action_index in legal_actions:

        action = ACTIONS[action_index]

        # BOMB

        if action == "BOMB":

            if current_danger_time != -1:
                continue

            if can_escape_after_bomb(game_state):
                safe_actions.append(action_index)

            continue

        # Movement / WAIT

        nx, ny = movement_targets[action]

        danger_time = danger_map[nx, ny]

        # Immediately lethal destination.
        if danger_time != -1 and danger_time <= 1:
            continue

        if current_danger_time != -1:

            if action_preserves_escape(
                game_state,
                action,
                danger_map,
            ):
                
                if action_is_escape_safe(
                    game_state,
                    action,
                    danger_map,
                ):
                    safe_actions.append(action_index)

        else:
            safe_actions.append(action_index)

    if safe_actions:
        if return_info:
            return safe_actions, False
        return safe_actions

    # Emergency fallback.
    emergency_actions = [
        action_index
        for action_index in legal_actions
        if ACTIONS[action_index] != "BOMB"
    ]

    if not emergency_actions:
        emergency_actions = legal_actions

    if return_info:
        return emergency_actions, True

    return emergency_actions

def action_preserves_escape(
    game_state,
    action,
    danger_map,
):
    """
    Check whether taking `action` leaves a time-aware escape route.

    The DQN still chooses the action. This function only rejects
    actions from which no temporally safe continuation can be found.
    """

    _, _, _, (x, y) = game_state["self"]

    deltas = {
        "UP": (0, -1),
        "RIGHT": (1, 0),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "WAIT": (0, 0),
    }

    dx, dy = deltas[action]

    start_x = x + dx
    start_y = y + dy

    field = game_state["field"]

    opponent_positions = {
        tuple(other[3])
        for other in game_state["others"]
    }

    if not is_tile_safe_at_time(
        game_state,
        start_x,
        start_y,
        1,
    ):
        return False

    directions = [
        (0, -1),
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, 0),
    ]

    queue = deque([
        (start_x, start_y, 1)
    ])

    visited = {
        (start_x, start_y, 1)
    }

    max_time = 6

    while queue:

        cx, cy, time_step = queue.popleft()

        if is_conservative_escape_target(
            game_state,
            cx,
            cy,
        ):
            return True
        
        if time_step >= max_time:
            continue

        next_time = time_step + 1

        for dx, dy in directions:

            nx = cx + dx
            ny = cy + dy

            if (
                nx < 0
                or ny < 0
                or nx >= field.shape[0]
                or ny >= field.shape[1]
            ):
                continue

            if field[nx, ny] != 0:
                continue

            if (nx, ny) in opponent_positions:
                continue

            if not is_tile_safe_at_time(
                game_state,
                nx,
                ny,
                next_time,
            ):
                continue

            state = (nx, ny, next_time)

            if state in visited:
                continue

            visited.add(state)

            queue.append(state)

    return False

def is_conservative_escape_target(
    game_state,
    x,
    y,
    extra_bomb=None,
):
    """
    Return True only if (x, y) lies outside the blast geometry
    of every currently known bomb.

    Path traversal remains time-aware, but an escape is considered
    complete only after reaching a tile that does not depend on
    waiting for a bomb/explosion to disappear.
    """

    field = game_state["field"]

    bombs = list(game_state["bombs"])

    if extra_bomb is not None:
        bombs.append(extra_bomb)

    if game_state["explosion_map"][x, y] > 0:
        return False

    for (bomb_x, bomb_y), _ in bombs:

        blast_tiles = get_blast_tiles(
            field,
            bomb_x,
            bomb_y,
            s.BOMB_POWER,
        )

        if (x, y) in blast_tiles:
            return False

    return True

def action_is_escape_safe(
    game_state,
    action,
    danger_map,
):
    """
    Return True when an action makes temporal progress toward safety.

    The function does not choose the action. It only prevents the DQN
    from wasting scarce escape time while standing in a predicted
    blast region.
    """

    _, _, _, (x, y) = game_state["self"]

    deltas = {
        "UP": (0, -1),
        "RIGHT": (1, 0),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "WAIT": (0, 0),
    }

    dx, dy = deltas[action]

    nx = x + dx
    ny = y + dy

    current_danger = danger_map[x, y]
    next_danger = danger_map[nx, ny]

    # Best case: this action immediately leaves all predicted blast zones.
    if next_danger == -1:
        return True

    # Waiting while threatened does not make escape progress.
    if action == "WAIT":
        return False

    # Moving into a tile that explodes earlier than agent's current tile is not progress.
    if (
        current_danger != -1
        and next_danger < current_danger
    ):
        return False

    return True

def act(self, game_state: dict) -> str:
    """
    Select an action using the DQN.

    During training:
        epsilon-greedy exploration

    During evaluation:
        greedy neural-network policy

    The safety mask restricts selectable actions, but the
    learned network determines which safe action is selected.
    """

    state_vector = game_state_to_vector(game_state)

    available_actions, used_fallback = get_safe_action_indices(
        game_state,
        return_info=True
    )

    if used_fallback:
        self.safety_fallback_count += 1

    danger_map = build_danger_map(game_state)
    _, _, _, (x, y) = game_state["self"]

    if danger_map[x, y] != -1:
        self.danger_action_count += 1

    if game_state["step"] % 100 == 0:
        self.logger.info(
            f"SAFETY_DIAG "
            f"round={game_state['round']} "
            f"step={game_state['step']} "
            f"danger_actions={self.danger_action_count} "
            f"safety_fallbacks={self.safety_fallback_count}"
        )

    # Exploration during training.
    if (
        self.train
        and random.random() < self.epsilon
    ):
        return ACTIONS[
            random.choice(available_actions)
        ]

    state_tensor = torch.tensor(
        state_vector,
        dtype=torch.float32,
        device=self.device,
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = self.policy_net(
            state_tensor
        )[0]

    best_action_index = max(
        available_actions,
        key=lambda i: q_values[i].item(),
    )

    return ACTIONS[best_action_index]


def state_to_features(game_state):
    """
    Engineered state representation for the DQN.

    Components:
        1. local_tiles
        2. target_type
        3. target_direction
        4. opponent_direction
        5. danger_level
        6. escape_direction
        7. bomb_available
        8. bomb_value
        9. opponent_in_blast
        10. bomb_escape_possible
        11. normalized_coin_distance
    """

    if game_state is None:
        return None

    field = game_state["field"]
    _, _, bombs_left, (x, y) = game_state["self"]

    bombs = game_state["bombs"]
    coins = game_state["coins"]

    bomb_positions, opponent_positions = get_occupied_positions(game_state)

    directions = [
        ((0, -1), "UP"),
        ((1, 0), "RIGHT"),
        ((0, 1), "DOWN"),
        ((-1, 0), "LEFT"),
    ]

    # Local tile types

    def classify_tile(nx, ny):
        """
        0 = FREE
        1 = WALL
        2 = CRATE
        3 = BOMB
        4 = OPPONENT
        """

        if (
            nx < 0
            or ny < 0
            or nx >= field.shape[0]
            or ny >= field.shape[1]
        ):
            return 1

        if (nx, ny) in bomb_positions:
            return 3

        if (nx, ny) in opponent_positions:
            return 4

        if field[nx, ny] == -1:
            return 1

        if field[nx, ny] == 1:
            return 2

        return 0

    local_tiles = (
        classify_tile(x, y - 1),   # UP
        classify_tile(x + 1, y),   # RIGHT
        classify_tile(x, y + 1),   # DOWN
        classify_tile(x - 1, y),   # LEFT
    )


    # Find direction toward nearest opponent

    def find_opponent_target():
        """
        Find the first movement direction on a shortest path toward
        the nearest opponent.

        Opponent tiles themselves are not walkable, so the BFS targets
        a reachable free tile adjacent to an opponent.
        """

        if not opponent_positions:
            return None

        queue = deque([
            ((x, y), None)
        ])

        visited = {(x, y)}

        while queue:
            (cx, cy), first_direction = queue.popleft()

            # Is the agent adjacent to an opponent?
            for dx, dy in [
                (0, -1),
                (1, 0),
                (0, 1),
                (-1, 0),
            ]:
                if (cx + dx, cy + dy) in opponent_positions:
                    if first_direction is None:
                        return "HERE"

                    return first_direction

            for (dx, dy), direction_name in directions:
                nx = cx + dx
                ny = cy + dy

                if (nx, ny) in visited:
                    continue

                if not is_walkable(game_state, nx, ny):
                    continue

                visited.add((nx, ny))

                next_first_direction = (
                    direction_name
                    if first_direction is None
                    else first_direction
                )

                queue.append(
                    ((nx, ny), next_first_direction)
                )

        return None

    # Find shortest path to a useful crate-bombing position

    def count_crates_hit_from(px, py):
        """
        Number of crates that a bomb placed at (px, py) would destroy,
        using the framework's blast geometry.
        """

        blast_tiles = get_blast_tiles(
            field,
            px,
            py,
            3
        )

        return sum(
            1
            for bx, by in blast_tiles
            if field[bx, by] == 1
        )


    def find_crate_target():
        """
        BFS through reachable free tiles.

        The target is not a crate itself. The target is the nearest
        reachable tile from which placing a bomb would hit a crate.
        """

        queue = deque([
            ((x, y), None)
        ])

        visited = {(x, y)}

        while queue:
            (cx, cy), first_direction = queue.popleft()

            if count_crates_hit_from(cx, cy) > 0:
                if first_direction is None:
                    return "HERE"

                return first_direction

            for (dx, dy), direction_name in directions:

                nx = cx + dx
                ny = cy + dy

                if (nx, ny) in visited:
                    continue

                if not is_walkable(game_state, nx, ny):
                    continue

                visited.add((nx, ny))

                next_first_direction = (
                    direction_name
                    if first_direction is None
                    else first_direction
                )

                queue.append(
                    ((nx, ny), next_first_direction)
                )

        return None

    # Unified objective

    coin_target_direction, coin_distance = (
        get_coin_target_and_distance(game_state)
    )

    if coin_target_direction is not None:

        target_type = "COIN"
        target_direction = coin_target_direction

    else:

        crate_target_direction = find_crate_target()

        if crate_target_direction is not None:

            target_type = "CRATE"
            target_direction = crate_target_direction

        else:

            target_type = "NONE"
            target_direction = "NONE"

    if coin_distance is None:
        normalized_coin_distance = 0.0
    else:
        normalized_coin_distance = min(
            coin_distance,
            10
        ) / 10.0

    # Opponent target

    opponent_direction = find_opponent_target()

    if opponent_direction is None:
        opponent_direction = "NONE"

    # Danger level

    danger_map = build_danger_map(game_state)

    current_danger_time = danger_map[x, y]

    if current_danger_time == -1:
        danger_level = "SAFE"

    elif current_danger_time <= 1:
        danger_level = "IMMEDIATE"

    else:
        danger_level = "FUTURE"

    # Escape direction

    escape_direction = find_escape_direction(
        game_state,
        danger_map,
    )

    # Bomb availability

    bomb_available = int(bool(bombs_left))

    # Bomb value

    crates_hit = count_crates_hit_from(x, y)

    if crates_hit == 0:
        bomb_value = 0

    elif crates_hit == 1:
        bomb_value = 1

    else:
        bomb_value = 2

    # Opponent bomb value

    current_blast_tiles = get_blast_tiles(
        field,
        x,
        y,
        3
    )

    opponent_in_blast = int(
        any(
            opponent_position in current_blast_tiles
            for opponent_position in opponent_positions
        )
    )

    # Can the agent safely escape after placing a bomb?

    bomb_escape_possible = can_escape_after_bomb(
        game_state
    )

 

    return (
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
    )

def features_to_vector(features):
    """
    Convert the engineered categorical state into a fixed numerical
    vector for the neural network.

    Uses one-hot encoding so categorical values are not given an
    artificial numerical ordering.
    """

    if features is None:
        return None

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

    vector = []

    
    # 1. Local tiles
    # Each of the four neighboring tiles has one of:
    # 0 FREE
    # 1 WALL
    # 2 CRATE
    # 3 BOMB
    # 4 OPPONENT

    for tile in local_tiles:
        one_hot = [0.0] * 5
        one_hot[tile] = 1.0
        vector.extend(one_hot)

    
    # 2. Resource target type

    target_types = [
        "NONE",
        "COIN",
        "CRATE",
    ]

    vector.extend(
        1.0 if target_type == value else 0.0
        for value in target_types
    )

    # 3. Resource target direction

    target_directions = [
        "UP",
        "RIGHT",
        "DOWN",
        "LEFT",
        "HERE",
        "NONE",
    ]

    vector.extend(
        1.0 if target_direction == value else 0.0
        for value in target_directions
    )

    # 4. Opponent direction

    opponent_directions = [
        "UP",
        "RIGHT",
        "DOWN",
        "LEFT",
        "HERE",
        "NONE",
    ]

    vector.extend(
        1.0 if opponent_direction == value else 0.0
        for value in opponent_directions
    )

    # 5. Danger level

    danger_levels = [
        "SAFE",
        "FUTURE",
        "IMMEDIATE",
    ]

    vector.extend(
        1.0 if danger_level == value else 0.0
        for value in danger_levels
    )

    # 6. Escape direction

    escape_directions = [
        "UP",
        "RIGHT",
        "DOWN",
        "LEFT",
        "WAIT",
        "NONE",
    ]

    vector.extend(
        1.0 if escape_direction == value else 0.0
        for value in escape_directions
    )

    # 7. Bomb availability

    vector.append(float(bomb_available))

    # 8. Bomb value

    bomb_values = [0, 1, 2]

    vector.extend(
        1.0 if bomb_value == value else 0.0
        for value in bomb_values
    )

    # 9. Opponent currently in hypothetical blast

    vector.append(float(opponent_in_blast))

    # 10. Escape possible after placing bomb

    vector.append(float(bomb_escape_possible))

    # 11. Distance to nearest reachable coin

    vector.append(
        float(normalized_coin_distance)
    )

    return np.asarray(vector, dtype=np.float32)

def game_state_to_vector(game_state):

    if game_state is None:
        return None

    features = state_to_features(game_state)

    return features_to_vector(features)

def find_escape_direction(
    game_state,
    danger_map,
    extra_bomb=None,
):
    """
    Find the first action on a temporally safe path out of the
    blast geometry of the relevant bombs.
    """

    field = game_state["field"]
    _, _, _, (start_x, start_y) = game_state["self"]

    opponent_positions = {
        tuple(other[3])
        for other in game_state["others"]
    }

    directions = [
        ((0, -1), "UP"),
        ((1, 0), "RIGHT"),
        ((0, 1), "DOWN"),
        ((-1, 0), "LEFT"),
        ((0, 0), "WAIT"),
    ]

    if danger_map[start_x, start_y] == -1:
        return "NONE"

    queue = deque([
        (start_x, start_y, 0, None)
    ])

    visited = {
        (start_x, start_y, 0)
    }

    max_time = 6

    while queue:

        cx, cy, time_step, first_action = queue.popleft()

        if (
            time_step > 0
            and is_conservative_escape_target(
                game_state,
                cx,
                cy,
                extra_bomb=extra_bomb,
            )
        ):
            return first_action

        if time_step >= max_time:
            continue

        next_time = time_step + 1

        for (dx, dy), action_name in directions:

            nx = cx + dx
            ny = cy + dy

            if (
                nx < 0
                or ny < 0
                or nx >= field.shape[0]
                or ny >= field.shape[1]
            ):
                continue

            if field[nx, ny] != 0:
                continue

            if (nx, ny) in opponent_positions:
                continue

            if not is_tile_safe_at_time(
                game_state,
                nx,
                ny,
                next_time,
                extra_bomb=extra_bomb,
            ):
                continue

            state = (
                nx,
                ny,
                next_time,
            )

            if state in visited:
                continue

            visited.add(state)

            next_first_action = (
                action_name
                if first_action is None
                else first_action
            )

            queue.append(
                (
                    nx,
                    ny,
                    next_time,
                    next_first_action,
                )
            )

    return "NONE"

def can_escape_after_bomb(game_state):
    """
    Test whether the agent can escape after hypothetically placing
    a bomb at its current position.

    Returns:
        1 if a safe escape path exists
        0 otherwise
    """

    _, _, bombs_left, (x, y) = game_state["self"]

    # If no bomb can currently be placed, the question is irrelevant.
    if not bombs_left:
        return 0

    # Simulate a new bomb using the framework's initial timer.
    hypothetical_bomb = (
        (x, y),
        3
    )

    danger_map = build_danger_map(
        game_state,
        extra_bomb=hypothetical_bomb
    )

    escape_direction = find_escape_direction(
        game_state,
        danger_map,
        extra_bomb=hypothetical_bomb,
    )

    return int(escape_direction != "NONE")


def build_danger_map(game_state, extra_bomb=None):
    """
    Return the earliest bomb timer affecting each tile.

    Values:
        -1 = no predicted bomb danger
         0 = currently dangerous explosion
        >0 = earliest active bomb timer affecting that tile

    extra_bomb:
        Optional ((x, y), timer) used later to simulate placing
        our own bomb.
    """
    field = game_state["field"]
    bombs = list(game_state["bombs"])
    explosion_map = game_state["explosion_map"]

    danger_map = np.full(field.shape, -1, dtype=int)

    # Current explosions are immediate danger.
    for px in range(field.shape[0]):
        for py in range(field.shape[1]):
            if explosion_map[px, py] > 0:
                danger_map[px, py] = 0

    if extra_bomb is not None:
        bombs.append(extra_bomb)

    # Consider every active bomb, not just the nearest one.
    for (bomb_x, bomb_y), timer in bombs:

        blast_tiles = get_blast_tiles(
            field,
            bomb_x,
            bomb_y,
            3
        )

        danger_time = int(timer) + 1

        for bx, by in blast_tiles:

            current = danger_map[bx, by]

            if current == -1 or danger_time < current:
                danger_map[bx, by] = danger_time

    return danger_map

def is_tile_safe_at_time(
    game_state,
    x,
    y,
    time_step,
    extra_bomb=None,
):
    """
    Return True if tile (x, y) is traversable and non-lethal
    at the specified future time.

    time_step=1 means after the agent's next action.
    """

    field = game_state["field"]

    bombs = list(game_state["bombs"])

    if extra_bomb is not None:
        bombs.append(extra_bomb)


    current_explosion_remaining = int(
        game_state["explosion_map"][x, y]
    )

    # explosion_map tells us for how many upcoming actions the
    # currently dangerous explosion remains relevant.
    if (
        current_explosion_remaining > 0
        and time_step <= current_explosion_remaining
    ):
        return False

    # Active bombs

    for (bomb_x, bomb_y), timer in bombs:

        detonation_time = int(timer) + 1

        # Bomb physically blocks its own tile before detonation.
        if (
            (x, y) == (bomb_x, bomb_y)
            and time_step < detonation_time
        ):
            return False

        # Explosion is dangerous for EXPLOSION_TIMER steps:
       
        explosion_end = (
            detonation_time
            + s.EXPLOSION_TIMER
            - 1
        )

        if (
            detonation_time
            <= time_step
            <= explosion_end
        ):
            blast_tiles = get_blast_tiles(
                field,
                bomb_x,
                bomb_y,
                s.BOMB_POWER,
            )

            if (x, y) in blast_tiles:
                return False

    return True

def get_blast_tiles(field, bomb_x, bomb_y, power):
    blast_tiles = {(bomb_x, bomb_y)}

    directions = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    ]

    for dx, dy in directions:
        for distance in range(1, power + 1):
            nx = bomb_x + dx * distance
            ny = bomb_y + dy * distance

            if nx < 0 or ny < 0:
                break

            if nx >= field.shape[0] or ny >= field.shape[1]:
                break

            # Walls stop the explosion.
            if field[nx, ny] == -1:
                break

            blast_tiles.add((nx, ny))

    return blast_tiles
