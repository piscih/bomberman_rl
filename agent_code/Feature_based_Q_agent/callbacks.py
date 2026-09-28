from collections import deque
import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Q-learning parameters
ALPHA = 0.1
GAMMA = 0.9

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995


def setup(self):
    
    """
    Setup the Q-learning agent.
    """
    self.alpha = ALPHA
    self.gamma = GAMMA
    self.epsilon = EPSILON_START

    if self.train or not os.path.isfile("my-saved-model.pkl"):
        self.logger.info("Setting up Q-table from scratch.")
        self.model = {}

    else:
        self.logger.info("Loading Q-table from saved state.")

        with open("my-saved-model.pkl", "rb") as file:
            data = pickle.load(file)

        self.model = data["q_table"]
        self.epsilon = data.get("epsilon", EPSILON_MIN)

def get_q_values(self, state):
    """
    Return the Q-values for all six actions.

    If this state has never been encountered, initialise it with zeros.
    """

    if state not in self.model:
        self.model[state] = np.zeros(len(ACTIONS), dtype=np.float32)

    return self.model[state]

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

def get_safe_action_indices(game_state):
    """
    Return legal actions after removing obviously suicidal choices.

    This is a safety filter, not a complete planning algorithm.
    If no action passes the filter, fall back to all legal actions.
    """

    legal_actions = get_legal_action_indices(game_state)

    _, _, _, (x, y) = game_state["self"]

    danger_map = build_danger_map(game_state)

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
            if can_escape_after_bomb(game_state):
                safe_actions.append(action_index)

            continue

        # Movement / WAIT
        nx, ny = movement_targets[action]

        danger_time = danger_map[nx, ny]

        if danger_time == -1 or danger_time > 1:
            safe_actions.append(action_index)

    if safe_actions:
        return safe_actions

    return legal_actions

def act(self, game_state: dict) -> str:

    state = state_to_features(game_state)
    q_values = get_q_values(self, state)

    available_actions = get_safe_action_indices(game_state)

    if self.train and random.random() < self.epsilon:
        return ACTIONS[random.choice(available_actions)]

    max_q = max(
        q_values[i]
        for i in available_actions
    )

    best_actions = [
        i
        for i in available_actions
        if q_values[i] == max_q
    ]

    return ACTIONS[random.choice(best_actions)]


def state_to_features(game_state):
    """
    
    V3.1a: ten-component opponent-aware state representation.

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

    # Find shortest-path direction to a reachable coin

    def find_coin_target():
        """
        BFS through currently walkable tiles.

        Returns the first movement direction on a shortest path
        to a reachable visible coin.

        Returns None if no visible coin is currently reachable.
        """

        if not coins:
            return None

        coin_positions = set(coins)

        queue = deque([
            ((x, y), None)
        ])

        visited = {(x, y)}

        while queue:
            (cx, cy), first_direction = queue.popleft()

            if (cx, cy) in coin_positions:
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
                # already standing on a useful bombing tile.
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

    coin_target_direction = find_coin_target()

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
        danger_map
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

    # Can agent safely escape after placing a bomb?

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
    )

def find_escape_direction(game_state, danger_map):
    """
    Find the first action on a time-aware path to safety.

    Returns:
        "UP", "RIGHT", "DOWN", "LEFT", "WAIT", or "NONE"
    """

    field = game_state["field"]
    _, _, _, (start_x, start_y) = game_state["self"]

    bomb_positions, opponent_positions = get_occupied_positions(game_state)

    directions = [
        ((0, -1), "UP"),
        ((1, 0), "RIGHT"),
        ((0, 1), "DOWN"),
        ((-1, 0), "LEFT"),
        ((0, 0), "WAIT"),
    ]

    # If the current tile has no predicted danger, no escape is needed.
    if danger_map[start_x, start_y] == -1:
        return "NONE"

    queue = deque([
        (start_x, start_y, 0, None)
    ])

    # Time matters, so (x, y, time) is the state.
    visited = {
        (start_x, start_y, 0)
    }

    # Bomb timers are at most 4 in the normal environment.
    max_time = 5

    while queue:

        cx, cy, time_step, first_action = queue.popleft()

        # Don't classify the initial tile at t=0 as the solution.
        if time_step > 0 and danger_map[cx, cy] == -1:
            return first_action

        if time_step >= max_time:
            continue

        next_time = time_step + 1

        for (dx, dy), action_name in directions:

            nx = cx + dx
            ny = cy + dy

            # Movement validity
            if (
                nx < 0
                or ny < 0
                or nx >= field.shape[0]
                or ny >= field.shape[1]
            ):
                continue

            if field[nx, ny] != 0:
                continue

            # Active bombs block movement.
            # Exception:
            # If agent is currently standing on it's own bomb tile,
            # allowed to move away from it.
            if (nx, ny) in bomb_positions and (nx, ny) != (start_x, start_y):
                continue

            if (nx, ny) in opponent_positions:
                continue

            # Temporal safety

            tile_danger_time = danger_map[nx, ny]

            if tile_danger_time != -1:
                # If the tile explodes at or before the time agent
                # arrive there, it cannot use it.
                if tile_danger_time <= next_time:
                    continue

            state = (nx, ny, next_time)

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
                    next_first_action
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
        danger_map
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
        Optional ((x, y), timer) used to simulate placing
        the agent's own bomb.
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

    # every active bomb is considered, not just the nearest one.
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


