
import os
import pickle
import random
from collections import deque

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

OPPOSITES = {
    'UP': 'DOWN',
    'DOWN': 'UP',
    'LEFT': 'RIGHT',
    'RIGHT': 'LEFT',
}

BLAST_RADIUS = 3
BOMB_TIMER = 3
NAV_OPPONENT_RADIUS = 3

MOVE_DELTAS = {
    'UP': (0, -1),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'RIGHT': (1, 0),
}


def setup(self):

    self.model_file = "q_table2.pkl"
    self.reset_training = False
    self.last_action = 'WAIT'

    if not self.reset_training and os.path.isfile(self.model_file):

        with open(self.model_file, "rb") as file:
            saved = pickle.load(file)

        if isinstance(saved, dict) and 'q_table' in saved:
            self.q_table = saved['q_table']
            self.epsilon = saved.get('epsilon', 1.0)
            self.rounds_trained = saved.get('rounds_trained', 0)
        else:
            self.q_table = saved
            self.epsilon = 1.0
            self.rounds_trained = 0

        print(
            f"Q-TABLE LOADED: {len(self.q_table)} states, "
            f"epsilon={self.epsilon:.3f}, "
            f"rounds_trained={self.rounds_trained} <<<"
        )

    else:

        self.q_table = {}
        self.epsilon = 1.0
        self.rounds_trained = 0

        print(">>> NO Q-TABLE FOUND - STARTING EMPTY <<<")

        if not getattr(self, 'train', True):
            print(
                "TESTING WITH AN EMPTY, "
                "UNTRAINED Q-TABLE"
            )


def in_bounds(field, x, y):

    return (
        0 <= x < field.shape[0]
        and
        0 <= y < field.shape[1]
    )


def get_neighbors(x, y):

    return [
        ((x, y - 1), 'UP'),
        ((x + 1, y), 'RIGHT'),
        ((x, y + 1), 'DOWN'),
        ((x - 1, y), 'LEFT'),
    ]

# Tile Bocked Check : 

def is_tile_blocked(
    field,
    tx,
    ty,
    bombs,
    explosion_map,
    others=None
):

    if others is None:
        others = []

    if not in_bounds(field, tx, ty):
        return True

    if field[tx, ty] != 0:
        return True

    if any(
        (bx, by) == (tx, ty)
        for (bx, by), _ in bombs
    ):
        return True

    if explosion_map[tx, ty] > 0:
        return True

    if (tx, ty) in others:
        return True

    return False


# Bomb blast cells calculation:
def get_blast_cells(field, x, y):

    cells = [(x, y)]

    for dx, dy in [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]:

        for distance in range(
            1,
            BLAST_RADIUS + 1
        ):

            nx = x + dx * distance
            ny = y + dy * distance

            if not in_bounds(field, nx, ny):
                break

            if field[nx, ny] == -1:
                break

            cells.append((nx, ny))

            if field[nx, ny] == 1:
                break

    return cells


# time aware danger map construction:
def build_danger_map(
    field,
    bombs,
    explosion_map
):

    danger_map = np.full(
        field.shape,
        99,
        dtype=int
    )

    danger_map[explosion_map > 0] = 0

    if not bombs:
        return danger_map

    bomb_data = []

    for (bx, by), timer in bombs:

        if not in_bounds(field, bx, by):
            continue

        bomb_data.append(
            {
                "pos": (bx, by),
                "timer": max(1, int(timer))
            }
        )

    changed = True

    while changed:

        changed = False

        for i, bomb in enumerate(bomb_data):

            bx, by = bomb["pos"]

            explosion_time = bomb["timer"]

            blast_cells = get_blast_cells(
                field,
                bx,
                by
            )

            for j, other_bomb in enumerate(bomb_data):

                if i == j:
                    continue

                if other_bomb["pos"] in blast_cells:

                    if other_bomb["timer"] > explosion_time:

                        other_bomb["timer"] = explosion_time

                        changed = True

    danger_map = np.full(
        field.shape,
        99,
        dtype=int
    )

    danger_map[explosion_map > 0] = 0

    for bomb in bomb_data:

        bx, by = bomb["pos"]

        explosion_time = bomb["timer"]

        for cx, cy in get_blast_cells(
            field,
            bx,
            by
        ):

            danger_map[cx, cy] = min(
                danger_map[cx, cy],
                explosion_time
            )

    return danger_map


def classify_danger(tte):

    if tte == 0:
        return 'EXPLODING'

    if tte <= 2:
        return 'IMMINENT'

    if tte <= 3:
        return 'DELAYED'

    return 'SAFE'


# BFS for navigation and target selection:
def bfs_first_step(
    field,
    bombs,
    danger_map,
    targets,
    start_pos,
    avoid_danger=True,
    others=None
):

    if others is None:
        others = []

    if not targets:
        return 'NONE'

    queue = deque([
        (start_pos, [])
    ])

    visited = {
        start_pos
    }

    while queue:

        (cx, cy), path = queue.popleft()

        if (cx, cy) in targets:

            if path:
                return path[0]

            return 'NONE'

        for (nx, ny), direction in get_neighbors(
            cx,
            cy
        ):

            if not in_bounds(
                field,
                nx,
                ny
            ):
                continue

            if (nx, ny) in visited:
                continue

            if field[nx, ny] != 0:
                continue

            if any(
                (bx, by) == (nx, ny)
                for (bx, by), _ in bombs
            ):
                continue

            if (nx, ny) in others:
                continue

            if (
                avoid_danger
                and
                danger_map[nx, ny] <= 3
            ):
                continue

            visited.add(
                (nx, ny)
            )

            queue.append(
                (
                    (nx, ny),
                    path + [direction]
                )
            )

    return 'NONE'



# Time aware escape search for bomb placement:
def _time_aware_escape_search(
    x,
    y,
    field,
    bombs,
    others,
    danger_map,
    safety_margin=0
):

    threshold = 3 + safety_margin

    queue = deque([
        ((x, y), 0, None)
    ])

    visited = {
        ((x, y), 0)
    }

    while queue:

        (cx, cy), time, first_action = queue.popleft()

        if (
            time > 0
            and
            danger_map[cx, cy] > threshold
        ):

            return first_action

        if time >= BOMB_TIMER:
            continue

        for (nx, ny), action in get_neighbors(
            cx,
            cy
        ):

            if not in_bounds(
                field,
                nx,
                ny
            ):
                continue

            if field[nx, ny] != 0:
                continue

            if any(
                (bx, by) == (nx, ny)
                for (bx, by), _ in bombs
            ):
                continue

            if (nx, ny) in others:
                continue

            next_time = time + 1

            if danger_map[nx, ny] <= next_time:
                continue

            state = (
                (nx, ny),
                next_time
            )

            if state in visited:
                continue

            visited.add(state)

            next_first_action = (
                action
                if first_action is None
                else first_action
            )

            queue.append(
                (
                    (nx, ny),
                    next_time,
                    next_first_action
                )
            )

    return None

# Find escape route after placing a bomb:
def find_bomb_escape(
    x,
    y,
    field,
    bombs,
    explosion_map=None,
    others=None
):

    if explosion_map is None:
        explosion_map = np.zeros_like(field)

    if others is None:
        others = []

    simulated_bombs = list(bombs) + [
        ((x, y), BOMB_TIMER)
    ]

    danger_map = build_danger_map(
        field,
        simulated_bombs,
        explosion_map
    )

    return _time_aware_escape_search(
        x,
        y,
        field,
        bombs,
        others,
        danger_map,
        safety_margin=1
    )

# Find if escape is possible after placing a bomb:
def can_escape_bomb(
    x,
    y,
    field,
    bombs,
    explosion_map=None,
    others=None
):

    return (
        find_bomb_escape(
            x,
            y,
            field,
            bombs,
            explosion_map,
            others
        )
        is not None
    )

# Find escape action based on current game state and danger map:
def get_escape_action(
    game_state,
    danger_map=None,
    others=None
):

    if game_state is None:
        return None

    field = game_state['field']

    bombs = game_state['bombs']

    explosion_map = game_state[
        'explosion_map'
    ]

    _, _, _, (x, y) = game_state[
        'self'
    ]

    if others is None:

        others = [
            xy
            for _, _, _, xy
            in game_state['others']
        ]

    if danger_map is None:

        danger_map = build_danger_map(
            field,
            bombs,
            explosion_map
        )

    return _time_aware_escape_search(
        x,
        y,
        field,
        bombs,
        others,
        danger_map,
        safety_margin=0
    )



# Bomb quality assessment for decision making:
def get_bomb_quality(
    x,
    y,
    field,
    bombs,
    others,
    bomb_possible,
    danger_map,
    explosion_map=None,
    opponent_safety_radius=2
):

    if not bomb_possible:
        return 0

    if danger_map[x, y] <= 3:
        return 0

    crate_hits = 0
    enemy_hits = 0

    for dx, dy in [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]:

        for distance in range(
            1,
            BLAST_RADIUS + 1
        ):

            nx = x + dx * distance
            ny = y + dy * distance

            if not in_bounds(
                field,
                nx,
                ny
            ):
                break

            if field[nx, ny] == -1:
                break

            if (nx, ny) in others:
                enemy_hits += 1

            if field[nx, ny] == 1:
                crate_hits += 1
                break

    if (
        crate_hits == 0
        and
        enemy_hits == 0
    ):
        return 0

    if not can_escape_bomb(
        x,
        y,
        field,
        bombs,
        explosion_map=explosion_map,
        others=others
    ):
        return 0

    if enemy_hits >= 1:
        return 3

    if any(
        abs(ox - x) + abs(oy - y)
        <= opponent_safety_radius
        for ox, oy in others
    ):
        return 0

    if crate_hits >= 2:
        return 2

    if crate_hits >= 1:
        return 1

    return 0



# Convert game state to feature representation for Q-learning:
def state_to_features(
    game_state,
    last_action='WAIT'
):

    if game_state is None:
        return None

    field = game_state['field']

    bombs = game_state['bombs']

    explosion_map = game_state[
        'explosion_map'
    ]

    coins = game_state['coins']

    _, _, bomb_possible, (x, y) = (
        game_state['self']
    )

    others = [
        xy
        for _, _, _, xy
        in game_state['others']
    ]

    danger_map = build_danger_map(
        field,
        bombs,
        explosion_map
    )

    danger = classify_danger(
        danger_map[x, y]
    )

    blocked_up = int(
        is_tile_blocked(
            field,
            x,
            y - 1,
            bombs,
            explosion_map,
            others
        )
    )

    blocked_down = int(
        is_tile_blocked(
            field,
            x,
            y + 1,
            bombs,
            explosion_map,
            others
        )
    )

    blocked_left = int(
        is_tile_blocked(
            field,
            x - 1,
            y,
            bombs,
            explosion_map,
            others
        )
    )

    blocked_right = int(
        is_tile_blocked(
            field,
            x + 1,
            y,
            bombs,
            explosion_map,
            others
        )
    )

   
    danger_direction = 'NONE'

    if danger != 'SAFE':

        escape = get_escape_action(
            game_state,
            danger_map=danger_map,
            others=others
        )

        danger_direction = (
            escape
            if escape is not None
            else 'NONE'
        )

  
    target_direction = 'NONE'

    if danger in [
        'SAFE',
        'DELAYED'
    ]:

        if coins:

            coin_targets = set(coins)

            queue = deque([
                ((x, y), 0)
            ])

            visited = {
                (x, y)
            }

            reachable_coins = []

            while queue:

                (cx, cy), distance = queue.popleft()

                if (cx, cy) in coin_targets:

                    reachable_coins.append(
                        (
                            (cx, cy),
                            distance
                        )
                    )

                for (nx, ny), _ in get_neighbors(
                    cx,
                    cy
                ):

                    if not in_bounds(
                        field,
                        nx,
                        ny
                    ):
                        continue

                    if (nx, ny) in visited:
                        continue

                  
                    if field[nx, ny] != 0:
                        continue

                  
                    if any(
                        (bx, by) == (nx, ny)
                        for (bx, by), _ in bombs
                    ):
                        continue

                  
                    if (nx, ny) in others:
                        continue

                 
                    if danger_map[nx, ny] <= 3:
                        continue

                    visited.add(
                        (nx, ny)
                    )

                    queue.append(
                        (
                            (nx, ny),
                            distance + 1
                        )
                    )

            if reachable_coins:

                reachable_coins.sort(
                    key=lambda item: (
                        item[1],
                        item[0]
                    )
                )

                closest_coin = (
                    reachable_coins[0][0]
                )

                target_direction = bfs_first_step(
                    field,
                    bombs,
                    danger_map,
                    {closest_coin},
                    (x, y),
                    avoid_danger=True,
                    others=others
                )

        if target_direction == 'NONE':

            crate_targets = set()

            crates_x, crates_y = np.where(
                field == 1
            )

            for cx, cy in zip(
                crates_x,
                crates_y
            ):

                for dx, dy in [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1)
                ]:

                    nx = cx + dx
                    ny = cy + dy

                    if (
                        in_bounds(
                            field,
                            nx,
                            ny
                        )
                        and
                        field[nx, ny] == 0
                    ):

                        crate_targets.add(
                            (nx, ny)
                        )

            uncontested = {
                t
                for t in crate_targets
                if not any(
                    abs(t[0] - ox)
                    +
                    abs(t[1] - oy)
                    <= NAV_OPPONENT_RADIUS
                    for ox, oy in others
                )
            }

            search_targets = (
                uncontested
                if uncontested
                else crate_targets
            )

            target_direction = bfs_first_step(
                field,
                bombs,
                danger_map,
                search_targets,
                (x, y),
                avoid_danger=True,
                others=others
            )

        if (
            target_direction == 'NONE'
            and
            others
        ):

            opp_targets = set()

            for ox, oy in others:

                for dx, dy in [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1)
                ]:

                    nx = ox + dx
                    ny = oy + dy

                    if (
                        in_bounds(
                            field,
                            nx,
                            ny
                        )
                        and
                        field[nx, ny] == 0
                        and
                        (nx, ny) not in others
                    ):

                        opp_targets.add(
                            (nx, ny)
                        )

            target_direction = bfs_first_step(
                field,
                bombs,
                danger_map,
                opp_targets,
                (x, y),
                avoid_danger=True,
                others=others
            )


    bomb_quality = get_bomb_quality(
        x,
        y,
        field,
        bombs,
        others,
        bomb_possible,
        danger_map,
        explosion_map
    )

    return (
        danger,
        danger_direction,
        target_direction,
        bomb_quality,
        blocked_up,
        blocked_down,
        blocked_left,
        blocked_right,
        last_action,
    )


# Valid action determination based on game state:
def get_valid_actions(game_state):

    if game_state is None:
        return ['WAIT']

    field = game_state['field']

    bombs = game_state['bombs']

    explosion_map = game_state[
        'explosion_map'
    ]

    others = [
        xy
        for _, _, _, xy
        in game_state['others']
    ]

    _, _, bomb_possible, (x, y) = (
        game_state['self']
    )

    valid_actions = []

    for (nx, ny), action in get_neighbors(
        x,
        y
    ):

        if not is_tile_blocked(
            field,
            nx,
            ny,
            bombs,
            explosion_map,
            others
        ):

            valid_actions.append(
                action
            )

    valid_actions.append('WAIT')

    if bomb_possible:
        valid_actions.append('BOMB')

    return valid_actions



# Safety assessment for movement actions:
def movement_is_safe(
    game_state,
    action,
    danger_map
):

    field = game_state['field']

    _, _, _, (x, y) = (
        game_state['self']
    )

    dx, dy = MOVE_DELTAS[action]

    nx = x + dx
    ny = y + dy

    if not in_bounds(
        field,
        nx,
        ny
    ):
        return False

    arrival_time = 1

    if danger_map[nx, ny] <= arrival_time:
        return False

    return True

# Safe action filtering based on game state and valid actions:
def get_safe_actions(
    game_state,
    valid_actions
):

    if game_state is None:
        return ['WAIT']

    field = game_state['field']

    bombs = game_state['bombs']

    explosion_map = game_state[
        'explosion_map'
    ]

    others = [
        xy
        for _, _, _, xy
        in game_state['others']
    ]

    _, _, bomb_possible, (x, y) = (
        game_state['self']
    )

    danger_map = build_danger_map(
        field,
        bombs,
        explosion_map
    )

    safe_actions = []

    for action in valid_actions:


        if action in MOVE_DELTAS:

            dx, dy = MOVE_DELTAS[action]

            nx = x + dx
            ny = y + dy

            if not movement_is_safe(
                game_state,
                action,
                danger_map
            ):
                continue

            if (nx, ny) in others:
                continue

            safe_actions.append(
                action
            )

        elif action == 'WAIT':

            if danger_map[x, y] > 1:

                safe_actions.append(
                    'WAIT'
                )


        elif action == 'BOMB':

            if not bomb_possible:
                continue

            escape = find_bomb_escape(
                x,
                y,
                field,
                bombs,
                explosion_map=explosion_map,
                others=others
            )

            if escape is not None:

                safe_actions.append(
                    'BOMB'
                )

    return safe_actions



# Act method for the Q-learning agent, determining the next action based on the current game state and learned Q-values:
def act(self, game_state):

    previous_action = getattr(
        self,
        'last_action',
        'WAIT'
    )

    features = state_to_features(
        game_state,
        last_action=previous_action
    )

    if features is None:

        self.last_action = 'WAIT'

        return 'WAIT'

    valid_actions = get_valid_actions(
        game_state
    )

    if not valid_actions:

        self.last_action = 'WAIT'

        return 'WAIT'


    safe_actions = get_safe_actions(
        game_state,
        valid_actions
    )

  
    # emergency fallback if no safe actions are available:
    if not safe_actions:

        field = game_state['field']

        bombs = game_state['bombs']

        explosion_map = game_state[
            'explosion_map'
        ]

        _, _, _, (x, y) = game_state[
            'self'
        ]

        danger_map = build_danger_map(
            field,
            bombs,
            explosion_map
        )

        movement_actions = [
            action
            for action in valid_actions
            if action in MOVE_DELTAS
        ]

        if movement_actions:

            def safety_value(action):

                dx, dy = MOVE_DELTAS[action]

                nx = x + dx
                ny = y + dy

                return danger_map[nx, ny]

            best_safety = max(
                safety_value(action)
                for action in movement_actions
            )

            safe_actions = [
                action
                for action in movement_actions
                if safety_value(action)
                == best_safety
            ]

        else:

            safe_actions = valid_actions


    if features not in self.q_table:

        self.q_table[features] = [
            0.0
            for _ in ACTIONS
        ]

    q_values = self.q_table[
        features
    ]

   
# epsilon-greedy action selection based on Q-values and safe actions:
    epsilon = getattr(
        self,
        'epsilon',
        0.05
    )

    if (
        self.train
        and
        random.random() < epsilon
    ):

        selected_action = random.choice(
            safe_actions
        )

    else:

        safe_indices = [
            ACTIONS.index(action)
            for action in safe_actions
        ]

        best_value = max(
            q_values[index]
            for index in safe_indices
        )

        best_actions = [
            action
            for action in safe_actions
            if q_values[
                ACTIONS.index(action)
            ] == best_value
        ]

        opposite = OPPOSITES.get(
            previous_action
        )

        non_reversing = [
            action
            for action in best_actions
            if action != opposite
        ]

        if non_reversing:

            selected_action = random.choice(
                non_reversing
            )

        else:

            selected_action = random.choice(
                best_actions
            )

   # final action
    self.last_action = selected_action

    return selected_action