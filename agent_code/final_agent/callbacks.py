import os
from collections import deque
import numpy as np
import torch
from .dqn_model import DQNResNet

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
N_ACTIONS = 6
ROWS = 17
COLS = 17
BLAST_RADIUS = 3
BOMB_TIMER = 4
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


def in_bounds(x, y):
    return 0 <= x < COLS and 0 <= y < ROWS


def get_bomb_positions(bombs):
    return {tuple(position) for position, _ in bombs}


def get_other_positions(others):
    return {tuple(player[3]) for player in others}


def calculate_blast_cells(field, bomb_position, radius=BLAST_RADIUS):
    bx, by = bomb_position
    blast = {(bx, by)}
    for dx, dy in DIRECTIONS:
        for step in range(1, radius + 1):
            nx, ny = bx + dx * step, by + dy * step
            if not in_bounds(nx, ny) or field[nx, ny] == -1:
                break
            blast.add((nx, ny))
            if field[nx, ny] == 1:
                break
    return blast


def build_danger_map(game_state):
    field, bombs = game_state['field'], game_state['bombs']
    danger_time = np.full((COLS, ROWS), np.inf, dtype=np.float32)
    explosion_map = game_state.get('explosion_map', np.zeros((COLS, ROWS)))
    danger_time[explosion_map > 0] = 0.0

    bomb_data = {}
    for position, timer in bombs:
        pos = tuple(position)
        bomb_data[pos] = {'timer': float(timer), 'blast': calculate_blast_cells(field, pos)}

    changed = True
    while changed:
        changed = False
        for position, data in bomb_data.items():
            timer, blast = data['timer'], data['blast']
            for other_position, other_data in bomb_data.items():
                if other_position != position and other_position in blast:
                    new_timer = min(timer, other_data['timer'])
                    if new_timer < timer:
                        data['timer'] = timer = new_timer
                        changed = True

    for data in bomb_data.values():
        timer = max(0.0, data['timer'])
        for x, y in data['blast']:
            danger_time[x, y] = min(danger_time[x, y], timer)

    return danger_time


def compute_safe_reachable(game_state, start_position=None, max_depth=20, danger_time=None):
    field = game_state['field']
    if start_position is None:
        start_position = tuple(game_state['self'][3])
    if danger_time is None:
        danger_time = build_danger_map(game_state)

    bomb_positions = get_bomb_positions(game_state['bombs'])
    other_positions = get_other_positions(game_state['others'])

    sx, sy = start_position
    queue = deque([(sx, sy, 0)])
    visited = {(sx, sy)}
    safe_tiles = set()

    while queue:
        x, y, distance = queue.popleft()
        if danger_time[x, y] > distance:
            safe_tiles.add((x, y))
        if distance >= max_depth:
            continue

        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny) or (nx, ny) in visited:
                continue
            if field[nx, ny] != 0 or (nx, ny) in bomb_positions or (nx, ny) in other_positions:
                continue
            arrival_time = distance + 1
            if danger_time[nx, ny] <= arrival_time:
                continue

            visited.add((nx, ny))
            queue.append((nx, ny, arrival_time))

    return safe_tiles


def has_safe_escape(game_state):
    field = game_state['field']
    sx, sy = game_state['self'][3]

    simulated_bombs = list(game_state['bombs']) + [((sx, sy), BOMB_TIMER)]
    simulated_state = dict(game_state, bombs=simulated_bombs)

    danger_time = build_danger_map(simulated_state)
    bomb_positions = get_bomb_positions(simulated_bombs)
    other_positions = get_other_positions(game_state['others'])

    queue = deque([(sx, sy, 0)])
    visited = {(sx, sy)}

    while queue:
        x, y, distance = queue.popleft()
        if danger_time[x, y] <= distance:
            continue

        if distance > 0:
            continuation_exists = False
            for dx, dy in DIRECTIONS:
                cx, cy = x + dx, y + dy
                if not in_bounds(cx, cy) or field[cx, cy] != 0:
                    continue
                if (cx, cy) in bomb_positions or (cx, cy) in other_positions:
                    continue
                if danger_time[cx, cy] > distance + 1:
                    continuation_exists = True
                    break

            if continuation_exists or danger_time[x, y] > BOMB_TIMER:
                return True

        if distance >= BOMB_TIMER - 1:
            continue

        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny) or (nx, ny) in visited or field[nx, ny] != 0:
                continue
            if (nx, ny) in bomb_positions or (nx, ny) in other_positions:
                continue
            arrival = distance + 1
            if danger_time[nx, ny] <= arrival:
                continue

            visited.add((nx, ny))
            queue.append((nx, ny, arrival))

    return False


def nearest_coin_distance(game_state, start_position=None):
    field = game_state['field']
    coins = {tuple(c) for c in game_state['coins']}
    if not coins:
        return None, np.full((COLS, ROWS), -1.0, dtype=np.float32)

    if start_position is None:
        start_position = tuple(game_state['self'][3])

    sx, sy = start_position
    bomb_positions = get_bomb_positions(game_state['bombs'])
    other_positions = get_other_positions(game_state['others'])

    distance_map = np.full((COLS, ROWS), np.inf, dtype=np.float32)
    queue = deque([(sx, sy)])
    distance_map[sx, sy] = 0
    nearest_distance = None

    while queue:
        x, y = queue.popleft()
        current_distance = distance_map[x, y]

        if (x, y) in coins:
            nearest_distance = int(current_distance)
            break

        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny) or distance_map[nx, ny] != np.inf:
                continue
            if field[nx, ny] != 0 or (nx, ny) in bomb_positions or (nx, ny) in other_positions:
                continue

            distance_map[nx, ny] = current_distance + 1
            queue.append((nx, ny))

    normalized = np.full((COLS, ROWS), -1.0, dtype=np.float32)
    reachable = np.isfinite(distance_map)
    normalized[reachable] = 1.0 - np.minimum(distance_map[reachable] / 17.0, 1.0)

    return nearest_distance, normalized


def prepare_state(game_state):
    danger_time = build_danger_map(game_state)
    self_position = tuple(game_state['self'][3])
    safe_tiles = compute_safe_reachable(game_state, self_position, max_depth=12, danger_time=danger_time)
    coin_distance, coin_map = nearest_coin_distance(game_state)
    features, mask = extract_features_and_mask(game_state, precomputed_danger=danger_time, precomputed_safe_tiles=safe_tiles, precomputed_coin_map=coin_map)
    return features, mask, danger_time, coin_distance


def extract_features_and_mask(game_state, precomputed_danger=None, precomputed_safe_tiles=None, precomputed_coin_map=None):
    field = game_state['field']
    sx, sy = game_state['self'][3]
    bombs, others, coins = game_state['bombs'], game_state['others'], game_state['coins']

    bomb_positions = get_bomb_positions(bombs)
    other_positions = get_other_positions(others)

    danger_time = build_danger_map(game_state) if precomputed_danger is None else precomputed_danger
    safe_tiles = compute_safe_reachable(game_state, (sx, sy), max_depth=12, danger_time=danger_time) if precomputed_safe_tiles is None else precomputed_safe_tiles
    coin_distance_map = nearest_coin_distance(game_state)[1] if precomputed_coin_map is None else precomputed_coin_map

    mask = np.zeros(N_ACTIONS, dtype=np.float32)

    for idx, (dx, dy) in enumerate(DIRECTIONS):
        nx, ny = sx + dx, sy + dy
        if in_bounds(nx, ny) and field[nx, ny] == 0 and (nx, ny) not in bomb_positions and (nx, ny) not in other_positions:
            if danger_time[nx, ny] > 1:
                mask[idx] = 1.0

    current_danger = danger_time[sx, sy]
    if current_danger > 2 and len(safe_tiles) > 1:
        mask[4] = 1.0
    if np.sum(mask[:4]) == 0:
        mask[4] = 1.0

    has_bomb = bool(game_state['self'][2])
    if has_bomb and (sx, sy) not in bomb_positions and has_safe_escape(game_state):
        mask[5] = 1.0

    features = np.zeros((12, COLS, ROWS), dtype=np.float32)
    features[0] = (field == -1).astype(np.float32)
    features[1] = (field == 1).astype(np.float32)

    for cx, cy in coins:
        if in_bounds(cx, cy):
            features[2, cx, cy] = 1.0

    features[3, sx, sy] = 1.0

    for ox, oy in other_positions:
        if in_bounds(ox, oy):
            features[4, ox, oy] = 1.0

    for (bx, by), timer in bombs:
        if in_bounds(bx, by):
            features[5, bx, by] = 1.0
            features[6, bx, by] = np.clip(float(timer) / BOMB_TIMER, 0.0, 1.0)

    explosion_map = game_state.get('explosion_map', np.zeros((COLS, ROWS)))
    features[7] = (explosion_map > 0).astype(np.float32)

    finite = np.isfinite(danger_time)
    danger_feature = np.zeros((COLS, ROWS), dtype=np.float32)
    danger_feature[finite] = np.clip(1.0 - (danger_time[finite] / BOMB_TIMER), 0.0, 1.0)
    features[8] = danger_feature

    if has_bomb:
        features[9].fill(1.0)

    for x, y in safe_tiles:
        features[10, x, y] = 1.0

    features[11] = coin_distance_map

    return features, mask


def setup(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.policy_net = DQNResNet(input_channels=12, num_actions=6).to(self.device)

    model_path = os.path.join(os.path.dirname(__file__), "dqn_model.pt")

    if os.path.isfile(model_path):
        try:
            self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.logger.info("Loaded dqn_model.pt")
        except RuntimeError as exc:
            raise RuntimeError("dqn_model.pt is incompatible with the final 12-channel DQN. Delete the old model and retrain from scratch.") from exc
    else:
        if getattr(self, "train", False):
            self.logger.info("No dqn_model.pt found. Starting fresh training.")
        else:
            raise FileNotFoundError(f"Missing trained model: {model_path}")

    self.policy_net.eval()
    self.epsilon = 0.0

    self.cached_features = None
    self.cached_mask = None
    self.cached_danger = None
    self.cached_coin_distance = None


def act(self, game_state: dict) -> str:
    features, mask, danger_time, coin_distance = prepare_state(game_state)

    self.cached_features = features
    self.cached_mask = mask
    self.cached_danger = danger_time
    self.cached_coin_distance = coin_distance

    if getattr(self, "train", False):
        epsilon = getattr(self, "epsilon", 0.0)
        if np.random.random() < epsilon:
            valid_indices = np.where(mask == 1.0)[0]
            if len(valid_indices) > 0:
                return ACTIONS[np.random.choice(valid_indices)]

    with torch.no_grad():
        state_t = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.policy_net(state_t).squeeze(0).cpu().numpy()

    q_values[mask == 0.0] = -1e9
    action_idx = int(np.argmax(q_values))

    return ACTIONS[action_idx]