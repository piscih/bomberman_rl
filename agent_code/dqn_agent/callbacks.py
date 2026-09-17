import os
import collections
import numpy as np
import torch
from .dqn_model import DQNResNet

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.policy_net = DQNResNet(input_channels=10, num_actions=6).to(self.device)
    
    model_path = os.path.join(os.path.dirname(__file__), "dqn_model.pt")
    if os.path.exists(model_path):
        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy_net.eval()
    
    self.epsilon = 0.0  # Set to >0 in train.py during training

def act(self, game_state: dict) -> str:
    features = state_to_features(game_state)
    mask = get_valid_action_mask(game_state)
    
    if hasattr(self, 'train') and self.train and np.random.rand() < self.epsilon:
        valid_indices = np.where(mask == 1.0)[0]
        if len(valid_indices) == 0:
            return 'WAIT'
        return ACTIONS[np.random.choice(valid_indices)]

    with torch.no_grad():
        state_tensor = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.policy_net(state_tensor).squeeze(0).cpu().numpy()
        
        q_values[mask == 0.0] = -1e9
        return ACTIONS[np.argmax(q_values)]

    
def state_to_features(game_state: dict) -> np.ndarray:
    if game_state is None:
        return np.zeros((10, 17, 17), dtype=np.float32)
        
    features = np.zeros((10, 17, 17), dtype=np.float32)
    arena = game_state['field']
    
    features[0] = (arena == 1).astype(np.float32)   # Walls
    features[1] = (arena == -1).astype(np.float32)  # Crates
    
    for x, y in game_state['coins']:
        features[2, x, y] = 1.0
        
    for (bx, by), timer in game_state['bombs']:
        ch = 3 + min(timer, 3)
        features[ch, bx, by] = 1.0
        
    for x in range(17):
        for y in range(17):
            if game_state['explosion_map'][x, y] > 0:
                features[7, x, y] = 1.0
                
    for _, _, _, (ox, oy) in game_state['others']:
        features[8, ox, oy] = 1.0
        
    _, _, _, (sx, sy) = game_state['self']
    features[9, sx, sy] = 1.0
    
    return features

def get_valid_action_mask(game_state: dict) -> np.ndarray:
    mask = np.zeros(6, dtype=np.float32)
    if game_state is None:
        return mask
        
    arena = game_state['field']
    _, _, bombs_left, (sx, sy) = game_state['self']
    
    moves = {'UP': (sx, sy - 1), 'RIGHT': (sx + 1, sy), 'DOWN': (sx, sy + 1), 'LEFT': (sx - 1, sy)}
    
    for idx, act_name in enumerate(['UP', 'RIGHT', 'DOWN', 'LEFT']):
        nx, ny = moves[act_name]
        if 0 <= nx < 17 and 0 <= ny < 17 and arena[nx, ny] == 0:
            mask[idx] = 1.0
            
    mask[4] = 1.0
    
    if bombs_left > 0:
        targets_crate = any(
            0 <= sx+dx < 17 and 0 <= sy+dy < 17 and arena[sx+dx, sy+dy] == -1
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]
        )
        targets_opp = any(
            abs(ox - sx) + abs(oy - sy) <= 3 and (ox == sx or oy == sy)
            for _, _, _, (ox, oy) in game_state['others']
        )
        
        if (targets_crate or targets_opp) and _has_safe_escape(game_state, (sx, sy)):
            mask[5] = 1.0
            
    return mask

def _has_safe_escape(game_state: dict, start_pos: tuple) -> bool:
    arena = game_state['field'].copy()
    arena[start_pos[0], start_pos[1]] = 1
    
    sim_bombs = game_state['bombs'] + [(start_pos, 4)]
    danger_zones = set()
    for (bx, by), _ in sim_bombs:
        danger_zones.add((bx, by))
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            for r in range(1, 4):
                nx, ny = bx + dx*r, by + dy*r
                if 0 <= nx < 17 and 0 <= ny < 17:
                    if arena[nx, ny] == 1:
                        break
                    danger_zones.add((nx, ny))
                    if arena[nx, ny] == -1:
                        break

    queue = collections.deque([(start_pos[0], start_pos[1], 0)])
    visited = set([(start_pos[0], start_pos[1], 0)])
    
    while queue:
        x, y, t = queue.popleft()
        if t >= 4 and (x, y) not in danger_zones:
            return True
        if t > 5:
            continue
            
        moves = [(-1,0), (1,0), (0,-1), (0,1)] if t == 0 else [(0,0), (-1,0), (1,0), (0,-1), (0,1)]
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 17 and 0 <= ny < 17 and arena[nx, ny] == 0:
                if (nx, ny, t + 1) not in visited:
                    visited.add((nx, ny, t + 1))
                    queue.append((nx, ny, t + 1))
                    
    return False