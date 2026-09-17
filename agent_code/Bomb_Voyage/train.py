import collections
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from .callbacks import state_to_features, get_valid_action_mask, ACTIONS
from .dqn_model import DQNResNet

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
LR = 3e-4
MEMORY_SIZE = 100_000
TARGET_UPDATE_FREQ = 2500
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 1_000_000

# Reward Weights
REWARD_KILL = 40.0
REWARD_COIN = 0.5
REWARD_CRATE = 0.2
REWARD_WAIT_PENALTY = -0.1
REWARD_CLOSER_TO_OPPONENT = 0.2

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done, mask):
        self.buffer.append((state, action, reward, next_state, done, mask))
        
    def sample(self, batch_size):
        return zip(*random.sample(self.buffer, batch_size))
        
    def __len__(self):
        return len(self.buffer)

def setup_training(self):
    self.replay_buffer = ReplayBuffer(MEMORY_SIZE)
    self.target_net = DQNResNet(input_channels=10, num_actions=6).to(self.device)
    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
    self.steps_done = 0

    checkpoint_path = os.path.join(os.path.dirname(__file__), "train_checkpoint.pt")
    model_path = os.path.join(os.path.dirname(__file__), "dqn_model.pt")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)
        print(f"[Training Setup] Loaded checkpoint from step {self.steps_done}.")
    elif os.path.exists(model_path):
        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        print("[Training Setup] Loaded pre-trained dqn_model.pt weights.")

    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None:
        return
        
    reward = 0.0
    if "KILLED_OPPONENT" in events:
        reward += REWARD_KILL
    if "COIN_COLLECTED" in events:
        reward += REWARD_COIN
    if "CRATE_DESTROYED" in events:
        reward += REWARD_CRATE
    if self_action == 'WAIT':
        reward += REWARD_WAIT_PENALTY
    if "KILLED_SELF" in events or "GOT_KILLED" in events:
        reward -= 50.0  # Dominates any coin/crate rewards    

    if new_game_state is not None and old_game_state.get('others') and new_game_state.get('others'):
        old_sx, old_sy = old_game_state['self'][3]
        new_sx, new_sy = new_game_state['self'][3]
        
        old_min_dist = min([abs(old_sx - ox) + abs(old_sy - oy) for _, _, _, (ox, oy) in old_game_state['others']])
        new_min_dist = min([abs(new_sx - ox) + abs(new_sy - oy) for _, _, _, (ox, oy) in new_game_state['others']])
        
        if new_min_dist < old_min_dist:
            reward += REWARD_CLOSER_TO_OPPONENT
        elif new_min_dist > old_min_dist:
            reward -= REWARD_CLOSER_TO_OPPONENT

    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state) if new_game_state is not None else np.zeros((10, 17, 17), dtype=np.float32)
    
    action_idx = ACTIONS.index(self_action) if (self_action is not None and self_action in ACTIONS) else 4
    next_mask = get_valid_action_mask(new_game_state) if new_game_state is not None else np.zeros(6, dtype=np.float32)
    is_terminal = (new_game_state is None)
    
    self.replay_buffer.push(old_features, action_idx, reward, new_features, is_terminal, next_mask)
    
    self.steps_done += 1
    calculated_eps = EPS_START - (self.steps_done / EPS_DECAY_STEPS) * (EPS_START - EPS_END)
    
    if self.steps_done > 400_000:
        self.epsilon = max(0.20, calculated_eps)
    else:
        self.epsilon = max(EPS_END, calculated_eps)
    
    if self.steps_done % 4 == 0:
        _optimize_model(self)

def end_of_round(self, last_game_state, last_action, events):
    game_events_occurred(self, last_game_state, last_action, None, events)
    _optimize_model(self)
    
    model_path = os.path.join(os.path.dirname(__file__), "dqn_model.pt")
    torch.save(self.policy_net.state_dict(), model_path)
    checkpoint_path = os.path.join(os.path.dirname(__file__), "train_checkpoint.pt")
    checkpoint = {
        'model_state_dict': self.policy_net.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'steps_done': self.steps_done
    }
    torch.save(checkpoint, checkpoint_path)

def _optimize_model(self):
    if len(self.replay_buffer) < BATCH_SIZE:
        return
        
    states, actions, rewards, next_states, dones, next_masks = self.replay_buffer.sample(BATCH_SIZE)
    
    states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
    next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
    next_masks_t = torch.tensor(np.array(next_masks), dtype=torch.float32, device=self.device)
    q_values = self.policy_net(states_t).gather(1, actions_t)
    
    with torch.no_grad():
        next_q_policy = self.policy_net(next_states_t)
        next_q_policy[next_masks_t == 0.0] = -1e9
        best_actions = next_q_policy.argmax(dim=1, keepdim=True)
        
        next_q_target = self.target_net(next_states_t).gather(1, best_actions)
        expected_q = rewards_t + (1 - dones_t) * GAMMA * next_q_target

    loss = nn.SmoothL1Loss()(q_values, expected_q)
    
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
    self.optimizer.step()
    if self.steps_done % TARGET_UPDATE_FREQ == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())