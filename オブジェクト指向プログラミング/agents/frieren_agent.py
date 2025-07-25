"""
フリーレン専用の強化学習エージェント（DQN）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


class FrierenRLAgent:
    """フリーレン専用の強化学習エージェント（魔法特化）"""
    
    def __init__(self, state_size=16, action_size=4, learning_rate=0.0005):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999  # よりゆるやかに減衰
        self.learning_rate = learning_rate
        
        # フリーレン特有の特性
        self.magic_power = 100  # 魔力値
        self.experience_bonus = 1.2  # 経験による補正
        
        # DQNモデル
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        
        # 修正: optimizerを一度だけ作成して再利用
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        self.update_target_network()
    
    def _build_model(self):
        """DQNモデルの構築"""
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model
    
    def update_target_network(self):
        """ターゲットネットワークの更新"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """経験の保存"""
        self.memory.append((state, action, reward, next_state, done))
    
    def get_action(self, state):
        """行動選択（ε-greedy）"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        """経験リプレイによる学習"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        # 修正: より効率的なテンソル処理
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for e in batch:
            states.append(e[0])
            actions.append(e[1])
            rewards.append(e[2])
            next_states.append(e[3])
            dones.append(e[4])
        
        # numpy配列に変換してからテンソル化
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(np.array(dones))
        
        # Q値計算
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        # 損失計算と最適化
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 修正: 既存のoptimizerを使用
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 