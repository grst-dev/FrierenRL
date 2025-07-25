"""
フェルン専用の強化学習エージェント（Q学習）
"""

import numpy as np
import random
from collections import defaultdict


class FernRLAgent:
    """フェルン専用の強化学習エージェント（回復・サポート特化）"""
    
    def __init__(self, state_size=16, action_size=4):
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1
        
        # フェルン特有の特性
        self.healing_efficiency = 1.5  # 回復効率
        self.party_awareness = 0.8     # パーティ状況認識度
    
    def get_action(self, state, party_status):
        """パーティ状況を考慮した行動選択"""
        # パーティの危険度を計算
        danger_level = sum(1 for char in party_status if char['hp'] < 30) / len(party_status)
        
        # 危険度が高い場合は回復行動を優先
        if danger_level > 0.5 and random.random() < 0.7:
            return 2  # 回復行動
        
        # 通常のε-greedy選択
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """Q値の更新"""
        current_q = self.q_table[state][action]
        next_max_q = np.max(self.q_table[next_state])
        target = reward + (0 if done else self.gamma * next_max_q)
        self.q_table[state][action] += self.alpha * (target - current_q) 