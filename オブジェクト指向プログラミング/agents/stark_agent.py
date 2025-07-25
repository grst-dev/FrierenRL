"""
シュタルク専用の強化学習エージェント（Q学習）
"""

import numpy as np
import random
from collections import defaultdict


class StarkRLAgent:
    """シュタルク専用の強化学習エージェント（物理戦闘特化）"""
    
    def __init__(self, state_size=16, action_size=4):
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.alpha = 0.15
        self.gamma = 0.95
        self.epsilon = 0.2
        
        # シュタルク特有の特性
        self.courage_level = 0.3  # 勇気レベル（戦闘中に上昇）
        self.fear_factor = 0.7    # 恐怖による行動制限
    
    def get_action(self, state, boss_hp_ratio):
        """ボスのHP状況と勇気レベルを考慮した行動選択"""
        # ボスのHPが低いほど勇気が湧く
        if boss_hp_ratio < 0.3:
            self.courage_level = min(1.0, self.courage_level + 0.1)
        
        # 勇気レベルが高い場合は攻撃的になる
        if self.courage_level > 0.7 and random.random() < 0.8:
            return 0  # 攻撃行動
        
        # 恐怖により防御的になることがある
        if random.random() < self.fear_factor and boss_hp_ratio > 0.8:
            return 1  # 防御行動
        
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """Q値の更新"""
        current_q = self.q_table[state][action]
        next_max_q = np.max(self.q_table[next_state])
        target = reward + (0 if done else self.gamma * next_max_q)
        self.q_table[state][action] += self.alpha * (target - current_q) 