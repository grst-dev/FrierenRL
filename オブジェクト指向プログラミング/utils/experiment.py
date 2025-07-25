"""
比較実験機能
"""

import random
import numpy as np
from env import EnhancedFrierenAdventureEnv


class RandomAgent:
    """ランダムエージェント（比較用）"""
    
    def __init__(self, action_size=4):
        self.action_size = action_size
    
    def get_action(self, state, *args):
        """ランダム行動選択"""
        return random.randint(0, self.action_size - 1)


def random_agent_battle(env, num_battles=100):
    """ランダムエージェントでの戦闘実験"""
    print("\n=== ランダムエージェント戦闘実験 ===")
    
    # ランダムエージェントを設定
    random_agents = [RandomAgent() for _ in range(3)]
    env.set_agents(random_agents)
    
    wins = 0
    total_turns = []
    
    for battle in range(num_battles):
        state = env.reset()
        done = False
        turn = 0
        
        while not done and turn < 50:
            turn += 1
            next_state, rewards, done, info = env.step()
            state = next_state
        
        if not env.boss.alive:
            wins += 1
            total_turns.append(turn)
    
    win_rate = wins / num_battles
    avg_turns = np.mean(total_turns) if total_turns else 0
    
    print(f"ランダムエージェント結果:")
    print(f"勝率: {win_rate:.3f} ({wins}/{num_battles})")
    print(f"平均ターン数: {avg_turns:.1f}")
    
    return win_rate, avg_turns


def compare_learning_effect(env, trained_agents, num_battles=100):
    """学習済みエージェントとランダムエージェントの比較"""
    print("\n=== 学習効果の比較実験 ===")
    
    # 学習済みエージェントでの戦闘
    env.set_agents(trained_agents)
    trained_wins = 0
    trained_turns = []
    
    for battle in range(num_battles):
        state = env.reset()
        done = False
        turn = 0
        
        while not done and turn < 50:
            turn += 1
            next_state, rewards, done, info = env.step()
            state = next_state
        
        if not env.boss.alive:
            trained_wins += 1
            trained_turns.append(turn)
    
    trained_win_rate = trained_wins / num_battles
    trained_avg_turns = np.mean(trained_turns) if trained_turns else 0
    
    # ランダムエージェントでの戦闘
    random_win_rate, random_avg_turns = random_agent_battle(env, num_battles)
    
    # 比較結果
    print(f"\n=== 比較結果 ===")
    print(f"学習済みエージェント:")
    print(f"  勝率: {trained_win_rate:.3f} ({trained_wins}/{num_battles})")
    print(f"  平均ターン数: {trained_avg_turns:.1f}")
    print(f"ランダムエージェント:")
    print(f"  勝率: {random_win_rate:.3f}")
    print(f"  平均ターン数: {random_avg_turns:.1f}")
    print(f"改善率:")
    print(f"  勝率改善: {(trained_win_rate - random_win_rate) / random_win_rate * 100:.1f}%")
    print(f"  ターン数改善: {(random_avg_turns - trained_avg_turns) / random_avg_turns * 100:.1f}%")
    
    return {
        'trained_win_rate': trained_win_rate,
        'trained_avg_turns': trained_avg_turns,
        'random_win_rate': random_win_rate,
        'random_avg_turns': random_avg_turns
    } 