"""
拡張されたフリーレン冒険環境
"""

import numpy as np
import random
from .party_character import EnhancedPartyCharacter
from .boss_enemy import EnhancedBossEnemy


class EnhancedFrierenAdventureEnv:
    """拡張されたフリーレン冒険環境"""
    
    def __init__(self):
        self.party = [
            EnhancedPartyCharacter('フリーレン', 'mage', hp=90, mp=100),
            EnhancedPartyCharacter('フェルン', 'healer', hp=80, mp=70),
            EnhancedPartyCharacter('シュタルク', 'warrior', hp=120, mp=30)
        ]
        self.boss = EnhancedBossEnemy()
        self.turn = 0
        self.battle_log = []
        
        # キャラクターの位置情報を追加
        self.character_positions = {
            'フリーレン': [0, 0],
            'フェルン': [1, 0], 
            'シュタルク': [2, 0],
            'ボス': [5, 5]  # ボスの初期位置
        }
        
        # 行動ログ（状況, 行動, キャラ名, ターン）
        self.action_log = []
        
        # 強化学習エージェント（外部から注入）
        self.agents = None
        
        # 修正: target_network同期の頻度設定
        self.target_sync_freq = 10  # より頻繁に同期
    
    def set_agents(self, agents):
        """エージェントを設定"""
        self.agents = agents
    
    def reset(self):
        """環境リセット"""
        for char in self.party:
            char.hp = char.max_hp
            char.mp = char.max_mp
            char.alive = True
        
        self.boss = EnhancedBossEnemy()
        self.turn = 0
        self.battle_log = []
        self.action_log = []  # ログもリセット
        
        # 位置情報もリセット
        self.character_positions = {
            'フリーレン': [0, 0],
            'フェルン': [1, 0], 
            'シュタルク': [2, 0],
            'ボス': [5, 5]
        }
        
        return self._get_state()
    
    def _get_state(self):
        """状態取得"""
        state = []
        for char in self.party:
            state.extend([
                char.hp / char.max_hp,
                char.mp / char.max_mp,
                1.0 if char.alive else 0.0,
                char.experience / 100.0
            ])
        state.extend([
            self.boss.hp / self.boss.max_hp,
            1.0 if self.boss.alive else 0.0,
            1.0 if self.boss.rage_mode else 0.0,
            self.turn / 50.0
        ])
        return np.array(state, dtype=np.float32)
    
    def _manhattan_distance(self, pos1, pos2):
        """マンハッタン距離を計算"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _can_attack_target(self, attacker_name, target_name):
        """攻撃可能かどうかをマンハッタン距離で判定"""
        if attacker_name not in self.character_positions or target_name not in self.character_positions:
            return False
        
        attacker_pos = self.character_positions[attacker_name]
        target_pos = self.character_positions[target_name]
        
        # マンハッタン距離が1以下なら攻撃可能
        return self._manhattan_distance(attacker_pos, target_pos) <= 1
    
    def _move_character(self, char_name, new_pos):
        """キャラクターの位置を移動"""
        if char_name in self.character_positions:
            self.character_positions[char_name] = new_pos
    
    def _get_hp_level(self, char):
        """HP区分（0:低, 1:中, 2:高）"""
        ratio = char.hp / char.max_hp
        if ratio < 0.3:
            return 0
        elif ratio < 0.7:
            return 1
        else:
            return 2
    
    def _get_dist_level(self, char_name):
        """ボスとの距離区分（0:近, 1:中, 2:遠）"""
        pos = self.character_positions[char_name]
        boss_pos = self.character_positions['ボス']
        dist = self._manhattan_distance(pos, boss_pos)
        if dist <= 1:
            return 0
        elif dist <= 3:
            return 1
        else:
            return 2
    
    def step(self, actions=None):
        """1ステップ実行"""
        if actions is None and self.agents is not None:
            # エージェントによる行動選択
            state = self._get_state()
            actions = []
            
            # フリーレン
            actions.append(self.agents[0].get_action(state))
            
            # フェルン（パーティ状況考慮）
            party_status = [{'hp': char.hp, 'alive': char.alive} for char in self.party]
            actions.append(self.agents[1].get_action(tuple(state), party_status))
            
            # シュタルク（ボス状況考慮）
            boss_hp_ratio = self.boss.hp / self.boss.max_hp if self.boss.alive else 0
            actions.append(self.agents[2].get_action(tuple(state), boss_hp_ratio))
        
        rewards = [0, 0, 0]
        
        # パーティの行動実行
        for i, action in enumerate(actions):
            if not self.party[i].alive:
                continue
            
            char_name = self.party[i].name
            # --- 状況ログの記録 ---
            hp_level = self._get_hp_level(self.party[i])
            dist_level = self._get_dist_level(char_name)
            self.action_log.append(((hp_level, dist_level), action, char_name, self.turn))
            # ---
            damage_dealt = self.party[i].take_action(action)
            
            if action == 0 or action == 3:  # 攻撃またはスキル
                # マンハッタン距離による当たり判定
                if self._can_attack_target(char_name, 'ボス'):
                    if damage_dealt > 0:
                        self.boss.receive_damage(damage_dealt)
                        rewards[i] += damage_dealt / 10
                        self.battle_log.append(f"{char_name}の攻撃がボスに命中！ダメージ: {damage_dealt}")
                    else:
                        self.battle_log.append(f"{char_name}の攻撃が失敗...")
                else:
                    # 攻撃範囲外の場合
                    rewards[i] -= 2  # ペナルティ
                    self.battle_log.append(f"{char_name}の攻撃が範囲外...")
                    
            elif action == 1:  # 防御
                # 防御行動では位置を調整（ボスに近づく）
                current_pos = self.character_positions[char_name]
                boss_pos = self.character_positions['ボス']
                
                # ボスに向かって1マス移動
                dx = 1 if boss_pos[0] > current_pos[0] else (-1 if boss_pos[0] < current_pos[0] else 0)
                dy = 1 if boss_pos[1] > current_pos[1] else (-1 if boss_pos[1] < current_pos[1] else 0)
                
                new_pos = [current_pos[0] + dx, current_pos[1] + dy]
                self._move_character(char_name, new_pos)
                self.battle_log.append(f"{char_name}が防御しながら移動")
                
            elif action == 2:  # 回復
                if self.party[i].role == 'healer':
                    for char in self.party:
                        if char.alive and char.hp < char.max_hp:
                            char.heal(damage_dealt)
                            rewards[i] += 5
                            self.battle_log.append(f"{char_name}が{char.name}を回復")
                            break
        
        # ボスの行動
        if self.boss.alive:
            # ボスの位置を更新
            self.boss.set_position(self.character_positions['ボス'])
            attacks = self.boss.select_target_and_attack(self.party, self.character_positions)
            for target, damage in attacks:
                # ボスの攻撃もマンハッタン距離で判定
                if self._can_attack_target('ボス', target.name):
                    target.receive_damage(damage)
                    self.battle_log.append(f"ボスの攻撃が{target.name}に命中！ダメージ: {damage}")
                    if not target.alive:
                        rewards[self.party.index(target)] -= 50
                else:
                    self.battle_log.append(f"ボスの攻撃が{target.name}に外れた...")
        
        # 勝利/敗北判定
        done = not self.boss.alive or all(not char.alive for char in self.party)
        
        if not self.boss.alive:
            rewards = [r + 100 for r in rewards]
        elif all(not char.alive for char in self.party):
            rewards = [r - 100 for r in rewards]
        
        # 協力ボーナス
        alive_count = sum(1 for char in self.party if char.alive)
        if alive_count >= 2:
            rewards = [r + 2 for r in rewards]
        
        self.turn += 1
        
        return self._get_state(), rewards, done, {'battle_log': self.battle_log}
    
    def train_agents(self, episodes=1000, batch_size=64):
        """エージェント学習"""
        if self.agents is None:
            raise ValueError("エージェントが設定されていません")
        
        scores = []
        win_rates = []
        wins = 0
        
        for episode in range(episodes):
            state = self.reset()
            total_reward = 0
            done = False
            step_count = 0
            
            while not done and step_count < 100:
                # 行動選択
                actions = []
                for i, agent in enumerate(self.agents):
                    if hasattr(agent, 'get_action'):
                        if i == 0:  # フリーレン
                            actions.append(agent.get_action(state))
                        elif i == 1:  # フェルン
                            party_status = [{'hp': char.hp, 'alive': char.alive} for char in self.party]
                            actions.append(agent.get_action(tuple(state), party_status))
                        else:  # シュタルク
                            boss_hp_ratio = self.boss.hp / self.boss.max_hp if self.boss.alive else 0
                            actions.append(agent.get_action(tuple(state), boss_hp_ratio))
                
                # 環境ステップ
                next_state, rewards, done, _ = self.step(actions)
                
                # 学習
                for i, agent in enumerate(self.agents):
                    if hasattr(agent, 'remember'):  # DQN
                        agent.remember(state, actions[i], rewards[i], next_state, done)
                        if len(agent.memory) > batch_size:
                            agent.replay(batch_size)
                    elif hasattr(agent, 'update'):  # Q学習
                        agent.update(tuple(state), actions[i], rewards[i], tuple(next_state), done)
                
                state = next_state
                total_reward += sum(rewards)
                step_count += 1
                
                # 修正: target_networkの同期頻度を改善
                if (hasattr(self.agents[0], 'update_target_network') and 
                    step_count % self.target_sync_freq == 0):
                    self.agents[0].update_target_network()
            
            scores.append(total_reward)
            if not self.boss.alive:
                wins += 1
            
            # 定期的な統計表示
            if episode % 100 == 0:
                win_rate = wins / (episode + 1)
                win_rates.append(win_rate)
                avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
                print(f"Episode {episode}: Avg Score = {avg_score:.2f}, Win Rate = {win_rate:.3f}")
        
        return scores, win_rates 
    
    def get_battle_info(self):
        """戦闘情報を取得（デバッグ用）"""
        info = {
            'turn': self.turn,
            'character_positions': self.character_positions.copy(),
            'party_hp': [char.hp for char in self.party],
            'boss_hp': self.boss.hp,
            'battle_log': self.battle_log[-5:] if self.battle_log else []  # 最新5件
        }
        return info
    
    def test_manhattan_distance(self):
        """マンハッタン距離のテスト"""
        print("=== マンハッタン距離テスト ===")
        
        # テストケース1: 隣接している場合
        pos1 = [0, 0]
        pos2 = [1, 0]
        distance = self._manhattan_distance(pos1, pos2)
        can_attack = self._can_attack_target('フリーレン', 'ボス')
        print(f"位置1: {pos1}, 位置2: {pos2}")
        print(f"マンハッタン距離: {distance}")
        print(f"攻撃可能: {can_attack}")
        
        # テストケース2: 離れている場合
        pos1 = [0, 0]
        pos2 = [3, 3]
        distance = self._manhattan_distance(pos1, pos2)
        print(f"位置1: {pos1}, 位置2: {pos2}")
        print(f"マンハッタン距離: {distance}")
        
        # 現在の位置情報を表示
        print(f"現在の位置情報: {self.character_positions}") 