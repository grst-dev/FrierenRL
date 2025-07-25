"""
拡張されたボス敵
"""

import random


class EnhancedBossEnemy:
    """拡張されたボス敵"""
    
    def __init__(self, name="魔王", hp=300, attack_patterns=None):
        self.name = name
        self.max_hp = hp
        self.hp = hp
        self.alive = True
        self.turn_count = 0
        self.attack_patterns = attack_patterns or ['normal_attack', 'aoe_attack', 'magic_attack']
        self.rage_mode = False
        self.position = [5, 5]  # ボスの位置情報を追加
    
    def set_position(self, pos):
        """位置を設定"""
        self.position = pos
    
    def get_position(self):
        """位置を取得"""
        return self.position
    
    def _manhattan_distance(self, pos1, pos2):
        """マンハッタン距離を計算"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _can_attack_target(self, target_pos):
        """攻撃可能かどうかをマンハッタン距離で判定"""
        return self._manhattan_distance(self.position, target_pos) <= 1
    
    def select_target_and_attack(self, party, party_positions=None):
        """攻撃対象選択と攻撃実行"""
        alive_party = [char for char in party if char.alive]
        if not alive_party:
            return []
        
        self.turn_count += 1
        
        # HP50%以下で激怒モード
        if self.hp / self.max_hp <= 0.5:
            self.rage_mode = True
        
        # 攻撃パターン選択
        if self.rage_mode and self.turn_count % 3 == 0:
            return self._aoe_attack(alive_party, party_positions)
        elif len(alive_party) == 1:
            return self._focused_attack(alive_party[0], party_positions)
        else:
            return self._strategic_attack(alive_party, party_positions)
    
    def _strategic_attack(self, alive_party, party_positions=None):
        """戦略的攻撃（優先度: ヒーラー > メイジ > 戦士）"""
        healers = [char for char in alive_party if char.role == 'healer']
        mages = [char for char in alive_party if char.role == 'mage']
        
        if healers:
            target = random.choice(healers)
        elif mages:
            target = random.choice(mages)
        else:
            target = random.choice(alive_party)
        
        # 位置情報がある場合は当たり判定を行う
        if party_positions and target.name in party_positions:
            target_pos = party_positions[target.name]
            if not self._can_attack_target(target_pos):
                return []  # 攻撃範囲外
        
        damage = 35 + random.randint(-5, 10)
        return [(target, damage)]
    
    def _aoe_attack(self, alive_party, party_positions=None):
        """全体攻撃"""
        attacks = []
        damage = 25 + random.randint(-5, 5)
        
        for char in alive_party:
            # 位置情報がある場合は当たり判定を行う
            if party_positions and char.name in party_positions:
                target_pos = party_positions[char.name]
                if self._can_attack_target(target_pos):
                    attacks.append((char, damage))
            else:
                # 位置情報がない場合は全員に攻撃
                attacks.append((char, damage))
        
        return attacks
    
    def _focused_attack(self, target, party_positions=None):
        """集中攻撃"""
        # 位置情報がある場合は当たり判定を行う
        if party_positions and target.name in party_positions:
            target_pos = party_positions[target.name]
            if not self._can_attack_target(target_pos):
                return []  # 攻撃範囲外
        
        damage = 50 + random.randint(-10, 15)
        return [(target, damage)]
    
    def receive_damage(self, damage):
        """ダメージ処理"""
        self.hp = max(0, self.hp - damage)
        if self.hp <= 0:
            self.alive = False 