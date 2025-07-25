"""
拡張されたパーティキャラクター
"""


class EnhancedPartyCharacter:
    """拡張されたパーティキャラクター"""
    
    def __init__(self, name, role, hp=100, mp=50):
        self.name = name
        self.role = role
        self.max_hp = hp
        self.hp = hp
        self.max_mp = mp
        self.mp = mp
        self.alive = True
        self.experience = 0
        
        # 役割別の特殊能力
        if role == 'mage':
            self.magic_damage_multiplier = 2.0
            self.mana_efficiency = 0.8
        elif role == 'healer':
            self.healing_multiplier = 1.5
            self.support_bonus = 1.2
        elif role == 'warrior':
            self.physical_damage_multiplier = 1.3
            self.defense_bonus = 1.2
    
    def take_action(self, action, target=None):
        """行動実行"""
        if not self.alive:
            return 0
        
        if action == 0:  # 攻撃
            if self.role == 'mage' and self.mp >= 10:
                damage = 40 * self.magic_damage_multiplier
                self.mp -= 10
                return damage
            else:
                damage = 20 * getattr(self, 'physical_damage_multiplier', 1.0)
                return damage
        elif action == 1:  # 防御
            return 0
        elif action == 2:  # 回復
            if self.role == 'healer' and self.mp >= 5:
                heal_amount = 25 * getattr(self, 'healing_multiplier', 1.0)
                self.mp -= 5
                return heal_amount
            return 0
        elif action == 3:  # スキル
            if self.mp >= 15:
                self.mp -= 15
                if self.role == 'mage':
                    return 60 * self.magic_damage_multiplier
                elif self.role == 'warrior':
                    return 45 * self.physical_damage_multiplier
                elif self.role == 'healer':
                    return 35  # 全体回復
            return 0
        
        return 0
    
    def receive_damage(self, damage):
        """ダメージ処理"""
        actual_damage = damage * (1 - getattr(self, 'defense_bonus', 0) * 0.1)
        self.hp = max(0, self.hp - actual_damage)
        if self.hp <= 0:
            self.alive = False
        return actual_damage
    
    def heal(self, amount):
        """回復処理"""
        self.hp = min(self.max_hp, self.hp + amount)
    
    def restore_mp(self, amount):
        """MP回復"""
        self.mp = min(self.max_mp, self.mp + amount) 