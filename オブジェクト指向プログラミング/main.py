import numpy as np
import pygame as pg
import matplotlib.pyplot as plt
import random
import os
import time
import csv
import datetime
import configparser
import json
from logging import getLogger, config
from abc import ABCMeta, abstractmethod

# リファクタリングされたモジュール
from agents import FrierenRLAgent, FernRLAgent, StarkRLAgent
from env import EnhancedFrierenAdventureEnv
from ui import Grid, Graph
from utils import Func
from utils.experiment import compare_learning_effect

# エージェントの持つべき属性の抽象クラス
class Attribute(metaclass=ABCMeta):
    @abstractmethod
    def get_name(self): pass

    @abstractmethod
    def get_no(self): pass

    @abstractmethod
    def coord(self): pass

# エージェントの取り得る行動の抽象クラス
class Behavior(metaclass=ABCMeta):
    @abstractmethod
    def move(self): pass

    @abstractmethod
    def reset(self): pass

# 環境定義のクラス
class Environment:
    # 修正: マジックナンバーを定数として定義
    GRID = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 1],
                     [1, 0, 2, 2, 0, 1, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 2, 0, 2, 2, 0, 0, 2, 2, 1],
                     [1, 0, 0, 0, 0, 2, 2, 0, 2, 0, 0, 1],
                     [1, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 1],
                     [1, 0, 0, 2, 2, 0, 2, 2, 2, 0, 2, 1],
                     [1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1],
                     [1, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

class Agent(Attribute, Behavior):
    def __init__(self, name: str, no: int, x: int, y: int):
        self.__name = name
        self.__no = no
        self.__x = x
        self.__y = y

    def get_name(self) -> str:
        return self.__name

    def get_no(self) -> int:
        return self.__no

    @property
    def coord(self):
        return self.__x, self.__y

    @coord.setter
    def coord(self, xy: list):
        self.__x = xy[0]
        self.__y = xy[1]

    def collision(self, diff_y: int, diff_x: int) -> int:
        if Environment.GRID[self.__y + diff_y][self.__x + diff_x] == Grid.ROAD:
            return True
        else:
            return False

    def move(self, vertical: int, horizontal: int):
        if self.collision(vertical, horizontal):
            self.__y += vertical
            self.__x += horizontal

    def action(self, num: int):
        if num == 0:  # 上
            self.move(-1, 0)
        elif num == 1:  # 下
            self.move(1, 0)
        elif num == 2:  # 左
            self.move(0, -1)
        elif num == 3:  # 右
            self.move(0, 1)

    def strategy(self):
        pass

    def reset(self):
        pass

class Friend(Agent):
    def __init__(self, name: str, no: int, x: int, y: int):
        super().__init__(name, no, x, y)
        self.__init_x = x
        self.__init_y = y

    def action(self, num: int):
        # 修正: 定数を使用
        Environment.GRID[self.coord[1]][self.coord[0]] = Grid.ROAD
        if num == 0:  # 上
            self.move(-1, 0)
        elif num == 1:  # 下
            self.move(1, 0)
        elif num == 2:  # 左
            self.move(0, -1)
        elif num == 3:  # 右
            self.move(0, 1)
        Environment.GRID[self.coord[1]][self.coord[0]] = Grid.HUMAN

    def strategy(self, coord: list):
        if (3 > coord[0] - self.__init_x > 0) and (3 > coord[1] - self.__init_y > 0):
            return 1
        elif (-3 < coord[0] - self.__init_x < 0) and (3 > coord[1] - self.__init_y > 0):
            return 3
        elif (-3 < coord[0] - self.__init_x < 0) and (-3 < coord[1] - self.__init_y < 0):
            return 0
        elif(3 > coord[0] - self.__init_x > 0) and (-3 < coord[1] - self.__init_y < 0):
            return 2
        else:
            return random.randint(0, 3)

    def reset(self):
        # 修正: 定数を使用
        Environment.GRID[self.coord[1]][self.coord[0]] = Grid.ROAD
        self.coord = [self.__init_x, self.__init_y]
        Environment.GRID[self.coord[1]][self.coord[0]] = Grid.HUMAN

class Enemy(Agent):
    def __init__(self, name: str, no: int, x: int, y: int):
        super().__init__(name, no, x, y)
        self.__init_x = x
        self.__init_y = y

    def action(self, num: int):
        # 修正: 定数を使用
        Environment.GRID[self.coord[1]][self.coord[0]] = Grid.ROAD
        if num == 0:  # 左上
            self.move(-1, -1)
        elif num == 1:  # 右下
            self.move(1, 1)
        elif num == 2:  # 左下
            self.move(1, -1)
        elif num == 3:  # 右上
            self.move(-1, 1)
        Environment.GRID[self.coord[1]][self.coord[0]] = Grid.ALIEN

    def reset(self):
        # 修正: 定数を使用
        Environment.GRID[self.coord[1]][self.coord[0]] = Grid.ROAD
        self.coord = [self.__init_x, self.__init_y]
        Environment.GRID[self.coord[1]][self.coord[0]] = Grid.ALIEN

def main():
    """メイン関数（グリッドワールドシミュレーション）"""
    # 外部設定ファイルの読み込み
    conf = configparser.ConfigParser()
    
    # 設定ファイルが存在しない場合の処理
    if not os.path.exists('./conf/conf.ini'):
        print("設定ファイルが見つかりません: ./conf/conf.ini")
        print("デフォルト設定で実行します...")
        # デフォルト設定を設定
        conf.add_section('ENVIRONMENT')
        conf.set('ENVIRONMENT', 'AGT1_COORDX', '1')
        conf.set('ENVIRONMENT', 'AGT1_COORDY', '1')
        conf.set('ENVIRONMENT', 'AGT2_COORDX', '2')
        conf.set('ENVIRONMENT', 'AGT2_COORDY', '1')
        conf.set('ENVIRONMENT', 'AGT3_COORDX', '3')
        conf.set('ENVIRONMENT', 'AGT3_COORDY', '1')
        conf.set('ENVIRONMENT', 'ENEMY_COORDX', '10')
        conf.set('ENVIRONMENT', 'ENEMY_COORDY', '10')
        
        conf.add_section('SYSTEM')
        conf.set('SYSTEM', 'MAX_STEP', '50')
        conf.set('SYSTEM', 'MAX_EPISODE', '10')
    else:
        conf.read('./conf/conf.ini')

    # ログファイル出力のための設定
    if not os.path.exists('./conf/log_config.json'):
        print("ログ設定ファイルが見つかりません: ./conf/log_config.json")
        print("デフォルトログ設定で実行します...")
        log_conf = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": "INFO",
                    "stream": "ext://sys.stdout"
                }
            },
            "root": {
                "handlers": ["console"],
                "level": "INFO"
            }
        }
    else:
        with open('./conf/log_config.json', 'r') as f:
            log_conf = json.load(f)
    
    # ログディレクトリが存在しない場合は作成
    log_dir = 'log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"ログディレクトリを作成しました: {log_dir}")
    
    config.dictConfig(log_conf)
    logger = getLogger(__name__)

    # エージェント生成
    human = []
    alien = []
    
    # フリーレン（mage）
    tmp_x = int(conf.get('ENVIRONMENT', 'AGT1_COORDX', fallback='1'))
    tmp_y = int(conf.get('ENVIRONMENT', 'AGT1_COORDY', fallback='1'))
    human.append(Friend('フリーレン', 1, tmp_x, tmp_y))
    Environment.GRID[tmp_y][tmp_x] = Grid.HUMAN
    
    # フェルン（healer）
    tmp_x = int(conf.get('ENVIRONMENT', 'AGT2_COORDX', fallback='2'))
    tmp_y = int(conf.get('ENVIRONMENT', 'AGT2_COORDY', fallback='1'))
    human.append(Friend('フェルン', 2, tmp_x, tmp_y))
    Environment.GRID[tmp_y][tmp_x] = Grid.HUMAN
    
    # シュタルク（warrior）
    tmp_x = int(conf.get('ENVIRONMENT', 'AGT3_COORDX', fallback='3'))
    tmp_y = int(conf.get('ENVIRONMENT', 'AGT3_COORDY', fallback='1'))
    human.append(Friend('シュタルク', 3, tmp_x, tmp_y))
    Environment.GRID[tmp_y][tmp_x] = Grid.HUMAN
    
    # エイリアン
    tmp_x = int(conf.get('ENVIRONMENT', 'ENEMY_COORDX', fallback='10'))
    tmp_y = int(conf.get('ENVIRONMENT', 'ENEMY_COORDY', fallback='10'))
    alien.append(Enemy('エイリアン', 1, tmp_x, tmp_y))
    Environment.GRID[tmp_y][tmp_x] = Grid.ALIEN
    
    num_human = 3
    num_alien = 1
    contact = [0, 0, 0]
    history = [[] for _ in range(num_human)]
    
    # 実行ステップ数やエピソード数をカウントする変数
    step = 0
    episode = 0
    max_step = int(conf.get('SYSTEM', 'MAX_STEP', fallback='50'))
    max_episode = int(conf.get('SYSTEM', 'MAX_EPISODE', fallback='10'))

    # グリッドの生成
    grid = Grid(Environment.GRID.shape)

    # グラフのインスタンス生成
    graph = Graph('Simulation result', 'Number of episodes', 'Number of contacts', num_human)

    # ログファイル保存用
    func = Func()

    # 実行するか否かを問う開始ダイアログボックス
    if input('シミュレーションを実行しますか？ (y/n): ') == 'y':
        logger.info('エージェントシミュレーションが開始されました')
        
        # エージェントシミュレーション
        while episode < max_episode:
            logger.info('エピソード%dを実行します', episode)
            
            while step < max_step:
                # グリッドの表示内容設定
                grid.reflect(Environment.GRID, human, alien)
                # グリッド表示の更新
                pg.display.update()
                
                # Do the action
                for i in range(0, num_human, 1):
                    human[i].action(human[i].strategy(alien[0].coord))
                for j in range(0, num_alien, 1):
                    alien[j].action(random.randint(0, 3))
                
                # humanとalienで当たり判定
                for i in range(0, num_human, 1):
                    for j in range(0, num_alien, 1):
                        if (abs(human[i].coord[0] - alien[j].coord[0])) <= 1 and \
                                (abs(human[i].coord[1] - alien[j].coord[1])) <= 1:
                            contact[i] += 1
                            break
                
                step += 1
                time.sleep(0.01)

            # エピソードの終了処理
            for i in range(0, num_human, 1):
                history[i].append(contact[i])
            
            # 初期化
            step = 0
            for i in range(0, num_human, 1):
                contact[i] = 0
            
            # エピソード数のカウントアップ
            episode += 1
            
            # 全エージェントのリスポーン
            for i in range(0, num_human, 1):
                human[i].reset()
            for j in range(0, num_alien, 1):
                alien[j].reset()
            
            graph.data(history)
            plt.pause(.001)

        # 衝突履歴のhistoryをログファイルとして出力
        func.save(np.array(history))
        # グラフの自動保存
        graph.save(func.get_log_date() + '_graph.png')
        
        print('シミュレーションが終了しました．')
        logger.info('エージェントシミュレーションを終了しました')
        
        # pygameの終了処理
        pg.quit()
        # グラフを画面に表示
        graph.show()
        graph.close()
        
        # プロセス終了時のクリーンアップ
        import atexit
        atexit.register(pg.quit)

def moving_average(data, window_size=20):
    import numpy as np
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def run_frieren_training():
    """フリーレン冒険シミュレーションの学習実行"""
    print("=== フリーレン冒険パーティ強化学習開始 ===")
    
    # 環境とエージェントの初期化
    env = EnhancedFrierenAdventureEnv()
    agents = [
        FrierenRLAgent(state_size=16, action_size=4),
        FernRLAgent(state_size=16, action_size=4),
        StarkRLAgent(state_size=16, action_size=4)
    ]
    
    # エージェントを環境に設定
    env.set_agents(agents)
    
    # 学習実行
    scores, win_rates = env.train_agents(episodes=1000, batch_size=64)
    
    # 結果可視化
    import numpy as np
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.3, label='Raw')
    plt.plot(moving_average(scores), color='red', label='Moving Avg')
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    episodes = range(0, len(win_rates) * 100, 100)
    plt.plot(episodes, win_rates, alpha=0.3, label='Raw')
    if len(win_rates) > 5:
        plt.plot(episodes[len(episodes)-len(moving_average(win_rates, 3)):], moving_average(win_rates, 3), color='red', label='Moving Avg')
    plt.title('Win Rate')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("\n=== 学習後のテスト戦闘 ===")
    test_battle(env)
    
    # --- 追加: 状況ごとの行動選択割合ヒートマップ ---
    from ui.plot import Graph
    print("\n=== 状況ごとの行動選択割合ヒートマップ ===")
    graph = Graph('Action Heatmap', '', '', 1)
    # 全キャラまとめて
    graph.plot_action_heatmap(env.action_log, char_name=None, show=True)
    # キャラごと
    for char in ['フリーレン', 'フェルン', 'シュタルク']:
        graph.plot_action_heatmap(env.action_log, char_name=char, show=True)
    # ---
    
    # 修正: 比較実験を追加
    print("\n=== 学習効果の比較実験 ===")
    compare_learning_effect(env, agents, num_battles=50)

def test_battle(env):
    """学習後のテスト戦闘"""
    # 探索率を0にしてテスト
    for agent in env.agents:
        if hasattr(agent, 'epsilon'):
            agent.epsilon = 0
    
    state = env.reset()
    done = False
    turn = 0
    
    print(f"戦闘開始！")
    print(f"パーティ: フリーレン(HP:{env.party[0].hp}), フェルン(HP:{env.party[1].hp}), シュタルク(HP:{env.party[2].hp})")
    print(f"ボス: {env.boss.name}(HP:{env.boss.hp})")
    
    while not done and turn < 50:
        turn += 1
        next_state, rewards, done, info = env.step()
        print(f"\nターン {turn}:")
        print(f"パーティ: フリーレン(HP:{env.party[0].hp}), フェルン(HP:{env.party[1].hp}), シュタルク(HP:{env.party[2].hp})")
        print(f"ボス: {env.boss.name}(HP:{env.boss.hp})")
        state = next_state
    
    if not env.boss.alive:
        print(f"\n🎉 勝利！ {turn}ターンでボスを撃破しました！")
    elif all(not char.alive for char in env.party):
        print(f"\n💀 敗北... パーティが全滅しました...")
    else:
        print(f"\n⏰ 時間切れ...")

def test_manhattan_distance_system():
    """マンハッタン距離システムのテスト"""
    print("=== マンハッタン距離システムテスト ===")
    
    # 環境を初期化
    env = EnhancedFrierenAdventureEnv()
    
    # テスト用のダミーエージェントを作成
    class DummyAgent:
        def get_action(self, state, *args):
            return random.randint(0, 3)  # ランダムな行動
    
    # ダミーエージェントを設定
    dummy_agents = [DummyAgent() for _ in range(3)]
    env.set_agents(dummy_agents)
    
    # マンハッタン距離のテスト
    env.test_manhattan_distance()
    
    # 簡単な戦闘テスト
    print("\n=== 戦闘テスト ===")
    state = env.reset()
    done = False
    turn = 0
    
    print(f"初期状態:")
    print(f"パーティ: フリーレン(HP:{env.party[0].hp}), フェルン(HP:{env.party[1].hp}), シュタルク(HP:{env.party[2].hp})")
    print(f"ボス: {env.boss.name}(HP:{env.boss.hp})")
    print(f"位置情報: {env.character_positions}")
    
    # 5ターンの戦闘をシミュレート
    while not done and turn < 5:
        turn += 1
        next_state, rewards, done, info = env.step()
        
        print(f"\nターン {turn}:")
        print(f"パーティ: フリーレン(HP:{env.party[0].hp}), フェルン(HP:{env.party[1].hp}), シュタルク(HP:{env.party[2].hp})")
        print(f"ボス: {env.boss.name}(HP:{env.boss.hp})")
        print(f"位置情報: {env.character_positions}")
        
        # バトルログを表示
        if info.get('battle_log'):
            print("バトルログ:")
            for log in info['battle_log'][-3:]:  # 最新3件
                print(f"  - {log}")
        
        state = next_state
    
    print(f"\n戦闘終了: {'勝利' if not env.boss.alive else '敗北' if all(not char.alive for char in env.party) else '継続'}")

if __name__ == "__main__":
    print("=== フリーレン協力型グリッドワールド強化学習シミュレーション ===")
    print("実行モードを選択してください:")
    print("1. マンハッタン距離システムテスト")
    print("2. グリッドワールドシミュレーション")
    print("3. フリーレン強化学習シミュレーション")
    print("4. 全て実行")
    
    try:
        choice = input("選択してください (1-4): ").strip()
        
        if choice == "1":
            test_manhattan_distance_system()
        elif choice == "2":
            main()
        elif choice == "3":
            run_frieren_training()
        elif choice == "4":
            # マンハッタン距離システムのテストを追加
            test_manhattan_distance_system()
            
            # まずグリッドワールドシミュレーションを実行
            main()
            # 続けてフリーレン強化学習シミュレーションを実行
            run_frieren_training()
        else:
            print("無効な選択です。デフォルトで全て実行します。")
            test_manhattan_distance_system()
            main()
            run_frieren_training()
            
    except KeyboardInterrupt:
        print("\nプログラムが中断されました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("デフォルトで全て実行します。")
        test_manhattan_distance_system()
        main()
        run_frieren_training() 