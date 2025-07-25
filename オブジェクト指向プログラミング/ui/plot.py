"""
グラフ描画クラス
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Graph:
    """グラフ描画クラス"""
    
    def __init__(self, name: str, x_axis: str, y_axis: str, num: int):
        self.__num = num
        self.__fig = plt.figure(figsize=(5*self.__num, 5))
        self.__ax = []
        for i in range(0, self.__num, 1):
            self.__ax.append(self.__fig.add_subplot(1, self.__num, i+1))
            self.__ax[i].set_title(name + ' no.' + str(i))
            self.__ax[i].set_xlabel(x_axis)
            self.__ax[i].set_ylabel(y_axis)
    
    def data(self, history):
        """データの描画"""
        for i in range(0, self.__num, 1):
            self.__ax[i].plot(history[i], color='blue')
    
    def save(self, filename):
        """グラフの保存"""
        plt.savefig(filename)
    
    def show(self):
        """グラフの表示"""
        plt.show()
    
    def close(self):
        """グラフの閉じる"""
        plt.close() 

    def plot_action_heatmap(self, action_log, char_name=None, save_path=None, show=True):
        """
        状況（HP区分×距離区分）ごとの行動選択割合ヒートマップを描画
        action_log: [((hp_level, dist_level), action, char_name, turn), ...]
        char_name: 指定したキャラのみ可視化（Noneなら全員分集計）
        """
        import matplotlib
        # 利用可能な日本語フォントを優先的に指定（なければ自動でfallback）
        matplotlib.rcParams['font.family'] = ['IPAexGothic', 'Noto Sans CJK JP', 'Yu Gothic', 'MS Gothic', 'sans-serif']
        # 状況×行動のカウント
        count = np.zeros((3*3, 4))  # HP区分3×距離区分3, 行動4種
        total = np.zeros(3*3)
        for (hp_level, dist_level), action, name, turn in action_log:
            if (char_name is not None) and (name != char_name):
                continue
            idx = hp_level * 3 + dist_level
            count[idx, action] += 1
            total[idx] += 1
        # 割合に変換
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(count, total[:, None], out=np.zeros_like(count), where=total[:, None]!=0)
        # ラベル
        yticklabels = [f"HP:{['低','中','高'][i//3]}, 距離:{['近','中','遠'][i%3]}" for i in range(9)]
        xticklabels = ['攻撃', '防御', '回復', 'スキル']
        plt.figure(figsize=(8, 6))
        sns.heatmap(ratio, annot=True, cmap='Blues', xticklabels=xticklabels, yticklabels=yticklabels, fmt='.2f')
        plt.xlabel('行動（攻撃／防御／回復／スキル）')
        plt.ylabel('状況（HP区分, 距離区分）')
        title = f'状況ごとの行動選択割合ヒートマップ' + (f'（{char_name}）' if char_name else '（全キャラ）')
        plt.title(title)
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close() 