"""
ログ機能クラス
"""

import csv
import datetime


class Func:
    """ログ機能クラス"""
    
    def __init__(self):
        self.__log_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save(self, history):
        """エージェントの衝突履歴をcsvファイルに出力"""
        file_name = open("./log_steps_" + self.__log_date + ".csv", 'w', newline='')
        writer = csv.writer(file_name)
        writer.writerows(history.T)
        file_name.close()
    
    def get_log_date(self):
        """インスタンス生成時のdateの取得"""
        return self.__log_date
    
    def re_date(self):
        """実行時分秒の更新メソッド"""
        self.__log_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 