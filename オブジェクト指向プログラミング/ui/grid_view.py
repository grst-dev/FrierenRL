"""
グリッド表示クラス
"""

import pygame as pg
import os


class Grid:
    """グリッド表示クラス"""
    
    # 修正: マジックナンバーを定数として定義
    ROAD = 0      # 通路の番号
    WALL = 1      # 壁の番号
    OBST = 2      # 障害物の番号
    HUMAN = 9     # 人間
    ALIEN = 8     # エイリアン
    CS = 50       # 単位長さの定義
    
    def __init__(self, grid_shape):
        self.grid_shape = grid_shape
        self.SCR_X = grid_shape[1] * self.CS  # 矩形のX軸の長さ定義
        self.SCR_Y = grid_shape[0] * self.CS  # 矩形のY軸の長さ定義
        
        pg.init()  # PyGameの初期化（一度だけコール）
        self.rect = pg.Rect(0, 0, self.SCR_X, self.SCR_Y)  # 矩形の大きさ宣言
        self.screen = pg.display.set_mode(self.rect.size)
        pg.display.set_caption(u'Grid world')
        self.screen.fill((255, 255, 255))
        
        # 修正: キャラクター画像の動的ロード
        self.char_img_map = self._load_character_images()
    
    def _load_character_images(self):
        """キャラクター画像の動的ロード"""
        char_img_map = {}
        images_dir = 'images'
        
        # 画像ディレクトリが存在しない場合は作成
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
            print(f"画像ディレクトリを作成しました: {images_dir}")
        
        # キャラクター名とファイル名のマッピング
        char_files = {
            'フリーレン': 'frieren.png',
            'フェルン': 'fern.png',
            'シュタルク': 'stark.png',
            'エイリアン': 'aura.png'  # アウラの画像
        }
        
        for char_name, filename in char_files.items():
            filepath = os.path.join(images_dir, filename)
            if os.path.exists(filepath):
                try:
                    char_img_map[char_name] = pg.transform.scale(
                        pg.image.load(filepath), (self.CS, self.CS)
                    )
                except pg.error as e:
                    print(f"画像読み込みエラー {filepath}: {e}")
                    char_img_map[char_name] = self._create_default_image(char_name)
            else:
                # 警告ではなく、デフォルト画像を生成
                char_img_map[char_name] = self._create_default_image(char_name)
        
        return char_img_map
    
    def _create_default_image(self, char_name="キャラクター"):
        """デフォルト画像（色分けされた円形）を生成"""
        surface = pg.Surface((self.CS, self.CS), pg.SRCALPHA)
        
        # キャラクター名に応じて色を設定
        color_map = {
            'フリーレン': (255, 100, 100),  # 赤（魔法使い）
            'フェルン': (100, 255, 100),    # 緑（ヒーラー）
            'シュタルク': (100, 100, 255),  # 青（戦士）
            'エイリアン': (255, 255, 100),  # 黄（敵）
        }
        
        color = color_map.get(char_name, (150, 150, 150))  # デフォルトはグレー
        pg.draw.circle(surface, color, (self.CS//2, self.CS//2), self.CS//3)
        
        # キャラクター名の頭文字を表示
        font = pg.font.Font(None, self.CS//4)
        text = font.render(char_name[0], True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.CS//2, self.CS//2))
        surface.blit(text, text_rect)
        
        return surface
    
    def reflect(self, grid_data, human=None, alien=None):
        """グリッドの表示と各アイテムの表示"""
        for x in range(0, grid_data.shape[1]):
            for y in range(0, grid_data.shape[0]):
                tmp_rect = pg.Rect(x*self.CS, y*self.CS, self.CS, self.CS)
                
                # 通路の着色と枠線の着色
                if grid_data[y][x] == self.ROAD:
                    pg.draw.rect(self.screen, (255, 255, 255), tmp_rect)
                    pg.draw.rect(self.screen, (0, 0, 0), tmp_rect, 1)
                # 壁の着色
                elif grid_data[y][x] == self.WALL:
                    pg.draw.rect(self.screen, (0, 0, 0), tmp_rect)
                # 障害物の着色
                elif grid_data[y][x] == self.OBST:
                    pg.draw.rect(self.screen, (0, 0, 0), tmp_rect)
                # エージェント画像の描画
                elif grid_data[y][x] == self.HUMAN and human is not None:
                    # humanリストから該当座標のキャラを探す
                    for h in human:
                        if h.coord == (x, y):
                            # 修正: 動的画像ロード
                            char_name = h.get_name()
                            img = self.char_img_map.get(char_name, self._create_default_image())
                            self.screen.blit(img, tmp_rect)
                            break
                    pg.draw.rect(self.screen, (0, 0, 0), tmp_rect, 1)
                elif grid_data[y][x] == self.ALIEN and alien is not None:
                    # alienリストから該当座標のキャラを探す
                    for a in alien:
                        if a.coord == (x, y):
                            # 修正: 動的画像ロード
                            char_name = a.get_name()
                            img = self.char_img_map.get(char_name, self._create_default_image())
                            self.screen.blit(img, tmp_rect)
                            break
                    pg.draw.rect(self.screen, (0, 0, 0), tmp_rect, 1) 