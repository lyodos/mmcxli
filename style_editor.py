#!/usr/bin/env python3

# The MIT License

# Copyright (c) 2024 Lyodos

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# 以下に定める条件に従い、本ソフトウェアおよび関連文書のファイル（以下「ソフトウェア」）の複製を取得するすべての人に対し、ソフトウェアを無制限に扱うことを無償で許可します。これには、ソフトウェアの複製を使用、複写、変更、結合、掲載、頒布、サブライセンス、および/または販売する権利、およびソフトウェアを提供する相手に同じことを許可する権利も無制限に含まれます。

# 上記の著作権表示および本許諾表示を、ソフトウェアのすべての複製または重要な部分に記載するものとします。

# ソフトウェアは「現状のまま」で、明示であるか暗黙であるかを問わず、何らの保証もなく提供されます。ここでいう保証とは、商品性、特定の目的への適合性、および権利非侵害についての保証も含みますが、それに限定されるものではありません。作者または著作権者は、契約行為、不法行為、またはそれ以外であろうと、ソフトウェアに起因または関連し、あるいはソフトウェアの使用またはその他の扱いによって生じる一切の請求、損害、その他の義務について何らの責任も負わないものとします。 

import wx
import math
import json
import copy

import numpy as np

import logging
import inspect

####

class AxesEditPanel(wx.Panel):
    def __init__(
        self, 
        parent, 
        id = -1,
        manager = None,
        axes_pane_size: tuple = (360, 360), # pane のサイズ（もし複数あるならば各々の）
        border_at: list = [50.0, 50.0], # = (x, y) であるとき dim0, dim1 各軸について、 [-x, x], [-y, y] が描画範囲になる
        tick_interval = None,
        tab_id: int = None, # 何番目のタブに属するか。カーソルの当たり判定に使う
        **kwargs,
    ):
        super().__init__(parent, id = id, **kwargs)

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        self.logger.debug("Initializing...")
        
        self.manager = manager # SampleManagerPanel インスタンスを指定
        self.tab_id = tab_id

        self.axes_pane_size = axes_pane_size
        self.border_at = border_at
        self.tick_interval = tick_interval if tick_interval is not None else 10 ** int(math.log10(max(border_at)) - 0.17)

        # Manager が保持しているカラーパレットを覗けるようにしておく
        self.slot_fill_palette = self.manager.slot_fill_palette
        self.slot_color_palette = self.manager.slot_color_palette
        self.slot_border_palette = self.manager.slot_border_palette

        self.active_slot_index = self.manager.active_slot_index

        # Axes pane で手動入力した特徴量、およびファイルからロードして計算した圧縮埋め込みを保持するリスト
        # 全体の長さは manager の max slots に一致し、各要素は圧縮次元数に一致する長さの数値リストである。
        # つまり、現在の値はスロットごとに 1 つしか保持しない。
        self.file_features_list = [] # ファイルをロードして計算した圧縮埋め込み（正本は manager が保持）
        self.handmade_features_list = [] # Axes pane で手動入力する特徴量
        
        for slot in self.manager.slot_list:
            self.file_features_list.append(None)
            # 単なる None 初期化ではなく、手動入力値をロードした場合に対応させる
            if slot.emb_handmade is not None:
                self.handmade_features_list.append(slot.emb_handmade.flatten().tolist()) 
#                self.handmade_features_list.append([float(slot.emb_handmade[0, 0]), float(slot.emb_handmade[0, 1])]) 
                
            else:
                self.handmade_features_list.append([None]*self.manager.dim_comp) 
            
        # self.handmade_features_list の内容が更新され、埋め込みの再計算が必要である場合 True となるフラグ
        self.need_recalc_features = [False]*self.manager.max_slots 
        
        
        # 1 ペインで 2 軸ずつ担当。圧縮埋め込みの次元数がさらに多い場合は pane_23, pane_45, ... と続けて定義する
        self.pane_01 = AxesPane(
            self, 
            dim = (0, 1), 
            size = self.axes_pane_size, 
            host = self,
            tab_id = self.tab_id,
            border_at = border_at[0:2], 
            tick_interval = self.tick_interval,
            bg_image_path = "./images/emb_dim_01.png",
        )
        self.pane_01_sizer = wx.BoxSizer(wx.VERTICAL)
        self.pane_01_sizer.Add(self.pane_01, 0, wx.RIGHT | wx.BOTTOM, 5)

        # グリッドを作る。ポジションを指定して pane を追加
        self.bag_sizer = wx.GridBagSizer(vgap = 0, hgap = 0) # 行間 0 px, 列間 0 px
        self.bag_sizer.Add(self.pane_01_sizer, pos = (0, 0), flag = wx.ALL | wx.ALIGN_CENTER, border = 0) 

        self.bag_size = self.bag_sizer.GetSize()
        self.bag_sizer.SetMinSize(self.bag_size) # 最低サイズ。これがないと  assertion 'size >= 0' failed in GtkScrollbar
        
        # 統括 sizer を作り、すべての部品を配置していく。
        self.root_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.root_sizer.Add(self.bag_sizer, 0, wx.EXPAND | wx.LEFT, 0)
        self.SetSizer(self.root_sizer)
        self.Layout()

        self.timer = wx.Timer(self) # wx.Timer クラスで、指定間隔での処理を実行する。
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.timer.Start(137) # タイマーは手動で起動する必要がある。単位 ms


    def update(self, event):
        self.active_slot_index = self.manager.active_slot_index # アクティブスロットは外側で更新されるので反映

        # 全部のスロットについて状態更新が走る
        for i, slot in enumerate(self.manager.slot_list):
            # EditPanel で扱うべき変数について、Slot から現在の値を取得する
            # ファイルからのロード有無は、slot.style_abs_path is not None でも slot.style_comp is not None でも調べられる
            if slot.style_abs_path is not None:
                self.file_features_list[i] = slot.style_comp[0, :].tolist()
            else:
                self.file_features_list[i] = None 

            # 手動入力特徴量の有無のフラグはこちらで作るので、実態に応じてスロット側へ反映させる
            if self.handmade_features_list[i][0] is not None:
                slot.has_handmade_features = True
            else:
                slot.has_handmade_features = False

            # 各スロットの self.handmade_features_list の値が更新されていたら、numpy array に変換して格納＆伸長
            # AxesPane 内のマウスクリック（下記）により True が入るので、処理を行ってからフラグを False に戻す
            if self.need_recalc_features[i] == True:
                if self.handmade_features_list[i][0] is not None:
                    # 手動入力値を作った
                    slot.emb_handmade = feature_to_array(self.handmade_features_list[i])
                    slot.emb_expand = self.manager.sess_SD.run(
                        ['emb'], 
                        {'comp': slot.emb_handmade},
                    )[0] # (1, self.dim_comp) -> (1, self.dim_style)
                else:
                    # 手動入力値を消した → 初期値に戻す
                    slot.emb_handmade = None
                    slot.emb_expand = copy.deepcopy(self.manager.style_silent)
                
                slot.plot_embedding() # embedding の圧縮表現と復元も作ったので、plot に反映する
                self.overwrite_handmade_styles(i, slot)
                
                # スロットのラベルの更新も必要
                slot.update_style_label()
                
                slot.Refresh()  # 再描画をトリガー
                self.need_recalc_features[i] = False


    # 設定をファイルとして保存
    # 同名のメソッドが StyleSlotPanel にもあるが、そちらはファイル由来の embedding を入れる機能なので使い分ける
    
    def overwrite_handmade_styles(
        self, 
        slot_index,
        slot,
    ):
        try:
            # 絶対パスから相対パスに変換。絶対パスを json に残すと個人情報保護や複数マシン間での共有にリスク
            # ファイルを開くたびに config ファイルを更新する
            if self.handmade_features_list[slot_index][0] is not None:
                # アクティブスロットに手動入力特徴量がある
                self.manager.style_portfolio[slot_index]["emb_handmade"] = slot.emb_handmade.tolist()
                self.manager.style_portfolio[slot_index]["emb_expand"] = slot.emb_expand.tolist()
            else:
                # アクティブスロットの手動入力特徴量を消した
                self.manager.style_portfolio[slot_index]["emb_handmade"] = None
                self.manager.style_portfolio[slot_index]["emb_expand"] = None
            self.manager.style_portfolio[slot_index]["handmade_style_name"] = slot.handmade_style_name
            
            with open(self.manager.style_portfolio_path, 'w') as f:
                json.dump(self.manager.style_portfolio, f, indent = 4)
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Style gallery was saved to '{self.manager.style_portfolio_path}'")
        except:
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Failed to save the style gallery.")


####

# None を 0 に置換して、サイズ (1, n) の numpy float32 array に変換する関数
def feature_to_array(feature_list: list):
    cleaned_list = [0 if x is None else x for x in feature_list] # None を 0 に置換
    numpy_array = np.array(cleaned_list, dtype = np.float32)
    numpy_array = numpy_array.reshape((1, -1))
    return numpy_array


####

# この Pane の直接の親は AxesEditPanel であり、さらにその親が StyleManagerPanel

# どのスロットがアクティブであるかの情報は StyleManagerPanel が持つ。
# 実際の圧縮表現のスタイル値は StyleManagerPanel の中の slot が持つ。

# 筋を通すならば、現在のアクティブスロットの n 次元圧縮埋め込みを Manager から受け取って保持するのは AxesEditPanel 
# AxesPane クラスの中には埋め込みの変数は保持せず、AxesEditPanel やさらに Manager が持つ変数の書き換えに専念する。


class AxesPane(wx.Panel):
    def __init__(
        self, 
        parent, 
        id = -1,
        host = None, # AxesEditPanel
        tab_id: int = None, # 何番目のタブに属するか。カーソルの当たり判定に使う
        dim: tuple = (0, 1), # スタイル潜在空間の、どの次元（2 つ）を担当するか。
        size: tuple = (720, 720),
        border_at: list = [20.0, 20.0], # 必ず中心を (x, y) = (0, 0)  として、描画範囲の右端、上端となる値を [x, y] で指定
        bg_image_path: str = None,
        draw_grid: bool = True, # グリッド線も描画する
        tick_interval: float = 1.0, # 単位は埋め込みのサイズに等しい。下で pixel 単位に換算して描画する
        tick_height: int = 2, # 上下（左右）に tick_height px ずつの tickmark を描く
        label_with_mouse: bool = False, # マウスに追随して現在値のラベルも移動する。Windows で不用意にブリンクするので不採用
        **kwargs,
    ):
        super().__init__(parent, id = id, size = size, **kwargs)

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        # フレームの背景スタイルを設定する。この処理は wx.BufferedPaintDC(self) を使用してフリッカーを抑えるためのもの
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

        self.host = host # AxesEditPanel インスタンスを指定
        self.tab_id = tab_id

        self.light_gray_color = wx.Colour(215, 215, 225)
        self.night_gray_color = wx.Colour(23, 23, 27) # この色は背景画像に合わせてあるので、任意には変更できない。

        self.dim = dim
        self.bg_image_path = bg_image_path
        self.draw_grid = draw_grid
        self.tick_interval = tick_interval
        self.tick_h = tick_height
        self.label_with_mouse = label_with_mouse

        self.client_size = size # ウィンドウのクライアント領域のサイズ
        self.border_at = border_at
        self.mag_base = [s//2 / b for s, b in zip(self.client_size, self.border_at)] # プロット時の拡大率を計算
        self.mag_coef: float = 1.0 # （そのうち実装したい）マウススクロールで拡大率を変える機能
        self.tick_interval_px = [
            self.client_size[0] / (self.border_at[0]*2) * self.tick_interval,
            self.client_size[1] / (self.border_at[1]*2) * self.tick_interval,
        ]
        self.mouse_client_pos = None # 現在の Panel クライアント領域の左上からの、マウスの相対座標
        self.mouse_coord = (0, 0) # 現在の Panel クライアント領域の中央からの、マウスの相対座標
        self.mouse_inside = False # 現在のパネルにマウスが入っているか
        self.close_pos_threshold = 10 # マウスからの距離がこの閾値よりも近い点に当たり判定を付ける

        # マウスカーソル上に現在のマウス位置を表示する
        self.initial_mouse_pos = (7, 7)  # 表示場所
        self.mouse_pos_label = wx.StaticText(self, label = f"Axes {self.dim[0]}/{self.dim[1]}")
        self.mouse_pos_label.SetFont(wx.Font(11, wx.MODERN, wx.NORMAL, wx.BOLD))
        self.mouse_pos_label.SetPosition(self.initial_mouse_pos) 
        self.mouse_pos_label.SetBackgroundColour(self.night_gray_color)
        self.mouse_pos_label.SetForegroundColour(self.light_gray_color)

        self.mouse_pos_sizer = wx.BoxSizer(wx.VERTICAL)
        self.mouse_pos_sizer.Add(self.mouse_pos_label, 0, wx.ALL, 7) # ここのマージンが 0 だとラベルが初期配置されない不具合
        self.SetSizer(self.mouse_pos_sizer)
        
        self.SetBackgroundColour(self.night_gray_color)

        # 背景画像と軸線を描画したビットマップを作成
        self.background_bitmap = self.create_background_bitmap()
        
        self.Bind(wx.EVT_PAINT, self.on_paint) # 画像をパネル内に貼り付ける
        self.Bind(wx.EVT_LEFT_UP, self.on_mouse_up) # マウスの左ボタンのクリック「終了」タイミング
        self.Bind(wx.EVT_RIGHT_UP, self.on_mouse_right_up) # マウスの右ボタンのクリック「終了」タイミング
        
        self.timer = wx.Timer(self) # wx.Timer クラスで、指定間隔での処理を実行する。
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.timer.Start(57) # タイマーは手動で起動する必要がある。単位 ms

    # update はマウスの動きをキャプチャする機能。対応するスタイル空間上の座標を self.mouse_value で取れる。

    # ただし現在、Linux で以下のメッセージが出る
    # Debug: ScreenToClient cannot work when toplevel window is not shown
    
    def update(self, event):
        if self.GetTopLevelParent().active_tab == self.tab_id:
            # タブが開いているときのみ実行する。さもないと他のタブでもカーソルの形が変わってしまう
            self.client_size = self.GetClientSize() 
            mouse_abs_pos = wx.GetMousePosition() # wx.GetMousePosition() ならばウィンドウ外でもマウス位置を取れる
            self.mouse_client_pos = self.ScreenToClient(mouse_abs_pos) # 現在の Panel クライアント領域の左上からの相対座標
            # 現在の Panel クライアント領域の中央からの相対座標
            self.mouse_coord = [
                self.mouse_client_pos.x - self.client_size[0]/2, 
                self.mouse_client_pos.y - self.client_size[1]/2
            ]
            # 拡大率を考慮した、現在のマウス位置に対応するスタイル空間の座標。なお y 軸が上下逆になるので注意
            self.mouse_value = [
                round(self.mouse_coord[0] / self.mag_base[0], 2), 
                round(-self.mouse_coord[1] / self.mag_base[1], 2)
            ]
            # ウィンドウ内にマウスがあるかどうかを判定
            if 0 <= self.mouse_client_pos.x < self.client_size[0] and 0 <= self.mouse_client_pos.y < self.client_size[1]:
                self.mouse_inside = True
            else:
                self.mouse_inside = False
        
            # カーソルに追随するようにマウス位置ラベルの位置を調整
            if self.mouse_inside:
                # 十字カーソルに変更
                self.SetCursor(wx.Cursor(wx.CURSOR_CROSS))
                # ラベルには現在の座標値を記載する
                self.mouse_pos_label.SetLabel(
                    f"Axes {self.dim[0]}/{self.dim[1]}\n{self.mouse_value[0]}, {self.mouse_value[1]}"
                )
                # マウス位置ラベルをカーソルに追随させるか。現在デフォルトは False
                if self.label_with_mouse is True:
                    if self.mouse_coord[0] >= 0:
                        # カーソルの左に表示
                        self.mouse_pos_label.SetPosition(
                            (self.mouse_client_pos.x - self.mouse_pos_label.GetSize()[0] - 15, self.mouse_client_pos.y - 20)
                        ) 
                    else:
                        # カーソルの右に表示
                        self.mouse_pos_label.SetPosition(
                            (self.mouse_client_pos.x + 20, self.mouse_client_pos.y - 20)
                        ) 
            else:
                # 通常の矢印カーソルに戻す
                self.SetCursor(wx.Cursor(wx.CURSOR_ARROW))
                # ラベルには現在の座標値を記載しない
                self.mouse_pos_label.SetLabel(f"Axes {self.dim[0]}/{self.dim[1]}")
                self.mouse_pos_label.SetPosition(self.initial_mouse_pos) 

            self.Refresh()  # 再描画をトリガー。そのままだとフリッカーが発生するので、on_paint で対策している


    def on_paint(self, event):
        panel_width, panel_height = self.GetSize() 
        mouse_abs_pos = wx.GetMousePosition() # wx.GetMousePosition() ならばウィンドウ外でもマウス位置を取れる
        self.mouse_client_pos = self.ScreenToClient(mouse_abs_pos) # 現在の Panel クライアント領域の左上からの相対座標
        
        dc = wx.AutoBufferedPaintDC(self) # wx.PaintDC だと Windows でフリッカーが発生するため切り替えた
        dc.DrawBitmap(self.background_bitmap, 0, 0, True) # create_background_bitmap で作成済みの背景画像を描画

        size = 8 # 非アクティブのポイントの表示サイズ
        focus_size = 14 # アクティブ（選択中ないし近傍）なポイントの表示サイズ
        
        # まず、ファイルからロードした圧縮埋め込みの座標を表示する。非アクティブなスロットも（色を変えて）同時表示する
        for i, file_feature in enumerate(self.host.file_features_list):
            if file_feature is not None and self.mouse_client_pos is not None:
                cl_pos_x = round(file_feature[0] * self.mag_base[0] + panel_width/2)
                cl_pos_y = round(panel_height/2 - file_feature[1] * self.mag_base[1]) # クライアント領域の Y 座標は反転
                dist = ((self.mouse_client_pos[0] - cl_pos_x)**2 + (self.mouse_client_pos[1] - cl_pos_y)**2) ** 0.5
                if i == self.host.active_slot_index:
                    # アクティブなスロット
                    if dist < self.close_pos_threshold:
                        # カーソルから一定距離以内のポイントだけ目立たせる
                        self.draw_arrow(
                            dc, 
                            (cl_pos_x, cl_pos_y), # (x, y) 描画対象のクライアント領域における座標
                            panel_width,
                            panel_height,
                            draw_line = False, # 原点からポイントへの線を描画する
                            draw_perpendiculars = True, # ポイントから x 軸、y 軸への垂線を描画する
                            pen = wx.Pen(wx.Colour(55, 245, 225), 2, wx.PENSTYLE_SHORT_DASH),
                            arrow_pen = None,
                            arrow_fill = wx.Brush(wx.Colour(0, 0, 0), wx.TRANSPARENT),
                            arrow_type = "rect",
                            arrow_size = focus_size,
                        )
                    else:
                        self.draw_arrow(
                            dc, 
                            (cl_pos_x, cl_pos_y), # (x, y) 描画対象のクライアント領域における座標
                            panel_width,
                            panel_height,
                            draw_line = False, # 原点からポイントへの線を描画する
                            draw_perpendiculars = True, # ポイントから x 軸、y 軸への垂線を描画する
                            pen = wx.Pen(wx.Colour(self.host.slot_color_palette[i]), 1),
                            arrow_pen = wx.Pen(wx.Colour(self.host.slot_color_palette[i]), 2),
                            arrow_fill = wx.Brush(wx.Colour(self.host.slot_border_palette[i])),
                            arrow_type = "rect",
                            arrow_size = focus_size,
                        )
                else:
                    # 現在アクティブでないスロットたち
                    if dist < self.close_pos_threshold:
                        # カーソルから一定距離以内のポイントだけ目立たせる
                        self.draw_arrow(
                            dc, 
                            (cl_pos_x, cl_pos_y), # (x, y) 描画対象のクライアント領域における座標
                            panel_width,
                            panel_height,
                            draw_line = False, # 原点からポイントへの線を描画する
                            draw_perpendiculars = True, # ポイントから x 軸、y 軸への垂線を描画する
                            pen = wx.Pen(wx.Colour(self.host.slot_color_palette[i]), 2, wx.PENSTYLE_LONG_DASH),
                            arrow_pen = wx.Pen(wx.Colour(self.host.slot_color_palette[i]), 2),
                            arrow_fill = wx.Brush(wx.Colour(self.host.slot_color_palette[i]), wx.TRANSPARENT),
                            arrow_type = "rect",
                            arrow_size = focus_size,
                        )
                    else:
                        self.draw_arrow(
                            dc, 
                            (cl_pos_x, cl_pos_y), # (x, y) 描画対象のクライアント領域における座標
                            panel_width,
                            panel_height,
                            draw_line = False, # 原点からポイントへの線を描画する
                            draw_perpendiculars = True, # ポイントから x 軸、y 軸への垂線を描画する
                            pen = wx.Pen(wx.Colour(self.host.slot_color_palette[i]), 1, wx.PENSTYLE_DOT),
                            arrow_pen = wx.Pen(wx.Colour(self.host.slot_color_palette[i]), 2),
                            arrow_fill = wx.Brush(wx.Colour(self.host.slot_border_palette[i])),
                            arrow_type = "rect",
                            arrow_size = size,
                        )

        # 次に、手動入力した座標を表示する（現在の選択スロットのみが対象）。
        # 現在の選択スロットの情報は host に存在する。ただしどのスロットも未選択の場合がある
        for i, feature_elements in enumerate(self.host.handmade_features_list):
            current_slot_color = wx.Colour(self.host.slot_color_palette[i]) # このスロットのテーマ色
            handmade_features = feature_elements[self.dim[0]:self.dim[1]+1]
            # handmade_features の値は None の場合がありうるので、まず判定が必要
            if isinstance(handmade_features[0], (int, float)) and isinstance(handmade_features[1], (int, float)):
                cl_pos_X = round(handmade_features[0] * self.mag_base[0] + panel_width/2)
                cl_pos_Y = round(panel_height/2 - handmade_features[1] * self.mag_base[1]) # クライアント領域の Y 座標は反転
                # 距離で描画スタイルを変える
                dist = ((self.mouse_client_pos[0] - cl_pos_X)**2 + (self.mouse_client_pos[1] - cl_pos_Y)**2) ** 0.5
                # 描画スタイルはアクティブスロットとそれ以外で変化する
                if i == self.host.active_slot_index:
                    if dist < self.close_pos_threshold:
                        self.draw_arrow(
                            dc, 
                            (cl_pos_X, cl_pos_Y), # (x, y) 描画対象のクライアント領域における座標
                            panel_width,
                            panel_height,
                            draw_line = True, # 原点からポイントへの線を描画する
                            draw_perpendiculars = False, # ポイントから x 軸、y 軸への垂線を描画する
                            pen = wx.Pen(wx.Colour(55, 245, 225), 3, wx.PENSTYLE_SHORT_DASH),
                            arrow_pen = None,
                            arrow_fill = wx.Brush(current_slot_color, wx.TRANSPARENT),
                            arrow_type = "circle",
                            arrow_size = 9,
                        )
                    else:
                        self.draw_arrow(
                            dc, 
                            (cl_pos_X, cl_pos_Y), # (x, y) 描画対象のクライアント領域における座標
                            panel_width,
                            panel_height,
                            draw_line = True, # 原点からポイントへの線を描画する
                            draw_perpendiculars = False, # ポイントから x 軸、y 軸への垂線を描画する
                            pen = wx.Pen(current_slot_color, 2, wx.PENSTYLE_SHORT_DASH),
                            arrow_pen = wx.Pen(current_slot_color, 2),
                            arrow_fill = wx.Brush(wx.Colour(0, 0, 0), wx.TRANSPARENT),
                            arrow_type = "circle",
                            arrow_size = 9,
                        )
                else:
                    if dist < self.close_pos_threshold:
                        self.draw_arrow(
                            dc, 
                            (cl_pos_X, cl_pos_Y), # (x, y) 描画対象のクライアント領域における座標
                            panel_width,
                            panel_height,
                            draw_line = True, # 原点からポイントへの線を描画する
                            draw_perpendiculars = False, # ポイントから x 軸、y 軸への垂線を描画する
                            pen = wx.Pen(wx.Colour(self.host.slot_color_palette[i]), 2, wx.PENSTYLE_LONG_DASH),
                            arrow_pen = wx.Pen(wx.Colour(self.host.slot_color_palette[i]), 2),
                            arrow_fill = wx.Brush(wx.Colour(self.host.slot_color_palette[i]), wx.TRANSPARENT),
                            arrow_type = "circle",
                            arrow_size = 7,
                        )
                    else:
                        self.draw_arrow(
                            dc, 
                            (cl_pos_X, cl_pos_Y), # (x, y) 描画対象のクライアント領域における座標
                            panel_width,
                            panel_height,
                            draw_line = True, # 原点からポイントへの線を描画する
                            draw_perpendiculars = False, # ポイントから x 軸、y 軸への垂線を描画する
                            pen = wx.Pen(wx.Colour(self.host.slot_color_palette[i]), 1, wx.PENSTYLE_LONG_DASH),
                            arrow_pen = wx.Pen(wx.Colour(self.host.slot_color_palette[i]), 2),
                            arrow_fill = wx.Brush(wx.Colour(self.host.slot_border_palette[i])),
                            arrow_type = "circle",
                            arrow_size = 4,
                        )


    # ちなみに、手動入力したポイントは 〇 および原点からの線で、ファイルから読み込んだポイントは □ および軸への垂線で表す

    # dc を指定してポイントへの矢印を描画する
    def draw_arrow(
        self, 
        dc, 
        cl_pos,
        panel_width,
        panel_height,
        draw_line: bool = True, # 原点からポイントへの線を描画する
        draw_perpendiculars: bool = False, # ポイントから x 軸、y 軸への垂線を描画する
        pen = None,
        arrow_pen = None,
        arrow_fill = None,
        arrow_type: str = "rect",
        arrow_size: int = 8,
    ):
        # line
        dc.SetPen(pen)
        if draw_line is True:
            dc.DrawLine(panel_width//2, panel_height//2, cl_pos[0], cl_pos[1]) # ラインを描画
        if draw_perpendiculars is True:
            dc.DrawLine(cl_pos[0], cl_pos[1], cl_pos[0], panel_height//2) # x 軸への垂線を描画
            dc.DrawLine(cl_pos[0], cl_pos[1], panel_width//2, cl_pos[1]) # y 軸への垂線を描画

        # arrow
        if arrow_pen is None:
            arrow_pen = pen
        dc.SetPen(arrow_pen)
        dc.SetBrush(arrow_fill)
        if arrow_type == "rect":
            dc.DrawRectangle(cl_pos[0] - arrow_size//2, cl_pos[1] - arrow_size //2, arrow_size, arrow_size) 
        elif arrow_type == "circle":
            dc.DrawCircle(cl_pos[0], cl_pos[1], arrow_size) 
        else:
            pass


    # dc を指定して星形を描く（現在は使用していない）
    def draw_star(
        self, 
        dc, 
        x, 
        y, 
        radius: float = 30, 
        inner_radius: float = 12, # 星の内側の頂点の半径
        vertices: int = 5,
    ):
        if vertices < 2:
            return
        angle = (1 * math.pi) / vertices
        points = []

        for i in range(vertices*2):
            r = inner_radius if i % 2 == 0 else radius
            points.append(
                (
                    round(x + r * math.cos(i * angle + math.pi/2)), 
                    round(y + r * math.sin(i * angle + math.pi/2)),
                )
            )
        
        dc.DrawPolygon(points, xoffset = 0, yoffset = 0)

    ####

    # マウスの左クリック（の、ボタンを押し終えた瞬間）
    def on_mouse_up(self, event):
        if self.GetTopLevelParent().active_tab == self.tab_id:
            # AxesEditPanel の変数の値を、ここから書き変える 
            panel_width, panel_height = self.GetSize() 
            self.host.handmade_features_list[self.host.active_slot_index][self.dim[0]] = round((self.mouse_client_pos[0] - panel_width/2) / self.mag_base[0], 4)
            self.host.handmade_features_list[self.host.active_slot_index][self.dim[1]] = round((panel_height/2 - self.mouse_client_pos[1]) / self.mag_base[1], 4)
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Clicked {self.mouse_client_pos} -> {self.host.handmade_features_list}")
            self.host.need_recalc_features[self.host.active_slot_index] = True # feature の再計算をトリガー
            self.Refresh()  # 再描画をトリガー
        event.Skip() 


    # マウスの右クリック（の、ボタンを押し終えた瞬間）
    def on_mouse_right_up(self, event):
        if self.GetTopLevelParent().active_tab == self.tab_id:
            # 手動入力した埋め込み座標と、現在のマウス位置との距離を測定し、一定以下なら座標を消去
            # ただしアクティブスロットのみが対象
            handmade_features = self.host.handmade_features_list[self.host.active_slot_index][self.dim[0]:self.dim[1]+1]
            if handmade_features[0] is not None:
                panel_width, panel_height = self.GetSize() 
                cl_pos_x = round(handmade_features[0] * self.mag_base[0] + panel_width/2)
                cl_pos_y = round(panel_height/2 - handmade_features[1] * self.mag_base[1]) # クライアント領域の Y 座標は反転
                dist = ((self.mouse_client_pos[0] - cl_pos_x)**2 + (self.mouse_client_pos[1] - cl_pos_y)**2) ** 0.5
                # 以下が、距離が一定以内の場合 → 消去
                if dist < self.close_pos_threshold:
                    self.host.handmade_features_list[self.host.active_slot_index][self.dim[0]] = None
                    self.host.handmade_features_list[self.host.active_slot_index][self.dim[1]] = None
                    self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Right clicked {self.mouse_client_pos} -> {self.host.handmade_features_list}")
                    self.host.need_recalc_features[self.host.active_slot_index] = True # feature の再計算をトリガー
            self.Refresh()  # 再描画をトリガー
        event.Skip() 


    # ここは背景画像とグリッドをビットマップとして作成する機能で、各種イベントハンドラとは独立している。
    def create_background_bitmap(self):
        # 背景画像の読み込みと描画
        img = wx.Image(self.bg_image_path, wx.BITMAP_TYPE_ANY)
        # 画像をパネルに合わせてスケールする（アス比は元画像から変わる）。これがないと原寸のまま指定位置に配置される
        img = img.Scale(self.client_size[0], self.client_size[1], wx.IMAGE_QUALITY_HIGH)
        bmp = wx.Bitmap(img)

        # デバイスコンテキストの作成
        dc = wx.MemoryDC()
        dc.SelectObject(bmp)
        
        # パネルのサイズをピクセル単位で取得
        panel_width, panel_height = self.GetSize() 
        # こちらは画像のサイズ。ただしスケール済みなので基本的には panel に一致するはず
        img_width = bmp.GetWidth()
        img_height = bmp.GetHeight()
        # 画像の中心がパネルの中心に来るよう描画位置を設定。なおパネルの左上が原点 (0, 0) である。
        x = (panel_width - img_width) // 2
        y = (panel_height - img_height) // 2

        dc.DrawBitmap(bmp, x, y, True)

        # ここから、グリッド線の描画
        if self.draw_grid:
            dc.SetPen(wx.Pen(self.light_gray_color))  # ペン
            dc.SetTextForeground(self.light_gray_color)  # 文字色

            dc.DrawLine(0, panel_height // 2, panel_width, panel_height // 2) # パネル中央に x 軸の線を描画
            dc.DrawLine(panel_width // 2, 0, panel_width // 2, panel_height) # パネル中央に y 軸の線を描画

            # self.tick_interval ピクセルおきに tickmark と数値ラベルを描画
            n_ticks_x = round(panel_width // self.tick_interval_px[0]) # tickmark 本数（pixel / pixel）
            n_ticks_y = round(panel_height // self.tick_interval_px[1])

            # x 軸の tickmark と数値ラベルを描画する
            for i in range(-n_ticks_x // 2, n_ticks_x // 2 + 1):
                x = round(panel_width // 2 + i * self.tick_interval_px[0]) # 整数でないと線を引けないらしい
                dc.DrawLine(x, panel_height // 2 - self.tick_h, x, panel_height // 2 + self.tick_h)
                dc.DrawText(
                    str(round(i * self.tick_interval)), 
                    x - 7, # 残念ながらラベルは容易には中央寄せできない
                    panel_height // 2 + 5,
                )

            # y 軸の tickmark と数値ラベルを描画する
            for j in range(-n_ticks_y // 2, n_ticks_y // 2 + 1):
                y = round(panel_height // 2 - j * self.tick_interval_px[1])
                dc.DrawLine(panel_width // 2 - self.tick_h, y, panel_width // 2 + self.tick_h, y)
                dc.DrawText(
                    str(round(j * self.tick_interval)), 
                    panel_width // 2 + 10, 
                    y - 7,
                )

        return bmp
