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
import wx.lib.scrolledpanel as scrolled

import json
import os
import colorsys
import copy

import logging
import inspect

import numpy as np
rng = np.random.default_rng(2141)

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg

import onnxruntime as ort # 予め ort-gpu を入れること。 Opset 17 以上が必要
#pip install ort-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-12/pypi/simple/
so = ort.SessionOptions()
so.log_severity_level = 0

from style_slot import StyleSlotPanel
from style_editor import AxesEditPanel
from utils import plot_embedding_cube

# 設定ファイルのパス
STYLE_PORTFOLIO_PATH = "./styles/style_portfolio.json"


# スタイル埋め込みの圧縮次元から本来次元へのデコードを統括するクラス

class StyleManagerPanel(scrolled.ScrolledPanel):
    def __init__(
        self, 
        parent, 
        backend = None, # SoundControl クラスのインスタンスをここに指定。self.current_target_style にアクセスする。
        id = -1,
        host = None,
        tab_id: int = None,
        dim_style: int = 128,
        dim_comp: int = 2,
        comp_v_range: tuple | list = (50, 50), # 本当は config から触るべきなのだが、audio backend を通すのもなんか変
        model_device = "cpu",
        style_compressor_ckpt: str = "./weights/pumap_encoder_2dim.onnx",
        style_decoder_ckpt: str = "./weights/pumap_decoder_2dim.onnx",
        max_slots: int = 8, # 最大いくつのスタイル埋め込みスロットを保持するか。
        restore_slot: bool = True, # 前回終了時に読み込んでいたスタイルファイルを自動で再ロードするよう試みる
        portfolio_path: str = None, # config を保存するときのファイル名
        **kwargs,
    ):
        super().__init__(parent, id = id, **kwargs)

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        self.logger.debug("Initializing...")
        self.host = host
        self.backend = backend
        self.tab_id = tab_id
        
        self.dim_style = dim_style
        self.dim_comp = dim_comp
        self.comp_v_range = comp_v_range # 圧縮埋め込みの値の範囲
        
        self.max_slots = max_slots
        self.active_slot_color = wx.Colour(252, 235, 245)
        self.hover_slot_color = wx.Colour(252, 247, 250)
        self.nonactive_slot_color = wx.Colour(255, 255, 255)

        self.style_portfolio_path = portfolio_path if portfolio_path is not None else STYLE_PORTFOLIO_PATH
        # 圧縮スタイルを管理する設定ファイルのロード
        self.style_portfolio = self.load_make_styles_gallery(self.style_portfolio_path)

        self.need_refresh = False # 強制再描画のトリガー

        # 初期スタイルとして「ほぼ無音のサンプルを style encoder に放り込んで得た style vector」をロードしておく。
        # なお参照先の存在は保証されるが、実際に無音スタイルであるためには、本クラス以前に SampleManagerPanel の初期化が必要
        self.style_silent = np.round(self.backend.candidate_style_list[1], 4) # 1 は sampler 用の初期スタイル

        # 最終的に VC に使う style embedding はロード時に計算するが、とりあえず初期スタイルとして self.style_silent
        self.style_result = copy.deepcopy(self.style_silent)
        self.backend.candidate_style_list[0] = copy.deepcopy(self.style_silent) # audio backend 側にも反映


        #### ネットワーク初期化。VC 動作に割り込まないよう、CPU 上で独自に作っておく。
        
        # なお実際に session run させる場面は slot 側に実装されていることに注意

        # 埋め込み計算用のデバイス設定
        if model_device == "cpu":
            self.onnx_provider_list = ['CPUExecutionProvider']
        else:
            self.onnx_provider_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.style_compressor_ckpt = style_compressor_ckpt
        self.style_decoder_ckpt = style_decoder_ckpt
        
        self.logger.debug("Initializing StyleCompressor...")
        self.sess_SCE = ort.InferenceSession(
            self.style_compressor_ckpt, 
            providers  = self.onnx_provider_list,
        ) # 入力は (batch, dim_style = 128)

        self.logger.debug("Initializing StyleDecoder...")
        self.sess_SD = ort.InferenceSession(
            self.style_decoder_ckpt, 
            providers  = self.onnx_provider_list,
        ) # 入力は (batch, dim_comp = 2)

        # ダミーデータをネットワークに通して試運転する。この結果は「無音時に対応する埋め込み」となる
        comp_silent = self.sess_SCE.run(
            ['comp'], 
            {'emb': self.style_silent},
        )[0] # (1, self.dim_style) -> (1, self.dim_comp)
        # 伸長結果は捨てる。実は無音のスタイル埋め込みを圧縮して伸長しても、無音の埋め込みにはならない
        _ = self.sess_SD.run(
            ['emb'], 
            {'comp': comp_silent},
        )[0] # (1, self.dim_comp) -> (1, self.dim_style)

        #### 音声ファイル用のスロットパネルおよび、最終採用スタイルを画像表示するパネルの初期化

        # 最初に十分な数のカラーパレットを作っておく
        self.slot_fill_palette = self.make_slot_colors(
            n = self.max_slots,
            saturation = 0.03,
            brightness = 1.0,
        )
        # アクティブスロット用に、少し濃い色のパレットも作る
        self.slot_color_palette = self.make_slot_colors(
            n = self.max_slots,
            saturation = 0.2,
            brightness = 1.0,
        )
        #  境界線用に、さらに濃い色のパレットも作る
        self.slot_border_palette = self.make_slot_colors(
            n = self.max_slots,
            saturation = 0.5,
            brightness = 1.0,
        )
        
        self.active_slot_index = 0 # 最初からスロット 0 を選択状態にする。常にどこか 1 つが選択される
        # 初期版では複数のスロットをアクティブにできたが、ロジックが複雑すぎて管理できなくなったので廃止した
        
        #### 子パネルの初期化
        
        # 個別のスタイル設定ファイルとその埋め込みを管理するスロットを作成。ネットワークの定義後に行うこと。
        self.slot_list = []
        for i in range(self.max_slots):
            # リストの要素として作るのが楽。
            self.slot_list.append(
                StyleSlotPanel(
                    self, 
                    manager = self, 
                    slot_index = i, 
                    dim_style = self.dim_style,
                    dim_comp = self.dim_comp,
                    fill_color = self.slot_fill_palette[i],
                    theme_color = self.slot_color_palette[i],
                    border_color = self.slot_border_palette[i],
                    restore_slot = restore_slot,
                    style = wx.BORDER_STATIC, 
                )
            )
        
        # さらに 2 次元のグリッドを編集するパネルを作る。StyleSlotPanel よりも後でないとエラーになる
        # 現在は圧縮埋め込みを 2 次元に固定しているが、小改修で n 次元にできる（ただし n % 2 == 0 が条件）
        self.axes_panel = AxesEditPanel(
            self, 
            manager = self, 
            style = wx.BORDER_NONE, 
            axes_pane_size = (640, 640),
            border_at = self.comp_v_range,#[50.0, 50.0], # 各軸の描画範囲
            tab_id = self.tab_id,
        )

        # さらに、スタイル毎の slot の横に、実際に VC に用いるスタイルベクトルの作成結果を表示するパネルを作る。
        # 最新版のスタイルの計算結果は常に self.style_result に格納されるものとする。
        self.result_panel = ResultEmbeddingPanel(self, manager = self, style = wx.BORDER_NONE)

        # 複数のスロットを垂直配置する sizer 
        self.slots_sizer = wx.BoxSizer(wx.VERTICAL)
        for i in range(self.max_slots):
            self.slots_sizer.Add(self.slot_list[i], 0, wx.ALL, 0) 

        # 右側に最終採用版の埋め込みを配置するための sizer
        self.result_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.result_sizer.Add(self.slots_sizer, 0, wx.EXPAND | wx.ALL, 5)
        self.result_sizer.Add(self.axes_panel, 0, wx.EXPAND | wx.ALL, 5)
        self.result_sizer.Add(self.result_panel, 1, wx.GROW, 5)

        # 統括用の sizer である self.root_sizer を作り、すべての部品を配置していく。
        self.root_sizer = wx.BoxSizer(wx.VERTICAL)
        self.root_sizer.Add(self.result_sizer, 1, wx.EXPAND | wx.ALL, 0)
        self.SetSizer(self.root_sizer) # （既存の子要素があれば削除して）統括用 sizer を self 配下に加える
        self.Layout()

        self.timer = wx.Timer(self) # wx.Timer クラスで、指定間隔での処理を実行する。
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.timer.Start(100) # タイマーは手動で起動する必要がある。単位 ms

        self.logger.debug("Initialized.")


    def update(self, event):
        # 最終的な VC 用の埋め込みを決める。
        # 実際に埋め込みを保持するのは slot 側であり、StyleManagerPanel は「現在のアクティブスロット番号」のみ記憶する
        if self.active_slot_index is not None:
            active_slot = self.slot_list[self.active_slot_index]
            # ファイルからロードしたスタイルがある場合はロード直後のオリジナルを優先
            if active_slot.style_abs_path is not None:
                self.style_result = active_slot.style_from_file
            else:
                self.style_result = active_slot.emb_expand
        else:
            self.style_result = self.style_silent # アクティブスロットが指定されていないとき

        self.backend.candidate_style_list[0] = copy.deepcopy(self.style_result)

        if self.need_refresh:
            # 常に全スロットの再描画を行う
            for slot in self.slot_list:
                slot.Layout()
                slot.Refresh() 
            self.need_refresh = False

        event.Skip()


    # パネルを左クリックしてアクティブにする処理。実際には StyleSlotPanel 側から呼び出す。
    def on_panel_click(
        self, 
        event, 
        slot_index: int,  # 何番目のスロットをクリックしたか
    ):
        # 非アクティブなスロット上のパネルをクリック → 現在のアクティブを外す
        if self.active_slot_index != slot_index and self.active_slot_index is not None:
            self.slot_list[self.active_slot_index].SetBackgroundColour(
                self.slot_list[self.active_slot_index].fill_color # 非選択時テーマ色
            )
            self.slot_list[self.active_slot_index].left_border_panel.SetBackgroundColour(
                self.slot_list[self.active_slot_index].fill_color # 非選択時ボーダー色
            )
        # しかる後、active slot を変更
        self.active_slot_index = slot_index
        self.slot_list[slot_index].SetBackgroundColour(
            self.slot_list[slot_index].theme_color # アクティブ色
        )
        self.slot_list[slot_index].left_border_panel.SetBackgroundColour(
            self.slot_list[slot_index].border_color # アクティブボーダー色
        )

        for slot in self.slot_list:
            slot.Refresh() # 常に全スロットの再描画を行う


    # 右クリックの場合は、あくまで当該の非アクティブ化に専念する
    def on_panel_right_click(
        self, 
        event, 
        slot_index: int,
    ):
        if self.active_slot_index == slot_index:
            self.slot_list[self.active_slot_index].SetBackgroundColour(
                self.slot_list[self.active_slot_index].fill_color # 非選択時テーマ色
            )
            self.slot_list[self.active_slot_index].left_border_panel.SetBackgroundColour(
                self.slot_list[self.active_slot_index].fill_color # 非選択時ボーダー色
            )
            self.active_slot_index = 0 # 初期状態は 0

        for slot in self.slot_list:
            slot.Refresh() # 常に全スロットの再描画を行う


    # カーソルが侵入したとき色を変える。ただし、Windows でしか動作しない（wxWidgets そのもののバグらしい）
    def on_panel_hover(self, event, slot_index: int):
        if self.active_slot_index != slot_index:
            self.slot_list[slot_index].SetBackgroundColour(
                self.slot_list[slot_index].theme_color # アクティブ色
            )
            self.slot_list[slot_index].left_border_panel.SetBackgroundColour(
                self.slot_list[slot_index].theme_color # アクティブボーダー色
            )
        self.slot_list[slot_index].Refresh()

    # カーソルが離れたとき色を戻す
    def on_panel_unhover(self, event, slot_index: int):
        if self.active_slot_index != slot_index:
            self.slot_list[slot_index].SetBackgroundColour(
                self.slot_list[slot_index].fill_color # 非選択時テーマ色
            )
            self.slot_list[slot_index].left_border_panel.SetBackgroundColour(
                self.slot_list[slot_index].fill_color # 非選択時ボーダー色
            )
        self.slot_list[slot_index].Refresh()



    # 指定した名称のスタイル一覧をロードする。ファイルがない場合、作成する
    def load_make_styles_gallery(
        self,
        x,
    ):
        if os.path.exists(x):
            with open(x, "r") as f:
                return json.load(f)
        else:
            conf_list = []
            for i in range(self.max_slots):
                dict = {
                    "slot_index": i,
                    "last_selected_file": None,
                    "file_style_name": None,
                    "emb_file": None,
                    "emb_comp": None,
                    "emb_recon": None,
                    "handmade_style_name": None,
                    "emb_handmade": None,
                    "emb_expand": None,
                }
                conf_list.append(dict)
            return conf_list


    # いい感じに色相が離散したカラーパレットを作る機能
    
    def make_slot_colors(
        self,
        n: int = 15,
        hue_step: int = 137, # 137 で 16, 139 で 7, 127 で 21 → 127 のほうが最小ステップはでかいが、色が固まって見える
        saturation: float = 0.2,
        brightness: float = 1.0,
        brightness_coef: float = 0.993,
        test_palette: bool = False,
    ):
        num_colors = n
        hue_values = [(hue_step * i) % 360 for i in range(num_colors)]
        
        # 均等な分割ができるかどうかのテスト
        if test_palette:
            hue_values.sort()
            min_diff = float('inf')  # 初期値として無限大を設定する
            # 隣り合った成分間の差分を計算し、最小値を求める
            for i in range(1, n):
                diff = hue_values[i] - hue_values[i - 1]
                if diff < min_diff:
                    min_diff = diff
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Minimal distance of the color palette is {min_diff}")
    
        # HSV から RGB への変換と wxPython のカラー設定
        slot_color_list = []
        for i, hue in enumerate(hue_values):
            hue_normalized = hue / 360.0
            rgb_color = colorsys.hsv_to_rgb(hue_normalized, saturation, brightness*brightness_coef**i)
            slot_color_list.append(
                [round(rgb_color[0] * 255), round(rgb_color[1] * 255), round(rgb_color[2] * 255)],
            )

        return slot_color_list


####

# 計算した style vector を、最終的に VC に使う embedding として集約表示するパネル
# 最新の埋め込みそのものはここには無く、manager こと StyleManagerPanel の style_result に常に最新版が格納されている前提

class ResultEmbeddingPanel(wx.Panel):
    def __init__(
        self, 
        parent, 
        id = -1,
        manager = None, # 呼び出し元である StyleManagerPanel
        figsize: tuple = (110, 200),
        **kwargs,
    ):
        super().__init__(parent, id = id, **kwargs)

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        self.logger.debug("Initializing...")
        self.manager = manager # StyleManagerPanel インスタンスを指定
        self.figsize = figsize

        self.result_style_label = wx.StaticText(self) # 選択中のファイルを表示するテキスト
        self.result_style_label.SetLabel("Calculated style")
        
        # embedding は self.manager.style_result に格納されるが、
        # 再描画を必要な場合のみ行うために、現在の style embedding をキャッシュしておく
        self.style_cache = copy.deepcopy(self.manager.style_result)

        self.canvas_cube = self.plot_result_embedding(self.manager.style_result, figsize = self.figsize) 

        self.style_sizer = wx.BoxSizer(wx.VERTICAL)
        self.style_sizer.Add(self.canvas_cube, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 0)
        self.style_sizer.Add(self.result_style_label, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)

        self.root_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.root_sizer.Add(self.style_sizer, 1, wx.ALL | wx.ALIGN_CENTER, 0)
        self.SetSizer(self.root_sizer)

        self.SetToolTip('Style embedding used in VC')

        self.timer = wx.Timer(self) # wx.Timer クラスで、指定間隔での処理を実行する。
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.timer.Start(99) # タイマーは手動で起動する必要がある。単位 ms

    # キャッシュと現在の選択スタイルを比較し、変化している場合のみ再描画を行う
    def update(self, event):
        if not np.allclose(self.style_cache, self.manager.style_result, atol = 1e-8):
            # 壊す前に、ウィジェット全体の sizer の中での順番を記録しておく
            index = self.style_sizer.GetChildren().index(self.style_sizer.GetItem(self.canvas_cube))
            # Detach() と Destroy() で現在のスライダーを削除して作り直す
            self.style_sizer.Detach(self.canvas_cube)
            self.canvas_cube.Destroy()

            self.canvas_cube = self.plot_result_embedding(self.manager.style_result, figsize = self.figsize)
            self.style_cache = copy.deepcopy(self.manager.style_result)
            self.canvas_cube.draw()

            # 統括 sizer に配置し直す
            self.style_sizer.Insert(
                index, self.canvas_cube, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 0,
            )
            self.Layout()


    # スタイル埋め込みをプロットする
    def plot_result_embedding(
        self,
        style,
        figsize: tuple,
        aspect: float = 1,
        cmap: str = "bwr",
    ):
        self.fig_cube, self.ax_cube = plot_embedding_cube(
            style,
            figsize = figsize,
            aspect = aspect,
            cmap = cmap,
        )
        self.fig_cube.patch.set_facecolor((1, 1, 1, 1)) # rgba Embedding の画像表示の背景色

        # FigureCanvasWxAgg は matplotlib と wx の連携用のパーツ
        return FigureCanvasWxAgg(self, -1, self.fig_cube)


####

# 以下は、デバッグ用にスクリプトで実行するときだけ利用される独自の audio backend であり、内容は全くない

class Backend(wx.Panel):
    def __init__(
        self, 
        parent, 
        id = -1,
        **kwargs,
    ):
        super().__init__(parent, id = id, **kwargs)
        self.candidate_style_list = [np.zeros((1, 128), dtype = np.float32)] * 3 # expanded, sampler, full-128
        self.comp_v_range = [50, 50]


# 以下はスクリプトで実行するときだけ実行される

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title = "Style Manager Test", size = (1280, 880))

        self.pos = self.GetScreenPosition()
        self.size = self.GetSize() # フレーム自身のサイズを変数に格納しておく

        self.active_tab = None # 実際には notebook がないので、情報としては意味がない
        
        self.backend = Backend(self)
        self.panel = StyleManagerPanel(
            self, 
            self.backend, 
            host = self,
            style_compressor_ckpt = "./weights/pumap_encoder_2dim.onnx",
            style_decoder_ckpt = "./weights/pumap_decoder_2dim.onnx",
        )
        self.panel.SetSize(self.size)
        self.panel.SetupScrolling()

        # ステータスバーを作成
        self.sb = self.CreateStatusBar(number = 4)
        self.sb.SetStatusText('Ready', i = 0) # ステータスバーに文字を表示させる
        self.sb_size = self.sb.GetSize() # ステータスバーの高さを変数に格納しておく

        self.Show()

        self.timer = wx.Timer(self) # wx.Timer クラスで、指定間隔での処理を実行する。
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.timer.Start(37) # タイマーは手動で起動する必要がある。単位 ms


    def update(self, event):
        self.size = self.GetSize()
        self.pos = self.GetScreenPosition()
        self.panel_pos = self.panel.GetScreenPosition()
        self.sb_size = self.sb.GetSize()

        self.panel.SetSize((self.sb_size[0], self.size[1] - self.sb_size[1] - (self.panel_pos[1] - self.pos[1])))
        self.panel.FitInside()
        self.Layout()

        self.sb.SetStatusText(
            f"Frame pos: {self.pos}, panel pos: {self.panel_pos}", 
            i = 0,
        )
        self.sb.SetStatusText(
            f"Frame: {self.GetSize()}, panel: {self.panel.GetSize()}, sb: {self.sb_size}", 
            i = 1,
        )
        if self.panel.active_slot_index is not None:
            self.sb.SetStatusText(
                f"Selected style: '{self.panel.slot_list[self.panel.active_slot_index].file_style_name}'",
                i = 2,
            )
        self.sb.SetStatusText(
            f"Active slot: {self.panel.active_slot_index}",
            i = 3,
        )


if __name__ == "__main__":
    log_name = f"./logs/handmade_style_latest.log" 
    logging.basicConfig(
        filename = log_name, 
        level = logging.DEBUG, 
        format = '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    )
    # 一部の「うるさい」ライブラリについて、個別のログレベルを設定
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)

    app = wx.App(False)
    frame = MainFrame()
    app.MainLoop()

