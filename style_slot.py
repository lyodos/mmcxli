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
import os
import json
import copy
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
import numpy as np

from datetime import datetime

import logging
import inspect

from utils import plot_embedding_cube, truncate_string, sanitize_filename

import numpy as np
import csv


class StyleSlotPanel(wx.Panel):
    def __init__(
        self, 
        parent, 
        id = -1,
        manager = None,
        slot_index: int = None, # slot リストの中でこのパネルが何番目の要素であるか
        dim_style: int = 128,
        dim_comp: int = 2,
        fill_color = None, # 塗りのテーマカラーを指定する
        theme_color = None, # 線のテーマカラーを指定する
        border_color = None, # 境界線用の、さらに濃い色のテーマカラーを指定する
        restore_slot: bool = True,
        **kwargs,
    ):
        super().__init__(parent, id = id, **kwargs)

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        self.manager = manager # StyleManagerPanel インスタンスを指定
        self.slot_index = slot_index # このスロットが全体の何番目か。アプリケーションの起動中は変化しない前提
        self.dim_style = dim_style
        self.dim_comp = dim_comp
        self.fill_color = wx.Colour(fill_color)
        self.theme_color = wx.Colour(theme_color)
        self.border_color = wx.Colour(border_color)

        self.os_sep = os.path.sep # 現在の OS で有効なパス区切り文字

        # スロットが現在保持しているスタイル（ファイル由来）。最初はスタイルが未選択
        self.style_abs_path = None # ここが None か有効な文字列かで、ファイルロードの有無のフラグとする
        self.file_style_name = "" # 最初はスタイル名が空欄
        
        # ファイルから直接ロードした埋め込みを保持する。最初はゼロ初期化
        self.style_from_file = copy.deepcopy(self.manager.style_silent)

        # 直接ロードした埋め込みを self.dim_style → self.dim_comp 次元に圧縮エンコードしたのが self.style_comp 
        # こちらは None で初期化（値が入っていると editor pane で描画が発生するため）
        self.style_comp = None

        # self.style_comp をデコーダで self.dim_style 次元に戻したものが self.style_recon
        self.style_recon = copy.deepcopy(self.manager.style_silent)

        self.has_handmade_features = False # 手入力した特徴量を持つか。打ち込み値自体は AxesEditPanel で持つ
        self.handmade_style_name = "" # 手動で入力したスタイル名を、csv ロード時に退避させるための変数

        # 手入力した特徴量
        self.emb_handmade = None
        # 手入力した特徴量をネットワークで伸長したもの
        self.emb_expand = copy.deepcopy(self.manager.style_silent)


        #### パネル部品の定義
        
        # スタイル選択ボタン
        self.load_btn = wx.Button(self, label = "Load...")
        self.load_btn.SetMinSize((60, -1)) # 最小幅を設定
        self.load_btn.SetMaxSize((60, -1)) # 最大幅を設定
        self.load_btn.Bind(wx.EVT_BUTTON, self.load_style) # self.style_abs_path が未設定（None）だと . がデフォルト位置
        
        # スタイル選択解除ボタン
        self.unload_btn = wx.Button(self, label = "Unload")
        self.unload_btn.SetMinSize((60, -1)) # 最小幅を設定
        self.unload_btn.SetMaxSize((60, -1)) # 最大幅を設定
        self.unload_btn.Bind(wx.EVT_BUTTON, self.unload_file) # self.style_abs_path が未設定（None）だと . がデフォルト位置

        self.file_buttons_sizer = wx.BoxSizer(wx.VERTICAL)
        self.file_buttons_sizer.Add(self.load_btn, 0, wx.BOTTOM | wx.ALIGN_CENTER_HORIZONTAL, 0)
        self.file_buttons_sizer.Add(self.unload_btn, 0, wx.TOP | wx.ALIGN_CENTER_HORIZONTAL, 5)
        # いったん使わないので隠しておく
        self.unload_btn.Hide()

        # スタイル保存ボタン
        self.save_btn = wx.Button(self, label = "Save...")
        self.save_btn.SetMinSize((60, -1)) # 最小幅を設定
        self.save_btn.SetMaxSize((60, -1)) # 最大幅を設定
        self.save_btn.Bind(wx.EVT_BUTTON, self.save_file)
        self.save_btn.Enable() # 最初は何もロードされていないが、手動入力したスタイルを保存するために押せる

        # 選択中のスタイルを表示するテキスト
        self.style_label = wx.StaticText(self) 
        self.style_label.SetLabel(f"Slot {self.slot_index}\nEnter style name")
        self.style_label.SetMinSize((155, -1)) # Text の最小幅を設定（自動調整を適用させないため）
        self.style_label.SetMaxSize((155, -1)) # Text の最大幅を設定
        self.style_label.SetToolTip(f'Load a pre-calculated style embedding ({self.dim_style} dim) from a csv file')

        # ダブルクリックすると編集（ただし読み込んだスタイルがあるときは編集無効）
        self.style_label.Bind(wx.EVT_LEFT_DCLICK, self.on_label_double_click) 

        # スタイル名の部分をクリック／右クリックしてアクティブ／非アクティブにするイベントハンドラ
        self.style_label.Bind(wx.EVT_LEFT_DOWN, lambda event: self.manager.on_panel_click(event, self.slot_index))
        self.style_label.Bind(wx.EVT_RIGHT_DOWN, lambda event: self.manager.on_panel_right_click(event, self.slot_index))
        self.style_label.Bind(wx.EVT_ENTER_WINDOW, lambda event: self.manager.on_panel_hover(event, self.slot_index))
        self.style_label.Bind(wx.EVT_LEAVE_WINDOW, lambda event: self.manager.on_panel_unhover(event, self.slot_index))

        self.label_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.label_sizer.Add(self.style_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 0)

        # 埋め込みのプロットを格納する sizer で、最初に空で初期化して後から要素を加える
        self.plot_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # 選択ボタン、選択中ファイル表示の集合をまとめた sizer を作る
        self.widgets_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.widgets_sizer.Add(self.file_buttons_sizer, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.widgets_sizer.Add(self.save_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.widgets_sizer.Add(self.label_sizer, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 5)
        self.widgets_sizer.Add(self.plot_sizer, 0, wx.ALL | wx.ALIGN_TOP, 5) 

        # スロットの外郭が欲しいので、左側にもう 1 つパネルを作って包む
        self.left_border_panel = wx.Panel(self, wx.ID_ANY, size = (10, -1))

        # 統括用の sizer を作り、すべての部品を配置していく。
        self.root_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.root_sizer.Add(self.left_border_panel, 0, wx.EXPAND, 0) 
        self.root_sizer.Add(self.widgets_sizer, 0, wx.EXPAND, 5) 
        self.SetSizer(self.root_sizer)

        self.root_sizer.SetMinSize((435, -1)) # 最小幅を設定

        # 各パネル背景色は、スロットがアクティブか否か（StyleManagerPanel の active_slot_index で判定）で変化させる。
        # リアルタイムの更新処理も update として実装する
        if self.manager.active_slot_index == self.slot_index:
            self.SetBackgroundColour(self.theme_color)
            self.left_border_panel.SetBackgroundColour(self.border_color)
        else:
            self.SetBackgroundColour(self.fill_color)
            self.left_border_panel.SetBackgroundColour(self.fill_color)

        
        # なお self.manager.style_portfolio[self.slot_index] にスタイル辞書があり、["last_selected_file"] 等のキーを持つ
        # 初期状態だと、slot_index 以外は None である
        if restore_slot:
            # 前回の選択ファイルを復旧する場合
            if self.manager.style_portfolio[self.slot_index]["last_selected_file"] is not None:
                abs_path_raw = self.manager.style_portfolio[self.slot_index]["last_selected_file"]
                self.style_abs_path = self.load_csv_to_slot(abs_path_raw) 
                # 以下、ロードに成功した場合のみ、埋め込みの計算等が必要
                if self.style_abs_path is not None:
                    self.update_styles_from_file()

            # TODO 
            # また名前が復元できていない。
            if self.manager.style_portfolio[self.slot_index]["emb_handmade"] is not None:
                self.emb_handmade = np.array(self.manager.style_portfolio[self.slot_index]["emb_handmade"], dtype = np.float32)
                self.emb_expand = np.array(self.manager.style_portfolio[self.slot_index]["emb_expand"], dtype = np.float32)
                self.handmade_style_name = self.manager.style_portfolio[self.slot_index]["handmade_style_name"]
                # さらに、AxesEditPanel にも反映させる必要がある。
                # が、AxesEditPanel はこの時点でまだ初期化されていないので、AxesEditPanel 側で処理する
                self.has_handmade_features = True

            # ラベルのアップデートは？どうやら、内部変数は入っているがプロット、ラベル更新がトリガーされていない
            self.update_style_label()
            self.plot_embedding()
        
        # self.update は作成済みのプロットを再描画する。
        self.timer = wx.Timer(self) # wx.Timer クラスで、指定間隔での処理を実行する。
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.timer.Start(37) # タイマーは手動で起動する必要がある。単位 ms

        self.logger.debug(f"Slot {self.slot_index} was initialized.")

    ####

    # update は背景色の整合処理だけを行う
    def update(self, event):
        if self.manager.active_slot_index == self.slot_index:
            self.SetBackgroundColour(self.theme_color)
            self.left_border_panel.SetBackgroundColour(self.border_color)
        else:
            self.SetBackgroundColour(self.fill_color)
            self.left_border_panel.SetBackgroundColour(self.fill_color)
        self.Layout()


    def load_style(self, event):
        self._load_style()
    
    
    # 作成済みのスタイルが記載された（csv 形式）ファイルを選択してロードする
    def _load_style(self):
        previous_path = copy.deepcopy(self.style_abs_path)
        
        # ダイアログ初期値を前回ファイルにする
        last_relpath = self.manager.style_portfolio[self.slot_index]["last_selected_file"]
        if last_relpath is not None:
            last_relpath = last_relpath.replace('/', self.os_sep).replace('\\', self.os_sep) 
            abs_path = os.path.abspath(last_relpath)
            initial_dir = os.path.dirname(abs_path)
            initial_filename = os.path.basename(abs_path)
        else:
            initial_dir = "."
            initial_filename = ""
        
        with wx.FileDialog(
            self, 
            "Open embedding csv file", 
            defaultDir = initial_dir,
            defaultFile = initial_filename,
            wildcard = "CSV files (*.csv;*.txt)|*.csv;*.txt",
            style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return # キャンセルの場合はロード処理に入らず、現在の選択ファイルが保持される
            
            abs_path_raw = fileDialog.GetPath()
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Selecting... {abs_path_raw}")
            self.style_abs_path = self.load_csv_to_slot(abs_path_raw) 
            # 以下、ロードに成功した場合のみ、埋め込みの計算等が必要
            if self.style_abs_path is not None:
                self.update_styles_from_file()

        self.manager.need_refresh = True


    # 実際に csv からデータを読み込み、 self.file_style_name, self.style_from_file, self.is_file_loaded を更新する
    # 形が正規化されていなくても、コンマないし改行で区切った数値が、ファイル内に計 self.dim_style 要素あれば読み込み可能
    def load_csv_to_slot(self, file_path):
        if file_path is not None:
            file_path = file_path.replace('/', self.os_sep).replace('\\', self.os_sep) 
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Selected CSV: {os.path.basename(file_path)}")
            try:
                data_list = [] # csv が何行あるかは全く保証されていない前提なので、いったん行ごとに array 化する
                # ファイルを開き、csv の行を反復処理するイテレータを返す。各行のデータは for 文でリストとして取得
                with open(file_path, 'r') as f:
                    reader = csv.reader(f) 
                    for i, row in enumerate(reader):
                        # 各行のデータを float32 に変換してリストに追加
                        data = np.array(row, dtype = np.float32)
                        data_list.append(data)
                
                # 合計したセル数が self.dim_style であるかをチェック
                if len(data_list[-1]) == self.dim_style:
                    # CSV のデータが 1 行に記されている (1, 128)
                    valid_data = data_list[-1]
                    is_valid_length = True
                else:
                    # 複数行の CSV である場合（ (128, 1) の場合も含む）は結合が必要
                    data_concat = np.concatenate(data_list)
                    if data_concat.shape[-1] == self.dim_style:
                        valid_data = data_concat
                        is_valid_length = True
                    else:
                        is_valid_length = False
                
                # 要素数が self.dim_style になっていれば有効なデータとみなして進む
                if is_valid_length:
                    self.file_style_name = os.path.splitext(os.path.basename(file_path))[0] # 拡張子を抜いたファイル名
                    self.style_from_file = valid_data.reshape(1, self.dim_style)
                    self.logger.debug(f"({inspect.currentframe().f_code.co_name}) '{self.file_style_name}' is a valid style ({self.style_from_file.shape})")
                    return file_path
                else:
                    # CSV 要素数の違いで失敗した場合は警告ダイアログを表示し、現状変更はされない
                    wx.MessageBox(f"The style csv must contains {self.dim_style} numerical values!")
                    return None
            except:
                # CSV のフォーマットが不正でロードすらできない場合は警告ダイアログを表示し、現状変更はされない
                wx.MessageBox(f"Element size of the style csv is not {self.dim_style} or ill-formed!")
                return None
        else:
            # 有効なファイルパスを与えない場合は何もしない（通常はファイルが選択される前提であり、このルートには入らない）
            return None


    # ファイルロードに成功した場合だけ実行される、ファイル由来の各種スタイル変数とラベル、ボタン類の更新
    def update_styles_from_file(self):
            # ファイルからの圧縮埋め込みおよび、復元の self.dim_style 次元埋め込みも計算
            self.style_comp = self.manager.sess_SCE.run(
                ['comp'], 
                {'emb': self.style_from_file},
            )[0] # (1, self.dim_style) -> (1, self.dim_comp)
            self.style_recon = self.manager.sess_SD.run(
                ['emb'], 
                {'comp': self.style_comp},
            )[0] # (1, self.dim_comp) -> (1, self.dim_style)
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Compressed style: {np.round(self.style_comp, 3)}")
            self.plot_embedding() # embedding の圧縮表現と復元も作ったので、plot に反映する

            self.update_style_label() # スロットが選択中のファイルを示すラベル文字列を更新する
            self.unload_btn.Show() # ロードして初めて、アンロードするボタンが必要になる
            self.save_btn.Disable() # 保存ボタンは押させないようにする

            # ホストである StyleManagerPanel のアクティブスロットを更新する。ただしこの挙動が最善かどうかは不明
            self.manager.active_slot_index = self.slot_index
            self.SetBackgroundColour(self.theme_color) # アクティブなスロットとして背景色を変更する
            self.left_border_panel.SetBackgroundColour(self.border_color) # アクティブなスロットとして背景色を変更する
            
            # 設定をファイルとして保存
            self.overwrite_styles_gallery() 


    # ファイルをロードされているスロットを未選択に戻す
    # なお依然としてアクティブスロットではあり続ける
    
    def unload_file(self, event):
        if self.style_abs_path is not None:

            self.style_abs_path = None 
            self.file_style_name = "" # 最初のスタイル名に戻す
            self.update_style_label()

            self.unload_btn.Hide()
            self.save_btn.Enable() # 保存ボタンを押せるように戻す

            # 埋め込みの値も再初期化する
            self.style_from_file = copy.deepcopy(self.manager.style_silent)
            self.style_comp = None
            self.style_recon = copy.deepcopy(self.manager.style_silent)
    
            self.overwrite_styles_gallery() # 設定をファイルとして保存
            
            self.Refresh()
            # 実はここでもう 1 回、今度は style_editor 側の overwrite がトリガーされる。更新されるキーは微妙に異なる
            self.manager.need_refresh = True 
            self.manager.axes_panel.need_recalc_features[self.slot_index] = True # slot.style_comp 再計算をトリガー
            
            # 見ての通り、手動入力した座標はいったん消滅する。ただし Editor から次の瞬間に書き戻される。
            # また plot_embedding は AxesEditPanel の update 側にトリガーされる


    # ロードしたスタイルのラベルをセットする。ロード時とアンロード時に走る
    def update_style_label(
        self,
    ):
        if self.style_abs_path is not None:
            # ロード完了フラグが立っている場合
            text = f"\nLoaded style from file:\n{truncate_string(self.file_style_name, max = 21)}\nHandmade features:\n({self.emb_handmade})\n"
            tooltip_text = 'Click: activate, Right-click: deactivate'
        else:
            # ロードされていない（あるいは失敗した）場合 → 手動入力側の情報を表示
            current_handmade = np.round(self.emb_handmade, 3) if self.emb_handmade is not None else ""
            if self.handmade_style_name == "":
                text = f"Slot {self.slot_index} {current_handmade}\nEnter style name"
            else:
                text = f"Slot {self.slot_index} {current_handmade}\nStyle: '{self.handmade_style_name}'"
            tooltip_text = f'Load a pre-calculated style embedding ({self.dim_style}) from a csv file'

        self.style_label.SetLabel(text)
        self.style_label.SetToolTip(tooltip_text)
        self.root_sizer.Layout()


    # 選択中のスタイル埋め込み、およびその圧縮形をプロットする。
    # 初期バージョンでは plot_sizer を消して再作成していたが、これは非効率なので外側で維持し続けることにした。
    def plot_embedding(self):
        # Detach() と Destroy() で現在の sizer とプロットの Canvas オブジェクトを Window から削除
        if hasattr(self, 'canvas_file'):
            self.plot_sizer.Detach(self.canvas_file)
            self.canvas_file.Destroy()
            del self.canvas_file # Destroy は Window からの削除なので、変数自体は手動で消す必要がある
        if hasattr(self, 'canvas_comp'):
            self.plot_sizer.Detach(self.canvas_comp)
            self.canvas_comp.Destroy()
            del self.canvas_comp

        # 描画対象が存在しない場合は評価を終了
        if self.style_abs_path is None and self.has_handmade_features is False:
            return
        
        # 以下、いずれかの feature が作成ないしロードされている場合のみ評価
        
        self.fig_file, self.ax_file = plot_embedding_cube(
            self.style_from_file,
            figsize = (55, 100),
            aspect = 1,
            cmap = "bwr",
        )

        # 圧縮表現に表示するデータは、直接入力値がファイル由来よりも優先される
        if self.has_handmade_features:
            cube_for_plot = self.emb_handmade
        else:
            cube_for_plot = np.zeros((1, self.dim_comp), dtype = np.float32) # ただし、作図はするが隠される
        v_range_max = max([abs(i) for i in self.manager.comp_v_range])
        
        self.fig_comp, self.ax_comp = plot_embedding_cube(
            cube_for_plot,
            figsize = (55, 100),
            image_reshape = (1, 2), # n = 2 のときの圧縮表現
            aspect = 1,
            v_range = [-v_range_max, v_range_max],
            cmap = "bwr", # "jet"
        )
        self.fig_comp.patch.set_facecolor((0.2, 0.2, 0.24, 1))

        # FigureCanvasWxAgg は matplotlib と wx の連携用のパーツ
        self.canvas_file = FigureCanvasWxAgg(self, -1, self.fig_file)
        self.canvas_comp = FigureCanvasWxAgg(self, -1, self.fig_comp)
        self.plot_sizer.Add(self.canvas_file, 0, wx.TOP | wx.BOTTOM | wx.ALIGN_CENTER_VERTICAL, 0)
        self.plot_sizer.Add(self.canvas_comp, 0, wx.TOP | wx.BOTTOM | wx.ALIGN_CENTER_VERTICAL, 0)

        # file 由来と手動入力の特徴量が併存する場合、VC は file 由来が優先されるが、表示は両方が対象
        if self.style_abs_path is None:
            self.canvas_file.Hide() 
        if self.has_handmade_features is False:
            self.canvas_comp.Hide() 

        if hasattr(self.manager, "root_sizer"):
            self.manager.root_sizer.Layout()
        self.Refresh()  # 再描画をトリガー


    # スロットで計算して保持している 128 次元のスタイルベクトルを csv に「名前をつけて保存」する
    def save_file(self, event):
        # self.file_style_name が "" でない→手動で名前を入れている
        if len(self.file_style_name) > 0:
            default_name = self.file_style_name
        else:
            now = datetime.now()
            default_name = f"STY{self.dim_style}_{now.strftime('%Y-%m-%d_%H.%M.%S')}"
        
        default_extension = 'csv'
        wildcard = "Csv files (*.csv)|*.csv|Text files (*.txt)|*.txt|All files (*.*)|*.*"
        dlg = wx.FileDialog(
            self, 
            message = "Save style as ...", 
            defaultDir = "./styles" or '.', 
            defaultFile = f"{default_name}.{default_extension}", 
            wildcard = wildcard,
            style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            with open(path, 'w', newline = '') as f:
                writer = csv.writer(f)
                writer.writerow(np.round(self.style_recon.astype(float), 4).flatten().tolist()) 
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Expanded style embedding was saved at: {path}")
    
    
    # こちらは現在のスタイル一覧の全体を json ファイルとして保存
    def overwrite_styles_gallery(self):
        try:
            if self.style_abs_path is not None:
                # 絶対パスから相対パスに変換。絶対パスを json に残すと個人情報保護や複数マシン間での共有にリスク
                rel_path = os.path.relpath(self.style_abs_path, os.getcwd()) 
            else:
                rel_path = None

            # ファイルを開くたびに config ファイルを更新する
            self.manager.style_portfolio[self.slot_index]["last_selected_file"] = copy.deepcopy(rel_path)
            self.manager.style_portfolio[self.slot_index]["file_style_name"] = copy.deepcopy(self.file_style_name)
            self.manager.style_portfolio[self.slot_index]["handmade_style_name"] = copy.deepcopy(self.handmade_style_name)

            if self.style_from_file is not None:
                self.manager.style_portfolio[self.slot_index]["emb_file"] = self.style_from_file.tolist()
                if self.style_comp is not None:
                    self.manager.style_portfolio[self.slot_index]["emb_comp"] = self.style_comp.tolist()
                if self.style_recon is not None:
                    self.manager.style_portfolio[self.slot_index]["emb_recon"] = self.style_recon.tolist()
            else:
                # ファイルからロードされていないときは問答無用で None 
                self.manager.style_portfolio[self.slot_index]["emb_file"] = None
                self.manager.style_portfolio[self.slot_index]["emb_comp"] = None
                self.manager.style_portfolio[self.slot_index]["emb_recon"] = None
            
            # 手動入力した特徴量を反映させる機能は、ここではなく AxesEditPanel のメソッドとして用意されている
            
            with open(self.manager.style_portfolio_path, 'w') as f:
                json.dump(self.manager.style_portfolio, f, indent = 4)
            
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Style gallery was saved to '{self.manager.style_portfolio_path}'")
        except:
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Failed to save the style gallery.")


    # StaticText を TextCtrl に置き換えて編集可能にする
    def on_label_double_click(self, event):
        text_rect = self.style_label.GetRect()
        text_value = "New style name" if self.handmade_style_name == "" else self.handmade_style_name
        self.label_text_ctrl = wx.TextCtrl(
            self.style_label.GetParent(), # self, 
            wx.ID_ANY, 
            value = text_value, 
            pos = (text_rect.x, text_rect.y),
            size = (text_rect.width, -1),#(text_rect.width, text_rect.height),
            style = wx.TE_PROCESS_ENTER,
        )
        if self.style_abs_path is not None:
            self.label_text_ctrl.Hide() # スタイルがロード済みの場合も編集パネルを作るが、触れないようにする
        else:
            # StaticTextを非表示にし、TextCtrlをフォーカスさせる
            self.style_label.Hide()
            self.label_sizer.Add(self.label_text_ctrl, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 0)

            self.label_text_ctrl.SetFocus()
            self.label_text_ctrl.SetSelection(0, len(text_value)) # テキスト全体を選択状態にする

            # Enterキーで確定するためのイベントハンドラを設定
            self.label_text_ctrl.Bind(wx.EVT_TEXT_ENTER, self.on_label_text_enter)
            # フォーカスを失ったときのイベントハンドラを設定。ただし、パネルの sizer 全体がフォーカス範囲になってしまう
            self.label_text_ctrl.Bind(wx.EVT_KILL_FOCUS, self.on_label_text_enter)


    def on_label_text_enter(self, event):
        # Enter キーが押されたら、TextCtrl の内容を取得して StaticTextに 反映し、TextCtrl を（削除→）隠蔽する
        self.handmade_style_name = sanitize_filename(self.label_text_ctrl.GetValue())
        
        # ラベル更新。ファイルがロードされているとそもそも編集に入れないので、そのケースは考えなくていい
        current_handmade = np.round(self.emb_handmade, 3) if self.emb_handmade is not None else ""
        if self.handmade_style_name == "":
            text = f"Slot {self.slot_index} {current_handmade}\nEnter style name"
        else:
            text = f"Slot {self.slot_index} {current_handmade}\nStyle: '{self.handmade_style_name}'"
        self.style_label.SetLabel(text) 

        self.label_sizer.Detach(self.label_text_ctrl)
        self.label_text_ctrl.Unbind(wx.EVT_TEXT_ENTER)
        self.label_text_ctrl.Unbind(wx.EVT_KILL_FOCUS)
        self.label_text_ctrl.Hide() # 最初は Destroy() していたが Linux でエラーになるので、隠すだけにした
        self.style_label.Show()
        
        self.overwrite_styles_gallery() # 設定をファイルとして保存


