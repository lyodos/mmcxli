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
import os
import math
import copy
import csv


from datetime import datetime

from utils import truncate_string

import logging
import inspect

import numpy as np


#### （高度）話者スタイルの 128 次元を全部手動で制御する。ジョーク機能の一種


class FullManagerPanel(scrolled.ScrolledPanel):
    def __init__(
        self, 
        parent, 
        backend = None, # SoundControl クラスのインスタンスをここに指定。先にバックエンドが初期化されている必要がある。
        id = -1,
        dim_style: int = 128,
        nrow: int = 18, # スライダーを行列に配置するときの行数
        val_range: list = [-4, 4],
        **kwargs,
    ):
        super().__init__(parent, id = id, **kwargs)

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        self.logger.debug("Initializing...")

        self.os_sep = os.path.sep # 現在の OS で有効なパス区切り文字

        self.sc = backend
        self.dim_style = dim_style
        self.nrow = nrow
        self.val_range = val_range

        # スロットが現在保持しているスタイル（ファイル由来ないし直接入力）。最初はスタイルが未選択なので None で初期化
        self.is_file_loaded = False
        self.style_abs_path = None 
        self.style_name = "" # 最初はスタイル名が空欄

        # self.style_from_file はファイルから直接ロードした埋め込み、style_current は編集物を保持する。最初はゼロ初期化
        self.style_from_file = np.zeros((1, self.dim_style), dtype = np.float32)
        self.style_current = np.zeros((1, self.dim_style), dtype = np.float32)
        
        ####
        
        # スタイル選択ボタン
        self.load_btn = wx.Button(self, label = "Load...")
        self.load_btn.SetMinSize((70, -1)) # 最小幅を設定
#        self.load_btn.SetMaxSize((70, -1)) # 最大幅を設定
        self.load_btn.Bind(wx.EVT_BUTTON, self.select_file) # self.style_abs_path が未設定（None）だと . がデフォルト位置

        # スタイル保存ボタン。なお本来は「名前をつけて保存」にする必要があるため、現在のイベントハンドラは暫定
        self.save_btn = wx.Button(self, label = "Save as...")
        self.save_btn.SetMinSize((70, -1)) # 最小幅を設定
#        self.save_btn.SetMaxSize((70, -1)) # 最大幅を設定
        self.save_btn.Bind(wx.EVT_BUTTON, self.save_file)
        self.save_btn.Disable()
        save_tooltip_text = f'You can save the style after you change at least one slier value.'
        self.save_btn.SetToolTip(save_tooltip_text)
        
        # 選択中のスタイルを表示するテキスト
        self.style_label = wx.StaticText(self) 
        self.style_label.SetLabel(f"Enter style name")
        self.style_label.SetMinSize((150, -1)) # Text の最小幅を設定（自動調整を適用させないため）
#        self.style_label.SetMaxSize((150, -1)) # Text の最大幅を設定
        self.style_label.SetToolTip(f'Load a pre-calculated style embedding ({self.dim_style} dim) from a csv file')

        # ダブルクリックすると編集
#        self.style_label.Bind(wx.EVT_LEFT_DCLICK, self.on_label_double_click) 

        self.style_label_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.style_label_sizer.Add(self.style_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 0)
        
        # クリアボタン
        self.clear_btn = wx.Button(self, label = "Clear all")
        self.clear_btn.SetMinSize((70, -1)) # 最小幅を設定
#        self.clear_btn.SetMaxSize((70, -1)) # 最大幅を設定
        self.clear_btn.Bind(wx.EVT_BUTTON, self.clear) # self.style_abs_path が未設定（None）だと . がデフォルト位置
        self.clear_btn.Disable()

        # 現在のマイク入力からリアルタイムで計算して埋め込みを表示させるか
        self.monitor_checkbox = wx.CheckBox(self, wx.ID_ANY, "Realtime input monitor (when the target style is 'from myself')")
        self.monitor_checkbox.SetValue(False)
        self.monitor_checkbox.Disable()

        # 選択ボタン、選択中ファイル表示の集合をまとめた sizer を作る
        self.widgets_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.widgets_sizer.Add(self.load_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.widgets_sizer.Add(self.style_label_sizer, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.widgets_sizer.Add(self.save_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.widgets_sizer.Add(self.clear_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.widgets_sizer.Add(self.monitor_checkbox, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        # スライダーたち
        self.slider_list = []
        for i in range(self.dim_style):
            self.slider_list.append(
                DimensionSlider(
                    self, 
                    dim = i, 
                    host = self, 
                    value = float(self.style_current[0, i]),
                    val_range = self.val_range, # 99 percentile がほぼ 4
                    mult = 100, # 0.01 刻みで入力可能になる
                    sldr_size = (105, -1), # 145
                    show_label = True,
                    label_horizontal = True,
                )
            )

        #### Sizer

        # グリッドを作る。ポジションを指定して追加できる
        self.bag_sizer = wx.GridBagSizer(vgap = 0, hgap = 0) # 行間 0 px, 列間 0 px
        
        # chunk size 設定
        for i in range(self.dim_style):
            self.bag_sizer.Add(
                self.slider_list[i], 
                pos = (i%self.nrow, int(i/self.nrow)), 
                flag = wx.RIGHT, 
                border = 5,
            )

        self.bag_size = self.bag_sizer.GetSize()
        self.bag_sizer.SetMinSize(self.bag_size) # 最低サイズ。これがないと  assertion 'size >= 0' failed in GtkScrollbar
        
        # 統括 sizer を作り、すべての部品を配置していく。
        self.root_sizer = wx.BoxSizer(wx.VERTICAL)
        self.root_sizer.Add(self.widgets_sizer, proportion = 0, flag = wx.EXPAND, border = 5) 
        self.root_sizer.Add(self.bag_sizer, 0, wx.EXPAND | wx.ALL, border = 5)
        self.SetSizer(self.root_sizer)
        self.Layout()

        self.timer = wx.Timer(self) # wx.Timer クラスで、指定間隔での処理を実行する。
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.timer.Start(100) # タイマーは手動で起動する必要がある。単位 ms

        self.logger.debug("Initialized.")


    def update(self, event):
        # 試験的に、現在の SE が計算した最新のスタイルでスライダーをリアルタイム更新してみる
        if self.sc.efx_control.auto_encode == True:
            self.monitor_checkbox.Enable()
        else:
            self.monitor_checkbox.SetValue(False)
            self.monitor_checkbox.Disable()

        if self.monitor_checkbox.GetValue() == True:
            self.style_current = copy.deepcopy(self.sc.efx_control.style_vect)
            self.set_sldr_vals(self.sc.efx_control.style_vect)

        # バックエンド側に現在のスタイルを反映させる（選ぶかどうかは向こうが決める）
        self.sc.candidate_style_list[2] = copy.deepcopy(self.style_current)


    # ローカルのスタイル csv ファイルを処理するイベントハンドラ
    # 実際に csv からデータを読み込み、 self.style_name, self.style_from_file, self.is_file_loaded を更新する
    # 形が正規化されていなくても、コンマないし改行で区切った数値が、ファイル内に計 self.dim_style 要素あれば読み込み可能

    def load_csv_to_slot(self, file_path):
        if file_path is not None:
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) CSV name: {os.path.basename(file_path)}")
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
                    valid_data = data_list[-1]
                    is_valid_length = True
                else:
                    data_concat = np.concatenate(data_list)
                    if data_concat.shape[-1] == self.dim_style:
                        valid_data = data_concat
                        is_valid_length = True
                    else:
                        is_valid_length = False
                
                # 長さが self.dim_style 要素になっていれば有効なデータとみなして進む
                if is_valid_length:
                    self.style_name = os.path.splitext(os.path.basename(file_path))[0] # 拡張子を抜いたファイル名
                    self.style_from_file = valid_data.reshape(1, self.dim_style)
                    self.logger.debug(f"({inspect.currentframe().f_code.co_name}) '{self.style_name}' is a valid style ({self.style_from_file.shape})")
                    # ファイルロードの完了フラグ
                    self.is_file_loaded = True 
                else:
                    # 失敗した場合は、現状変更されない
                    wx.MessageBox(f"The style csv must contains {self.dim_style} numerical values") # 警告ダイアログを表示
            except:
                self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Style file with valid shape was not specified, or failed to load.")
        else:
            pass # 有効なファイルパスを与えない場合は何もしない（通常はファイルが選択される前提であり、このルートには入らない）


    # 作成済みのスタイルが記載された（csv 形式）ファイルを選択してロードする
    def select_file(self, event):
        previous_path = copy.deepcopy(self.style_abs_path)
        with wx.FileDialog(
            self, 
            "Open embedding csv file. This operation will overwrite existing slider values", 
            defaultDir = os.path.dirname(self.style_abs_path) if self.style_abs_path is not None and os.path.exists(self.style_abs_path) else ".",
            wildcard = "CSV files (*.csv;*.txt)|*.csv;*.txt",
            style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return # キャンセルの場合はロード処理に入らず、現在の選択ファイルが保持される
            
            abs_path_raw = fileDialog.GetPath()
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Selecting... {abs_path_raw}")
            if abs_path_raw is not None:
                # OS 依存のパス区切り文字を修正。なおドライブレター等はうまく処理できない
                self.style_abs_path = abs_path_raw.replace('/', self.os_sep).replace('\\', self.os_sep) 

            self.load_csv_to_slot(self.style_abs_path) # ファイルをロードしてメモリ上の変数に格納するところまで実行
            # 失敗した場合はここで止まり、self.style_abs_path 以外のインスタンス変数の値は変更されない

            # 以下、ロードに成功した場合のみ
            if self.is_file_loaded:
                # 選択中のファイルを示すラベル文字列を更新する
                self.update_style_label(self.style_name)
                # ここに現在のスタイルバッファを更新する処理を書く
                self.set_sldr_vals(self.style_from_file)
                self.style_current = copy.deepcopy(self.style_from_file)
                self.save_btn.Disable()
            else:
                self.style_abs_path = previous_path # ロードに失敗した場合の後処理として、パスの変数を復元して終わり

        self.clear_btn.Show()


    def set_sldr_vals(self, style):
        for i, element in enumerate(self.slider_list):
            element.value = min(
                self.val_range[1], 
                max(
                    style[0, i], 
                    self.val_range[0],
                ),
            )
            element.sldr.SetValue(round(element.value*element.sldr_mult))
            element.value_label.SetLabel(f'{element.value: >5.2f}')

        self.save_btn.Enable()
        self.clear_btn.Enable()


    # ロードしたスタイルのラベルをセットする
    def update_style_label(
        self,
        file_name,
    ):
        if hasattr(self, 'style_label'):
            if self.is_file_loaded:
                text = f"Loaded style file:\n{truncate_string(file_name, max = 27)}" # ロード完了フラグが立っている場合
                tooltip_text = 'You can change the name when you save the style file.'
            else:
                if self.style_name == "":
                    text = f"Style name is unset" # ロードされていない（あるいは失敗した）場合
                else:
                    text = f"Style name: {self.style_name}"
                tooltip_text = f'Load a pre-calculated style embedding ({self.dim_style}) from a csv file'
            self.style_label.SetLabel(text)
            self.style_label.SetToolTip(tooltip_text)
        self.root_sizer.Layout()


    # 最終的に VC に使用するスタイル埋め込みを csv に保存する
    def save_file(self, event):
        if len(self.style_name) > 0:
            default_name = self.style_name
        else:
            now = datetime.now()
            default_name = "from_scratch_style_" + now.strftime("%Y-%m-%d_%H.%M.%S")
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
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Full style is saved at: {path}")
            with open(path, 'w', newline = '') as f:
                writer = csv.writer(f)
                writer.writerow(np.round(self.style_current.astype(float), 4).flatten().tolist()) 
                # numpy 配列をリストに変換して書き込む
                # ただし float32 を float にキャストしてから round しないと、書き出した桁数がおかしくなる


    # ファイル未選択に戻す
    def clear(self, event):
        self.is_file_loaded = False
        self.style_abs_path = None 
        self.style_name = "" # 選択中のファイルを示すラベル文字列を空欄に戻す
        self.update_style_label(self.style_name)
        self.style_from_file = np.zeros((1, self.dim_style), dtype = np.float32) # 埋め込みの値も初期化する
        self.style_current = np.zeros((1, self.dim_style), dtype = np.float32) # 埋め込みの値も初期化する
        self.set_sldr_vals(self.style_from_file)

        self.save_btn.Disable()
        self.clear_btn.Disable()

        self.Refresh()


####

class DimensionSlider(wx.Panel):
    def __init__(
        self,
        parent,
        dim: int,
        host = None,
        value: float = 0.0, 
        val_range: list = [-30, 30],
        mult: int = 100, # 0.01 刻みで入力可能になる
        sldr_size: tuple = (105, -1),
        show_label: bool = True,
        label_horizontal: bool = True,
        distance: int = 10, # distance from label to slider
        **kwargs, 
    ): 
        super().__init__(parent, id = wx.ID_ANY, **kwargs)

        self.parent = parent
        self.dim = dim
        self.host = host
        self.val_range = val_range
        self.sldr_mult = mult
        self.sldr_size = sldr_size
        self.show_label = show_label
        self.label_horizontal = label_horizontal
        self.value = value

        self.white_color = wx.Colour(255, 255, 255)
        self.light_gray_color = wx.Colour(215, 215, 225)
        self.night_gray_color = wx.Colour(23, 23, 27) # この色は元画像書き出しに合わせてあるので、任意には変更できない。

        # スタイルのパラメータ値を編集するスライダー
        self.sldr = wx.Slider(
            self, 
            value = round(self.value*self.sldr_mult), 
            minValue = math.floor(self.val_range[0] * self.sldr_mult), 
            maxValue = math.ceil(self.val_range[1] * self.sldr_mult),
            size = self.sldr_size,
            style = wx.SL_HORIZONTAL,
        )
        self.sldr.Bind(wx.EVT_SLIDER, self.on_sldr_change)
        # スライダーを右クリックすると原点に戻す
        self.sldr.Bind(wx.EVT_RIGHT_DOWN, self.on_right_click_slider)

        # Dim を表示
        if self.show_label:
            self.label = wx.StaticText(self)
            self.label.SetLabel(f'{self.dim:0>3.0f}')
            if self.dim % 10 >= 5:
                self.label.SetBackgroundColour(wx.Colour(255, 235, 245)) 

        # 現在のパラメータ値を表示
        self.value_label = wx.StaticText(self)
        self.value_label.SetLabel(f'{self.value: >5.2f}')

        # ダブルクリックすると編集
        self.value_label.Bind(wx.EVT_LEFT_DCLICK, self.on_value_double_click) 

        if self.label_horizontal == True:
            self.sizer = wx.BoxSizer(wx.HORIZONTAL)
            if self.show_label:
                self.sizer.Add(self.label, 0, flag = wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border = 0)
            self.sizer.Add(self.sldr, 0, flag = wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border = 0)
            self.sizer.Add(self.value_label, 0, flag = wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border = 0)
        else:
            self.sizer = wx.BoxSizer(wx.VERTICAL)
            if self.show_label:
                self.sizer.Add(self.label, 0, flag = wx.BOTTOM | wx.ALIGN_CENTER_HORIZONTAL, border = distance)
            self.sizer.Add(self.sldr, 0, flag = wx.BOTTOM | wx.ALIGN_CENTER_HORIZONTAL, border = 0)
            self.sizer.Add(self.value_label, 0, flag = wx.RIGHT | wx.ALIGN_RIGHT, border = 0)

        self.SetSizer(self.sizer)
        self.Layout()


    def on_sldr_change(self, event):
        self.value = self.sldr.GetValue() / self.sldr_mult
        self.value_label.SetLabel(f'{self.value: >5.2f}')
        self.host.style_current[0, self.dim] = copy.deepcopy(self.value)
        self.host.save_btn.Enable()
        self.host.clear_btn.Enable()

        event.Skip()
    
    
    def on_right_click_slider(self, event):
        slider_rect = self.sldr.GetScreenRect()
        mouse_pos = wx.GetMousePosition()
        
        # Check if the right click happened within the slider's screen rectangle
        if slider_rect.Contains(mouse_pos):
            self.value = 0.0
            self.sldr.SetValue(0)
            self.value_label.SetLabel(f'{self.value: >5.2f}')

        event.Skip()
    
    ####
    
    # StaticText を TextCtrl に置き換えて編集可能にする
    def on_value_double_click(self, event):
        text_rect = self.value_label.GetRect()
        value_str = str(self.value)
        self.value_ctrl = wx.TextCtrl(
            self.value_label.GetParent(), # self, 
            wx.ID_ANY, 
            value = value_str, 
            pos = (text_rect.x, text_rect.y),
            size = (text_rect.width, -1),
            style = wx.TE_PROCESS_ENTER,
        )
        # StaticTextを非表示にし、TextCtrlをフォーカスさせる
        self.value_label.Hide()
        self.sizer.Add(self.value_ctrl, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 0)
        self.value_ctrl.SetFocus()
        self.value_ctrl.SetSelection(0, len(value_str)) # テキスト全体を選択状態にする

        # Enterキーで確定するためのイベントハンドラを設定
        self.value_ctrl.Bind(wx.EVT_TEXT_ENTER, self.on_label_text_enter)
        # フォーカスを失ったときのイベントハンドラを設定。ただし、パネルの sizer 全体がフォーカス範囲になってしまう
        self.value_ctrl.Bind(wx.EVT_KILL_FOCUS, self.on_label_text_enter)


    def on_label_text_enter(self, event):
        # Enter キーが押されたら、TextCtrl の内容を取得して StaticTextに 反映し、TextCtrl を（削除→）隠蔽する
        self.value = round(float(filter_valid_characters(self.value_ctrl.GetValue())) / self.sldr_mult, 2)
        self.value_label.SetLabel(f'{self.value: >5.2f}')
        self.sizer.Detach(self.value_ctrl)
        self.value_ctrl.Unbind(wx.EVT_TEXT_ENTER)
        self.value_ctrl.Unbind(wx.EVT_KILL_FOCUS)
        self.value_ctrl.Hide() # 最初は Destroy() していたが Linux でエラーになるので、隠すだけにした
        self.value_label.Show()

####

def filter_valid_characters(input_str):
    valid_chars = set("0123456789-.")
    filtered_chars = [ch for ch in input_str if ch in valid_chars]
    filtered_str = ''.join(filtered_chars)

    return filtered_str

