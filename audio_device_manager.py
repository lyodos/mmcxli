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

import logging
import inspect

class SoundIOTable(wx.ListCtrl):
    def __init__(
        self, 
        parent, 
        backend,
    ):
        wx.ListCtrl.__init__(self, parent, -1, style = wx.LC_REPORT) # 行ごとに横区切り線

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.logger.debug("Initializing ...")

        self.backend = backend
        self.lab_dict = {
            "name": "Device Name", 
            "index": "ID", 
            "hostapi": "API", 
            "max_input_channels": "In ch",
            "max_output_channels": "Out ch",
            "default_low_input_latency": "Latency (low I)",
            "default_low_output_latency": "Latency (low O)",
            "default_high_input_latency": "Latency (high I)",
            "default_high_output_latency": "Latency (high O)",
            "default_samplerate": "SR (default)",
        }

        # 以下の self.backend.dicts_dev_raw は sd.query_devices() の返り値。[0] がチャンネル数 0 も含めた全デバイスの辞書
        self.keys = self.backend.dicts_dev_raw[0].keys()
        self.col_names = ["", "In", "Out"]
        self.InsertColumn(0, self.col_names[0], width = 40)
        self.InsertColumn(1, self.col_names[1], width = 40)
        self.InsertColumn(2, self.col_names[2], width = 40)
        
        # 列を作成
        for col, lab in enumerate(self.keys):
            self.col_names.append(lab)
            if lab == "name":
                self.InsertColumn(col+3, self.lab_dict[lab], width = wx.LIST_AUTOSIZE)
                self.SetColumnWidth(col+3, 300)
            elif lab in self.lab_dict.keys():
                self.InsertColumn(col+3, self.lab_dict[lab], width = wx.LIST_AUTOSIZE_USEHEADER)
            else:
                self.InsertColumn(col+3, lab, width = wx.LIST_AUTOSIZE_USEHEADER) # self.lab_dict に対応がない場合
        
        # 作成済みの列に行を追加
        for row in range(len(self.backend.dicts_dev_raw)):
            self.InsertItem(index = row, label = str(row))
            for col, lab in enumerate(self.col_names):
                self.SetItem(row, col, str(self.backend.dicts_dev_raw[row].get(lab)))
            # まず In, Out の初期値を - 無効で上書き
            self.SetItem(row, 1, "-")
            self.SetItem(row, 2, "-")
            if row == self.backend.dev_ids_in_use[0]:
                self.SetItem(row, 1, str(1))
            if row == self.backend.dev_ids_in_use[1]:
                self.SetItem(row, 2, str(1))
            self.SetItem(row, 0, str(row)) # 最後に、冒頭の ID 列に機械的に連番を入れる
            
    # なお、SetItem メソッドを使えば作成済みのテーブルを書き換えられる。
    def update_device_list(
        self,
    ):
        # 全デバイス辞書の要素 = テーブルの行ごとに処理
        for row in range(len(self.backend.dicts_dev_raw)):

            if row == self.backend.dev_ids_in_use[0]:
                self.SetItem(row, 1, str(1))
            else:
                self.SetItem(row, 1, "-")

            if row == self.backend.dev_ids_in_use[1]:
                self.SetItem(row, 2, str(1))
            else:
                self.SetItem(row, 2, "-")


####

# サウンドデバイスの選択インターフェースである SoundIOChoice は表示部品を持つので wx.Panel を継承する必要がある。

# 既知の問題：Windows で WASAPI と他の API の相互切り替えを試みると異常動作し、アプリの強制終了が必要。
# ただし単に作者のマシンで WASAPI の環境が正しく設定されていないだけかもしれない。

# あるいは WASAPI だけ対応するサンプリング周波数がずれている可能性。
# この場合、サンプリング周波数の異なるバックエンドの作り直しが本アプリケーションの仕様上困難であるため、
# 手動で config を書き変えてから起動してくださいという話になる。


class SoundIOChoice(wx.Panel):
    def __init__(
        self, 
        parent, 
        backend, 
        **kwargs,
    ):
        super().__init__(parent, id = wx.ID_ANY, **kwargs) # parent として通常は wx.Panel（wx.Frame でも可）を指定する。

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.logger.debug("Initializing ...")

        self.backend = backend # こいつは参照渡しなので、backend を本クラスのインスタンスから再定義できる。

        # 音声デバイスの table list を作成。parent の指定が必須。
        self.dev_table = SoundIOTable(self, self.backend)

        self.make_api_choice() # 部品定義を全てメソッドにした。 self.io_choice_sizer がここで作られる。
        self.make_device_choice()  # 部品定義を全てメソッドにした。
        
        self.block_roll_panel = BlockRollPanel(self, self.backend) # audio backend のサイズ自体を変更できるスライダー
        
        # 選択肢（およびデバイス一覧表）をその中に配置するための、BoxSizer を作成

        # デバイス選択の各パーツを垂直配置する孫 sizer
        self.device_sizer = wx.BoxSizer(wx.VERTICAL) 
        self.device_sizer.Add(self.api_rbx, 0, flag = wx.TOP, border = 5) # api
        self.device_sizer.Add(self.io_choice_sizer, 0, flag = wx.TOP, border = 5) # device
        # ブロックサイズとデバイス選択を水平配置する子 sizer
        self.select_sizer = wx.BoxSizer(wx.HORIZONTAL) 
        self.select_sizer.Add(self.device_sizer, 0, flag = wx.ALL | wx.ALIGN_BOTTOM, border = 0) # api + device
        self.select_sizer.AddStretchSpacer(1) # なお self.root_sizer.Add するとき EXPAND しないとスペースが作られない
        self.select_sizer.Add(self.block_roll_panel, 0, flag = wx.TOP | wx.RIGHT | wx.ALIGN_BOTTOM, border = 5) # blocksize
        
        # 統括 sizer を作成し、すべてを配置
        self.root_sizer = wx.BoxSizer(wx.VERTICAL) 
        self.root_sizer.Add(self.select_sizer, proportion = 0, flag = wx.EXPAND | wx.LEFT, border = 5)
        self.root_sizer.Add(self.dev_table, proportion = 1, flag = wx.GROW | wx.LEFT | wx.TOP, border = 10)
        self.SetSizer(self.root_sizer)
        self.root_sizer.Fit(self) 
        self.Layout()


    # API 選択肢のラジオボタンを作る
    def make_api_choice(self):
        self.api_rbx = wx.RadioBox(
            self, 
            id = wx.ID_ANY, 
            label = "Audio API", 
#            size = (300, -1),
            choices = [str(x) for x in self.backend.apis_installed], 
            style = wx.LC_REPORT | wx.LC_HRULES | wx.RA_HORIZONTAL,
        )
        self.api_rbx.SetSelection(self.backend.api_id_in_use)
        self.api_rbx.Bind(wx.EVT_RADIOBOX, self.on_choice_api)

        # 不許可の API を選べないようにする
        for i, api_dict in enumerate(self.backend.dicts_apis):
            if i not in self.backend.apis_allowed:
                self.api_rbx.EnableItem(i, False)
            # ツールチップの定義
            self.api_rbx.SetItemToolTip(i, 'Select audio API')


    # デバイス選択肢を作成し、画面表示部品を作る
    def make_device_choice(self):
        # まず入出力それぞれについて、チャンネル数が 1 以上のデバイスだけを残す。
        # self.backend.dicts_dev_raw は scan() の最初に作成されるデバイス一覧の辞書
        # API を指定して self.backend.dev_avbl_on_api （現在選択中の API で利用可能なデバイス ID）の値で絞り込む。
        # さらに、self.backend.dev_strict_i_names, self.backend.dev_strict_o_names に名称が含まれないデバイスを落とす。
        self.i_dicts_avbl = []
        self.i_ids_avbl = []
        self.o_dicts_avbl = []
        self.o_ids_avbl = []
        for i, device in enumerate(self.backend.dicts_dev_raw):
            if i in self.backend.dev_avbl_on_api and device["max_input_channels"] > 0 and self.backend.strict_avbl_i[i]:
                self.i_dicts_avbl.append(device)
                self.i_ids_avbl.append(i)
            if i in self.backend.dev_avbl_on_api and device["max_output_channels"] > 0 and self.backend.strict_avbl_o[i]:
                self.o_dicts_avbl.append(device)
                self.o_ids_avbl.append(i)

        # 以下の SetSelection において、使用不可であるはずのデバイスが.backend.dev_ids_in_use[0 or 1] の
        # 選択値になっていた場合にエラー。これはもっと手前の段階で対処する必要がある。

        self.cho_i = wx.Choice(
            self, 
            id = wx.ID_ANY,
#            size = (350, 50), # ただし Windows では Choice は縦サイズが最小値になってしまうので無効
            choices = [str(x) for x in self.i_ids_avbl], 
            name = "Select_input_device",
        )
        self.cho_i.SetMinSize((300, 50))
        self.cho_i.SetMaxSize((400, 50))
        self.cho_i.SetSelection(self.i_ids_avbl.index(self.backend.dev_ids_in_use[0])) # 選択値をセット
        # 選択肢データは int で保持されるが、デバイス名称を文字列として保持し表示させる。
        for n, device in enumerate(self.i_dicts_avbl):
            self.cho_i.SetString(
                n, 
                str(self.i_ids_avbl[n]) + " [" + self.backend.apis_installed[device["hostapi"]] + "] " + 
                " " + device["name"] + 
                " (" + str(device["max_input_channels"]) + " in, " + str(device["max_output_channels"]) + " out)",
            )
        # ユーザーが任意のタイミングで選択肢を選んだときのイベントリスナを追加
        self.cho_i.Bind(wx.EVT_CHOICE, self.OnChoice_i)

        self.cho_o = wx.Choice(
            self, 
            id = wx.ID_ANY,
            choices = [str(x) for x in self.o_ids_avbl], 
            name = "Select_input_device",
        )
        self.cho_o.SetMinSize((300, 50))
        self.cho_o.SetMaxSize((400, 50))
        self.cho_o.SetSelection(self.o_ids_avbl.index(self.backend.dev_ids_in_use[1])) # 選択値をセット
        for n, device in enumerate(self.o_dicts_avbl):
            self.cho_o.SetString(
                n, 
                str(self.o_ids_avbl[n]) + " [" + self.backend.apis_installed[device["hostapi"]] + "] " + 
                " " + device["name"] + 
                " (" + str(device["max_input_channels"]) + " in, " + str(device["max_output_channels"]) + " out)",
            )
        self.cho_o.Bind(wx.EVT_CHOICE, self.OnChoice_o)

        # sizer を作って、i/o それぞれを配置しておく
        self.io_choice_sizer = wx.BoxSizer(wx.HORIZONTAL) 
        self.io_choice_sizer.Add(self.cho_i, 0, flag = wx.GROW, border = 0)
        self.io_choice_sizer.Add(self.cho_o, 0, flag = wx.GROW, border = 0)


    # 選択 API を変えたときのイベントハンドラ

    # 使う関数は change_device() だが api_pref に値を指定し、かつ device を None で代入。
    # 選んだ API によって device がサポートされている場合はなるべく変えたくないので、その処理は足す必要。
    def on_choice_api(self, event):
        self.backend.api_pref = self.backend.apis_installed[self.api_rbx.GetSelection()]
        self.logger.debug(f"Switching audio API to '{self.backend.api_pref}...")
        self.backend.change_device(
            sr_proc = None,#self.backend.sr_proc,
            sr_out = None,
            device = None, 
            n_ch_proc = None,#self.backend.n_ch_proc,
            n_ch_max = None,#self.backend.n_ch_max,
            latency = None,#self.backend.latency,
            api_pref = self.backend.api_pref, # api_pref に値を指定し、かつ device を None で代入。
        )
        self.logger.debug(f"Audio API switched to {self.backend.apis_installed[self.backend.api_id_in_use]} (device ids: {self.backend.dev_ids_in_use})")


    # 選択デバイスを変えたときのイベントハンドラ
    
    # GetSelection() が取得するのは「現在選択肢にあるデバイスの中での順番」なので「全デバイスの中での番号」に換算。

    # 理論上は self.backend.dev_ids_in_use の値を変えると、対応 API、サンプリング周波数、チャンネル数も変える必要がある。
    # これらの値を変更してから放り込むか、change_device() 内で自動対応させるか？
    # → 入力の API を変えると出力の API も連動して変える必要がある。なので change_device() 内で対応させる方が楽。
    # ただし現在の UI では API を絞り込んだ時点で入出力が必ず同じ API になるので、どちらでも実はいい。

    def OnChoice_i(self, event):
        self.backend.dev_ids_in_use[0] = self.i_ids_avbl[self.cho_i.GetSelection()] # self.i_ids_avbl は API ごとに作り直し
        self.logger.debug(f"Switching input device to '{self.backend.dev_ids_in_use}'...")
        self.dev_table.update_device_list()
        self.backend.change_device(
            sr_proc = self.backend.sr_proc,
            sr_out = None,#self.backend.sr_out,
            device = self.backend.dev_ids_in_use, 
            n_ch_proc = None,#self.backend.n_ch_proc,
            n_ch_max = None,#self.backend.n_ch_max,
            api_pref = None,#self.backend.api_pref,
            latency = None,#self.backend.latency,
        )
        self.logger.debug(f"Sound I/O devices switched to {self.backend.dev_ids_in_use} (default is {self.backend.dev_ids_default})")

    def OnChoice_o(self, event):
        self.backend.dev_ids_in_use[1] = self.o_ids_avbl[self.cho_o.GetSelection()]
        self.logger.debug(f"Switching output device to '{self.backend.dev_ids_in_use}'...")
        self.dev_table.update_device_list()
        self.backend.change_device(
            sr_proc = self.backend.sr_proc,
            sr_out = None,#self.backend.sr_out,
            device = self.backend.dev_ids_in_use, 
            n_ch_proc = None,#self.backend.n_ch_proc,
            n_ch_max = None,#self.backend.n_ch_max,
            api_pref = None,#self.backend.api_pref,
            latency = None,#self.backend.latency,
        )
        self.logger.debug(f"Sound I/O devices switched to {self.backend.dev_ids_in_use} (default is {self.backend.dev_ids_default})")


####

# blocksize をホットに変更する。なお、sr_out の途中変更は影響範囲が大きすぎるので今の所検討していない。
# 再三書いているようにこいつの挙動は著しく不安定なので、通常は config 側でブロックサイズを書き換えてから起動すべき

class BlockRollPanel(wx.Panel):
    def __init__(
        self, 
        parent, 
        backend, 
        **kwargs,
    ):
        super().__init__(parent, id = wx.ID_ANY, **kwargs)

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.logger.debug("Initializing ...")

        self.backend = backend
        
        # block_roll_size を通じて blocksize を制御するスライダー
        self.block_roll_coef: float = 1
        self.block_roll_default = math.ceil(self.backend.block_roll_size) # ただし block_roll_size 自体が int を保証される
        self.block_roll_sldr = wx.Slider(
            self, 
            value = self.block_roll_default, 
            minValue = 2, # 最低ロール量 40 ms
            maxValue = 10, # 最大ロール量 200 ms 
            size = (300, -1),
            style = wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS,
        )
        for value in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            self.block_roll_sldr.SetTick(value)
        self.block_roll_sldr.Bind(wx.EVT_SLIDER, self.on_block_roll_sldr_change)
        self.block_roll_sldr.SetToolTip('[WARNING] Hot-resizing of the audio backend blocksize makes loud noise and potentially cause the application to crash!')
        self.block_roll_box = wx.StaticBox(
            self, 
            wx.ID_ANY, 
            "(Advanced) Audio blocksize"
        )
        self.block_roll_text = wx.StaticText(self) # ラベルとして使用する StaticText を作成
#        self.block_roll_text.SetLabel(f'20 * {self.block_roll_default} ms / iteration [Default: * {self.block_roll_default}]')
        self.block_roll_text.SetLabel(f'{self.backend.blocksize} (20*{self.backend.block_roll_size} ms / iteration) [Default: *{self.block_roll_default}]')
        self.root_sizer = wx.StaticBoxSizer(self.block_roll_box, wx.VERTICAL)
        self.root_sizer.Add(self.block_roll_text, 0, flag = wx.LEFT | wx.TOP, border = 10)
        self.root_sizer.Add(self.block_roll_sldr, 0, flag = wx.EXPAND | wx.ALL, border = 5)

        # 注意喚起のためテキスト色を赤に
        self.block_roll_box.SetForegroundColour(wx.Colour(255, 50, 0))
        self.block_roll_text.SetForegroundColour(wx.Colour(255, 50, 0))
        
        self.SetSizer(self.root_sizer)
        self.root_sizer.Fit(self) 
        self.Layout()


    # これは単に内部変数を弄るだけではなく、stream の再作成が必要
    def on_block_roll_sldr_change(self, event):
        if hasattr(self, 'block_roll_sldr') and self.block_roll_sldr is not None:
            self.backend.block_roll_size = self.block_roll_sldr.GetValue()
            self.backend.block_sec = self.backend.block_roll_size*0.02 # 実時間で何秒おきに VC を呼び出すか
            # 実際に backend を作るときの blocksize をここで確定
            self.backend.blocksize = int(self.backend.block_sec * self.backend.sr_out) 
            self.backend.update_vc_config("blocksize", int(self.backend.blocksize), sub_dict = "backend", save = False)
        
            self.block_roll_text.SetLabel(f'{self.backend.blocksize} (20*{self.backend.block_roll_size} ms / iteration) [Default: *{self.block_roll_default}]')
            self.backend.need_remake_stream = True # stream を作り直すので、ロックが必要
            
        # 実際に stream を再起動する処理は、backend に実装しておいてここから呼び出す。
        
        self.logger.debug(f"Switching audio blocksize to {self.backend.blocksize}. Remake backend")
        self.backend.change_device(
            sr_proc = None,#self.backend.sr_proc,
            sr_out = None,#self.backend.sr_out,
            device = None,#self.backend.dev_ids_in_use, 
            n_ch_proc = None,#self.backend.n_ch_proc,
            n_ch_max = None,#self.backend.n_ch_max,
            api_pref = None,#self.backend.api_pref,
            latency = None,#self.backend.latency,
        )
