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
from wx.adv import SplashScreen as SplashScreen
import signal
import os
import copy
import json

import logging
import inspect
import warnings

# hi dpi 対応
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except:
    pass


from audio_backend import SoundControl
from audio_device_manager import SoundIOChoice
from config_manager import load_make_app_config, load_make_vc_config
from vc_monitor_widgets import MonitorWidgets
from vc_control_widgets import FloatPanel
from vc_advanced_settings import AdvancedSettingsPanel
from sample_manager import SampleManagerPanel
from plot_content import PlotEmbeddingPanel
from style_manager import StyleManagerPanel
from style_full_manager import FullManagerPanel


# メニューバーの部品定義とイベントハンドラの作り込み。
# 最初は Frame 側で SetMenuBar() していたが、frame のメソッドをメニューから操作できない欠点があったため移管した。

class SoundAppMenu(): # フレーム直下に配置するメニューバー。
    def __init__(self, frame):
        super().__init__()

        self.frame = frame # 前のバージョンから変えた。属している親フレームを引数で指定する仕様に。
        self.menu_bar = wx.MenuBar()
        
        menu_file = wx.Menu() # wx.Menu() はメニューバー第一要素。いわゆる File のこと

        menu_SaveCurrentAppConf = menu_file.Append(wx.ID_ANY, '現在のアプリケーション設定を上書き保存')
        menu_SaveCurrentVCConf = menu_file.Append(wx.ID_ANY, '現在の VC 設定を上書き保存')
        # 属している親フレームの自作メソッドを割り当て
        self.frame.Bind(wx.EVT_MENU, self.on_save_app_conf, menu_SaveCurrentAppConf) 
        self.frame.Bind(wx.EVT_MENU, self.on_save_vc_conf, menu_SaveCurrentVCConf) 

        quitItem = menu_file.Append(wx.ID_EXIT, '終了(Q)\tCtrl+Q')
        self.frame.Bind(wx.EVT_MENU, self.frame.on_frame_close, quitItem) # 属している親フレームのデフォルトメソッドを割り当て
        
        menu_edit = wx.Menu()
        menu_edit.Append(wx.ID_ANY, 'Copy')
        menu_edit.Append(wx.ID_ANY, 'Paste')
        
        # メニューの第一階層要素を追加、すなわちルートに append する。
        self.menu_bar.Append(menu_file, 'File') 
#        self.menu_bar.Append(menu_edit, 'Edit') 

        self.frame.SetMenuBar(self.menu_bar)
        
    def on_save_app_conf(self, event):
        self.frame.save_app_conf()
        
    def on_save_vc_conf(self, event):
        self.frame.save_vc_conf()



class Frame(wx.Frame):
    def __init__(
        self, 
        app,
        app_name = "Application",
        window_size = (1280, 960),
    ): 
        super().__init__(None, -1, app_name, size = window_size)

        # アイコンの設定
        icon = wx.Icon("images/MMCXLI-logo.ico", wx.BITMAP_TYPE_ICO) 
        self.SetIcon(icon)

        self.app = app

        self.app_config = self.app.app_config
        self.vc_config = self.app.vc_config
        # アプリ開始時点でのオリジナルの config を保持しておく
        self.app_config_orig = copy.deepcopy(self.app_config)
        self.vc_config_orig = copy.deepcopy(self.vc_config)

        self.style_from_sample = True # サンプル音声から 128 次元の埋め込みを計算するか

        #### アプリケーションウィンドウの初期配置
        
        # ステータスバーの作成
        self.sb = self.CreateStatusBar(number = 4)
        self.sb.SetStatusText('Ready', i = 0) # ステータスバーに文字を表示させる
        
        # メニューバーの作成
        SoundAppMenu(self)
        
        self.SetMinSize(self.app_config["window_min_size"]) # 描画領域の最小サイズを設定
        self.Bind(wx.EVT_SIZE, self.on_size)  # サイズ変更時の最小サイズを制約するイベントハンドラ
        self.size = self.GetSize() # フレーム自身のサイズを変数に格納しておく
        self.pos = self.GetScreenPosition()

        #### タブの作成

        # パネル内部の機能は、タブ用ルートパネルの初期化後に作り込む。
        # タブに所属させるパネルを作成するとき、 parent がタブ内のルートに揃わねばならないという制約があるため。

        self.notebook = wx.Notebook(self, wx.ID_ANY)

        # Notebook 内の各タブのルートとなるパネルを、タブの数だけ作成する。
        # なお各タブ用の root sizer は notebook には直接配置できないので、root panel を噛ませる。
        # つまり notebook -> *_tab_panel -> *_tab_sizer -> 個別機能パネル の階層性を持つ。
        
        # なお各タブに 1 つのパネルしか配置しない場合は notebook に直接 InsertPage で個別機能パネルをぶら下げてもいい。
        # ただし sizer がない分、配置の自由度が下がる。

        # 波形プロットを配置するパネル
        self.monitor_tab_panel = wx.Panel(self.notebook, wx.ID_ANY)
        self.monitor_tab_sizer = wx.BoxSizer(wx.VERTICAL)
        self.monitor_tab_panel.SetSizer(self.monitor_tab_sizer) 
        self.monitor_tab_sizer.Fit(self.monitor_tab_panel)

        # 音声デバイス管理のテーブルを配置するパネル
        self.io_tab_panel = wx.Panel(self.notebook, wx.ID_ANY)
        self.io_tab_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.io_tab_panel.SetSizer(self.io_tab_sizer)
        self.io_tab_sizer.Fit(self.io_tab_panel)

        # ContentVec の結果を表示するパネル
        if self.app_config["display_content"] is True:
            self.content_tab_panel = wx.Panel(self.notebook, wx.ID_ANY)
            self.content_tab_sizer = wx.BoxSizer(wx.VERTICAL)
            self.content_tab_panel.SetSizer(self.content_tab_sizer)

        # VC 制御部品を配置するパネル
        self.ctrl_tab_panel = wx.Panel(self.notebook, wx.ID_ANY)
        self.controller_tab_sizer = wx.BoxSizer(wx.VERTICAL)
        self.ctrl_tab_panel.SetSizer(self.controller_tab_sizer)

        # 話者スタイル編集を配置するパネル
        self.style_tab_panel = wx.Panel(self.notebook, wx.ID_ANY)
        self.style_tab_sizer = wx.BoxSizer(wx.VERTICAL)
        self.style_tab_panel.SetSizer(self.style_tab_sizer)

        # 話者スタイル編集を配置するパネル
        if self.style_from_sample is True:
            self.sampler_tab_panel = wx.Panel(self.notebook, wx.ID_ANY)
            self.sampler_tab_sizer = wx.BoxSizer(wx.VERTICAL)
            self.sampler_tab_panel.SetSizer(self.sampler_tab_sizer)
        
        # 話者スタイルの 128 dim 全部を手動で制御するパネル
        self.obsessed_tab_panel = wx.Panel(self.notebook, wx.ID_ANY)
        self.obsessed_tab_sizer = wx.BoxSizer(wx.VERTICAL)
        self.obsessed_tab_panel.SetSizer(self.obsessed_tab_sizer)

        # 各タブのルートとなるパネルを、notebook に配置する。この配置順がタブの並び順になる
        is_inserted_0 = self.notebook.InsertPage(index = 0, page = self.monitor_tab_panel, text = 'Monitor', select = True) 
        is_inserted_1 = self.notebook.InsertPage(index = 1, page = self.io_tab_panel, text = 'Audio Devices')
        if self.app_config["display_content"] is True:
            is_inserted_2 = self.notebook.InsertPage(index = 2, page = self.content_tab_panel, text = 'ContentVec')
            is_inserted_3 = self.notebook.InsertPage(index = 3, page = self.ctrl_tab_panel, text = 'Advanced Buffer Settings')
            if self.style_from_sample is True:
                is_inserted_4 = self.notebook.InsertPage(index = 4, page = self.sampler_tab_panel, text = 'Sampler')
            is_inserted_5 = self.notebook.InsertPage(index = 5, page = self.style_tab_panel, text = 'Style Editor')
            is_inserted_6 = self.notebook.InsertPage(index = 6, page = self.obsessed_tab_panel, text = 'Full Style Editor')
        else:
            is_inserted_2 = self.notebook.InsertPage(index = 2, page = self.ctrl_tab_panel, text = 'Advanced Buffer Settings')
            if self.style_from_sample is True:
                is_inserted_3 = self.notebook.InsertPage(index = 3, page = self.sampler_tab_panel, text = 'Sampler')
            is_inserted_4 = self.notebook.InsertPage(index = 4, page = self.style_tab_panel, text = 'Style Editor')
            is_inserted_5 = self.notebook.InsertPage(index = 5, page = self.obsessed_tab_panel, text = 'Full Style Editor')
        
        # 動的にアクティブタブを設定する
        self.active_tab = self.app_config["initial_active_tab"]
        self.notebook.SetSelection(self.active_tab)
        # 現在のアクティブタブ変数を更新するイベント
        self.notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.on_tab_changed)

        #### stream backend の作成

        # フレームの描画と独立したサイクルで音声入出力を行うバックエンド。
        # こいつは GUI がなく wx とは無関係（波形等の表示は専用クラスに任せる）
        self.sc = SoundControl(
            host = self,
            vc_config = self.vc_config,
            keep_voiced = self.vc_config["keep_voiced"], 
        )

        # input → output ではなく output → input の起動順でも動くが、遅延が 0.01 秒程度増える。
        self.sc.input_stream.start()
        self.sc.output_stream.start()
        
        #### 各機能パネルの初期化（stream backend の初期化が前提）
        
        # 以下は VC を構成する各機能がパネルとして実装されているので、配置先を指定 ＆ sizer に Add する必要がある。
        
        # 入出力を選択させる部品
        self.device_choice = SoundIOChoice(
            self.io_tab_panel, # parent
            backend = self.sc,
        )
        self.io_tab_sizer.Add(self.device_choice, proportion = 0, flag = wx.GROW | wx.ALL, border = 0) 
        
        # # 波形およびスペクトログラムをプロットするパネル（上の sc を受ける）を作成し、monitor tab 用の sizer に配置。
        self.monitor_widgets_panel = MonitorWidgets(
            self.monitor_tab_panel, # parent
            sc = self.sc, # backend
            tab_id = 0, # 何番目のタブに属するか。リアルタイム描画の必要性判定に使う
        )
        self.monitor_tab_sizer.Add(self.monitor_widgets_panel, proportion = 0, flag = wx.GROW | wx.ALL, border = 0) 
        self.monitor_widgets_panel.SetMinSize(self.size) # 最低保証サイズ
        self.monitor_widgets_panel.SetupScrolling() # こいつだけはスクロールを有効化する必要がある

        # ContentVec embedding プロット用のパネル（必要な場合のみ）
        if self.app_config["display_content"] is True:
            self.emb_panel = PlotEmbeddingPanel(
                self.content_tab_panel, # parent
                self.sc.efx_control, # host
                "buf_emb", # target_name: target buffer on the host
                hop_sec = 0.02, # 描画対象である ContentVec が 1 frame で何秒に相当するか
            ) 
            self.content_tab_sizer.Add(self.emb_panel, proportion = 0, flag = wx.GROW | wx.ALL, border = 0)

        # VC の制御を行う部品のパネル。ここまでに backend と efx_control を定義済みであること。
        self.controller_panel = AdvancedSettingsPanel(
            parent = self.ctrl_tab_panel,
            backend = self.sc, # SoundControl インスタンス。先にバックエンド側ストリームが開始している必要がある。
        )
        self.controller_tab_sizer.Add(self.controller_panel, proportion = 0, flag = wx.GROW | wx.ALL, border = 0) 
        self.controller_panel.SetMinSize(self.size) # 最低サイズ。これがないと  assertion 'size >= 0' failed in GtkScrollbar
        self.controller_panel.SetupScrolling() # こいつだけはスクロールを有効化する必要がある

        # サンプラー
        if self.style_from_sample:
            self.sampler_panel = SampleManagerPanel(
                parent = self.sampler_tab_panel,
                backend = self.sc, # SoundControl インスタンス。先にバックエンド側ストリームが開始している必要がある。
                harmof0_ckpt = self.vc_config["model"]["harmof0_ckpt"],
                SE_ckpt = self.vc_config["model"]["SE_ckpt"],
                max_slots = self.app_config["max_slots"], 
                portfolio_path = self.app_config["sample_portfolio_path"], 
            )
            self.sampler_tab_sizer.Add(self.sampler_panel, proportion = 0, flag = wx.GROW | wx.ALL, border = 0)
            self.sampler_panel.SetMinSize(self.size) # 最低サイズ。これがないと  assertion 'size >= 0' failed in GtkScrollbar
            self.sampler_panel.SetupScrolling() # こいつだけはスクロールを有効化する必要がある

        # スタイル編集
        self.style_panel = StyleManagerPanel(
            self.style_tab_panel, 
            backend = self.sc, 
            tab_id = 4, # 何番目のタブに属するか。カーソルの当たり判定に使う
            style_compressor_ckpt = self.vc_config["model"]["style_compressor_ckpt"],
            style_decoder_ckpt = self.vc_config["model"]["style_decoder_ckpt"],
            max_slots = self.app_config["max_slots"], 
            restore_slot = self.app_config["restore_slot"],
            portfolio_path = self.app_config["style_portfolio_path"], 
        )
        self.style_tab_sizer.Add(self.style_panel, proportion = 0, flag = wx.GROW | wx.ALL, border = 0)
        self.style_panel.SetMinSize(self.size) # 最低サイズ。これがないと  assertion 'size >= 0' failed in GtkScrollbar
        self.style_panel.SetupScrolling() # こいつだけはスクロールを有効化する必要がある

        # 全次元スタイル編集
        self.obsessed_panel = FullManagerPanel(
            self.obsessed_tab_panel, 
            backend = self.sc, 
        )
        self.obsessed_tab_sizer.Add(self.obsessed_panel, proportion = 0, flag = wx.GROW | wx.ALL, border = 0)
        self.obsessed_panel.SetMinSize(self.size) # 最低サイズ。これがないと  assertion 'size >= 0' failed in GtkScrollbar
        self.obsessed_panel.SetupScrolling() # スクロールを有効化する


        #### 緊急対応ボタンや基礎的なモニターを集めたパネル（タブよりも上位）

        self.float_control_panel = FloatPanel(self, self.sc)
        
        ####

        # 統括用の sizer である self.root_sizer を作り、タブのルートである notebook を配置していく。
        self.root_sizer = wx.BoxSizer(wx.VERTICAL)
        self.root_sizer.Add(self.float_control_panel, 0, wx.EXPAND | wx.ALL, border = 0)
        self.root_sizer.Add(self.notebook, 0, wx.EXPAND | wx.ALL, border = 0)
        self.SetSizer(self.root_sizer)
        self.Layout()
        
        # パネル内部の機能を初期化し終えたので、後は終了処理や、主窓上の独立したループ処理などを定義。

        # 以下はステータスバー（plot とも stream とも別サイクルの更新処理）を、主窓に入れる場合のみ
        self.timer = wx.Timer(self) # wx.Timer クラスで、指定間隔での処理を実行する。
        # wx.EvtHandler 由来の Bind() メソッドで、イベントの種類とハンドラ・メソッドを紐付ける。
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.timer.Start(37) # タイマーは手動で起動する必要がある。単位 ms

        # メインウィンドウを閉じたときの挙動として、アプリの終了を追加
        self.Bind(wx.EVT_CLOSE, self.on_frame_close)

    ####

    # 現在、update はステータスバーの更新作業に特化している。
    def update(self, event):

        self.size = self.GetSize()
        self.pos = self.GetScreenPosition()

        self.sb.SetStatusText(
            "dBFS (I/O): {: >7.1f} dB, {: >7.1f} dB (thresh: {: >7.1f} dB)".format(
                self.sc.input_dBFS, 
                self.sc.output_dBFS,
                self.sc.VC_threshold,
            ), 
            i = 0,
        )

        self.sb.SetStatusText(
            "{:_>7.2f}->{:_>7.2f} s | VC lap {:_>5.1f} ms | RTF {: >6.3f}".format(
                self.sc.head_i/self.sc.sr_out,
                self.sc.head_o/self.sc.sr_out,
                self.sc.efx_control.vc_lap, # 120 ms くらい → le_proc = 64 だと 180 ms まで伸びる
                self.sc.efx_control.vc_lap / (1000 * self.sc.blocksize / self.sc.sr_out), 
            ), 
            i = 1,
        )

        self.sb.SetStatusText(
            "(CE{:_>5.1f} | w2s{:_>5.1f} | SE{:_>5.1f} | f0n{:_>5.1f} | D{:_>5.1f} )".format(
                self.sc.efx_control.CE_lap, # 10 ないし 20 ms
                self.sc.efx_control.harmof0_lap,
                self.sc.efx_control.SE_lap, # 10 前後
                self.sc.efx_control.f0n_lap, # 20 ms まで伸びる → 少し len_proc の影響を受ける
                self.sc.efx_control.decode_lap, # 50--90 ms くらい → 少し len_proc の影響を受ける
            ), 
            i = 2,
        )

        self.sb.SetStatusText(
            "Plot lapse: wav {: >4.2f} ms, spec {: >4.2f}/{: >4.2f} ms".format(
                self.monitor_widgets_panel.wav_i_panel.lapse,
                self.monitor_widgets_panel.mel_pre_panel.lapse,
                self.monitor_widgets_panel.mel_post_panel.lapse,
            ), 
            i = 3,
        )


    def on_tab_changed(self, event):
        # 現在のアクティブなタブの番号を取得
        self.active_tab = self.notebook.GetSelection()


    # 現在の app_config を上書き保存する。上のメニューから呼び出せる。
    def save_app_conf(self):
        with open(self.app.app_config_path, 'w') as f:
            json.dump(self.app_config, f, indent = 4)


    # 現在の vc_config を上書き保存する。上のメニューから呼び出せる。
    def save_vc_conf(self):
        with open(self.app.vc_config_path, 'w') as f:
            json.dump(self.vc_config, f, indent = 4)


    # プログラム内で vc_config 由来の設定値を更新した時、メモリ上の元の dict に書き戻す。
    # save を指定した場合は、update された vc_config をローカルの json ファイルに上書きする処理まで行う。
    # ちなみに target_dict の更新は常に inplace で実行される
    def update_vc_config(
        self,
        key,
        value,
        target_dict: dict = None,
        sub_dict: str = None,
        save: bool = True,
    ):
        if target_dict is None:
            target_dict = self.vc_config
        
        if sub_dict is not None:
            target_dict[sub_dict][key] = value # 一部のキーは ["model"] や ["style"] 等のサブ辞書に入っている
        else:
            target_dict[key] = value
        
        if save:
            self.save_vc_conf()


    def on_size(self, event):
        # ウィンドウサイズが最小サイズより小さい場合、最小サイズに制限する
        width, height = self.GetSize()
        min_width, min_height = self.GetMinSize()
        
        if width < min_width or height < min_height:
            self.SetSize(max(width, min_width), max(height, min_height))
        event.Skip()


    # メインウィンドウを閉じたときの挙動
    def on_frame_close(self, event):
        self._on_frame_close()


    # ウィンドウを閉じたときの挙動には、アプリケーションの終了処理まで含まれている
    # TODO Windows においてアプリケーションを数十分以上起動すると、終了処理が正しく走らなくなる。
    def _on_frame_close(self):
        self.sc.input_stream.stop() 
        self.sc.input_stream.close() 
        self.sc.output_stream.stop() 
        self.sc.output_stream.close() 
        # サンプラーは独自のオーディオストリームを持つので（贅沢だねぇ）、ご退場願う
        if self.style_from_sample:
            self.sampler_panel.output_stream.stop() 
            self.sampler_panel.output_stream.close() 
        self.sc.terminate() # audio backend 自体を終了
        self.Destroy() # frame 自体を終了
        self.app.ExitMainLoop() # アプリケーションを終了


# 警告メッセージをログファイルにリダイレクトするためのフック
def _warning_handler(message, category, filename, lineno, file = None, line = None):
    logging.warning(
        f'{filename}({lineno}): {category.__name__}: {message}',
        stack_info = True
    )


class MyApp(wx.App):
    def OnInit(self):
        # スプラッシュスクリーンを表示。wx.Adv を使わないと画像が即時ロードされない等の厄介な問題がある。
        # 現在 Windows では以下のコードで動くが、linux だと表示されない。
        if str(os.name) == "nt":
            splash = SplashScreen(
                wx.Bitmap("./images/MMCXLI-logo-256.png", wx.BITMAP_TYPE_PNG), 
                wx.adv.SPLASH_CENTRE_ON_SCREEN | wx.adv.SPLASH_TIMEOUT,
                6000, 
                None, 
                -1
            )
            splash.Show()

        # ログや設定のフォルダがなければ明示的に作成しておく
        os.makedirs("./logs", exist_ok = True)
        os.makedirs("./configs", exist_ok = True)
        os.makedirs("./styles", exist_ok = True)

        # ログの吐き出し設定
        log_name = f"./logs/app_latest.log" 
        logging.basicConfig(
            filename = log_name, 
            level = logging.DEBUG, 
            format = '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
            encoding = 'utf-8',
        )
        
        # 一部の「うるさい」ライブラリについて、個別のログレベルを設定
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('numba').setLevel(logging.WARNING)
        # 警告フックを設定
        warnings.showwarning = _warning_handler

        #### VC 用 config 辞書のロード、または（ない場合）作成。バックエンドより前にロードが必要

        self.app_config_path = "./configs/app_config.json"
        self.vc_config_path = "./configs/vc_config.json"
        self.app_config = load_make_app_config(self.app_config_path, debug = True)
        self.vc_config = load_make_vc_config(self.vc_config_path, debug = True)

        self.frame = Frame(self, self.app_config["application_name"], self.app_config["window_size"])

        self.frame.Show()

        # メインフレームが表示された後、スプラッシュスクリーンを消去する
        if str(os.name) == "nt":
            wx.CallLater(2000, self.CloseSplashScreen, splash)

        return True


    def CloseSplashScreen(self, splash):
        if splash:
            splash.Destroy()


    # Ctrl + C が押されたときの処理。
    # アプリ自体の終了（ExitMainLoop）もフレーム側に実装されている。お作法的に良いのかどうかは知らん
    def sigint_handler(self, signum, frame):
        self.frame._on_frame_close()


####


def main():
    pid = os.getpid()
    print(f"Process ID: {pid}")

    application = MyApp(redirect = False)
    signal.signal(signal.SIGINT, application.sigint_handler) # Ctrl + C のシグナルハンドラを登録
    application.MainLoop()


if __name__ == '__main__':
    main() 

