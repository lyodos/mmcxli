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
import sounddevice as sd
import json
import os
import logging
import inspect

from datetime import datetime

import copy

import numpy as np
rng = np.random.default_rng(2141)

import onnxruntime as ort # 予め ort-gpu を入れること。 Opset 17 以上が必要
#pip install ort-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-12/pypi/simple/
so = ort.SessionOptions()
so.log_severity_level = 0


from utils import truncate_string
from sample_player_widgets import SamplePlayerWidgets
from sample_slot import AudioSlotPanel, ResultEmbeddingPanel


# s44 現在、計算したスタイル埋め込みを JSON に保存できるが、前回計算したスタイルは自動ではロードされない。
# この辺はもう少し真面目にスタイル編集画面を設計してから考えるべきだろう。

# ただし、現状では前に選択したフォルダが保存されないので、これはなんとかしたい。

# 設定ファイルのパス
SAMPLE_PORTFOLIO_PATH = "./styles/sample_portfolio.json"


# サンプルのロードと埋め込み計算を統括するクラス

class SampleManagerPanel(scrolled.ScrolledPanel):
    def __init__(
        self, 
        parent, 
        backend = None, # SoundControl クラス
        id = -1,
        host = None,
        mute_direct_out: bool = True, # 音声デバイスへの、backend を介さない直接出力をミュートする
        model_device = "cpu",
        harmof0_ckpt: str = None,
        SE_ckpt: str = None,
        initial_sec: float = 8.0, # 埋め込み計算に使う初期化用（ダミー）データの秒数
        ch_map: list = [0], # 入力信号のどのチャンネルを、処理関数に流すかを決めるマップ（下記）
        max_slots: int = 8, # 最大いくつの音声ファイルを保持するか。3 以上だとなぜか UI の反応が鈍くなる
        portfolio_path: str = None, # config を保存するときのファイル名
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(parent, id = id, **kwargs)

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        self.logger.debug("Initializing...")
        self.debug = debug
        self.host = host
        self.backend = backend # このソースを直接評価するときは、backend は下で作る

        # 以下は audio backend インスタンス変数から拾う 
        self.sr_out: float = self.backend.sr_out
        self.sr_proc: float = self.backend.sr_proc
        self.blocksize: int = copy.deepcopy(self.backend.blocksize) # この変数は OutputStream の作り直しに必要
        self.ch_map = ch_map
        # 流すデータの種類に注意。音声チャンクを (blocksize, n_ch) の np.float64 で用意する。

        self.max_slots = max_slots
        self.active_slot_color = wx.Colour(252, 235, 245)
        self.hover_slot_color = wx.Colour(252, 247, 250)
        self.nonactive_slot_color = wx.Colour(255, 255, 255)

        self.sample_portfolio_path = portfolio_path if portfolio_path is not None else SAMPLE_PORTFOLIO_PATH
        self.initial_audio_name: str = "dummy" # 最初の選択中ファイル（ダミー）の名前
        self.file_audio_name: str = "dummy" # 現在の選択中ファイルの名前
        # file_audio_name という変数は AudioSlotPanel にもあり、ファイルを選択するとこちら側に反映される。

        # 再生設定
        self.initial_sec: float = initial_sec # 秒数で定義
        self.max_sec = self.backend.sampler_max_sec
        self.cs = None # 実際に再生する current sample。ただし最初は None で初期化する
        self.cs_name: str = "" # 実際に再生するサンプルの名前。set_sample の内部でのみ書き換えが可能。
        self.cs_sec: float = self.initial_sec  # 単位は seconds であり、まずダミーの秒数で初期化する。
        self.play_position: int = 0 # こちらの単位は秒ではなくサンプル
        self.mute_direct_out = mute_direct_out # 本パネルの OutputStream からデバイスへの直接出力をミュートするか
        self.playing = False
        self.repeat = False
        self.sldr_updatable = False # 再生ポジションのスライダーが更新可能か否か。callback の評価中はロックされる

        #### waveform buffer の初期ダミーデータ作成

        # 再生用（sr_out）とスタイル埋め込みの計算用（16khz）のバッファがある
        self.len_wav_play = int(self.backend.sr_out * self.initial_sec) # 44100 Hz ベースかつ channel last
        self.initial_audio_play = (rng.random((self.len_wav_play, len(self.ch_map)), dtype = np.float32) - 0.5) * 2e-5
        self.len_wav_i16 = int(self.backend.sr_proc * self.initial_sec) # 16000 Hz ベースかつ time last
        self.buf_wav_i16 = (rng.random((len(self.ch_map), self.len_wav_i16), dtype = np.float32) - 0.5) * 2e-5

        ####
        
        # ローカルの音声ファイルを選択するための設定ファイルのロード。
        self.sample_portfolio = self.load_make_sample_styles(self.sample_portfolio_path)

        #### ネットワーク初期化。VC 動作に割り込まないよう、CPU 上で独自に作っておく。

        # 埋め込み計算用のデバイス設定
        if model_device == "cpu":
            self.onnx_provider_list = ['CPUExecutionProvider']
        else:
            self.onnx_provider_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.harmof0_ckpt = harmof0_ckpt
        self.SE_ckpt = SE_ckpt
        
        self.logger.debug("Initializing HarmoF0...")
        # HarmoF0 の変換器の定義。返り値でスペクトログラムも取れる。
        self.sess_HarmoF0 = ort.InferenceSession(
            self.harmof0_ckpt, 
            providers  = self.onnx_provider_list,
            session_options = so,
        )
        
        self.logger.debug("Initializing Style Encoder...")
        # 入力は (batch, 1, dim_spec, n_frame >= 80) の 4D テンソルに変えないと受けられない。
        self.sess_SE = ort.InferenceSession(
            self.SE_ckpt, 
            providers  = self.onnx_provider_list,
        )

        # ダミーデータをネットワークに通して試運転する。この結果は「無音時に対応する埋め込み」となる
        real_F0, activation, real_N, spec_chunk = self.sess_HarmoF0.run(
            ['freq_t', 'act_t', 'energy_t', 'spec'], 
            {"input": self.buf_wav_i16},
        )
        # スペクトログラムを用いてスタイル埋め込みを計算する。ただし末尾（時間）次元が 4 の倍数でないと動作しない
        spec_size_by_four = (spec_chunk.shape[-1] // 4) * 4
        self.style_silent = self.sess_SE.run(
            ['output'], 
            {'input': spec_chunk[:, np.newaxis, 48:, :spec_size_by_four]},
        )[0] # (1, 128)

        # ダミーデータを再生サンプルにロードしておく
        self.set_sample(self.initial_audio_name, self.initial_sec, self.initial_audio_play)

        #### 初期スタイルの作成
        
        # VC に使う style embedding は音声のロード時に計算するが、とりあえず初期値として self.style_silent を使う
        self.style_result = self.style_silent
        self.backend.candidate_style_list[1] = copy.deepcopy(self.style_silent) # audio backend 側にも反映

        # 平均埋め込みの計算に各 embedding を何割ずつ反映するか（いったん 1 固定）
        self.mix_coef_list = [1.0]*self.max_slots # 長さは常に全スロット数分
        # 下に active_coef_list というものも作る。これは「アクティブなスロットの成分だけ抜き出した」可変長のリスト

        #### 音声ファイル用のスロットパネルおよび、最終採用スタイルの表示パネルの初期化

        # 個別の音声ファイルとその埋め込みを管理するスロットを作成。ネットワークの定義後に行うこと。
        # リストの要素として作るのが楽。
        self.slot_list = []
        self.active_slot_index = None
        self.active_list = []
        for i in range(self.max_slots):
            self.slot_list.append(
                AudioSlotPanel(self, manager = self, slot_index = i, style = wx.BORDER_STATIC)
            )
        
        # さらに、オーディオ毎の slot の横に、実際に VC に用いるスタイルベクトルの作成結果を表示するパネル
        self.result_panel = ResultEmbeddingPanel(self, host = self, style = wx.BORDER_NONE)

        #### パネル部品の初期化

        # サンプル音声の再生ボタンやスライダーを集めたパネル。定義は別のソースファイルにある
        self.player_panel = SamplePlayerWidgets(self, debug = self.debug) 
        # スライダーを壊して再作成するメソッドについては、AudioSlotPanel からも触れるように代入しておく
        self.remake_sldr = self.player_panel.remake_sldr 
        # 再生を手動停止するメソッドについても、AudioSlotPanel からも触れるように代入しておく
        self._stop_sound  = self.player_panel._stop_sound 

        # 複数のスロットを垂直配置する sizer 
        self.slots_sizer = wx.BoxSizer(wx.VERTICAL)
        for i in range(self.max_slots):
            self.slots_sizer.Add(self.slot_list[i], proportion = 0, flag = wx.ALL, border = 0) 

        # スロットたちの右側に、最終採用版の埋め込みを配置するための sizer も作る
        self.samples_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.samples_sizer.Add(self.slots_sizer, 0, wx.EXPAND | wx.ALL, border = 5)
        self.samples_sizer.Add(self.result_panel, 1, wx.GROW, border = 5)

        # 統括用の sizer である self.root_sizer を作り、すべての部品を配置していく。
        self.root_sizer = wx.BoxSizer(wx.VERTICAL)
        self.root_sizer.Add(self.player_panel, 0, wx.EXPAND | wx.BOTTOM, border = 5)
        self.root_sizer.Add(self.samples_sizer, 1, wx.EXPAND | wx.ALL, border = 0)
        self.SetSizer(self.root_sizer)
#        self.root_sizer.Fit(self) 
        self.Layout()
        self.root_size = self.root_sizer.GetSize()

        self.SetupScrolling()

        #### ストリームの開始
        
        # 本インスタンスから出力デバイスに音声を直接送る経路部分のストリームを作る。
        # 音声デバイス自体は backend と重複してもいいが、sr, チャンネル数、blocksize, dtype は厳密に一致させる
        # なお set_sample 後でないと output_callback で使う変数が揃わない
        self.output_stream = sd.OutputStream(
            samplerate = self.backend.sr_out, # scan() で作成される
#            device = self.backend.dev_ids_in_use, # scan() で作成される
            # query_devices() で得られる max_input_channels ないし max_output_channels が設定可能な最大値。最小値は 1
            channels = self.backend.n_ch_in_use[2], 
            dtype = 'float32', # 'float32' and 'float64' use [-1.0, +1.0]. 'uint8' is an unsigned 8 bit format
            blocksize = self.backend.blocksize, # 標準は 0  だが、だいたい 128 くらいになる。
#            latency = self.backend.latency, #0.1, # 単位は秒もしくは 'high' 'low'
            callback = self.output_callback, # コールバックは backend 用ではなく、このクラス内で作る専用品
#            extra_settings = self.backend.api_specific_settings,
        )
        self.output_stream.start()

        self.timer = wx.Timer(self) # wx.Timer クラスで、指定間隔での処理を実行する。
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.timer.Start(100) # タイマーは手動で起動する必要がある。単位 ms

        self.logger.debug("Initialized.")

    ####
    
    # self.update は（作成済みの）スライダーおよびボタンの状態を、現在の再生ヘッドに整合させる。
    # さらに、現在 active な slots の平均埋め込みの計算機能も持つ。

    # 【重要】ボタン状態整合を callback 内と update で 2 回呼ぶと segmentation fault になるので update 側だけ

    def update(self, event):
        if self.play_position <= len(self.cs) and self.sldr_updatable:
            if hasattr(self.player_panel, 'pos_sldr') and self.player_panel.pos_sldr is not None:
                try:
                    self._update_pos_sldr() # スライダーの状態整合をここだけに設けることにした。
                except:
                    self.logger.debug(
                        "({}) Cannot update the slider. hasattr: {}, is_none: {}".format(
                            inspect.currentframe().f_code.co_name,
                            hasattr(self.player_panel, 'pos_sldr'), 
                            self.player_panel.pos_sldr is None,
                        )
                    )
            self._update_btn_state() # いずれにせよボタンの play/pause は整合させる
        
        self.Layout()
        self.root_size = self.root_sizer.GetSize() # 現在の部品全体のサイズを格納した変数の値を更新
        
        # Audio file がロードされて埋め込みが計算済みの slot があれば、平均埋め込みを計算して使用値を更新
        self.active_list = []
        self.active_style_list = []
        self.active_coef_list = []
        for i, slot in enumerate(self.slot_list):
            if slot.is_active_checkbox.GetValue() is True:
                self.active_list.append(i)
                self.active_style_list.append(copy.deepcopy(slot.style))
                # 平均埋め込みの計算に各 embedding を何割ずつ反映するか
                self.active_coef_list.append(self.mix_coef_list[i]) 
            
        sum_coef = sum(self.active_coef_list)
        w_quotient = sum_coef if sum_coef > 0 else 1.0
        
        if len(self.active_list) > 1:
            # arr_listの各要素を結合し、新しい軸を追加することで平均を計算しやすくする
            weighted_array_list = [coef * arr / w_quotient for coef, arr in zip(self.active_coef_list, self.active_style_list)]
            combined_array = np.stack(weighted_array_list, axis = 0)
            # 各次元ごとに平均を計算し、新しい (1, 128) の配列を生成したものが、最終的に VC に使う style embedding 
            self.style_result = np.sum(combined_array, axis = 0)
        elif len(self.active_list) > 0:
            self.style_result = self.active_style_list[0]
        else:
            self.style_result = self.style_silent

        # というわけで、self.style_result が VC 用の埋め込みとして backend に送られる
        self.backend.candidate_style_list[1] = copy.deepcopy(self.style_result)


    # 再生状態の変数は変えずに、ボタンの状態を整合させる。もっぱら update から呼ばれる。
    # 最初は _send_sound, _stop_sound からも毎回呼び出していたが、Linux で再生終了処理時に segmentation fault に
    def _update_btn_state(self):
        if hasattr(self.player_panel, 'play_btn') and self.player_panel.play_btn is not None:
            if self.playing is True:
                if self.player_panel.play_btn.GetLabel() != "Pause":
                    self.player_panel.play_btn.SetLabel("Pause") # 再生中："Pause" でなければいけない
            else:
                if self.player_panel.play_btn.GetLabel() != "Play":
                    self.player_panel.play_btn.SetLabel("Play") # 停止中もしくは一時停止中："Play" でなければならない
        self.player_panel.play_btn.Refresh()


    # 再生状態の変数は変えずに、スライダーの状態を整合させる。update から呼ばれる。
    # ただしスライダーの有効長が現在のサンプルに合致している必要がある。
    # 以前は output_callback からも毎回呼び出していたが、Linux で再生時に勝手にスライダーを作り直し？再生が止まった。
    # なお、音声ファイルの切り替え処理の最終段階で、self.player_panel.pos_text が存在しない状況で _update_pos_sldr が
    # 走るタイミングがあり、ここでエラーが必ず出る。なので、try で包んでおくこと。
    def _update_pos_sldr(self):
        if hasattr(self.player_panel, 'pos_sldr') and self.player_panel.pos_sldr is not None:
            try:
                self.player_panel.pos_sldr.SetValue(int(self.play_position / self.backend.sr_out * self.player_panel.pos_sldr_mult))
            except:
                pass
        
        if hasattr(self.player_panel, 'pos_text') and self.player_panel.pos_text is not None:
            try:
                self.player_panel.pos_text.SetLabel(
                    label = "{}\nPlaying: {: >7.2f} / {: >7.2f} sec".format(
                        truncate_string(self.cs_name, max = 29),
                        self.play_position / self.backend.sr_out, 
                        self.cs_sec,
                    ),
                )
            except:
                pass


    # パネルを左クリックして active にする処理。実際には AudioSlotPanel 側から呼び出す。
    # これ現在 text につけているが、パネルにつけられないか？
    def on_panel_click(
        self, 
        event, 
        slot_index: int, 
        unselect: bool = False, # 現在アクティブなものを左クリックすると非アクティブ化するか
        inactivate_other: bool = True, # 同時に 1 スロットしか active にしない
    ):
        if self.active_slot_index == slot_index:
            if unselect is True:
                # 現在アクティブなスロット上のパネルをクリックした場合に、当該を非アクティブ化するオプション
                self.slot_list[slot_index].SetBackgroundColour(self.nonactive_slot_color) # 背景色に戻す
                self.active_slot_index = None
        else:
            if self.playing is False:
                # （別の）サンプルの再生中は操作を受け付けない
                if self.slot_list[slot_index].is_file_loaded is True:
                    # 現在非アクティブなスロット上のパネルをクリックした場合→有効な音声があれば、当該をアクティブ化
                    # オプション次第では、現在のアクティブを非アクティブ化してから処理に入る
                    if inactivate_other and self.active_slot_index is not None:
                        self.slot_list[self.active_slot_index].SetBackgroundColour(self.nonactive_slot_color) 
                    # しかる後、active slot を変更
                    self.active_slot_index = slot_index
                    self.slot_list[slot_index].SetBackgroundColour(self.active_slot_color) # アクティブ色
                    # 再生サンプルにも反映する。いきなり set_sample するのではなく、まず選択ファイルとして登録する。
                    self.file_audio_name = copy.deepcopy(self.slot_list[slot_index].file_audio_name)
                    self.file_sec = copy.deepcopy(self.slot_list[slot_index].file_sec)
                    self.file_audio_play = self.slot_list[slot_index].file_audio_play
                    self.set_sample(
                        self.file_audio_name, 
                        self.file_sec, 
                        self.file_audio_play,
                    )
                    self.remake_sldr() # 再生位置スライダーを再作成する（こいつが必要なため、再生中は処理に入れない）

        for slot in self.slot_list:
            slot.Refresh() # 常に全スロットの再描画を行う


    # 右クリックの場合は、あくまで当該の非アクティブ化に専念する
    def on_panel_right_click(
        self, 
        event, 
        slot_index: int,
    ):
        if self.active_slot_index == slot_index:
            self.slot_list[self.active_slot_index].SetBackgroundColour(self.nonactive_slot_color) # 背景色に戻す
            self.active_slot_index = None # 初期状態は None

        for slot in self.slot_list:
            slot.Refresh() # 常に全スロットの再描画を行う


    # カーソルが侵入したとき色を変える。ただし、Windows でしか動作しない（wxWidgets そのもののバグらしい）。
    def on_panel_hover(self, event, slot_index: int):
        if self.active_slot_index == slot_index:
            self.slot_list[slot_index].SetBackgroundColour(self.hover_slot_color) # アクティブ色
        else:
            self.slot_list[slot_index].SetBackgroundColour(self.hover_slot_color) # アクティブ色
        self.slot_list[slot_index].Refresh()


    # カーソルが離れたとき色を戻す。
    def on_panel_unhover(self, event, slot_index: int):
        if self.active_slot_index == slot_index:
            self.slot_list[slot_index].SetBackgroundColour(self.active_slot_color) # アクティブ色
        else:
            self.slot_list[slot_index].SetBackgroundColour(self.nonactive_slot_color) # 背景色に戻す
        self.slot_list[slot_index].Refresh()

    
    ####

    # 実際に再生する音声を current sample、すなわち OutputStream から見える場所である self.cs 変数にセットする。
    # このメソッドを呼び出すポイントは SampleManagerPanel に 3 つ、AudioSlotPanel に 1 つ存在する。
    def set_sample(
        self,
        file_audio_name, # 引数としてサンプルの名称（ファイル名）、
        file_sec, # 秒数、
        file_audio_data, # そして音声の実体である array (time, n_ch) が必要
    ):
        # 現在セットされているサンプルの名前を反映→再生中にファイルを選択し直した場合の、再反映タイミング決定に使う
        self.cs_name = file_audio_name
        self.cs_sec = file_sec
        # この段階でチャンネル数を OutputStream に合わせている。現在、サラウンド等のマルチチャンネルに非対応
        if self.backend.n_ch_in_use[0] == 2 and file_audio_data.shape[1] == 1:
            self.cs = np.hstack((file_audio_data, file_audio_data)) # モノラル→ステレオ
        elif self.backend.n_ch_in_use[0] == 1 and file_audio_data.shape[1] == 2:
            self.cs = file_audio_data[:, 0] # ステレオ→ ch0 だけ送る
        else:
            self.cs = file_audio_data

        self.play_position = int(0) # いずれにせよ再生ヘッドを 0 戻しする

        self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Set as: '{self.cs_name}' ({self.cs_sec} sec): {self.cs.shape}")
    
    # なお set_sample は再生中に評価してはいけないので、強制的に再生を停止させる仕様になっている。
    
    # 現在 self.play_position の値を書き換える機能は self.set_sample （ゼロ戻しのみ）、
    # output_callback, および self.player_panel._on_pos_sldr_change に存在する

    ####

    # 音声のコールバック関数であるが、スライダーや再生状態の変数の制御ループもここに書く。
    # 最初はスライダーの状態整合処理を独立した update メソッドにして wx.Timer で呼んでいたが、
    # スライダーが一時的に操作できないタイミングで update が走ると segmentation fault になることが判明した

    # 以下の音声出力は「本クラス→出力デバイス」の直接経路。
    # ただし実際には mute_direct_out = True を指定し、こちらからは直接音が鳴らないようにする
    
    def output_callback(self, outdata, frames, time, status):
        # 最初に、バックエンドの blocksize が変わっていたら Stream 自体を作り直す機能
        if self.blocksize != self.backend.blocksize:
            self.remake_stream() # つまり output_callback 自体もいったん生まれ変わる必要がある

        # フラグのリセット
        self.sldr_updatable = False # 送信処理中は position slider の更新を無効化
        reset_play_position = False # 現在のコールバックを処理した瞬間に終端に達する場合のみ True にするフラグ
        # キューに送る用のチャンクを初期化
        chunk0 = np.zeros((self.backend.blocksize, self.backend.n_ch_in_use[0])) # ダミーとして供給する無音サンプル
        chunk = np.zeros_like(chunk0) # これから中身の声を入れる
        
        # 再生ヘッドをまず調べる
        remains = int(len(self.cs) - self.play_position) # まだ送られていないサンプルの長さ（最低 1）
        
        # playing である場合のみ、キュー（to backend）に送るサンプルを作成
        # 加えて mute_direct_out == False の場合のみ、自前の音声出力である outdata に声の入ったサンプルを供給する
        if self.playing:
            if remains <= 0:
                # ヘッドがサンプルの有効長を超えている → 再生できないのでリセット処理に進む
                reset_play_position = True
                self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Head {self.play_position} exceeds the sample length ({remains} samples remaining)")
            elif remains <= self.backend.blocksize:
                # 現ブロックを最後に再生が終わる → 残りサンプルをチャンクに当てはめて送る
                reset_play_position = True
                self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Reached the last chunk (played {self.play_position} samples, {remains} remaining)")
                chunk[:remains, :] = self.cs[-remains:, :]
                self.backend.queueP.put_nowait(copy.deepcopy(chunk)) # VC バックエンドに queueP を通じてデータを送出
            else:
                # まだ 1 ブロック分以上再生すべきサンプルが残っている
                chunk[:] = self.cs[int(self.play_position):int(self.play_position + self.backend.blocksize), :]
                self.backend.queueP.put_nowait(copy.deepcopy(chunk))
                self.play_position += self.backend.blocksize # ブロック分だけヘッドを進める

            # 音声デバイスへの音声チャンクの直接送出
            if self.mute_direct_out is False:
                outdata[:] = copy.deepcopy(chunk[:])
            else:
                outdata[:] = copy.deepcopy(chunk0[:]) # ミュートする場合も無音サンプルの送出は必要

            # 現ブロックを最後に再生が終わる場合のヘッド処理。
            # callback 内でサンプルを再ロードしたりスライダーを弄ったりする機能は、これ以降に組み込むこと
            # self.play_position に値を代入する機能は他にも set_sample, _stop_sound, _on_pos_sldr_change にある
            if reset_play_position is True:
                if self.repeat is False:
                    self.player_panel._stop_sound() # リピートしない場合、再生を停止する。ヘッドも内部で 0 戻しされる
                else:
                    self.play_position = int(0) # リピートする場合、ヘッドの 0 戻しだけ行う
                
        else:
            # 再生中でない場合は常に無音サンプルの送出
            outdata[:] = copy.deepcopy(chunk0[:])

            # 選択中のファイル（ホストにコピーした名前）と再生中のサンプル名が異なったら、再生サンプルに反映させる必要
            # 再生中である場合は原則として、ヘッドが終端に達するこの瞬間にのみ self.sample を走らせる
            
            # 現在、この下のどこかにバグがあり、再生中に UI の左クリックでアクティブサンプルを切り替えると固まる。
            # どうやらスライダーまでは作られているが、スライダーの文字部分を作る段階でエラーを出している。
            # 仕方ないので、再生中はそもそも左クリックを無視する仕様にして強引に対処した。
            
            if self.cs_name != self.file_audio_name:
                self.logger.debug("Change samples from {} to {}".format(self.cs_name, self.file_audio_name))
                self.player_panel._stop_sound() # サンプル切り替え前に、必ず再生を停止する。ヘッドも内部で 0 戻し
                self.set_sample(self.file_audio_name, self.file_sec, self.file_audio_play)
                self.remake_sldr() # 再生位置のスライダーを作り直す
                self._update_btn_state() # ここにボタンの状態整合を手動で入れる必要がある

        # ここでスライダー表示をアップデートしていたが、 segmentation fault の原因になるため削除した。
        self.sldr_updatable = True # 最後に timer による update を可能に戻す


    # さらに、VC 本体側の backend が OutputStream を作り直したときに、こちらも blocksize を変えて再作成する。
    # トリガーとしては self.backend.sr_out と、こちらで保持している変数が一致しない場合。callback 内で判定させる。

    def remake_stream(
        self,
    ):
        self.terminate()
        self.blocksize: int = copy.deepcopy(self.backend.blocksize) # 更新が必要
        self.output_stream = sd.OutputStream(
            samplerate = self.backend.sr_out, # scan() で作成される
            channels = self.backend.n_ch_in_use[2], 
            dtype = 'float32', # 'float32' and 'float64' use [-1.0, +1.0]. 'uint8' is an unsigned 8 bit format
            blocksize = self.backend.blocksize, # 標準は 0  だが、だいたい 128 くらいになる。
            callback = self.output_callback, # コールバックは backend ではなく、このクラス内で作る専用品
        )
        self.output_stream.start()
    
    
    # ストリームを停止させる処理
    def terminate(
        self,
    ) -> None:
        self.output_stream.stop()
        self.output_stream.close() 


    # 指定した名称の config ファイルがない場合、作成する
    def load_make_sample_styles(
        self,
        x,
    ):
        if os.path.exists(x):
            with open(x, "r") as f:
                return json.load(f)
        else:
            config_list = []
            for i in range(self.max_slots):
                slot_config = {
                    "slot_index": i,
                    "last_selected_file": None # 音声ファイル相対パス。なお OS ごとに区切り文字が変わるので注意
                }
                config_list.append(slot_config)
            return config_list

    # 本当はさらに、保存した self.sample_portfolio から次回アプリ起動時に状態を復元したい。でも
    # 面倒くさいのでまだ実装していない。
    
    # なぜ面倒かというとファイルの移動・削除への対処を考慮する必要があり、しかもロード動作はかなり重い。
    # なのでアプリの起動後に、バックグラウンドでロードさせる等の方策を検討しなければならない
    # さらに、環境によっては 3 つ以上のサンプルをロードするとアプリが不安定になる。
    # また、ロード済みのサンプルを unload する処理をまだ実装していない
    
    # なので、自動で復元するのは諦め、ファイル選択のダイアログ初期値を前回ファイルにするのが落としどころでは


####

# 以下は、player.py をスクリプトで実行するときだけ利用される独自の audio backend

import queue

class Backend(wx.Panel):
    def __init__(
        self, 
        parent, 
        id = -1,
        **kwargs,
    ):
        super().__init__(parent, id = id, **kwargs)

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.logger.debug("Initializing audio stream backend...")

        self.mic_mix: float = 0.5 # マイクからくる音と sample player からくる音のミックス比（1.0 でマイクのみ）
        self.sample_amp: float = 1.0 # sample player からくる音の音量を絞る
        self.blocksize = 2048*3
        self.queueP = queue.Queue() # P は sample player -> InputStream のミックス用信号
        self.sr_out: float = 44100 # 実は int だと librosa.resample でバグる
        self.sr_proc: float = 16000
        self.n_ch_in_use = [2, 1, 2]
        self.VC_threshold = -40.0
        
        self.output_stream = sd.OutputStream(
            samplerate = self.sr_out, # scan() で作成される
            channels = self.n_ch_in_use[2], 
            dtype = 'float32', # 'float32' and 'float64' use [-1.0, +1.0]. 'uint8' is an unsigned 8 bit format
            blocksize = self.blocksize, # 標準は 0  だが、だいたい 128 くらいになる。
            latency = "low", # 単位は秒もしくは 'high' 'low'
            callback = self.callback, # コールバックは backend ではなく、このクラス内で作る専用品
        )
        self.output_stream.start() # 

    def callback(
        self,
        outdata, 
        frames, 
        time, 
        status,
    ):
        if status:
            self.logger.debug(f"{status}")
        data_p = np.zeros((frames, self.n_ch_in_use[2])) # self.n_ch_in_use[0] が入力 ch 数
        if self.queueP.empty() is False:
            side_wav = self.queueP.get() # ここは wait が必要。でないとこちらの反応が速すぎてサンプルが落ちる
            for i in list(range(self.n_ch_in_use[2])):
                data_p[:, i] = side_wav[: , i] * self.sample_amp # n_ch_in_use[0] = 2 だったら i = 0, 1 の 2 チャンネル分
        outdata[:] = data_p[:]


# 以下は、単体テスト用に player.py をスクリプトで実行するときだけ実行される。

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title = "Style Calculator", size = (1280, 800))

        self.pos = self.GetScreenPosition()
        self.size = self.GetSize() # フレーム自身のサイズを変数に格納しておく

        self.backend = Backend(self)
        self.panel = SampleManagerPanel(self, self.backend, debug = True)
        self.panel.SetSize(self.size)

        # ステータスバーを作成
        self.sb = self.CreateStatusBar(number = 4)
        self.sb.SetStatusText('Ready', i = 0) # ステータスバーに文字を表示させる
        self.sb_size = self.sb.GetSize() # ステータスバーの高さを変数に格納しておく

        self.Show()

        self.timer = wx.Timer(self) # wx.Timer クラスで、指定間隔での処理を実行する。
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.timer.Start(37) # タイマーは手動で起動する必要がある。単位 ms

    def update(self, event):
        # まずパネルの高さと位置を取得。
        self.size = self.GetSize()
        self.pos = self.GetScreenPosition()
        self.panel_pos = self.panel.GetScreenPosition()
        self.sb_size = self.sb.GetSize()

        # self.pos, self.panel_pos は現在のディスプレイ内での絶対位置。
        # しかも、パネル開始位置はタイトルバーの高さ分（37 px）下にずれるらしい。
        # 内部の各パネルの高さを動的に設定するとき、この高さを差し引かないといけない。
        # さらに Windows では panel の開始位置が Frame よりも右にズレる（Linux では同じ）ため、
        # Frame ではなく status bar の幅を基準にして panel の幅を定義しないと、右側の垂直スクロールバーが隠れる。
        self.panel.SetSize((self.sb_size[0], self.size[1] - self.sb_size[1] - (self.panel_pos[1] - self.pos[1])))
        # ここで self.panel.SetupScrolling() として再セットアップしてはいけない。スクロール位置がリセットされてしまう
        self.Layout()

        self.sb.SetStatusText(
            f"Frame pos: {self.pos}, panel pos: {self.panel_pos}", 
            i = 0,
        )
        self.sb.SetStatusText(
            f"Frame: {self.GetSize()}, panel: {self.panel.GetSize()}, root: {self.panel.root_size}, sb: {self.sb_size}", 
            i = 1,
        )
        self.sb.SetStatusText(
            f"Playing '{self.panel.cs_name}', Selected audio: '{self.panel.file_audio_name}'",
            i = 2,
        )
        self.sb.SetStatusText(
            f"Active slots: {self.panel.active_list}",
            i = 3,
        )

        pass


if __name__ == "__main__":

    now = datetime.now()
    log_name = f"./logs/sample_style_test.log" 
    logging.basicConfig(
        filename = log_name, 
        level = logging.DEBUG, 
        format = '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    )
    
    # matplotlibのログレベルを設定（適宜）
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    
    app = wx.App(False)
    frame = MainFrame()
    app.MainLoop()



