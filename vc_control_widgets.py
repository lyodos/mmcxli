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

import soundfile as sf
import os
import math

import copy

import numpy as np

import librosa
from pydub import AudioSegment

import threading

import logging
import inspect


from audio_level_meter import InputLevelMeterPanel

####

# VC の設定パネルは基本設定（上に常時表示する部分）と、高度なバッファ設定（タブ内に表示する部分）がある。
# どちらもこのソース内のクラスとして用意されている。

# 以下は常時表示する部分の部品定義で、次に定義する FloatPanel の要素として呼び出される。

class BasicVCSettingsPanel(wx.Panel):
    def __init__(
        self, 
        parent, 
        backend = None, # SoundControl クラスのインスタンスをここに指定。先にバックエンドが初期化されている必要がある。
        id = -1,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(parent, id = id, **kwargs)
        self.debug = debug
        self.sc = backend

        self.b_color = wx.Colour(255, 255, 255) 
        
        #### VC 制御

        # VC の閾値を制御するスライダー。初期状態は -40 dBFS
        self.VC_threshold_sldr = wx.Slider(
            self, 
            value = round(self.sc.VC_threshold), # -40
            minValue = -80,
            maxValue = 0,
            size = (240, -1),
            style = wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS,
        )
        self.VC_threshold_sldr.SetTickFreq(20) # 20 dBFS
        self.VC_threshold_sldr.Bind(wx.EVT_SLIDER, self.on_VC_threshold_sldr_change)
        # VC_threshold には周囲に囲みを作り、タイトルラベルを付ける
        self.VC_threshold_box = wx.StaticBox(self, wx.ID_ANY, 'VC threshold (dBFS)')
        self.VC_threshold_sizer = wx.StaticBoxSizer(self.VC_threshold_box, wx.HORIZONTAL)
        self.VC_threshold_sizer.Add(self.VC_threshold_sldr, proportion = 0, flag = wx.EXPAND | wx.ALL, border = 5)

        # マイク音声にプリアンプを掛けるスライダー
        self.mic_amp_coef: float = 1 # スライダーで 1 dB ずつ動かせるように調整
        self.mic_amp_default = math.ceil(self.sc.mic_amp / self.mic_amp_coef)
        self.mic_amp_sldr = wx.Slider(
            self, 
            value = self.mic_amp_default,
            minValue = self.sc.efx_control.mic_amp_range[0], 
            maxValue = self.sc.efx_control.mic_amp_range[1],
            size = (240, -1),
            style = wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS,
        )
        self.mic_amp_sldr.SetTickFreq(10) # 10 dB
        for value in [-6, -3, -1, 0, 1, 3, 6]:
            self.mic_amp_sldr.SetTick(value)
        self.mic_amp_sldr.Bind(wx.EVT_SLIDER, self.on_mic_amp_sldr_change)
        self.mic_amp_sldr.Bind(wx.EVT_RIGHT_DOWN, self.on_mic_amp_sldr_right)
        self.mic_amp_box = wx.StaticBox(self, wx.ID_ANY, 'Mic preamp (dB)')
        self.mic_amp_sizer = wx.StaticBoxSizer(self.mic_amp_box, wx.HORIZONTAL)
        self.mic_amp_sizer.Add(self.mic_amp_sldr, proportion = 0, flag = wx.EXPAND | wx.ALL, border = 5)

        # VC 音高を制御するスライダー。単位は半音で、デフォルトでは上下それぞれ 15 半音までサポート
        # 先に、想定レンジを外れていないかチェックする
        if self.sc.efx_control.pitch_shift > self.sc.efx_control.pitch_range[1]:
            pitch_shift_init_value = int(self.sc.efx_control.pitch_range[1])
        elif self.sc.efx_control.pitch_shift < self.sc.efx_control.pitch_range[0]:
            pitch_shift_init_value = int(self.sc.efx_control.pitch_range[0])
        else:
            pitch_shift_init_value = round(self.sc.efx_control.pitch_shift)
        self.pitch_shift_sldr = wx.Slider(
            self, 
            value = pitch_shift_init_value,
            minValue = self.sc.efx_control.pitch_range[0], 
            maxValue = self.sc.efx_control.pitch_range[1],
            size = (270, -1),
            style = wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS,
        )
        for value in [-12, -9, -6, -3, -1, 0, 1, 3, 6, 9, 12]:
            self.pitch_shift_sldr.SetTick(value)
        self.pitch_shift_sldr.Bind(wx.EVT_SLIDER, self.on_pitch_shift_sldr_change)
        self.pitch_shift_sldr.Bind(wx.EVT_RIGHT_DOWN, self.on_pitch_shift_sldr_right)
        self.pitch_shift_box = wx.StaticBox(self, wx.ID_ANY, 'Pitch shift (semitone)')
        self.pitch_shift_sizer = wx.StaticBoxSizer(self.pitch_shift_box, wx.HORIZONTAL)
        self.pitch_shift_sizer.Add(self.pitch_shift_sldr, proportion = 0, flag = wx.EXPAND | wx.ALL, border = 5)

        #### スイッチ

        # 自分自身の style embedding を使うか、サンプル音声由来の style embedding を使うか
        self.switch_style_array = ('to sample', 'from myself')
        self.switch_style_rbx = wx.RadioBox(
            self, 
            wx.ID_ANY, 
            'Target style',
            choices = self.switch_style_array, 
            style = wx.RA_VERTICAL,
        )
        self.switch_style_rbx.Bind(wx.EVT_RADIOBOX, self.on_switch_style)
        self.switch_style_rbx.SetSelection(int(self.sc.efx_control.auto_encode)) # 初期状態は "Sample"
        self.switch_style_sizer = wx.BoxSizer(wx.VERTICAL)
        self.switch_style_sizer.Add(self.switch_style_rbx)

        # 相対音高（harmoF0 + shift）を使うか、絶対音高（ターゲットstyleから推測）を使うか
        self.pitch_mode_array = ('source', 'target')
        self.pitch_mode_rbx = wx.RadioBox(
            self, 
            wx.ID_ANY, 
            'Pitch mode',
            choices = self.pitch_mode_array, 
            style = wx.RA_VERTICAL,
        )
        self.pitch_mode_rbx.Bind(wx.EVT_RADIOBOX, self.on_pitch_mode)
        self.pitch_mode_rbx.SetSelection(int(self.sc.efx_control.absolute_pitch)) # 初期状態は "absolute"
        self.pitch_mode_sizer = wx.BoxSizer(wx.VERTICAL)
        self.pitch_mode_sizer.Add(self.pitch_mode_rbx)

        # 音量の元ネタにソーススペクトログラムの実測 energy を使うか、f0n_predictor による推定値を使うか
        self.energy_mode_array = ('same as source', 'estimate')
        self.energy_mode_rbx = wx.RadioBox(
            self, 
            wx.ID_ANY, 
            'Energy mode',
            choices = self.energy_mode_array, 
            style = wx.RA_VERTICAL,
        )
        self.energy_mode_rbx.Bind(wx.EVT_RADIOBOX, self.on_energy_mode)
        self.energy_mode_rbx.SetSelection(int(self.sc.efx_control.estimate_energy)) # 初期状態は "absolute"
        self.energy_mode_sizer = wx.BoxSizer(wx.VERTICAL)
        self.energy_mode_sizer.Add(self.energy_mode_rbx)

        # 変換先話者スタイルの選択
        self.style_mode_array = ('2-dim', 'Sample', 'Full-128')
        self.style_mode_rbx = wx.RadioBox(
            self, 
            wx.ID_ANY, 
            'Style control',
            choices = self.style_mode_array, 
            style = wx.RA_VERTICAL,
        )
        self.style_mode_rbx.Bind(wx.EVT_RADIOBOX, self.on_style_mode)
        self.style_mode_rbx.SetSelection(int(self.sc.style_mode))
        self.style_mode_sizer = wx.BoxSizer(wx.VERTICAL)
        self.style_mode_sizer.Add(self.style_mode_rbx)

        #### フェーダー類を集めた sizer を作る
        self.faders_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.faders_sizer.Add(self.mic_amp_sizer, proportion = 0, flag = wx.ALL, border = 5)
        self.faders_sizer.Add(self.VC_threshold_sizer, proportion = 0, flag = wx.ALL, border = 5)
        self.faders_sizer.Add(self.switch_style_sizer, proportion = 0, flag = wx.ALL, border = 5)
        self.faders_sizer.Add(self.pitch_mode_sizer, proportion = 0, flag = wx.ALL, border = 5)
        self.faders_sizer.Add(self.pitch_shift_sizer, proportion = 0, flag = wx.ALL, border = 5)
        self.faders_sizer.Add(self.energy_mode_sizer, proportion = 0, flag = wx.ALL, border = 5)
        self.faders_sizer.Add(self.style_mode_sizer, proportion = 0, flag = wx.ALL, border = 5)

        # 統括 sizer を作る。ここは VERTICAL でないと AddStretchSpacer が効かない
        self.root_sizer = wx.BoxSizer(wx.VERTICAL)
        self.root_sizer.Add(self.faders_sizer, 0, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(self.root_sizer)
        self.root_sizer.Fit(self) 
        self.Layout()

        # 背景色のセット
        self.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_MENU))

####

    # ノイズ閾値を変えるスライダーの値の変更
    def on_VC_threshold_sldr_change(self, event):
        if hasattr(self, 'VC_threshold_sldr') and self.VC_threshold_sldr is not None:
            self.sc.VC_threshold = float(self.VC_threshold_sldr.GetValue())
            self.sc.update_vc_config("VC_threshold", float(self.sc.VC_threshold), save = False)
        event.Skip(False)


    # マイク音声にプリアンプを掛けるスライダーの値の変更
    def on_mic_amp_sldr_change(self, event):
        if hasattr(self, 'mic_amp_sldr') and self.mic_amp_sldr is not None:
            self.sc.mic_amp = float(self.mic_amp_sldr.GetValue())
            self.sc.update_vc_config("mic_amp", round(self.sc.mic_amp), save = False)
        event.Skip(False)

    # マイク音声にプリアンプを掛けるスライダー上での右クリックで、原点（0）に値を戻す
    def on_mic_amp_sldr_right(self, event):
        # Check if the right click happened within the slider's screen rectangle
        slider_rect = self.mic_amp_sldr.GetScreenRect()
        mouse_pos = wx.GetMousePosition()
        if slider_rect.Contains(mouse_pos):
            self.sc.mic_amp = 0.0
            self.mic_amp_sldr.SetValue(0)
        event.Skip(False)


    # ピッチシフトを変えるスライダーの値の変更
    def on_pitch_shift_sldr_change(self, event):
        if hasattr(self, 'pitch_shift_sldr') and self.pitch_shift_sldr is not None:
            self.sc.efx_control.pitch_shift = self.pitch_shift_sldr.GetValue()
            self.sc.update_vc_config("pitch_shift", self.sc.efx_control.pitch_shift, save = False)
        event.Skip(False)

    # ピッチシフトを変えるスライダー上での右クリックで、原点（0）に値を戻す
    # ただし、現在は「スライダー周辺領域」でしか動作せず、スライダー内では on_pitch_shift_sldr_change が発火してしまう
    def on_pitch_shift_sldr_right(self, event):
        # Check if the right click happened within the slider's screen rectangle
        slider_rect = self.pitch_shift_sldr.GetScreenRect()
        mouse_pos = wx.GetMousePosition()
        if slider_rect.Contains(mouse_pos):
            self.sc.efx_control.pitch_shift = 0
            self.pitch_shift_sldr.SetValue(0)
        event.Skip(False)


    # 自分自身の style embedding を使うか、サンプル音声由来の style embedding を使うか
    def on_switch_style(self, event):
        self.sc.efx_control.auto_encode = bool(self.switch_style_rbx.GetSelection())
        self.sc.update_vc_config("auto_encode", bool(self.sc.efx_control.auto_encode), save = False)
        event.Skip(False)

    # 相対音高（harmoF0 + shift）を使うか、絶対音高（ターゲットstyleから推測）を使うか
    def on_pitch_mode(self, event):
        self.sc.efx_control.absolute_pitch = bool(self.pitch_mode_rbx.GetSelection())
        self.sc.update_vc_config("absolute_pitch", bool(self.sc.efx_control.absolute_pitch), save = False)
        event.Skip(False)

    # 音量の元ネタにソーススペクトログラムの実測 energy を使うか、f0n_predictor による推定値を使うか
    def on_energy_mode(self, event):
        self.sc.efx_control.estimate_energy = bool(self.energy_mode_rbx.GetSelection())
        self.sc.update_vc_config("estimate_energy", bool(self.sc.efx_control.estimate_energy), save = False)
        event.Skip(False)

    # 変換先話者スタイルの選択
    def on_style_mode(self, event):
        self.sc.style_mode = int(self.style_mode_rbx.GetSelection())
        self.sc.update_vc_config("style_mode", self.sc.style_mode, save = False)
        event.Skip(False)


####

# フロート表示する部分は、上の情報表示部分と下の基本設定部品群（BasicVCSettingsPanel）に分かれる

class FloatPanel(wx.Panel):
    def __init__(
        self, 
        parent, 
        backend = None, # SoundControl クラスのインスタンスをここに指定。先にバックエンドが初期化されている必要がある。
        id = -1,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(parent, id = id, **kwargs)
        self.debug = debug
        self.sc = backend

        self.b_color = [wx.Colour(105, 108, 113), wx.Colour(79, 118, 182), wx.Colour(171, 52, 77), wx.Colour(252, 188, 0)] 
        # VC OFF, VC ON, VC ON(timeout), offline conversion
        self.status_text_color = [wx.Colour(189, 198, 192), wx.Colour(222, 220, 225)] # VC OFF、VC ON
        self.mute_btn_color = [wx.Colour(255, 255, 255), wx.Colour(242, 249, 167)] # 通常時、ミュート中
        
        # 現在の VC 待機状態
        self.status_text = wx.StaticText(self, label = "(THROUGH)")
        self.status_font = wx.Font(22, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        self.status_text.SetFont(self.status_font)
        self.status_text.SetForegroundColour(wx.Colour(255, 50, 0))
        self.status_text.SetMinSize((440, -1))
        self.status_text.SetMaxSize((440, -1))
        
        # 入力信号レベルと VC 閾値
        self.level_panel = InputLevelMeterPanel(self, backend = self.sc)
        
        # とにかく出力音声をミュートする
        self.mute_font = wx.Font(16, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        self.mute_btn = wx.ToggleButton(self, wx.ID_ANY, 'Mute')
        self.mute_btn.SetMinSize((160, 50)) # 最小幅を設定
        self.mute_btn.SetMinSize((160, 50)) # 最大幅を設定
        self.mute_btn.Bind(wx.EVT_TOGGLEBUTTON, self.switch_mute)
        self.mute_btn.SetFont(self.mute_font)
        self.mute_btn.SetBackgroundColour(self.mute_btn_color[int(self.sc.mute)]) 

        # 現在の遅延量を文字表示する
        self.delay_num_text = wx.StaticText(self, label = f"{len(self.sc.queueA): >4.0f}", style = wx.ALIGN_CENTER)
        self.delay_font = wx.Font(18, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        self.delay_num_text.SetFont(self.delay_font)
        self.delay_num_text.SetForegroundColour(self.status_text_color[1]) 
        self.delay_text = wx.StaticText(self, label = "blocks delay", style = wx.ALIGN_CENTER)
        self.delay_font = wx.Font(8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        self.delay_text.SetFont(self.delay_font)
        self.delay_text.SetForegroundColour(self.status_text_color[1]) 
        
        self.delay_sizer = wx.BoxSizer(wx.VERTICAL)
        self.delay_sizer.Add(self.delay_num_text, 0, wx.ALL | wx.RIGHT, 0)
        self.delay_sizer.Add(self.delay_text, 0, wx.ALL | wx.RIGHT, 0)
#        self.delay_sizer.SetMinSize((100, -1)) # 最小幅を設定
#        self.delay_sizer.SetMinSize((100, -1)) # 最大幅を設定

        # 現時点のキューをリリースして遅延を一気に取り戻す
        self.release_queue_btn = wx.Button(self, wx.ID_ANY, 'Skip delay')
        self.release_queue_btn.SetMinSize((80, 50)) # 最小幅を設定
        self.release_queue_btn.SetMinSize((80, 50)) # 最大幅を設定
        self.release_queue_btn.Bind(wx.EVT_BUTTON, self.on_release_queue)
        
        # 緊急対応用のボタンを集めた sizer を作る
        self.panic_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.panic_sizer.Add(self.status_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.panic_sizer.Add(self.level_panel, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.panic_sizer.AddStretchSpacer(1)
        self.panic_sizer.Add(self.mute_btn, proportion = 0, flag = wx.ALL | wx.ALIGN_CENTER_VERTICAL, border = 5)
        self.panic_sizer.AddStretchSpacer(1)
        self.panic_sizer.Add(self.delay_sizer, proportion = 0, flag = wx.ALL | wx.ALIGN_CENTER_VERTICAL, border = 0)
        self.panic_sizer.Add(self.release_queue_btn, proportion = 0, flag = wx.ALL | wx.ALIGN_CENTER_VERTICAL, border = 5)

        # 上で作った 基本 VC 設定のパネルを作って配置
        self.basic_vc_settings_panel = BasicVCSettingsPanel(self, self.sc)

        # 統括 sizer を作る。ここは VERTICAL でないと AddStretchSpacer が効かない
        self.root_sizer = wx.BoxSizer(wx.VERTICAL)
        self.root_sizer.Add(self.panic_sizer, 0, wx.EXPAND | wx.ALL, 0)
        self.root_sizer.Add(self.basic_vc_settings_panel, 0, wx.EXPAND | wx.ALL, 0)
        self.SetSizer(self.root_sizer)
        self.root_sizer.Fit(self) 
        self.Layout()

        # フロートパネルにファイルをドラッグアンドドロップした時の挙動。オフライン VC 処理を掛ける
        self.SetDropTarget(AudioDropTarget(self.sc, host = self, max_sec = self.sc.offline_max_sec))
        
        # 背景色のセット
        self.SetBackgroundColour(self.b_color[int(self.sc.vc_now)]) 

        self.timer = wx.Timer(self) # wx.Timer クラスで、指定間隔での処理を実行する。
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.timer.Start(50) # タイマーは手動で起動する必要がある。単位 ms


    # 消音ボタンを押した時のイベント
    def switch_mute(self, event):
        if self.sc.mute:
            self.sc.mute = False
            self.mute_btn.SetLabel("Mute")
        else:
            self.sc.mute = True
            self.mute_btn.SetLabel("ミュートを解除")
        self.mute_btn.SetBackgroundColour(self.mute_btn_color[int(self.sc.mute)]) 


    def update(self, event):

        # フロー画面の背景色は、VC の ON/OFF および時間超過の警告に応じて切り替える
        if self.sc.offline_conversion_now == True:
            self.status_text.SetForegroundColour(wx.Colour(23, 23, 27)) 
            self.SetBackgroundColour(self.b_color[3]) 
        else:
            self.status_text.SetForegroundColour(self.status_text_color[int(self.sc.vc_now)]) 
            if self.sc.efx_control.vc_lap / (1000 * self.sc.blocksize / self.sc.sr_out) > 1:
                self.SetBackgroundColour(self.b_color[2]) 
                if self.sc.vc_now:
                    self.status_text.SetLabel("VC Running (timeout)")
            else:
                self.SetBackgroundColour(self.b_color[int(self.sc.vc_now)]) 
                if self.sc.vc_now:
                    self.status_text.SetLabel("VC Running")
                else:
                    self.status_text.SetLabel("(THROUGH)")

        # ミュート状態の更新
        if self.sc.mute is True:
            if self.mute_btn.GetLabel() != "ミュートを解除":
                self.mute_btn.SetLabel("ミュートを解除")
        else:
            if self.mute_btn.GetLabel() != "Mute":
                self.mute_btn.SetLabel("Mute")
        
        self.delay_num_text.SetLabel(f"{len(self.sc.queueA)}")

        self.Refresh()
        self.Layout()


    # 現時点のキューをリリースして遅延を一気に取り戻す
    def on_release_queue(self, event):
        while len(self.sc.queueA) > 0:
            _, _ = self.sc.queueA.popleft()


####

# ファイルをドラッグ＆ドロップしたときに、オフラインで VC を掛ける。現在のターゲットスタイルに従う

# なお、ファイルが一定時間以上の場合に制限を掛ける機能は、現在冒頭だけ読む乱暴なものである。
# 本当は無音部を検知して max_sec 以下の chunk に切り分けて処理すべきなのだが、
# 面倒くさいので未実装である

class AudioDropTarget(wx.FileDropTarget):
    def __init__(
        self, 
        backend,
        host,
        max_sec: float = 16.0, # これよりも長い音声は冒頭のみ読み込んで処理される
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.sc = backend # audio backend
        self.host = host # Float Panel
        self.max_sec = max_sec
        self.os_sep = os.path.sep # 現在の OS で有効なパス区切り文字

    def OnDropFiles(self, x, y, filenames):
        self.success = False
        self.skip_orig = copy.deepcopy(self.sc.skip_always)

        # 背景を変えるギミックのために、変換の非同期実行が必要
        self.sc.offline_conversion_now = True
        # ファイルの処理を非同期で行う
        threading.Thread(target = self.process, args = (filenames,)).start()
        return True # TypeError: invalid result from AudioDropTarget.OnDropFiles(), a 'bool' is expected not 'NoneType'


    def process(self, filenames):
        for i, file_path in enumerate(filenames):
            try:
                if file_path.lower().endswith(('.wav', '.ogg', '.mp3', '.m4a', '.flac', '.opus')):
                    self.sc.skip_always = True # 一時的にリアルタイム VC を迂回し InferenceSession のリソースを優先確保
                    self.host.status_text.SetLabel(f"Converting [{i+1}/{len(filenames)}]")
                    
                    # OS 依存のパス区切り文字を修正。なおドライブレター等はうまく処理できない
                    offline_load_path = file_path.replace('/', self.os_sep).replace('\\', self.os_sep) 
                    self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Offline VC of '{offline_load_path}' is running...")

                    # 入力ファイルのあるフォルダにサブディレクトリを作り、変換後のファイルを同名で保存
                    load_dir_name = os.path.dirname(offline_load_path)
                    load_file_name = os.path.basename(offline_load_path)
                    save_dir_name = os.path.join(load_dir_name, "converted")
                    os.makedirs(save_dir_name, exist_ok = True)

                    # soundfile をロードしたら 2 種類のバッファに保存する。リサンプリングは time last が必要。
                    # sf は返り値は channel last だが、librosa は timel last なので注意
                    file_audio_raw, file_orig_fs = self.read_audio(
                        file_path, 
                        max_sec = self.max_sec,
                    ) # (time, channel)

                    # embedding 計算用のバッファに保存 self.host.sr_proc (16000 Hz) で作る
                    file_audio_16 = librosa.resample(
                        file_audio_raw,  # time last の状態で音声をリサンプルする
                        orig_sr = file_orig_fs, 
                        target_sr = self.sc.sr_proc,
                        res_type = "polyphase",
                        axis = -1,
                    ) # こちらは time last で保持する

                    # 出力は self.sc.sr_dec こと 24k になる。
                    converted_audio_24 = self.sc.efx_control.convert_offline(file_audio_16)

                    # 書き出す音声ファイルのパスを設定
                    output_file_path = os.path.join(save_dir_name, os.path.splitext(load_file_name)[0] + ".wav")
                    self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Writing '{output_file_path}'...")
                    sf.write(output_file_path, converted_audio_24.T, self.sc.efx_control.sr_dec, subtype = 'FLOAT')

                    # もし self.max_sec より長い音声が来たら、エラー処理が必要
                    self.success = True
                else:
                    wx.MessageBox(
                        "Unsupported file format. Please drop an audio file ('.wav', '.ogg', '.mp3', '.m4a', '.flac', '.opus').", 
                        "Error", 
                        wx.OK | wx.ICON_ERROR,
                    )
                    self.success = False
            except Exception as e:
                wx.MessageBox(f"An error occurred while dropping files: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
                self.success = False

        self.sc.offline_conversion_now = False
        self.sc.skip_always = self.skip_orig
        
        return self.success


    def read_audio(
        self, 
        file_path: str, 
        max_sec: float = 16.0,
    ):
        _, file_extension = os.path.splitext(file_path) # 拡張子はドット . を含む

        if file_extension.lower() in ['.wav', '.flac', '.ogg']:
            audio, sr = librosa.load(file_path, sr = None, mono = True, duration = max_sec)
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]
            return audio.astype('float32'), int(sr)

        elif file_extension.lower() in ['.mp3', '.m4a', '.opus']:
            audio = AudioSegment.from_file(file_path)
            sr = audio.frame_rate
            audio = audio.set_channels(1) if audio.channels == 2 else audio  # Ensure mono audio
            audio = audio.set_sample_width(2)    # Set sample width to 16-bit (adjust as needed)
        
            if max_sec < len(audio) / 1000:
                audio = audio[:max_sec * 1000] # 操作はミリ秒単位で実施する必要がある

            audio = audio.get_array_of_samples()
            audio = np.array(audio, dtype = 'float32') / 32768.0  # Normalize to float32
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]

            return audio, int(sr)  # Assume 16kHz sample rate for pydub
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
