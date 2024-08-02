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

from utils import truncate_string

#### 再生ボタン類とスライダー制御をまとめた部品クラス。

class SamplePlayerWidgets(wx.Panel):
    def __init__(
        self,
        parent,
        id = -1,
        debug: bool = False,
    ):
        super().__init__(parent, id = id)
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        self.host = parent
        self.debug = debug

        self.pos_sldr_mult = 1000 # wx のスライダーは int しか取れないので、移動量を水増しする必要がある。

        # 現在の duration に基づいて slider を作成ないし再作成する。
        self._make_btns() # 再生ボタン等を作る
        self._make_pos_sldr_and_text() # スライダーと文字表示を集めた self.player_sldr_sizer も作られる
        # play_position は self.host の値を弄るので注意。
        self._make_sample_amp_sldr()

        # 再生制御用の部品の集合をまとめた sizer を作る
        self.root_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.root_sizer.Add(self.player_btns_sizer, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 5)
        self.root_sizer.Add(self.player_sldr_sizer, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 0) 
        self.root_sizer.AddStretchSpacer(1) # なお self.root_sizer.Add するとき EXPAND しないとスペースが作られない
        self.root_sizer.Add(self.sample_amp_sizer, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.SetBackgroundColour(self.host.nonactive_slot_color) 
        self.SetSizer(self.root_sizer)

        self.logger.debug(f"({inspect.currentframe().f_back.f_code.co_name}) Initialized.")


    def _make_sample_amp_sldr(self):
        # sample player 再生音声の音量を制御するスライダー。初期状態は 1.0 = 100%
        self.sample_sldr_mult = 100 # wx のスライダーは int しか取れないので、移動量を水増しする必要がある。
        self.sample_amp_sldr = wx.Slider(
            self, 
            value = round(self.host.backend.sample_amp * self.sample_sldr_mult),
            minValue = 0, 
            maxValue = self.sample_sldr_mult, # 内部状態は 0--100 の int だが host.backend 側の値は 0--1 の float
            size = (150, -1),
            style = wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS,
        )
        self.sample_amp_sldr.SetTickFreq(round(self.sample_sldr_mult / 4))
        self.sample_amp_sldr.Bind(wx.EVT_SLIDER, self.on_sample_amp_sldr_change)
        # 周囲に囲みを作り、タイトルラベルを付ける
        self.sample_amp_box = wx.StaticBox(self, wx.ID_ANY, 'Sample player volume (%)')
        self.sample_amp_sizer = wx.StaticBoxSizer(self.sample_amp_box, wx.HORIZONTAL)
        self.sample_amp_sizer.Add(self.sample_amp_sldr, proportion = 0, flag = wx.EXPAND | wx.ALL, border = 5)


    def _make_btns(self):
        # 再生ボタン
        self.play_btn = wx.Button(self, label = "Play")
        self.play_btn.SetMinSize((75, 50)) # 最小幅を設定
        self.play_btn.SetMaxSize((75, 50)) # 最大幅を設定
        self.play_btn.Bind(wx.EVT_BUTTON, self.send_sound)
        # 停止ボタン
        self.stop_btn = wx.Button(self, label = "Stop")
        self.stop_btn.SetMinSize((75, 50)) # 最小幅を設定
        self.stop_btn.SetMaxSize((75, 50)) # 最大幅を設定
        self.stop_btn.Bind(wx.EVT_BUTTON, self.stop_sound)
        # リピート再生するか否かのトグルボタン。本当はトグルスイッチが欲しいが、wx には標準で入っていないらしい
        self.repeat_btn = wx.ToggleButton(self, -1, label = "Repeat")
        self.repeat_btn.SetMinSize((75, 40)) # 最小幅を設定
        self.repeat_btn.SetMaxSize((75, 40)) # 最大幅を設定
        self.repeat_btn.Bind(wx.EVT_TOGGLEBUTTON, self.switch_repeat)
        # ボタンを集めた sizer
        self.player_btns_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.player_btns_sizer.Add(self.play_btn, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5) 
        self.player_btns_sizer.Add(self.stop_btn, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5) 
        self.player_btns_sizer.Add(self.repeat_btn, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 0) 


    def _make_pos_sldr_and_text(self):
        if self.debug:
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Player slider maxValue = {math.ceil(self.host.cs_sec * self.pos_sldr_mult)}")
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Selected sample: '{truncate_string(self.host.cs_name, max = 23)}'")
        # 再生ヘッドの文字表示
        self.pos_text = wx.StaticText(
            self, 
            label = "{}\nPlaying: {: >7.2f} / {: >7.2f} sec".format(
                truncate_string(self.host.cs_name, max = 29),
                self.host.play_position / self.host.sr_out, 
                self.host.cs_sec,
            ),
        )
        # 再生ヘッドのスライダー
        self.pos_sldr = wx.Slider(
            self, 
            value = int(self.host.play_position / self.host.sr_out * self.pos_sldr_mult), 
            minValue = 0, 
            maxValue = math.ceil(self.host.cs_sec * self.pos_sldr_mult), # self.host.cs_sec は set_sample が作る
            size = (600, -1),
            style = wx.SL_HORIZONTAL | wx.SL_AUTOTICKS,
        )
        self.pos_sldr.SetTickFreq(self.pos_sldr_mult)
        self.pos_sldr.Bind(wx.EVT_SLIDER, self._on_pos_sldr_change) # スライダーのイベントハンドラ。文字の変更も含む
        # スライダーと文字表示を集めた sizer
        self.player_sldr_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.player_sldr_sizer.Add(self.pos_sldr, proportion = 0, flag = wx.ALL | wx.ALIGN_CENTER_VERTICAL, border = 5) 
        self.player_sldr_sizer.Add(self.pos_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)


    # イベントハンドラで、再生ヘッドに関するスライダーおよび文字の値を直接弄る。
    # このメソッドは self.host.play_position を直接弄ることに注意。
    def _on_pos_sldr_change(self, event):
        if hasattr(self, 'pos_sldr') and self.pos_sldr is not None:
            self.host.play_position = int(self.pos_sldr.GetValue() * self.host.sr_out // self.pos_sldr_mult)
            self.pos_sldr.SetValue(int(self.host.play_position / self.host.sr_out * self.pos_sldr_mult))
        # 文字を弄る処理も、対象が存在するかの検査を入れておく
        if hasattr(self, 'pos_text') and self.pos_text is not None:
            self.pos_text.SetLabel(
                label = "{}\nPlaying: {: >7.2f} / {: >7.2f} sec".format(
                    truncate_string(self.host.cs_name, max = 29),
                    self.host.play_position / self.host.sr_out, 
                    self.host.cs_sec,
                ),
            )
        self.root_sizer.Layout()


    # サンプラー付随プレイヤーの出力音量を変えるスライダーの値の変更
    def on_sample_amp_sldr_change(self, event):
        if hasattr(self, 'sample_amp_sldr') and self.sample_amp_sldr is not None:
            self.host.backend.sample_amp = self.sample_amp_sldr.GetValue() / self.sample_sldr_mult # ここは float
            self.sample_amp_sldr.SetValue(round(self.host.backend.sample_amp * self.sample_sldr_mult))

    # 上記の 3 つのイベントハンドラは設定の変更値を書き戻す必要がないので、そのような機能を組み込んでいない


    # スライダー部分を壊して再作成するメソッド。
    # いったん SampleManagerPanel に登録後、AudioSlotPanel の load_file から呼ばれる
    def remake_sldr(self):
        # 壊す前に、ウィジェット全体の sizer の中での順番を記録しておく
        index = self.root_sizer.GetChildren().index(self.root_sizer.GetItem(self.player_sldr_sizer))
        # Detach() と Destroy() で現在のスライダーを削除して作り直す
        self.root_sizer.Detach(self.player_sldr_sizer)
        self.player_sldr_sizer.Destroy()
        self.pos_sldr.Destroy()
        self.pos_text.Destroy()

        # 現在の duration に基づいて slider を作り直す。host から変数を参照している。
        self._make_pos_sldr_and_text() 
        # 統括 sizer に配置し直す
        self.root_sizer.Insert(index, self.player_sldr_sizer, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 0)
        self.root_sizer.Layout()

        if self.debug:
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Player slider was discarded and re-initialized.")


    # 再生／一時停止ボタンを押した時のイベント。self.host.playing の状態を切り替えるだけ。
    def send_sound(self, event):
        self._send_sound() # イベントハンドラ用のメソッドと、任意呼び出し用のメソッドを用意しておく


    # 任意呼び出し用メソッドは、self.host.playing の書き換えとボタン表示状態の整合を実装する
    def _send_sound(self):
        if not self.host.playing:
            if self.debug:
                self.logger.debug(f"({inspect.currentframe().f_code.co_name}) play")
            self.host.playing = True
        else:
            if self.debug:
                self.logger.debug(f"({inspect.currentframe().f_code.co_name}) pause")
            self.host.playing = False


    # 停止ボタンを押した時のイベント。常にヘッドを 0 戻しする
    def stop_sound(self, event):
        self._stop_sound() # イベントハンドラ用のメソッドと、任意呼び出し用のメソッドを用意しておく


    # 任意呼び出し用メソッドは、self.host.playing の書き換えとボタン表示状態の整合を実装する
    def _stop_sound(self):
        self.host.playing = False
        self.host.play_position = int(0) # ヘッドを 0 戻しする
        if self.debug:
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) stop sound")


    # リピートボタンを押した時のイベント
    def switch_repeat(self, event):
        if self.host.repeat:
            self.host.repeat = False
        else:
            self.host.repeat = True

