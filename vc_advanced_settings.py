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
import math

import logging
import inspect


from utils import pred_contentvec_len


#### （高度）VC 変換処理のチャンクサイズを制御する。初期値は AudioEfx 作成時に入る

# なお、ここで制御する変数は本来、 self.sc.efx_control の他変数の値に応じて設定可能範囲の下限が変わる。
# しかし計算が面倒なので、現状では余裕を見て一定以上の長さで選択させるようになっている。

class AdvancedSettingsPanel(scrolled.ScrolledPanel):
    def __init__(
        self, 
        parent, 
        backend = None, # SoundControl クラスのインスタンスをここに指定。先にバックエンドが初期化されている必要がある。
        id = -1,
        host = None,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(parent, id = id, **kwargs)

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        self.logger.debug("Initializing...")
        self.debug = debug
        self.host = host
        self.sc = backend

        ####
        
        # ContentVec にどの程度の秒数のサンプルを入れるか。
        # 直接的には ContentVec に放り込むべき 16000 Hz waveform サンプル数 = len_embedder_input で決まるが、
        # self.len_embedder_input = int((self.len_content * 320 + 80)) で内部的に計算されるので操作すべきは self.len_content
        # self.len_content は「結果として何フレーム分の content を得たいか」を指定する引数。
        # なので、結果的にスライダーのスケールは f0n_predictor および decoder と同じになる。
        self.len_content_coef: float = 10 # 内部変数は 1 単位 20 ms だが細かすぎるので、スライダー 1 単位で 200 ms に粗くした
        self.len_content_default = math.ceil(self.sc.efx_control.len_content / self.len_content_coef)
        self.len_content_sldr = wx.Slider(
            self, 
            value = self.len_content_default, 
            minValue = 1, 
            maxValue = math.ceil(self.sc.efx_control.n_buffer_spec / (2 * self.len_content_coef)), # 最大 4 秒 
            size = (240, -1),
            style = wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS,
        )
        for value in [1, 2, 3, 4, 5, 10, 15, 20]:
            self.len_content_sldr.SetTick(value)
        self.len_content_sldr.Bind(wx.EVT_SLIDER, self.on_len_content_sldr_change)
        self.len_content_box = wx.StaticBox(
            self, 
            wx.ID_ANY, 
            "ContentVec"
        )
        self.len_content_text = wx.StaticText(self) # ラベルとして使用する StaticText を作成
        self.len_content_text.SetLabel(f'{self.len_content_coef*0.02} s * {self.len_content_default} chunks [Default {self.len_content_default}]')
        self.len_content_sizer = wx.StaticBoxSizer(self.len_content_box, wx.VERTICAL)
        self.len_content_sizer.Add(self.len_content_text, proportion = 0, flag = wx.LEFT | wx.TOP, border = 10)
        self.len_content_sizer.Add(self.len_content_sldr, proportion = 0, flag = wx.EXPAND | wx.ALL, border = 5)
        
        # 最低 1 frame = 20 ms 必要で、最大値はバッファ秒数（self.len_wav_i16 / 16000）だが、これも spec に便宜的に揃える
        # 厳密には、minValue は 1 frame = 20*self.len_content_coef = 200 ms 決め打ちではなく
        # backend の blocksize から動的に決めたほうがいい。
        # ただしリアルタイム VC で blocksize をこれ以上大きくしても意義が薄れてくるため、現状の 1 frame ですでに十分大きい。
        
        # ContentVec の計算時に終端が有効な埋め込みを返さないため、入力信号を予め折り返す量。
        # 内部変数は self.sc.content_expand_rate であり、レンジは 0 以上 1 以下。
        # 0 だと折り返しなし。1 だと元信号と同じ長さの折り返し量になり、これが理論上限である。
        # ただしスライダーの操作単位では 1 足してから 100 倍、つまり 100% から 200% までになるよう定義している。
        self.content_expand_coef: float = 100
        self.content_expand_default = round((self.sc.content_expand_rate + 1) * self.content_expand_coef) # [0, 1] -> [100, 200]
        self.content_expand_sldr = wx.Slider(
            self, 
            value = self.content_expand_default, 
            minValue = math.ceil(1 * self.content_expand_coef), # 最小 content_expand_rate = 0 で、スライダーの 100% に該当
            maxValue = round(2 * self.content_expand_coef), # 最大 200%
            size = (240, -1),
            style = wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS,
        )
        self.content_expand_sldr.SetTickFreq(20) # 20% ずつ操作
        self.content_expand_sldr.Bind(wx.EVT_SLIDER, self.on_content_expand_sldr_change)
        self.content_expand_box = wx.StaticBox(
            self, 
            wx.ID_ANY, 
            "ContentVec end flip size"
        )
        self.content_expand_text = wx.StaticText(self) # ラベルとして使用する StaticText を作成
        self.content_expand_text.SetLabel(f'{self.content_expand_default}%')
        self.content_expand_sizer = wx.StaticBoxSizer(self.content_expand_box, wx.VERTICAL)
        self.content_expand_sizer.Add(self.content_expand_text, proportion = 0, flag = wx.LEFT | wx.TOP, border = 10)
        self.content_expand_sizer.Add(self.content_expand_sldr, proportion = 0, flag = wx.EXPAND | wx.ALL, border = 5)
        
        # self.len_w2m もここに反映する
        # ContentVec が self.len_content で決まっていたように、self.len_w2m は間接的に操作する必要がある。
        # これも多分、1 chunk を 10 ms * 10 (coef) として何チャンク放り込むかで制御したほうがいい。
        self.len_spec_coef: float = 10 # 変数は 1 chunk で 10 ms だが細かすぎるので、スライダー 1 単位で 10*10 ms に調整
        self.len_spec_default = math.ceil(self.sc.efx_control.len_spec / self.len_spec_coef)
        self.len_spec_sldr = wx.Slider(
            self, 
            value = self.len_spec_default, 
            minValue = 2, 
            maxValue = math.ceil(self.sc.efx_control.n_buffer_spec / (4 * self.len_spec_coef)), # 最大 4 秒 
            size = (240, -1),
            style = wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS,
        )
        for value in [2, 3, 4, 5, 10, 15, 20]:
            self.len_spec_sldr.SetTick(value)
        self.len_spec_sldr.Bind(wx.EVT_SLIDER, self.on_len_spec_sldr_change)
        self.len_spec_box = wx.StaticBox(
            self, 
            wx.ID_ANY, 
            "Wav2spec"
        )
        self.len_spec_text = wx.StaticText(self) # ラベルとして使用する StaticText を作成
        self.len_spec_text.SetLabel(f'{self.len_spec_coef*0.01} s * {self.len_spec_default} chunks [Default {self.len_spec_default}]')
        self.len_spec_sizer = wx.StaticBoxSizer(self.len_spec_box, wx.VERTICAL)
        self.len_spec_sizer.Add(self.len_spec_text, proportion = 0, flag = wx.LEFT | wx.TOP, border = 10)
        self.len_spec_sizer.Add(self.len_spec_sldr, proportion = 0, flag = wx.EXPAND | wx.ALL, border = 5)

        # f0n_predictor と decoder はいずれも投入データが buf_emb であり、バッファ長は n_buffer_spec に依存する。
        
        # f0n_predictor のチャンクサイズを制御するスライダー。
        # f0n_predictor に放り込むべき content フレーム数（20 ms hop）
        # 実際の変数は 1 chunk で 20 ms だが、単位が細かすぎるのでスライダー変化率を 10 倍に調整（スライダー 1 で 200 ms）
        # 最低値は 1 で、最大値は spectrogram バッファ秒数（ self.sc.efx_control.n_buffer_spec * 0.01）を超えない程度
        self.len_f0n_coef: float = 10 
        self.len_f0n_default = math.ceil(self.sc.efx_control.len_f0n_predictor / self.len_f0n_coef)
        self.len_f0n_sldr = wx.Slider(
            self, 
            value = self.len_f0n_default, 
            minValue = 1, 
            maxValue = math.ceil(self.sc.efx_control.n_buffer_spec / (2 * self.len_f0n_coef)), # 最大 4 秒 
            size = (240, -1),
            style = wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS,
        )
        for value in [1, 2, 3, 4, 5, 10, 15, 20]:
            self.len_f0n_sldr.SetTick(value)
        self.len_f0n_sldr.Bind(wx.EVT_SLIDER, self.on_len_f0n_sldr_change)
        self.len_f0n_box = wx.StaticBox(
            self, 
            wx.ID_ANY, 
            "F0n predictor"
        )
        self.len_f0n_text = wx.StaticText(self) # ラベルとして使用する StaticText を作成
        self.len_f0n_text.SetLabel(f'{self.len_f0n_coef*0.02} s * {self.len_f0n_default} chunks [Default {self.len_f0n_default}]')
        self.len_f0n_sizer = wx.StaticBoxSizer(self.len_f0n_box, wx.VERTICAL)
        self.len_f0n_sizer.Add(self.len_f0n_text, proportion = 0, flag = wx.LEFT | wx.TOP, border = 10)
        self.len_f0n_sizer.Add(self.len_f0n_sldr, proportion = 0, flag = wx.EXPAND | wx.ALL, border = 5)

        # VC デコードのチャンクサイズを制御するスライダー。
        # VC decoder に放り込むべき content フレーム数（20 ms hop）
        # 最低値は 1 で、最大値は spectrogram バッファ秒数（ self.sc.efx_control.n_buffer_spec * 0.01）を超えない程度
        self.len_proc_coef: float = 10
        self.len_proc_default = math.ceil(self.sc.efx_control.len_proc / self.len_proc_coef)
        self.len_proc_sldr = wx.Slider(
            self, 
            value = self.len_proc_default, 
            minValue = 1, 
            maxValue = math.ceil(self.sc.efx_control.n_buffer_spec / (2 * self.len_proc_coef)), # 最大 4 秒 
            size = (240, -1),
            style = wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS,
        )
        for value in [1, 2, 3, 4, 5, 10, 15, 20]:
            self.len_proc_sldr.SetTick(value)
        self.len_proc_sldr.Bind(wx.EVT_SLIDER, self.on_len_proc_sldr_change)
        self.len_proc_box = wx.StaticBox(
            self, 
            wx.ID_ANY, 
            f'Decoder'
        )
        self.len_proc_text = wx.StaticText(self) # ラベルとして使用する StaticText を作成
        self.len_proc_text.SetLabel(f'{self.len_proc_coef*0.02} s * {self.len_proc_default} chunks [Default {self.len_proc_default}]')
        self.len_proc_sizer = wx.StaticBoxSizer(self.len_proc_box, wx.VERTICAL)
        self.len_proc_sizer.Add(self.len_proc_text, proportion = 0, flag = wx.LEFT | wx.TOP, border = 10)
        self.len_proc_sizer.Add(self.len_proc_sldr, proportion = 0, flag = wx.EXPAND | wx.ALL, border = 5)
        
        # self.len_style_encoder は多分本番で使わないかな？
        
        # さらに、最終出力音声をクロスフェードするかの切り替え。
        
        #### バッファの置換サイズ

        # F0n predictor のバッファ更新を、ロール分だけ入れるか、計算したチャンクを全部入れるか。デフォルトは True で全部
        self.substitute_f0n_pred_array = ('roll only', 'all')
        self.substitute_f0n_pred_rbx = wx.RadioBox(
            self, 
            wx.ID_ANY, 
            'f0n_predictor buffer update',
            choices = self.substitute_f0n_pred_array, 
            size = (240, -1),
            style = wx.RA_VERTICAL,
        )
        self.substitute_f0n_pred_rbx.Bind(wx.EVT_RADIOBOX, self.on_substitute_f0n_pred)
        self.substitute_f0n_pred_rbx.SetSelection(int(self.sc.efx_control.substitute_all_for_f0n_pred))
        self.substitute_f0n_pred_sizer = wx.BoxSizer(wx.VERTICAL)
        self.substitute_f0n_pred_sizer.Add(self.substitute_f0n_pred_rbx)

        # Spectrogram のバッファ更新を、ロール分だけ入れるか、計算したチャンクを全部入れるか。デフォルトは True で全部
        self.substitute_spec_array = ('roll only', 'all')
        self.substitute_spec_rbx = wx.RadioBox(
            self, 
            wx.ID_ANY, 
            'Spectrogram buffer update',
            choices = self.substitute_spec_array, 
            size = (240, -1),
            style = wx.RA_VERTICAL,
        )
        self.substitute_spec_rbx.Bind(wx.EVT_RADIOBOX, self.on_substitute_spec)
        self.substitute_spec_rbx.SetSelection(int(self.sc.efx_control.substitute_all_for_spec))
        self.substitute_spec_sizer = wx.BoxSizer(wx.VERTICAL)
        self.substitute_spec_sizer.Add(self.substitute_spec_rbx)

        # Content のバッファ更新を、ロール分だけ入れるか、計算したチャンクを全部入れるか。デフォルトは False でロールだけ
        self.substitute_content_array = ('roll only', 'all')
        self.substitute_content_rbx = wx.RadioBox(
            self, 
            wx.ID_ANY, 
            'ContentVec buffer update',
            choices = self.substitute_content_array, 
            size = (240, -1),
            style = wx.RA_VERTICAL,
        )
        self.substitute_content_rbx.Bind(wx.EVT_RADIOBOX, self.on_substitute_content)
        self.substitute_content_rbx.SetSelection(int(self.sc.efx_control.substitute_all_for_content))
        self.substitute_content_sizer = wx.BoxSizer(wx.VERTICAL)
        self.substitute_content_sizer.Add(self.substitute_content_rbx)

        # Decoder 出力は今のところ全部入れている

        # 最終出力時のクロスフェード量を制御するスライダー。
        # f0n_predictor に放り込むべき content フレーム数（20 ms hop）
        # 実際の変数は 1 単位 1 sample だが、単位が細かすぎるのでスライダー変化率を 64 倍に調整
        # 最低値は 1 で、最大値は spectrogram バッファ秒数（ self.sc.efx_control.n_buffer_spec * 0.01）を超えない程度
        self.len_fade_coef: int = int(self.sc.sr_out / 1000)
        self.len_fade_default = int(self.sc.efx_control.cross_fade_samples / self.len_fade_coef)
        self.len_fade_sldr = wx.Slider(
            self, 
            value = self.len_fade_default, 
            minValue = 0, 
            maxValue = min(20, int(self.sc.blocksize / (self.len_fade_coef))), # blocksize より大きな fade はありえない
            size = (240, -1),
            style = wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS,
        )
        for value in [0, 1, 2, 3, 4, 5, 10, 15, 20]:
            self.len_fade_sldr.SetTick(value)
        self.len_fade_sldr.Bind(wx.EVT_SLIDER, self.on_len_fade_sldr_change)
        self.len_fade_box = wx.StaticBox(
            self, 
            wx.ID_ANY, 
            f"Output cross-fade size"
        )
        self.len_fade_box.SetToolTip('The cross-fade adds delay, but needed for a gapless realtime VC, especially in a singing voice')
        self.len_fade_text = wx.StaticText(self) # ラベルとして使用する StaticText を作成
        self.len_fade_text.SetLabel(f'{self.len_fade_coef*self.len_fade_default} samples ({round(1000*self.len_fade_coef*self.len_fade_default/self.sc.sr_out)} ms) [Default {self.len_fade_coef*self.len_fade_default}]')
        self.len_fade_sizer = wx.StaticBoxSizer(self.len_fade_box, wx.VERTICAL)
        self.len_fade_sizer.Add(self.len_fade_text, proportion = 0, flag = wx.LEFT | wx.TOP, border = 10)
        self.len_fade_sizer.Add(self.len_fade_sldr, proportion = 0, flag = wx.EXPAND | wx.ALL, border = 5)

        #### Help
        
        self.spec_help_source = {
            "en": "Spectrograms is calculated as a by-product of the HarmoF0 module, which estimates the fundamental frequency (F0) of the original speech. Volume is also included. Meanwhile, the calculation of a speaker style vector depends on a spectrogram, but we can define the target style in advance. It means that the real-time estimation of the source speaker style is unnecessary.",
            "ja": "音声をスペクトログラムに変換するモジュールは、元発話のピッチ（F0）を推定する HarmoF0 モデルの前処理の一部であるため、ここで F0 および音量も同時推定される。作成したスペクトログラムから話者スタイルを抽出できるが、変換先スタイルが既知である場合はリアルタイム処理は不要である。なお HarmoF0 のアルゴリズム上、入力系列長が短くても精度は十分に出る。",
        }
        self.spec_help_panel = wx.Panel(self)
        self.spec_help_text = wx.StaticText(self.spec_help_panel, label = self.spec_help_source["en"])
        self.spec_help_text.SetSize((240, -1))
        self.spec_help_text.SetMaxSize((240, -1))
        self.spec_help_text.Wrap(235)  # 250ピクセルで折り返す
        self.spec_help_panel.SetToolTip(self.spec_help_source["ja"])
        
        self.content_help_source = {
            "en": "The ContentVec module, which extracts speech content, directly takes a 16000 Hz waveform as input. A desired input sequence length is around 2 to 3 seconds.\nSince it's a Transformer model, extending the sequence doesn't increase computation time linearly, but occupies VRAM.",
            "ja": "ContentVec は 16000 Hz の音声を直接入力に取り、発話内容の埋め込みを抽出する。2 ないし 3 秒以上の入力長が望ましい。なお内部構造に Transformer を含むため、入力を長くしても処理時間は線形には増えないが VRAM を著しく消費する。",
        }
        self.content_help_panel = wx.Panel(self)
        self.content_help_text = wx.StaticText(self.content_help_panel, label = self.content_help_source["en"])
        self.content_help_text.SetSize((240, -1))
        self.content_help_text.SetMaxSize((240, -1))
        self.content_help_text.Wrap(235)  # 250ピクセルで折り返す
        self.content_help_panel.SetToolTip(self.content_help_source["ja"])

        self.f0n_help_source = {
            "en": "There are two ways to determine the target pitch and energy for VC: 1) the 'source' mode based on the original speech, and 2) 'target' mode, which uses a dedicated network (f0n predictor).\nNo real-time computation for the former, thereby significantly saving time. For the latter, the f0n predictor ideally requires a minimum input duration of 2 (preferably 4) seconds.",
            "ja": "変換後の然るべきピッチ（F0）および音量（energy）を決めるとき、HarmoF0 で推定した（元発話の）値をシフトさせる方法（Pitch mode: 'source'）と、ContentVec 特徴量および変換先の話者スタイルから推定する方法（Pitch mode: 'target'）とがある。前者は推定に使う F0n predictor ネットワークをリアルタイムで要求しないので、計算時間を大幅に節約できる。後者では F0n predictor の入力は最低 2 秒、できれば 4 秒欲しい。",
        }
        self.f0n_help_panel = wx.Panel(self)
        self.f0n_help_text = wx.StaticText(self.f0n_help_panel, label = self.f0n_help_source["en"])
        self.f0n_help_text.SetSize((240, -1))
        self.f0n_help_text.SetMaxSize((240, -1))
        self.f0n_help_text.Wrap(235)  # 250ピクセルで折り返す
        self.f0n_help_panel.SetToolTip(self.f0n_help_source["ja"])

        self.proc_help_source = {
            "en": "With enough GPU capacity, decoding longer sequences than the default (0.8 seconds) can slightly enhance sound quality.\nOutput audio mismatch from the previous iteration leads to popping noise without cross-fading. Although it adds delay to the VC, less than 5 ms is not recommended.",
            "ja": "GPU に余裕があれば、デフォルトの 0.8 秒よりも長くすることで音質が若干改善する。イテレーション間で音高や音量が完全には一致しないため、クロスフェード量が 0 だとプチノイズが発生する。クロスフェード分だけ追加の遅延が生じるが、5 ms 以上は確保しないとノイズが目立つようになる。",
        }
        self.proc_help_panel = wx.Panel(self)
        self.proc_help_text = wx.StaticText(self.proc_help_panel, label = self.proc_help_source["en"])
        self.proc_help_text.SetSize((240, -1))
        self.proc_help_text.SetMaxSize((240, -1))
        self.proc_help_text.Wrap(235)  # 250ピクセルで折り返す
        self.proc_help_panel.SetToolTip(self.proc_help_source["ja"])
        
        #### Sizer

        # グリッドを作る。ポジションを指定して追加できる
        self.bag_sizer = wx.GridBagSizer(vgap = 0, hgap = 0) # 行間 0 px, 列間 0 px
        
        # chunk size 設定
        self.bag_sizer.Add(self.len_spec_sizer, pos = (0, 0), flag = wx.ALL, border = 5)
        self.bag_sizer.Add(self.len_content_sizer, pos = (0, 1), flag = wx.ALL, border = 5)
        self.bag_sizer.Add(self.len_f0n_sizer, pos = (0, 2), flag = wx.ALL, border = 5)
        self.bag_sizer.Add(self.len_proc_sizer, pos = (0, 3), flag = wx.ALL, border = 5)
        # Buffer 置換サイズ設定
        self.bag_sizer.Add(self.substitute_spec_sizer, pos = (1, 0), flag = wx.ALL, border = 5)
        self.bag_sizer.Add(self.substitute_content_sizer, pos = (1, 1), flag = wx.ALL, border = 5)
        self.bag_sizer.Add(self.substitute_f0n_pred_sizer, pos = (1, 2), flag = wx.ALL, border = 5)
        # ContentVec フリップおよび出力クロスフェード設定
        self.bag_sizer.Add(self.content_expand_sizer, pos = (2, 1), flag = wx.ALL, border = 5)
        self.bag_sizer.Add(self.len_fade_sizer, pos = (2, 3), flag = wx.ALL, border = 5)
        # Help テキスト
        self.bag_sizer.Add(self.spec_help_panel, pos = (3, 0), flag = wx.ALL, border = 5)
        self.bag_sizer.Add(self.content_help_panel, pos = (3, 1), flag = wx.ALL, border = 5)
        self.bag_sizer.Add(self.f0n_help_panel, pos = (3, 2), flag = wx.ALL, border = 5)
        self.bag_sizer.Add(self.proc_help_panel, pos = (3, 3), flag = wx.ALL, border = 5)

        self.bag_size = self.bag_sizer.GetSize()
        self.bag_sizer.SetMinSize(self.bag_size) # 最低サイズ。これがないと  assertion 'size >= 0' failed in GtkScrollbar
        
        # 統括 sizer を作り、すべての部品を配置していく。
        self.root_sizer = wx.BoxSizer(wx.VERTICAL)
        self.root_sizer.Add(self.bag_sizer, 0, wx.EXPAND | wx.ALL, border = 5)
        self.SetSizer(self.root_sizer)
        self.root_sizer.Fit(self) 
        self.Layout()

        # self.timer = wx.Timer(self) # wx.Timer クラスで、指定間隔での処理を実行する。
        # self.Bind(wx.EVT_TIMER, self.update, self.timer)
        # self.timer.Start(37) # タイマーは手動で起動する必要がある。単位 ms

        self.logger.debug("Initialized.")

    ####

    def update(self, event):
        pass


    # Wav2spec チャンクサイズを変えるスライダーの値の変更
    def on_len_spec_sldr_change(self, event):
        if hasattr(self, 'len_spec_sldr') and self.len_spec_sldr is not None:
            self.sc.efx_control.len_spec = int(self.len_spec_sldr.GetValue() * self.len_spec_coef)
            self.sc.efx_control.len_w2m = math.ceil(self.sc.efx_control.len_spec * 160)
            val = math.ceil(self.sc.efx_control.len_spec / self.len_spec_coef)
            self.len_spec_sldr.SetValue(val)
            self.len_spec_text.SetLabel(f'{self.len_spec_coef*0.01} s * {val} chunks [Default {self.len_spec_default}]')
            self.sc.update_vc_config("len_spec", self.sc.efx_control.len_spec, save = False)

    # ContentVec チャンクサイズを変えるスライダーの値の変更
    # ただし、self.len_content_input = int((self.len_content * 320 + 80)) がサンプル単位の実際の入力サイズ
    def on_len_content_sldr_change(self, event):
        if hasattr(self, 'len_content_sldr') and self.len_content_sldr is not None:
            self.sc.efx_control.len_content = int(self.len_content_sldr.GetValue() * self.len_content_coef)
            # ContentVec 入力および出力のフレームサイズ予測値も更新
            self.sc.efx_control.len_embedder_input = int((self.sc.efx_control.len_content * 320 + 80))
            self.sc.efx_control.len_embedder_output = pred_contentvec_len(self.sc.efx_control.len_embedder_input) 
            val = math.ceil(self.sc.efx_control.len_content / self.len_content_coef)
            self.len_content_sldr.SetValue(val)
            self.len_content_text.SetLabel(f'{self.len_content_coef*0.02} s * {val} chunks [Default {self.len_content_default}]')
            self.sc.update_vc_config("len_content", self.sc.efx_control.len_content, save = False)

    # ContentVec の終端の flip 量を変えるスライダーの値の変更
    def on_content_expand_sldr_change(self, event):
        if hasattr(self, 'content_expand_sldr') and self.content_expand_sldr is not None:
            val = self.content_expand_sldr.GetValue()
            self.sc.content_expand_rate = float(val) / self.content_expand_coef - 1
            self.content_expand_text.SetLabel(f'{val}%')
            self.sc.update_vc_config("content_expand_rate", self.sc.content_expand_rate, save = False) # backend にある

    # F0/energy 推定時のチャンクサイズを変えるスライダーの値の変更
    def on_len_f0n_sldr_change(self, event):
        if hasattr(self, 'len_f0n_sldr') and self.len_f0n_sldr is not None:
            self.sc.efx_control.len_f0n_predictor = int(self.len_f0n_sldr.GetValue() * self.len_f0n_coef)
            val = math.ceil(self.sc.efx_control.len_f0n_predictor / self.len_f0n_coef)
            self.len_f0n_sldr.SetValue(val)
            self.len_f0n_text.SetLabel(f'{self.len_f0n_coef*0.02} s * {val} chunks [Default {self.len_f0n_default}]')
            self.sc.update_vc_config("len_f0n_predictor", self.sc.efx_control.len_f0n_predictor, save = False)

    # デコード時のチャンクサイズを変えるスライダーの値の変更
    def on_len_proc_sldr_change(self, event):
        if hasattr(self, 'len_proc_sldr') and self.len_proc_sldr is not None:
            self.sc.efx_control.len_proc = int(self.len_proc_sldr.GetValue() * self.len_proc_coef)
            val = math.ceil(self.sc.efx_control.len_proc / self.len_proc_coef)
            self.len_proc_sldr.SetValue(val)
            self.len_proc_text.SetLabel(f'{self.len_proc_coef*0.02} s * {val} chunks [Default {self.len_proc_default}]')
            self.sc.update_vc_config("len_proc", self.sc.efx_control.len_proc, save = False)


    # VC 出力のクロスフェード量を変えるスライダーの値の変更
    def on_len_fade_sldr_change(self, event):
        if hasattr(self, 'len_fade_sldr') and self.len_fade_sldr is not None:
            self.sc.efx_control.cross_fade_samples = int(self.len_fade_sldr.GetValue() * self.len_fade_coef)
            val = int(self.sc.efx_control.cross_fade_samples / self.len_fade_coef)
            self.len_fade_sldr.SetValue(val)
            self.len_fade_text.SetLabel(f'{self.len_fade_coef*val} samples ({round(1000*self.len_fade_coef*val/self.sc.sr_out)} ms) [Default {self.len_fade_coef*self.len_fade_default}]')
            self.sc.efx_control.need_remake_kernel = True # クロスフェード用のカーネルを弄るので、ロックが必要
            self.sc.update_vc_config("cross_fade_samples", self.sc.efx_control.cross_fade_samples, save = False)


    # 入力スペクトログラムのバッファ更新を、ロール分だけ入れるか、計算したチャンクを全部入れるか
    def on_substitute_spec(self, event):
        self.sc.efx_control.substitute_all_for_spec = bool(self.substitute_spec_rbx.GetSelection())
        self.substitute_spec_rbx.SetSelection(int(self.sc.efx_control.substitute_all_for_spec))
        self.sc.update_vc_config("substitute_all_for_spec", self.sc.efx_control.substitute_all_for_spec, save = False)

    # Content のバッファ更新を、ロール分だけ入れるか、計算したチャンクを全部入れるか
    def on_substitute_content(self, event):
        self.sc.efx_control.substitute_all_for_content = bool(self.substitute_content_rbx.GetSelection())
        self.substitute_content_rbx.SetSelection(int(self.sc.efx_control.substitute_all_for_content))
        self.sc.update_vc_config("substitute_all_for_content", self.sc.efx_control.substitute_all_for_content, save = False)

    # F0n predictor のバッファ更新を、ロール分だけ入れるか、計算したチャンクを全部入れるか
    def on_substitute_f0n_pred(self, event):
        self.sc.efx_control.substitute_all_for_f0n_pred = bool(self.substitute_f0n_pred_rbx.GetSelection())
        self.substitute_f0n_pred_rbx.SetSelection(int(self.sc.efx_control.substitute_all_for_f0n_pred))
        self.sc.update_vc_config("substitute_all_for_f0n_pred", self.sc.efx_control.substitute_all_for_f0n_pred, save = False)

####
