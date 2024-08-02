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

# hi dpi 対応
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except:
    pass

from plot_waveform import PlotWaveformPanel
from plot_spectrogram import PlotSpecPanel

from plot_content import PlotEmbeddingPanel

#### 再生ボタン類とスライダー制御をまとめた部品クラス。

class MonitorWidgets(scrolled.ScrolledPanel):
    def __init__(
        self,
        parent,
        id = -1,
        sc = None,
        tab_id: int = None, # 何番目のタブに属するか。リアルタイム描画の必要性判定に使う
        show_content: bool = False,
    ):
        super().__init__(parent, id = id)
        self.sc = sc #  SoundControl クラスのインスタンスを指定
        self.tab_id = tab_id
        self.show_content = show_content

        # 波形プロット用のパネル（上の sc を受ける）を作成し、monitor tab 用の sizer に配置。
        self.wav_i_panel = PlotWaveformPanel(
            self, # parent
            backend = self.sc, 
            queue_name = "wq_input",
            sr = self.sc.sr_out,
            channel = [0, 1],
            plot_window_sec = 4.0,
            update_ms = 71,
        ) 

        self.wav_o_panel = PlotWaveformPanel(
            self, # parent
            backend = self.sc, 
            queue_name = "wq_output",
            sr = self.sc.sr_out,
            channel = [0, 1],
            plot_window_sec = 4.0,
            update_ms = 71,
        ) 

        # Mel spec プロット用のパネル
        self.mel_pre_panel = PlotSpecPanel(
            self, # parent
            host = self.sc.efx_control,
            target_name = "buf_spec_p",
            sr = 16000, #self.sc.sr_proc, # spec 変換前の waveform データが何 Hz 定義だったか
            hop_size = self.sc.efx_control.hop_size, # 描画対象の 1 frame が、waveform 何サンプルに相当するか。
            dim_spec = self.sc.efx_control.dim_spec,
            f_min = self.sc.efx_control.spec_fmin,
            f_max = self.sc.efx_control.spec_fmax,
            pitch_contour = "buf_f0_all",
#            pitch_contour = "buf_f0_real",
            channel = 0,#self.sc.efx_control.spec_rt_i, # 0: L, 1: R. Multi-channel is not supported yet.
        ) 
        
        # Output 側の Mel を描画するパネル
        self.mel_post_panel = PlotSpecPanel(
            self, # parent
            host = self.sc.efx_control,
            target_name = "buf_spec_o",
            sr = 16000,#self.sc.efx_control.sr_proc, # 変換前の waveform データが何 Hz 定義だったか
            hop_size = self.sc.efx_control.hop_size, # 描画対象の 1 frame が、waveform 何サンプルに相当するか。
            dim_spec = self.sc.efx_control.dim_spec, # 通常は 80 ただし、target tensor があるので自動取得は可能
            f_min = self.sc.efx_control.spec_fmin,
            f_max = self.sc.efx_control.spec_fmax,
            pitch_contour = "buf_f0_pred",
            channel = 0,#self.sc.efx_control.spec_rt_o, # 0: L, 1: R. Multi-channel is not supported yet.
        ) 
        
        if self.show_content:
            # ContentVec embedding プロット用のパネル
            self.emb_panel = PlotEmbeddingPanel(
                self, # parent
                self.sc.efx_control, # host
                "buf_emb", # target_name
                hop_sec = 0.02, # 描画対象である ContentVec が 1 frame で何秒に相当するか
            ) 
        
        ####
        
        self.info_i_sizer = wx.BoxSizer(wx.VERTICAL)
        self.info_i_text = wx.StaticText(self) # 入力音声情報を表示するテキスト
        self.info_i_text.SetLabel(f"Input signal:\n{self.sc.sr_out} Hz\nn_ch: {self.sc.n_ch_in_use[0]}")
        self.info_i_sizer.Add(self.info_i_text, flag = wx.GROW | wx.ALL, border = 0)

        self.info_o_sizer = wx.BoxSizer(wx.VERTICAL)
        self.info_o_text = wx.StaticText(self) # 入力音声情報を表示するテキスト
        self.info_o_text.SetLabel(f"Output signal:\n{self.sc.sr_out} Hz\nn_ch: {self.sc.n_ch_in_use[2]}")
        self.info_o_sizer.Add(self.info_o_text, flag = wx.GROW | wx.ALL, border = 0)

        self.spec_rt_i_sizer = wx.BoxSizer(wx.VERTICAL)
        self.spec_rt_i_rbx = wx.RadioBox(
            self, 
            wx.ID_ANY, 
            "Realtime plot", 
            choices = ["always", "with VC"], 
            majorDimension = 1,
        )
        self.spec_rt_i_rbx.Bind(wx.EVT_RADIOBOX, self.on_spec_rt_i)
        self.spec_rt_i_sizer.Add(self.spec_rt_i_rbx, flag = wx.GROW | wx.ALL, border = 0)
        self.spec_rt_i_rbx.SetSelection(int(self.sc.efx_control.spec_rt_i)) # 初期状態
        self.spec_rt_i_rbx.SetToolTip('VC needs the real-time calculation of the spectrogram when the pitch mode is "source" or the energy mode is "same as source"')
        
        self.spec_rt_o_sizer = wx.BoxSizer(wx.VERTICAL)
        self.spec_rt_o_rbx = wx.RadioBox(
            self, 
            wx.ID_ANY, 
            "Realtime plot", 
            choices = ["always", "with VC", "none"], 
            majorDimension = 1,
        )
        self.spec_rt_o_rbx.Bind(wx.EVT_RADIOBOX, self.on_spec_rt_o)
        self.spec_rt_o_sizer.Add(self.spec_rt_o_rbx, flag = wx.GROW | wx.ALL, border = 0)
        self.spec_rt_o_rbx.SetSelection(int(self.sc.efx_control.spec_rt_o)) # 初期状態
        self.spec_rt_o_rbx.SetToolTip('Extra calculation time (~ 20 ms)')
        
        ####
        
        # グリッドを作る。ポジションを指定して追加できる
        self.bag_sizer = wx.GridBagSizer(vgap = 0, hgap = 0) # 行間 0 px, 列間 0 px

        self.bag_sizer.Add(self.info_i_sizer, pos = (0, 0), flag = wx.ALL | wx.ALIGN_CENTER_VERTICAL, border = 0) 
        self.bag_sizer.Add(self.spec_rt_i_sizer, pos = (1, 0), flag = wx.ALL | wx.ALIGN_CENTER_VERTICAL, border = 0) 
        self.bag_sizer.Add(self.info_o_sizer, pos = (2, 0), flag = wx.ALL | wx.ALIGN_CENTER_VERTICAL, border = 0) 
        self.bag_sizer.Add(self.spec_rt_o_sizer, pos = (3, 0), flag = wx.ALL | wx.ALIGN_CENTER_VERTICAL, border = 0) 

        self.bag_sizer.Add(self.wav_i_panel, pos = (0, 1), flag = wx.ALL | wx.ALIGN_CENTER_VERTICAL, border = 0) 
        self.bag_sizer.Add(self.mel_pre_panel, pos = (1, 1), flag = wx.ALL | wx.ALIGN_CENTER_VERTICAL, border = 0) 
        self.bag_sizer.Add(self.wav_o_panel, pos = (2, 1), flag = wx.ALL | wx.ALIGN_CENTER_VERTICAL, border = 0) 
        self.bag_sizer.Add(self.mel_post_panel, pos = (3, 1), flag= wx.ALL | wx.ALIGN_CENTER_VERTICAL, border = 0) 
        
        if self.show_content:
            self.bag_sizer.Add(self.emb_panel, pos = (4, 1), flag = wx.ALL | wx.ALIGN_CENTER_VERTICAL, border = 0) 

        self.bag_size = self.bag_sizer.GetSize()
        self.bag_sizer.SetMinSize(self.bag_size) # 最低サイズ。これがないと  assertion 'size >= 0' failed in GtkScrollbar
        
        # 統括 sizer を作り、すべての部品を配置していく。
        self.root_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.root_sizer.Add(self.bag_sizer, 0, wx.EXPAND | wx.LEFT, border = 20)
        self.SetSizer(self.root_sizer)
        self.root_sizer.Fit(self) 
        self.Layout()

#        self.SetBackgroundColour(wx.Colour(255, 255, 255)) 
        self.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_MENU))

        self.SetupScrolling() # こいつだけはスクロールを有効化する必要がある

        self.timer = wx.Timer(self) # wx.Timer クラスで、指定間隔での処理を実行する。
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.timer.Start(37) # タイマーは手動で起動する必要がある。単位 ms

    def update(self, event):
        self.info_i_text.SetLabel(f"Input:\n{self.sc.sr_out} Hz\nn_ch: {self.sc.n_ch_in_use[0]}")
        self.info_o_text.SetLabel(f"Output:\n{self.sc.sr_out} Hz\nn_ch: {self.sc.n_ch_in_use[2]}")
    
        self.Refresh()
        self.Layout()
        self.Update()
    
    def on_spec_rt_i(self, event):
        self.sc.efx_control.spec_rt_i = self.spec_rt_i_rbx.GetSelection() # ["always", "with VC"] = [0, 1]
        # ここに、新しい設定値を vc_config に書き戻す処理を入れた。即座に json の上書き保存はしない。
        self.sc.update_vc_config("spec_rt_i", self.sc.efx_control.spec_rt_i, save = False)
#        self.spec_rt_i_rbx.SetSelection(int(self.sc.efx_control.spec_rt_i))

    def on_spec_rt_o(self, event):
        self.sc.efx_control.spec_rt_o = self.spec_rt_o_rbx.GetSelection() # ["always", "with VC", "none"] = [0, 1, 2]
#        self.spec_rt_o_rbx.SetSelection(int(self.sc.efx_control.spec_rt_o))
        self.sc.update_vc_config("spec_rt_o", self.sc.efx_control.spec_rt_o, save = False)


# お行儀を考えると、Frame のメソッドとして書くのではなく、ConfManager みたいなクラスを用意して機能をまとめるべきだろうが

