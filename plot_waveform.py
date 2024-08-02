#!/usr/bin/env python3

# The MIT License

# Copyright (c) 2024 Lyodos

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# 以下に定める条件に従い、本ソフトウェアおよび関連文書のファイル（以下「ソフトウェア」）の複製を取得するすべての人に対し、ソフトウェアを無制限に扱うことを無償で許可します。これには、ソフトウェアの複製を使用、複写、変更、結合、掲載、頒布、サブライセンス、および/または販売する権利、およびソフトウェアを提供する相手に同じことを許可する権利も無制限に含まれます。

# 上記の著作権表示および本許諾表示を、ソフトウェアのすべての複製または重要な部分に記載するものとします。

# ソフトウェアは「現状のまま」で、明示であるか暗黙であるかを問わず、何らの保証もなく提供されます。ここでいう保証とは、商品性、特定の目的への適合性、および権利非侵害についての保証も含みますが、それに限定されるものではありません。作者または著作権者は、契約行為、不法行為、またはそれ以外であろうと、ソフトウェアに起因または関連し、あるいはソフトウェアの使用またはその他の扱いによって生じる一切の請求、損害、その他の義務について何らの責任も負わないものとします。 

import time
import queue

import logging
import inspect

import numpy as np

import wx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg

# hi dpi 対応
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except:
    pass

# 要検討：self.backend.n_ch_in_use[2] は「出力をプロットに用いるよう決め打ちした場合のチャンネル数」であり、
# もしプロット対象のデータを別のキューから取得するように経路を変更すると、チャンネル数が合わなくなる虞がある。
# channel 引数（self.channel）でそのへんをうまく調整できるのだが、現在はまだ処理に反映されていない。

# waveform を図示するためのデータは wq から block_size 単位で送られてくる。
# 全部保持するわけではなく、表示用に間引いたデータだけを保持させる。
# sr_proc ではなく sr_out の世界で完結させる。

class PlotWaveformPanel(wx.Panel):
    def __init__(
        self, 
        parent, 
        id = -1, 
        backend = None, # SoundControl クラスのインスタンスを指定。
        queue_name: str = None, # parent に含まれるキューの名称（例："wq_input"）を文字列で
        sr: int = None,
        channel: int | list = [0, 1],
        down_sample: int = 32, # プロット時にサンプルを間引く。最低 1 = 全サンプルを描画。デフォルト 10
        plot_window_sec: float = 4.0, # プロットに描画する時間範囲で、単位はミリ秒
        figsize: tuple = (800, 120),
        update_ms: int = 100, # プロットの更新間隔で、単位はミリ秒
        **kwargs,
    ):
        super().__init__(parent, id = id, **kwargs) # parent として通常は wx.Panel（wx.Frame でも可）を指定する。

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.logger.debug("Initializing ...")

        plt.ion() # interactive mode をオンにする。こいつがないとグラフを Timer で更新しない。場所は図の作成よりも上。

        self.backend = backend # こいつは参照渡しなので、backend を本クラスのインスタンスから再定義できる。
        self.queue_name = queue_name
        self.sr = sr if sr is not None else self.backend.sr_out
        self.figsize = figsize
        
        # 波形を図示する際の間引きと描画範囲を定義。1000 フレーム 2 ch の描画におよそ 1 ms を要する
        self.channel = channel
        self.down_sample = down_sample 
        self.plot_window_sec = plot_window_sec # 単位ミリ秒。200 ms で 44100 だと 8820
        # キューから受け取った self.backend.blocksize の長さのサンプルを、解像度を落とした後、self.buffer に貯めていく
        self.buffer_length = int(self.plot_window_sec * self.sr / self.down_sample) # 802
        # プロットすべき波形データ（固定サイズとし、古いサンプルは捨てる）を、まずゼロで初期化しておく。
        self.buffer = np.zeros((self.buffer_length, len(self.channel)))
        self.update_ms = update_ms # 波形プロットの更新間隔。単位ミリ秒

        self.lapse = 0.0

        ## 初期状態の plot を作る。安全を考えると、下はクラス化した方がいい。

        px = 1/plt.rcParams['figure.dpi']
        plt.rc('font', size = 6)          # controls default text sizes
        plt.rc('axes', titlesize = 6)     # font size of the axes title
        plt.rc('axes', labelsize = 6)     # font size of the x and y labels
        plt.rc('xtick', labelsize = 6)    # font size of the tick labels
        plt.rc('ytick', labelsize = 6)    # font size of the tick labels
        plt.rc('legend', fontsize = 6)    # legend font size
        plt.rc('figure', titlesize = 10)  # font size of the figure title

        self.figure, self.ax = plt.subplots(
            figsize = (self.figsize[0]*px, self.figsize[1]*px),
        )
        self.figure.patch.set_facecolor(
            (
                wx.SystemSettings.GetColour(wx.SYS_COLOUR_MENU)[0]/255, 
                wx.SystemSettings.GetColour(wx.SYS_COLOUR_MENU)[1]/255, 
                wx.SystemSettings.GetColour(wx.SYS_COLOUR_MENU)[2]/255, 
                1
            )
        ) # rgba で指定。現在は plot 外のエリア色を、アプリケーションの背景色に合わせるようになっている
        plt.subplots_adjust(left = 0.07, right = 0.98, bottom = 0.2, top = 0.9)
        
        self.lines = self.ax.plot(
            self.buffer, # (time, ch) の 2d tensor
            linewidth = 0.85,
        )
        
        # チャンネル番号を凡例表示する
        if len(self.channel) > 0:
            self.legends = self.ax.legend(
                ['ch {}'.format(c) for c in self.channel], 
                loc = 'lower left', 
                ncol = len(self.channel),
            )
            self.legends.get_frame().set_linewidth(0.0)# 凡例の枠線を非表示にする
            self.legends.get_frame().set_facecolor('none') # 凡例の背景色を非表示にする
        
        self.ax.axis((0, self.buffer_length, -1, 1))
        xtick_divide = 5 # プロット範囲が何秒分であっても 0%, 25%, 50%, 75%, 100% に目盛り
        self.ax.set_xticks(np.linspace(0, self.buffer_length, xtick_divide))
        self.ax.set_xticklabels(np.linspace(-self.plot_window_sec, 0.0, xtick_divide).round(3)) # 「x 秒前」になる
        # y 軸の tickmark 
        self.ax.set_yticks(np.linspace(-1.0, 1.0, 5))
        self.ax.yaxis.grid(True)
        self.ax.tick_params(
            bottom = False, 
            top = False, 
            labelbottom = True, 
            right = False, 
            left = False, 
            labelleft = True,
        )
        self.ax.set_facecolor((
                wx.SystemSettings.GetColour(wx.SYS_COLOUR_MENU)[0]/255, 
                wx.SystemSettings.GetColour(wx.SYS_COLOUR_MENU)[1]/255, 
                wx.SystemSettings.GetColour(wx.SYS_COLOUR_MENU)[2]/255, 
                1
            )
        )
        
        # FigureCanvasWxAgg は matplotlib と wx の連携用のパーツ
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.plot_sizer = wx.BoxSizer(wx.VERTICAL)
        self.plot_sizer.Add(self.canvas, proportion = 0, flag = wx.TOP)
        self.SetSizer(self.plot_sizer)
        self.plot_sizer.Fit(self) # Sizer を親要素である Panel のサイズに自動適合

        # wx.Timer クラスで、指定間隔での処理を実行する。
        self.timer = wx.Timer(self) 
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.timer.Start(self.update_ms)

        self.logger.debug("Initialized.")


    # 毎フレームのプロットデータを作るメソッド。上で handler として使われ、self.timer にバインドしている。
    # plot_spectrogram は更新処理のたびに与えたデータを素直に全部描画するが、こちらは
    # waveform そのものではなく適度に間引いたデータを独自に保持する。なので roll が入っている。
    
    def update(self, event):
        self.time0 = time.perf_counter_ns() # time in nanosecond

        # メソッドを 1 回呼び出すごとに、キューが空になるまで get してプロットデータを充填する手続きを回す
        while True:
            try:
                temp = getattr(self.backend, self.queue_name).get_nowait()
            except queue.Empty:
                break
            # self.down_sample でサンプルを時間方向に間引く（エイリアシングは発生するが、速度のために妥協）
            temp = temp[::self.down_sample, :] # 本当は range(self.channel) で回すほうが安全
            self.buffer = np.roll(self.buffer, -len(temp), axis = 0) # 既存の self.buffer を最初の軸について左にロール
            self.buffer[-len(temp):, :] = temp # ロールして古いデータが入った部分を最新のデータに置換。
        
        for column, line in enumerate(self.lines):
            line.set_ydata(self.buffer[:, column]) 

        self.lapse = round((time.perf_counter_ns() - self.time0)/1e+6, 2) # time in millisecond
