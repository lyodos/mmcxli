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
import copy
import numpy as np

import logging
import inspect

import wx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg

# hi dpi 対応
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except:
    pass


class PlotEmbeddingPanel(wx.Panel):
    def __init__(
        self, 
        parent, # parent として通常は wx.Panel（wx.Frame でも可）を指定する。
        host, # 描画対象の tensor を要素として保有しているインスタンスの実体を指定
        target_name: str = "buf_emb", # host 内の、描画ターゲットとする torch tensor (1, n_frame, dim_emb) の名前
        hop_sec: float = 0.02, # 描画対象の 1 frame が実時間で何秒に相当するか。
        channel: int = 0, # list ではなく int で与える。マルチチャンネルに対応していない
        from_sec: float = None, # 何秒前から現在までのデータを保持・描画するか。単位は秒
        figsize: tuple = (800, 800),
        v_range: tuple = (-2, 2), # 画像表示するときのカラースケールが対応すべき下限上限
        update_ms: int = 47, # プロットの更新間隔で、単位はミリ秒。wx.Timer なので厳密ではない
        id: int = -1, 
        **kwargs,
    ):
        super().__init__(parent, id = id, **kwargs)

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.logger.debug("Initializing ...")

        # plot 対象を定義後、 plt.ion() で interactive 化し、 update() で更新。
        plt.ion() # interactive mode をオンにする。こいつがないとグラフを Timer で更新しない。場所は図の作成よりも上。

        self.host = host
        self.target_name = target_name # 名前だけ渡しておいて、呼び出しのたびに中身を取得
        # raw_input が描画対象のテンソルの実体
        # ここで入れた長さのうち、最新の from_sec 秒に相当するデータだけが Panel 内で保持される
        raw_input = copy.deepcopy(getattr(self.host, self.target_name)) # (ch, dim_emb, n_frame) or (dim_emb, n_frame)
        if raw_input.ndim == 2:
            raw_input = raw_input[np.newaxis, :, :] # 本来趣旨では ndim = 3 だけ受け入れるべきである

        self.channel = min(channel, raw_input.shape[0] - 1) # どのチャンネルを描画するか
        self.dim_emb = raw_input.shape[1] # この時点で (n_ch, dim_emb, n_frame) が保証される
        self.hop_sec: float = hop_sec

        self.logger.debug(f"Hop: {self.hop_sec} seconds per frame, bins: {self.dim_emb}")

        # 元データの総フレーム長。time last が保証されるので dim 2 を取り出せばいい
        self.orig_frame_len: float =  raw_input.shape[2]
        # プロットに描画する時間範囲で、単位は秒。引数で与えなければ元データから計算
        self.from_sec: float = from_sec if from_sec is not None else self.hop_sec * self.orig_frame_len
        # 実際に描画データとして保持すべきフレーム長。例：4 s で 1 frame = 0.02 s だと、200 frames
        self.frame_len: int = int(self.from_sec / self.hop_sec) + 1 # ただし意図的に 1 frame を水増し
        self.logger.debug(f"Plot range: {self.from_sec} sec, {self.frame_len} frames")
        
        if self.frame_len > self.orig_frame_len:
            # こちらのバッファが入力データよりも長い→後ろ側に代入する
            self.tensor = np.zeros((1, self.dim_emb, self.frame_len), dtype = np.float32) 
            self.tensor[:, :, -self.orig_frame_len:] = raw_input[self.channel:self.channel+1, :, :]
        else:
            # こちらのバッファよりも入力データのほうが長い→入力データの後ろの方だけ使う
            self.tensor = raw_input[self.channel:self.channel+1, :, -self.frame_len:] 
        
        self.v_range = v_range 
        self.figsize = figsize
        self.update_ms = update_ms
        self.lapse = 0.0 # （テスト用）プロット所要時間を記録。単位はミリ秒
        
        ## 初期状態の plot を作る。安全を考えると、下はクラス化した方がいい。

        px = 1/plt.rcParams['figure.dpi'] # 1/100 （マシンによって異なる）
        plt.rc('font', size = 6)          # default text sizes
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
        ) # plot 外のエリア色を、アプリケーションの背景色に合わせる場合
        plt.subplots_adjust(left = 0.07, right = 0.98, bottom = 0.2, top = 0.9) # 軸ラベル分の余白確保が必要

        self.content_imshow = self.ax.imshow(
            self.tensor[self.channel, :, :], 
            aspect = 'auto', # aspect{'auto', 'equal'} or float
            interpolation = 'none', # 'none' もしくは 'nearest'
            vmin = self.v_range[0], 
            vmax = self.v_range[1],
            cmap = 'seismic', # bwr
            origin = 'upper',
        )

        xtick_divide = 5 # プロット範囲が何秒分であっても 0%, 25%, 50%, 75%, 100% に目盛り
        self.ax.set_xticks(np.linspace(0, self.frame_len, xtick_divide))
        self.ax.set_xticklabels(np.linspace(-self.from_sec, 0.0, xtick_divide).round(3))
        # y 軸の tickmark は省略

        self.ax.tick_params(
            bottom = True, 
            top = True, 
            labelbottom = True, 
            right = True, 
            left = True, 
            labelleft = True,
        )

        # FigureCanvasWxAgg は matplotlib と wx の連携用のパーツ
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.root_sizer = wx.BoxSizer(wx.VERTICAL)
        self.root_sizer.Add(self.canvas, proportion = 0, flag = wx.TOP)
        self.SetSizer(self.root_sizer)
        self.root_sizer.Fit(self) # Sizer を親要素である Panel のサイズに自動適合

        # wx.Timer クラスで、指定間隔での処理を実行する。
        self.timer = wx.Timer(self) 
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.timer.Start(self.update_ms)

        self.logger.debug("Initialized.")


    # ロールする機能は PlotEmbeddingPanel には持たせないことにした。そこにある tensor を素直に描画する機能に特化。
    def update(self, event):
        self.time0 = time.perf_counter_ns() # time in nanosecond

        raw_input = copy.deepcopy(getattr(self.host, self.target_name)) 

        if raw_input.ndim == 2:
            raw_input = raw_input[np.newaxis, :, :]

        if self.frame_len > raw_input.shape[2]:
            # こちらのバッファが入力データよりも長い→後ろ側に代入する
            self.tensor[:, :, -raw_input.shape[2]:] = raw_input[self.channel, :, :]
        else:
            # こちらのバッファよりも入力データのほうが長い→入力データの後ろの方だけ使う
            self.tensor = raw_input[self.channel, :, -self.frame_len:] 

        self.content_imshow.set_data(
            self.tensor[self.channel, :, :], 
        ) 
        
        # 強制再描画。これがないと Linux では画面が更新されない場合がある
        self.Refresh()
        self.Update()

        self.lapse = (time.perf_counter_ns() - self.time0)/1e+6 # time in millisecond

