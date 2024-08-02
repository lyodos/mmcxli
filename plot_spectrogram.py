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
import math
import librosa

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


from utils import hz_to_onehot


contour_color_cycle = [
    (0.96, 0.99, 0.98, 0.95), # しろ
    (0.6, 0.9, 0.98, 0.94), # そらいろ
    (1, 0.9, 0.1, 0.92), # きいろ
]


# 前提として PlotSpecPanel() を初期化する時点で host は target_name を作成済み、かつ update しても大きさ一定の必要あり
# target_name で取得される対象テンソルは (1, n_frame, dim_emb) の channel last を想定していたが、
# どうやら time last に統一するほうがソースが簡潔になる

# 構造は ContentVec の描画用クラスに近いが、こちらはプロット範囲の秒数指定が存在せず、与えたテンソルの全範囲を描画する
# 実際何秒分のデータであるかは、テンソルのサイズと元データのサンプリング周波数、hop size から推定する

# Y 軸の設計をみて分かるとおり、現在は log spectrogram のプロットに特化している。
# マルチチャンネルを突っ込むと、 channel 引数に指定した 1 つのチャンネルだけが描画される

# なお update_ms を小さく（30 ms 台など）すると、それだけで Panel の更新動作が極端に遅くなるので注意。


class PlotSpecPanel(wx.Panel):
    def __init__(
        self, 
        parent, # parent として通常は wx.Panel（wx.Frame でも可）を指定する。
        host, # 描画対象の tensor を保有するインスタンスを指定
        target_name: str = "buf_spec_p", # host 内の、描画ターゲットとする tensor (batch, dim_spec, n_frame) の名前
        sr: int = 16000, # spec 変換前の waveform データが何 Hz 定義だったか
        hop_size: int = 160, # 描画対象の 1 frame が、waveform 何サンプルに相当するか。秒数を求めるには sr が必要
        dim_spec: int = 352, # ただし、target tensor があるので自動取得は可能
        bins_per_octave: int = 48, # これは HarmoF0 で使うような、対数スペクトログラムのみ存在する指標（mel にはない）
        f_min: int = None,
        f_max: int = None,
        channel: int = 0, # list ではなく int で与える。マルチチャンネルに対応していない
        figsize: tuple = (800, 170), # プロットエリアの横、縦。単位は px
        v_range: tuple = (-50, 40), # スペクトログラムを画像表示するときのカラースケールが対応すべき下限上限
        pitch_contour: str = None, # f0 抽出結果をオーバーレイ表示。host 内の numpy array (batch, n_frame) の名称指定
        update_ms: int = 150, # プロットの更新間隔で、単位はミリ秒。wx.Timer なので厳密ではない。
        id = -1, 
        **kwargs,
    ):
        super().__init__(parent, id = id, **kwargs)

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.logger.debug("Initializing ...")

        # plot 対象を定義後、 plt.ion() で interactive 化し、 update() で更新。
        plt.ion() # interactive mode をオンにする。こいつがないとグラフを Timer で更新しない。場所は図の作成よりも上

        self.host = host
        self.target_name = target_name # 名前だけ渡しておいて、呼び出しのたびに中身を取得
        # self.tensor が描画対象のテンソルの実体。常に time last が前提
        self.tensor = copy.deepcopy(getattr(self.host, self.target_name)) # 3D (ch, dim_spec, n_frame)

        self.sr = sr
        self.hop_size = hop_size
        
        self.dim_spec = dim_spec
        self.bins_per_octave = bins_per_octave
        # bins 上限は、HarmoF0 に算入したため取得が厄介になった
        self.f_min = f_min if dim_spec is not None else 0 # ただし実際の HarmoF0 では 0 ではない
        self.f_max = f_max if dim_spec is not None else self.sr//2 # ただし実際の HarmoF0 では sr//2 ではない
        
        self.logger.debug(f"Hop: {self.hop_size}, bins: {self.dim_spec} ({self.f_min}--{self.f_max})")

        # プロットするフレームの元データの長さは、最初に投入したテンソルの長さに従う
        self.frame_len =  self.tensor.shape[2]
        # プロットに描画する時間範囲で、単位は秒
        self.time_range = self.hop_size * self.frame_len / self.sr

        self.logger.debug(f"Time range: {self.time_range} seconds, frame length: {self.frame_len}")
        
        # 同時にプロットする F0 の値。時間解像度を自動調整して spec 側に合わせられる。
        # なお contour はマルチチャンネルも許容する。そのため、(ch, time) の 2D に強制的に変更する処理が入る。
        self.pitch_contour_name = pitch_contour
        if self.pitch_contour_name is not None:
            pitch_data = copy.deepcopy(getattr(self.host, self.pitch_contour_name))
            if pitch_data.ndim == 3:
                pitch_data = pitch_data[:, 0, :]
            if pitch_data.ndim == 1:
                pitch_data = pitch_data[np.newaxis, :] # 強制的に 2D 化する
            
            # pitch contour の末尾次元サイズを pitch_adj_rate 倍して spec の時間に合わせる
            self.pitch_adj_rate: float = 1.0
            # 時間解像度が違う場合、adjust rate を決めてデータをリサンプリングする
            if pitch_data.shape[1] != self.tensor.shape[-1]:
                self.pitch_adj_rate = self.tensor.shape[-1] / pitch_data.shape[1]
                pitch_data = librosa.resample(
                    pitch_data, 
                    orig_sr = 1000, # sr を整数化しても値がずれないよう、適当な大きな値にしている
                    target_sr = round(1000*self.pitch_adj_rate),
                    res_type = "polyphase",
                )
                self.logger.debug(f"Pitch is adjusted by {self.pitch_adj_rate}. Adjusted shape: {pitch_data.shape}")
            else:
                pass
        
        # スペクトログラムはマルチチャンネル化していないため、ステレオを突っ込むと後のチャンネルだけ描画する
        self.channel = min(channel, self.tensor.shape[0] - 1) # どのチャンネルを描画するか
        self.v_range = v_range 
        self.figsize = figsize
        self.update_ms = update_ms
        self.lapse = 0.00 # （テスト用）プロット所要時間を記録。単位はミリ秒
        
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
        ) # rgba で指定。現在は plot 外のエリア色を、アプリケーションの背景色に合わせるようになっている
        plt.subplots_adjust(left = 0.07, right = 0.98, bottom = 0.15, top = 0.9) # 軸ラベル分の余白確保が必要

        self.spec_imshow = self.ax.imshow(
            self.tensor[self.channel, :, :], 
            aspect = 'auto', # aspect{'auto', 'equal'} or float
            interpolation = 'nearest', # 'none' もしくは 'nearest' それ以外は非推奨
            vmin = self.v_range[0], 
            vmax = self.v_range[1],
            cmap = 'inferno',
            origin = 'lower',
        )
        
        # 設計上は contour は 1 本とは限らないので、ループで 1 本ずつ描画を追加していく
        self.contour_cmap = contour_color_cycle # ファイル冒頭で定義したもの
        if self.pitch_contour_name is not None:
            self.contour_plot = []
            # Pitch contour の方は、別々の Line2D オブジェクトを束ねたリストとして作成する
            for i in range(pitch_data.shape[0]):
                line, = self.ax.plot(
                    np.arange(pitch_data.shape[1]), # x 
                    hz_to_onehot(pitch_data[i, :]), # y 
                    color = self.contour_cmap[i],
                    linewidth = 0.5, # 本当は線幅や線種も変えられるが、今のところ実装していない
                )
                self.contour_plot.append(line)

        xtick_divide = 5 # プロット範囲が何秒分であっても 0%, 25%, 50%, 75%, 100% に目盛り
        self.ax.set_xticks(np.linspace(0, self.frame_len, xtick_divide))
        self.ax.set_xticklabels(np.linspace(-self.time_range, 0.0, xtick_divide).round(3)) # 「x 秒前」になる
        # y 軸の tickmark 
        self.ax_bins = [(i+1)*100 for i in range(9)] + [(i+1)*1000 for i in range(4)]
        self.ax_bins_label = [str(100)] + [""]*8 + [str(1000)] + [""] + [""] + [str(4000)]
        self.ax.set_yticks(
            [math.log((hz + 1e-7) / self.f_min) / math.log(2.0**(1.0 / self.bins_per_octave)) + 0.5 for hz in self.ax_bins], 
            labels = self.ax_bins_label,
            minor = False,
        )

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


    # self.update_ms が大きい時、以下のメソッドの何処かが、パネル全体の更新処理を著しく重くしている。

    # ロールする機能は PlotSpecPanel には持たせないことにした。そこにある tensor を素直に描画する機能に特化。
    def update(self, event):
        self.time0 = time.perf_counter_ns() # time in nanosecond
        
        data = copy.deepcopy(getattr(self.host, self.target_name))
        self.spec_imshow.set_data(
            data[self.channel, :, :], 
        ) 

        if self.pitch_contour_name is not None:
            pitch = copy.deepcopy(getattr(self.host, self.pitch_contour_name))
            if pitch.ndim == 3:
                pitch = pitch[:, 0, :]
            if pitch.ndim == 1:
                pitch = pitch[np.newaxis, :] # 強制的に 2D 化する

            pitch = librosa.resample(
                pitch, 
                orig_sr = 1000, # sr を整数化しても値がずれないよう、適当な大きな値にしている
                target_sr = round(1000*self.pitch_adj_rate),
                res_type = "polyphase",
            )
            for i, line in enumerate(self.contour_plot):
                line.set_data(
                    np.arange(pitch.shape[1]), 
                    hz_to_onehot(pitch[i, :]), 
                ) 

        self.lapse = round((time.perf_counter_ns() - self.time0)/1e+6, 2) # time in millisecond
