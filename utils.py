#!/usr/bin/env python3

# The MIT License

# Copyright (c) 2024 Lyodos

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# 以下に定める条件に従い、本ソフトウェアおよび関連文書のファイル（以下「ソフトウェア」）の複製を取得するすべての人に対し、ソフトウェアを無制限に扱うことを無償で許可します。これには、ソフトウェアの複製を使用、複写、変更、結合、掲載、頒布、サブライセンス、および/または販売する権利、およびソフトウェアを提供する相手に同じことを許可する権利も無制限に含まれます。

# 上記の著作権表示および本許諾表示を、ソフトウェアのすべての複製または重要な部分に記載するものとします。

# ソフトウェアは「現状のまま」で、明示であるか暗黙であるかを問わず、何らの保証もなく提供されます。ここでいう保証とは、商品性、特定の目的への適合性、および権利非侵害についての保証も含みますが、それに限定されるものではありません。作者または著作権者は、契約行為、不法行為、またはそれ以外であろうと、ソフトウェアに起因または関連し、あるいはソフトウェアの使用またはその他の扱いによって生じる一切の請求、損害、その他の義務について何らの責任も負わないものとします。 

import numpy as np

import re



# [-1, 1] の範囲を想定した信号を、デシベルフルスケール（dBFS）単位の音量に変換する

def to_dBFS(
    x,
    dig = 1,
):
    rms = np.sqrt(np.mean(x**2))
    return round(20 * np.log10(rms + 1e-8), dig)


# Hz 単位の周波数を、352 bins の one-hot feature map に戻す

def hz_to_onehot(
    hz: np.array, 
    fmin: float = 27.5,
    fmax: float = 4186,
    freq_bins: int = 352, 
    bins_per_octave: int = 48,
):
    indices = np.log((hz + 1e-7) / fmin) / np.log(2.0**(1.0 / bins_per_octave)) + 0.5
    indices = np.clip(indices, 0, freq_bins)
    return indices


# ContentVec に 16k wav を通した時のサイズ変化の予測

def pred_contentvec_len(length):
    return (length - 80) // 320


# 不連続な音声チャンク間で、サイン 2 乗によるクロスフェードを掛けるためのカーネル
# blocksize + extra で定義する
# ／＼ ではなく ／--＼
# -- の長さが extra であり、--＼ の長さが blocksize

def make_cross_extra_kernel(
    blocksize: tuple = (1, 2048),
    extra: int = 128, # -- の長さ
    divide: bool = False, # 作成したカーネルを「／」と「--＼」に分割する
): 
    assert extra > 0 and extra < blocksize[-1], "'extra' should be shorter than the blocksize"
    curve_length = extra*2 # 曲線部分 ／＼ のサンプル長
    time_array = np.linspace(0, np.pi, curve_length) # 曲線部分 ／＼ の基底
    
    if len(blocksize) == 2:
        sin_array_list = [time_array for _ in range(blocksize[0])]
    elif len(blocksize) == 3:
        sin_array_list = [[time_array for _ in range(blocksize[1])] for _ in range(blocksize[0])]
    elif len(blocksize) == 4:
        sin_array_list = [[[time_array for _ in range(blocksize[2])] for _ in range(blocksize[1])] for _ in range(blocksize[0])]
    else:
        raise NotImplementedError
    
    curve_array = np.sin(np.array(sin_array_list)) ** 2  # 曲線部分 ／＼ の実際の形を作成
    plateau_array = np.zeros(blocksize)[..., :(blocksize[-1]-extra)] + 1 # 平らな部分 -- すなわち係数 1 の array を作成

    if divide is False:
        return np.concatenate([curve_array[..., :extra], plateau_array, curve_array[..., -extra:]], axis = -1)
    else:
        current_half = np.concatenate([curve_array[..., :extra], plateau_array], axis = -1) # --＼
        previous_half = curve_array[..., -extra:] # ／
        return [current_half, previous_half]

####

# ビープ音を発生させる。音声デバイスのテストと、スペクトログラムのプロット時の bin の高さ確認に使う

def make_beep(
    sampling_freq: int = 44100,
    frequency: float = 440, 
    duration: float = 1, # 秒
    beep_rate: float = 0.1, # 音が出るパートの割合。1 だと常に鳴る。0.1 だと 0.1 秒鳴ってから 0.9 秒沈黙する
    level: float = 0.2, # beep 部分の音量
    n_channel: int = 1,
):
    num_samples = int(sampling_freq * duration)
    time = np.linspace(0, duration, num_samples, endpoint = False) # 時間軸を作成
    signal = level * np.sin(2 * np.pi * frequency * time) # まずは切れ目のない正弦波を作成。チャンネルは 1 つだけ
    assert beep_rate > 0, "beep_rate must be (0, 1]"
    if beep_rate < 1.0:
        signal[int(num_samples * beep_rate):] *= 0.0

    return np.stack([signal] * n_channel, axis = -1) # (time, n_channel) の numpy array で返す


def make_beep(
    sampling_freq: int = 44100,
    frequency: float = 440, 
    duration: float = 1, # 秒
    beep_rate: float = 0.1, # 音が出るパートの割合。1 だと常に鳴る。0.1 だと 0.1 秒鳴ってから 0.9 秒沈黙する
    level: float = 0.2, # beep 部分の音量
    n_channel: int = 1,
    channel_last: bool = True, # True で (time, n_channel)、False で (n_channel, time)
    dtype: np.dtype = np.float64, # 出力の dtype
):
    num_samples = int(sampling_freq * duration)
    time = np.linspace(0, duration, num_samples, endpoint = False) # 時間軸を作成
    signal = level * np.sin(2 * np.pi * frequency * time) # まずは切れ目のない正弦波を作成。チャンネルは 1 つだけ
    assert beep_rate > 0, "beep_rate must be (0, 1]"
    if beep_rate < 1.0:
        signal[int(num_samples * beep_rate):] *= 0.0

    signal = np.stack([signal] * n_channel, axis = -1) # (num_samples, n_channel) の numpy array を作成
    
    if not channel_last:
        signal = signal.T # (n_channel, num_samples) へ変換
    
    return signal.astype(dtype) # dtype を変換して返す


# 文字列が max 文字より長い場合、中間を省略して max 文字以下にする

def truncate_string(
    input_string, 
    max: int = 30, 
    replace_str: str = "...",
):
    if len(input_string) > max:
        half_length = (max - 3) // 2  # 省略部分の文字数
        truncated_string = input_string[:half_length] + replace_str + input_string[-half_length:]
        return truncated_string
    else:
        return input_string


# UNIXで使える文字を残し、それ以外はアンダースコアに置き換え

def sanitize_filename(
    input_string: str,
    allow_jp: bool = True,
):
    if allow_jp:
        # 日本語（ひらがな、カタカナ、漢字）、英数字、記号を許可
        sanitized = re.sub(r'[^ぁ-んァ-ン一-龯a-zA-Z0-9._-]', '_', input_string)
    else:
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', input_string)
    
    # 先頭や末尾のアンダースコアを削除
    sanitized = sanitized.strip('_')
    return sanitized

####

# プロット関係

import matplotlib
import matplotlib.cm as cm
matplotlib.use("Agg") # plt の前に指定
import matplotlib.pylab as plt


def plot_spectrogram_harmof0(
    spectrogram,
    f0 = None,
    act = None,
    figsize = (600, 200),
    aspect = None,
    v_range: tuple = (-50, 40),
    cmap = "inferno",
    plot_colorbar: bool = False,
):
    px = 1/plt.rcParams['figure.dpi'] # 1/100
    plt.rc('font', size = 6)          # controls default text sizes
    plt.rc('axes', titlesize = 6)     # fontsize of the axes title
    plt.rc('axes', labelsize = 6)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize = 6)    # fontsize of the tick labels
    plt.rc('ytick', labelsize = 6)    # fontsize of the tick labels
    plt.rc('legend', fontsize = 6)    # legend fontsize
    plt.rc('figure', titlesize = 10)  # fontsize of the figure title

    fig, ax = plt.subplots(
        figsize = (figsize[0]*px, figsize[1]*px),
    )
    plt.subplots_adjust(left = 0.05, right = 0.99, bottom = 0.11, top = 0.94)

    im = ax.imshow(
        spectrogram, 
        aspect = "auto" if aspect is None else aspect, 
        origin = "lower", 
        interpolation = 'none',
        cmap = matplotlib.colormaps[cmap],
        vmin = v_range[0],
        vmax = v_range[1],
    )
    
    # HarmoF0 activation sequence
    if act is not None:
        ln_act = ax.plot(
            np.linspace(start = 0, stop = spectrogram.shape[-1], num = f0.shape[-1]), 
            350*np.squeeze(act), 
            linewidth = 0.8, 
            linestyle = "dashed",
            color = (0.3, 0.5, 1.0),
        )
    
    # HarmoF0 pitch sequence
    if f0 is not None:
        ln_f0 = ax.plot(
            np.linspace(start = 0, stop = spectrogram.shape[-1], num = f0.shape[-1]), 
            np.squeeze(hz_to_onehot(f0)), 
            linewidth = 0.75, 
            color = (1, 1, 1),
        )

    if plot_colorbar:
        plt.colorbar(im, ax = ax, orientation = 'vertical', aspect = 50)
    
    fig.canvas.draw()
    plt.close()
    return fig, ax


# 話者スタイルの埋め込みを 128 ならば 16*8 といったビットマップで表現する

def plot_embedding_cube(
    embedding,
    figsize = (200, 200),
    image_reshape = (16, 8), # 1 次元配列を画像形状に変形するときのサイズ
    aspect = 1,
    v_range: tuple | list = (-2, 2),
    cmap = "bwr",
    transparent_margin: bool = False,
):
    px = 1/plt.rcParams['figure.dpi'] # e.g. 1/100
    plt.rc('font', size = 6)          # controls default text sizes
    plt.rc('axes', titlesize = 6)     # fontsize of the axes title
    plt.rc('axes', labelsize = 6)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize = 6)    # fontsize of the tick labels
    plt.rc('ytick', labelsize = 6)    # fontsize of the tick labels
    plt.rc('legend', fontsize = 6)    # legend fontsize
    plt.rc('figure', titlesize = 10)  # fontsize of the figure title

    fig, ax = plt.subplots(
        figsize = (figsize[0]*px, figsize[1]*px),
    )
    plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.05, top = 0.95)

    image_shape = image_reshape # 画像の形状を指定
    image_array = embedding.reshape(image_shape) # 1 次元配列を画像形状に変形

    im = ax.imshow(
        image_array,
        aspect = "auto" if aspect is None else aspect, 
        origin = "upper", 
        interpolation = 'none',
        cmap = matplotlib.colormaps[cmap],
        vmin = v_range[0],
        vmax = v_range[1],
    )

    if transparent_margin:
        fig.patch.set_alpha(0)

    fig.canvas.draw()
    plt.axis('off')
    plt.close()
    return fig, ax

