#!/usr/bin/env python3

# The MIT License

# Copyright (c) 2024 Lyodos

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# 以下に定める条件に従い、本ソフトウェアおよび関連文書のファイル（以下「ソフトウェア」）の複製を取得するすべての人に対し、ソフトウェアを無制限に扱うことを無償で許可します。これには、ソフトウェアの複製を使用、複写、変更、結合、掲載、頒布、サブライセンス、および/または販売する権利、およびソフトウェアを提供する相手に同じことを許可する権利も無制限に含まれます。

# 上記の著作権表示および本許諾表示を、ソフトウェアのすべての複製または重要な部分に記載するものとします。

# ソフトウェアは「現状のまま」で、明示であるか暗黙であるかを問わず、何らの保証もなく提供されます。ここでいう保証とは、商品性、特定の目的への適合性、および権利非侵害についての保証も含みますが、それに限定されるものではありません。作者または著作権者は、契約行為、不法行為、またはそれ以外であろうと、ソフトウェアに起因または関連し、あるいはソフトウェアの使用またはその他の扱いによって生じる一切の請求、損害、その他の義務について何らの責任も負わないものとします。 

import os
import math
import copy # 再代入を想定したインスタンス変数（mutable: list, dict, bytearray, set）は deepcopy で渡す必要がある。
import time

import logging
import inspect

import numpy as np
rng = np.random.default_rng(2141)

import librosa

import onnxruntime as ort # 予め ort-gpu を入れること。 Opset 17 以上が必要
#pip install ort-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-12/pypi/simple/
so = ort.SessionOptions()
so.log_severity_level = 3


from utils import pred_contentvec_len, make_cross_extra_kernel, make_beep


class AudioEfx:
    def __init__(
        self, 
        sc, # Audio backend のこと。SoundControl クラスのインスタンスをここに指定。
        vc_config, # ロード（もしくは main.py で作成）した dict を指定
        hop_size: int = 160, # wav2spec での信号の時間フレームの圧縮。例：16000 Hz を 1/100 s 間隔のスペクトログラムに → 160 倍
        dim_spec: int = 352,
        ch_map: list = [0], # 入力信号のどのチャンネルを、処理関数に流すかを決めるマップ（下記）
        bypass: bool = False, # 入力信号をそのまま返す（ただし VC の処理は走るので重さは変わらない）
    ):
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        self.vc_config = vc_config
        
        # バックエンド作成前に、利用可能な GPU の数を見ておく

        # 特定の GPU を決め打ちで使いたい場合は CUDA_VISIBLE_DEVICESを設定
        if self.vc_config["model"]["model_device"] == "cuda:0":
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            self.onnx_provider_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.device = "cuda:0"
        elif self.vc_config["model"]["model_device"] == "cuda:1":
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            self.onnx_provider_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.device = "cuda:1"
        elif self.vc_config["model"]["model_device"] == "cpu":
            self.onnx_provider_list = ['CPUExecutionProvider']
            self.device = "cpu"
        else:
            self.onnx_provider_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.device = "cuda" # つまり "cuda:2" 以降のデバイス指定は無視される
        self.logger.debug(f"ONNX Runtime provider: {str(self.onnx_provider_list)}")

        # 以下の変数は実際にロードされる重みそのものではなく、その相対パスを示す文字列
        self.CE_ckpt = self.vc_config["model"]["CE_ckpt"]
        self.harmof0_ckpt = self.vc_config["model"]["harmof0_ckpt"]
        self.SE_ckpt = self.vc_config["model"]["SE_ckpt"]
        self.f0n_ckpt = self.vc_config["model"]["f0n_ckpt"]
        self.decoder_ckpt = self.vc_config["model"]["decoder_ckpt"]
        
        #### Audio settings
        
        self.sc = sc # stream backend
        self.sr_proc = self.sc.sr_proc
        self.sr_dec = self.vc_config["backend"]["sr_decode"]
        self.hop_size = hop_size
        self.dim_spec = dim_spec
        self.bypass = bypass # 推論負荷は掛けるが、入力音声を出力音声にバイパス。テスト専用機能で config からは触れない

        # backend に返すときに前のイテレーションの復元音声とクロスフェードする長さ（単位：sample）
        self.cross_fade_samples = self.vc_config["cross_fade_samples"] 

        # 入力信号のどのチャンネルを、処理関数に流すかを決めるマップ。config ではなく backend 作成前に内部で決める
        self.ch_map = ch_map
        if self.sc.n_ch_in_use[1] != len(self.ch_map):
            self.logger.warning("The channel map configurations differ between the audio backend and the VC engine.")

        # チャンネルマッピングの例（ただし現在は ch0 の 1 つしか VC していない）：
        # [0] 入力信号が何チャンネルであれ、ch 0 だけ取り出して VC の ch 0 に回す（内部チャンネルが 1 つ）
        # [1] 入力信号が何チャンネルであれ、ch 1 だけ取り出して〜
        # [0, 2] 入力信号が何チャンネルであれ、ch 0 と ch 2 を取り出して VC の ch 0, 1 に回す（内部チャンネルが 2 つ）
        # VC 側は元信号のチャンネル構成を知らないが、ch_map を逆に適用すれば AudioEfx 側で復元はできる。

        #### VC settings
        
        # 以下は初期値を config から取得し、いったん VC が立ち上がった後は GUI 上のコントローラから制御する
        self.mic_amp_range = self.vc_config["mic_amp_range"]
        self.auto_encode = self.vc_config["auto_encode"]
        self.pitch_shift = self.vc_config["pitch_shift"]
        self.pitch_range = self.vc_config["pitch_range"]
        self.absolute_pitch = self.vc_config["absolute_pitch"]
        self.estimate_energy = self.vc_config["estimate_energy"] 

        # f0n_predictor による F0 および energy の間接推定が必要か否か（内部的に更新すべきフラグ）
        if self.absolute_pitch is False and self.estimate_energy is False:
            self.need_pred_f0n = False
        else:
            self.need_pred_f0n = True

        self.substitute_all_for_spec = self.vc_config["substitute_all_for_spec"]
        self.substitute_all_for_content = self.vc_config["substitute_all_for_content"]
        self.substitute_all_for_f0n_pred = self.vc_config["substitute_all_for_f0n_pred"]
        self.spec_rt_i = self.vc_config["spec_rt_i"]
        self.spec_rt_o = self.vc_config["spec_rt_o"]
        self.activation_threshold = self.vc_config["activation_threshold"]

        #### buffer settings and definition

        # スペクトログラム等に変換した特徴量（いずれも hop = 10 ms のもの）溜めておくバッファの長さ。
        # 秒数ではなく frame 数で与える（これらを起点に他のパラメータを計算するので、厳密な値が必要）
        self.n_buffer_spec = self.vc_config["n_buffer_spec"] # 単位は spectrogram frames

        # waveform を溜めておくバッファの長さ。こちらは実時間（秒数）で定義
        self.sec_wav_buffer = self.vc_config["sec_wav_buffer"]
        # 当然、stream のブロックサイズより小さいことはありえない。なお長い分にはあまり問題ない。
        self.logger.debug(f"'sec_wav_buffer' is {self.sec_wav_buffer}: must be longer than {self.sc.blocksize / self.sc.sr_out}")
        self.sec_wav_buffer = max(self.sec_wav_buffer, self.sc.blocksize / self.sc.sr_out)
        
        # style vector を作るとき、メルスペクトログラム（hop = 10 ms）を何フレーム放り込むか。
        # 他の特徴量よりも長い必要があり、3 ないし 4 秒 = 300--400 frame 程度ないと安定しない。また無音に弱い。
        # ただし「リアルタイムで計算する必要がない」
        self.len_style_encoder = self.vc_config["len_style_encoder"] 
        # 単位は spectrogram frames （なお構造上 self.n_buffer_spec 以下）

        # pitch, energy のリアルタイム推定時、content（hop = 20 ms）を何フレーム放り込むか。len_proc よりは長くしたい
        self.len_f0n_predictor = self.vc_config["len_f0n_predictor"] 
        # 単位は content frames（これも blocksize を下回れない）


        # 1 回の変換で VC decoder に放り込むべき content フレーム数（hop = 20 ms）
        self.len_proc = self.vc_config["len_proc"] 
        # 当たり前だが decode 時の長さ（20 ms * frames）は callback の 1 ブロック分の実時間より短くできない
        # さらに現在、余裕を見て最低 2 倍は用意するようになっている
        # 一方、バッファの秒数を超える量を復号することも不可能
        self.logger.debug(f"Content blocks for decoder input is {self.len_proc}: must be shorter/longer than {int(self.sec_wav_buffer / 0.02)} / {math.ceil(100*self.sc.blocksize / self.sc.sr_out)}")
        
        self.len_proc = max(
            min(
                self.len_proc, 
                int(self.sec_wav_buffer / 0.02),
            ), 
            math.ceil(100*self.sc.blocksize / self.sc.sr_out),
        )
        
        # 上の設定値に基づき、実際の各バッファ変数の長さを決めていく。
        
        # self.buf_wav_i は波形を収容する ndarray だが、backend と異なり float32 かつ (channel, time) の time last 形式
        # 音声入出力から VC エンジン向けの channel mapping は、self.buf_wav_i への代入時に行う。
        self.len_wav_i   = int(self.sc.sr_out * self.sec_wav_buffer) # オーディオ入出力（44100 or 48000 Hz）ベース
        self.len_wav_i16 = int(16000 * self.sec_wav_buffer) # 16000 Hz ベース
        self.logger.debug(f"    - I/O audio blocksize: {self.sc.blocksize} samples ({self.sc.blocksize/self.sc.sr_out: >6.3f} sec: {self.sc.sr_out} Hz)")
        self.logger.debug(f"    - Buffer for raw input audio: {self.len_wav_i} samples ({self.sec_wav_buffer: >5.1f} sec: {self.sc.sr_out} Hz)")
        self.logger.debug(f"    - Buffer for ContentVec: {self.len_wav_i16} samples ({self.sec_wav_buffer: >5.1f} sec: {16000} Hz)")


        # スペクトログラム変換
        self.len_spec = self.vc_config["len_spec"]
        self.spec_fmin = self.vc_config["spec_fmin"]
        self.spec_fmax = self.vc_config["spec_fmax"]

        # 当然、buffer size より長くなることはありえない。
        # さりとて、callback を 1 回呼び出す間に進む実時間 = self.sc.blocksize / self.sc.sr_out よりは長い必要がある。
        # （さもないとスペクトログラムを plot するときに隙間が生じる）
        self.logger.debug(f"Wav2spec input is {self.len_spec} chunks ({math.ceil(self.len_spec*self.hop_size)} samples): must be shorter than {self.n_buffer_spec} chunks ({self.len_wav_i16} samples) and longer than {math.ceil(100 * self.sc.blocksize / self.sc.sr_out)} chunks")
        
        self.len_spec = max(
            min(
                self.len_spec, 
                self.n_buffer_spec,
            ), 
            math.ceil(100 * self.sc.blocksize / self.sc.sr_out),
        )

        # len_w2m は self.buf_wav_i16 の末尾の幾つまでのサンプルを、実際に wav2spec に放り込むか。
        self.len_w2m = math.ceil(self.len_spec*self.hop_size)


        # ContentVec
        # 次いで、self.buf_wav_i16 の末尾の幾つまでのサンプルを ContentVec に放り込むかを決める。
        # len_content は「結果として何フレーム分の content を得たいか」を指定する引数。最低 1 frame = 20 ms 必要
        self.len_content = self.vc_config["len_content"] # (>= 1) 
        
        # 当然、buffer size より長くなることはありえない。
        # さりとて、callback を 1 回呼び出す間に進む実時間よりは長い必要がある。
        self.logger.debug(f"ContentVec input is {self.len_content} chunks ({int((self.len_content * 320 + 80))} samples): must be shorter than {self.len_wav_i16} samples and longer than {math.ceil(50 * self.sc.blocksize / self.sc.sr_out)} chunks")

        self.len_content = max(
            min(
                self.len_content, 
                self.len_wav_i16,
            ), 
            math.ceil(50 * self.sc.blocksize / self.sc.sr_out),
        )
        
        # サンプル単位の実際の入力サイズは self.len_embedder_input であり、len_content から逆算して決める。
        self.len_embedder_input = int((self.len_content * 320 + 80))
        
        # ContentVec 生出力のフレームサイズ予測値
        self.len_embedder_output = pred_contentvec_len(self.len_embedder_input) 
        #   生出力は (x - 80) // 320 frames に。入力は 16000 Hz なので、生出力は約 20 ms の時間解像度を持つ。
        #   ここから分かる通り、ContentVec の順伝播には最低 400 samples の 16k waveform (= 25 ms) が必要。
        self.logger.debug(f"ContentVec output is {self.len_embedder_output} samples")

        #### Make buffer arrays
        
        # 上の設定値に基づき、実際にバッファを作っていく。
        # この時点で input_stream は作成済みであるが開始していないので、まずダミーデータを作る
        self.buf_wav_i   = (rng.random((len(self.ch_map), self.len_wav_i  ), dtype = np.float32) - 0.5) * 2e-5
        self.buf_wav_i16 = (rng.random((len(self.ch_map), self.len_wav_i16), dtype = np.float32) - 0.5) * 2e-5
        # 出力音声も buffer に貯める。入力音声と同じ長さ (self.len_wav_i)、同じサンプリング周波数 (self.sc.sr_out)
        self.buf_wav_o   = (rng.random((len(self.ch_map), self.len_wav_i  ), dtype = np.float32) - 0.5) * 2e-5

        # ContentVec においてバックグラウンドノイズだけの入力は荒れるので、疑似信号を冒頭に足す
        self.sine_input = make_beep(
            sampling_freq = self.sc.sr_proc, 
            frequency = 220, 
            duration = 0.1,
            beep_rate = 1.0,
            level = 0.2,
            n_channel = len(self.ch_map),
            channel_last = False,
            dtype = np.float32,
        )
        # 正弦波よりもランダムノイズのほうが、VC したときの聴感がまろやかになる。
        self.rand_input = (rng.random((len(self.ch_map), self.sc.blocksize), dtype = np.float32) - 0.5) * 0.4

        # spec buffer の信号下限値（-50）は正確には PlotSpecPanel の v_range[0] だが、変数アクセスが面倒なので手入力した
        self.buf_spec_p = rng.random((len(self.ch_map), self.dim_spec, self.n_buffer_spec), dtype = np.float32) - 50 
        self.buf_spec_o = np.zeros((len(self.ch_map), self.dim_spec, self.n_buffer_spec), dtype = np.float32) - 50 
        # content embedding を記録するバッファ。
        self.buf_emb = np.zeros((len(self.ch_map), 768, self.n_buffer_spec//2), dtype = np.float32)
        # なお spec は 10 ms だが content は 20 ms 解像度なので、ContentVec のバッファは半分に切り詰めていることに注意

        # さらに、抽出もしくは予測した f0 および energy を保管するバッファも作る。いずれも spec と同じ 10 ms 解像度
        self.buf_f0_real = np.zeros((len(self.ch_map), self.n_buffer_spec), dtype = np.float32) + 440.0
        self.buf_energy_real = np.zeros((len(self.ch_map), self.n_buffer_spec), dtype = np.float32)
        self.buf_activation = np.zeros((len(self.ch_map), self.n_buffer_spec), dtype = np.float32)
        self.buf_f0_pred = np.zeros((len(self.ch_map), self.n_buffer_spec), dtype = np.float32) + 440.0
        self.buf_energy_pred = np.zeros((len(self.ch_map), self.n_buffer_spec), dtype = np.float32)

        # 以下はプロット用に、f0 の実測と予測を合わせたもの
        self.buf_f0_all = np.concatenate((self.buf_f0_real, self.buf_f0_pred), axis = 0)

        # イテレーションごとのバッファ巻取り量の設定
        self.logger.debug(f"Buffer roll size: {self.sc.block_roll_size*2} frames for spectrogram, {self.sc.block_roll_size} for ContentVec, and {self.sc.block_roll_size*2} for F0 and Energy")

        # 計算が必要な場合は、 callback 1 回の実時間を援用すればいい。
        # スペクトログラムが hop = 10 ms: 1 秒で 100 frames 進むとき、self.sc.blocksize / self.sc.sr_out 秒で何フレーム進むか？
        # f0 および energy のバッファのロール量も、10 ms なので spec 用の変数が使い回せる。
        # 一方 content は hop = 20 ms: 1 秒で 50 frames 進むとき、self.sc.blocksize / self.sc.sr_out 秒で何フレーム進むか？
        
        # 蛇足： block_roll_size や blocksize は VC 側の変数とも捉えられるが、stream の作成に blocksize が必要なので、
        # backend の変数として持たせる方がたぶん都合がいい。

        #### 処理を記録する変数の定義

        self.proc_head = 0 # バックエンドから何サンプル取り込んだか（入力デバイスのサンプリング周波数準拠）
        self.pre_lap: float = 0.0 # 1 回の推論呼び出しにおいて、取り込んだ音声を VC 用に前処理するときの所要時間
        self.vc_lap: float = 0.0
        self.post_lap: float = 0.0
        self.total_end_time = time.perf_counter_ns() # 前のイテレーションの終了時刻を記録する

        #### ネットワークの初期化
        
        # pitch tracker (& wav2spec), Style Encoder, ContentVec, f0n_predictor, decoder の順に作成＆テスト
        # すべてバッファに入れてから次の工程に進むようにしたので、順番は変えてもいい

        # バッファに計算したチャンクを入れるとき、大雑把に 3 つの選択肢がある
        # 1. 全部入れる
        # 2. roll size 分だけ切り出して入れる
        # 3. roll size + 一定の遷移期間のフェードインを手前に作って入れる
        
        # wav_i 等は roll size だけ入れても滑らかに繋がるが、
        # ContentVec 等は当フレームの結果に前後のフレームの状態が依存するため、チャンクを全部突っ込むと表示が崩れる。
        # しかし roll size 分だけ切り出すと、バッファに入れたとき前の iteration のフレームとの間が不連続になる。

        self.logger.debug(f"Initializing HarmoF0...")
        self.sess_HarmoF0 = ort.InferenceSession(
            self.harmof0_ckpt, 
            providers  = self.onnx_provider_list,
            session_options = so,
        )
        test_HarmoF0 = True
        if test_HarmoF0:
            _, _, _, _ = self.sess_HarmoF0.run(
                ['freq_t', 'act_t', 'energy_t', 'spec'], 
                {"input": self.buf_wav_i16[:, -self.len_w2m:]},
            )
            time0 = time.perf_counter_ns() # time in nanosecond
            real_F0, activation, real_N, spec_chunk = self.sess_HarmoF0.run(
                ['freq_t', 'act_t', 'energy_t', 'spec'], 
                {"input": self.buf_wav_i16[:, -self.len_w2m:]},
            )
            self.harmof0_lap = (time.perf_counter_ns() - time0)/1e+6
            self.logger.debug(f"    - {str(list(self.buf_wav_i16[:, -self.len_w2m:].shape))} was converted {str(list(spec_chunk.shape))} and {str(list(real_F0.shape))} in {self.harmof0_lap: >7.2f} ms")
            self.logger.debug(f"    - (HarmoF0 RTF: {self.harmof0_lap / (self.len_w2m/16): >7.4f})")
            self.spec_chunk_size = spec_chunk.shape[-1]
            spec_chunk = spec_chunk[:, :, 2:] # 最初の 2 点はゴミなので削る
            self.buf_spec_p[:, :, -spec_chunk.shape[2]:] = copy.deepcopy(spec_chunk)


        self.logger.debug(f"Initializing Style Encoder...")
        # 入力は (batch, 1, dim_spec, n_frame >= 80) の 4D テンソル
        self.sess_SE = ort.InferenceSession(
            self.SE_ckpt, 
            providers  = self.onnx_provider_list,
            session_options = so,
        )
        test_SE = True
        if test_SE:
            _ = self.sess_SE.run(
                ['output'], 
                {'input': self.buf_spec_p[:, 48:, -self.len_style_encoder:][:, np.newaxis, :, :]},
            )[0]
            time0 = time.perf_counter_ns() # time in nanosecond
            self.style_silent = self.sess_SE.run(
                ['output'], 
                {'input': self.buf_spec_p[:, 48:, -self.len_style_encoder:][:, np.newaxis, :, :]},
            )[0]
            self.SE_lap = (time.perf_counter_ns() - time0)/1e+6
            self.logger.debug(f"    - {str(list(self.buf_spec_p[:, 48:, -self.len_style_encoder:][:, np.newaxis, :, :].shape))} was converted {str(list(self.style_silent.shape))} in {self.SE_lap: >7.2f} ms")
            self.logger.debug(f"    - (Style RTF: {self.SE_lap / (self.len_style_encoder*10): >7.4f})")
            # 出力は時間次元を持たない (batch, 128) ので入力長は自由だが、なるべく長めに通したほうが安定する。

        # style_vect は VC 時のターゲット話者スタイルとなる変数。
        self.style_vect = self.style_silent # とりあえず「無音を埋め込んだベクトル」である style_silent で初期化する。
        # このスタイルは config に入れておく
        self.vc_config["style"]["style_silent"] = self.style_silent.tolist()
        
        # 厳密には、必ず test_SE を実行しておかないと self.style_silent が作成されないので、上の書き方はよくない


        self.logger.debug(f"Initializing ContentVec...")
        self.sess_CE = ort.InferenceSession(
            self.CE_ckpt, 
            providers  = self.onnx_provider_list,
            session_options = so,
        )
        test_CE = True
        if test_CE:
            _ = self.sess_CE.run(
                ['last_hidden_state'], 
                {'input': self.buf_wav_i16[:, -self.len_embedder_input:]},
            )[0]
            time0 = time.perf_counter_ns() # time in nanosecond
            content0 = self.sess_CE.run(
                ['last_hidden_state'], 
                {'input': self.buf_wav_i16[:, -self.len_embedder_input:]},
            )[0]
            # なお content は ContentVec そのままではなく time last に変換する必要
            content0 = content0.transpose(0, 2, 1)
            self.CE_lap = (time.perf_counter_ns() - time0)/1e+6
            self.logger.debug(f"    - {str(list(self.buf_wav_i16[:, -self.len_embedder_input:].shape))} ({self.len_embedder_input/16000: >6.3f} sec) was converted to {str(list(content0.shape))} in {self.CE_lap: >7.2f} ms") 
            self.logger.debug(f"    - (Content RTF: {self.CE_lap / (self.len_embedder_input/32): >7.4f})")


        self.logger.debug(f"Initializing F0/Energy Predictor...")
        # 入力は content, style で (batch, 768, n_frame), (batch, 128) の各サイズを持つ。ただし batch > 1 は動作が非保証
        self.sess_f0n = ort.InferenceSession(
            self.f0n_ckpt, 
            providers  = self.onnx_provider_list,
            session_options = so,
        )
        test_f0n = True
        if test_f0n:
            _, _ = self.sess_f0n.run(
                ['pred_F0', 'pred_N'], 
                {
                    'content': self.buf_emb[:, :, -self.len_f0n_predictor:], 
                    'style': self.style_vect,
                },
            )
            time0 = time.perf_counter_ns() # time in nanosecond
            pred_F0, pred_N = self.sess_f0n.run(
                ['pred_F0', 'pred_N'], 
                {
                    'content': self.buf_emb[:, :, -self.len_f0n_predictor:], 
                    'style': self.style_vect,
                },
            )
            self.f0n_lap = (time.perf_counter_ns() - time0)/1e+6
            self.logger.debug(f"    - {str(list(self.buf_emb[:, :, -self.len_f0n_predictor:].shape))}, {str(list(self.style_vect.shape))} was converted {str(list(pred_F0.shape))} and {str(list(pred_N.shape))} in {self.f0n_lap: >7.2f} ms")
            self.logger.debug(f"    - (f0n RTF: {self.f0n_lap / (self.len_f0n_predictor*20): >7.4f})")


        self.logger.debug(f"Initializing the decoder...")
        self.sess_dec = ort.InferenceSession(
            self.decoder_ckpt, 
            providers  = self.onnx_provider_list,
            session_options = so,
        )
        test_vocoder = True
        if test_vocoder:
            _ = self.sess_dec.run(
                ['output'], 
                {
                    'content': self.buf_emb[:, :, -self.len_proc:],
                    'pitch': self.buf_f0_pred[:, -self.len_proc*2:],
                    'energy': self.buf_energy_pred[:, -self.len_proc*2:],
                    'style': self.style_vect,
                },
            )[0].squeeze(1)
            time0 = time.perf_counter_ns() # time in nanosecond
            tensor_recon = self.sess_dec.run(
                ['output'], 
                {
                    'content': self.buf_emb[:, :, -self.len_proc:],
                    'pitch': self.buf_f0_pred[:, -self.len_proc*2:],
                    'energy': self.buf_energy_pred[:, -self.len_proc*2:],
                    'style': self.style_vect,
                },
            )[0].squeeze(1)
            self.decode_lap = (time.perf_counter_ns() - time0)/1e+6
            self.logger.debug(f"    - {str(list(self.buf_emb[:, :, -self.len_proc:].shape))} ({self.len_proc / 50: >6.3f} sec) was converted {str(list(tensor_recon.shape))} in {self.decode_lap: >7.2f} ms.")
            self.logger.debug(f"    - (Decoder RTF: {self.decode_lap / (self.len_proc*20): >7.4f})")
        
        
        ####
        
        # デコードされた出力音声をバッファに貯める前に、デコーダの周波数から出力用周波数に変換しておく
        wav_o = librosa.resample(tensor_recon, orig_sr = self.sr_dec, target_sr = self.sc.sr_out, res_type = "polyphase")
        # さらに出力音声をスペクトログラムに変えて貯める際は、サンプリング周波数を 16k にしてから wav2spec
        wav_o_for_spec = librosa.resample(tensor_recon, orig_sr = self.sr_dec, target_sr = self.sr_proc, res_type = "polyphase")
        
        self.logger.debug(f"Decoded tensor: {tensor_recon.shape[1]} samples ({self.sr_dec} Hz) -> {wav_o.shape[1]} samples ({self.sc.sr_out} Hz) for output -> {wav_o_for_spec.shape} samples for spectrogram.")
        
        #### クロスフェード関係の変数

        # 最後に、出力音声のクロスフェード処理を作る。
        self.o_cross_kernel = make_cross_extra_kernel(
            (len(self.ch_map), self.sc.blocksize),
            extra = self.cross_fade_samples,
            divide = False,
        )
        # callback から返るべきサンプル長が、クロスフェードのサイズだけ水増しされている
        self.previous_output = np.zeros((len(self.ch_map), (self.sc.blocksize+self.cross_fade_samples))) 
        # blocksize を動的に変更するとカーネルの再作成が必要なので、そのための管理用フラグを作っておく
        self.need_remake_kernel = False

        # ContentVec にもクロスフェードをかけるカーネルを開発中に試したが、声がダブってしまう問題があるので止めた。

    ####

    def inference(
        self,
        in_block, # ここは backend の blocksize と厳密に一致している前提。2 ブロック同時に来た場合の反応は未検証
        skip: bool = False, # skip は inference の引数で、VC パートの重い処理をすっ飛ばして入力をそのまま出力。
        dBFS: float = -60, # 入力音声レベルに応じてスタイルを mix する機能に使う
    ):
        self.send_time0 = time.perf_counter_ns() # 現フレームの開始時刻

        # tensor_i は入力ブロックを float32 tensor に変換し (ch, time) の次元順に転置したデータ
        tensor_i = copy.deepcopy(in_block).astype(np.float32).transpose(1, 0) # (batch, blocksize)
        in_blocksize = tensor_i.shape[1] # この変数はデコード後の処理でも頻繁に使う

        # self.buf_wav_i には入力サンプリング周波数のまま格納。
        # 左にちょうど 1 ブロック分ロールしてから、古いデータが入った部分を最新のデータに置換する
        self.buf_wav_i = np.roll(self.buf_wav_i, -in_blocksize, axis = 1) # (1, 2097152)
        # 2 ch (0, 1) の入力を内部処理チャンネル数に投影する。デフォルトでは ch_map = [0] なので ch 0 だけ使用。
        for c_proc, c_i in enumerate(self.ch_map):
            self.buf_wav_i[c_proc, -tensor_i.shape[1]:] = copy.deepcopy(tensor_i[c_i, :])
        
        # 次に 16k に変換して、self.buf_wav_i16 に投入
        tensor_i16 = librosa.resample(tensor_i, orig_sr = self.sc.sr_out, target_sr = self.sr_proc, res_type = "polyphase")
        self.buf_wav_i16 = np.roll(self.buf_wav_i16, -tensor_i16.shape[1], axis = 1)
        for c_proc, c_i in enumerate(self.ch_map):
            self.buf_wav_i16[c_proc, -tensor_i16.shape[1]:] = copy.deepcopy(tensor_i16[c_i, :])

        # 前処理の遅延量を記録。 wav2spec は含んでいない
        self.pre_lap = (time.perf_counter_ns() - self.send_time0)/1e+6

        #### ここから VC パート。skip は inference の引数で、VC パートの重い処理をすっ飛ばして入力を出力に垂れ流す。

        # 入力信号スペクトログラムの計算は VC をスキップしない場合、もしくはリアルタイム更新が "always" の場合
#        if skip == False or self.spec_rt_i == 0:
        if ((skip == False and self.spec_rt_i == 1) and (self.absolute_pitch is False or self.estimate_energy is False)) or self.spec_rt_i == 0:
            time0 = time.perf_counter_ns()
            # HarmoF0 で正解ピッチを計算し、同時に Wav2spec してバッファに入れる。VC を適用する場合は省略できない
            real_F0, activation, real_N, spec_chunk = self.sess_HarmoF0.run(
                ['freq_t', 'act_t', 'energy_t', 'spec'], 
                {"input": self.buf_wav_i16[:, -self.len_w2m:]},
            ) # すべて 10 ms 解像度。spec_chunk は time last
            # spec, f0, energy, activation のバッファを更新する。ただし計算したチャンクを全て代入するか、最新部分だけか選ぶ
            self.buf_spec_p = np.roll(self.buf_spec_p, -self.sc.block_roll_size*2, axis = 2)
            self.buf_f0_real = np.roll(self.buf_f0_real, -self.sc.block_roll_size*2, axis = 1)
            self.buf_energy_real = np.roll(self.buf_energy_real, -self.sc.block_roll_size*2, axis = 1)
            self.buf_activation = np.roll(self.buf_activation, -self.sc.block_roll_size*2, axis = 1)
            if self.substitute_all_for_spec is True:
                self.buf_spec_p[:, :, -spec_chunk.shape[2]:] = spec_chunk
                self.buf_f0_real[:, -real_F0.shape[1]:] = real_F0
                self.buf_energy_real[:, -real_N.shape[1]:] = real_N
                self.buf_activation[:, -activation.shape[1]:] = activation
            else:
                self.buf_spec_p[:, :, -self.sc.block_roll_size*2:] = spec_chunk[:, :, -self.sc.block_roll_size*2:]
                self.buf_f0_real[:, -self.sc.block_roll_size*2:] = real_F0[:, -self.sc.block_roll_size*2:]
                self.buf_energy_real[:, -self.sc.block_roll_size*2:] = real_N[:, -self.sc.block_roll_size*2:]
                self.buf_activation[:, -self.sc.block_roll_size*2:] = activation[:, -self.sc.block_roll_size*2:]
            self.harmof0_lap = (time.perf_counter_ns() - time0)/1e+6

            # 以下はプロット用に、f0 の実測と予測を合わせたもの
            self.buf_f0_all = np.concatenate((self.buf_f0_real, self.buf_f0_pred), axis = 0)

        # ここからの工程は VC を適用する場合のみ必要
        if skip == False:
            # 16k buffer から ContentVec を計算する
            time0 = time.perf_counter_ns()
            # ContentVec は後端の情報が失われるため、発話を折り返した情報をでっち上げて計算する
            # これを入れずに「あーー」とか「おー」とか同じ音を長く続けると、音量がチャンクごとに減衰する |＼|＼|＼|＼ 
            if self.sc.content_expand_rate > 0:
                signal16 = self.buf_wav_i16[:, -self.len_embedder_input:]
                # 時間軸方向に反転した配列を作成
                signal16_rev = np.flip(signal16, axis = 1)[:, :int(self.len_embedder_input*self.sc.content_expand_rate)]
                # 元の配列と反転した配列を連結
#                signal16_cat = np.concatenate((signal16, copy.deepcopy(signal16_rev)), axis = 1)
                signal16_cat = np.concatenate((self.rand_input, signal16, copy.deepcopy(signal16_rev)), axis = 1)
                content0 = self.sess_CE.run(
                    ['last_hidden_state'], 
                    {'input': signal16_cat},
                )[0] # ["last_hidden_state"]
                content0 = content0.transpose(0, 2, 1)
                # concat して通した ContentVec から本来の部分だけに戻す
                content0 = content0[:, :, :self.len_embedder_output]
            else:
                content0 = self.sess_CE.run(
                    ['last_hidden_state'], 
                    {'input': self.buf_wav_i16[:, -self.len_embedder_input:]},
                )[0] # ["last_hidden_state"]
                content0 = content0.transpose(0, 2, 1)

            self.buf_emb = np.roll(self.buf_emb, -self.sc.block_roll_size, axis = 2) 
            # 経験上、ネットワークに通した全サンプルを使った方が音質が安定する
            if self.substitute_all_for_content is True:
                self.buf_emb[:, :, -content0.shape[2]:] = content0
            else:
                self.buf_emb[:, :, -self.sc.block_roll_size:] = content0[:, :, -self.sc.block_roll_size:]
            self.CE_lap = (time.perf_counter_ns() - time0)/1e+6

            # 話者スタイルの算出。出力は時間のない (batch, 128)
            time0 = time.perf_counter_ns()
            if self.auto_encode:
                self.style_vect = self.sess_SE.run(
                    ['output'], 
                    {'input': self.buf_spec_p[:, 48:, -self.len_style_encoder:][:, np.newaxis, :, :]},
                )[0]
            else:
                self.style_vect = self.sc.current_target_style # 他の GUI クラスから触るため、backend がスタイルを持つ
            self.SE_lap = (time.perf_counter_ns() - time0)/1e+6

            # f0n_predictor による F0 および energy の間接推定。入力に content + style vector が必要である。
            time0 = time.perf_counter_ns() # time in nanosecond
            if self.absolute_pitch is False and self.estimate_energy is False:
                self.need_pred_f0n = False
            else:
                self.need_pred_f0n = True
            # 必要な場合のみ、最新時点の pred_F0, pred_N を作成する。不要なら計算をパス
            if self.need_pred_f0n:
                pred_F0, pred_N = self.sess_f0n.run(
                    ['pred_F0', 'pred_N'], 
                    {
                        'content': self.buf_emb[:, :, -self.len_f0n_predictor:], 
                        'style': self.style_vect,
                    },
                )
                # バッファを更新。こちらは直接推定値と異なり、全部代入するか、roll 部分だけ代入するかで結果が変化する。
                self.buf_f0_pred = np.roll(self.buf_f0_pred, -self.sc.block_roll_size*2, axis = 1)
                self.buf_energy_pred = np.roll(self.buf_energy_pred, -self.sc.block_roll_size*2, axis = 1)
                if self.substitute_all_for_f0n_pred is True:
                    self.buf_f0_pred[:, -pred_F0.shape[1]:] = pred_F0
                    self.buf_energy_pred[:, -pred_N.shape[1]:] = pred_N
                else:
                    self.buf_f0_pred[:, -self.sc.block_roll_size*2:] = pred_F0[:, -self.sc.block_roll_size*2:]
                    self.buf_energy_pred[:, -self.sc.block_roll_size*2:] = pred_N[:, -self.sc.block_roll_size*2:]
            self.f0n_lap = (time.perf_counter_ns() - time0)/1e+6

            # 以下はプロット用に、f0 の実測と予測を合わせたもの
            self.buf_f0_all = np.concatenate((self.buf_f0_real, self.buf_f0_pred), axis = 0)

            # デコーダについても末尾を flip して入れてみたが、録音したサンプルが全く変わらないことが判明した。
            time0 = time.perf_counter_ns() # time in nanosecond
            if self.absolute_pitch:
                pitch_chunk = self.buf_f0_pred[:, -self.len_proc*2:] * 2**((self.pitch_shift) / 12)
            else:
                pitch_chunk = self.buf_f0_real[:, -self.len_proc*2:] * 2**((self.pitch_shift) / 12)
            if self.estimate_energy:
                energy_chunk = self.buf_energy_pred[:, -self.len_proc*2:]
            else:
                energy_chunk = self.buf_energy_real[:, -self.len_proc*2:]
            tensor_recon = self.sess_dec.run(
                ['output'], 
                {
                    'content': self.buf_emb[:, :, -self.len_proc:],
                    'pitch': pitch_chunk,
                    'energy': energy_chunk,
                    'style': self.style_vect,
                },
            )[0].squeeze(1)
            # デコーダの出力は resample が必要
            tensor_recon = librosa.resample(
                tensor_recon, orig_sr = self.sr_dec, target_sr = self.sc.sr_out, res_type = "polyphase",
            )
            self.decode_lap = (time.perf_counter_ns() - time0)/1e+6

        self.vc_end_time = time.perf_counter_ns() # time in nanosecond
        
        # skip と異なり bypass では VC の重い処理自体は行われるが、結果をバイパスして入力値を返す。
        if self.bypass or skip == True:
            tensor_recon = copy.deepcopy(self.buf_wav_i[:, -(self.sc.blocksize+self.cross_fade_samples):])
        
        
        # クロスフェード
        # 本来は周回ごとに以下のように信号が到着する
        # 前   |￣￣￣￣|
        # 今  ＿＿＿＿＿|￣￣￣￣|
        # 次 ＿＿＿＿＿＿＿＿＿＿|￣￣￣￣|
        
        # これらを滑らかに繋ぐため ／￣￣￣＼ のカーネルを掛ける。つまり self.cross_fade_samples の区間で上がり、
        # (self.sc.blocksize - self.cross_fade_samples) の plateau 後に、 self.cross_fade_samples の区間で下がる。
        # 前 ／￣￣￣＼ 
        # 今 ＿＿＿＿／￣￣￣＼
        #            |       | VC の戻りが、 self.cross_fade_samples だけ追加で遅延する。仕方ないね

        # なお、VC でデコードする長さは self.sc.blocksize + self.cross_fade_samples より長い必要がある

        if self.cross_fade_samples > 0:
            # もし cross fade のサイズが変更されていたら、カーネルを作り直す
            if self.need_remake_kernel or self.o_cross_kernel.shape[-1] != (self.sc.blocksize+self.cross_fade_samples):
                self.o_cross_kernel = make_cross_extra_kernel(
                    (len(self.ch_map), self.sc.blocksize), # (1, 6144)
                    extra = self.cross_fade_samples, # 512
                    divide = False,
                ) # カーネル全体の長さは (self.sc.blocksize + self.cross_fade_samples) となるので注意
                self.need_remake_kernel = False
            # クロスフェード用にカーネルを適用した再構成テンソルを作る。冒頭と中盤を現在の周回で、末尾を次の周回で使う。
            tensor_recon = tensor_recon[:, -(self.sc.blocksize+self.cross_fade_samples):] * self.o_cross_kernel # (1, 6656)
            
        # 出力音声は入力音声と同じ長さ、同じサンプリング周波数 (self.sc.sr_out) の buffer に貯める前提。
        # なお移動量は（sr_out の世界で）厳格に 1 block 分
        # 出力はいったんブロック全部をバッファに放り込み、backend に返すときにクロスフェードする。
        self.buf_wav_o = np.roll(self.buf_wav_o, -in_blocksize, axis = 1)
        # バッファのうち、ロールして最初から巻き戻ってきた部分の信号を消去する。
        self.buf_wav_o[:, -self.sc.blocksize:] *= 0.0
        # クロスフェード用カーネルが適用された信号を足す
        self.buf_wav_o[:, -tensor_recon.shape[1]:] += tensor_recon

        # 出力音声もスペクトログラムを計算する（ただし skip する周回では省略）
        if (skip == False and self.spec_rt_o == 1) or self.spec_rt_o == 0:
            # ただし表示タブが 0 つまり monitor のときだけ必要。いったん backend に戻らないと Frame を参照できない
            if self.sc.host.GetTopLevelParent().active_tab == 0:
                wav_o_for_spec = librosa.resample(
                    self.buf_wav_o[:, -(self.sc.blocksize+self.cross_fade_samples):], 
                    orig_sr = self.sc.sr_out, 
                    target_sr = self.sr_proc, 
                    res_type = "polyphase",
                )
                recon_F0, recon_act, recon_N, recon_spec = self.sess_HarmoF0.run(
                    ['freq_t', 'act_t', 'energy_t', 'spec'], 
                    {"input": wav_o_for_spec[:, -self.len_w2m:].astype(np.float32)},
                ) # time last
                self.buf_spec_o = np.roll(self.buf_spec_o, -self.sc.block_roll_size*2, axis = 2)
                if self.substitute_all_for_spec is True or recon_spec.shape[-1] < self.sc.block_roll_size*2:
                    self.buf_spec_o[:, :, -recon_spec.shape[2]:] = recon_spec
                else:
                    self.buf_spec_o[:, :, -self.sc.block_roll_size*2:] = recon_spec[:, :, -self.sc.block_roll_size*2:]
    
        # ラップタイムの計測
        self.post_lap = (time.perf_counter_ns() - self.vc_end_time)/1e+6 # Ryzen 3700X で 8--19 ms 程度（非コンパイル時）
        # vc_lap が実際の所要時間を表す指標
        self.vc_lap = (time.perf_counter_ns() - self.send_time0)/1e+6 
        # total_lap は「前のフレーム終了から現フレーム終了まで」なので、「VC 所要時間＋次のコールバックまでの待ち時間」
        self.total_lap = (time.perf_counter_ns() - self.total_end_time)/1e+6
        self.total_end_time = time.perf_counter_ns() 

        self.proc_head += in_blocksize
        self.retro_samples = int(0.05*self.sc.sr_out) # 再構成音声の最後の部分が低品質な恐れがあるため、過去部分を返す
        
        if self.cross_fade_samples > 0 or self.retro_samples > 0:
            return self.buf_wav_o[:, -(self.sc.blocksize+self.cross_fade_samples+self.retro_samples):-(self.cross_fade_samples+self.retro_samples)].T
        else:
            return self.buf_wav_o[:, -self.sc.blocksize:].T
        

    def __call__(
        self, 
        in_block,
    ):
        return self.inference(in_block)


    # (batch, time) の numpy array を読み込み、現在の変換設定に従って全体を変換する
    
    def convert_offline(
        self,
        tensor_i16,
    ):
        real_F0, activation, real_N, spec_chunk = self.sess_HarmoF0.run(
            ['freq_t', 'act_t', 'energy_t', 'spec'], 
            {"input": tensor_i16},
        )
        # 末尾（時間）次元を 4 の倍数に切り詰める
        spec_size_by_four = (spec_chunk.shape[-1] // 4) * 4

        content0 = self.sess_CE.run(
            ['last_hidden_state'], 
            {'input': tensor_i16},
        )[0] # ["last_hidden_state"]
        content0 = content0.transpose(0, 2, 1)
        # ここは現在ステレオ対応

        # 話者スタイルの算出。出力は時間のない (batch, 128)
        if self.auto_encode:
            style_vect = self.sess_SE.run(
                ['output'], 
                {'input': spec_chunk[:, np.newaxis, 48:, :spec_size_by_four]},
            )[0]
        else:
            style_vect = self.sc.current_target_style
        
        if style_vect.shape[0] == 1 and spec_chunk.shape[0] > 1:
            style_vect = np.tile(style_vect, (spec_chunk.shape[0], 1))
        elif style_vect.shape[0] != spec_chunk.shape[0]:
            style_vect = np.tile(style_vect[0, :], (spec_chunk.shape[0], 1))

        pred_F0, pred_N = self.sess_f0n.run(
            ['pred_F0', 'pred_N'], 
            {
                'content': content0, 
                'style': style_vect,
            },
        )

        if self.absolute_pitch:
            pitch_chunk = pred_F0 * 2**((self.pitch_shift) / 12)
        else:
            pitch_chunk = real_F0 * 2**((self.pitch_shift) / 12)

        if self.estimate_energy:
            energy_chunk = pred_N
        else:
            energy_chunk = real_N

        tensor_recon = self.sess_dec.run(
            ['output'], 
            {
                'content': content0,
                'pitch': pitch_chunk[:, -content0.shape[2]*2:],
                'energy': energy_chunk[:, -content0.shape[2]*2:],
                'style': style_vect,
            },
        )[0].squeeze(1)
        
        return np.clip(tensor_recon, -1, 1) # 出力は self.sr_dec こと 24k になる。

