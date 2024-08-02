#!/usr/bin/env python3

# The MIT License

# Copyright (c) 2024 Lyodos

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# 以下に定める条件に従い、本ソフトウェアおよび関連文書のファイル（以下「ソフトウェア」）の複製を取得するすべての人に対し、ソフトウェアを無制限に扱うことを無償で許可します。これには、ソフトウェアの複製を使用、複写、変更、結合、掲載、頒布、サブライセンス、および/または販売する権利、およびソフトウェアを提供する相手に同じことを許可する権利も無制限に含まれます。

# 上記の著作権表示および本許諾表示を、ソフトウェアのすべての複製または重要な部分に記載するものとします。

# ソフトウェアは「現状のまま」で、明示であるか暗黙であるかを問わず、何らの保証もなく提供されます。ここでいう保証とは、商品性、特定の目的への適合性、および権利非侵害についての保証も含みますが、それに限定されるものではありません。作者または著作権者は、契約行為、不法行為、またはそれ以外であろうと、ソフトウェアに起因または関連し、あるいはソフトウェアの使用またはその他の扱いによって生じる一切の請求、損害、その他の義務について何らの責任も負わないものとします。 

import sys
import numpy as np
import copy # 再代入を想定したインスタンス変数（mutable: list, dict, bytearray, set）は deepcopy で渡す必要がある。
import queue
import collections
import threading
import os
from datetime import datetime
from socket import gethostname
from hashlib import md5
import json

import logging
import inspect

from utils import to_dBFS, make_beep
from vc_engine import AudioEfx

# hi dpi 対応
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except:
    pass

import sounddevice as sd
import soundfile as sf

from audio_device_check import device_test_spawn, device_test_strict


class SoundControl:
    def __init__(
        self, 
        host, # このクラスを呼び出すときの親になる Frame を指定
        vc_config,
        api_pref: str = None, # 文字列 "ALSA" や "ASIO" など。
        # 最初は [入力, 出力] で準備していたが、PortAudio の仕様上、入出力に異なる API を使用しない
        generate_sine: bool = False, # True でテスト用の正弦波。下流の信号の途切れをテストするために使う。
        beep: bool = False, # True でテスト用の正弦波を 1 秒おきに 0.1 秒だけ鳴らす。時間ズレの検証に使う
        skip_always: bool = False, # 全てのサンプルを VC モデルに掛けずに素通しする。
        never_skip: bool = False, # 閾値以下の音声でも必ず VC を掛ける。絶対に身バレしたくない人向け
        bypass: bool = False, # skip と異なり、音声は入力から出力にそのまま流すが、 VC の処理負荷はかける。
        **kwargs,
    ):
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.logger.debug("Initializing ...")

        self.host = host
        self.vc_config = vc_config
        self.update_vc_config = self.host.update_vc_config # vc_config をアップデートするメソッド

        # 以下はテスト用の機能なので config に含めない
        self.generate_sine = generate_sine 
        self.beep = beep
        self.skip_always = skip_always
        self.never_skip = never_skip
        self.bypass = bypass

        #### 基本パラメータの定義
        
        # audio backend の blocksize は固定する。基本的には 128 が最も遅延が少ないが、VC だと短すぎて使いモノにならない。
        self.n_ch_proc =  self.vc_config["backend"]["n_ch_proc"]
        # block_roll_size のほうに値が指定される場合、実際の blocksize は sr_out から計算
        self.block_roll_size = self.vc_config["backend"]["block_roll_size"] 
        # blocksize を「hop = 20 ms の content を 1 回の VC 推論で何個分 roll するか」で決める
        self.blocksize = self.vc_config["backend"]["blocksize"] # 通常は None で初期化

        self.VC_threshold = self.vc_config["VC_threshold"]
        self.dispose_silent_blocks = self.vc_config["dispose_silent_blocks"] 
        self.keep_voiced = self.vc_config["keep_voiced"]

        # 音声サンプルを読み込む処理において、これよりも長い秒数は冒頭のみ処理
        self.sampler_max_sec = self.vc_config["sampler_max_sec"]
        # ファイルに対するオフライン VC における最大秒数。これを超えるサンプルは冒頭のみロードされる
        self.offline_max_sec = self.vc_config["offline_max_sec"]
        
        # 本当は以下の変数は VC の実行クラスに持たせるべき。ただし操作パネル側を同時に書き変える必要があるので、後で作業
        self.cross_fade_samples = self.vc_config["cross_fade_samples"]
        self.content_expand_rate = self.vc_config["content_expand_rate"] 
        
        self.mic_amp: float = self.vc_config["mic_amp"]
        self.sample_amp: float = 1.0
        
        # self.record_every が一定以上の値で record_output_audio == True が入る。
        # record_output_audio == True のとき、record_every 秒ごとに出力音声を切ってファイルに保存する。
        self.record_every = self.vc_config["record_every"]
        if self.record_every > 1e-5:
            self.record_input_audio = True # TODO: 現在、Ctrl + C すると終了処理に最後のバッファの保存が含まれない
            self.record_output_audio = True
        else:
            self.record_input_audio = False
            self.record_output_audio = False
        
        # 内部変数の初期化

        self.all_input_buffer = [] # 入力音声をひたすら貯める。ただし self.record_input_audio = True の場合のみデータが入る
        self.all_output_buffer = [] # 出力音声をひたすら貯める。ただし self.record_output_audio = True の場合のみデータが入る
        
        self.input_dBFS = self.VC_threshold - 10 # これは適当な初期値を入れているだけ
        self.output_dBFS = self.VC_threshold - 10

        self.mute = False # これは UI 経由で出力をミュートするボタンのための内部変数。出力信号に直接介入する
        self.first_time = True # stream を開始後、アプリが完全に立ち上がるまでの状態管理に使う
        self.is_voice = self.keep_voiced + 1 # 現在のフレームが voiced であるかどうか。keep_voiced = 1 ならここに 2 が入る
        self.vc_now: bool = False # 実際に現在 VC が掛かっている状態か。画面のタリー表示に使う
        self.offline_conversion_now: bool = False # 現在オフライン変換が走っている状態か。
        
        # ここから、別のインスタンスやプロセスと信号をやり取りするためのキューを定義
        self.queueA = collections.deque(maxlen = 2048) # A は backend 内の InputStream → (queue → OutputStream) で使用
        self.wq_input = queue.Queue() # wq_input は InputStream -> plot_waveform
        self.wq_output = queue.Queue() # wq_output は OutputStream -> plot_waveform
        self.queueP = queue.Queue() # P は sample player -> InputStream のミックス用信号

        # 現在の VC の変換先スタイル。ここでは問答無用でゼロ初期化し、 SampleManagerPanel の初期化時に書き換える。
        self.current_target_style = np.zeros((1, 128), dtype = np.float32) # ここが vc_engine から読まれる
        # 下のいずれかを毎フレーム反映させる
        self.candidate_style_list = [np.zeros((1, 128), dtype = np.float32)] * 3 # expanded, sampler, full-128
        self.style_mode = self.vc_config["style_mode"] # 最初は潜在空間からの expand で作るモード

        self.head_i: int = 0
        self.head_o: int = 0

        #### オーディオデバイス検査
        
        # 使用可能なオーディオデバイスを厳格にスキャンして一覧化しておく。新しいデバイスでアプリを起動する際に実行。
        # マシン名の一意なハッシュ値を名前に持つ json ファイルを削除すれば、次にアプリを起動する際に再スキャン
        # ただし現在、とりあえずマシンの既定音声デバイスでアプリを立ち上げるようになっている（立ち上げ後に変更は可能）
        self.machine_md5 = md5(gethostname().encode()).hexdigest()
        self.device_info_path = f"./configs/StrictDeviceInfo-{self.machine_md5}.json"

        # マシン名のハッシュが一致するチェック結果がなければ検査を実施。
        if not os.path.isfile(self.device_info_path):
            if str(os.name) == 'nt':
                self.strict_report = device_test_strict() # Windows は子プロセスの初期化が遅いので並列化はかえって不利
            else:
                self.strict_report = device_test_spawn() # Linux では対象デバイスごとにプロセスを分けて高速スキャンする
        else:
            self.logger.debug("Loading previous result of the strict device check...")
            with open(self.device_info_path, 'r') as handle:
                self.strict_report = json.load(handle)

        self.strict_avbl_i      = self.strict_report["strict_avbl_i"]
        self.strict_avbl_o      = self.strict_report["strict_avbl_o"]
        self.dev_strict_i_names = self.strict_report["dev_strict_i_names"]
        self.dev_strict_o_names = self.strict_report["dev_strict_o_names"]
        self.dev_strict_i_apis  = self.strict_report["dev_strict_i_apis"]
        self.dev_strict_o_apis  = self.strict_report["dev_strict_o_apis"]

        # 下で定義される scan メソッドを用い、sounddevice のデフォルトオーディオ設定をインスタンス変数に入れておく。
        self.scan(
            sr_proc = self.vc_config["backend"]["sr_proc"], # ここに入れた値を元に、self.sr_proc が self.scan 内で作成される
            sr_out = self.vc_config["backend"]["sr_out"], # デフォルトは None → self.scan 内で self.sr_out が作成される
            device = None, # init 時は device を指定しない
            n_ch_proc = self.vc_config["backend"]["n_ch_proc"], 
            n_ch_max = self.vc_config["backend"]["n_ch_max"],
            latency = self.vc_config["backend"]["latency"], # よほどの特殊事情がなければ "low"
            api_pref = api_pref,
        ) 
        # 実際に内部処理に回すチャンネル数を格納した変数 self.n_ch_in_use は self.scan 内部で作られる。
        # ただし [in, proc, out] の 3 要素を持つので、必要な要素だけ取り出して Stream に流し込む必要がある。
        
        # 通常は blocksize を blocksize 引数でなく block_roll_size で与える。
        # このとき blocksize を計算する処理が scan 以降、AudioEfx より前に必要
        if self.blocksize is None:
            if self.block_roll_size is not None:
                self.block_sec = self.block_roll_size*0.02 # 実時間で何秒おきに VC を呼び出すか
                self.blocksize = int(self.block_sec * self.sr_out) # 実際に backend を作るときの blocksize をここで確定
        else:
            pass # TODO 万が一、どちらも None だったときの処理をまだ考えていない
        self.need_remake_stream = False # 
        
        # dummy input として、長さが frames である正弦波を作る。開始フェーズは self.head_i で決まる
        self.data_sine = make_beep(
            sampling_freq = self.sr_out, # scan しないと作られない
            frequency = 440, # あくまで内部テスト用の機能なので config では指定できない
            beep_rate = 0.1, # 音が出るパートの割合
            level = 0.2, # beep 部分の音量
            n_channel = self.n_ch_in_use[0],
        )
        
        # AudioEfx のインスタンスは backend のインスタンス変数として定義し、使用時は output_stream 内部から逐次呼び出す。
        self.efx_control = AudioEfx(
            sc = self,
            vc_config = self.vc_config,
            hop_size = 160, # ここは使用する wav2spec のモジュールによって値が固定される。現在の仕様では 160
            dim_spec = 352, # ここは HarmoF0 の仕様で決まっている。
            ch_map = list(range(self.n_ch_in_use[1])), # 入力信号のどのチャンネルを、処理関数に流すかを決めるマップ
            bypass = self.bypass,
        ) 
        
        self.timestamp_at_start = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') # アプリ起動時刻

        # スキャン＆選択したデバイスで stream を初期化する
        
        self.input_stream = sd.InputStream(
            samplerate = self.sr_out, # scan() で作成される。in と out で同じサンプリング周波数を使うので名前は sr_out 
            device = self.dev_ids_in_use, # scan() で作成される
            # query_devices() で得られる max_input_channels ないし max_output_channels が設定可能な最大値。最小値は 1
            channels = self.n_ch_in_use[0], 
            # channels の値が n_channels と一致しないと outdata[:] = indata[:, mapping] で、
            # IndexError: index n is out of bounds for axis 1 with size n と叱られる
            dtype = 'float32', # 'float32' and 'float64' use [-1.0, +1.0]. 'uint8' is an unsigned 8 bit format
            blocksize = self.blocksize,
            latency = self.latency, # 単位は秒もしくは 'high' 'low'
            callback = self.input_callback,
            extra_settings = self.api_specific_settings,
        )
        
        self.output_stream = sd.OutputStream(
            samplerate = self.sr_out,
            device = self.dev_ids_in_use, 
            channels = self.n_ch_in_use[2], 
            dtype = 'float32',
            blocksize = self.blocksize,
            latency = self.latency,
            callback = self.output_callback,
            extra_settings = self.api_specific_settings,
        )

        # ここまでが __init__ の定義
        # ストリームの初回起動はメイン関数側に任せる。
        # input → output ではなく output → input の起動順でも動くが、遅延が 0.01 秒程度増える。


    ####
    
    # 入力用コールバック
    def input_callback(
        self,
        indata, 
        frames, 
        time, 
        status,
    ):
        if status:
            self.logger.info(status)

        # data_p は無音 + sample player
        data_p = np.zeros((frames, self.n_ch_in_use[0])) # self.n_ch_in_use[0] が入力 ch 数
        
        # こちらは、sample player から信号が来ている場合にミックスする。どんな場合でも高々 1 block
        if self.queueP.empty() is False:
            side_wav = self.queueP.get() # ここは wait が必要。でないとこちらの反応が速すぎてサンプルが落ちる
            if frames == side_wav.shape[0]:
                for i in list(range(self.n_ch_in_use[0])):
                    data_p[:, i] = side_wav[: , i] # n_ch_in_use[0] = 2 だったら i = 0, 1 の 2 チャンネル分
            else:
                self.logger.warning(f"({inspect.currentframe().f_code.co_name}) Sample player blocksize{side_wav.shape[0]} does not match the one of the audio backend {frames}")

        # どうやら mic input は断続的（indata の block が常には存在しない）らしい。
        # なので inputStream と outputStream を分離する設計の場合、indata を突っ込むだけだと音声が途切れる。
        if self.generate_sine or self.beep:
            data_send = (self.data_sine[:frames, :] + indata.copy()*10**(self.mic_amp/20)) + data_p*self.sample_amp
            data_send = np.clip(data_send, -1, 1)
            self.data_sine = np.roll(self.data_sine, -frames, axis = 0)
        else:
            # mic preamp はここに掛かる
            data_send = indata.copy()*10**(self.mic_amp/20) + data_p*self.sample_amp

        # VC 用の level 計算は mix 後に行うよう仕様変更した
        self.input_dBFS = to_dBFS(data_send)
        if self.input_dBFS > self.VC_threshold:
            self.is_voice = self.keep_voiced + 1 # フラグの値を「音声あり」としてリセット
        else:
            self.is_voice -= 1 # 閾値以下の信号レベルと判定されたら、ブロックごとに値を 1 ずつ下げる
            self.is_voice = max(0, self.is_voice) # ただし 0 が下限
            # 例 keep_voiced = 1 だったら、初期が 2、閾値以下になった直後のブロックで 1、次のブロックで 0 となって無音判定
        if self.head_o <= 0:
            self.is_voice = 0 # VC エンジンが立ち上がる前は常に無音判定
        
        # 音声ブロックに加え、音量レベルが閾値以上かどうかを、キューに乗せて流す
        self.queueA.append((data_send, copy.deepcopy(self.is_voice))) # # queueA は backend -> AudioEfx
        self.wq_input.put(data_send) # wq_input は backend -> plot_waveform

        self.head_i += frames # 入力がどこまで処理されたかのヘッド位置を進める

    # TODO 現在 mute は出力のカットに入っているが、実は input 側も介入させる方が安全
    # 遅延が極めて大きい時、「Mute ボタンを押した瞬間にマイクに入っていた音声」が復活しうるためである。
    # あるいは mute 処理に付随して、バッファの強制解放を入れておくか。

    ####
    
    # 出力用コールバック

    # dispose_silent_blocks を True にすると、room noise しか含まれていない（音量が VC_threshold 以下の）信号が到着した場合、
    # キューが空になる 1つ手前まで、もしくは音声が含まれている信号が来るまで audio_data を取り出し続け、
    # 途中を捨てて最後の block だけ出力に回す。つまり信号処理の遅延分を、声がない場所で回復運転して取り戻すことができる。

    # ただし、このオプションは現在 GUI からオンオフする機能を書いていない
    
    # ちなみに outdata を与えない場合、Windows では無音となって特に問題ないが Linux ではキーッという強烈なノイズが発生する
    
    def output_callback(
        self, 
        outdata, 
        frames, 
        time, 
        status,
    ):
        if status:
            self.logger.info(status)

        # 現在の VC の変換先スタイルを更新。
        # これはいわゆる音声処理ではないが、評価タイミングが各 callback の冒頭だと好都合なのでここにある
        self.current_target_style = self.candidate_style_list[int(self.style_mode)]

        audio_data = np.zeros((frames, self.n_ch_in_use[2]), dtype = 'float32')
        if len(self.queueA) > 0:
            audio_data, is_voice = self.queueA.popleft() # 最低 1 回は、最も古いキューを pop する操作が入る
            if self.dispose_silent_blocks:
                # 喋っていないときのサンプルを捨てるオプション
                while is_voice <= 0 and len(self.queueA) > 0:
                    audio_data, is_voice = self.queueA.popleft()
                    self.head_o += frames # 捨てたサンプル分ヘッドを進める
            
            # 以下で VC 推論関数にデータを投入する。ただし実際に処理するか（self.vc_now）は skip や bypass のフラグ依存
            if is_voice > 0:
                self.vc_now = False if self.skip_always or self.bypass else True
                result = self.efx_control.inference(audio_data, skip = self.skip_always, dBFS = self.input_dBFS)
            else:
                self.vc_now = False
                result = self.efx_control.inference(audio_data, skip = not self.never_skip, dBFS = self.input_dBFS)

            # stream の作り直し中であった場合は処理が変化する → ただし、厳密にはロジックがまだ完成していない
            if self.need_remake_stream is False or self.first_time:
                outdata[:] = result * (1 - int(self.mute)) # 緊急避難である mute はここで掛かる
            else:
                # blocksize を動的に変える操作中はサイズが不一致になるため、出力音声を無音としてでっちあげる
                # しかし sample player 側も措置が必要で、そちらのロジックが未完成なので現在 sample player が落ちる
                outdata[:] = np.zeros((frames, self.n_ch_in_use[2]), dtype = 'float32')

            # wq_output は output waveform plot
            self.wq_output.put(outdata[:, list(range(self.n_ch_in_use[2]))] * (1 - int(self.mute))) 

            # 入力音声を録音する機能 → 出力との比較でタイミングを揃えたいので、入力音声だが output_callback 側に実装した
            # ただし遅延量の厳密な測定には、input_callback 側に置いた方が便利なので、将来的に切り替え可能にしたい。
            if self.record_input_audio:
                self.all_input_buffer.append(audio_data.copy()) # 起動時から（もしくは捨てて以降）の入力音声のバッファに追加
                # バッファの長さが self.record_every を超えるごとに、ファイルに保存
                if len(self.all_input_buffer) >= self.record_every * self.sr_out // self.blocksize:
                    filename = f'i_{self.timestamp_at_start}_{self.head_o:011}.ogg'
                    threading.Thread(
                        target = sf.write, 
                        args = (
                            filename, 
                            np.concatenate(self.all_input_buffer), 
                            int(self.sr_out),
                        )
                    ).start()
                    self.all_input_buffer = []

            # 出力音声を録音する機能
            if self.record_output_audio:
                self.all_output_buffer.append(outdata.copy()) 
                if len(self.all_output_buffer) >= self.record_every * self.sr_out // self.blocksize:
                    filename = f'o_{self.timestamp_at_start}_{self.head_o:011}.ogg'
                    threading.Thread(
                        target = sf.write, 
                        args = (
                            filename, 
                            np.concatenate(self.all_output_buffer), 
                            int(self.sr_out),
                        )
                    ).start()
                    self.all_output_buffer = []

            self.output_dBFS = to_dBFS(outdata)
            self.head_o += frames
            
        elif self.head_i <= 0:
            outdata[:] = np.zeros((frames, self.n_ch_in_use[2])) # head_i が 0 つまり InputStream の稼働前は、無音を返す必要。
        else:
            pass
        
        if self.head_o > 0:
            self.first_time = False # このフラグ現実装は本当に「VC エンジンの準備完了」をとらえているのか？

    ####

    # オーディオデバイスをスキャンするメソッドの定義。コードの粒度的にはクラスを分けた方がいいかも
    # InputStream と OutputStream が分かれているので、両方とも手動でリセットが必要

    # sounddevice の機能を利用して、オーディオデバイスを（再）スキャンする。原則 None で決め打ちしたい項目だけ値を指定すること。
    def scan(
        self,
        sr_proc: int = None,
        sr_out: int = None, # 出力のサンプリング周波数は、特に指定がなければデフォルトに合わせる予定。
        device: list = None, # [0, 34] や [33, 33] のように、[入力, 出力] のデバイス番号（0 始まり）を指定。
        n_ch_proc: int = 1, # 内部処理用のチャンネル数を指定。基本的には int: 1
        n_ch_max: int = 2, # 入出力デバイスが多チャンネルサポートでも、指定したチャンネル数までしか使わない。
        latency = 'low',
        api_pref: str = None, # 文字列
        wasapi_exclusive: bool = False, # （"api_pref" == "Windows WASAPI" の場合のみ）排他モードを使用するか
        **kwargs,
    ):

        if sd.default.device[0] < 0:
            self.logger.warning(f"({inspect.currentframe().f_code.co_name}) Python 'sounddevice' cannot find the system default input. Connect a mic.")
            if sd.default.device[1] < 0:
                self.logger.warning(f"({inspect.currentframe().f_code.co_name}) Python 'sounddevice' cannot find the system default output. Connect an audio device.")
            sys.exit("Error: no audio device")

        # まず sd のクエリで音声入出力デバイスの辞書情報を一覧する
        self.dicts_dev_raw = sd.query_devices() # tuple であり専用クラスにも属する
        self.dev_names_all = [d['name'] for d in self.dicts_dev_raw]

        # 既知の問題
        # Linux 上で strict check を走らせると、pulseaudio 上の仮想 IO が有効なのに使用不可として誤判定されることがある。
        # 基本的にメインプロセス上で sd.query_devices() したときに返るデフォルトデバイスは「有効なはず」なので、
        # strict check の結果に関わらず self.dicts_dev_raw のデフォルトデバイスの ID は使用可に上書きする仕様にした。
        # この上書き処理は scan() ごとに、デバイスの API と名前でクエリを掛けて実施する必要がある（ID は変化しうる）。

        # オーディオ API ごとの「対応する音声デバイスの辞書」を、組み込みメソッドを用いて取得。
        self.dicts_apis = sd.query_hostapis()
        self.apis_installed = [x['name'] for x in self.dicts_apis] # 現在のシステムにインストールされている API の名称一覧
        # デフォルト API の選択は「デフォルトの出力ないし入力デバイスが持つ hostapi キーの値」
        self.default_api_id = self.dicts_dev_raw[sd.default.device[1]]["hostapi"] # 0 これは immutable

        # この段階では、インストールされているが対応デバイスがない API も表示される。たとえば Ubuntu の "OSS" 
        self.logger.debug(f"({inspect.currentframe().f_code.co_name}) All devices:\n{self.dicts_dev_raw}")
        self.logger.debug(f"({inspect.currentframe().f_code.co_name}) APIs installed: {self.apis_installed}")

        # 予めすべての API について、その内でのデフォルトデバイスおよび、実際に利用可能な推奨デバイスを指定。

        # self.force_enable とは、sd.query_hostapis() で返る「API ごとのデフォルトデバイス」が 
        # strict check に不合格だった場合に、そのデバイスを無理やり使用可能とマークし直すか？
        # とりあえず Linux （正確には ALSA のある環境）でのみ許可しておく。
        self.force_enable = True if "ALSA" in self.apis_installed else False # 厳密には pulseaudio の有無で判定すべき
        
        # 上記の正しい作法が分からんので、Linux の音声周りに詳しい人にレビューしてもらいたいところ

        self.default_ids_by_api = [] # [[1, 0], [10, 8]] # みたいなリストを API の数だけ連ねる
        self.apis_allowed = [] # [0, 1, 3, 4] # みたいな使用可能 API 番号だけを示すもの
        for i, d in enumerate(self.dicts_apis):
            # 現在の API でリストされている（必ずしも使用可能と限らない）デバイスの ID
            def_io = [d["default_input_device"], d["default_output_device"]]
            self.default_ids_by_api.append(def_io)
            if def_io[0] < 0:
                # 当該 API のためのデフォルト（入力）デバイスがないので、使用不能と判断
                self.logger.debug(f"({inspect.currentframe().f_code.co_name}) | * Default input device for '{d['name']}' is {def_io[0]}, but invalid")
                if def_io[1] < 0:
                    self.logger.debug(f"({inspect.currentframe().f_code.co_name}) | * Default output device for '{d['name']}' is {def_io[1]}, but invalid")
            else:
                self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Default '{d['name']}' devices are {def_io} (i/o)")
                # デバイス ID よりも名前でマッチングさせたい。
                def_io_names = [self.dicts_dev_raw[def_io[0]]['name'], self.dicts_dev_raw[def_io[1]]['name']]
                # これも初期にはデフォルトデバイスが定義されているならば使用可能な API と判断していたが、危険すぎる。
                enable_this_api = 0

                # strict な入力名称リストの中で、現在の API の入力であるものだけを抽出
                # 注意：strict check 後に API をインストール、アンインストールすると、ap == i によるマッチングはズレる。
                strict_name_list_by_api = [na for na, ap in zip(self.dev_strict_i_names, self.dev_strict_i_apis) if ap == i]
                if def_io_names[0] in strict_name_list_by_api:
                    # デフォルト入力は厳密な意味で使用可能
                    self.logger.debug(f"({inspect.currentframe().f_code.co_name}) |-- {def_io_names[0]} is available")
                    enable_this_api = enable_this_api + 1
                else:
                    # デフォルト入力デバイスが strict check で不合格だった場合
                    # self.force_enable が True である場合のみ有効化
                    if self.force_enable == True:
                        self.dev_strict_i_names.append(def_io_names[0])
                        self.dev_strict_i_apis.append(i)
                        self.strict_avbl_i[def_io[0]] = True
                        self.logger.debug(f"({inspect.currentframe().f_code.co_name}) |-- Adding default input for {d['name']}, '{def_io_names[0]}', to the allow-list")
                        enable_this_api = enable_this_api + 1
                    else:
                        self.logger.debug(f"({inspect.currentframe().f_code.co_name}) |-- Default input for {d['name']}, '{def_io_names[0]}', cannot be enabled")

                # strict な出力名称リストの中で、現在の API の出力であるものだけを抽出
                strict_name_list_by_api = [na for na, ap in zip(self.dev_strict_o_names, self.dev_strict_o_apis) if ap == i]
                if def_io_names[1] in strict_name_list_by_api:
                    self.logger.debug(f"({inspect.currentframe().f_code.co_name}) |-- {def_io_names[1]} is available")
                    enable_this_api = enable_this_api + 1
                else:
                    if self.force_enable == True:
                        self.dev_strict_o_names.append(def_io_names[1])
                        self.dev_strict_o_apis.append(i)
                        self.strict_avbl_o[def_io[1]] = True # デバイスの使用許可の bool list の値を書き換え
                        self.logger.debug(f"({inspect.currentframe().f_code.co_name}) |-- Adding default output for {d['name']}, '{def_io_names[1]}', to the allow-list")
                        enable_this_api = enable_this_api + 1
                    else:
                        self.logger.debug(f"({inspect.currentframe().f_code.co_name}) |-- Default output for {d['name']}, '{def_io_names[1]}', cannot be enabled")

                # 入出力ともデフォルトデバイスが使用可能と判断された場合のみ、当該 API を使用許可する
                if enable_this_api >= 2:
                    self.apis_allowed.append(i)
        self.logger.debug(f"({inspect.currentframe().f_code.co_name}) APIs allowed to use: {self.apis_allowed}")

        # ユーザーが選んだ、もしくはデフォルトの API を反映させて、対応するデバイスを選ぶ処理

        # self.api_pref は初期化時は「上で作った許可リストの先頭」の API 名だが、ユーザーが GUI で API を選んだ場合はその名称。
        # "MME" とか "ALSA" とかの単一文字列。
        self.api_pref = api_pref if api_pref is not None else self.dicts_apis[self.apis_allowed[0]]['name'] 

        # self.dev_ids_default はユーザーが選択した API についての、現在のシステム上でのデフォルトデバイス。
        # api_pref -> self.api_pref に基づいて self.dev_ids_default の値が書き換わるのでディープコピー必要

        # ユーザーが選択した API が利用可能ならば、その中でのデフォルトデバイスを選択。
        # ただしデフォルトデバイスが strict test で利用禁止と判明している場合、当該 API の使用を禁止する仕様とした。
        self.dev_ids_default = self.default_ids_by_api[self.apis_installed.index(self.api_pref)]
        # 選択した API で利用可能なデバイス ID。
        self.dev_avbl_on_api = copy.deepcopy(self.dicts_apis[self.apis_installed.index(self.api_pref)]["devices"]) # list

        # ストリームで実際に使うべきオーディオデバイス。ユーザー指定値が存在する場合、ここでオーバーライド
        # ここは deep copy を使わないといけない。
        if device is not None:
            self.dev_ids_in_use = copy.deepcopy(device)
        else: 
            self.dev_ids_in_use = copy.deepcopy(self.dev_ids_default)

        self.logger.debug(f"({inspect.currentframe().f_code.co_name}) User-specified API is '{self.api_pref}' where the device ids {self.dev_ids_in_use} (input, output) are used.")

        # 遅延量は常に 'low' でいいだろう。
        self.latency = latency if latency is not None else 'low'
        
        #### ここから下の機能は、「実際に選択されたデバイス self.dev_ids_in_use」に準拠して話がすすむ。

        # 選択したデバイスの辞書を取り出す。ただし Window には ["index"] つまり番号がないので工夫が必要だった。
        self.dict_i = self.dicts_dev_raw[self.dev_ids_in_use[0]] # 選択したデバイスの情報が記載された辞書
        self.dict_o = self.dicts_dev_raw[self.dev_ids_in_use[1]]

        # self.api_id_in_use は選択したデバイスで実際に使用される API で、self.api_pref のような文字列ではなく id 整数
        self.api_id_in_use = self.dict_i["hostapi"] # 入出力に違う API を使うケースはサポートしない
        
        # API 特異的な設定項目
        self.wasapi_exclusive = wasapi_exclusive
        if self.api_id_in_use == "Windows WASAPI":
            self.api_specific_settings = sd.WasapiSettings(exclusive = self.wasapi_exclusive)
        else: 
            self.api_specific_settings = None

        # サンプリング周波数の調整。In は常にデバイスのデフォルトに合わせる。
        # 注：現在のデバイスでサポートされているか否かの調査が必要。
        self.sr_proc = sr_proc # 内部処理用
        self.sr_out = sr_out if sr_out is not None else self.dict_o["default_samplerate"] # 出力用
        # sd のデフォルト周波数をセットする。
        self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Sampling rates for internal signal processing: {self.sr_proc} Hz, the output device: {self.sr_out} Hz. The rates CANNOT be changed while the application is running.")

        # ここからチャンネル数の調整

        # 選択されたデバイスが持つチャンネル No. の一覧
        self.map_i = list(range(0, self.dict_i['max_input_channels'], 1)) # [0, 1] 
        self.map_o = list(range(0, self.dict_o['max_output_channels'], 1)) # [0, 1, 2, 3, 4, 5]
        # 有効な入力 or 出力がない場合、継続不可能
        if len(self.map_i) <= 0:
            sys.exit("Error: the selected input device has no available channels")
        if len(self.map_o) <= 0:
            sys.exit("Error: the selected output device has no available channels")

        self.n_ch_proc = n_ch_proc if n_ch_proc is not None else 1 # 内部処理用のチャンネル数を指定。基本的には 1
        self.n_ch_max = n_ch_max if n_ch_max is not None else 2
        # 入出力デバイスが多チャンネルサポートでも、指定したチャンネル数までしか使わない。
        # self.n_ch_in_use が callback で内部処理に使うべきチャンネル数。[in, process, out] の 3 要素。
        # TODO 出力の ch 6 にマップしたい場合などもあるので、高度な設定としてルーティングを決め打ちできるようにしたい。
        self.n_ch_in_use = [
            min(self.n_ch_max, len(self.map_i)), 
            self.n_ch_proc, 
            min(self.n_ch_max, len(self.map_o)),
        ]
        self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Number of channels (in, process, out): {self.n_ch_in_use}")


    ####
    
    # （外部からの）デバイス device 再指定を受けて stream, callback を作り直す。
    # このデバイスには、InputStream, OutputStream を作り直す機能がある。
    # 入出力で利用可能なチャンネル数が変わることがあるので注意。
    def change_device(
        self,
        sr_proc: float,
        sr_out: float,
        device: list, 
        n_ch_proc: int,
        n_ch_max: int,
        api_pref: str,
        latency,
        **kwargs,
    ) -> None:

        self.scan(
            sr_proc = sr_proc, # self.sr_proc は scan 内部に再セット機能あり
            sr_out = sr_out, # self.sr_out は scan 内部に再セット機能あり
            device = device,
            n_ch_proc = n_ch_proc, 
            n_ch_max = n_ch_max,
            api_pref = api_pref,
            latency = latency, 
        )

        self.terminate()

        # いったん input と output の両ストリームを作り直している。
        # 本当は変更がある方だけ作り直すべきだが、信号が切れてもいいなら両方リセットして新規に作るほうが楽だろう。
        self.input_stream = sd.InputStream(
            samplerate = self.sr_out, 
            device = self.dev_ids_in_use, 
            channels = self.n_ch_in_use[0],
            dtype = 'float32',
            blocksize = self.blocksize, 
            latency = self.latency,
            callback = self.input_callback, 
            extra_settings = self.api_specific_settings,
        )
        self.output_stream = sd.OutputStream(
            samplerate = self.sr_out, 
            device = self.dev_ids_in_use, 
            channels = self.n_ch_in_use[2],
            dtype = 'float32',
            blocksize = self.blocksize, 
            latency = self.latency,
            callback = self.output_callback,
            extra_settings = self.api_specific_settings,
        )
        self.input_stream.start()
        self.output_stream.start()
        self.need_remake_stream = False # フラグを戻す

        self.logger.debug(f"({inspect.currentframe().f_code.co_name})  Latency settings (i/o) = {self.input_stream.latency} / {self.output_stream.latency}")

        # 既知の問題：無効な入力デバイスを指定するとプログラムが停止する。
        # Expression 'err' failed in 'src/hostapi/alsa/pa_linux_alsa.c', line: 3355
        # なお出力デバイスの場合、音は聞こえないがプログラムは停止しない。


    def terminate(
        self,
    ) -> None:
        self.input_stream.stop()
        self.output_stream.stop()
        self.input_stream.close() 
        self.output_stream.close() 
