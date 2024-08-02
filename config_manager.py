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
import json
import shutil
import logging
import inspect

# 注意

# 以下は「vc_config.json や app_config.json が見当たらないときに factory default の値を入れて（再）作成する」
# ための機能である。
# アプリケーションの挙動をカスタマイズしたいときは、このソースではなく config フォルダにある json ファイルを
# 書き換えること。


####

# 指定した名称の config ファイルがない場合、作成する
def load_make_app_config(
    file_path,
    debug: bool = False,
    save: bool = True,
):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        root_dict = {} # 空の辞書オブジェクトを作成

        # アプリケーション名
        root_dict["application_name"] = "MMCXLI"

        # ウィンドウの初期サイズ
        root_dict["window_size"] = (1280, 960)

        # ウィンドウの最小サイズ（ドラッグしてもこれ以下のサイズにならない）
        root_dict["window_min_size"] = (1200, 650)

        # インターフェース表示言語（ただし現在英語しか作っていない）
        root_dict["lang"] = "en"
        
        # 発話クリップを読み込んでスタイル埋め込みを計算するときの最大同時ロード数
        # TODO Sample load の場合は、なぜか 3 つ以上ロードすると動作が遅くなり、クラッシュの危険性。
        root_dict["max_slots"] = 8

        # 前回終了時に読み込んでいたスタイルファイルを自動で再ロードするよう試みる
        root_dict["restore_slot"] = True 
        
        # サンプルマネージャが管理するサンプル一覧の保存先パス
        root_dict["sample_portfolio_path"] = "./styles/sample_portfolio.json"
        
        # スタイルマネージャが管理するスタイル一覧の保存先パス
        root_dict["style_portfolio_path"] = "./styles/style_portfolio.json"

        # ContentVec の抽出結果をリアルタイムするタブを作るか？基本的にテスト用
        root_dict["display_content"] = False

        # 起動時のアクティブタブの番号（0 は monitor）
        root_dict["initial_active_tab"] = 0 


        # 設定をファイルとして保存。現在、保存先ファイル名はハードコーディングされている
        if save:
            try:
                with open(file_path, 'w') as f:
                    json.dump(root_dict, f, indent = 4)
                if debug:
                    logging.debug(f"  [load_make_vc_config] Application config was saved to '{file_path}'")
            except:
                if debug:
                    logging.debug("Failed to save the application config")

        return root_dict


####


# 初回起動時にこの関数を用いて vc_config.json ファイルが作成・保存される。以降は起動時にファイルから設定が読み込まれる。
# GUI上で変更された設定は「ファイル＞現在の VC 設定を上書き保存」で反映できる。
# vc_config.json ファイルを削除すると次回起動時、ここにある初期設定で再作成される。

# 指定した名称の config ファイルがない場合、作成する
def load_make_vc_config(
    file_path,
    debug: bool = False,
    save: bool = True, # 元ファイルがない場合、新規作成した json を書き出しておく。
):
    if os.path.exists(file_path):
        backup_file_path = file_path + '.bak'
        shutil.copyfile(file_path, backup_file_path) # ロードするときは必ず backup を作成する
        
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        root_dict = {} # 空の辞書オブジェクトを作成

        # 設定ファイルを開いた人への説明
        root_dict["description_en"] = "The settings are loaded upon startup. The changes made in the GUI are reflected with 'File > Save Current VC Settings'. You can delete this file and restore it with the factory default values on the next startup."
        
        #### backend
        
        root_dict["backend"] = {} # 空の辞書オブジェクトを作成

        # この変数は CUDA デバイスではなく、Python sounddevice のオーディオデバイスを意味する
        root_dict["backend"]["device"] = None
        # Python sounddevice のストリーム作成時に指定する latency 
        root_dict["backend"]["latency"] = "low" # ただし通常は "low" で決め打ちされる
        
        # Sounddevice の stream blocksize は、VC 推論の呼び出し間隔を決める重要なパラメータである。

        # blocksize は通常、VC 側の Content ロール量を使って定義する。1 単位が 20 ms に相当する
        root_dict["backend"]["block_roll_size"] = 7 
        
        # 実際に SoundControl 初期化に使う変数は blocksize （sample 数そのもの）だが、
        # これは None で指定しておいて、backend のサンプリング周波数が決まった後で内部的に計算する。
        root_dict["backend"]["blocksize"] = None # なお Python の None は json では自動で null にマップされる

        # Python sounddevice では通常は 128 の倍数を使うが、ロールの関係で無視する。
        # blocksize = 1536 で 遅延が知覚され始め 2048 だと明らか。
        # ちなみに 2048*3 は 44.1 kHz だと 140 ms であり、この場合 VC が概ね 130 ms を超えると音が切れまくる。
        # 呼び出し間隔を 2049*2 に上げると、ハイスペック PC でないと underrun が頻発するので厳しい。
        # 逆に 2049*4 だと pre/post のバッファ読み書きの遅延が顕著になる。
        
        # 最初は ProcessPoolExecutor 等を試したが、逆にオーバーヘッドが増える。
        # ただし input と output のストリームを分離しないと、処理が遅れた瞬間にエラーで止まるので、分離は必須。
        # ちなみに音声入出力デバイスを再スキャンした場合、self.sc を作り直すのではなく self.sc 内で stream を再起動する仕様。

        # api_pref は、scan で作成される。これを config に反映させるべきかは要検討。
        # 複数のマシンで設定ファイルを共有する場合に対応できなくなるためである。

        # ContentVec や HarmoF0 の入力 sr 
        root_dict["backend"]["sr_proc"] = 16000 # この値は内部ネットワークの仕様で固定されるので、一般ユーザーは弄らない
        # VC 部と audio backend の連結、そしてアプリ外とのやり取りにも使う sr （入出力ともに）
        # 通常は sr_out = None としてデバイス側デフォルトに任せるが、必要な場合は 44100 や 48000 等の値を入れてもいい
        root_dict["backend"]["sr_out"] = None 
        # VC モデルにおいて vocoder が返す音声のサンプリング周波数（sr_out とは必ずしも一致しない）
        root_dict["backend"]["sr_decode"] = 24000 # この値は vocoder の現在の仕様で固定されるので、一般ユーザーは弄らない
        
        # VC 処理の音声チャンネルの数。VC の設計上は多チャンネル化できるが通常はモノラル
        root_dict["backend"]["n_ch_proc"] = 1 
        # 入出力デバイスが多チャンネル対応（例: 32 ch）でも、このチャンネル数までしか扱わない
        root_dict["backend"]["n_ch_max"] = 2 
        
        # 原理上は複数マイクの声を、それぞれ異なるターゲット話者スタイルに向けて VC して返すようなルーティングも可能だが、
        # きわめて処理が面倒なのでいったん考えないことにする。

        #### model settings

        root_dict["model"] = {} # 空の辞書オブジェクトを作成

        root_dict["model"]["model_device"] = "cuda" # GPU を決め打ちする場合は "cuda:0" や "cuda:1" など。あるいは "cpu"
        # TODO 現在 "cuda:2" 以降の GPU デバイスを決め打ちする処理が実装されていない。が、そこまで必要かな？

        # 実のところ Nuitka の --onefile でバンドルさせないためには、絶対パスにしたほうがいいのかもしれない？
        root_dict["model"]["harmof0_ckpt"] = "./weights/harmof0.onnx"
        root_dict["model"]["CE_ckpt"] = "./weights/hubert500.onnx"
        root_dict["model"]["f0n_ckpt"] = "./weights/f0n_predictor_hubert500.onnx"
        root_dict["model"]["SE_ckpt"] = "./weights/style_encoder_304.onnx"
        root_dict["model"]["decoder_ckpt"] = "./weights/decoder_24k.onnx"
        root_dict["model"]["style_compressor_ckpt"] = "./weights/pumap_encoder_2dim.onnx"
        root_dict["model"]["style_decoder_ckpt"] = "./weights/pumap_decoder_2dim.onnx"

        # harmoF0 で使用する周波数の最低最高値。元々は PyTorch のモデル内に定義されていたが ONNX 化で情報を取れなくなった
        root_dict["spec_fmin"] = 27.5
        root_dict["spec_fmax"] = 4371.3394 # 27.5*2^(351/48) = 4371.3394

        #### vc
        
        # マイク入力のプリアンプを画面で設定可能な範囲。単位は dB
        root_dict["mic_amp"] = 0.0 # 0.0 で事前調整なし（1 倍）
        root_dict["mic_amp_range"] = (-20, 20)

        # 入力信号の block 内の平均 dBFS 値がこれ以下の場合、VC を適用しない
        root_dict["VC_threshold"] = -40.0
        # ただし音量が小さくても、activation の値がこの閾値を超える（喋りが含まれている）場合は skip せずに VC を実行する
        root_dict["activation_threshold"] = 0.7

        # VC_threshold 以下のルームノイズを、OutputStream で一部捨てる
        root_dict["dispose_silent_blocks"] = False
        # VC_threshold を下回ってからも、ここに決めた block 数は voiced と判定し続ける
        root_dict["keep_voiced"] = 1
        
        # 入力信号をリアルタイムでスペクトログラム変換するか？# ["always", "with VC"] = [0, 1]
        root_dict["spec_rt_i"] = 1
        # 出力信号をスペクトログラム変換するか？# ["always", "with VC", "none"] = [0, 1, 2]
        root_dict["spec_rt_o"] = 1

        # 以下は ContentVec を適用するときの、後端の折り返し量。この量は buffer size に依存しないが、計算時間に影響を及ぼす
        # 0.0 で折り返しなし。負値は無効。上限は 1（元の信号より長く折り返せない） 
        root_dict["content_expand_rate"] = 0.1

        # True であれば VC 先話者のスタイルを入力音声から計算（オウム返しになる）False ならば既知のスタイルを与えて VC する
        root_dict["auto_encode"] = False 

        # ソース音声からの相対音高（harmoF0 + shift）を使うか、ターゲット話者スタイルから推測する「絶対音高モード」を使うか
        root_dict["absolute_pitch"] = True
        # 後者の方が計算は重いが、新規スタイルの作成時はこちらにしないとスタイル固有の音高の目安が分からない。
        # 前者は自分の声からのピッチシフトを厳密に反映したい場合、たとえば歌唱時に使う

        # 半音いくつ分ピッチを変えるか。上で「絶対音高モード」を選んだ場合、ターゲット話者スタイルの音高にさらに加算される
        root_dict["pitch_shift"] = 0.0 
        # ピッチ変更を画面で設定可能な範囲。単位は半音
        root_dict["pitch_range"] = (-18, 18)

        # 音量を推定するか、入力音声のスペクトログラムから単純計算した power を使うか
        root_dict["estimate_energy"] = False # 

        # 変換先話者スタイルの選択。
        root_dict["style_mode"] = 0 # ('2-dim', 'Sample', 'Full-128')

        #### vc buffer 
        
        # 入力から受け取った waveform を溜めておくバッファの長さ（単位：秒）
        root_dict["sec_wav_buffer"] = 16.0

        # hop = 10 ms のスペクトログラムを、1 回の inference で何フレーム分計算するか。すなわち 100 フレームで 1 秒
        root_dict["len_spec"] = 20 # ただし整数、できれば 10 の倍数で指定すること
        # 入力から受け取って spectrogram に変換したデータを溜めておくバッファのフレーム長。
        root_dict["n_buffer_spec"] = 400

        # 何フレーム分のサンプルを、ContentVec に突っ込むか。Decoder と同じ 1 frame = 20 ms で計算
        # ContentVec の 1 回の計算あたり何フレーム分を得るか。Decoder と同じ 1 frame = 20 ms 換算
        root_dict["len_content"] = 100 
        
        # hop = 10 ms の Spec をいくつ放り込んで style vector を作るか。最低 80
        # 話者スタイルを計算する spectrogram フレーム長。Conv2d の制約があり最低 80（0.8 秒）
        root_dict["len_style_encoder"] = 200

        # F0, energy を推定する content（20 ms / frame、50 frames で 1s）
        root_dict["len_f0n_predictor"] = 80 # hop = 20 ms の content をいくつ放り込んで pitch, energy を計算するか。

        # VC decoder に放り込むべき content フレーム数（20 ms / frame すなわち 50 frames で 1 秒）
        root_dict["len_proc"] = 30
        
        # backend に返すときに前のイテレーションの復元音声とクロスフェードする量
        root_dict["cross_fade_samples"] = 352

        # 閾値で skip する場合は、ここは False の方が整合性が取れる
        # 入力を spectrogram, f0 に変えてバッファに追加するとき、計算分を全部代入するか
        root_dict["substitute_all_for_spec"] = True 
        # ContentVec をバッファに追加するとき、計算分を全部代入するか
        root_dict["substitute_all_for_content"] = True 
        # f0n の推定値を～
        root_dict["substitute_all_for_f0n_pred"] = False 
        
        # Sampler の音声サンプルをロードする処理における最大秒数。これを超えるサンプルは冒頭のみロードされる
        root_dict["sampler_max_sec"] = 16.0

        # ファイルに対するオフライン VC における最大秒数。これを超えるサンプルは冒頭のみロードされる
        root_dict["offline_max_sec"] = 30.0
        
        #### log
        
        # 0 より大きな数であれば、起動時からの音声を秒数ごとにカレントディレクトリに録音
        # TODO: 保存先フォルダの設定
        root_dict["record_every"] = 0.0 # 単位は秒

        #### style
        
        # # 圧縮埋め込みの値の範囲
        # root_dict["comp_v_range"] = [-50, 50] 
        
        style_dict = {} # 空の辞書オブジェクトを作成

        root_dict["style"] = style_dict # 実際にスタイルが入るのはもっと後。あるいは別の style_config に独立させる方向で


        # 設定をファイルとして保存。現在の仕様では保存先ファイル名はハードコーディングされている
        if save:
            try:
                with open(file_path, 'w') as f:
                    json.dump(root_dict, f, indent = 4)
                if debug:
                    logging.debug(f"  [load_make_vc_config] VC config was saved to '{file_path}'")
            except:
                if debug:
                    logging.debug("Failed to save VC config")

        return root_dict

