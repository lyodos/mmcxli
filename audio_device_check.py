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
import os
from contextlib import redirect_stdout
import multiprocessing
from itertools import compress

import logging
import inspect

from socket import gethostname
from hashlib import md5
import json

import sounddevice as sd

# デバイスの出入力が有効か否かの厳格なテスト。実際にストリームを作って信号を流してみる

# 現状の問題：そもそも Device unavailable [PaErrorCode -9985] の状態でも、非対応のサンプリング周波数で
# テストを掛けると Invalid sample rate [PaErrorCode -9997] が返ってしまう。

def device_test_strict():
    # Linux の ALSA ではエラーが波及して本体のストリーム有効化に失敗してしまうことがあるため、別プロセスに投げた。
#    print("\n  Inspecting audio input/output devices on the current system...")
    logging.debug("(device_test_strict) Inspecting audio input/output devices on the current system...")

    parent_conn_44, child_conn_44 = multiprocessing.Pipe()
    parent_conn_48, child_conn_48 = multiprocessing.Pipe()
    p_44 = multiprocessing.Process(target = device_test_child_process, args = (44100, True, child_conn_44))
    p_48 = multiprocessing.Process(target = device_test_child_process, args = (48000, True, child_conn_48))

    p_44.start() # 別プロセスで開始
    p_48.start() # 別プロセスで開始
    p_44.join() # 終了まで待機
    p_48.join() # 終了まで待機

    _44_names, _44_apis, strict_avbl_44_i, strict_avbl_44_o = parent_conn_44.recv()
    _48_names, _48_apis, strict_avbl_48_i, strict_avbl_48_o = parent_conn_48.recv()
    strict_avbl_sum_i = [x + y for x, y in zip(strict_avbl_44_i, strict_avbl_48_i)]
    strict_avbl_sum_o = [x + y for x, y in zip(strict_avbl_44_o, strict_avbl_48_o)]

    p_44.terminate()
    p_48.terminate()

    logging.debug(f"    - Strictly available input devices  (44.1 kHz): {strict_avbl_44_i}")
    logging.debug(f"    - Strictly available input devices  (48   kHz): {strict_avbl_48_i}")
    logging.debug(f"    - Strictly available input devices    (code)  : {strict_avbl_sum_i}")
    logging.debug(f"    - Strictly available output devices (44.1 kHz): {strict_avbl_44_o}")
    logging.debug(f"    - Strictly available output devices (48   kHz): {strict_avbl_48_o}")
    logging.debug(f"    - Strictly available output devices   (code)  : {strict_avbl_sum_o}")

    strict_avbl_i = [x > -4 for x in strict_avbl_sum_i]
    strict_avbl_o = [x > -4 for x in strict_avbl_sum_o]
    dev_strict_i_names = list(compress(_44_names, strict_avbl_i)) # 許可名称
    dev_strict_i_apis  = list(compress(_44_apis,  strict_avbl_i)) # 上 API
    dev_strict_o_names = list(compress(_44_names, strict_avbl_o))
    dev_strict_o_apis  = list(compress(_44_apis,  strict_avbl_o))

    dict = {
        "strict_avbl_i": strict_avbl_i, 
        "strict_avbl_o": strict_avbl_o, 
        "dev_strict_i_names": dev_strict_i_names, 
        "dev_strict_o_names": dev_strict_o_names, 
        "dev_strict_i_apis": dev_strict_i_apis , 
        "dev_strict_o_apis": dev_strict_o_apis,
    }

    machine_name = gethostname()
    machine_md5 = md5(machine_name.encode()).hexdigest()

    with open("./configs/StrictDeviceInfo-" + machine_md5 + ".json", 'w') as handle:
        json.dump(dict, handle, indent = 4)

    return(dict)


def device_test_child_process(
    samplerate: int,
    showPaErrorCodes: bool,
    conn, # 結果を格納して親プロセスから取り出すための pipe connection
) -> list:

    dicts_dev_raw = sd.query_devices()
    dev_names = [d["name"] for d in dicts_dev_raw]
    dev_apis = [d["hostapi"] for d in dicts_dev_raw]
    # self.dicts_dev_raw に全デバイスの辞書を格納済みとする
    max_ch_i = []
    max_ch_o = []
    for i in list(range(len(dicts_dev_raw))):
        max_ch_i.append(dicts_dev_raw[i]["max_input_channels"])
        max_ch_o.append(dicts_dev_raw[i]["max_output_channels"])

    strict_avbl_i = []
    strict_avbl_o = []

    def test_input_callback(indata, frames, time, status) -> None:
        if status:
            logging.debug(status)

    def test_output_callback(outdata, frames, time, status) -> None:
        if status:
            logging.debug(status)
        outdata[:, 0] = list((x - 0.5) * 0.2 for x in np.random.rand(frames))
    
    # input 
    for i, nch in enumerate(max_ch_i):
        if nch > 0:
            try:
                sd.check_input_settings(device = i, samplerate = samplerate)
                _ = sd.InputStream(
                    samplerate = samplerate,
                    device = i,
                    callback = test_input_callback,
                )
                strict_avbl_i.append(1)
            except Exception as e:
                strict_avbl_i.append(-2)
                if showPaErrorCodes:
                    logging.debug(f"  * Device {i}, input, {samplerate} Hz: {e}")
                else:
                    with redirect_stdout(open(os.devnull, 'w')):
                        print(e, file = sys.stdout)
        else:
            logging.debug(f"  - Input {i}: 0 channels available")
            strict_avbl_i.append(0)

    logging.debug(f"  - Input devices (strict): {strict_avbl_i}")

    # output
    for i, nch in enumerate(max_ch_o):
        if nch > 0:
            try:
                sd.check_output_settings(device = i, samplerate = samplerate)
                _ = sd.OutputStream(
                    samplerate = samplerate,
                    device = i,
                    callback = test_output_callback,
                )
                strict_avbl_o.append(1)
            except Exception as e:
                strict_avbl_o.append(-2)
                if showPaErrorCodes:
                    logging.debug(f"  * Device {i}, output, {samplerate} Hz: {e}")
                else:
                    with redirect_stdout(open(os.devnull, 'w')):
                        print(e, file = sys.stdout)
        else:
            logging.debug(f"  - Output {i}: 0 channels available")
            strict_avbl_o.append(0)

    logging.debug(f"  - Output devices (strict): {strict_avbl_o}")

    # 実際に対応することが確認されたデバイスの id を、
    ids_strict_i = []
    ids_strict_o = []
    for i, b in enumerate(strict_avbl_i):
        if b > 0:
            ids_strict_i.append(i)
    for i, b in enumerate(strict_avbl_o):
        if b > 0:
            ids_strict_o.append(i)

    conn.send((dev_names, dev_apis, strict_avbl_i, strict_avbl_o))


####

# 以下は Linux （正確には Windows 以外の環境）で使うデバイス検査関数

# 実はデバイスごとのストリーミングテストも、孫プロセスに分離する必要があるっぽい。
# そうしないと、最初のデバイスのテスト時のエラーが次のデバイスの立ち上げを邪魔して、以降が全部 False になる。

# 既知の問題：pulseaudio のデフォルトデバイスでは  
# Error opening InputStream: Illegal combination of I/O devices [PaErrorCode -9993]
# つまり

# なお正式な Python の作法では、spawn する子プロセスが内部関数ではいけない。
# AttributeError: Can't pickle local object 'device_test_spawn.<locals>.test_input_grandson'
# しかし内部関数としても、なぜか Linux では動作した。

def test_input_grandson(i, nch, samplerate, conn):

    # 以下は子プロセスで発生する ALSA 関係の stderr をヌルっと捨てる
    devnull = open(os.devnull, 'w')
    os.dup2(devnull.fileno(), 2)  # stderrを/dev/nullにリダイレクト
    devnull.close()
    
    from sounddevice import InputStream

    if nch > 0:
        try:
            def _input_callback(indata, frames, time, status) -> None:
                if status:
                    with redirect_stdout(open(os.devnull, 'w')):
                        print(status, file = sys.stdout)
            
            test_stream = InputStream(
                samplerate = samplerate,
                device = i, 
                channels = nch, 
                callback = _input_callback,
            )
            test_stream.close() 
            logging.debug(f"  - Device {i} (input), {samplerate} Hz: OK")
            conn.send(1)
        
        except Exception as e:
            logging.debug(f"  * Device {i} (input), {samplerate} Hz: {str(e)}")
            conn.send(-2)
    
    else:
        logging.debug(f"  - Device {i} (input), {samplerate} Hz: no channels available")
        conn.send(0)


def test_output_grandson(i, nch, samplerate, conn):

    devnull = open(os.devnull, 'w')
    os.dup2(devnull.fileno(), 2) 
    devnull.close()
    
    from sounddevice import OutputStream

    if nch > 0:
        try:
            def _output_callback(outdata, frames, time, status) -> None:
                if status:
                    with redirect_stdout(open(os.devnull, 'w')):
                        print(status, file = sys.stdout)
#                        print(status, file = sys.stderr)
                outdata[:, 0] = list(0.2 * (x - 0.5) for x in np.random.rand(frames))
            
            test_stream = OutputStream(
                samplerate = samplerate,
                device = i, 
                channels = nch, 
                callback = _output_callback,
            )
            test_stream.close() 
            logging.debug(f"  - Device {i} (output), {samplerate} Hz: OK")
            conn.send(1)
        
        except Exception as e:
            logging.debug(f"  * Device {i} (output), {samplerate} Hz: {str(e)}")
            conn.send(-2)
    
    else:
        logging.debug(f"  - Device {i} (output), {samplerate} Hz: no channels available")
        conn.send(0)


####

def device_test_spawn() -> list:
    logging.debug("(device_test_spawn) Inspecting audio devices on the current system...")

    dicts_dev_raw = sd.query_devices()
    dev_names = [d["name"] for d in dicts_dev_raw]
    dev_apis = [d["hostapi"] for d in dicts_dev_raw]
    # 現在認識されているデバイスごとに、最大入出力数を記録しておく。
    max_ch_i = []
    max_ch_o = []
    for i in list(range(len(dicts_dev_raw))):
        max_ch_i.append(dicts_dev_raw[i]["max_input_channels"])
        max_ch_o.append(dicts_dev_raw[i]["max_output_channels"])
    logging.debug(f"({inspect.currentframe().f_code.co_name}) Maximum input channel sizes are {max_ch_i}")
    logging.debug(f"({inspect.currentframe().f_code.co_name}) Maximum output channel sizes are {max_ch_o}")

    # チャンネル数 0 のデバイスに対して子プロセスを作るのは無駄なので、ここを改善する。
    # だいたい半分のリソースで済むようになる。

    connections = [tuple(multiprocessing.Pipe()) for x in max_ch_i]
    logging.debug("Starting child processes to inspect the condition of '44100 Hz, input'")
    process_list_44_i = []
    result_list_44_i = [-2] * len(max_ch_i)
    for i, nch in enumerate(max_ch_i):
        if nch > 0:
            process_list_44_i.append(
                multiprocessing.Process(
                    target = test_input_grandson, 
                    args = (i, nch, 44100, connections[i][1]),
                )
            )
            process_list_44_i[i].start() # 別プロセスで開始
        else:
            process_list_44_i.append([]) # 要素数がずれないように、空リストを挿入

    for i, nch in enumerate(max_ch_i):
        if nch > 0:
            process_list_44_i[i].join() # 終了まで待機
            result_list_44_i[i] = connections[i][0].recv() # recv() は 1 回しか評価できない。
            process_list_44_i[i].terminate()
        elif nch == 0:
            result_list_44_i[i] = 0
        else:
            pass
    logging.debug("All child processes of '44100 Hz, input' ended")


    logging.debug("Starting child processes to inspect the condition of '48000 Hz, input'")
    process_list_48_i = []
    result_list_48_i = [-2] * len(max_ch_i)
    for i, nch in enumerate(max_ch_i):
        if nch > 0:
            process_list_48_i.append(
                multiprocessing.Process(
                    target = test_input_grandson, 
                    args = (i, nch, 48000, connections[i][1]),
                )
            )
            process_list_48_i[i].start()
        else:
            process_list_48_i.append([])

    for i, nch in enumerate(max_ch_i):
        if nch > 0:
            process_list_48_i[i].join()
            result_list_48_i[i] = connections[i][0].recv()
            process_list_48_i[i].terminate()
        elif nch == 0:
            result_list_48_i[i] = 0
        else:
            pass
    logging.debug("All child processes of '48000 Hz, input' ended")

    # 出力も 44o と 48o ごとに調査

    logging.debug("Starting child processes to inspect the condition of '44100 Hz, output'")
    process_list_44_o = []
    result_list_44_o = [-2] * len(max_ch_o)
    for i, nch in enumerate(max_ch_o):
        if nch > 0:
            process_list_44_o.append(
                multiprocessing.Process(
                    target = test_output_grandson, 
                    args = (i, nch, 44100, connections[i][1]),
                )
            )
            process_list_44_o[i].start()
        else:
            process_list_44_o.append([])

    for i, nch in enumerate(max_ch_o):
        if nch > 0:
            process_list_44_o[i].join()
            result_list_44_o[i] = connections[i][0].recv()
            process_list_44_o[i].terminate()
        elif nch == 0:
            result_list_44_o[i] = 0
        else:
            pass
    logging.debug("All child processes of '44100 Hz, output' ended")


    logging.debug("Starting child processes to inspect the condition of '48000 Hz, output'")
    process_list_48_o = []
    result_list_48_o = [-2] * len(max_ch_o)
    for i, nch in enumerate(max_ch_o):
        if nch > 0:
            process_list_48_o.append(
                multiprocessing.Process(
                    target = test_output_grandson, 
                    args = (i, nch, 48000, connections[i][1]),
                )
            )
            process_list_48_o[i].start()
        else:
            process_list_48_o.append([])

    for i, nch in enumerate(max_ch_o):
        if nch > 0:
            process_list_48_o[i].join()
            result_list_48_o[i] = connections[i][0].recv()
            process_list_48_o[i].terminate()
        elif nch == 0:
            result_list_48_o[i] = 0
        else:
            pass
    logging.debug("All child processes of '48000 Hz, output' ended")

    result_list_sum_i = [x + y for x, y in zip(result_list_44_i, result_list_48_i)]
    result_list_sum_o = [x + y for x, y in zip(result_list_44_o, result_list_48_o)]

    logging.debug(f"({inspect.currentframe().f_code.co_name}) Strictly available input devices  (44.1 kHz): {result_list_44_i}")
    logging.debug(f"({inspect.currentframe().f_code.co_name}) Strictly available input devices  (48   kHz): {result_list_48_i}")
    logging.debug(f"({inspect.currentframe().f_code.co_name}) Strictly available input devices    (code)  : {result_list_sum_i}")
    logging.debug(f"({inspect.currentframe().f_code.co_name}) Strictly available output devices (44.1 kHz): {result_list_44_o}")
    logging.debug(f"({inspect.currentframe().f_code.co_name}) Strictly available output devices (48   kHz): {result_list_48_o}")
    logging.debug(f"({inspect.currentframe().f_code.co_name}) Strictly available output devices   (code)  : {result_list_sum_o}")

    strict_avbl_i = [x > -4 for x in result_list_sum_i]
    strict_avbl_o = [x > -4 for x in result_list_sum_o]
    dev_strict_i_names = list(compress(dev_names, strict_avbl_i)) # 許可名称
    dev_strict_i_apis  = list(compress(dev_apis,  strict_avbl_i)) # 上 API
    dev_strict_o_names = list(compress(dev_names, strict_avbl_o))
    dev_strict_o_apis  = list(compress(dev_apis,  strict_avbl_o))

    dict = {
        "strict_avbl_i": strict_avbl_i, 
        "strict_avbl_o": strict_avbl_o, 
        "dev_strict_i_names": dev_strict_i_names, 
        "dev_strict_o_names": dev_strict_o_names, 
        "dev_strict_i_apis": dev_strict_i_apis , 
        "dev_strict_o_apis": dev_strict_o_apis,
    }

    machine_name = gethostname()
    machine_md5 = md5(machine_name.encode()).hexdigest()

    with open("./configs/StrictDeviceInfo-" + machine_md5 + ".json", 'w') as handle:
        json.dump(dict, handle, indent = 4)

    return(dict)
