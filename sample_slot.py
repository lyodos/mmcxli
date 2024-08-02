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
import librosa
from pydub import AudioSegment

import os
import json
import copy
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
import numpy as np
import csv

from datetime import datetime

import logging
import inspect

from utils import plot_spectrogram_harmof0, plot_embedding_cube, truncate_string


class ImageDropTarget(wx.FileDropTarget):
    def __init__(self, host):
        super().__init__()
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.host = host # host というのは AudioSlotPanel インスタンスのこと

    def OnDropFiles(self, x, y, filenames):
        try:
            file_path = filenames[0]
            if file_path.lower().endswith(('.wav', '.ogg', '.mp3', '.m4a', '.flac', '.opus')):
                self.host.current_file_path_raw = file_path
                # OS 依存のパス区切り文字を修正。なおドライブレター等はうまく処理できない
                self.host.current_file_path = self.host.current_file_path_raw.replace('/', self.host.os_sep).replace('\\', self.host.os_sep) 
                self.logger.debug(f"({inspect.currentframe().f_code.co_name}) selecting... {self.host.current_file_path}")
                # ファイルをロードしてメモリ上の変数に格納するところまで実行
                self.host.load_file(self.host.current_file_path, self.host.manager.sr_out, self.host.manager.sr_proc) 
                return True
            else:
                wx.MessageBox("Unsupported file format. Please drop an audio file (wav, ogg, mp3).", "Error", wx.OK | wx.ICON_ERROR)
                return False
        except Exception as e:
            wx.MessageBox(f"An error occurred while dropping files: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
            return False


class AudioSlotPanel(wx.Panel):
    def __init__(
        self, 
        parent, 
        id = -1,
        manager = None,
        slot_index: int = None, # slot リストの中でこのパネルが何番目の要素であるか
        **kwargs,
    ):
        super().__init__(parent, id = id, **kwargs)
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        self.manager = manager # SampleManagerPanel インスタンスを指定
        self.slot_index = slot_index
        self.os_sep = os.path.sep # 現在の OS で有効なパス区切り文字
        
        self.is_file_loaded = False # まず file load されていない状態で初期化
        
        # アクティブにするか（VC 用のスタイル埋め込み計算に反映させるか）のチェックボックス
        self.is_active_checkbox = wx.CheckBox(self, wx.ID_ANY, '')
        self.is_active_checkbox.Disable()
        self.is_active_checkbox.SetMinSize((20, -1)) # 最小幅を設定
        self.is_active_checkbox.Bind(wx.EVT_CHECKBOX, self.on_active_checkbox_click) # 単にパネルへの伝播を止める
        
        self.current_file_path = None # 最初はファイルが未選択なので None で初期化
        self.current_file_label = wx.StaticText(self) # 選択中のファイルを表示するテキスト
        self.current_file_label.SetLabel("Audio file: unset")
        self.current_file_label.SetMinSize((155, -1)) # Text の最小幅を設定（自動調整を適用させないため）
        self.current_file_label.SetMaxSize((155, -1)) # Text の最大幅を設定
        self.current_file_label.SetToolTip('Load audio and calculate the speaker style')

        # ファイル選択ボタン。これを定義するには self.current_file_path が定義済み（未選択の場合は None）である必要
        self.open_btn = wx.Button(self, label = "Load...")
        self.open_btn.SetMinSize((60, -1)) # 最小幅を設定
        self.open_btn.SetMaxSize((60, -1)) # 最大幅を設定
        self.open_btn.Bind(wx.EVT_BUTTON, self.select_file) # self.current_file_path が未選択だと . がデフォルト位置

        # スタイル書き出しボタン
        self.export_btn = wx.Button(self, label = "Export style as...")
#        self.export_btn.SetMinSize((70, -1)) # 最小幅を設定
#        self.export_btn.SetMaxSize((70, -1)) # 最大幅を設定
        self.export_btn.Bind(wx.EVT_BUTTON, self.export_style)
        if self.is_file_loaded is False:
            self.export_btn.Hide()


        # スタイルのミックス量
        
        # 話者スタイルをミックスするときに当該スロットが持つ係数を制御するスライダー。初期状態は 1.0
        self.mix_coef_sldr_mult = 100 # wx のスライダーは int しか取れないので、移動量を水増しする必要がある。
        self.mix_coef_sldr = wx.Slider(
            self, 
            value = round(self.manager.mix_coef_list[self.slot_index] * self.mix_coef_sldr_mult), 
            minValue = 0, 
            maxValue = self.mix_coef_sldr_mult, # 内部状態は 0--100 の int だが、manager 側の値は 0--1 の float に変更
            size = (140, -1),
            style = wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS,
        )
        self.mix_coef_sldr.SetTickFreq(round(self.mix_coef_sldr_mult / 4))
        self.mix_coef_sldr.Bind(wx.EVT_SLIDER, self.on_mix_coef_sldr_change)

        self.mix_coef_label = wx.StaticText(self)
        self.mix_coef_label.SetLabel('Style mix (%)')
        self.mix_coef_label.SetToolTip('Load audio and calculate the speaker style')

        self.mix_coef_sizer = wx.BoxSizer(wx.VERTICAL)
        self.mix_coef_sizer.Add(self.mix_coef_label, 0, wx.LEFT | wx.ALIGN_CENTER_HORIZONTAL, 0)
        self.mix_coef_sizer.Add(self.mix_coef_sldr, 0, wx.LEFT | wx.ALIGN_CENTER_HORIZONTAL, 0)

        self.label_export_sizer = wx.BoxSizer(wx.VERTICAL)
        self.label_export_sizer.Add(self.current_file_label, 0, wx.ALL | wx.ALIGN_LEFT, 0)
        self.label_export_sizer.Add(self.export_btn, 0, wx.ALL | wx.ALIGN_LEFT, 0)
        self.label_export_sizer.Add(self.mix_coef_sizer, 0, wx.TOP | wx.ALIGN_LEFT, 15)

        # sizer を隠すときは、それが属しているオブジェクトを指定する必要がある
        if self.is_file_loaded is False:
            self.label_export_sizer.Hide(self.mix_coef_sizer, recursive = True)
        
        # ファイル選択ボタン、選択中ファイル表示の集合をまとめた sizer を作る
        self.widgets_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.widgets_sizer.Add(self.open_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.widgets_sizer.Add(self.label_export_sizer, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        
        # 統括用の sizer を作り、すべての部品を配置していく。なお再生制御はここではなく、ホスト側が 1 つだけ持つ
        self.root_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.root_sizer.Add(self.widgets_sizer, 0, wx.EXPAND, 5) 
        self.root_sizer.Add(self.is_active_checkbox, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.SetBackgroundColour(self.manager.nonactive_slot_color)
        
        self.SetSizer(self.root_sizer) # （既存の子要素があれば削除して）統括用 sizer を AudioSlotPanel 配下に加える

        # パネルをクリックしてアクティブにするイベント。イベントハンドラに追加引数を渡すためラムダ式にしている
        self.current_file_label.Bind(wx.EVT_LEFT_DOWN, lambda event: self.manager.on_panel_click(event, self.slot_index))
        self.current_file_label.Bind(wx.EVT_RIGHT_DOWN, lambda event: self.manager.on_panel_right_click(event, self.slot_index))
        self.current_file_label.Bind(wx.EVT_ENTER_WINDOW, lambda event: self.manager.on_panel_hover(event, self.slot_index))
        self.current_file_label.Bind(wx.EVT_LEAVE_WINDOW, lambda event: self.manager.on_panel_unhover(event, self.slot_index))


        # スロットにファイルをドラッグアンドドロップした時の挙動
        self.SetDropTarget(ImageDropTarget(self))

        # self.update は作成済みのプロットを再描画する。
        self.timer = wx.Timer(self) # wx.Timer クラスで、指定間隔での処理を実行する。
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.timer.Start(57) # タイマーは手動で起動する必要がある。単位 ms

        self.logger.debug(f"({inspect.currentframe().f_back.f_code.co_name}) AudioSlotPanel {self.slot_index} was initialized.")
    
    
    def update(self, event):
        if hasattr(self, 'canvas_spec') and self.canvas_spec is not None:
            self.canvas_spec.draw()

        if self.is_file_loaded is False:
            self.export_btn.Hide()
            self.label_export_sizer.Hide(self.mix_coef_sizer, recursive = True)
        else:
            self.export_btn.Show()
            self.label_export_sizer.Show(self.mix_coef_sizer, recursive = True)
        
        self.Layout()

    
    #### ローカルの音声ファイルを選択するイベントハンドラ
    
    def select_file(self, event):
        # ダイアログ初期値を前回ファイルにする
        last_relpath = self.manager.sample_portfolio[self.slot_index]["last_selected_file"]
        if last_relpath is not None:
            last_relpath = last_relpath.replace('/', self.os_sep).replace('\\', self.os_sep) 
            abs_path = os.path.abspath(last_relpath)
            initial_dir = os.path.dirname(abs_path)
            initial_filename = os.path.basename(abs_path)
        else:
            initial_dir = "."
            initial_filename = ""
        
        with wx.FileDialog(
            self, 
            "Open audio file", 
            defaultDir = initial_dir,
            defaultFile = initial_filename,
            wildcard = "Audio files (*.wav;*.mp3;*.ogg)|*.wav;*.mp3;*.ogg",
            style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return # キャンセルの場合はロード処理に入らず、現在の選択ファイルが保持される
            
            self.current_file_path_raw = fileDialog.GetPath()
            # OS 依存のパス区切り文字を修正。なおドライブレター等はうまく処理できない
            self.current_file_path = self.current_file_path_raw.replace('/', self.os_sep).replace('\\', self.os_sep) 
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Selecting... {self.current_file_path}")

            # ファイルをロードしてメモリ上の変数に格納するところまで実行
            self.load_file(self.current_file_path, self.manager.sr_out, self.manager.sr_proc) 
        event.Skip(False)


    # 選択したファイルを実際にロードする
    # 音声のロードに成功すれば self.is_file_loaded = True になり、self.file_sec に有効値が入る
    # ただし音声は self.file_audio_store/play に格納され、まだ実際の再生サンプル self.manager.cs には反映されない。
    # 再生中に実行された場合の安全のため、self.load_file までは自動的に実行され self.file_audio_store が更新されるが、
    # self.manager.cs 変数の実際の書き換えは self.manager.playing == False の場合しか実行できないようにする。

    def load_file(
        self,
        file_path, # select_file() が走った状態なので、file_path の指す内容は有効な音声ファイルのはず。
        sr_out: float,
        sr_proc: float,
    ):
        if file_path is not None:
            self.is_file_loaded = False # まず file load されていない状態に戻す
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) file_path: {file_path}")
            # サンプルを変える場合、再生を停止する。ヘッドも内部で 0 戻しされる
            if self.manager.playing:
                self.manager._stop_sound() 
            try:
                # soundfile をロードしたら 2 種類のバッファに保存する。リサンプリングは time last が必要。
                # sf は返り値は channel last だが、librosa は timel last なので注意
                file_audio_raw, file_orig_fs = self.read_audio(
                    file_path, 
                    max_sec = self.manager.max_sec,
                ) # (time, channel)
                self.file_audio_name = os.path.basename(file_path)
                # sf.read を使う場合、以下が shape[0] を使うようになる
                self.file_sec: float = file_audio_raw.shape[1] / file_orig_fs # 単位：秒
                
                # TODO ここは重い処理なので、実際には threading を使って非同期化し、その間に「変換中」の画面表示を出すべき

                # 再生用のバッファに保存 self.manager.sr_out (44100 Hz) で作る
                self.file_audio_play = librosa.resample(
                    file_audio_raw,  # time last の状態で音声をリサンプルする
                    orig_sr = file_orig_fs, 
                    target_sr = sr_out,
                    res_type = "polyphase",
                    axis = -1,
                ).T # 再生サンプルは channel last で保持する
                
                # embedding 計算用のバッファに保存 self.manager.sr_proc (16000 Hz) で作る
                self.file_audio_store = librosa.resample(
                    file_audio_raw,  # time last の状態で音声をリサンプルする
                    orig_sr = file_orig_fs, 
                    target_sr = sr_proc,
                    res_type = "polyphase",
                    axis = -1,
                ) # 加工用サンプルは time last で保持する

                self.calculate_embedding() # ロードしたら、さっそく embedding を計算する。self.style が作成される
                self.plot_embedding() # embedding が計算できたら、plot に反映する
                self.is_active_checkbox.Enable() # さらに、アクティブスロットのチェックボックスを選択状態にする
                self.is_active_checkbox.SetValue(True)

                # サンプル一覧を更新＆ファイル保存。現在、保存先ファイル名はハードコーディングされている
                try:
                    # 絶対パスから相対パスに変換。絶対パスを json に残すと個人情報保護や複数マシン間での共有にリスク
                    rel_path = os.path.relpath(file_path, os.getcwd()) 
                    # ファイルを開くたびに config ファイルを更新する
                    self.manager.sample_portfolio[self.slot_index]["last_selected_file"] = rel_path
                    self.manager.sample_portfolio[self.slot_index]["is_active"] = True
                    self.manager.sample_portfolio[self.slot_index]["mix_coef"] = self.manager.mix_coef_list[self.slot_index]
                    self.manager.sample_portfolio[self.slot_index]["embedding"] = self.style.tolist()

                    # サンプルロードに伴い、自動的にサンプル一覧のファイルが保存される
                    # これ、他の場所からも呼び出すと思うので、manager 側に保存メソッドとして書き出した方がいいだろう
                    with open(self.manager.sample_portfolio_path, 'w') as f:
                        json.dump(self.manager.sample_portfolio, f, indent = 4)
                    self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Sample portfolio was updated: '{self.manager.sample_portfolio_path}'")
                except:
                    self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Failed to update the sample portfolio.")
                
                # 音声のロードに成功したら、ホスト側にも選択ファイルとして登録する。
                self.manager.file_audio_name = copy.deepcopy(self.file_audio_name)
                self.manager.file_sec = copy.deepcopy(self.file_sec)
                self.manager.file_audio_play = self.file_audio_play

                # さらに self.manager.playing == False の場合のみ、manager における実際の再生サンプルの反映まで行う。
                # もし True だったら、再生中のサンプルの末尾に達した瞬間に callback から評価させる
                # （ホストにおいて self.file_audio_name と self.cs_name が異なることがフラグとなる）
                self.manager.set_sample(self.file_audio_name, self.file_sec, self.file_audio_play) # サンプル反映
                self.manager.remake_sldr() # 再生位置スライダーを再作成する
                # （初期化時は手動で部品を作るのでここを使わない）

                self.is_file_loaded = True # ファイルロードの完了フラグ
            except:
                self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Failed to load the audio sample {file_path}")
                # 対応する音声が存在しない／正常にロードできない場合は config file をリネームして末尾に ".bak" を付加
                ex_config_name = self.manager.sample_portfolio_path.replace('.json', '.json.bak')
                os.rename(self.manager.sample_portfolio_path, ex_config_name)
                self.logger.debug(f"({inspect.currentframe().f_code.co_name}) {self.manager.sample_portfolio_path} was renamed to {ex_config_name}")
        else:
            pass # 有効なファイルパスを与えない場合は何もしない（通常はファイルが選択される前提であり、このルートには入らない）
        
        # ファイルロード試行の後、選択中のファイルを示すラベル文字列を更新する
        # 正確に言うと、ロードした音声の embedding 計算／プロットに失敗してもファイル名は更新される。一応これは仕様である。
        self.update_file_label(self.file_audio_name)
        
        self.Refresh()
        self.manager.Refresh()
        self.manager.SetupScrolling()  # スクロール設定を再度適用


    def read_audio(
        self, 
        file_path: str, 
        max_sec: float = 16.0,
    ):
        _, file_extension = os.path.splitext(file_path) # 拡張子はドット . を含む

        if file_extension.lower() in ['.wav', '.flac', '.ogg']:
            audio, sr = librosa.load(file_path, sr = None, mono = True, duration = max_sec)
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]
            return audio.astype('float32'), int(sr)

        elif file_extension.lower() in ['.mp3', '.m4a', '.opus']:
            audio = AudioSegment.from_file(file_path)
            sr = audio.frame_rate
            audio = audio.set_channels(1) if audio.channels == 2 else audio  # Ensure mono audio
            audio = audio.set_sample_width(2)    # Set sample width to 16-bit (adjust as needed)
        
            if max_sec < len(audio) / 1000:
                audio = audio[:max_sec * 1000] # 操作はミリ秒単位で実施する必要がある

            audio = audio.get_array_of_samples()
            audio = np.array(audio, dtype = 'float32') / 32768.0  # Normalize to float32
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]

            return audio, int(sr)  # Assume 16kHz sample rate for pydub
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

    # スロットが選択中であるファイルのラベルをセットする。なおテキストエリアのサイズは最初に十分量を確保して固定
    def update_file_label(
        self,
        file_name,
    ):
        self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Updating file label")

        if hasattr(self, 'current_file_label'):
            if self.is_file_loaded:
                text = f"\nLoaded audio file:\n{truncate_string(file_name, max = 23)}\n" # ロード完了フラグが立っている場合
                tooltip_text = f"'{file_name}'\nClick: activate, Right-click: deactivate"
            else:
                text = "Audio file: unset" # ロードされていない（あるいは失敗した）場合
                tooltip_text = 'Load audio and calculate the speaker style'
        self.current_file_label.SetLabel(text)
        self.current_file_label.SetToolTip(tooltip_text)
        self.root_sizer.Layout()


    # TODO portfolio の保存は複数回呼び出されるので、部品化してホストに置くべき
    
    def on_active_checkbox_click(self, event):
        if self.is_active_checkbox.GetValue() == True:
            self.manager.sample_portfolio[self.slot_index]["is_active"] = True
        else:
            self.manager.sample_portfolio[self.slot_index]["is_active"] = False
        try:
            with open(self.manager.sample_portfolio_path, 'w') as f:
                json.dump(self.manager.sample_portfolio, f, indent = 4)
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Sample portfolio was updated: '{self.manager.sample_portfolio_path}'")
        except:
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Failed to update the sample portfolio.")
        event.Skip(False)


    #### 埋め込みの計算

    # スタイル混合比のスライダーのイベントハンドラ
    
    # ここに portfolio の値も更新する処理を入れる
    def on_mix_coef_sldr_change(self, event):
        self.manager.mix_coef_list[self.slot_index] = self.mix_coef_sldr.GetValue() / self.mix_coef_sldr_mult 
        self.manager.sample_portfolio[self.slot_index]["mix_coef"] = self.manager.mix_coef_list[self.slot_index]
        try:
            with open(self.manager.sample_portfolio_path, 'w') as f:
                json.dump(self.manager.sample_portfolio, f, indent = 4)
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Sample portfolio was updated: '{self.manager.sample_portfolio_path}'")
        except:
            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Failed to update the sample portfolio.")
        event.Skip(False)

    # TODO portfolio の保存は複数回呼び出されるので、部品化してホストに置くべき

    # 選択中のオーディオデータのスタイル埋め込みを計算する。なお、末尾（時間）次元は 4 の倍数でないと動作しない
    def calculate_embedding(self):
        self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Calculating embedding")
        self.real_F0, self.activation, self.real_N, self.spectrogram = self.manager.sess_HarmoF0.run(
            ['freq_t', 'act_t', 'energy_t', 'spec'], 
            {"input": self.file_audio_store},
        )
        # 末尾（時間）次元を 4 の倍数に切り詰める
        self.spec_size_by_four = (self.spectrogram.shape[-1] // 4) * 4
        self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Spectrogram for SE: {self.spectrogram[:, np.newaxis, 48:, :self.spec_size_by_four].shape}")

        self.style = self.manager.sess_SE.run(
            ['output'], 
            {'input': self.spectrogram[:, np.newaxis, 48:, :self.spec_size_by_four]},
        )[0]


    # 選択中のオーディオデータのスタイル埋め込みをプロットする。
    def plot_embedding(self):
        self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Plotting 128-dim style embedding")

        if hasattr(self, 'fig_spec') or hasattr(self, 'fig_cube'):
            # Detach() と Destroy() で現在の sizer とプロットの Canvas オブジェクトを削除して作り直す
            self.widgets_sizer.Detach(self.plot_sizer)
            self.plot_sizer.Destroy()
            self.canvas_spec.Destroy()
            self.canvas_cube.Destroy()

        self.fig_spec, self.ax_spec = plot_spectrogram_harmof0(
            self.spectrogram[0, :, :],
            f0 = self.real_F0, 
            act = self.activation, 
            figsize = (600, 200),
            aspect = None,
            v_range = (-50, 40),
            cmap = "inferno",
        )
        self.fig_spec.patch.set_facecolor((1, 1, 1, 1)) # (rgba) Spectrogram の画像表示の背景色

        self.fig_cube, self.ax_cube = plot_embedding_cube(
            self.style,
            figsize = (120, 200),
            aspect = 1,
            cmap = "bwr",
        )
        self.fig_cube.patch.set_facecolor((1, 1, 1, 1)) # (rgba) Embedding の画像表示の背景色

        # FigureCanvasWxAgg は matplotlib と wx の連携用のパーツ
        self.canvas_spec = FigureCanvasWxAgg(self, -1, self.fig_spec)
        self.canvas_cube = FigureCanvasWxAgg(self, -1, self.fig_cube)
        self.plot_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.plot_sizer.Add(self.canvas_spec, 0, wx.TOP | wx.BOTTOM | wx.ALIGN_CENTER_VERTICAL, 5)
        self.plot_sizer.Add(self.canvas_cube, 0, wx.TOP | wx.BOTTOM | wx.ALIGN_CENTER_VERTICAL, 5)
        # self.plot_sizer は widgets_sizer の配下に加える。
        self.widgets_sizer.Add(self.plot_sizer, 0, wx.EXPAND | wx.ALIGN_TOP, 5) 

        self.manager.root_sizer.Layout()


    # スロットで計算して保持している 128 次元のスタイル埋め込みを csv に保存する
    def export_style(self, event):
        if self.is_file_loaded is True:
            default_extension = '.csv'
            wildcard = "Csv files (*.csv)|*.csv|Text files (*.txt)|*.txt|All files (*.*)|*.*"
            dlg = wx.FileDialog(
                self, 
                message = "Save style as ...", 
                defaultDir = "./styles" or '.', 
                defaultFile = os.path.splitext(self.file_audio_name)[0] + default_extension, 
                wildcard = wildcard,
                style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
            )
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                with open(path, 'w', newline = '') as f:
                    writer = csv.writer(f)
                    writer.writerow(np.round(self.style.astype(float), 4).flatten().tolist()) 
                    # numpy 配列をリストに変換して書き込む
                    # ただし float32 を float にキャストしてから round しないと、書き出した桁数がおかしくなる

                self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Sample style was saved at: {path}")
        event.Skip(False)


####

# 計算した style vector ないし、複数話者について平均したものを、最終的に VC に使う embedding として集約表示する

class ResultEmbeddingPanel(wx.Panel):
    def __init__(
        self, 
        parent, 
        id = -1,
        host = None,
        **kwargs,
    ):
        super().__init__(parent, id = id, **kwargs)
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        self.host = host # SampleManagerPanel インスタンスを指定

        self.result_style_label = wx.StaticText(self) # 選択中のファイルを表示するテキスト
        self.result_style_label.SetLabel("Calculated style")
        self.result_style_label.SetToolTip('Used as VC style embedding')
        
        # 再計算を必要な場合のみ行うために、現在の style embedding をキャッシュしておく
        self.style_cache = copy.deepcopy(self.host.style_result)

        # embedding はとりあえず self.host.style_silent で代用する
        self.canvas_cube = self.plot_result_embedding() 

        # スタイル書き出しボタン
        self.export_btn = wx.Button(self, label = "Export style as...")
        self.export_btn.Bind(wx.EVT_BUTTON, self.export_style)

        self.style_sizer = wx.BoxSizer(wx.VERTICAL)
        self.style_sizer.Add(self.canvas_cube, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 0)
        self.style_sizer.Add(self.result_style_label, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)
        self.style_sizer.Add(self.export_btn, 0, wx.TOP | wx.ALIGN_CENTER_HORIZONTAL, 10)
        
        # 統括用の sizer を作り、すべての部品を配置していく
        self.root_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.root_sizer.Add(self.style_sizer, 1, wx.ALL | wx.ALIGN_CENTER, 0)
        self.SetSizer(self.root_sizer) 

        # self.update は作成済みのプロットを再描画する
        self.timer = wx.Timer(self) # wx.Timer クラスで、指定間隔での処理を実行する。
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.timer.Start(99) # タイマーは手動で起動する必要がある。単位 ms

        self.logger.debug(f"({inspect.currentframe().f_back.f_code.co_name}) Initialized.")
    
    
    def update(self, event):
        # 必要な場合のみ再描画を行う
        if not np.allclose(self.style_cache, self.host.style_result, atol = 1e-8):
            # 壊す前に、ウィジェット全体の sizer の中での順番を記録しておく
            index = self.style_sizer.GetChildren().index(self.style_sizer.GetItem(self.canvas_cube))
            # Detach() と Destroy() で現在のスライダーを削除して作り直す
            self.style_sizer.Detach(self.canvas_cube)
            self.canvas_cube.Destroy()

            self.canvas_cube = self.plot_result_embedding() # embedding はとりあえず self.host.style_silent で代用する
            self.style_cache = copy.deepcopy(self.host.style_result)
            self.canvas_cube.draw()

            # 統括 sizer に配置し直す
            self.style_sizer.Insert(index, self.canvas_cube, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 0)
            self.style_sizer.Layout()

        self.Layout()


    # 最終的に VC に使用するスタイル埋め込みをプロットする。
    def plot_result_embedding(self):
        self.fig_cube, self.ax_cube = plot_embedding_cube(
            self.host.style_result,
            figsize = (120, 200),
            aspect = 1,
            cmap = "bwr",
        )
        self.fig_cube.patch.set_facecolor((1, 1, 1, 1))
        # FigureCanvasWxAgg は matplotlib と wx の連携用のパーツ
        return FigureCanvasWxAgg(self, -1, self.fig_cube)


    # 最終的に VC に使用するスタイル埋め込みを csv に保存する
    def export_style(self, event):
        # 現在の日付時刻を取得
        now = datetime.now()
        formatted_date_time = now.strftime("%Y-%m-%d_%H.%M.%S")
        final_style_name = "style_" + formatted_date_time
        default_extension = '.csv'
        wildcard = "Csv files (*.csv)|*.csv|Text files (*.txt)|*.txt|All files (*.*)|*.*"
        dlg = wx.FileDialog(
            self, 
            message = "Save style as ...", 
            defaultDir = "./styles" or '.', 
            defaultFile = final_style_name + default_extension, 
            wildcard = wildcard,
            style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            with open(path, 'w', newline = '') as f:
                writer = csv.writer(f)
                writer.writerow(np.round(self.host.style_result.astype(float), 4).flatten().tolist()) 

            self.logger.debug(f"({inspect.currentframe().f_code.co_name}) Mixed sample style was saved at: {path}")

