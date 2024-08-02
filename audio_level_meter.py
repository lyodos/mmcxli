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



class InputLevelMeterPanel(wx.Panel):
    def __init__(
        self, 
        parent, 
        backend = None, # SoundControl クラスのインスタンスをここに指定。先にバックエンドが初期化されている必要がある。
        id = -1,
        size: tuple = (280, 50),
        min_level: int = -100,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(parent, id = id, size = size, **kwargs)
        self.debug = debug
        self.sc = backend

        # 背景スタイルを設定する。この処理は wx.BufferedPaintDC(self) を使用してフリッカーを抑えるためのもの
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

        self.client_size = size # ウィンドウのクライアント領域のサイズ
        self.left_margin: int = 20
        self.meter_h: int = 8
        self.min_level = min_level
        
        self.b_color = [wx.Colour(105, 108, 113), wx.Colour(79, 118, 182), wx.Colour(171, 52, 77)] # VC OFF, VC ON, (timeout) 
        self.light_gray_color = wx.Colour(215, 215, 225)
        self.night_gray_color = wx.Colour(23, 23, 27) # この色は元画像書き出しに合わせてあるので、任意には変更できない。

        self.G_color =  wx.Colour(64, 180, 76)
        self.Y_color =  wx.Colour(255, 204, 0)
        self.R_color =  wx.Colour(219, 50, 54)

        # 画像をパネル内に貼り付ける
        self.Bind(wx.EVT_PAINT, self.on_paint)
        
        self.timer = wx.Timer(self) # wx.Timer クラスで、指定間隔での処理を実行する。
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.timer.Start(37) # タイマーは手動で起動する必要がある。単位 ms


    def update(self, event):
        self.Refresh()


    def on_paint(self, event):
        panel_width, panel_height = self.GetSize() 

        border_pen = wx.Pen(wx.Colour(225, 225, 225), 2)
        bg_fill = wx.Brush(self.night_gray_color)
        # 背景色と同じ色で、インジケータの上からグリッド線を描画し、ブロック風に見せる
        bg_grid_pen = wx.Pen(self.night_gray_color, 2)

        dc = wx.AutoBufferedPaintDC(self) # wx.PaintDC だと Windows でフリッカーが発生するため切り替えた

        # フロー画面の背景色は、VC の ON/OFF および時間超過の警告に応じて切り替える
        if self.sc.efx_control.vc_lap / (1000 * self.sc.blocksize / self.sc.sr_out) > 1:
            dc.SetBackground(wx.Brush(self.b_color[2]))
        else:
            dc.SetBackground(wx.Brush(self.b_color[int(self.sc.vc_now)]))

        dc.Clear()
        
        # 外郭
        dc.SetPen(border_pen)
        dc.SetBrush(bg_fill)
        dc.DrawRectangle(1, 1, panel_width - 2, panel_height - 2) 

        # インジケータの色を、現在のオーディオレベルに応じて変える
        rect_width = int(
            max(
                0, 
                2*(min(self.sc.input_dBFS, 0) - self.min_level)
            )
        )
        # たぶん、エリアごとに色を変えて描画するといい
        dc.SetPen(wx.Pen(self.G_color, 0))
        dc.SetBrush(wx.Brush(self.G_color))
        dc.DrawRectangle(
            self.left_margin, (panel_height - self.meter_h) // 2, 
            rect_width, self.meter_h,
        ) 

        # インジケータの色を、現在のオーディオレベルに応じて変える
        rect_width = int(
            max(
                0, 
                2*(min(self.sc.input_dBFS, 0) + 20)
            )
        )
        if rect_width > 0:
            dc.SetPen(wx.Pen(self.Y_color, 0))
            dc.SetBrush(wx.Brush(self.Y_color))
            dc.DrawRectangle(
                int(2*(-20 - self.min_level)) + self.left_margin, (panel_height - self.meter_h) // 2, 
                rect_width, self.meter_h,
            ) 

        # インジケータの色を、現在のオーディオレベルに応じて変える
        rect_width = int(
            max(
                0, 
                2*(min(self.sc.input_dBFS, 0) + 6)
            )
        )
        if rect_width > 0:
            dc.SetPen(wx.Pen(self.R_color, 0))
            dc.SetBrush(wx.Brush(self.R_color))
            dc.DrawRectangle(
                int(2*(-6 - self.min_level)) + self.left_margin, (panel_height - self.meter_h) // 2, 
                rect_width, self.meter_h,
            ) 

        # ブロック区切り用縦グリッド線
        dc.SetPen(bg_grid_pen)
        grid_level_list = list(range(-80, 1, 4))
        for pos in grid_level_list:
            dc.DrawLine(
                int(2*(pos - self.min_level) + self.left_margin), 5, 
                int(2*(pos - self.min_level) + self.left_margin), panel_height - 5,
            )

        # 目盛り用縦グリッド線
        dc.SetFont(wx.Font(8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        dc.SetPen(border_pen)
        dc.SetTextForeground(self.light_gray_color)  # 文字色
        grid_level_list = [-80, -60, -40, -20, 0]
        for pos in grid_level_list:
            dc.DrawLine(
                int(2*(pos - self.min_level)) + self.left_margin + 1, panel_height - 9, 
                int(2*(pos - self.min_level)) + self.left_margin + 1, panel_height - 8,
            )
            dc.DrawText(
                f"{pos: >4.0f}", 
                int(2*(pos - self.min_level)) + self.left_margin - 10, 
                2,
            )
        
        # 現在の VC 閾値
        dc.SetPen(wx.Pen(self.light_gray_color, 1, wx.PENSTYLE_DOT))
        dc.SetBrush(wx.Brush(self.b_color[1], wx.TRANSPARENT))
        dc.DrawRectangle(
            int(2*(self.sc.VC_threshold - self.min_level)) + self.left_margin, (panel_height - self.meter_h - 5) // 2, 
            int(-2*self.sc.VC_threshold) + 2, self.meter_h + 7,
        ) 


####

# 以下は最初にテストで作ったもので、グラフィックスではなくテキストでレベル表示する。

class InputLevelTextPanel(wx.Panel):
    def __init__(
        self, 
        parent, 
        backend = None, # SoundControl クラスのインスタンスをここに指定。先にバックエンドが初期化されている必要がある。
        id = -1,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(parent, id = id, **kwargs)
        self.debug = debug
        self.sc = backend

        self.b_color = [wx.Colour(105, 108, 113), wx.Colour(79, 118, 182), wx.Colour(171, 52, 77)] # VC OFF, VC ON, (timeout) 
        self.status_text_color = [wx.Colour(189, 198, 192), wx.Colour(222, 220, 225)] # VC OFF、VC ON

        # 入力信号レベルと VC 閾値
        # 後でグラフィックスに変更
        self.level_text = wx.StaticText(self, label = f"Input {self.sc.input_dBFS: >7.1f} dBFS (thresh: {self.sc.VC_threshold: >7.1f} dBFS)", style = wx.ALIGN_CENTER)
        self.level_font = wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        self.level_text.SetFont(self.level_font)
        self.level_text.SetForegroundColour(self.status_text_color[0]) 
        
        # 統括 sizer を作る。ここは VERTICAL でないと AddStretchSpacer が効かない
        self.root_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.root_sizer.Add(self.level_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 0)
        self.SetSizer(self.root_sizer)
        self.Layout()

        # 背景色のセット
        self.SetBackgroundColour(self.b_color[int(self.sc.vc_now)]) 

        self.timer = wx.Timer(self) # wx.Timer クラスで、指定間隔での処理を実行する。
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.timer.Start(50) # タイマーは手動で起動する必要がある。単位 ms


    def update(self, event):

        self.level_text.SetLabel(f"Input {self.sc.input_dBFS: >7.1f} dBFS (thresh: {self.sc.VC_threshold: >7.1f} dBFS)")
        self.level_text.SetForegroundColour(self.status_text_color[int(self.sc.vc_now)]) 

        # フロー画面の背景色は、VC の ON/OFF および時間超過の警告に応じて切り替える
        if self.sc.efx_control.vc_lap / (1000 * self.sc.blocksize / self.sc.sr_out) > 1:
            self.SetBackgroundColour(self.b_color[2]) 
        else:
            self.SetBackgroundColour(self.b_color[int(self.sc.vc_now)]) 

        self.Refresh()
        self.Layout()
