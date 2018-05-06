# 参考URL : http://aidiary.hatenablog.com/entry/20110607/1307449007

import wave
import struct
import numpy as np
import matplotlib.pyplot as plt, urllib
#%matplotlib inline
from saveAudio import play, save

def createSinWave (amplitude, f0, fs, length):
    """
    振幅amplitude、基本周波数f0、サンプリング周波数 fs、
    長さlength秒の正弦波を作成して返す
    これで作られるデータはwavファイルと同じバイナリデータ
    """
    data = []
    # [-1.0, 1.0]の小数値が入った波を作成
    sample_index = int(length * fs)
    for n in range(sample_index):  # nはサンプルインデックス
        s = amplitude * np.sin(2 * np.pi * f0 * n / fs)
        # 振幅が大きい時はクリッピング
        if s > 1.0:  s = 1.0
        if s < -1.0: s = -1.0
        data.append(s)
    # [-32768, 32767]の整数値に変換
    data = [int(x * 32767.0) for x in data]
    # plt.plot(data[0:100]); plt.show()
    # バイナリに変換
    data = struct.pack("h" * len(data), *data)  # listに*をつけると引数展開される
    return data

if __name__ == "__main__" :
    allData =b""
    #freqList = [262, 294, 330, 349, 392, 440, 494, 523]  # ドレミファソラシド
    #freqList = [2093.005, 2349.318, 2637.020, 2793.826, 3135.963, 3520.000, 3951.066, 4186.009]  # ドレミファソラシド
    freqList = [32.703, 65.406, 130.813, 261.626, 523.251, 1046.502, 2093.005, 4186.009]
    for f in freqList:
        data = createSinWave(1.0, f, 10000.0, 1.0)
        play(data, 10000, 16)
        allData += data
    filename = "./wav3/do*8.wav"
    save(allData, 10000, 16, filename)
