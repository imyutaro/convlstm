import wave
import struct
import numpy as np
from pylab import *
from saveAudio import save, play

def createCombinedWave (amplitude, freqList, fs, length):
    """
    振幅amplitude、基本周波数f0、サンプリング周波数 fs、
    長さlength秒の正弦波を作成して返す
    freqListの正弦波を合成した波を返す
    これで作られるデータはwavファイルと同じバイナリデータ
    """
    data = []
    amp = float(amplitude) / len(freqList)
    # [-1.0, 1.0]の小数値が入った波を作成
    sample_index = int(length * fs)
    for n in range(sample_index):  # nはサンプルインデックス
        s = 0.0
        for f in freqList:
            s += amp * np.sin(2 * np.pi * f * n / fs)
        # 振幅が大きい時はクリッピング
        if s > 1.0:  s = 1.0
        if s < -1.0: s = -1.0
        data.append(s)
    # [-32768, 32767]の整数値に変換
    data = [int(x * 32767.0) for x in data]
    # バイナリに変換
    data = struct.pack("h" * len(data), *data)  # listに*をつけると引数展開される
    return data

if __name__ == "__main__" :
    """
    # 和音
    chordList = [(262, 330, 392),  # C（ドミソ）
                 (294, 370, 440),  # D（レファ#ラ）
                 (330, 415, 494),  # E（ミソ#シ）
                 (349, 440, 523),  # F（ファラド）
                 (392, 494, 587),  # G（ソシレ）
                 (440, 554, 659),  # A（ラド#ミ）
                 (494, 622, 740)]  # B（シレ#ファ#）
    chordList = [
    (392, 494, 587), #G
    (494, 622, 740), #B
    (494, 622, 740), #B
    (440, 554, 659), #A
    (392, 494, 587), #G
    (392, 494, 587), #G
    (392, 494, 587), #G
    (440, 554, 659), #A

    (494, 622, 740), #B
    (440, 554, 659), #A
    (440, 554, 659), #A
    (440, 554, 659), #A
    (440, 554, 659), #A
    (392, 494, 587), #G
    (262, 330, 392), #C
    (392, 494, 587), #G
    ]
    """

    chordList = [
    (1046.502,),
    ]
    allData = b""
    fs = 44100
    for freqList in chordList:
        data = createCombinedWave(1.0, freqList, fs, 4.0)
        #play(data, fs, 16)
        allData += data
    filename = "./tmp/complicated_chord/1046.502*4.wav"
    save(allData, fs, 16, filename)
"""
(261.626+440.000+987.767)*4-(440.000+1396.913+2093.005)*4.wav
(130.813,220.000,493.883),
(698.456,220.000,1046.502),

(130.813+587.330+493.883)*4-(698.456+220.000+1046.502)*4
(130.813,587.330,493.883),
(698.456,220.000,1046.502),
"""
