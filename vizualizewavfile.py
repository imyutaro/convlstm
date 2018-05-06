# 参照URL:http://aidiary.hatenablog.com/entry/20110519/1305808715
import wave
from numpy import *
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt, librosa, librosa.display, urllib
import os

def printWaveInfo(wf):
    """WAVEファイルの情報を取得"""
    print("チャンネル数:", wf.getnchannels())
    print("サンプル幅:", wf.getsampwidth())
    print("サンプリング周波数:", wf.getframerate())
    print("フレーム数:", wf.getnframes())
    print("パラメータ:", wf.getparams())
    print("長さ（秒）:", float(wf.getnframes()) / wf.getframerate())
    print("\n")

if __name__ == '__main__':
    file_path = './wavSrc/instrument/same_instrument/guitar_acoustic_021-022-050.wav'
    #file_path = './wavSrc/instrument/same_instrument/flute_synthetic_000-051-100.wav'
    #file_path = './wavSrc/instrument/same_instrument/flute_synthetic_000-036-100.wav'

    #file_path='./wavSrc/single+chord/2349.318+/2349.318+195.998*4(fs=44100).wav'
    #file_path='./wavSrc/single+chord/2349.318+/2349.318+1046.502*4(fs=44100).wav'
    #file_path='./wavSrc/single+chord/2349.318+/2349.318+659.255*4(fs=44100).wav'
    #file_path='./wavSrc/single+chord/2349.318+/2349.318+329.628*4(fs=44100).wav'
    #file_path='./wavSrc/single+chord/2349.318+/2349.318+97.999*4(fs=44100).wav'
    #file_path='./wavSrc/single+chord/2349.318+/440.000+2793.826*4(fs=44100).wav'
    #file_path='./wavSrc/single+chord/2349.318+/174.614+2793.826*4(fs=44100).wav'

    wf = wave.open(file_path, "r")
    print(file_path)
    printWaveInfo(wf)

    buffer = wf.readframes(wf.getnframes())
    print(len(buffer))  # バイト数 = 1フレーム2バイト x フレーム数

    # bufferはバイナリなので2バイトずつ整数（-32768から32767）にまとめる
    data = frombuffer(buffer, dtype="int16")
    # プロット
    #plt.plot(data)
    #plt.plot(data[0:16000])
    #plt.plot(data[0:int(16000/50)])
    #plt.plot(data[0:int(16000/50)])
    #plt.show()
    #plt.plot(data[int(16000/50):int(16000/50*2)])
    #plt.show()
    max=np.max(data)
    min=np.min(data)

    #"""

    for i in range(10):
        fig=plt.figure(figsize=(20, 50))
        plt.subplot(10, 1, i + 1)
        #plt.plot(data[int(16000/50*i):int(16000/50*(i+1))])
        plt.plot(data[int(16000/50*i):int(16000/50*(i+3))])
        plt.ylim([min,max])
        plt.show()
    #fig.savefig('./myThesis/06_experiment/audio.png', bbox_inches='tight')
    #"""
    """
    for i in range(4):
        plt.plot(data[int(16000*i):int(16000*(i+1))])
        plt.ylim([min,max])
        plt.show()
    """
