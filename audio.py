import wave
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

def sampleRate(wf):
    """sampling rateを返す"""
    return wf.getframerate()

def saveWav(audio, dir_path, file_name, data_format='filtered'):
    """引数のnumpyの2次元配列のファイルもしくはwavファイルを保存する．filterに通されたdataはnumpyの2次元配列"""
    if not os.path.isdir(dir_path+'filtered/'):
        os.mkdir(dir_path+'filtered/') # make dir for shortform csv

    if data_format=='wav':
        audio_data, fs = librosa.load(dir_path+file_name, sr=None)
    else:
        # instrumentのsampling rateは16000
        # 一般的なCDのsampling rateは44100
        #sr=44100
        sr=16000

    audio_data = librosa.db_to_amplitude(audio_data)
    print(audio_data.shape)

    y_hat = librosa.istft(audio_data)
    # librosa.istftにはstftのshape,(freq, time)のnumpy配列を入力する．
    print(y_hat.shape)


    librosa.output.write_wav(dir_path+'filtered/'+file_name+'.wav', y_hat, sr=fs)

def mkdb(audio_data):
    D = librosa.amplitude_to_db(librosa.stft(audio_data, n_fft=1024, hop_length=512), ref=np.max)
    D = np.hstack((D, D))
    D = np.hstack((D, D))
