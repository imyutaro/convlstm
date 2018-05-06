# 参考サイト：https://qiita.com/martin-d28jp-love/items/34161f2facb80edd999f

#%matplotlib inline #jupyterの表示に必要
import matplotlib.pyplot as plt, librosa, librosa.display, urllib
import numpy as np
import wave
import os

# spectral.py
def melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                   power=2.0, **kwargs):
    '''
    y : np.ndarray [shape=(n,)] or None
        audio time-series

    sr : number > 0 [scalar]
        sampling rate of `y`

    S : np.ndarray [shape=(d, t)]
        spectrogram</#>

    n_fft : int > 0 [scalar]
        length of the FFT window

    hop_length : int > 0 [scalar]
        number of samples between successive frames.
        See `librosa.core.stft`

    power : float > 0 [scalar]
        Exponent for the magnitude melspectrogram.
        e.g., 1 for energy, 2 for power, etc.
    '''
    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length,
                            power=power)

    # Build a Mel filter
    mel_basis = filters.mel(sr, n_fft, **kwargs)

    return np.dot(mel_basis, S)

def _spectrogram(y=None, S=None, n_fft=1024, hop_length=512, power=1):

    if S is not None:
        # Infer n_fft from spectrogram shape
        n_fft = 2 * (S.shape[0] - 1)
    else:
        # Otherwise, compute a magnitude spectrogram from input
        S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length))**power

    return S, n_fft


################


file_path = "./createWave/wav4/FT/2349.318+659.255*4(fs=44100).wav"
x, fs = librosa.load(file_path, sr=44100)
wf = wave.open(file_path, "r")
file_name = os.path.splitext(file_path)[0]
print(file_name)
librosa.display.waveplot(x, sr=fs, color='blue')
#C = librosa.feature.chroma_cqt(y=x, sr=fs)
#librosa.display.specshow(C, y_axis='chroma', x_axis='time')
#plt.colorbar()


fs = wf.getframerate()
print(
"file_path:", file_path,
"\nmfcc shape:", librosa.feature.mfcc(x, sr=fs, n_mfcc = 128).shape,
"\nサンプリング周波数:", wf.getframerate(),
"\nx(audio time series):", x,
"\nx.shape:", x.shape,
"\nx type:", type(x),
"\nfs(sampling rate):", fs
)


#1. 音声データを適当な長さのフレームに分割する
#2. Window関数を適応し、離散フーリエ変換して絶対値の2乗を取り、周波数スペクトルを得る
#   - STFT
#3. 対数をとる

stft = np.abs(librosa.stft(x, n_fft=1024, hop_length=512))**2
# window size : 1024 ms
# hop length : 512 ms
# - number audio of frames between STFT columns. If unspecified, defaults win_length / 4.
# win length : If unspecified, defaults to win_length = n_fft.
D = librosa.amplitude_to_db(librosa.stft(x), ref=np.max)
print(D.shape)
print(librosa.stft(x).shape)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')

librosa.display.specshow(stft, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
print(stft.shape)
print(stft)

# numpyを保存
np.savetxt(str(file_name)+".csv", stft, delimiter=',')


log_stft = librosa.power_to_db(stft)
# Take the logarithm of
# librosa.power_to_db
# - Convert a power spectrogram (amplitude squared) to decibel (dB) units.
#   This computes the scaling 10 * log10(S / ref) in a numerically stable way.

librosa.display.specshow(log_stft, sr=fs, x_axis='time', y_axis='hz')
plt.colorbar()

print(log_stft.shape)
# 4. メル周波数で均等になるようBINを集めてスムージングする
#   - binパワー計算済みのSを利用して、メルフィルタバンクをあてる
melsp = librosa.feature.melspectrogram(S=log_stft)
librosa.display.specshow(melsp, sr=fs, x_axis='time', y_axis="hz")
plt.colorbar()
print("melsp:", melsp,
"\nmelsp shape:", melsp.shape,
"\nbin power:", log_stft)

# 5. 離散コサイン変換する（低次項を取る）
# デフォルト n_mfcc = 20bin
#mfccs = librosa.feature.mfcc(S=melsp, n_mfcc=128)
mfccs = librosa.feature.mfcc(S=melsp, n_mfcc=20)
print("mfccs shape", mfccs.shape)

# 標準化して可視化
import sklearn
import matplotlib.cm as cm
mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
librosa.display.specshow(mfccs, sr=fs, x_axis='time', y_axis="mel")
print(np.amax(mfccs), " ", np.amin(mfccs))

#plt.pcolor(mfccs, vmin=-4, vmax=4)
#plt.clim(-4,4)
plt.colorbar()
plt.show()

print(
"mfccs:", mfccs,
"\nmfccs type:", type(mfccs),
"\nmfccs shape:", mfccs.shape
)

"""
plt.xlabel("hz(frecuency)")
plt.ylabel("mel")
plt.plot(mfccs[0, :], label="曲1_[0]")
plt.xlabel("hz(frecuency)")
plt.ylabel("mel")
plt.plot(mfccs[1, :], label="曲1_[1]")
plt.xlabel("hz(frecuency)")
plt.ylabel("mel")
plt.plot(mfccs[2, :], label="曲1_[2]")

# pmgで保存
librosa.display.specshow(mfccs, sr=fs)
plt.savefig(str(file_name)+".jpg", bbox_inches="tight", pad_inches=0.0)
"""

# numpyを保存
np.savetxt(str(file_name)+".csv", mfccs, delimiter=',')


####################


file_path = "./createWave/FCGAm.wav"
x, fs = librosa.load(file_path, sr=44100)
wf = wave.open(file_path, "r")
file_name = os.path.splitext(file_path)[0]
#librosa.display.waveplot(x, sr=fs, color='blue')

print(
"file_path:", file_path,
"\nmfcc shape:", librosa.feature.mfcc(x, sr=fs, n_mfcc = 128).shape,
"\nサンプリング周波数:", wf.getframerate(),
"\nx(audio time series):", x,
"\nx.shape:", x.shape,
"\nx type:", type(x),
"\nfs(sampling rate):", fs
)


#1. 音声データを適当な長さのフレームに分割する
#2. Window関数を適応し、離散フーリエ変換して周波数スペクトルを得る
#   - STFT
#3. 対数をとる

stft = np.abs(librosa.stft(x, n_fft=1024, hop_length=512))**2
log_stft = librosa.power_to_db(stft)

librosa.display.specshow(log_stft, sr=fs, x_axis='time', y_axis='hz')
plt.colorbar()


# 4. メル周波数で均等になるようBINを集めてスムージングする
#   - binパワー計算済みのSを利用して、メルフィルタバンクをあてる
melsp = librosa.feature.melspectrogram(S=log_stft)
librosa.display.specshow(melsp, sr=fs, x_axis='time', y_axis="hz")
plt.colorbar()
print("melsp:", melsp,
"\nmelsp shape:", melsp.shape,
"\nbin power:", log_stft)

# 5. 離散コサイン変換する（低次項を取る）
# デフォルト n_mfcc = 20bin
mfccs = librosa.feature.mfcc(S=melsp, n_mfcc=128)
print("mfccs shape", mfccs.shape)

# 標準化して可視化
import sklearn
mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
librosa.display.specshow(mfccs, sr=fs, x_axis='time', y_axis="mel")
plt.colorbar()
print(
"mfccs:", mfccs,
"\nmfccs type:", type(mfccs),
"\nmfccs shape:", mfccs.shape
)

"""
plt.xlabel("hz(frecuency)")
plt.ylabel("mel")
plt.plot(mfccs[0, :], label="曲1_[0]")
plt.xlabel("hz(frecuency)")
plt.ylabel("mel")
plt.plot(mfccs[1, :], label="曲1_[1]")
plt.xlabel("hz(frecuency)")
plt.ylabel("mel")
plt.plot(mfccs[2, :], label="曲1_[2]")

# pmgで保存
librosa.display.specshow(mfccs, sr=fs)
plt.savefig(str(file_name)+".jpg", bbox_inches="tight", pad_inches=0.0)
"""

# numpyを保存
np.savetxt(str(file_name)+".csv", mfccs, delimiter=',')
