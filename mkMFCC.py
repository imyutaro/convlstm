#%matplotlib inline #jupyterの表示に必要
import matplotlib.pyplot as plt, librosa, librosa.display, urllib
import numpy as np
import wave
import os
import sys

#dir_path = './wavSrc/music/'
dir_path = './wavSrc/time_change/chord/instruments/'
files = []
count = 0
i = 0
"""
args = sys.argv
print(len(args))
if len(args) > 1:
    dir_path = args[1]
    files = os.listdir(dir_path)
"""

file_list = os.listdir(dir_path)
tmp = []
for file_name in file_list:
    _, ext = os.path.splitext(file_name)
    if ext=='.wav':
        tmp.append(file_name)
        count +=1
if count == 0:
    print('There is no data')
    sys.exit(1)
file_list = tmp

print(file_list)
if not os.path.isdir(dir_path+'mfcc/'):
    os.mkdir(dir_path+'mfcc/') # make dir for shortform csv

fig=plt.figure(figsize=(20, 5*count))
for file_name in file_list:
    path, ext = os.path.splitext(file_name)
    if ext == ".wav":
        file_path = dir_path+file_name
        x, fs = librosa.load(file_path, sr=44100)
        wf = wave.open(file_path, "r")
        file_name = os.path.splitext(file_name)[0] #拡張子のwavを取るため
        print(file_name)
        stft = np.abs(librosa.stft(x, n_fft=1024, hop_length=512))**2
        print(stft.shape)
        print(stft)
        print(stft.shape)
        print(librosa.stft(x).shape)
        plt.subplot(count, 1, i + 1)
        i += 1

        log_stft = librosa.power_to_db(stft)
        print(log_stft.shape)
        # 4. メル周波数で均等になるようBINを集めてスムージングする
        #   - binパワー計算済みのSを利用して、メルフィルタバンクをあてる
        melsp = librosa.feature.melspectrogram(S=log_stft)
        #librosa.display.specshow(melsp, sr=fs, x_axis='time', y_axis="hz")
        #title=file_name+'_melsp'
        #plt.title(file_name)
        #plt.colorbar()

        # 5. 離散コサイン変換する（低次項を取る）
        # デフォルト n_mfcc = 20bin
        mfccs = librosa.feature.mfcc(S=melsp, n_mfcc=24)
        print("mfccs shape", mfccs.shape)

        # 一致判定
        flag =  np.allclose(mfccs, librosa.feature.mfcc(y=x, sr=fs ,n_mfcc=24))
        # 結果表示
        print(flag) # True or false

        # 標準化して可視化
        import sklearn
        import matplotlib.cm as cm
        mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
        librosa.display.specshow(mfccs, sr=fs, x_axis='time', y_axis="mel")
        title=file_name+'_mfcc'
        plt.title(title)
        plt.colorbar()
        #mfccs = np.hstack((mfccs, mfccs))
        #mfccs = np.hstack((mfccs, mfccs))
        print("mfccs shape_long", mfccs.shape)
        #np.savetxt(dir_path+'mfcc/'+file_name+'_mfcc.csv', mfccs, delimiter=',')
fig.savefig(dir_path+'mfcc2.png', bbox_inches='tight')
#plt.show()

"""
fig=plt.figure(figsize=(20, 5*count))
for file_name in file_list:
    path, ext = os.path.splitext(file_name)
    if ext == ".wav":
        file_path = dir_path+file_name
        x, fs = librosa.load(file_path, sr=None)
        wf = wave.open(file_path, "r")
        file_name = os.path.splitext(file_name)[0] #拡張子のwavを取るため
        print(file_name)
        plt.subplot(count, 1, i + 1)
        i += 1

        # 4. メル周波数で均等になるようBINを集めてスムージングする
        #   - binパワー計算済みのSを利用して、メルフィルタバンクをあてる
        melsp = librosa.feature.melspectrogram(y=x, sr=fs, n_mels=128)
        #librosa.display.specshow(melsp, sr=fs, x_axis='time', y_axis="hz")
        #title=file_name+'_melsp'
        #plt.title(file_name)
        #plt.colorbar()

        # 5. 離散コサイン変換する（低次項を取る）
        # デフォルト n_mfcc = 20bin
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(melsp), n_mfcc=128)
        print("mfccs shape", mfccs.shape)

        # 一致判定
        flag =  np.allclose(mfccs, librosa.feature.mfcc(y=x, sr=fs ,n_mfcc=128))
        # 結果表示
        print(flag) # True or false

        # 標準化して可視化
        import sklearn
        import matplotlib.cm as cm
        #mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
        librosa.display.specshow(mfccs, sr=fs, x_axis='time')
        title=file_name+'_mfcc'
        plt.title(title)
        plt.colorbar()
        #mfccs = np.hstack((mfccs, mfccs))
        #mfccs = np.hstack((mfccs, mfccs))
        #print("mfccs shape_long", mfccs.shape)
        np.savetxt(dir_path+'mfcc/'+file_name+'_mfcc.csv', mfccs, delimiter=',')
fig.savefig(dir_path+'mfcc2.png', bbox_inches='tight')
#plt.show()
"""
