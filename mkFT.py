# 参考サイト：https://qiita.com/martin-d28jp-love/items/34161f2facb80edd999f

#%matplotlib inline #jupyterの表示に必要
import matplotlib.pyplot as plt, librosa, librosa.display, urllib
import numpy as np
import os
import sys

#dir_path='./wavSrc/tmp/nsynth/nsynth-test/guitar_acoustic/'
#dir_path='./wavSrc/tmp/nsynth/nsynth-test/bass_electronic/'
#dir_path='./wavSrc/tmp/nsynth/nsynth-test/bass_synthetic/'
#dir_path='./wavSrc/tmp/nsynth/nsynth-test/brass_acoustic/'
#dir_path='./wavSrc/tmp/nsynth/nsynth-test/flute_acoustic/'
dir_path='./wavSrc/tmp/nsynth/nsynth-test/flute_synthetic/'
#dir_path='./wavSrc/tmp/nsynth/nsynth-test/guitar_acoustic/'
#dir_path='./wavSrc/tmp/nsynth/nsynth-test/guitar_electronic/'
#dir_path='./wavSrc/tmp/nsynth/nsynth-test/keyboard_acoustic/'
#dir_path='./wavSrc/tmp/nsynth/nsynth-test/keyboard_electronic/'
#dir_path='./wavSrc/tmp/nsynth/nsynth-test/keyboard_synthetic/'
#dir_path='./wavSrc/tmp/nsynth/nsynth-test/mallet_acoustic/'
#dir_path='./wavSrc/tmp/nsynth/nsynth-test/organ_electronic/'
#dir_path='./wavSrc/tmp/nsynth/nsynth-test/reed_acoustic/'
#dir_path='./wavSrc/tmp/nsynth/nsynth-test/string_acoustic/'
#dir_path='./wavSrc/tmp/nsynth/nsynth-test/vocal_acoustic/'
#dir_path='./wavSrc/tmp/nsynth/nsynth-test/vocal_synthetic/'

#dir_path='./wavSrc/instrument/same_instrument/'

#dir_path = './wavSrc/tmp/'
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
file_list.sort()

print('\n\n', file_list, '\n')
"""
if not os.path.isdir(dir_path+'FT/'):
    os.mkdir(dir_path+'FT/') # make dir for shortform csv
"""

#fig=plt.figure(figsize=(20, 5*count))
for file_name in file_list:
    fig=plt.figure(figsize=(20, 5))
    path, ext = os.path.splitext(file_name)
    if ext == ".wav":
        file_path = dir_path+file_name
        x, fs = librosa.load(file_path, sr=None)
        file_name = os.path.splitext(file_name)[0] #拡張子のwavを取るため
        #data = np.abs(librosa.stft(x, n_fft=1024, hop_length=512))**2
        data = librosa.stft(x, n_fft=1024, hop_length=512)
        #plt.subplot(count, 1, i + 1)
        #i += 1

        print(file_name)
        #print(data)
        print(data.shape)
        #print(type(data))
        #librosa.display.specshow(data, y_axis='linear', x_axis='time')
        #librosa.display.specshow(data, y_axis='log', x_axis='time')
        # numpyを保存
        #np.savetxt(dir_path+'FT/'+file_name+'_ft.csv', data, delimiter=',')

        data = librosa.amplitude_to_db(data, ref=np.max)
        #data = D[0:400,:]
        data = np.hstack((data, data))
        data = np.hstack((data, data))
        print(data.shape)
        #librosa.display.specshow(data, y_axis='linear', x_axis='time')
        librosa.display.specshow(data, y_axis='log', x_axis='time')
        #plt.pcolor(data, cmap='magma')

        plt.title(file_name)
        plt.colorbar(format='%+2.0f dB')
        plt.show()
        #np.savetxt(dir_path+'FT/'+file_name+'_D.csv', data, delimiter=',')
        """
        C = librosa.feature.chroma_cqt(y=x, sr=fs)
        C = C[:,0:688*2]
        tempo, beat_f = librosa.beat.beat_track(y=x, sr=fs, trim=False)
        beat_f = librosa.util.fix_frames(beat_f, x_max=C.shape[1])
        Csync = librosa.util.sync(C, beat_f, aggregate=np.median)
        beat_t = librosa.frames_to_time(beat_f, sr=fs)
        librosa.display.specshow(Csync, y_axis='chroma', x_axis='time', x_coords=beat_t)
        plt.colorbar()
        """
#fig.savefig(dir_path+'FT.png', bbox_inches='tight')
#plt.show()
