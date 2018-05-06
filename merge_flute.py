import matplotlib.pyplot as plt, librosa, librosa.display, urllib
import numpy as np
import os
import sys

def saveWav(audio, dir_path, file_name, fs):
    if not os.path.isdir(dir_path+'merge/'):
        os.mkdir(dir_path+'merge/') # make dir for shortform csv

    #audio = librosa.db_to_amplitude(audio)
    #audio = np.sqrt(audio)
    print(audio.shape)

    y_hat = librosa.istft(audio)
    # librosa.istftにはstftのshapeの(freq, time)のものを入力する．
    print(y_hat.shape)

    # instrumentのsampling rateは16000
    # 一般的なCDのsampling rateは44100
    #sr=44100
    #sr=16000
    file_name=os.path.splitext(file_name)[0]
    librosa.output.write_wav(dir_path+'merge/'+file_name+'_tmp.wav', y_hat, sr=fs)


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
dir_path = './wavSrc/instrument/tmp/'
plt.rcParams['font.size'] = 18

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
count += 1

file_path1 = 'flute_synthetic_000-104-025.wav'
file_path2 = 'flute_acoustic_002-081-075.wav'
file_path3='flute_synthetic_000-100-075.wav'
file_path4='flute_synthetic_000-101-127.wav'

fig=plt.figure(figsize=(20, 8*count))
x1, fs1 = librosa.load(dir_path+file_path1, sr=None)
stft1 = librosa.stft(x1, n_fft=1024, hop_length=512)
#stft1 = np.abs(librosa.stft(x1, n_fft=1024, hop_length=512))**2
x2, fs2 = librosa.load(dir_path+file_path2, sr=None)
stft2 = librosa.stft(x2, n_fft=1024, hop_length=512)
#stft2 = np.abs(librosa.stft(x2, n_fft=1024, hop_length=512))**2
x3, fs3 = librosa.load(dir_path+file_path3, sr=None)
stft3 = librosa.stft(x3, n_fft=1024, hop_length=512)
#stft3 = np.abs(librosa.stft(x3, n_fft=1024, hop_length=512))**2
x4, fs4 = librosa.load(dir_path+file_path4, sr=None)
stft4 = librosa.stft(x3, n_fft=1024, hop_length=512)
#stft3 = np.abs(librosa.stft(x3, n_fft=1024, hop_length=512))**2


for _ in range(4):
    stft2=np.hstack((stft2,stft2))
    stft3=np.hstack((stft3,stft3))
stft2=stft2[:,0:stft1.shape[1]]
stft3=stft3[:,0:stft1.shape[1]]
stft=stft1+stft2+stft3+stft4
file_name = os.path.splitext(file_path1)[0] +'+'+os.path.splitext(file_path3)[0]
print(stft.shape,'\n',stft1.shape,'\n',stft2.shape,'\n')


saveWav(audio=stft1,dir_path=dir_path,file_name=file_path1, fs=fs1)
#saveWav(audio=stft2,dir_path=dir_path,file_name=file_path2, fs=fs2)
saveWav(audio=stft3,dir_path=dir_path,file_name=file_path3, fs=fs3)
saveWav(audio=stft,dir_path=dir_path,file_name='merge', fs=fs3)

"""
D1 = librosa.amplitude_to_db(stft1, ref=np.max)
D1 = D1[0:120,:]
D2 = librosa.amplitude_to_db(stft2, ref=np.max)
D2 = D2[0:120,:]
D = np.hstack((D1, D2))
D=np.hstack((D,D))
print(D.shape)
"""

plt.subplot(count, 1, 1)
#librosa.display.specshow(stft, y_axis='linear', x_axis='time', cmap='gray_r')
librosa.display.specshow(librosa.amplitude_to_db(stft,ref=np.max),y_axis='log', x_axis='time', cmap='gray_r')
#librosa.display.specshow(librosa.amplitude_to_db(stft,ref=np.max),y_axis='log', x_axis='time')
#plt.gca().invert_yaxis()
plt.title(file_name)
plt.colorbar(format='%+2.0f dB')

plt.subplot(count, 1, 2)
#librosa.display.specshow(stft1, y_axis='linear', x_axis='time', cmap='gray_r')
librosa.display.specshow(librosa.amplitude_to_db(stft1,ref=np.max),y_axis='log', x_axis='time', cmap='gray_r')
#librosa.display.specshow(librosa.amplitude_to_db(stft1,ref=np.max),y_axis='log', x_axis='time')
#plt.gca().invert_yaxis()
plt.colorbar(format='%+2.0f dB')
plt.title(file_path1)

plt.subplot(count, 1, 3)
#librosa.display.specshow(stft2, y_axis='linear', x_axis='time', cmap='gray_r')
librosa.display.specshow(librosa.amplitude_to_db(stft2,ref=np.max),y_axis='log', x_axis='time', cmap='gray_r')
#librosa.display.specshow(librosa.amplitude_to_db(stft2,ref=np.max),y_axis='log', x_axis='time')
#plt.gca().invert_yaxis()
plt.colorbar(format='%+2.0f dB')
plt.title(file_path2)

plt.subplot(count, 1, 4)
#librosa.display.specshow(stft3, y_axis='linear', x_axis='time', cmap='gray_r')
librosa.display.specshow(librosa.amplitude_to_db(stft3,ref=np.max),y_axis='log', x_axis='time', cmap='gray_r')
#librosa.display.specshow(librosa.amplitude_to_db(stft3,ref=np.max),y_axis='log', x_axis='time')
#plt.gca().invert_yaxis()
plt.colorbar(format='%+2.0f dB')
plt.title(file_path3)

plt.subplot(count, 1, 5)
#librosa.display.specshow(stft3, y_axis='linear', x_axis='time', cmap='gray_r')
librosa.display.specshow(librosa.amplitude_to_db(stft4,ref=np.max),y_axis='log', x_axis='time', cmap='gray_r')
#librosa.display.specshow(librosa.amplitude_to_db(stft3,ref=np.max),y_axis='log', x_axis='time')
#plt.gca().invert_yaxis()
plt.colorbar(format='%+2.0f dB')
plt.title(file_path4)

#np.savetxt(dir_path+file_name+'_tmp.csv', stft, delimiter=',')
#fig.savefig(dir_path+'FT.png', bbox_inches='tight')
plt.show()
