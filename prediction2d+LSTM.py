# https://github.com/jkatsuta/viz_nn/blob/master/viz_cnn.ipynb
#"""not to show on display on GPU server
import matplotlib
matplotlib.use('Agg')
#"""
import numpy as np
import matplotlib.pyplot as plt, librosa, librosa.display, urllib
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import model_from_json
from keras.utils import plot_model
from keras import backend as K
#import loadCsv as load
import loadWav as load
import sys
import os

plt.rcParams['font.size'] = 18
#log_name = '01190211filter=3'
#log_name='01231253filter=6'
#src_dir = './wavSrc/music/'

#src_dir='./wavSrc/time_change/instrument/'
#src_dir='./wavSrc/time_change/instrument/test_data/'
#log_name='01190211filter=3' #good result
#log_name='02051039filter=3'

#src_dir='./wavSrc/instrument/same_instrument/'
#src_dir='./wavSrc/instrument/same_instrument/test_data/'
#log_name='01310916filter=4'
#log_name='02031526filter=1'
#log_name='02031526filter=5'
#log_name='02051654filter=3'
#log_name='02052255filter=6' #filter=1 or 5 or 6

src_dir='./wavSrc/single+chord/2349.318+/'
log_name='02060415filter=4'


#src_dir='./wavSrc/instrument/'
#src_dir='./wavSrc/instrument/test_data/'
#log_name='01291308filter=6'
#log_name='01291739filter=3'
#log_name='01291830filter=4' #1層のconv kernelsize=(50,50)でのやつ
#log_name='01291739filter=3' #1層のconv kernelsize=(50,0)でのやつ
#log_name='01300028filter=' #1層のconv kernelsize=(100,0)でのやつ

fav_label = src_dir+"fav.txt"

output_dir = './output/my2dCNN+LSTM/'

print('\n\nlog_name : ', log_name,
'\nsrc_dir : ', src_dir,
'\noutput_dir : ', output_dir)

#os.mkdir("sample")

def saveWav(audio, dir_path, file_name, sr=16000):

    if not os.path.isdir(dir_path+'filtered/'):
        os.mkdir(dir_path+'filtered/') # make dir for shortform csv

    audio = librosa.db_to_amplitude(audio, ref=np.max)
    #audio = librosa.db_to_amplitude(audio)
    print(audio.shape)

    audio = np.sqrt(audio)
    y_hat = librosa.istft(audio, hop_length=512)
    # librosa.istftにはstftのshapeの(freq, time)のものを入力する．
    print(y_hat.shape)

    # instrumentのsampling rateは16000
    # 一般的なCDのsampling rateは44100
    #sr=44100
    #sr=16000
    librosa.output.write_wav(dir_path+'filtered/'+file_name+'.wav', y_hat, sr=sr)

def recall_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def viz_layer_all_filter(l_num=None, model=None, input_data=None, log_name='tmp', data_file='None'):

    if l_num is None:
        return print("l_num is None")
    if model is None:
        return print("model is None")
    if input_data is None:
        return print("input_data is None")

    """maxpoolingの層では実行できない
    w1 = model.layers[l_num].get_weights()[0] # l_num層目のレイヤーのweights
    b1 = model.layers[l_num].get_weights()[1] # とbiasのshape
    print(
    "\nw1.shape ", w1.shape,
    "\nb1.shape ", b1.shape,
    "\ntrain_music.shape ", train_music.shape
    )
    """
    # l_num層目までを取得
    layers = model.layers.copy()[:l_num]
    # 新しいmodel(model2)にlayerを入れる
    model2 = Sequential()
    for layer in layers:
        model2.add(layer)

    model2.compile(loss='binary_crossentropy', optimizer='adam')  # compileするだけ (loss, optimizerは使わない)

    features = model2.predict(input_data)
    print(l_num, "層目, layer name : ", model.layers[l_num])
    print("features.shape : ",features.shape)

    max = np.amax(features)
    min = np.amin(features)
    print("max : ", max)
    print("min : ", min)

    for i, img in enumerate(features):
        max = np.amax(img)
        min = np.amin(img)
        #print("max : ", max)
        #print("min : ", min)
        print('img.shape : ', img.shape)
        #index = 0
        index = img.shape[1] # 表示させたいfilter番号まですべてのfilter表示させる.
        for index in range(index):
            audio = np.array([])
            file_name = os.path.splitext(data_file[i][0])[0] #拡張子のwavを取るため
            #fig = plt.figure(figsize=(20, 5*features.shape[1]))
            fig = plt.figure(figsize=(int(20*features.shape[3]/688), 5*features.shape[1]))
            #print("features.shape : ",features.shape)
            #for j in range(img.shape[0]):
            for j in range(img.shape[0]):
                plt.subplot(features.shape[1], 1, j + 1)
                #plt.gca().invert_yaxis()
                #plt.pcolor(img[j,index], vmin=min, vmax=max, cmap='gray_r')
                librosa.display.specshow(img[j,index], y_axis='log', x_axis='time')
                #librosa.display.specshow(img[j,index], y_axis='linear', x_axis='time')
                plt.colorbar(format='%+2.0f dB')
                if j==0:
                    title = data_file[i][0]+'_layer'+str(l_num)+'filter_no'+str(index)
                    plt.title(title, fontsize=20)
                    audio=img[j,index]
                else:
                    #audio=np.vstack((audio, img[j,index]))
                    audio=np.hstack((audio, img[j,index]))
            print('audio.shape : ', audio.shape)
            #saveWav(audio=audio, dir_path=output_dir, file_name=file_name+'_input'+str(i+1)+'-'+str(j+1)+'_layer'+str(l_num)+'_filter_no'+str(index))
            fig.savefig(output_dir+'layeroutput/'+log_name+'_'+file_name+'_input'+str(i+1)+'_layer'+str(l_num)+'_filter_no'+str(index)+'.png', bbox_inches='tight')

def viz_layer_1_filter(l_num=None, model=None, input_data=None, log_name='tmp', data_file='None'):

    if l_num is None:
        return print("l_num is None")
    if model is None:
        return print("model is None")
    if input_data is None:
        return print("input_data is None")

    # l_num層目までを取得
    layers = model.layers.copy()[:l_num]
    # 新しいmodel(model2)にlayerを入れる
    model2 = Sequential()
    for layer in layers:
        model2.add(layer)

    model2.compile(loss='binary_crossentropy', optimizer='adam')  # compileするだけ (loss, optimizerは使わない)

    features = model2.predict(input_data)
    print(l_num, "層目, layer name : ", model.layers[l_num])
    print("features.shape : ",features.shape)

    max = np.amax(features)
    min = np.amin(features)
    print("max : ", max)
    print("min : ", min)

    for i, img in enumerate(features):
        fig = plt.figure(figsize=(20, 5*features.shape[1]))
        max = np.amax(img)
        min = np.amin(img)
        index = 3# 表示したいfilter番号,今は3個目のfilterを表示させている.
        for j, img in enumerate(img):
            plt.subplot(features.shape[1], 1, j + 1)
            plt.gca().invert_yaxis()
            plt.pcolor(img[index], vmin=min, vmax=max, cmap='gray_r')
            plt.colorbar()
            if j==0:
                title = data_file[i][0]+'_layer'+str(l_num)+'filter_no'+str(index)
                plt.title(title, fontsize=20)
        file_name = os.path.splitext(data_file[i][0])[0] #拡張子のwavを取るため
        fig.savefig(output_dir+'layeroutput/'+log_name+'_'+file_name+'_input'+str(i+1)+'_layer'+str(l_num)+'_filter_no'+str(index)+'.png', bbox_inches='tight')

def visualize_filter(model, l_num=1):
    from sklearn.preprocessing import MinMaxScaler
    # 最初の畳み込み層の重みを取得
    # tf => (nb_row, nb_col, nb_channel, nb_filter)
    # th => (nb_filter, nb_channel, nb_row, nb_col)
    W = model.layers[l_num].get_weights()[0]
    print(W.shape)
    #W = model.layers[1].get_weights()

    # 次元を並べ替え
    #if K.image_dim_ordering() == 'tf':
        # (nb_filter, nb_channel, nb_row, nb_col)
    W = W.transpose(3, 2, 0, 1)

    nb_filters, nb_channel, nb_row, nb_col = W.shape

    print(W.shape)

    """
    # 32個（手抜きで固定）のフィルタの重みを描画
    fig=plt.figure(figsize=(20, 5*4))
    for i in range(nb_filters):
        # フィルタの画像
        im = W[i, 0]

        # 重みを0-255のスケールに変換
        scaler = MinMaxScaler(feature_range=(0, 255))
        im = scaler.fit_transform(im)

        plt.subplot(1, nb_filters, i + 1)
        plt.axis('off')
        plt.imshow(im, cmap='gray_r')
    """
    #""" to make line graph


    for i, W in enumerate(W):
        count=0
        fig=plt.figure(figsize=(20*2, 5*4*nb_filters))
        for j, W in enumerate(W):
            print('W.shape : ', W.shape)
            print('count : ', count)
            x = np.linspace(start=0, stop=W.shape[1]-1, num=W.shape[1])
            print('x.shape : ', x.shape)
            y = W[0]
            print('y.shape : ', y.shape)
            plt.subplot(nb_filters, 1, count+1)
            plt.plot(x,y,linewidth=6)
            count+=1
        fig.savefig(output_dir+'layeroutput/graph_channel'+str(i)+'.png', bbox_inches='tight')
    #plt.show()
    #"""


def filtered(l_num=None, model=None, input_data=None, log_name='tmp', data_file='None'):
    # l_num層目までを取得
    input = model.layers.copy()[0]
    layers = model.layers.copy()[l_num]
    # 新しいmodel(model2)にlayerを入れる
    model2 = Sequential()
    model2.add(input)
    model2.add(layers)
    model2.compile(loss='binary_crossentropy', optimizer='adam')  # compileするだけ (loss, optimizerは使わない)

    features = model2.predict(input_data)
    print(l_num, "層目, layer name : ", model.layers[l_num])
    print("features.shape : ",features.shape)
    max = np.amax(features)
    min = np.amin(features)
    print("max : ", max)
    print("min : ", min)

    for i, img in enumerate(features):
        max = np.amax(img)
        min = np.amin(img)
        #print("max : ", max)
        #print("min : ", min)
        print(img.shape)
        #index = 0
        index = img.shape[1] # 表示させたいfilter番号まですべてのfilter表示させる.
        for index in range(index):
            audio = np.array([])
            file_name = os.path.splitext(data_file[i][0])[0] #拡張子のwavを取るため
            fig = plt.figure(figsize=(20, 5*features.shape[1]))
            for j in range(img.shape[0]):
                plt.subplot(features.shape[1], 1, j + 1)
                #plt.gca().invert_yaxis()
                #plt.pcolor(img[j,index], vmin=min, vmax=max, cmap='gray_r')
                librosa.display.specshow(img[j,index], y_axis='log', x_axis='time', cmap='gray_r')
                #plt.colorbar()
                plt.colorbar(format='%+2.0f dB')

                if j==0:
                    title = data_file[i][0]+'_layer'+str(l_num)+'filter_no'+str(index)
                    plt.title(title, fontsize=20)
                    audio=img[j,index]
                else:
                    audio=np.vstack((audio, img[j,index]))
                    #audio=np.hstack((audio, img[j,index]))
            print('audio.shape : ', audio.shape)
            #saveWav(audio=audio, dir_path=output_dir, file_name=file_name+'_input'+str(i+1)+'-'+str(j+1)+'_layer'+str(l_num)+'_filter_no'+str(index))
            fig.savefig(output_dir+'layeroutput/'+log_name+'_'+file_name+'_input'+str(i+1)+'_layer'+str(l_num)+'_filter_no'+str(index)+'.png', bbox_inches='tight')


# モデルを読み込む
model = None
with open(output_dir+'model'+log_name+'.json') as f:
    model = model_from_json(f.read())

# 学習結果を読み込む
model.load_weights(output_dir+'weight'+log_name+'.h5')

#from keras.utils import plot_model
#plot_model(model, to_file=output_dir+log_name+'_model.png', show_shapes=True)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', recall_score])


input_shape = model.layers.copy()[0].input_shape

print('\n{:-^60}'.format('train'))
train_music, train_label, data_file = load.time_dim3_overlap(dir_path = src_dir, fav_label=fav_label)
#train_music, train_label, data_file = load.for_prediction(dir_path=src_dir, fav_label=fav_label, input_shape=input_shape, OVERLAP=False, data_format='dB')
train_music = np.expand_dims(train_music, axis=2)

print('train_music.shape : ', train_music.shape) #=>  (sample, time_step, channel, time, freq) = (8, 5, 1, 688, 120)

score = model.evaluate(train_music, train_label, verbose=0)
print('Test loss :', score[0])
print('Test accuracy :', score[1])
print('test mse:', score[2])
print('test recall:', score[3])

features = model.predict(train_music)
print("features.shape : ", features.shape)
print("max : ", np.amax(features))
print("min : ", np.amin(features))

print(features.shape)

print('\n{:-^60}'.format('predict'))
print('No.   data_name   output   answer')
for i in range(features.shape[0]):
    print('{:-^60}'.format(''))
    print('{0:02d}'.format(i+1), data_file[i], features[i], train_label[i])
print('{:-^60}\n'.format(''))

"""
count = 0
wav_name = []
evaluate = []

files = os.listdir(src_dir)
for file_name in files:
    path, ext = os.path.splitext(file_name)
    if ext == ".wav":
        count += 1
        wav_name.append(file_name)
if count == 0:
    print('\nThere is no data\n')
    sys.exit(1)

for file_name in wav_name:
    tmp_evaluate = 0
    sample = 1
    for i in range(features.shape[0]):
        if file_name == data_file[i]:
            tmp_evaluate += float(features[i][0])
            sample += 1
    evaluate.append(tmp_evaluate/sample)

print('\n{:-^60}'.format('evaluate'))
print('No.   data_name   evaluate')
for i, file_name in enumerate(wav_name):
    print('{:-^60}'.format(''))
    print('{0:02d}'.format(i+1), file_name, evaluate[i])
print('{:-^60}\n'.format(''))
"""

#"""
# Kerasでは、以下のように簡単にレイヤーごとのパラメータが取得できる
lays = model.layers # list of the layers

for i, l in enumerate(lays):
    print(i, l)

for i, img in enumerate(train_music):
    fig = plt.figure(figsize=(20, 5*train_music.shape[1]))
    max = np.amax(train_music[i])
    min = np.amin(train_music[i])
    #print("max : ", max)
    #print("min : ", min)
    #print(img.shape)
    for j, img in enumerate(img):
        plt.subplot(train_music.shape[1], 1, j + 1)
        #plt.gca().invert_yaxis()
        #plt.pcolor(img[0], vmin=min, vmax=max, cmap='magma')
        librosa.display.specshow(img[0], y_axis='log', x_axis='time')
        #librosa.display.specshow(img[0], y_axis='linear', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        if j==0:
            plt.title(data_file[i][0], fontsize=20)
            audio=img[0]
        else:
            audio=np.vstack((audio, img[0]))
        #plt.show()
    file_name = os.path.splitext(data_file[i][0])[0] #拡張子のwavを取るため
    fig.savefig(output_dir+'layeroutput/'+log_name+'_'+file_name+'_input'+str(i+1)+'_inputlayer.png', bbox_inches='tight')
    #saveWav(audio=audio, dir_path=output_dir, file_name=file_name+'_input'+str(i+1)+'_inputlayer')
#"""
"""
for i,_ in enumerate(lays):
    if i+1<=8 and i+1>=2:
        #7層目まで出力
        viz_layer_all_filter(l_num=i+1, model=model, input_data=train_music, log_name=log_name, data_file=data_file)

for i,_ in enumerate(lays):
    if i+1<=8 and i+1>=2:
        #7層目まで出力
        viz_layer_all_filter(l_num=i+1, model=model, input_data=train_music, log_name=log_name, data_file=data_file)
"""
#"""
visualize_filter(model, l_num=1)
visualize_filter(model, l_num=4)
visualize_filter(model, l_num=7)
visualize_filter(model, l_num=10)
#"""
#"""
viz_layer_all_filter(l_num=2, model=model, input_data=train_music, log_name=log_name, data_file=data_file)
viz_layer_all_filter(l_num=5, model=model, input_data=train_music, log_name=log_name, data_file=data_file)
viz_layer_all_filter(l_num=8, model=model, input_data=train_music, log_name=log_name, data_file=data_file)
viz_layer_all_filter(l_num=11, model=model, input_data=train_music, log_name=log_name, data_file=data_file)
#"""
"""
train = train_music[1]
train = np.expand_dims(train, axis=0)
print(train.shape)
filtered(l_num=4, model=model, input_data=train, log_name=log_name, data_file=data_file[1])
"""
