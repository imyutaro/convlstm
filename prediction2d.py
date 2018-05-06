# https://github.com/jkatsuta/viz_nn/blob/master/viz_cnn.ipynb
#"""not yo show on display on GPU server
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

#src_dir='./wavSrc/time_change/instrument/test_data/'
#log_name='02051010filter=1'
#log_name='02051057filter=1'

src_dir='./wavSrc/instrument/same_instrument/'
#log_name='01311856filter=3'
#log_name='02011129filter=1'
#log_name='02011241filter=8'
#log_name='02021037filter=1'
#log_name='02021440filter=3'
log_name='02051735filter=4'

fav_label=src_dir+'fav.txt'
output_dir='./output/my2dCNN/'

print('\n\nlog_name : ', log_name,
'\nsrc_dir : ', src_dir,
'\noutput_dir : ', output_dir)

def saveWav(audio, dir_path, file_name):

    if not os.path.isdir(dir_path+'filtered/'):
        os.mkdir(dir_path+'filtered/') # make dir for shortform csv

    #audio = librosa.db_to_amplitude(audio, ref=np.max)
    print(audio.shape)

    y_hat = librosa.istft(audio, hop_length=512)
    # librosa.istftにはstftのshapeの(freq, time)のものを入力する．
    print(y_hat.shape)

    # sampling rateは16000
    librosa.output.write_wav(dir_path+'filtered/'+file_name+'.wav', y_hat, sr=16000)

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
        #print('img.shape'.img.shape)
        #index = 0
        file_name = os.path.splitext(data_file[i][0])[0] #拡張子のwavを取るため
        index = img.shape[0] # 表示させたいfilter番号まですべてのfilter表示させる.
        #fig = plt.figure(figsize=(20, 5*features.shape[1]))
        fig = plt.figure(figsize=(int(20*img.shape[2]/688), 5*img.shape[0]))
        #print("features.shape : ",features.shape)
        for index in range(index):
            audio = np.array([])
            plt.subplot(img.shape[0], 1, index+1)
            #plt.gca().invert_yaxis()
            #plt.pcolor(img[index], vmin=min, vmax=max, cmap='gray_r')
            librosa.display.specshow(img[index], y_axis='log', x_axis='time')
            #librosa.display.specshow(img[index], y_axis='linear', x_axis='time', cmap='gray_r')
            plt.colorbar(format='%+2.0f dB')
            plt.title('filter_no %d' % (index))
            audio=img[index]
            print('audio.shape : ', audio.shape)
            saveWav(audio=audio, dir_path=output_dir, file_name=file_name+'_input'+str(i+1)+'_layer'+str(l_num)+'_filter_no'+str(index))
        title = file_name+'_layer'+str(l_num)
        plt.suptitle(title)
        plt.subplots_adjust(hspace=0.3)
        fig.savefig(output_dir+'layeroutput/'+log_name+'_'+file_name+'_input'+str(i+1)+'_layer'+str(l_num)+'.png', bbox_inches='tight')

def viz_layer(l_num=None, model=None, input_data=None, log_name='tmp', data_name='None'):

    if l_num==None:
        return print("l_num is None")
    elif model==None:
        return print("model is None")
    #elif input_data==None:
        #return print("input_data is None")

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

    #model2.compile(loss='categorical_crossentropy', optimizer='adam')  # compileするだけ (loss, optimizerは使わない)
    model2.compile(loss='binary_crossentropy', optimizer='adam')  # compileするだけ (loss, optimizerは使わない)

    features = model2.predict(input_data)
    print(l_num, "層目, layer name : ", model.layers[l_num])
    print("features.shape : ",features.shape)

    max = np.amax(features)
    min = np.amin(features)
    print("max : ", max)
    print("min : ", min)


    fig = plt.figure(figsize=(20, 70))
#    fig = plt.figure(figsize=(20, 10))
    #plt.title(2,3,str(l_num)+'層目 '+str(model.layers[l_num])
    #plt.figure(figsize=(20, 50))
    index = 126
    #max = np.amax(features)
    #min = np.amin(features)
    for i, img in enumerate(features):
        #plt.subplot(2, 5, i + 1) # for music
        #plt.subplot(1, 4, i + 1) # for chord
        #plt.subplot(4, 2, i + 1) # for simple chord
        print(data_name[i])
        max = np.amax(img[index])
        min = np.amin(img[index])
        #iindex = img.argmax(axis=0)
        print("max : ", max)
        print("min : ", min)
        plt.subplot(input_data.shape[0], 1, i + 1)
        #plt.axis('off')
         #img = np.reshape(img, (img.shape[1],img.shape[0]))
        #librosa.display.specshow(img, x_axis='time', y_axis="mel")
        #librosa.display.specshow(img, x_axis='linear', y_axis="time")
        plt.pcolor(img[index], vmin=min, vmax=max, cmap="magma")
        plt.xticks( np.arange(0, input_data.shape[3], 10) )
        title = data_name[i][0]+'_filter '+str(index)
        plt.title(title)
        plt.colorbar()
        np.savetxt('./output/my2dCNN/layeroutput/filter_no'+str(index)+'_layer'+str(l_num)+'_'+data_name[i][0]+'.csv', img[index], delimiter=',')
    #plt.show()
    fig.savefig('./output/my2dCNN/layeroutput/'+log_name+'_layer'+str(l_num)+'.png', bbox_inches='tight')
    print("finish saving layer"+str(l_num))
"""
def visualize_filter(layer_name, filter_index, num_loops=200):
    #指定した層の指定したフィルタの出力を最大化する入力画像を勾配法で求める

    # 指定した層の指定したフィルタの出力の平均
    activation_weight = 1.0
    if layer_name == 'predictions':
        # 出力層だけは2Dテンソル (num_samples, num_classes)
        activation = activation_weight * K.mean(layer.output[:, filter_index])
    else:
        # 隠れ層は4Dテンソル (num_samples, row, col, channel)
        activation = activation_weight * K.mean(layer.output[:, :, :, filter_index])

    # 層の出力の入力画像に対する勾配を求める
    # 入力画像を微小量変化させたときの出力の変化量を意味する
    # 層の出力を最大化したいためこの勾配を画像に足し込む
    grads = K.gradients(activation, input_tensor)[0]

    # 正規化トリック
    # 画像に勾配を足し込んだときにちょうどよい値になる
    grads /= (K.sqrt(K.mean(K.square(grads))) + K.epsilon())

    # 画像を入力して層の出力と勾配を返す関数を定義
    iterate = K.function([input_tensor], [activation, grads])

    # ノイズを含んだ画像（4Dテンソル）から開始する
    x = np.random.random((1, img_height, img_width, 3))
    x = (x - 0.5) * 20 + 128

    # 勾配法で層の出力（activation_value）を最大化するように入力画像を更新する
    cache = None
    for i in range(num_loops):
        activation_value, grads_value = iterate([x])
        # activation_valueを大きくしたいので画像に勾配を加える
        step, cache = rmsprop(grads_value, cache)
        x += step
        print(i, activation_value)

    # 画像に戻す
    img = deprocess_image(x[0])

    return img
"""
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
    count=0
    fig=plt.figure(figsize=(20*2, 5*4*nb_filters*nb_channel))
    for i, W in enumerate(W):
        for j, W in enumerate(W):
            print('W.shape : ', W.shape)
            print('count : ', count)
            x = np.linspace(start=0, stop=W.shape[1]-1, num=W.shape[1])
            print('x.shape : ', x.shape)
            y = W[0]
            print('y.shape : ', y.shape)
            plt.subplot(nb_filters*nb_channel, 1, count+1)
            plt.plot(x,y,linewidth=6)
            count+=1
    #plt.show()
    #"""
    fig.savefig(output_dir+'layeroutput/graph'+str(l_num)+'.png', bbox_inches='tight')

# モデルを読み込む
model = None
with open('./output/my2dCNN/model'+log_name+'.json') as f:
    model = model_from_json(f.read())
#plot_model(model, to_file='model.png', show_shapes=True)
# 学習結果を読み込む
model.load_weights('./output/my2dCNN/weight'+log_name+'.h5')

print('\n{:-^60}'.format('train'))
train_music, train_label, data_file = load.dim3(dir_path=src_dir, fav_label=fav_label)
input_shape = (None, train_music.shape[1], train_music.shape[2], train_music.shape[3])

model.summary()

#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', recall_score])

score = model.evaluate(train_music, train_label, verbose=0)
print('Test loss :', score[0])
print('Test accuracy :', score[1])
print('test mse:', score[2])
print('test recall:', score[3])

features = model.predict(train_music)
print("features.shape : ", features.shape)
print("max : ", np.amax(features))
print("min : ", np.amin(features))

print('\n{:-^60}'.format('predict'))
print('No.   data_name   output   answer')
for i in range(features.shape[0]):
    print('{:-^60}'.format(''))
    print('{0:02d}'.format(i+1), data_file[i], features[i], train_label[i])
print('{:-^60}\n'.format(''))

#"""
# Kerasでは、以下のように簡単にレイヤーごとのパラメータが取得できる
lays = model.layers # list of the layers
for i, l in enumerate(lays):
    print(i, l)

max = np.amax(train_music)
min = np.amin(train_music)
print("max : ", max)
print("min : ", min)

fig = plt.figure(figsize=(20, 5*train_music.shape[0]))
for i, img in enumerate(train_music):
    #plt.subplot(2, 5, i + 1) # for music
    #plt.subplot(1, 4, i + 1) # for chord
    #plt.subplot(4, 2, i + 1) # for simple chord
    plt.subplot(train_music.shape[0], 1, i + 1)
    #img = np.reshape(img, (img.shape[1],img.shape[0]))
    librosa.display.specshow(img[0], x_axis='time', y_axis="log")
    #librosa.display.specshow(img, x_axis='linear', y_axis="time")
    #plt.pcolor(img[0], vmin=min, vmax=max, cmap="magma")
    plt.title(data_file[i][0])
    plt.colorbar()
    file_name = os.path.splitext(data_file[i][0])[0] #拡張子のwavを取るため
    print('img[0].shape : ', img[0].shape)
    saveWav(audio=img[0], dir_path=output_dir, file_name=file_name+'_inputlayer')
plt.subplots_adjust(hspace=0.4)
#plt.show()
fig.savefig('./output/my2dCNN/layeroutput/'+log_name+'_input_layer.png', bbox_inches='tight')
#"""
"""
for i,_ in enumerate(lays):
    if i+1<=8 and i+1>=2:
        #7層目まで出力
        viz_layer(l_num=i+1, model=model, input_data=train_music, log_name=log_name, data_name=data_name)
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
