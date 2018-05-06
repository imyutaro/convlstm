# https://github.com/jkatsuta/viz_nn/blob/master/viz_cnn.ipynb
"""not yo show on display on GPU server
import matplotlib
matplotlib.use('Agg')
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import model_from_json
import loadCsv as load
from keras.utils import plot_model

plt.rcParams['font.size'] = 18
log_name = '12211238filter=3'
img_dir = "./wavSrc/wav7/FT/"
#img_dir = "./wavSrc/music/shortform/"


def viz_layer(l_num=None, model=None, input_data=None):

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
    max = np.amax(features)
    min = np.amin(features)
    print("max : ", max)
    print("min : ", min)

    print(l_num, "層目, layer name : ", model.layers[l_num])
    print("features.shape : ",features.shape)
    #plt.figure(figsize=(20, 10))
    #fig = plt.figure(figsize=(20, 70))
    plt.figure(figsize=(20, 70))
    for i, img in enumerate(features):
        #plt.subplot(2, 5, i + 1) # for music
        #plt.subplot(1, 4, i + 1) # for chord
        #plt.subplot(4, 2, i + 1) # for simple chord
        plt.subplot(input_data.shape[0], 1, i + 1)
        #plt.axis('off')
         #img = np.reshape(img, (img.shape[1],img.shape[0]))
        #librosa.display.specshow(img, x_axis='time', y_axis="mel")
        #librosa.display.specshow(img, x_axis='linear', y_axis="time")
        plt.pcolor(img, vmin=min, vmax=max, cmap="magma")
        plt.xticks( np.arange(0, img.shape[1], 10) )
        plt.colorbar()
    plt.show()
    #fig.savefig("./layeroutput/layer"+str(l_num)+".png")

# モデルを読み込む
model = None
"""
with open('./output/my1dCNN_LSTM_exp/model11301027.json') as f:
    model = model_from_json(f.read())

with open('./output/my1dCNN_exp/model11301046.json') as f:
    model = model_from_json(f.read())
"""

# モデルを読み込む
model = None
with open('./output/my1dCNN/model'+log_name+'.json') as f:
    model = model_from_json(f.read())

plot_model(model, to_file='model.png', show_shapes=True)

# 学習結果を読み込む
model.load_weights('./output/my1dCNN/weight'+log_name+'.h5')

from keras.utils import plot_model

plot_model(model, to_file='./model.png')


#train_music, train_label, test_music, test_label, data_file = load.dim3_test(dir_path = img_dir)
train_music, train_label, test_music, test_label, data_file = load.dim3_easyarray(dir_path = img_dir)
num =  len(np.where(train_music[0]>1)[0])
print("\n", num, "\n ", train_music.shape)

"""
img_dir = './wavSrc/wav/AmFCG.csv'
#data_file = img_dir+'input_label.txt'
train_music = np.loadtxt(img_dir, delimiter=',').T
train_music = np.array([train_music[0:690,:]])
train_label = np.array([[0,1]])
"""

model.summary()

#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

score = model.evaluate(train_music, train_label, verbose=0)
print('Test loss :', score[0])
print('Test accuracy :', score[1])

features = model.predict(train_music)
print("features.shape : ", features.shape)
max = np.amax(features)
min = np.amin(features)
print("max : ", max)
print("min : ", min)

print('\n----------predict-------------------------------------')
print('data_name   output   answer')
for i in range(features.shape[0]):
    print('------------------------------------------------------')
    print(data_file[i], features[i], train_label[i])
print('------------------------------------------------------\n')

import matplotlib.pyplot as librosa, librosa.display, urllib

# Kerasでは、以下のように簡単にレイヤーごとのパラメータが取得できる
lays = model.layers # list of the layers

for i, l in enumerate(lays):
    print(i, l)

#plt.figure(figsize=(20, 10))
#fig = plt.figure(figsize=(20, 70))
plt.figure(figsize=(20, 70))
for i, img in enumerate(train_music):
    #plt.subplot(2, 5, i + 1) # for music
    #plt.subplot(1, 4, i + 1) # for chord
    #plt.subplot(4, 2, i + 1) # for simple chord
    plt.subplot(train_music.shape[0], 1, i + 1)
     #img = np.reshape(img, (img.shape[1],img.shape[0]))
    #librosa.display.specshow(img, x_axis='time', y_axis="mel")
    #librosa.display.specshow(img, x_axis='linear', y_axis="time")
    #plt.axis('off')
    plt.pcolor(img, vmin=min, vmax=max, cmap="magma")
    plt.colorbar()
plt.show()
#fig.savefig("./layeroutput/input_layer.png")

for i,_ in enumerate(lays):
    if i+1<=8:
        #7層目まで出力
        viz_layer(l_num=i+1, model=model, input_data=train_music)

"""
plt.figure(figsize=(20, 10))
for ind, val in enumerate(w1):
    plt.subplot(2, 8, ind + 1)
    im = val.reshape((128, 128))
    plt.axis("off")
    #plt.imshow(im, cmap='coolwarm',interpolation='nearest')
    librosa.display.specshow(im)
"""
