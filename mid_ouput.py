#http://d.hatena.ne.jp/natsutan/20170219/1487515186
import matplotlib
matplotlib.use('Agg')

import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras import backend as K
import loadWav as load
import matplotlib.pyplot as librosa, librosa.display, urllib
plt.rcParams['font.size'] = 18

src_dir='./wavSrc/instrument/test_data/'
log_name='01291308filter=6'

fav_label = src_dir+"fav.txt"

output_dir = './output/my2dCNN+LSTM/'
print('\n\nlog_name : ', log_name,
'\nsrc_dir : ', src_dir,
'\noutput_dir : ', output_dir)

# モデルを読み込む
model = None
with open(output_dir+'model'+log_name+'.json') as f:
    model = model_from_json(f.read())

# 学習結果を読み込む
model.load_weights(output_dir+'weight'+log_name+'.h5')
model.summary()
"""
# load image
images = np.empty([0, 28, 28], np.float32)
img_ori = Image.open('data/I.png')
img_gray = ImageOps.grayscale(img_ori)

img_ary = np.asarray(img_gray)
img_ary = 255 - img_ary
images = np.append(images, [img_ary], axis=0)

images = images.reshape(1, 28, 28, 1)
"""
train_music, train_label, data_file = load.time_dim3_overlap(dir_path = src_dir, fav_label=fav_label)
train_music = np.expand_dims(train_music, axis=2)
print(train_music.shape)
"""
# predict
ret = model.predict(images, 1, 1)
print(ret)
"""
# output
l_num=7
train = train_music[1]
train = np.expand_dims(train, axis=0)
print(train.shape)
get_layer_output = K.function([model.layers[0].input],
                                  [model.layers[l_num].output])
layer_output = get_layer_output([train,])
print(layer_output[0].shape)
#np.save('output/convolution2d_out.npy', layer_output[0], allow_pickle=False)

fig = plt.figure(figsize=(20,5*train.shape[1]))
for img in layer_output[0]:
    for i, img in enumerate(img):
        print(img.shape)
        plt.subplot(train.shape[1], 1, i + 1)
        librosa.display.specshow(img[0], y_axis='log', x_axis='time', cmap='gray_r')

fig.savefig(output_dir+'layeroutput/'+log_name+'_output'+str(l_num)+'.png', bbox_inches='tight')
