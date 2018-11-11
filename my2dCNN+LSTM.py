# http://aidiary.hatenablog.com/entry/20161120/1479640534

import glob
import numpy as np
import os.path
import sys
from datetime import datetime
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Concatenate, Reshape, ActivityRegularization, concatenate
from keras.layers.recurrent import LSTM
from keras.layers import TimeDistributed
from keras import regularizers
from keras.preprocessing import sequence
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping
from keras import backend as K
#import loadCsv as load
import loadWav as load

#src_path='./wavSrc/time_change/instrument/'
#src_path = './wavSrc/instrument/same_instrument/'
src_path='./wavSrc/single+chord/2349.318+/'

#test_src_path = src_path+'test_data/01/'
test_src_path = src_path
#test_src_path = src_path+'test_data/'
fav_label = src_path+'fav.txt'
test_fav_label = test_src_path+'fav.txt'
log_time = datetime.now()
log_time = log_time.strftime('%m%d%H%M')
logs_dirpath = './output/my2dCNN+LSTM/' + log_time

print('\n\nlog_name : ', log_time,
'\nsrc_dir : ', src_path,
'\ntest_src_dir : ', test_src_path,
'\nlogs_dirpath : ', logs_dirpath)

batch_size = 5
PADDING = 'same'
CHANNEL_ORDER = 'channels_first'
KERNELSIZE=(1,50)
POOLSIZE=(1,2)

def recall_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def build_cnn(input_shape):

    inputs = Input(batch_shape=input_shape)
    nb_filters = 128

    # kernel_size=(縦,横)=(列,行)=(row,col)
    conv2d_1 = Convolution2D(filters=nb_filters, kernel_size=(4,1), strides=(1, 1), padding=PADDING, data_format=CHANNEL_ORDER)(inputs)
    relu_1 = Activation('relu')(conv2d_1)
    maxpool_1 = MaxPooling2D(pool_size=(4,1), data_format=CHANNEL_ORDER)(relu_1)
    print('\nconv2d_1.shape : ',conv2d_1.shape)
    print('maxpool_1.shape',maxpool_1.shape,'\n')

    conv2d_2 = Convolution2D(filters=nb_filters*2, kernel_size=(4,1), strides=(1, 1), padding=PADDING, data_format=CHANNEL_ORDER)(maxpool_1)
    relu_2 = Activation('relu')(conv2d_2)
    maxpool_2 = MaxPooling2D(pool_size=(2,1), data_format=CHANNEL_ORDER)(relu_2)
    print('\nconv2d_2.shape : ',conv2d_2.shape)
    print('maxpool_2.shape',maxpool_2.shape,'\n')

    conv2d_3 = Convolution2D(filters=nb_filters*2, kernel_size=(4,1), strides=(1, 1), padding=PADDING, data_format=CHANNEL_ORDER)(maxpool_2)
    relu_3 = Activation('relu')(conv2d_3)
    maxpool_3 = MaxPooling2D(pool_size=(2,1), data_format=CHANNEL_ORDER)(relu_3)
    print('\nconv2d_3.shape : ',conv2d_3.shape)
    print('maxpool_3.shape',maxpool_3.shape,'\n')

    conv2d_4 = Convolution2D(filters=nb_filters*4, kernel_size=(4,1), strides=(1, 1), padding=PADDING, data_format=CHANNEL_ORDER)(maxpool_3)
    relu_4 = Activation('relu')(conv2d_4)
    print('\nconv2d_4.shape : ',conv2d_4.shape)

    global_ave_pool_1 = GlobalAveragePooling2D()(relu_4)
    global_max_pool_1 = GlobalMaxPooling2D()(relu_4)
    '''
    conv2d_4 = MaxPooling2D(pool_size=(32,1))(conv2d_4)
    print('\nconv2d_4.shape : ',conv2d_4.shape)
    regL2 = Reshape((11,))(conv2d_4)
    print('\nregL2.shape : ',regL2.shape, '\n')
    regL2 = ActivityRegularization(l2=0.01)(conv2d_4)
    #regL2 = Dense(512, W_regularizer = regularizers.l2(0.01), activation='relu')(conv2d_4)

    print(
    '\n--shape--',
    '\nglobal_ave_pool_1 : ',global_ave_pool_1.shape,
    '\nglobal_max_pool_1 : ',global_max_pool_1.shape,
    '\nregL2 : ',regL2.shape,
    '\n---------\n'
    )

    merged_vector = Concatenate([global_ave_pool_1, global_max_pool_1, regL2], axis=1)
    '''
    merged_vector = concatenate([global_ave_pool_1, global_max_pool_1])
    #dense_2 = Dense(1532, activation='relu')(merged_vector)
    dense_2 = Dense(1532)(merged_vector)
    relu_5 = Activation('relu')(dence_2_4)

    #model.add(Dropout(0.5))

    dense_3 = Dense(2048)(dense_2)
    dense_4 = Dense(2048)(dense_3)
    dense_5 = Dense(1)(dense_4)
    softmax_1 = Activation('sigmoid')(dence_5)

    model = Model(inputs=[inputs], outputs=[softmax_1])

    model.summary()

    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def build_cnn_test(input_shape, nb_filters=None):

    if nb_filters == None:
        return print('nb_filters is None')

    inputs = Input(batch_shape=input_shape)

    # kernel_size=(縦,横)=(列,行)=(row,col)
    conv2d_1 = TimeDistributed(Convolution2D(filters=nb_filters, kernel_size=KERNELSIZE, strides=(1, 1), padding=PADDING, data_format=CHANNEL_ORDER), name='conv_1')(inputs)
    conv2d_1_relu = TimeDistributed(Activation('relu'), name='activation_1')(conv2d_1)
    maxpool_1 = TimeDistributed(MaxPooling2D(pool_size=POOLSIZE, data_format=CHANNEL_ORDER), name='maxpool_1')(conv2d_1_relu)

    print('\nconv2d_1.shape : ',conv2d_1.shape)
    print('maxpool_1.shape',maxpool_1.shape,'\n')

    conv2d_2 = TimeDistributed(Convolution2D(filters=nb_filters, kernel_size=KERNELSIZE, strides=(1, 1), padding=PADDING, data_format=CHANNEL_ORDER), name='conv_2')(maxpool_1)
    conv2d_2_relu = TimeDistributed(Activation('relu'), name='activation_2')(conv2d_2)
    maxpool_2 = TimeDistributed(MaxPooling2D(pool_size=POOLSIZE, data_format=CHANNEL_ORDER), name='maxpool_2')(conv2d_2_relu)

    print('\nconv2d_2.shape : ',conv2d_2.shape)
    print('maxpool_2.shape',maxpool_2.shape,'\n')

    conv2d_3 = TimeDistributed(Convolution2D(filters=nb_filters, kernel_size=KERNELSIZE, strides=(1, 1), padding=PADDING, data_format=CHANNEL_ORDER), name='conv_3')(maxpool_2)
    conv2d_3_relu = TimeDistributed(Activation('relu'), name='activation_3')(conv2d_3)
    maxpool_3 = TimeDistributed(MaxPooling2D(pool_size=POOLSIZE, data_format=CHANNEL_ORDER), name='maxpool_3')(conv2d_3_relu)

    print('\nconv2d_3.shape : ',conv2d_3.shape)
    print('maxpool_3.shape',maxpool_3.shape,'\n')

    conv2d_4 = TimeDistributed(Convolution2D(filters=nb_filters, kernel_size=KERNELSIZE, strides=(1, 1), padding=PADDING, data_format=CHANNEL_ORDER), name='conv_4')(maxpool_3)
    conv2d_4_relu = TimeDistributed(Activation('relu'), name='activation_4')(conv2d_4)
    maxpool_4 = TimeDistributed(MaxPooling2D(pool_size=POOLSIZE, data_format=CHANNEL_ORDER), name='maxpool_4')(conv2d_4_relu)

    print('\nconv2d_4.shape : ',conv2d_4.shape)
    print('maxpool_4.shape',maxpool_4.shape,'\n')

    dense_1 = TimeDistributed(Flatten(), name='flatten')(maxpool_4)

#    dense_1 = Dense(2048)(dense_1)
#    dense_1 = Dense(1024)(dense_1)
    dense_1 = TimeDistributed(Dense(512), name='dense_1')(dense_1)
    dense_1_relu = TimeDistributed(Activation('relu'), name='activation_5')(dense_1)
    lstm_1 = LSTM(128, return_sequences=False, name='lstm_1')(dense_1_relu)
    dense_2 = Dense(1, name='predictions')(lstm_1)
    dense_2_sigmoid = Activation('sigmoid', name='predict_activation')(dense_2)

    model = Model(inputs=[inputs], outputs=[dense_2_sigmoid])

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', recall_score])

    return model


def build_cnn_test2(input_shape, nb_filters=None):

    if nb_filters is None:
        return print('nb_filters is None')

    inputs = Input(batch_shape=input_shape)

    # kernel_size=(縦,横)=(列,行)=(row,col)
    conv2d_1 = TimeDistributed(Convolution2D(filters=nb_filters, kernel_size=(50,1), strides=(1, 1), padding=PADDING, data_format=CHANNEL_ORDER), name='conv_1')(inputs)
    conv2d_1_relu = TimeDistributed(Activation('relu'), name='activation_1')(conv2d_1)
    maxpool_1 = TimeDistributed(MaxPooling2D(pool_size=(4,1), data_format=CHANNEL_ORDER), name='maxpool_1')(conv2d_1_relu)

    print('\nconv2d_1.shape : ',conv2d_1.shape)
    print('maxpool_1.shape',maxpool_1.shape,'\n')
    """
    conv2d_2 = TimeDistributed(Convolution2D(filters=nb_filters, kernel_size=(50,1), strides=(1, 1), padding=PADDING, data_format=CHANNEL_ORDER), name='conv_2')(maxpool_1)
    conv2d_2_relu = TimeDistributed(Activation('relu'), name='activation_2')(conv2d_2)
    maxpool_2 = TimeDistributed(MaxPooling2D(pool_size=(2,1), data_format=CHANNEL_ORDER), name='maxpool_2')(conv2d_2_relu)

    print('\nconv2d_2.shape : ',conv2d_2.shape)
    print('maxpool_2.shape',maxpool_2.shape,'\n')

    conv2d_3 = TimeDistributed(Convolution2D(filters=nb_filters, kernel_size=(50,1), strides=(1, 1), padding=PADDING, data_format=CHANNEL_ORDER), name='conv_3')(maxpool_2)
    conv2d_3_relu = TimeDistributed(Activation('relu'), name='activation_3')(conv2d_3)
    maxpool_3 = TimeDistributed(MaxPooling2D(pool_size=(2,1), data_format=CHANNEL_ORDER), name='maxpool_3')(conv2d_3_relu)

    print('\nconv2d_3.shape : ',conv2d_3.shape)
    print('maxpool_3.shape',maxpool_3.shape,'\n')

    conv2d_4 = TimeDistributed(Convolution2D(filters=nb_filters, kernel_size=(50,1), strides=(1, 1), padding=PADDING, data_format=CHANNEL_ORDER), name='conv_4')(maxpool_3)
    conv2d_4_relu = TimeDistributed(Activation('relu'), name='activation_4')(conv2d_4)
    maxpool_4 = TimeDistributed(MaxPooling2D(pool_size=(2,1), data_format=CHANNEL_ORDER), name='maxpool_4')(conv2d_4_relu)

    print('\nconv2d_4.shape : ',conv2d_4.shape)
    print('maxpool_4.shape',maxpool_4.shape,'\n')
    """
    dense_1 = TimeDistributed(Flatten(), name='flatten')(maxpool_1)

#    dense_1 = Dense(2048)(dense_1)
#    dense_1 = Dense(1024)(dense_1)
    dense_1 = TimeDistributed(Dense(512), name='dense_1')(dense_1)
    dense_1_relu = TimeDistributed(Activation('relu'), name='activation_5')(dense_1)
    lstm_1 = LSTM(256, return_sequences=False, name='lstm_1')(dense_1_relu)
    dense_2 = Dense(1, name='predictions')(lstm_1)
    dense_2_sigmoid = Activation('sigmoid', name='predict_activation')(dense_2)

    model = Model(inputs=[inputs], outputs=[dense_2_sigmoid])

    model.summary()

    # マルチクラスの出力を行う場合, cross_entropyは非推奨
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])

    return model

def main(nb_filters):

    model = build_cnn_test(input_shape, nb_filters)

    tb_cb = TensorBoard(log_dir=logs_dirpath, histogram_freq=1)
    es_cb = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

    history = model.fit(train_music, train_label, verbose=1, nb_epoch=100, batch_size=batch_size, validation_data=(test_music, test_label), shuffle=True, callbacks=[tb_cb])

    score = model.evaluate(test_music, test_label, verbose=1)
    print('test loss:', score[0])
    print('test acc:', score[1])
    print('test mse:', score[2])
    print('test recall:', score[3])

    model_path = './output/my2dCNN+LSTM/weight'+log_time+'filter='+str(nb_filters)+'.h5'
    model.save(model_path)  # creates a HDF5 file at model_path

    # save as JSON
    model_json=model.to_json()
    with open('./output/my2dCNN+LSTM/model'+log_time+'filter='+str(nb_filters)+'.json',mode='w') as f:
        f.write(model_json)

    K.clear_session()

if __name__=='__main__':

    train_music = []
    train_label = []
    test_music = []
    test_label = []

    print('\n{:-^60}'.format('train'))
    train_music, train_label, data_file = load.time_dim3_overlap(dir_path=src_path, fav_label=fav_label)
    print('\n{:-^60}'.format('test'))
    test_music, test_label, _ = load.time_dim3_overlap(dir_path=test_src_path, fav_label=test_fav_label)
    #train_music = sequence.pad_sequences(train_music, dtype='float32')
    #test_music = sequence.pad_sequences(test_music, dtype='float32')
    train_music = np.expand_dims(train_music, axis=2)
    test_music = np.expand_dims(test_music, axis=2)
    #input_shape = (None, train_music.shape[1], 1, train_music.shape[2], train_music.shape[3])
    input_shape = (None, train_music.shape[1], 1, train_music.shape[3], train_music.shape[4])
    # input_shape = (batch_size, time_step, channel, frame number, freq number)
    print('input_shape : ', input_shape)
    print('train_music.shape : ', train_music.shape)

    for nb_filters in range(1,7):
        main(nb_filters=nb_filters)

    #main(8)
    #main(16)
    #main(32)
    #main(64)
    #main(128)
