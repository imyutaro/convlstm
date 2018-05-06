# http://aidiary.hatenablog.com/entry/20161120/1479640534

import numpy as np
from datetime import datetime
from keras.models import Model
from keras.layers import Convolution1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense, Input, concatenate, Reshape
from keras import regularizers
from keras.preprocessing import sequence
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping
from keras import backend as K
import loadCsv as load

#img_dir = './music/shortform/'
#img_dir = "./wavSrc/wav2/"
img_dir = "./wavSrc/wav6/FT/"
fav_label = img_dir+'fav.txt'

log_time = datetime.now()
log_time = log_time.strftime("%m%d%H%M")
logs_dirpath = "./output/my1dCNN/" + log_time

def build_cnn(input_shape, nb_filters=128):

    inputs = Input(batch_shape=input_shape)
    #inputs = Input(input_shape)
    print("input_shape : ", input_shape,
    "\ninputs : ", inputs)

    conv1d_1 = Convolution1D(filters=nb_filters, kernel_size=4, strides=1, padding ='valid')(inputs)
    conv1d_1_relu = Activation('relu')(conv1d_1)
    maxpool_1 = MaxPooling1D(pool_size = 4)(conv1d_1_relu)

    conv1d_2 = Convolution1D(filters=nb_filters*2, kernel_size=4, strides=1, padding ='valid')(maxpool_1)
    conv1d_2_relu = Activation('relu')(conv1d_2)
    maxpool_2 = MaxPooling1D(pool_size = 2)(conv1d_2_relu)

    conv1d_3 = Convolution1D(filters=nb_filters*2, kernel_size=4, strides=1, padding ='valid')(maxpool_2)
    conv1d_3_relu = Activation('relu')(conv1d_3)
    maxpool_3 = MaxPooling1D(pool_size = 2)(conv1d_3_relu)

    conv1d_4 = Convolution1D(filters=nb_filters*4, kernel_size=4, strides=1, padding ='valid')(maxpool_3)
    conv1d_4_relu = Activation('relu')(conv1d_4)

    global_ave_pool_1 = GlobalAveragePooling1D()(conv1d_4_relu)
    global_max_pool_1 = GlobalMaxPooling1D()(conv1d_4_relu)
    dense_1 = Convolution1D(filters=nb_filters*4, kernel_size=4, strides=1, kernel_regularizer = regularizers.l2(0.01), padding ='valid', activation="relu")(maxpool_3)
    #dense_1 = Dense(nb_filters*2, kernel_regularizer = regularizers.l2(0.01), activation="relu")(conv1d_4)
    #maxpool_4 = MaxPooling1D(pool_size = 2)(dense_1)
    dense_1 = Flatten()(dense_1)

    merged_vector = concatenate([global_ave_pool_1, global_max_pool_1, dense_1], axis=1)

    #model.add(Dropout(0.5))

    dense_2 = Dense(2048)(merged_vector)
    dense_2 = Dense(2048)(dense_2)
    dense_3 = Dense(2048)(dense_2)
    dense_3 = Dense(1024)(dense_3)
    dense_4 = Dense(1)(dense_3)
    dence_4_sigmoid = Activation('sigmoid')(dense_4)

    model = Model(inputs=[inputs], outputs=[dense_4_sigmoid])

    model.summary()

    # マルチクラスの出力を行う場合, cross_entropyは非推奨
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def build_cnn_test(input_shape, nb_filters=128):

    inputs = Input(batch_shape=input_shape)
    #inputs = Input(input_shape)
    print("input_shape : ", input_shape,
    "\ninputs : ", inputs)

    conv1d_1 = Convolution1D(filters=nb_filters, kernel_size=4, strides=1, padding ='valid', kernel_initializer='random_uniform')(inputs)
    conv1d_1_relu = Activation('relu')(conv1d_1)
    maxpool_1 = MaxPooling1D(pool_size = 4)(conv1d_1_relu)

    conv1d_2 = Convolution1D(filters=nb_filters, kernel_size=4, strides=1, padding ='valid', kernel_initializer='random_uniform')(maxpool_1)
    conv1d_2_relu = Activation('relu')(conv1d_2)
    maxpool_2 = MaxPooling1D(pool_size = 2)(conv1d_2_relu)

    conv1d_3 = Convolution1D(filters=nb_filters, kernel_size=4, strides=1, padding ='valid', kernel_initializer='random_uniform')(maxpool_2)
    conv1d_3_relu = Activation('relu')(conv1d_3)
    maxpool_3 = MaxPooling1D(pool_size = 2)(conv1d_3_relu)

    conv1d_4 = Convolution1D(filters=nb_filters, kernel_size=4, strides=1, padding ='valid', kernel_initializer='random_uniform')(maxpool_3)
    conv1d_4_relu = Activation('relu')(conv1d_4)
    """
    global_ave_pool_1 = GlobalAveragePooling1D()(conv1d_4)
    global_max_pool_1 = GlobalMaxPooling1D()(conv1d_4)
    dense_1 = Convolution1D(filters=nb_filters*4, kernel_size=4, strides=1, kernel_regularizer = regularizers.l2(0.01), padding ='valid', activation="relu")(maxpool_3)
    #dense_1 = Dense(nb_filters*2, kernel_regularizer = regularizers.l2(0.01), activation="relu")(conv1d_4)
    #maxpool_4 = MaxPooling1D(pool_size = 2)(dense_1)
    dense_1 = Flatten()(dense_1)

    merged_vector = concatenate([global_ave_pool_1, global_max_pool_1, dense_1], axis=1)
    """
    #model.add(Dropout(0.5))

    #dense_2 = Dense(2048)(merged_vector)
    maxpool_4 = MaxPooling1D(pool_size = 2)(conv1d_4_relu)
    dense_1 = Flatten()(maxpool_4)
    dense_2 = Dense(2048, kernel_initializer='random_uniform')(dense_1)
    dense_3 = Dense(2048, kernel_initializer='random_uniform')(dense_2)
    dense_3 = Dense(1024, kernel_initializer='random_uniform')(dense_3)
    dense_4 = Dense(1)(dense_3)
    dence_4_sigmoid = Activation('sigmoid')(dense_4)

    model = Model(inputs=[inputs], outputs=[dense_4_sigmoid])

    model.summary()

    # マルチクラスの出力を行う場合, cross_entropyは非推奨
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def main(input_shape, nb_filters):

    model = build_cnn_test(input_shape, nb_filters)

    tb_cb = TensorBoard(log_dir=logs_dirpath, histogram_freq=1)
    es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')

    history = model.fit(train_music, train_label, verbose=1, nb_epoch=100, batch_size=10, validation_data=(train_music, train_label), shuffle=True, callbacks=[tb_cb])

    score = model.evaluate(test_music, test_label, verbose=1)
    print('test loss:', score[0])
    print('test acc:', score[1])

    model_path = "./output/my1dCNN/weight"+log_time+"filter="+str(nb_filters)+".h5"
    model.save(model_path)  # creates a HDF5 file at model_path

    # save as JSON
    model_json=model.to_json()
    with open("./output/my1dCNN/model"+log_time+"filter="+str(nb_filters)+".json",mode='w') as f:
        f.write(model_json)

    K.clear_session()

if __name__=='__main__':

    train_music = []
    train_label = []
    test_music = []
    test_label = []

    #nb_filters = 128
    #nb_filters = 120

    #train_music, train_label, test_music, test_label, data_file = load.dim3_test(dir_path = img_dir, fav_label=fav_label)
    train_music, train_label, test_music, test_label, data_file = load.dim3_easyarray(dir_path = img_dir, fav_label=fav_label)
    input_shape = (None,train_music.shape[1],train_music.shape[2])

    print("train_music.shape : ", train_music.shape)
    print("train_label.shape", train_label.shape)


    for nb_filters in range(6):
        nb_filters += 1
        main(input_shape=input_shape, nb_filters=nb_filters)

    main(input_shape=input_shape, nb_filters=32)
    main(input_shape=input_shape, nb_filters=64)
    main(input_shape=input_shape, nb_filters=120)
