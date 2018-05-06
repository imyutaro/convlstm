
import numpy as np
from datetime import datetime
from keras.models import Model
from keras.layers import Convolution1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense, Input, concatenate, Reshape
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras import regularizers
from keras.preprocessing import sequence
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping
from keras import backend as K
import loadCsv as load
import simple_input

def myLSTM(input_shape, nb_filters=128):
    inputs = Input(batch_shape=input_shape)

    conv1d_1 = TimeDistributed(Convolution1D(filters=nb_filters, kernel_size=4, strides=1, padding ='valid', activation="relu"))(inputs)
    maxpool_1 = TimeDistributed(MaxPooling1D(pool_size = 4))(conv1d_1)

    conv1d_2 = TimeDistributed(Convolution1D(filters=nb_filters*2, kernel_size=4, strides=1, padding ='valid', activation="relu"))(maxpool_1)
    maxpool_2 = TimeDistributed(MaxPooling1D(pool_size = 2))(conv1d_2)

    lstm_1 = LSTM(128, return_sequences=True)(maxpool_2)
    dence_1 = Dense(128, activation="relu")(lstm_1)
    dence_2 = Dense(2, activation="softmax")(dence_1)

    model = Model(inputs=[inputs], outputs=[dence_2])

    model.summary()

    # マルチクラスの出力を行う場合, cross_entropyは非推奨
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


if __name__=='__main__':

    img_dir = './music/shortform/'
    data_file = 'input_label.txt'

    train_music = []
    train_label = []
    test_music = []
    test_label = []

    input_shape = (None,599,128)
    #input_shape = (None,250,1)
    nb_filters = 128

    train_music, train_label, test_music, test_label = load.dim3_exp(dir_name = img_dir)

    #train_music, train_label, test_music, test_label = simple_input.sin_input()
    #train_music = sequence.pad_sequences(train_music, dtype='float32')
    #test_music = sequence.pad_sequences(test_music, dtype='float32')
    #train_music = np.expand_dims(train_music, axis=2)
    #test_music = np.expand_dims(test_music, axis=2)

    print("train_music.shape : ", train_music.shape)
    print("train_label.shape", train_label.shape)

    model = myLSTM(input_shape, nb_filters)
    #model = myLSTM(input_shape, nb_filters=3)

    log_time = datetime.now()
    logs_dirpath = "./tmp/logs/" + log_time.strftime("%m%d%H%M")

    tb_cb = TensorBoard(log_dir=logs_dirpath, histogram_freq=1, write_images=True)
    es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')

    history = model.fit(train_music, train_label, verbose=1, nb_epoch=100, batch_size=32, validation_data=(train_music, train_label), shuffle=False, callbacks=[es_cb, tb_cb])

    score = model.evaluate(test_music, test_label, verbose=1)
    print('test loss:', score[0])
    print('test acc:', score[1])

    model.save('./tmp/my_model.h5')  # creates a HDF5 file 'my_model.h5'

    # save as JSON
    model_json=model.to_json()
    with open("model.json",mode='w') as f:
        f.write(model_json)

    K.clear_session()
