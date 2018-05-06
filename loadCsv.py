import numpy as np
import os.path
import sys

input_data = []
audio_list = []
split_list = []
label_list = []
data_name = []

def fav_list(fav_label=None):
    if not os.path.exists(fav_label):
        with open(fav_label, 'w') as f:
            f.write('')
            # if fav_lavel text file does not exist, make empty fav_label text file

    with open(fav_label) as fav_label:
        l_file = fav_label.readlines()
        fav = []
        for l in l_file:
            split_l = l.rstrip('\n').split(",")
            fav.append(split_l[-1])
    return fav

def show_input(data_name, label_list, data_shape):
    if data_name is None:
        return print('data_name is None\n')
    print('\n{:-^60}'.format('input'))
    print('No.    data_name    answer')
    for i in range(data_name.shape[0]):
        print('{:-^60}'.format(''))
        print('{0:02d}'.format(i+1), data_name[i], "   ", label_list[i])
    print('{:-^60}'.format(''))
    print('{:-^60}'.format(''))
    print('shape:',data_shape)
    print('{:-^60}\n'.format(''))

def form(file_path, data_format='power'):
    import matplotlib.pyplot as plt, librosa, librosa.display, urllib
    import sklearn

    x, fs = librosa.load(file_path, sr=None)
    if data_format=='power':
        stft = np.abs(librosa.stft(x, n_fft=1024, hop_length=512))**2
        data = stft
        #return : shape=(1 + n_fft/2, t)なので転置する
        #data = librosa.amplitude_to_db(stft, ref=np.max).T
        #data = librosa.power_to_db(stft).T # stftの対数をとる
    else:
        data = librosa.feature.mfcc(y=x, sr=fs ,n_mfcc=128).T
        #return : shape=(n_mfcc, t)なので転置する
        # data = sklearn.preprocessing.scale(data, axis=1)
        '''sklearn.preprocessing.scale(data, axis=1)について
        スケーリング
        取りうる値の大小が著しく異なる特徴量を入れると結果が悪くなることがある．
        このようなときは平均を引いて，標準偏差で割るとよくなることがある．場合によっては悪くなることもある．
        参照：http://sucrose.hatenablog.com/entry/2013/04/19/014258
        '''
    return data

def dim2(dir_path=None, fav_label=None):

    if dir_path == None:
        return print("dir_path is None\n")

    fav = fav_list(fav_label=fav_label)
    files = os.listdir(dir_path)# ファイル名のリストを取得
    count = 0

    for file_name in files:
        path, ext = os.path.splitext(file_name)
        if ext == ".csv":
            tmp_shape = np.loadtxt(dir_path+file_name, delimiter=',').shape[1]
            tmp_label = [[0 for i in range(2)] for j in range(tmp_shape)]
            if file_name in fav:
                tmp_label = [ [0,1] for j in range(tmp_shape)]
            else:
                tmp_label = [ [1,0] for j in range(tmp_shape)]
            if count == 0:
                label_list = tmp_label
                audio_list = np.loadtxt(dir_path+file_name, delimiter=',')
            else :
                label_list.extend(tmp_label)
                audio_list = np.append(audio_list, np.loadtxt(dir_path+file_name, delimiter=','), axis=1)
            count += 1
        else: pass

    label_list = np.array(label_list)
    return audio_list.T, label_list

def dim3(dir_path=None, FRAME_NUM=599, fav_label=None):

    #return inputdata shape is (number of csv files, FRAME_NUM, FREQ)
    #return inputdata shape is (number of csv files, frequency dimention)
    if dir_path == None:
        return print("dir_path is None\n")

    files = os.listdir(dir_path)
    count = 0
    for file_name in files:
        path, ext = os.path.splitext(file_name)
        if ext == ".csv":
            count += 1
    if count == 0:
        print('\nThere is no data\n')
        sys.exit(1)

    fav = fav_list(fav_label=fav_label)
    FRAME_NUM = int(688)
    # FREQ:frequency dimention
    FREQ = 120
    files = os.listdir(dir_path)# ファイル名のリストを取得
    count = 0

    for file_name in files:
        path, ext = os.path.splitext(file_name)
        if ext == '.csv':
            tmp_label = [[0 for i in range(1)] for j in range(10)]
            if file_name in fav:
                tmp_label = [1]
            else:
                tmp_label = [0]

            #audio_list = np.loadtxt(dir_path+file_name, delimiter=',').T
            #audio_list = [audio_list[0:FRAME_NUM,0:FREQ]]
            #audio_list[0][audio_list[0] < np.amax(audio_list[0])/2]=0

            audio_list = np.loadtxt(dir_path+file_name, delimiter=',')
            audio_list = [audio_list[0:FREQ,0:FRAME_NUM]]
            if count == 0:
                label_list = [tmp_label]
                input_data = np.array([audio_list])
                data_name = np.array([file_name])
            else :
                label_list.extend([tmp_label])
                input_data = np.vstack((input_data, [audio_list]))
                data_name = np.append(data_name, [file_name])
            count += 1
        else: pass

    label_list = np.array(label_list)
    data_name = data_name.reshape(data_name.shape[0],1)
    show_input(data_name=data_name, label_list=label_list, data_shape=input_data.shape)
    return input_data, label_list, data_name

def time_dim3_overlap(dir_path=None, FRAME_NUM=599, fav_label=None, OVERLAP=True):

    #return shape is (10, FRAME_NUM, 128)
    #return shape is (music number, music time, MFCC dimention)
    if dir_path == None:
        return print("dir_path is None\n")

    files = os.listdir(dir_path)
    count = 0
    for file_name in files:
        path, ext = os.path.splitext(file_name)
        if ext == ".csv":
            count += 1
    if count == 0:
        print('\nThere is no data\n')
        sys.exit(1)

    overlap=8 # 1/overlapでoverlapしていく

    fav = fav_list(fav_label=fav_label)
    FRAME_NUM = int(688*2)
    WINDOW = int(688)
    stride = int(WINDOW/overlap)
    print('overlap ',overlap)
    print('FRAME_NUM/stride', FRAME_NUM/stride)
    print('WINDOW/stride', WINDOW/stride)
    print('stride WINDOW/overlap', WINDOW/overlap)
    print('windoe/stride ',WINDOW/stride)
    TIME_STEP = int(FRAME_NUM/stride-WINDOW/stride+WINDOW/stride/overlap)
    print('TIME_STEP', TIME_STEP)
    FREQ = 120
    files = os.listdir(dir_path)# ファイル名のリストを取得
    count = 0

    for file_name in files:
        path, ext = os.path.splitext(file_name)
        if ext == '.csv':
            tmp_label = [[0 for i in range(1)] for j in range(10)]
            if file_name in fav:
                tmp_label = [1]
            else:
                tmp_label = [0]
            audio_list = np.loadtxt(dir_path+file_name, delimiter=',').T
            split_list = np.array([audio_list[0:WINDOW,0:FREQ]])
            #audio_list = np.loadtxt(dir_path+file_name, delimiter=',')
            #split_list = np.array([audio_list[0:FREQ,0:WINDOW]])
            for i in range(1,TIME_STEP):
                split_list = np.vstack((split_list, [audio_list[stride*i:stride*i+WINDOW,0:FREQ]]))
                #split_list = np.vstack((split_list, [audio_list[0:FREQ,stride*i:stride*i+WINDOW]]))
            if count == 0:
                label_list = [tmp_label]
                input_data = np.array([split_list])
                data_name = np.array([file_name])
            else :
                label_list.extend([tmp_label])
                input_data = np.vstack((input_data, [split_list]))
                data_name = np.append(data_name, [file_name])
            count += 1
        else: pass

    label_list = np.array(label_list)
    data_name = data_name.reshape(data_name.shape[0],1)

    show_input(data_name=data_name, label_list=label_list, data_shape=input_data.shape)

    return input_data, label_list, data_name

def for_prediction(input_shape, dir_path=None, fav_label=None, OVERLAP=False, data_format='mfcc'):

    #return shape is (number of csv files, time_step, FRAME_NUM, FREQ)
    #return shape is (music number, music time, MFCC dimention)
    if dir_path == None:
        return print("dir_path is None\n")

    files = os.listdir(dir_path)
    count = 0
    for file_name in files:
        path, ext = os.path.splitext(file_name)
        if ext == ".wav":
            count += 1
    if count == 0:
        print('\nThere is no data\n')
        sys.exit(1)

    fav = fav_list(fav_label=fav_label)
    TIME_STEP = input_shape[1]
    FRAME_NUM = input_shape[3]
    # 172frames = 1s
    FREQ = input_shape[4]
    print('inputshape', input_shape)

    overlap=1 #defaultではoverlapしない
    label_list = []
    if OVERLAP:
        overlap=4 # 1/overlapでoverlapしていく

    files = os.listdir(dir_path)# ファイル名のリストを取得
    count = 0
    for file_name in files:
        path, ext = os.path.splitext(file_name)
        if ext == '.wav':
            tmp_label = [[0 for i in range(1)] for j in range(10)]
            if file_name in fav:
                tmp_label = [1]
            else:
                tmp_label = [0]

            audio_list = form(file_path=dir_path+file_name, data_format=data_format)
            if audio_list.shape[0]<FRAME_NUM:
                for i in range(int(FRAME_NUM/audio_list.shape[0])):
                    audio_list = np.vstack((audio_list,audio_list))
            start=0
            sample = int(audio_list.shape[0]/(FRAME_NUM/overlap*TIME_STEP))
            print(audio_list.shape)
            for sample in range(sample):
                if start+int(FRAME_NUM/overlap*TIME_STEP)+FRAME_NUM < audio_list.shape[0]:
                    split_list = np.array([audio_list[start:start+FRAME_NUM,0:FREQ]])
                    for i in range(1,TIME_STEP):
                        split_list = np.vstack((split_list, [audio_list[start+int(FRAME_NUM/overlap*i):start+int(FRAME_NUM/overlap*i)+FRAME_NUM,0:FREQ]]))
                    if count == 0:
                        label_list = [tmp_label]
                        input_data = np.array([split_list])
                        data_name = np.array([file_name])
                    else :
                        label_list.extend([tmp_label])
                        input_data = np.vstack((input_data, [split_list]))
                        data_name = np.append(data_name, [file_name])
                start += int(FRAME_NUM/overlap*TIME_STEP)
                count += 1
        else: pass
    label_list = np.array(label_list)
    data_name = data_name.reshape(data_name.shape[0],1)
    show_input(data_name=data_name, label_list=label_list, data_shape=input_data.shape)
    return input_data, label_list, data_name

if __name__=='__main__':

    src_dir = './wavSrc/time_change/chord/instruments/time_change_instrument/'
    fav_label = src_dir+"fav.txt"

    train_music = []
    train_label = []
    test_music = []
    test_label = []
    fav = []
    count = 0

    input_shape = (None, 5, 1, 688, 513)
    #train_music, train_label, data_name = time_dim3_overlap(dir_path=src_dir, fav_label=fav_label, input_shape=input_shape, OVERLAP=True, data_format='mfcc')
    train_music, train_label, data_name = time_dim3_overlap(dir_path=src_dir, fav_label=fav_label, OVERLAP=True)
    print("train_music.shape : ", train_music.shape)
    print("train_label.shape", train_label.shape)
    """
    #print("\n shape", train_music.shape,"\n train_music ", train_music)

    print(train_music[0,0,:,:].shape)

    # to see each csv input data with time_dim3_easyarray
    for i in range(train_music.shape[0]):
        for j in range(train_music.shape[1]):
            path, ext = os.path.splitext(data_name[i][0])
            #np.savetxt("./wavSrc/tmp/"+path+"_"+str(j)+".csv", train_music[i][j], delimiter=',')
            #np.savetxt(src_dir+path+"_"+str(j)+".csv", train_music[i][j], delimiter=',')
    """
    """
    # to see each csv input data with other function
    for i in range(train_music.shape[0]):
        np.savetxt("./wavSrc/tmp/"+data_name[i][0], train_music[i], delimiter=',')
        print(data_name[i][0])
    """

    """
    for i in range(train_music.shape[0]):
        num =  len(np.where(train_music[i]>np.amax(train_music[i])/2)[0])
        print("\ni:num ",i, ":", num,
        "\n ave ", np.average(train_music),
        "\n max/2 ", np.amax(train_music)/2,
        "\n max ", np.amax(train_music),
        "\n min ", np.amin(train_music))

                        print(mfcc_list.shape)
                        num =  len(np.where(mfcc_list>np.amax(mfcc_list)/2)[0])
                        print("\nnum ", num)
    """
