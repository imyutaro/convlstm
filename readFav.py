import os.path
import numpy as np

def fav_list(fav_label = None):
    with open(fav_label) as fav_label:
        l_file = fav_label.readlines()
        fav = []
        for l in l_file:
            split_l = l.rstrip('\n').split(",")
            fav.append(split_l[-1])
    return fav

def dim3_easyarray(dir_path = None, frame_num=599, fav_label = None):

    #return shape is (10, frame_num, 128)
    #return shape is (music number, music time, MFCC dimention)
    if dir_path == None:
        return print('dir_path is None\n')
    elif fav_label == None:
        return print('fav.txt is None\n')

    fav = fav_list(fav_label = fav_label)
    mfcc_list = []
    label_list = []
    data_file = []
    frame_num=690
    #1曲のsample数分1,0(好き嫌い)をつけないといけない
    # data数 : 10
    # sample数 : csvの時間長 588
    # class数 : 2
    # MFCCの次元 : 128
    # dataのshapeは(10, 599, 128)
    # labelのshapeは(10, 599, 2)
    files = os.listdir(dir_path)# ファイル名のリストを取得
    count = 0

    for file_name in files:
        path, ext = os.path.splitext(file_name)
        if ext == ".csv":
            tmp_shape = frame_num
            tmp_label = [[0 for i in range(1)] for j in range(10)]
            if file_name in fav:
                tmp_label = [1]
            else:
                tmp_label = [0]
            if count == 0:
                label_list = [tmp_label]
                mfcc_list = np.loadtxt(dir_path+file_name, delimiter=',').T
                mfcc_list = [mfcc_list[0:frame_num,0:120]]
                mfcc_list[0][mfcc_list[0] < np.amax(mfcc_list[0])/2]=0
                data_file = np.array([file_name])
            else :
                label_list.extend([tmp_label])
                tmp = np.loadtxt(dir_path+file_name, delimiter=',').T
                tmp = [tmp[0:frame_num,0:120]]
                tmp[0][tmp[0] < np.amax(tmp[0])/2]=0
                mfcc_list = np.vstack((mfcc_list, tmp))
                data_file = np.append(data_file, [file_name])
            count += 1
        else: pass

    label_list = np.array(label_list)
    data_file = data_file.reshape(data_file.shape[0],1)
    print(data_file)

    return mfcc_list, label_list, mfcc_list, label_list, data_file


#img_dir = './createWave/wav3/FT/'
#img_dir = './createWave/wav4/FT/'
#img_dir = './createWave/wav2/'
#img_dir = './music/shortform/'
img_dir='./wavSrc/instrument/same_instrument/'
fav_label = img_dir+'fav.txt'

train_music = []
train_label = []
test_music = []
test_label = []
#fav = []
"""
with open(fav_label, 'r', encoding='UTF-8') as fav_label:
    for line in fav_label:
        print(line.rstrip())
"""

    #diff = to_categorical(max_index)
#print("type fav", type(fav)) # list
"""
train_music, train_label, test_music, test_label, data_name = dim3_easyarray(dir_path=img_dir, fav_label=fav_label)

print('\n------------input-------------------------------------')
print('data_name    answer')
for i in range(data_name.shape[0]):
    print('------------------------------------------------------')
    print(data_name[i], "   ", train_label[i])
print('------------------------------------------------------\n')
"""

fav=fav_list(fav_label=fav_label)
print(fav)
