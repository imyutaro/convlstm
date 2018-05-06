# 参考URL:http://chachay.hatenablog.com/entry/2016/10/03/215841

from pydub import AudioSegment
import os
import os.path
import glob
import sys
import numpy as np
np.set_printoptions(threshold=np.inf) # numpyをprintした際の省略を無くす

def time_toShortCsv(dir_path=None, file_list=None):

    if not os.path.isdir(dir_path+'time_shortform/'):
        os.mkdir(dir_path+'time_shortform/') # make dir for shortform csv

    frame_num = 688
    split_time = int(frame_num/4)
    time_step = 10
    freq = 400
    for file_name in file_list:
        audio_list = np.loadtxt(dir_path+file_name, delimiter=',').T
        num = int(audio_list.shape[0]/frame_num)
        print(audio_list.shape)
        for sample in range(num):
            start = frame_num*sample
            split_list = np.array([audio_list[start+split_time*0:start+split_time*4,0:freq]])
            for i in range(1,time_step):
                if start+split_time*i+split_time*4 < audio_list.shape[0]:
                    split_list = np.vstack((split_list, [audio_list[start+split_time*i:start+split_time*i+split_time*4,0:freq]]))
            if split_list.shape==(time_step, frame_num, freq):
                no_ext = os.path.splitext(file_name)[0]
                np.save(dir_path+no_ext+'_{0:d}'.format(sample)+'.npy', split_list)
            print(split_list.shape)

def toShortCsv(dir_path=None, file_list=None):

    frame_num = 688*2
    #split_time = int(frame_num/4)
    time_step = 10
    freq = 400
    csvlen = 0
    #maxlen = 4

    if not os.path.isdir(dir_path+'shortform/'):
        os.mkdir(dir_path+'shortform/') # make dir for shortform csv

    for file_name in file_list:
        audio_list = np.loadtxt(dir_path+file_name, delimiter=',').T
        num = int(audio_list.shape[0]/frame_num)
        num = 4
        print(audio_list.shape)
        for sample in range(num):
            #audio_list[audio_list < np.amax(audio_list)/2]=0
            #audio_list[audio_list < np.amax(audio_list)/4]=0
            start = frame_num*sample
            if start+frame_num < audio_list.shape[0]:
                split_list = np.array(audio_list[start:start+frame_num,0:freq])
                no_ext = os.path.splitext(file_name)[0]
                np.savetxt(dir_path+'shortform/'+no_ext+'_{0:d}'.format(sample)+'.csv', split_list.T, delimiter=',')
            print('split_list.shape', split_list.shape, 'start+frame_num', start+frame_num)


def tochannel_ShortCsv(dir_path=None, file_list=None):

    frame_num = 688*2
    #split_time = int(frame_num/4)
    time_step = 10
    freq = 400
    csvlen = 0
    #maxlen = 4

    if not os.path.isdir(dir_path+'channel/'):
        os.mkdir(dir_path+'channel/') # make dir for shortform csv

    for file_name in file_list:
        audio_list = np.loadtxt(dir_path+file_name, delimiter=',').T
        num = int(audio_list.shape[0]/frame_num)
        num = 4
        print(audio_list.shape)
        for sample in range(num):
            #audio_list[audio_list < np.amax(audio_list)/2]=0
            #audio_list[audio_list < np.amax(audio_list)/4]=0
            start = frame_num*sample
            if start+frame_num < audio_list.shape[0]:
                split_list = np.array(audio_list[start:start+frame_num,0:freq])
                no_ext = os.path.splitext(file_name)[0]
                np.savetxt(dir_path+'channel/'+no_ext+'_{0:d}'.format(sample)+'.csv', split_list.T, delimiter=',')
            print('split_list.shape', split_list.shape, 'start+frame_num', start+frame_num)

if __name__=='__main__':

    dir_path = './wavSrc/RWC/MIDI/wav/RM-P001/'

    count = 0
    file_list = []
    file_list = os.listdir(dir_path)

    tmp = []
    for file_name in file_list:
        _, ext = os.path.splitext(file_name)
        if ext=='.csv':
            tmp.append(file_name)
            count +=1
    if count == 0:
        print('There is no data')
        sys.exit(1)
    file_list = tmp

    files = os.listdir(dir_path)
    toShortCsv(dir_path=dir_path, file_list=file_list)
