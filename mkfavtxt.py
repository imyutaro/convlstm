from pydub import AudioSegment
import os
import os.path
import glob
import sys

dir_path = '/Users/yutaro/lab/B4_research/workspace/wavSrc/RWC/MIDI/wav/RM-P001/shortform/tmp/'

count = 0
file_list = []
file_list = os.listdir(dir_path)

tmp = []
for file_name in file_list:
    _, ext = os.path.splitext(file_name)
    if ext=='.csv':
        tmp.append(file_name)
        count +=1

with open(dir_path+'fav.txt', mode = 'w', encoding = 'utf-8') as fh:
    for tmp in tmp:
        fh.write(tmp+'\n')
