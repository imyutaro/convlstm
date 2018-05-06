# 参考URL:http://chachay.hatenablog.com/entry/2016/10/03/215841

from pydub import AudioSegment
import os

def toShortwav(filepath = None, start = 0, end = 10*1000):
    """
    pydub does things in miliseconds, so 10 second is 10 * 1000 miliseconds.
    default cut filepath file between 0s(start) ~ 10s(end)
    """
    if filepath==None:
        return filepath
    else:
        sound = AudioSegment.from_file(filepath, format="wav")
        shortform = sound[start:end] #cut sound from start ~ to end
        # cut後の秒の表示
        print(len(shortform))
        return shortform

if __name__=="__main__":

    file_path = "./music/完全.wav"
    no_ext = os.path.splitext(file_name)[0]
    shortform = toShortwav(file_path)
    if shortform==None:
        print("filepath == None\n")
    else:
        added.export(no_ext+"_tmp.wav", format="wav")
