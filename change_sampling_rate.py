# https://qiita.com/Dsuke-K/items/2ad4945a81644db1e9ff

import soundfile as sf
import wave

def change_sampling_rate(file_path = None):


    data, samplerate = sf.read(file_path)
    sf.write(file_path, data, 44100)

    print(data.shape)

    wf = wave.open(file_path, "r")
    print("サンプリング周波数:", wf.getframerate())

    # stereo音源なら
    # l_channel = data[:,0]
    # r_channel = data[:,1]

if __name__ == "__main__":
    file_path = "./music/squall48khz.wav"
    change_sampling_rate(file_path)
