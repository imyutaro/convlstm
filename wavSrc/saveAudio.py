import wave

def save(data, fs, bit, filename):
    """波形データをWAVEファイルへ出力"""
    wf = wave.open(filename, "w")
    wf.setnchannels(1)
    wf.setsampwidth(bit//8)
    wf.setframerate(fs)
    wf.writeframes(data)
    wf.close()

def play(data, fs, bit):
    import pyaudio
    # ストリームを開く
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=int(fs),
                    output= True)
    # チャンク単位でストリームに出力し音声を再生
    chunk = 1024
    sp = 0  # 再生位置ポインタ
    buffer = data[sp:sp+chunk]
    while buffer != b'':
        stream.write(buffer)
        sp = sp + chunk
        buffer = data[sp:sp+chunk]
    stream.close()
    p.terminate()

if __name__ == "__main__" :
    from createSinWave import createSinWave
    from createCombineWave import createCombinedWave
    allData = b""
    freqList = [262, 294, 330, 349, 392, 440, 494, 523]  # ドレミファソラシド
    for f in freqList:
        data = createSinWave(1.0, f, 8000.0, 1.0)
        play(data, 8000, 16)
        allData += data
    save(allData, 8000, 16, "sin.wav")
