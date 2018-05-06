# ref>http://aidiary.hatenablog.com/entry/20120225/1330179868
import numpy as np
import matplotlib.pyplot as plt

def hz2mel(f):
    """Hzをmelに変換"""
    return 1127.01048 * np.log(f / 700.0 + 1.0)

def mel2hz(m):
    """melをhzに変換"""
    return 700.0 * (np.exp(m / 1127.01048) - 1.0)

def melFilterBank(fs, nfft, numChannels):
    """メルフィルタバンクを作成"""
    # ナイキスト周波数（Hz）
    fmax = fs / 2
    # ナイキスト周波数（mel）
    melmax = hz2mel(fmax)
    # 周波数インデックスの最大数
    nmax = nfft / 2
    # 周波数解像度（周波数インデックス1あたりのHz幅）
    df = fs / nfft
    # メル尺度における各フィルタの中心周波数を求める
    dmel = melmax / (numChannels + 1)
    melcenters = np.arange(1, numChannels + 1) * dmel
    # 各フィルタの中心周波数をHzに変換
    fcenters = mel2hz(melcenters)
    # 各フィルタの中心周波数を周波数インデックスに変換
    indexcenter = np.round(fcenters / df)
    # 各フィルタの開始位置のインデックス
    indexstart = np.hstack(([0], indexcenter[0:numChannels - 1]))
    # 各フィルタの終了位置のインデックス
    indexstop = np.hstack((indexcenter[1:numChannels], [nmax]))

    filterbank = np.zeros((numChannels, int(nmax)))
    for c in np.arange(0, numChannels):
        # 三角フィルタの左の直線の傾きから点を求める
        increment= 1.0 / (indexcenter[c] - indexstart[c])
        for i in np.arange(indexstart[c], indexcenter[c]):
            i=int(i)
            filterbank[c, i] = (i - indexstart[c]) * increment
        # 三角フィルタの右の直線の傾きから点を求める
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in np.arange(indexcenter[c], indexstop[c]):
            i=int(i)
            filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)

    return filterbank, fcenters

# メルフィルタバンクを作成
nfft=1024
fs=16000
#fs=41100
numChannels = 20  # メルフィルタバンクのチャネル数
df = fs / nfft   # 周波数解像度（周波数インデックス1あたりのHz幅）
filterbank, fcenters = melFilterBank(fs, nfft, numChannels)

# メルフィルタバンクのプロット
plt.rcParams['font.size'] = 38
plt.figure(figsize=(30,10), dpi=80)
for c in np.arange(0, numChannels):
    plt.plot(np.arange(0, nfft / 2) * df, filterbank[c])
plt.xlabel('Hz')
#plt.ylabel('Magnitude of critical band filter')
plt.ylabel('Magnitude')
plt.savefig("melfilterbank.png")
plt.show()
