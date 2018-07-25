# README
#### スライド
- slide.pdf, slide.pptx
  研究発表のためのスライド

#### 前処理
- mkMFCC.py
  audioをMFCCに変換する
- mkFT.py
  audioを短時間フーリエ変換する
- mp3towav.py
  mp3ファイルをwavファイルに変換する
- loadWav.py
  wavファイルを読み込みcsvファイルに変換・保存する
  wavファイルを読み込みモデルの入力に変換する
- loadCsv.py
  csvファイルを読み込みモデルの入力に変換する
- readFav.py
  楽曲に割り当てた好きな部分をラベルとして教師信号を作成する


#### モデル
- my2dCNN+LSTM.py
  モデル
- prediction2d+LSTM.py
  学習したモデルを使って予測結果を出力する
