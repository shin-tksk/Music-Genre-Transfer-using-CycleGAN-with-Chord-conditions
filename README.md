# Music-Genre-Transfer-using-CycleGAN-with-Chord-conditions

- CycleGANを用いて異なる音楽ジャンル間でのドメイン変換を実現するモデルの改良を行った。
- コード情報を学習に組み込むことでジャンル変換後の移調を抑えたり、変換精度を上げることを目的としている。
- コード情報を取得するために、tonal centroid features(調性中心特徴)を使用する。
- コード情報ありとコード情報なしの学習結果と変換後の移調率を評価する。
- 評価には別で学習したジャンル分類器を使用している。

## Tonal Centroid Features

楽曲のコードを検出するために、tonal centroid features(調性中心特徴)を使用。
（音程ベクトルをTonnetzに基づき五度圏、短三度圏、長三度圏にマッピングする）

MIDIデータから１２次元の音程ベクトルを算出して、L1正規化した後、6×12の三角関数基底の変換行列をかけて、2×3次元の音程空間にマッピングします。

計算式は以下のようになる。

<img src="imgs/tonal.png" width="500px"/>

以下に例として音程空間上の各ルート音のメジャーコードの座標を示す。

<img src="imgs/major_5th.png" width="250px"/><img src="imgs/major_ma3th.png" width="250px"/><img src="imgs/major_mi3th.png" width="250px"/>

音程空間上で入力データと各コード（メジャー、マイナー）のユークリッド距離を計算してコードを検出する。

<img src="imgs/distance.png" width="300px"/>

ζは楽曲データの音程空間の座標を表し、ζ'は距離を測りたいコードの座標を表す。

## Model Architecture

本研究のモデルは、2つのGANをサイクル的に構成し学習を行っている（CycleGAN）

<img src="imgs/model.png" width="700px"/>

GenはGenerator、DisはDiscriminator、AとBは2つのジャンルを示す。青と赤の矢印は2方向のジャンル変換、黒の矢印はDiscriminatorへの入力を表す。
x、x hat、x tlideはそれぞれ、実データ、変換後のデータ、再変換後のデータを表す。

GeneratorとDiscriminatorの構造は以下のようになっている。

<img src="imgs/gen.png" width="600px"/>
<img src="imgs/dis.png" width="350px"/>

## Genre Classfier

ジャンル変換の精度を評価するために分類器を別で学習させる。構造は以下のようになっている。

<img src="imgs/cla.png" width="400px"/>

分類器の分類精度は以下のようになっている。

<img src="imgs/cla_result.png" width="500px"/>

## Versions

- Python 3.7.0
- Numpy 1.19.5
- Scipy 1.7.2
- TensorFlow-gpu 1.14.0
- prttey midi 0.2.9
- Pypianoroll 1.0.4

## Datasets

この研究では、クラシック、ジャズ、ポップスの3つのジャンルを使用しています。
学習時はデータ数が少ないジャンルに合わせて学習を行う。

データ数は以下のようになっている。

<img src="imgs/data.png" width="500px"/>

以下に前処理の手順を説明します。

1.　pretty midiとpypianorollを二つのパッケージを用いて、MIDIデータをnumpy配列に変換する。最小単位は４８部音符とし、音高はC0からC8の84音を使用する。入力データは４小節であり、192×84の行列となる。

2.　MIDIデータのト全てのトラックをピアノトラックに落とし込む。またドラムトラックや交響曲は扱いが難しいので使用しない。

3.　ベロシティ（音量）は全て100に固定して2値の行列を作成する。

4.　4/4拍子以外の楽曲出ないものや拍子記号が曲中で変わるものを除外する。

以上の工程により、1つのデータは[192,84,1]となる。（3つ目の要素はチャネル数を表す）













