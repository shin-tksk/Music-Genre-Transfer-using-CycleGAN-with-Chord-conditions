# Music-Genre-Transfer-using-CycleGAN-with-Chord-conditions

- CycleGANを用いて異なる音楽ジャンル間でのドメイン変換を実現するモデルを構築した。
- コード情報を学習に組み込むことでジャンル変換後の移調を抑えたり、変換精度を上げることを目的としている。
- コード情報を取得するために、tonal centroid features(調性中心特徴)を使用する。
- 評価には別で学習したジャンル分類器を使用している。

# Model Architecture

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
