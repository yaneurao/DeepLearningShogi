# DeepLearningShogi(dlshogi)
[![pypi](https://img.shields.io/pypi/v/dlshogi.svg)](https://pypi.python.org/pypi/dlshogi)

将棋でディープラーニングの実験をするためのプロジェクトです。

基本的にAlphaGo/AlphaZeroの手法を参考に実装していく方針です。

検討経緯、実験結果などは、随時こちらのブログに掲載していきます。

http://tadaoyamaoka.hatenablog.com/

## ダウンロード
[Releases](https://github.com/TadaoYamaoka/DeepLearningShogi/releases)からダウンロードできます。

最新のモデルファイルは、[棋神アナリティクス](https://kishin-analytics.heroz.jp/lp/)でご利用いただけます。

## Value lossの重み付け

`dlshogi.train --value-loss-min-weight 0.5`は、教師評価値から変換された期待勝率`q`に応じて
各局面のvalue lossを重み付けします。範囲は0～1、デフォルトは1（従来通り）です。

```text
w = a + (1 - a) * 4 * q * (1 - q)
value loss = sum(w * ((1 - val_lambda) * BCE(result) + val_lambda * BCE(q))) / sum(w)
```

`a`が指定値です。中央`q=0.5`で重み1、端`q=0,1`で重み`a`になります。
`q`にはデータローダーの教師value（evalfix有効時は補正後）を使い、学習中のモデル予測値や
勝敗ラベルからは重みを計算しません。HCPE・HCPE3の両方に適用します。
policy loss、通常モデル・SWAモデルのテストlossとaccuracyは変更しません。

`--batches-per-update`併用時は、更新1回分の全ミニバッチを通した重み合計で正規化します。
現在のローダーと同じく不完全なミニバッチは扱わず、末尾の蓄積グループは実際のバッチ数を使います。
重み合計が0の場合はvalue勾配を0とし、policyの学習は継続します。
`a=1`では従来の計算経路を維持します。`a<1`かつ勾配蓄積時は、policyとvalueの逆伝播を
分けて計算しvalue勾配を別バッファに保持するため、追加の計算時間とモデル勾配相当のメモリが必要です。
ミニバッチの入力や計算グラフを蓄積数分保持することはありません。

この変更はPythonのみで、ネイティブ拡張の再ビルドは不要です。

## Policy教師の混合

`dlshogi.train`の`--policy-mix`は、HCPE3の選択手だけの正解分布と、visitNumから作る分布を混ぜる比率です。
範囲は0〜1、既定値は1（従来動作）です。`--temperature`を適用した後の分布に対して、
`(1-policy_mix) * one_hot(selectedMove16) + policy_mix * visit_distribution`を各レコードで計算し、その後で重複局面を平均します。
0は選択手だけ、0.1は選択手90%とvisit分布10%、1はvisit分布だけです。
選択手が最大visitNumの手と異なる棋譜では、temperature=0とpolicy-mix=0は異なります。
hcpe教師、value教師、テストの採点は変更しません。既存キャッシュは選択手を保持しないため、1以外は`--cache`と併用できません。

```powershell
python setup.py build_ext --inplace --force
python -m dlshogi.train train.hcpe3 test.hcpe --temperature 0.1 --policy-mix 0.1
```

更新後はC++拡張モジュールの再ビルドが必要です。対応するC++コンパイラ、Cython、NumPy、setuptoolsのあるPython環境で実行してください。

## ソース構成
|フォルダ|説明|
|:---|:---|
|cppshogi|Aperyを流用した将棋ライブラリ（盤面管理、指し手生成）、入力特徴量作成|
|dlshogi|ニューラルネットワークの学習（Python）|
|dlshogi/utils|ツール類|
|selfplay|MCTSによる自己対局|
|test|テストコード|
|usi|対局用USIエンジン|
|usi_onnxruntime|OnnxRuntime版ビルド用プロジェクト|

## ビルド環境
### USIエンジン、自己対局プログラム
#### Windowsの場合
* Windows 11 64bit
* Visual Studio 2022
#### Linuxの場合
* Ubuntu 18.04 LTS / 20.04 LTS
* g++
#### Windows、Linux共通
* CUDA 12.1
* cuDNN 8.9
* TensorRT 8.6

※CUDA 10.0以上であれば変更可

### 学習部
上記USIエンジンのビルド環境に加えて以下が必要
* [Pytorch](https://pytorch.org/) 1.6以上
* Python 3.7以上 ([Anaconda](https://www.continuum.io/downloads))
* CUDA (PyTorchに対応したバージョン)
* cuDNN (CUDAに対応したバージョン)

## 謝辞
* 将棋の局面管理、合法手生成に、[Apery](https://github.com/HiraokaTakuya/apery)のソースコードを使用しています。
* モンテカルロ木探索の実装は囲碁プログラムの[Ray+Rn](https://github.com/zakki/Ray)の実装を参考にしています。
* 探索部の一部にLeela Chess Zeroのソースコードを流用しています。
* 王手生成などに、[やねうら王](https://github.com/yaneurao/YaneuraOu)のソースコードを流用しています。

## ライセンス
ライセンスはGPL3ライセンスとします。
