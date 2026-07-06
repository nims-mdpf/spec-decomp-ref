# spec-decomp-ref
Spectral Decomposition via Reference Data
参照データを用いたスペクトル分解ツール

## データの準備

### 解析対象のデータ
```{this repo}/data/*.csv```に解析対象のCSVファイルを置く．  
複数ファイルに対応可能（{this repo}/data/*.csvのファイルはすべて読み込まれる）．  
一つ一つのcsvファイルは2カラムデータ形式となっている必要あり．

### 参照データ
```{this repo}/data/reference.csv```に参照データCSVシートを置く．  
↑```reference.csv```という名前は必須．

### 設定ファイル
```{this repo}/data/config.json```に解析設定のJSONファイルを置く．  
↑```config.json```という名前は必須．
テンプレートを以下に示す：
```
{
    "model": {
        "peak" : "VoigtPeaks",
        "background" : "LinearBackground",
        "noise" : "GaussianNoise",
        "regularization" : "BayesianInformationCriterion"
    },
    "states" : ["CaO", "CaCO3"],
    "is_common": {
        "ratio" : true,
        "shift" : false,
        "sigma" : false,
        "gamma" : true
    },
    "bounds" : {
        "ratio" : [0, 1],
        "shift" : [-0.5, 0.5],
        "sigma" : [0.3, 0.8],
        "gamma" : [0.1, 0.4]
    }
}
```


## Dockerコンテナの作成
```{this repo}/container/```ディレクトリに移動後：
```
docker build -t ref-spe-image .
```

```{this repo}/```ディレクトリに移動後：
```
docker run -it --rm -v `pwd`:/app ref-spe-image bash
```
解析プログラムの実行
```
python src/main.py
```

## 出力結果
```{this repo}/out/```に解析結果が吐き出される．

### 設計概要

![設計](/img/アーキテクチャ設計.png)
