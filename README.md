# spec-decomp-ref
Spectral Decomposition via Reference Data
参照データを用いたスペクトル分解ツール

## Requirements

* **OS**: Ubuntu 22.04 LTS
* **Language**: Python >= 3.9

## Installation
```{this repo}/```ディレクトリに移動後：
```bash
uv venv
source .venv/bin/activate
uv sync
```
インストールによる不具合について責任は負いかねます。

## Usage
### データの準備

#### 解析対象のデータ
```{this repo}/data/*.csv```に解析対象のCSVファイルを置く．  
複数ファイルに対応可能（{this repo}/data/*.csvのファイルはすべて読み込まれる）．  
一つ一つのcsvファイルは2カラムデータ形式となっている必要あり．

#### 参照データ
```{this repo}/data/reference.csv```に参照データCSVシートを置く．  
↑```reference.csv```という名前は必須．

#### 設定ファイル
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

### 解析プログラムの実行
```bash
python src/main.py
```

## 出力結果
```{this repo}/out/```に解析結果が吐き出される．

## 設計概要

![設計](/img/architecture_design.png)

## References
[1] R. Murakami, H. Tanaka, H. Shinotsuka, K. Nagata, H. Shouno, H. Yoshikawa, "Development of multiple core-level XPS spectra decomposition method based on the Bayesian information criterion", Journal of Electron Spectroscopy and Related Phenomena. 245 (2020) 147003. https://doi.org/10.1016/j.elspec.2020.147003.

## Author
* **Ryo Murakami, Hiroshi Shinotsuka**
* NIMS
