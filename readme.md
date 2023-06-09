# 配布データと応募用ファイル作成方法の説明

本コンペティションで配布されるデータと応募用ファイルの作成方法について説明する.

1. [配布データ](#配布データ)
2. [応募用ファイルの作成方法](#応募用ファイルの作成方法)
<br>
<br>

## 配布データ

配布されるデータは以下の通り.

- [readme](#readme)
- [学習用データ](#学習用データ)
- [動作確認用のプログラム](#動作確認用のプログラム)
- [応募用サンプルファイル](#応募用サンプルファイル)
<br>
<br>

### readme

本ファイルである"readme.md"のことで, 配布用データの説明と応募用ファイルの作成方法を説明したドキュメント. マークダウン形式で, プレビューモードで見ることを推奨する.
<br>
<br>

### 学習用データ

学習用データは"train.zip"で, 解凍すると以下のようなディレクトリ構造のデータが作成される.

```bash
train
├─ road.csv
├─ search_data.csv
├─ search_unspec_data.csv
└─ train.csv
```
内容は以下の通り.

- [道路構造データ](#道路構造データ)
- [トラフィックカウンターデータ](#トラフィックカウンタートラカンデータ)
- [ドラぷらルート検索データ](#ドラぷらルート検索データ)
<br>
<br>

#### 道路構造データ

- road.csv
  - ICの名前や場所, 区間(始点ICと終点IC)の距離などの属性データ. サイズ:1Mb

|  No |  column           |  型      |  説明   |  値例  |
| :---|  :---             |  :---:   |  :---  |  :---  |
|  1  |  start_name       | str      |  区間の始点ICの名称  |  所沢  |
|  2  |  end_name         | str      |  区間の終点ICの名称  |  大泉JCT  |
|  3  |  start_code       | int      |  区間の始点ICコード  |  1800006  |
|  4  |  end_code         | int      |  区間の終点ICコード  |  1110210  |
|  5  |  start_pref_code  | int      |  区間の始点ICがおける[県コード](https://nlftp.mlit.go.jp/ksj/gml/codelist/PrefCd.html)  |  11  |
|  6  |  end_pref_code    | int      |  区間の終点ICがおける[県コード](https://nlftp.mlit.go.jp/ksj/gml/codelist/PrefCd.html)  |  13  |
|  7  |  start_lat        | float    |  区間の始点ICの緯度  |  35.80615  |
|  8  |  end_lat          | float    |  区間の終点ICの緯度  |  35.75582  |
|  9  |  start_lng        | float    |  区間の始点ICの経度  |  139.535511  |
|  10 |  end_lng          | float    |  区間の終点ICの経度  |  139.601514  |
|  11 |  start_degree     | int      |  区間の始点ICが接続する他ICの数  |  2  |
|  12 |  end_degree       | int      |  区間の終点ICが接続する他ICの数  |  4  |
|  13 |  direction        | str      |  方向  |  上り  |
|  14 |  KP               | float    |  トラフィックカウンタの[KP](#kpキロポストとは)(km)  |  10.21  |
|  15 |  start_KP         | float    |  始点ICのKP(km)  |  43.7  |
|  16 |  end_KP           | float    |  終点ICのKP(km)  |  35.7  |
|  17 |  limit_speed      | int      |  区間の制限速度(km/h)  |  100  |
|  18 |  road_code        | int      |  道路コード(関越道:1800, 館山道:1130)  |  1800  |

#### KP(キロポスト)とは
*KPは起点からの距離を表す. 関越道起点は大泉JCT（練馬IC）, 館山道起点は篠崎IC（京葉道路）としている.*
<br>
<br>

#### トラフィックカウンター(トラカン)データ

- train.csv
  - 各道路区間を通過する自動車の速度, 台数などを記録した時系列データ. サイズ:64Mb

|  No  |  column           |  型      |  説明  |  値例  |
| :--- |  :---             |  :---:   |  :---  |  :---  |
|  1   |  datetime         | str      |  日時  |  2021-04-08 00:00:00  |
|  2   |  start_code       | int      |  区間の始点ICコード  |  1800006  |
|  3   |  end_code         | int      |  区間の終点ICコード  |  1110210  |
|  4   |  allCars          | int      |  1時間内の全車線の通過台数の合計  |  568  |
|  5   |  OCC              | float    |  1時間内の全車線の占有率(%)  |  1.6  |
|  6   |  speed            | float    |  1時間内の全車線の平均速度  |  87  |
|  7   |  is_congestion    | int      |  渋滞あり・なし(目的変数)  |  0 or 1  |

*※speedが40以下の場合にis_congestionが1(渋滞あり)と定義する.*

<br>
<br>

#### ドラぷらルート検索データ

- search_data.csv
  - 各道路区間の時間指定**あり**検索数. サイズ:40Mb

|  No |  column           |  型      |  説明  |  値例  |
| :---|  :---             |  :---:   |  :---  |  :---  |
|  1  |  datetime         | str      |  日時 |  2021-04-08 00:00:00  |
|  2  |  start_code       | int      |  区間の始点ICコード  |  1800006  |
|  3  |  end_code         | int      |  区間の終点ICコード  |  1110210  |
|  4  |  search_1h        | int      |  1時間ごとに集計したドラぷらルート検索データ(時間指定あり検索数)  |  4  |

<br>
<br>


- search_unspec_data.csv
  - 各道路区間の時間指定**なし**検索数. サイズ:2Mb  

|  No |  column           |  型      |  説明  |  値例  |
| :---|  :---             |  :---:   |  :---  |  :---  |
|  1  |  date             | str      |  日付  |  2021-04-08  |
|  2  |  start_code       | int      |  区間の始点ICコード  |  1800006  |
|  3  |  end_code         | int      |  区間の終点ICコード  |  1110210  |
|  4  |  search_unspec_1d | int      |  前日分として集計したドラぷらルート検索データ(時間指定なし検索数)  |  4  |

<br>
<br>


### 動作確認用のプログラム

予測を行うプログラムの動作を確認するためのプログラム. "run_test.zip"が本体で, 解凍すると, 以下のようなディレクトリ構造のデータが作成される.

```bash
run_test
└─ run.py
```

使用方法等については[応募用ファイルの作成方法](#応募用ファイルの作成方法)を参照すること.
<br>
<br>

### 応募用サンプルファイル

応募用のサンプルファイル. 実体は"sample_submit.zip"で, 解凍すると以下のようなディレクトリ構造のデータが作成される.

```bash
sample_submit
├── model
│   └── ...
├── src
│   └── predictor.py
└── requirements.txt
```

sample_submit/modelは空のディレクトリとなる. sample_submit/requirements.txtには何も書いていない. 詳細や作成方法については[応募用ファイルの作成方法](#応募用ファイルの作成方法)を参照すること.
<br>
<br>

## 応募用ファイルの作成方法

学習済みモデルを含めた, 予測を実行するためのソースコード一式をzipファイルでまとめたものとする.
<br>
<br>

### ディレクトリ構造

以下のようなディレクトリ構造となっていることを想定している.

```bash
sample_submit
├── model              必須: 学習済モデルを置くディレクトリ
│   └── ...
├── src                必須: Pythonのプログラムを置くディレクトリ
│   ├── predictor.py   必須: 投稿時に予測プログラムとして呼び出されるファイル
│   └── ...            任意: その他のファイル (ディレクトリ作成可能)
└── requirements.txt   任意: 追加で必要なライブラリ一覧
```

- 学習済みモデルの格納場所は"model"ディレクトリを想定している.
  - 学習済みモデルを使用しない場合でも空のディレクトリを作成する必要がある.
  - 名前は必ず"model"とすること.
- Pythonのプログラムの格納場所は"src"ディレクトリを想定している.
  - 学習済みモデル等を読み込んで推論するためのメインのソースコードは"predictor.py"を想定している.
    - ファイル名は必ず"predictor.py"とすること.
  - その他予測を実行するために必要なファイルがあれば作成可能である.
  - ディレクトリ名は必ず"src"とすること.
- 実行するために追加で必要なライブラリがあれば, その一覧を"requirements.txt"に記載することで, 評価システム上でも実行可能となる.
  - インストール可能で実行可能かどうか予めローカル環境で試しておくこと.
  - 評価システムの実行環境については, [*こちら*](https://github.com/signatelab/runtime-gpu)を参照すること.
<br>
<br>

### predictor.pyの実装方法

以下のクラスとメソッドを実装すること.
<br>
<br>

#### ScoringService

予測実行のためのクラス. 以下のメソッドを実装すること.
<br>
<br>

##### get_model

モデルを取得するメソッド. 以下の条件を満たす必要がある.

- クラスメソッドであること.
- 引数としてmodel_path(str型)を指定すること.
  - 学習済みモデルが格納されているディレクトリのパスが渡される.
- 引数としてinference_df(DataFrame型)を指定すること.
  - 予測対象の期間よりも過去のDataFrameが推論補助データとして渡される. 渡されるDataFrameは下記.
    - ローカルでの検証時: 2021-04-08以降, かつ検証実行時に指定したstart_date以前DataFrame
    - 暫定評価時: 2021-04-08 ~ 2022-07-31のDataFrame
    - 最終評価時: 2021-04-08 ~ 2023-03-31のDataFrame
- 学習済みモデルの読み込みに成功した場合はTrueを返す.
  - モデル自体は任意の名前(例えば"model")で保存しておく.
<br>
<br>

##### predict

予測を実行するメソッド. 以下の条件を満たす必要がある.

- クラスメソッドであること.
- 引数としてinput(DataFrame型)を指定すること.
  - inputには**当日**の[学習用データ](#学習用データ)をマージしたものがDataFrameで渡される.
  - `get_model`メソッドで読み込んだ学習済みモデルや推論補助データを用いて**翌日**の渋滞予測を行う想定である.
  - inputに与えられるDataFrameのフォーマットは以下.


|  No |  column           |  型       |
| :---|  :---             |  :---:    |
|  1  |  datetime         |  datetime |
|  2  |  start_code       |  int      |
|  3  |  end_code         |  int      |
|  4  |  OCC              |  float    |
|  5  |  allCars          |  int      |
|  6  |  speed            |  float    |
|  7  |  is_congestion    |  int      |
|  8  |  start_name       |  str      |
|  9  |  end_name         |  str      |
|  10 |  start_pref_code  |  int      |
|  11 |  end_pref_code    |  int      |
|  12 |  start_lat        |  float    |
|  13 |  end_lat          |  float    |
|  14 |  start_lng        |  float    |
|  15 |  end_lng          |  float    |
|  16 |  start_degree     |  float    |
|  17 |  end_degree       |  float    |
|  18 |  KP               |  float    |
|  19 |  direction        |  str      |
|  20 |  start_KP         |  float    |
|  21 |  end_KP           |  float    |
|  22 |  limit_speed      |  int      |
|  23 |  road_code        |  int      |
|  24 |  search_1h        |  float    |
|  25 |  search_unspec_1d |  float    |

<br>
<br>

- 予測結果を**DataFrame型**で返すように作成すること. フォーマットは以下.

|  No |  column           |  型            |
| :---|  :---             |  :---:         |
|  1  |  datetime         |  datetime/str  |
|  2  |  start_code       |  int/str       |
|  3  |  end_code         |  int/str       |
|  4  |  prediction       |  int           |

- 各要素は以下を満たす.
  - "datetime", "start_code", "end_code"は, ある区間のある日時を表すindexである.
  - "prediction"は, 各indexの翌日が渋滞しているかどうかを予測したもの(0か1)とすること.

<br>
<br>

##### 実装例
以下は実装例. 翌日が土曜日もしくは日曜日の場合にのみ, 渋滞と予測する単純な処理である.

```Python
import pandas as pd

class ScoringService(object):
    @classmethod
    def get_model(cls, model_path, inference_df):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            inference_df: Past data not subject to prediction.

        Returns:
            bool: The return value. True for success.
        """
        cls.model = None
        cls.data = inference_df

        return True


    @classmethod
    def predict(cls, input):
        """Predict method

        Args:
            input: meta data of the sample you want to make inference from (DataFrame)

        Returns:
            prediction: Inference for the given input. Return columns must be ['datetime', 'start_code', 'end_code'](DataFrame).
        
        Tips:
            You can use past data by writing "cls.data".
        """
        
        prediction = input.copy()
        prediction['weekday'] = pd.to_datetime(prediction['datetime'], format='%Y-%m-%d').apply(lambda x: x.weekday())
        prediction['prediction'] = prediction['weekday'].apply(lambda x: 1 if x==4 or x==5 else 0)
        prediction = prediction[['datetime', 'start_code', 'end_code', 'prediction']]

        return prediction
```

応募用サンプルファイル"sample_submit.zip"も参照すること.
<br>
<br>

### 推論テスト

予測を行うプログラムが実装できたら, 推論テスト(正常に動作するかの確認)を行う.
推論テストを行うには下記のようなディレクトリ構造を作成する.

```bash
run_test
├── sample_submit
│   ├── model
│   │   └── ...
│   ├── src
│   │   └── predictor.py
│   └── requirements.txt
├── train
│   ├─ road.csv
│   ├─ search_data.csv
│   ├─ search_unspec_data.csv
│   └─ train.csv
└── run.py

```

<br>
<br>

#### 環境構築

評価システムと[同じ環境](https://github.com/signatelab/runtime-gpu)を用意する.
<br>
<br>

#### 予測の実行手順

1. [推論テストのディレクトリ構造](#推論テスト)を参照し, 同じ構造のファイルを作成
2. src/predictor.pyに, 予測プログラムを実装する.
    - 機械学習モデルを作成する場合, 下記手順で実装する.
      - 配布データを[input](#predict)のフォーマットに変形し学習を行い, modelディレクトリ内に作成したモデルを格納する.
      - modelディレクトリから学習済みモデルを呼び出し, 再学習や予測を行う処理をpredictor.pyの`predict`メソッドに実装する.
    - inference_dfを使用する場合, `get_model`メソッド内で受け取ったinference_dfを`predict`メソッドで呼び出すことができる. (デフォルトの場合`cls.data`に格納されている)
3. [動作確認用のプログラム](#動作確認用のプログラム)を用いて予測を実行し, 最後に出力される精度(F1_score)を確認する. 下記コマンドを実行することで処理が行われる.


```bash
$ cd /path/run_test
$ python run.py  --exec-path /path/to/submit/src --data-dir /path/to/train --start-date start_date --end-date end_date
...
```

- 引数"--exec-path"には実装した予測プログラム("predictor.py")が存在するパス名を指定する.
- 引数"--data-dir"には配布された学習用データ"train"のパス名を指定する.
- 引数"--start-date"には検証用データを作成するときの最初の日付(2021-04-08 ~ 2022-07-31)をYYYY-MM-ddのフォーマットで指定する. (指定しない場合は2021-04-08)
- 引数"--end-date"には検証用データを作成するときの最後の(2021-04-08 ~ 2022-07-31)の日付をYYYY-MM-ddのフォーマットで指定する. (指定しない場合は2022-07-31)
  - "--start-date"で指定した日付よりも後の日付を設定すること.

"run.py"の実行に成功すると, 検証用のデータを作成し実装したプログラム(predictor.py)による予測が行われ, 最後に精度が出力される. 
プログラム上のエラーがない場合でも, モデルの読み込みに失敗したときや, 予測結果のフォーマットが正しくないときは途中でエラーのメッセージが返され, 止まる. 
具体例は下記.
- `get_model`メソッドを呼んだときにTrueを返さない.
- 予測結果がDataFrame型でない.
- `predict`メソッドから出力されるpredictionのcolumnが"datetime", "start_code", "end_code", "prediction"でない.

投稿する前にエラーが出ずに実行が成功することを確認すること.
<br>
<br>

### 応募用ファイルの作成

上記の[ディレクトリ構造](#ディレクトリ構造)となっていることを確認して, zipファイルとして圧縮する.