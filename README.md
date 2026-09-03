# Titanic Survival Prediction

## 1. プロジェクト概要

Kaggle の Titanic データセットを用いて、乗客の生存可否（`Survived`）を予測する二値分類プロジェクトです。  
単純な学習だけでなく、**特徴量設計・前処理・複数モデル比較・交差検証による評価**までを一連で実施し、最終的に提出用ファイルを生成しています。

扱っている課題は、欠損値やカテゴリ変数を含む表形式データに対して、どの前処理とモデルがより安定して高い予測性能を出せるかを検証することです。

## 2. 分析・開発のアプローチ

### 使用データ
- 学習データ: [`train.csv`](./train.csv)
- テストデータ: [`test.csv`](./test.csv)

### 前処理
`train_submission.py` では、`ColumnTransformer` + `Pipeline` を使って前処理を統合しています。

- 数値列
  - `SimpleImputer(strategy="median")`
  - `StandardScaler()`
- カテゴリ列
  - `SimpleImputer(strategy="most_frequent")`
  - `OneHotEncoder(handle_unknown="ignore")`

### 特徴量
`make_features()` 内で以下を作成しています。

- `Title`: 氏名 (`Name`) から敬称を抽出し、主要カテゴリ以外は `Rare` に集約
- `FamilySize`: `SibSp + Parch + 1`
- `IsAlone`: `FamilySize == 1` のフラグ
- `Deck`: `Cabin` 先頭文字（欠損は `U`）

また、`Survived`, `PassengerId`, `Name`, `Ticket`, `Cabin` は学習特徴量から除外しています（`errors="ignore"`）。

### 使用モデル・分析手法
候補モデル（[`train_submission.py`](./train_submission.py)）:
- Logistic Regression
- Random Forest (`n_estimators=500`, `min_samples_leaf=2`, `max_features="sqrt"`)

評価:
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- `cross_validate` による複数指標同時評価
  - Accuracy
  - ROC AUC
  - F1

最終的に ROC AUC 平均が最も高いモデルを選択し、全学習データで再学習後に `test.csv` へ予測を実施します。

### 評価方法
- 5-fold Stratified CV の平均値・標準偏差で比較
- 評価結果を [`model_evaluation.csv`](./model_evaluation.csv) に保存

## 3. 主な結果

[`model_evaluation.csv`](./model_evaluation.csv) に記録された結果:

- **Random Forest**
  - accuracy_mean: **0.8373**
  - accuracy_std: 0.0124
  - roc_auc_mean: **0.8803**
  - f1_mean: 0.7721
- **Logistic Regression**
  - accuracy_mean: 0.8294
  - accuracy_std: 0.0178
  - roc_auc_mean: 0.8723
  - f1_mean: **0.7730**

要約:
- ROC AUC と Accuracy は Random Forest が優位
- F1 は Logistic Regression がわずかに高い
- 本実装では ROC AUC を基準に Random Forest を選択し、提出ファイルを作成

## 4. ディレクトリ構成

主要ファイルのみ抜粋:

- [`titanic.ipynb`](./titanic.ipynb): Notebook ベースの分析・検討
- [`titanic_2_no_skills.ipynb`](./titanic_2_no_skills.ipynb): 比較用 Notebook
- [`train_submission.py`](./train_submission.py): 前処理〜学習〜評価〜提出ファイル生成を一括実行するスクリプト
- [`train.csv`](./train.csv): 学習データ
- [`test.csv`](./test.csv): テストデータ
- [`model_evaluation.csv`](./model_evaluation.csv): モデル比較結果
- [`submission.csv`](./submission.csv): 最終提出ファイル
- [`submission_no_skills.csv`](./submission_no_skills.csv): 比較用提出ファイル
- [`requirements.txt`](./requirements.txt): 実行依存パッケージ

## 5. 実行方法

### 必要な環境
- Python 3.10+（推奨）
- pip

### インストール
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 実行手順
```bash
python train_submission.py
```

実行後に以下が更新・生成されます:
- `model_evaluation.csv`
- `submission.csv`

## 6. 詳細資料

- Notebook:
  - [`titanic.ipynb`](./titanic.ipynb)
  - [`titanic_2_no_skills.ipynb`](./titanic_2_no_skills.ipynb)
- スクリプト:
  - [`train_submission.py`](./train_submission.py)

## 7. 使用技術

- 言語
  - Python
  - Jupyter Notebook
- 主要ライブラリ（[`requirements.txt`](./requirements.txt)）
  - pandas==2.3.3
  - scikit-learn==1.6.1
- ML / 統計手法
  - 二値分類（Logistic Regression, Random Forest）
  - 交差検証（Stratified K-Fold）
  - 指標評価（Accuracy / ROC AUC / F1）
  - 欠損値補完、標準化、One-Hot Encoding
