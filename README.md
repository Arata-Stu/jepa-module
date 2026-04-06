# JEPA Module (Event + JEPA)

イベントデータ向け JEPA 事前学習のプロトタイプ実装です。

## 想定環境

- Python 3.11
- macOS (開発・テスト)
- 仮想環境: `./env`

## セットアップ

```bash
source env/bin/activate
python --version
pip install -r requirements.txt
```

## ディレクトリ構成

```text
.
├── src/
│   ├── event/
│   │   └── representations.py      # voxel grid 変換などイベント表現
│   └── jepa/
│       ├── masks/                  # マスク生成/適用
│       ├── models/                 # ViT encoder / predictor
│       │   └── utils/              # attention block / patch embed
│       └── utils/                  # tensor helper
├── scripts/
│   └── train_step1_pretrain.py     # step1: 同時刻マスク予測
├── docs/
│   └── step1_prototype.md
├── requirements.txt
└── legacy/
```

## Step1 の実行

```bash
source env/bin/activate
python scripts/train_step1_pretrain.py \
  --model-size tiny \
  --steps 200 \
  --batch-size 8 \
  --height 128 --width 128 \
  --t-bins 8 --patch-size 16 --tubelet-size 2 \
  --normalize-voxel --normalize-targets
```

## 補足

- 現在の `scripts/train_step1_pretrain.py` はランダムイベント生成を使う最小プロトタイプです。
- 実データ (GEN4 / DSEC) に移行する場合は、同スクリプト内の `sample_voxel_batch` をデータローダに置き換えてください。
