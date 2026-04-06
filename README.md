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
├── configs/
│   └── train_step1.yaml            # Hydra設定
├── docs/
│   └── step1_prototype.md
├── requirements.txt
└── legacy/
```

## Step1 の実行

```bash
source env/bin/activate
python scripts/train_step1_pretrain.py \
  model_size=tiny \
  steps=200 \
  batch_size=8 \
  height=128 width=128 \
  t_bins=8 patch_size=16 tubelet_size=2 \
  normalize_voxel=true normalize_targets=true
```

## 補足

- 現在の `scripts/train_step1_pretrain.py` はランダムイベント生成を使う最小プロトタイプです。
- 実データ (GEN4 / DSEC) に移行する場合は、同スクリプト内の `sample_voxel_batch` をデータローダに置き換えてください。
- 表現崩壊対策として、`SIGReg` / `VICReg` 系正則化をオプションで有効化できます。

### 追加正則化の例

```bash
source env/bin/activate
python scripts/train_step1_pretrain.py \
  collapse_strategy=vicreg \
  model_size=tiny \
  steps=200 \
  vicreg_std_weight=0.1 \
  vicreg_cov_weight=0.01 \
  vicreg_use_target=true
```

### Collapse Strategy 切り替え

- `collapse_strategy=ema_stopgrad`: EMA teacher + stop-grad（デフォルト）
- `collapse_strategy=vicreg`: VICReg 系正則化
- `collapse_strategy=sigreg`: SIGReg 正則化

```bash
source env/bin/activate
python scripts/train_step1_pretrain.py collapse_strategy=ema_stopgrad steps=200
python scripts/train_step1_pretrain.py collapse_strategy=vicreg steps=200 vicreg_std_weight=0.1 vicreg_cov_weight=0.01
python scripts/train_step1_pretrain.py collapse_strategy=sigreg steps=200 sigreg_weight=0.01
```
