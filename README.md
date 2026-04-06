# JEPA Module (Event + JEPA)

イベントデータ向け JEPA 事前学習（Step1）のプロトタイプ実装です。

## 想定環境

- Python 3.11
- macOS（開発・テスト）
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
│   │   ├── representations.py      # voxel grid 変換などイベント表現
│   │   └── data/                   # データセット/コレータ/バッチ供給
│   │       ├── n_imagenet.py
│   │       ├── providers.py
│   │       └── __init__.py
│   └── jepa/
│       ├── masks/                  # マスク生成/適用
│       ├── models/                 # ViT encoder / predictor
│       ├── utils/                  # distributed / scheduler utility
│       └── regularizers.py         # SIGReg / VICReg
├── scripts/
│   └── train_step1_pretrain.py     # step1: 同時刻マスク予測
├── configs/
│   └── train_step1.yaml            # Hydra 設定
├── docs/
│   ├── step1_prototype.md
│   └── n_imagenet_data_design.md
└── requirements.txt
```

## Step1 実行例

### synthetic（デフォルト）

```bash
source env/bin/activate
python scripts/train_step1_pretrain.py \
  data.source=synthetic \
  model_size=tiny \
  steps=200 \
  batch_size=8 \
  height=128 width=128 \
  t_bins=8 patch_size=16 tubelet_size=2 \
  normalize_voxel=true normalize_targets=true
```

### N-ImageNet

```bash
source env/bin/activate
python scripts/train_step1_pretrain.py \
  data.source=n_imagenet \
  data.n_imagenet.split=train \
  data.n_imagenet.train_list=/absolute/path/to/train_list.txt \
  data.n_imagenet.root_dir=/absolute/path/to/N_Imagenet \
  data.n_imagenet.compressed=true \
  batch_size=8 \
  data.num_workers=4
```

## Collapse Strategy 切り替え

- `collapse_strategy=ema_stopgrad`: EMA teacher + stop-grad（デフォルト）
- `collapse_strategy=vicreg`: VICReg 系正則化
- `collapse_strategy=sigreg`: SIGReg 正則化

```bash
source env/bin/activate
python scripts/train_step1_pretrain.py collapse_strategy=ema_stopgrad steps=200
python scripts/train_step1_pretrain.py collapse_strategy=vicreg steps=200 vicreg_std_weight=0.1 vicreg_cov_weight=0.01
python scripts/train_step1_pretrain.py collapse_strategy=sigreg steps=200 sigreg_weight=0.01
```

## Scheduler（Warmup + Cosine）

```bash
source env/bin/activate
python scripts/train_step1_pretrain.py \
  scheduler.enabled=true \
  scheduler.warmup_steps=1000 \
  scheduler.start_lr=1.0e-6 \
  scheduler.final_lr=1.0e-6 \
  scheduler.update_weight_decay=true \
  scheduler.final_weight_decay=0.001
```

## Distributed（DDP）

```bash
source env/bin/activate
torchrun --standalone --nproc_per_node=2 scripts/train_step1_pretrain.py \
  distributed.enabled=true \
  data.source=n_imagenet \
  data.n_imagenet.train_list=/absolute/path/to/train_list.txt
```

## 補足

- 学習データは `data.source` で切り替えます（`synthetic` / `n_imagenet`）。
- `n_imagenet` では list file（1行1サンプルの npz パス）を読み込みます。
- 学習出力は timestamp ごとに `outputs/train/YYYY-MM-DD/HH-MM-SS/` 配下へまとまります。
- 例:
  - checkpoint: `outputs/train/.../.../checkpoints/step_000100.pt`
  - metrics: `outputs/train/.../.../logs/train_metrics.csv`
  - eval metrics（有効時）: `outputs/train/.../.../logs/eval_metrics.csv`
  - tensorboard: `outputs/train/.../.../logs/tensorboard/`
  - hydra log: `outputs/train/.../.../*.log`

### TensorBoard

```bash
source env/bin/activate
tensorboard --logdir outputs/train
```

- 記録される主な値:
  - `train/loss`, `train/recon`, `train/sig`, `train/std`, `train/cov`
  - `train/lr`, `train/weight_decay`
  - `eval/*`（`eval.enabled=true` のとき）
