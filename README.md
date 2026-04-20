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
│   │       ├── dsec.py
│   │       ├── n_imagenet.py
│   │       ├── providers.py
│   │       └── __init__.py
│   └── jepa/
│       ├── masks/                  # マスク生成/適用
│       ├── models/                 # ViT encoder / predictor
│       ├── utils/                  # distributed / scheduler utility
│       └── regularizers.py         # SIGReg / VICReg
├── scripts/
│   └── train_jepa_pretrain.py     # step1: 同時刻マスク予測
├── configs/
│   └── train_jepa.yaml             # Hydra 設定（JEPA）
├── docs/
│   ├── step1_prototype.md
│   └── n_imagenet_data_design.md
└── requirements.txt
```

## Step1 実行例

### synthetic（動作確認用）

```bash
source env/bin/activate
python scripts/train_jepa_pretrain.py \
  data.source=synthetic \
  model_size=tiny \
  steps=200 \
  batch_size=8 \
  height=240 width=320 \
  t_bins=10 patch_size=16 tubelet_size=2 \
  temporal_mix.enabled=true \
  temporal_mix.short_t=1 \
  normalize_voxel=true normalize_targets=true
```

### Pretrain Mixed（デフォルト設定そのまま）

```bash
source env/bin/activate
python scripts/train_jepa_pretrain.py
```

- デフォルトでは `data.pretrain_mixed.augment.enabled=true`（RRC含む）と
  `scheduler.enabled=true` で実行されます。
- 無効化したい場合は `data.pretrain_mixed.augment.enabled=false scheduler.enabled=false` を指定してください。

### 途中再開（checkpoint resume）

```bash
source env/bin/activate

# 明示パスから再開
python scripts/train_jepa_pretrain.py \
  resume_from=/absolute/path/to/checkpoints/step_050000.pt \
  steps=100000

# out_dir 内の最新 checkpoint から自動再開
python scripts/train_jepa_pretrain.py \
  hydra.run.dir=/absolute/path/to/previous/run \
  out_dir=/absolute/path/to/previous/run/checkpoints \
  log_dir=/absolute/path/to/previous/run/logs \
  metrics_file=/absolute/path/to/previous/run/logs/train_metrics.csv \
  auto_resume=true \
  steps=100000
```

- checkpoint には model / optimizer / step / RNG state が保存されます。
- `ema_stopgrad` の場合は teacher(EMA) も保存され、同状態から再開されます。
- `resume_load_optimizer=false` や `resume_load_rng_state=false` で部分再開も可能です。

### N-ImageNet

```bash
source env/bin/activate
python scripts/train_jepa_pretrain.py \
  data.source=n_imagenet \
  data.n_imagenet.split=train \
  data.n_imagenet.train_list=/absolute/path/to/train_list.txt \
  data.n_imagenet.root_dir=/absolute/path/to/N_Imagenet \
  data.n_imagenet.compressed=true \
  batch_size=8 \
  data.num_workers=4
```

### N-ImageNet Linear Probe（下流分類・凍結エンコーダ）

```bash
source env/bin/activate
python scripts/train_n_imagenet_linear_probe.py \
  pretrained.checkpoint=/absolute/path/to/pretrain/step_100000.pt \
  model.model_size=tiny \
  model.height=240 model.width=320 \
  model.t_bins=10 model.patch_size=16 model.tubelet_size=2 \
  data.n_imagenet.train_list=/absolute/path/to/train_list.txt \
  data.n_imagenet.val_list=/absolute/path/to/val_list.txt \
  data.n_imagenet.root_dir=/absolute/path/to/N_Imagenet \
  batch_size=64
```

- デフォルトで `data.n_imagenet.force_full_event_input=true` のため、1ファイル内イベントを全量入力します（slice 無効）。
- 学習対象は線形 head のみで、encoder は checkpoint 読み込み後に凍結されます。
- `pretrained.encoder_key=teacher_encoder`（既定）で teacher を優先し、無ければ `encoder` へフォールバックします。
- `model.*`（`height/width/t_bins/patch_size/tubelet_size/temporal_mix.*`）は pretrain 時と一致させてください。

### DSEC

```bash
source env/bin/activate
python scripts/train_jepa_pretrain.py \
  data.source=dsec \
  data.dsec.root_dir=/absolute/path/to/dsec_root \
  data.dsec.split=train \
  data.dsec.split_config=/absolute/path/to/train_val_test_split.yaml \
  data.dsec.load_events=true \
  data.dsec.load_rgb=false \
  data.dsec.load_labels=false \
  batch_size=8 \
  data.num_workers=4
```

### Pretrain Mixed（DSEC + Gen4 + N-ImageNet, 自己教師あり）

```bash
source env/bin/activate
python scripts/train_jepa_pretrain.py \
  data.source=pretrain_mixed \
  eval.enabled=false \
  t_bins=10 \
  temporal_mix.enabled=true \
  temporal_mix.short_t=1 \
  temporal_mix.image_prob=0.5 \
  temporal_mix.short_mode=sum \
  data.pretrain_mixed.dsec.enabled=true \
  data.pretrain_mixed.dsec.root_dir=/absolute/path/to/dsec-downsampled \
  data.pretrain_mixed.gen4.enabled=true \
  data.pretrain_mixed.gen4.root_dir=/absolute/path/to/gen4-downsampled \
  data.pretrain_mixed.n_imagenet.enabled=true \
  data.pretrain_mixed.n_imagenet.root_dir=/absolute/path/to/n-imagenet-downsample \
  data.pretrain_mixed.n_imagenet.manifest_file=/absolute/path/to/n_imagenet_train_manifest.txt \
  data.pretrain_mixed.events_per_sample_min=20000 \
  data.pretrain_mixed.events_per_sample_max=80000 \
  data.pretrain_mixed.window_duration_us_min=200000 \
  data.pretrain_mixed.window_duration_us_max=1200000 \
  data.pretrain_mixed.duration_sources=[dsec,gen4] \
  data.pretrain_mixed.weights.dsec=1.0 \
  data.pretrain_mixed.weights.gen4=1.0 \
  data.pretrain_mixed.weights.n_imagenet=1.0 \
  data.pretrain_mixed.augment.enabled=true \
  data.pretrain_mixed.augment.hflip_prob=0.5 \
  data.pretrain_mixed.augment.max_shift=8 \
  data.pretrain_mixed.augment.time_flip_prob=0.2 \
  data.pretrain_mixed.augment.polarity_flip_prob=0.2 \
  data.pretrain_mixed.augment.random_resized_crop.enabled=true \
  data.pretrain_mixed.augment.random_resized_crop.prob=0.5 \
  data.pretrain_mixed.augment.random_resized_crop.scale_min=0.8 \
  data.pretrain_mixed.augment.random_resized_crop.scale_max=1.0 \
  data.pretrain_mixed.augment.random_resized_crop.preserve_aspect=true
```

- `weights` でデータセット混合比率を制御できます。
- `weight <= 0` の source は除外され、`root_dir` 未設定でもエラーになりません。
- `*.manifest_file` を設定すると、ディレクトリ再帰走査をスキップして list ファイルから読み込みます（`.txt` / `.csv` / `.npy`）。
- `window_duration_us_*` はシーケンス系（例: DSEC/Gen4）の時間窓可変サンプリング用です。
- `events_per_sample_*` はイベント数ベースの可変切り出しです。
- 停止シーン対策には `min_events_in_window` と `min_event_rate_eps`（events/sec）を使えます。
  条件を満たさない窓は同一ファイル内で `max_window_attempts` 回まで再サンプリングします。
- `data.pretrain_mixed.augment.enabled=true` でイベント段階の拡張を有効化できます。
  `hflip/max_shift/time_flip/polarity_flip` と `random_resized_crop.*`（ランダム矩形crop→座標リサイズ）が使えます。
  `random_resized_crop.preserve_aspect=true` でセンサ比を維持した等方スケーリングになります。
- `t_bins=10 + temporal_mix.short_t=1` で 10bin/1bin を学習中に混在できます。
- H5読み込み失敗時は別サンプルへ自動リトライし、worker が落ちにくい挙動にしています。
- 範囲外 index の追跡は `data.pretrain_mixed.debug_index_check=true` で有効化できます。
  詳細を即停止で見たいときは `data.pretrain_mixed.debug_raise_on_oob=true` を併用します。
- DataLoader 周りで `pin_memory` スレッドのエラーが出る場合は、まず
  `data.num_workers=0 data.pin_memory=false` で安定動作を確認し、その後 `num_workers` を増やしてください。

### Event Voxel 可視化（赤青白）

```bash
source env/bin/activate
python scripts/visualize_pretrain_voxels.py \
  data.source=pretrain_mixed
```

- 出力先: `outputs/train/.../.../event_viz/`
- `index.csv` に `source` / 元ファイル `path` / サンプル時間幅（`sample_time_span_us/sec`）/ 保存画像パスを出力します。
- この可視化スクリプトは DataLoader を使わず dataset を直接読むため、原因切り分け時に安定です。

## Collapse Strategy 切り替え

- `collapse_strategy=ema_stopgrad`: EMA teacher + stop-grad（デフォルト）
- `collapse_strategy=vicreg`: VICReg 系正則化
- `collapse_strategy=sigreg`: SIGReg 正則化

```bash
source env/bin/activate
python scripts/train_jepa_pretrain.py collapse_strategy=ema_stopgrad steps=200
python scripts/train_jepa_pretrain.py collapse_strategy=vicreg steps=200 vicreg_std_weight=0.1 vicreg_cov_weight=0.01
python scripts/train_jepa_pretrain.py collapse_strategy=sigreg steps=200 sigreg_weight=0.01
```

## Ablation 実行コマンド集

公平比較のため、まず以下を固定するのがおすすめです。

- `seed`
- `steps`
- `batch_size`
- `lr`, `weight_decay`, scheduler 設定
- `height/width/t_bins/patch_size/tubelet_size`
- 下流評価プロトコル（同じ checkpoint 選択ルール）

### 共通ベース（pretrain_mixed）

```bash
source env/bin/activate

BASE_ARGS="\
data.source=pretrain_mixed \
eval.enabled=false \
steps=50000 \
seed=0 \
batch_size=8 \
t_bins=10 \
temporal_mix.enabled=true \
temporal_mix.short_t=1 \
temporal_mix.image_prob=0.5 \
temporal_mix.short_mode=sum"
```

### 1. 学習手法（JEPA / MEM / MAE）

JEPA（このリポジトリ本体）

```bash
python scripts/train_jepa_pretrain.py ${BASE_ARGS} \
  collapse_strategy=ema_stopgrad \
  hydra.run.dir=outputs/ablations/method_jepa_seed0
```

MEM / MAE（`tmp/mem` 実装）

このリポジトリの `scripts/train_jepa_pretrain.py` は JEPA 系のみ実装です。  
`MEM/MAE` は `tmp/mem` の実装を使って比較してください。

```bash
cd tmp/mem/mem

# MEM: MAEフラグを0、VAE checkpointを指定
torchrun --standalone --nproc_per_node=1 run_mem_pretraining.py \
  --config ../configs/nimagenet.conf \
  --expweek 2026-04 \
  --expname mem_seed0 \
  --model pt_vit \
  --MAE 0 \
  --data_path /absolute/path/to/processed_dataset \
  --discrete_vae_weight_path /absolute/path/to/vae_checkpoint.pth \
  --output_dir ../../../outputs/ablations/method_mem_seed0 \
  --log_dir ../../../outputs/ablations/method_mem_seed0/tb

# MAE: MAEフラグを1（離散VAEは不要）
torchrun --standalone --nproc_per_node=1 run_mem_pretraining.py \
  --config ../configs/nimagenet.conf \
  --expweek 2026-04 \
  --expname mae_seed0 \
  --MAE 1 \
  --data_path /absolute/path/to/processed_dataset \
  --output_dir ../../../outputs/ablations/method_mae_seed0 \
  --log_dir ../../../outputs/ablations/method_mae_seed0/tb
```

### 2. 勾配更新方式（JEPA 内）

```bash
python scripts/train_jepa_pretrain.py ${BASE_ARGS} \
  collapse_strategy=ema_stopgrad \
  hydra.run.dir=outputs/ablations/collapse_ema_stopgrad_seed0

python scripts/train_jepa_pretrain.py ${BASE_ARGS} \
  collapse_strategy=vicreg \
  vicreg_std_weight=0.1 vicreg_cov_weight=0.01 \
  hydra.run.dir=outputs/ablations/collapse_vicreg_seed0

python scripts/train_jepa_pretrain.py ${BASE_ARGS} \
  collapse_strategy=sigreg \
  sigreg_weight=0.01 \
  hydra.run.dir=outputs/ablations/collapse_sigreg_seed0
```

### 3. 前処理（true / false）

```bash
# hflip ON / OFF
python scripts/train_jepa_pretrain.py ${BASE_ARGS} \
  data.pretrain_mixed.augment.enabled=true \
  data.pretrain_mixed.augment.hflip_prob=0.5 \
  hydra.run.dir=outputs/ablations/aug_hflip_on_seed0

python scripts/train_jepa_pretrain.py ${BASE_ARGS} \
  data.pretrain_mixed.augment.enabled=true \
  data.pretrain_mixed.augment.hflip_prob=0.0 \
  hydra.run.dir=outputs/ablations/aug_hflip_off_seed0

# time flip ON / OFF
python scripts/train_jepa_pretrain.py ${BASE_ARGS} \
  data.pretrain_mixed.augment.enabled=true \
  data.pretrain_mixed.augment.time_flip_prob=0.2 \
  hydra.run.dir=outputs/ablations/aug_timeflip_on_seed0

python scripts/train_jepa_pretrain.py ${BASE_ARGS} \
  data.pretrain_mixed.augment.enabled=true \
  data.pretrain_mixed.augment.time_flip_prob=0.0 \
  hydra.run.dir=outputs/ablations/aug_timeflip_off_seed0

# polarity flip ON / OFF
python scripts/train_jepa_pretrain.py ${BASE_ARGS} \
  data.pretrain_mixed.augment.enabled=true \
  data.pretrain_mixed.augment.polarity_flip_prob=0.2 \
  hydra.run.dir=outputs/ablations/aug_polflip_on_seed0

python scripts/train_jepa_pretrain.py ${BASE_ARGS} \
  data.pretrain_mixed.augment.enabled=true \
  data.pretrain_mixed.augment.polarity_flip_prob=0.0 \
  hydra.run.dir=outputs/ablations/aug_polflip_off_seed0

# random resized crop ON / OFF（aspect固定）
python scripts/train_jepa_pretrain.py ${BASE_ARGS} \
  data.pretrain_mixed.augment.enabled=true \
  data.pretrain_mixed.augment.random_resized_crop.enabled=true \
  data.pretrain_mixed.augment.random_resized_crop.prob=0.5 \
  data.pretrain_mixed.augment.random_resized_crop.scale_min=0.8 \
  data.pretrain_mixed.augment.random_resized_crop.scale_max=1.0 \
  data.pretrain_mixed.augment.random_resized_crop.preserve_aspect=true \
  hydra.run.dir=outputs/ablations/aug_rrc_on_seed0

python scripts/train_jepa_pretrain.py ${BASE_ARGS} \
  data.pretrain_mixed.augment.enabled=true \
  data.pretrain_mixed.augment.random_resized_crop.enabled=false \
  hydra.run.dir=outputs/ablations/aug_rrc_off_seed0
```

### 4. 蓄積イベント数（events_per_sample）sweep

```bash
for N in 20000 40000 80000; do
  python scripts/train_jepa_pretrain.py ${BASE_ARGS} \
    data.pretrain_mixed.events_per_sample_min=${N} \
    data.pretrain_mixed.events_per_sample_max=${N} \
    hydra.run.dir=outputs/ablations/events_${N}_seed0
done
```

## Scheduler（Warmup + Cosine）

```bash
source env/bin/activate
python scripts/train_jepa_pretrain.py \
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
torchrun --standalone --nproc_per_node=2 scripts/train_jepa_pretrain.py \
  distributed.enabled=true \
  data.source=n_imagenet \
  data.n_imagenet.train_list=/absolute/path/to/train_list.txt
```

## 補足

- 学習データは `data.source` で切り替えます（`pretrain_mixed` / `synthetic` / `n_imagenet` / `dsec`）。
- `n_imagenet` では list file（1行1サンプルの npz パス）を読み込みます。
- `dsec` は DSEC-Detection 構造（`images/events/object_detections`）を読み込みます。
- `dsec` では `data.dsec.load_events/load_rgb/load_labels` でロード対象を切り替えできます。
- `pretrain_mixed` は downsample 後 H5 を混合する「事前学習専用」ローダです（`eval.enabled=false`）。
- 学習の入力解像度は `320x240` 前提です。`320x180`（Gen4）はローダ内で中央パディングして統一します。
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
