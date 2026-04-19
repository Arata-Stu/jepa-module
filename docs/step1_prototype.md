# Step1 Prototype (Masked Prediction)

`scripts/train_jepa_pretrain.py` は、イベントデータ向け JEPA 事前学習の `step1` を単体で試すための最小プロトタイプです。

- 入力: voxel grid（`synthetic` / `n_imagenet` / `dsec`）
- 目的: 同一時刻内でマスク領域のトークン特徴を予測
- モデル: `src/jepa` の ViT encoder + predictor

## 実行例

### synthetic

```bash
source env/bin/activate
python scripts/train_jepa_pretrain.py \
  data.source=synthetic \
  model_size=tiny \
  steps=200 \
  batch_size=8 \
  height=128 width=128 \
  t_bins=8 patch_size=16 tubelet_size=2 \
  normalize_voxel=true normalize_targets=true
```

### n_imagenet

```bash
source env/bin/activate
python scripts/train_jepa_pretrain.py \
  data.source=n_imagenet \
  data.n_imagenet.split=train \
  data.n_imagenet.train_list=/absolute/path/to/train_list.txt \
  data.n_imagenet.root_dir=/absolute/path/to/N_Imagenet \
  data.n_imagenet.compressed=true \
  batch_size=8
```

### dsec

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
  batch_size=8
```

## 補足

- データセット処理は `src/event/data/` に分離してあります。
- `NImageNetEventsDataset` がイベント読み出し、`NImageNetVoxelCollator` が voxel 化を担当します。
- `DSECEventsDataset` / `DSECVoxelCollator` を追加し、`data.dsec.load_events/load_rgb/load_labels` でロード対象を切り替えできます。
- `step1` では同時刻マスク予測までを対象とし、`step2`（時刻シフト + フロー利用）は未実装です。
- `step2`（時刻シフト + フロー利用）を入れる前段として、まず `step1` の loss が安定して下がるかを確認する用途を想定しています。
- 設定は `configs/train_jepa.yaml` で管理し、CLIは `key=value` で上書きします。
- 崩壊抑制の切替は `collapse_strategy=ema_stopgrad|vicreg|sigreg` です。
- 学習率/Weight Decay スケジューラは `scheduler.enabled=true` で Warmup + Cosine を有効化できます。
- 分散学習は `distributed.enabled=true` で有効化し、`torchrun` 経由で起動します。
- 出力は `outputs/train/YYYY-MM-DD/HH-MM-SS/` にまとまり、`checkpoints/` と `logs/train_metrics.csv` を同一 timestamp で追跡できます。
- TensorBoard は `logging.tensorboard.enabled=true` で有効になり、`logs/tensorboard/` にイベントが保存されます。
- 評価ログは `eval.enabled=true` のときに `logs/eval_metrics.csv` と `eval/*` スカラーへ記録されます。
