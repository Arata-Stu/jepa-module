# Step1 Prototype (Masked Prediction)

`scripts/train_step1_pretrain.py` は、イベントデータ向け JEPA 事前学習の `step1` を単体で試すための最小プロトタイプです。

- 入力: ランダムイベントから作る voxel grid（実データローダの代替）
- 目的: 同一時刻内でマスク領域のトークン特徴を予測
- モデル: `src/jepa` の ViT encoder + predictor

## 実行例

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

- 実データを使う場合は `sample_voxel_batch` を置き換えるだけで学習ループ本体は再利用できます。
- `step2`（時刻シフト + フロー利用）を入れる前段として、まず `step1` の loss が安定して下がるかを確認する用途を想定しています。
- 設定は `configs/train_step1.yaml` で管理し、CLIは `key=value` で上書きします。
- 崩壊抑制の切替は `collapse_strategy=ema_stopgrad|vicreg|sigreg` です。
