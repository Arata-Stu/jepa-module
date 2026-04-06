# Step1 Prototype (Masked Prediction)

`scripts/train_step1_pretrain.py` は、イベントデータ向け JEPA 事前学習の `step1` を単体で試すための最小プロトタイプです。

- 入力: ランダムイベントから作る voxel grid（実データローダの代替）
- 目的: 同一時刻内でマスク領域のトークン特徴を予測
- モデル: `src/jepa` の ViT encoder + predictor

## 実行例

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

- 実データを使う場合は `sample_voxel_batch` を置き換えるだけで学習ループ本体は再利用できます。
- `step2`（時刻シフト + フロー利用）を入れる前段として、まず `step1` の loss が安定して下がるかを確認する用途を想定しています。
