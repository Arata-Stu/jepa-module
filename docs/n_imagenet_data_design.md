# N-ImageNet Data Loader Design

## 目的

- Step1 事前学習で `data.source=n_imagenet` を指定した際に、イベント列から voxel grid を生成して学習へ供給する。
- データ処理を `src/event/data/` に集約し、学習ループ本体と分離する。

## 実装配置

- `src/event/data/n_imagenet.py`
  - `NImageNetEventsDataset`: list file を読み、`npz` または `h5` からイベント列 `(x, y, pol, time)` を返す。
  - `NImageNetVoxelCollator`: バッチ化時に voxel grid へ変換する。
- `src/event/data/providers.py`
  - `NImageNetVoxelBatchProvider`: `DataLoader` をラップして `next_batch()` を提供。
  - `DistributedSampler` 対応（`distributed.enabled=true` のとき rank ごとに shard）。
  - `SyntheticVoxelBatchProvider`: 既存の synthetic 供給器。
- `scripts/train_step1_pretrain.py`
  - `build_batch_provider(...)` で `data.source` に応じて provider を切り替え。

## データフロー

1. `NImageNetEventsDataset` が list file を 1 行ずつ読み込み、サンプルパスを決定。
2. `__getitem__` で npz を読み込み、必要に応じて以下を実施。
   - 時間/インデックススライス（`data.n_imagenet.slice.*`）
   - 学習時拡張（左右反転、座標シフト）
   - 画角外イベントの除去
3. `NImageNetVoxelCollator` がサンプルごとに `VoxelGrid.convert(...)` を適用し、`inputs` を `(B, 1, T, H, W)` で返す。
4. 学習ループは `batch_provider.next_batch()["inputs"]` のみを参照して forward する。

## list file 仕様

- 1 行につき 1 サンプル（npz のパス）
- 例:

```text
/data/N_Imagenet/class_a/sample_0001.npz
/data/N_Imagenet/class_b/sample_0107.npz
```

- 相対パスの場合は `data.n_imagenet.root_dir` を先頭に付与して解決する。

## 主要設定（Hydra）

- `data.source`: `synthetic` or `n_imagenet`
- `data.n_imagenet.train_list`, `data.n_imagenet.val_list`
- `data.n_imagenet.compressed`: 圧縮イベント形式の読み分け
- `data.n_imagenet.sensor_height`, `data.n_imagenet.sensor_width`
- `data.n_imagenet.rescale_to_voxel_grid`
- `data.n_imagenet.slice.*`, `data.n_imagenet.augment.*`

## 今後の拡張点

- GEN4 / DSEC 用 dataset を `src/event/data/` に同様のインターフェースで追加
- Step2（時刻シフト予測）用に、同一サンプルから複数時刻窓を返す dataset へ拡張
