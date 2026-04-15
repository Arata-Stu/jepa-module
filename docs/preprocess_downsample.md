# Preprocess Downsample Guide

`scripts/downsample_1mpx.py` と `scripts/downsample_dsec.py` は、イベント H5 を空間ダウンサンプルする前処理スクリプトです。

- 共通ロジック:
  - 極性 `p` は処理時に `-1/+1` へ変換して積算し、保存時に `0/1` へ戻します。
  - 出力は一時ファイル（`*.tmp`）に書き、成功時のみ最終名へ `rename` します。
  - 途中失敗後の再実行時は stale tmp を削除して再開します。
- 並列実行:
  - ルート一括モードでは `--num_processes N` で `spawn` プロセス並列が使えます。

## 1MPX: `scripts/downsample_1mpx.py`

### 単一ファイル処理

```bash
python scripts/downsample_1mpx.py \
  --input_path /path/to/input.h5 \
  --output_path /path/to/output_1mpx.h5 \
  --output_height 360 \
  --output_width 640
```

### データセット一括処理（train/test/val）

`dataset_root/<split>/` 配下の `.h5` を探索して処理します。

```bash
python scripts/downsample_1mpx.py \
  --dataset_root /path/to/dataset_root \
  --splits train test val \
  --output_suffix _1mpx.h5 \
  --num_processes 4
```

主なオプション:

- `--output_root /path/to/out`: 出力先を別ディレクトリへ変更（相対構造を維持）
- `--recursive`: split 配下を再帰探索
- `--overwrite`: 既存出力を上書き
- `--tmp_suffix .tmp`: 一時ファイルの suffix
- `--num_events_per_chunk 100000`: チャンクサイズ調整（`--chunk_size` は互換エイリアス）
- `--write_ms_to_idx`: 出力 H5 に `ms_to_idx` を書く

## DSEC: `scripts/downsample_dsec.py`

### 単一ファイル処理

```bash
python scripts/downsample_dsec.py \
  --input_path /path/to/events.h5 \
  --output_path /path/to/events_2x.h5
```

### DSEC ルート一括処理（複数シーケンス）

`<dsec_root>/<split>/<sequence>/events/left/events.h5` を探索して処理します。

```bash
python scripts/downsample_dsec.py \
  --dataset_root /path/to/DSEC_ROOT \
  --splits train test \
  --output_name events_2x.h5 \
  --num_processes 4
```

主なオプション:

- `--output_root /path/to/out`: 出力先を別ディレクトリへ変更（相対構造を維持）
- `--overwrite`: 既存出力を上書き
- `--tmp_suffix .tmp`: 一時ファイルの suffix
- `--num_events_per_chunk 100000`: チャンクサイズ調整
- `--dsec_root`: `--dataset_root` の互換エイリアス

## 既存出力と再実行

- 既定では、最終出力ファイルが既に存在する場合はスキップします。
- `--overwrite` を付けると再生成します。
- 中断後に tmp が残っている場合は、次回実行時に削除してから再処理します。

## 学習ローダとの連携（DSEC）

`scripts/train_step1_pretrain.py` の DSEC 設定で、読み込むイベントファイルを切り替え可能です。

```yaml
data:
  dsec:
    downsample: true
    downsample_event_file: events_2x.h5
```

- `downsample: false` のとき `events.h5`
- `downsample: true` のとき `downsample_event_file`（既定: `events_2x.h5`）
