#!/usr/bin/env bash
set -euo pipefail

# Multi-duration attention rollout evaluation for three JEPA checkpoints.
# - Evaluates each checkpoint on the same fixed scene set (manifest)
# - Sweeps accumulation window from 1ms to 200ms
# - Separates temporal_mix.enabled=false/true per checkpoint

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# Data/source settings
SOURCE_NAME="${SOURCE_NAME:-gen4}"               # gen4 or dsec
SOURCE_ROOT_DIR="${SOURCE_ROOT_DIR:-/mnt/data/arata/gen4-downsampled}"
SOURCE_SPLIT="${SOURCE_SPLIT:-}"
SOURCE_SPLITS="${SOURCE_SPLITS:-${SOURCE_SPLIT:-train}}" # e.g. "train" or "train,val"
SOURCE_FILE_GLOB="${SOURCE_FILE_GLOB:-*_4x.h5}" # for gen4 default

# Scene-set settings
NUM_SCENES="${NUM_SCENES:-32}"
MANIFEST_PATH="${MANIFEST_PATH:-/tmp/attention_eval_${SOURCE_NAME}_${NUM_SCENES}.txt}"
REBUILD_MANIFEST="${REBUILD_MANIFEST:-auto}"     # auto|1|0

# Visualization settings
NUM_SAMPLES="${NUM_SAMPLES:-32}"
SEED="${SEED:-42}"
OUT_BASE="${OUT_BASE:-outputs/rollout_duration_sweep}"
DURATIONS_MS=(${DURATIONS_MS:-1 2 5 10 20 50 100 150 200})

# Checkpoints
CKPT_200MS="${CKPT_200MS:-/home/apollo-22/Arata_ws/jepa-module/outputs/train/2026-04-26/200ms/checkpoints/step_680000.pt}"
CKPT_20MS="${CKPT_20MS:-/home/apollo-22/Arata_ws/jepa-module/outputs/train/2026-05-01/20ms/checkpoints/step_880000.pt}"
CKPT_VARIABLE="${CKPT_VARIABLE:-/home/apollo-22/Arata_ws/jepa-module/outputs/train/dsec-gen4-pretrain/02-44-35/checkpoints/step_830000.pt}"

MODELS=(
  "fixed200ms|${CKPT_200MS}|false"
  "fixed20ms|${CKPT_20MS}|false"
  "variable|${CKPT_VARIABLE}|true"
)

manifest_exists="0"
if [[ -f "${MANIFEST_PATH}" ]]; then
  manifest_exists="1"
fi

if [[ -z "${SOURCE_SPLITS}" ]]; then
  echo "[ERROR] SOURCE_SPLITS must be non-empty (e.g. SOURCE_SPLITS=train or SOURCE_SPLITS=train,val)" >&2
  exit 1
fi

should_rebuild="0"
if [[ "${REBUILD_MANIFEST}" == "1" ]]; then
  should_rebuild="1"
elif [[ "${REBUILD_MANIFEST}" == "auto" && "${manifest_exists}" == "0" ]]; then
  should_rebuild="1"
fi

if [[ "${should_rebuild}" == "1" ]]; then
  IFS=',' read -r -a split_list <<< "${SOURCE_SPLITS}"
  tmp_all="$(mktemp /tmp/attention_eval_all.XXXXXX)"
  : > "${tmp_all}"
  valid_split_count=0

  for split in "${split_list[@]}"; do
    split_trimmed="$(echo "${split}" | xargs)"
    [[ -z "${split_trimmed}" ]] && continue
    split_dir="${SOURCE_ROOT_DIR%/}/${split_trimmed}"
    if [[ ! -d "${split_dir}" ]]; then
      echo "[WARN] split directory not found, skip: ${split_dir}" >&2
      continue
    fi
    valid_split_count=$((valid_split_count + 1))
    echo "[INFO] rebuilding manifest from ${split_dir} (glob=${SOURCE_FILE_GLOB})"
    find "${split_dir}" -type f -name "${SOURCE_FILE_GLOB}" >> "${tmp_all}"
  done

  if [[ "${valid_split_count}" -lt 1 ]]; then
    rm -f "${tmp_all}"
    echo "[ERROR] no valid split directories under root=${SOURCE_ROOT_DIR} for SOURCE_SPLITS=${SOURCE_SPLITS}" >&2
    echo "[HINT] set SOURCE_ROOT_DIR/SOURCE_SPLITS correctly, or set REBUILD_MANIFEST=0 with MANIFEST_PATH=<existing_manifest>" >&2
    exit 1
  fi

  tmp_sorted="$(mktemp /tmp/attention_eval_sorted.XXXXXX)"
  sort -u "${tmp_all}" > "${tmp_sorted}"
  sed -n "1,${NUM_SCENES}p" "${tmp_sorted}" > "${MANIFEST_PATH}"
  rm -f "${tmp_all}" "${tmp_sorted}"
fi

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "[ERROR] manifest not found: ${MANIFEST_PATH}" >&2
  exit 1
fi

scene_count="$(grep -vc '^[[:space:]]*$' "${MANIFEST_PATH}" || true)"
if [[ "${scene_count}" -lt 1 ]]; then
  echo "[ERROR] manifest has no entries: ${MANIFEST_PATH}" >&2
  exit 1
fi

echo "[INFO] source=${SOURCE_NAME} root=${SOURCE_ROOT_DIR} splits=${SOURCE_SPLITS}"
echo "[INFO] manifest=${MANIFEST_PATH} scenes=${scene_count}"
echo "[INFO] durations_ms=${DURATIONS_MS[*]}"
echo "[INFO] out_base=${OUT_BASE}"

splits_override=("data.pretrain_mixed.${SOURCE_NAME}.splits=[${SOURCE_SPLITS}]")

for model in "${MODELS[@]}"; do
  IFS="|" read -r model_name ckpt tmix <<< "${model}"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[WARN] checkpoint not found, skip: ${ckpt}"
    continue
  fi

  for ms in "${DURATIONS_MS[@]}"; do
    us=$((ms * 1000))
    run_dir="${OUT_BASE}/${model_name}/${ms}ms"
    echo "[RUN] model=${model_name} temporal_mix=${tmix} duration=${ms}ms ckpt=${ckpt}"

    "${PYTHON_BIN}" scripts/visualize_attention_rollout.py \
      pretrained.checkpoint="${ckpt}" \
      temporal_mix.enabled="${tmix}" \
      data.source=pretrain_mixed \
      data.pretrain_mixed.${SOURCE_NAME}.enabled=true \
      data.pretrain_mixed.${SOURCE_NAME}.root_dir="${SOURCE_ROOT_DIR}" \
      data.pretrain_mixed.${SOURCE_NAME}.manifest_file="${MANIFEST_PATH}" \
      "${splits_override[@]}" \
      data.pretrain_mixed.weights.gen4=0.0 \
      data.pretrain_mixed.weights.dsec=0.0 \
      data.pretrain_mixed.weights.n_imagenet=0.0 \
      data.pretrain_mixed.weights.${SOURCE_NAME}=1.0 \
      data.pretrain_mixed.random_slice=false \
      data.pretrain_mixed.augment.enabled=false \
      data.pretrain_mixed.window_duration_us_min="${us}" \
      data.pretrain_mixed.window_duration_us_max="${us}" \
      data.pretrain_mixed.duration_sources="[${SOURCE_NAME}]" \
      visualize.num_samples="${NUM_SAMPLES}" \
      visualize.weighted_sampling=false \
      seed="${SEED}" \
      hydra.run.dir="${run_dir}"
  done
done

echo "[DONE] duration sweep completed"
