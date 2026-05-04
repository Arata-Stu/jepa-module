#!/usr/bin/env bash
set -euo pipefail

# Compose side-by-side comparison images/videos from rollout sweep outputs.
# Expected input layout (from run_attention_rollout_duration_sweep.sh):
#   <OUT_BASE>/<MODEL>/<DURATION>/attention_rollout/*.png
#
# Example:
#   bash scripts/compose_rollout_comparison.sh outputs/rollout_duration_sweep

OUT_BASE="${1:-outputs/rollout_duration_sweep}"

MODEL_A="${MODEL_A:-fixed200ms}"
MODEL_B="${MODEL_B:-fixed20ms}"
MODEL_C="${MODEL_C:-variable}"

# Optional display labels (used only when ADD_LABELS=1)
LABEL_A="${LABEL_A:-fixed200ms}"
LABEL_B="${LABEL_B:-fixed20ms}"
LABEL_C="${LABEL_C:-variable}"

OUT_SUBDIR="${OUT_SUBDIR:-_compare}"
FPS="${FPS:-4}"
ADD_LABELS="${ADD_LABELS:-0}" # 0|1

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[ERROR] ffmpeg not found in PATH" >&2
  exit 1
fi

if [[ ! -d "${OUT_BASE}" ]]; then
  echo "[ERROR] OUT_BASE not found: ${OUT_BASE}" >&2
  exit 1
fi

model_a_root="${OUT_BASE%/}/${MODEL_A}"
model_b_root="${OUT_BASE%/}/${MODEL_B}"
model_c_root="${OUT_BASE%/}/${MODEL_C}"

if [[ ! -d "${model_a_root}" || ! -d "${model_b_root}" || ! -d "${model_c_root}" ]]; then
  echo "[ERROR] model directory missing under OUT_BASE=${OUT_BASE}" >&2
  echo "        required: ${MODEL_A}, ${MODEL_B}, ${MODEL_C}" >&2
  exit 1
fi

tmp_dir="$(mktemp -d /tmp/rollout_compare.XXXXXX)"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

durations=()
while IFS= read -r d; do
  durations+=("${d}")
done < <(find "${model_a_root}" -mindepth 1 -maxdepth 1 -type d -name '*ms' | sort)

if [[ "${#durations[@]}" -eq 0 ]]; then
  echo "[ERROR] no duration directories found under ${model_a_root}" >&2
  exit 1
fi

for dpath in "${durations[@]}"; do
  duration="$(basename "${dpath}")"
  dir_a="${model_a_root}/${duration}/attention_rollout"
  dir_b="${model_b_root}/${duration}/attention_rollout"
  dir_c="${model_c_root}/${duration}/attention_rollout"

  if [[ ! -d "${dir_a}" || ! -d "${dir_b}" || ! -d "${dir_c}" ]]; then
    echo "[WARN] skip duration=${duration} (missing one of attention_rollout dirs)"
    continue
  fi

  out_root="${OUT_BASE%/}/${OUT_SUBDIR}/${duration}"
  out_frames="${out_root}/frames"
  mkdir -p "${out_frames}"

  files_a=()
  files_b=()
  files_c=()
  while IFS= read -r f; do files_a+=("${f}"); done < <(find "${dir_a}" -maxdepth 1 -type f -name '*.png' | sort)
  while IFS= read -r f; do files_b+=("${f}"); done < <(find "${dir_b}" -maxdepth 1 -type f -name '*.png' | sort)
  while IFS= read -r f; do files_c+=("${f}"); done < <(find "${dir_c}" -maxdepth 1 -type f -name '*.png' | sort)

  n_a="${#files_a[@]}"
  n_b="${#files_b[@]}"
  n_c="${#files_c[@]}"
  n_common="${n_a}"
  if [[ "${n_b}" -lt "${n_common}" ]]; then n_common="${n_b}"; fi
  if [[ "${n_c}" -lt "${n_common}" ]]; then n_common="${n_c}"; fi

  if [[ "${n_common}" -lt 1 ]]; then
    echo "[WARN] no frames for duration=${duration} (a=${n_a}, b=${n_b}, c=${n_c})"
    continue
  fi

  echo "[INFO] duration=${duration} paired_frames=${n_common} (a=${n_a}, b=${n_b}, c=${n_c})"

  for ((i=0; i<n_common; i++)); do
    in_a="${files_a[$i]}"
    in_b="${files_b[$i]}"
    in_c="${files_c[$i]}"
    base_name="$(basename "${in_a}")"
    out_img="${out_frames}/${base_name}"

    if [[ "${ADD_LABELS}" == "1" ]]; then
      ffmpeg -hide_banner -loglevel error -y \
        -i "${in_a}" -i "${in_b}" -i "${in_c}" \
        -filter_complex "\
[0:v]drawtext=text='${LABEL_A}':x=16:y=16:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.55[v0];\
[1:v]drawtext=text='${LABEL_B}':x=16:y=16:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.55[v1];\
[2:v]drawtext=text='${LABEL_C}':x=16:y=16:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.55[v2];\
[v0][v1][v2]hstack=inputs=3[out]" \
        -map "[out]" -frames:v 1 "${out_img}"
    else
      ffmpeg -hide_banner -loglevel error -y \
        -i "${in_a}" -i "${in_b}" -i "${in_c}" \
        -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[out]" \
        -map "[out]" -frames:v 1 "${out_img}"
    fi
  done

  out_video="${out_root}/comparison_${duration}.mp4"
  ffmpeg -hide_banner -loglevel error -y \
    -framerate "${FPS}" \
    -pattern_type glob -i "${out_frames}/*.png" \
    -c:v libx264 -pix_fmt yuv420p "${out_video}"

  echo "[DONE] duration=${duration} video=${out_video}"
done

echo "[DONE] all available durations processed"
