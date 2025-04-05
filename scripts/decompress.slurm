#!/bin/bash
#SBATCH --job-name=decompress-sae
#SBATCH --time=24:00:00
#SBATCH --partition=highgpu
#SBATCH --cpus-per-task=128
#SBATCH --output=slurm/decompress-%j.out
#SBATCH --error=slurm/decompress-%j.out

set -euo pipefail

A="$HOME/git/activation-cache-exporter"
B="$HOME/git/train-saes"

decompress_file() {
  src_file="$1"
  split="$2"
  base_name=$(basename "$src_file" .gz)
  mkdir -p "$B/data/$split"
  echo "Decompressing: $src_file -> $B/data/$split/$base_name"
  gunzip -c "$src_file" > "$B/data/$split/$base_name"
}

export A B
export -f decompress_file

for split in train test val; do
  mkdir -p "$B/data/$split"
  find "$A/data/$split" -name '*.parquet.gz' \
    | xargs -n 1 -P "$(nproc)" -I {} bash -c 'decompress_file "$0" "'"$split"'"' {}
done

scancel "$SLURM_JOB_ID"