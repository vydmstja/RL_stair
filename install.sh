#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cyclo_dir="${1:-/workspace/cyclo_lab}"

if [[ ! -d "${cyclo_dir}/source/cyclo_lab/cyclo_lab" ]]; then
  echo "Cyclo Lab checkout not found: ${cyclo_dir}" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
backup_dir="${cyclo_dir}/.rl_stair_backup/${timestamp}"
mkdir -p "${backup_dir}"

while IFS= read -r -d '' source_file; do
  relative_path="${source_file#${repo_dir}/overlay/}"
  target_file="${cyclo_dir}/${relative_path}"
  if [[ -f "${target_file}" ]]; then
    mkdir -p "${backup_dir}/$(dirname "${relative_path}")"
    cp -a "${target_file}" "${backup_dir}/${relative_path}"
  fi
  mkdir -p "$(dirname "${target_file}")"
  cp -a "${source_file}" "${target_file}"
done < <(find "${repo_dir}/overlay" -type f -print0)

echo "RL_stair overlay installed in: ${cyclo_dir}"
echo "Backup of replaced files: ${backup_dir}"
echo "Training task: Cyclo-Velocity-Stairs-K1-Rev1-v0"
