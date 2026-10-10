#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
assets_dir="${R3M_ASSETS:-/mnt/khanh/work/r3m}"
export PYTHONPATH="${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export NO_ALBUMENTATIONS_UPDATE=1
exec "${R3M_PYTHON:-${assets_dir}/.venv/bin/python}" -m lerobot.r3m_groot.rollout \
    --assets "${assets_dir}" "$@"
