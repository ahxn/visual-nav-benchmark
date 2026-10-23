#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

venv_dir="${VENV_DIR:-.venv}"
python_bin="$venv_dir/bin/python"
episode_count="${EPISODE_COUNT:-5}"
start_seed="${START_SEED:-11}"
mode="${1:-synthetic}"

if [[ ! -x "$python_bin" ]]; then
    echo "missing $python_bin; run scripts/setup_cpu_cluster.sh first" >&2
    exit 1
fi

case "$mode" in
    test)
        "$python_bin" -m pytest -q
        ;;
    synthetic)
        "$python_bin" -m visual_nav \
            --episodes "$episode_count" \
            --seed "$start_seed" \
            --methods oracle_depth noisy_depth \
            --output results/cluster_synthetic
        ;;
    thor-classical)
        "$python_bin" -m visual_nav.thor_cli \
            --headless \
            --scenes FloorPlan2 FloorPlan202 FloorPlan302 FloorPlan402 \
            --episodes "$episode_count" \
            --seed "$start_seed" \
            --methods simulator_depth classical_stereo classical_sparse_features \
            --output results/cluster_thor_classical.json
        ;;
    thor-all)
        "$python_bin" -m visual_nav.thor_cli \
            --headless \
            --device cpu \
            --scenes FloorPlan2 FloorPlan202 FloorPlan302 FloorPlan402 \
            --episodes "$episode_count" \
            --seed "$start_seed" \
            --methods simulator_depth classical_stereo classical_sparse_features learned_metric_depth \
            --output results/cluster_thor_all_cpu.json
        ;;
    *)
        echo "usage: $0 {test|synthetic|thor-classical|thor-all}" >&2
        exit 2
        ;;
esac
