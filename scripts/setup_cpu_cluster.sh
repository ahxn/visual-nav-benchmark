#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python3}"
venv_dir="${VENV_DIR:-.venv}"
install_thor=false
install_deep=false

for option in "$@"; do
    case "$option" in
        --with-ai2thor)
            install_thor=true
            ;;
        --with-deep)
            install_thor=true
            install_deep=true
            ;;
        *)
            echo "usage: $0 [--with-ai2thor] [--with-deep]" >&2
            exit 2
            ;;
    esac
done

"$python_bin" -m venv "$venv_dir"
venv_python="$venv_dir/bin/python"
"$venv_python" -m pip install --upgrade pip
"$venv_python" -m pip install -e ".[vision,dev]"

if [[ "$install_thor" == true ]]; then
    "$venv_python" -m pip install -e ".[ai2thor]"
fi

if [[ "$install_deep" == true ]]; then
    # Install the CPU wheel first so the project never pulls a CUDA runtime.
    "$venv_python" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    "$venv_python" -m pip install transformers accelerate
fi

"$venv_python" -m pytest -q
echo "CPU environment ready at $venv_dir"
