#!/usr/bin/env bash
#
# One-shot setup on a new server:
#   1. clone ULD + open-unlearning at the exact upstream commits we tested
#   2. overlay our 2 ULD configs + 5 open-unlearning files on top
#
# Idempotent — safe to re-run; existing repos are left alone.
#
# After setup completes, run:   bash run.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ULD_COMMIT="858608c"          # UCSB-NLP-Chang/ULD@main release commit
OU_COMMIT="4ad738a"           # locuslab/open-unlearning@main

clone_pinned() {
    local url="$1" dir="$2" commit="$3"
    if [ -d "$dir/.git" ]; then
        echo "  → $dir already exists; leaving as-is"
        return
    fi
    echo "  → clone $url → $dir @ $commit"
    git clone "$url" "$dir"
    git -C "$dir" checkout "$commit"
}

echo "[1/2] Cloning upstream repos at pinned commits"
# Use SSH by default — some servers firewall HTTPS (port 443) but allow SSH
# (port 22). If you don't have a GitHub SSH key set up, override:
#   USE_HTTPS=1 bash setup.sh
if [ "${USE_HTTPS:-0}" = "1" ]; then
    ULD_URL="https://github.com/UCSB-NLP-Chang/ULD.git"
    OU_URL="https://github.com/locuslab/open-unlearning.git"
else
    ULD_URL="git@github.com:UCSB-NLP-Chang/ULD.git"
    OU_URL="git@github.com:locuslab/open-unlearning.git"
fi
clone_pinned "$ULD_URL"  ULD              "$ULD_COMMIT"
clone_pinned "$OU_URL"   open-unlearning  "$OU_COMMIT"

echo "[2/2] Overlaying our changes"
# rsync would be cleaner; cp -a works portably and is enough here.
cp -a "$ROOT/overlay/ULD/."             "$ROOT/ULD/"
cp -a "$ROOT/overlay/open-unlearning/." "$ROOT/open-unlearning/"

echo
echo "Setup done. Layout:"
ls -d "$ROOT/ULD" "$ROOT/open-unlearning" "$ROOT/run.sh" 2>/dev/null
echo
echo "Next:"
echo "  huggingface-cli login            # Llama-3.1 / open-unlearning models need auth"
echo "  python -m pip install --user tf-keras lm-eval hydra-colorlog peft"
echo "  bash run.sh                      # full pipeline (Phase A+B+C, all splits)"
echo "  SKIP_PHASE_A=1 ONLY=forget01 bash run.sh   # quick smoke test"
