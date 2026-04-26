#!/usr/bin/env bash
#
# One-click ULD on Llama-3.1-8B-Instruct, evaluated with the open-unlearning
# TOFU benchmark. Runs all three forget splits.
#
# Phases (idempotent — each step skips if its output already exists):
#   A. Eval the open-unlearning retain-model checkpoints to produce the
#      retain-side TOFU_EVAL.json that the unlearned-model eval needs as
#      reference (forget_quality KS test).
#   B. Train a ULD assistant for each forget split, against the
#      open-unlearning TOFU full-finetune (skips ULD's own Phase 0/1).
#   C. Eval each ULD model (base + assistant) using the open-unlearning
#      TOFU evaluator and write per-split TOFU_EVAL.json.
#
# Prereqs:
#   - huggingface-cli login (Llama-3.1 weights are gated)
#   - System Python 3.10 with: torch, transformers, peft, hydra-core, deepspeed,
#     pytorch_lightning, datasets (already present in /usr/bin/python).
#   - flash-attn is NOT required — we force attn_implementation=sdpa at the CLI.
#
# Usage:
#   bash run_uld_open_unlearning_8b.sh
#   GPU=0 bash run_uld_open_unlearning_8b.sh
#   ONLY=forget05 bash run_uld_open_unlearning_8b.sh   # run a single split
#
# Hardware: tested on a single A100-40GB.

set -euo pipefail

#####################################
# Config
#####################################
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ULD_REPO="${ULD_REPO:-${ROOT}/ULD}"
OU_REPO="${OU_REPO:-${ROOT}/open-unlearning}"
PY="${PY:-/usr/bin/python}"

GPU="${GPU:-0}"
HF_BASE_PREFIX="${HF_BASE_PREFIX:-open-unlearning/tofu_Llama-3.1-8B-Instruct}"
# Tokenizer source — defaults to the open-unlearning mirror, which is public.
# meta-llama/* is gated and requires a separate access request; if you have
# access, you can override: HF_TOKENIZER=meta-llama/Llama-3.1-8B-Instruct
HF_TOKENIZER="${HF_TOKENIZER:-open-unlearning/tofu_Llama-3.1-8B-Instruct_full}"

NUM_LAYER="${NUM_LAYER:-4}"          # assistant uses 4 of base's 32 layers
LORA_R="${LORA_R:-16}"
ULD_WEIGHT="${ULD_WEIGHT:--0.8}"
ULD_TOPF="${ULD_TOPF:-0.01}"

TRAIN_BS="${TRAIN_BS:-4}"
TRAIN_GA="${TRAIN_GA:-4}"
TRAIN_LR="${TRAIN_LR:-1e-3}"
TRAIN_EP="${TRAIN_EP:-10}"

MODELS_ROOT="${MODELS_ROOT:-${ULD_REPO}/outputs_trained_models/llama3_8b}"

splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

# Optional: ONLY=forget01 to run a single split
if [ -n "${ONLY:-}" ]; then
    new_splits=()
    for sp in "${splits[@]}"; do
        case "$sp" in "$ONLY"*) new_splits+=("$sp");; esac
    done
    splits=("${new_splits[@]}")
fi

mkdir -p "$MODELS_ROOT"

echo "============================================================"
echo "ULD × open-unlearning, Llama-3.1-8B-Instruct"
echo "  ULD repo            : $ULD_REPO"
echo "  open-unlearning repo: $OU_REPO"
echo "  python              : $PY"
echo "  GPU                 : $GPU"
echo "  num_layer / lora_r  : $NUM_LAYER / $LORA_R"
echo "  ULD weight / topF   : $ULD_WEIGHT / $ULD_TOPF"
echo "  splits              : ${splits[*]}"
echo "============================================================"

#####################################
# Phase A — retain-model eval (reference for forget_quality)
#####################################
echo
if [ "${SKIP_PHASE_A:-0}" = "1" ]; then
    echo "[Phase A] SKIPPED (SKIP_PHASE_A=1) — forget_quality will be omitted in eval"
else
    echo "[Phase A] Reference TOFU_EVAL.json for each retain model"
    cd "$OU_REPO"
    # First try the precomputed bundle on HF (open-unlearning/eval dataset).
    # That covers all three retain splits in seconds — only fall back to
    # local re-eval (FORCE_PHASE_A_RECOMPUTE=1) if the bundle download fails.
    need_download=0
    for sp in "${splits[@]}"; do
        retain=$(echo "$sp" | cut -d' ' -f3)
        out_json="${OU_REPO}/saves/eval/tofu_Llama-3.1-8B-Instruct_${retain}/TOFU_EVAL.json"
        [ -f "$out_json" ] || need_download=1
    done
    if [ "$need_download" = "1" ] && [ "${FORCE_PHASE_A_RECOMPUTE:-0}" != "1" ]; then
        echo "  → Downloading precomputed retain eval logs (open-unlearning/eval)"
        "$PY" setup_data.py --eval_logs
    fi
    for sp in "${splits[@]}"; do
        forget=$(echo "$sp" | cut -d' ' -f1)
        holdout=$(echo "$sp" | cut -d' ' -f2)
        retain=$(echo "$sp" | cut -d' ' -f3)
        task="tofu_Llama-3.1-8B-Instruct_${retain}"
        out_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"
        if [ -f "$out_json" ]; then
            echo "  → OK $retain: $out_json"
            continue
        fi
        echo "  → Recomputing $retain → $out_json"
        CUDA_VISIBLE_DEVICES="$GPU" "$PY" src/eval.py \
            experiment=eval/tofu/default \
            model=Llama-3.1-8B-Instruct \
            model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_${retain}" \
            model.model_args.attn_implementation=sdpa \
            model.tokenizer_args.pretrained_model_name_or_path="${HF_TOKENIZER}" \
            forget_split="$forget" \
            holdout_split="$holdout" \
            task_name="$task"
    done
fi

#####################################
# Phase B — train ULD assistants
#####################################
echo
echo "[Phase B] Train ULD assistants"
cd "$ULD_REPO"
export PYTHONPATH="${ULD_REPO}:${PYTHONPATH:-}"
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
for sp in "${splits[@]}"; do
    forget=$(echo "$sp" | cut -d' ' -f1)
    out_dir="${MODELS_ROOT}/uld_assistant_${forget}"
    if find "$out_dir" -name "checkpoint-*" -type d 2>/dev/null | grep -q .; then
        echo "  → SKIP $forget: assistant already trained at $out_dir"
        continue
    fi
    echo "  → Train ULD assistant for $forget → $out_dir"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" scripts/hf_forget_train.py \
        project="llama3_8b_uld_${forget}" \
        data=tofu_chat3 \
        data.dataset.split="${forget}_perturbed" \
        data_mode=forget_more_retain_perturb \
        model=llama-3-8b \
        model.model_path="${HF_BASE_PREFIX}_full" \
        model.tokenizer_path="${HF_TOKENIZER}" \
        model_mode=uld \
        model_mode.num_layer="$NUM_LAYER" \
        model_mode.Lora.r="$LORA_R" \
        unlearn_loss=remember+uniform \
        unlearn_loss.retain_weight=5.0 \
        trainer.batch_size="$TRAIN_BS" \
        trainer.gradient_accumulation_steps="$TRAIN_GA" \
        trainer.learning_rate="$TRAIN_LR" \
        trainer.max_epochs="$TRAIN_EP" \
        trainer.strategy=gpu \
        OUTPUTMODELDIR="$out_dir" \
        postfix=uld \
        "hydra.run.dir=outputs/tune_log/llama3_8b_uld_${forget}/\${now:%Y-%m-%d_%H-%M-%S}"
done

#####################################
# Phase C — eval ULD with open-unlearning metrics
#####################################
echo
echo "[Phase C] Eval ULD with open-unlearning TOFU metrics"
cd "$OU_REPO"
for sp in "${splits[@]}"; do
    forget=$(echo "$sp" | cut -d' ' -f1)
    holdout=$(echo "$sp" | cut -d' ' -f2)
    retain=$(echo "$sp" | cut -d' ' -f3)
    out_dir="${MODELS_ROOT}/uld_assistant_${forget}"

    # pick the latest checkpoint-*
    ck=$(find "$out_dir" -name "checkpoint-*" -type d \
        | awk -F'checkpoint-' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -z "$ck" ]; then
        echo "  ✗ ERROR: no checkpoint found in $out_dir"; exit 1
    fi

    task="tofu_Llama-3.1-8B-Instruct_${forget}_ULD"
    retain_json="${OU_REPO}/saves/eval/tofu_Llama-3.1-8B-Instruct_${retain}/TOFU_EVAL.json"
    eval_json="${OU_REPO}/saves/eval/${task}/TOFU_EVAL.json"

    if [ -f "$eval_json" ]; then
        echo "  → SKIP $forget: $eval_json exists"
        continue
    fi

    # If retain JSON is missing (Phase A skipped), pass null so eval still runs
    # — forget_quality will be omitted but the other metrics still compute.
    if [ -f "$retain_json" ]; then
        retain_arg="retain_logs_path=$retain_json"
    else
        retain_arg="retain_logs_path=null"
        echo "  ! retain_logs_path=null (forget_quality will be omitted)"
    fi

    echo "  → Eval ULD on $forget (assistant=$ck) → $eval_json"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" src/eval.py \
        experiment=eval/tofu/default \
        model=Llama-3.1-8B-Instruct_ULD \
        model.model_args.pretrained_model_name_or_path="${HF_BASE_PREFIX}_full" \
        model.model_args.assistant_path="$ck" \
        model.model_args.weight="$ULD_WEIGHT" \
        model.model_args.top_logit_filter="$ULD_TOPF" \
        model.model_args.attn_implementation=sdpa \
        model.tokenizer_args.pretrained_model_name_or_path="${HF_TOKENIZER}" \
        forget_split="$forget" \
        holdout_split="$holdout" \
        $retain_arg \
        task_name="$task"
done

#####################################
# Summary
#####################################
echo
echo "============================================================"
echo "DONE — open-unlearning TOFU metrics for ULD on Llama-3.1-8B"
echo "============================================================"
for sp in "${splits[@]}"; do
    forget=$(echo "$sp" | cut -d' ' -f1)
    json="${OU_REPO}/saves/eval/tofu_Llama-3.1-8B-Instruct_${forget}_ULD/TOFU_EVAL.json"
    if [ -f "$json" ]; then
        echo
        echo "$forget : $json"
        "$PY" - <<EOF
import json, sys
data = json.load(open("$json"))
def get(k, *path):
    cur = data.get(k, {})
    for p in path: cur = cur.get(p, {}) if isinstance(cur, dict) else {}
    return cur if not isinstance(cur, dict) else cur.get("agg_value", cur)
def show(k):
    v = data.get(k, None)
    if isinstance(v, dict) and "agg_value" in v: print(f"  {k:32s} = {v['agg_value']:.4f}")
    elif isinstance(v, (int, float)): print(f"  {k:32s} = {v:.4f}")
for k in ("forget_quality","model_utility","forget_truth_ratio",
         "forget_Q_A_Prob","forget_Q_A_ROUGE","privleak","extraction_strength"):
    show(k)
EOF
    else
        echo "$forget : MISSING ($json)"
    fi
done
