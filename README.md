# ULD on Llama-3.1-8B-Instruct, evaluated with open-unlearning's TOFU benchmark

Glue code that lets you train **ULD** ([Reversing the Forget-Retain Objectives,
arXiv:2406.08607](https://arxiv.org/abs/2406.08607)) on Llama-3.1-8B-Instruct
and evaluate it inside [`open-unlearning`](https://github.com/locuslab/open-unlearning)'s
TOFU benchmark — all 7 metrics
(`forget_quality / model_utility / forget_truth_ratio / forget_Q_A_Prob /
forget_Q_A_ROUGE / privleak / extraction_strength`).

## What's in this repo

```
.
├── setup.sh                    # clones UCSB-NLP-Chang/ULD + locuslab/open-unlearning
│                                 at pinned commits, then overlays our changes
├── run.sh                      # one-click Phase A + B + C
└── overlay/
    ├── ULD/                    # 2 new files (Llama-3-8B + Llama-3 chat-template
    │                             data config)
    └── open-unlearning/        # 1 new model handler (ULDForCausalLM), 1 new
                                  model config, plus 3 small fixes (registry
                                  hook + bf16-numpy casts)
```

Upstream pins (last tested):

- `UCSB-NLP-Chang/ULD` @ `858608c`
- `locuslab/open-unlearning` @ `4ad738a`

## Quick start (new server)

```bash
git clone https://github.com/JingyueCong/uld_in_openunlearn.git
cd uld_in_openunlearn

bash setup.sh                   # clones + overlays — idempotent
huggingface-cli login           # Llama-3.1 weights are gated
python -m pip install --user tf-keras lm-eval hydra-colorlog peft

# Validate end-to-end on one split first (~30-60 min on A100):
SKIP_PHASE_A=1 ONLY=forget01 bash run.sh

# Full run — all 3 splits, all 7 metrics (~2-3h on A100):
bash run.sh
```

## What `run.sh` does

| Phase | What | Time on A100 40GB |
|---|---|---|
| A | Download retain-model TOFU eval JSONs (reference for `forget_quality`) | ~5 sec |
| B | Train ULD assistant for each forget split (4-layer + LoRA r=16, 10 epoch) | ~30-45 min |
| C | Evaluate `ULDForCausalLM(base, assistant)` with open-unlearning's TOFU evaluator | ~1.5-2 h |

Each phase is idempotent — re-running skips work whose output already exists.

## Useful env vars

| var | default | meaning |
|---|---|---|
| `GPU` | `0` | CUDA_VISIBLE_DEVICES |
| `SKIP_PHASE_A` | `0` | `1` skips retain JSON download (loses `forget_quality`) |
| `ONLY` | _(unset)_ | run a single split, e.g. `ONLY=forget01` |
| `NUM_LAYER` | `4` | assistant layer count (base has 32) |
| `LORA_R` | `16` | LoRA rank for assistant |
| `ULD_WEIGHT` | `-0.8` | logit-difference coefficient |
| `ULD_TOPF` | `0.01` | relative top-logit filter for assistant |
| `TRAIN_BS` / `TRAIN_GA` | `4` / `4` | per-device batch size / grad accum (lower if OOM) |
| `HF_BASE_PREFIX` | `open-unlearning/tofu_Llama-3.1-8B-Instruct` | HF id prefix for `_full` / `_retain*` |

## How it works

ULD does a logit difference at inference:

```
final_logits = base_logits + weight × filtered(assistant_logits)
```

- `base` = `open-unlearning/tofu_Llama-3.1-8B-Instruct_full` (already finetuned on
  full TOFU by open-unlearning — we skip ULD's Phase 0/1)
- `assistant` = a 4-layer LoRA model trained on the same TOFU split with
  `remember+uniform` loss (CE on forget, KL-to-uniform on retain). Trained by
  ULD's own trainer (`ULD/scripts/hf_forget_train.py`).
- The `ULDForCausalLM` wrapper in `open-unlearning/src/model/uld.py` plugs into
  open-unlearning's `MODEL_REGISTRY` and overrides `forward()` to compute the
  combined logits. KV cache is disabled because base and assistant have
  different layer counts; for TOFU's short sequences this is acceptable.

## Hardware

Tested on a single A100 40GB. `TRAIN_BS=2 TRAIN_GA=8` if you OOM during Phase B.

Phase C is the bottleneck (no KV cache → O(n²) generation per token); switching
the base to KV-cache mode would cut Phase C from ~1.5h to ~30 min, but would
require slightly more wrapper plumbing — not yet implemented.
