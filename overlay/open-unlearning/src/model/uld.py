"""ULD (Unlearning via Logit Difference) wrapper for open-unlearning.

At inference time:
    final_logits = base_logits + weight * filtered_assistant_logits

The assistant has fewer transformer layers than the base, so its KV cache
shape differs. We disable KV caching (recompute every step) to keep this
simple and correct — TOFU sequences are short enough that the O(n^2) cost
is acceptable.

The class subclasses LlamaForCausalLM purely to inherit a working
`generate()` and tokenization plumbing. Override behavior:
  - forward(): runs base + assistant, combines logits, returns
    CausalLMOutputWithPast with past_key_values=None.
  - from_pretrained(): also loads the assistant via `assistant_path`.

Assistant loading supports two on-disk shapes:
  1. A merged HF model directory (config.json + safetensors).
  2. A LoRA adapter directory with a sibling `../fullmodel/` directory
     (the layout produced by ULD/scripts/hf_forget_train.py with
     model_mode=uld). We auto-load fullmodel and merge the LoRA.
"""
import os
import logging
from typing import Optional

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger("model.uld")


def _relative_top_filter(scores: torch.Tensor, relative_top: float):
    """Mask out (in log-space) tokens whose normalized score is far below
    the top of the distribution. Returns (scores, mask) where mask is True
    for filtered-out positions. Adapted from ULD/uld/model/gen_util.py."""
    min_tokens_to_keep = max(int(relative_top * scores.shape[-1]), 1)
    scores_normalized = scores.log_softmax(dim=-1)
    sorted_logits, _ = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + float(np.log(relative_top))
    probs_thresh = torch.min(min_thresh, probs_thresh).unsqueeze(-1)
    mask = scores_normalized < probs_thresh
    return scores, mask


def _load_assistant(
    assistant_path: str,
    torch_dtype: Optional[torch.dtype],
    attn_implementation: Optional[str],
) -> torch.nn.Module:
    extra = {}
    if torch_dtype is not None:
        extra["torch_dtype"] = torch_dtype
    if attn_implementation is not None:
        extra["attn_implementation"] = attn_implementation

    adapter_cfg = os.path.join(assistant_path, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        # ULD layout: ../fullmodel + this LoRA dir.
        full_dir = os.path.normpath(os.path.join(assistant_path, "..", "fullmodel"))
        if not os.path.isdir(full_dir):
            raise FileNotFoundError(
                f"Found LoRA adapter at {assistant_path} but no sibling "
                f"`fullmodel/` directory at {full_dir}. ULD-trained "
                f"assistants save the small base model under ../fullmodel."
            )
        from peft import PeftModel  # local import: peft is optional

        logger.info(f"ULD: loading assistant fullmodel from {full_dir}")
        base = AutoModelForCausalLM.from_pretrained(full_dir, **extra)
        logger.info(f"ULD: merging LoRA adapter from {assistant_path}")
        peft = PeftModel.from_pretrained(base, assistant_path, **extra)
        return peft.merge_and_unload()

    # Already-merged HF model directory.
    logger.info(f"ULD: loading assistant (merged) from {assistant_path}")
    return AutoModelForCausalLM.from_pretrained(assistant_path, **extra)


class ULDForCausalLM(LlamaForCausalLM):
    """LlamaForCausalLM whose forward() output is the ULD logit difference."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        assistant_path: str = None,
        weight: float = -0.8,
        top_logit_filter: float = 0.01,
        **kwargs,
    ):
        if assistant_path is None or assistant_path == "???":
            raise ValueError(
                "ULDForCausalLM.from_pretrained requires `assistant_path` "
                "(set in model_args)."
            )
        # Load the base via the parent class so we inherit tokenizer/config
        # plumbing exactly as a normal Llama model would.
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Load assistant on the same dtype / attn impl as base.
        torch_dtype = kwargs.get("torch_dtype", None)
        attn_impl = kwargs.get("attn_implementation", None)
        assistant = _load_assistant(assistant_path, torch_dtype, attn_impl)
        # Place assistant on the base's device. open-unlearning loads with
        # device_map="cuda" so base is on cuda:0.
        device = next(model.parameters()).device
        assistant.to(device)
        assistant.eval()

        # Stash on the base. Use object.__setattr__ to bypass nn.Module's
        # auto-registration into model.modules() (we don't want generate()
        # or saving routines to walk into the assistant).
        object.__setattr__(model, "_uld_assistant", assistant)
        model._uld_weight = float(weight)
        model._uld_top_filter = float(top_logit_filter)

        # KV cache for the base would diverge from a non-existent assistant
        # cache, so disable caching globally for this model.
        model.generation_config.use_cache = False
        model.config.use_cache = False

        logger.info(
            f"ULDForCausalLM ready: base={pretrained_model_name_or_path} "
            f"assistant={assistant_path} weight={weight} "
            f"top_logit_filter={top_logit_filter}"
        )
        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
        **kwargs,
    ):
        # Always recompute fully (assistant has fewer layers, so the
        # base's KV cache can't be reused for the assistant; rather than
        # juggling two caches, recompute both).
        base_out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        assist_out = self._uld_assistant(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        base_logits = base_out.logits
        assist_logits = assist_out.logits.to(base_logits.device)

        if self._uld_top_filter > 0.0:
            base_logits, mask = _relative_top_filter(base_logits, self._uld_top_filter)
            # Zero out the assistant's contribution where base is filtered.
            assist_logits = assist_logits.clone()
            assist_logits[mask] = 0.0
            logits = base_logits + self._uld_weight * assist_logits
        else:
            logits = base_logits + self._uld_weight * assist_logits

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # Override: never carry past_key_values through, and always feed the
        # full input_ids each step (matches use_cache=False).
        kwargs.pop("past_key_values", None)
        kwargs["use_cache"] = False
        out = super().prepare_inputs_for_generation(
            input_ids, past_key_values=None, **kwargs
        )
        # Newer transformers may set `input_ids` to only the last token
        # when past_key_values is present; force full sequence here.
        out["input_ids"] = input_ids
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            out["attention_mask"] = kwargs["attention_mask"]
        out["past_key_values"] = None
        out["use_cache"] = False
        return out
