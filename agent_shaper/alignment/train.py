# DPO training loop (Direct Preference Optimization).
#
# Requires optional deps: `uv sync --extra alignment`
#
# Usage:
#   uv run python -m agent_shaper.alignment.train --model_name <hf-id> --dataset_name <hf-id>

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sequence summed log-probabilities over masked positions (DPO-style)."""
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    mask = mask[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    per_token_logps = per_token_logps * mask
    return per_token_logps.sum(dim=-1)


class DPOLoss(nn.Module):
    """Direct Preference Optimization loss (Rafailov et al., 2023).

    Loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

    where log_ratio = log(pi(y|x) / pi_ref(y|x))
    """

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,    # einops: "b"
        policy_rejected_logps: torch.Tensor,  # einops: "b"
        ref_chosen_logps: torch.Tensor,        # einops: "b"
        ref_rejected_logps: torch.Tensor,      # einops: "b"
    ) -> tuple[torch.Tensor, dict]:
        # Validate: all inputs must be 1-D with a consistent batch size.
        # einsum signature for each input:  b ->
        batch = einops.parse_shape(policy_chosen_logps, "b")["b"]
        for name, tensor in (
            ("policy_rejected_logps", policy_rejected_logps),
            ("ref_chosen_logps", ref_chosen_logps),
            ("ref_rejected_logps", ref_rejected_logps),
        ):
            parsed = einops.parse_shape(tensor, "b")
            if parsed["b"] != batch:
                raise ValueError(
                    f"{name} batch {parsed['b']} != policy_chosen batch {batch}"
                )

        chosen_logratios = policy_chosen_logps - ref_chosen_logps    # einops: "b"
        rejected_logratios = policy_rejected_logps - ref_rejected_logps  # einops: "b"
        logits = chosen_logratios - rejected_logratios               # einops: "b"

        if self.label_smoothing > 0:
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )  # einops: "b"
        else:
            losses = -F.logsigmoid(self.beta * logits)               # einops: "b"

        chosen_rewards = self.beta * chosen_logratios.detach()       # einops: "b"
        rejected_rewards = self.beta * rejected_logratios.detach()   # einops: "b"

        metrics = {
            "chosen_rewards": chosen_rewards.mean().item(),                        # einsum: "b ->"
            "rejected_rewards": rejected_rewards.mean().item(),                    # einsum: "b ->"
            "margins": (chosen_rewards - rejected_rewards).mean().item(),          # einsum: "b ->"
            "accuracy": (chosen_rewards > rejected_rewards).float().mean().item(), # einsum: "b ->"
        }

        return losses.mean(), metrics  # einsum: "b ->"


class DPOLossEinsum(nn.Module):
    """DPO loss (Rafailov et al., 2023) — einops reductions.

    Every intermediate tensor is a 1-D batch vector:
      inputs / logratios / logits / losses / rewards : b
      all metric scalars and the returned loss       : b -> (einops.reduce mean)
    """

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,    # b
        policy_rejected_logps: torch.Tensor,  # b
        ref_chosen_logps: torch.Tensor,        # b
        ref_rejected_logps: torch.Tensor,      # b
    ) -> tuple[torch.Tensor, dict]:
        batch = einops.parse_shape(policy_chosen_logps, "b")["b"]
        for name, tensor in (
            ("policy_rejected_logps", policy_rejected_logps),
            ("ref_chosen_logps", ref_chosen_logps),
            ("ref_rejected_logps", ref_rejected_logps),
        ):
            parsed = einops.parse_shape(tensor, "b")
            if parsed["b"] != batch:
                raise ValueError(
                    f"{name} batch {parsed['b']} != policy_chosen batch {batch}"
                )

        chosen_logratios: torch.Tensor = policy_chosen_logps - ref_chosen_logps        # b
        rejected_logratios: torch.Tensor = policy_rejected_logps - ref_rejected_logps  # b
        logits: torch.Tensor = chosen_logratios - rejected_logratios                   # b

        if self.label_smoothing > 0:
            losses: torch.Tensor = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )  # b
        else:
            losses = -F.logsigmoid(self.beta * logits)  # b

        chosen_rewards: torch.Tensor = self.beta * chosen_logratios.detach()     # b
        rejected_rewards: torch.Tensor = self.beta * rejected_logratios.detach() # b

        metrics = {
            "chosen_rewards": einops.reduce(chosen_rewards, "b -> ", "mean").item(),
            "rejected_rewards": einops.reduce(rejected_rewards, "b -> ", "mean").item(),
            "margins": einops.reduce(
                chosen_rewards - rejected_rewards, "b -> ", "mean"
            ).item(),
            "accuracy": einops.reduce(
                (chosen_rewards > rejected_rewards).float(), "b -> ", "mean"
            ).item(),
        }

        return einops.reduce(losses, "b -> ", "mean"), metrics  # b ->


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(
    model_name: str,
    device: str,
    gradient_checkpointing: bool = True,
    bf16: bool = True,
):
    dtype = torch.bfloat16 if bf16 else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model = model.to(device)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    return model, tokenizer


def load_ref_model(model_name: str, device: str, bf16: bool = True):
    dtype = torch.bfloat16 if bf16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def forward_pass(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    return compute_logprobs(outputs.logits, labels=input_ids, mask=response_mask)


@dataclass
class PreferenceBatch:
    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    chosen_response_mask: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor
    rejected_response_mask: torch.Tensor

    def to(self, device: torch.device) -> PreferenceBatch:
        return PreferenceBatch(
            chosen_input_ids=self.chosen_input_ids.to(device),
            chosen_attention_mask=self.chosen_attention_mask.to(device),
            chosen_response_mask=self.chosen_response_mask.to(device),
            rejected_input_ids=self.rejected_input_ids.to(device),
            rejected_attention_mask=self.rejected_attention_mask.to(device),
            rejected_response_mask=self.rejected_response_mask.to(device),
        )


def _encode_pair(
    tokenizer,
    prompt: str,
    completion: str,
    max_length: int,
) -> tuple[list[int], list[int]]:
    """Returns input_ids and response_mask (1 on completion tokens only)."""
    if tokenizer.chat_template is not None:
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        full_text = f"{prompt}{completion}"
        prompt_prefix = prompt

    full_enc = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_tensors=None,
    )
    prompt_enc = tokenizer(
        prompt_prefix,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_tensors=None,
    )

    input_ids = full_enc["input_ids"]
    attn = full_enc["attention_mask"]
    prompt_len = len(prompt_enc["input_ids"])
    response_mask = [0] * len(input_ids)
    for i in range(prompt_len, len(input_ids)):
        response_mask[i] = attn[i]
    return input_ids, response_mask


def _collate_preference_batch(features: list[dict]) -> PreferenceBatch:
    pad_id = features[0]["pad_token_id"]
    max_len = max(len(f["chosen_input_ids"]) for f in features)
    max_len = max(max_len, max(len(f["rejected_input_ids"]) for f in features))

    def pad_seq(ids: list[int], mask: list[int], fill_len: int):
        pad_n = fill_len - len(ids)
        ids = ids + [pad_id] * pad_n
        am = [1] * (fill_len - pad_n) + [0] * pad_n
        rm = mask + [0] * pad_n
        return ids, am, rm

    chosen_ids, chosen_am, chosen_rm = [], [], []
    rej_ids, rej_am, rej_rm = [], [], []

    for f in features:
        c_ids, c_am, c_rm = pad_seq(f["chosen_input_ids"], f["chosen_response_mask"], max_len)
        r_ids, r_am, r_rm = pad_seq(f["rejected_input_ids"], f["rejected_response_mask"], max_len)
        chosen_ids.append(c_ids)
        chosen_am.append(c_am)
        chosen_rm.append(c_rm)
        rej_ids.append(r_ids)
        rej_am.append(r_am)
        rej_rm.append(r_rm)

    return PreferenceBatch(
        chosen_input_ids=torch.tensor(chosen_ids, dtype=torch.long),
        chosen_attention_mask=torch.tensor(chosen_am, dtype=torch.long),
        chosen_response_mask=torch.tensor(chosen_rm, dtype=torch.float),
        rejected_input_ids=torch.tensor(rej_ids, dtype=torch.long),
        rejected_attention_mask=torch.tensor(rej_am, dtype=torch.long),
        rejected_response_mask=torch.tensor(rej_rm, dtype=torch.float),
    )


def create_dataloader(
    dataset_name: str,
    tokenizer,
    split: str,
    max_samples: int | None,
    max_length: int,
    batch_size: int,
    shuffle: bool,
):
    raw = load_dataset(dataset_name, split=split)
    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))

    def row_to_features(ex):
        if "prompt" in ex:
            prompt = ex["prompt"]
        else:
            question = ex.get("question", "")
            system = ex.get("system", "")
            prompt = f"{system}\n\n{question}".strip() if system else question
        chosen = ex["chosen"]
        rejected = ex["rejected"]
        c_ids, c_rm = _encode_pair(tokenizer, prompt, chosen, max_length)
        r_ids, r_rm = _encode_pair(tokenizer, prompt, rejected, max_length)
        return {
            "chosen_input_ids": c_ids,
            "chosen_response_mask": c_rm,
            "rejected_input_ids": r_ids,
            "rejected_response_mask": r_rm,
            "pad_token_id": tokenizer.pad_token_id,
        }

    mapped = raw.map(row_to_features, remove_columns=raw.column_names)

    return DataLoader(
        mapped,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_preference_batch,
    )


def train_step(
    policy_model,
    ref_model,
    batch: PreferenceBatch,
    loss_fn: DPOLoss,
    optimizer,
    max_grad_norm: float,
    gradient_accumulation_steps: int,
    step_in_accumulation: int,
) -> dict[str, float]:
    device = next(policy_model.parameters()).device
    batch = batch.to(device)

    policy_chosen_logps = forward_pass(
        policy_model,
        batch.chosen_input_ids,
        batch.chosen_attention_mask,
        batch.chosen_response_mask,
    )
    policy_rejected_logps = forward_pass(
        policy_model,
        batch.rejected_input_ids,
        batch.rejected_attention_mask,
        batch.rejected_response_mask,
    )

    with torch.no_grad():
        ref_chosen_logps = forward_pass(
            ref_model,
            batch.chosen_input_ids,
            batch.chosen_attention_mask,
            batch.chosen_response_mask,
        )
        ref_rejected_logps = forward_pass(
            ref_model,
            batch.rejected_input_ids,
            batch.rejected_attention_mask,
            batch.rejected_response_mask,
        )

    loss, metrics = loss_fn(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        ref_chosen_logps=ref_chosen_logps,
        ref_rejected_logps=ref_rejected_logps,
    )

    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()

    grad_norm_val = None
    if (step_in_accumulation + 1) % gradient_accumulation_steps == 0:
        grad_norm_val = clip_grad_norm_(policy_model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    metrics["loss"] = loss.item()
    if grad_norm_val is not None:
        metrics["grad_norm"] = grad_norm_val.item()

    return metrics


@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    ref_model_name: str | None = None
    dataset_name: str = "Intel/orca_dpo_pairs"
    dataset_split: str = "train"
    max_samples: int | None = None
    max_length: int = 1024
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-7
    weight_decay: float = 0.0
    num_epochs: int = 1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    beta: float = 0.1
    label_smoothing: float = 0.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    bf16: bool = True
    gradient_checkpointing: bool = True
    output_dir: str = "./dpo_checkpoints"
    save_model: bool = False


def main(cfg: Config) -> None:
    seed_everything(cfg.seed)

    ref_name = cfg.ref_model_name or cfg.model_name

    policy_model, tokenizer = load_model(
        cfg.model_name,
        cfg.device,
        gradient_checkpointing=cfg.gradient_checkpointing,
        bf16=cfg.bf16,
    )
    ref_model = load_ref_model(ref_name, cfg.device, bf16=cfg.bf16)

    dataloader = create_dataloader(
        dataset_name=cfg.dataset_name,
        tokenizer=tokenizer,
        split=cfg.dataset_split,
        max_samples=cfg.max_samples,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    loss_fn = DPOLoss(beta=cfg.beta, label_smoothing=cfg.label_smoothing)

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    steps_per_epoch = max(1, len(dataloader) // cfg.gradient_accumulation_steps)
    num_training_steps = steps_per_epoch * cfg.num_epochs
    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step + 1) / float(max(1, num_warmup_steps + 1))
        return max(
            0.0,
            float(num_training_steps - step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    start_time = time.time()
    policy_model.train()

    last_metrics: dict[str, float] = {}

    for epoch in range(cfg.num_epochs):
        metrics_sum: dict[str, float] = {}
        metrics_count = 0

        for batch_idx, batch in enumerate(dataloader):
            metrics = train_step(
                policy_model=policy_model,
                ref_model=ref_model,
                batch=batch,
                loss_fn=loss_fn,
                optimizer=optimizer,
                max_grad_norm=cfg.max_grad_norm,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                step_in_accumulation=batch_idx,
            )

            metrics_count += 1
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metrics_sum[key] = metrics_sum.get(key, 0.0) + float(value)

            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                scheduler.step()
                global_step += 1
                last_metrics = {k: v / metrics_count for k, v in metrics_sum.items()}
                last_metrics["learning_rate"] = scheduler.get_last_lr()[0]
                lr = last_metrics["learning_rate"]
                loss_v = last_metrics.get("loss", float("nan"))
                print(
                    f"step={global_step} epoch={epoch + 1} loss={loss_v:.4f} "
                    f"lr={lr:.2e} elapsed_h={(time.time() - start_time) / 3600:.2f}"
                )
                metrics_sum = {}
                metrics_count = 0

        remaining = len(dataloader) % cfg.gradient_accumulation_steps
        if remaining != 0:
            clip_grad_norm_(policy_model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            if metrics_count > 0:
                last_metrics = {k: v / metrics_count for k, v in metrics_sum.items()}
                last_metrics["learning_rate"] = scheduler.get_last_lr()[0]

    print("Training complete.")
    if last_metrics:
        print(f"final_loss={last_metrics.get('loss', float('nan')):.4f}")

    if cfg.save_model:
        output_path = Path(cfg.output_dir) / f"dpo_{cfg.model_name.split('/')[-1]}"
        output_path.mkdir(parents=True, exist_ok=True)
        policy_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"Saved policy to {output_path}")


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Train with DPO loss.")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--ref_model_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_model", action="store_true")

    args = parser.parse_args()
    cfg = Config()
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key, value)

    main(cfg)


if __name__ == "__main__":
    main_cli()
