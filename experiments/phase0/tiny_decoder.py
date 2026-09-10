#!/usr/bin/env python3
"""Portable Phase 0 workload for timing and torch.profiler practice."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "phase0" / "tiny-decoder-fixed-v1.json"


class TinyDecoder(nn.Module):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        model = config["model"]
        assert isinstance(model, dict)
        vocab_size = int(model["vocab_size"])
        hidden_size = int(model["hidden_size"])
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=int(model["attention_heads"]),
            dim_feedforward=int(model["ffn_hidden_size"]),
            dropout=float(model["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.TransformerEncoder(layer, num_layers=int(model["layers"]))
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)
        if model["tied_input_output_embeddings"]:
            self.output.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.token_embedding(token_ids)
        seq_len = token_ids.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=token_ids.device, dtype=hidden.dtype
        )
        hidden = self.layers(hidden, mask=causal_mask, is_causal=True)
        return self.output(self.final_norm(hidden))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[index]


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    with record_function("train_step"):
        optimizer.zero_grad(set_to_none=True)
        with record_function("forward"):
            logits = model(inputs)
        with record_function("loss"):
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            optimizer.step()
    return loss


def load_config(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_device(device: torch.device) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = config["model"]
    workload_config = config["workload_config"]
    optimizer_config = config["optimizer"]
    measurement = config["measurement"]
    assert isinstance(model_config, dict)
    assert isinstance(workload_config, dict)
    assert isinstance(optimizer_config, dict)
    assert isinstance(measurement, dict)

    device = torch.device(args.device)
    validate_device(device)
    if workload_config["dtype"] != "float32":
        raise SystemExit("This fixed Phase 0 entrypoint currently supports float32 only.")

    seed = int(workload_config["seed"])
    torch.manual_seed(seed)
    model = TinyDecoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(optimizer_config["learning_rate"]))

    batch_size = int(workload_config["batch_size"])
    sequence_length = int(workload_config["sequence_length"])
    vocab_size = int(model_config["vocab_size"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    all_tokens = torch.randint(
        0, vocab_size, (batch_size, sequence_length + 1), generator=generator
    )
    inputs = all_tokens[:, :-1].to(device)
    targets = all_tokens[:, 1:].to(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    warmup_steps = int(measurement["warmup_steps"])
    measured_steps = int(measurement["measured_steps"])
    for _ in range(warmup_steps):
        train_step(model, optimizer, inputs, targets)
    synchronize(device)

    durations = []
    last_loss = None
    for _ in range(measured_steps):
        synchronize(device)
        start = time.perf_counter()
        last_loss = train_step(model, optimizer, inputs, targets)
        synchronize(device)
        durations.append(time.perf_counter() - start)

    assert last_loss is not None
    median_step = statistics.median(durations)
    metrics = {
        "workload": config["workload"],
        "torch_version": torch.__version__,
        "device": str(device),
        "dtype": workload_config["dtype"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "median_step_time_ms": median_step * 1000,
        "p95_step_time_ms": percentile(durations, 0.95) * 1000,
        "tokens_per_second": batch_size * sequence_length / median_step,
        "last_loss": float(last_loss.detach().cpu()),
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "peak_memory_note": (
            "torch.cuda.max_memory_allocated"
            if device.type == "cuda"
            else "Not reported: no equivalent reliable CPU/MPS peak API is used."
        ),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    if args.profile:
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        ) as profiler:
            train_step(model, optimizer, inputs, targets)
            synchronize(device)

        sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        table = profiler.key_averages().table(sort_by=sort_key, row_limit=25)
        (args.output_dir / "profiler_table.txt").write_text(table, encoding="utf-8")
        profiler.export_chrome_trace(str(args.output_dir / "trace.json"))
        print(table)
        print(f"Trace written to {args.output_dir / 'trace.json'}")


if __name__ == "__main__":
    main()
