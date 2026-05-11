# Design: Optimize NFT Wan22 I2V Config for H200 GPU Utilization

**Date:** 2026-05-11
**Status:** Approved
**Goal:** Maximize training throughput by increasing per_device_batch_size to utilize unused H200 memory (~100GB idle out of 141GB)

## Problem

Current NFT LoRA Wan22 I2V training on 8× H200 GPUs uses only ~40GB per GPU (28% utilization). The root cause is `per_device_batch_size=1`, which processes one sample at a time during both rollout inference and optimization.

## Current Config

| Parameter | Value | Effect |
|---|---|---|
| `per_device_batch_size` | 1 | 1 sample per GPU per micro-batch |
| `unique_sample_num_per_epoch` | 48 | 48 unique prompts per epoch |
| `group_size` | 16 | 16 samples per prompt |
| `num_batches_per_epoch` | 96 | (48×16) / (8×1) = 96 rollout batches |
| `gradient_accumulation_steps` | 192 | 96 × 2 timesteps = 192 |
| `gradient_step_per_epoch` | 1 | 1 optimizer step per epoch |
| `num_train_timesteps` | 2 | NFT timesteps per sample |
| GPU memory used | ~40 GB / 141 GB | 28% utilization |

## Proposed Change

| Parameter | Old | New | Rationale |
|---|---|---|---|
| `per_device_batch_size` | 1 | **4** | Process 4 samples simultaneously per GPU |
| `gradient_accumulation_steps` | 192 | **48** | 96/4=24 batches × 2 timesteps = 48 |
| `num_batches_per_epoch` | 96 | **24** | (48×16) / (8×4) = 24 rollout batches |

All other parameters unchanged: `group_size=16`, `unique_sample_num_per_epoch=48`, `num_train_timesteps=2`, `gradient_step_per_epoch=1`.

## Memory Estimate (per_device_batch_size=4)

| Component | Memory |
|---|---|
| transformer (5B, bf16) | ~10 GB |
| transformer_2 (5B, bf16, frozen) | ~10 GB |
| VAE | ~0.5 GB |
| EMA (LoRA rank=128) | ~0.5 GB |
| PickScore reward model | ~1.5 GB |
| Optimizer states (AdamW, LoRA only) | ~1 GB |
| Activations (batch_size=4, 720×1280) | ~20-30 GB |
| CUDA overhead | ~5-10 GB |
| **Total** | **~50-70 GB** |

Target: ~50-70GB out of 141GB — conservative with headroom for backward pass spikes.

## Throughput Improvement

- **Rollout**: 4× fewer inference calls → ~3-4× speedup
- **Optimize**: 4× fewer micro-batches (24 vs 96), each processing 4 samples → ~3-4× speedup
- **Overall**: ~3-4× more epochs per wall-clock hour

## Invariants Preserved

- Same total samples per epoch (768)
- Same optimizer steps per epoch (1)
- Same effective learning rate (same total gradient accumulation)
- Same advantage computation (same group_size, same prompts)
- Same model, LoRA config, resolution

## Risk Mitigation

- If `per_device_batch_size=4` OOMs during backward pass, fall back to 2 (~45-55GB)
- The Wan2 I2V inference method already supports batch_size > 1 (processes `len(prompt)` items)
- NFT optimize loop iterates over samples individually, so `per_device_batch_size` affects the outer batch dimension

## Implementation

Config-only change to `examples/nft/lora/wan22/i2v.yaml`:
1. Change `per_device_batch_size: 1` → `4`

`gradient_accumulation_steps: auto` will automatically recompute to 48 (24 batches × 2 timesteps).
`num_batches_per_epoch` is derived and will automatically become 24.

No code changes required.
