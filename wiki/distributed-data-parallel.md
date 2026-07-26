# Distributed data parallel (DDP)

The simplest way to use more than one GPU, and the baseline everything else in
[lecture 13](13-efficient-training.md) is measured against. The idea is in the name: the
**data** is distributed, the model is not.

## How it works

Every GPU holds a complete, synchronized copy of the model and its optimizer. The dataset is
split into as many slices as there are GPUs — with 1,000 examples and four GPUs, each gets 250
(≈32:42). Then, per step (slides 25–30):

1. **Start synchronized.** Every GPU has an identical copy of the model.
2. **Forward pass in parallel.** Each GPU runs its own slice, so each produces *different*
   activations.
3. **Backward pass.** Each GPU therefore produces a *different* gradient — they saw different
   data.
4. **Synchronize with an [all-reduce](collective-communication.md).** The four gradients are
   summed and the sum is distributed back to all four GPUs, so every optimizer now holds the
   gradient accumulated over the whole batch.
5. **Update.** Every optimizer applies the same update to the same weights, so the copies stay
   synchronized for the next step.

Step 4 is the only communication, and it costs **2 bytes per parameter**, because the gradients
are FP16 under [mixed precision](mixed-precision-training.md) (≈16:31). In practice the
all-reduce is overlapped with the backward pass — gradients for upstream layers are communicated
while the lower layers are still computing (slide 27).

## What it is good for, and where it breaks

DDP is straightforward and it parallelizes computation well. Its problem is memory, and the
problem is pure waste: **every GPU stores the full model, the full gradient and the full
optimizer state** (≈17:17). Nothing about that replication is necessary — the four copies of the
optimizer state are identical.

From the [per-parameter budget](gpu-memory-for-training.md), with Adam and mixed precision, that
is 16 bytes per parameter — written on slide 31 as $(2+2+K)\Psi$ with $K = 12$. For a 7.5-billion
parameter model:

| | Memory consumed | $\Psi$ = 7.5B, $N_d$ = 64 |
|---|---|---|
| Baseline (DDP) | $(2+2+K)\Psi$ | **120 GB** |

120 GB **on each of the 64 GPUs**, to train one 7.5B model. Adding GPUs does not help; it
replicates the problem.

That is the entire setup for [ZeRO](zero-and-fsdp.md), which keeps DDP's data-parallel structure
and shards the replicated state instead of copying it — reaching 1.9 GB per GPU on the same
model at stage 3, and, for stages 1 and 2, doing so **for no extra communication at all**.

## Related pages

- [Lecture 13 — Efficient training](13-efficient-training.md)
- [ZeRO and FSDP](zero-and-fsdp.md) — what to use instead, and why stage 2 is free
- [Collective communication](collective-communication.md) — the all-reduce that makes DDP work
- [GPU memory for training](gpu-memory-for-training.md) — where the 16 bytes per parameter come from
- [Mixed precision training](mixed-precision-training.md)
