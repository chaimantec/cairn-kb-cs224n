# ZeRO and FSDP

**ZeRO** — the *Zero Redundancy Optimizer*, released by Microsoft as part of the DeepSpeed
project (≈18:48) — is the answer to [DDP](distributed-data-parallel.md)'s memory waste. DDP
replicates the model, the gradients and the optimizer state on every GPU; ZeRO shards them, so
each GPU holds a slice and the pieces are reassembled by communication when needed. It comes in
three stages, which shard progressively more, and the headline result from
[lecture 13](13-efficient-training.md) is that **the first two are free** — they cut memory
several-fold without moving a single extra byte.

**FSDP** (fully sharded data parallel) is ZeRO stage 3, under PyTorch's name for it (≈28:52).

## The three stages

With $\Psi$ the parameter count, $N_d$ the number of GPUs, and $K = 12$ the optimizer's bytes
per parameter (slides 31–39):

| Stage | Sharded | Memory consumed | $\Psi$ = 7.5B, $N_d$ = 64 |
|---|---|---|---|
| Baseline (DDP) | nothing | $(2+2+K)\Psi$ | **120 GB** |
| **Stage 1** ($P_{os}$) | optimizer state | $2\Psi + 2\Psi + \dfrac{K\Psi}{N_d}$ | **31.4 GB** |
| **Stage 2** ($P_{os+g}$) | + gradients | $2\Psi + \dfrac{(2+K)\Psi}{N_d}$ | **16.6 GB** |
| **Stage 3** ($P_{os+g+p}$) | + parameters | $\dfrac{(2+2+K)\Psi}{N_d}$ | **1.9 GB** |

Note what stays un-sharded at each stage. At stage 1 every GPU still has all the parameters
($2\Psi$) and its own full gradient ($2\Psi$); only the 12-byte optimizer state is divided. At
stage 3 nothing is replicated, and the whole 16 bytes per parameter is divided by $N_d$.

## Stage 1 — optimizer state sharding

Each GPU keeps the full FP16 parameters and computes a full gradient on its own data slice, but
holds only **a shard of the optimizer state**, and is responsible for updating only the
parameters in that shard (slide 32). A step runs (≈19:33–21:06, slide 33):

1. Each worker computes the gradient on its subset of the data.
2. A [reduce-scatter](collective-communication.md) gives each worker the *complete* gradient for
   its own parameter shard — GPU 0 sends GPU 1 the chunk GPU 1 owns, and so on.
3. Each worker updates its shard, using its slice of the optimizer state.
4. An **all-gather** redistributes the updated parameters so every GPU is synchronized again.

If you have a network with eight parameters on four GPUs, each GPU maintains optimizer state for
two parameters instead of all eight, updates those two, and gets the other six back from its
peers (≈21:56–22:42).

**Why this is free.** An all-reduce *is* a reduce-scatter followed by an all-gather (slide 34).
DDP pays for one all-reduce; stage 1 pays for a reduce-scatter plus an all-gather. Same
communication, one quarter the memory. The lecturer's advice is unhedged: "we basically saved
memory for free… so you should just always use this" (≈24:14).

## Stage 2 — also shard the gradients

The complication is that each worker still needs the full gradient for its own data slice, but no
longer has memory to hold it (slide 35). The solution is to **never instantiate the full gradient
vector** (slide 36).

The backward pass proceeds layer by layer. When the worker reaches layer $j$, it takes the
upstream gradient, computes the gradient for layer $j$'s parameters, **immediately sends it to
whichever worker owns layer $j$** (a [reduce](collective-communication.md)), and deallocates the
temporary buffer (≈25:01–26:34, slide 37). Then each owner updates its shard and an all-gather
resynchronizes (slide 38).

A reduce plus an all-gather is again the same total as DDP's all-reduce, so **stage 2 is also
free** (≈27:21). Stages 1 and 2 together are the "just do it" recommendation in the lecture's
flowchart.

## Stage 3 / FSDP — shard the parameters too

When even the model will not fit, shard the parameters as well (slide 39). This is the point
where the free lunch ends: slide 40 states the caveat outright, because a GPU that holds no
complete layer must fetch one before it can use it.

**Setup.** The model is divided into **FSDP units** — groups of consecutive layers (slide 42).
Each unit's parameters are flattened into a single **FlatParameter**: on slide 43 a linear
layer's 12-entry weight matrix and 3-entry bias are concatenated into one 15-element array, plus
a padding slot so it divides evenly, and sharded across 16 ranks.

**Forward pass** (≈30:24, slide 44): all-gather the full parameters for the unit, run the forward
pass, then discard the parameter shards you do not own.

**Backward pass** (≈31:11, slide 45): all-gather the unit's parameters again, compute each GPU's
gradient on its own data, then reduce-scatter so the full gradient for each shard reaches the GPU
that owns it. Each GPU then updates its own shard (slide 46).

So the total is **two all-gathers plus a reduce-scatter**, against DDP's single all-reduce
(≈34:15, slide 48).

## Why the overhead is smaller than it looks

The all-gather for the *next* unit is prefetched while the current unit's forward pass is still
running (≈55:13). Slide 48's timeline shows three tracks — CPU, GPU compute stream and GPU
communication stream — with AG1 overlapping FWD0, AG2 overlapping FWD1, and parameter-free
(memory release) blocks interleaved. Communication hides behind computation, so for a
sufficiently large model the overhead is "really not that bad". This is the diagram a student
asks to have walked through, and the walkthrough at ≈55:13–59:04 is the clearest part of the
lecture on FSDP.

The same overlapping happens in the backward pass, where reduce-scatters hide behind backward
computation (≈58:18).

## Two practical wrinkles

**FSDP unit 0 is never freed.** It stays resident rather than being released like the others —
an implementation detail of FSDP, visible in the timeline (≈59:04).

**The sharding policy is architecture-specific, and it matters.** PyTorch's FSDP wrapper requires
you to supply one, and different ways of grouping layers into units give different communication
overheads — it makes sense to put consecutive layers in the same unit, so that one all-gather
loads several layers at once rather than communicating constantly (≈59:51, ≈1:01:26). Because
everyone uses Transformers, well-tuned policies exist for them. The warning for CS224N students
is explicit: if your final project invents a new architecture — "subquadratic attention,
whatever" — it may not run efficiently simply because no good sharding policy exists for it
(≈1:00:38).

A student asks whether the discarded weights are cached or re-streamed each time. The answer:
there may be some caching in the system, but to the user it behaves as though everything is
thrown away and streamed back per layer — which is precisely why the sharding policy matters
(≈1:01:26).

## What ZeRO does not solve

Sharding attacks parameters, gradients and optimizer state. It does **not** touch
[activations](gpu-memory-for-training.md), which scale with the batch size and are frequently
what actually triggers the out-of-memory error (≈35:50). For those you need gradient/activation
checkpointing, or, failing that, [LoRA](lora.md).

## Related pages

- [Lecture 13 — Efficient training](13-efficient-training.md)
- [Distributed data parallel](distributed-data-parallel.md) — the baseline this improves on
- [Collective communication](collective-communication.md) — the four operations, and the identity
  that makes stages 1 and 2 free
- [GPU memory for training](gpu-memory-for-training.md) — where the 16 bytes per parameter come from
- [Mixed precision training](mixed-precision-training.md)
- [LoRA](lora.md) — the next move when stage 3 is still not enough
