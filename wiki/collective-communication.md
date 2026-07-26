# Collective communication

Multi-GPU training is mostly a question of which bytes move between which GPUs and when. The
operations that move them are the MPI **collectives**, and
[lecture 13](13-efficient-training.md) introduces four of them — one at a time, each at the
moment a training scheme needs it. Understanding [DDP](distributed-data-parallel.md) and
[ZeRO/FSDP](zero-and-fsdp.md) is largely a matter of knowing these four and one identity
relating them.

Throughout, imagine four GPUs (ranks 0–3), each holding its own array of values.

## All-reduce

Every rank contributes a value; the values are combined (summed); **every rank receives the
result**.

```
rank0: in0   rank1: in1   rank2: in2   rank3: in3
                  ↓
rank0: out   rank1: out   rank2: out   rank3: out        out[i] = sum(inX[i])
```

This is the one DDP uses (slide 28). Each GPU computes a gradient on its own slice of data, and
the all-reduce gives all of them the summed gradient, so every optimizer applies the same update
and the models stay synchronized. Communication cost is **2 bytes per parameter**, since the
gradients are FP16 (≈16:31).

## Reduce-scatter

Values are combined as in an all-reduce, but each rank receives **only its own shard** of the
result rather than the whole thing.

```
rank0: [in0 in1 in2 in3]   …each rank holds a full array…
                  ↓
rank0: out0    rank1: out1    rank2: out2    rank3: out3
                                          outY[i] = sum(inX[Y*count+i])
```

Used by [ZeRO stage 1](zero-and-fsdp.md) (slide 33): each worker computes the full gradient for
its data, then reduce-scatters so each ends up with the complete gradient for exactly the
parameters it is responsible for updating (≈20:20).

## All-gather

No combining. Each rank starts with one piece; **every rank ends up holding all the pieces**,
concatenated.

```
rank0: in0    rank1: in1    rank2: in2    rank3: in3
                  ↓
rank0: [in0 in1 in2 in3]   …and identically on every other rank…
                                          out[Y*count+i] = inY[i]
```

Used to restore synchronization after each GPU has updated its own shard of the parameters
(≈21:06, slide 33), and in FSDP to materialize a layer's full parameters before using it.

## Reduce

Like all-reduce, but the combined result lands on **one** rank — the root — rather than all of
them.

```
rank0: in0   rank1: in1   rank2: in2   rank3: in3
                  ↓
              rank2: out   (root)          out[i] = sum(inX[i])
```

This is what [ZeRO stage 2](zero-and-fsdp.md) uses (slide 37): as the backward pass produces the
gradient for layer $j$, it is sent straight to the single GPU that owns layer $j$, and the local
copy is deallocated (≈26:34).

## The identity that makes ZeRO free

The single most useful fact in the lecture:

$$\text{all-reduce} = \text{reduce-scatter} \;+\; \text{all-gather}$$

An all-reduce is *implemented* as a reduce-scatter followed by an all-gather (slide 34, ≈23:27).

This is why ZeRO stages 1 and 2 cost nothing. DDP already pays for one all-reduce. Stage 1 needs
a reduce-scatter (to give each GPU the gradient for its shard) and an all-gather (to
resynchronize parameters afterwards) — which is *the same communication*, just with the
intermediate step exposed rather than hidden. So you shard the optimizer state, cut memory
roughly 4×, and put no extra bytes on the wire (≈24:14). Stage 2 substitutes a reduce for the
reduce-scatter and is likewise free (≈27:21).

[ZeRO stage 3 / FSDP](zero-and-fsdp.md) is where the identity stops helping: it needs an
all-gather in the forward pass, another in the backward pass, and a reduce-scatter for the
gradients — two all-gathers and a reduce-scatter against DDP's single all-reduce, which is
genuinely more (≈34:15, slide 48).

## Summary

| Operation | Combines? | Who gets the result | Used by |
|---|---|---|---|
| **All-reduce** | yes | everyone | [DDP](distributed-data-parallel.md) |
| **Reduce-scatter** | yes | each rank gets its own shard | ZeRO stage 1, FSDP backward |
| **All-gather** | no | everyone gets everything | ZeRO stages 1–3, FSDP forward |
| **Reduce** | yes | one root rank | ZeRO stage 2 |

## Related pages

- [Lecture 13 — Efficient training](13-efficient-training.md)
- [Distributed data parallel](distributed-data-parallel.md)
- [ZeRO and FSDP](zero-and-fsdp.md)
- [GPU memory for training](gpu-memory-for-training.md)
