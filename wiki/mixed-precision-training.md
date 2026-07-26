# Mixed precision training

Training a network in FP16 instead of FP32 halves the memory it takes and speeds up the matrix
multiplies, but done naively it destroys the model. Mixed precision is the recipe that keeps
the savings and avoids the destruction: run the expensive operations in half precision, keep
the numerically delicate one — the weight update — in full precision. It is the first thing
[lecture 13](13-efficient-training.md) tells you to do, and the advice is unconditional:
"always use mixed precision training… you barely ever see a hit in performance" (≈35:50).

## What a floating-point format buys you

A number in FP32 is 32 bits: one sign bit, **8 exponent bits**, **23 mantissa bits**. So each
parameter costs **4 bytes** (slides 4–5). The value is

$$(-1)^B \times 2^{E-127} \times \left(1 + \sum_{i=1}^{23} b_{23-i} 2^{-i}\right)$$

Slide 7 names the two parts, and the naming is what makes the rest of the topic legible:

- the exponent term $2^{E-127}$ is **range** — how small and how large a magnitude you can
  reach at all;
- the mantissa sum is **precision** — how finely you can distinguish numbers that are close
  together. Formally, between consecutive powers of two FP32 can represent

$$[2^k,\ 2^k(1+\epsilon),\ 2^k(1+2\epsilon),\ \ldots,\ 2^{k+1}] \quad \text{where } \epsilon = 2^{-23}$$

The three formats in this lecture differ only in how they split their bits:

| Format | Sign | Exponent (range) | Mantissa (precision) | Bytes |
|---|---|---|---|---|
| FP32 | 1 | 8 | 23 | 4 |
| FP16 | 1 | 5 | 10 | 2 |
| bfloat16 | 1 | **8** | 7 | 2 |

## Why casting everything to FP16 fails

FP16 cuts *both* range and precision (slide 8), and each loss causes a distinct failure
(slides 9–12).

**Gradients underflow.** `torch.finfo(torch.float16)` gives a smallest normal of
`6.10352e-05` (slide 10); anything smaller becomes exactly zero. This is not a corner case:
slide 11 plots the real distribution of activation gradient magnitudes during training, and
**more than half the gradients sit below FP16's representable range** (≈5:33). They do not
become slightly wrong — they become zero, and the parameters they belong to stop learning.
At the other end, large values become NaNs.

**Updates lose precision.** With 10 mantissa bits, `1.0001` rounds to `1` (slide 10). The
reported epsilon is `0.000976562` — the smallest number you can add to 1 and still see a
change. Since a weight update is typically a tiny quantity added to a much larger weight, this
is exactly the arithmetic FP16 is worst at.

## The recipe

Slides 13–16 build the answer in two moves.

**Move 1 — master weights** (≈6:19). Keep an FP32 copy of the parameters, called the master
weights, and cast down for the compute:

1. Maintain a copy of model parameters in FP32 (master weights)
2. Run forward pass in FP16
3. Compute gradient in FP16
4. Copy gradient into FP32
5. Update master weights in FP32
6. Copy into the FP16 version

This fixes the update-precision problem — the addition now happens in FP32 — but not the range
problem, because the gradients were still *computed* in FP16 and underflowed before step 4 ever
ran (slide 15).

**Move 2 — loss scaling** (≈8:35). Before the backward pass, multiply the loss by a large
constant (say 1000). By the chain rule every gradient is scaled by the same factor, shifting the
whole distribution right, out of the underflow region. After upcasting to FP32, divide it back
out. The full recipe (slide 16) is:

1. Maintain a copy of model parameters in FP32 (master weights)
2. Run forward pass in FP16
3. **Scale loss by a large value**
4. Compute gradient in FP16
5. Copy gradient into FP32 **and divide by the scale factor**
6. Update master weights in FP32
7. Copy into the FP16 version

A student raises the obvious objection to master weights — copying 32-bit versions between
host and GPU is slow — and the answer is that the I/O can usually be overlapped with the
forward and backward passes, so it only bites for very small networks (≈7:04).

## In PyTorch

FP16 needs a `GradScaler` for the loss scaling and an `autocast` context for the forward pass
(slide 17):

```python
scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

`scaler.step()` unscales the gradients first, and **skips the optimizer step entirely if they
contain infs or NaNs**. That matters, because the scale factor is not free to choose: scale by
too much and you overflow into NaNs, so the scaler has to adapt it across iterations to the
network's own dynamics — which the lecturer calls out as the annoying part (≈9:22–10:08).

## bfloat16, which removes the scaling

The reason loss scaling exists is FP16's narrow range. So spend the bits differently: give the
exponent the **same 8 bits as FP32** and let precision take the loss, leaving **7 mantissa
bits**. That is bfloat16 — "brain float 16" — and it has FP32's dynamic range in half the
memory (≈11:03, slide 20). It turns out neural network training needs range far more than
precision, so this trade is almost free.

With bfloat16 there is no `GradScaler` and no scale/unscale bookkeeping (slide 21):

```python
for input, target in data:
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda"):
        output = model(input)
        loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

The one catch is hardware. bfloat16 needs a recent NVIDIA architecture — H100, A100 or A6000;
on an older GPU it is unavailable (≈11:49). Check with `torch.cuda.is_bf16_supported()`
(slide 50).

## What it is worth

Slide 22, finetuning DistilBERT for sentiment classification on a single A100:

| Precision | Training speed (3 epochs) | Predictive acc. (test set) | Memory allocated |
|---|---|---|---|
| Float64 | 24.59 min | 92.14% | 10.42 GB |
| Float32 | 21.75 min | 89.92% | 5.37 GB |
| Float16 | 5.23 min | 50.08% | 2.87 GB |
| Float16-mixed | 7.25 min | 92.15% | 4.31 GB |
| Bfloat16-mixed | 7.45 min | 92.61% | 4.46 GB |

The Float16 row is the whole argument for this page: pure half precision is the fastest and
smallest option and it **collapses to 50.08% accuracy**, which for binary sentiment
classification is chance. Both mixed rows recover full accuracy at roughly a third of FP32's
time and memory. Bfloat16-mixed is fractionally the most accurate of all five, including
Float64 — the lecturer attributes that to a regularizing effect from the reduced precision
(≈13:22). The speedup comes from half-precision matrix multiplies being faster, not from the
memory saving.

## Related pages

- [Lecture 13 — Efficient training](13-efficient-training.md), where this is Part 1
- [GPU memory for training](gpu-memory-for-training.md) — mixed precision is why parameters cost
  2 bytes and the optimizer 12
- [ZeRO and FSDP](zero-and-fsdp.md) — the next lever, once precision is dealt with
- [Gradient descent](gradient-descent.md) and [backpropagation](backpropagation.md) — what is
  being computed in FP16 here
