# Lecture 13 — Efficient training

The systems lecture. Shikhar Murty opens by saying it "has nothing to do with natural language
at all, but hopefully it's going to be useful for final projects" (≈0:06), and that is exactly
what it is: how to get a model that does not fit on your GPU to train on the GPUs you actually
have. It comes in three parts — [mixed precision training](mixed-precision-training.md),
multi-GPU training with [DDP](distributed-data-parallel.md) and
[ZeRO/FSDP](zero-and-fsdp.md), and [parameter-efficient finetuning](parameter-efficient-finetuning.md)
via [LoRA](lora.md) (slide 2).

Everything in it converges on one flowchart, which appears twice (slides 50 and 65) and which
the lecturer tells students to wake up for if they slept through the rest (≈52:52). It is
reproduced at the end of this page.

**Slide-by-slide text of this deck: [65 slides](../raw/slides/13-efficient-training.md)** —
printed slide numbers match PDF pages 1:1.

Slides PDF: [Lecture 12 — efficient neural network training](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture12-training-shikhar.pdf) ·
[Full transcript](../raw/transcripts/13-efficient-training.md)

> **A note on numbering.** This lecture sits at **position 13** in the playlist this knowledge
> base follows; the video title and the deck both call it "Lecture 12", and the lecturer opens
> with "welcome to lecture 12". Repo files use the position. The lecturer is **Shikhar Murty**,
> a PhD student in the Stanford NLP group, who also gives the later Reasoning and Agents lecture.

## Part 1 — Floating point, and why a neural network cares

The lecture starts further down than most students expect: with how a number is stored. An
FP32 float is 32 bits — one sign bit, **8 exponent bits**, and **23 mantissa bits** — so every
parameter costs **4 bytes** of GPU memory (≈2:25, slides 4–5). The value it denotes is

$$(-1)^B \times 2^{E-127} \times \left(1 + \sum_{i=1}^{23} b_{23-i} 2^{-i}\right)$$

and slide 7 labels the two halves of that formula with the words the rest of the lecture uses:
the exponent term is **range**, the mantissa sum is **precision**. More exponent bits means you
can reach smaller and larger magnitudes; more mantissa bits means you can tell nearby numbers
apart.

FP16 halves the memory by cutting both: **5 exponent bits and 10 mantissa bits** (slide 8). So
the obvious response to a CUDA out-of-memory error — cast everything to FP16 — costs you both
range and precision at once (≈3:59), and both losses bite:

- **Range.** Very small numbers flush to zero and very large ones become NaNs. `torch.finfo`
  reports FP16's smallest normal as `6.10352e-05` (slide 10); anything below that is zero. The
  NVIDIA histogram on slide 11 shows the actual distribution of activation gradients during
  training, and **more than half of them fall below the FP16 representable range** (≈5:33).
  Gradients silently becoming zero is not a rounding detail — it is training that quietly stops.
- **Precision.** With 10 mantissa bits, `1.0001` rounds to `1` (slide 10), so weight updates
  are imprecise (slide 12).

## Mixed precision, in two moves

The fix, developed across slides 13–16, keeps FP16 for the expensive parts and FP32 for the
part that must be exact. **Master weights** (≈6:19): hold an FP32 copy of the parameters,
run the forward and backward passes in FP16, then upcast the gradient to FP32, update the FP32
master weights, and copy back down.

That fixes precision but not range — the gradients were computed in FP16, so they underflowed
before you ever upcast them (slide 15). The second move is **loss scaling**: multiply the loss
by a large constant before the backward pass, which shifts the whole gradient distribution
right, out of the underflow region; then divide it back out after upcasting to FP32 (≈8:35,
slide 16). In PyTorch this is `GradScaler` plus an `autocast` context (slide 17).

The full story, with the recipe and the PyTorch code, is on
[mixed precision training](mixed-precision-training.md).

**bfloat16** removes the need for the scaling entirely. It spends **8 exponent bits** — the
same dynamic range as FP32 — and keeps only **7 mantissa bits** (slide 20), on the bet that
neural network training needs range far more than it needs precision. It turns out to be a good
bet, and the code loses the `GradScaler` (slide 21). The caveat is hardware: bfloat16 needs a
recent NVIDIA architecture — H100, A100, A6000 (≈11:49).

Slide 22 is the empirical case, fine-tuning DistilBERT for sentiment classification on a single
A100:

| Precision | Training speed (3 epochs) | Predictive acc. (test set) | Memory allocated |
|---|---|---|---|
| Float64 | 24.59 min | 92.14% | 10.42 GB |
| Float32 | 21.75 min | 89.92% | 5.37 GB |
| Float16 | 5.23 min | 50.08% | 2.87 GB |
| Float16-mixed | 7.25 min | 92.15% | 4.31 GB |
| Bfloat16-mixed | 7.45 min | 92.61% | 4.46 GB |

Read the third row first: **pure FP16 collapses to 50.08% accuracy** — that is the underflow
problem, not a subtle degradation. Both mixed rows recover full accuracy at about a third of
FP32's time and memory. Bfloat16-mixed is fractionally the *best* of all five on accuracy, which
the lecturer attributes to a regularizing effect from the reduced precision (≈13:22). The speedup
itself comes from matrix multiplies being faster in half precision.

## Part 2 — What actually occupies GPU memory

Before multi-GPU training makes sense, you need the memory accounting — and the lecturer flags
that his first version of it is "somewhat of a lie" he will fix later (≈14:10). Per parameter,
with mixed precision and Adam (slides 24, 31):

| Item | Precision | Bytes/parameter |
|---|---|---|
| Model parameters | FP16 | 2 |
| Gradients | FP16 | 2 |
| Master weights | FP32 | 4 |
| Adam momentum $m_t$ | FP32 | 4 |
| Adam variance $v_t$ | FP32 | 4 |

That last block is the surprise: the **optimizer costs 12 bytes per parameter**, three times the
model itself, because Adam carries a momentum and a variance term per parameter and both live in
FP32 (≈14:57, ≈18:03). Slide 31 writes the total as $(2+2+K)\Psi$ with $K = 12$, and for a
$\Psi = 7.5\text{B}$-parameter model that is **120 GB on every single GPU**.

The lie, fixed at slide 49, is that this leaves out **model activations**, which must be kept
for the backward pass and which **scale with the batch size** (≈35:02). That is what actually
stops you raising the batch size, and — importantly — none of the sharding techniques in this
lecture help with it (≈35:50).

Details on [GPU memory for training](gpu-memory-for-training.md).

## Distributed data parallel, and its memory problem

[DDP](distributed-data-parallel.md) is the baseline: every GPU holds a full synchronized copy
of the model and optimizer, the *dataset* is split, each GPU computes gradients on its own
slice, and an **all-reduce** merges the gradients so every optimizer sees the sum (slides
25–30). Communication is 2 bytes per parameter, since the gradients are FP16 (≈16:31).

It works and it scales badly: that 120 GB is replicated on every GPU (slide 31). Nothing about
holding four identical copies of the optimizer state is necessary.

## ZeRO: shard the state instead of replicating it

ZeRO — **Zero Redundancy Optimizer**, from Microsoft's DeepSpeed project (≈18:48) — shards the
replicated state across GPUs in three escalating stages (slides 31–48):

| Stage | What is sharded | Memory ($\Psi$ = 7.5B, $N_d$ = 64) |
|---|---|---|
| Baseline (DDP) | nothing | $(2+2+K)\Psi$ = **120 GB** |
| **Stage 1** ($P_{os}$) | optimizer state | $2\Psi + 2\Psi + \frac{K\Psi}{N_d}$ = **31.4 GB** |
| **Stage 2** ($P_{os+g}$) | + gradients | $2\Psi + \frac{(2+K)\Psi}{N_d}$ = **16.6 GB** |
| **Stage 3** ($P_{os+g+p}$) | + parameters | $\frac{(2+2+K)\Psi}{N_d}$ = **1.9 GB** |

The striking result is that **stages 1 and 2 are free**. The identity that makes it work is
that an all-reduce is exactly a reduce-scatter followed by an all-gather (≈23:27, slide 34) —
which is precisely the communication stage 1 needs. So you get a 4× memory reduction for the
same bytes on the wire, and the lecturer's advice is blunt: "you should just always use this"
(≈24:14). Stage 2 shards gradients too by never instantiating the full gradient vector — each
layer's gradient is sent to its owning GPU the moment it is computed in the backward pass, then
the memory is freed (≈25:01, slides 36–37) — and costs a reduce plus an all-gather, again the
same total (≈27:21).

**Stage 3 is not free.** Also called **FSDP** (fully sharded data parallel), it shards the
parameters themselves, so a GPU no longer holds any complete layer and must all-gather one
before it can use it — in the forward pass *and* again in the backward pass, plus a
reduce-scatter to route gradients. That is two all-gathers and a reduce-scatter against DDP's
single all-reduce (≈34:15, slide 48). You do it when the model does not otherwise fit.

The overhead is smaller in practice than it looks, because the all-gather for the next FSDP unit
is **prefetched during the current unit's forward pass** — slide 48's timeline shows AG1
overlapping FWD0 (≈55:13). This is the diagram a student asks him to walk through step by step
at ≈54:25, and the walkthrough runs to ≈59:04.

Two practical wrinkles come out of that walkthrough: FSDP unit 0 is never freed, an
implementation detail of FSDP (≈59:04); and the **sharding policy is architecture-specific**.
PyTorch's FSDP wrapper requires one, Transformers have well-tuned policies because everyone uses
them, and a novel architecture in a final project may simply not run efficiently for want of a
good policy (≈59:51–1:00:38).

Full treatment on [ZeRO and FSDP](zero-and-fsdp.md); the four operations these stages are built
from are on [collective communication](collective-communication.md).

## Part 3 — When it still does not fit: parameter-efficient finetuning

If mixed precision, ZeRO stage 3, batch size 1 and gradient checkpointing all fail, you stop
updating every parameter (≈37:22). In **full finetuning** you update all of $\phi_o$; GPT-3 has
175 billion parameters, and you must store a separate full copy per task (slide 57). Since
$|\Delta\phi| = |\phi_o|$, that is both a training and a storage problem.

Beyond necessity, the lecture gives two further arguments (slides 52–55). Modern models are
heavily overparameterized, so finetuning a small subset can *generalize better* on a small
dataset (≈38:53). And there is a resource argument: slide 53 charts training compute for the
largest AI models doubling every ~3.4 months against global compute capacity doubling every
~1.5 years, with the two curves crossing around 2026 — with the consequence that AI development
concentrates in the best-funded organizations, and their value systems get embedded in the
models (≈39:40–40:26). Slide 54 adds the environmental cost, citing Cornell scientists' 2021
estimate that training GPT-3 emitted the equivalent of running a coal power plant for 10
straight hours, and slide 55 brings it home with CS234: had every student used the more
efficient of two equally good RL algorithms, the class would have saved 880 kilowatt-hours,
about a US household's monthly consumption (≈41:58).

## LoRA

[LoRA](lora.md) — low-rank adaptation — rests on the observation that weight updates during
adaptation have a low "intrinsic rank" (Aghajanyan et al. 2020, slide 59). So rather than learn
a full-rank update, constrain it to a product of two thin matrices. For a pretrained weight
matrix $W_0 \in \mathbb{R}^{d \times k}$, frozen:

$$W_0 + \Delta W = W_0 + \alpha BA$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d,k)$.
Only $A$ and $B$ are trainable. The scalar $\alpha$ trades off pretrained knowledge against
task-specific knowledge — set it to 1 for an even trade, higher if the task is something the
model has never seen, lower if you want to disturb it as little as possible (≈49:46).

Two properties make it practical. As $r$ grows, LoRA converges towards full finetuning, so $r$
is a slider (≈47:27). And there is **no additional inference latency**: because the update is
just an additive term, you swap tasks by subtracting $BA$ and adding $B'A'$ (slide 60, ≈48:14).

The results justify the defaults. On GPT-3, LoRA with **4.7M trainable parameters** scores 73.4
on WikiSQL and 91.7 on MNLI-m, against full finetuning's **175,255.8M parameters** scoring 73.8
and 89.5 — matching or beating it on all three datasets at roughly 1/37,000th of the trainable
parameters (slide 63). The ablations on slide 64 give the two hyperparameters: adapting **$W_q$
and $W_v$** beats adapting any single matrix, and rank is remarkably cheap — on WikiSQL,
$W_q, W_v$ scores 73.4 at $r=1$ and 73.5 at $r=64$, so a very small $r$ is already competitive
(≈52:05).

## The flowchart

Slides 50 and 65, and the lecturer's own summary of the whole lecture (≈52:52, ≈54:25):

1. **Always use mixed precision training.** You barely ever see a hit in accuracy or F1.
2. **Always use bfloat16** if `torch.cuda.is_bf16_supported()` — i.e. on Ampere-generation
   hardware (H100, A100, A6000).
3. **Does batch size 1 fit on a single GPU?**
   - **Yes** → try a larger batch size, and use **ZeRO stage 2**. It is free; there is no reason
     not to.
   - **No** → try **ZeRO stage 3 (FSDP)**, and gradient/activation checkpointing.
4. **Still out of memory?** → **use LoRA**, with the query and value matrices, **rank 8** and
   **alpha 1** as a starting point.

Slide 65 compresses it further: "start with Llama 7B + bfloat16 + ZeRO Stage-3 (or FSDP) +
LoRA."

One scoping caveat, from a student question at ≈53:39: **all of the multi-GPU material assumes
more than one GPU.** On a single GPU you are left with quantization, and the lecturer doubts you
can finetune the larger models at all that way.

## Related pages

- [Mixed precision training](mixed-precision-training.md) — FP16, bfloat16, master weights, loss scaling
- [GPU memory for training](gpu-memory-for-training.md) — the per-parameter accounting and what blows it up
- [Collective communication](collective-communication.md) — all-reduce, reduce-scatter, all-gather, reduce
- [Distributed data parallel](distributed-data-parallel.md) — the baseline multi-GPU setup
- [ZeRO and FSDP](zero-and-fsdp.md) — sharding optimizer state, gradients and parameters
- [Parameter-efficient finetuning](parameter-efficient-finetuning.md) — the family, and why it exists
- [LoRA](lora.md) — low-rank adaptation in full
- [Final project guidance](final-project-guidance.md) — the lecture is explicitly aimed at final projects
- [Pretraining and fine-tuning](pretraining-and-finetuning.md) — where finetuning was introduced
- [Transformer](transformer.md) and [self-attention](self-attention.md) — the $W_q$/$W_v$ matrices LoRA adapts
