# Parameter-efficient finetuning (PEFT)

Finetuning normally means updating every parameter in the model. Parameter-efficient finetuning
means **freezing almost all of them and training a small number of new or selected parameters
instead**. It is the last resort in [lecture 13](13-efficient-training.md)'s flowchart — what
you do when [mixed precision](mixed-precision-training.md),
[ZeRO stage 3](zero-and-fsdp.md), a batch size of 1 and gradient checkpointing have all failed
(≈37:22) — and also, on its own merits, often the better choice.

The lecture develops [LoRA](lora.md) as its worked example; this page is the surrounding case.

## Why not just finetune everything

**It does not fit.** In full finetuning you update all of $\phi_o$, so
$|\Delta\phi| = |\phi_o|$. GPT-3 has 175 billion parameters (slide 57). The
[memory budget](gpu-memory-for-training.md) makes the consequence concrete: a trainable
parameter costs 16 bytes (weight, gradient, master weight, Adam momentum, Adam variance), while
a frozen one costs 2. Freezing is worth about 8× — far more than the parameter count alone
suggests, because it deletes the gradient and the entire optimizer state, not just the weight.

**It does not store.** You learn a separate $\Delta\phi$ the size of the whole model for *every*
downstream task, which is "expensive and challenging for storing and deploying many independent
instances" (slide 57, ≈43:31).

**It may not even be better.** State-of-the-art models are massively overparameterized, so
with a small dataset, updating a small subset can *generalize better* than updating everything —
a regularizing effect the lecture observes again in LoRA's results (≈38:53, ≈51:18, slide 52).

## The general form

Encode the task-specific increment with a much smaller set of parameters $\Theta$ (slide 58):

$$\Delta\phi = \Delta\phi(\Theta), \qquad |\Theta| \ll |\phi_o|$$

so that finding $\Delta\phi$ becomes optimizing over $\Theta$:

$$\max_\Theta \sum_{(x,y)} \sum_{t=1}^{|y|} \log\left(P_{\phi_o+\Delta\phi(\Theta)}(y_t \mid x, y_{<t})\right)$$

against full finetuning's objective over all of $\phi$ (slide 56):

$$\max_\phi \sum_{(x,y)} \sum_{t=1}^{|y|} \log(P_\phi(y_t \mid x, y_{<t}))$$

You now search a much smaller space, the result is small enough to keep on disk per task, and it
should require less compute (≈44:17).

## The family

The lecture names several methods without developing them (≈49:46), and slide 63 benchmarks
them against each other on GPT-3:

| Method | What is trained | Trainable params (GPT-3) | WikiSQL | MNLI-m |
|---|---|---|---|---|
| Full finetuning | everything | 175,255.8M | 73.8 | 89.5 |
| **BitFit** | only the bias terms | 14.2M | 71.3 | 91.0 |
| PreEmbed | prefix embeddings | 3.2M | 63.1 | 88.6 |
| PreLayer | prefix at every layer | 20.2M | 70.1 | 89.5 |
| Adapter$^H$ | inserted adapter layers | 7.1M | 71.9 | 89.8 |
| Adapter$^H$ | inserted adapter layers | 40.1M | 73.2 | 91.5 |
| **LoRA** | low-rank update matrices | **4.7M** | **73.4** | **91.7** |
| LoRA | low-rank update matrices | 37.7M | 74.0 | 91.6 |

LoRA at 4.7M trainable parameters matches or beats full finetuning at 175 billion, and does so
with the fewest parameters of any method in the table. That is why the lecture spends its time
there. Slide 63 adds the scaling picture: across trainable-parameter budgets from roughly
$10^6$ to $10^{8.5}$, LoRA is the highest curve throughout on both WikiSQL and MultiNLI-matched,
while the prefix-based methods peak early and then *decline*.

Note that this KB's [pretraining and finetuning](pretraining-and-finetuning.md) page covers
prefix tuning and LoRA as they were introduced in lecture 9; this page and
[LoRA](lora.md) are the fuller treatment.

## The argument beyond necessity

Slides 53–55 make an efficiency-as-a-value case that goes past any single model fitting on any
single GPU (≈39:40–41:58):

- **Compute growth is unsustainable.** Training compute for the largest AI models doubles about
  every 3.4 months; global compute capacity doubles about every 1.5 years. Slide 53's chart has
  the two curves crossing around 2026.
- **Concentration follows cost.** As training costs rise, AI development concentrates in the
  best-funded organizations — and, the slide asks, *whose value systems* then get embedded in
  the AI of tomorrow?
- **The field optimizes for accuracy, not efficiency.** Slide 54 counts papers at three
  conferences by what they target. At ACL 2018: 16 accuracy, 0 efficiency. CVPR 2019: 13 vs 2.
  NeurIPS 2018: 9 vs 4.
- **There is an environmental cost.** Cornell scientists estimated in 2021 that training GPT-3
  emitted the equivalent of running a coal power plant for 10 straight hours (slide 54). Closer
  to home, slide 55 reports that in a Stanford reinforcement learning class (CS234) of more than
  200 students, two algorithms performed equally well but one used far more power — had everyone
  used the efficient one, the class would have saved **880 kilowatt-hours**, about what a typical
  American household uses in a month.

> The lecturer also gives a spoken figure for GPT-3's emissions — "1.1 million tons" — which he
> hedges himself as "or some such number" (≈41:12), and which does not appear on the slide. The
> deck's claim is the coal-power-plant comparison only. Prefer the slide.

## Related pages

- [LoRA](lora.md) — the method this lecture develops, in full
- [Lecture 13 — Efficient training](13-efficient-training.md), where this is Part 3
- [GPU memory for training](gpu-memory-for-training.md) — why freezing a parameter saves 8×
- [Pretraining and fine-tuning](pretraining-and-finetuning.md) — where finetuning was introduced
- [Instruction finetuning](instruction-finetuning.md) — a different axis: *what* you finetune on
- [ZeRO and FSDP](zero-and-fsdp.md) — what you try before reaching for PEFT
