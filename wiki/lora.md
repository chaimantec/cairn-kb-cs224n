# LoRA (low-rank adaptation)

LoRA is the [parameter-efficient finetuning](parameter-efficient-finetuning.md) method
[lecture 13](13-efficient-training.md) develops in full. Instead of learning a full-rank update
to each weight matrix, it constrains the update to be the product of two thin matrices — so you
train a few million parameters rather than a few hundred billion, store a few megabytes per
task, and pay nothing at inference time.

## The observation it rests on

When large language models are finetuned, the updates to the weights have a low **intrinsic
rank** (Aghajanyan et al. 2020, cited on slide 59). The lecturer puts it as: look at the
geometric structure of the gradients and they tend to be low-rank (≈45:04). If the update you
are looking for is effectively low-rank anyway, there is no reason to search the full-rank space
for it.

## The method

Take a pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ and **freeze it**. Constrain
its update to a low-rank decomposition (slide 59):

$$W_0 + \Delta W = W_0 + \alpha BA$$

where

- $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ are the only **trainable**
  parameters,
- $r \ll \min(d, k)$ is the **rank**, the knob controlling capacity,
- $\alpha$ is a scalar trading off the knowledge already in the pretrained model against the
  task-specific knowledge you are adding.

Concretely, the input $x$ (of width $d$) flows down two paths — through the frozen $W_0$, and
through the narrow $A \rightarrow B$ bottleneck — and the two outputs are summed to give $h$.
The bottleneck is the point: information passing through the adapter is squeezed through $r$
dimensions.

The parameter saving is $dk$ against $r(d+k)$. For a 768×768 matrix at $r = 8$ that is 589,824
against 12,288, about 48×.

> **LoRA also appears in lecture 9, with different letters.** That deck (slide 34) writes the
> correction as $W + AB$ with $A \in \mathbb{R}^{d \times k}$, $B \in \mathbb{R}^{k \times d}$
> and rank $k$ — so $A$ and $B$ are swapped relative to this page and the rank has a different
> name. Same method, each page in its own lecture's notation. See
> [pretraining and fine-tuning](pretraining-and-finetuning.md#full-versus-parameter-efficient-fine-tuning).

### What $\alpha$ does

Think of it as how much you are willing to disturb the model (≈46:40, ≈49:46):

- $\alpha = 0$ — no change at all.
- $\alpha$ small — you want only a small amount of task-specific knowledge added.
- $\alpha = 1$ — an even trade between pretrained and task-specific knowledge. **This is what
  people typically use.**
- $\alpha > 1$ — appropriate if the task is something the pretrained model has no idea about.

## In code

The whole method is a couple of lines in the forward pass (slide 61):

```python
input_dim = 768   # e.g., the hidden size of the pre-trained model
output_dim = 768  # e.g., the output size of the layer
rank = 8          # the rank 'r' for the low-rank adaptation

W = ...  # from pretrained network, shape input_dim x output_dim

W_A = nn.Parameter(torch.empty(input_dim, rank))   # LoRA weight A
W_B = nn.Parameter(torch.empty(rank, output_dim))  # LoRA weight B

nn.init.kaiming_uniform_(W_A, a=math.sqrt(5))
nn.init.zeros_(W_B)

def regular_forward_matmul(x, W):
    h = x @ W
    return h

def lora_forward_matmul(x, W, W_A, W_B):
    h = x @ W                        # regular matrix multiplication
    h += x @ (W_A @ W_B) * alpha     # use scaled LoRA weights
    return h
```

Note `nn.init.zeros_(W_B)`: because $B$ starts at zero, $BA = 0$ at initialization, so the
adapted model *begins* exactly as the pretrained one. You freeze the model parameters, compute
$h$ as before, and add the offset term — and that is all, repeated for every weight matrix in
every layer you choose to adapt (≈49:00).

## Two properties that make it practical

**$r$ is a slider up to full finetuning.** As you increase the number of trainable parameters,
training LoRA converges towards training the original model (slide 60, ≈47:27). You are not
choosing a fundamentally weaker method, only a cheaper point on a continuum.

**No additional inference latency.** Because the update is a plain additive term, it can be
folded into the weights. To switch tasks, recover $W_0$ by subtracting $BA$ and add a different
$B'A'$ (slide 60, ≈48:14). Contrast adapters, which insert extra layers into the forward pass
and therefore cost something at every inference. Storing these small matrices per task is also
far cheaper than storing a full $\Delta\phi$.

## Results

On GPT-2 medium and large, on the E2E NLG Challenge (slide 62), LoRA on GPT-2 M with **0.35M**
trainable parameters scores BLEU 70.4, against full finetuning's **354.92M** parameters scoring
68.2 — better, with a thousandth of the trainable parameters.

On GPT-3 (slide 63):

| Model & Method | Trainable params | WikiSQL Acc. (%) | MNLI-m Acc. (%) | SAMSum R1/R2/RL |
|---|---|---|---|---|
| GPT-3 (FT) | 175,255.8M | 73.8 | 89.5 | 52.0/28.0/44.5 |
| GPT-3 (BitFit) | 14.2M | 71.3 | 91.0 | 51.3/27.4/43.5 |
| GPT-3 (PreEmbed) | 3.2M | 63.1 | 88.6 | 48.3/24.2/40.5 |
| GPT-3 (PreLayer) | 20.2M | 70.1 | 89.5 | 50.8/27.3/43.5 |
| GPT-3 (Adapter$^H$) | 7.1M | 71.9 | 89.8 | 53.0/28.9/44.8 |
| GPT-3 (Adapter$^H$) | 40.1M | 73.2 | 91.5 | 53.2/29.0/45.1 |
| **GPT-3 (LoRA)** | **4.7M** | **73.4** | **91.7** | **53.8/29.8/45.9** |
| GPT-3 (LoRA) | 37.7M | 74.0 | 91.6 | 53.4/29.2/45.1 |

LoRA matches or exceeds the finetuning baseline on all three datasets. Sometimes it is
*better* than full finetuning, which the lecturer attributes to the regularizing effect of
updating only a small subset (≈51:18).

## The two hyperparameters

Slide 64 answers both with ablations at a fixed budget of 18M trainable parameters.

**Which weight matrices?** Apply LoRA to the self-attention weight matrices — and specifically
to the **query and value** projections (≈52:05):

| Weight type | $W_q$ | $W_k$ | $W_v$ | $W_o$ | $W_q, W_k$ | $W_q, W_v$ | $W_q, W_k, W_v, W_o$ |
|---|---|---|---|---|---|---|---|
| Rank $r$ | 8 | 8 | 8 | 8 | 4 | 4 | 2 |
| WikiSQL (±0.5%) | 70.4 | 70.0 | 73.0 | 73.2 | 71.4 | **73.7** | **73.7** |
| MultiNLI (±0.1%) | 91.0 | 90.8 | 91.0 | 91.3 | 91.3 | 91.3 | **91.7** |

Adapting both $W_q$ and $W_v$ gives the best performance overall — and note that spreading the
same budget across two matrices at rank 4 beats concentrating it in one at rank 8.

**What rank?** A very small one:

| | Weight type | $r=1$ | $r=2$ | $r=4$ | $r=8$ | $r=64$ |
|---|---|---|---|---|---|---|
| WikiSQL (±0.5%) | $W_q$ | 68.8 | 69.6 | 70.5 | 70.4 | 70.0 |
| WikiSQL (±0.5%) | $W_q, W_v$ | 73.4 | 73.3 | 73.7 | 73.8 | 73.5 |
| WikiSQL (±0.5%) | $W_q, W_k, W_v, W_o$ | 74.1 | 73.7 | 74.0 | 74.0 | 73.9 |
| MultiNLI (±0.1%) | $W_q$ | 90.7 | 90.9 | 91.1 | 90.7 | 90.7 |
| MultiNLI (±0.1%) | $W_q, W_v$ | 91.3 | 91.4 | 91.3 | 91.6 | 91.4 |
| MultiNLI (±0.1%) | $W_q, W_k, W_v, W_o$ | 91.2 | 91.7 | 91.7 | 91.5 | 91.4 |

Rank 1 is already competitive: $W_q, W_v$ scores 73.4 at $r=1$ and 73.5 at $r=64$ on WikiSQL.
Whatever structure the update needs, it is genuinely low-dimensional — which is the original
observation, confirmed. As the lecturer notes, this is much smaller than the hidden-state
dimensions of most models (≈52:05).

## What to actually do

The lecture's parting recommendation (≈54:25):

- Apply LoRA to the **query matrix and the value matrix**.
- Set **rank = 8**.
- Set **alpha = 1**.

"That's a good starting point… just do that and you should be good to go."

## Related pages

- [Parameter-efficient finetuning](parameter-efficient-finetuning.md) — the wider family and the case for it
- [Lecture 13 — Efficient training](13-efficient-training.md)
- [Self-attention](self-attention.md) and [Transformer](transformer.md) — where $W_q$ and $W_v$ live
- [GPU memory for training](gpu-memory-for-training.md) — why freezing saves so much more than it seems
- [Pretraining and fine-tuning](pretraining-and-finetuning.md) — LoRA's first, briefer appearance in lecture 9
- [ZeRO and FSDP](zero-and-fsdp.md) — what you try before this
