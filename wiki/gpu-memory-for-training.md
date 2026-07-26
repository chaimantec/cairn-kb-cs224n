# GPU memory for training

"Can I train this model on this GPU?" is answered by a short piece of arithmetic that most
people meet only when they hit a CUDA out-of-memory error.
[Lecture 13](13-efficient-training.md) sets it out per parameter, and every technique in that
lecture is an attack on one line of the table.

## The per-parameter budget

Assume [mixed precision training](mixed-precision-training.md) with Adam — the standard setup
(slides 24, 31):

| Item | Precision | Bytes per parameter |
|---|---|---|
| Model parameters | FP16 | 2 |
| Gradients | FP16 | 2 |
| Master weights | FP32 | 4 |
| Adam momentum $m_t$ | FP32 | 4 |
| Adam variance $v_t$ | FP32 | 4 |
| **Total** | | **16** |

The line that surprises people is the optimizer. Adam keeps a momentum and a variance term for
**every** parameter, and because you are training in mixed precision both must be held in FP32
alongside the FP32 master weights — so the optimizer state is **12 bytes per parameter**, three
times the model's own 2 (≈14:57, ≈18:03). The lecturer's own reaction: "when I first saw this a
few years back, I was very surprised to see that optimizers also need memory" (≈14:10).

The Adam update equations that require this state (slide 24):

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla w_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla w_t)^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t+\epsilon}}\hat{m}_t$$

where $\nabla w_t$ is the minibatch gradient and $\eta$ the learning rate.

Slide 31 writes the total as $(2 + 2 + K)\Psi$, where $\Psi$ is the parameter count and
$K = 12$ is the optimizer's bytes per parameter. For $\Psi = 7.5$ billion that is **120 GB** —
and under plain [DDP](distributed-data-parallel.md) it is 120 GB on *every* GPU, which is the
entire motivation for [ZeRO](zero-and-fsdp.md).

## The term the first version leaves out

The lecturer flags his own accounting as "somewhat of a lie" when he introduces it (≈14:10)
and corrects it at slide 49. The missing term is **model activations** — the intermediate values
saved during the forward pass because the backward pass needs them.

$$\text{GPU memory} = \text{parameters} + \text{gradients} + \text{master weights} + \text{Adam momentum} + \text{Adam variance} + \textbf{activations}$$

Activations differ from everything else in the table in one crucial way: they **scale with the
batch size** (≈35:02). Every other line is fixed once you choose a model. This is why raising
the batch size is what triggers the out-of-memory error, and why the error moves when you change
it.

Two consequences worth remembering:

- Activations are stored in FP16 or bfloat16 under mixed precision, like the parameters.
- **None of the ZeRO stages shard activations** (≈35:50). Sharding attacks parameters,
  gradients and optimizer state; the activation term survives all three. The tool for that one
  is gradient (activation) checkpointing, which the lecture mentions as a step in the flowchart
  (≈37:22) but does not develop.

## How each technique attacks the table

| Technique | What it reduces |
|---|---|
| [Mixed precision](mixed-precision-training.md) | parameters and gradients, 4 bytes → 2 |
| [ZeRO stage 1](zero-and-fsdp.md) | optimizer state, divided by the number of GPUs |
| [ZeRO stage 2](zero-and-fsdp.md) | + gradients |
| [ZeRO stage 3 / FSDP](zero-and-fsdp.md) | + parameters |
| Gradient checkpointing | activations (recompute instead of store) |
| [LoRA](lora.md) | gradients *and* optimizer state — only $A$ and $B$ are trainable |

The last row is worth spelling out, because it is why
[parameter-efficient finetuning](parameter-efficient-finetuning.md) saves so much more than its
parameter count suggests. Freezing a parameter removes its gradient (2 bytes) and its entire
12-byte optimizer state, leaving only the 2-byte weight. The frozen model costs about an eighth
of what a trained one does.

## Related pages

- [Lecture 13 — Efficient training](13-efficient-training.md)
- [Mixed precision training](mixed-precision-training.md)
- [ZeRO and FSDP](zero-and-fsdp.md)
- [Distributed data parallel](distributed-data-parallel.md)
- [Gradient descent](gradient-descent.md) — where Adam and the adaptive optimizers are introduced
