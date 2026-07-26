---
title: Lecture 13 — Efficient Neural Network Training (slide deck)
lecture: 13
slides: 65 printed / 65 pages in the PDF
source_pdf: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture12-training-shikhar.pdf
note: |
  Lecturer is Shikhar Murty. The deck's own title is "Lecture 12: Efficient Neural Network
  Training"; the Cairn catalog lists it at **position 13**, and repo files use the catalog
  position. Printed slide numbers match PDF page numbers 1:1, with no gaps and no offset.
  Five pages (1, 53, 54, 55, 62) print no number but occupy their position in the sequence, so
  slide N is page N throughout.
---

# Lecture 13 — Efficient Neural Network Training: slide-by-slide

Text and figures of
[`cs224n-spr2024-lecture12-training-shikhar.pdf`](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture12-training-shikhar.pdf),
transcribed from the deck. Diagrams and plots are described in prose since the KB is
read as text.

Companion pages: [wiki page for this lecture](../../wiki/13-efficient-training.md) ·
[transcript](../transcripts/13-efficient-training.md)

## Contents

The three section-title slides are 3, 23 and 51.

| Slides | Section |
| ------ | ------- |
| 1–2 | Title and lecture plan; milestone announcements |
| 3–22 | §1 Mixed Precision Training: FP32/FP16 bit layouts, the value formula, range vs. precision, gradient underflow, master weights, loss scaling, the PyTorch AMP recipe, Bfloat16 |
| 23–30 | §2 Multi-GPU Training: what sits in GPU VRAM, Adam's optimizer state, DDP and its all-reduce |
| 31–34 | ZeRO Stage-1 — optimizer-state sharding; the collective operations compared (all-reduce, reduce-scatter, all-gather) |
| 35–38 | ZeRO Stage-2 — optimizer state + gradient sharding |
| 39–48 | ZeRO Stage-3 / Full FSDP — parameters sharded too; FSDP units, FlatParameter, the communication/compute timeline |
| 49–50 | Model activations added back to the memory calculation; the multi-GPU decision flowchart |
| 51–55 | §3 Parameter-Efficient Finetuning — motivation: compute growth, accuracy vs. efficiency, environmental cost |
| 56–61 | Full finetuning objective; LoRA's low-rank update and its code |
| 62–64 | LoRA in practice: GPT-2 and GPT-3 results, which matrices to adapt, which rank |
| 65 | Summary flowchart |

---

## Slide 1 — Title

**Natural Language Processing with Deep Learning — CS224N/Ling284.** Below the Stanford arch
logo: **Shikhar Murty**, *Lecture 12: Efficient Neural Network Training*.

## Slide 2 — Lecture Plan

**Lecture 12: Efficient Neural Network Training - hopefully useful for final projects!**
1. Mixed Precision Training [20 mins]
2. Multi-GPU Training with DDP / FSDP [40 mins]
3. Parameter Efficient Finetuning: LoRA [20 mins]

- Announcements
  - Proposal grades coming out today
  - Final project milestone details is out today!
    - Worth 5% of overall grade
    - Due on May 21st (12 days to work on this)
    - Max 2 pages
    - **Use this as a forcing function to get work done for your final project!**

## Slide 3 — Section title: Mixed Precision Training

## Slide 4 — Floating Points 101

A bit-layout diagram for **FP32**: one pink **S** (sign) box, then eight green **E** boxes
labelled "Exponent (8 bits)", then a run of light-blue **F** boxes labelled "Digits / Mantissa
(23 bits)" (shown as seven boxes, an ellipsis, and a final box to represent all 23).

## Slide 5 — Floating Points 101

Same FP32 bit-layout diagram as slide 4. An arrow now points down from the sign bit to a pink
highlighted box reading **"Memory requirement: 4 bytes."**

## Slide 6 — Floating Points 101

Same FP32 bit-layout diagram as slide 4, now with curved arrows from the **S**, **E**, and **F**
fields down to the value formula:

$$(-1)^B \times 2^{E-127} \times \left(1 + \sum_{i=1}^{23} b_{23-i} 2^{-i}\right)$$

The arrow from **S** points to the $(-1)^B$ term, the arrow from the exponent field points to
$2^{E-127}$, and the arrow from the mantissa field points to the summation $\sum_{i=1}^{23}$.

## Slide 7 — Floating Points 101

Same diagram and formula as slide 6, with two additional pink labels: **"range"** under the
$2^{E-127}$ term and **"precision"** under the mantissa summation term. Below, the line:

"Can represent $[2^k,\ 2^k(1+\epsilon),\ 2^k(1+2\epsilon),\ \ldots,\ 2^{k+1}]$ where
$\epsilon = 2^{-23}$"

## Slide 8 — Floating Points 101

Two bit-layout diagrams stacked for comparison, no formula this time:
- **FP16**: pink **sign** box, five green boxes labelled "Exponent (5 bits)", ten light-blue
  boxes labelled "Digits / Mantissa (10 bits)".
- **FP32**: pink **sign** box, eight green boxes labelled "Exponent (8 bits)", light-blue boxes
  (seven shown, ellipsis, one more) labelled "Digits / Mantissa (23 bits)".

## Slide 9 — Training Neural Networks in Half-Precision?

Same FP16/FP32 bit-layout comparison as slide 8, now with two bullets below:
- Standard Neural Network Training: Model parameters and gradients represented in FP32 (CUDA
  OOM errors with large models).
- Possible solution: Use FP16!

## Slide 10 — Training Neural Networks in Half-Precision?

Same two bullets as slide 9, with two more added:
- Less range: Roughly 2e-14 to 2e15 on both sides
- Smaller precision leads to rounding errors: 1.0001 is 1 in half precision

Below, a terminal screenshot:
```
>>> torch.finfo(torch.float16)

finfo(resolution=0.001, min=-65504, max=65504,
eps=0.000976562, smallest_normal=6.10352e-05,
tiny=6.10352e-05, dtype=float16)
```
Below that, the start of the FP16 bit-layout diagram (sign + exponent boxes only shown at the
bottom of the slide).

## Slide 11 — Training Neural Networks in Half-Precision?

Same bullets as slide 10, plus a "For Neural Net training:" sub-bullet: **"Gradients can
underflow"** (in red).

To the right, a histogram chart titled with axes **"Percentage of all activation gradient
values"** (y-axis, log-scaled: 1/512, 1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8,
16, 32, 64) against **log₂(magnitude)** (x-axis, roughly −75 to 16, with tick labels at −75,
−45, −38, −34, −32, −30, −28, −26, −24, −23, −22, −21, −20, −19, −18, −17, −16, −15, −14, −13,
−12, −11, −10, −9, −8, −6, −4, −2, 0, 2, 4, 6, 8, 10, 12, 14, 15, 16). One data series, a green
bar histogram: a tall isolated bar at 0 reaching 64, then a gap, then a roughly bell-shaped
cluster of bars rising from near −45 to a peak of about 4 around −27 to −29, tapering back down
past −13. Two vertical reference lines and bracket annotations mark ranges rather than
additional series: a red line at about −24 labelled on its left **"Become zero in FP16"** and on
its right, together with a blue line at about −14, **"FP16 denorms"**; a black line at 15 marks
the right edge of the **"FP16 Representable range"** bracket spanning from the red line to the
black line.

## Slide 12 — Training Neural Networks in Half-Precision?

Same bullets as slide 11, with a further sub-bullet added under "For Neural Net training:":
**"Weight updates are imprecise"** (in red). No chart on this slide.

## Slide 13 — Solution: Mixed Precision Training

"Still use FP16, but use FP32 for neural network updates!"

Left, a dataflow diagram (credited **Sharang et al. 2018**): **float2half** converts
Master-Weights (F32) into FP16 weights and activations. These FP16 weights and activations feed
**FWD**, which outputs FP16 activations. FP16 weights and FP16 activation-gradients feed
**BWD-Actv**, which outputs FP16 activation gradients. FP16 activations and FP16
activation-gradients feed **BWD-Weight**, which outputs FP16 weight gradients. The FP16 weight
gradient flows into **Weight Update** (green box), which also takes the F32 Master-Weights
directly; **Weight Update** outputs Updated Master-Weights in F32, which loop back (via
float2half) to start the next iteration.

Right, **"Take-2"**:
1. Maintain a copy of model parameters in FP32 (Master weights)
2. Run forward pass in FP16
3. Compute gradient in FP16
4. Copy gradient into FP32
5. Update master weights in FP32 *[fixes weight update issue!]*
6. Copy into FP16 version

## Slide 14 — Solution: Mixed Precision Training

Same title and opening line as slide 13, but the right-hand list is replaced by the histogram
chart from slide 11 (activation-gradient magnitude distribution with the "Become zero in FP16" /
"FP16 denorms" / "FP16 Representable range" annotations), shown alone and enlarged, with no
"Take-2" list and no diagram.

## Slide 15 — Solution: Mixed Precision Training

Same title, opening line, and dataflow diagram as slide 13, with the same "Take-2" list, plus a
new line below in red: **"Here, gradients can still underflow (small gradients will become
exactly 0)."**

## Slide 16 — Solution: Mixed Precision Training

Same title, opening line, and dataflow diagram as slide 13/15. The list is now retitled
**"Recipe for Mixed-Precision Training"** and gains a new step 3:
1. Maintain a copy of model parameters in FP32 (Master weights)
2. Run forward pass in FP16
3. Scale loss by a large value (to artificially increase gradient)
4. Compute gradient in FP16
5. Copy gradient into FP32 and divide by scale factor
6. Update master weights in FP32 *[fixes weight update issue!]*
7. Copy into FP16 version

## Slide 17 — Mixed Precision Training in PyTorch

A code screenshot:

```python
# Creates model and optimizer in default precision
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

# Creates a GradScaler once at the beginning of training.
scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()

        # Runs the forward pass with autocasting.
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()
```

Source: <https://pytorch.org/docs/stable/notes/amp_examples.html>

## Slide 18 — Can we get rid of gradient scaling?

Same FP16/FP32 bit-layout diagrams as slide 8. Below: "We need scaling because FP16 has a small
range compared to FP32", followed by the same `torch.finfo(torch.float16)` terminal screenshot
as slide 10.

## Slide 19 — Can we get rid of gradient scaling?

Same content as slide 18 (diagrams + "We need scaling…" line), minus the terminal screenshot,
plus a new line: "💡 Can we allocate 8 bits for exponent (same range) while sacrificing
precision?"

## Slide 20 — Greater Dynamic Range with Bfloat16

Three bit-layout diagrams stacked:
- **FP16**: sign, Exponent (5 bits), Digits / Mantissa (10 bits).
- **FP32**: sign, Exponent (8 bits), Digits / Mantissa (23 bits).
- **BFloat16**: sign, Exponent (8 bits), Digits / Mantissa (7 bits).

Below: "Greater Dynamic Range with Bfloat16: can represent much smaller numbers and much larger
numbers (no INF / NaNs)"

## Slide 21 — Bfloat16 does not need GradScalars

A code screenshot:

```python
# Creates model and optimizer in default precision
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

for input, target in data:
    optimizer.zero_grad()

    # Enables autocasting for the forward pass (model + loss)
    with torch.autocast(device_type="cuda"):
        output = model(input)
        loss = loss_fn(output, target)

    # Exits the context manager before backward()
    loss.backward()
    optimizer.step()
```

Source: <https://pytorch.org/docs/stable/amp.html#torch.autocast>

## Slide 22 — Greater Dynamic Range with Bfloat16

A horizontal bar chart, **"Training speed (3 epochs)"**, with x-axis **"Training speed in
minutes"** labelled at 0, 6.5, 13, 19.5, 26, plus two extra numeric columns to the right of the
bars, **"Predictive acc. (test set)"** and **"Memory allocated"**. Five bars, one per precision
setting:

| Precision | Training speed | Predictive acc. (test set) | Memory allocated |
|---|---|---|---|
| Float64 | 24.59 min | 92.14% | 10.42 GB |
| Float32 | 21.75 min | 89.92% | 5.37 GB |
| Float16 | 5.23 min | 50.08% | 2.87 GB |
| Float16-mixed | 7.25 min | 92.15% | 4.31 GB |
| Bfloat16-mixed | 7.45 min | 92.61% | 4.46 GB |

Caption: "Results from finetuning DistilBERT for sentiment classification on a single A100
GPU." Source: <https://sebastianraschka.com/blog/2023/llm-mixed-precision-copy.html>

## Slide 23 — Section title: Multi-GPU Training

## Slide 24 — Multi-GPU Training

Left: a single **GPU:0** box, receiving an arrow from a **Data** cylinder, containing a small
neural-net diagram and, below it, an **Optimizer** box.

Right: "What's stored on GPU VRAM?" — "NN: Model parameters (in FP16)" — "Optimizer: Master
weights (FP32) + Adam momentum (FP32) + Adam variance (FP32) +"

Below, labelled **"Adam Optimizer"**, the update equations:

$$m_t = \beta_1 * m_{t-1} + (1-\beta_1) * \nabla w_t$$
$$v_t = \beta_2 * v_{t-1} + (1-\beta_2) * (\nabla w_t)^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t+\epsilon}} * \hat{m}_t$$

An arrow points from $\nabla w_t$ in the first equation to a pink label reading **"minibatch
gradient."**

## Slide 25 — The Basics: Distributed Data Parallel (DDP)

Header line: "Each GPU has a synchronized copy of the model with its own slice of the data."
Four GPU boxes are laid out in a 2×2 grid, each a different colour (tan **GPU:0**, red
**GPU:2**, green **GPU:1**, yellow **GPU:3**), each fed by its own data cylinder and each
containing an identical small neural-net diagram plus an **Optimizer** box.

## Slide 26 — The Basics: Distributed Data Parallel (DDP)

Same 2×2 four-GPU layout as slide 25, now labelled **"Forward Pass in parallel"** — the
neural-net box inside each of the four GPUs is highlighted (lightened fill) to show all four
GPUs running their forward pass simultaneously and independently.

## Slide 27 — The Basics: Distributed Data Parallel (DDP)

Same four-GPU layout, now captioned **"Run Backward pass while communicating gradients for
upstream parameters."** Curved connector lines are drawn between the GPUs: a tan arc across the
top connects GPU:0 and GPU:2; a teal vertical line connects GPU:0 and GPU:1; an orange vertical
line connects GPU:2 and GPU:3; a red arc across the bottom connects GPU:1 and GPU:3 — a ring
topology used to pass gradients between GPUs as the backward pass proceeds layer by layer.

## Slide 28 — The Basics: Distributed Data Parallel (DDP)

Same four-GPU layout and ring connectors as slide 27. To the right, a diagram titled **"'All
Reduce' Operation":** four coloured input blocks in0 (blue), in1 (red), in2 (green), in3
(yellow) sit under columns labelled rank 0–rank 3; an arrow points to four output columns rank
0–rank 3, each holding an identical **out** block. Caption under the diagram: "out[i] =
sum(inX[i])." Below: "Communication overhead: 2 bytes per parameter (gradients in FP16)."

## Slide 29 — The Basics: Distributed Data Parallel (DDP)

The plain four-GPU layout (no ring connectors), with each **Optimizer** box now containing small
multicoloured tick marks (grey/pink/teal) to represent accumulated gradient contributions.
Caption: "At this point, the optimizer has the cumulated gradient from all GPUs!"

## Slide 30 — The Basics: Distributed Data Parallel (DDP)

Same four-GPU layout as slide 29. Caption: "Update model parameters to **maintain
synchronization**" (the phrase "maintain synchronization" is in purple).

## Slide 31 — Unfortunately, Naive DDP has poor memory scaling

A memory-layout diagram: three bars labelled **gpu₀**, **gpuᵢ**, **gpu_{N-1}** (with ellipses
between them for the omitted GPUs), each bar stacked from thin blue and orange strips on top of
a large green region. To the right, a table:

| | Memory Consumed | (K=12, Ψ=7.5B, N_d=64) |
|---|---|---|
| Baseline | $(2+2+K) * \Psi$ | 120GB |

Legend below the bars:
- Blue: 2 bytes for FP16 parameters
- Orange: 2 bytes for FP16 backward pass gradients
- Green: 4 bytes for FP32 master weights
- Green: 4 bytes for FP32 Adam momentum
- Green: 4 bytes for FP32 Adam variance

## Slide 32 — ZeRO Stage-1: Optimizer State Sharding (P_os)

Same memory-layout diagram as slide 31, with a second row added: **P_os**, where each GPU's bar
keeps the full-width blue and orange strips but the green region is now a thin sliver (only a
shard of the optimizer state resides on each GPU). Table:

| | Memory Consumed | (K=12, Ψ=7.5B, N_d=64) |
|---|---|---|
| Baseline | $(2+2+K) * \Psi$ | 120GB |
| P_os | $2\Psi + 2\Psi + \dfrac{K*\Psi}{N_d}$ | 31.4GB |

- Each GPU has the full set of FP16 model parameters, and computes the gradient on its subset of
  data
- Each GPU has a sharded copy of the full optimizer state.
- Each GPU is responsible for updating a shard of the full parameters

## Slide 33 — ZeRO Stage-1: Optimizer State Sharding (P_os)

Left: the plain four-GPU layout (GPU:0–GPU:3, each with data cylinder, neural net, and Optimizer
box), no ring connectors.

Right:
- Each worker computes gradient on its subset of data.
- Perform a reduce-scatter so that each worker gets the full gradient corresponding to their
  parameter shard: a diagram shows rank 0–rank 3 each holding a full input block (in0 blue, in1
  red, in2 green, in3 yellow); an arrow points to four output columns where rank 0 holds only
  **out0**, rank 1 only **out1**, rank 2 only **out2**, rank 3 only **out3** (each a differently
  positioned single block, the rest of that column blank). Caption: "outY[i] =
  sum(inX[Y*count+i])."
- Each worker updates its parameters
- Perform an all-gather to synchronize params: a second diagram shows rank 0 holding only in0
  (blue), rank 1 only in1 (red), rank 2 only in2 (green), rank 3 only in3 (yellow); an arrow
  points to four output columns where every rank now holds all four blocks stacked (blue, red,
  green, yellow). Caption: "out[Y*count+i] = inY[i]."

## Slide 34 — ZeRO Stage-1: Optimizer State Sharding (P_os)

Left: the same plain four-GPU layout as slide 33.

Right, three side-by-side diagrams (credited to the Meta engineering FSDP blog post) comparing
collective operations, each showing GPU columns A, B, C, D:
- **All Reduce**: four columns A, B, C, D (light blue) → arrow down → all four columns become
  identical, each holding A+B+C+D (dark blue).
- **Reduce-Scatter**: each column is subdivided into four rows (A0–A3, B0–B3, C0–C3, D0–D3,
  shown as a 4×4 grid in pink/magenta shades) → arrow down → each GPU ends up holding only one
  reduced row: GPU 1 holds A0+B0+C0+D0, GPU 2 holds A1+B1+C1+D1, GPU 3 holds A2+B2+C2+D2, GPU 4
  holds A3+B3+C3+D3, with the rest of each column blank.
- **All-gather**: each GPU starts holding only its own single block (A in GPU 1, B in GPU 2, C
  in GPU 3, D in GPU 4, shown in purple shades) → arrow down → every GPU ends up holding all
  four blocks A, B, C, D stacked together.

Source: <https://engineering.fb.com/2021/07/15/open-source/fsdp/attachment/fsdp-graph-2a/>

A pink label with an arrow pointing at the All Reduce diagram reads **"We used all-reduce for
DDP."** Below: "Communication overhead: 2 bytes per parameter (gradients in FP16)" and "We saved
memory for free!"

## Slide 35 — ZeRO Stage-2: Optimizer State + gradient sharding (P_os+g)

Same memory-layout diagram as slide 32, with a third row added: **P_os+g**, where each GPU's bar
now keeps only the full-width blue strip, while both the orange and green regions are reduced to
thin slivers. Table:

| | Memory Consumed | (K=12, Ψ=7.5B, N_d=64) |
|---|---|---|
| Baseline | $(2+2+K) * \Psi$ | 120GB |
| P_os | $2\Psi + 2\Psi + \dfrac{K*\Psi}{N_d}$ | 31.4GB |
| P_os+g | $2\Psi + \dfrac{(2+K)*\Psi}{N_d}$ | 16.6GB |

Below: "Along with sharing optimizer state, can we also shard gradients? 🤔 Complexity: We still
need the full gradient for the worker's data slice!"

## Slide 36 — ZeRO Stage-2: Optimizer State + gradient sharding (P_os+g)

Same diagram and table as slide 35, with an added **"Solution:"**
- Never instantiate the full gradient vector!
- Send gradient to the "GPU in charge" as soon as the gradient for a shard is made available in
  the backward pass

## Slide 37 — ZeRO Stage-2: Optimizer State + gradient sharding (P_os+g)

Left: the plain four-GPU layout (GPU:0–GPU:3).

Right:
- Worker performs a backward pass layer-by-layer in the computation graph
- Suppose worker is at layer-j:
  - Take upstream gradient, compute gradient for parameters at layer-j,
  - immediately send the gradients to the correct worker (reduce)
  - deallocate memory for parameter gradient.

Below, a reduce diagram: rank 0–rank 3 each hold an input block (in0 blue, in1 red, in2 green,
in3 yellow); an arrow points to a single output held only by rank 2, labelled **"(root)"**,
containing **out**. Caption: "out[i] = sum(inX[i])."

## Slide 38 — ZeRO Stage-2: Optimizer State + gradient sharding (P_os+g)

Same left-hand four-GPU layout and the same bullet list as slide 37, with two more bullets
added:
- Worker updates its param shard using corresponding gradient + state
- Perform an all-gather to synchronize

Below, an all-gather diagram: rank 0 holds only in0 (blue), rank 1 only in1 (red), rank 2 only
in2 (green), rank 3 only in3 (yellow); an arrow points to four output columns where every rank
now holds all four blocks stacked (blue, red, green, yellow). Caption: "out[Y*count+i] = inY[i]."

## Slide 39 — ZeRO Stage-3 (Full FSDP): When even the model parameters won't fit

Same memory-layout diagram as slide 35, with a fourth row added: **P_os+g+p**, where every GPU's
bar is reduced to thin slivers of all three colours (parameters, gradients, and optimizer states
are all sharded). A legend at the bottom identifies the three colours: blue = Parameters, orange
= Gradients, green = Optimizer States. Table:

| | Memory Consumed | (K=12, Ψ=7.5B, N_d=64) |
|---|---|---|
| Baseline | $(2+2+K) * \Psi$ | 120GB |
| P_os | $2\Psi + 2\Psi + \dfrac{K*\Psi}{N_d}$ | 31.4GB |
| P_os+g | $2\Psi + \dfrac{(2+K)*\Psi}{N_d}$ | 16.6GB |
| P_os+g+p | $\dfrac{(2+2+K)*\Psi}{N_d}$ | 1.9GB |

## Slide 40 — ZeRO Stage-3 (Full FSDP): When even the model parameters won't fit

Same diagram, legend and table as slide 39, with an added note: "Caveat: So far, communication
overhead was 'free'. With Full FSDP, this is no longer the case."

## Slide 41 — ZeRO Stage-3 (Full FSDP): When even the model parameters won't fit

Same memory-layout diagram, legend and table as slide 39 (Baseline 120GB, P_os 31.4GB, P_os+g
16.6GB, P_os+g+p 1.9GB). To the right: "High-level sketch:" — "1. Divide model parameters into
FSDP units."

## Slide 42 — ZeRO Stage-3 (Full FSDP): When even the model parameters won't fit

A diagram showing a six-layer model (**layer0**–**layer5**, stacked top to bottom with arrows
between them) being **"Wrap & Shard"**-ed (grey arrow) into **FSDP Unit0**: layer0 sits alone at
the top; layer1 and layer2 are grouped together inside a nested box labelled **FSDP Unit1**;
layer3 sits alone; layer4 and layer5 are grouped together inside a nested box labelled **FSDP
Unit2**. All four elements — layer0, FSDP Unit1, layer3 and FSDP Unit2 — are contained inside the
outer **FSDP Unit0** box, connected top-to-bottom by arrows.

To the right, a dotted **"Exec"** arrow leads to two columns, **Forward** and **Backward**,
showing what happens when Unit1 (layer1 + layer2) executes: in the Forward column, top-to-bottom
— **gather full params** → a box containing layer1 and layer2 → **free peer shards**. In the
Backward column, top-to-bottom — **synchronize gradients** → **free peer shards** → a box
containing layer1 and layer2 → **gather full params** (i.e., the backward column runs the same
four steps in reverse order to the forward column).

Text: "High-level sketch: 1. Divide model parameters into FSDP units"

## Slide 43 — ZeRO Stage-3 (Full FSDP): When even the model parameters won't fit

A diagram titled **FlatParameter**: a linear model has a **weight** matrix whose 12 entries are
numbered 0–11 (laid out 4 wide × 3 tall: row 0 = [0,1,2,3], row 1 = [4,5,6,7], row 2 =
[8,9,10,11]) and a **bias** vector with 3 entries numbered 12–14. These are concatenated into one
flat array, indices 0 through 14 in order, plus a padding slot at the end (dashed). This
FlatParameter is then split into a **"Full Sharding (F = 16)"** group of 6 local shards, drawn as
two rows of three boxes: top row **0, 1, …, 7**, bottom row **8, 9, …, 15**, each box holding a
small "Local Shard" (dashed sub-box) of the flat array — box 0 holds element 0, box 1 holds
element 1, box 7 holds element 7, box 8 holds element 8, box 9 holds element 9, and box 15 holds
only padding (dashed, empty). All six boxes are enclosed by a dashed red **"Sharding Group"**
outline. Legend: pink/red dashed = Sharding Group, orange dashed = Local Shard, blue dashed =
Padding, boxed "i" = Rank i.

Text: "High-level sketch: 1. Divide model parameters into FSDP units. **2. Shard each unit
across multiple GPUs**"

## Slide 44 — ZeRO Stage-3 (Full FSDP): When even the model parameters won't fit

A timeline diagram with three horizontal tracks — **CPU**, **GPU Comp. Stream**, **GPU Comm.
Stream** — split into a **Forward** phase and a **Backward** phase (divided by a dotted vertical
line). The CPU track shows a sequence of small numbered cells (unit IDs, e.g. 0 0 1 0 2 2 2 2 0 1
1 1 0 0) with dashed diagonal lines connecting each to matching cells on the compute and
communication tracks below. The GPU Comp. Stream shows compute blocks labelled **FWD0**,
**FWD1**, **FWD0**, **FWD2**, **FWD2** in the forward half and **BWD2**, **BWD0**, **BWD1**,
**BWD0** in the backward half (colours: light blue = Forward Comp. (FWD), light green = Backward
Comp. (BWD)). The GPU Comm. Stream shows, in the forward half, four pink **All-Gather (AG)**
blocks **AG0, AG1, AG2, AG2**, and in the backward half, purple **Reduce-Scatter (RS)** blocks
**RS2, AG1, RS1, RS0** (one further all-gather appears interleaved). Legend: pink = All-Gather
(AG), purple = Reduce-Scatter (RS), light blue = Forward Comp. (FWD), light green = Backward
Comp. (BWD), yellow = Parameter Free, "i" = FSDP Unit i.

Below, an all-gather diagram (same style as slide 33/38): rank 0 holds only in0 (blue), rank 1
only in1 (red), rank 2 only in2 (green), rank 3 only in3 (yellow); an arrow points to four output
columns where every rank now holds all four stacked blocks. Caption: "out[Y*count+i] = inY[i]."
A pink label reads **"all-gather."**

Text: "High-level sketch: 1. Divide model parameters into FSDP units. 2. Shard each unit across
multiple GPUs. **3. Run forward pass:** - perform an **all-gather** so each GPU gets all pieces
of a module. - Run forward pass - Discard param shards"

## Slide 45 — ZeRO Stage-3 (Full FSDP): When even the model parameters won't fit

Same timeline diagram as slide 44. Below it, a reduce-scatter diagram (same style as slide 33):
rank 0–rank 3 each hold a full input block (in0 blue, in1 red, in2 green, in3 yellow); an arrow
points to four output columns where rank 0 holds only **out0**, rank 1 only **out1**, rank 2 only
**out2**, rank 3 only **out3**. Caption: "outY[i] = sum(inX[Y*count+i])." A pink label reads
**"reduce-scatter."**

Text: "High-level sketch: 1. Divide model parameters into FSDP units. 2. Shard each unit across
multiple GPUs. 3. Run forward pass. **4. Run backward pass:** - perform an **all-gather** to get
all pieces of module, - Each GPU computes gradient for its data chunk - Do a **reduce-scatter**
to send full gradient piece to the right GPU"

## Slide 46 — ZeRO Stage-3 (Full FSDP): When even the model parameters won't fit

Same timeline diagram as slides 44–45, now shown alone (no reduce-scatter/all-gather sub-diagram
below it).

Text: "High-level sketch: 1. Divide model parameters into FSDP units. 2. Shard each unit across
multiple GPUs. 3. Run forward pass. 4. Run backward pass. **5. Each GPU updates its own shard
using the full gradient received earlier.**"

## Slide 47 — ZeRO Stage-3 (Full FSDP): When even the model parameters won't fit

Left: the same timeline diagram as slides 44–46. Right: the three-panel All Reduce /
Reduce-Scatter / All-gather comparison diagram from slide 34 (columns A, B, C, D).

Text: "Communication overhead recap: 1. DDP: All reduce. 2. ZeRO Stage-1 / 2: Reduce-Scatter +
All-gather [memory saved for free]"

## Slide 48 — ZeRO Stage-3 (Full FSDP): When even the model parameters won't fit

Same timeline diagram as slide 47 (the three-panel All Reduce / Reduce-Scatter / All-gather
comparison diagram is no longer shown). Text: "Communication overhead recap: 1. DDP: All reduce.
2. ZeRO Stage-1 / 2: Reduce-Scatter + All-gather [memory saved for free]. **3. ZeRO Stage-3:
All-gather + Reduce-scatter + All-gather [More overhead!]**" (the stage-3 line is in red).

## Slide 49 — Revisiting GPU memory calculation

Centered text:

"GPU memory consumption := Model parameters (in FP16) + Gradients (FP16) + Master weights (FP32)
+ Adam momentum (FP32) + Adam variance (FP32) +" followed by, highlighted in yellow: **"Model
Activations (This scales with the batch size)!"**

## Slide 50 — Multi-GPU Training Optimizations Recap

A flowchart of grey rounded boxes. Top box: "Always use Mixed-Precision Training / Always use
BFloat16 if `torch.cuda.is_bf16_supported()`" (the code snippet highlighted in yellow). An arrow
leads down to "Does batch-size = 1 fit on single GPU?" This branches two ways:
- **Yes** → "Try larger batch size / use ZeRO Stage-2"
- **No** → "Does ZeRO Stage-3 solve OOM errors??", which branches **No** → "Try Parameter-Efficient
  Finetuning!"

## Slide 51 — Section title: Parameter-Efficient Finetuning

Subtitle: *"Adapted from slides by Diyi Yang"*

## Slide 52 — From fine-tuning to parameter-efficient fine-tuning (PEFT)

Two side-by-side diagrams of a network-block stack (a transformer-block-style icon: horizontal
bars for layers, a diamond-shaped attention icon in a blue box, and an orange self-attention box
with small **Q**, **K**, **V** squares and curved arrows between them, all connected by upward
arrows).
- Left, **"Full Fine-tuning / Update all model parameters"**: every element — bars, blue
  attention box, orange Q/K/V box, and arrows — is drawn in solid colour, indicating the whole
  stack is being updated.
- Right, **"Parameter-efficient Fine-tuning / Update a small subset of model parameters"**: most
  elements are faded to light grey/tan, with only the top blue attention box left in solid
  colour, indicating just a small subset of parameters is trainable while the rest stay frozen.

Right-hand text: "Why fine-tune *only some* parameters?
1. Fine-tuning all parameters is impractical with large models
2. State-of-the-art models are massively over-parameterized → Parameter-efficient fine-tuning
   matches performance of full fine-tuning"

## Slide 53 — Why do we need efficient adaptation?

*(This page prints no slide number, but sits unambiguously between numbered pages 52 and 56.)*

- Exponential growth in maximum training compute for largest AI models (~2x every 3.4 months)
  vs. global compute capacity (~2x every 1.5 years)
- Clearly unsustainable rate of growth in AI computing scale, forecasted to slow a lot in the
  next few years.
- **As costs of training go up, AI development becomes concentrated in only the most well-funded
  organizations, especially in industry.**
  - Concentrating market power could lead to only a few dominant interests controlling a global
    technology – **whose value systems** are embedded in the AI of tomorrow?

Right, a line chart titled **"Unsustainable growth at current rates"**: y-axis **FLOPS /
FLOPS-days**, log-scaled from 1.0E+20 to 1.0E+28; x-axis **Year**, 2022 to 2032 (gridlines at
2022, 2024, 2026, 2028, 2030, 2032). Two data series:
- **Global computing capacity (FLOPS)** — blue line, starting at roughly 1.0E+22 in 2022 and
  rising gently to roughly 1.0E+24 by 2032.
- **Largest AI training compute (FLOPS-days)** — red line, starting at roughly 1.0E+20 in 2022
  and rising steeply throughout, crossing the blue line around 2026 and reaching roughly 1.0E+28
  by 2032.

Source: "How much of AI progress is from scaling compute? And how far will it scale? (by
'jack', AI Progress Essay Contest)"

Footer: "Slides credit to Benji Xie, Regina Wang and Pranav Gurusankar"

## Slide 54 — Accuracy vs Efficiency: What are we focusing on?

*(This page also prints no slide number, sitting between 53 and 55.)*

1. Emphasis on accuracy over efficiency in current AI paradigm
   - "Is the tradeoff between efficiency and accuracy linear? It's quite that simple… [Ang et
     al., 2022]" [as printed on the slide; the sense appears to be that it is *not* that simple,
     though the text itself reads this way]
2. **Hidden environmental costs of training (and fine tuning) LLMs**
   - *Most large players are non-transparent about the costs of training their models.*
   - *Cornell scientists in 2021 estimated that training GPT-3 was equivalent in carbon
     emissions to running a coal power plant for 10 straight hours.*

Right, a grouped bar chart, y-axis **Number of papers** (0–16), x-axis three
conference/venue-years: **ACL 2018**, **CVPR 2019**, **NeurIPS 2018**. Four series per group
(legend: Accuracy = orange, Efficiency = green, Both = yellow/gold, Other = blue):
- **ACL 2018**: Accuracy ≈ 16, Efficiency ≈ 0, Both ≈ 2, Other ≈ 2.
- **CVPR 2019**: Accuracy ≈ 13, Efficiency ≈ 2, Both ≈ 2, Other ≈ 3.
- **NeurIPS 2018**: Accuracy ≈ 9, Efficiency ≈ 4, Both ≈ 7, Other ≈ 1.

Caption: "AI papers tend to target accuracy rather than efficiency. The figure shows the
proportion of papers that target accuracy, efficiency, both or other from a sample of 60 papers
from top AI conferences ([Green AI])."

Footer: "Slides credit to Benji Xie, Regina Wang and Pranav Gurusankar"

## Slide 55 — (untitled quote slide)

*(This page also prints no slide number, sitting between 54 and 56.)*

Block quote: "At Stanford, for example, more than 200 students in a class on reinforcement
learning were asked to implement common algorithms for a homework assignment. Though two of the
algorithms performed equally well, one used far more power.

If all the students had used the more efficient algorithm, the researchers estimated they would
have reduced their collective power consumption by **880 kilowatt-hours — about what a typical
American household uses in a month.**"

Below: "An example using CS234 in *Towards the Systematic Reporting of the Energy and Carbon
Footprints of Machine Learning*." (linked)

Footer: "Slides credit to Benji Xie, Regina Wang and Pranav Gurusankar"

## Slide 56 — Full Finetuning

- Assume we have a pre-trained autoregressive language model $P_\phi(y|x)$
  - E.g., GPT based on Transformer
- Adapt this pretrained model to downstream tasks (e.g., summarization, NL2SQL, reading
  comprehension)
  - Training dataset of context-target pairs $\{(x_i,y_i)\}_{i=1,\ldots,N}$
- During full fine-tuning, we update $\phi_o$ to $\phi_o + \Delta\phi$ by following the gradient
  to maximize the conditional language modeling objective

$$\max_\phi \sum_{(x,y)} \sum_{t=1}^{|y|} \log(P_\phi(y_t \mid x, y_{<t}))$$

## Slide 57 — Full-Finetuning

- For each downstream task, we learn a different set of parameters $\Delta\phi$
  - $|\Delta\phi| = |\phi_o|$
  - GPT-3 has a $|\phi_o|$ of 175 billion
  - Expensive and challenging for storing and deploying many independent instances
- Can we do better?

## Slide 58 — Full-Finetuning

Same bullets as slide 57, with two more added:
- Key idea: encode the task-specific parameter increment $\Delta\phi = \Delta\phi(\Theta)$ by a
  smaller-sized set of parameters $\Theta$, $|\Theta| \ll |\phi_o|$
- The task of finding $\Delta\phi$ becomes optimizing over $\Theta$

$$\max_\Theta \sum_{(x,y)} \sum_{t=1}^{|y|} \log\left(P_{\phi_o+\Delta\phi(\Theta)}(y_t \mid x, y_{<t})\right)$$

## Slide 59 — Low-rank-parameterized update matrices

- Updates to the weights have a low "intrinsic rank" during adaptation (Aghajanyan et al. 2020)
- $W_0 \in \mathbb{R}^{d\times k}$: a pretrained weight matrix
- Constrain its update with a low-rank decomposition:

$$W_0 + \Delta W = W_0 + \alpha BA$$

where $B \in \mathbb{R}^{d\times r}$, $A \in \mathbb{R}^{r\times k}$, $r \ll \min(d,k)$
- $\alpha$ is the tradeoff between pre-trained "knowledge" and task-specific "knowledge"
- Only A and B contain **trainable** parameters

Right, a diagram: at the top, a vector **h**; below it a circled **+** with two diagonal arrows
converging into it, one rising from the blue **Pretrained Weights** box (labelled
$W_0 \in \mathbb{R}^{d\times k}$) on the left, the other rising from an orange trapezoid labelled
$B \in \mathbb{R}^{d\times r}$ on the right. The **B** trapezoid narrows down to a bracketed
**r**, connecting to a second orange trapezoid below it labelled $A \in \mathbb{R}^{r\times k}$,
which widens back out. At the bottom, a vector **x** with a brace labelled **d** underneath;
two arrows rise from x, one straight up into the Pretrained Weights box, the other into the
**A** trapezoid. This depicts input x feeding both the frozen pretrained weight matrix $W_0$ and
the low-rank adapter path (through $A$ then $B$), whose outputs are summed to produce h.

## Slide 60 — Low-rank-parameterized update matrices

Same diagram as slide 59. New bullets:
- As one increase the number of trainable parameters, training LoRA converges to training the
  original model
- **No additional inference latency:** when switching to a different task, recover $W_0$ by
  subtracting $BA$ and adding a different $B'A'$
- Often LoRA is applied to the weight matrices in the self-attention module

## Slide 61 — Low-rank-parameterized update matrices

A code screenshot:

```python
input_dim = 768  # e.g., the hidden size of the pre-trained model
output_dim = 768  # e.g., the output size of the layer
rank = 8  # The rank 'r' for the low-rank adaptation

W = ...  # from pretrained network with shape input_dim x output_dim

W_A = nn.Parameter(torch.empty(input_dim, rank))  # LoRA weight A
W_B = nn.Parameter(torch.empty(rank, output_dim))  # LoRA weight B

# Initialization of LoRA weights
nn.init.kaiming_uniform_(W_A, a=math.sqrt(5))
nn.init.zeros_(W_B)

def regular_forward_matmul(x, W):
    h = x @ W
    return h

def lora_forward_matmul(x, W, W_A, W_B):
    h = x @ W  # regular matrix multiplication
    h += x @ (W_A @ W_B)*alpha  # use scaled LoRA weights
    return h
```

Source: <https://lightning.ai/pages/community/article/lora-llm/>

## Slide 62 — LoRA in practice

*(This page prints no slide number, but sits unambiguously between numbered pages 61 and 63.)*

A results table, **GPT-2 medium (M) and large (L) with different adaptation methods on the E2E
NLG Challenge:**

| Model & Method | # Trainable Parameters | BLEU | NIST | MET | ROUGE-L | CIDEr |
|---|---|---|---|---|---|---|
| GPT-2 M (FT)* | 354.92M | 68.2 | 8.62 | 46.2 | 71.0 | 2.47 |
| GPT-2 M (Adapter$^L$)* | 0.37M | 66.3 | 8.41 | 45.0 | 69.8 | 2.40 |
| GPT-2 M (Adapter$^L$)* | 11.09M | 68.9 | 8.71 | 46.1 | 71.3 | 2.47 |
| GPT-2 M (Adapter$^H$) | 11.09M | 67.3±.6 | 8.50±.07 | 46.0±.2 | 70.7±.2 | 2.44±.01 |
| GPT-2 M (FT$^{Top2}$)* | 25.19M | 68.1 | 8.59 | 46.0 | 70.8 | 2.41 |
| GPT-2 M (PreLayer)* | 0.35M | 69.7 | 8.81 | 46.1 | 71.4 | 2.49 |
| GPT-2 M (LoRA) | 0.35M | 70.4±.1 | 8.85±.02 | 46.8±.2 | 71.8±.1 | 2.53±.02 |
| GPT-2 L (FT)* | 774.03M | 68.5 | 8.78 | 46.0 | 69.9 | 2.45 |
| GPT-2 L (Adapter$^L$) | 0.88M | 69.1±.1 | 8.68±.03 | 46.3±.0 | 71.4±.2 | 2.49±.0 |
| GPT-2 L (Adapter$^L$) | 23.00M | 68.9±.3 | 8.70±.04 | 46.1±.1 | 71.3±.2 | 2.45±.02 |
| GPT-2 L (PreLayer)* | 0.77M | 70.3 | 8.85 | 46.2 | 71.7 | 2.47 |
| GPT-2 L (LoRA) | 0.77M | 70.4±.1 | 8.89±.02 | 46.8±.2 | 72.0±.2 | 2.47±.02 |

(In the source table the best score in each column is printed in bold; the LoRA rows carry most
of the bolded values.)

Caption: "GPT-2 medium (M) and large (L) with different adaptation methods on the E2E NLG
Challenge. For all metrics, higher is better. LoRA outperforms several baselines with comparable
or fewer trainable parameters"

Citation: Hu, Edward J., Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
Lu Wang, and Weizhu Chen. "Lora: Low-rank adaptation of large language models." *arXiv preprint
arXiv:2106.09685* (2021).

## Slide 63 — LoRA in practice

A second results table, GPT-3:

| Model & Method | # Trainable Parameters | WikiSQL Acc. (%) | MNLI-m Acc. (%) | SAMSum R1/R2/RL |
|---|---|---|---|---|
| GPT-3 (FT) | 175,255.8M | 73.8 | 89.5 | 52.0/28.0/44.5 |
| GPT-3 (BitFit) | 14.2M | 71.3 | 91.0 | 51.3/27.4/43.5 |
| GPT-3 (PreEmbed) | 3.2M | 63.1 | 88.6 | 48.3/24.2/40.5 |
| GPT-3 (PreLayer) | 20.2M | 70.1 | 89.5 | 50.8/27.3/43.5 |
| GPT-3 (Adapter$^H$) | 7.1M | 71.9 | 89.8 | 53.0/28.9/44.8 |
| GPT-3 (Adapter$^H$) | 40.1M | 73.2 | 91.5 | 53.2/29.0/45.1 |
| GPT-3 (LoRA) | 4.7M | 73.4 | 91.7 | 53.8/29.8/45.9 |
| GPT-3 (LoRA) | 37.7M | 74.0 | 91.6 | 53.4/29.2/45.1 |

Right: "LoRA matches or exceeds the fine-tuning baseline on all three datasets"

Below, two line/scatter charts sharing the x-axis **$\log_{10}$ # Trainable Parameters** (6–11).
Five series in each, per the legend (**Method**): **Fine-Tune** (blue dot), **PrefixEmbed**
(orange +), **PrefixLayer** (teal star), **Adapter(H)** (orange/red ×), **LoRA** (pink downward
triangle).

- Left, **WikiSQL** (y-axis **Validation Accuracy**, ≈0.55–0.75):
  - Fine-Tune: a single point at the far right, around $x\approx11$, $y\approx0.74$.
  - PrefixEmbed: rises from ≈0.565 at $x=6$ to a peak ≈0.63 around $x=6.5$, then falls back to
    ≈0.55 by $x=7$.
  - PrefixLayer: rises from ≈0.67 at $x=6.3$ to a peak ≈0.70 around $x=7$, then declines to
    ≈0.65 by $x=8$.
  - Adapter(H): roughly flat, ≈0.70–0.73, from $x=6.5$ to $x=8.5$.
  - LoRA: the highest curve throughout, roughly flat at ≈0.73–0.75 from $x=6.5$ to $x=8.5$.
- Right, **MultiNLI-matched** (y-axis **Validation Accuracy**, ≈0.84–0.92), same five series and
  colours:
  - Fine-Tune: a single point at $x\approx11$, $y\approx0.895$.
  - PrefixEmbed: rises from ≈0.845 at $x=6$ to ≈0.885 around $x=6.7$, then declines to ≈0.86 by
    $x=7.5$.
  - PrefixLayer: rises to a peak ≈0.895 around $x=7$, then declines to ≈0.875 by $x=8$.
  - Adapter(H): roughly flat, ≈0.89–0.915, from $x=6.5$ to $x=8.5$.
  - LoRA: the highest curve throughout, roughly flat at ≈0.915–0.92 from $x=6.5$ to $x=8.5$.

Text: "LoRA exhibits better scalability and task performance."

## Slide 64 — Understanding low-rank adaptation

Top table, **"Which weight matrices in Transformers should we apply LoRA to?"** (header: "# of
Trainable Parameters = 18M"):

| Weight Type / Rank $r$ | $W_q$ (8) | $W_k$ (8) | $W_v$ (8) | $W_o$ (8) | $W_q, W_k$ (4) | $W_q, W_v$ (4) | $W_q, W_k, W_v, W_o$ (2) |
|---|---|---|---|---|---|---|---|
| WikiSQL (±0.5%) | 70.4 | 70.0 | 73.0 | 73.2 | 71.4 | **73.7** | **73.7** |
| MultiNLI (±0.1%) | 91.0 | 90.8 | 91.0 | 91.3 | 91.3 | 91.3 | **91.7** |

Right: "Adapting both Wq and Wv gives the best performance overall."

Bottom table, **"What is the optimal rank $r$ for LoRA?"**

| | Weight Type | $r=1$ | $r=2$ | $r=4$ | $r=8$ | $r=64$ |
|---|---|---|---|---|---|---|
| WikiSQL (±0.5%) | $W_q$ | 68.8 | 69.6 | 70.5 | 70.4 | 70.0 |
| WikiSQL (±0.5%) | $W_q, W_v$ | 73.4 | 73.3 | 73.7 | 73.8 | 73.5 |
| WikiSQL (±0.5%) | $W_q, W_k, W_v, W_o$ | 74.1 | 73.7 | 74.0 | 74.0 | 73.9 |
| MultiNLI (±0.1%) | $W_q$ | 90.7 | 90.9 | 91.1 | 90.7 | 90.7 |
| MultiNLI (±0.1%) | $W_q, W_v$ | 91.3 | 91.4 | 91.3 | 91.6 | 91.4 |
| MultiNLI (±0.1%) | $W_q, W_k, W_v, W_o$ | 91.2 | 91.7 | 91.7 | 91.5 | 91.4 |

Right: "LoRA already performs competitively with a very small $r$"

## Slide 65 — Summarizing everything:

The same flowchart as slide 50: top box "Always use Mixed-Precision Training / Always use
BFloat16 if `torch.cuda.is_bf16_supported()`" (code snippet highlighted yellow) → "Does
batch-size = 1 fit on single GPU?", branching **Yes** → "Try larger batch size / use ZeRO
Stage-2"; **No** → "Does ZeRO Stage-3 solve OOM errors??", branching **No** → **"Try LoRA!"**
(this final box's text differs from slide 50, which read "Try Parameter-Efficient Finetuning!").

Below: "**Even simple: start with Llama 7B + bfloat16 + ZeRO Stage-3 (or FSDP) + LoRA 🤞**"
