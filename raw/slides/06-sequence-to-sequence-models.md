---
title: Lecture 6 — LSTM RNNs and Neural Machine Translation (slide deck)
lecture: 6
slides: 56 printed / 56 pages in the PDF
source_pdf: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture06-fancy-rnn.pdf
note: Printed slide numbers match PDF page numbers 1:1. Four full-bleed image pages (19, 43, 44, 48) print no number but occupy their position in the sequence, so slide N is page N throughout. Slides 4–18 repeat lecture 5's material as a recap.
---

# Lecture 6 — LSTM RNNs and Neural Machine Translation: slide-by-slide

Text and figures of
[`cs224n-spr2024-lecture06-fancy-rnn.pdf`](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture06-fancy-rnn.pdf),
transcribed from the deck. Diagrams and plots are described in prose since the KB is
read as text.

**Slide numbers vs PDF pages.** The deck has 56 pages and the printed numbers run 1–56
with no gaps and no offset: **slide N is PDF page N**. Four pages (19, 43, 44, 48) are
full-bleed images and print no number, but they sit in sequence — page 18 prints "18"
and page 20 prints "20" — so nothing is missing.

**Slides 4–18 are a recap of lecture 5.** Perplexity (4–5), the vanishing-gradient
intuition (6–11), the proof sketch (12–13), why it matters (14–15), exploding gradients
and clipping (16–17) and the setup for LSTMs (18) all appeared in
[lecture 5's deck](05-recurrent-neural-networks.md) as slides 49–50 and 51–63. They are
transcribed here in brief with a pointer, except where lecture 6 adds something new
(slide 15 adds the ~7-token rule of thumb; slide 18 drops lecture 5's forward pointer).

The title slide calls this *Lecture 6: LSTM RNNs and Neural Machine Translation*; the
course catalog lists it as *Sequence to Sequence Models*.

Companion pages: [wiki page for this lecture](../../wiki/06-sequence-to-sequence-models.md) ·
[transcript](../transcripts/06-sequence-to-sequence-models.md)

## Contents

| Slides | Section |
| ------ | ------- |
| 1–2 | Title and lecture plan |
| 3–5 | Recap: LMs and RNNs; perplexity and the perplexity table |
| 6–17 | §1 Vanishing and exploding gradients (recap of lecture 5, slides 51–62) |
| 18 | Setting up the fix: a vanilla RNN rewrites its hidden state every step |
| 19–21 | §2 LSTMs — the Apple WWDC framing, the history, hidden state vs cell state |
| 22 | **The LSTM equations** — the three gates, new cell content, cell and hidden state |
| 23–24 | The LSTM drawn as a diagram; "the + sign is the secret" |
| 25 | How the LSTM solves vanishing gradients — ~100 timesteps rather than ~7 |
| 26–27 | Vanishing gradients beyond RNNs: ResNet, DenseNet, HighwayNet |
| 28–32 | §3 Other RNN uses: tagging, sentence encoding, conditional generation |
| 33 | §4 Bidirectional and multi-layer RNNs: motivation ("terribly exciting") |
| 34–37 | Bidirectional RNNs — diagram, equations, and when they do not apply |
| 38–40 | Multi-layer (stacked) RNNs, and how deep they are in practice |
| 41 | LSTMs: real-world success, and their displacement by Transformers |
| 42–44 | §5 Machine Translation defined; the 1950s |
| 45–47 | 1990s–2010s: Statistical Machine Translation, and why it was hard |
| 48 | 2014: Neural Machine Translation hits MT research (the meteor slide) |
| 49–51 | §6 What is NMT? The seq2seq model; seq2seq is versatile |
| 52–53 | NMT as a conditional LM; training a seq2seq system end-to-end |
| 54 | The multi-layer deep encoder-decoder net, and the conditioning bottleneck |
| 55 | NMT: the first big success story of NLP deep learning |
| 56 | In summary — four practical takeaways |

---

## Slide 1 — Title

**Natural Language Processing with Deep Learning — CS224N/Ling284**

Christopher Manning. Lecture 6: LSTM RNNs and Neural Machine Translation.

## Slide 2 — Lecture Plan

1. Exploding and vanishing gradients (20 mins)
2. Long Short-Term Memory RNNs (LSTMs) (20 mins)
3. Other uses of RNNs (5 mins)
4. Bidirectional and multi-layer RNNs (15 mins)
5. Machine translation (10 mins)
6. Neural machine translation introduction (10 mins)

- Final Projects: Next Tuesday, part of the lecture is about choosing final projects
  - It's fine to just work on Ass 3 and to delay thinking about projects until next week!
    - Ass 3 is about neural machine translation (and LSTMs)

## Slide 3 — Recap

- **Language Model**: A system that **predicts the next word**

- **Recurrent Neural Network**: A family of neural networks that:
  - Take **sequential input of any length;** apply the **same weights on each step**
  - Can optionally produce output on each step

- Recurrent Neural Network ≠ Language Model
  - RNNs can be used for many other things (see later)

- **Language Modeling** is a traditional **subcomponent** of many NLP tasks, all those
  involving **generating text** or **estimating the probability of text**:
  - Now everything in NLP is being rebuilt upon Language Modeling: GPT-3 is an LM!
  - Language modeling can be done with different models, e.g., *n*-grams or transformers

## Slide 4 — Evaluating Language Models

Identical to [lecture 5 slide 49](05-recurrent-neural-networks.md). Perplexity:

  perplexity = ∏_{t=1..T} ( 1 / P_LM(**x**⁽ᵗ⁺¹⁾ | **x**⁽ᵗ⁾, …, **x**⁽¹⁾) )^{1/T}
  = exp( (1/T) Σ_{t=1..T} − log **ŷ**⁽ᵗ⁾_{**x**_{t+1}} ) = exp(J(θ))

> **Lower** perplexity is better!

## Slide 5 — RNNs greatly improved perplexity over what came before

The same results table as [lecture 5 slide 50](05-recurrent-neural-networks.md):

| Model | Perplexity |
| --- | --- |
| Interpolated Kneser-Ney 5-gram (Chelba et al., 2013) | 67.6 |
| RNN-1024 + MaxEnt 9-gram (Chelba et al., 2013) | 51.3 |
| RNN-2048 + BlackOut sampling (Ji et al., 2015) | 68.3 |
| Sparse Non-negative Matrix factorization (Shazeer et al., 2015) | 52.9 |
| LSTM-2048 (Jozefowicz et al., 2016) | 43.7 |
| 2-layer LSTM-8192 (Jozefowicz et al., 2016) | 30 |
| **Ours small** (LSTM-2048) | 43.9 |
| **Ours large** (2-layer LSTM-2048) | 39.8 |

Source: https://research.fb.com/building-an-efficient-neural-language-model-over-a-billion-words/

## Slide 6 — 1. Problems with RNNs: Vanishing and Exploding Gradients

Section title. The four-step chain **h**⁽¹⁾ → **h**⁽²⁾ → **h**⁽³⁾ → **h**⁽⁴⁾ linked by
**W**, with J⁽⁴⁾(θ) rising from **h**⁽⁴⁾. (= lecture 5 slide 51.)

## Slides 7–10 — Vanishing gradient intuition

The chain-rule expansion built up one factor at a time, identical to
[lecture 5 slides 52–55](05-recurrent-neural-networks.md):

- Slide 7: ∂J⁽⁴⁾/∂**h**⁽¹⁾ = **?**
- Slide 8: = (∂**h**⁽²⁾/∂**h**⁽¹⁾) × (∂J⁽⁴⁾/∂**h**⁽²⁾)
- Slide 9: = (∂**h**⁽²⁾/∂**h**⁽¹⁾) × (∂**h**⁽³⁾/∂**h**⁽²⁾) × (∂J⁽⁴⁾/∂**h**⁽³⁾)
- Slide 10: = (∂**h**⁽²⁾/∂**h**⁽¹⁾) × (∂**h**⁽³⁾/∂**h**⁽²⁾) × (∂**h**⁽⁴⁾/∂**h**⁽³⁾) × (∂J⁽⁴⁾/∂**h**⁽⁴⁾)

## Slide 11 — Vanishing gradient intuition (the problem)

Each Jacobian factor is boxed, with the question *What happens if these are small?*

> **Vanishing gradient problem:** When these are small, the gradient signal gets smaller
> and smaller as it backpropagates further

## Slide 12 — Vanishing gradient proof sketch (linear case)

Identical to [lecture 5 slide 57](05-recurrent-neural-networks.md). With σ the identity,
∂**h**⁽ᵗ⁾/∂**h**⁽ᵗ⁻¹⁾ = **W**_h, so for ℓ = *i* − *j*:

  ∂J⁽ⁱ⁾(θ)/∂**h**⁽ʲ⁾ = (∂J⁽ⁱ⁾(θ)/∂**h**⁽ⁱ⁾) **W**_h^ℓ

*If W_h is "small", then this term gets exponentially problematic as ℓ becomes large.*

Source: Pascanu et al, 2013, http://proceedings.mlr.press/v28/pascanu13.pdf

## Slide 13 — Vanishing gradient proof sketch (eigenvalues)

Identical to [lecture 5 slide 58](05-recurrent-neural-networks.md). If the eigenvalues
λ₁ … λ_n of **W**_h are all less than 1 (*sufficient but not necessary*), then in the
eigenvector basis

  (∂J⁽ⁱ⁾(θ)/∂**h**⁽ⁱ⁾) **W**_h^ℓ = Σ_{i=1..n} c_i λ_i^ℓ **q**_i ≈ **0** (for large ℓ)

For nonlinear σ the proof requires λ_i < γ for some γ dependent on dimensionality and σ.

## Slide 14 — Why is vanishing gradient a problem?

Identical to [lecture 5 slide 59](05-recurrent-neural-networks.md), with a near-loss
J⁽²⁾(θ) and a distant loss J⁽⁴⁾(θ) both backpropagating to **h**⁽¹⁾:

> Gradient signal from far away is lost because it's much smaller than gradient signal
> from close-by.
>
> So, model weights are basically updated only with respect to **near effects**, not
> **long-term effects**.

## Slide 15 — Effect of vanishing gradient on RNN-LM

- **LM task:** *When she tried to print her tickets, she found that the printer was out
  of toner. She went to the stationery store to buy more toner. It was very overpriced.
  After installing the toner into the printer, she finally printed her ________*

- To learn from this training example, the RNN-LM needs to **model the dependency**
  between *"tickets"* on the 7th step and the target word *"tickets"* at the end.

- But if the gradient is small, the model **can't learn this dependency**
  - So, the model is **unable to predict similar long-distance dependencies** at test time

- **In practice, a simple RNN will only condition ~7 tokens back** [vague rule-of-thumb]

(The last bullet is new here — lecture 5's version of this slide, slide 60, does not
have it. It is the number slide 25 contrasts LSTMs against.)

## Slide 16 — Why is exploding gradient a problem?

Identical to [lecture 5 slide 61](05-recurrent-neural-networks.md):

  θ^new = θ^old − α ∇_θ J(θ)

- This can cause **bad updates**: too large a step, reaching a weird and bad parameter
  configuration (with large loss)
  - You think you've found a hill to climb, but suddenly you're in Iowa
- In the worst case, **Inf** or **NaN** in your network (then restart from a checkpoint)

A small inset plot of cost against θ shows gradient-descent steps converging toward a
minimum, with a large red arrow shooting off the bottom-right of the slide and a "?"
— the step that overshoots.

## Slide 17 — Gradient clipping: solution for exploding gradient

Identical to [lecture 5 slide 62](05-recurrent-neural-networks.md):

```
Algorithm 1  Pseudo-code for norm clipping
   ĝ ← ∂E/∂θ
   if ‖ĝ‖ ≥ threshold then
       ĝ ← (threshold / ‖ĝ‖) ĝ
   end if
```

- **Intuition**: take a step in the same direction, but a smaller step
- In practice, **remembering to clip gradients is important**, but exploding gradients
  are an easy problem to solve

## Slide 18 — How to fix the vanishing gradient problem?

- The main problem is that *it's too difficult for the RNN to learn to preserve
  information over many timesteps*.

- In a vanilla RNN, the hidden state is constantly being **rewritten**

  **h**⁽ᵗ⁾ = σ( **W**_h **h**⁽ᵗ⁻¹⁾ + **W**_x **x**⁽ᵗ⁾ + **b** )

- Could we design an RNN with separate **memory** which is added to?

## Slide 19 — 2. LSTMs: Apple WWDC Keynote 2016

A full-bleed screenshot of the Apple WWDC 2016 keynote: Craig Federighi on stage beside
a slide showing a large iOS-style rounded-square icon with the numeral **3**. (The
lecture uses this as evidence of how mainstream LSTMs had become.)

## Slide 20 — Long Short-Term Memory RNNs (LSTMs)

- A type of RNN proposed by **Hochreiter and Schmidhuber in 1997** as a solution to the
  problem of vanishing gradients
  - Everyone cites that paper but really a crucial part of the modern LSTM is from
    **Gers et al. (2000)** 💜
- Only started to be recognized as promising through the work of S's student **Alex
  Graves c. 2006**
  - Work in which he also invented **CTC (connectionist temporal classification)** for
    speech recognition
- Became really well-known after **Hinton brought it to Google in 2013**
  - Following Graves having been a postdoc with Hinton

Sources:
Hochreiter and Schmidhuber, 1997. Long short-term memory. https://www.bioinf.jku.at/publications/older/2604.pdf
Gers, Schmidhuber, and Cummins, 2000. Learning to Forget: Continual Prediction with LSTM. https://dl.acm.org/doi/10.1162/089976600300015015
Graves, Fernandez, Gomez, and Schmidhuber, 2006. Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural nets. https://www.cs.toronto.edu/~graves/icml_2006.pdf

## Slide 21 — Long Short-Term Memory RNNs (hidden state and cell state)

- On step *t*, there is a **hidden state h**⁽ᵗ⁾ and a **cell state c**⁽ᵗ⁾
  - Both are vectors length *n*
  - The cell stores **long-term information**
  - The LSTM can **read**, **erase**, and **write** information from the cell
    - The cell becomes conceptually rather like RAM in a computer

- The selection of which information is erased/written/read is controlled by three
  corresponding **gates**  *(gates are calculated things whose values are probabilities)*
  - The gates are also vectors of length *n*
  - On each timestep, each element of the gates can be **open** (1), **closed** (0), or
    somewhere in-between
  - The gates are **dynamic**: their value is computed based on the current context

## Slide 22 — Long Short-Term Memory (LSTM) — the equations

We have a sequence of inputs *x*⁽ᵗ⁾, and we will compute a sequence of hidden states
*h*⁽ᵗ⁾ and cell states *c*⁽ᵗ⁾. On timestep *t*:

  **f**⁽ᵗ⁾ = σ( **W**_f **h**⁽ᵗ⁻¹⁾ + **U**_f **x**⁽ᵗ⁾ + **b**_f )
  **i**⁽ᵗ⁾ = σ( **W**_i **h**⁽ᵗ⁻¹⁾ + **U**_i **x**⁽ᵗ⁾ + **b**_i )
  **o**⁽ᵗ⁾ = σ( **W**_o **h**⁽ᵗ⁻¹⁾ + **U**_o **x**⁽ᵗ⁾ + **b**_o )

  **c̃**⁽ᵗ⁾ = tanh( **W**_c **h**⁽ᵗ⁻¹⁾ + **U**_c **x**⁽ᵗ⁾ + **b**_c )
  **c**⁽ᵗ⁾ = **f**⁽ᵗ⁾ ⊙ **c**⁽ᵗ⁻¹⁾ + **i**⁽ᵗ⁾ ⊙ **c̃**⁽ᵗ⁾
  **h**⁽ᵗ⁾ = **o**⁽ᵗ⁾ ⊙ tanh **c**⁽ᵗ⁾

Each line is labelled in a box on the left:

- **Forget gate** (**f**): controls what is kept vs forgotten, from previous cell state
- **Input gate** (**i**): controls what parts of the new cell content are written to cell
- **Output gate** (**o**): controls what parts of cell are output to hidden state
- **New cell content** (**c̃**): this is the new content to be written to the cell
- **Cell state** (**c**): erase ("forget") some content from last cell state, and write
  ("input") some new cell content
- **Hidden state** (**h**): read ("output") some content from the cell

Two further annotations: the σ on the three gate lines is boxed and labelled
**Sigmoid function**: *all gate values are between 0 and 1*; a brace down the right side
reads *All these are vectors of same length n*; and at the bottom, *Gates are applied
using element-wise (or Hadamard) product:* ⊙.

## Slide 23 — Long Short-Term Memory (LSTM) — the picture

You can think of the LSTM equations visually like this:

The standard Christopher Olah diagram: three repeating green cells **A** in a row, each
taking x_{t−1}, x_t, x_{t+1} from below and emitting h_{t−1}, h_t, h_{t+1} above. The
middle cell is drawn in full: a horizontal line runs straight across the top (the cell
state), crossed first by a ⊗ (the forget gate multiply) and then by a ⊕ (the input
add); below it four yellow neural-network layer boxes — σ, σ, tanh, σ — compute the
gates and the candidate content, feeding pointwise ⊗ operations, and a tanh on the cell
state combines with the last σ to produce h_t.

A legend distinguishes: yellow box = Neural Network Layer; pink circle = Pointwise
Operation; plain arrow = Vector Transfer; merging arrows = Concatenate; splitting arrow
= Copy.

Source: http://colah.github.io/posts/2015-08-Understanding-LSTMs/

## Slide 24 — Long Short-Term Memory (LSTM) — the picture, annotated

The same single-cell diagram, with each part labelled and an output layer added on top:

  **ŷ** = softmax(**Uh** + **b**₂) ∈ ℝ^{|V|}

Labels, following the cell state left to right along the top: **Forget some cell
content** (the ⊗ where c_{t−1} enters), **Write some new cell content** (the ⊕), and
**Output some cell content to the hidden state** (the ⊗ after tanh). Along the bottom,
from h_{t−1} and x_t: **Compute the forget gate** (f_t), **Compute the input gate**
(i_t), **Compute the new cell content** (c̃_t), **Compute the output gate** (o_t).

A magenta callout points at the ⊕: **The + sign is the secret!**

Source: http://colah.github.io/posts/2015-08-Understanding-LSTMs/

## Slide 25 — How does LSTM solve vanishing gradients?

- The LSTM architecture makes it **much easier** for an RNN to **preserve information
  over many timesteps**
  - e.g., if the forget gate is set to 1 for a cell dimension and the input gate set to
    0, then the information of that cell is preserved indefinitely.
  - In contrast, it's harder for a vanilla RNN to learn a recurrent weight matrix *W_h*
    that preserves info in the hidden state
  - In practice, you get about **100 timesteps** rather than about **7**

- However, there are alternative ways of creating more direct and linear pass-through
  connections in models for long distance dependencies

## Slide 26 — Is vanishing/exploding gradient just an RNN problem?

- No! It can be a problem for all neural architectures (including **feed-forward** and
  **convolutional** neural networks), especially **very deep** ones.
  - Due to chain rule / choice of nonlinearity function, gradient can become vanishingly
    small as it backpropagates
  - Thus, lower layers are learned very slowly (i.e., are hard to train)
- Another solution: lots of new deep feedforward/convolutional architectures **add more
  direct connections** (thus allowing the gradient to flow)

For example:
- **Residual connections** aka "ResNet"
- Also known as **skip-connections**
- The **identity connection preserves information** by default
- This makes **deep** networks much **easier to train**

The figure (He et al. 2015, Figure 2, "Residual learning: a building block") shows input
**x** passing through weight layer → relu → weight layer to give ℱ(**x**), while a curved
identity arrow carries **x** around them to a ⊕, producing ℱ(**x**) + **x** → relu.

"Deep Residual Learning for Image Recognition", He et al, 2015. https://arxiv.org/pdf/1512.03385.pdf

## Slide 27 — Is vanishing/exploding gradient just a RNN problem? (other methods)

Other methods:

- **Dense connections** aka "DenseNet"
- Directly connect each layer to all future layers!

  The figure (Huang et al. 2017, Figure 1) shows "A 5-layer dense block with a growth
  rate of *k* = 4. Each layer takes all preceding feature-maps as input." — five
  coloured feature-map stacks with curved arrows from every earlier layer into every
  later one.

- **Highway connections** aka "HighwayNet"
- Similar to residual connections, but the identity connection vs the transformation
  layer is controlled by a **dynamic gate**
- Inspired by LSTMs, but applied to deep feedforward/convolutional networks

  The figure ("Highway Circuit") shows *x* entering an "Information Highway", splitting
  into a transform path *H* gated by *T*, and a carry path gated by *C*, recombining at
  a ⊕ to give *y*.

- **Conclusion**: Though vanishing/exploding gradients are a general problem, **RNNs are
  particularly unstable** due to the repeated multiplication by the **same** weight
  matrix [Bengio et al, 1994]

Sources: "Densely Connected Convolutional Networks", Huang et al, 2017, https://arxiv.org/pdf/1608.06993.pdf ·
"Highway Networks", Srivastava et al, 2015, https://arxiv.org/pdf/1505.00387.pdf ·
"Learning Long-Term Dependencies with Gradient Descent is Difficult", Bengio et al. 1994, http://ai.dinfo.unifi.it/paolo//ps/tnn-94-gradient.pdf

## Slide 28 — 3. Other RNN uses: RNNs can be used for sequence tagging

e.g., **part-of-speech tagging**, named entity recognition

An RNN runs left to right over *the startled cat knocked over the vase*, emitting DT,
JJ, NN, VBN, IN, DT, NN. (= lecture 5 slide 66.)

## Slide 29 — RNNs can be used as a sentence encoder model (the question)

e.g., for **sentiment classification**. An RNN over *overall I enjoyed the movie a lot*
with a **Sentence encoding** feeding the label **positive**. *How to compute sentence
encoding?* (= lecture 5 slide 67.)

## Slide 30 — RNNs can be used as a sentence encoder model (basic way)

**Basic way**: Use final hidden state. (= lecture 5 slide 68.)

## Slide 31 — RNNs can be used as a sentence encoder model (better way)

**Usually better**: Take element-wise max or mean of all hidden states.
(= lecture 5 slide 69.)

## Slide 32 — RNN-LMs can be used to generate text based on other information

e.g., **speech recognition**, machine translation, summarization. An audio waveform
conditions an RNN-LM that generates *what's the weather* from `<START>`.

> This is an example of a *conditional language model*.
> We'll see Machine Translation as an example in more detail

(= lecture 5 slide 71.)

## Slide 33 — 4. Bidirectional and Multi-layer RNNs: motivation

Task: Sentiment Classification.

A forward RNN runs over *the movie was terribly exciting !*, its hidden states combined
by element-wise mean/max into a **Sentence encoding** that predicts **positive**. The
hidden state above *terribly* is boxed and called out:

> We can regard this hidden state as a representation of the word *"terribly"* in the
> context of this sentence. We call this a *contextual representation*.

A second callout on the right:

> These contextual representations only contain information about the *left* context
> (e.g. *"the movie was"*).
>
> **What about *right* context?**
>
> In this example, *"exciting"* is in the right context and this modifies the meaning of
> *"terribly"* (from negative to positive)

## Slide 34 — Bidirectional RNNs (the diagram)

Three rows over the same sentence *the movie was terribly exciting !*:

- **Forward RNN** (red) running left to right along the bottom
- **Backward RNN** (green) running right to left in the middle
- **Concatenated hidden states** (grey) at the top, each stacking the forward and
  backward state for that position

The concatenated state above *terribly* is boxed:

> This contextual representation of "terribly" has both left and right context!

## Slide 35 — Bidirectional RNNs (the equations)

On timestep *t*:

  Forward RNN:  **h**→⁽ᵗ⁾ = RNN_FW( **h**→⁽ᵗ⁻¹⁾, **x**⁽ᵗ⁾ )
  Backward RNN: **h**←⁽ᵗ⁾ = RNN_BW( **h**←⁽ᵗ⁺¹⁾, **x**⁽ᵗ⁾ )
  Concatenated hidden states: **h**⁽ᵗ⁾ = [ **h**→⁽ᵗ⁾ ; **h**←⁽ᵗ⁾ ]

Annotations: RNN_FW is boxed —

> This is a general notation to mean "compute one forward step of the RNN" — it could be
> a simple RNN or LSTM computation.

*Generally, these two RNNs have separate weights.* And on the concatenation:

> We regard this as "the hidden state" of a bidirectional RNN. This is what we pass on to
> the next parts of the network.

## Slide 36 — Bidirectional RNNs: simplified diagram

The same sentence with a single row of hidden states joined by **two-way** arrows.

> The two-way arrows indicate bidirectionality and the depicted hidden states are
> assumed to be the concatenated forwards+backwards states

## Slide 37 — Bidirectional RNNs (when they apply)

- Note: bidirectional RNNs are only applicable if you have access to the **entire input
  sequence**
  - They are **not** applicable to Language Modeling, because in LM you *only* have left
    context available.

- If you do have entire input sequence (e.g., any kind of encoding), **bidirectionality
  is powerful** (you should use it by default).

- For example, **BERT** (**Bidirectional** Encoder Representations from Transformers) is
  a powerful pretrained contextual representation system **built on bidirectionality**.
  - You will learn more about **transformers**, including BERT, in a couple of weeks!

## Slide 38 — Multi-layer RNNs

- RNNs are already "deep" on one dimension (they unroll over many timesteps)
- We can also make them "deep" in another dimension by **applying multiple RNNs** — this
  is a multi-layer RNN.
- This allows the network to compute **more complex representations**
  - The **lower RNNs** should **compute lower-level features** and the **higher RNNs**
    should compute **higher-level features**.
- Multi-layer RNNs are also called *stacked RNNs*.

(Illustrated with three stacked ice-cream cones.)

## Slide 39 — Multi-layer RNNs (the diagram)

Three rows of RNN hidden states — **RNN layer 1** at the bottom taking the words *the
movie was terribly exciting !*, then **RNN layer 2**, then **RNN layer 3** — each layer
running left to right and feeding upward.

> The hidden states from RNN layer *i* are the inputs to RNN layer *i*+1

## Slide 40 — Multi-layer RNNs in practice

- Multi-layer or stacked RNNs allow a network to compute **more complex representations**
  — they work better than just have one layer of high-dimensional encodings!
  - The **lower RNNs** should **compute lower-level features** and the **higher RNNs**
    should compute **higher-level features**.
- **High-performing RNNs are usually multi-layer** (but aren't as deep as convolutional
  or feed-forward networks)
- For example: In a 2017 paper, Britz et al. find that for Neural Machine Translation,
  **2 to 4 layers** is best for the encoder RNN, and **4 layers** is best for the
  decoder RNN
  - Often 2 layers is a lot better than 1, and 3 might be a little better than 2
  - Usually, **skip-connections**/**dense-connections** are needed to train deeper RNNs
    (e.g., **8 layers**)
- **Transformer**-based networks (e.g., BERT) are usually deeper, like **12 or 24 layers**.
  - You will learn about Transformers later; they have a lot of skipping-like connections

"Massive Exploration of Neural Machine Translation Architecutres", Britz et al, 2017. https://arxiv.org/pdf/1703.03906.pdf

## Slide 41 — LSTMs: real-world success

- In **2013–2015**, LSTMs started achieving state-of-the-art results
  - Successful tasks include handwriting recognition, speech recognition, machine
    translation, parsing, and image captioning, as well as language models
  - **LSTMs** became the **dominant approach** for most NLP tasks

- **Now (2019–2024)**, **Transformers** have become dominant for all tasks
  - For example, in **WMT** (a Machine Translation conference + competition):
    - In WMT 2014, there were **0** neural machine translation systems (!)
    - In **WMT 2016**, the summary report contains "**RNN**" **44** times (and these
      systems won)
    - In WMT 2019: "**RNN**" **7** times, "**Transformer**" **105** times

Sources: "Findings of the 2016 Conference on Machine Translation (WMT16)", Bojar et al. 2016, http://www.statmt.org/wmt16/pdf/W16-2301.pdf ·
"Findings of the 2018 Conference on Machine Translation (WMT18)", Bojar et al. 2018, http://www.statmt.org/wmt18/pdf/WMT028.pdf ·
"Findings of the 2019 Conference on Machine Translation (WMT19)", Barrault et al. 2019, http://www.statmt.org/wmt18/pdf/WMT028.pdf

## Slide 42 — 5. Machine Translation

**Machine Translation (MT)** is the task of translating a sentence *x* from one language
(the **source language**) to a sentence *y* in another language (the **target language**).

  *x:  L'homme est né libre, et partout il est dans les fers*
    ↓
  *y:  Man is born free, but everywhere he is in chains*

— Rousseau

## Slide 43 — The early history of MT: 1950s

- Machine translation research began in the **early 1950s** on machines less powerful
  than high school calculators (before term "A.I." coined!)
- Concurrent with foundational work on automata, formal languages, probabilities, and
  information theory
- MT heavily funded by military, but basically just simple rule-based systems doing word
  substitution
- Human language is more complicated than that, and varies more across languages!
- Little understanding of natural language syntax, semantics, pragmatics
- Problem soon appeared intractable

1 minute video showing 1954 MT: https://youtu.be/K-HfpsHPmvw

## Slide 44 — The early history of MT: 1950s (the newsreel)

A full-bleed still from the Paramount News reel: black-and-white footage of a room-sized
computer with the title card **ELECTRONIC "BRAIN" Translates RUSSIAN to ENGLISH**.

## Slide 45 — 1990s-2010s: Statistical Machine Translation

- Core idea: Learn a **probabilistic model** from **data**
- Suppose we're translating French → English.
- We want to find **best English sentence *y*, given French sentence *x***

  argmax_y P(y|x)

- Use Bayes Rule to break this down into **two components** to be learned separately:

  = argmax_y P(x|y) P(y)

  - P(x|y) — **Translation Model**: Models how words and phrases should be translated
    (*fidelity*). Learned from parallel data.
  - P(y) — **Language Model**: Models how to write good English (*fluency*). Learned
    from monolingual data.

## Slide 46 — What happens in translation isn't trivial to model!

A word-alignment figure: the German *Morgen | fliege | ich | nach Kanada | zur Konferenz*
is linked by crossing arrows to the English *Tomorrow | I | will fly | to the conference
| in Canada* — the crossings showing that the word order does not correspond.

Below, a Chinese sentence and its translations:

> 1519年600名西班牙人在墨西哥登陆，去征服几百万人口的阿兹特克帝国，初次交锋他们损兵三分之二。

Reference: In 1519, six hundred Spaniards landed in Mexico to conquer **the Aztec Empire**
**with a population of a few million**. They lost two thirds of their soldiers in the
first clash.

- translate.google.com (2009): 1519 600 Spaniards landed in Mexico, *millions of people
  to conquer the Aztec empire*, the first two-thirds of soldiers against their loss.
- translate.google.com (2013): 1519 600 Spaniards landed in Mexico *to conquer the Aztec
  empire, hundreds of millions of people*, the initial confrontation loss of soldiers
  two-thirds.
- translate.google.com (2015): 1519 600 Spaniards landed in Mexico, *millions of people
  to conquer the Aztec empire*, the first two-thirds of the loss of soldiers they clash.

## Slide 47 — 1990s–2010s: Statistical Machine Translation (the cost)

- SMT was a **huge research field**
- The best systems were **extremely complex**
  - Hundreds of important details
- Systems had many **separately-designed subcomponents**
  - Lots of **feature engineering**
    - Need to design features to capture particular language phenomena
  - Required compiling and maintaining **extra resources**
    - Like tables of equivalent phrases
  - Lots of **human effort** to maintain
    - Repeated effort for each language pair!

## Slide 48 — 2014

A full-bleed image of a meteor striking the Earth, labelled **Neural Machine
Translation** (the meteor) and **MT research** (the planet), dated **2014**, captioned
*(dramatic reenactment)*.

## Slide 49 — 6. What is Neural Machine Translation?

- **Neural Machine Translation (NMT)** is a way to do Machine Translation with a *single
  end-to-end neural network*

- The neural network architecture is called a **sequence-to-sequence** model (aka
  **seq2seq**) and it involves *two* RNNs

## Slide 50 — Neural Machine Translation (NMT): the sequence-to-sequence model

The canonical seq2seq diagram, left to right:

- **Encoder RNN** reads the source sentence (input) *il  m'  a  entarté*. Its final
  hidden state is boxed in orange: *Encoding of the source sentence. Provides initial
  hidden state for Decoder RNN.*
- **Decoder RNN** starts from `<START>` and generates the target sentence (output)
  *he hit me with a pie `<END>`*, taking an **argmax** at each step and feeding the
  emitted word back in as the next step's input (drawn as dotted pink arrows).

Callouts:
> **Encoder RNN** produces an **encoding** of the source sentence.
>
> **Decoder RNN** is a Language Model that generates target sentence, *conditioned on
> encoding*.
>
> Note: This diagram shows **test time** behavior: decoder output is fed in ⋯▸ as next
> step's input

## Slide 51 — Sequence-to-sequence is versatile!

- The general notion here is an **encoder-decoder** model
  - One neural network takes input and produces a neural representation
  - Another network produces output based on that neural representation
  - If the input and output are sequences, we call it a seq2seq model

- Sequence-to-sequence is useful for *more than just MT*
- Many NLP tasks can be phrased as sequence-to-sequence:
  - **Summarization** (long text → short text)
  - **Dialogue** (previous utterances → next utterance)
  - **Parsing** (input text → output parse as sequence)
  - **Code generation** (natural language → Python code)

## Slide 52 — Neural Machine Translation (NMT) as a conditional LM

- The **sequence-to-sequence** model is an example of a **Conditional Language Model**
  - **Language Model** because the decoder is predicting the next word of the target
    sentence *y*
  - **Conditional** because its predictions are *also* conditioned on the source
    sentence *x*

- NMT directly calculates P(y|x):

  P(y|x) = P(y₁|x) P(y₂|y₁, x) P(y₃|y₁, y₂, x) … P(y_T | y₁, …, y_{T−1}, x)

  The last factor is braced and labelled *Probability of next target word, given target
  words so far and source sentence x*.

- **Question:** How to train an NMT system?
- **(Easy) Answer:** Get a big parallel corpus…
  - But there is now exciting work on "unsupervised NMT", data augmentation, etc.

## Slide 53 — Training a Neural Machine Translation system

The same encoder–decoder diagram, now at training time: the source sentence *il m' a
entarté* comes from the corpus and so does the target sentence — the decoder is fed
`<START> he hit me with a pie` from the corpus rather than its own output. Each decoder
step produces ŷ_t and a loss J_t; J₁, J₄ and J₇ are boxed and annotated *= negative log
prob of "he"*, *of "with"*, and *of `<END>`*.

  J = (1/T) Σ_{t=1..T} J_t = J₁ + J₂ + J₃ + J₄ + J₅ + J₆ + J₇

Fat blue arrows show the gradient flowing back from every loss, through the decoder, and
on through the encoder.

> **Seq2seq is optimized as a single system.** Backpropagation operates *"end-to-end"*.

## Slide 54 — Multi-layer deep encoder-decoder machine translation net

[Sutskever et al. 2014; Luong et al. 2015]

A three-layer stacked encoder–decoder. The **Encoder: Builds up sentence meaning** (dark
red cells, each drawn as a small column of numbers like 0.1 / 0.6 / −0.1 / −0.4 / 0.2)
reads the **Source sentence** *Die Proteste waren am Wochenende eskaliert `<EOS>`*. The
blue column at `<EOS>` carries across into the **Decoder** (green cells), which emits
**Translation generated** *The protests escalated over the weekend `<EOS>`*, with pink
curved arrows showing **Feeding in last word** — each generated word fed back as the
next input.

> The hidden states from RNN layer *i* are the inputs to RNN layer *i*+1

Handwritten at the bottom of the slide, under the single blue column joining encoder to
decoder: **Conditioning = Bottleneck**.

## Slide 55 — NMT: the first big success story of NLP Deep Learning

Neural Machine Translation went from a **fringe research attempt** in **2014** to the
**leading standard method** in **2016**

- **2014**: First seq2seq paper published [Sutskever et al. 2014]

- **2016**: Google Translate switches from SMT to NMT — and by 2018 everyone had
  - https://www.nytimes.com/2016/12/14/magazine/the-great-ai-awakening.html

  (Logos: Microsoft, SYSTRAN, Google, Facebook, Baidu, NetEase, Tencent, Sogou.)

- This was amazing!
  - **SMT** systems, built by **hundreds** of engineers over many **years**, were
    outperformed by NMT systems trained by **small groups** of engineers in a few
    **months**

## Slide 56 — In summary

Lots of new information today! What are some of the **practical takeaways**?

1. **LSTMs are powerful** (illustrated with the Olah LSTM diagram)
2. **Clip your gradients** (illustrated with the Pascanu error-surface plot — a smooth
   loss surface with a sheer cliff face, and a trajectory that gets flung off it)
3. **Use bidirectionality when possible** (illustrated with the simplified bidirectional
   diagram over *the movie was terribly exciting !*)
4. **Encoder-Decoder Neural Machine Translation Systems work very well** (illustrated
   with the multi-layer deep encoder-decoder net from slide 54)
