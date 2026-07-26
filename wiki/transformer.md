# Transformer

The architecture built on top of [self-attention](self-attention.md) that has underpinned
almost every state-of-the-art NLP model since 2018 — and, increasingly, models outside NLP
too. Introduced by Vaswani et al. (2017), "Attention Is All You Need," and covered in
[lecture 8](08-self-attention-and-transformers.md) (slides 6, 33–61).

## Why it displaced recurrence

By the time this lecture is taught, the case is already made empirically rather than just
architecturally (slides 7–14). On the original paper's own machine translation benchmark,
the Transformer (big) reached 28.4 BLEU on WMT 2014 English-German and 41.8 on
English-French, at a fraction of the training cost (in FLOPs) of the ensembled
recurrent/convolutional systems it beat. Transformer-based models have since taken over
SuperGLUE, dominate large-language-model leaderboards, and — via Vision Transformers and
AlphaFold2 — extended well outside NLP into image classification and protein folding.
Kaplan et al. (2020)'s scaling laws show Transformer language-modeling loss falling
smoothly, as a power law, as model size, training data, and compute all increase together,
with no sign of the trend stopping at the scales tested.

The underlying reason recurrence lost is covered under
[self-attention](self-attention.md#why-move-beyond-recurrence): RNNs have $O(n)$
unparallelizable steps and $O(n)$ interaction distance between distant words; self-attention
has $O(1)$ of each, at the cost of $O(n^2 \cdot d)$ compute per layer versus recurrence's
$O(n \cdot d^2)$ — a trade that favors self-attention whenever sequence length $n$ is small
relative to model dimensionality $d$ (slide 18).

## Multi-head attention

Single-headed self-attention forces one query/key/value projection to capture every reason
a word might want to look at another word — but there are often several *different*
reasons at once. Manning's running example (lecture 8, ≈42:38, slide 44–45): representing
"learned" in *I went to Stanford, CS224N, and learned*, you might want to attend to
"Stanford, CS224N" for one kind of information (what was learned) and to "I … went … and
learned" for another (syntactically related words). A single averaged attention operation
struggles to do both at once.

**Multi-head attention** runs several independent self-attention "heads" in parallel, each
with its own query, key and value matrices, and combines their outputs (slide 45):

$$Q_\ell, K_\ell, V_\ell \in \mathbb{R}^{d \times d/h}, \qquad \ell = 1, \dots, h$$
$$\text{output}_\ell = \operatorname{softmax}\left(XQ_\ell K_\ell^\top X^\top\right) X V_\ell, \qquad \text{output}_\ell \in \mathbb{R}^{d/h}$$
$$\text{output} = Y[\text{output}_1; \dots; \text{output}_h], \qquad Y \in \mathbb{R}^{d \times d}$$

Each head projects down to a *lower*-dimensional space ($d/h$ instead of $d$), so — worked
through visually in the lecture (≈49:30–51:53) — running $h$ heads costs no more than
running one full-dimensional head; the total compute is the same, just reshaped. Nobody
guarantees the heads learn to specialize in different things, but empirically they do,
"just like we hope different dimensions in our feed-forward layers will learn different
things because of lack of symmetry" (≈44:58). After training, individual heads can often be
identified as doing something specific — some pick out syntactic dependencies, others do a
kind of global context-averaging — though what a head is "looking at" gets harder to
interpret the deeper into the network it sits, since by then it has already absorbed context
from everywhere else (≈55:45). Parameters are independent at every block; the number of
heads is typically kept constant across blocks, chosen so each head gets a reasonable
number of dimensions to work with (often around 64, so $h \approx d / 64$) — nobody has
found a strong reason to vary head count by layer, though after training you can often zero
out some heads with little effect (≈53:24–54:58).

## Scaled dot-product attention

A subtler fix, needed once you start stacking many self-attention layers with
[layer normalization](#residual-connections-and-layer-normalization) between them (slide
37). After layer norm, each vector's elements have mean 0 and variance 1 — but a dot product
between two such vectors still has variance that scales with the model dimensionality
$d_k$ (mean of a sum of $d_k$ terms is $d_k \cdot 0 = 0$; variance of a sum of $d_k$
independent terms is $d_k \cdot 1 = d_k$). As $d_k$ grows, dot products can become very
large in either direction, which pushes softmax inputs to extremes and shrinks gradients.
The fix is a single division:

$$\text{Output} = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

dividing by $\sqrt{d_k}$ resets the variance of the dot product back to 1, keeping the
attention distribution closer to uniform at initialization rather than saturated toward one
key (lecture 8, ≈56:30–58:02).

## Residual connections and layer normalization

Two general deep-network optimization tricks, applied around both the self-attention and
the feed-forward sublayer, and usually drawn together as a single "Add & Norm" box in
Transformer diagrams (slide 33).

**Residual connections** (slide 34) — covered in general, with the loss-landscape picture,
under [vanishing and exploding gradients](vanishing-and-exploding-gradients.md); in the
Transformer the form is $x_\ell = F(x_{\ell-1}) + x_{\ell-1}$, where $F$ is whichever
sublayer (self-attention or feed-forward) sits at that point. Deep networks are surprisingly
bad at learning the identity function on their own, so adding the raw input back in means
each layer only has to learn a *residual* correction, and gradients have a direct path
around any layer that is struggling to learn.

**Layer normalization** [Ba et al., 2016] (slides 35–36) addresses a different problem: the
input distribution to a given layer keeps shifting as earlier layers' parameters update
during training, which makes that layer hard to train against a moving target. The fix is
to renormalize each individual word vector, at every layer, to zero mean and unit standard
deviation — computed **per word**, not shared across the words in a sequence or across a
training batch (both were explicitly asked about in lecture and confirmed no, ≈1:05:46):

$$\mu^\ell = \frac{1}{H} \sum_{i=1}^H a_i^\ell \qquad \sigma^\ell = \sqrt{\frac{1}{H} \sum_{i=1}^H \left(a_i^\ell - \mu^\ell\right)^2}$$
$$x^{\ell\prime} = \frac{x^\ell - \mu^\ell}{\sigma^\ell + \epsilon}$$

with $\epsilon$ a small constant to avoid dividing by zero when there's little variation,
and optional learned scale/offset parameters to stretch the normalized vector back out
afterward (their practical importance is small — see the course's lecture notes). This is
why layer norm was invented as a *replacement* for batch normalization: batch norm shares
statistics across a batch, which makes a given example's forward pass depend, undesirably,
on which other examples happen to be in the same batch.

## Encoder, decoder, and encoder-decoder

Assembling multi-head attention, the feed-forward sublayer, and Add & Norm around each
gives one Transformer **block**; the full architecture repeats a block some number of times
(6, in the original paper) and comes in three shapes (lecture 8, ≈1:08:52–1:12:45):

- **Transformer encoder** — self-attention with **no masking** in every block, since an
  encoder is meant to build a representation of a complete input (e.g. a source sentence)
  and every word should be free to attend to every other word, in both directions.
- **Transformer decoder** — self-attention **with** the future-masking described under
  [self-attention](self-attention.md#3-looking-at-the-future), because a decoder generates
  one token at a time and must not see ahead. This is the architecture behind
  decoder-only language models.
- **Transformer encoder-decoder** — the original architecture the "Attention Is All You
  Need" paper actually presents, and the one used for machine translation. The decoder gets
  a *third* sublayer per block, **cross-attention**, inserted between the masked
  self-attention and the feed-forward layer: its queries come from the decoder's own
  vectors $z_i$, but its keys and values are drawn from the encoder's output $h_i$,
$$k_i = Kh_i \qquad v_i = Vh_i \qquad q_i = Qz_i$$

  so every decoder position can attend over the *entire* encoded source sentence — the
  direct Transformer analogue of the encoder-decoder attention covered under
  [attention](attention.md), just computed with the query-key-value machinery of
  self-attention instead of a bespoke scoring function.

A final linear layer projects each decoder output vector up to vocabulary-sized logits, and
a softmax turns those into a probability distribution over the next word — the same
generation step every language model in the course uses (slides 53–55).

## Drawbacks and variants

The lecture is explicit that the Transformer, as just described, is not the end of the
search for a better architecture (≈41:51, slides 57–61):

- **Quadratic compute in self-attention.** Computing every pair of word interactions costs
  $O(n^2)$ in the sequence length $n$ — a real step *backwards* from recurrence's $O(n)$
  cost, and the reason context-length limits remain a practical constraint even in the
  largest 2024-era language models (a model dimensionality around a thousand makes $n^2
  \cdot d$ manageable at $n \approx 30$, but explodes once $n$ reaches the tens of
  thousands). **Linformer** [Wang et al., 2020] projects the keys and values down to a
  lower-dimensional space before attention, cutting the quadratic term; **BigBird**
  [Zaheer et al., 2021] replaces full all-pairs attention with a mix of local-window,
  global, and random interactions. In practice, though, at the scale of GPT-3 or ChatGPT
  most compute is spent elsewhere than in the attention operation itself, so whether
  eliminating quadratic attention is actually necessary remains an open question (≈1:15:51).
- **Position representations.** Absolute sinusoidal or learned position indices are not
  obviously the best choice — see the relative-position, dependency-syntax, and rotary
  (RoPE) alternatives named under
  [self-attention](self-attention.md#1-no-notion-of-sequence-order).
- **Architectural modifications, mostly, don't help.** A large controlled study, "Do
  Transformer Modifications Transfer Across Implementations and Applications?" (Narang et
  al.), tested roughly two dozen proposed changes — different nonlinearities, normalization
  schemes, embedding-tying strategies, mixture-of-experts variants — and found that *most*
  do not meaningfully improve performance once evaluated consistently. The original
  architecture plus a small number of confirmed improvements (e.g. changing the
  feed-forward sublayer's nonlinearity) has had unusually lasting power (≈1:16:13).

## Related pages

- [Self-attention](self-attention.md) — the query-key-value mechanism and the three fixes
  (position, nonlinearity, masking) this architecture is built from.
- [Attention](attention.md) — the general mechanism, including the encoder-decoder
  attention that cross-attention specializes.
- [Vanishing and exploding gradients](vanishing-and-exploding-gradients.md) — residual
  connections, covered there in general and applied here.
- [Lecture 8 — Self-Attention and Transformers](08-self-attention-and-transformers.md)
