# Attention

The idea that let a decoder look back at *any* part of an encoder's output instead of
relying on a single fixed-size summary vector. Introduced in
[lecture 7](07-attention-final-projects-and-llm-intro.md) (slides 7–28) as the fix for the
sequence-to-sequence [bottleneck](seq2seq-and-encoder-decoder.md), and generalized there
into a technique used everywhere in deep learning — including as the core building block
of [self-attention](self-attention.md) and the [Transformer](transformer.md) the very next
lecture.

## The bottleneck it fixes

A vanilla seq2seq model stuffs everything useful about the source sentence into one hidden
vector, which the decoder then has to draw on for the entire translation (lecture 7,
≈13:20). That's plausible for a four-word sentence and implausible for a forty-word one —
you can make the hidden state bigger or use a multi-layer LSTM, but it's still "a very
questionable thing to do," and not how a human translator works: a person re-reads earlier
parts of the source sentence as they translate, rather than trying to hold the whole thing
in their head at once (≈14:09).

Attention gives the decoder exactly that: on each decoder step, a direct connection back to
every encoder hidden state, so it can look at whichever part of the source is relevant right
now (≈14:54).

## The mechanism

On decoder step $t$, with encoder hidden states $h_1, \dots, h_N$ and decoder hidden state
$s_t$ (slide 22):

1. **Attention scores.** Compare $s_t$ against every encoder hidden state:
$$e^t = \left[s_t^\top h_1, \dots, s_t^\top h_N\right] \in \mathbb{R}^N$$

2. **Attention distribution.** Softmax the scores into a probability distribution over
   source positions:

$$\alpha^t = \operatorname{softmax}(e^t) \in \mathbb{R}^N$$

3. **Attention output.** Take the weighted sum of encoder hidden states this distribution
   implies:

$$a_t = \sum_{i=1}^{N} \alpha^t_i h_i \in \mathbb{R}^h$$

4. **Combine with the decoder state.** Concatenate the attention output with the decoder's
   own hidden state, $[a_t; s_t] \in \mathbb{R}^{2h}$, and proceed as in the non-attention
   seq2seq model — multiply by an output matrix and softmax to choose the next word.

Worked example (lecture 7, ≈16:29, slide 7–21): translating the French *il a m'entarté*
("he pied me") into "he hit me with a pie." On the first decoder step the attention
distribution puts nearly all its weight on *il* ("he"), producing "he." On later steps the
distribution shifts onto *entarté* ("pied") for "hit," "with," "a," and "pie" — the pie-ing
verb spreads its attention weight across the whole English verb phrase it maps to. Slide 23
shows this as a soft alignment matrix, and the point Manning makes is that nobody trained an
alignment system directly: **the network learned this alignment by itself** as a side effect
of learning to translate well.

## Why it matters

Attention was invented in 2014, in the earliest neural machine translation work — genuinely
novel, unlike the RNNs, LSTMs and feed-forward networks around it, which were all invented
before 2000 and simply waited for enough data and compute (≈12:33). It significantly
improved machine translation performance, and effectively every machine translation system
since has used it. Beyond the accuracy gain, it offers several things at once (slide 23):

- **Solves the bottleneck problem** — the decoder can look directly at the source instead
  of routing everything through one fixed-size vector.
- **Helps with the [vanishing gradient](vanishing-and-exploding-gradients.md) problem** —
  attention gives shortcut connections to every encoder hidden state, the same kind of
  direct path that motivates residual connections.
- **Gives interpretability for free** — inspecting the attention distribution shows what
  the model was focusing on, producing a soft word alignment nobody explicitly trained.

## History: three ways to score attention

The dot product $s^\top h$ (used above) is the simplest scoring function, but not the only
one, and the lecture presents the history slightly out of order (≈31:19). Given some values
$h_1, \dots, h_N \in \mathbb{R}^{d_1}$ and a query $s \in \mathbb{R}^{d_2}$ (slide 25–26):

- **Additive attention** [Bahdanau, Cho, and Bengio 2014] — the *original* form, and the
  first neural machine translation system to use attention (a University of Montreal team,
  with a far more modest compute budget than Google's contemporaneous eight-layer-deep pure
  LSTM system). A small feed-forward network scores each pair:
$$e_i = v^\top \tanh(W_1 h_i + W_2 s) \in \mathbb{R}$$

  with $W_1 \in \mathbb{R}^{d_3 \times d_1}$, $W_2 \in \mathbb{R}^{d_3 \times d_2}$, and
  $v \in \mathbb{R}^{d_3}$ all learned; $d_3$ is a hyperparameter. Despite the name, it's
  really just a little neural net, not literally "addition" — Manning calls "additive
  attention" a fairly weird name for it. It's more complex and slower than the alternatives
  below, but a later large-scale architecture search (Britz et al. 2017) found that, with
  good hyperparameter tuning, it can actually outperform them.
- **Multiplicative (bilinear) attention** [Luong, Pham, and Manning 2015] — proposed the
  following year as a simpler alternative:
$$e_i = s^\top W h_i \in \mathbb{R}, \qquad W \in \mathbb{R}^{d_2 \times d_1}$$

  A learned matrix sits between the two vectors, so they no longer need to align
  dimension-by-dimension — $W$ can learn to match, say, "where the decoder stores word
  meaning" against "where the encoder stores word meaning" even if those live in different
  coordinates. Manning still thinks "bilinear attention" is the better name, but
  "multiplicative attention" is what stuck.
- **Reduced-rank multiplicative attention** — $W$ has $d_1 \times d_2$ parameters, which
  gets large fast (a length-1000 hidden state means a million parameters). Factor it as two
  skinny matrices, $U \in \mathbb{R}^{k \times d_2}$ and $V \in \mathbb{R}^{k \times d_1}$
  with $k \ll d_1, d_2$:
$$e_i = s^\top(U^\top V)h_i = (Us)^\top(Vh_i)$$

  A little linear algebra shows this is exactly the same as projecting both vectors into a
  lower-dimensional space and dot-producting them there — which is precisely what
  [self-attention in Transformers](self-attention.md) does with its query and key
  projections (≈30:33).
- **Basic dot-product attention** — $e_i = s^\top h_i \in \mathbb{R}$, the simplest case
  (equivalent to the multiplicative form with $W$ fixed to the identity, so it requires
  $d_1 = d_2$).

## The general definition

Attention outgrew machine translation almost immediately (slide 27–28). Stated generally:
given a set of vector **values** and a vector **query**, attention is a technique to
compute a weighted sum of the values, dependent on the query — you'd say "the query
*attends to* the values." In the seq2seq case above, each decoder hidden state is the query
and the encoder hidden states are the values.

Two ways to think about what this buys you:

- The weighted sum is a **selective summary** of the values, where the query determines
  which values to focus on.
- It's a way to obtain a **fixed-size representation of an arbitrary set of
  representations**, dependent on some other representation.

Manning's summary: attention has become *"the powerful, flexible, general way [to do]
pointer and memory manipulation in all deep learning models"* — a genuinely new idea from
after 2010, and it came out of machine translation. The next lecture pushes this to its
logical extreme: instead of one query attending to a separate set of values, every word in
a sequence attends to every other word in the *same* sequence — see
[self-attention](self-attention.md).

## Related pages

- [Self-attention](self-attention.md) — attention applied within a single sequence, the
  core building block of the Transformer.
- [Transformer](transformer.md) — the architecture built out of self-attention.
- [Sequence-to-sequence and encoder-decoder models](seq2seq-and-encoder-decoder.md) — the
  bottleneck problem attention was invented to solve.
- [Vanishing and exploding gradients](vanishing-and-exploding-gradients.md) — the other
  problem attention's shortcut connections help with.
- [Lecture 7 — Attention, Final Projects and LLM Intro](07-attention-final-projects-and-llm-intro.md)
