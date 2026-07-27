# Connectionist Temporal Classification (CTC)

A loss function for sequence-to-sequence problems where the input is much longer than the output,
the alignment between them is unknown, and that alignment is **monotonic**. It is the training
objective behind classical speech recognition, handwriting recognition, and the speech
[brain-computer interface](brain-computer-interfaces.md) in
[lecture 14](14-brain-computer-interfaces.md) (slides 46–48, 53–55, ≈53:22). Fan notes that
CS224S covers it properly; what follows is the working version a CS224N student needs.

## The problem CTC solves

Take the neural speech decoder. The input is one feature vector per **20 ms** bin — thousands of
frames for a sentence. The output is a phoneme string a few dozen symbols long. Three things are
true at once:

1. **The lengths do not match**, by orders of magnitude.
2. **You do not know which frames produced which symbol.** The training data pairs a whole
   recording with a whole sentence, not frame with phoneme. Hand-labelling the alignment is
   infeasible.
3. **The alignment is monotonic.** The first frames correspond to the first phonemes and never
   the last.

### Why not an encoder-decoder

The default CS224N answer to a sequence-to-sequence problem is an
[encoder-decoder with attention](seq2seq-and-encoder-decoder.md). Fan rejects it explicitly, and
the reason is worth internalizing: encoder-decoder models permit **arbitrary** alignment between
input and output. That is exactly the right power for
[machine translation](machine-translation.md), where a target word may depend on a source word
anywhere in the sentence. Here it is power you do not need — and every unnecessary degree of
freedom has to be paid for in training data, which a single-participant BCI study does not have
(≈51:47, slide 45).

CTC buys the monotonic case and nothing more.

## The blank symbol and the collapsing rule

CTC lets the network emit **one symbol per input frame**, so input and output are the same length
by construction. The output alphabet is extended with one extra symbol, the **blank**, written
$\epsilon$.

A frame-level output is then converted to the final sequence by two steps, in this order
(slide 47):

1. **Merge repeated symbols.**
2. **Delete the blanks.**

So `h h e ε ε l l l ε l l o` collapses to `hello`.

The order matters, and the blank is what makes genuine repeats expressible. Without $\epsilon$,
merging repeats would make `hello` unspellable — the two `l`s would collapse into one. With it,
a blank between them (`l l l ε l l`) protects the boundary.

## Training: marginalize over all alignments

Because many frame-level paths collapse to the same string, no single path is the "right" label.
Slide 48 traces three different alignments — `εheleεelleεo`, `heεllεlleεo`, `hhellεεloo` — all
collapsing to `hello`.

CTC therefore defines the probability of the target as the **sum over every valid alignment**:

$$p(Y \mid X) = \sum_{A \in \mathcal{A}_{X,Y}} \prod_{t=1}^{T} p_t(a_t \mid X)$$

where $X$ is the input, $Y$ the target sequence, $T$ the number of input frames, $\mathcal{A}_{X,Y}$
the set of frame-level alignments that collapse to $Y$, and $p_t(a_t \mid X)$ the network's
probability of emitting symbol $a_t$ at frame $t$. The inner product scores one alignment
step by step; the outer sum marginalizes over all of them. Training maximizes this, so the
network never has to be told where the boundaries are — it learns an alignment as a by-product of
learning the sequence.

The sum is over exponentially many paths and is computed by dynamic programming. Fan skips the
derivation in lecture (≈55:40) and so does this page; the point to carry is that the alignment is
*latent*, not supervised.

## Inference: beam search over a collapsing alphabet

At test time the network produces, for each frame, a distribution over symbols — in the BCI case
a **phoneme probability** column at every 20 ms step (slide 53). The task is

$$\mathbf{Y}^* = \arg\max_{\mathbf{Y}} P(\mathbf{Y} \mid \mathbf{X})$$

and, as in [Assignment 3](decoding-algorithms.md), you approximate it with **beam search**:
maintain the top-$k$ hypotheses, extend each by every symbol, keep the best $k$ again.

CTC adds one wrinkle that ordinary beam search does not have, and slides 54–55 are built around
it. Because hypotheses are scored *after collapsing*, **different extensions can merge into the
same prefix** — a path ending `…ε a` and a path ending `…a a` both collapse to the same string.
A correct CTC beam search must therefore sum the probabilities of merging paths rather than
treating them as separate beam entries. Fan flags the caveat and moves on ("I'm not going to
expand on it too much here", ≈58:47); the failure mode if you ignore it is a beam that wastes
slots on duplicates and systematically under-scores the hypotheses with many alignments.

## What sits on top

CTC gives you the most likely *phoneme* sequence. Turning that into words needs a pronunciation
dictionary at minimum, and does much better with a language model folded into the same search —
because `I can spoke` is a perfectly plausible phoneme string and an implausible sentence. See
[language models in decoding](language-models-in-decoding.md) for the shallow-fusion objective,
the word insertion bonus, and the two-model n-gram-then-Transformer arrangement the BCI uses.

## When to reach for CTC

Use it when all three hold: unknown alignment, large length mismatch, and monotonicity. Speech
recognition and handwriting recognition are the classic cases (slide 46) and the reason the BCI
could borrow the technique wholesale. Do **not** use it when the output can reorder relative to
the input — translation, summarization, most generation — because monotonicity is an assumption
CTC bakes in, not a preference it expresses.

## Related pages

- [Sequence-to-sequence and encoder-decoder models](seq2seq-and-encoder-decoder.md) — the more
  general and more expensive alternative.
- [Decoding algorithms](decoding-algorithms.md) — beam search in its usual form.
- [Language models in decoding](language-models-in-decoding.md) — what turns phonemes into
  sentences.
- [LSTM](lstm.md) — and the GRU, the network CTC is trained on top of here.
- [Lecture 14 — Brain-computer interfaces](14-brain-computer-interfaces.md)
