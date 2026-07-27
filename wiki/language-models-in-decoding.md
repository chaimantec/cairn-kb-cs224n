# Language models in decoding

When a system decodes text from a noisy signal — speech, handwriting, or neural activity — the
acoustic or neural model alone is not enough. It scores how well a hypothesis matches the signal,
not whether the hypothesis is a sentence anyone would say. A **language model** supplies the
second half of that judgement, and how you fold it into the search is a real engineering problem
with a latency budget attached.

Covered in [lecture 14](14-brain-computer-interfaces.md), slides 56–59 (≈59:34–1:03:22).

## The problem in one example

Fan's example is the whole argument. The neural decoder emits a phoneme sequence; run it through
a pronunciation dictionary and you might get **"I can spoke"**. That is a perfectly good phoneme
string. It is not a sentence (≈59:34).

Nothing in the signal model can rule it out, because the signal model was never asked whether the
output was grammatical. A language model can — see
[language modeling](language-modeling.md) for what $P(\mathbf{Y})$ means and how it is estimated.

## The fused objective

Instead of maximizing the signal model's score alone, the decoder maximizes a product of three
terms (slide 56):

$$\mathbf{Y}^* = \arg\max_{\mathbf{Y}} P(\mathbf{Y} \mid \mathbf{X})^{\alpha} \times P(\mathbf{Y}) \times L(\mathbf{Y})^{\gamma}$$

- $P(\mathbf{Y} \mid \mathbf{X})$ — the neural (or acoustic) model's probability of word sequence
  $\mathbf{Y}$ given signal $\mathbf{X}$, raised to a tunable weight $\alpha$.
- $P(\mathbf{Y})$ — the **language model's** probability of the sentence, factorized by the chain
  rule in the usual way, $P(\mathbf{Y}) = P(y_1)P(y_2 \mid y_1)P(y_3 \mid y_2, y_1)\cdots$
- $L(\mathbf{Y})^{\gamma}$ — the **word insertion bonus**, with $L(\mathbf{Y})$ the length of the
  hypothesis and $\gamma$ a tunable weight.

$\alpha$ and $\gamma$ are hyperparameters tuned on held-out data.

### Why the word insertion bonus is necessary

Every additional word multiplies in another probability less than one, so $P(\mathbf{Y})$ shrinks
monotonically with length. That is a property of the chain-rule factorization, not a fact about
language: "longer sentences will have smaller probabilities than shorter sentences — that's just
the property of how you decompose this probability" (≈1:00:19).

Left alone, a language model fused this way biases the decoder toward truncated output. The
insertion bonus adds a length-proportional reward that cancels the bias. It is the same
length-normalization problem that afflicts plain beam search — see
[decoding algorithms](decoding-algorithms.md) — solved with an explicit tunable term rather than
by dividing by length.

## Two language models, at two different speeds

The interesting design decision is that the BCI uses **both** an n-gram and a Transformer, at
different points, because they have different latency profiles.

### First pass — an n-gram model, inside the real-time loop

The decoder consumes one **20 ms** bin at a time and must finish all its work before the next one
arrives. Within that budget it beam-searches phoneme probabilities into candidate words and scores
them — perhaps 100 hypotheses — against an **[n-gram language model](n-gram-language-models.md)**
(a 5-gram in the deployed system, slide 59). The scored hypotheses are pruned to the top-$k$ and
fed back for the next bin (slide 57).

The reason is not that the n-gram is better. It is that scoring against it is a **memory
lookup** — "you can just load everything into memory and all the evaluation is just a memory
lookup, so it's really quick" — whereas a Transformer LM such as GPT-3 cannot answer 100 queries
in 20 ms (≈1:01:50). This is a case where a 1990s model is the correct choice in 2024, for a
reason that has nothing to do with quality.

### Second pass — a Transformer, rescoring

Once a full sentence has been decoded, the constraint relaxes. The system keeps the **n-best
hypotheses** — about 100 sentences — and rescores them with a
**[Transformer](transformer.md) language model**, which has around half a second to work
(≈1:03:22, slide 58). Slide 58's illustration: `P(I can speak) = 0.95` against
`P(I can spoke) = 0.01`, and the better sentence wins.

The division of labour is the point. The cheap model runs where latency is binding and only has
to keep the right answer *somewhere* in the beam; the expensive model runs once, offline relative
to the streaming loop, and only has to rank a short list.

## The general pattern

This structure — a signal model, a fast language model fused into the search, and a strong
language model rescoring an n-best list — is standard in speech recognition and predates the BCI
work by decades. What the lecture adds is a clean statement of *why each stage is where it is*,
which is a latency argument rather than an accuracy argument. When you meet a pipeline with a
weak model in the inner loop and a strong one on the outside, this is usually the reason.

## Related pages

- [Connectionist Temporal Classification](connectionist-temporal-classification.md) — where the
  phoneme probabilities being searched come from.
- [Decoding algorithms](decoding-algorithms.md) — beam search, and the length bias in its plain
  form.
- [n-gram language models](n-gram-language-models.md) — the first-pass model.
- [Transformer](transformer.md) — the rescoring model.
- [Language modeling](language-modeling.md) — what $P(\mathbf{Y})$ is.
- [Lecture 14 — Brain-computer interfaces](14-brain-computer-interfaces.md)
