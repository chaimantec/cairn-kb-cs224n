# GloVe

GloVe is the word vector model built at Stanford — **Pennington, Socher and Manning,
EMNLP 2014** — and it is the model whose vectors Manning actually demos in class. Its
design question is precise: **how do you get a count-based method to produce the
linear meaning components that word2vec exhibits?** Covered in
[lecture 2](02-word-vectors-and-language-models.md) (≈40:27–45:07), on **slides
21–23**.

Slides: [21–23, transcribed](../raw/slides/02-word-vectors-and-language-models.md) ·
[wordvecs2 PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture02-wordvecs2.pdf) ·
[Full transcript](../raw/transcripts/02-word-vectors-and-language-models.md)

## The motivation

Two things were true at once around 2013–2014. The count-based tradition — SVD
over co-occurrence matrices, latent semantic analysis, Doug Rohde's work — had a
lot of machinery and had already shown linear semantic structure, but never worked
very well. Meanwhile word2vec worked well and its vectors supported analogies.
See [distributional semantics](distributional-semantics.md) for that lineage.

So the goal was a model in which adding or subtracting a vector corresponds to a
meaning difference, built on a matrix of counts (≈41:13).

## The key idea: ratios of co-occurrence probabilities

Pennington's insight is that **ratios** of co-occurrence probabilities encode
meaning components, in a way that individual probabilities do not (≈42:00).

Manning's example uses *ice* and *steam*. Think about which words occur near each
(≈42:00):

- Near *ice*: *solid* is likely; *gas* is not; *water* is likely; a random word is
  not.
- Near *steam*: *gas* is likely; *solid* is not; *water* is likely; a random word
  is not.

Looking at any single one of these probabilities does not give you a meaning
component — you just get a number that is large or small. But take the **ratio** of
the two co-occurrence probabilities,

$$\frac{P(x \mid \text{ice})}{P(x \mid \text{steam})}$$

where $P(x \mid w)$ is the probability of seeing word $x$ near word $w$, and structure
appears (≈42:46):

- For *solid* the ratio is large.
- For *gas* the ratio is small.
- For *water* and for a random word, the ratio is approximately 1.

So that ratio picks out a direction in the space corresponding to the
solid/liquid/gas dimension of physics, while words irrelevant to the distinction
sit at ratio ≈ 1. Manning notes this was the hand-waving conception, and that when
you actually run the counts on real data it works out — you get factors of roughly
10 in both directions for the discriminating words, and numbers near 1 for the
others (≈43:32).

**Slide 22** gives the measured numbers, which are worth seeing because the effect is
larger than "roughly 10" suggests in one direction and the control column is *fashion*
rather than a random word:

| | solid | gas | water | fashion |
| - | ----- | --- | ----- | ------- |
| $P(x \mid \text{ice})$ | $1.9 \times 10^{-4}$ | $6.6 \times 10^{-5}$ | $3.0 \times 10^{-3}$ | $1.7 \times 10^{-5}$ |
| $P(x \mid \text{steam})$ | $2.2 \times 10^{-5}$ | $7.8 \times 10^{-4}$ | $2.2 \times 10^{-3}$ | $1.8 \times 10^{-5}$ |
| **ratio** | $\mathbf{8.9}$ | $\mathbf{8.5 \times 10^{-2}}$ | $1.36$ | $0.96$ |

The two discriminating words land two orders of magnitude apart from each other, while
*water* (related to both) and *fashion* (related to neither) both sit near 1 — so a
ratio near 1 does not mean "unrelated", it means "does not discriminate".

## From ratios to a model

The engineering step follows almost immediately (≈44:19). If you want a *ratio* of
probabilities to correspond to a *difference* of vectors, take logs: a log turns
the ratio into a subtraction. So build a **log-bilinear** model in which the dot
product of two word vectors models the log co-occurrence probability:

$$w_i \cdot w_j = \log P(i \mid j)$$

and with vector differences,

$$w_x \cdot (w_a - w_b) = \log \frac{P(x \mid a)}{P(x \mid b)}$$

The second line (both are on **slide 23**) is the whole point: the difference between
two word vectors *is* the log of the ratio of their co-occurrence probabilities —
exactly the linear meaning component you wanted.

That is the essence of GloVe: fit the dot products to log co-occurrence counts.
Manning notes the real model adds bias terms and frequency thresholds, and
explicitly skips them as unimportant to the intuition (≈44:19). For completeness,
slide 23 gives the actual loss:

$$J = \sum_{i,j=1}^{V} f(X_{ij}) \left( w_i^{\top} \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

where $X_{ij}$ is the number of times word $j$ occurs in the context of word $i$, $V$ is
the vocabulary size, and $w_i$ and $\tilde{w}_j$ are the two vectors GloVe keeps per word
(the same center/outside split as [word2vec](word2vec.md)). It is a weighted
least-squares fit of the dot product, plus per-word bias terms $b_i$ and $\tilde{b}_j$,
to $\log X_{ij}$. The weighting function $f$ rises
from zero, grows roughly linearly, then **saturates flat at 1.0** past a cutoff, which
is what stops extremely frequent pairs from dominating the fit — the "frequency
threshold" Manning skips over. Slide 23 lists the payoff as fast training and
scalability to huge corpora.

His summary is that the basic intuition — what it takes to get linear meaning
components — is the part worth knowing.

## In practice

The demo in lecture 2 (≈11:45 onward) uses 100-dimensional GloVe vectors loaded via
gensim, and Manning flags up front that strictly speaking these are not word2vec
vectors, though they behave in exactly the same way (≈12:32). Everything shown
there — *croissant* returning brioche, baguette and focaccia;
$\text{king} - \text{man} + \text{woman} \approx \text{queen}$;
Australia:beer::Russia:vodka — is GloVe output. The model was built in 2014,
which is why it cannot answer questions about the last decade of politics
(≈19:34).

On evaluation (≈50:29), GloVe scores at the top of the table Manning shows,
above plain SVD, SVD over log counts, CBOW and skip-gram. See
[evaluating word vectors](evaluating-word-vectors.md).

## Related pages

- [distributional semantics](distributional-semantics.md) — the count-based
  tradition GloVe comes out of, including SVD and Rohde's COALS
- [word2vec](word2vec.md) — the prediction-based model GloVe was trying to match
- [evaluating word vectors](evaluating-word-vectors.md) — the comparison table and
  what it means
- [lecture 2](02-word-vectors-and-language-models.md) — the full narrative,
  including the notebook demo
