# GloVe

GloVe is the word vector model built at Stanford by Manning and the postdoc
Jeffrey Pennington, and it is the model whose vectors Manning actually demos in
class. Its design question is precise: **how do you get a count-based method to
produce the linear meaning components that word2vec exhibits?** Covered in
[lecture 2](02-word-vectors-and-language-models.md) (≈40:27–45:07).

Slides: [wordvecs2](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture02-wordvecs2.pdf) ·
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
the two co-occurrence probabilities, P(x | ice) / P(x | steam), and structure
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

## From ratios to a model

The engineering step follows almost immediately (≈44:19). If you want a *ratio* of
probabilities to correspond to a *difference* of vectors, take logs: a log turns
the ratio into a subtraction. So build a **log-bilinear** model in which the dot
product of two word vectors models the log co-occurrence probability:

> `w_iᵀ w̃_j ≈ log P(i | j)`

Then the difference between two word vectors corresponds to the log of the ratio of
their co-occurrence probabilities — which is exactly the linear meaning component
you wanted.

That is the essence of GloVe: fit the dot products to log co-occurrence counts.
Manning notes the real model adds bias terms and frequency thresholds, and
explicitly skips them as unimportant to the intuition (≈44:19). His summary is that
the basic intuition — what it takes to get linear meaning components — is the part
worth knowing.

## In practice

The demo in lecture 2 (≈11:45 onward) uses 100-dimensional GloVe vectors loaded via
gensim, and Manning flags up front that strictly speaking these are not word2vec
vectors, though they behave in exactly the same way (≈12:32). Everything shown
there — *croissant* returning brioche, baguette and focaccia; `king − man + woman ≈
queen`; Australia:beer::Russia:vodka — is GloVe output. The model was built in 2014,
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
