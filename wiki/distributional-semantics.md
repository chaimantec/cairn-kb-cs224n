# Distributional semantics

Distributional semantics is the foundational commitment of this course: a word's
meaning is represented by the contexts it appears in, rather than by a link to the
thing it refers to. Everything from word2vec to modern language models rests on
it. The idea is introduced in [lecture 1](01-intro-and-word-vectors.md) (≈35:01)
and its count-based branch is developed in
[lecture 2](02-word-vectors-and-language-models.md) (≈33:28–40:27).

## The two theories of meaning

The standard linguistic account pairs a symbol with an idea or a thing —
*signifier* and *signified* — which is called **denotational semantics** (lecture
1, ≈28:50). The meaning of *tree* is all the trees in the world. The same notion
is used for programming languages, where symbols like `while` and `if` have
denotations. It is a perfectly reasonable theory of meaning, and it was never
obvious how to make a computer do anything with it.

**Distributional semantics** takes the opposite tack: represent a word's meaning
by the distribution of contexts it occurs in. Manning quotes J.R. Firth's line
from 1957, "you shall know a word by the company it keeps," and notes the idea
also traces back to philosophical work by Wittgenstein (lecture 1, ≈35:01). If you
want to know what *banking* means, collect sentences that use it — "government
debt problems turning into banking crises" — and let the words that surround it
*be* its meaning.

This is a strange thing to believe about meaning if you think about it too hard,
and it is also the reason the field works. Words are the observable trace of how
people use language, and the co-occurrence statistics are dense enough to
reconstruct a great deal of semantics from nothing but text.

## Why the pre-neural representations failed

**WordNet** was the standard resource for computational meaning before this — a
fancy thesaurus of synonyms and *is-a* hierarchies (a panda is a kind of carnivore,
which is a placental mammal). Manning's critique (lecture 1, ≈30:23):

- It misses nuance. WordNet lists *proficient* as a synonym for *good*, but of all
  the things you would call good, "that was a proficient shot" sounds wrong.
- It is incomplete, and cannot keep up with slang or new usage.
- It is entirely human-made, so it is expensive and never finished.

**One-hot vectors** were the other option: index the vocabulary, and represent
each word as a vector that is 1 in its own position and 0 everywhere else. These
are also called **localist representations**, because each word is represented at
exactly one point in the vector (lecture 1, ≈34:16). Manning's objection is
decisive (≈32:42): they encode no similarity whatsoever. *Motel* and *hotel* are
just two different positions. Their dot product is zero, so the vectors are
formally **orthogonal** — as unrelated as any two words can be. And with a
realistic vocabulary the vectors are enormous.

You can bolt similarity on from the outside by consulting a separate resource —
web search called this **query expansion** — but similarity is never intrinsic to
the representation (≈33:30). The idea that follows is to learn representations
that have similarity built in.

## Dense vectors

The replacement is a short **dense** vector per word, with real-valued components.
Manning's slide uses eight dimensions so it fits on the page; in practice these
are a few hundred to a couple of thousand dimensions, not the half-million of a
full vocabulary (lecture 1, ≈35:47). Similarity is the **dot product**, which
grows when corresponding components share a sign and have large magnitude
(≈36:32).

These are called **word vectors**, **embeddings**, or neural word representations
— "embedding" because each word is placed at a position in a high-dimensional
space, whose dimensionality is the length of the vector (≈37:19).

Manning stresses that high-dimensional spaces are not just big versions of
two-dimensional ones (≈38:05). In 2-D, you are only near something if both
coordinates match. In high dimensions a point can be simultaneously close to many
different things along different dimensions — which is how a single vector for
*star* can sit near both *nebula* and *celebrity* (≈42:43). This is worth
internalizing, because the intuition that a word must "pick" one region of the
space is simply wrong.

Since nobody can look at 300 dimensions, visualizations project down with
**t-SNE**, a nonlinear dimensionality reduction that tends to work better for
these representations than PCA (≈43:29). Zooming into such a plot shows real
structure: countries near nationality terms, and a verb region with fine internal
organization — verbs of communication together, *come* and *go* together, forms
of *have*, forms of *be*, with *become* and *remain* near *be* because they take
the same kind of state complements (≈39:37–40:22).

On how many dimensions to use (≈44:16): things start working around 100, 300 was
the standard for years because it worked well, and as models and data have grown
it has become common to use 1,000 or even 2,000.

## The count-based branch

There is an obvious alternative to predicting contexts: just *count* them. Build a
matrix of how often each word occurs in the context of each other word (lecture 2,
≈33:28). With the toy corpus "I like deep learning / I like NLP / I enjoy flying"
and a one-word window, every cell is 0 or 1 except where *I like* occurs twice.
Each row of that matrix is already a word vector.

People did exactly this, and the vectors are ungainly (lecture 2, ≈35:01). With
400,000 words in the vocabulary the matrix is 400,000 × 400,000, which is vastly
worse than the 400,000 × 100 of learned vectors. So the standard move is to reduce
the dimensionality, which points at PCA and — since it works for a matrix of any
shape — the **singular value decomposition** (≈36:35).

SVD factors the matrix into three: two orthonormal matrices *U* and *V*, and a
diagonal matrix of **singular values** ordered by size, which act as weights on
the different dimensions. Keeping only the largest singular values and zeroing the
rest gives a low-dimensional representation of each word (≈38:08).

A student notes the example matrix should be square if it is
vocabulary-by-vocabulary; Manning agrees, and mentions the non-square variant is
words-versus-documents, which he skipped over (lecture 2, ≈45:07).

### Latent semantic analysis and COALS

This was all explored well before neural word vectors, under the name **latent
semantic analysis**, and Manning's verdict is that it "sort of half worked but
never worked very well" (lecture 2, ≈38:08). Running SVD on raw counts does not
produce good vectors.

The thread was kept alive, particularly in psychology, and in the early 2000s a
graduate student named Doug Rohde improved on it substantially with an algorithm
the transcript renders as COALS (lecture 2, ≈38:53). His improvements, several of
which reappear in word2vec:

- **Log the frequencies** rather than using raw counts.
- **Ramp the window**, so that nearer context words count for more than distant
  ones.
- Use **Pearson correlations** instead of counts.

Manning's aside is a nice piece of history (≈40:27): Rohde effectively discovered
the linear semantic components that later made word2vec famous. A figure in his
dissertation shows "doer of an event" as a linear direction in the space that you
can follow from a verb to the noun for the person who does it. He did not become
famous for it, because at the time nobody was paying attention.

That observation — that count-based methods can also yield linear meaning
structure — is what led directly to [GloVe](glove.md).

## Which branch won

Neither, exactly. The count-based and prediction-based approaches turned out to be
closely related, and GloVe was built specifically to get the linear structure of
word2vec out of a co-occurrence matrix. The evaluation table in lecture 2 (≈50:29)
shows plain SVD performing terribly, SVD over log counts already reasonable, then
CBOW and skip-gram, then GloVe — see
[evaluating word vectors](evaluating-word-vectors.md).

## Related pages

- [word2vec](word2vec.md) — the prediction-based implementation
- [GloVe](glove.md) — count-based, but engineered for linear meaning components
- [word senses and polysemy](word-senses-and-polysemy.md) — what one vector per
  word costs you
- [evaluating word vectors](evaluating-word-vectors.md) — how these approaches
  compare
