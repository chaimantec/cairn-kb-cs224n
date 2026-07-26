# word2vec

word2vec is the first algorithm the course teaches, and the simplest thing in it
that genuinely works: from nothing but a large body of raw text, it learns a dense
vector for every word such that words used in similar contexts end up near each
other. It was introduced by Tomas Mikolov and colleagues at Google in 2013
([lecture 1](01-intro-and-word-vectors.md), ≈46:35). It was neither the first
method for learning word vectors — others go back to around the turn of the
millennium — nor the last, but it was simple and fast, which is why it caught
everyone's attention.

Covered across [lecture 1](01-intro-and-word-vectors.md) (setup, objective
function, gradient derivation) and
[lecture 2](02-word-vectors-and-language-models.md) (variants, negative sampling,
sampling distribution).

Slide text: [lecture 1, slides 25–36](../raw/slides/01-intro-and-word-vectors.md) ·
[lecture 2, slides 7–14](../raw/slides/02-word-vectors-and-language-models.md).
PDFs: [wordvecs1](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture01-wordvecs1.pdf),
[wordvecs2](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture02-wordvecs2.pdf)

## The setup

Start with a large corpus — just a long list of words. Give every word *type* a
vector; a type is the word *problems* wherever it occurs, as opposed to a *token*,
which is one particular occurrence (lecture 1, ≈48:55). Then slide a window
through every position in the text. At each position one word is the **center
word** and the words around it within the window are the **outside words** —
Manning's example uses two words on each side (≈50:29).

At each position, use the similarity between the center vector and each outside
vector to compute the probability that those words co-occur, then nudge the
vectors to make the co-occurrences that actually happened more probable. Move to
the next position and repeat.

Manning is careful that the model is weak on purpose (lecture 1, ≈51:15): given
*banking*, you cannot really predict that *into* came before it. The goal is only
to do as well as possible — *crisis* should be likely near *banking*, *skillet*
should not.

## The objective function

Written as a likelihood, you get a product over every position in the text and
every word in the window at that position. Writing $T$ for the number of positions
in the corpus, $m$ for the half-width of the window, $w_t$ for the center word at
position $t$, and $\theta$ for every parameter of the model stacked into one long
vector, that is (slide 28)

$$L(\theta) = \prod_{t=1}^{T} \prod_{\substack{-m \le j \le m \\ j \ne 0}} P(w_{t+j} \mid w_t ; \theta)$$

Three conventional adjustments turn it into the objective actually minimized
(lecture 1, ≈52:03–54:23):

1. **A minus sign.** For entirely arbitrary historical reasons everyone minimizes
   rather than maximizes, which is why the algorithms are called gradient
   *descent*.
2. **A logarithm.** The enormous product is awkward for the math; taking logs
   turns products into sums.
3. **Division by the number of words.** Otherwise the value grows with corpus
   size.

The result is the **average negative log likelihood** (slide 28):

$$J(\theta) = -\frac{1}{T} \log L(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \sum_{\substack{-m \le j \le m \\ j \ne 0}} \log P(w_{t+j} \mid w_t ; \theta)$$

Minimizing it maximizes the probability of the words that actually appear in
context — the slide's own gloss is "minimizing objective function ⟺ maximizing
predictive accuracy".

The probability itself is defined entirely in terms of the word vectors — there
are no other parameters in the model (lecture 1, ≈55:11; slide 29). For a center
word $c$ and an outside word $o$,

$$P(o \mid c) = \frac{\exp(u_o^{\top} v_c)}{\sum_{w \in V} \exp(u_w^{\top} v_c)}$$

where $v_c$ is the center vector of $c$, $u_o$ the outside vector of $o$, and $V$
the vocabulary. This is a **softmax** over the dot products; see
[softmax and cross-entropy](softmax-and-cross-entropy.md). A high dot product
means high co-occurrence probability. Manning flags that this notion of
"similarity" is a strange one (lecture 1, ≈56:44): it has to make *hotel* and
*motel* similar, but it also has to let *the* appear before *student*, so *the*
ends up similar to essentially every noun.

## Two vectors per word

Each word $w$ gets **two** vectors: $v_w$ for when it is the center word, $u_w$ for
when it is an outside word. The reason is purely mathematical convenience (lecture 1,
≈1:02:11; explained fully in lecture 2, ≈25:49). If a word used one vector for
both roles, then while summing over every candidate outside word for the softmax
denominator you would eventually reach the center word itself, producing a
quadratic term — the exponential of a vector dotted with itself — where every
other term in the sum is linear. That is not intractable, just messy, so the
authors kept the two sets disjoint.

Manning notes it actually works slightly better to tie them, but in practice
people estimate the two separately and **average** them at the end (lecture 2,
≈21:57). The two end up very close anyway, because sliding through the text visits
every pair in both configurations: *octopus* as center with *legs* outside, then a
few steps later *legs* as center with *octopus* outside (lecture 2, ≈22:42).

The parameter count is just the vectors. With $d$-dimensional vectors and a
vocabulary of $V$ words, $\theta$ is the stacked column vector of all of them, so
$\theta \in \mathbb{R}^{2dV}$ (slide 31). With a 400,000-word vocabulary and
100-dimensional vectors that is $400{,}000 \times 2 \times 100 =$ **80 million
parameters** (lecture 1, ≈1:02:56).

## It is a bag of words model

word2vec knows nothing about sentence structure, and does not even distinguish
left from right — it predicts the same probability for a word wherever in the
window it sits (lecture 2, ≈10:58). All it knows is which words tend to occur near
which other words. It also learns exactly one vector per word type, so it cannot
represent a word's meaning in context; that limitation is discussed in
[word senses and polysemy](word-senses-and-polysemy.md) and is only lifted later
in the course with contextual representations.

## Training it

The vectors are learned by optimization (lecture 1, ≈1:00:37). Initialize every
vector with small **random** numbers — this matters, because initializing to zero
creates symmetries that prevent learning entirely (lecture 2, ≈7:53). Then compute
the gradient of the objective with respect to every parameter and walk downhill,
in practice with stochastic gradient descent over mini-batches. See
[gradient descent](gradient-descent.md).

The gradient with respect to the center vector, derived by hand in lecture 1
(≈1:05:16–1:18:42), comes out as

$$\frac{\partial}{\partial v_c} \log P(o \mid c) = u_o - \sum_{x \in V} P(x \mid c)\, u_x$$

which Manning reads as **observed minus expected**: the outside vector that
actually occurred, minus the average outside vector the model currently predicts,
weighted by how likely it thinks each word is. When the model's expectation
matches observation the derivative is zero, which is the optimum. The same form
recurs throughout these derivations. Note that the transcript is unreliable for
the notation here — the slides are the authority.

## The family of variants

The 2013 paper describes several methods (lecture 2, ≈26:34). Two model shapes:

- **Skip-gram** — predict the outside words from the center word. This is the
  version presented in the lectures: simpler, and works well.
- **Continuous bag of words (CBOW)** — predict the center word from all the
  context words.

And separately, choices of training loss:

- **Naive softmax** — the full normalization over the vocabulary, as above.
  Manning notes it is entirely doable on modern hardware, but was expensive when
  the paper was written: 400,000 words, each needing a dot product of 100- or
  300-dimensional vectors and an exponentiation (lecture 2, ≈27:20).
- **Hierarchical softmax** — mentioned but not explained in the lecture.
- **Negative sampling** — explained below.

A practical note for anyone looking at old course materials: previous offerings of
CS224N had students implement word2vec from scratch as assignment 2, but the
Spring 2024 offering dropped it because the quarter is shorter, so the old
assignment 2 is not the one to work from (lecture 2, ≈25:03).

## Skip-gram with negative sampling

Instead of normalizing over the whole vocabulary, train a handful of simple **binary**
logistic regressions, each distinguishing a true pair (the center word with a word from
its context window) from "noise" pairs (the center word with a random word) — lecture 2,
≈28:51, **slides 11–12**. The softmax is replaced by the **logistic function**, which
maps any real number to a probability in (0, 1).

**Slide 12** states the objective in the course's notation, for $K$ negative samples:

$$J_{\text{neg-sample}}(u_o, v_c, U) = -\log \sigma(u_o^{\top} v_c) - \sum_{k \in \{K \text{ sampled indices}\}} \log \sigma(-u_k^{\top} v_c)$$

where $u_k$ is the outside vector of the $k$-th sampled word and $\sigma$ is the
logistic function, also given on slide 12 as

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

with the aside "(we'll become good friends soon)". The slide annotates it "**sigmoid rather
than softmax**". The first term drives the real outside word's probability up; the
sum drives the sampled words' probabilities down.

Manning points out that "sigmoid" merely means s-shaped, and there are infinitely
many s-shaped functions — the one actually used is the logistic function (lecture
2, ≈29:37). The negative terms work by exploiting its symmetry: negating the
argument flips you to the other side of the curve, so the same expression drives
the sampled words toward low probability (≈30:25).

**How to sample the negatives** matters, and is a nice detail (lecture 2, ≈31:10).
Sampling uniformly from 400,000 words is wrong, because frequency varies
enormously; what you want is roughly the **unigram distribution** — how common
each word is on its own — under which you would draw *the* about 10% of the time.
But the standard word2vec recipe raises the unigram probability to the power
**3/4**. Manning asks the class why and confirms the answer (≈32:41): it somewhat
increases the probability of less frequent words, moving partway from true
relative frequency toward uniform. You want to move some distance toward uniform
sampling but not all the way — all the way would correspond to an exponent of
zero.

Typically only five or ten negative samples are used per positive example
(≈31:10). Slide 12 writes the sampling distribution as

$$P(w) = \frac{U(w)^{3/4}}{Z}$$

where $U(w)$ is the unigram distribution and $Z$ the normalizing constant that makes
the result sum to one.

### Why this makes the gradients sparse

An aside on **slides 13–14** that is easy to skip but explains why the method is fast in
practice. In any one window there are at most $2m + 1$ words plus $2km$ negative samples,
where $m$ is the window half-width and $k$ the number of negatives per word, so the
gradient of that window's loss, $\nabla_{\theta} J_t(\theta) \in \mathbb{R}^{2dV}$, is
**almost entirely zero** — non-zero blocks only for the handful of words actually
present.

The engineering consequence (slide 14): you should only update the word vectors that
actually appear, which needs either sparse matrix update operations touching just certain
**rows** of the embedding matrices $U$ and $V$, or a hash of word vectors. Slide 14 flags
"rows not columns in actual DL packages!" and notes that with millions of word vectors
and distributed training, avoiding gigantic update messages matters.

## What the learned vectors know

See [lecture 2](02-word-vectors-and-language-models.md) for the notebook demo, and
[evaluating word vectors](evaluating-word-vectors.md) for how this is measured. In
brief: nearest neighbours are sensible (*croissant* → brioche, baguette,
focaccia), and the vectors support **analogies** — $\text{king} - \text{man} + \text{woman} \approx \text{queen}$ —
because meaning differences show up as directions in the space, which was the most
celebrated property discovered about them (lecture 2, ≈15:38). Manning is candid
that essentially nobody uses these vectors directly anymore (≈17:58), but the
recipe — define a simple prediction objective over lots of text, optimize it, and
get useful structure out — is the one the rest of the course builds on.

## Related pages

- [distributional semantics](distributional-semantics.md) — the idea word2vec
  implements, and the count-based alternative
- [GloVe](glove.md) — the Stanford model that combines counting with the linear
  structure word2vec exhibits
- [gradient descent](gradient-descent.md) — how the optimization actually runs
- [softmax and cross-entropy](softmax-and-cross-entropy.md) — the probability
  function and the loss
