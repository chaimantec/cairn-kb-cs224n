# Softmax, the logistic function, and cross-entropy loss

These are the small pieces of machinery that turn raw scores into probabilities and
probabilities into a training signal. They appear everywhere in the course, so it is
worth being solid on them. Softmax is derived in
[lecture 1](01-intro-and-word-vectors.md) (≈57:30–59:50, **slide 30**); the logistic
function appears in [lecture 2](02-word-vectors-and-language-models.md) with negative
sampling (≈29:37, **slide 12**) and again in the neural classifier (≈1:10:32, **slides
38 and 41**); cross-entropy loss is covered at the end of lecture 2
(≈1:11:19–1:13:43, **slide 39**).

Slide text: [lecture 1 slides](../raw/slides/01-intro-and-word-vectors.md) ·
[lecture 2 slides](../raw/slides/02-word-vectors-and-language-models.md)

## Softmax

The problem softmax solves: a dot product between two word vectors is a similarity
score, but it is an **unbounded real number** — it can be positive or negative
(lecture 1, ≈57:30). What you want is a probability distribution. Softmax gets you
there in two steps:

1. **Exponentiate.** `e^x` is positive for any real `x`, so this makes every score
   positive (≈57:30).
2. **Normalize.** Compute the numerator for every possible outcome, sum them, and
   divide through. Manning describes this as turning the numbers into a
   distribution "in the dumbest way possible" (≈57:30).

> `softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)`

The result sums to one by construction (≈58:18). In word2vec this gives a
distribution over which context word appears, from nothing but dot products — see
[word2vec](word2vec.md).

Manning's gloss on the name (≈59:05): it is called soft**max** because
exponentiating amplifies the largest values, so the biggest score dominates — but
it is **soft** because smaller items still receive some probability rather than
being zeroed out. He points out the name is a bit odd, since a real max picks out
exactly one thing while softmax turns a bunch of real numbers into a distribution.

Its ubiquity is the thing to take away: any time you want to turn a vector in ℝⁿ
into probabilities, you push it through a softmax (≈59:50).

## The logistic (sigmoid) function

> `σ(x) = 1 / (1 + e^(−x))`

The logistic function maps any real number to a probability between 0 and 1
(lecture 2, ≈29:37). It is what you use when there is a single score to convert
rather than a set of competing ones — a binary decision rather than a choice among
a vocabulary.

Manning makes a terminological point worth keeping (≈29:37): **"sigmoid" only means
s-shaped**, and there are infinitely many s-shaped functions. The one actually used
is the logistic function, so the two words get used interchangeably even though
"sigmoid" is the looser term.

A useful property, exploited by negative sampling: the function is symmetric about
its midpoint, so negating the argument reflects you to the other side of the curve.
That is why the same expression can be used to push true context words toward high
probability and sampled words toward low probability (lecture 2, ≈30:25).

The logistic function also serves as the nonlinearity in the neural classifier at
the end of lecture 2 (≈1:10:32), and a single logistic unit is what makes binary
logistic regression a rough model of a neuron (≈1:16:05).

## Softmax versus logistic

They are the same idea at different arities. Softmax produces a distribution over
many mutually exclusive outcomes; the logistic function produces a probability for
one binary outcome. In word2vec, naive softmax normalizes over the whole vocabulary,
while negative sampling replaces that with a handful of independent logistic
decisions — which is precisely what makes it cheaper (lecture 2, ≈28:51).

## Cross-entropy loss

Everything in lectures 1 and 2 is framed as log likelihood and negative log
likelihood, but when you start writing PyTorch in assignment 2 you will use
**cross-entropy loss**. Manning flags this explicitly as a note for learning ahead
(lecture 2, ≈1:11:19), because the mismatch in vocabulary is confusing if nobody
tells you they are the same thing.

Cross entropy comes from information theory. Given a true distribution *p* and your
model's distribution *q*, the cross entropy is the expectation under *p* of the
negative log of *q*:

> `H(p, q) = − Σ_c p(c) log q(c)`

The special case that matters is when your labels are **one-hot** — ground truth,
gold, or target data where the correct class has probability 1 and everything else
0 (≈1:12:56). Manning's example: labelling "I visit Paris every spring" with
probability 1 for *location* and 0 for *not a location*. Then every term in the
summation multiplies by zero except one, and all that is left is the log
probability your model assigns to the correct class.

Which is exactly the negative log likelihood you have been minimizing all along
(≈1:13:43). Cross-entropy loss and negative log likelihood are the same objective
in this setting; the only thing that changes is the name and the library function.
Slide 39 says it directly — "Use this in PyTorch! `torch.nn.CrossEntropyLoss()`" — and
adds the caveat that cross entropy *can* be used with a more interesting *p*, but for
now you just want it as the loss in PyTorch.

## Seen again as an output layer

Lecture 4's neural dependency parser is the first place in the course where this is used as
the output layer of a real system rather than as a derivation. Its architecture ends

    y = softmax(Uh + b₂)

over the three possible transitions { Shift, Left-Arc_r, Right-Arc_r }, and the log loss —
cross-entropy error — is **backpropagated all the way into the embeddings**, so the word,
part-of-speech and dependency-label vectors are themselves learned by it (lecture 4,
slide 44). The hidden layer beneath it exists precisely so that a *linear* softmax is
enough to separate the classes; see [transition-based parsing](transition-based-parsing.md).

Note also that Manning deliberately avoids differentiating the loss in lecture 3, computing
gradients of the raw score *s* instead, because the derivative of the logistic is an
Assignment 2 question (lecture 3, slide 27).

## Related pages

- [word2vec](word2vec.md) — where the softmax and negative sampling live
- [gradient descent](gradient-descent.md) — differentiating through these
  functions, including the chain rule through log and exp
- [activation functions](activation-functions.md) — the hidden-layer non-linearities, as
  opposed to these output-layer functions
- [backpropagation](backpropagation.md) — how this loss reaches the parameters
- [lecture 1](01-intro-and-word-vectors.md) — softmax derivation
- [lecture 2](02-word-vectors-and-language-models.md) — logistic function, neural
  classifier, cross-entropy
- [lecture 4](04-dependency-parsing.md) — softmax as the parser's output layer
