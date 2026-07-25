# Gradient descent

Gradient descent is the mechanism that turns "we have an objective function" into
"we have trained parameters", and it is how every model in this course is
actually fit. Introduced at the end of [lecture
1](01-intro-and-word-vectors.md) (≈1:00:37–1:19:30) and completed at the start of
[lecture 2](02-word-vectors-and-language-models.md) (≈3:09–8:41).

Slides: [wordvecs1](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture01-wordvecs1.pdf),
[wordvecs2](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture02-wordvecs2.pdf) ·
Notes: [gradient notes](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/readings/gradient-notes.pdf),
[review of differential calculus](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/readings/review-differential-calculus.pdf)

## The idea

You have a cost function you want to minimize. You compute its gradient, which
tells you which direction is downhill. You take a small step that way. You repeat
(lecture 2, ≈3:09).

Manning's picture in lecture 1 (≈1:01:23) is a two-dimensional surface: start at
the top left, compute the derivatives, find they point down and a bit to the right,
walk that way, recompute, find they now point down and further to the right, walk
again, and eventually reach the minimum. In one dimension this is so simple it
barely needs calculus. The reason calculus is genuinely required is that in many
dimensions the gradient at different points can head in quite different
directions (lecture 2, ≈3:58).

## The update rule

> `θ_new = θ_old − α ∇J(θ)`

α is the **step size** or **learning rate**, and in practice it is a very small
number — Manning quotes 10⁻³, 10⁻⁴, or even 10⁻⁵ (lecture 2, ≈4:46).

The learning rate has to be small for a concrete reason (≈5:33): the gradient tells
you the downhill direction *at the point where you computed it*, and that direction
stops being right as soon as you move. Walk too far and you sail past the minimum;
with a really large step you can land somewhere worse than where you started. So
you take little steps and recompute often.

## Stochastic gradient descent

Plain gradient descent is never used (lecture 2, ≈5:33). The problem is cost:
evaluating the objective and its gradient over the *entire* training set before
taking one step is enormously expensive, so you would wait a very long time to make
a single update.

**Stochastic gradient descent** instead samples a small subset of the data,
pretends that subset is the whole dataset, and uses its gradient as the direction
to walk (≈6:21). This is also called **mini-batch** gradient descent. The gradient
you get is a noisy, inaccurate estimate of the true gradient.

*(The transcript garbles the mini-batch size Manning quotes at ≈6:21 — it reads
"16 or 2". Take the actual figure from the
[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture02-wordvecs2.pdf).)*

The interesting part is that SGD is not merely a cheaper approximation you tolerate
(≈7:07). Manning's point is that neural networks often work *better* with noise in
the system: the noise gives the optimization jiggle and moves things around, so
stochastic gradient descent is both dramatically faster *and* a better optimizer
for neural networks than the exact version. This is one of the recurring surprises
of the field — the sloppy method wins on quality too.

## Initialization matters

You must initialize parameters with small **random** numbers (lecture 2, ≈7:53). If
you leave all the word vectors at zero, nothing works at all: identical starting
values create symmetries that the gradient cannot break, since every unit receives
the same updates forever. So random initialization is not a nicety, it is what makes
learning possible.

## Computing the gradients

For word2vec, the parameters are all the word vectors concatenated — with two
vectors per word, one for the center role and one for the outside role (lecture 1,
≈1:02:11). For a 400,000-word vocabulary at 100 dimensions that is 80 million
parameters to differentiate with respect to (≈1:02:56). Manning's comment: that is a
lot of parameters to fiddle, but we have big computers.

Lecture 1 works the gradient out by hand (≈1:05:16–1:18:42). The pieces of calculus
it relies on are worth listing, because the same moves recur all course:

- The derivative of a **sum** is the sum of the derivatives, so a gradient over a
  big summed objective decomposes into per-term gradients (≈1:08:26).
- **log of a quotient** splits into log numerator minus log denominator (≈1:09:13).
- log and exp **cancel** (≈1:10:02).
- The derivative of `uᵀv` with respect to `v` is `u`. Manning justifies it
  componentwise: the dot product expands to `u₁v₁ + u₂v₂ + u₃v₃ + …`, so
  differentiating with respect to `v₁` leaves `u₁` and kills every other term; do
  that along the whole vector and you get `u` back (≈1:10:49–1:11:34).
- The **chain rule**, applied twice — once for the outer log, once for the exp
  inside the sum (≈1:12:20–1:15:33).

The result for the center vector is `u_o − Σ_x P(x | c) · u_x`, which Manning reads
as **observed minus expected** (≈1:17:56). It compares the outside vector that
actually occurred against the weighted average of what the model predicted. When
expectation matches observation the derivative is zero and you have reached a
maximum. He notes this observed-minus-expected form "you see quite a bit in these
kinds of derivations" (≈1:18:42) — it is worth recognizing on sight.

Manning's closing remark is the practical one (≈1:19:30): he wants you to
understand how this works, but you will quickly find that computers do it for you
and you will not compute these by hand on a regular basis. The transcript is
unreliable for the notation in this section — several symbols come through as
"[Music]" or wrong subscripts — so use the slides.

## Related pages

- [word2vec](word2vec.md) — the objective these gradients are taken of
- [softmax and cross-entropy](softmax-and-cross-entropy.md) — the functions inside
  the objective
- [lecture 1](01-intro-and-word-vectors.md) — the full hand derivation
- [lecture 2](02-word-vectors-and-language-models.md) — SGD, learning rates,
  initialization
