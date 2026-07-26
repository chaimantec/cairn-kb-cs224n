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

Writing $\theta$ for the vector of all parameters and $J(\theta)$ for the cost function,
one step of gradient descent is (lecture 1, slide 33)

$$\theta^{\text{new}} = \theta^{\text{old}} - \alpha \nabla_{\theta} J(\theta)$$

or, one parameter at a time (slide 33),

$$\theta_j^{\text{new}} = \theta_j^{\text{old}} - \alpha \frac{\partial}{\partial \theta_j^{\text{old}}} J(\theta)$$

$\alpha$ is the **step size** or **learning rate**, and in practice it is a very small
number — Manning quotes $10^{-3}$, $10^{-4}$, or even $10^{-5}$ (lecture 2, ≈4:46).

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

*(The captions garble the mini-batch size Manning quotes at ≈6:21 — they read "16 or
2" — and the slide does not give a number either
([lecture 1 slide 39](../raw/slides/01-intro-and-word-vectors.md), repeated as
lecture 2 slide 6), so this KB does not state one.)*

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

Lecture 5 adds the scale rule that goes with this (slide 11): weights should be drawn
$\sim \mathrm{Uniform}(-r, r)$ with $r$ chosen so that numbers get neither too big nor too
small under repeated matrix multiplication, hidden-layer biases 0, and output biases at the
optimal value if the weights were 0. **Xavier initialization** picks the variance from the
layer sizes:

$$\mathrm{Var}(W_i) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$

where $n_{\text{in}}$ is the fan-in (previous layer size) and $n_{\text{out}}$ the fan-out
(next layer size).
Manning notes this used to be treated as quite important and that the need for it largely
goes away later, once **layer normalization** is used — though you must still initialize to
*something* (lecture 5, ≈20:57).

## Optimizers

Lecture 5's slide 12. Plain SGD "will work just fine" for almost any problem if you fiddle
enough — but fiddling means hand-tuning the learning rate, often with a schedule that starts
higher and halves every $k$ epochs, plus various other complications (≈22:29).

**Adaptive optimizers** avoid that. For each parameter they accumulate a measure of what the
gradient has been in the past, and use it to scale that parameter's step — so you get
**differential per-parameter learning rates**, and much less sensitivity to hyperparameters
(≈23:14). The family, in the order the slide gives it:

- **AdaGrad** — the simplest member, co-invented by John Duchi at Stanford. Nice, but "tends
  to stall early".
- **RMSprop**
- **Adam** — *"a fairly good, safe place to begin in many cases"*, and the one used in
  Assignment 2. If you remember nothing else, remember Adam (≈24:46).
- **AdamW**
- **NAdamW** — can be better with word vectors and for speed (Nesterov acceleration).

Start them at an initial learning rate of around **0.001**; most have other hyperparameters
too.

The **W** variants are worth knowing about specifically for word vectors: word vectors are
updated *sparsely*, because particular words only turn up occasionally, and some optimizers
have properties tuned for that (≈23:59). Beyond this there is a whole family of further ideas
— momentum, Nesterov acceleration — that belong to a convex optimization class.

## Gradient clipping

The fix for **exploding gradients** (lecture 5, slide 62). If the norm of the gradient
exceeds a threshold, scale it down *before* applying the update. The slide's pseudo-code,
from Pascanu et al. 2013, with $E$ the error and $\hat{g}$ the gradient about to be
applied:

$$
\begin{aligned}
& \hat{g} \leftarrow \frac{\partial E}{\partial \theta} \\
& \textbf{if } \lVert \hat{g} \rVert \ge \text{threshold} \textbf{ then} \\
& \qquad \hat{g} \leftarrow \frac{\text{threshold}}{\lVert \hat{g} \rVert} \hat{g} \\
& \textbf{end if}
\end{aligned}
$$

The intuition is **same direction, smaller step**. Thresholds around 5, 10 or 20 for the
gradient norm are typical (lecture 6, ≈19:33). Without it, one very large gradient produces
an update that lands somewhere arbitrary and high-loss — Manning's image is "you think you've
found a hill to climb, but suddenly you're in Iowa" — and in the worst case gives Inf or NaN
and forces a restart from a checkpoint.

He is candid that this "isn't high-falutin math, really — it's a crude hack", but it works,
it is often essential, and it is one of lecture 6's four named takeaways (slide 56). See
[vanishing and exploding gradients](vanishing-and-exploding-gradients.md).

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
- **log of a quotient** splits into log numerator minus log denominator,
  $\log(a/b) = \log a - \log b$ (≈1:09:13).
- log and exp **cancel**: $\log \exp(x) = x$ (≈1:10:02).
- The derivative of $u^{\top} v$ with respect to $v$ is $u$. Manning justifies it
  componentwise: the dot product expands to $u_1 v_1 + u_2 v_2 + u_3 v_3 + \cdots$, so
  differentiating with respect to $v_1$ leaves $u_1$ and kills every other term; do
  that along the whole vector and you get $u$ back (≈1:10:49–1:11:34).
- The **chain rule**, applied twice — once for the outer log, once for the exp
  inside the sum (≈1:12:20–1:15:33).

The result for the center vector $v_c$ is

$$\frac{\partial}{\partial v_c} \log P(o \mid c) = u_o - \sum_{x \in V} P(x \mid c)\, u_x$$

which Manning reads as **observed minus expected** (≈1:17:56). It compares the outside vector that
actually occurred against the weighted average of what the model predicted. When
expectation matches observation the derivative is zero and you have reached a
maximum. He notes this observed-minus-expected form "you see quite a bit in these
kinds of derivations" (≈1:18:42) — it is worth recognizing on sight.

Manning's closing remark is the practical one (≈1:19:30): he wants you to
understand how this works, but you will quickly find that computers do it for you
and you will not compute these by hand on a regular basis. The transcript is
unreliable for the notation in this section — several symbols come through as
"[Music]" or wrong subscripts — so use the slides.

## Getting the gradients for an arbitrary network

Lectures 1 and 2 derive gradients for one specific objective. Lecture 3 answers the
general question — how do you compute $\nabla_{\theta} J(\theta)$ for *any* function — and
gives two answers that are the same answer (lecture 3, slide 9): **by hand**, using
[matrix calculus](matrix-calculus.md), and **algorithmically**, using
[backpropagation](backpropagation.md) over a computation graph. One thing worth carrying
back to this page: in deep learning $\theta$ includes the **data representation** itself, so the
word vectors are updated by the same rule as everything else.

## Related pages

- [word2vec](word2vec.md) — the objective these gradients are taken of
- [softmax and cross-entropy](softmax-and-cross-entropy.md) — the functions inside
  the objective
- [matrix calculus](matrix-calculus.md) — Jacobians, the chain rule, and the shape
  convention that makes the update rule a plain subtraction
- [backpropagation](backpropagation.md) — computing these gradients automatically
- [vanishing and exploding gradients](vanishing-and-exploding-gradients.md) — what goes wrong
  over long sequences, and why clipping is needed
- [regularization and dropout](regularization-and-dropout.md) — the other half of lecture 5's
  practical-technique set
- [lecture 1](01-intro-and-word-vectors.md) — the full hand derivation
- [lecture 2](02-word-vectors-and-language-models.md) — SGD, learning rates,
  initialization
- [lecture 3](03-backpropagation-and-neural-networks.md) — gradients for an arbitrary
  network, by hand and algorithmically
- [lecture 5](05-recurrent-neural-networks.md) — Xavier initialization, the adaptive
  optimizer family, and gradient clipping
