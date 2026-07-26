# Regularization and dropout

Covered in the first quarter of [lecture 5](05-recurrent-neural-networks.md) (slides 7–9),
as part of the set of practical techniques that turned deep networks from a fifteen-year
dead end into something that works. The section is short but contains one genuinely
important claim: **the textbook picture of overfitting is not what modern neural network
practitioners believe.**

## The classic view

A full loss function includes a regularization term over all parameters $\theta$. The most
common is **L2 regularization** (slide 7):

$$J(\theta) = \frac{1}{N} \sum_{i=1}^{N} -\log \left( \frac{e^{f_{y_i}}}{\sum_c e^{f_c}} \right) + \lambda \sum_k \theta_k^2$$

where $N$ is the number of training examples, $f_c$ the score the model gives class $c$,
$y_i$ the correct class for example $i$, and $\lambda$ the regularization strength.
The term says: among models that explain the data, prefer the one with **small parameter
weights**. Classically its job is to prevent **overfitting** — the model does very well on
the training data but generalizes badly to data it has not seen. The textbook picture on the
left of slide 7 shows training error falling monotonically as model "power" increases, while
test error falls and then **rises**; the gap is overfitting, and shrinking the parameters
numerically is meant to reduce it.

Manning notes that regularization theory belongs to the machine learning classes — CS229 —
and does not develop it here (≈8:32).

## The modern view

This is the part that matters (≈10:54–12:28). Slide 7's second bullet:

> **Now:** Regularization produces models that **generalize well** when we have a "big" model.
> We do not care that our models overfit on the training data, even though they are **hugely**
> overfit.

Manning states it more bluntly in lecture: *"This is not a picture that modern neural network
people believe at all… We don't believe that overfitting exists anymore, but what we are
concerned about is models that will generalize well to different data."*

The reasoning. In classical statistics, training billions of parameters would be absurd —
you cannot estimate them, so you get noise. What is actually found is that yes, you cannot
estimate the numbers well, but "what you get is a kind of an interesting averaging function
from all these myriad numbers". If you keep training a huge network past the point where it
looks like it is starting to overfit, the **validation** loss keeps going down too.

So in current practice models are trained until they fit the training set essentially
perfectly — "maybe it's 0.007 loss or something, but you can train them to get zero loss,
because you've got such rich models you can perfectly fit, memorize, the entire training
set." Classically that is a disaster; with modern large networks it is not, **provided the
regularization was done well**, because the model still generalizes.

The right-hand plot on slide 7 is the empirical evidence: a measured zero-one loss curve
where the test error rises to a bump and then **falls again**, continuing down toward the
training curve.

## Why dropout, and not L2

The flip side of the modern view (≈13:14): L2 and L1 regularization **are not strong enough**
to produce that generalization effect on big models. So the field turned to other methods, of
which "everyone's favorite is dropout".

## Dropout

**Srivastava, Hinton, Krizhevsky, Sutskever and Salakhutdinov, 2012/JMLR 2014** (slides 8–9).

**At training time**, for each data point each time, randomly set inputs to 0 with
probability $p$ — the **dropout ratio**. Slide 8 gives $p = 0.5$ typically, and less (around
0.15) for the input layer. Technically you sample a random mask of zeros and ones each time
and take its **Hadamard product** with the data, so different components disappear on
different passes (≈14:01).

**At test time** you drop nothing, and multiply all weights by $1 - p$ to compensate for the
fact that you used to be dropping things. Slide 8's arithmetic makes this concrete: where a
training pass with $x_2$ and $x_4$ zeroed computes

$$y = w_1 x_1 + w_3 x_3 + b$$

the test-time computation is

$$y = (1 - p)(w_1 x_1 + w_2 x_2 + w_3 x_3 + w_4 x_4)$$

A practical caveat from lecture (≈13:14): the formulation on slide 8 is the **original**
one; the version used in Assignment 2 is the way deep learning packages now do it, and a
couple of details differ.

## Three ways to understand why it works

Manning gives all three (≈15:33–17:04), and slide 9 carries the headline: *Prevents Feature
Co-adaptation = Good Regularization! **Use it everywhere!***

1. **No feature co-adaptation.** The model cannot become reliant on, say, component 17 of a
   vector, because component 17 sometimes randomly disappears. If there are other features
   that would let it work out what to do, it has to learn to use those too. Slide 9's
   phrasing: "a feature cannot only be useful in the presence of particular other features."

2. **An implicit model ensemble.** Training with dropout is like training with a huge
   ensemble — the ensemble of the power set, an exponential number of possible dropout
   patterns, all at once. Slide 9 calls this a form of **model bagging**.

3. **A middle ground between naïve Bayes and logistic regression.** In a single layer:
   naïve Bayes sets each feature's weight independently, from data statistics alone, with no
   regard to the other features; logistic regression sets every weight in the context of all
   the others; **dropout sits in between**, setting weights in the context of *some* of the
   other features, with different ones absent at different times.

The current framing, following **Wager, Wang and Liang (2013)** at Stanford, is that dropout
is a strong **feature-dependent regularizer**, and that paper gives theoretical results for
why to think of it that way (slide 9, ≈17:04).

## Implementation note

Dropout is a good example of the vectorization rule from the same lecture (slide 10): you do
**not** write a for loop over positions setting some to zero. You build a mask vector and
take an element-wise product (≈18:38). See [gradient descent](gradient-descent.md) for the
rest of the practical-technique set — initialization, optimizers and gradient clipping.

## Related pages

- [Gradient descent](gradient-descent.md) — initialization, adaptive optimizers, and gradient
  clipping, the other practical techniques from this part of the lecture.
- [Softmax, the logistic function, and cross-entropy](softmax-and-cross-entropy.md) — the loss
  the regularizer is added to.
- [Backpropagation](backpropagation.md) — what the dropout mask sits inside.
- [Lecture 5 — Recurrent Neural Networks](05-recurrent-neural-networks.md) — the lecture this
  comes from.
