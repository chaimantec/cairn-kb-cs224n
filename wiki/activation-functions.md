# Activation functions (non-linearities)

The element-wise function *f* in a neural network layer **h** = *f*(**Wx** + **b**).
Lecture 2 used it without explanation; [lecture 3](03-backpropagation-and-neural-networks.md)
slides 6–8 say what the options are, why the choice matters, and why the layer would be
pointless without one.

Primary source: lecture 3, slides 6–8
([slide text](../raw/slides/03-backpropagation-and-neural-networks.md)), ≈7:48–16:18.

## Why a non-linearity is needed at all

Neural networks do **function approximation** — you want to learn some very complex
function, from a piece of text to its meaning, or from an image to an interpretation
(≈14:45). A matrix multiply is a linear transform, and the fatal property of linear
transforms is that they **compose into a single linear transform**: W₁W₂**x** = W**x**
(slide 8). So a deep network of pure matrix multiplies could be collapsed to one layer and
has exactly the representational power of one layer. Stacking buys nothing.

Insert a non-linearity between the layers and the network can approximate arbitrarily
complex functions. Slide 8 illustrates this twice: a set of scattered points fitted by
progressively more flexible curves, and a two-class point cloud separated first by a
straight diagonal boundary that misclassifies many points, then by a curved boundary that
follows the clusters.

One aside from Manning that is easy to miss, and is a real result rather than a throwaway:
in terms of *representational* power, multi-layer linear networks give you nothing — but in
terms of *learning* they do behave differently, and there is a body of theoretical work
studying linear neural networks for exactly that reason (≈16:18).

## Why the very first neuron could not learn

The original McCulloch and Pitts unit of 1943 was a **threshold** (slide 6):

    1(Wx > θ)  =  1(Wx − θ > 0)

It fires or it does not. The problem is visible in its plot: the function is flat on both
sides of the threshold, so its slope is zero everywhere it is defined. With no gradient
there is nothing for **gradient-based learning** to descend, which is why later work moved
to activations with slopes (≈8:34). Manning's image for the whole enterprise is skiing
during spring break: given slopes, you can work out where it is steeper and head down
(≈9:19).

## The family

Slide 7 lays them out side by side.

| Function | Definition | Range | Notes |
| --- | --- | --- | --- |
| **logistic** ("sigmoid") | 1 / (1 + exp(−*z*)) | (0, 1) | still used where you want a probability |
| **tanh** | (e^z − e^−z)/(e^z + e^−z) | (−1, 1) | a rescaled logistic — see below |
| **hard tanh** | −1 if *z* < −1; *z* if −1 ≤ *z* ≤ 1; 1 if *z* > 1 | [−1, 1] | piecewise-linear, avoids exponentials |
| **ReLU** | max(*z*, 0) | [0, ∞) | the default first thing to try |
| **Leaky ReLU** | small positive slope (e.g. 0.01*z*) below zero | ℝ | removes the dead zone |
| **Parametric ReLU** | *az* below zero, *a* learned | ℝ | the negative slope is a parameter |
| **Swish** | *z* · logistic(*z*) | ℝ | common in Transformers |
| **GELU** | *z* · P(X ≤ *z*), X ~ N(0,1) ≈ *z* · logistic(1.702*z*) | ℝ | common in Transformers (BERT, RoBERTa) |

### tanh is a rescaled logistic

Worth knowing because the exponential formulas make it look unrelated (slide 7):

    tanh(z) = 2 · logistic(2z) − 1

It is the same function, stretched by two and moved down by one — twice as steep, and
mapping to [−1, 1] instead of [0, 1]. Manning suggests treating it as a math exercise if
your calculus is rusty (≈10:06). The motivation for preferring it over the logistic was
that the logistic's output is always non-negative, which tends to push things toward bigger
numbers.

The motivation for **hard tanh** is cheapness: computing tanh means computing exponentials,
which are slow, and a piecewise-linear approximation with slope one in the middle works in
many cases (≈10:54).

### ReLU, and why it works despite looking wrong

ReLU is what hard tanh leads to, and **the first thing to try for a deep network**
(slide 7). It looks like it should fail on the lecture's own terms: in the negative region
the gradient is zero, so the unit is dead. Manning says it still feels slightly perverse to
him (≈11:40), but two things make it work:

- Although an individual neuron is dead whenever its input goes negative, across a network
  enough units are alive, and the pattern of which are alive gives a kind of
  **specialization**.
- In the positive region the slope is always exactly **one**, which gives very clean,
  productive **backward flow of gradients** — the property that matters for
  [backpropagation](backpropagation.md) (≈12:27).

It trains quickly, performs well, became the default for years, and is what the course uses
in Assignment 2 — including in the neural dependency parser's hidden layer,
*h* = ReLU(**Wx** + **b**₁) ([lecture 4](04-dependency-parsing.md), slide 44).

### The dead zone, and what came after

Eventually people had second thoughts about an activation that is dead over half its range
(≈12:27). **Leaky ReLU** gives the negative half a straight line with a small slope;
**Parametric ReLU** makes that slope a learned parameter. More recently **Swish** and
**GELU** turn up in Transformer models — both look essentially like *y* = *x* for positive
inputs, with a funny bit of curve below zero that restores some slope, curving the opposite
way to what you might expect (≈13:58).

## Practical summary

- **Deep network, no other information:** start with **ReLU**.
- **Need a probability out:** **logistic**, as in the lecture's binary NER classifier
  (lecture 3, slide 5). For a distribution over several classes use
  [softmax](softmax-and-cross-entropy.md).
- **Recurrent networks:** **tanh** — the course uses it in Assignment 3 (≈10:06).
- **Transformer-family models:** **GELU** or **Swish**.

## Related pages

- [Backpropagation](backpropagation.md) — where *f*′(**z**) enters, as the diagonal local
  gradient diag(*f*′(**z**)).
- [Matrix calculus](matrix-calculus.md) — why the Jacobian of an element-wise activation is
  diagonal.
- [Softmax, the logistic function, and cross-entropy](softmax-and-cross-entropy.md) — the
  output-layer functions, as opposed to the hidden-layer ones here.
- [Lecture 3](03-backpropagation-and-neural-networks.md) — the lecture this comes from.
