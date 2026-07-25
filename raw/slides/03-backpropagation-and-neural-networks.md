---
title: Lecture 3 — Neural net learning: Gradients by hand (matrix calculus) and algorithmically (backpropagation) (slide deck)
lecture: 3
slides: 85
source_pdf: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture03-neuralnets.pdf
note: Printed slide numbers match the PDF page numbers exactly, 1–85.
---

# Lecture 3 — Gradients by Hand and Algorithmically: slide-by-slide

Text and figures of all 85 slides of
[`cs224n-spr2024-lecture03-neuralnets.pdf`](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture03-neuralnets.pdf),
transcribed from the deck. Cite these as "slide N" — the printed number equals the
PDF page number. Diagrams and plots are described in prose since the KB is read as
text.

**Two numbering quirks in the deck itself.** The title slide calls this "Lecture 3",
but the plan on slide 10 is headed "Lecture 4: Gradients by hand and algorithmically",
and slide 6 is headed "7. Neural computation" — both are leftovers from an earlier
year's ordering, where the recap material on slides 3–9 belonged to a different lecture.
The section numbering that actually applies to this deck is the plan on slide 10:
1. Introduction, 2. Matrix calculus, 3. Backpropagation.

Many slides are **build steps** — the same slide re-shown with one more line revealed.
They are transcribed individually so that a citation to any one of them resolves, but
the substantive content of a run (e.g. slides 17–21, or 59–69) is the last slide in it.

Companion pages: [wiki page for this lecture](../../wiki/03-backpropagation-and-neural-networks.md) ·
[transcript](../transcripts/03-backpropagation-and-neural-networks.md)

## Contents

| Slides | Section |
| ------ | ------- |
| 1 | Title |
| 2 | §1 Introduction: Assignment 2, readings, PyTorch tutorial |
| 3–5 | Recap: a neural net as several logistic regressions; matrix notation; the NER window classifier |
| 6–8 | Neural computation: the McCulloch & Pitts unit, non-linearities old and new, why non-linearities are needed |
| 9 | Recap: stochastic gradient descent, and the two ways to get gradients |
| 10 | Lecture plan |
| 11–12 | §2 Matrix calculus: framing, and the Math 51 reference |
| 13–15 | Gradients: one input, *n* inputs, and the Jacobian |
| 16 | The chain rule: multiply derivatives, multiply Jacobians |
| 17–21 | Worked Jacobian: element-wise activation → diag(*f*′(**z**)) |
| 22–25 | Other Jacobians: ∂(**Wx**+**b**)/∂**x**, /∂**b**, and ∂(**u**ᵀ**h**)/∂**u** |
| 26–37 | The full worked example: ∂s/∂**b** in four steps (break up, chain rule, write Jacobians, simplify) |
| 38–40 | Re-using computation: the shared prefix, and δ the upstream gradient |
| 41–44 | Derivative with respect to a matrix: the shape convention, δᵀ**x**ᵀ, and why the transposes |
| 45 | Deriving the local input gradient ∂z_i/∂W_ij = x_j |
| 46–47 | Jacobian form vs shape convention, and the two ways to work |
| 48 | §3 Backpropagation: what is left to add |
| 49–52 | Computation graphs, forward propagation, and passing gradients backwards |
| 53–58 | A single node: upstream × local = downstream; nodes with multiple inputs |
| 59–69 | The numeric example *f*(*x*,*y*,*z*) = (*x*+*y*)·max(*y*,*z*) worked end to end |
| 70–71 | Gradients sum at outward branches |
| 72–74 | Node intuitions: + distributes, max routes, * switches |
| 75–77 | Efficiency: compute all gradients at once, not one at a time |
| 78 | Back-prop in a general computation graph; fprop and bprop have the same complexity |
| 79–82 | Automatic differentiation and the forward/backward API |
| 83 | Manual gradient checking with the numeric gradient |
| 84 | Summary |
| 85 | Why learn all these details about gradients? |

---

## Slide 1 — Title

"Natural Language Processing with Deep Learning — CS224N/Ling284". Christopher
Manning. "Lecture 3: Neural net learning: Gradients by hand (matrix calculus) and
algorithmically (the backpropagation algorithm)".

## Slide 2 — 1. Introduction

Assignment 2 makes sure you really understand the math of neural networks … then we'll
let the software do it! It also teaches us about dependency parsing.

We'll go through all the math quickly today, but this is the one week of quarter to most
work through the readings!!!

This will be a tough week for some! → Make sure to get help if you need it:
visit office hours, read tutorial materials on the syllabus!

Thursday will be mainly linguistics! Some people find that tough too. 😉

**PyTorch tutorial:** 3:30pm Friday Apr 12 in Gates B01. A great chance to get an intro
to PyTorch, a key deep learning package, used in Ass 2+!

## Slide 3 — Where we ended last time: A neural network = running several logistic regressions at the same time

"Before we know it, we have a multilayer neural network…."

The figure is the standard four-layer feed-forward diagram: layer L₁ holds inputs
x₁, x₂, x₃ plus a +1 bias unit (blue circles); layer L₂ holds three units plus a +1
bias; layer L₃ holds two units plus a +1 bias; layer L₄ holds two output units emitting
h_{W,b}(x). Every unit in a layer connects to every unit in the next.

Margin note: *This allows us to re-represent and compose our data multiple times and to
learn a classifier that is highly non-linear in terms of the original inputs* (but,
typically, is linear in terms of the pre-final layer representations).

## Slide 4 — Matrix notation for a layer

We have

    a₁ = f(W₁₁x₁ + W₁₂x₂ + W₁₃x₃ + b₁)
    a₂ = f(W₂₁x₁ + W₂₂x₂ + W₂₃x₃ + b₂)
    etc.

In matrix notation

    z = Wx + b
    a = f(z)

Activation *f* is applied element-wise:

    f([z₁, z₂, z₃]) = [f(z₁), f(z₂), f(z₃)]

The figure shows inputs x₁, x₂, x₃ and a +1 unit fully connected to outputs a₁, a₂, a₃,
with one arrow labelled W₁₂ (the weight from x₂ into a₁) and the bias arrow into a₃
labelled b₃.

## Slide 5 — NER: Binary classification for center word being location

We do supervised training and want high score if it's a location.

    J_t(θ) = σ(s) = 1 / (1 + e^{−s})        ← predicted model probability of class
    s = uᵀh
    h = f(Wx + b)
    x (input) ∈ R^{5d}

The right-hand figure is the window classifier drawn bottom to top: a 5*d*-long input
vector, an 8-unit hidden layer, a single scalar *s*, then a single output dot. A margin
note reads "*f* = Some element-wise non-linear function, e.g., logistic, tanh, ReLU".
Below the input: x = [ x_museums, x_in, x_Paris, x_are, x_amazing ], labelled
"Embedding of 1-hot words". A small sigmoid curve is inset next to the σ formula.

## Slide 6 — 7. Neural computation

A textbook line drawing of three biological neurons, labelled with dendrites, soma,
axon, myelin sheath and terminal buttons, showing the axon of one cell terminating on
the dendrites of the next.

Caption: **Original McCulloch & Pitts 1943 threshold unit:**

    1(Wx > θ) = 1(Wx − θ > 0)

This function has no slope, so, no **gradient-based learning**.

To the right, a plot of the step function: flat at 0 for negative argument, jumping to 1
at zero and flat thereafter.

## Slide 7 — Non-linearities, old and new

Five activation functions across the top, each with its formula and plot:

- **logistic ("sigmoid")**: *f*(*z*) = 1 / (1 + exp(−*z*)); S-curve from 0 to 1,
  crossing 0.5 at *z* = 0, plotted over −6 to 6.
- **tanh**: tanh(*z*) = (e^z − e^{−z}) / (e^z + e^{−z}); S-curve from −1 to 1.
- **hard tanh**: HardTanh(*x*) = −1 if *x* < −1; *x* if −1 ≤ *x* ≤ 1; 1 if *x* > 1.
  Plotted as the piecewise-linear approximation drawn over the smooth tanh.
- **ReLU** (Rectified Linear Unit): ReLU(*z*) = max(*z*, 0); flat at 0 to the left of
  the origin, then the 45° line.
- **Leaky ReLU / Parametric ReLU**: two lines with a small negative slope to the left of
  the origin — Leaky ReLU *y* = 0.01*x*, Parametric ReLU *y* = *ax* — and *y* = *x* to
  the right.

Text below:

tanh is just a rescaled and shifted sigmoid (2 × as steep, [−1,1]):

    tanh(z) = 2·logistic(2z) − 1

Logistic and tanh are still used (e.g., logistic to get a probability).

However, now, for deep networks, the first thing to try is ReLU: it trains quickly and
performs well due to good gradient backflow.

ReLU has a negative "dead zone" that recent proposals mitigate.

GELU/Swish often used with Transformers (BERT, RoBERTa, etc.)

Two newer functions in the right column, each plotted:

- **Swish** (arXiv:1710.05941): swish(*x*) = *x* · logistic(*x*)
- **GELU** (arXiv:1606.08415): GELU(*x*) = *x* · P(X ≤ *x*), X ~ N(0,1)
  ≈ *x* · logistic(1.702*x*)

## Slide 8 — Non-linearities (i.e., "*f*" on previous slide): Why they're needed

Neural networks do function approximation, e.g., regression or classification.

- Without non-linearities, deep neural networks can't do anything more than a linear
  transform.
- Extra layers could just be compiled down into a single linear transform: W₁W₂x = Wx
- But, with more layers that include non-linearities, they can approximate any complex
  function!

Right: three stacked plots of the same scattered blue × data points over [0,1], fitted
by a red curve of increasing flexibility — a single smooth step, then a sine-like wave,
then a wiggly curve that passes through nearly every point.

Bottom left: two classification plots of the same green/red point cloud. The first is
split by a straight diagonal decision boundary that misclassifies many points; the
second by a curved boundary that wraps around the green cluster.

## Slide 9 — Remember: Stochastic Gradient Descent

Update equation:

    θ^{new} = θ^{old} − α ∇_θ J(θ)         ← α = step size or learning rate

i.e., for each parameter: θ_j^{new} = θ_j^{old} − α ∂J(θ)/∂θ_j^{old}

In deep learning, θ includes the data representation (e.g., word vectors) too!

How can we compute ∇_θ J(θ)?

1. By hand
2. Algorithmically: the backpropagation algorithm

## Slide 10 — Lecture Plan

"Lecture 4: Gradients by hand and algorithmically" *(deck's own heading — see the note
at the top of this file)*

1. Introduction (10 mins)
2. Matrix calculus (35 mins)
3. Backpropagation (35 mins)

Key Learning: The mathematics and practical implementation of how neural networks are
trained by backpropagation.

## Slide 11 — Computing Gradients by Hand

- **Matrix calculus:** Fully vectorized gradients
  - "Multivariable calculus is just like single-variable calculus if you use matrices"
  - Much faster and more useful than non-vectorized gradients
  - But doing a non-vectorized gradient can be good for intuition; recall the first
    lecture for an example
  - **Lecture notes and matrix calculus notes cover this material in more detail**
  - **You might also review Math 51, which has an online textbook:**
    http://web.stanford.edu/class/math51/textbook.html

## Slide 12 — (Math 51 textbook screenshot)

Two page images side by side. Left: the cover of *Linear Algebra, Multivariable Calculus,
and Modern Applications*, "Math 51 course text prepared by the Stanford University Math
Department", last modified on April 3, 2024.

Right: page 697 of that text, section **G. Neural networks and the multivariable Chain
Rule (optional)**. It opens with two quotations:

> "…research on neural networks was held up for 20 years until somebody remembered the
> Chain Rule!" — T. Griffiths, Luce Professor of Information Technology, Consciousness,
> and Culture (Princeton)

> "…linear algebra simplifies the complex processes that underlie neural network
> operations, […] allowing [AI models] to efficiently process vast amounts of data,
> recognize patterns, and make decisions." — Medium article "The Crucial Role of
> Mathematics in AI Development"

Subsection G.1, "Mathematical model of a neural network as a composition of functions",
develops the cat-classifier example: a function *f*_cat : **R**^{1000} → **R** on a
1000-pixel image returning > 0.5 for a cat and < 0.5 otherwise, built as a composition
of intermediate vector-valued functions. It gives the general form

    R^n --f₁--> R^{d₁} --f₂--> R^{d₂} --f₃--> … --f_{N−1}--> R^{d_{N−1}} --f_N--> R^m

calling **R**^n the input layer, **R**^m the output layer, and each **R**^{d_i} in
between a hidden layer; a composition of *N* functions is an *N*-layer neural network,
so a single-layer network has no hidden layers and no function composition at all. Each
x_{i,j} is a *node* or *unit* — "hence the name 'neural network'" — and the page notes
that many people dislike the brain analogy and call the whole thing a *multi-layer
perceptron* instead. Footer: Copyright © 2021 Stanford University Department of
Mathematics; "NOTICE RE UPLOADING TO WEBSITES: This content is protected and may not be
shared, uploaded or distributed."

## Slide 13 — Gradients

- Given a function with 1 output and 1 input

      f(x) = x³

- It's gradient (slope) is its derivative

      df/dx = 3x²

  "How much will the output change if we change the input a bit?"

  At *x* = 1 it changes about 3 times as much: 1.01³ = 1.03
  At *x* = 4 it changes about 48 times as much: 4.01³ = 64.48

## Slide 14 — Gradients

- Given a function with 1 output and *n* inputs

      f(x) = f(x₁, x₂, …, x_n)

- Its gradient is a vector of partial derivatives with respect to each input

      ∂f/∂x = [ ∂f/∂x₁, ∂f/∂x₂, …, ∂f/∂x_n ]

## Slide 15 — Jacobian Matrix: Generalization of the Gradient

- Given a function with ***m* outputs** and *n* inputs

      f(x) = [ f₁(x₁, x₂, …, x_n), …, f_m(x₁, x₂, …, x_n) ]

- It's Jacobian is an ***m* x *n* matrix** of partial derivatives

      ∂f/∂x = ⎡ ∂f₁/∂x₁  …  ∂f₁/∂x_n ⎤
              ⎢    ⋮      ⋱     ⋮     ⎥
              ⎣ ∂f_m/∂x₁ …  ∂f_m/∂x_n ⎦

Boxed in red at the right: (∂f/∂x)_{ij} = ∂f_i/∂x_j

## Slide 16 — Chain Rule

- For composition of one-variable functions: **multiply derivatives**

      z = 3y
      y = x²
      dz/dx = (dz/dy)(dy/dx) = (3)(2x) = 6x

- For multiple variables functions: **multiply Jacobians**

      h = f(z)
      z = Wx + b
      ∂h/∂x = (∂h/∂z)(∂z/∂x) = …

## Slide 17 — Example Jacobian: Elementwise activation Function

    h = f(z), what is ∂h/∂z?        h, z ∈ R^n
    h_i = f(z_i)

*(First build step — the question only.)*

## Slide 18 — Example Jacobian: Elementwise activation Function

Same setup, plus: Function has *n* outputs and *n* inputs → *n* by *n* Jacobian.

## Slide 19 — Example Jacobian: Elementwise activation Function

Same setup, plus the first line of the derivation:

    (∂h/∂z)_{ij} = ∂h_i/∂z_j = (∂/∂z_j) f(z_i)        definition of Jacobian

## Slide 20 — Example Jacobian: Elementwise activation Function

Adds the case split:

    (∂h/∂z)_{ij} = ∂h_i/∂z_j = (∂/∂z_j) f(z_i)        definition of Jacobian
                 = { f′(z_i)  if i = j
                   { 0        if otherwise            regular 1-variable derivative

## Slide 21 — Example Jacobian: Elementwise activation Function

Completes the result:

    ∂h/∂z = ⎛ f′(z₁)        0    ⎞
            ⎜        ⋱            ⎟  = diag(f′(z))
            ⎝   0        f′(z_n) ⎠

## Slide 22 — Other Jacobians

    (∂/∂x)(Wx + b) = W

## Slide 23 — Other Jacobians

    (∂/∂x)(Wx + b) = W
    (∂/∂b)(Wx + b) = I   (Identity matrix)

## Slide 24 — Other Jacobians

    (∂/∂x)(Wx + b) = W
    (∂/∂b)(Wx + b) = I   (Identity matrix)
    (∂/∂u)(uᵀh) = hᵀ

Margin note: *Fine print: This is the correct Jacobian. Later we discuss the "shape
convention"; using it the answer would be* **h**.

## Slide 25 — Other Jacobians

The same three results, plus:

- Compute these at home for practice!
  - Check your answers with the lecture notes

## Slide 26 — Back to our Neural Net!

    s = uᵀh
    h = f(Wx + b)
    x (input)

The figure repeats the window-classifier stack from slide 5: the 5*d* input vector at
the bottom, an 8-unit hidden layer above it, and the scalar *s* at the top, with
x = [ x_museums, x_in, x_Paris, x_are, x_amazing ].

## Slide 27 — Back to our Neural Net!

- Let's find ∂s/∂**b**
  - Really, we care about the gradient of the loss J_t but we will compute the gradient
    of the score for simplicity (and because the logistic is in Ass 1!)

Same equations and figure as slide 26.

## Slide 28 — 1. Break up equations into simple pieces

    s = uᵀh                    s = uᵀh
    h = f(Wx + b)      →       h = f(z)
                               z = Wx + b
    x (input)                  x (input)

Carefully define your variables and keep track of their dimensionality!

## Slide 29 — 2. Apply the chain rule

    s = uᵀh                    ∂s/∂b = (∂s/∂h)(∂h/∂z)(∂z/∂b)
    h = f(z)
    z = Wx + b
    x (input)

## Slide 30 — 2. Apply the chain rule

Same as slide 29, with `s = uᵀh` boxed on the left and the matching factor ∂s/∂**h**
boxed on the right.

## Slide 31 — 2. Apply the chain rule

Same, with `h = f(z)` boxed and the factor ∂**h**/∂**z** boxed.

## Slide 32 — 2. Apply the chain rule

Same, with `z = Wx + b` boxed and the factor ∂**z**/∂**b** boxed.

## Slide 33 — 3. Write out the Jacobians

The chain rule ∂s/∂**b** = (∂s/∂**h**)(∂**h**/∂**z**)(∂**z**/∂**b**), with a boxed
reminder at bottom left:

    Useful Jacobians from previous slide
    (∂/∂u)(uᵀh) = hᵀ
    (∂/∂z)(f(z)) = diag(f′(z))
    (∂/∂b)(Wx + b) = I

## Slide 34 — 3. Write out the Jacobians

First factor substituted: an arrow under ∂s/∂**h** points to **u**ᵀ.

## Slide 35 — 3. Write out the Jacobians

Second factor substituted: **u**ᵀ diag(*f*′(**z**)).

## Slide 36 — 3. Write out the Jacobians

Third factor substituted:

    = uᵀ diag(f′(z)) I

## Slide 37 — 3. Write out the Jacobians

Simplified:

    = uᵀ diag(f′(z)) I
    = uᵀ ⊙ f′(z)

Margin note: ⊙ = Hadamard product = element-wise multiplication of 2 vectors to give
vector.

## Slide 38 — Re-using Computation

- Suppose we now want to compute ∂s/∂**W**
  - Using the chain rule again:

        ∂s/∂W = (∂s/∂h)(∂h/∂z)(∂z/∂W)

## Slide 39 — Re-using Computation

    ∂s/∂W = (∂s/∂h)(∂h/∂z)(∂z/∂W)
    ∂s/∂b = (∂s/∂h)(∂h/∂z)(∂z/∂b)

The first two factors are printed in blue in both lines. Caption: The same! Let's avoid
duplicated computation …

## Slide 40 — Re-using Computation

    ∂s/∂W = δ (∂z/∂W)
    ∂s/∂b = δ (∂z/∂b) = δ
    δ = (∂s/∂h)(∂h/∂z) = uᵀ ⊙ f′(z)

δ is the upstream gradient ("error signal")

## Slide 41 — Derivative with respect to Matrix: Output shape

- What does ∂s/∂**W** look like?        **W** ∈ R^{n×m}
- 1 output, *nm* inputs: 1 by *nm* Jacobian?
  - Inconvenient to then do θ^{new} = θ^{old} − α ∇_θ J(θ)

## Slide 42 — Derivative with respect to Matrix: Output shape

Same two bullets, plus:

- Instead, we leave pure math and use the **shape convention**: the shape of the
  gradient is the shape of the parameters!
- So ∂s/∂**W** is *n* by *m*:

      ⎡ ∂s/∂W₁₁   …   ∂s/∂W_{1m} ⎤
      ⎢    ⋮       ⋱       ⋮      ⎥
      ⎣ ∂s/∂W_{n1} …  ∂s/∂W_{nm} ⎦

## Slide 43 — Derivative with respect to Matrix

- What is ∂s/∂**W** = δ (∂**z**/∂**W**)
  - δ is going to be in our answer
  - The other term should be **x** because **z** = **Wx** + **b**

- Answer is: ∂s/∂**W** = δᵀ**x**ᵀ

δ is upstream gradient ("error signal") at **z**
**x** is local input signal

## Slide 44 — Why the Transposes?

    ∂s/∂W    =     δᵀ        xᵀ
    [n × m]     [n × 1]   [1 × m]

             = ⎡δ₁⎤                    ⎡ δ₁x₁  …  δ₁x_m ⎤
               ⎢⋮ ⎥ [x₁, …, x_m]  =    ⎢  ⋮     ⋱    ⋮   ⎥
               ⎣δ_n⎦                   ⎣ δ_nx₁ …  δ_nx_m⎦

- Hacky answer: this makes the dimensions work out!
  - Useful trick for checking your work!
- Full explanation in the lecture notes
  - Each input goes to each output — you want to get outer product

## Slide 45 — Deriving local input gradient in backprop

- For ∂**z**/∂**W** in our equation:

      ∂s/∂W = δ (∂z/∂W) = δ (∂/∂W)(Wx + b)

- Let's consider the derivative of a single weight W_ij
- W_ij only contributes to z_i
  - For example: W₂₃ is only used to compute z₂ not z₁

        ∂z_i/∂W_ij = (∂/∂W_ij) W_{i·}·x + b_i
                   = (∂/∂W_ij) Σ_{k=1}^{d} W_ik x_k = x_j

The boxed figure at the right is a small two-hidden-unit network drawn in red: inputs
x₁, x₂, x₃ and a +1 unit at the bottom, fully connected up to h₁ = *f*(z₁) and
h₂ = *f*(z₂), which both feed the scalar *s* at the top. Three arrows label specific
parameters: u₂ on the edge from h₂ to *s*, W₂₃ on the edge from x₃ to h₂, and b₂ on the
edge from the +1 unit to h₂.

## Slide 46 — What shape should derivatives be?

- Similarly, ∂s/∂**b** = **h**ᵀ ⊙ *f*′(**z**) is a row vector
  - But shape convention says our gradient should be a column vector because **b** is a
    column vector …
- Disagreement between Jacobian form (which makes the chain rule easy) and the shape
  convention (which makes implementing SGD easy)
  - We expect answers in the assignment to follow the **shape convention**
  - But Jacobian form is useful for computing the answers

## Slide 47 — What shape should derivatives be?

Two options for working through specific problems:

1. Use Jacobian form as much as possible, reshape to follow the shape convention at the
   end:
   - What we just did. But at the end transpose ∂s/∂**b** to make the derivative a
     column vector, resulting in δᵀ
2. Always follow the shape convention
   - Look at dimensions to figure out when to transpose and/or reorder terms
   - The error message δ that arrives at a hidden layer has the same dimensionality as
     that hidden layer

## Slide 48 — 3. Backpropagation

We've almost shown you backpropagation

- It's taking derivatives and using the (generalized, multivariate, or matrix) chain rule

One more concept:

- We **re-use** derivatives computed for higher layers in computing derivatives for
  lower layers to minimize computation

## Slide 49 — Computation Graphs and Backpropagation

- Software represents our neural net equations as a graph
  - Source nodes: inputs
  - Interior nodes: operations

    s = uᵀh
    h = f(z)
    z = Wx + b
    x (input)

The graph is a left-to-right chain of four circular nodes: **x** enters a "•"
(multiplication) node that also takes **W** from below; its output enters a "+" node
that also takes **b**; that enters an "*f*" node; that enters a second "•" node that also
takes **u**; an arrow leaves to the right.

## Slide 50 — Computation Graphs and Backpropagation

Same, with a third bullet — Edges pass along result of the operation — and the edges now
labelled: **x** → [•] → **Wx** → [+] → **z** → [*f*] → **h** → [•] → *s*.

## Slide 51 — Computation Graphs and Backpropagation

The same labelled graph, with a large red box overlaid on the slide reading **"Forward
Propagation"**.

## Slide 52 — Backpropagation

- Then go backwards along edges
  - Pass along **gradients**

The same chain, now with blue arrows running right to left underneath it: ∂s/∂s enters
from the right into the [•] node, ∂s/∂**h** flows back into [*f*], ∂s/∂**z** back into
[+], and ∂s/∂**b** drops down to **b**.

## Slide 53 — Backpropagation: Single Node

- Node receives an "upstream gradient"
- Goal is to pass on the correct "downstream gradient"

Boxed: **h** = *f*(**z**)

A single large circle labelled *f*, with **z** entering from the left and **h** leaving
to the right along the top; underneath, a blue arrow enters from the right labelled
∂s/∂**h** (Upstream gradient) and leaves to the left labelled ∂s/∂**z** (Downstream
gradient).

## Slide 54 — Backpropagation: Single Node

Adds:

- Each node has a **local gradient**
  - The gradient of its output with respect to its input

The circle now carries ∂**h**/∂**z** inside it, labelled Local gradient.

## Slide 55 — Backpropagation: Single Node

Adds the chain rule, boxed in red at the lower left with the label "Chain rule!":

    ∂s/∂z = (∂s/∂h)(∂h/∂z)

## Slide 56 — Backpropagation: Single Node

Adds the summary line:

- [downstream gradient] = [upstream gradient] x [local gradient]

## Slide 57 — Backpropagation: Single Node

- What about nodes with multiple inputs?

Boxed: **z** = **Wx**

A single "*" node with two incoming arrows, **W** from the upper left and **x** from the
lower left, and **z** leaving to the right.

## Slide 58 — Backpropagation: Single Node

- Multiple inputs → multiple local gradients

The same node now carries two local gradients, ∂**z**/∂**W** and ∂**z**/∂**x**, and
sends two downstream gradients back along its two input edges:

    ∂s/∂W = (∂s/∂z)(∂z/∂W)
    ∂s/∂x = (∂s/∂z)(∂z/∂x)

with the single upstream gradient ∂s/∂**z** arriving from the right.

## Slide 59 — An Example

Boxed:

    f(x, y, z) = (x + y) max(y, z)
    x = 1, y = 2, z = 0

*(Question only — the rest is built up over slides 60–69.)*

## Slide 60 — An Example

Forward prop steps

    a = x + y
    b = max(y, z)
    f = ab

The graph: *x* and *y* both feed a [+] node; *y* and *z* both feed a [max] node; the two
outputs feed a [*] node whose output leaves to the right. Note that *y* branches to both
the + and the max.

## Slide 61 — An Example

The same graph with forward values on the edges: *x* = 1 and *y* = 2 into [+] giving 3;
*y* = 2 and *z* = 0 into [max] giving 2; 3 and 2 into [*] giving **6**.

## Slide 62 — An Example

Adds the first local gradients, with the [+] node circled in blue:

    ∂a/∂x = 1    ∂a/∂y = 1

## Slide 63 — An Example

Adds, with the [max] node circled:

    ∂b/∂y = 1(y > z) = 1     ∂b/∂z = 1(z > y) = 0

## Slide 64 — An Example

Adds, with the [*] node circled:

    ∂f/∂a = b = 2     ∂f/∂b = a = 3

## Slide 65 — An Example

Backprop begins: ∂f/∂f = 1 is written in blue on the output edge.

## Slide 66 — An Example

The [*] node's local gradients are boxed, and its two downstream gradients are written
in blue on the incoming edges: 1*2 = 2 on the edge from [+], and 1*3 = 3 on the edge
from [max]. Caption: upstream * local = downstream.

## Slide 67 — An Example

The [max] node's local gradients are boxed, and its downstream gradients are written on
its inputs: 3*1 = 3 on the *y* edge and 3*0 = 0 on the *z* edge.

## Slide 68 — An Example

The [+] node's local gradients are boxed, and its downstream gradients are written on
its inputs: 2*1 = 2 on the *x* edge and 2*1 = 2 on the *y* edge.

## Slide 69 — An Example

The final answers, in blue at the left:

    ∂f/∂x = 2
    ∂f/∂y = 3 + 2 = 5
    ∂f/∂z = 0

Note that *y*'s gradient is the **sum** of the 2 arriving from the [+] branch and the 3
arriving from the [max] branch.

## Slide 70 — Gradients sum at outward branches

A schematic: one node on the left whose output branches into two nodes on the right.
Black arrows run left to right (forward); blue arrows run right to left (backward), and
the two backward arrows meet at the left-hand node where a blue **+** marks their sum.

## Slide 71 — Gradients sum at outward branches

The same figure, plus the equations:

    a = x + y
    b = max(y, z)      ∂f/∂y = (∂f/∂a)(∂a/∂y) + (∂f/∂b)(∂b/∂y)
    f = ab

## Slide 72 — Node Intuitions

- **+** "distributes" the upstream gradient to each summand

The example graph, with the upstream gradient 2 at the [+] node's output copied
unchanged onto both of its inputs (2 on the *x* edge, 2 on the *y* edge).

## Slide 73 — Node Intuitions

Adds:

- **max** "routes" the upstream gradient

The upstream 3 at the [max] node goes entirely to *y* (the larger input) and 0 to *z*.

## Slide 74 — Node Intuitions

Adds:

- **\*** "switches" the upstream gradient

The [*] node's upstream gradient 1 arrives, and each input edge receives it multiplied
by the *other* input's forward value — 2 on the branch whose sibling carried 2, 3 on the
branch whose sibling carried 3.

## Slide 75 — Efficiency: compute all gradients at once

- Incorrect way of doing backprop:
  - First compute ∂s/∂**b**

The chain graph **x** → [*] → [+] → [*f*] → [•], with **W**, **b**, **u** feeding in
from below, and a single blue backward path running from the right end to **b**.

## Slide 76 — Efficiency: compute all gradients at once

Adds:

  - Then independently compute ∂s/∂**W**
  - Duplicated computation!

A second, red backward path is drawn running the whole length of the graph to **W**,
overlapping the blue one along its entire shared stretch.

## Slide 77 — Efficiency: compute all gradients at once

- Correct way:
  - Compute all the gradients at once
  - Analogous to using δ when we computed gradients by hand

Now a single green backward path runs from the right to the [+] node; the blue arrow
drops from there to **b** and the red arrow continues one node further to **W** — the
shared prefix traversed once.

## Slide 78 — Back-Prop in General Computation Graph

1. **Fprop:** visit nodes in topological sort order
   - Compute value of node given predecessors
2. **Bprop:**
   - initialize output gradient = 1
   - visit nodes in reverse order:
     Compute gradient wrt each node using gradient wrt successors

     {y₁, y₂, … y_n} = successors of *x*

     ∂z/∂x = Σ_{i=1}^{n} (∂z/∂y_i)(∂y_i/∂x)

Done correctly, big O() complexity of fprop and bprop is **the same**.

In general, our nets have regular layer-structure and so we can use matrices and
Jacobians…

The figure is a tangled directed acyclic graph of dark red nodes, inputs *x* at the
bottom and a single scalar output *z* at the top, with black arrows going up (fprop) and
dashed magenta arrows going down (bprop); an intermediate row is labelled y₁, y₂, …, y_n.

## Slide 79 — Automatic Differentiation

- The gradient computation can be automatically inferred from the symbolic expression of
  the fprop
- Each node type needs to know how to compute its output and how to compute the gradient
  wrt its inputs given the gradient wrt its output
- Modern DL frameworks (Tensorflow, PyTorch, etc.) do backpropagation for you but mainly
  leave layer/node writer to hand-calculate the local derivative

Two figures on the left: the same small graph drawn twice, once with solid black
forward arrows and once with dashed magenta arrows reversed.

## Slide 80 — Backprop Implementations

A code screenshot:

```python
class ComputationalGraph(object):
    #...
    def forward(inputs):
        # 1. [pass inputs to input gates...]
        # 2. forward the computational graph:
        for gate in self.graph.nodes_topologically_sorted():
            gate.forward()
        return loss # the final gate in the graph outputs the loss
    def backward():
        for gate in reversed(self.graph.nodes_topologically_sorted()):
            gate.backward() # little piece of backprop (chain rule applied)
        return inputs_gradients
```

## Slide 81 — Implementation: forward/backward API

A "*" node with scalar inputs *x* and *y* and output *z* (x, y, z are scalars), beside a
code screenshot:

```python
class MultiplyGate(object):
    def forward(x,y):
        z = x*y
        return z
    def backward(dz):
        # dx = ... #todo
        # dy = ... #todo
        return [dx, dy]
```

Two callouts: the argument `dz` of `backward` is annotated ∂L/∂z, and the returned
`[dx, dy]` is annotated ∂L/∂x.

## Slide 82 — Implementation: forward/backward API

The same node, with the code completed:

```python
class MultiplyGate(object):
    def forward(x,y):
        z = x*y
        self.x = x # must keep these around!
        self.y = y
        return z
    def backward(dz):
        dx = self.y * dz # [dz/dx * dL/dz]
        dy = self.x * dz # [dz/dy * dL/dz]
        return [dx, dy]
```

## Slide 83 — Manual Gradient checking: Numeric Gradient

- For small *h* (≈ 1e-4),

      f′(x) ≈ ( f(x + h) − f(x − h) ) / 2h

- Easy to implement correctly
- But approximate and **very** slow:
  - You have to recompute *f* for **every parameter** of our model
- Useful for checking your implementation
  - In the old days, we hand-wrote everything, doing this everywhere was the key test
  - Now much less needed; you can use it to check layers are correctly implemented

## Slide 84 — Summary

**We've mastered the core technology of neural nets!** 🎉🎉🎉

- **Backpropagation:** recursively (and hence efficiently) apply the chain rule along
  computation graph
  - [downstream gradient] = [upstream gradient] x [local gradient]
- **Forward pass:** compute results of operations and save intermediate values
- **Backward pass:** apply chain rule to compute gradients

## Slide 85 — Why learn all these details about gradients?

- **Modern deep learning frameworks compute gradients for you!**
  - Come to the PyTorch introduction this Friday!
- But why take a class on compilers or systems when they are implemented for you?
  - Understanding what is going on under the hood is useful!
- Backpropagation doesn't always work perfectly out of the box
  - Understanding why is crucial for debugging and improving models
  - See Karpathy article (in syllabus):
    https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
  - Example in future lecture: exploding and vanishing gradients
