# Lecture 3 — Backpropagation and Neural Networks

This is the lecture that explains how neural networks are actually trained, and it is
deliberately the most mathematical one in the course. It has two halves that arrive at
the same place from different directions: first computing gradients **by hand**, using
matrix calculus, and then computing them **algorithmically**, using the backpropagation
algorithm over a computation graph. Manning's framing is that backpropagation is not a
separate piece of magic — it is the chain rule, applied along a graph, with intermediate
results cached so nothing is computed twice (≈44:57). Along the way it fills in the
non-linearities that lecture 2 left as an unexplained *f*.

**Slide-by-slide text of this deck: [85 slides](../raw/slides/03-backpropagation-and-neural-networks.md)**
— cite slide numbers from there; the printed slide number equals the PDF page number.

Slides PDF: [Lecture 3 — neuralnets](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture03-neuralnets.pdf) ·
Notes: [2019 notes 03 — neural nets](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/readings/cs224n-2019-notes03-neuralnets.pdf) ·
[Gradient notes](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/readings/gradient-notes.pdf) ·
[Review of differential calculus](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/readings/review-differential-calculus.pdf) ·
[Full transcript](../raw/transcripts/03-backpropagation-and-neural-networks.md)

The plan on slide 10 is: introduction (10 mins), matrix calculus (35 mins),
backpropagation (35 mins). Manning tells students explicitly that **this is the one week
of the quarter to work through the readings** (slide 2), and that the material carries
directly into Assignment 2. Two labels in the deck are leftovers from an earlier year's
ordering — slide 10 is headed "Lecture 4" and slide 6 "7. Neural computation" — which is
worth knowing if you are matching slides to lectures.

## Where lecture 2 left off

A neural network is several logistic regressions run at the same time, stacked (slide 3).
The difference from the single logistic regression you meet in a statistics class is that
there, you choose the input features yourself; here, only the very end is pinned down by
the objective function, and everything in the middle is a chance for the network to learn
by itself what features would be useful to the layers downstream (≈3:56). Manning stops to
say this is worth sitting with, because that self-organization of intermediate
representations is what makes neural networks more powerful than other machine learning in
most circumstances (≈4:43).

A layer in matrix notation (slide 4) is

    z = Wx + b
    a = f(z)

where the activation *f* is applied element-wise, one scalar at a time
(≈7:01). The running example for the whole lecture is the named entity window
classifier from lecture 2 (slide 5): a five-word window, each word's embedding
concatenated into **x** ∈ ℝ^5d, one hidden layer, and a dot product **u**ᵀ**h** producing
a single score *s* that feeds a logistic function. Manning computes gradients of the
**score** rather than the loss throughout, because the loss's derivative is an Assignment
2 question he does not want to give away (slide 27, ≈31:45).

## Non-linearities: why, and which

The very first artificial neuron, McCulloch and Pitts' 1943 threshold unit, fired when
**Wx** exceeded a threshold θ (slide 6). That function is flat on both sides, so it has no
slope anywhere — and with no gradient there is nothing for gradient-based learning to
descend. Manning's image for gradient-based learning is skiing during spring break: you
work out where it is steeper and head down (≈9:19). Everything since has been about having
usable slopes.

Slide 7 lays out the family: the **logistic (sigmoid)**, still used where you want a
probability; **tanh**, which is literally a rescaled logistic — stretched by two and moved
down by one — despite the exponentials making that non-obvious (≈10:06); **hard tanh**,
a cheap piecewise-linear stand-in that avoids computing exponentials; **ReLU**,
max(*z*, 0); **Leaky** and **Parametric ReLU**, which give the negative half a small
slope; and **Swish** and **GELU**, common in recent Transformer models.

ReLU is the one to try first for deep networks. Manning notes it still feels slightly
perverse to him that it works, since a unit is dead any time it goes negative — but across
a network enough units stay alive, which gives a kind of specialization, and the constant
slope of one in the positive region produces very clean backward flow of gradients
(≈12:27). See [activation functions](activation-functions.md) for the full comparison.

The reason non-linearities are needed at all is compositional (slide 8): a matrix multiply
is a linear transform, and composing linear transforms gives another linear transform —
W₁W₂**x** = W**x** — so a deep network of pure matrix multiplies has exactly the
representational power of a single layer. With non-linearities in between, the network can
approximate arbitrarily complex functions. Manning adds an aside that is easy to miss:
*linear* neural networks give you no extra representational power but do have interesting
*learning* properties, and there is a body of theoretical work on them for that reason
(≈16:18).

## Matrix calculus

The article of faith for this half is: **multivariable calculus is just like
single-variable calculus if you use matrices** (slide 11, ≈20:08). Manning points at
Stanford's Math 51 textbook, whose Appendix G covers neural networks and the multivariable
chain rule — while noting it is page 697 of a dense book and he doubts anyone gets there
(slide 12, ≈19:21).

The ladder (slides 13–15):

- One input, one output: the gradient is the ordinary derivative. For *f*(*x*) = *x*³ it
  is 3*x*², and the derivative tells you how much the output moves when you nudge the
  input — at *x* = 4 the slope is 48, so a change of 0.01 in the input is magnified
  roughly 48 times (≈22:26).
- *n* inputs, one output: the gradient is a vector of partial derivatives, one per input.
- *n* inputs, *m* outputs: the gradient is the **Jacobian**, an *m* × *n* matrix with
  (∂**f**/∂**x**)_ij = ∂f_i/∂x_j. This is the shape a neural network layer has, which is
  why Jacobians are the working object here (≈24:43).

The chain rule (slide 16) generalizes exactly as the mantra promises: for one variable you
multiply derivatives, for many variables you **multiply Jacobians**.

Four Jacobians do nearly all the work (slides 17–24):

| Expression | Jacobian |
| --- | --- |
| **h** = *f*(**z**), element-wise | diag(*f*′(**z**)) |
| ∂/∂**x** (**Wx** + **b**) | **W** |
| ∂/∂**b** (**Wx** + **b**) | **I** |
| ∂/∂**u** (**u**ᵀ**h**) | **h**ᵀ |

The element-wise one is worth understanding rather than memorizing (slides 17–21): because
output *i* depends only on input *i*, every off-diagonal entry is zero — changing z_j
cannot move h_i — so the Jacobian is diagonal, with *f*′(z_i) down the diagonal
(≈27:51). See [matrix calculus](matrix-calculus.md) for the derivations worked out.

## The worked example, and δ

Slides 26–37 compute ∂s/∂**b** in four steps, and the steps are the method:

1. **Break the equations into simple pieces**, defining each variable and tracking its
   dimensionality: *s* = **u**ᵀ**h**, **h** = *f*(**z**), **z** = **Wx** + **b**.
2. **Apply the chain rule**: ∂s/∂**b** = ∂s/∂**h** · ∂**h**/∂**z** · ∂**z**/∂**b**.
3. **Write out the Jacobians** from the table above: **u**ᵀ · diag(*f*′(**z**)) · **I**.
4. **Simplify**: **u**ᵀ ⊙ *f*′(**z**), where ⊙ is the **Hadamard product**, element-wise
   multiplication of two vectors giving a vector — not a dot product, which would collapse
   them to a scalar (≈34:07).

Then comes the idea the rest of the lecture turns on. Computing ∂s/∂**W** needs
∂s/∂**h** · ∂**h**/∂**z** · ∂**z**/∂**W**, and the first two factors are *identical* to
the ones just computed for **b** (slides 38–39). So name that shared prefix

    δ = ∂s/∂h · ∂h/∂z = uᵀ ⊙ f′(z)

the **upstream gradient** or **error signal**, compute it once, and reuse it (slide 40).
For the bias, ∂**z**/∂**b** is the identity, so ∂s/∂**b** is just δ. For the weights,
∂s/∂**W** = δᵀ**x**ᵀ — the upstream gradient times the local input signal, an outer
product (slide 43).

Why the transposes? The honest answer Manning gives first is that it makes the dimensions
work out, which is a genuinely useful check on your own work (slide 44). The real reason
is that every input contributes to every output, so you want an outer product. Slide 45
derives it properly for a single weight: W_ij is used only in computing z_i, and within
that it multiplies only x_j, so ∂z_i/∂W_ij = x_j.

## Jacobian form vs the shape convention

This is the part Manning flags as a reliable source of confusion, and it is a genuine
collision between mathematics and engineering convenience (slides 41–42, 46–47).

Strictly, **W** ∈ ℝ^{n×m} has *nm* inputs and *s* is one output, so ∂s/∂**W** is a
1 × *nm* Jacobian — a very long row vector. That is correct and useless: the SGD update
subtracts a multiple of the gradient from the parameters, and you want the gradient shaped
like the parameters so you can just subtract (≈38:49). So deep learning leaves pure math
and adopts the **shape convention**: the gradient has the shape of the parameters, and
∂s/∂**W** is written *n* × *m*.

The two forms disagree, and each is better at one job — Jacobian form makes the chain rule
work, the shape convention makes implementing SGD easy. Two workable strategies (slide 47):
do all the math in Jacobian form and reshape at the very end, or follow the shape
convention throughout and transpose or reorder terms as the dimensions demand. **The
assignment expects answers in the shape convention.** A useful invariant when doing the
second: the error signal δ arriving at a hidden layer has the same dimensionality as that
hidden layer.

## Backpropagation

Software represents the network's equations as a **computation graph** (slide 49): source
nodes are inputs, interior nodes are operations, and edges carry each operation's result.
Running it left to right is **forward propagation** and just computes functions
(slides 50–51). Running it right to left, passing gradients back along the edges, is
**backpropagation** (slide 52), and it starts from ∂s/∂s = 1.

The rule at a single node is the whole algorithm (slides 53–56). A node receives an
**upstream gradient** from its output side and knows its own **local gradient** — the
derivative of its output with respect to its input. It passes back the **downstream
gradient**:

    [downstream gradient] = [upstream gradient] × [local gradient]

which is just the chain rule drawn as a picture. A node with several inputs has several
local gradients and sends a downstream gradient back along each edge (slides 57–58).

Slides 59–69 work this end to end on *f*(*x*, *y*, *z*) = (*x* + *y*)·max(*y*, *z*) at
*x* = 1, *y* = 2, *z* = 0. The forward pass gives *a* = 3, *b* = 2, *f* = 6. The backward
pass gives ∂f/∂x = 2, ∂f/∂y = 5, ∂f/∂z = 0. Manning then checks each answer numerically by
nudging an input, which is worth following because it makes the abstraction concrete:
moving *x* to 1.1 gives 3.1 × 2 = 6.2, a rise of 0.2 for a change of 0.1, so the gradient
is 2 (≈54:22); moving *y* to 2.1 gives 3.1 × 2.1 = 6.51, a rise of about 0.5, so the
gradient is 5 (≈55:12).

That *y* result carries the rule that trips people up: **gradients sum at outward
branches** (slides 70–71). Because *y* feeds both the `+` node and the `max` node, its
total gradient is the sum of what comes back along both paths — 2 + 3 = 5.

Three intuitions for what the common node types do to a gradient (slides 72–74):

- **`+` distributes** the upstream gradient unchanged to each summand.
- **`max` routes** it — all of it to the larger input, nothing to the others.
- **`*` switches** it — each input gets the upstream gradient times the *other* input's
  forward value.

## Doing it efficiently, and doing it automatically

The wrong way to backpropagate is to compute ∂s/∂**b**, then independently ∂s/∂**W**, and
so on, because the paths share a long prefix and you would walk it repeatedly
(slides 75–76). The right way is to compute all gradients in one backward sweep, which is
exactly the δ trick from the hand-derivation expressed as an algorithm (slide 77).

The general algorithm (slide 78) works on any directed acyclic graph, not just neat layer
structures: topologically sort the nodes and compute values forward; then initialize the
output gradient to 1 and visit nodes in reverse order, computing each node's gradient from
its successors' gradients. The correctness check to remember is that **forward and backward
passes have the same big-O complexity** — if your backward pass costs more than your
forward pass, you are recomputing something (≈1:02:09).

In principle the whole backward pass can be derived from the symbolic form of the forward
pass, which is **automatic differentiation** (slide 79). Theano, out of the Université de
Montréal, tried exactly that and did the entire thing for you. It did not fully win —
too heavyweight, or too awkward, or people just prefer writing their own Python — so modern
frameworks are in one sense *less* automated: PyTorch and TensorFlow manage the graph and
run both passes, but whoever writes a layer must supply its forward computation and its
**local gradient** by hand, and the framework trusts it (slides 80–82, ≈1:03:47). One
implementation detail matters: `forward` must stash its inputs on the object, because
`backward` receives only the upstream gradient and cannot produce the downstream one
without knowing the values it was evaluated at (≈1:07:37).

Since you are writing that local gradient yourself, you can get it wrong, so check it
numerically (slide 83):

    f′(x) ≈ ( f(x + h) − f(x − h) ) / 2h

with *h* ≈ 10⁻⁴. Use the **two-sided** estimate, not the one-sided one taught in calculus
class — it is much more accurate and stable (≈1:09:59). It is easy to get right and far too
slow to train with, since it costs a re-evaluation of *f* per parameter, but it is the
standard way to confirm a hand-written layer is correct.

## Why learn this if the framework does it

Manning's closing argument (slide 85): frameworks compute gradients for you, and that is
precisely why high school students can now do deep learning projects for science fairs. But
backpropagation does not always work out of the box, and understanding why is what lets you
debug and improve models — the example he promises for a later lecture is **exploding and
vanishing gradients** in recurrent networks (≈1:12:20). The syllabus reading on this point
is Karpathy's "Yes you should understand backprop".

## Related pages

- [Backpropagation](backpropagation.md) — the algorithm on its own, with the computation
  graph, the node rule, and the node intuitions.
- [Matrix calculus](matrix-calculus.md) — Jacobians, the chain rule, the four standard
  Jacobians, and the shape convention.
- [Activation functions](activation-functions.md) — the full family and why non-linearity
  is necessary.
- [Gradient descent](gradient-descent.md) — the update rule these gradients feed, from
  lectures 1 and 2.
- [Softmax, the logistic function, and cross-entropy](softmax-and-cross-entropy.md) — the
  loss whose derivative Manning deliberately leaves as an assignment question.
- [Lecture 4 — Dependency Parsing](04-dependency-parsing.md) — where this machinery is put
  to work in a real classifier.
