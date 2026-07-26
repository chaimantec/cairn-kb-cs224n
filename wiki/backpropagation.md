# Backpropagation

Backpropagation is how the gradients that train a neural network are computed. Manning's
deflationary summary is worth holding onto: it is **only two things** — you apply the chain
rule, and you store intermediate results so you never recompute the same quantity twice
(≈44:57 in [lecture 3](03-backpropagation-and-neural-networks.md)). Its invention made
people famous because it gave an effective learning algorithm, but there is no third idea
hiding in it.

Primary source: lecture 3, slides 48–85
([slide text](../raw/slides/03-backpropagation-and-neural-networks.md)).

## The computation graph

Software represents a network's equations as a graph (slide 49): **source nodes** are
inputs, **interior nodes** are operations, and **edges** carry the result of each operation
to the next. The lecture's running network, with input $x$, weight matrix $W$, bias $b$,
non-linearity $f$, hidden layer $h$ and scalar score $s$,

$$s = u^{\top} h, \qquad h = f(z), \qquad z = Wx + b$$

becomes the chain $x \to [\cdot] \to Wx \to [+] \to z \to [f] \to h \to [\cdot] \to s$, with
$W$, $b$ and $u$ entering from below (slide 50).

Running the graph left to right is **forward propagation**: it computes the function, and
it saves the intermediate values (slide 51). Running it right to left, passing gradients
back along the edges, is **backpropagation** (slide 52). The backward pass is seeded with
$\partial s / \partial s = 1$.

## The rule at a single node

This is the entire algorithm, and everything else is bookkeeping (slides 53–56). A node
computing $h = f(z)$ has three gradients associated with it:

- the **upstream gradient** $\dfrac{\partial s}{\partial h}$, handed to it from the output
  side;
- its **local gradient** $\dfrac{\partial h}{\partial z}$, the derivative of its own output
  with respect to its own input — this is the only part that knows what the node does;
- the **downstream gradient** $\dfrac{\partial s}{\partial z}$, which it computes and passes
  back.

The relation between them is the chain rule drawn as a picture:

$$\underbrace{\frac{\partial s}{\partial z}}_{\text{downstream}} = \underbrace{\frac{\partial s}{\partial h}}_{\text{upstream}} \times \underbrace{\frac{\partial h}{\partial z}}_{\text{local}}$$

A node with several inputs, such as $z = Wx$, simply has one local gradient per input
and sends one downstream gradient back along each incoming edge (slides 57–58).

## Gradients sum at outward branches

The rule people get wrong. If a variable feeds more than one downstream node, gradient
flows back to it along every one of those paths, and its total gradient is their **sum**.
For a variable $y$ feeding two nodes $a$ and $b$ (slides 70–71):

$$\frac{\partial f}{\partial y} = \frac{\partial f}{\partial a}\frac{\partial a}{\partial y} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial y}$$

In the lecture's worked example (slides 59–69),
$f(x, y, z) = (x + y) \cdot \max(y, z)$ at $x = 1$, $y = 2$, $z = 0$, the variable $y$ feeds
both the `+` node and the `max` node. It receives 2 back from one and 3 from the other, so
$\partial f/\partial y = 5$ — while $\partial f/\partial x = 2$ and
$\partial f/\partial z = 0$. Manning verifies this numerically: nudging $y$ to 2.1 gives
$3.1 \times 2.1 = 6.51$, up about 0.5 from 6, which is a gradient of 5 (≈55:12).

## What the common nodes do to a gradient

Three intuitions worth memorizing, because they let you sanity-check a backward pass by
eye (slides 72–74):

| Node | Effect on the upstream gradient |
| --- | --- |
| `+` | **distributes** it — each summand gets the same gradient, unchanged |
| `max` | **routes** it — all of it goes to the largest input, zero to the others |
| `*` | **switches** it — each input gets the gradient times the *other* input's forward value |

## Computing all gradients at once

The naive approach — compute $\partial s/\partial b$, then separately $\partial s/\partial W$,
then $\partial s/\partial x$ — walks the shared part of the graph once per parameter
(slides 75–76). The correct approach is a single backward sweep that computes every gradient
together (slide 77). This is exactly the $\delta$ trick from the by-hand derivation in the
same lecture:

$$\delta = \frac{\partial s}{\partial h} \cdot \frac{\partial h}{\partial z}$$

is the shared **upstream gradient**, or *error signal*, computed once and reused for both
$\partial s/\partial b$ and $\partial s/\partial W$ (slide 40). See
[matrix calculus](matrix-calculus.md).

## The general algorithm

Backpropagation is not restricted to tidy layer structures; it works on any directed
acyclic graph (slide 78):

1. **Fprop** — visit nodes in topological sort order, computing each node's value from its
   predecessors.
2. **Bprop** — initialize the output gradient to 1, then visit nodes in reverse topological
   order, computing each node's gradient from its successors' gradients:

$$\frac{\partial z}{\partial x} = \sum_{i=1}^{n} \frac{\partial z}{\partial y_i} \frac{\partial y_i}{\partial x} \qquad \text{for } \{y_1, \dots, y_n\} \text{ the successors of } x$$

The correctness invariant to remember: done correctly, **fprop and bprop have the same
big-O complexity**. If your backward pass costs asymptotically more than your forward pass,
you are recomputing work you already did (≈1:02:09).

## Automatic differentiation, and what frameworks actually do

In principle the backward pass can be derived from the symbolic form of the forward pass —
**automatic differentiation** (slide 79). **Theano**, out of the Université de Montréal,
tried exactly this: you stated the forward computation symbolically and it produced the
backward pass for you. It did not fully win, whether because it was too heavyweight, or
awkward for unusual cases, or because people prefer writing their own Python (≈1:03:47).

So modern frameworks are in one sense *less* automated. PyTorch and TensorFlow manage the
computation graph and run both passes, but whoever writes a layer or node must supply its
forward computation **and its local gradient**, and the framework calls that and assumes it
is correct (slides 79–80). One implementation detail follows from this (slides 81–82): the
`forward` method must stash its inputs on the object —

```python
def forward(x, y):
    z = x*y
    self.x = x   # must keep these around!
    self.y = y
    return z

def backward(dz):
    dx = self.y * dz
    dy = self.x * dz
    return [dx, dy]
```

— because `backward` receives only the upstream gradient and cannot produce a downstream
gradient without knowing the values the function was evaluated at (≈1:07:37).

## Gradient checking

Since you write the local gradient by hand, you can get it wrong. The check is a numeric
gradient (slide 83):

$$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}$$

with the perturbation $h$ around $10^{-4}$ — there is no magic value, it depends on the
function. Compare against what your backward pass produces and expect agreement to within
about $10^{-2}$.

Two points Manning stresses. Use the **two-sided** estimate, not the one-sided
$\big(f(x+h) - f(x)\big)/h$ taught in calculus classes: evaluating equally on both sides
is much more accurate and stable (≈1:09:59). And this is a *check*, not a training method —
it costs a full re-evaluation of $f$ for **every parameter**, which is precisely the cost
backpropagation exists to avoid. In the days before frameworks, when everything was written
by hand, doing this everywhere was the key test; now it is mainly for confirming a
newly-written layer is correct.

## Why understand it if the framework does it

Frameworks compute gradients for you, which is why deep learning projects turn up at high
school science fairs (slide 85). Manning's argument for learning it anyway is that
backpropagation does not always work out of the box, and understanding why is what lets you
debug and improve models. The concrete example he promises for a later lecture is
**exploding and vanishing gradients** in recurrent networks (≈1:12:20). The syllabus
recommends Karpathy's "Yes you should understand backprop".

## Backpropagation through time

The RNN case (lecture 5, slides 41–43) is where the outward-branch rule above stops being a
detail and becomes the whole algorithm. A [recurrent neural
network](recurrent-neural-networks.md) applies the *same* weight matrix $W_h$ at every
timestep, so the question is how to differentiate with respect to a **repeated** weight:

> The gradient with respect to a repeated weight is the **sum** of the gradient with respect
> to each time it appears.

$$\frac{\partial J^{(t)}}{\partial W_h} = \sum_{i=1}^{t} \left. \frac{\partial J^{(t)}}{\partial W_h} \right|_{(i)}$$

where $J^{(t)}$ is the loss at timestep $t$ and $|_{(i)}$ marks the contribution from the
$i$-th application of $W_h$. The justification is exactly "gradients sum at outward
branches". Think of the single $W_h$ as being copied by an identity function into
$W_h^{(1)}, W_h^{(2)}, \dots$ at each timestep. Identity copies
have partial derivative 1 with respect to each other, so applying the multivariable chain
rule leaves each term multiplied by 1, and what remains is a plain sum (≈1:06:36). Slide 43
writes the cancellation out.

The algorithm — backpropagate over timesteps $i = t, \dots, 0$, summing gradients as you go —
is **backpropagation through time** (Werbos 1988).

In practice it is often **truncated** after around 20 timesteps for training efficiency
(slide 43). The asymmetry is worth noting: the *forward* pass still uses the full context;
only the backward pass is cut short (≈1:08:09).

Over long sequences this repeated multiplication is also where backpropagation breaks down —
see [vanishing and exploding gradients](vanishing-and-exploding-gradients.md), and note the
conclusion that RNNs are particularly unstable precisely because the repeated matrix is the
**same** one every step.

## Where it appears in this course

- [Lecture 3](03-backpropagation-and-neural-networks.md) — introduced and derived in full.
- [Lecture 4](04-dependency-parsing.md) — used in practice: the log loss of the neural
  dependency parser's softmax is backpropagated all the way into the word, POS-tag and
  dependency-label embeddings (slide 44).
- [Lecture 5](05-recurrent-neural-networks.md) — backpropagation through time, and the
  vanishing/exploding gradient analysis of what the repeated matrix does to the signal.
- [Lecture 6](06-sequence-to-sequence-models.md) — backpropagating end-to-end through a
  [seq2seq model](seq2seq-and-encoder-decoder.md), from the decoder's losses right back
  through the encoder.

## Related pages

- [Matrix calculus](matrix-calculus.md) — the by-hand version of the same computation, and
  where $\delta$ comes from.
- [Gradient descent](gradient-descent.md) — what consumes the gradients once computed.
- [Activation functions](activation-functions.md) — why ReLU's constant slope of one gives
  such clean gradient backflow.
- [Vanishing and exploding gradients](vanishing-and-exploding-gradients.md) — what happens to
  these gradients over many steps, and how it is fixed.
- [Recurrent neural networks](recurrent-neural-networks.md) — the architecture BPTT unrolls.
