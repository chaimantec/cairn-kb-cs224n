# Matrix calculus for neural networks

The mathematics you need to differentiate a neural network by hand. The mantra Manning
gives, and repeats whenever a step looks hard, is that **multivariable calculus is just
like single-variable calculus if you use matrices** (slide 11, ≈20:08 in
[lecture 3](03-backpropagation-and-neural-networks.md)). Several of the results below are
easiest to get by trusting that and checking the shapes afterwards.

Primary source: lecture 3, slides 11–47
([slide text](../raw/slides/03-backpropagation-and-neural-networks.md)). The course's own
further reading is the [gradient notes](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/readings/gradient-notes.pdf)
and the [review of differential calculus](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/readings/review-differential-calculus.pdf);
Manning also points at Stanford's Math 51 textbook, whose Appendix G covers neural networks
and the multivariable chain rule — while noting it is page 697 of a very dense book
(slide 12).

## The ladder: derivative → gradient → Jacobian

**One input, one output** (slide 13). The gradient is the ordinary derivative. For
*f*(*x*) = *x*³ it is 3*x*², and the useful reading of it is "how much does the output move
if I nudge the input?" At *x* = 1 the slope is 3, so 1.01³ ≈ 1.03; at *x* = 4 the slope is
48, so a nudge of 0.01 is magnified about 48 times.

***n* inputs, one output** (slide 14). The gradient is a vector of partial derivatives, one
per input:

    ∂f/∂x = [ ∂f/∂x₁, ∂f/∂x₂, …, ∂f/∂xₙ ]

***n* inputs, *m* outputs** (slide 15). The gradient becomes the **Jacobian**, an
*m* × *n* matrix:

    (∂f/∂x)ᵢⱼ = ∂fᵢ/∂xⱼ

This is the shape a neural network layer has — *n* inputs, *m* outputs — which is why
Jacobians are the working object for the rest of the derivation (≈24:43). (Manning notes in
passing that the mathematician it is named after was German, so it "should really be
Yacobi", but nobody says that in the US.)

## The chain rule

Neural networks are compositions of functions, so gradients come from the chain rule
(slide 16). It generalizes exactly as the mantra promises:

- **One variable: multiply derivatives.** With *z* = 3*y* and *y* = *x*²,
  d*z*/d*x* = (d*z*/d*y*)(d*y*/d*x*) = 3 · 2*x* = 6*x*.
- **Many variables: multiply Jacobians.** With **h** = *f*(**z**) and **z** = **Wx** + **b**,
  ∂**h**/∂**x** = (∂**h**/∂**z**)(∂**z**/∂**x**).

## The four Jacobians you actually need

| Expression | Jacobian | Slide |
| --- | --- | --- |
| **h** = *f*(**z**), *f* applied element-wise | diag(*f*′(**z**)) | 21 |
| ∂/∂**x** (**Wx** + **b**) | **W** | 22 |
| ∂/∂**b** (**Wx** + **b**) | **I** (identity) | 23 |
| ∂/∂**u** (**u**ᵀ**h**) | **h**ᵀ | 24 |

Two of these are the mantra doing its work: in one variable the derivative of *wx* + *b*
with respect to *x* is *w*, and with respect to *b* is 1 — so in matrix calculus they are
**W** and **I**. Manning's advice is not to overthink it (≈29:24). The identity rather than
a scalar 1 reflects that **b** is a vector.

The element-wise one is worth deriving rather than memorizing (slides 17–21). The layer has
*n* inputs and *n* outputs, so the Jacobian is *n* × *n*. By definition its (*i*, *j*) entry
is ∂h_i/∂z_j. But h_i = *f*(z_i) depends only on z_i, so when *i* ≠ *j*, changing z_j does
not move h_i at all and the entry is zero. Everything off the diagonal vanishes, and the
diagonal holds the ordinary one-variable derivative *f*′(z_i) — hence diag(*f*′(**z**))
(≈27:51).

A footnote on the fourth (slide 24): **h**ᵀ is the *correct Jacobian*. Under the shape
convention discussed below the answer would be written **h**.

## The worked derivation, and δ

Slides 26–37 compute ∂s/∂**b** for the network *s* = **u**ᵀ**h**, **h** = *f*(**z**),
**z** = **Wx** + **b**. The four steps are the general method:

1. **Break the equations into simple pieces.** Define each variable and *keep track of its
   dimensionality* — slide 28's only instruction.
2. **Apply the chain rule:** ∂s/∂**b** = ∂s/∂**h** · ∂**h**/∂**z** · ∂**z**/∂**b**.
3. **Substitute the Jacobians:** **u**ᵀ · diag(*f*′(**z**)) · **I**.
4. **Simplify:** **u**ᵀ ⊙ *f*′(**z**).

Here ⊙ is the **Hadamard product** — element-wise multiplication of two vectors giving
another vector of the same size. Manning flags it because it rarely appears in a regular
math course but constantly in neural networks, and because it is *not* a dot product, which
would collapse the two vectors to a scalar (≈34:07).

Now compute a second gradient and the point of the exercise appears (slides 38–40):

    ∂s/∂W = ∂s/∂h · ∂h/∂z · ∂z/∂W
    ∂s/∂b = ∂s/∂h · ∂h/∂z · ∂z/∂b

The first two factors are identical. Name them

    δ = ∂s/∂h · ∂h/∂z = uᵀ ⊙ f′(z)

the **upstream gradient** or **error signal**. Compute it once and reuse it. Then
∂s/∂**b** = δ (since ∂**z**/∂**b** = **I**), and ∂s/∂**W** = δᵀ**x**ᵀ (slide 43) — the
upstream gradient times the local input signal, which is an outer product.

This reuse is the same idea that [backpropagation](backpropagation.md) implements as an
algorithm; the hand derivation is where it first shows up.

### Why the transposes

Manning gives the honest answer first: δᵀ**x**ᵀ makes the dimensions work out, and checking
dimensions is a genuinely useful way to catch your own errors (slide 44). [*n* × *m*] on the
left, [*n* × 1][1 × *m*] on the right. The real reason is that every input contributes to
every output, so what you want is an **outer product**.

Slide 45 derives it properly for a single weight. W_ij is used only in computing z_i — W₂₃
contributes to z₂ and not z₁ — and within that sum it multiplies only x_j. So

    ∂zᵢ/∂Wᵢⱼ = ∂/∂Wᵢⱼ Σₖ Wᵢₖxₖ = xⱼ

## Jacobian form vs the shape convention

The collision between mathematics and engineering convenience, and the thing Manning warns
will confuse people (slides 41–42, 46–47).

Strictly, **W** ∈ ℝ^{n×m} contributes *nm* inputs and *s* is a single output, so ∂s/∂**W**
is a **1 × *nm* Jacobian** — one very long row vector. That is mathematically right and
practically useless, because the SGD update

    θ_new = θ_old − α ∇_θ J(θ)

wants to subtract the gradient from the parameters, and the parameters are a matrix. Turning
them into "a God Almighty row vector" to do it is inconvenient (≈38:49).

So deep learning adopts the **shape convention**: *the shape of the gradient is the shape of
the parameters*. ∂s/∂**W** is written as an *n* × *m* matrix of ∂s/∂W_ij entries (slide 42),
and the update is a plain element-wise subtraction.

The two conventions genuinely disagree, and each is better at one job: **Jacobian form makes
the chain rule work**; **the shape convention makes implementing SGD easy**. Slide 46 gives
the concrete clash — ∂s/∂**b** = **h**ᵀ ⊙ *f*′(**z**) is a row vector in Jacobian form, but
**b** is a column vector, so the shape convention wants a column.

Two workable strategies (slide 47):

1. Work in **Jacobian form** throughout, then transpose/reshape at the very end to satisfy
   the shape convention. (This is what the lecture does — the final answer for ∂s/∂**b**
   becomes δᵀ.)
2. Follow the **shape convention** at every stage, transposing and reordering terms as the
   dimensions demand.

**Assignment 2 expects answers in the shape convention**, though Jacobian form is often the
easier route to computing them. A useful invariant when working in shape convention: the
error message δ arriving at a hidden layer has the same dimensionality as that hidden layer.

## Related pages

- [Backpropagation](backpropagation.md) — the same computation as an algorithm over a graph.
- [Gradient descent](gradient-descent.md) — the update rule these gradients feed, and the
  "observed minus expected" gradient from lecture 1.
- [Lecture 3](03-backpropagation-and-neural-networks.md) — the full lecture this comes from.
