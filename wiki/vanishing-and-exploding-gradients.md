# Vanishing and exploding gradients

The problem that makes vanilla [recurrent neural networks](recurrent-neural-networks.md)
much weaker in practice than they look on paper, and the reason the [LSTM](lstm.md) exists.
Covered in [lecture 5](05-recurrent-neural-networks.md) (slides 51–63) and re-derived at the
start of [lecture 6](06-sequence-to-sequence-models.md) (slides 6–18), which then generalizes
it beyond RNNs.

## The intuition

Lecture 5's slides 52–56 build it one chain-rule factor at a time. To get the gradient of a
loss at step 4 with respect to a hidden state at step 1, you multiply Jacobians along the
chain:

$$\frac{\partial J^{(4)}}{\partial h^{(1)}} = \frac{\partial h^{(2)}}{\partial h^{(1)}} \times \frac{\partial h^{(3)}}{\partial h^{(2)}} \times \frac{\partial h^{(4)}}{\partial h^{(3)}} \times \frac{\partial J^{(4)}}{\partial h^{(4)}}$$

where $h^{(t)}$ is the hidden state at step $t$ and $J^{(t)}$ the loss there.
Slide 56 asks the question that matters: *what happens if these are small?*

> **Vanishing gradient problem:** when these factors are small, the gradient signal gets
> smaller and smaller as it backpropagates further.

## The proof sketch

Slides 57–58 make it precise in the linear case. Recall
$h^{(t)} = \sigma(W_h h^{(t-1)} + W_x x^{(t)} + b_1)$, with $W_h$ the recurrent weight
matrix and $W_x$ the input weight matrix. If $\sigma$ were the identity, then by the chain
rule

$$\frac{\partial h^{(t)}}{\partial h^{(t-1)}} = \operatorname{diag}\left(\sigma'(\cdot)\right) W_h = I W_h = W_h$$

exactly. So the gradient of a loss at step $i$ with respect to a hidden state at an earlier
step $j$, writing $\ell = i - j$ for the distance between them, carries a factor of
$W_h^{\ell}$:

$$\frac{\partial J^{(i)}(\theta)}{\partial h^{(j)}} = \frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} \prod_{j < t \le i} \frac{\partial h^{(t)}}{\partial h^{(t-1)}} = \frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} W_h^{\ell}$$

What is wrong with $W_h^{\ell}$? Write it in the eigenvector basis of $W_h$, with
eigenvalues $\lambda_1, \dots, \lambda_n$ and eigenvectors $q_1, \dots, q_n$. If all
eigenvalues are less than 1 — a **sufficient but not necessary** condition, as the slide
notes —

$$\frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} W_h^{\ell} = \sum_{i=1}^{n} c_i \lambda_i^{\ell} q_i \approx 0 \quad \text{for large } \ell$$

because $\lambda_i^{\ell} \to 0$ (the $c_i$ are the coefficients of the upstream gradient in
that basis; the slide reuses $i$ as the summation index). For nonlinear activations the
argument is essentially the same, with the condition $\lambda_i < \gamma$ for some $\gamma$
depending on dimensionality and $\sigma$. Source: **Pascanu et al. (2013)**, "On the
difficulty of training recurrent neural networks".

Manning's version in lecture 6 (≈13:19) is the compact statement: either all eigenvalues are
below one and the gradient shrinks as you go further back, or some are above one and it
grows — and unless things sit precisely at a largest eigenvalue of about one, you get one or
the other, and both are bad.

## Why the vanishing case is a problem

You might argue it should not be (lecture 6, ≈14:06): all else being equal the closest words
*are* the most relevant, so updating mostly with respect to nearby losses is not obviously
wrong. Manning concedes the point and then says the effect is far too severe. Slide 59's
statement:

> Gradient signal from far away is lost because it's much smaller than gradient signal from
> close by. So model weights are updated only with respect to **near effects**, not
> **long-term effects**.

Slide 60's example makes it concrete:

> *When she tried to print her tickets, she found that the printer was out of toner. She went
> to the stationery store to buy more toner. It was very overpriced. After installing the
> toner into the printer, she finally printed her ________*

A human predicts "tickets" with near certainty. The model must connect "tickets" at step 7
with the target at the end — about 20-odd words later. If the gradient between those
positions is too small, the dependency is never **learned**, and so at test time the model
cannot predict long-distance dependencies at all. From the local context alone, "she finally
printed her ___" could be her paper, her invitation, her novel (lecture 6, ≈15:41).

**The number.** Lecture 6's slide 15 adds a bullet lecture 5's version does not have:

> In practice, a simple RNN will only condition ~7 tokens back [vague rule-of-thumb]

Manning uses this to deflate the whole architecture. [*n*-gram
models](n-gram-language-models.md) maxed out at 5-grams because of sparsity and storage. "In
theory we've now got a much better solution; in practice, because of vanishing gradients,
we're only kind of getting the equivalent of 8-grams. So we haven't made that much progress,
it feels like" (≈17:13).

## Exploding gradients, and clipping

The mirror image (lecture 5, slides 61–62), and by comparison an easy problem. If the
gradient becomes too large, the SGD step
$\theta^{\text{new}} = \theta^{\text{old}} - \alpha \nabla_{\theta} J(\theta)$ becomes too
large, and you land in a weird, bad, high-loss part of parameter space. Manning's line for
it: *"You think you've found a hill to climb, but suddenly you're in Iowa."* Worst case you
get **Inf** or **NaN** in the network and have to restart from an earlier checkpoint.

**Gradient clipping** is the fix: if the norm of the gradient exceeds a threshold, scale it
down before applying the update. Slide 62's pseudo-code, with $E$ the error and $\hat{g}$ the
gradient about to be applied:

$$
\begin{aligned}
& \hat{g} \leftarrow \frac{\partial E}{\partial \theta} \\
& \textbf{if } \lVert \hat{g} \rVert \ge \text{threshold} \textbf{ then} \\
& \qquad \hat{g} \leftarrow \frac{\text{threshold}}{\lVert \hat{g} \rVert} \hat{g} \\
& \textbf{end if}
\end{aligned}
$$

The intuition is *same direction, smaller step*. Thresholds are typically around 5, 10 or 20
for the gradient norm (lecture 6, ≈19:33). Manning is candid that this "isn't high-falutin
math, really — it's a crude hack" — but it works, it is often essential, and remembering to
do it is one of the lecture's four named takeaways (lecture 6, slide 56). See
[gradient descent](gradient-descent.md).

## The architectural diagnosis

Slide 63 states what has to change:

> The main problem is that *it's too difficult for the RNN to learn to preserve information
> over many timesteps*. In a vanilla RNN, the hidden state is constantly being **rewritten**.

$$h^{(t)} = \sigma\left(W_h h^{(t-1)} + W_x x^{(t)} + b\right)$$

Manning expands on this in lecture 6 (≈20:20): you take the previous hidden vector, multiply
it by a matrix that changes it entirely, and add in the input. If what you want is "there is
useful stuff in $h^{(t-1)}$, please keep it around for a while", then learning weights that mostly
preserve what was there is not an obvious thing for gradient descent to find.

Two families of fix are named:

1. **An RNN with separate memory that is added to** — the [LSTM](lstm.md).
2. **More direct and linear pass-through connections** — attention and residual connections,
   both developed in later lectures.

## Why the LSTM fixes it

The short version, spelled out under [LSTM](lstm.md): the cell state is updated by
$c^{(t)} = f^{(t)} \odot c^{(t-1)} + i^{(t)} \odot \tilde{c}^{(t)}$ — **additively**, where
$f^{(t)}$ is the forget gate, $i^{(t)}$ the input gate and $\tilde{c}^{(t)}$ the candidate
new cell content. In a simple RNN the next hidden state
comes out of multiplicative operations, which makes preserving information hard; **the plus
sign is the secret** (lecture 6, slide 24 and ≈41:19). Set the forget gate to 1 and the input
gate to 0 for some cell dimension and that dimension is preserved indefinitely. The result:
about **100 timesteps** of effective memory rather than about 7 (slide 25).

Asked whether the additive path helps with *exploding* gradients too, Manning says yes — for
the same reason, because you are not doing a long sequence of multiplies (lecture 6, ≈43:39).

## Not just an RNN problem

Lecture 6's slides 26–27 generalize the whole thing, and this is the part that connects the
lecture to computer vision.

Any very deep network suffers this, feed-forward and convolutional included, because of the
chain rule and the choice of nonlinearity. If gradients vanish in a deep network, the lower
layers get very little signal, so their parameters barely update, so they learn nothing, so
the network does not work — which is a large part of why deep networks were stuck in the
early 2000s (≈45:10). This is the same story lecture 5 tells about the fifteen-year gap
(slide 6).

The fix is the same idea applied vertically: **add more direct connections** so the gradient
can flow.

- **Residual connections / "ResNet"**, also called **skip-connections** (He et al. 2015). A
  block computes $\mathcal{F}(x)$, and an identity connection carries $x$ around it to be
  added back: $\mathcal{F}(x) + x$. The identity connection **preserves information by default**, and this is what
  made deep computer vision models much more learnable than plain networks (≈46:44).
- **Dense connections / "DenseNet"** (Huang et al. 2017). Connect each layer directly to all
  later layers — every layer takes all preceding feature maps as input.
- **Highway connections / "HighwayNet"** (Srivastava et al. 2015). Like residual connections,
  but the balance between the identity path and the transformation is controlled by a
  **dynamic gate** — explicitly inspired by LSTMs, applied to deep feed-forward and
  convolutional networks.

Slide 27's conclusion is the sentence to keep:

> Though vanishing/exploding gradients are a general problem, **RNNs are particularly
> unstable** due to the repeated multiplication by the **same** weight matrix
> [Bengio et al, 1994].

That "same" is the crux. In a deep feed-forward network the layers have different parameters,
so it is not quite raising one matrix to a power; in an RNN it exactly is, which is why the
eigenvalue argument bites so hard (≈44:25).

## Related pages

- [LSTM](lstm.md) — the fix, and why the plus sign is where it lives.
- [Recurrent neural networks](recurrent-neural-networks.md) — the architecture with the
  problem.
- [Backpropagation](backpropagation.md) — the chain-rule machinery being unrolled here.
- [Gradient descent](gradient-descent.md) — the update step that clipping modifies.
- [Activation functions](activation-functions.md) — the $\sigma$ whose derivative appears in
  every Jacobian factor.
- [Lecture 5 — Recurrent Neural Networks](05-recurrent-neural-networks.md)
- [Lecture 6 — Sequence to Sequence Models](06-sequence-to-sequence-models.md)
