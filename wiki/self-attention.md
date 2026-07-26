# Self-attention

[Attention](attention.md) applied *within* a single sequence rather than from a decoder to
an encoder: every word attends to every other word in the same input (or output). This is
the core building block of the [Transformer](transformer.md), introduced in
[lecture 8](08-self-attention-and-transformers.md) (slides 22–50) as the direct replacement
for recurrence.

## Why move beyond recurrence

The Transformer authors had three design goals (slide 17): minimize computational
complexity per layer, minimize the path length between any pair of words (to make
long-range dependencies learnable), and maximize how much computation can be parallelized.
Recurrent networks — even with attention bolted on — fail the last two:

- **Linear interaction distance.** An RNN encodes useful *linear locality* (nearby words
  like "tasty pizza" do often affect each other), but two words that are far apart in the
  sequence, like "chef" and "was" in *the chef who went to the stores … was*, need
  $O(\text{sequence length})$ RNN steps to interact. Gradients have to propagate through
  every intervening step, which is exactly the setup for the
  [vanishing gradient](vanishing-and-exploding-gradients.md) problem (lecture 8, ≈4:46–7:03).
- **Non-parallelizability.** An RNN's forward and backward passes have $O(\text{sequence
  length})$ *unparallelizable* operations — you can't compute the hidden state at position 5
  before you've computed it at positions 1–4, no matter how many GPU cores you have. This
  gets worse as sequences grow, and increasingly limits how much data you can train on
  (≈7:03–8:37).

Self-attention solves both at once: any word can attend to any other word regardless of
distance, so the maximum interaction distance is $O(1)$, and — because every word's
attention computation is independent of every other word's — the number of unparallelizable
operations does not grow with sequence length either (≈9:25, ≈10:57).

## The mechanism: query, key, value

Think of attention as a *fuzzy* or approximate hashtable (slide 28): a real hashtable maps
each query (hash) to exactly one key-value pair; self-attention lets a query match *every*
key to varying degrees, and returns a weighted sum of the values, weighted by how well the
query matched each key.

Given a sequence of words $w_1, \dots, w_n$ and a shared embedding matrix $E$ (lecture 8,
≈14:03–16:21, slide 29):

1. **Compute query, key, and value for every word.** Three learned weight matrices
   $Q, K, V \in \mathbb{R}^{d \times d}$ each transform a word's embedding $x_i$ into a
   differently-purposed vector:
$$q_i = Qx_i \qquad k_i = Kx_i \qquad v_i = Vx_i$$

   Every word plays all three roles at once — it offers a key and a value to be looked up
   by others, and issues a query of its own.
2. **Score all pairs.** Dot-product the query for word $i$ against the key for word $j$:

$$e_{ij} = q_i \cdot k_j$$

   computed for every pair $(i, j)$ in the sequence at once.
3. **Softmax to a distribution.** Normalize each word's scores over all the words it could
   attend to:

$$\alpha_{ij} = \operatorname{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}$$

4. **Weighted sum of values.**

$$\text{output}_i = \sum_j \alpha_{ij} v_j$$

**Why learn separate $Q$ and $K$ matrices**, when $q_i^\top k_j$ is really one matrix in the
middle? A student asks this directly (≈18:41): using $Q$ and $K$ separately is a low-rank
factorization of that combined matrix, done for computational efficiency — and it also has
a real expressive benefit, since it means the network is not forced to look at itself: if
$Q$ and $K$ were both the identity, $e_{ii}$ would just be a word dotted with itself, which
tends to be large. Because $Q$ and $K$ are learned independently, the model can instead
learn *whether* a word should attend to itself, rather than having that forced on it
(≈19:27–20:14).

**Vectorized form** (slide 30). Stacking all the word embeddings into a matrix
$X \in \mathbb{R}^{n \times d}$:
$$Q = XW^Q \qquad K = XW^K \qquad V = XW^V$$
$$\text{Output} = \operatorname{softmax}\left(QK^\top\right)V$$
one big matrix multiply for the whole sequence, rather than a for-loop over word pairs —
this is what makes self-attention GPU-friendly.

## Three things self-attention needs before it can replace an RNN

Self-attention as defined above has three real problems, which lecture 8 fixes one at a
time before assembling the "minimal self-attention building block" (slide 26, 31, ≈21:01).

### 1. No notion of sequence order

Self-attention is an operation on a *set* of vectors — nowhere does a word's position enter
the computation. "Zuko made his uncle tea" and "his uncle made Zuko tea" get **identical**
representations, which is clearly wrong (≈21:46–22:32).

The fix is to inject **position vectors** $p_i \in \mathbb{R}^d$, one per sequence index,
and add them to the input embeddings before computing queries, keys and values — only once,
at the very first layer (slide 41):
$$v_i = \tilde{v}_i + p_i \qquad q_i = \tilde{q}_i + p_i \qquad k_i = \tilde{k}_i + p_i$$

Two ways to construct $p_i$ (slides 42, 25–27):

- **Sinusoidal position representations** (the original Transformer choice) — concatenate
  sine and cosine functions of varying periods:
$$p_i = \begin{pmatrix} \sin(i / 10000^{2 \cdot 1/d}) \\ \cos(i / 10000^{2 \cdot 1/d}) \\ \vdots \\ \sin(i / 10000^{2 \cdot (d/2)/d}) \\ \cos(i / 10000^{2 \cdot (d/2)/d}) \end{pmatrix}$$

  The pitch was that periodicity would let position matter less in absolute terms and let
  the model extrapolate to longer sequences than it was trained on — but in practice that
  extrapolation doesn't really work (≈24:48–25:33).
- **Learned absolute position embeddings** — just learn a $d \times n_{\max}$ matrix as a
  parameter, one column per position, the same way every other parameter is learned. Simple
  and "feels more deep-learning," but it hard-caps the model at $n_{\max}$ words: pass it
  anything longer and it crashes (≈25:33–26:19). This is a large part of why context-length
  limits are still a real constraint even in the largest language models — "can I fit this
  prompt into ChatGPT" is not a solved problem (≈27:51).

More flexible schemes exist, and are only briefly named here (slide 27, 43, 58): **relative
position** representations, which encode how far apart two words are rather than their
absolute index [Shaw et al., 2018]; representations tied to **dependency syntax**, so that
words close in the parse tree are treated as close [Wang et al., 2019]; and **Rotary
Position Embeddings (RoPE)** [Su et al., 2021].

### 2. No nonlinearities

With no element-wise nonlinearity anywhere, stacking self-attention layers just keeps
re-averaging value vectors — several layers of self-attention collapse into something that
behaves like one big self-attention operation, none of the usual benefit of depth (≈30:19).

The fix is the same one every other architecture uses: apply a feed-forward network
(a two-layer MLP with a ReLU) to *each word's output vector independently*, right after
self-attention:
$$m_i = \text{MLP}(\text{output}_i) = W_2 \cdot \operatorname{ReLU}(W_1 \times \text{output}_i + b_1) + b_2$$
This is applied position-wise — every word goes through the same feed-forward network, but
independently of every other word, which keeps it just as parallelizable as the attention
step itself (≈31:06–32:39).

### 3. Looking at the future

For any task where you're defining a probability distribution over a sequence — language
modeling, or generating a translation — the model can't be allowed to see the answer it's
trying to predict. Restricting the keys and queries to only past words at each step would
work but kills parallelizability, so instead the **whole** $n \times n$ score matrix is
computed as normal, and future positions are masked out afterward (slide 49, ≈33:25):
$$e_{ij} = \begin{cases} q_i^\top k_j & j < i \\ -\infty & j \ge i \end{cases}$$
Softmaxing $-\infty$ sends its weight to zero, so a masked position contributes nothing to
the weighted average — the model can compute everything in parallel and still never see the
future.

This masking is a **decoder-only** device: an encoder over a complete source sentence (as
in machine translation) wants every word to see every other word, so it uses no masking at
all; a decoder generating one token at a time needs it, because at generation time each
step really does have to be conditioned only on what came before (≈35:41). See
[LSTM](lstm.md) for the equivalent unidirectional-vs-bidirectional distinction in recurrent
models.

## Assembling the minimal building block

Put the three fixes together with the base mechanism and you get what lecture 8 calls a
minimal self-attention architecture (slide 26): embed the input, add position embeddings,
run self-attention (masked if generation requires it), pass each output through a
feed-forward layer, and stack that block some number of times before a final prediction
layer. Nobody actually uses exactly this in practice — the [Transformer](transformer.md)
adds several more pieces on top — but it's the right way to understand *why* each of those
additional pieces exists.

## Related pages

- [Attention](attention.md) — the general mechanism this specializes: attention within one
  sequence rather than between an encoder and a decoder.
- [Transformer](transformer.md) — multi-head attention, scaled dot-product attention,
  residual connections and layer norm, and the full encoder/decoder architecture built on
  top of self-attention.
- [Vanishing and exploding gradients](vanishing-and-exploding-gradients.md) — the RNN
  problem that linear interaction distance is a version of.
- [LSTM](lstm.md) — the recurrent building block self-attention replaces, including its own
  unidirectional/bidirectional distinction.
- [Lecture 8 — Self-Attention and Transformers](08-self-attention-and-transformers.md)
