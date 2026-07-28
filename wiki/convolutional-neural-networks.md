# Convolutional neural networks for NLP

A convolution computes a representation for **every $n$-gram in a sentence** by sliding a small
learned filter along the sequence. It is the architectural opposite of a
[recurrent network](recurrent-neural-networks.md), which builds one representation per position
out of the entire prefix, and it predates [Transformers](transformer.md) as the main alternative
to recurrence for sentence-level tasks.

Covered in [lecture 17](17-convnets-and-treernns.md) (slides 4–34). Christopher Manning is candid
that this is a historical detour — "in the last five years when Transformers have dominated,
there hasn't been much use of convolutional neural networks for NLP" (lecture 17, ≈4:43) — but
two of its ideas, per-position feed-forward layers and normalization, live on inside the
Transformer.

## The core operation

Take a sentence of $n$ words with $k$-dimensional word vectors $\mathbf{x}_i \in \mathbb{R}^k$,
concatenated as $\mathbf{x}_{1:n} = \mathbf{x}_1 \oplus \mathbf{x}_2 \oplus \cdots \oplus
\mathbf{x}_n$. A **filter** (or **kernel**) $\mathbf{w} \in \mathbb{R}^{hk}$ spans a window of
$h$ words. Apply it at every position:

$$c_i = f\big(\mathbf{w}^{T}\mathbf{x}_{i:i+h-1} + b\big)$$

where $b$ is a bias and $f$ a nonlinearity such as a sigmoid or ReLU. The resulting vector

$$\mathbf{c} = [c_1, c_2, \ldots, c_{n-h+1}] \in \mathbb{R}^{n-h+1}$$

is the filter's **feature map** — one number per window. Slide 9 of the lecture-17 deck works a
full numeric example: seven words, 4-dimensional vectors, one $3\times 4$ filter, giving the five
values $-1.0, -0.5, -3.6, -0.2, 0.3$ before bias and nonlinearity.

Nothing about this respects linguistic structure. Every window of width $h$ gets the same
treatment whether or not it is a constituent, which is why Manning says "there's nothing
linguistically or cognitively especially plausible here" (lecture 17, ≈5:28) — contrast
[tree recursive networks](tree-recursive-neural-networks.md), which take the opposite position.

## Shape control

- **Narrow convolution** (no padding): $n$ words and width $h$ give $n - h + 1$ outputs, so the
  representation shrinks.
- **Padding**: append zero vectors at each end to keep the output length equal to $n$.
- **Wide convolution**: pad by $h-1$ at each end, producing *more* outputs than inputs — seven
  words become nine with a trigram filter and padding of two (lecture 17, ≈10:10).
- **Multiple filters (channels)**: $m$ filters produce $m$ feature maps, so each position is
  described by an $m$-dimensional vector. Whether that is larger or smaller than $k$ is a design
  choice.

## Pooling

A feature map has one value per position, but classification needs a fixed-size vector. Pooling
collapses it.

**Max pooling**, $\hat{c} = \max\{\mathbf{c}\}$, is the default. The intuition that makes it
click is to read a filter as a *feature detector* — one might match first-person language ("I",
"my", "we", "our"), another speech and thinking verbs ("think", "say", "told"). What you want to
know is whether the feature fires anywhere at all, "regardless of whether it's in the first,
second, third, or fourth position" (lecture 17, ≈12:29). Max pooling asks exactly that, and as a
side effect makes the output length-invariant.

**Average pooling** fits a different reading of a filter — as measuring a graded property such as
casualness across the whole text. You can concatenate both, but if you pick one, max pooling
generally wins for the kinds of features neural networks learn (lecture 17, ≈13:16).

Three refinements, all more common in speech and vision than in text:

| Variant | What it does | Why |
| --- | --- | --- |
| **Stride** $s$ | Advance the filter $s$ positions at a time | Adjacent windows overlap heavily and are nearly redundant, especially for wide filters |
| **Local max pooling** | Max within fixed windows rather than globally | A feature firing at four places in a long sentence should count more than once |
| **$k$-max pooling** | Keep the $k$ largest activations per filter, not necessarily adjacent | Detects that something matched in several places, while staying fixed-size |
| **Dilation** | Build a window from positions $1, 3, 5$ rather than $1, 2, 3$ | Widens the receptive field without widening the filter |

## In PyTorch

`nn.Conv1d`, where `out_channels` is the number of filters and `kernel_size` is $h$. Pair it with
`nn.BatchNorm1d` for normalization, and a max over the time dimension for pooling (lecture 17,
≈13:16–14:03).

## Normalization: BatchNorm vs LayerNorm

Both rescale activations to zero mean and unit variance — "like the familiar Z-transform of
statistics" — and differ only in which axis the statistics are computed over:

- **LayerNorm** computes statistics across the feature dimensions of each instance independently.
  This is the one standard in [Transformers](transformer.md).
- **BatchNorm** normalizes across all elements and items in a batch, for each feature
  independently. This is the one standard in CNNs, and was invented first (Ioffe and Szegedy,
  2015).

BatchNorm additionally makes a model much less sensitive to parameter initialization, since
activations are rescaled automatically, and tends to make learning rates easier to tune (lecture
17, slide 28, ≈30:23).

## Size-1 convolutions

A kernel of width 1 looks vacuous and is not: it is a fully connected linear layer *across
channels* at a single position, mapping a word's channel vector to a new one. It can reduce many
channels to few, and it adds depth with very few parameters compared to a fully connected layer
over the whole sentence (Lin, Chen and Yan, *Network in Network*, 2013).

This is the direct ancestor of the per-position feed-forward sublayer inside a
[Transformer](transformer.md) block, as Manning points out: "that's sort of actually what we also
have with the fully connected layers in Transformers … you've got a fully connected layer that's
just at one subword token position, and calculates a new representation for it" (lecture 17,
≈32:41).

## Two landmark systems

**Yoon Kim (2014), single-layer CNN for sentence classification.** Filters of width 3, 4 and 5,
100 feature maps each, max-over-time pooling to one number per filter, concatenate into
$\mathbf{z} = [\hat{c}_1, \ldots, \hat{c}_m]$, and classify with a single softmax layer
$y = \mathrm{softmax}(W^{(S)}\mathbf{z} + b)$. It matched or beat the best contemporary models on
seven [sentiment](sentiment-analysis.md) and classification datasets. Two things to remember from
it: the [dropout](regularization-and-dropout.md) confound in its baseline comparison, and the
**channel-doubling** trick for the word-vector fine-tuning pitfall — see
[lecture 17](17-convnets-and-treernns.md#the-word-vector-fine-tuning-pitfall).

**Conneau et al. (2017), VD-CNN.** A vision-style network for text: characters embedded in 16
dimensions over a fixed 1024-character window, then convolutional blocks (two
[conv → BatchNorm → ReLU] sublayers each, width 3) alternating with local pooling that halves
length and doubles channels — 64, 128, 256, 512 — then $k$-max pooling with $k = 8$ and three
fully connected layers, with ResNet-style shortcuts at each stage. The finding: the deepest model
tested, 29 layers, was best on every dataset, reaching state of the art on several purely from
characters with no pre-trained word vectors.

## Where CNNs stand now

The honest summary from slide 27: CNNs are good for classification, need zero padding for short
phrases, are "somewhat implausible/hard to interpret," and are **easy to parallelize on GPUs** —
which is their real advantage over an RNN, whose hidden state at position $t$ cannot be computed
before position $t-1$. That parallelism argument is the same one
[self-attention](self-attention.md) makes, which is part of why Transformers displaced both.

Traffic now runs the other way: Vision Transformers are taking over in vision, "though some
papers argue that CNNs and Transformers have complementary advantages, and you can usefully use
both" (slide 27).

## See also

- [Lecture 17 — ConvNets and TreeRNNs](17-convnets-and-treernns.md) — the source lecture.
- [Recurrent neural networks](recurrent-neural-networks.md) — the architecture CNNs were
  positioned against.
- [Transformers](transformer.md) and [self-attention](self-attention.md) — what replaced both.
- [Tree recursive neural networks](tree-recursive-neural-networks.md) — the structured
  alternative from the same lecture.
- [Sentiment analysis](sentiment-analysis.md) — the task these models are evaluated on.
