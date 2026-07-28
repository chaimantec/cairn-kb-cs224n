# Tree recursive neural networks (TreeRNNs)

A family of models that computes the meaning of a sentence by composing vectors **up a parse
tree**: each parent representation is a learned function of its two children, applied recursively
from the words to the root. Developed at Stanford by Richard Socher, Christopher Manning, Andrew
Ng and colleagues between roughly 2010 and 2015 — "they were sort of the Stanford brand"
([lecture 17](17-convnets-and-treernns.md), ≈42:48).

They are no longer competitive, and the lecture says so plainly. They are worth understanding
because they are the clearest neural implementation of
[compositionality](compositionality.md), and because the one thing they still do better than
Transformers — negation — is a live problem.

## Recursive vs. recurrent

"The difference between 'recursive' and 'recurrent' is sort of a fake difference, right, they
both come from the same 'recur' word" (lecture 17, ≈49:03). The real distinction is the shape of
the recursion:

- A [recurrent network](recurrent-neural-networks.md) recurs **along the sequence**. The state at
  position $t$ summarizes the whole prefix $1..t$, so you cannot get a representation of a phrase
  without everything that preceded it, and the vector tends to over-weight the last few words.
- A recursive network recurs **up a tree**. Each node's vector depends only on its children, so
  *the country of my birth* has a representation that is a fact about that phrase alone.

The cost is that a tree network needs a tree — either given by a parser or learned jointly, as
below.

## Why trees: recursion in language

Slide 35 of the lecture-17 deck reproduces Hauser, Chomsky and Fitch (*Science*, 2002), whose
thesis is that the faculty of language in the narrow sense consists of recursion alone, and that
this is the uniquely human component. The linguistic observation is hierarchical nesting — the
same structure appearing inside itself:

> [The person standing next to [the man from [the company that purchased [the firm that you used
> to work at]]]]

Context-free grammar permits this to nest without bound, like nested `if` statements in a
programming language. Manning is careful about the empirical claim: people neither produce nor
understand unbounded recursion, and in practice "no one's going to go more than eight deep," but
the structure of the language does not appear to have a depth limit, and recursion is "a very
powerful prior for language structure" (lecture 17, ≈45:07–45:52).

Real corpora bear this out. The Penn Treebank parse of *Analysts said Mr. Stronach wants to
resume a more influential role in running the company* contains four verb phrases nested inside
one another.

## The simple TreeRNN

One small network does two jobs at once (slides 42–43). Given the vectors $c_1$ and $c_2$ of two
candidate children, it produces the parent vector and a score for whether merging them is a
plausible constituent:

$$p = \tanh\!\left(W \begin{bmatrix} c_1 \\ c_2 \end{bmatrix} + b\right)$$

$$\text{score} = U^{T} p$$

$W$ is the composition matrix, $b$ a bias, $U$ a learned scoring vector, and — crucially — **the
same $W$ is used at every node of the tree**, exactly as an RNN reuses its parameters at every
timestep (lecture 17, ≈50:35). Training is by gradient descent through the whole tree.

### Greedy parsing with the score

Because the network scores merges, it can build the tree it needs (slides 44–47, ≈51:22):

1. Score every adjacent pair of items in the current sequence.
2. Commit to the highest-scoring merge, replacing the pair with its parent vector.
3. Re-score the pairs the merge created, and repeat until one node remains.

On *The cat sat on the mat*: (*The*, *cat*) scores 3.1 and (*the*, *mat*) 2.3, against 0.1–0.4
for the alternatives, so *The cat* merges first, then *the mat*, then *on* + *the mat* at 3.6,
then *sat*, then the root. The result is a binary parse *and* a vector at every node. A whole
tree's score is the sum of its node decisions,

$$s(x, y) = \sum_{n \in \mathrm{nodes}(y)} s_n$$

for sentence $x$ and parse $y$. Socher, Manning and Ng (ICML 2011) won a best-paper award for
this, and the phrase vectors transferred usefully to sentence classification.

This is a *constituency* parser, and contrasts with the
[transition-based dependency parsers](transition-based-parsing.md) of lecture 4: it builds
phrase structure rather than head-dependent arcs, and it is greedy rather than beam-searched.

### What one matrix cannot do

The limitation is structural (slide 48, ≈52:56–53:42, ≈55:15):

- **No real interaction between the input words.** Concatenate-and-multiply is additive: each
  child contributes independently through its own block of $W$, and neither can modulate the
  other.
- **The same composition function for every syntactic relation.** But an adjective modifying a
  noun and a verb taking an object are different operations. In *the red ball*, *red* supplies
  attributes of the noun; in *kick the ball*, the object plays a completely different role.

## The Recursive Neural Tensor Network (RNTN)

Socher, Perelygin, Wu, Chuang, Manning, Ng and Potts (2013) fix this by letting the children
interact **multiplicatively**. Learn a stack of in-between matrices — a three-dimensional tensor
$V^{[1:d]}$, one slice per output dimension — and add a bilinear form to the standard linear
layer:

$$p = f\!\left(\begin{bmatrix} b \\ c \end{bmatrix}^{T} V^{[1:2]} \begin{bmatrix} b \\ c
\end{bmatrix} + W \begin{bmatrix} b \\ c \end{bmatrix}\right)$$

where $b$ and $c$ are the child vectors, $W$ the ordinary composition matrix and $f$ a
nonlinearity. The new first term is a vector times a matrix times a vector, so each child's
values scale the other's contribution — "mediated multiplicative interactions between word or
phrase vectors" (lecture 17, ≈1:02:17). The illustration in the deck uses two slices; in general
the number of slices is the parent dimensionality.

The line of work continued into **TreeLSTMs** (Tai, Socher, Manning 2015), which apply
[LSTM](lstm.md) gating to the tree-structured case and "work even better" (slide 49).

## Results, and the one that matters

Evaluated on [sentiment analysis](sentiment-analysis.md) over the Stanford Sentiment Treebank:

| Model | Sentence labels | Treebank labels |
| --- | --- | --- |
| Bigram naive Bayes | ~79 | ~83.2 |
| RNN (simple TreeRNN) | ~78.2 | ~82.4 |
| MV-RNN | ~80 | ~82.9 |
| **RNTN** | ~79.8 | **85.4** |

(Slides 52 and 57; accuracies in percent.) The early tree models beat *unigram* naive Bayes but
not the bigram version, "because a lot of the extra information that you want to capture for
sentiment analysis, you can get from bigrams" — *not good*, *somewhat interesting* (lecture 17,
≈1:02:17). Only the RNTN pulls clearly ahead.

The result Manning cares about is **negation** (slide 59). Negating a *positive* sentence is easy
for anything, because *not* is itself a negative word, so all four models show a decrease in
positive activation ($-0.16$ for bigram NB through $-0.54$ for the RNTN). Negating a *negative*
sentence should push sentiment up, and requires genuine composition: bigram NB, the simple RNN
and MV-RNN all produce essentially zero change ($-0.01$, $-0.01$, $+0.01$), while the RNTN moves
correctly by $+0.25$. *It's definitely not dull* comes out positive even though *dull* is
strongly negative and *not* is a negative word on its own.

## Why they lost

The closing diagnosis (lecture 17, ≈1:10:00–1:10:48):

> these models had a strictly context-free backbone, and the only information flow was
> tree-structured … whereas in the Transformer you've got this attention function, where, at
> every position, you're looking at every other position, and so you can have much more general
> information flow.

[Self-attention](self-attention.md) is an all-pairs channel; a tree gives each node only its two
children. The tree also has to be right, and errors propagate. Against that, "to the extent that
you actually want to carefully model the sort of semantics of human language — what modifies
what, and how does negation or quantifiers in a sentence behave — in some sense these models were
more right," and Manning says the negation result "still isn't captured as well by any of the
current Transformer models" (≈1:09:14).

The open question he leaves is whether the two can be combined: "something that's a bit more
tree-structured, while still more flexible, like a Transformer" (≈1:11:35). The same tension —
whether a model that never manipulates structure can be said to compose meaning — is the
substance of [lecture 18](18-nlp-linguistics-philosophy.md).

## See also

- [Lecture 17 — ConvNets and TreeRNNs](17-convnets-and-treernns.md) — the source lecture.
- [Compositionality](compositionality.md) — the principle these models implement.
- [Convolutional neural networks](convolutional-neural-networks.md) — the unstructured
  alternative from the same lecture.
- [Sentiment analysis](sentiment-analysis.md) — the evaluation task, and the treebank.
- [Transition-based parsing](transition-based-parsing.md) and
  [dependency grammar](dependency-grammar.md) — the other parsing tradition in this course.
