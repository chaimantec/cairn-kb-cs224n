# Transition-based dependency parsing

How to build a dependency parse with a sequence of cheap local decisions instead of a
search over trees. This is the method Assignment 2 implements, and the one that made
neural dependency parsing practical.

Primary source: [lecture 4](04-dependency-parsing.md), slides 30–49
([slide text](../raw/slides/04-dependency-parsing.md)).

## Where it sits among the alternatives

Slide 30 lists four families:

1. **Dynamic programming** — Eisner (1996) gives an $O(n^3)$ algorithm, for a sentence of
   $n$ words, by producing parse items with heads at the ends rather than the middle.
2. **Graph algorithms** — build a minimum spanning tree over scored dependencies.
   McDonald et al.'s (2005) MSTParser is $O(n^2)$ and scores dependencies independently with
   an ML classifier.
3. **Constraint satisfaction** — eliminate edges failing hard constraints (Karlsson 1990).
4. **Transition-based parsing**, also called deterministic dependency parsing — greedy
   choice of attachments guided by a machine learning classifier. MaltParser (Nivre et al.
   2008) proved it highly effective, and fast.

The fourth is what the course uses, because it gives a very simple machine learning
mechanism and is therefore good for an assignment (≈53:29).

## The arc-standard transition system

Joakim Nivre's system (Nivre 2003; slides 31–32) is roughly a shift-reduce parser — the
kind taught in a compilers class — except that the reduce actions are specialized to create
dependencies with the head on the left or the right (≈54:16).

The parser state, a **configuration**, has three parts:

- a **stack** $\sigma$, written with its top to the **right**, starting as `[ROOT]`;
- a **buffer** $\beta$, written with its top to the **left**, starting as the input sentence;
- a set $A$ of dependency arcs, starting empty.

The trick to remember is that the two data structures are written with their tops facing
each other (≈55:01). The transitions, for input words $w_1, \dots, w_n$ and a relation
label $r$ (slide 32):

$$\text{Start:} \quad \sigma = [\text{ROOT}], \quad \beta = w_1, \dots, w_n, \quad A = \emptyset$$

$$
\begin{aligned}
\textbf{1. Shift} \quad && \sigma,\ w_i \mid \beta,\ A \quad &\Rightarrow \quad \sigma \mid w_i,\ \beta,\ A \\
\textbf{2. Left-Arc}_r \quad && \sigma \mid w_i \mid w_j,\ \beta,\ A \quad &\Rightarrow \quad \sigma \mid w_j,\ \beta,\ A \cup \{r(w_j, w_i)\} \\
\textbf{3. Right-Arc}_r \quad && \sigma \mid w_i \mid w_j,\ \beta,\ A \quad &\Rightarrow \quad \sigma \mid w_i,\ \beta,\ A \cup \{r(w_i, w_j)\}
\end{aligned}
$$

$$\text{Finish:} \quad \sigma = [w], \quad \beta = \emptyset$$

**Shift** moves the first buffer word onto the stack. **Left-Arc** makes the top of the
stack the head of the item below it; **Right-Arc** makes the item below the head of the
top. Either way the arc is recorded and the dependent is popped. Parsing ends when the
buffer is empty and the stack holds ROOT alone.

### Worked example: *I ate fish*

Slides 33–34, ≈55:49–58:56:

| Action | Stack | Buffer | Arc added |
| --- | --- | --- | --- |
| Start | `[root]` | I ate fish | |
| Shift | `[root] I` | ate fish | |
| Shift | `[root] I ate` | fish | |
| Left-Arc | `[root] ate` | fish | `nsubj(ate → I)` |
| Shift | `[root] ate fish` | | |
| Right-Arc | `[root] ate` | | `obj(ate → fish)` |
| Right-Arc | `[root]` | | `root([root] → ate)` |

Final: $A = \{\text{nsubj}(\text{ate} \to \text{I}),\ \text{obj}(\text{ate} \to \text{fish}),\ \text{root}([\text{root}] \to \text{ate})\}$.

The deck's "Nota bene" is the point of the whole exercise: at every step Manning made the
**correct** next transition, but a real parser has to work out which one that is, by
exploring or inferring (slide 34). Different choices build different trees — had he chosen
Right-Arc at the first opportunity, *ate* would have become a dependent of *I* (≈59:43). So
**the sequence of transitions is the parse**.

Note also that arc-standard only ever builds **projective** trees (slide 38); see
[dependency grammar](dependency-grammar.md) for what that excludes and how others cope.

## Choosing the next action

Slide 35: each action is predicted by a discriminative classifier over the legal moves — at
most 3 untyped choices, or $|R| \times 2 + 1$ when the arcs are typed, for a set $R$ of
relation labels. In the simplest form there is **no search whatsoever**: predict, commit,
move on. A beam search keeping $k$ good parse prefixes at each step is slower but better.

The payoff is complexity. Each word is handled a constant number of times, so this is a
**linear time** parsing algorithm — against the cubic time you pay to fully consider the
parses of a context-free grammar (≈1:01:19). Nivre's essential result was that machine
learning is good enough that greedy prediction, with no search at all, still produces a very
accurate parser (≈1:02:04). MaltParser's accuracy sits a little below the state of the art
but it gives very fast linear-time parsing — good enough to parse the web.

## Evaluation: UAS and LAS

Slide 37, on *She saw the video lecture*. Write each word's head as an index (0 = ROOT):

| # | Word | Gold head | Gold label | Parsed head | Parsed label |
| --- | --- | --- | --- | --- | --- |
| 1 | She | 2 | nsubj | 2 | nsubj |
| 2 | saw | 0 | root | 0 | root |
| 3 | the | 5 | det | 4 | det |
| 4 | video | 5 | nn | 5 | nsubj |
| 5 | lecture | 2 | obj | 2 | ccomp |

$$\text{Acc} = \frac{\#\ \text{correct deps}}{\#\ \text{of deps}}$$

- **UAS** — *unlabeled attachment score*: fraction of words assigned the correct head. Here
  4 of 5, so **80%**.
- **LAS** — *labeled attachment score*: correct head **and** correct relation label. Here
  2 of 5, so **40%**.

## The three problems with symbolic features

Nivre's 2005 parser used an older-style symbolic classifier — logistic regression or an SVM
— over **indicator features**: conjunctions of one to three elements of the configuration
(slide 36). Writing $s_1$ and $s_2$ for the top two stack items, $\mathrm{lc}(\cdot)$ for a
leftmost child, and `.w`, `.t`, `.l` for a token's word, POS tag and dependency label, the
configuration with *good* on top of the stack and *has* below it fires features like

$$
\begin{aligned}
& s_1.w = \text{good} \ \wedge\  s_1.t = \text{JJ} \\
& s_2.w = \text{has} \ \wedge\  s_2.t = \text{VBZ} \ \wedge\  s_1.w = \text{good} \\
& \mathrm{lc}(s_2).t = \text{PRP} \ \wedge\  s_2.t = \text{VBZ} \ \wedge\  s_1.t = \text{JJ}
\end{aligned}
$$

These produce a binary vector of dimension $10^6$–$10^7$. Slide 39 names the three problems:

1. **Sparse** — conjoining particular words gives millions of features, each seen almost
   never. A feature might fire ten times in a million sentences (≈1:03:37).
2. **Incomplete** — some word combinations simply never occur in training, so the
   corresponding features are missing rather than merely rare.
3. **Expensive** — the one that surprises people: **more than 95% of parsing time went into
   computing the features**, not into the machine learning decision (≈1:06:45).

## The neural parser (Chen and Manning 2014)

The fix is a dense representation, and it delivers two distinct wins.

### First win: distributed representations

Slide 41. Each word becomes a $d$-dimensional dense vector, so a configuration never seen in
training still resembles configurations that were. The step further is to embed the
**part-of-speech tags and dependency labels** as well: real POS tag sets are much
finer-grained than noun/verb/adjective, and NNS (plural noun) should end up near NN
(singular noun), just as `nummod` should end up near `amod` (≈1:09:53). These smaller
discrete sets exhibit their own similarities. See
[distributional semantics](distributional-semantics.md).

The configuration becomes a vector by extracting a fixed set of tokens from the stack and
buffer (slide 42) — the top two stack items $s_1$, $s_2$, the first buffer item $b_1$, and
their leftmost and rightmost already-built children $\mathrm{lc}(s_1)$, $\mathrm{rc}(s_1)$,
$\mathrm{lc}(s_2)$, $\mathrm{rc}(s_2)$ — then looking
up the word, POS and dependency-label embedding of each and **concatenating** them. Missing
slots are $\emptyset$. This is the same concatenation move as the five-word window classifier of
lecture 2 (≈1:11:24).

### Second win: a non-linear classifier

Slide 43. Traditional ML classifiers — Naïve Bayes, SVMs, logistic regression, softmax —
only give **linear decision boundaries**. A neural network with a hidden layer learns
non-linear ones. See [activation functions](activation-functions.md).

The architecture (slide 44) is a plain feed-forward multi-class classifier, with $W$ and
$b_1$ the hidden layer's weights and bias and $U$ and $b_2$ the output layer's:

$$
\begin{aligned}
x &= \text{lookup + concat of the extracted tokens} \\
h &= \operatorname{ReLU}(W x + b_1) \\
y &= \operatorname{softmax}(U h + b_2) \quad \text{over } \{\text{Shift}, \text{Left-Arc}_r, \text{Right-Arc}_r\}
\end{aligned}
$$

The hidden layer re-represents the input — moving it around in an intermediate vector space
— so that a linear softmax can separate it. The log loss (cross-entropy error) is
**backpropagated all the way into the embeddings**, so the word, POS and label vectors are
themselves learned. See [backpropagation](backpropagation.md) and
[softmax, the logistic function, and cross-entropy](softmax-and-cross-entropy.md).

### Results

Slide 40, English parsing to Stanford Dependencies:

| Parser | UAS | LAS | sent./s |
| --- | --- | --- | --- |
| MaltParser | 89.8 | 87.2 | 469 |
| MSTParser | 91.4 | 88.1 | 10 |
| TurboParser | **92.3** | 89.6 | 8 |
| Chen & Manning 2014 | 92.0 | **89.7** | **654** |

As accurate as the best graph-based parsers, and roughly 50–80× faster than them. The
counter-intuitive part: you might expect real-valued matrix arithmetic to be slower than
symbolic feature lookup, but because the symbolic models spent nearly all their time
computing features, the neural version came out **faster and more accurate at the same
time** (≈1:08:19). Slide 45's summary: this was the first simple and successful neural
dependency parser.

## What came after

**Scaling it (slide 46).** Google developed the model further — bigger and deeper networks
with better-tuned hyperparameters, beam search, and global CRF-style inference over the
whole decision sequence — producing SyntaxNet and the **Parsey McParseFace** model in 2016,
billed as "the world's most accurate parser":

| Method | UAS | LAS (PTB WSJ SD 3.3) |
| --- | --- | --- |
| Chen & Manning 2014 | 92.0 | 89.7 |
| Weiss et al. 2015 | 93.99 | 92.05 |
| Andor et al. 2016 | 94.61 | 92.79 |

**Graph-based parsers (slides 47–48).** The other approach: compute a score for every
possible head of every word — $n^2$ candidate dependencies in a sentence of length $n$ — then
find the best tree with a minimum spanning tree algorithm, which also rules out cycles and
disconnected fragments. For *The big cat sat*, picking the head of *big* means scoring arcs
from ROOT (0.5), *The* (0.3), *cat* (2.0) and *sat* (0.8), and taking *cat*. Doing this well
needs good **contextual** representations of each word token, which the course builds in
later lectures.

**Dozat and Manning (2017)** (slide 49) revived graph-based parsing in a neural setting with
a biaffine scoring model, reaching **95.74 UAS / 94.08 LAS** — a bit over a percent better
than Parsey McParseFace. It is slower than the transition-based parsers, because of those
$n^2$ dependencies, and it is the parser in **Stanza**, Stanford's open-source NLP software
(≈1:18:22).

## Related pages

- [Dependency grammar](dependency-grammar.md) — what these parsers are producing.
- [Syntactic ambiguity](syntactic-ambiguity.md) — why the decisions are hard.
- [Backpropagation](backpropagation.md) — how the neural parser is trained.
- [Lecture 4](04-dependency-parsing.md) — the full lecture.
