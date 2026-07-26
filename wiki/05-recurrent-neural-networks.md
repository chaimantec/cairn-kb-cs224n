# Lecture 5 — Recurrent Neural Networks

The deck calls this lecture *Language Models and Recurrent Neural Networks*, and the order
matters: the task comes first and the architecture is introduced to serve it. Manning
flags language modeling on slide 2 as **"the most important concept in the class"**, the
thing that leads to BERT, GPT-3 and ChatGPT. The lecture establishes four things: what a
language model is and the two equivalent ways of defining it; how *n*-gram language models
worked and exactly where they broke; that a **recurrent neural network** can consume a
context of any length with one shared weight matrix, which is what a fixed window cannot
do; and that vanilla RNNs have a **vanishing gradient** problem severe enough to negate
that advantage in practice. The last point is the setup for
[lecture 6](06-sequence-to-sequence-models.md).

Before any of that, the first quarter of the lecture is a grab-bag of practical neural-net
technique — regularization, dropout, vectorization, initialization, optimizers — the pieces
that turned deep networks from a 15-year dead end into something that works.

**Slide-by-slide text of this deck: [72 slides](../raw/slides/05-recurrent-neural-networks.md)**
— printed slide numbers match PDF pages 1:1.

Slides PDF: [Lecture 5 — rnnlm](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture05-rnnlm.pdf) ·
Notes: [2019 notes 05 — LM and RNN](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/readings/cs224n-2019-notes05-LM_RNN.pdf) ·
[Full transcript](../raw/transcripts/05-recurrent-neural-networks.md)

## Why deep networks took fifteen years to work

Backpropagation was worked out in the 80s and 90s, but essentially every trained network
had exactly **one** hidden layer, because nobody could make more layers work (≈3:55). Slide
6 quotes Bengio et al. (2006) saying so in print: deep networks "were generally found to be
not better, and often worse" than one- or two-layer networks, and because that is a
negative result it was barely reported. Manning's framing is worth keeping: the fixes that
eventually worked look underwhelming — "fairly little introductions of new ideas and tweaks
of things" — but that handful of tweaks turned around a field that had been stuck for
fifteen years and that nearly everyone had abandoned (≈7:00).

## Regularization, and the modern view of overfitting

The classic story (slide 7) is that regularization exists to prevent **overfitting**:
training error falls monotonically, test error falls and then rises, and an L2 penalty
$\lambda \sum_k \theta_k^2$ — added to the loss, with $\lambda$ controlling its strength —
keeps the parameters small enough that the rise is smaller.

Manning says flatly that modern neural network people **do not believe this picture**
(≈10:54). With billions of parameters, classical statistics says you cannot estimate them
at all — but what you get instead is "a kind of an interesting averaging function from all
these myriad numbers". If you keep training a huge network past the point where it looks
like overfitting, the validation loss goes down as well. Today models are trained until
they memorize the training set almost perfectly — essentially zero loss — and that is not
considered a disaster, provided regularization was done well, because the model still
generalizes (≈12:28). Slide 7 shows both pictures side by side: the textbook curve on the
left, and a measured curve on the right where test error bumps up and then falls again.

The flip side: L1 and L2 are not strong enough regularizers to get that effect, which is
why the field turned to **dropout**.

## Dropout

At training time, for each example, randomly zero some of the inputs to each layer — you
sample a mask of zeros and ones and take its **Hadamard product** with the data (≈14:01).
Slide 8 gives a dropout ratio of $p = 0.5$ typically, less (around 0.15) for the input
layer. At test time you drop nothing and rescale the weights by $1 - p$ instead.

Manning gives three ways to think about why it works (≈15:33–17:04):

1. **No feature co-adaptation.** The model cannot become reliant on component 17 of a
   vector, because sometimes component 17 randomly disappears, so it must learn to use
   whatever other features would also serve.
2. **An implicit ensemble.** You are effectively training the ensemble of the power set —
   an exponential number of possible dropout patterns — all at once.
3. **A middle ground between naïve Bayes and logistic regression.** Naïve Bayes sets each
   feature weight independently of the others; logistic regression sets every weight in the
   context of all the others; dropout sets weights in the context of *some* of the others.

Current practice, following Wager, Wang and Liang (2013) at Stanford, is to regard dropout
as a strong **feature-dependent regularizer** (slide 9).

One practical note: the formulation on slide 8 is the original one, and the version in
Assignment 2 is the way deep learning packages now do it. The details differ slightly
(≈13:14).

## Vectorization, initialization, optimizers

**Vectorization** (slide 10): no for loops, ever. The measured example gives 639 µs per
loop for a Python for loop versus 53.8 µs using a single C × N matrix — an order of
magnitude on CPU, two to three orders of magnitude once you move to GPUs. Manning's rule:
if you are writing a for loop over anything but superficial input processing, you have
almost certainly made a mistake (≈18:38).

**Initialization** (slide 11): weights must start as small random numbers, not zeros.
Zeros leave the network symmetric — every element of the matrix undergoes the same
operation, so a whole vector of features behaves like one feature copied many times, and
there is no gradient signal that breaks the tie (≈19:26). The scale matters: small enough
not to blow up under repeated multiplication, large enough not to vanish. **Xavier
initialization** sets $\mathrm{Var}(W_i) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$, where
$n_{\text{in}}$ is the fan-in and $n_{\text{out}}$ the fan-out. Manning notes this used to be treated as
important and largely goes away later, once **layer normalization** removes the need to be
careful — though you must still initialize to *something*.

**Optimizers** (slide 12): plain SGD works for almost any problem if you fiddle enough, but
"fiddling enough" means step sizes and learning-rate schedules. The adaptive family —
AdaGrad, RMSprop, Adam, AdamW, NAdamW — accumulates a per-parameter measure of past
gradients and scales each parameter's step by it, which buys margins of safety against bad
hyperparameters. AdaGrad (co-invented by John Duchi at Stanford) is the simplest and "tends
to stall early"; **Adam** is the good default and the one used in Assignment 2; the W
variants can help specifically with word vectors, which are updated sparsely because
particular words only turn up occasionally (≈23:59). Start around learning rate 0.001. See
[gradient descent](gradient-descent.md).

## What a language model is

Two equivalent definitions (slides 13–14):

1. A system that puts a **probability distribution over the next word**, given the preceding
   words, $P(x^{(t+1)} \mid x^{(t)}, \dots, x^{(1)})$, where $x^{(t+1)}$ ranges over the
   vocabulary and the probabilities sum to one.
2. A system that **assigns a probability to a piece of text**.

The second follows from the first by the chain rule of probability, for a text of $T$ words:

$$P\left(x^{(1)}, \dots, x^{(T)}\right) = \prod_{t=1}^{T} P\left(x^{(t)} \mid x^{(t-1)}, \dots, x^{(1)}\right)$$

Each factor of the decomposition is exactly what definition 1 provides (≈26:23).

This is not a 2022 invention. Language models have been central to NLP since at least the
80s and the idea goes back to the 50s (≈27:10). Your phone keyboard's next-word suggestions
and Google's query autocomplete are language models (slides 15–16). See
[language modeling](language-modeling.md).

## n-gram language models

The pre-2012 answer (slides 17–22). Make a **Markov assumption** — the next word depends
only on the preceding $n - 1$ words — and estimate by counting in a corpus:

$$P(w \mid \text{students opened their}) = \frac{\operatorname{count}(\text{students opened their } w)}{\operatorname{count}(\text{students opened their})}$$

Slide 19's worked example: if "students opened their" occurred 1000 times, "students opened
their books" 400 and "students opened their exams" 100, then
$P(\text{books} \mid \cdot) = 0.4$ and
$P(\text{exams} \mid \cdot) = 0.1$. The slide's own annotation is the objection — *should we have discarded
the "proctor" context?* Having thrown away "as the proctor started the clock", the model
prefers *books*, when the discarded context made *exams* far more likely (≈33:23).

Two problems that get worse as *n* grows (slides 20–21):

- **Sparsity.** Unseen numerator → probability zero, which is fatal because any computation
  involving it collapses to zero; patched by **smoothing** (add a small $\delta$, e.g. 0.25,
  to every count). Unseen denominator → the whole distribution is undefined; patched by
  **backoff**, conditioning on a shorter context instead.
- **Storage.** You must store counts for every *n*-gram seen; the number grows
  exponentially in context size.

These conflicting pressures are why $n$ maxed out around 5. Google's *n*-gram release, built
on a trillion-word web corpus, stopped at $n = 5$ (≈38:00).

The generation demo (slides 23–26) is the memorable part. From "today the", sample
repeatedly from the stored distributions and you get: *today the price of gold per ton,
while production of shoe lasts and shoe industry, the bank intervened just after it
considered and rejected an IMF demand to rebuild depleted European stocks, sept 30 end
primary 76 cts a share.* Manning's verdict: **surprisingly grammatical, but incoherent**
(≈41:54). And a historical note worth keeping — "scale will solve everything" is not a new
story; it is exactly what people said in the early 2010s about collecting ten trillion words
for a better *n*-gram model (≈43:25). See
[*n*-gram language models](n-gram-language-models.md).

## The fixed-window neural language model

The obvious neural fix (slides 27–30) reuses the window classifier from lecture 2: take a
fixed window of words, look up embeddings, concatenate, hidden layer, softmax — except the
softmax is now over the whole vocabulary rather than a binary label. Writing $e^{(i)}$ for
the embedding of the $i$-th window word, $W$ and $b_1$ for the hidden layer, $U$ and $b_2$
for the output layer, and $|V|$ for the vocabulary size:

$$e = \left[e^{(1)}; e^{(2)}; e^{(3)}; e^{(4)}\right]$$

$$h = f(W e + b_1)$$

$$\hat{y} = \operatorname{softmax}(U h + b_2) \in \mathbb{R}^{|V|}$$

This is roughly Bengio et al. (2000/2003). It removes the sparsity problem and the storage
cost. It did not take off at the time, because with a fixed window it was not that different
from an *n*-gram model, neural nets were hard to run without GPUs, and you could get more
out of counting *n*-grams over hundreds of billions of words (≈47:16).

Two problems remain (slide 30). The window is still fixed and no fixed window is ever big
enough. And — the more interesting objection — **there is no symmetry in how inputs are
processed**: $x^{(1)}$ and $x^{(2)}$ are multiplied by completely different sub-parts of $W$, so what
the model learns about "student" in position 1 is learned separately from what it learns
about "student" in position 2, even though the evidence they provide is the same (≈48:48).

## Recurrent neural networks

The core idea (slide 31): **apply the same weights repeatedly**. A simple RNN language
model (slide 32) is

$$e^{(t)} = E x^{(t)}$$

$$h^{(t)} = \sigma\left(W_h h^{(t-1)} + W_e e^{(t)} + b_1\right)$$

$$\hat{y}^{(t)} = \operatorname{softmax}\left(U h^{(t)} + b_2\right) \in \mathbb{R}^{|V|}$$

where $E$ is the embedding matrix, $W_h$ the recurrent weight matrix, $W_e$ the input weight
matrix, and $h^{(0)}$ the initial hidden state, which can simply be zeros. The nonlinearity
$\sigma$ has most commonly been **tanh** for RNNs, since it is balanced across positive and
negative (≈54:14).
See [activation functions](activation-functions.md).

The hidden state accumulates a memory of everything seen so far, and crucially the *same*
W_h and W_e are applied at every position, so "student" contributes the same way wherever
it occurs (≈54:14). The advantages on slide 33 follow: any-length input, information from
many steps back, **model size does not grow** with context length, and symmetry across
positions.

Two disadvantages, both of which shape the rest of the course:

- **Recurrent computation is slow.** You must compute one hidden vector at a time — which,
  as Manning notes, is literally the for loop he told you never to write (≈56:33).
- **Accessing information from many steps back is hard in practice.** The memory of distant
  words fades and recent words dominate the hidden state. Human beings forget too, he
  concedes, but simple RNNs forget "rather too quickly" (≈57:21).

See [recurrent neural networks](recurrent-neural-networks.md).

## Training an RNN language model

The loss at step $t$ is the cross-entropy between the predicted distribution and the true
next word, which for a one-hot target reduces to the negative log probability of that word;
the overall loss is the average over the $T$ positions (slide 34). See
[softmax and cross-entropy](softmax-and-cross-entropy.md).

$$J^{(t)}(\theta) = -\log \hat{y}^{(t)}_{x_{t+1}} \qquad\qquad J(\theta) = \frac{1}{T} \sum_{t=1}^{T} J^{(t)}(\theta)$$

Slides 35–39 walk this over *the students opened their exams*. The key procedural point,
named on slide 39, is **teacher forcing**: at each step the model predicts a distribution,
is scored against the actual next word, and then is fed **the actual next word** rather than
its own prediction. Manning is explicit that this is not free generation, and that its
limitation is that you never explore what the model would have generated on its own
(≈1:01:59).

Computing loss over an entire corpus at once is impossible memory-wise (slide 40), so the
data is cut into segments. Manning notes the linguistically tidy answer is sentences or
documents, but that in practice, for GPU efficiency, people just cut every 100 words so that
a batch of equal-length segments packs into a matrix (≈1:04:15). A student then makes the
sharp observation that this reintroduces a limit on context — and Manning agrees: **cutting
into segments is making a Markov assumption again** (≈1:13:35).

**Backpropagation through time** (slides 41–43). $W_h$ appears at every timestep, so the
question is the derivative with respect to a repeated weight. The answer is that the
gradient with respect to a repeated weight is the **sum** of the gradients with respect to
each place it appears,

$$\frac{\partial J^{(t)}}{\partial W_h} = \sum_{i=1}^{t} \left. \frac{\partial J^{(t)}}{\partial W_h} \right|_{(i)}$$

which is the "gradients sum at outward branches" rule from lecture 3, applied to
identity copies of $W_h$ whose partials with respect to each other are 1 (≈1:06:36). In
practice this is often **truncated** after ~20 timesteps: the forward pass still uses the
full context, only the backward pass is cut short (≈1:08:09). See
[backpropagation](backpropagation.md).

## Generating with an RNN-LM

Sampling (slide 44) works like the *n*-gram case: sample from $\hat{y}^{(t)}$ and feed the sampled word
back in as the next input — a **roll-out**. Manning notes he cheated in the earlier diagram
by starting from "the"; properly you start from a `<s>` pseudo-word with its own embedding,
and stop when you generate `</s>` (≈1:08:57). This, he points out, is exactly what ChatGPT
is doing, with a more complicated model — and because it is probabilistic, running it twice
gives different answers (≈1:10:31).

The demonstrations (slides 45–48) are trained on Obama speeches (a few hundred thousand
words), *Harry Potter*, recipes, and — the one that works best — **paint colour names**.
The paint model is **character-level**, and it is a *conditional* RNN: the hidden state is
initialized with the RGB values of a colour rather than with zeros, so it generates a name
for a given colour (≈1:16:41). The output includes Ghasty Pink, Navel Tan, Homestar Brown,
Stoner Blue, Burble Simp, Stanky Bean and Turdly (≈1:17:27).

## Evaluation: perplexity

Slide 49 defines **perplexity** as the inverse probability of the corpus, normalized by
number of words — equivalently $\exp(J(\theta))$, the exponential of the cross-entropy loss. Lower
is better. Slide 50's table shows the progression that matters: interpolated Kneser-Ney
5-gram at 67.6, RNN-based hybrids around 51, and LSTMs at 43.7 and 30. Manning treats this
table in more depth at the start of [lecture 6](06-sequence-to-sequence-models.md). See
[perplexity](perplexity.md).

## Vanishing and exploding gradients

Slides 51–56 build the intuition one chain-rule factor at a time:
$\partial J^{(4)} / \partial h^{(1)}$ is a product of Jacobians
$\partial h^{(t)} / \partial h^{(t-1)}$, and if those are small the gradient signal shrinks
the further back it propagates.

The proof sketch (slides 57–58) makes it precise in the linear case. If $\sigma$ is the
identity, $\partial h^{(t)} / \partial h^{(t-1)} = W_h$ exactly, so the gradient from step
$i$ back to step $j$ carries a factor of $W_h^{\ell}$ where $\ell = i - j$:

$$\frac{\partial J^{(i)}(\theta)}{\partial h^{(j)}} = \frac{\partial J^{(i)}(\theta)}{\partial h^{(i)}} W_h^{\ell}$$

Written in the eigenvector basis of $W_h$, with eigenvalues $\lambda_i$ and eigenvectors
$q_i$, that factor is $\sum_i c_i \lambda_i^{\ell} q_i$, which goes to zero for large $\ell$
whenever the eigenvalues are less than 1 (sufficient but not necessary). For nonlinear
$\sigma$ the same argument holds with $\lambda_i < \gamma$ for some $\gamma$ depending on
dimensionality and $\sigma$. Source: Pascanu et al. (2013).

**Why it matters** (slides 59–60): gradient signal from far away is much smaller than
gradient signal from close by, so the model is updated only with respect to near effects. On
the printer/toner example — *When she tried to print her tickets … she finally printed her
___* — the model must connect "tickets" at step 7 to the target at the end, and if the
gradient is too small it never learns that dependency, so it cannot predict long-distance
dependencies at test time either.

**Exploding gradients** (slides 61–62) are the mirror image and, as Manning puts it, an easy
problem: if the gradient norm exceeds a threshold, scale it down before the update —
**gradient clipping**. Same direction, smaller step. The worst case otherwise is Inf or NaN
and a restart from an earlier checkpoint. His line for the failure mode: "You think you've
found a hill to climb, but suddenly you're in Iowa."

Slide 63 states the diagnosis and the fix to come: in a vanilla RNN the hidden state is
**constantly being rewritten**, and the question is whether we can build an RNN with a
separate memory that is *added to* — which is the LSTM. See
[vanishing and exploding gradients](vanishing-and-exploding-gradients.md).

## Other things RNNs are for

Slide 64's recap is worth stating plainly: **an RNN is not a language model**. An RNN is a
family of architectures that take sequential input of any length, apply the same weights at
each step, and can optionally emit output at each step. RNNs are used for much more than
LMs, and LMs can be built with other models (especially Transformers).

Slide 65 gives the old and new answers to why language modeling matters. Old: it is a
benchmark, and a subcomponent of predictive typing, speech recognition, spelling correction,
machine translation, summarization and dialogue. New: **everything in NLP has been rebuilt
on it** — GPT-3, GPT-4, Claude Opus and Gemini Ultra are all language models.

The other uses (slides 66–71): **sequence tagging** (POS tagging, NER — a tag per position);
**sentence classification** (sentiment — either the final hidden state, or, usually better,
the element-wise max or mean of all hidden states); **as an encoder module** inside a larger
system, as for question answering; and **conditional language models**, where generation is
conditioned on some other input, such as audio for speech recognition. That last one is the
bridge to machine translation in the next lecture.

Slide 72 names what was built here: the **simple / vanilla / Elman RNN**. LSTMs, GRUs and
multi-layer RNNs come next.

## Related pages

- [Language modeling](language-modeling.md) — the task, both definitions, and why the course
  is organized around it.
- [*n*-gram language models](n-gram-language-models.md) — the Markov assumption, counting,
  sparsity, smoothing, backoff, and generation.
- [Recurrent neural networks](recurrent-neural-networks.md) — the architecture, RNN-LM
  training, teacher forcing, BPTT, and the bidirectional and stacked variants.
- [Perplexity](perplexity.md) — the metric, its relation to cross-entropy, and the numbers.
- [Vanishing and exploding gradients](vanishing-and-exploding-gradients.md) — the proof
  sketch, clipping, and the architectural fixes.
- [Lecture 6 — Sequence to Sequence Models](06-sequence-to-sequence-models.md) — LSTMs, and
  the first big application of RNNs.
- [Lecture 3 — Backpropagation and Neural Networks](03-backpropagation-and-neural-networks.md)
  — the machinery being unrolled here.
