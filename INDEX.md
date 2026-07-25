# CS224N — Natural Language Processing with Deep Learning (Stanford, Spring 2024)

CS224N is Stanford's course on deep learning for natural language processing,
taught here by **Christopher Manning** in Spring 2024, with guest lectures from
several researchers. It builds from the bottom up: word vectors, feed-forward
networks, recurrent networks and attention, then the methods that define the field
today — transformers, encoder-decoder models, pretraining and post-training of
large language models, adaptation, interpretability, and agents. Manning's stated
goals are that students learn the methods, gain some real understanding of human
language and why it is hard for computers, and come out able to build working
systems.

> **Coverage note.** This knowledge base covers **lectures 1 to 6**: wiki pages,
> timestamped transcripts, and full slide-by-slide text for each. Slide *URLs* for
> lectures 1–18 are listed in [sources.md](sources.md), so questions about later
> lectures can be answered by pointing at the right PDF, but there are no transcripts,
> wiki pages, or slide text for them yet. See [TODO.md](TODO.md) for what remains.
>
> **Citing sources.** Prefer citing a **slide number** for anything on a slide
> (equations, tables, definitions) and a **timestamp** for anything Manning says aloud
> (asides, worked reasoning, answers to student questions). The slide files in
> `raw/slides/` carry the numbers; the transcripts carry the timestamps.

## Lecture pages

- [Lecture 1 — Intro and Word Vectors](wiki/01-intro-and-word-vectors.md) — what
  the course covers and how it is graded; why language matters and how neural NLP
  progressed from 2014 machine translation to GPT-2 and ChatGPT; denotational vs
  distributional theories of meaning; why WordNet and one-hot vectors fail; the
  word2vec setup, objective function, and the full hand-derivation of its gradient.
- [Lecture 2 — Word Vectors and Language Models](wiki/02-word-vectors-and-language-models.md)
  — stochastic gradient descent and random initialization; why word2vec uses two
  vectors per word; skip-gram vs CBOW; negative sampling and the unigram^(3/4)
  sampling trick; the gensim/GloVe notebook demo and analogies; co-occurrence
  counts, SVD and GloVe; intrinsic vs extrinsic evaluation; word senses; and the
  first neural classifier, a window classifier for named entities.
- [Lecture 3 — Backpropagation and Neural Networks](wiki/03-backpropagation-and-neural-networks.md)
  — how neural networks are trained, in two halves. The non-linearities lecture 2 left
  unexplained and why a network without them collapses to one layer; gradients, Jacobians
  and the chain rule; the worked hand-derivation of ∂s/∂**b** and ∂s/∂**W** and the δ
  error signal; the Jacobian form vs shape convention collision; then computation graphs,
  the upstream × local = downstream rule, the (*x*+*y*)·max(*y*,*z*) worked example,
  automatic differentiation and numeric gradient checking.
- [Lecture 4 — Dependency Parsing](wiki/04-dependency-parsing.md) — constituency vs
  dependency views of sentence structure; why human language is *globally* ambiguous and
  the four headline ambiguity types; heads, dependents, typed relations, projectivity;
  Pāṇini to Tesnière and the rise of treebanks and Universal Dependencies; the
  arc-standard transition system worked through on *I ate fish*; UAS and LAS; and why
  Chen & Manning's neural parser beat symbolic feature parsers on accuracy *and* speed.
- [Lecture 5 — Recurrent Neural Networks](wiki/05-recurrent-neural-networks.md) — the
  lecture Manning calls the most important in the class. Regularization and the modern
  view that overfitting no longer matters, dropout, vectorization, Xavier initialization
  and the adaptive optimizers; what a language model *is*, in two equivalent definitions;
  *n*-gram models, their sparsity and storage failures, and the "today the price of gold"
  generation demo; the fixed-window neural LM and why it lacks symmetry across positions;
  the RNN, its training by teacher forcing, backpropagation through time, and roll-out
  generation (including the paint-colour names); perplexity; and the vanishing/exploding
  gradient problem with its eigenvalue proof sketch.
- [Lecture 6 — Sequence to Sequence Models](wiki/06-sequence-to-sequence-models.md) —
  perplexity explained properly, with Jelinek's 64-sided die; the ~7-token limit of a
  simple RNN; the LSTM in full — cell state, three gates, the equations, the history from
  Hochreiter and Schmidhuber through Graves to Hinton at Google, and why "the + sign is
  the secret"; vanishing gradients as a general deep-network problem, with ResNet,
  DenseNet and HighwayNet; bidirectional and stacked RNNs; machine translation from the
  1950s code-breaking analogy through statistical MT to the 2014 neural breakthrough; and
  the seq2seq encoder-decoder model, trained end-to-end.

## Topic pages

- [word2vec](wiki/word2vec.md) — the algorithm in full: center and outside words,
  the average-negative-log-likelihood objective, why there are two vectors per
  word, the 80-million-parameter count, skip-gram vs CBOW, naive softmax vs
  negative sampling, and how negative samples are drawn. **Start here for anything
  about how word vectors are learned.**
- [Distributional semantics](wiki/distributional-semantics.md) — the idea the whole
  course rests on: meaning as context. Denotational semantics and its limits,
  WordNet's failures, why one-hot vectors are orthogonal and useless, what dense
  embeddings buy you, how high-dimensional spaces behave differently from 2-D, and
  the count-based lineage through SVD, latent semantic analysis and Rohde's COALS.
- [GloVe](wiki/glove.md) — the Stanford model, and the one whose vectors are used
  in the class demo. Why *ratios* of co-occurrence probabilities encode meaning
  components, the ice/steam/solid/gas worked example, and how taking logs turns
  that ratio into a log-bilinear model with linear meaning directions.
- [Gradient descent](wiki/gradient-descent.md) — the update rule and why the
  learning rate must be small; why plain gradient descent is never used and
  stochastic/mini-batch gradient descent is both faster *and* a better optimizer;
  why zero initialization breaks learning; and the calculus used in the lecture-1
  derivation, including the "observed minus expected" gradient form.
- [Softmax, the logistic function, and cross-entropy](wiki/softmax-and-cross-entropy.md)
  — exponentiate-then-normalize and why the name "softmax" is slightly misleading;
  the logistic function and why "sigmoid" is the looser term; and why
  cross-entropy loss with one-hot labels is exactly the negative log likelihood
  you have been minimizing all along. **Read this if PyTorch's
  `CrossEntropyLoss` confuses you.**
- [Word senses and polysemy](wiki/word-senses-and-polysemy.md) — the cost of one
  vector per word. The *pike* and *field* examples, per-sense clustering and the
  four senses of *jaguar*, why a single vector is a frequency-weighted
  superposition of senses, Manning's argument that discrete senses are the wrong
  model anyway, and the sparse-coding result that recovers senses from one vector.
- [Evaluating word vectors](wiki/evaluating-word-vectors.md) — intrinsic vs
  extrinsic evaluation, a distinction that recurs all course. Analogy scoring (and
  Manning admitting the demo was cherry-picked), similarity against human
  judgments with the actual score table, the model comparison from plain SVD up to
  GloVe, and named entity recognition as the downstream task.
- [Backpropagation](wiki/backpropagation.md) — the algorithm, reduced to its two ideas:
  the chain rule, and caching intermediate results. Computation graphs, the
  upstream × local = downstream rule at a single node, why gradients **sum** at outward
  branches, what `+`, `max` and `*` each do to a gradient, the same-big-O invariant,
  what frameworks automate and what they leave you to write, and numeric gradient
  checking. **Start here for anything about how a network is trained.**
- [Matrix calculus](wiki/matrix-calculus.md) — the by-hand version: gradient → Jacobian,
  multiplying Jacobians for the chain rule, the four Jacobians that do all the work, the
  δ error signal and why ∂s/∂**W** = δᵀ**x**ᵀ is an outer product, and the **shape
  convention** vs Jacobian form — which is the thing that most often confuses people, and
  what Assignment 2 expects.
- [Activation functions](wiki/activation-functions.md) — the *f* in *f*(**Wx** + **b**).
  Why a network without one collapses to a single linear transform, why the 1943 threshold
  unit could not learn, and the whole family: logistic, tanh (a rescaled logistic), hard
  tanh, ReLU and why its dead zone works anyway, Leaky/Parametric ReLU, Swish and GELU.
- [Dependency grammar](wiki/dependency-grammar.md) — heads and dependents, typed
  relations, trees and the fake ROOT, projectivity and the five ways to cope with
  crossing arcs, the history from Pāṇini through Tesnière, and why treebanks beat
  hand-written grammars — including the arrival of evaluation itself.
- [Transition-based parsing](wiki/transition-based-parsing.md) — the arc-standard system
  (stack, buffer, Shift/Left-Arc/Right-Arc) worked through step by step on *I ate fish*;
  UAS and LAS with the scoring example; the three problems with indicator features
  (sparse, incomplete, and 95% of runtime); and the Chen & Manning neural parser, its
  results table, and the graph-based alternative up to Dozat & Manning.
- [Syntactic ambiguity](wiki/syntactic-ambiguity.md) — why parsing is a prediction
  problem. The global-vs-local ambiguity argument against programming languages, the four
  headline types with their newspaper examples, Catalan growth in the four-PP *Wall Street
  Journal* sentence, and why this is what defeated hand-written grammars.
- [Language modeling](wiki/language-modeling.md) — the concept Manning calls the most
  important in the class. The two equivalent definitions and why the chain rule makes them
  the same, why it is not a 2022 invention, the old and new answers to why it matters, the
  four ways the course builds one, conditional language models, and the distinction that a
  recurrent neural network is *not* a language model. **Start here for anything about what
  an LM is.**
- [*n*-gram language models](wiki/n-gram-language-models.md) — how language models worked
  from 1975 to 2012. The Markov assumption and counting, the "students opened their"
  example and what discarding the proctor context costs, both sparsity problems with
  smoothing and backoff, storage, why *n* stopped at 5, the generation demo, and why
  "scale will solve everything" is an old story.
- [Recurrent neural networks](wiki/recurrent-neural-networks.md) — the architecture and the
  fixed-window problem it solves; the equations; advantages and the two disadvantages that
  shape the rest of the course; training with teacher forcing and 100-word segments;
  backpropagation through time and truncation; roll-out generation; sequence tagging,
  sentence encoding and encoder uses; and the bidirectional and stacked variants with the
  rule that bidirectionality cannot be used for generation.
- [LSTM](wiki/lstm.md) — parsing the name, the history from Hochreiter and Schmidhuber 1997
  through the Gers forget gate to Hinton at Google, hidden state vs cell state, the full
  gate equations, why the cell and hidden state are separate (with the *King of Prussia*
  example), and why the additive update gives ~100 timesteps of memory instead of ~7.
  **The + sign is the secret.**
- [Vanishing and exploding gradients](wiki/vanishing-and-exploding-gradients.md) — the
  chain-rule intuition, the eigenvalue proof sketch for W_h^ℓ, the printer/tickets example,
  the ~7-token rule of thumb, gradient clipping, and the generalization beyond RNNs to
  ResNet, DenseNet and HighwayNet — plus why RNNs are worse than deep feed-forward networks
  (the repeated matrix is the *same* one).
- [Perplexity](wiki/perplexity.md) — the standard LM metric. The evaluation logic, the
  formula, the identity perplexity = exp(cross-entropy), the warning about logarithm base,
  Fred Jelinek's 64-sided-die explanation of why it exists at all, and the full results
  table from Kneser-Ney at 67.6 down to LSTMs at 30.
- [Machine translation](wiki/machine-translation.md) — where NLP started. The 1950s
  code-breaking analogy and why it flopped, statistical MT and the Bayes-rule split into a
  translation model and a language model, why that split let the translation model stay
  dumb, the Aztec Empire sentence Google Translate kept failing, and the 2014 neural
  breakthrough.
- [Sequence-to-sequence and encoder-decoder models](wiki/seq2seq-and-encoder-decoder.md) —
  the two-RNN architecture, seq2seq as a conditional language model, training end-to-end
  with teacher forcing in the decoder, the "Conditioning = Bottleneck" note that attention
  later solves, why you cannot just stack layers on the source, and the tasks beyond MT
  the pattern covers.
- [Regularization and dropout](wiki/regularization-and-dropout.md) — L2 and the classic
  overfitting picture, and Manning's claim that modern practitioners do not believe it;
  why models are now trained to memorize the training set; dropout's mechanics at train and
  test time; and the three explanations of why it works, including the naïve-Bayes /
  logistic-regression middle ground.

## Raw materials

- [`raw/slides/`](raw/slides/) — **the full text of every slide, with slide numbers**,
  for lecture 1 ([40 slides](raw/slides/01-intro-and-word-vectors.md)), lecture 2
  ([47 slides](raw/slides/02-word-vectors-and-language-models.md)), lecture 3
  ([85 slides](raw/slides/03-backpropagation-and-neural-networks.md)), lecture 4
  ([49 slides](raw/slides/04-dependency-parsing.md)), lecture 5
  ([72 slides](raw/slides/05-recurrent-neural-networks.md)) and lecture 6
  ([56 slides](raw/slides/06-sequence-to-sequence-models.md)). Each file opens
  with a section-to-slide-range table, then transcribes every slide in order —
  including the equations, the tables of numbers, the margin annotations, and prose
  descriptions of the diagrams and plots. For lectures 1–3, 5 and 6 **the printed slide
  number equals the PDF page number**, so "slide 28" is page 28. **Lecture 4 is the
  exception:** its printed numbers run 1–49 but the PDF has only 45 pages, because printed
  slides 4, 5, 8 and 13 were hidden in the source deck and never exported — cite the printed
  number and expect it to sit a few pages later in the PDF. One further caveat: **lecture
  6's slides 4–18 repeat lecture 5's slides 49–63** as a recap, and are transcribed in brief
  with pointers rather than in full — the exception is slide 15, which adds the "~7 tokens
  back" rule of thumb that slide 25 then contrasts LSTMs against. Use these files when a
  learner asks where something is in the slides, wants an equation exactly as written,
  or when the transcript is unclear — the slides are the authority.
- [`raw/transcripts/`](raw/transcripts/) — lecture transcripts for lectures 1 to 6,
  grouped into paragraphs each prefixed with an `[MM:SS]` timestamp. Use these to
  point a learner at the exact moment something is explained ("Manning covers this
  around 42:00"), or to quote him directly — they read as sentences, so they quote
  cleanly. These are auto-generated captions that have been **copy-edited**:
  punctuation and sentence boundaries added, filler and false starts removed, and
  mis-heard vocabulary restored (*word2vec* arrived as "word Tove" and "word DEC",
  *CBOW* as "sibo", *COALS* as "Kohl's", *ReLU* as "value", *tanh* as "10 H", and in
  lecture 4 *parsing* itself as "paing" and *parser* as "paa"; in lectures 5 and 6
  *n-gram* as "engram", *Xavier* as "harier", *Hadamard* as "hadam mod", *eigenvalue* as
  "ion value", and the names *Bengio*, *Jelinek*, *Feigenbaum*, *Hochreiter*, *Gers* and
  *Olah* all beyond recognition). No content was added,
  removed or reordered, and every timestamp is preserved. Student questions are marked
  in italics. Where a garble could not be resolved from the slides, the text carries an
  inline `[Ed: …]` note saying so instead of guessing — treat those as known gaps, and
  prefer the slide. Each header notes what remains unreliable in that lecture.
- [`raw/transcripts/original/`](raw/transcripts/original/) — the untouched verbatim
  captions, kept only for reference. **Prefer the edited transcripts above**; reach for
  these only to check exactly what the speech recognizer produced.
- [`sources.md`](sources.md) — the full inventory of course documents, with a
  canonical URL for each: lecture slides for lectures 1–18, supplementary readings
  (the 2019 course notes, the gradient and differential-calculus reviews, the
  self-attention and transformers notes, the Python review), the assignment 2–4
  handouts, the final-project handouts, and 43 further papers the syllabus links
  to on arxiv and the ACL Anthology. **The slides are the authority wherever the
  transcript is garbled**, especially for mathematical notation — cite the URL and
  let the learner open it.

  The PDF binaries are deliberately not committed here. This knowledge base is
  read by an agent that navigates markdown and cannot extract anything from a
  PDF blob in the repo, so the canonical course URL is the useful artifact and
  committing the decks would have added over 100MB to every clone. Wiki pages link
  slides directly at `web.stanford.edu`.

## How this KB is organized

See [AGENTS.md](AGENTS.md) for the conventions — relative links only, every claim
traceable to a transcript timestamp or a slide, and never inventing course content.
Where a transcript is genuinely unclear, the pages say so rather than filling the
gap.
