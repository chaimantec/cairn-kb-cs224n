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

> **Coverage note.** This knowledge base currently has wiki pages for
> **lectures 1 and 2 only**. Slides and handouts for lectures 1–18 are in
> `raw/pdfs/`, so questions about later lectures can be answered by pointing at
> those PDFs, but there are no transcripts or wiki pages for them yet. See
> [TODO.md](TODO.md) for what remains.

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

## Raw materials

- [`raw/transcripts/`](raw/transcripts/) — verbatim lecture transcripts for
  lectures 1 and 2, grouped into paragraphs each prefixed with an `[MM:SS]`
  timestamp. Use these to point a learner at the exact moment something is
  explained, or to quote Manning directly. Note that these are auto-generated
  captions: technical terms are frequently mangled (*word2vec* appears as "word
  Tove", *PyTorch* as "py talk", *CBOW* as "sibo"), and the dictated mathematical
  notation in lecture 1 is unreliable. Each lecture page ends with a note on the
  specific distortions in that transcript.
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
