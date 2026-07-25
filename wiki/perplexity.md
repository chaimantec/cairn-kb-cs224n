# Perplexity

The standard evaluation metric for [language models](language-modeling.md). Defined on
lecture 5's slide 49, and explained properly — including where it came from and why it is a
slightly odd choice — at the start of
[lecture 6](06-sequence-to-sequence-models.md) (≈2:26–6:22).

## The evaluation logic

Before the formula, the reasoning (lecture 6, ≈2:26). A language model scores a piece of text
and says how likely it is. Our standard for what counts as text in a language is text
produced by human beings. So take a **fresh** piece of human-written text — not text the
model was trained on — show it to the model, and ask it to predict each successive word. The
better it does, the better a language model it is, because it more accurately predicts
human-written text.

This works only because a next-word predictor can also score whole texts, via the chain rule.
See [language modeling](language-modeling.md).

## The definition

    perplexity = ∏_{t=1..T} ( 1 / P_LM( x⁽ᵗ⁺¹⁾ | x⁽ᵗ⁾, …, x⁽¹⁾ ) )^{1/T}

Take the model's probability at each position, **invert** it, take the product across all
positions, and take the geometric average — the 1/T exponent normalizes by number of words,
so texts of different lengths are comparable. Manning's inversion example: a probability of
0.002 becomes 500 (lecture 6, ≈3:12).

**Lower perplexity is better.**

## It is the exponential of cross-entropy

The identity on slide 49 is the one to remember, because the course otherwise works in
negative log likelihoods:

    perplexity = ∏ₜ ( 1 / ŷ⁽ᵗ⁾_{x_{t+1}} )^{1/T}
               = exp( (1/T) Σₜ − log ŷ⁽ᵗ⁾_{x_{t+1}} )
               = exp( J(θ) )

So if you already have a per-word negative log likelihood, just exponentiate it (lecture 6,
≈4:02). J(θ) here is exactly the training loss of an
[RNN language model](recurrent-neural-networks.md). See
[softmax, the logistic function, and cross-entropy](softmax-and-cross-entropy.md).

**A warning about logarithm base.** Perplexity numbers depend on the base used for the
logarithm and exponential. Base 2 was traditional, coming from thinking in bits; natural logs
are now common. Numbers computed under different bases are not comparable, so you need to
know which one is being used (lecture 6, ≈4:02).

## Why perplexity rather than cross-entropy

Manning says outright that "from a modern perspective it kind of makes no sense" to use
perplexity (lecture 6, ≈4:48). The reason it exists is historical.

In the era of symbolic AI — the days of John McCarthy and Ed Feigenbaum doing logic-based
systems — a group at IBM including **Fred Jelinek** began applying probabilistic methods to
speech recognition. Jelinek's own account, as Manning tells it, was that in the late 70s and
early 80s none of the AI people he was trying to talk to understood how to do real math or
knew any information theory, so cross-entropy rate was not going to land. He needed something
simpler.

Exponentiating gives a number with a concrete interpretation: **perplexity is equivalent to
how many uniform choices you are choosing between.** A perplexity of 64 is like rolling a
64-sided die each time you guess the next word, and your chance of getting a particular face
is your chance of guessing right (≈6:22). That is what made it explicable, and it stuck.

## The numbers

Lecture 5's slide 50 and lecture 6's slide 5 show the same table, from a Facebook Research
post on building an efficient LM over a billion words:

| Model | Perplexity |
| --- | --- |
| Interpolated Kneser-Ney 5-gram (Chelba et al., 2013) | 67.6 |
| RNN-1024 + MaxEnt 9-gram (Chelba et al., 2013) | 51.3 |
| RNN-2048 + BlackOut sampling (Ji et al., 2015) | 68.3 |
| Sparse Non-negative Matrix factorization (Shazeer et al., 2015) | 52.9 |
| LSTM-2048 (Jozefowicz et al., 2016) | 43.7 |
| 2-layer LSTM-8192 (Jozefowicz et al., 2016) | 30 |
| **Ours small** (LSTM-2048) | 43.9 |
| **Ours large** (2-layer LSTM-2048) | 39.8 |

Manning's reading of it (lecture 6, ≈7:07):

- The best-known [*n*-gram](n-gram-language-models.md) smoothing method,
  **interpolated Kneser-Ney**, gives about **67**.
- Early RNNs could **not** beat *n*-grams on their own. They only won in combination with
  something else — a symbolic maximum entropy model — giving around **51**.
- Real progress came with [**LSTMs**](lstm.md): **43.7**, and **30** for a two-layer model.
- Halving perplexity from about 60 to 30 corresponds to reducing cross-entropy by about
  **one bit**.

Two pieces of context he adds. Historically, when he started in NLP, perplexities were
three-figure numbers — 150 was common — so 67 was already enormous progress (≈7:07). And by
modern standards even 30 is very high: the best current models reach **perplexities in the
single digits**, meaning they very often guess exactly the right word — though never always,
since nobody can predict what a person will say next in many circumstances (≈8:40).

## Related pages

- [Language modeling](language-modeling.md) — what is being evaluated, and why scoring text
  follows from next-word prediction.
- [*n*-gram language models](n-gram-language-models.md) — the 67.6 baseline.
- [LSTM](lstm.md) — what moved the number to 30.
- [Softmax, the logistic function, and cross-entropy](softmax-and-cross-entropy.md) — the loss
  perplexity exponentiates.
- [Evaluating word vectors](evaluating-word-vectors.md) — the intrinsic/extrinsic distinction
  from earlier in the course.
- [Lecture 6 — Sequence to Sequence Models](06-sequence-to-sequence-models.md)
