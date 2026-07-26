# Language modeling

Manning calls language modeling **"the most important concept in the class"** on
[lecture 5](05-recurrent-neural-networks.md)'s slide 2, and the annotation on that slide
says why: it is what leads to BERT, GPT-3 and ChatGPT. Everything from lecture 5 onward is
either a way of building a language model or a use of one.

## Two definitions, and why they are the same

**As next-word prediction** (lecture 5, slide 13). Given a sequence of words
$x^{(1)}, \dots, x^{(t)}$, compute the probability distribution over the next word:

$$P\left(x^{(t+1)} \mid x^{(t)}, \dots, x^{(1)}\right)$$

where $x^{(t+1)}$ can be any word in the vocabulary $V$, and the probabilities over $V$ sum
to one. A system that does this is a **language model**. The canonical example throughout the
course is *the students opened their ___*, with *books*, *laptops*, *exams* and *minds* as
candidate continuations.

**As a probability over text** (lecture 5, slide 14). A language model is a system that
assigns a probability to a piece of text $x^{(1)}, \dots, x^{(T)}$:

$$
\begin{aligned}
P\left(x^{(1)}, \dots, x^{(T)}\right) &= P\left(x^{(1)}\right) \cdot P\left(x^{(2)} \mid x^{(1)}\right) \cdots P\left(x^{(T)} \mid x^{(T-1)}, \dots, x^{(1)}\right) \\
&= \prod_{t=1}^{T} P\left(x^{(t)} \mid x^{(t-1)}, \dots, x^{(1)}\right)
\end{aligned}
$$

The second definition follows from the first by the **chain rule of probability**: each
factor in the decomposition is exactly what definition 1 provides (lecture 5, ≈26:23). This
is not a technicality — it is why a next-word predictor can be used to score text, which is
what makes evaluation by [perplexity](perplexity.md) possible and what makes a language
model usable as a component inside a translation system.

## It is not a new idea

Language models have been central to NLP since at least the 1980s, and the idea goes back to
at least the 1950s — Claude Shannon introduced *n*-grams while working out information
theory, and his 1951 paper uses the (etymologically correct) term "digram" (lecture 5,
≈27:10 and ≈30:16). Manning is explicit that language models "weren't something that got
invented in 2022 with ChatGPT".

You use them daily (lecture 5, slides 15–16): phone keyboard next-word suggestions, which
are traditionally compact and not very good so they can run in little memory; and Google's
query autocomplete.

## Why it matters — the old answer and the new one

Lecture 5's slide 65 puts both:

**Old answer.** Language modeling is a benchmark task that measures progress on predicting
language use, and a subcomponent of many NLP tasks — anything that involves generating text
or estimating the probability of text. The list: predictive typing, speech recognition,
handwriting recognition, spelling and grammar correction, authorship identification, machine
translation, summarization, dialogue.

**New answer.** *Everything in NLP has now been rebuilt upon language modeling.* GPT-3 is an
LM, GPT-4 is an LM, Claude Opus is an LM, Gemini Ultra is an LM. We now instruct language
models to do language understanding and reasoning tasks for us.

The old answer is not merely historical. [Statistical machine
translation](machine-translation.md) factored translation into a translation model and a
language model precisely so that the language model could carry most of the work — the
translation model could stay simple and know nothing about target-language word order or
grammar, because the language model $P(y)$ over target sentences $y$ handled fluency
(lecture 6, ≈1:00:41).

## Ways to build one

The course covers four, in order:

1. **[*n*-gram language models](n-gram-language-models.md)** — count *n*-grams in a corpus
   and normalize. What the field used from roughly 1975 to 2012.
2. **Fixed-window neural language models** — concatenate word embeddings from a fixed
   window, hidden layer, softmax over the vocabulary. Roughly Bengio et al. (2000/2003).
   Removes the sparsity and storage problems but keeps the fixed window, and processes each
   position with different weights (lecture 5, slides 27–30).
3. **[Recurrent neural networks](recurrent-neural-networks.md)** — one shared weight matrix
   applied at every position, so context length is unbounded and the model size does not
   grow with it. Lecture 5's main subject; improved by the [LSTM](lstm.md) in lecture 6.
4. **Transformers** — flagged repeatedly as what came next and what dominates now, but
   taught in a later lecture.

A distinction lecture 5's slide 64 makes explicitly and that is easy to lose: **a recurrent
neural network is not a language model**. An RNN is an architecture; a language model is a
task. RNNs do many other things, and language models are built many other ways.

## Conditional language models

A **conditional language model** generates text conditioned on some other input rather than
from nothing (lecture 5, slide 71). The RNN-LM is initialized or conditioned with a
representation of that input, and generation proceeds as usual. Examples in the course:

- **Speech recognition** — condition on audio, generate the transcript (lecture 5, slide 71).
- **Paint colour naming** — condition the initial hidden state on RGB values, generate a name
  a character at a time (lecture 5, ≈1:16:41).
- **Machine translation** — condition on the source sentence. Lecture 6 makes this the formal
  description of seq2seq: a language model because the decoder predicts the next target word,
  conditional because it is also conditioned on the source (slide 52). See
  [seq2seq and encoder-decoder models](seq2seq-and-encoder-decoder.md).

## Training and generation

Both are covered in detail under [recurrent neural
networks](recurrent-neural-networks.md), but the two named ideas belong here:

- **Teacher forcing** — during training, after scoring the model's prediction, feed it the
  *actual* next word rather than its own prediction (lecture 5, slide 39). It makes training
  simple because you always know the right answer, at the cost of never exploring what the
  model would have generated on its own (≈1:01:59).
- **Roll-out** — during generation, feed the sampled word back in as the next input and
  repeat until an end-of-sequence symbol (lecture 5, slide 44). Manning notes this is exactly
  what ChatGPT does, and that because it is probabilistic, running it twice gives different
  answers (≈1:10:31).

## Related pages

- [*n*-gram language models](n-gram-language-models.md) — the pre-neural approach and where
  it broke.
- [Recurrent neural networks](recurrent-neural-networks.md) — the RNN-LM, its training, and
  its variants.
- [Perplexity](perplexity.md) — how language models are evaluated.
- [Softmax, the logistic function, and cross-entropy](softmax-and-cross-entropy.md) — the
  output layer and the loss.
- [Lecture 5 — Recurrent Neural Networks](05-recurrent-neural-networks.md)
- [Lecture 6 — Sequence to Sequence Models](06-sequence-to-sequence-models.md)
