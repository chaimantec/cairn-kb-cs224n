# n-gram language models

The way [language models](language-modeling.md) were built from about 1975 until about 2012
(lecture 5, ≈28:42). Worth understanding not as history but because the two ways they fail —
sparsity and storage — are exactly what the neural models were built to fix, and because the
generation demo shows how far pure counting gets you.

## Definition

An ***n*-gram** is a chunk of *n* consecutive words (lecture 5, slide 17). For *the students
opened their*: unigrams are *the*, *students*, *opened*, *their*; bigrams are *the students*,
*students opened*, *opened their*; trigrams are *the students opened*, *students opened
their*; the four-gram is the whole thing.

Manning's aside for anyone with a classics education: the naming is horrific, because *gram*
is a Greek root and so should take Greek numerals — properly *monograms* and *digrams*.
Shannon's 1951 paper does use "digram"; the idea died there and everyone says "bigram"
(≈29:29). "It's kind of cute, I like it, a nice practical notation."

## The Markov assumption and counting

Two steps (lecture 5, slide 18). First assume the next word $x^{(t+1)}$ depends only on the
preceding $n - 1$ words — throw the rest of the context away:

$$P\left(x^{(t+1)} \mid x^{(t)}, \dots, x^{(1)}\right) \approx P\left(x^{(t+1)} \mid x^{(t)}, \dots, x^{(t-n+2)}\right)$$

Then, by the definition of conditional probability, that equals the probability of an
$n$-gram over the probability of an $(n-1)$-gram — and both are estimated by counting in a
large corpus:

$$\approx \frac{\operatorname{count}\left(x^{(t+1)}, x^{(t)}, \dots, x^{(t-n+2)}\right)}{\operatorname{count}\left(x^{(t)}, \dots, x^{(t-n+2)}\right)}$$

There is no training. You count, and you divide (≈38:45).

## The worked example, and what it gets wrong

Slide 19. Learning a 4-gram model on *as the proctor started the clock, the students opened
their ___*, you discard everything but "students opened their". If in the corpus:

- "students opened their" occurred 1000 times
- "students opened their **books**" occurred 400, so
  $P(\text{books} \mid \text{students opened their}) = 400/1000 = \mathbf{0.4}$
- "students opened their **exams**" occurred 100, so
  $P(\text{exams} \mid \text{students opened their}) = 100/1000 = \mathbf{0.1}$

The slide's own annotation is the objection: *should we have discarded the "proctor"
context?* Knowing a proctor started a clock makes *exams* much more likely, but the model
threw that away and so prefers *books*, which is simply more common (≈33:23). Manning's
verdict is that looking at the immediately prior words is not terrible — those are the most
helpful words to look at — but it is clearly primitive.

## Sparsity

Two distinct failures (slide 20), both raised by students in the lecture (≈33:23–34:10):

**Sparsity problem 1 — zero numerator.** The continuation $w$ never occurred after that
context, so $P(w \mid \cdot) = 0$. Manning stresses why zero is especially bad in probability: once
you have a zero, any computation involving it instantly goes to zero (≈34:56). And unseen
continuations are common — *students opened their accounts*, or, if it is a biology
dissection class, *students opened their frogs*.

*Partial fix:* **smoothing**. Add a small $\delta$ to every count — e.g. $\delta = 0.25$, so
things never seen get a count of 0.25 and things seen once get 1.25, and nothing is
impossible (≈35:41).

**Sparsity problem 2 — zero denominator.** The context itself never occurred, so no
distribution can be computed at all.

*Partial fix:* **backoff**. Condition on a shorter context: if "students opened their" is
unseen, use "opened their"; if that fails, "their" (≈36:26).

## Storage

You must store a count for every *n*-gram observed, and the number of *n*-grams grows
exponentially with context size (slide 21).

## Why n stopped at 5

The two pressures conflict (≈37:14). Longer context gives a better estimate in principle,
but makes both sparsity and storage worse — sparsity so much worse that you almost
necessarily hit zeros. In practice things maxed out at **5**, with occasional 6- and 7-grams.
Google's famous *n*-gram release from the 2000s, built on a **trillion-word** web corpus,
gave counts up to $n = 5$ and stopped there (≈38:00).

Compare this with the ~7-token effective limit of a simple RNN, which lecture 6 uses to
argue that vanilla RNNs did not actually buy much: "in practice, because of vanishing
gradients, we're only kind of getting the equivalent of 8-grams" (lecture 6, ≈17:13).

## Generating text

Slides 22–26. Build a trigram model over a 1.7-million-word Reuters corpus — a few seconds
on a laptop, since there is no training — and sample repeatedly. From *today the*, the model
gives `company 0.153, bank 0.153, price 0.077, italian 0.039, emirate 0.039, …`. Draw a
uniform random number, sample, append, repeat.

Note the coarse granularity of those probabilities: they are counts divided by a total, so
in a small corpus they cluster on values corresponding to "occurred once", "occurred twice",
"occurred four times" (≈39:33). The slide flags this as a sparsity symptom in its own right.

The full sample (slide 26):

> *today the price of gold per ton, while production of shoe lasts and shoe industry, the
> bank intervened just after it considered and rejected an IMF demand to rebuild depleted
> European stocks, sept 30 end primary 76 cts a share.*

Manning's assessment (≈41:54): **surprisingly grammatical** — "the bank intervened just
after it considered and rejected an IMF demand" makes sense as a piece of text — but
completely **incoherent**. The conclusion on the slide: you need to consider more than three
words at a time, but increasing $n$ worsens sparsity and model size.

## "Scale will solve everything" is an old story

A point worth remembering when reading modern scaling claims (≈43:25). The argument people
make today was made in the early 2010s about *n*-grams: not good enough on 10 million words
with a trigram model? Use 100 million and a four-gram. Not enough? A trillion words and a
5-gram. "Gee, wouldn't it be good if we could collect 10 trillion words of text." Manning's
gloss: "it turns out that sometimes you can do better with better models as well as simply
scale."

## What replaced it

Two problems went away with the fixed-window neural LM (lecture 5, slide 30): no sparsity
problem, and no need to store observed *n*-grams — only the network's parameters. Two
remained: the window is still fixed, and different positions are processed by different
weights. Both are what [recurrent neural networks](recurrent-neural-networks.md) fix.

On evaluation, [perplexity](perplexity.md) makes the comparison concrete: the best *n*-gram
smoothing known, **interpolated Kneser-Ney**, reached about 67.6, against 43.7 and 30 for
LSTMs (lecture 6, slide 5).

## Related pages

- [Language modeling](language-modeling.md) — the task these models perform.
- [Recurrent neural networks](recurrent-neural-networks.md) — what replaced them.
- [Perplexity](perplexity.md) — how the comparison is measured, including Kneser-Ney's number.
- [Language models in decoding](language-models-in-decoding.md) — where an *n*-gram model is
  still the right choice in 2024: scoring it is a memory lookup, so it is the only kind of LM
  that fits inside the speech BCI's 20 ms real-time budget.
- [Lecture 5 — Recurrent Neural Networks](05-recurrent-neural-networks.md)
- [Lecture 14 — Brain-computer interfaces](14-brain-computer-interfaces.md)
