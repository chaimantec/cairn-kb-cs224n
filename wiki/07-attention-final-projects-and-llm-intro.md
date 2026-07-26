# Lecture 7 — Attention, Final Projects and LLM Intro

A hinge lecture with two distinct halves. The first is technical and short: a five-minute
close-out of [machine translation](machine-translation.md) evaluation, then **attention** —
the fix for the seq2seq bottleneck that [lecture 6](06-sequence-to-sequence-models.md) left
unsolved, and the idea that the rest of the course, starting with next lecture's
Transformer, builds directly on top of. The second half is a practical briefing on the
CS224N final project: how to choose and scope one, how to find data and research topics,
and how to actually get a neural network to train.

**Slide-by-slide text of this deck: [73 slides](../raw/slides/07-attention-final-projects-and-llm-intro.md)**
— printed slide numbers match PDF pages 1:1.

Slides PDF: [Lecture 7 — final-project](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture07-final-project.pdf) ·
[Full transcript](../raw/transcripts/07-attention-final-projects-and-llm-intro.md)

## Evaluating machine translation

Slides 4–6 close out the machine translation material with **BLEU**, the standard
automatic evaluation metric — how it scores *n*-gram overlap against reference
translations, why it's imperfect, and what BLEU numbers mean in practice (20s for a
roughly-gettable translation, 50s–60s for a strong modern neural system). See
[evaluating machine translation: BLEU](evaluating-machine-translation.md) for the full
treatment, including the worked scoring example on slide 5 and the 2013–2019 BLEU trend
chart on slide 6 that visibly shows neural MT overtaking the statistical approaches it
replaced.

## Attention

Slides 7–28 are the technical core of the lecture. Manning motivates attention from the
seq2seq **bottleneck**: cramming an entire source sentence into one fixed-size vector is
implausible for anything but a short sentence, and isn't how a human translator works — a
person keeps glancing back at the source as they translate (≈13:20–14:09). Attention gives
the decoder a direct connection to every encoder hidden state at every step, worked through
visually across slides 10–21 with the running French example *il a m'entarté* → "he hit me
with a pie," where the model's learned attention distribution tracks which source word each
output word actually corresponds to — a soft word alignment nobody explicitly trained
(≈24:21, slide 23).

The lecture then works through the history of how attention scores actually get computed
— dot-product, [Bahdanau et al.'s additive form](attention.md#history-three-ways-to-score-attention)
(the original, 2014), and [Luong, Pham, and Manning's multiplicative/bilinear form](attention.md#history-three-ways-to-score-attention)
(2015, with Manning himself a co-author) — and closes by generalizing attention beyond
machine translation entirely: given any set of values and a query, attention computes a
weighted, query-dependent summary of those values. That general definition is what
[self-attention](self-attention.md) specializes the very next lecture. Full treatment,
including all three scoring variants and their trade-offs, is at
[attention](attention.md).

## Final projects and research practice

The remaining ~40 minutes (slides 29–72) shift from technical content to practical
guidance: choosing between the default project (a minBERT implementation, open-ended
within a guided scope) and a custom project; the five common project shapes; where to find
a research topic and where to find data; the train/tune/dev/test discipline that keeps a
project's results trustworthy; and a short, concrete checklist for actually getting a
neural network to train (start tiny, verify near-100% on a handful of examples, then scale
up, and don't fear overfitting the training set — regularize against the dev set instead).
Full treatment, including the named past-project examples (Deep Poetry, Word2Bits, the
CodeLlama-on-Fortran project, and others) at
[final project guidance](final-project-guidance.md).

## Related pages

- [Evaluating machine translation: BLEU](evaluating-machine-translation.md)
- [Attention](attention.md) — the mechanism, its three scoring variants, and its general
  definition.
- [Final project guidance](final-project-guidance.md) — choosing, scoping, and running a
  project; finding data; training discipline.
- [Machine translation](machine-translation.md) — the task this lecture's BLEU section
  evaluates.
- [Sequence-to-sequence and encoder-decoder models](seq2seq-and-encoder-decoder.md) — the
  bottleneck problem attention solves.
- [Self-attention](self-attention.md) — where attention goes next, in
  [lecture 8](08-self-attention-and-transformers.md).
