# Evaluating machine translation: BLEU

How [machine translation](machine-translation.md) quality gets measured, covered at the
start of [lecture 7](07-attention-final-projects-and-llm-intro.md) (slides 4–6) as a
five-minute prelude before the lecture moves on to attention. Assignment 3 uses BLEU
directly.

## Why it's hard

Translation doesn't have one right answer the way an image-classification label does.
Any sentence can be translated many valid ways, with different word choices and different
word orders, so you can't just check a candidate translation's words off against a single
reference sentence in order (lecture 7, ≈5:36–6:22). Before BLEU, the only real evaluation
method was **human judgment** — still the gold standard, and still widely used, because
automatic measures carry their own biases and blind spots — but human evaluation is slow,
and doesn't fit into a training loop where you want to iterate quickly (≈2:27).

## BLEU

**BLEU** — **B**i**l**ingual **E**valuation **U**nderstudy — was proposed by IBM and is
still, decades later, the most common automatic measure (≈3:13). The idea: given one or
more human-written **reference translations** for a source sentence, score a candidate
machine translation by how much it **overlaps** with the references, using overlapping
1-, 2-, 3-, and 4-grams (the choice of 4 isn't special — three or five would also be
reasonable, but four was judged a good length), combined as a geometric mean of *n*-gram
precision, plus a penalty for translations that are suspiciously short (Papineni et al.,
2002; slide 4). The short-translation penalty exists because, without it, a system could
game the score by only translating the easy parts of a sentence and leaving out anything
difficult — the parts it does translate would still score well on precision even though
the translation as a whole is worse (≈7:09).

The original design assumed **several** reference translations per source sentence, to
sample the space of valid translations and get reasonable coverage of the many ways a
sentence could correctly be rendered; in practice today it's not uncommon to see just one
reference, with the argument that on a large enough test set, good translations will still
tend to match more often than bad ones, in a roughly probabilistic sense (≈4:48). Slide 5
works a fully-annotated example: a machine translation of a news item about a Guam
bio-threat scare is checked against four independent human references, with each matching
span of words circled and connected by lines back to whichever reference it overlaps.

**What BLEU misses.** Because it's matching *n*-grams rather than meaning, BLEU is
"useful but imperfect": a genuinely good translation can score poorly if its word choices
simply don't happen to match the reference, and a bad translation — words in completely
the wrong grammatical role — can still pick up some credit if enough of them individually
match somewhere in the reference. It's harder to accidentally match for larger *n*, so
longer overlaps carry more real signal, but the crudeness at the *n*-gram level is a known,
accepted limitation (≈7:09).

## Reading BLEU numbers

BLEU is theoretically bounded between 0 and 100, but 100 is unreachable in practice, given
how many valid ways there are to translate the same sentence (≈7:57). As a rough guide:

- **20s** — a translation from which you can roughly understand what the source document
  was about.
- **30s–40s** — translations that are getting much better.
- **50s–60s** — not unusual for modern neural machine translation systems; unheard of in the
  statistical-MT era.

Slide 6 charts BLEU on the same benchmark from 2013 to 2019: phrase-based and syntax-based
statistical MT both crawl upward from about 20 to the mid-20s over six years, while neural
MT — starting out *below* both systems in 2015 — overtakes them by 2016 and reaches roughly
42 by 2019, a far steeper trajectory than either statistical approach ever achieved. See
[machine translation](machine-translation.md) for the history behind that crossover.

## Related pages

- [Machine translation](machine-translation.md) — the task BLEU measures, and the
  statistical-to-neural transition the BLEU chart illustrates.
- [Attention](attention.md) — what lecture 7 covers immediately after this evaluation
  prelude.
- [Lecture 7 — Attention, Final Projects and LLM Intro](07-attention-final-projects-and-llm-intro.md)
