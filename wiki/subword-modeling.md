# Subword modeling

How modern NLP represents words it has never seen. Instead of fixing a vocabulary of whole
words and mapping everything else to a single unknown token, subword modeling learns a
vocabulary of *parts* of words, so that any string can be spelled out as a sequence of known
pieces. Covered in [lecture 9](09-pretraining.md), slides 3–6.

## The problem: a finite vocabulary is a bad assumption

Every model in the course up to lecture 9 — [word2vec](word2vec.md), the
[RNN language models](recurrent-neural-networks.md), the [Transformer](transformer.md) —
assumes a vocabulary $V$ of tens of thousands of words, built from the training set, with all
novel words at test time mapped to a single `UNK` token. Slide 3 lays out what that throws
away:

| word | vocab mapping | embedding |
| --- | --- | --- |
| *hat* | *hat* | its own vector |
| *learn* | *learn* | its own vector |
| *taaaaasty* (variation) | `UNK` | the same grey vector |
| *laern* (misspelling) | `UNK` | the same grey vector |
| *Transformerify* (novel item) | `UNK` | the same grey vector |

A human reads all three of the bottom rows without difficulty. *Transformerify* is
transparent from **derivational morphology** — the *-ify* suffix turns a noun into a verb
meaning "make more like that noun" — and the model has no access to that at all, because it
never saw the exact character string (≈3:13). And the supply of such strings never runs out:
"language is always doing this, right? People are always coming up with new words, and there's
new domains, and young people are always making new words" (≈3:13).

English makes this look like a minor annoyance. Slide 4 puts up a Wiktionary conjugation table
for the Swahili verb *-ambia* ("to tell") to show that it is not. Swahili verbs inflect for
tense, mood, definiteness, negation, person and noun class, giving hundreds of conjugations per
verb. Under a whole-word vocabulary each of the 300-plus forms would get an independent vector,
"which makes no sense, because the 300 conjugations obviously have a lot in common and differ by
sort of meaningful extents" — and the vocabulary would have to be enormous, which is bad for
both efficiency and learning (≈4:45).

## Byte-pair encoding

The dominant modern approach is to learn a vocabulary of **subword tokens** and split every
word — at training *and* test time — into a sequence of known subwords. Slide 5 gives the
byte-pair encoding (BPE) algorithm, which builds that vocabulary from data:

1. Start with a vocabulary containing only characters and an "end-of-word" symbol.
2. Using a corpus of text, find the most common adjacent pair of symbols $a, b$; add $ab$ to
   the vocabulary as a new subword.
3. Replace instances of that pair with the new subword; repeat until the vocabulary reaches
   the desired size.

The base case is what removes `UNK` entirely: as long as the vocabulary contains every
character, any string whatsoever can be represented, in the worst case as one token per
character (slide 6). Everything above that base is bought by frequency — common sequences get
promoted into single tokens.

BPE was originally developed for [machine translation](machine-translation.md) (Sennrich et
al., 2016); a closely related method, **WordPiece** (Wu et al., 2016), is what pretrained
models such as [BERT](bert.md) use.

## What the result looks like

Slide 6 re-runs slide 3's table under a subword vocabulary. Common words survive as single
tokens; rarer strings decompose, sometimes intuitively and sometimes not, and every row now
gets real learned vectors rather than one shared `UNK`:

| word | vocab mapping | embeddings |
| --- | --- | --- |
| *hat* | `hat` | one |
| *learn* | `learn` | one |
| *taaaaasty* | `taa## aaa## sty` | three |
| *laern* | `la## ern##` | two |
| *Transformerify* | `Transformer## ify` | two |

The `##` marks a continuation — "don't add a space next" — so the original string can be
reconstructed exactly. Downstream, the model simply sees more tokens: an RNN takes `taa`, does
its update, takes `aaa`, does its update, takes `sty`. The lecture's point is that this lets
the model *process constructions* like elongated spelling, rather than being blind to them,
"instead of just seeing the entire word *tasty* and not knowing what it means" (≈8:36).

## Practical consequences

- **Everything is a subword token.** Once this is adopted, "all of our inputs now are subword
  tokens" — later slides draw them as words only for readability (≈40:57). Punctuation, `...`
  and runs of hyphens are tokens too, because modern practice is to preprocess text as little
  as possible, so that it matches what people will actually feed the system. This is a
  deliberate reversal of the earlier word2vec-era instinct to strip punctuation out (≈10:54).
- **The model cannot tell a whole word from a fragment.** Both are just indices into the
  embedding matrix, treated identically (≈12:24).
- **Tokenization minimizes the number of subwords**, splitting into the maximal pieces
  available, because sequence length is the real bottleneck in a Transformer — a few long
  tokens beat many short common ones (≈13:59).
- **Long words are handled by frequency, not by rule.** A long word that occurs often ends up
  in the vocabulary as a single token; one that doesn't, doesn't. "The statistics speak really
  well for themselves" (≈12:24).
- **If a word becomes several tokens, there is no canonical single vector for it.** Asked how
  to represent *tasty* when it is three tokens, the lecture's answer is that during processing
  they simply stay separate; if you need one vector afterwards you might average the three
  contextual vectors or take the last, but "at that point it's unclear what to do" (≈9:22).

Subword granularity also shapes pretraining objectives. [SpanBERT](bert.md#extensions-of-bert)
exists partly because masking a single subword of a long word makes the prediction nearly
trivial — the surrounding subwords give it away — so masking whole contiguous spans is a
harder and more useful task (≈54:02).

## Related pages

- [Lecture 9 — Pretraining](09-pretraining.md) — where this is taught, as the prerequisite to
  everything else in the lecture.
- [word2vec](word2vec.md) — the model whose fixed vocabulary this replaces.
- [BERT and masked language modeling](bert.md) — uses WordPiece; its masking rate is quoted in
  subword tokens.
- [GPT and in-context learning](gpt-and-in-context-learning.md) — GPT was trained with BPE at
  40,000 merges.
- [Machine translation](machine-translation.md) — where byte-pair encoding was first used in
  NLP.
