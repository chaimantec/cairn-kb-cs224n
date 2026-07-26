# Machine translation

Translating a sentence $x$ from a **source language** into a sentence $y$ in a **target
language** (lecture 6, slide 42). The slide's example is Rousseau: *L'homme est né libre, et
partout il est dans les fers* → *Man is born free, but everywhere he is in chains*.

Introduced in the second half of [lecture 6](06-sequence-to-sequence-models.md) (slides
42–55) and the subject of Assignment 3. It matters to the course for three reasons: it is
where NLP began; it is where [language models](language-modeling.md) first did most of the
work in a real system; and it is where deep learning's first big NLP success happened.

## The 1950s: where NLP started

Machine translation predates both artificial intelligence and NLP as fields (lecture 6,
≈55:15). Research began in the **early 1950s**, before the term "A.I." was coined, on
machines less powerful than high school calculators, concurrent with the foundational work
on automata, formal languages, probability and information theory.

Manning's account of *why* is the memorable part (≈56:00). Computers were developed during
the Second World War for two purposes: calculating artillery tables, and **code breaking**.
As the war gave way to the Cold War, with concerns on both sides about keeping up with the
other's science, someone had the idea that **translation between languages might be like code
breaking**. That thought reached the relevant people and the science funding agencies, and a
great deal of money was poured in.

It was, after some impressive-looking cooked-up demos, a complete flop. Slide 43 lists the
reasons; Manning emphasizes two (≈57:35):

- **Nobody knew anything about the structure of human languages.** The Chomsky hierarchy had
  not been invented; the formal properties of languages had not been explored. (Compare
  [lecture 4](04-dependency-parsing.md), where the Chomsky hierarchy appears as an argument
  that finite-state mechanisms cannot represent human language.)
- **The machines were hopeless.** "The little power brick for your laptop has more computing
  power inside it than the big mainframe computers they used to be using in those days."

So systems were simple lexicons plus rule-based word substitution, and MT was heavily
military-funded but soon appeared intractable. Slide 44 shows the Paramount News reel:
*ELECTRONIC "BRAIN" Translates RUSSIAN to ENGLISH*.

## 1990s–2010s: statistical machine translation

The field revived once people started building **empirical models over lots of data**
(≈58:20). The core idea (slide 45): learn a **probabilistic model** from data. We want the
best target sentence $y$ given source sentence $x$:

$$\arg\max_y P(y \mid x)$$

and **Bayes rule** breaks this into two components that can be learned separately:

$$= \arg\max_y P(x \mid y) \cdot P(y)$$

- **$P(x \mid y)$ — the translation model.** How words and phrases should be translated
  (*fidelity*). Learned from **parallel data**.
- **$P(y)$ — the language model.** How to write good English (*fluency*). Learned from
  **monolingual data**.

**Why this factorization helped** is the part worth understanding, since on its face it just
swaps $x$ and $y$ (≈1:00:41). It let the translation model stay *very simple*: essentially a
table of how words tend to get translated — see *homme*, emit "man" or "person" with some
probabilities — with **no knowledge of word order, grammar or structure in the target
language**. All of that moved into $P(y)$, a pure language model of exactly the kind lecture 5
builds. This is the clearest illustration of slide 65's "old answer" for why language
modeling matters: it is a **subcomponent** of other tasks.

**Parallel data** came from the places that produce it institutionally (≈59:07): the European
Union, which generates a huge amount among European languages; Hong Kong, for
English–Chinese; and the UN.

**The cost** (slide 47). SMT was a huge research field, and the best systems were extremely
complex: hundreds of important details, many separately-designed subcomponents, lots of
feature engineering to capture particular language phenomena, extra maintained resources like
tables of equivalent phrases, and a great deal of human effort — repeated for **each language
pair**.

## Why translation is hard to model

Slide 46 gives two illustrations.

**Word order does not correspond.** The German *Morgen | fliege | ich | nach Kanada | zur
Konferenz* aligns to English *Tomorrow | I | will fly | to the conference | in Canada* with
crossing arrows.

**Modification structure gets lost.** This is Manning's favourite failing example, and he
used it for years (≈1:01:29). The source is the Chinese translation of Jared Diamond's *Guns,
Germs, and Steel*. The correct reading:

> In 1519, six hundred Spaniards landed in Mexico to conquer **the Aztec Empire** **with a
> population of a few million**. They lost two thirds of their soldiers in the first clash.

Google Translate's attempts:

| Year | Output |
| --- | --- |
| 2009 | 1519 600 Spaniards landed in Mexico, *millions of people to conquer the Aztec empire*, the first two-thirds of soldiers against their loss. |
| 2013 | 1519 600 Spaniards landed in Mexico *to conquer the Aztec empire, hundreds of millions of people*, the initial confrontation loss of soldiers two-thirds. |
| 2015 | 1519 600 Spaniards landed in Mexico, *millions of people to conquer the Aztec empire*, the first two-thirds of the loss of soldiers they clash. |

The word choices are poor, but Manning's point is that the **real** failure is structural
(≈1:03:48). Chinese marks with an explicit particle, *de*, that "a few million people"
modifies "the Aztec Empire". The system loses that relation, and so the millions of people
become the ones doing the conquering. He also notes that 2013 looked like progress and 2015
had gone back downhill, so 2013 was luck rather than a better system (≈1:04:35). His
conclusion on the whole era: although some progress was made, these systems "just never
really worked all that great."

## 2014: neural machine translation

Slide 48 is a meteor labelled *Neural Machine Translation* striking a planet labelled *MT
research*, captioned "(dramatic reenactment)".

**Neural Machine Translation (NMT)** does the whole task with a **single end-to-end neural
network** (slide 49). Manning flags the generality of that idea (≈1:05:23): if you can build
one big system, put a loss function at the end, and backpropagate errors all the way back
down, you align *all* the learning with the final task — which earlier pipeline models could
not do.

The architecture is the **sequence-to-sequence** (seq2seq) model, using two RNNs — in practice
[LSTMs](lstm.md). It is covered in full under
[sequence-to-sequence and encoder-decoder models](seq2seq-and-encoder-decoder.md); the short
version is that an encoder RNN reads the source and produces an encoding, and a decoder RNN
is a language model that generates the target sentence conditioned on that encoding.

Training needs a **big parallel corpus** (slide 52). Manning mentions in passing that there is
now interesting work on "unsupervised NMT", where you have only a little information about how
the languages relate and not much parallel text, but does not cover it (≈1:12:22).

## Why NMT was the first big success story

Slide 55. NMT went from a **fringe research attempt in 2014** to the **leading standard method
in 2016**, when Google Translate switched from SMT to NMT; by 2018 everyone had. The logos on
the slide: Microsoft, SYSTRAN, Google, Facebook, Baidu, NetEase, Tencent, Sogou.

Manning places it third in deep learning's sequence of wins (≈1:13:54): **speech recognition**
first, **object recognition** in vision second, **machine translation** third.

What makes the story striking (≈1:15:26, and the slide's own emphasis): the SMT systems being
displaced had been worked on for about a decade, by **hundreds** of engineers, with millions
of lines of code and per-language-pair hacks. They were outperformed by NMT systems built by
**small groups** in a few **months**. The quality difference was obvious enough that people
noticed before Google announced it, which is what the *New York Times* piece linked on the
slide ("The Great A.I. Awakening") is about (≈1:16:13).

The displacement is visible in the WMT conference numbers on slide 41: **WMT 2014 had zero**
neural MT systems; WMT 2016's summary report mentions "RNN" **44** times and those systems
won; WMT 2019 mentions "RNN" **7** times and "Transformer" **105** times. So NMT displaced
SMT, and Transformers then displaced RNN-based NMT.

## Related pages

- [Sequence-to-sequence and encoder-decoder models](seq2seq-and-encoder-decoder.md) — the
  architecture that made NMT work.
- [Evaluating machine translation: BLEU](evaluating-machine-translation.md) — how MT quality
  gets measured, covered at the start of
  [lecture 7](07-attention-final-projects-and-llm-intro.md).
- [Attention](attention.md) — the fix for NMT's bottleneck, and the idea
  [Transformers](transformer.md) build on.
- [LSTM](lstm.md) — what the encoder and decoder actually are in practice.
- [Language modeling](language-modeling.md) — $P(y)$ in the SMT decomposition, and the decoder
  in the neural one.
- [Recurrent neural networks](recurrent-neural-networks.md) — including why the encoder can be
  bidirectional but the decoder cannot.
- [Lecture 6 — Sequence to Sequence Models](06-sequence-to-sequence-models.md)
