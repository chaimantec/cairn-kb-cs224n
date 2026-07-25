# Evaluating word vectors

Manning introduces evaluation as a distinction that "will come up again and again"
throughout the course, so it is worth learning properly the first time even though
the running example is word vectors. Covered in
[lecture 2](02-word-vectors-and-language-models.md) (≈45:52–52:03).

Slides: [wordvecs2](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture02-wordvecs2.pdf)

## Intrinsic versus extrinsic

**Intrinsic evaluation** scores performance on a specific, self-contained subtask
(≈46:38). It is fast to compute and helps you understand the component you are
building, but it is distant from what you actually want, so improving the number may
or may not help downstream.

**Extrinsic evaluation** takes a real task you care about — question answering,
document summarization, machine translation — plugs your component into a full
system, and measures accuracy on the real task (≈47:23). This is what you actually
want to know, but you have to run an entire system, and when the number moves it is
harder to see exactly why.

For word vectors specifically: measuring whether the vectors model word similarity
well is intrinsic; measuring whether they improve web search — so that *cell phone*
and *mobile phone* behave alike — is extrinsic (≈48:09).

Neither is sufficient alone, and the tension between them is the reason evaluation
recurs as a theme. A component that scores well intrinsically and does nothing for
your task is a common outcome.

## Intrinsic: word analogies

The first intrinsic evaluation is the analogy task — `a is to b as c is to what` —
scored over a fixed set of analogies, reporting the percentage the model gets right
(≈48:09).

Manning is refreshingly honest here: he admits he **cheated** in the notebook demo
by only showing analogies that work, and that if you play with the vectors yourself
you will find plenty that fail (≈48:09). The demo is a sales pitch; the scored set is
the evaluation. He notes GloVe does generally work, and shows a plot where the
male/female distinction is clean and linear (**slide 26**: brother–sister,
nephew–niece, uncle–aunt, man–woman, sir–madam, heir–heiress, king–queen,
emperor–empress, duke–duchess, earl–countess, all as near-parallel lines), but for
other relations it works sometimes and not others.

**Slide 25** gives the scoring rule, which is worth reading closely:

> `d = arg max_i [ (x_b − x_a + x_c)ᵀ x_i ] / ‖x_b − x_a + x_c‖`

Two details on that slide are easy to miss and both matter. First, the metric is
**cosine distance after addition**, not raw dot product — hence the normalization.
Second, marked with three exclamation marks on the slide: the search **discards the
input words**. Without that, `king − man + woman` would most often return *king*
itself, since the result vector stays nearest its own starting point. The slide also
raises the honest objection: *what if the information is there but not linear?* — in
which case a linear analogy test will score the vectors badly even though they encode
the relation.

## Intrinsic: word similarity against human judgments

The second intrinsic evaluation compares model similarity scores against **human**
similarity judgments (≈48:57). The data comes from psychologists asking
undergraduates to rate word pairs — here on a scale of 0 to 10 — and averaging the
answers across people. The examples Manning shows:

| Pair          | Human score |
| ------------- | ----------- |
| tiger–tiger   | 10          |
| book–paper    | 7.46        |
| plane–car     | 5.77        |
| stock–phone   | 1.62        |
| stock–jaguar  | 0.92        |

He calls it a noisy process, but it roughly captures how similar people think words
are (≈49:43). The dataset is named on **slide 27**: **WordSim353**. That slide's table
also includes tiger–cat 7.35, computer–internet 7.58, professor–doctor 6.62 and
stock–CD 1.31. You then ask your model to score the same pairs and measure the
**correlation** between the model's judgments and the humans'.

The results table (≈50:29) is the comparison worth remembering, and it lines up the
two traditions described in
[distributional semantics](distributional-semantics.md):

- Plain SVD over raw counts works **terribly**.
- SVD over **log** counts already works reasonably well.
- CBOW and skip-gram (the two [word2vec](word2vec.md) models) score better.
- [GloVe](glove.md) scores at the top.

**Slide 28** is that table, across five similarity datasets (SG = skip-gram; Size is
the training corpus in tokens):

| Model | Size | WS353 | MC | RG | SCWS | RW |
| ----- | ---- | ----- | -- | -- | ---- | -- |
| SVD | 6B | 35.3 | 35.1 | 42.5 | 38.3 | 25.6 |
| SVD-S | 6B | 56.5 | 71.5 | 71.0 | 53.6 | 34.7 |
| SVD-L | 6B | 65.7 | 72.7 | 75.1 | 56.5 | 37.0 |
| CBOW | 6B | 57.2 | 65.6 | 68.2 | 57.0 | 32.5 |
| SG | 6B | 62.8 | 65.2 | 69.7 | 58.1 | 37.2 |
| GloVe | 6B | 65.8 | 72.7 | 77.8 | 53.9 | 38.1 |
| SVD-L | 42B | 74.0 | 76.4 | 74.1 | 58.3 | 39.9 |
| **GloVe** | **42B** | **75.9** | **83.6** | **82.9** | **59.6** | **47.8** |
| CBOW | 100B | 68.4 | 79.6 | 75.4 | 59.4 | 45.5 |

The jump from raw SVD to scaled SVD is the striking part — 35.3 to 56.5 to 65.7 on
WS353 from nothing but preprocessing the counts, which says much of the gap between the
old count-based methods and neural word vectors was preprocessing rather than
architecture. The other lesson is in the last two rows: GloVe on 42B tokens beats CBOW
on 100B, so more data does not automatically win.

## Extrinsic: named entity recognition

The downstream task the lecture uses is **named entity recognition** (≈51:17):
finding names in text and classifying what type of thing they are. In "Chris Manning
lives in Palo Alto", you want *Chris Manning* tagged as a person and *Palo Alto* as a
place.

NER is a natural fit for testing word vectors, and the result is positive (≈52:03):
starting from a discrete, symbolic, probabilistic NER baseline, adding word vectors
to the system raises the scores — the GloVe row sits above the baseline row.

**Slide 29** is that table (F1 on a dev set, a test set, and the ACE and MUC7
benchmarks):

| Model | Dev | Test | ACE | MUC7 |
| ----- | --- | ---- | --- | ---- |
| Discrete | 91.0 | 85.4 | 77.4 | 73.4 |
| SVD | 90.8 | 85.7 | 77.3 | 73.7 |
| SVD-S | 91.0 | 85.5 | 77.6 | 74.3 |
| SVD-L | 90.5 | 84.8 | 73.6 | 71.5 |
| HPCA | 92.6 | **88.7** | 81.7 | 80.7 |
| HSMN | 90.5 | 85.7 | 78.7 | 74.7 |
| CW | 92.2 | 87.4 | 81.7 | 80.2 |
| CBOW | 93.1 | 88.2 | 82.2 | 81.1 |
| **GloVe** | **93.2** | 88.3 | **82.9** | **82.2** |

"Discrete" is the symbolic baseline. Worth noting how much smaller the extrinsic gains
are than the intrinsic ones: GloVe beats the baseline by about 2 points on Dev and 9 on
MUC7, and the plain SVD rows barely move the baseline at all despite differing hugely on
the similarity benchmarks. That gap is precisely the intrinsic/extrinsic caution above.

NER returns at the end of lecture 2 as the task for the first neural classifier, and
in that context Manning notes it is usually followed by **entity linking**, mapping
each found entity to a canonical form such as a Wikipedia page (≈1:04:22).

## Related pages

- [word2vec](word2vec.md) — CBOW and skip-gram, the models being scored
- [GloVe](glove.md) — the top-scoring model in the table
- [distributional semantics](distributional-semantics.md) — the SVD baselines and
  why log counts matter
- [lecture 2](02-word-vectors-and-language-models.md) — full context, including the
  demo Manning admits was cherry-picked
