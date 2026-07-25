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
male/female distinction is clean and linear, but for other relations it works
sometimes and not others.

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
are (≈49:43). You then ask your model to score the same pairs and measure the
**correlation** between the model's judgments and the humans'.

The results table (≈50:29) is the comparison worth remembering, and it lines up the
two traditions described in
[distributional semantics](distributional-semantics.md):

- Plain SVD over raw counts works **terribly**.
- SVD over **log** counts already works reasonably well.
- CBOW and skip-gram (the two [word2vec](word2vec.md) models) score better.
- [GloVe](glove.md) scores at the top.

The jump from raw to log counts is the striking part — it says much of the gap
between the old count-based methods and neural word vectors was preprocessing, not
architecture.

## Extrinsic: named entity recognition

The downstream task the lecture uses is **named entity recognition** (≈51:17):
finding names in text and classifying what type of thing they are. In "Chris Manning
lives in Palo Alto", you want *Chris Manning* tagged as a person and *Palo Alto* as a
place.

NER is a natural fit for testing word vectors, and the result is positive (≈52:03):
starting from a discrete, symbolic, probabilistic NER baseline, adding word vectors
to the system raises the scores substantially — the GloVe rows sit above the
baseline row.

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
