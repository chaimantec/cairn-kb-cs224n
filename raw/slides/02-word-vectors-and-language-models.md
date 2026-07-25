---
title: Lecture 2 — Word Vectors, Word Senses, and Neural Classifiers (slide deck)
lecture: 2
slides: 47
source_pdf: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture02-wordvecs2.pdf
note: Slide numbers printed on the slides match the PDF page numbers exactly. Slide 47 is blank.
---

# Lecture 2 — Word Vectors, Word Senses, and Neural Classifiers: slide-by-slide

Text and figures of all 47 slides of
[`cs224n-spr2024-lecture02-wordvecs2.pdf`](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture02-wordvecs2.pdf),
transcribed from the deck. Cite these as "slide N" — the printed number equals the
PDF page number. Diagrams and plots are described in prose since the KB is read as
text.

Note the deck's own title is "Word Vectors, Word Senses, and Neural Classifiers",
while the YouTube video is titled "Word Vectors and Language Models".

Companion pages: [wiki page for this lecture](../../wiki/02-word-vectors-and-language-models.md) ·
[transcript](../transcripts/02-word-vectors-and-language-models.md)

## Contents

| Slides | Section |
| ------ | ------- |
| 1 | Title |
| 2 | Lecture plan and the key goal |
| 3 | §1 Course organization |
| 4–6 | §2 Optimization basics: gradient descent, update rule, SGD |
| 7–9 | §3 Review of word2vec; parameters and computations; the "bag of words" point |
| 10–14 | §4 The word2vec algorithm family: skip-gram vs CBOW, negative sampling, sparse gradients |
| 15–20 | §5 Counting instead of predicting: co-occurrence matrices, SVD, COALS |
| 21–23 | GloVe: ratios of co-occurrence probabilities, the log-bilinear model and loss |
| 24–29 | §6 Evaluating word vectors: intrinsic vs extrinsic, analogies, WordSim353, NER |
| 30–33 | §7 Word senses: *pike*, sense clustering, superposition and sparse coding |
| 34–39 | §8 Classification and NER: window classifier, neural vs linear, cross-entropy loss |
| 40–46 | §9 Introducing neural networks: neurons, stacked logistic regressions, matrix notation, non-linearities |
| 47 | (blank) |

---

## Slide 1 — Title

"Natural Language Processing with Deep Learning — CS224N/Ling284". Christopher
Manning. "Lecture 2: Word Vectors, Word Senses, and Neural Classifiers".

## Slide 2 — Lecture Plan

"Lecture 2: Word Vectors, Word Senses, and Neural Network Classifiers"

1. Course organization (3 mins)
2. Optimization basics (5 mins)
3. Review of word2vec and looking at word vectors (12 mins)
4. More on word2vec (8 mins)
5. Can we capture the essence of word meaning more effectively by counting? (12m)
6. Evaluating word vectors (10 mins)
7. Word senses (10 mins)
8. Review of classification and how neural nets differ (10 mins)
9. Introducing neural networks (10 mins)

> Key Goal: To be able to read and understand word embeddings papers by the end of class

## Slide 3 — 1. Course Organization

- First assignment is due before class next Tuesday!
- Come to office hours/help sessions — they started yesterday (with an apology for
  a rescheduling mess-up). Come to discuss **final project ideas** as well as the
  assignments. Try to come early, often and off-cycle.
- TA office hours: 3-hour blocks Mon–Fri, with multiple TAs. Just show up.
  https://web.stanford.edu/class/cs224n/office_hours.html
- Instructor's office hours (in person by default): Monday 2–4pm, booked via
  Calendly, opening 2 weeks in advance. "I can't meet everyone, don't hog the
  slots!"

## Slide 4 — 2. Optimization: Gradient Descent

Identical in content to slide 37 of lecture 1: cost function J(θ) to minimize,
gradient descent as the algorithm, the idea being to compute the gradient and take
a small step in the direction of the negative gradient, repeatedly. Same convex
Cost-vs-θ figure with "Random initial value", "Learning step" arrows and the
"Minimum" at θ̂, and the same notes: "Our objectives may not be convex like this ☹
/ But life turns out to be okay ☺".

## Slide 5 — Gradient Descent (the update rule)

Identical to slide 38 of lecture 1.

> θ^new = θ^old − α ∇_θ J(θ)   (α = *step size* or *learning rate*)
>
> θ_j^new = θ_j^old − α ∂/∂θ_j^old J(θ)   (for each single parameter θ_j)

```python
while True:
    theta_grad = evaluate_gradient(J,corpus,theta)
    theta = theta - alpha * theta_grad
```

## Slide 6 — Stochastic Gradient Descent

Identical to slide 39 of lecture 1. J(θ) is a function of **all** windows in the
corpus (potentially billions), so ∇_θ J(θ) is very expensive; you would wait a
very long time before a single update; "**Very** bad idea for pretty much all
neural nets!". Solution: **stochastic gradient descent (SGD)** — repeatedly sample
windows and update after each one. Labelled "Mini Batch Gradient Descent".

```python
while True:
    window = sample_window(corpus)
    theta_grad = evaluate_gradient(J,window,theta)
    theta = theta - alpha * theta_grad
```

As in lecture 1, **no mini-batch size is given on the slide.**

## Slide 7 — 3. Review: Main idea of word2vec

- Start with random word vectors
- Iterate through each word position in the whole corpus
- Try to predict surrounding words using word vectors:
  P(o|c) = exp(u_oᵀv_c) / ∑_{w∈V} exp(u_wᵀv_c)
- **Learning:** Update vectors so they can predict actual surrounding words better
- Doing no more than this, this algorithm learns word vectors that capture well
  word similarity and meaningful directions in a word space!

The familiar `… problems turning into banking crises as …` strip with *banking* as
center word and probability arrows to each context position. A cartoon wizard in
the corner captioned **Magic!**

## Slide 8 — Word2vec parameters … and computations

Four matrix/vector diagrams left to right: **U** (outside) and **V** (center), both
drawn as matrices of dots; then **U · v₄ᵀ** labelled "dot product"; then
**softmax(U · v₄ᵀ)** labelled "probabilities".

- Boxed annotation: **"Bag of words" model!** pointing at → "The model makes the
  same predictions at each position"
- "We want a model that gives a reasonably high probability estimate to *all* words
  that occur in the context (at all often)"

## Slide 9 — Word2vec maximizes objective by putting similar words nearby in space

A large, dense 2-D scatter plot of a real word vector space with hundreds of words.
Legible clusters include: number words (*three*, *seven*, *eight*, *two*, *zero*,
*nine*) at upper left; first names (*gary*, *graham*, *michael*, *andrew*,
*david*, *richard*, *thomas*, *george*, *paul*, *peter*, *martin*, *raphael*) across
the top; days and months (*tuesday*, *saturday*, *christmas*, *summer*) at left;
academic subjects (*science*, *studies*, *study*, *biology*, *chemistry*,
*physics*, *mathematics*, *economics*, *statistics*, *calculus*) at right;
tech companies (*samsung*, *nokia*, *yahoo*, *google*, *koffice*, *openoffice*) at
lower left; and scientists (*newton*, *euler*, *archimedes*, *alhazen*) lower right.

## Slide 10 — 4. Word2vec algorithm family: More details

Subtitled "[Mikolov et al. 2013: 'Distributed Representations of Words and Phrases
and their Compositionality']".

**Why two vectors?** → Easier optimization. Average both at the end.
- But can implement the algorithm with just one vector per word … and it helps a bit

**Two model variants:**

1. **Skip-grams (SG)** — predict context ("outside") words (position independent)
   given center word
2. **Continuous Bag of Words (CBOW)** — predict center word from (bag of) context
   words

*We presented: **Skip-gram model***

**Loss functions for training:**

1. Naïve softmax (simple but expensive loss function, when many output classes)
2. More optimized variants like hierarchical softmax
3. Negative sampling

*So far, we explained **naïve softmax***

## Slide 11 — The skip-gram model with negative sampling

- The normalization term is computationally expensive (when many output classes) —
  the denominator of P(o|c) is annotated "A big sum over many words"
- Hence, in standard word2vec, you implement the skip-gram model with **negative
  sampling**
- Idea: train binary logistic regressions to differentiate a true pair (center word
  and a word in its context window) versus several "noise" pairs (the center word
  paired with a random word)

## Slide 12 — The skip-gram model with negative sampling [Mikolov et al. 2013]

- We take *K* negative samples (using word probabilities*)
- Maximize probability of real outside word; minimize probability of random words
- Using notation consistent with this class, we minimize:

> J_neg-sample(u_o, v_c, U) = − log σ(u_oᵀv_c) − ∑_{k ∈ {K sampled indices}} log σ(−u_kᵀv_c)

annotated "**sigmoid rather than softmax**".

- The logistic/sigmoid function: σ(x) = 1 / (1 + e^{−x}) — "(we'll become good
  friends soon)". Plotted as the familiar S-curve from −6 to 6, passing through 0.5
  at x = 0.
- \*Sample with **P(w) = U(w)^{3/4} / Z**, the unigram distribution U(w) raised to
  the 3/4 power
  - The power makes less frequent words be sampled a bit more often

## Slide 13 — Stochastic gradients with negative sampling [aside]

- We iteratively take gradients at each window for SGD
- In each window, we only have at most **2m + 1** words plus **2km** negative words
  with negative sampling, so ∇_θ J_t(θ) is very **sparse**!

Shows ∇_θ J_t(θ) ∈ ℝ^{2dV} as a tall column vector that is 0 almost everywhere,
with non-zero blocks only at ∇_{v_like}, ∇_{u_I}, and ∇_{u_learning}.

## Slide 14 — Stochastic gradients with negative sampling [aside] (continued)

- We might only update the word vectors that actually appear!
- Solution: either you need sparse matrix update operations to only update certain
  **rows** of full embedding matrices *U* and *V*, or you need to keep around a hash
  for word vectors
  - Boxed annotation: "Rows not columns in actual DL packages!"
- If you have millions of word vectors and do distributed computing, it is important
  to not have to send gigantic updates around!

Diagram: a |V| × *d* embedding matrix drawn as rows of dots.

## Slide 15 — 5. Why not capture co-occurrence counts directly?

"There's something weird about iterating through the whole corpus (perhaps many
times). Why don't we just accumulate all the statistics of what words appear near
each other??"

Building a co-occurrence matrix *X*:

- 2 options: windows vs. full document
- **Window:** similar to word2vec, use window around each word → captures some
  syntactic and semantic information ("word space")
- **Word-document** co-occurrence matrix will give general topics (all sports terms
  will have similar entries) leading to "Latent Semantic Analysis" ("document
  space")

## Slide 16 — Example: Window based co-occurrence matrix

- Window length 1 (more common: 5–10)
- Symmetric (irrelevant whether left or right context)
- Example corpus: "I like deep learning", "I like NLP", "I enjoy flying"

| counts | I | like | enjoy | deep | learning | NLP | flying | . |
| ------ | - | ---- | ----- | ---- | -------- | --- | ------ | - |
| **I** | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 |
| **like** | 2 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| **enjoy** | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| **deep** | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| **learning** | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| **NLP** | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| **flying** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| **.** | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 |

## Slide 17 — Co-occurrence vectors

- **Simple count co-occurrence vectors**
  - Vectors increase in size with vocabulary
  - Very high dimensional: require a lot of storage (though sparse)
  - Subsequent classification models have sparsity issues → Models are less robust
- **Low-dimensional vectors**
  - Idea: store "most" of the important information in a fixed, small number of
    dimensions: a dense vector
  - Usually **25–1000 dimensions**, similar to word2vec
  - How to reduce the dimensionality?

## Slide 18 — Classic Method: Dimensionality Reduction on X (HW1)

Singular Value Decomposition of co-occurrence matrix *X*. Factorizes *X* into
**UΣVᵀ**, where *U* and *V* are orthonormal (unit vectors and orthogonal).

Diagram: *X* on the left equals *U* · *Σ* · *Vᵀ*, with the retained rank-*k*
portions highlighted in blue and the discarded portions in yellow — a thin blue
column of *U*, the top-left corner of the diagonal *Σ*, and a blue row of *Vᵀ*.

- Retain only *k* singular values, in order to generalize.
- X̂ is the best rank *k* approximation to *X*, in terms of least squares.
- Classic linear algebra result. Expensive to compute for large matrices.

## Slide 19 — Hacks to X (several used in Rohde et al. 2005 in COALS)

- **Running an SVD on raw counts doesn't work well!!!**
- Scaling the counts in the cells can help ***a lot***
  - Problem: function words (*the, he, has*) are too frequent → syntax has too much
    impact. Some fixes:
    - log the frequencies
    - min(X, t), with t ≈ 100
    - Ignore the function words
- **Ramped windows** that count closer words more than further away words
- Use **Pearson correlations** instead of counts, then set negative values to 0
- Etc.

## Slide 20 — Interesting semantic patterns emerge in the scaled vectors

A 2-D plot from the COALS model. Verbs are plotted as open circles and their
agent nouns as filled dots, with green arrows drawn from verb to noun: DRIVE →
DRIVER, SWIM → SWIMMER, TEACH → TEACHER, MARRY → PRIEST. Other words plotted
include CLEAN, LEARN, TREAT, PRAY, JANITOR, STUDENT, DOCTOR, BRIDE. Crucially the
four arrows are all roughly parallel and of similar length.

> A meaning component (doer of event) becomes a linear meaning component in the
> space! This is the COALS model from Rohde et al. ms., 2005. *An Improved Model of
> Semantic Similarity Based on Lexical Co-Occurrence*

## Slide 21 — Encoding meaning components in vector differences

Subtitled "[GloVe: Pennington, Socher, and Manning, EMNLP 2014]", with a photo of
Jeffrey Pennington.

> Crucial insight: **Ratios of co-occurrence probabilities can encode meaning
> components.** We want to capture them as linear meaning components in a word
> vector space!

The schematic version of the table:

| | x = solid | x = gas | x = water | x = random |
| - | --------- | ------- | --------- | ---------- |
| P(x\|ice) | large | small | large | small |
| P(x\|steam) | small | large | large | small |
| P(x\|ice)/P(x\|steam) | large | small | ~1 | ~1 |

## Slide 22 — Encoding meaning components in vector differences (real numbers)

Same slide with actual measured co-occurrence probabilities. Note the fourth column
is *fashion* here rather than the schematic "random":

| | x = solid | x = gas | x = water | x = fashion |
| - | --------- | ------- | --------- | ----------- |
| P(x\|ice) | 1.9 × 10⁻⁴ | 6.6 × 10⁻⁵ | 3.0 × 10⁻³ | 1.7 × 10⁻⁵ |
| P(x\|steam) | 2.2 × 10⁻⁵ | 7.8 × 10⁻⁴ | 2.2 × 10⁻³ | 1.8 × 10⁻⁵ |
| P(x\|ice)/P(x\|steam) | **8.9** | 8.5 × 10⁻² | 1.36 | 0.96 |

## Slide 23 — GloVe: Encoding meaning components in vector differences

"[Pennington, Socher, and Manning, EMNLP 2014]"

> Q: How can we capture ratios of co-occurrence probabilities as linear meaning
> components in a word vector space?
>
> A: **Log-bilinear model:** w_i · w_j = log P(i|j)
>
> with vector differences: w_x · (w_a − w_b) = log [ P(x|a) / P(x|b) ]

**Loss:**

> J = ∑_{i,j=1}^{V} f(X_ij) ( w_iᵀ w̃_j + b_i + b̃_j − log X_ij )²

with a small plot of the weighting function *f*, which rises from 0, increases
roughly linearly, and then saturates flat at 1.0 beyond a cutoff.

- Fast training
- Scalable to huge corpora

## Slide 24 — 6. How to evaluate word vectors?

- A general concept of evaluation (in NLP): Intrinsic vs. extrinsic
- **Intrinsic:**
  - Evaluation on a specific/intermediate subtask
  - Fast to compute
  - Helps to understand that system
  - Not clear if really helpful unless correlation to real task is established
- **Extrinsic:**
  - Evaluation on a real task
  - Can take a long time to compute accuracy
  - Unclear if the subsystem is the problem or its interaction or other subsystems
  - If replacing exactly one subsystem with another improves accuracy → Winning!

## Slide 25 — Intrinsic word vector evaluation

- **Word Vector Analogies**: `a:b :: c:?`, e.g. `man:woman :: king:?`, evaluated as

> d = arg max_i [ (x_b − x_a + x_c)ᵀ x_i ] / ‖x_b − x_a + x_c‖

- Evaluate word vectors by how well their cosine distance after addition captures
  intuitive semantic and syntactic analogy questions
- **Discarding the input words from the search (!!!)**
- Problem: What if the information is there but not linear?

Small plot showing *man* → *woman* and *king* → (arrow) as two roughly parallel
vectors.

## Slide 26 — GloVe Visualization

A 2-D plot with dashed lines connecting male/female counterpart pairs, all roughly
parallel: brother–sister, nephew–niece, uncle–aunt, man–woman, sir–madam,
heir–heiress, king–queen, emperor–empress, duke–duchess, earl–countess.

## Slide 27 — Meaning similarity: Another intrinsic word vector evaluation

- Word vector distances and their correlation with human judgments
- Example dataset: **WordSim353**
  https://gabrilovich.com/resources/data/wordsim353/wordsim353.html

| Word 1 | Word 2 | Human (mean) |
| ------ | ------ | ------------ |
| tiger | cat | 7.35 |
| tiger | tiger | 10 |
| book | paper | 7.46 |
| computer | internet | 7.58 |
| plane | car | 5.77 |
| professor | doctor | 6.62 |
| stock | phone | 1.62 |
| stock | CD | 1.31 |
| stock | jaguar | 0.92 |

## Slide 28 — Correlation evaluation

Word vector distances and their correlation with human judgments, across five
similarity datasets (best in each column underlined; overall best in bold):

| Model | Size | WS353 | MC | RG | SCWS | RW |
| ----- | ---- | ----- | -- | -- | ---- | -- |
| SVD | 6B | 35.3 | 35.1 | 42.5 | 38.3 | 25.6 |
| SVD-S | 6B | 56.5 | 71.5 | 71.0 | 53.6 | 34.7 |
| SVD-L | 6B | 65.7 | 72.7 | 75.1 | 56.5 | 37.0 |
| CBOW† | 6B | 57.2 | 65.6 | 68.2 | 57.0 | 32.5 |
| SG† | 6B | 62.8 | 65.2 | 69.7 | 58.1 | 37.2 |
| GloVe | 6B | 65.8 | 72.7 | 77.8 | 53.9 | 38.1 |
| SVD-L | 42B | 74.0 | 76.4 | 74.1 | 58.3 | 39.9 |
| **GloVe** | **42B** | **75.9** | **83.6** | **82.9** | **59.6** | **47.8** |
| CBOW* | 100B | 68.4 | 79.6 | 75.4 | 59.4 | 45.5 |

This is the table Manning refers to when he says plain SVD works terribly while
SVD over scaled/log counts (SVD-S, SVD-L) already works reasonably.

## Slide 29 — Extrinsic word vector evaluation

- One example where good word vectors should help directly: **named entity
  recognition** — identifying references to a person, organization or location:
  "**Chris Manning** lives in **Palo Alto**."

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

"Discrete" is the symbolic baseline; GloVe is best or near-best on every column.

## Slide 30 — 7. Word senses and word sense ambiguity

- Most words have lots of meanings!
  - Especially common words
  - Especially words that have existed for a long time
- Example: **pike**
- Does one vector capture all these meanings or do we have a mess?

## Slide 31 — pike

- A sharp point or staff
- A type of elongated fish
- A railroad line or system
- A type of road
- The future (coming down the pike)
- A type of body position (as in diving)
- To kill or pierce with a pike
- To make one's way (pike along)
- In Australian English, pike means to pull out from doing something:
  - *I reckon he could have climbed that cliff, but he piked!*

## Slide 32 — Improving Word Representations Via Global Context And Multiple Word Prototypes (Huang et al. 2012)

- Idea: Cluster word windows around words, retrain with each word assigned to
  multiple different clusters bank₁, bank₂, etc.

A large 2-D word plot with the multi-prototype words boxed and subscripted.
Visible: **jaguar₁** near *luxury* and *convertible*; **jaguar₂** near *software*
and *microsoft*; **jaguar₃** near *keyboard*, *string*, *drum*, *bass*, *solo*,
*musical*; **jaguar₄** near *hunter*; **bank₁** near *finance*, *banking*,
*laundering*, *transaction*; also **attempt₁/attempt₂**, **approach₁/approach₂**,
**star₁/star₂**, and **canal₁**.

## Slide 33 — Linear Algebraic Structure of Word Senses, with Applications to Polysemy (Arora, …, Ma, …, TACL 2018)

- Different senses of a word reside in a **linear superposition** (weighted sum) in
  standard word embeddings like word2vec
- v_pike = α₁ v_pike₁ + α₂ v_pike₂ + α₃ v_pike₃
- Where α₁ = f₁ / (f₁ + f₂ + f₃), etc., for frequency *f*
- **Surprising result:** Because of ideas from *sparse coding* you can actually
  separate out the senses (providing they are relatively common)!

The recovered senses of **tie**, as five columns of nearest words:

| | | | | |
| - | - | - | - | - |
| trousers | season | scoreline | wires | operatic |
| blouse | teams | goalless | cables | soprano |
| waistcoat | winning | equaliser | wiring | mezzo |
| skirt | league | clinching | electrical | contralto |
| sleeved | finished | scoreless | wire | baritone |
| pants | championship | replay | cable | coloratura |

That is: the necktie/clothing sense, the league-standings sense, the drawn-game
sense, the cable-tie sense, and the musical-tie sense.

## Slide 34 — 8. Deep Learning Classification: Named Entity Recognition (NER)

- The task: **find** and **classify** names in text, by labeling word tokens, for
  example:

```
Last night , Paris Hilton wowed in a sequin gown .
              PER   PER

Samuel Quinn was arrested in the Hilton Hotel in Paris in April 1989 .
PER    PER                       LOC    LOC      LOC    DATE DATE
```

- Possible uses:
  - Tracking mentions of particular entities in documents
  - For question answering, answers are usually named entities
  - Relating sentiment analysis to the entity under discussion
- Often followed by **Entity Linking/Canonicalization** into a Knowledge Base such
  as Wikidata

## Slide 35 — Simple NER: Window classification using binary logistic classifier

- **Idea:** classify each word in its context window of neighboring words
- Train logistic classifier on hand-labeled data to classify center word {yes/no}
  for each class based on a **concatenation of word vectors** in a window
  - "Really, we usually use multi-class softmax, but we're trying to keep it simple ☺"
- **Example:** Classify "Paris" as +/− location in context of sentence with window
  length 2:

```
the   museums   in   Paris   are   amazing   to   see   .

X_window = [ x_museums  x_in  x_Paris  x_are  x_amazing ]ᵀ
```

- Resulting vector x_window = **x ∈ ℝ^{5d}**
- To classify all words: run classifier for each class on the vector centered on
  each word in the sentence

## Slide 36 — Classification review and notation

- Supervised learning: we have a **training dataset** consisting of **samples**
  {x_i, y_i}^N_{i=1}
- x_i are **inputs**, e.g. words (indices or vectors!), sentences, documents, etc.
  Dimension *d*.
- y_i are **labels** (one of *C* classes) we try to predict, for example:
  - classes: sentiment (+/−), named entities, buy/sell decision
  - {location, not-location}
  - other words
  - later: multi-word sequences

## Slide 37 — Neural classification

- Typical ML/stats softmax classifier: p(y|x) = exp(W_y · x) / ∑_{c=1}^{C} exp(W_c · x)
- Learned parameters θ are just elements of *W* (not input representation *x*,
  which has sparse symbolic features)
- Classifier gives **linear decision boundary**, which can be limiting
- A **neural network classifier** differs in that:
  - We learn **both** *W* **and (distributed!) representations** for words
  - The word vectors *x* re-represent one-hot vectors, moving them around in an
    intermediate layer vector space, for easy classification with a (linear) softmax
    classifier
    - Conceptually, we have an embedding layer: *x = Le*
  - We use deep networks — more layers — that let us re-represent and compose our
    data multiple times, giving a non-linear classifier

Two figures of a red/green 2-class scatter: the upper one separated by a single
straight line (much of it misclassified), the lower one carved up by a wiggly
non-linear boundary that fits the data. Side note: "But typically, it is linear
relative to the pre-final layer representation".

## Slide 38 — NER: Binary neural classifier for center word being location

We do supervised training and want high score if it's a location.

> J_t(θ) = σ(s) = 1 / (1 + e^{−s})    ← "predicted model probability of class"
>
> s = **u**ᵀ**h**
>
> **h** = f(**W** **x** + **b**)
>
> **x** (input)

with the sigmoid curve plotted, and the annotation "*f* = Some element-wise
non-linear function, e.g., logistic, tanh, ReLU".

Network diagram bottom to top: the input row of 5 word-vector blocks
`x = [x_museums x_in x_Paris x_are x_amazing]`, up into a narrower hidden layer
**h**, then to a single scalar, then to the output probability.

## Slide 39 — Training with "cross entropy loss"

- Until now, our objective was stated as to **maximize the probability of the
  correct class y** or equivalently to **minimize the negative log probability of
  that class** on training data
- Now restated in terms of **cross entropy**, a concept from **information theory**
- Let the true probability distribution be *p*; let our computed model probability
  be *q*
- The cross entropy is:

> H(p, q) = − ∑_{c=1}^{C} p(c) log q(c)

- Assuming a ground truth (or true or gold or target) probability distribution that
  is 1 at the right class and 0 everywhere else, *p* = [0, …, 0, 1, 0, …, 0], then:
- **Because of one-hot *p*, the only term left as our loss function is the negative
  log probability of the true class y_i: −log p(y_i|x_i)**

> **Use this in PyTorch! `torch.nn.CrossEntropyLoss()`**

with the note "Cross entropy can be used in other ways with a more interesting *p*,
but for now just know that you'll want to use it as the loss in PyTorch".

## Slide 40 — 9. Neural computation

A line drawing of three biological neurons, labelled with Dendrites, Soma, Axon,
Myelin sheath, and Terminal button, showing the axon of each neuron terminating on
the dendrites of the next.

## Slide 41 — A binary logistic regression unit is a bit similar to a neuron

"*f* = nonlinear activation function (e.g. sigmoid), *w* = weights, *b* = bias,
*h* = hidden, *x* = inputs"

> h_{w,b}(x) = f(wᵀx + b)
>
> f(z) = 1 / (1 + e^{−z})

Annotation on *b*: "We can have an 'always on' bias feature, which gives a class
prior, or separate it out, as a bias term". Diagram: inputs x₁, x₂, x₃ and a +1
bias unit feeding one circular unit that outputs h_{w,b}(x), with the note "*w*, *b*
are the parameters of this neuron i.e., this logistic regression model". The sigmoid
curve is plotted alongside.

## Slide 42 — A neural network = running several logistic regressions at the same time

"If we feed a vector of inputs through a bunch of logistic regression functions,
then we get a vector of outputs …"

Diagram: inputs x₁, x₂, x₃ and +1 (Layer L₁) fully connected to three output units.

> *But we don't have to decide ahead of time what variables these logistic
> regressions are trying to predict!*

## Slide 43 — A neural network = running several logistic regressions (composed)

"We can feed them into another logistic regression function, giving composed
functions"

Diagram: Layer L₁ (x₁, x₂, x₃, +1) → Layer L₂ (three units a₁⁽²⁾, a₂⁽²⁾, a₃⁽²⁾ and
+1) → Layer L₃, a single unit outputting h_{W,b}(x).

> *It is the final loss function that will direct what the intermediate hidden
> variables should be, so as to do a good job at predicting the targets for the next
> layer, etc.*

## Slide 44 — A neural network = running several logistic regressions (multilayer)

"Before we know it, we have a multilayer neural network…."

Diagram: four layers, L₁ (inputs) → L₂ → L₃ → L₄, the last emitting two outputs.

> *This allows us to re-represent and compose our data multiple times and to learn a
> classifier that is highly non-linear in terms of the original inputs* (but
> typically is linear in terms of the pre-final layer representations)

## Slide 45 — Matrix notation for a layer

We have

> a₁ = f(W₁₁x₁ + W₁₂x₂ + W₁₃x₃ + b₁)
> a₂ = f(W₂₁x₁ + W₂₂x₂ + W₂₃x₃ + b₂)
> etc.

In matrix notation

> z = Wx + b
> a = f(z)

Activation *f* is applied element-wise:

> f([z₁, z₂, z₃]) = [f(z₁), f(z₂), f(z₃)]

Diagram of one layer with W₁₂ and b₃ labelled on the corresponding edges.

## Slide 46 — Non-linearities (like f or sigmoid): Why they're needed

- Neural networks do function approximation, e.g. regression or classification
  - Without non-linearities, deep neural networks can't do anything more than a
    linear transform
  - Extra layers could just be compiled down into a single linear transform:
    W₁ W₂ x = Wx
  - But, with more layers that include non-linearities, they can approximate more
    complex functions!

Figures: the same red/green scatter with a linear boundary versus a curved one; and
three regression panels fitting the same scattered points with progressively more
flexible curves — a single sigmoid step, a smooth sine-like curve, and a wiggly
curve that interpolates nearly every point.

## Slide 47 — (blank)

Blank final page.
