# Word senses and polysemy

Most words mean several things, but word2vec and GloVe learn exactly one vector per
word. This page covers what that costs, the approaches to fixing it, and Manning's
argument that the obvious fix is based on a wrong model of how word meaning
actually works. Raised in questions during
[lecture 1](01-intro-and-word-vectors.md) (≈41:09, ≈45:48) and treated properly in
[lecture 2](02-word-vectors-and-language-models.md) (≈52:03–1:02:49).

Slides: [wordvecs2](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture02-wordvecs2.pdf)

## The scale of the problem

Manning's claim is that almost every word has many meanings, and the words that do
not are mostly very specialized scientific terms (lecture 2, ≈52:03). The famous
example is *bank* — the financial institution and the side of a river — and *star*
— the astronomical object and the celebrity.

His set piece is the word **pike**, which he uses because it is a word you do not
think of as ambiguous (≈52:49). The class comes up with, and he adds to:

- a kind of fish (slide 31: "a type of elongated fish")
- a spear — the medieval weapon (slide 31: "a sharp point or staff")
- a road — *pike* as shorthand for *turnpike*, so named because of the spiked
  barrier originally used to count people through
- a position in diving and gymnastics
- a verb, as in to attack someone with a pike (slide 31: "to kill or pierce with a
  pike")
- "coming down the pike", a metaphorical extension of the road sense that ends up
  meaning the future
- an Australian usage meaning to chicken out of something (≈55:07). Slide 31 gives the
  example sentence: *"I reckon he could have climbed that cliff, but he piked!"*

**Slide 31** lists two more the class did not produce: **a railroad line or system**,
and **to make one's way** ("pike along") — nine senses in total.

## What one vector per word actually gives you

A student asks in lecture 1 whether the similarity captured is context-dependent,
and Manning concedes it directly (≈41:56): the course is learning one vector per
word string, which absolutely does not capture meaning in context. The vector for
*star* will not tell you whether a given occurrence means a Hollywood star or an
astronomical one.

But the consolation is genuinely surprising, and follows from how high-dimensional
spaces behave (≈42:43): the single vector for *star* ends up close **both** to
astronomical words like *nebula* **and** to words like *celebrity*. It does not have
to choose, because in a high-dimensional space a point can be near different things
along different dimensions. This is the same point made in
[distributional semantics](distributional-semantics.md) about high-dimensional
geometry.

Asked directly whether each word has one embedding or several, Manning says one per
string of letters, and that you can think of it as an **average over the word's
senses** (lecture 1, ≈45:48) — which he then makes precise in lecture 2.

## Approach 1: learn a vector per sense

The direct fix: accept that words have several meanings, cluster the occurrences of
each word by the similarity of their contexts, treat each cluster as a sense, and
learn a separate vector for each (lecture 2, ≈54:20). This was done at Stanford in
2012, before word2vec — **slide 32** cites it as *Improving Word Representations Via
Global Context And Multiple Word Prototypes* (Huang et al. 2012), whose method is to
cluster the word windows around a word and retrain with that word assigned to
multiple clusters: bank₁, bank₂, and so on.

It works well. Manning shows *bank₁* and *bank₂* separated, and *jaguar* split four
ways (≈55:52):

- *jaguar₁* — the car, near *luxury* and *convertible*
- *jaguar₂* — near *software* and *Microsoft*, because Apple named a version of Mac
  OS X "Jaguar"
- *jaguar₃* — near *keyboard*, *solo*, *drum*, *bass*, because there is a Jaguar
  keyboard
- *jaguar₄* — the animal, near *hunter*, and Manning notes this sense that we think
  of as basic actually turns up rather *less* in text corpora

## Approach 2 (what is actually done): superposition

Despite that working, it is not what is done now (lecture 2, ≈57:26). Instead you
keep **one** vector per word, and what you learn is a weighted average of the sense
vectors you would have learned — weighted by the relative frequency of each sense.

This is commonly called a **superposition**. Manning's aside is that neural network
people like borrowing physics vocabulary, but it is simply a weighted average
(≈57:26).

## Why the sense model is the wrong model anyway

Manning then makes the linguistic argument that the sense-based view is broken
regardless (≈58:12). It is how dictionaries are built — sense 1, sense 2, sense 3 —
but ask how many senses a word has and every dictionary gives a different answer.

His example is *field* (≈59:43). A field can be:

- a place where you grow a crop
- a natural expanse — a rock field, an ice field
- a sporting field
- the mathematical sense

All of these have something to do with each other. The mathematical one is furthest
away; the physical ones are all sort of flat spaces. The sporting sense is clearly
different from the ice sense. But is an ice field really a *different sense* from a
rock field, or are you just modifying the same sense with a different material?
There is no principled answer, which is the point.

Cases like *bank* — where financial institution and river edge are genuinely far
apart — are the extreme, not the norm. Most words have meanings that shade into each
other, and cutting them into discrete senses is artificial. What you really have,
in Manning's framing, is something like a **probability density over the things a
word can mean** (≈1:00:29). Under that view, a vector that averages over contexts is
the more honest representation, not a compromise — and it points directly toward the
contextual word vectors covered later in the course.

## The sparse coding coda

There is a genuinely surprising result here (lecture 2, ≈1:00:29). Since the word
vector is the sum of sense vectors, standard linear algebra says you cannot recover
the individual senses from the single vector — the information is gone.

But because these vector spaces are so high-dimensional and sparse, ideas from
**sparse coding theory** can reconstruct the sense vectors from the one word vector —
provided the senses are relatively common.

**Slide 33** attributes this to *Linear Algebraic Structure of Word Senses, with
Applications to Polysemy* (Arora, …, Ma, …, TACL 2018) and writes the superposition
out precisely:

$$v_{\text{pike}} = \alpha_1 v_{\text{pike}_1} + \alpha_2 v_{\text{pike}_2} + \alpha_3 v_{\text{pike}_3}, \qquad \alpha_1 = \frac{f_1}{f_1 + f_2 + f_3}$$

and so on, where $f_i$ is the frequency of sense $i$ — which is exactly the
frequency-weighted average described above, made explicit.

The recovered senses of **tie** (≈1:02:03, slide 33) come out as five cleanly separated
columns of nearest words:

| clothing | league standings | drawn game | cable tie | musical |
| -------- | ---------------- | ---------- | --------- | ------- |
| trousers | season | scoreline | wires | operatic |
| blouse | teams | goalless | cables | soprano |
| waistcoat | winning | equaliser | wiring | mezzo |
| skirt | league | clinching | electrical | contralto |
| sleeved | finished | scoreless | wire | baritone |
| pants | championship | replay | cable | coloratura |

Manning points anyone who wants to understand the mechanism toward sparse coding
theory taught in Stanford's statistics department (≈1:01:15), and does not attempt to
teach it.

## How the course resolves it

Lecture 9 returns to this and supplies the answer the lecture-2 discussion points toward:
stop trying to give a word one vector at all. Its example is the minimal pair *I **record**
the **record***, where the verb and the noun are the same string. A
[word2vec](word2vec.md) embedding blends the two senses and cannot specialize; a pretrained
**contextual** model computes a different vector for each occurrence, from the sequence it
appears in (lecture 9, ≈16:15, ≈18:35).

The lecture grounds this in a Firth quote earlier than the famous one — "the complete meaning
of a word is always contextual, and no study of meaning apart from a complete context can be
taken seriously" (Firth 1935) — and makes it the motivation for pretraining whole models
rather than just embeddings. Note that one static embedding per vocabulary item still exists:
it is the model's *input*, and the contextual vectors are what the
[Transformer](transformer.md) computes on top of it (lecture 9, ≈27:52). See
[pretraining and fine-tuning](pretraining-and-finetuning.md) and [BERT](bert.md).

## Related pages

- [word2vec](word2vec.md) — the one-vector-per-type model
- [distributional semantics](distributional-semantics.md) — why high-dimensional
  spaces let one vector sit near several meanings
- [pretraining and fine-tuning](pretraining-and-finetuning.md) — contextual representations,
  which is how this is actually solved
- [BERT and masked language modeling](bert.md) — a pretrained encoder producing one vector per
  occurrence
- [lecture 2](02-word-vectors-and-language-models.md) — the pike discussion and
  the jaguar clustering results
- [lecture 9 — Pretraining](09-pretraining.md) — the *record* example and the Firth 1935 quote
