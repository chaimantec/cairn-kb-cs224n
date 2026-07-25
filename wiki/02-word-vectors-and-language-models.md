# Lecture 2 — Word Vectors and Language Models

This lecture finishes word vectors and opens the door to neural networks. It
completes the optimization story from lecture 1 (stochastic gradient descent,
random initialization), fills in the parts of word2vec that were skipped
(skip-gram versus CBOW, negative sampling, how to sample the negatives), then
takes the alternative route to word vectors through co-occurrence counts and SVD,
which leads to GloVe. It then covers how to tell whether word vectors are any
good, what to do about words with many meanings, and ends by building the
smallest useful neural classifier — a window classifier for named entities — as
the bridge into the rest of the course.

**Slide-by-slide text of this deck: [47 slides](../raw/slides/02-word-vectors-and-language-models.md)**
— cite slide numbers from there; the printed slide number equals the PDF page number.

Slides PDF: [Lecture 2 — wordvecs2](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture02-wordvecs2.pdf) ·
Notes: [2019 notes 02 — wordvecs2](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/readings/cs224n-2019-notes02-wordvecs2.pdf) ·
[Full transcript](../raw/transcripts/02-word-vectors-and-language-models.md)

The deck's own title is "Word Vectors, Word Senses, and Neural Classifiers" (the
YouTube video is titled "Word Vectors and Language Models"). Its plan (slide 2) runs:
course organization (3 mins), optimization basics (5), review of word2vec and looking
at word vectors (12), more on word2vec (8), can we capture word meaning more
effectively by counting? (12), evaluating word vectors (10), word senses (10), review
of classification and how neural nets differ (10), introducing neural networks (10).
Slide 2's stated goal: "To be able to read and understand word embeddings papers by
the end of class."

## Finishing the optimization story

Manning opens by admitting the previous lecture's handwritten derivation went
badly and pointing students at the cleanly written version on the website
(≈3:09). The loop he did not close: you have a cost function, you compute its
gradient to find which direction is downhill, you take a small step that way, and
you repeat. In one dimension this is trivially "walk downhill"; in many dimensions
the gradient at different points can point in quite different directions, which
is why the calculus is necessary (≈3:58).

Plain **gradient descent** subtracts a small multiple of the gradient from the
parameters, where the multiplier α is the step size or learning rate — typically
something like 10⁻³ to 10⁻⁵ (≈4:46). The learning rate must be small because a
large step can overshoot the minimum entirely and land somewhere worse than where
you started.

But plain gradient descent is never actually used (≈5:33). Computing the objective
over the entire training set before taking a single step is far too expensive.
Instead neural networks use **stochastic gradient descent**: sample a small
subset of the data, pretend it is the whole dataset, and use its noisy, inaccurate
gradient as the direction to walk. (The captions garble the mini-batch size Manning
quotes at ≈6:21 — they read "16 or 2" — and slide 6 gives no number, so the figure
is not recoverable from this course's materials.) This is also called mini-batch
gradient descent. Manning's observation is that SGD is not merely a faster
approximation: neural networks often work *better* with noise in the system,
because the jiggle moves things around, so SGD wins on both speed and quality
(≈7:07–7:53). See [gradient descent](gradient-descent.md).

One practical detail that matters (≈7:53): you must initialize word vectors with
small **random** numbers. Initializing everything to zero means nothing learns,
because identical starting values create symmetries the algorithm cannot break.

## What word2vec actually is

The only parameters are the outside and center word vectors, treated as disjoint
(≈10:12). At each position the model takes dot products between the center vector
and candidate outside vectors, softmaxes them into a distribution over the
vocabulary, and compares that to the word actually present — the difference is
the error signal.

This makes word2vec a **bag of words** model (≈10:58): it has no knowledge of
sentence structure, and does not even distinguish left from right. It predicts the
same probability for a word regardless of where in the window it sits. All it
knows is which words tend to appear in the neighbourhood of which other words.

### Why two vectors per word

A question from lecture 1 gets answered properly at ≈25:49. If a word used the
same vector as center and outside word, then while summing over every candidate
outside word for the softmax normalization you would eventually hit the center
word itself, producing a quadratic term — `exp` of the vector dotted with itself —
where every other term is linear. It is not intractable, just messier, so the
authors kept the two sets disjoint. Manning notes it actually works slightly
better to tie them, but in practice people estimate them separately and average
the two at the end (≈21:57). The two end up very close anyway, because as you
slide along the text every pair occurs in both configurations — *octopus* as
center with *legs* in context, then a few steps later *legs* as center with
*octopus* in context (≈22:42).

### Skip-gram, CBOW, and negative sampling

The 2013 paper describes a family of methods (≈26:34). Predicting the surrounding
words from the center word is **skip-gram** — the version presented in these
lectures, simpler and works fine. Predicting the center word from all the context
words is **continuous bag of words** (CBOW).

Separately there is the question of the loss used for training. What lecture 1
presented is **naive softmax**, summing over every word in the vocabulary. With
400,000 words, each requiring a dot product of 100- or 300-dimensional vectors
followed by an exponentiation, that is a lot of arithmetic — quite doable on
modern hardware, but expensive when the paper was written (≈27:20). The paper
therefore considers alternatives: hierarchical softmax, which Manning skips, and
**negative sampling**, which he explains (≈28:06).

The idea of skip-gram with negative sampling is to replace the full normalization
with a handful of simple logistic regressions: the true context word should score
high, and a few randomly chosen words should score low. The softmax is replaced by
the **logistic (sigmoid) function**, which maps any real number to a probability
between 0 and 1. Manning notes "sigmoid" only means s-shaped and there are
infinitely many such functions — the one actually used is the logistic function
(≈29:37). The negative terms exploit the symmetry of the logistic function:
negating the argument flips you to the other side of the curve, so the same
expression pushes the sampled words toward low probability (≈30:25).

How to choose the negatives matters (≈31:10). Sampling uniformly from all 400,000
words is wrong, because word frequency varies enormously — you want the **unigram
distribution**, roughly how common each word is, under which you would pick *the*
about 10% of the time. But the standard word2vec recipe does something better:
raise the unigram probability to the power of **3/4**. Manning puts the question
to the room and confirms the answer (≈32:41): this somewhat raises the probability
of less frequent words, moving partway from true relative frequency toward uniform
without going all the way. Going all the way would correspond to an exponent of
zero.

Full details on [word2vec](word2vec.md).

## The demo: what the vectors actually know

Manning loads word vectors in a Jupyter notebook using gensim — a package the
course does not otherwise use, but convenient for playing with vectors (≈11:45).
He flags that these are 100-dimensional **GloVe** vectors rather than strictly
word2vec vectors, though they behave the same way.

Comparing *bread* and *croissant* componentwise, the first components are both
negative, the second both positive, the third both negative and large, the fourth
both positive — visibly similar vectors (≈12:32). Using `most_similar`: *USA*
returns Canada, America, U.S.A, United States, Australia, with Manning noting it
is a little odd that Canada beats the accented spelling of the USA; *banana*
returns coconut, mango, bananas, potato, pineapple, fruit, with a mild bias toward
tropical fruit; *croissant* returns not *bread* but brioche, baguette and focaccia
(≈13:18–14:06).

Asking for the words most similar to the **negative** of a vector is not useful
(≈14:52). You might hope for antonyms; what you get is junk — strings you are not
sure are words at all, or names in other languages.

What does work is **analogies** (≈15:38). Start at the origin, add the vector for
*king*, subtract *man*, add *woman*, and ask for the nearest word: you get
*queen*. This was the most celebrated discovery about word vectors — that they
support meaning *components*, not just similarity — and it is read as "a is to b
as c is to what". Manning has some fun with it (≈17:58): Australia is to beer as
Russia is to *vodka*; pencil is to sketching as camera is to *photographing*.
Because the model was built in 2014 it cannot do recent politics, but Obama is to
Clinton as Reagan is to — the room guesses Bush, and the model answers *Nixon*,
which Manning thinks is fair (≈19:34). Syntactic relations work too, such as
tall/tallest/long. He is candid about how dated this is: "basically no one uses
this stuff anymore", but it was stunning at the time that so simple a model on so
simple a signal captured this much semantics, including cultural world knowledge
that goes beyond what most people would call word meaning (≈17:58, ≈21:08).

A student asks whether the difference between two vectors captures a relation like
"ruler" (≈21:08). Manning agrees: depending on which pairing you treat as the
analogy, the difference vector between *man* and *king* is a ruler relation, and
the one between *man* and *woman* is a gender relation.

## Counting instead of predicting

Having built a prediction model, Manning asks the obvious question: why not just
count? We have words and words in their contexts, so make a matrix of how often
each word occurs in the context of each other word (≈33:28). With the toy corpus
"I like deep learning / I like NLP / I enjoy flying" and a one-word window, every
cell is 0 or 1 except where *I like* occurs twice. Each row is then a word vector.

People have done exactly that, but the vectors are ungainly (≈35:01): with a
400,000-word vocabulary the matrix is 400,000 × 400,000, far worse than
400,000 × 100. So the standard move is to reduce the dimensionality, which points
at PCA and, for a matrix of any shape, the **singular value decomposition**
(≈36:35). SVD rewrites the matrix as a product of three matrices, two
orthonormal and one diagonal of singular values ordered by size. Zeroing out the
smallest singular values leaves a low-dimensional representation of each word
(≈38:08).

This was explored well before neural word vectors, under the name **latent
semantic analysis**, and it half worked but never worked well (≈38:53). A student
correctly points out that the example matrix should be square if it is
vocabulary-by-vocabulary; Manning agrees, noting the non-square case is
words-versus-documents, which he skipped (≈45:07).

The thread was kept alive in the early 2000s by a graduate student, Doug Rohde,
whose algorithm — COALS in the transcript's garbled rendering — improved on raw
counts by taking logs of frequencies, ramping the window so nearer words count for
more, and using Pearson correlations instead of counts (≈38:53–39:39). Manning's
aside is that Rohde effectively discovered linear semantic components before
word2vec did: a figure in his dissertation shows "doer of an event" as a linear
direction you can follow from a verb to the noun for the person who does it. He
did not become famous, because nobody was paying attention (≈40:27).

That thread is what led to GloVe, built at Stanford with Jeffrey Pennington
(≈41:13). See [distributional semantics](distributional-semantics.md) for the
count-based lineage and [GloVe](glove.md) for the model itself and the
ice/steam derivation at ≈42:00.

## Evaluating word vectors

Manning introduces a distinction that recurs all course long (≈45:52).
**Intrinsic** evaluation scores a specific internal subtask: fast to compute,
helps you understand the component, but distant from what you actually care about,
so improving it may or may not help. **Extrinsic** evaluation runs the whole
system on a real task — question answering, summarization, machine translation —
and measures downstream accuracy: what you actually want to know, but slow and
harder to attribute.

He admits he cheated in the demo by only showing analogies that work (≈48:09).
The honest intrinsic evaluations are a scored set of analogies, and word
similarity against **human judgments**: psychologists ask undergraduates to rate
pairs from 0 to 10 and average the answers, giving tiger/tiger 10, book/paper
7.46, plane/car 5.77, stock/phone 1.62, and stock/jaguar 0.92 (≈48:57). Models are
scored by how well their similarity judgments correlate with those. The results
table shows plain SVD working terribly, SVD over log counts already reasonable,
then CBOW and skip-gram, then GloVe (≈50:29).

For extrinsic evaluation the lecture uses **named entity recognition** —
identifying names and their types, so that in "Chris Manning lives in Palo Alto"
you tag *Chris Manning* as a person and *Palo Alto* as a place (≈51:17). Adding
word vectors to a discrete symbolic baseline NER system measurably raises the
numbers. Details on [evaluating word vectors](evaluating-word-vectors.md).

## Words have many meanings

Manning's set piece here is the word **pike** (≈52:49). The class produces: a
fish, a spear, a road (as in turnpike, so called because of the spiked barrier
used to count people through), a diving and gymnastics position, and a verb
meaning to attack with a pike. He adds "coming down the pike" as a metaphorical
extension of the road sense that ends up meaning the future, and an Australian
usage meaning to chicken out of something. His broader claim is that almost every
word has many meanings; only very specialized scientific terms do not (≈52:03).

One response is to give each sense its own vector: cluster the occurrences of a
word by context similarity, treat each cluster as a sense, and learn vectors for
those (≈54:20). Stanford did this in 2012, before word2vec. It works — *jaguar₁*
lands near luxury and convertible (the car), *jaguar₂* near software and Microsoft
(because Apple named a version of Mac OS Jaguar), *jaguar₃* near keyboard and
musical terms (there is a Jaguar keyboard), and *jaguar₄*, the animal sense, turns
up near *hunter* and is actually the rarest in text (≈55:52–57:26).

But that is not what is done now (≈57:26). Instead you keep one vector per word,
and what you learn is a weighted average of the sense vectors you would have
learned, weighted by the relative frequency of each sense. This is often called a
**superposition**, which Manning notes is physics vocabulary for what is simply a
weighted average.

He then argues the sense model is the wrong model anyway (≈58:12). Dictionaries
are built on it, but ask how many senses a word has and every dictionary gives a
different answer. Take *field*: a place where a crop grows, a rock field, an ice
field, a sporting field, the mathematical sense. These all have something to do
with each other; the sporting sense is clearly distinct from the ice sense, but is
an ice field really a different sense from a rock field, or just a different
modifier? What you really have is something like a probability density over
possible meanings, which makes an averaged vector a more honest model — and
points toward the contextual word vectors covered later in the course.

The surprising coda (≈1:00:29): standard linear algebra says you cannot recover
the individual sense vectors from their sum, but because these spaces are so high
dimensional and sparse, **sparse coding** can reconstruct them. Manning shows a
result where sense vectors for *tie* are recovered from the single word vector —
the clothing sense, the tied-game sense, the cable-tie sense, and the musical
sense, four or five out of five correct. He points to sparse coding theory taught
in Stanford's statistics department for anyone who wants to understand why. (The
transcript garbles the names of both the paper's authors and the statistician he
recommends, so this page does not attribute them.) See
[word senses and polysemy](word-senses-and-polysemy.md).

## The first neural classifier

The last fifteen minutes start on neural networks proper (≈1:02:49). Manning notes
word2vec is already a simple neural classifier — it predicts which words occur in
context — but uses NER as a cleaner example.

The task: label a word with its class, where the same words can have different
classes. *Paris Hilton* is a person; *Paris* in "I visit Paris every spring" is a
location (≈1:03:37). A third state is "not a named entity at all". In practice NER
is usually followed by **entity linking**, mapping the found entity to a canonical
form such as a Wikipedia page, which the lecture does not cover (≈1:04:22).

The architecture is a **window classifier** (≈1:05:07): take the word plus a
couple of words of context each side, look up their word vectors, and feed the
result into a classifier that decides location or not-location. This is supervised
learning — labelled examples `xᵢ` with classes `yᵢ`, where "I love Paris Hilton
greatly" is negative and "I visit Paris every spring" is positive, and it is
always the middle word being classified (≈1:06:39). He keeps it to two classes
deliberately, since the same example is reused in the next lecture.

The important contrast is with conventional classifiers (≈1:08:13). Logistic
regression, softmax classifiers, SVMs, naive Bayes — these are almost all
**linear** classifiers: you learn weights W, the inputs are fixed, and the
decision boundary is linear. A **neural** classifier learns the weights *and* the
distributed representations of the words, so it can move the words around in the
space. The consequence (≈1:09:45) is that the classifier is linear in terms of the
final re-represented vectors but nonlinear in terms of the original input space,
which lets it express much more complex functions.

Concretely (≈1:09:45), for "museums in Paris are amazing": look up the five word
vectors and concatenate them, giving 500 dimensions if the vectors are
100-dimensional. Multiply by a matrix and add a bias, then apply a nonlinearity
such as the logistic function — if W is 8 × 500 the result is a much smaller
hidden vector. Multiply that by another vector to get a score, and push the score
through the logistic function to get the probability that the middle word is a
location.

One forward-looking note (≈1:11:19): everything so far has been framed as log
likelihood and negative log likelihood, but in PyTorch you will use
**cross-entropy loss**. Cross entropy comes from information theory and compares a
true distribution p against your model's distribution q. In the special case where
the labels are one-hot — probability 1 for the right class, 0 elsewhere — every
term in the sum vanishes except the log probability your model assigns to the
correct class, which is exactly the log likelihood. See
[softmax and cross-entropy](softmax-and-cross-entropy.md).

Manning closes with his one obligatory slide of real neurons (≈1:13:43). A neuron
has many inputs from other neurons via synapses, combines them in the cell body,
and fires down a single axon if there is enough positive activation; real neurons
signal by spike rate, which artificial networks immediately reduced to a single
real-valued activation level. Binary logistic regression is a fair model of this:
inputs that are excitatory or inhibitory, summed into a level of excitation, then
passed through a nonlinearity (≈1:16:05). Neuroscientists reasonably object that
real neurons are far more complex, but the field has stuck with the simple version
because it turns into linear algebra cleanly. The step to a network is that
instead of one logistic regression you have many at once, and rather than having
to specify what each computes, you feed them into another logistic regression and
let learning work out something useful for the intermediate units to do
(≈1:17:42). That self-organization is the magic, and stacking more layers lets the
network re-represent its input in ways that make the final classification easier
(≈1:18:28).

## A note on source quality

As in lecture 1, the auto-generated captions mangled technical vocabulary
consistently: *word2vec* as "word de", "word DEC" and "watch ve"; *CBOW* as "sibo";
*GloVe* as "glav"; *COALS* as "Kohl's"; *Doug Rohde* as "Doug roie"; *Pearson* as
"piercon"; *brioche, baguette, focaccia* as "Brios bagette fatcha"; and the sampling
exponent 3/4 as "34".
[The transcript](../raw/transcripts/02-word-vectors-and-language-models.md) has been
copy-edited into readable sentences with those restored, and student questions marked;
the raw captions are kept at
[`original/`](../raw/transcripts/original/02-word-vectors-and-language-models.md).

The citations the captions destroyed have been recovered **from the slides**, not
guessed: Rohde et al. 2005 / COALS (slide 19–20), GloVe as Pennington, Socher and
Manning EMNLP 2014 (slides 21–23), Huang et al. 2012 for multi-prototype senses
(slide 32), and Arora et al. TACL 2018 for the sparse-coding result (slide 33).

One thing remains unrecoverable: the **mini-batch size** Manning quotes aloud at
≈6:21 comes through as "16 or 2", and slide 6 gives no number either, so this KB does
not state one.
