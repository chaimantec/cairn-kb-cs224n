# Lecture 1 — Intro and Word Vectors

This lecture makes the case that word meaning can be represented as a position in
a vector space, and then builds the first algorithm in the course that learns
such positions from raw text. It moves from what "meaning" even is, through why
the pre-neural representations (WordNet, one-hot vectors) were inadequate, to the
word2vec objective function and a hand-derivation of its gradient. By the end you
should understand how a pile of text plus calculus produces vectors that know
that *motel* and *hotel* are similar, with no supervision and no dictionary.

**Slide-by-slide text of this deck: [40 slides](../raw/slides/01-intro-and-word-vectors.md)**
— cite slide numbers from there; the printed slide number equals the PDF page number.

Slides PDF: [Lecture 1 — wordvecs1](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture01-wordvecs1.pdf) ·
Notes: [lecture 1 notes draft](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/readings/cs224n_winter2023_lecture1_notes_draft.pdf) ·
[Full transcript](../raw/transcripts/01-intro-and-word-vectors.md)

The lecture's own plan (slide 2) is: the course (10 mins), human language and word
meaning (15), word2vec introduction (15), word2vec objective function gradients
(25), optimization basics (5), looking at word vectors (10 or less). Slide 2 also
states the key takeaway in Manning's own words: "The (astounding!) result that word
meaning can be represented rather well by a (high-dimensional) vector of real
numbers."

## What the course is about

Manning gives three learning goals (≈4:03). The first is the foundations and
current methods of deep learning for NLP, built from the bottom up: word vectors
and feed-forward networks, then recurrent networks and attention, then the
methods that actually matter in 2024 — transformers, encoder-decoder models,
pretraining and post-training of large language models, adaptation,
interpretability, and agents. The second is an understanding of human language
itself and why it is hard for computers, on the grounds that for most students
this class is the only place they will encounter any linguistics (≈5:36). The
third is the ability to actually build systems, so that a graduate who is later
asked for a text classifier or an information-extraction pipeline knows they can
build it (≈6:21).

Grading is four assignments making up a little under half the grade, a final
project (either a default project with scaffolding or a fully custom one) making
up most of the rest, and a few percent for participation (≈7:08). Assignment 1 is
a deliberately easy Jupyter-notebook on-ramp; assignment 2 is where the math
lives and is, in Manning's words, "the scariest assignment in the whole class"
for some students, and also where PyTorch is introduced (≈8:42). On AI tools: the
course is happy for students to use LLMs as coding assistants, but asking
ChatGPT to answer assignment questions defeats the purpose (≈7:56).

## Why language, and why now

Manning's framing is that language is the differentiator between humans and their
nearest relatives rather than raw intelligence — chimpanzees use tools, plan, and
actually have better short-term memory than humans, yet humans took over the
planet because they could communicate (≈12:33). He then argues language does more
than communicate: it scaffolds higher-level thought and planning (≈13:20).
Writing, only about 5,000 years old, extended that to sharing knowledge across
time and space, which is what took humanity from simple metalworking to modern
technology in a very short span (≈14:07). He balances the knowledge-centric view
with a social one, quoting Herb Clark: the common misconception is that language
use has primarily to do with words and what they mean; it doesn't, it has
primarily to do with people and what they mean (≈16:28).

On the technical side, work on language by computer began in the 1950s, but the
step change came in the last decade with neural methods (≈17:15). Neural machine
translation started around 2014 and was deployed on services like Google by 2016
(≈18:00). Search engines shifted from keyword matching to actually answering
questions, typically with a retrieval network, a reranking network, and a reading
network that synthesizes an answer (≈20:20). GPT-2 in 2019 was the first model
that could simply generate fluent text, one word at a time conditioned on
everything before it; Manning's point about the stolen-nuclear-materials example
is that it is not just grammatical but demonstrably knows Cincinnati is in Ohio
and that the Department of Energy regulates nuclear materials (≈21:52–23:26).
ChatGPT and GPT-4 then made this controllable by instruction. Manning notes these
models are multimodal, and that Stanford coined **foundation models** as a
generalization of large language models to the same technology applied across
modalities — images, sound, DNA and RNA, seismic waves, any signal (≈25:01).

## What "meaning" means

The standard linguistic account pairs a symbol with an idea or thing — signifier
and signified — which is called **denotational semantics**; the meaning of *tree*
is all the trees in the world (≈28:50). This is a popular notion but it was never
obvious how to get it into a computer.

The pre-neural workaround was resources like **WordNet**, a fancy thesaurus of
synonyms and *is-a* relations (a panda is a kind of carnivore, which is a
placental mammal). Manning's critique (≈30:23) is that WordNet misses nuance —
it lists *proficient* as a synonym for *good*, but "that was a proficient shot"
sounds wrong — it is incomplete and cannot keep up with slang, and it is
entirely human-made and so expensive to maintain.

The alternative is **distributional semantics**, which represents a word's
meaning by the contexts it appears in. Manning quotes J.R. Firth's 1957 line
"you shall know a word by the company it keeps," and notes the idea traces back
further to philosophical work by Wittgenstein (≈35:01). If you want to know what
*banking* means, look at the words that occur around *banking* in real
sentences. This is the idea the whole course is built on; see
[distributional semantics](distributional-semantics.md).

## From one-hot to dense vectors

The naive way to put words into math is to index a vocabulary and represent each
word by a vector that is 1 in its own position and 0 everywhere else — a
**one-hot vector**. These are also called **localist representations**, because a
word is represented at exactly one point in the vector (≈34:16). The fatal
problem is that they encode no similarity at all: *motel* and *hotel* are simply
two different positions, and their dot product is zero, so formally the two
vectors are orthogonal and have nothing to do with each other (≈32:42). You can
bolt on an external similarity resource — web search called this query expansion
— but similarity is never intrinsic to the representation (≈33:30).

The fix is to learn a short **dense** vector per word instead, so that similarity
lives in the vectors themselves. Manning's slide uses eight dimensions to fit on
the page; in practice these are a few hundred to a couple of thousand
dimensions rather than the half-million of a full vocabulary (≈35:47). Similarity
is measured by dot product, which grows when corresponding components share a
sign and have large magnitude (≈36:32). These vectors are called **word
vectors**, **embeddings**, or neural word representations — "embedding" because
each word is placed as a point in a high-dimensional space (≈37:19).

Manning stresses that high-dimensional spaces behave very differently from
two-dimensional ones: in 2-D you are only near something if both coordinates
match, whereas in high dimensions a point can be close to many different things
along different dimensions, which is how one vector can capture several senses at
once (≈38:05). Since humans cannot see 300 dimensions, the visualizations use
t-SNE, a nonlinear dimensionality reduction that works better for these
representations than PCA (≈43:29). Zooming into such a plot shows countries
grouped with nationality terms, and a region of verbs with real internal
structure — verbs of communication clustering together, *come* and *go*
together, forms of *have*, forms of *be*, with *become* and *remain* sitting near
*be* because they take the same kind of state complements (≈39:37–40:22).

## Questions worth keeping

The Q&A at ≈41:09 raises the objection that similarity is context-dependent, and
Manning concedes it directly: for now the course learns exactly one vector per
word *type*, so it cannot represent a word's meaning in context — a Hollywood
star and an astronomical star get the same vector. Contextual representations
come later in the course. The consolation, and a genuinely surprising
consequence of high dimensionality, is that the single vector for *star* ends up
close both to astronomical words like *nebula* and to words like *celebrity*
(≈42:43). Asked whether each word has one embedding or several, he says one per
string of letters, and that you can think of it as an average over the word's
senses (≈45:48) — a point developed properly in
[word senses and polysemy](word-senses-and-polysemy.md).

On dimensionality (≈44:16): things start working around 100 dimensions, 300 was
the standard for a long time because it worked well, and as models and data have
grown it has become common to use 1,000 or even 2,000. On whether vector entries
are bounded: they are not bounded by the learning procedure, though people
sometimes length-normalize, and regularization tends to keep coefficients small
(≈45:01).

## word2vec

**word2vec** was introduced by Tomas Mikolov and colleagues at Google in 2013
(≈46:35). It was not the first method for learning word vectors — others go back
to around the turn of the millennium — but it was simple and fast, which is why
it caught on. The setup: take a large body of text (a **corpus** — and Manning
spends a cheerful minute on the fact that the plural is *corpora*, not "corpuses"
or "corpi", at ≈48:10), represent every word *type* by a vector, then slide
through every position in the text. At each position there is a **center word**
and the **outside words** in a window around it, two words each side in his
example. Use the similarity of the center and outside vectors to compute the
probability that those words actually co-occur, and adjust the vectors to make
the observed co-occurrences more probable (≈48:55–50:29).

Manning is explicit that the model is weak by design: seeing *banking* you cannot
predict that *into* precedes it, so the goal is only to do as well as possible —
*crisis* should be likely after *banking crisis*, *skillet* should not (≈51:15).

Turning that into math (≈52:03) gives a likelihood that is a product over every
position and every word in the window. Writing $T$ for the number of positions, $m$ for the
window half-width, $w_t$ for the center word at position $t$ and $\theta$ for all the
parameters (slide 28):

$$L(\theta) = \prod_{t=1}^{T} \prod_{\substack{-m \le j \le m \\ j \ne 0}} P(w_{t+j} \mid w_t ; \theta)$$

Three conventional adjustments follow: a
minus sign, because everyone minimizes rather than maximizes; a logarithm, which
turns the unwieldy product into a sum; and division by the number of words, so
the objective does not scale with corpus size. The result is the **average
negative log likelihood**, and minimizing it maximizes the probability of words
in context (≈53:37–54:23):

$$J(\theta) = -\frac{1}{T} \log L(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \sum_{\substack{-m \le j \le m \\ j \ne 0}} \log P(w_{t+j} \mid w_t ; \theta)$$

The probability itself is defined by the **softmax** of the dot product between
the outside and center vectors (≈55:57; slide 29). For a center word $c$ with center vector
$v_c$ and an outside word $o$ with outside vector $u_o$, over a vocabulary $V$:

$$P(o \mid c) = \frac{\exp(u_o^{\top} v_c)}{\sum_{w \in V} \exp(u_w^{\top} v_c)}$$

Manning notes the dot-product notion of
similarity is a strange one: it has to make *hotel* and *motel* similar, but also
has to let *the* precede *student*, so *the* ends up "similar" to essentially
every noun (≈56:44). See
[softmax and cross-entropy](softmax-and-cross-entropy.md) for why exponentiate-
then-normalize turns unbounded real numbers into a distribution, and why the name
is slightly misleading.

Crucially, the word vectors are the *only* parameters of the model (≈55:11).
There is no other machinery. And each word actually gets **two** vectors, one for
when it is the center word and one for when it is an outside word, purely because
it makes the math simpler (≈1:02:11). With a 400,000-word vocabulary and
100-dimensional vectors, that is $400{,}000 \times 2 \times 100 = 80$ million parameters.

Where do the vectors come from? You start with small random vectors and turn it
into an optimization problem: compute the gradient of the objective with respect
to every parameter and walk downhill, repeatedly, until the probabilities stop
improving (≈1:00:37–1:01:23). Manning calls it a miracle that random vectors plus
a pile of text plus calculus yields something useful — but it does. See
[gradient descent](gradient-descent.md).

## The gradient derivation

Manning spends the last fifteen minutes deriving the gradient by hand, prefacing
it with an apology to the students who know more math than he does and an
acknowledgement that others have not seen a math course in years (≈1:04:29). He
also admits the iPad handwriting is going badly and points to the neatly written
version on the course website (≈1:06:03) — the [lecture 1 slides
PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture01-wordvecs1.pdf) are the version
to read.

The derivation takes the partial derivative of the log probability with respect
to the center vector $v_c$. Splitting the log of the quotient gives a numerator
term and a denominator term (**slide 34**). The numerator is easy: log and exp
cancel, leaving the derivative of $u^{\top} v_c$ with respect to $v_c$, which is just $u$.
Manning justifies that componentwise — the dot product expands to
$u_1 v_1 + u_2 v_2 + \cdots$, so the derivative with respect to $v_1$ is $u_1$ and every
other term vanishes (≈1:10:49–1:11:34; slide 34's margin note reads "Each term is zero
except when $i = j$"). The denominator requires the chain rule twice: once for the log, once for
the exp inside the sum (≈1:12:20–1:15:33; **slide 35**, which flags "Important to
change index" — the inner sum has to be re-indexed before differentiating).

Recombining the two pieces reproduces the softmax, and the answer (**slide 36**) is

$$\frac{\partial}{\partial v_c} \log P(o \mid c) = u_o - \sum_{x \in V} P(x \mid c)\, u_x$$

which Manning reads as **observed minus expected** (≈1:17:56–1:18:42): the actual
outside vector minus the average outside vector the model currently predicts,
weighted by how likely it thinks each word is. The form recurs throughout these
derivations. When expectation equals observation the derivative is zero, which is
where the optimization stops. Finishing the job requires the same treatment for
the outside vectors, which he runs out of time for (≈1:19:30) and which is picked
up at the start of [lecture 2](02-word-vectors-and-language-models.md).

His closing note is that you should understand how this works, but that in
practice computers will compute these derivatives for you.

## A note on source quality

The captions are auto-generated, unpunctuated, and mangled technical terms badly —
*word2vec* as "word Tove" and "word DEC", *PyTorch* as "py talk", *t-SNE* as "tne".
[The transcript](../raw/transcripts/01-intro-and-word-vectors.md) has been copy-edited
into readable sentences with those terms restored; the raw captions are kept at
[`original/`](../raw/transcripts/original/01-intro-and-word-vectors.md) if you need to
check what was actually said.

Two places remain genuinely unreliable, and the slides are the authority for both:

- **The dictated mathematics** between ≈1:06:03 and ≈1:17:05 comes through with wrong
  subscripts and stray "[Music]" markers. Manning himself says the iPad handwriting
  was going badly and points at the website version (≈1:06:03). **Slides 33–36** are
  that version, written out neatly, and are transcribed in full on the
  [slide page](../raw/slides/01-intro-and-word-vectors.md).
- **One phrase at ≈49:42** ("we learn ve word vectors") was left uncorrected rather
  than guessed at.
