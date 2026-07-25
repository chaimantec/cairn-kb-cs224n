# Lecture 4 — Dependency Parsing

Manning opens by saying this lecture is "a 180" from Tuesday's (≈0:05): where lecture 3 was
all mathematics, this one is all linguistics — the structure of human sentences, how to
represent it, and how to get a machine to recover it. It establishes four things: that
sentences have structure and that recovering it is necessary for interpretation; that
**dependency grammar** is the representation this course uses; that a sentence can be
parsed by a sequence of cheap local decisions (**transition-based parsing**); and that
replacing the classifier that makes those decisions with a small neural network makes the
parser both more accurate and faster. The last of those is Assignment 2.

**Slide-by-slide text of this deck: [49 printed slides](../raw/slides/04-dependency-parsing.md)**
— cite the printed slide numbers. **Note:** the PDF has only 45 pages; printed slides 4, 5,
8 and 13 were hidden in the source deck and never exported, so printed numbers run ahead of
PDF pages (printed 40 is PDF page 36).

Slides PDF: [Lecture 4 — dep-parsing](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture04-dep-parsing.pdf) ·
Notes: [2019 notes 04 — dependency parsing](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/readings/cs224n-2019-notes04-dependencyparsing.pdf) ·
[Full transcript](../raw/transcripts/04-dependency-parsing.md)

The plan on slide 2 runs: syntactic structure, constituency and dependency (30 mins);
dependency grammar and treebanks (15); transition-based dependency parsing (15); neural
dependency parsing (20).

## Two views of sentence structure

**Constituency** — equivalently phrase structure grammar, equivalently context-free
grammar — organizes words into nested units (slide 3). Words belong to a few classes
(**parts of speech**): *cat* and *door* are nouns, *cuddly* an adjective, *the* a
determiner (linguists' term; you may also see *article*), *by* and *through* prepositions.
Words combine into phrases — *the cuddly cat* is a **noun phrase**, *by the door* a
**prepositional phrase** — and phrases combine into bigger phrases: *the cuddly cat by the
door* (≈4:44). Written as grammar rules this becomes NP → Det (Adj)\* N, NP → NP PP,
PP → P NP, and so on, where the Kleene star allows any number of adjectives (≈6:19).

**Dependency structure** represents the same facts differently (slide 7). Instead of
nesting, it asks for each word which other word it modifies, attaches to, or is an argument
of. For *Look in the large crate in the kitchen by the door*, the head of the whole thing is
*look*; *crate* is what the looking is in; *the*, *large*, *in the kitchen* and *by the
door* all modify *crate* (≈10:11). Manning notices while drawing this that he had got the
constituency version on the board wrong a moment earlier, and corrects it — a useful
illustration that the two representations really do encode the same attachment decisions
(≈11:43).

This course uses dependency structure for the parsers it builds. See
[dependency grammar](dependency-grammar.md).

## Why structure is needed at all

Humans communicate in a linear stream — a linear sequence of words on the page, and in
speech not even that, just a continuous sound stream with no white space between words
(≈12:31). To recover a meaning from that stream, a listener has to work out what modifies
what. A model has to do the same to interpret language correctly (slide 9).

The evidence that this is hard is **ambiguity**, and Manning's central claim is that human
language is not locally but **globally ambiguous** (≈17:56). Programming languages have
local ambiguities that are always resolved by rule — *else* attaches to the nearest *if* —
so there is never global ambiguity. Human language has no such rule: nothing in the string
decides, and making the sentence longer does not help. You are simply expected to read it
and use context and intelligence.

Four kinds of ambiguity get a headline each:

- **Prepositional phrase attachment** (slides 10–11): *Scientists count whales from space*
  — is the counting done from space (correct, and what the BBC article was about), or are
  they whales from space?
- **Coordination scope** (slides 14–15): *Shuttle veteran and longtime NASA executive Fred
  Gregory appointed to board* — one person or two? And *Doctor: No heart, cognitive issues*
  — does "no" scope over both?
- **Adjectival/adverbial modifier** (slide 16): *Students get first hand job experience*.
- **Verb phrase attachment** (slide 17): *Mutilated body washes up on Rio beach to be used
  for Olympics beach volleyball* — the beach, or the body?

PP attachment is the one that scales badly. In *The board approved its acquisition by Royal
Trustco Ltd. of Toronto for $27 a share at its monthly meeting*, four prepositional phrases
follow the object, and each must attach somewhere earlier (slide 12). The count of legal
analyses is a **Catalan number**, C_n = (2n)!/[(n+1)!n!] — exponential growth, and the same
series that shows up in polygon triangulation, for the same reason: the arcs cannot cross
(≈20:58). Manning's point is that a human reads a sentence like this over their corn flakes
without their brain exploding over the alternatives (≈21:43). See
[syntactic ambiguity](syntactic-ambiguity.md).

Structure is also useful in plainly practical ways. Slide 18 shows dependency paths used to
extract protein–protein interactions from biomedical text: from *KaiC interacts
rhythmically with SasA, KaiA and KaiB*, the path `KaiC ←nsubj interacts nmod:with→ SasA`
plus the conjunction arcs yields three interaction facts, which is how structured databases
of known interactions get built (≈27:51).

## Dependency grammar

A dependency is a binary asymmetric relation — an arrow — from a **head** (also governor,
superior, regent) to a **dependent** (modifier, inferior, subordinate) (slides 19–21). The
arrows are usually **typed** with a grammatical relation: `nsubj`, `obl`, `appos`, `case`,
`conj` and so on. Dependencies are generally taken to form a **tree**: connected, acyclic,
single-root. Conventionally a **fake ROOT** node is added so that every real word is the
dependent of exactly precisely one node, which makes parsing work out more cleanly
(slide 24, ≈38:43).

Two conventions worth knowing because they vary between sources: some people draw arrows
head→dependent and others dependent→head; this course follows Tesnière and draws them from
head to dependent (slide 24).

The history inverts the usual assumption (slides 22–23). Dependency grammar is the old idea
— Pāṇini's grammar of Sanskrit, somewhere between the 8th and 4th centuries BCE, and the
Arabic grammarians of the first millennium. Constituency/context-free grammar is the
recent invention: phrase structure grammars date to the 1940s (R.S. Wells, 1947) and were
canonicalized by Chomsky in the 1950s. Manning adds a digression aimed at the computer
scientists: the **Chomsky hierarchy** was not invented to torture CS103 students or to
explain formal language theory, but as an argument that finite-state mechanisms are
inadequate to represent the complexity of human language (≈34:51). And an aside that he
says blows his mind — Pāṇini's grammar was composed **orally** and transmitted that way for
centuries, which is part of why nobody knows what century he lived in (≈36:24).

## Treebanks, and why annotated data won

Early NLP tried to write grammars by hand, and it worked badly (≈39:29). Two reasons. First
coverage: beneath the canonical structures there is a very long tail of messy usage,
because people use language creatively — Yoda word order, or the 1990s fad for appending
*not* to the end of a sentence (≈41:02). Second, and bigger, ambiguity: a hand-written
grammar licenses all the parses of that four-PP sentence and gives you no reason to prefer
one. Statistics about how often particular words modify particular other words do give you
a reason.

So from the late 1980s through the 2000s the field built **treebanks** — the Penn Treebank
(Marcus et al. 1993), and now **Universal Dependencies**, which Manning has been heavily
involved with: over a hundred languages annotated with a uniform dependency formalism,
which makes it valuable for crosslinguistic and psycholinguistic work (slide 25, ≈42:35).
Assignment 2 trains on data of exactly this kind.

Building a treebank feels slower and less useful than writing a grammar, and is genuinely
slow, hard work — but it pays off in reusability (parsers, taggers and linguistic studies
all build on one annotation effort), broad coverage, frequency information, and, crucially,
**a way to evaluate systems** (slide 26). Manning underlines the last one: in the 1950s,
60s and 70s nobody had evaluation methods at all. You demonstrated a good parser by running
it, typing in a sentence, and saying *look, it worked*. Being able to say "here are a
thousand hand-parsed sentences, let us measure your parser against them" was a genuinely
revolutionary development (≈44:55).

## What information a parser can use

Slide 27 lists the four straightforward sources, illustrated on *Discussion of the
outstanding issues was completed*:

1. **Bilexical affinities** — are these two particular words a plausible head/dependent
   pair? *discussion → issues* is; *completed → the* is not.
2. **Dependency distance** — most dependencies are between nearby words.
3. **Intervening material** — dependencies rarely span verbs or punctuation.
4. **Valency of heads** — how many dependents, on which side, a head usually takes. *Broke*
   normally has something on its left (whoever broke it) and often something on its right,
   though *the cup broke* has nothing on the right; and it will not take an unbounded list
   of left arguments (≈48:04).

## Projectivity

A parse is **projective** if, with the words laid out in order and all arcs drawn above
them, no two arcs cross (slide 29). Dependencies derived from a CFG tree are necessarily
projective. Most syntactic structure is projective — but not all, and dependency theory
normally allows non-projective structures because you cannot get the semantics of certain
constructions right without them.

Two examples (slides 28–29): *I'll give a talk tomorrow on neural networks*, where *on
neural networks* modifies *talk* but *tomorrow* is an argument of *give*, so the arcs
cross; and *Who did Bill buy the coffee from yesterday?*, where *who* is the object of
*from* but has moved to the front. Slide 38 lists five ways to handle non-projectivity,
from declaring defeat, through adding a SWAP transition (which allows any non-projectivity,
by the logic of bubble sort), to moving to a graph-based parser that never imposed the
constraint. **The parser built in this course is projective-only** (≈53:29).

## Transition-based parsing

Of the four families on slide 30 — dynamic programming (Eisner 1996, O(n³)), graph
algorithms (MSTParser), constraint satisfaction, and transition-based parsing — the last is
what the course uses, because it is fast and its machine learning is simple.

The **arc-standard** system (Nivre 2003; slides 31–32) keeps three things: a **stack** σ
written with its top to the right, starting with ROOT; a **buffer** β written with its top
to the left, starting with the input sentence; and a set **A** of dependency arcs, starting
empty. Three actions:

- **Shift** — move the first word of the buffer onto the stack.
- **Left-Arc**_r — the top of the stack becomes the head of the item below it; add that arc
  and pop the dependent.
- **Right-Arc**_r — the item below the top becomes the head of the top; add that arc and pop
  the dependent.

Parsing finishes when the buffer is empty and the stack holds only ROOT.

Slides 33–34 run *I ate fish* through it: Shift, Shift, Left-Arc (adding
`nsubj(ate → I)`), Shift, Right-Arc (adding `obj(ate → fish)`), Right-Arc (adding
`root([root] → ate)`), done. The deck's "Nota bene" is the whole difficulty: at every step
Manning made the *correct* transition, but a parser has to work out which one that is
(≈58:56). Different choices build different trees — choosing Right-Arc instead would have
made *ate* a dependent of *I* (≈59:43).

So the question is how to choose, and the answer is a classifier over legal moves
(slide 35). This gives **linear-time parsing**, since each word is dealt with a constant
number of times — against the cubic time you pay to fully consider context-free parses
(≈1:01:19). In its simplest form there is **no search at all**: you just predict the next
transition each time and commit. A beam search keeping *k* prefixes helps, at a cost in
speed. Nivre's result was that machine learning is good enough that greedy prediction still
yields an accurate parser. See [transition-based parsing](transition-based-parsing.md).

## Evaluation: UAS and LAS

Slide 37 defines both on *She saw the video lecture*. Write each word's head as a number:
gold is 1→2, 2→0, 3→5, 4→5, 5→2. Compare a proposed parse arc by arc.

- **UAS** (unlabeled attachment score) — fraction of words given the correct head. In the
  slide's example 4 of 5, so 80%.
- **LAS** (labeled attachment score) — fraction given the correct head *and* the correct
  relation label. Only 2 of 5 there, so 40%.

## Why the neural parser wins

Nivre's 2005 parser used a symbolic feature-based classifier, with **indicator features**
that are conjunctions of elements of the configuration — "the top of the stack is *good*
and its POS is JJ", "the top of the stack is *good* and the second item is the verb *has*"
(slide 36). Three problems (slide 39):

1. **Sparse** — conjoining several particular words gives millions of features, most
   almost never seen. A feature might fire ten times in a million sentences (≈1:03:37).
2. **Incomplete** — some word combinations simply never appear in the training data, so
   the features for them are missing.
3. **Expensive** — and this is the one that surprises people: **more than 95% of parsing
   time went into computing the features**, not into the machine learning decision
   (≈1:06:45).

The neural parser (Chen and Manning 2014) fixes all three with a dense representation.
Two wins:

**Distributed representations** (slide 41). Words become *d*-dimensional embeddings, so a
configuration never seen before still resembles ones that were. The step beyond that is to
give **part-of-speech tags and dependency labels** their own embeddings too: real POS tag
sets are fine-grained, and NNS (plural noun) should end up close to NN (singular noun),
just as `nummod` should end up close to `amod` (≈1:09:53). See
[distributional semantics](distributional-semantics.md).

The configuration is turned into a vector by extracting a fixed set of tokens from the
stack and buffer — the top two stack items, the first buffer item, and their already-built
leftmost and rightmost children — then looking up the word, POS and dependency-label
embedding for each and **concatenating** them (slide 42). This is the same move as the
five-word window classifier of lecture 2 (≈1:11:24).

**A non-linear classifier** (slide 43). Everyone else's parsers used linear classifiers —
SVMs, logistic regression — which only give linear decision boundaries. The architecture
(slide 44) is a plain feed-forward network:

    h = ReLU(Wx + b₁)
    y = softmax(Uh + b₂)

over the three transition types, with the log loss backpropagated all the way into the
embeddings. The hidden layer re-represents the input so that a linear softmax can separate
it. See [backpropagation](backpropagation.md) and
[softmax, the logistic function, and cross-entropy](softmax-and-cross-entropy.md).

The result (slide 40) is a parser as accurate as the best graph-based parsers and far
faster than them:

| Parser | UAS | LAS | sent./s |
| --- | --- | --- | --- |
| MaltParser | 89.8 | 87.2 | 469 |
| MSTParser | 91.4 | 88.1 | 10 |
| TurboParser | **92.3** | 89.6 | 8 |
| Chen & Manning 2014 | 92.0 | **89.7** | **654** |

The counter-intuitive part, which Manning calls out: you might expect real-valued matrices
to be *slower* than symbolic feature lookup, but because the symbolic models spent nearly
all their time computing features, the neural version was faster and more accurate at the
same time (≈1:08:19).

## What happened next

Google took the model and scaled it — deeper networks, bigger vectors, better
hyperparameters, beam search, and global CRF-style inference over the decision sequence —
producing SyntaxNet and **Parsey McParseFace** in 2016, at 94.61 UAS (slide 46). Manning's
comment is that it blew his mind that dependency parsing got a full tech-press PR splash,
and that the silly name worked very well for media pickup (≈1:14:29).

**Graph-based parsers** take the other approach (slides 47–48): score every possible head
for every word — *n*² candidate dependencies — then find the best tree with a minimum
spanning tree algorithm, which also rules out cycles and disconnected pieces. Doing this
well needs good *contextual* representations of each word token, which the course develops
in later lectures. The neural version, **Dozat and Manning (2017)**, uses a biaffine
scoring model and reaches 95.74 UAS / 94.08 LAS — a bit over a percent better than Parsey
McParseFace — at the cost of being slower than transition-based parsers (slide 49). It is
the parser in **Stanza**, Stanford's open-source NLP software (≈1:18:22).

## Related pages

- [Dependency grammar](dependency-grammar.md) — heads and dependents, typed relations,
  projectivity, treebanks and Universal Dependencies.
- [Transition-based parsing](transition-based-parsing.md) — the arc-standard system, the
  neural classifier, UAS/LAS, and the graph-based alternative.
- [Syntactic ambiguity](syntactic-ambiguity.md) — the four ambiguity types, why human
  language is globally ambiguous, and Catalan growth.
- [Lecture 3 — Backpropagation and Neural Networks](03-backpropagation-and-neural-networks.md)
  — the machinery that trains this parser.
- [Distributional semantics](distributional-semantics.md) — why embedding POS tags and
  dependency labels works for the same reason embedding words does.
