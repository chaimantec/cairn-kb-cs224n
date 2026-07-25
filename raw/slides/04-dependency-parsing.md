---
title: Lecture 4 — Dependency Parsing (slide deck)
lecture: 4
slides: 49 printed / 45 pages in the PDF
source_pdf: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture04-dep-parsing.pdf
note: Printed slide numbers run 1–49, but slides 4, 5, 8 and 13 are not in the PDF (hidden in the source deck and not exported). The PDF therefore has 45 pages. Cite the printed numbers.
---

# Lecture 4 — Dependency Parsing: slide-by-slide

Text and figures of
[`cs224n-spr2024-lecture04-dep-parsing.pdf`](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture04-dep-parsing.pdf),
transcribed from the deck. Diagrams and plots are described in prose since the KB is
read as text.

**Slide numbers vs PDF pages.** The numbers printed on the slides run 1 to 49, but the
PDF has only 45 pages: **printed slides 4, 5, 8 and 13 are absent** — they were hidden
in the source deck and never exported, so no copy of them exists to transcribe. Every
other printed number is present exactly once. Headings below use the **printed** number,
which is what Manning and the course notes refer to; to find one in the PDF, subtract
the number of missing slides before it (printed 40 is PDF page 36, for instance).

Companion pages: [wiki page for this lecture](../../wiki/04-dependency-parsing.md) ·
[transcript](../transcripts/04-dependency-parsing.md)

## Contents

| Slides | Section |
| ------ | ------- |
| 1–2 | Title and lecture plan |
| 3, 6 | §1 Constituency: phrase structure grammar / CFGs |
| 7 | Dependency structure, the other view |
| 9 | Why sentence structure is needed for communication |
| 10–12 | Prepositional phrase attachment ambiguity; Catalan numbers |
| 14–16 | Coordination scope and adjectival/adverbial modifier ambiguity |
| 17 | Verb phrase attachment ambiguity |
| 18 | Dependency paths for semantic interpretation (protein–protein interaction) |
| 19–21 | §2 Dependency grammar: dependencies, typed arrows, head and dependent |
| 22–23 | History: Pāṇini, Tesnière, Hays |
| 24 | Arrow direction convention and the fake ROOT |
| 25–26 | The rise of annotated data and Universal Dependencies treebanks |
| 27 | Dependency conditioning preferences — the four sources of information |
| 28–29 | Dependency parsing as head selection; projectivity |
| 30 | §3 Four methods of dependency parsing |
| 31–32 | Greedy transition-based parsing [Nivre 2003]; the arc-standard transition system |
| 33–34 | "I ate fish" worked through the arc-standard parser |
| 35 | MaltParser: a classifier chooses the next action |
| 36 | Conventional (sparse, indicator) feature representation |
| 37 | Evaluation: UAS and LAS |
| 38 | Handling non-projectivity — five options |
| 39 | §4 Why a neural parser wins: the three problems with categorical features |
| 40 | Chen & Manning (2014) results table |
| 41–42 | First win: distributed representations; extracting tokens from a configuration |
| 43 | Second win: non-linear classifiers |
| 44 | The neural dependency parser model architecture |
| 45 | Dependency parsing for sentence structure |
| 46 | Further developments: Google's SyntaxNet / Parsey McParseFace |
| 47–49 | Graph-based dependency parsers; Dozat & Manning (2017) |

---

## Slide 1 — Title

"Natural Language Processing with Deep Learning — CS224N/Ling284". Christopher Manning.
"Lecture 4: Dependency Parsing".

## Slide 2 — Lecture Plan

Syntactic Structure and Dependency parsing

1. Syntactic Structure: Consistency and Dependency (30 mins)
2. Dependency Grammar and Treebanks (15 mins)
3. Transition-based dependency parsing (15 mins)
4. Neural dependency parsing (20 mins)

Key Learnings: Explicit linguistic structure and how a neural net can decide it

Reminders/comments:

- In Assignment 2, you build a neural dependency parser using PyTorch!
- Start installing and learning PyTorch (Ass 2 is quite scaffolded)
- Come to the PyTorch tutorial, Friday 3:30pm, Gates B01
- Final project discussions – **come meet with us**; focus of Tuesday class in week 4
  - Chris office hours Mon 2:00-4:00pm on Calendly:
    https://calendly.com/manning/cs224n-office-hours

*(The heading reads "Consistency and Dependency"; from the body of the lecture the
intended word is "Constituency".)*

## Slide 3 — 1. The linguistic structure of sentences – two views: Constituency = phrase structure grammar = context-free grammars (CFGs)

Phrase structure organizes words into nested constituents

**Starting unit: words**

    the, cat, cuddly, by, door

**Words combine into phrases**

    the cuddly cat,      by the door

**Phrases can combine into bigger phrases**

    the cuddly cat by the door

*(Printed slides 4 and 5 are not present in the PDF.)*

## Slide 6 — The linguistic structure of sentences – two views: Constituency = phrase structure grammar = context-free grammars (CFGs)

Phrase structure organizes words into nested constituents.

The slide lays out a grid of words showing how the pieces of a noun phrase stack up —
determiners in the left column, nouns in the second, adjectives/participles in the third,
and prepositional phrases in the fourth:

    the         cat
    a           dog
                large           in a crate
                barking         on the table
                cuddly          by the door
          large        barking
    talk to
    walked behind

The layout is the build-up of the constituent structure: determiner + (adjective)* +
noun + (PP)*, with *talk to* and *walked behind* at the bottom as verbs that such a noun
phrase can attach to.

## Slide 7 — Two views of linguistic structure: Dependency structure

- Dependency structure shows which words depend on (modify, attach to, or are arguments
  of) which other words.

The example sentence is printed in italics at the centre of the slide:

    Look in the large crate in the kitchen by the door

*(On the projected slide Manning draws the dependency arrows over this sentence by hand;
the exported PDF shows the bare sentence with no arcs. Printed slide 8 is not present in
the PDF.)*

## Slide 9 — Why is sentence structure needed for communication?

Humans communicate complex ideas by composing words together into bigger units to convey
complex meanings

Human listeners need to work out what modifies [attaches to] what

A model needs to understand sentence structure in order to be able to interpret language
correctly

## Slide 10 — Prepositional phrase attachment ambiguity

A screenshot of a BBC News page, in the Science & Environment section, with the headline

> **Scientists count whales from space**

By Jonathan Amos, BBC Science Correspondent. Partly visible behind it is another headline,
"San Jose cops kill man with knife".

## Slide 11 — Prepositional phrase attachment ambiguity

The same sentence printed twice with a photograph beside each reading:

- "Scientists count whales from space" — beside an aerial satellite-style photograph of
  two whales swimming at the surface of the ocean, marked with a green ✓. (The counting
  is done from space.)
- "Scientists count whales from space" — beside a fanciful image of a whale swimming
  through outer space among planets and nebulae, marked with a red ✗. (The whales are
  from space.)

## Slide 12 — PP attachment ambiguities multiply

- A key parsing decision is how we 'attach' various constituents
  - PPs, adverbial or participial phrases, infinitives, coordinations,

The example, with each prepositional phrase bracketed on its own line:

    The board approved [its acquisition] [by Royal Trustco Ltd.]
                                                    [of Toronto]
                                              [for $27 a share]
                                          [at its monthly meeting].

- Catalan numbers: C_n = (2n)! / [(n+1)! n!]
- An exponentially growing series, which arises in many tree-like contexts:
  - E.g., the number of possible triangulations of a polygon with *n*+2 sides
    - Turns up in triangulation of probabilistic graphical models (CS228)….

*(Printed slide 13 is not present in the PDF.)*

## Slide 14 — Coordination scope ambiguity

The same sentence printed twice, with space above each where the two bracketings are
drawn in the live lecture:

    Shuttle veteran and longtime NASA executive Fred Gregory appointed to board

    Shuttle veteran and longtime NASA executive Fred Gregory appointed to board

The two readings are: one person (a shuttle veteran *and* longtime NASA executive, namely
Fred Gregory) versus two people (a shuttle veteran, and separately Fred Gregory).

## Slide 15 — Coordination scope ambiguity

A photograph of a newspaper page (The News-Gazette, Nation/World section) under the
standfirst "PRESIDENT'S FIRST PHYSICAL", with the headline

> **Doctor: No heart, cognitive issues**

and a subhead "But Trump needs to reduce his cholesterol, lose weight", by Jill Colvin.
The ambiguity is whether "no" scopes over both "heart issues" and "cognitive issues", or
whether the doctor found "no heart".

## Slide 16 — Adjectival/Adverbial Modifier Ambiguity

A photograph of a newspaper page (The Pratt Tribune, Saturday, October 28, 2017), under
the standfirst "MENTORING DAY", with the headline

> **Students get first hand job experience**

The story by Gale Rose describes students visiting businesses around Pratt on Disability
Mentoring Day — 97 students from 12 schools, who "fanned out across Pratt and got first
hand … experience with various operations", including a visit to the Main Street Small
Animal Veterinary Clinic where they watched a snake eat a mouse. The ambiguity is in how
"first hand" groups: [first hand] [job experience] versus [first] [hand job] [experience].
Below the article is a campaign advertisement for a city commissioner.

## Slide 17 — Verb Phrase (VP) attachment ambiguity

A screenshot of a Guardian page, Rio de Janeiro, with the headline

> **Mutilated body washes up on Rio beach to be used for Olympics beach volleyball**

Timestamped 6/29/16, 1:48 PM. The ambiguity is whether the infinitival "to be used for
Olympics beach volleyball" attaches to *beach* or to the verb phrase *washes up*.

## Slide 18 — Dependency paths help extract semantic interpretation – simple practical example: extracting protein-protein interaction

A dependency tree drawn over the sentence "The results demonstrated that KaiC interacts
rhythmically with SasA, KaiA and KaiB", with *demonstrated* at the root:

- *demonstrated* —nsubj→ *results*, and *results* —det→ *The*
- *demonstrated* —ccomp→ *interacts*
- *interacts* —mark→ *that*, —nsubj→ *KaiC*, —advmod→ *rythmically* [sic],
  —nmod:with→ *SasA*
- *SasA* —case→ *with*, —conj:and→ *KaiA*, —conj:and→ *KaiB*, with *cc* on *and*

The arcs on the path used for extraction are drawn in red. Below, the three extracted
paths:

    KaiC ←nsubj interacts nmod:with → SasA
    KaiC ←nsubj interacts nmod:with → SasA conj:and→ KaiA
    KaiC ←nsubj interacts nmod:with → SasA conj:and→ KaiB

Citation: [Erkan et al. EMNLP 07, Fundel et al. 2007, etc.]

## Slide 19 — 2. Dependency Grammar and Dependency Structure

Dependency syntax postulates that syntactic structure consists of relations between
lexical items, normally binary asymmetric relations ("arrows") called **dependencies**

The tree for "Bills on ports and immigration were submitted by Senator Brownback,
Republican of Kansas" is drawn as an untyped tree with *submitted* at the top:

- *submitted* → *Bills*, *were*, *Brownback*
- *Bills* → *ports*; *ports* → *on*, *and*, *immigration*
- *Brownback* → *by*, *Senator*, *Republican*; *Republican* → *Kansas*;
  *Kansas* → *of*

## Slide 20 — Dependency Grammar and Dependency Structure

Same sentence and tree, now with the arrows labelled:

The arrows are commonly **typed** with the name of grammatical relations (subject,
prepositional object, apposition, etc.)

- *submitted* —nsubj:pass→ *Bills*, —aux→ *were*, —obl→ *Brownback*
- *Bills* —nmod→ *ports*; *ports* —case→ *on*, —cc→ *and*, —conj→ *immigration*
- *Brownback* —case→ *by*, —flat→ *Senator*, —appos→ *Republican*
- *Republican* —nmod→ *Kansas*; *Kansas* —case→ *of*

## Slide 21 — Dependency Grammar and Dependency Structure

Same tree, with the definitions:

An arrow connects a **head** (governor, superior, regent) with a **dependent** (modifier,
inferior, subordinate)

Usually, dependencies form a tree (a connected, acyclic, single-root graph)

## Slide 22 — Dependency Grammar/Parsing History

- The idea of dependency structure goes back a long way
  - To Pāṇini's grammar (composed c. 5th century BCE)
  - Basic approach of 1st millennium Arabic grammarians
- Constituency/context-free grammar is a new-fangled invention
  - 20th century invention (R.S. Wells, 1947; then Chomsky 1953, etc.)
- Modern dependency work is often sourced to Lucien Tesnière (1959)
  - Was dominant approach in "East" in 20th Century (Russia, China, …)
    - Good for free-er word order, inflected languages like Russian (or Latin!)
- Used in some of the earliest parsers in NLP, even in the US:
  - David Hays, one of the founders of U.S. computational linguistics, built early
    (first?) dependency parser (Hays 1962) and published on dependency grammar in
    *Language*

## Slide 23 — Pāṇini's grammar (composed c. 5th century BCE)

A photograph of an open birch-bark manuscript, both leaves densely written in Devanagari.

Gallery: http://wellcomeimages.org/indexplus/image/L0032691.html
CC BY 4.0 — File:Birch bark MS from Kashmir of the Rupavatra Wellcome L0032691.jpg
But this comes from much later – originally the grammar was **oral**

## Slide 24 — Dependency Grammar and Dependency Structure

A dependency analysis drawn in magenta as arcs above the sentence:

    ROOT Discussion of the outstanding issues was completed .

- Some people draw the arrows one way; some the other way!
  - Tesnière had them point from head to dependent – we follow that convention
- We usually add a fake ROOT so every word is a dependent of precisely 1 other node

## Slide 25 — The rise of annotated data & Universal Dependencies treebanks

Brown corpus (1967; PoS tagged 1979); Lancaster-IBM Treebank (starting late 1980s);
Marcus et al. 1993, The Penn Treebank, *Computational Linguistics*;
Universal Dependencies: http://universaldependencies.org/

Three screenshots of an annotation tool, each showing a sentence with its part-of-speech
tags in coloured boxes and labelled dependency arcs above:

- Sentence 76: "i think Miramar was a famous goat trainer or something ." —
  PRON VERB PROPN VERB DET ADJ NOUN NOUN CONJ NOUN PUNCT, with arcs nsubj, ccomp, nsubj,
  cop, det, amod, compound, cc, conj, punct.
- Sentence 77: "Why is the city called Miramar ?" — ADV AUX DET NOUN VERB PROPN PUNCT,
  with arcs advmod, auxpass, det, nsubjpass, xcomp, punct.
- Sentence 84: "Do you think there are any koreans in Miramar ?" —
  AUX PRON VERB PRON VERB DET PROPN ADP PROPN PUNCT, with arcs aux, nsubj, ccomp, expl,
  nsubj, det, nmod, case, punct.

## Slide 26 — The rise of annotated data

Starting off, building a treebank seems a lot slower and less useful than writing a
grammar (by hand)

But a treebank gives us many things

- Reusability of the labor
  - Many parsers, part-of-speech taggers, etc. can be built on it
  - Valuable resource for linguistics
- Broad coverage, not just a few intuitions
- Frequencies and distributional information
- A way to evaluate NLP systems

## Slide 27 — Dependency Conditioning Preferences

What are the straightforward sources of information for dependency parsing?

1. **Bilexical affinities** — The dependency [discussion → issues] is plausible
2. **Dependency distance** — Most (but not all) dependencies are between nearby words
3. **Intervening material** — Dependencies rarely span intervening verbs or punctuation
4. **Valency of heads** — How many dependents on which side are usual for a head?

Below, the same magenta arc diagram as slide 24 over
"ROOT Discussion of the outstanding issues was completed ."

## Slide 28 — Dependency Parsing

- A sentence is parsed by choosing for each word what other word (including ROOT) it is
  a dependent of
- Usually some constraints:
  - Only one word is a dependent of ROOT
  - Don't want cycles A → B, B → A
- This makes the dependencies a tree
- Final issue is whether arrows can cross (be **non-projective**) or not

The example, with arcs drawn above:

    ROOT   I  'll  give  a  talk  tomorrow  on  neural  networks

Here the arc from *talk* out to *on … networks* crosses the arc covering *tomorrow*.

## Slide 29 — Projectivity

- Definition of a **projective parse**: There are no crossing dependency arcs when the
  words are laid out in their linear order, with all arcs above the words
- Dependencies corresponding to a CFG tree must be **projective**
  - I.e., by forming dependencies by taking 1 child of each category as head
- Most syntactic structure is projective like this, but dependency theory normally does
  allow non-projective structures to account for displaced constituents
  - You can't easily get the semantics of certain constructions right without these
    nonprojective dependencies

The example sentence, hand-annotated, with *buy* marked "root":

    Who did Bill buy the coffee from yesterday ?

The arc from *from* back to *Who* is drawn in magenta and clearly crosses the other arcs
— the non-projective dependency.

## Slide 30 — 3. Methods of Dependency Parsing

1. **Dynamic programming**
   Eisner (1996) gives a clever algorithm with complexity O(n³), by producing parse items
   with heads at the ends rather than in the middle
2. **Graph algorithms**
   You create a Minimum Spanning Tree for a sentence
   McDonald et al.'s (2005) O(n²) MSTParser scores dependencies independently using an ML
   classifier (he uses MIRA, for online learning, but it can be something else)
   Neural graph-based parser: Dozat and Manning (2017) et seq. – very successful!
3. **Constraint Satisfaction**
   Edges are eliminated that don't satisfy hard constraints. Karlsson (1990), etc.
4. **"Transition-based parsing" or "deterministic dependency parsing"**
   Greedy choice of attachments guided by good machine learning classifiers
   E.g., MaltParser (Nivre et al. 2008). Has proven highly effective. And fast.

## Slide 31 — Greedy transition-based parsing [Nivre 2003]

*(A photograph of Joakim Nivre speaking at a lectern sits in the top right corner.)*

- A simple form of a greedy discriminative dependency parser
- The parser does a sequence of bottom-up actions
  - Roughly like "shift" or "reduce" in a shift-reduce parser – CS143, anyone?? – but the
    "reduce" actions are specialized to create dependencies with head on left or right
- The parser has:
  - a stack σ, written with top to the right
    - which starts with the ROOT symbol
  - a buffer β, written with top to the left
    - which starts with the input sentence
  - a set of dependency arcs A
    - which starts off empty
  - a set of actions

## Slide 32 — Basic transition-based dependency parser

    Start:  σ = [ROOT], β = w₁, …, w_n , A = ∅
    1. Shift        σ, w_i|β, A  ➜  σ|w_i, β, A
    2. Left-Arc_r   σ|w_i|w_j, β, A  ➜  σ|w_j, β, A∪{r(w_j,w_i)}
    3. Right-Arc_r  σ|w_i|w_j, β, A  ➜  σ|w_i, β, A∪{r(w_i,w_j)}
    Finish: σ = [w], β = ∅

## Slide 33 — Arc-standard transition-based parser

(there are other transition schemes …)
Analysis of "I ate fish"

The first three configurations are drawn as boxes, the stack in a grey outline and the
buffer in an orange outline:

- **Start** — stack `[root]`; buffer `I ate fish`
- **Shift** — stack `[root] I`; buffer `ate fish`
- **Shift** — stack `[root] I ate`; buffer `fish`

A cyan box at the right repeats the transition system from slide 32.

## Slide 34 — Arc-standard transition-based parser

Analysis of "I ate fish" — continued. Each row shows the configuration before and after
the transition:

- **Left Arc** — `[root] I ate` → `[root] ate`, with A += nsubj(ate → I)
- **Shift** — `[root] ate` + buffer `fish` → `[root] ate fish`, buffer empty
- **Right Arc** — `[root] ate fish` → `[root] ate`, with A += obj(ate → fish)
- **Right Arc** — `[root] ate` → `[root]`, with A += root([root] → ate), **Finish**

Final result, in magenta:

    A = { nsubj(ate → I), obj(ate → fish), root([root] → ate) }

A pink callout box: **Nota bene:** In this example I've at each step made the "correct"
next transition. But a parser has to work this out – by exploring or inferring!

## Slide 35 — MaltParser [Nivre and Hall 2005]

- We have left to explain how we choose the next action 🤷
  - Answer: Stand back, I know machine learning!
- Each action is predicted by a discriminative classifier (e.g., softmax classifier) over
  each legal move
  - Max of 3 untyped choices (max of |R| × 2 + 1 when typed)
  - Features: top of stack word, POS; first in buffer word, POS; etc.
- There is NO search (in the simplest form)
  - But you can profitably do a beam search if you wish (slower but better):
    - You keep *k* good parse prefixes at each time step
- The model's accuracy is *a bit* below the state of the art in dependency parsing, but
- It provides **very fast linear time parsing**, with high accuracy – great for parsing
  the web

## Slide 36 — Conventional Feature Representation

A configuration is drawn at the top: stack `ROOT has_VBZ good_JJ` with `He_PRP` hanging
off `has` by an nsubj arc, and buffer `control_NN …`.

An arrow points down to a long binary vector `0 0 0 1 0 0 1 0 … 0 0 1 0`, labelled
**binary, sparse — dim = 10⁶–10⁷**.

Feature templates: usually a combination of 1–3 elements from the configuration

**Indicator features** (boxed in red dots):

    s1.w = good ∧ s1.t = JJ
    s2.w = has ∧ s2.t = VBZ ∧ s1.w = good
    lc(s₂).t = PRP ∧ s₂.t = VBZ ∧ s₁.t = JJ
    lc(s₂).w = He ∧ lc(s₂).l = nsubj ∧ s₂.w = has

## Slide 37 — Evaluation of Dependency Parsing: (labeled) dependency accuracy

The sentence, with gold arcs drawn above:

    ROOT   She   saw   the   video   lecture
     0      1     2     3      4        5

Boxed at the right:

    Acc = # correct deps / # of deps

    UAS = 4 / 5 = 80%
    LAS = 2 / 5 = 40%

Two tables:

| Gold | | | |
| --- | --- | --- | --- |
| 1 | 2 | She | nsubj |
| 2 | 0 | saw | root |
| 3 | 5 | the | det |
| 4 | 5 | video | nn |
| 5 | 2 | lecture | obj |

| Parsed | | | |
| --- | --- | --- | --- |
| 1 | 2 | She | nsubj |
| 2 | 0 | saw | root |
| 3 | 4 | the | det |
| 4 | 5 | video | nsubj |
| 5 | 2 | lecture | ccomp |

## Slide 38 — Handling non-projectivity

- The arc-standard algorithm we just presented only builds projective dependency trees
- Possible directions to head:
  1. Just declare defeat on nonprojective arcs 🤷
  2. Use dependency formalism which only has projective representations
     - A CFG only allows projective structures; you promote head of projectivity
       violations
  3. Use a postprocessor to a projective dependency parsing algorithm to identify and
     resolve nonprojective links
  4. Add extra transitions that can model at least most non-projective structures (e.g.,
     add an extra SWAP transition will allow any non-projectivity, cf. bubble sort)
  5. Move to a parsing mechanism that does not use or require any constraints on
     projectivity (e.g., the graph-based MSTParser or Dozat and Manning (2017))

## Slide 39 — 4. Why do we gain from a neural dependency parser? Indicator Features Revisited

Categorical features are:

- **Problem #1:** sparse
- **Problem #2:** incomplete
- **Problem #3:** expensive to compute

More than 95% of parsing time is consumed by feature computation

*(The same red-dotted box of indicator features from slide 36 is repeated at the lower
left.)*

**Neural Approach:** learn a dense and compact feature representation

The same stack/buffer configuration is redrawn at the right, with an arrow down to a
short vector of real numbers `0.1 0.9 -0.2 0.3 … -0.1 -0.5`, labelled
**dense — dim = ~1000**.

## Slide 40 — A neural dependency parser [Chen and Manning 2014]

*(A photograph of Danqi Chen sits in the top right corner.)*

- Results on English parsing to Stanford Dependencies:
  - Unlabeled attachment score (UAS) = head
  - Labeled attachment score (LAS) = head and label

| Parser | UAS | LAS | sent. / s |
| --- | --- | --- | --- |
| MaltParser | 89.8 | 87.2 | 469 |
| MSTParser | 91.4 | 88.1 | 10 |
| TurboParser | **92.3** | 89.6 | 8 |
| C & M 2014 | 92.0 | **89.7** | **654** |

## Slide 41 — First win: Distributed Representations

- We represent each word as a *d*-dimensional dense vector (i.e., word embedding)
  - Similar words are expected to have close vectors.
- Meanwhile, **part-of-speech tags** (POS) and **dependency labels** are also represented
  as *d*-dimensional vectors.
  - The smaller discrete sets also exhibit many semantical similarities.

Boxed at the bottom:

    NNS (plural noun) should be close to NN (singular noun).
    nummod (numerical modifier) should be close to amod (adjective modifier).

The figure at the right is a schematic 2-D vector space in which each word is drawn as a
little three-cell vector: *was*, *were* and *is* cluster together at the upper left,
*come* and *go* cluster at the lower middle, and *good* sits alone at the right.

## Slide 42 — Extracting Tokens & vector representations from configuration

- We extract a set of tokens based on the stack / buffer positions:

The configuration at the top is the same one as slide 36: stack `ROOT has_VBZ good_JJ`,
`He_PRP` attached to `has` by nsubj, buffer `control_NN …`.

Below it, a table of the extracted tokens with three columns — word, POS, dep.:

| | word | POS | dep. |
| --- | --- | --- | --- |
| s₁ | good | JJ | ∅ |
| s₂ | has | VBZ | ∅ |
| b₁ | control | NN | ∅ |
| lc(s₁) | ∅ | ∅ | ∅ |
| rc(s₁) | ∅ | ∅ | ∅ |
| lc(s₂) | He | PRP | nsubj |
| rc(s₂) | ∅ | ∅ | ∅ |

The three columns are joined with + signs, and a large brace at the right is annotated:
*A concatenation of the vector representation of all these is the neural representation
of a configuration.*

## Slide 43 — Second win: Deep Learning classifiers are non-linear classifiers

- **Traditional ML classifiers** (including Naïve Bayes, SVMs, logistic regression and
  softmax classifier) are not very powerful classifiers: they only **give linear decision
  boundaries**
- But **neural networks** can use multiple layers to learn much more complex **nonlinear
  decision boundaries**

The same two classification plots as lecture 3 slide 8: the same green/red point cloud
split first by a straight diagonal boundary that misclassifies many points, then by a
curved boundary that follows the clusters.

## Slide 44 — Neural Dependency Parser Model Architecture (A simple feed-forward neural network multi-class classifier)

The network drawn bottom to top:

    Input layer x      lookup + concat
    Hidden layer h     h = ReLU(Wx + b₁)
    Output layer y     y = softmax(Uh + b₂)

The input layer is a row of coloured cells (yellow for word embeddings, red for POS, blue
for dependency labels), fed from the stack/buffer configuration box below it. Arrows
connect it fully to an 8-unit hidden layer, and that fully to a 6-unit output layer.

**Softmax probabilities** → { Shift , Left-Arc_r , Right-Arc_r }

Annotations:

- Log loss (cross-entropy error) will be back-propagated to the embeddings
- The hidden layer re-represents the input — it moves inputs around in an intermediate
  layer vector space—so it can be easily classified with a (linear) softmax
- **Wins:** Distributed representations! Non-linear classifier!

## Slide 45 — Dependency parsing for sentence structure

*(A photograph of Danqi Chen in the top right corner.)*

Chen & Manning (2014) showed that neural networks can accurately determine the structure
of sentences, supporting meaning interpretation

A parsed sentence with POS tags in coloured boxes and labelled arcs above:

    Markets  have  been  jolted  by  concerns  about  China .
     NNS     VBP   VBN    VBN    IN    NNS      IN     NNP  .

with arcs nsubjpass (jolted → Markets), aux (jolted → have), auxpass (jolted → been),
nmod (jolted → concerns), case (concerns → by), nmod (concerns → China), case
(China → about).

This paper was the first simple and successful neural dependency parser

The dense representations (and non-linear classifier) let it outperform other greedy
parsers in both accuracy and speed

## Slide 46 — Further developments in transition-based neural dependency parsing

This work was further developed and improved by others, including in particular at Google

- Bigger, deeper networks with better tuned hyperparameters
- Beam search
- Global, conditional random field (CRF)-style inference over the decision sequence

Leading to SyntaxNet and the Parsey McParseFace model (2016):
"The World's Most Accurate Parser"
https://research.googleblog.com/2016/05/announcing-syntaxnet-worlds-most.html

| Method | UAS | LAS (PTB WSJ SD 3.3) |
| --- | --- | --- |
| Chen & Manning 2014 | 92.0 | 89.7 |
| Weiss et al. 2015 | 93.99 | 92.05 |
| Andor et al. 2016 | 94.61 | 92.79 |

## Slide 47 — Graph-based dependency parsers

- Compute a score for every possible dependency for each word
  - Doing this well requires good "contextual" representations of each word token, which
    we will develop in coming lectures

The figure scores candidate heads for the word *big* in "ROOT The big cat sat": arcs into
*big* are drawn from ROOT (0.5), from *The* (0.3), from *cat* (2.0) and from *sat* (0.8).

Caption: e.g., picking the head for "big"

## Slide 48 — Graph-based dependency parsers

- Compute a score for every possible dependency (choice of head) for each word
  - Doing this well requires more than just knowing the two words
  - We need good "contextual" representations of each word token, which we will develop
    in the coming lectures
- Repeat the same process for each other word; find the best parse (MST algorithm)

The same figure, with the winning arc — *cat* → *big*, score 2.0 — now drawn in red.

## Slide 49 — A Neural graph-based dependency parser [Dozat and Manning 2017; Dozat, Qi, and Manning 2017]

- This paper revived interest in graph-based dependency parsing in a neural world
  - Designed a biaffine scoring model for neural dependency parsing
    - Also crucially uses a neural sequence model, something we discuss next week
- Really great results!
  - **But slower than the simple neural transition-based parsers**
    - There are n² possible dependencies in a sentence of length *n*

| Method | UAS | LAS (PTB WSJ SD 3.3) |
| --- | --- | --- |
| Chen & Manning 2014 | 92.0 | 89.7 |
| Weiss et al. 2015 | 93.99 | 92.05 |
| Andor et al. 2016 | 94.61 | 92.79 |
| **Dozat & Manning 2017** | **95.74** | **94.08** |
