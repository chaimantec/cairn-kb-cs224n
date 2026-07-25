# Dependency grammar

The representation of sentence structure that this course uses. Instead of grouping words
into nested phrases, a dependency analysis asks, for each word, **which other word it
depends on** — modifies, attaches to, or is an argument of — and labels that relation.

Primary source: [lecture 4](04-dependency-parsing.md), slides 7, 19–29
([slide text](../raw/slides/04-dependency-parsing.md)).

## Dependencies, heads and dependents

Dependency syntax postulates that syntactic structure consists of relations between lexical
items: binary asymmetric relations, drawn as arrows, called **dependencies** (slide 19). An
arrow connects a **head** — also called governor, superior, or regent — to a **dependent** —
also modifier, inferior, or subordinate (slide 21).

For *Look in the large crate in the kitchen by the door* (slide 7), the head of the whole
thing is *look*, because the sentence is a looking command. *Crate* is what the looking is
in; *the*, *large*, *in the kitchen* and *by the door* all modify *crate* (≈10:11).

The arrows are usually **typed** with the name of a grammatical relation (slide 20). On
*Bills on ports and immigration were submitted by Senator Brownback, Republican of Kansas*:

- `nsubj:pass` (submitted → Bills), `aux` (submitted → were), `obl` (submitted → Brownback)
- `nmod` (Bills → ports), `case` (ports → on), `cc` (ports → and), `conj` (ports → immigration)
- `case` (Brownback → by), `flat` (Brownback → Senator), `appos` (Brownback → Republican)
- `nmod` (Republican → Kansas), `case` (Kansas → of)

Manning is explicit that you are not expected to master the whole relation inventory
(≈30:59). What you should take away is the shape of the thing: it operates at the phrase
level and records what modifies what. The assignment does ask which word a given
prepositional phrase modifies, so that much you should be able to answer (≈31:44).

## Two conventions that vary between sources

- **Arrow direction.** Some people draw head → dependent, others dependent → head. This
  course follows Tesnière and points **from head to dependent** (slide 24, ≈37:57). If you
  read a paper whose diagrams look backwards, this is why.
- **What counts as the head.** Dependency grammar is not one uniquely defined theory.
  People disagree about which element to treat as head in various constructions, and
  linguists argue about the right analysis of all sorts of sentences. Manning's example, in
  answer to a student question: in Universal Dependencies, the copula *is* is **not** taken
  as the head of its sentence, though some frameworks do (≈51:10). His point is that
  disagreement about which analysis is right does not mean disagreement that there are
  units, phrases, modifiers and ambiguities to analyse (≈51:55).

## Trees and the fake ROOT

Dependencies are generally taken to form a **tree**: connected, acyclic, with a single root
(slide 21). Parsing a sentence then means choosing, for each word, which other word
(including ROOT) it is a dependent of (slide 28), subject to the constraints that only one
word is a dependent of ROOT and that there are no cycles A → B, B → A.

Conventionally a **fake ROOT** node is added at the front of the sentence, so that every
real word is the dependent of exactly one node — including the sentence's main word, which
becomes ROOT's single dependent (slide 24). This is a convenience that makes the parsing
algorithms work out more cleanly (≈38:43).

Normally the main word of a sentence is its **verb**, and the verb's arguments — subject,
object, prepositional phrase modifiers — are all its dependents (≈49:38). Not every language
requires a verb in a sentence, and even English has restricted verbless cases: in *easy as
pie*, the predicate adjective *easy* is the head (≈50:24).

## Projectivity

A parse is **projective** if, with the words in their linear order and all arcs drawn above
them, **no two arcs cross** (slide 29). Dependencies derived from a context-free grammar
tree are necessarily projective, because taking one child of each category as head can never
produce a crossing.

Most syntactic structure is projective, but not all, and dependency theory normally allows
**non-projective** structures because you cannot get the semantics of certain constructions
right without them. Two examples:

- *I'll give a talk tomorrow on neural networks* (slide 28) — *on neural networks* modifies
  *talk*, while *tomorrow* is an argument of *give*, so the two arcs cross (≈51:55).
- *Who did Bill buy the coffee from yesterday?* (slide 29) — *who* is the object of the
  preposition *from* but has been displaced to the front of the sentence.

Slide 38 lists five ways of coping:

1. Declare defeat on non-projective arcs.
2. Use a formalism that only has projective representations (a CFG), promoting the head of
   any violation.
3. Post-process a projective parse to identify and resolve non-projective links.
4. Add extra transitions — a **SWAP** transition allows any non-projectivity, on the logic
   of bubble sort.
5. Use a parsing mechanism that never imposed the constraint, such as a graph-based parser.

**The parser built in this course is projective-only** (≈53:29); the arc-standard system in
[transition-based parsing](transition-based-parsing.md) can only produce projective trees.

## History: dependency is the old idea

The usual assumption is backwards (slide 22). Dependency grammar is ancient and
constituency is the recent invention.

- **Pāṇini's grammar of Sanskrit**, composed somewhere between roughly the 8th and 4th
  centuries BCE — nobody knows exactly when — used a dependency analysis. Pāṇini is
  heralded as the first dependency grammarian, and really as the first person to write the
  grammar of a human language at all (≈33:17). The grammar was composed **orally** and
  transmitted that way for centuries, which is part of why the dating is so uncertain, and
  which Manning says blows his mind (≈36:24). The birch-bark manuscript on slide 23 post-
  dates the composition by about a millennium.
- **Arabic grammarians** of the first millennium took the same basic approach.
- **Constituency/context-free grammar** dates only to the 20th century: R.S. Wells in 1947,
  canonicalized by Chomsky in the 1950s.
- **Lucien Tesnière (1959)** is the usual source for modern dependency work. It was the
  dominant approach in the "East" — Russia, China — through the 20th century, and suits
  freer-word-order, inflected languages like Russian or Latin.
- **David Hays**, a founder of US computational linguistics, built an early (possibly the
  first) dependency parser in 1962.

Two asides for computer scientists. **Reed–Kellogg sentence diagramming**, widely taught in
American schools to anyone now over fifty, was a somewhat quirky form of dependency grammar
— heads with their dependents written underneath, using lines at different angles (≈30:11).
And the **Chomsky hierarchy** was not invented to torture CS103 students or to explain
formal language theory; it came out of an argument about human language, that finite-state
mechanisms are an inadequate formalism for its complexity (≈34:51).

## Treebanks and Universal Dependencies

Writing grammars by hand worked badly — the long tail of creative usage defeats coverage,
and a hand-written grammar gives no reason to prefer one parse over another (≈39:29). From
the late 1980s the field instead built **treebanks**: corpora of hand-annotated sentences.
The lineage on slide 25 runs Brown corpus (1967; POS-tagged 1979), Lancaster–IBM Treebank
(late 1980s), the Penn Treebank (Marcus et al. 1993), and now
[Universal Dependencies](http://universaldependencies.org/) — over a hundred languages
annotated with a uniform dependency formalism, which makes it valuable for crosslinguistic
and psycholinguistic work (≈42:35). Assignment 2 trains on data of this kind.

Slide 26 lists what a treebank buys that a grammar does not: **reusability** of the
annotation labour across parsers, taggers and linguistic studies; **broad coverage** rather
than a few intuitions; **frequencies and distributional information** to learn from; and a
way to **evaluate** systems. Manning underlines the last as the one that may seem comical
today but was transformative: through the 1950s, 60s and 70s nobody had evaluation methods,
and you demonstrated a good parser by running it, typing a sentence, and saying *look, it
worked*. Being able to measure a parser against a thousand hand-parsed sentences was a
revolutionary development of the late 1980s and 1990s (≈44:55).

## What a parser can condition on

Slide 27's four sources of information, illustrated on *Discussion of the outstanding issues
was completed*:

1. **Bilexical affinities** — is this pair of words a plausible head/dependent pair?
   *discussion → issues* is; *completed → the* is not.
2. **Dependency distance** — most dependencies are between nearby words.
3. **Intervening material** — dependencies rarely span an intervening verb or punctuation.
4. **Valency of heads** — how many dependents, on which side, this head usually takes.
   *Broke* usually has something on its left (who did the breaking) and often something on
   its right, though *the cup broke* has nothing on the right; and it will not take an
   unbounded list of arguments (≈48:04).

## Related pages

- [Transition-based parsing](transition-based-parsing.md) — how these structures are
  actually built, and how parses are scored.
- [Syntactic ambiguity](syntactic-ambiguity.md) — why the structure is hard to recover.
- [Lecture 4](04-dependency-parsing.md) — the full lecture.
