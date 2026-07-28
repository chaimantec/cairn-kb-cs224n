# Formal semantics and the denotational theory of meaning

The theory of meaning that dominated both linguistics and AI before distributional methods took
over: **the meaning of an expression is its denotation** — what it picks out in the world, or in
a formal model of the world. The meaning of *computer* is the set of computers
([lecture 18](18-nlp-linguistics-philosophy.md), ≈48:19).

This page covers the tradition, the machinery, why it was abandoned in practice, and what
[lecture 18](18-nlp-linguistics-philosophy.md) argues it still gets right. The rival account is
[distributional semantics](distributional-semantics.md), and the two are set against each other
on slide 35:

| | Model-theoretic semantics | Distributional semantics |
| --- | --- | --- |
| Meaning of a word is | its denotation in (a model of) the world | the contexts in which it occurs |
| Built from | logic, set theory, lambda calculus | co-occurrence statistics |
| Used by | symbolic AI, formal linguistics, ~1960–2017 | word vectors and everything after |

## Tarski's position, and Montague's break from it

From roughly 1940 to 1980 Alfred Tarski was the preeminent logician in the United States, and his
view was that meaning could not be studied through natural language at all, "because human
languages were, quote, 'impossibly incoherent'" (lecture 18, ≈50:40). Formal semantics was for
formal languages. The residue of that position is the shape of an introductory logic course: "in
the early weeks — weeks one and two of the logic class — you have some English sentences which you
translate into formal logic, and then, after that, you forget about human languages, and you just
start proving stuff about formal logical systems" (≈49:53).

**Richard Montague** (1930–1971), a student of Tarski's, rejected exactly this (slide 37):

> I reject the contention that an important theoretical difference exists between formal and
> natural languages.

He then showed how to build a formal semantics for natural language, and "Richard Montague's work
became the foundation of the work that's used in semantics in linguistics" — the picture taught in
Stanford's Linguistics 130 and 230 (≈51:27). Its formal core is the
[compositionality](compositionality.md) requirement stated as a homomorphism: the syntax is an
algebra, the semantics is an algebra, and there is a structure-preserving map from one to the
other (slide 39).

## The machinery

The worked example runs through slides 36, 40 and 41. Target logical form for *The red apple is on
the table*:

$$on(\iota(\lambda x(apple(x) \wedge red(x))),\, \iota(\lambda y.\, table(y)))$$

Here $\lambda x.\,\phi(x)$ is a function from an individual to a truth value — a *property* — and
$\iota$ is the definite-description operator, "the unique thing such that." So the formula says:
the unique red apple stands in the *on* relation to the unique table.

Slide 36's question is the one that matters for NLP: "but how do we get the latter from the
former? Other than by setting undergrads to work…"

The answer is a three-step pipeline (slides 40–41):

1. **Parse.** Produce a syntactic structure — S → NP VP, with NP → *The* N′, N′ → *red apple*, and
   VP → *is* PP.
2. **Lexical lookup.** Each word carries a lambda term: *the* is $\lambda P.\, \iota(P)$, *apple*
   is $\lambda x.\, apple(x)$, *red* is $\lambda P.\lambda x(P(x) \wedge red(x))$, *on* is
   $\lambda y.\lambda x(on(x,y))$, and *is* is the identity $\lambda P.\, P$.
3. **Semantic composition**, by a *rule-to-rule* correspondence in which each syntactic rule has a
   matching semantic one — e.g. $PP{:}\,\alpha(\beta) \to P{:}\,\alpha\ NP{:}\,\beta$ says that a
   PP's meaning is its preposition's meaning applied to its NP's meaning.

Apply the terms up the tree and the root is the formula above. Every step is determined by the
syntax; nothing is heuristic.

## What it was used for

"This was most of Natural Language Understanding, 1967–2017" is slide 42's title, and it is not
hyperbole about the era's database front-ends. The slide's example is *How many red cars in Palo
Alto does Kathy like?*, derived to

$$\lambda x.\, car(x) \wedge in'(paloalto)(x) \wedge red'(x) \wedge like(x)(kathy)$$

and then compiled straight into SQL to run against a database — a join over `Likes`, `Cars`,
`Locations` and `Reds` with a `count(*)`. Manning notes it is "approximately a slide — an actual
slide that I used to use in CS224N in the 2000s decade" (≈53:01–53:47).

The final phase was **semantic parsing**, in which the components stopped being hand-written:
"this same basic technology was incorporated into a machine-learning context, where your goal was
to start to learn various of these parts. You could not only learn the parser, but you could also
learn semantic meanings of words and learn composition rules" (≈54:33). The lineage on the slide
is Zettlemoyer & Collins (2005), Artzi & Zettlemoyer (2013), and Liang, Jordan & Klein (2013) —
with Percy Liang's early Stanford work in this area "before he was convinced to do neural
networks."

**Why it lost**: "these systems could actually work, and were used in limited domains, but they're
always extremely brittle" (≈55:19). A sentence outside the grammar, a word outside the lexicon, or
a construction the composition rules do not cover produces nothing at all rather than a degraded
answer — the failure mode a statistical system does not have.

## What survives

Three things, on lecture 18's own account.

**The vocabulary for evaluating neural systems.** Compositionality, systematic generalization,
stable meanings for symbols, and manipulating reference are named on slide 30 as the linguistic
concepts "increasingly central in the research program of deep learning AI."

**The question of whether neural models compose at all.** Slide 44 is a single question — "do
neural models provide suitable meaning (composition) functions?" — and the answer given is
genuinely open. See [compositionality](compositionality.md).

**Evidence about humans.** Slide 43 collects work suggesting people also compute meaning
hierarchically, "following mostly projective bottom-up trees" (Crain and Nakayama 1987; Pallier et
al. 2011; Ding et al. 2016; Hale et al. 2018), while noting the controversy.

## The counter-case: meaning as use

The lecture does not end on the denotational side. Its own position, argued from Wittgenstein's
*Philosophical Investigations* (slide 45), is that denotation is the wrong model of what meaning
*is*:

> You say: the point isn't the word, but its meaning, and you think of the meaning as a thing of
> the same kind as the word, though also different from the word. Here the word, there the
> meaning. The money, and the cow that you can buy with it. (But contrast: money, and its use.)

Against Bender and Koller's position that only form-plus-referent counts as meaning, Manning's two
claims (slide 46, ≈59:14):

1. **Meaning arises from connecting words to other things.** The real world is privileged but not
   uniquely so — virtual worlds ground meaning, and so do connections to other language.
2. **Meaning is gradient.** Not "you have the denotation or you don't" but how well you understand
   a word.

The *shehnai* example on slides 47–48 is the argument in miniature, and is worked through on the
[lecture 18 page](18-nlp-linguistics-philosophy.md#what-is-meaning).

## See also

- [Distributional semantics](distributional-semantics.md) — the theory that displaced this one.
- [Compositionality](compositionality.md) — the principle both traditions claim.
- [Lecture 18](18-nlp-linguistics-philosophy.md) — the source lecture.
- [Symbolic and neural AI](symbolic-and-neural-ai.md) — the wider version of the same split.
- [Dependency grammar](dependency-grammar.md) and
  [syntactic ambiguity](syntactic-ambiguity.md) — the parsing step this pipeline depends on.
