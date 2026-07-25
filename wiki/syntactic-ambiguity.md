# Syntactic ambiguity

Why recovering sentence structure is hard, and why it is a machine learning problem rather
than a rule-following one. The examples are the memorable part of
[lecture 4](04-dependency-parsing.md), but the argument underneath them is what matters:
human language is **globally** ambiguous in a way programming languages are not.

Primary source: lecture 4, slides 9–17
([slide text](../raw/slides/04-dependency-parsing.md)), ≈14:50–27:06.

## The setup: language is a linear stream

Humans communicate in a linear stream (≈12:31). In writing it is a linear sequence of words;
in speech it is not even that, just a continuous sound stream — there is no white space
between spoken words, and speakers pause at clause and sentence boundaries at best. To get
a meaning out of that stream, a listener has to work out **what modifies what** (slide 9).
A model has to do the same to interpret language correctly. Ambiguity is the evidence that
this step is real work rather than a formality.

## Global vs local ambiguity

Manning's central claim (≈17:56). In **programming languages** you have *local* ambiguities,
but they are always resolved by rule: `else` is construed with the nearest `if`; Python uses
indentation instead. There is never **global** ambiguity in a programming language — some
rule always decides.

**Human languages are not like that.** Nothing in the string resolves which reading is
correct, and making the sentence longer does not help. You are simply meant to read it and
use context and your intelligence to decide what is going on (≈18:42). This is why parsing
is a prediction problem: a hand-written grammar can *license* all the readings, but only
statistics about how often particular words modify particular other words give you a reason
to prefer one (≈41:49).

An aside worth keeping (≈24:00): the examples here are all English, and different languages
do not have all the same ambiguities. In Chinese, the prepositional phrase example below
does not arise, because a PP modifying the verb appears *before* the verb while the object
noun comes after, making the attachment unambiguous. That does not make Chinese
unambiguous — it has plenty of bad ambiguities of its own — just differently structured.

## The four types

### Prepositional phrase attachment

The most common source in English, because prepositional phrases are extremely common
(slides 10–11). *Scientists count whales from space* — is the counting done from space
(correct, and what the BBC article was about), or are they whales *from space*? Whenever a
PP follows a noun phrase that follows a verb, it is ambiguous which earlier element it
attaches to (≈17:09).

### Coordination scope

Slides 14–15. *Shuttle veteran and longtime NASA executive Fred Gregory appointed to
board*: either one person — Fred Gregory, who is both a shuttle veteran and a NASA
executive — or two, a shuttle veteran plus a longtime NASA executive named Fred Gregory
(≈22:29). Related is **apposition**, where a descriptive noun phrase names another, as in
*the author Fred Gregory*.

The same ambiguity arises without an explicit coordinator, using juxtaposition and a comma
(≈24:47): *Doctor: No heart, cognitive issues* — does *no* scope over both *heart issues*
and *cognitive issues*, or did the doctor find *no heart*?

### Adjectival/adverbial modifier

Slide 16. *Students get first hand job experience* — the grouping [first hand] [job
experience] versus the reading that comes to you if you have a smutty mind (≈25:33).

### Verb phrase attachment

Slide 17. *Mutilated body washes up on Rio beach to be used for Olympics beach volleyball* —
the infinitival phrase behaves much like a prepositional phrase in that it can attach to
different things: is it the beach that will be used for beach volleyball, or the body
(≈26:20)?

## Why the ambiguities multiply: Catalan numbers

The scaling argument, and the reason parsing cannot proceed by enumeration (slide 12). In a
real *Wall Street Journal* sentence:

    The board approved [its acquisition] [by Royal Trustco Ltd.]
                                                    [of Toronto]
                                              [for $27 a share]
                                          [at its monthly meeting].

four prepositional phrases follow the object, and each must attach to something earlier.
The attachments Manning works through (≈19:27): *by Royal Trustco Ltd.* modifies *the
acquisition*; *of Toronto* modifies *Royal Trustco Ltd.* — so a PP can modify an earlier PP,
or the noun phrase inside it; *for $27 a share* goes back to modifying *the acquisition*;
and *at its monthly meeting* goes all the way back up to the approval.

You do not get a free factorial choice of attachment points, because there is a restriction
that **the dependencies must not cross**: once you have gone back further, you must stay
equally far back or go further still (≈20:58). The resulting count is a **Catalan number**,

    C_n = (2n)! / [(n+1)! n!]

an exponentially growing series that shows up wherever a tree-like non-crossing constraint
does — the number of triangulations of a polygon with *n*+2 sides, and the triangulation of
probabilistic graphical models in CS228.

*(The transcript's spoken counts of readings for four and five prepositional phrases do not
match this formula, which gives C₄ = 14 and C₅ = 42, and the deck prints no table of values;
the [transcript](../raw/transcripts/04-dependency-parsing.md) marks the discrepancy at
≈21:43 rather than resolving it.)*

The point Manning draws from the example is not the arithmetic but the contrast with human
performance: people read sentences like this every morning over their corn flakes, and
their brains do not explode enumerating the alternatives. We just do it as we go along, and
it seems obvious (≈21:43).

## Why this matters practically

Two consequences run through the rest of the course.

**It is why hand-written grammars failed.** Early NLP wrote grammar rules and dictionaries,
and it worked badly for two reasons (≈39:29). Coverage: beneath the canonical structures
lies a long tail of creative usage — Yoda word order, or appending *not* to the end of a
sentence — that a hand-written grammar never captures. And, more importantly, ambiguity: the
grammar licenses every reading and ranks none. This is what drove the field toward annotated
[treebanks](dependency-grammar.md) and statistical models from the late 1980s.

**Resolving it is useful downstream.** Slide 18 shows dependency paths extracting
protein–protein interactions from biomedical text: from *The results demonstrated that KaiC
interacts rhythmically with SasA, KaiA and KaiB*, the path
`KaiC ←nsubj interacts nmod:with→ SasA`, plus the conjunction arcs to *KaiA* and *KaiB*,
yields three interaction facts. This is how structured databases of known interactions get
built (≈27:51).

Manning's summary: linguistic structure is useful, and it is syntactically very ambiguous, so
you should think of humans as **active interpreters** using contextual knowledge — of what
came earlier in the text, and of how the world works — to arrive at the right structure
(≈28:38).

## Related pages

- [Dependency grammar](dependency-grammar.md) — the representation whose arcs these
  ambiguities are about, and the non-crossing (projectivity) constraint.
- [Transition-based parsing](transition-based-parsing.md) — how a parser makes these
  attachment decisions in practice.
- [Lecture 4](04-dependency-parsing.md) — the full lecture.
