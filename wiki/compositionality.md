# Compositionality

**The meaning of a whole is a function only of the meanings of its syntactic parts and the manner
in which those parts are combined.** Known as *Frege's principle* — with the caveat
[lecture 18](18-nlp-linguistics-philosophy.md) puts on the slide itself, that it is "very unclear
that he either said or believed in it" (slide 38).

It is the organizing idea behind two otherwise unrelated parts of this course: the
[tree recursive networks](tree-recursive-neural-networks.md) of
[lecture 17](17-convnets-and-treernns.md), which try to implement it in vectors, and the
[formal semantics](formal-semantics.md) tradition of
[lecture 18](18-nlp-linguistics-philosophy.md), which formalized it in logic. It is also the
standard against which large language models are found wanting.

## Why it matters

Two payoffs, both stated on slide 38 of lecture 18.

**Exponential representational power.** If meanings compose, then a finite lexicon and a finite
set of combination rules generate unboundedly many sentence meanings. This is the same
observation von Humboldt made about form — that language must "make infinite use of finite means"
(lecture 18, ≈41:13) — applied to content. You can interpret a sentence you have never seen
because you have seen its parts.

**Systematic generalization**, the closely related property: "if a human or model can interpret a
noun phrase in subject position, then it should also be able to interpret it in object position"
(Fodor & Pylyshyn 1988). Understanding is systematic rather than item-by-item. The developmental
evidence is striking — children of **2 years 11 months** already generalize this way (Brooks &
Tomasello 1999) — and it is what makes rapid language acquisition possible.

Slide 30 of lecture 18 lists compositionality and systematic generalization first among the
"fundamental concepts of linguistics [that] are increasingly central in the research program of
deep learning AI," alongside stable meanings for symbols and manipulating reference.

## The logical version

In the [Montague](formal-semantics.md) tradition, compositionality is a structural requirement,
not a slogan. Partee's formulation of Montague (1970), quoted on slide 39 of lecture 18:

> The central idea is that anything that should count as a grammar should be able to be cast in
> the following form: the syntax is an algebra, the semantics is an algebra, and there is a
> homomorphism mapping elements of the syntactic algebra onto elements of the semantic algebra …
> It is the homomorphism requirement, **which is in effect the compositionality requirement**,
> that provides the most important constraint on UG in Montague's sense.

A homomorphism is a structure-preserving map: combine two syntactic pieces and interpret the
result, or interpret each piece and combine the interpretations, and you must get the same
answer. That is compositionality stated precisely.

Worked concretely (lecture 18, slides 40–41), *The red apple is on the table* parses to a tree,
each word gets a lexical entry — *the* is $\lambda P.\iota(P)$, *red* is
$\lambda P.\lambda x(P(x) \wedge red(x))$, *apple* is $\lambda x.\, apple(x)$ — and each node
applies its children by a rule-to-rule correspondence until the root reads

$$on(\iota(\lambda x(apple(x) \wedge red(x))),\, \iota(\lambda y.\, table(y)))$$

Nothing in that derivation is optional or heuristic: the syntax dictates the semantics at every
step.

## The vector version

[Lecture 17](17-convnets-and-treernns.md) asks the obvious question for a course about
representations: can you do the same thing with word vectors instead of logical forms?

The [TreeRNN](tree-recursive-neural-networks.md) answer is to make composition a *learned
function* applied at every node of a parse tree (slide 38 of lecture 17, ≈47:25):

$$p = \tanh\!\left(W \begin{bmatrix} c_1 \\ c_2 \end{bmatrix} + b\right)$$

where $c_1, c_2$ are the child vectors and the same $W$ is reused everywhere — the neural analogue
of a single, uniform composition rule. The hoped-for result is that *the country of my birth* ends
up near *the place where I was born* in the vector space, and near location words generally.

The limitation is instructive, because it is a *linguistic* one. A single $W$ means one
composition operation for every syntactic relation, and relations differ: in *the red ball*,
*red* supplies attributes of the noun; in *kick the ball*, the object plays an entirely different
role (lecture 17, ≈55:15). The **Recursive Neural Tensor Network** answers this by letting the
children interact multiplicatively,

$$p = f\!\left(\begin{bmatrix} b \\ c \end{bmatrix}^{T} V^{[1:2]} \begin{bmatrix} b \\ c
\end{bmatrix} + W \begin{bmatrix} b \\ c \end{bmatrix}\right)$$

so that each child's values can modulate the other's contribution rather than merely adding to it.

## The test case: negation

The cleanest empirical evidence that composition is doing real work, from lecture 17's
[sentiment](sentiment-analysis.md) experiments (slide 59):

- *Should have been funnier and more entertaining* is negative although every sentiment-bearing
  word in it is positive — the polarity is determined by what scopes over what.
- *It's definitely not dull* is positive although *dull* is strongly negative and *not* is, in
  corpus terms, a negative word.

Word-level and bigram models get the first kind of case (negating a positive) because *not* is
itself negatively distributed, but fail the second (negating a negative), showing changes in
activation of roughly zero. Only the RNTN, which composes up the tree, moves in the right
direction. Manning's assessment is that this specific result "still isn't captured as well by any
of the current Transformer models" (lecture 17, ≈1:09:14).

## Do neural language models compose?

[Lecture 18](18-nlp-linguistics-philosophy.md) devotes a whole slide to the question — slide 44 is
nothing but "Do neural models provide suitable meaning (composition) functions?" — and the answer
given is deliberately unresolved (≈56:07):

> in many ways, they seem to — they do an amazing job at understanding whatever sentences you put
> into them — but there are still some genuine concerns as to whether they're taking shortcuts, or
> working, to a certain extent, and don't actually have the same kind of compositional
> understanding, with systematic generalization, that human beings do.

Two pieces of evidence bear on it from elsewhere in the same lecture. The generalization slide
(slide 10) shows an LSTM learning a finite-automaton pattern from far fewer samples than a
Transformer needs — sample efficiency being exactly what compositional structure should buy you.
And the human-processing slide (slide 43) cites evidence that people compute meaning
hierarchically over mostly projective bottom-up trees (Crain and Nakayama 1987; Pallier et al.
2011; Ding et al. 2016; Hale et al. 2018), which "is obviously not what we're getting with
present-day Transformers."

The open question Manning leaves at the end of lecture 17 is whether the two can be combined:
"something that's a bit more tree-structured, while still more flexible, like a Transformer"
(≈1:11:35).

## See also

- [Formal semantics](formal-semantics.md) — the logical tradition that made compositionality
  precise.
- [Tree recursive neural networks](tree-recursive-neural-networks.md) — the neural
  implementation, and the negation results.
- [Distributional semantics](distributional-semantics.md) — the rival theory of word meaning,
  which says nothing about how meanings combine.
- [Lecture 17](17-convnets-and-treernns.md) and
  [lecture 18](18-nlp-linguistics-philosophy.md) — the two lectures this idea spans.
- [Syntactic ambiguity](syntactic-ambiguity.md) — why getting the structure right is prior to
  getting the meaning right.
