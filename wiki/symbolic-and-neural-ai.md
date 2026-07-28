# Symbolic AI and the cybernetics tradition

Two research programmes with different origins, and the distinction matters for a specific reason:
it is the frame in which "is language symbolic?" gets confused with "must a language processor be
symbolic?" [Lecture 18](18-nlp-linguistics-philosophy.md) separates those questions and answers
them differently (slides 24–34).

## Two lineages

**Symbolic AI** "dominated AI from the '60s until about 2010" (lecture 18, ≈25:46). John McCarthy
— a mathematician and logician, and the founder of Stanford's AI Lab — coined the term *artificial
intelligence*, and Manning stresses that this was a deliberate act of separation: "he very
explicitly chose a new name to disassociate what he was doing from the cybernetics approach"
(≈28:51). McCarthy "wanted to construct an artificial intelligence that looked like math and
logic." Marvin Minsky founded the parallel effort at MIT; Allen Newell and Herbert Simon were at
CMU.

**Cybernetics** was the older alternative, from the 1940s–50s, with Norbert Wiener at MIT as its
central figure. Its origins were "in sort of control and communication, so it's much nearer to an
electrical-engineering kind of background," and its aim was to unify control and communication in
animals and machines (≈31:11). The name comes from Greek *kybernetes*, "steersman" — the same root
as *Kubernetes*, and, through Latin, as *government*, "of course, it's a control system as well"
(≈31:58).

The claim worth keeping: **"in a very real sense, neural networks is a continuation of the
cybernetics tradition, rather than the AI tradition that started in the '50s and '60s"** (≈25:46).
The earliest neural networks — Frank Rosenblatt's Perceptron, physically wired, built for vision —
belong to that lineage, not to McCarthy's.

## The physical symbol system hypothesis

Newell and Simon's thesis, and the philosophical core of classical AI (lecture 18, ≈30:25):

> A physical symbol system has the necessary and sufficient means for general intelligent action.

Manning draws attention to how strong this is. *Sufficient* says a symbol system is enough for
general intelligence. **Necessary** says you cannot have general intelligence *without* one — "so
that was sort of the basis of classical AI," and it is the half that the success of neural
networks contradicts.

## An aside on hype

Slide 27 reproduces the *New York Times* of 8 July 1958 announcing Rosenblatt's Perceptron: "New
Navy Device Learns by Doing — Psychologist Shows Embryo of Computer Designed to Read and Grow
Wiser," reporting that the Navy "expects [it] will be able to walk, talk, see, write, reproduce
itself, and be conscious of its existence."

Read further down the same article and the demonstration was that the machine learned to
distinguish right-pointing from left-pointing arrows after fifty exposures (≈32:45–33:31). Manning
offers this "in case you think that AI hype is only a thing of the 2020s."

## Manning's position: the system versus its processor

This is the payoff of the section (slide 28, ≈33:31–35:48), and it is two claims that are usually
collapsed into one.

**Language is a symbolic system.** Obvious in writing, and true of speech as well: the phones of a
language are discrete categories, "and they're recognized in a symbolic way by language users." He
supports this with the history of the field — "all the pioneering work in categorical perception
in cognitive psychology is done with the sounds of human languages." So the *substrate* is
continuous (sound waves, or hand movements in sign languages) while the *structure* is symbolic.

**But the processor need not be.**

> The fact that humans use a symbol system for communication doesn't mean that the process that
> processes the symbols — the human brain — has to be a physical symbol system, and so, similarly,
> we don't have to design our NLP computer processors as physical symbol systems either. The brain
> is clearly much more like a neural network model, and probably neural models will scale better
> and capture language processing better than something that is a symbolic processor.

That is a rejection of the *necessary* clause of Newell and Simon while conceding everything the
symbolic tradition observed about language itself.

**Why would language be symbolic at all?** A functional hypothesis: **signalling reliability**.
"We could have just hummed at different frequencies, and that could have been used as our system
of communication," but discrete, well-separated targets are what let a listener recover the signal
when it degrades (≈35:48). Manning marks this as the dominant idea and offers it without
overclaiming — "which seems reasonable to me, but who knows."

## Stanford's Symbolic Systems, and why the name

A historical footnote with some content in it (slides 24, ≈26:32–28:05). Stanford is now unique in
having a **Symbolic Systems** program, and the name is due to **Jon Barwise** (1942–2000), who
"had a very strong belief that you needed to be dealing with meaning in the world, and the
connection between people's thinking and the world," and so refused to let it be called cognitive
science. Indiana had a Symbolic Systems program while Barwise was there; it reverted to cognitive
science after his death.

The distinction encoded in the name: cognitive science "focuses on the mind and intelligence as
naturally occurring phenomena," whereas Symbolic Systems "gives equal focus to human-constructed
systems that use symbols to communicate and to represent information" — human languages, logics,
programming languages — and to the systems that process them: brains, computers, complex social
systems.

## What linguistics is for now

Manning's answer is neither "the theories were right" nor "the theories were wrong" (slides 29–31,
≈36:34–38:52):

- **Not the fine detail.** "I don't think one necessarily wants to believe all the fine details of
  different linguistic theories," and particular formalisms "probably aren't the right thing to
  focus on in 2020s NLP."
- **Yes the broad picture.** "Most of our broad understanding of linguistics is right."
- **Above all, the questions.** "Linguistics gives us questions, concepts, and distinctions for
  examining languages and language acquisition and processing," and slide 29's own note is that
  "these tools are just as useful for studying computer-generated language."

Concretely, linguistics supplies the evaluation vocabulary: sentence structure, discourse
structure, natural language inference, hyperbole, translationese, prosody, morphology, indirect
speech acts, bridging anaphora, metaphor, reference, presupposition, stance, style. And slide 30
names the four concepts now most central to deep-learning research —
[**compositionality**](compositionality.md), **systematic generalization**, **stable meanings for
symbols**, **manipulating reference** — as "key to going from insect-level intelligence to
something like human intelligence."

That framing is worth quoting as Manning puts it: early neural work on vision and other sensory
processing "is sort of what gets you to insect-level intelligence. And if you want to get higher up
the chain than insect-level intelligence, then a lot of the questions and properties of linguistic
systems become increasingly relevant" (≈37:20–38:05).

## Language as a thinking tool

The section closes with two arguments that language is not only for communication.

**Wilhelm von Humboldt** (1767–1835) is cited by Chomsky for the claim that language makes
"infinite use of finite means," supporting a recursive, structured view (slide 32). Manning's own
reading, on the next slide and labelled as his own, is different: language "is no product
(*Ergon*), but an activity (*Energeia*)", and von Humboldt effectively distinguishes System 1
cognition — which he calls "acts of the spirit" — from System 2 "thinking", holding that System 2
thought requires "the fruitful extension of the mind through the symbols of language." His gloss:
we can obviously think without language, "but for the sort of more abstract, larger-scale thinking
that humans engage in … language gives a scaffolding inside the mind that makes that possible"
(≈41:59–43:34).

**Daniel Dennett's four grades** of intelligence, from *From Bacteria to Bach and Back*, ordered by
sample efficiency and competence (slide 34, ≈44:20–47:32):

| Grade | Competence | Example |
| --- | --- | --- |
| **Darwinian** | Fixed; improves only through genetic selection | bacteria, viruses |
| **Skinnerian** | Learns to respond to reinforcement | a lizard, a dog |
| **Popperian** | Learns models of the environment, so can plan — model-based RL | chimpanzees, and crows |
| **Gregorian** | Builds *thinking tools* | humans only |

Mathematics is a thinking tool, and so, Dennett suggests, is democracy, "but out of the space of
thinking tools, human language is the preeminent thinking tool that we have" — which makes humans
the only biological Gregorian intelligence.

## See also

- [Lecture 18](18-nlp-linguistics-philosophy.md) — the source lecture.
- [Formal semantics](formal-semantics.md) — what symbolic AI's theory of meaning actually looked
  like in practice.
- [Compositionality](compositionality.md) — the symbolic property neural models are measured
  against.
- [Distributional semantics](distributional-semantics.md) — the cybernetics-side theory of word
  meaning.
- [Tree recursive neural networks](tree-recursive-neural-networks.md) — an explicit attempt to put
  symbolic structure inside a neural network.
