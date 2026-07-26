---
title: Lecture 4 — Dependency Parsing
lecture: 4
video: https://www.youtube.com/watch?v=KVKvde-_MYc
source: YouTube auto-captions, copy-edited for readability
verbatim: false
original: original/04-dependency-parsing.md
slides: ../slides/04-dependency-parsing.md
---

# Lecture 4 — Dependency Parsing — transcript

Timestamps mark the start of each paragraph and can be cited directly ("Manning works
the *I ate fish* example around 55:49"). All 102 paragraph timestamps from the source
captions are preserved.

**This is an edited transcript.** The auto-generated captions had no punctuation and
destroyed the single most important word in the lecture: *parsing* arrived as "paing",
"passing" and "paring", and *parser* as "paa", "paard", "PA", "pases" and "parza".
*Pāṇini* came through as "panani", "panan", "pines" and "Pan's"; *Nivre* as "Nea" and
"Neo"; *Tesnière* as "tener"; *Parsey McParseFace* as "pzy mpas face"; *KaiC, SasA,
KaiA, KaiB* as "kisy", "sass a", "kai a" and "kai B"; *valency* as "veency"; *Kleene
star* as "clean star"; *Reed–Kellogg* as "read Kellogg"; and *ate* as "eight" throughout
the worked example. They have been copy-edited into sentences: punctuation added, false
starts and filler removed, mis-heard terms and names restored from context and checked
against the slides. **No content was added, removed, or reordered.** Student questions,
which the captions run together with Manning's speech, are marked in italics. The
verbatim captions are kept at
[`original/04-dependency-parsing.md`](original/04-dependency-parsing.md) for reference —
use this file unless you specifically need the raw output.

**Where the source is still unreliable**, the text carries an inline `[Ed:` note rather
than a silent guess. There are three: the name of the PhD student at 1:07:32, which the
captions drop entirely and the slides supply; the two counts of parses at 21:43, which
do not match the Catalan formula on the slide; and a magazine name at 1:14:29.

---

**[0:05]** Okay, hi everyone. So for today we're going to, I guess, do a 180 from where we
were on Tuesday. And so today I'm going to talk about syntactic structure, linguistic
structure of human language sentences, dependency parsing — and how, well, dependency, and
then how you go about building dependency parses. So we're solidly in the linguistic zone
today. How many people in the audience have done a linguistics class? Yay, okay, there's
some people have done linguistics classes, okay, great. And for the rest of you, well, this

**[0:51]** is your chance to see a little bit of human language structure, and if you like it
you can enroll in a linguistics class later on. So — oops — so Assignment 2 we handed out
on Tuesday. So in the second half of Assignment 2, your job to do is to build a neural
dependency parser using PyTorch. As we'll sort of come to later on, really the bit that you
have to build is just the machine learning bit of making decisions, and really we give you
most of the rest of the neural dependency parser. But this is also then a chance to remind
you that Assignment 2, in that second half, uses PyTorch, one of the leading deep

**[1:37]** learning frameworks. So if you're not familiar with that, it'd be a really good
idea to also go along to the Friday PyTorch tutorial — though we have tried to make
Assignment 2 so it's a fairly good place for learning PyTorch as you go along. We'll say
more soon about final projects, but you're certainly already encouraged to come and meet
with TAs or me about final projects, and we're putting up information about the TAs so you
can know more about them on the office hours page. Okay, so let's get straight into it and
start looking at linguistic structure. So in thinking about linguistic structure of human
languages

**[2:24]** there are two primary ways that people have thought about it. So one way is using
the idea that linguists normally call *phrase structure*, which is then represented in terms
of what computer scientists normally know as *context-free grammars*. So I'm going to spend
a couple of minutes going over that view of things, but actually it's not the main one that
I'm going to talk about in this class. I'm going to spend most of this class talking about
an alternative way of thinking about things called *dependency grammars*. There are actually
some correspondences you can make between the two ways of thinking about things, but I'm not
going to go into those here today. So for the constituency

**[3:11]** grammar or phrase structure version of things, the way that you go about thinking
about the structure of human languages is: well, there are words. Languages have lots of
words, hundreds of thousands of words. But it seems like a lot of the words — nearly all the
words, in fact — fall into a few basic classes that represent their nature and how they
behave in sentences. So for words like the examples here, we have nouns. So *cat* is a noun,
*door* is a noun, but something like *linguistics* is also a noun. So we have nouns. And then
we have other kinds of words, so something like *cuddly* is an adjective, a word that

**[3:58]** can modify nouns. And then we have *the*, for *the cuddly cat*. *The* is sort of a
slightly more complex one as to how to name. Normally in modern linguistics we refer to words
like that as a *determiner*. You might also see the name *article*. And sometimes, when people
try to shoehorn human language into eight part-of-speech categories, they say it's an
adjective — that doesn't really behave like regular adjectives. And then we have words like
*by* or *through* or *on* and *to* and ones like that, and so they're then prepositions. So we
have these classes, with lots of words fitting into each class, and so they're referred to
conventionally as

**[4:44]** *parts of speech*. But then once we've got words, we start putting them into bigger
units. So *the cuddly cat* is some kind of unit, and so it seems like this is an explication
of a noun, *cat*, and so this gets referred to as a *noun phrase*. And then *by the door* —
well, this is a phrase, but actually it has inside it *the door*, and that's a noun phrase.
But this bigger unit here of *by the door* is then a *prepositional phrase*. And we can
continue to build bigger units. So inside this we have this phrase that we've already looked
at, with the noun phrase and a

**[5:30]** prepositional phrase. But then we can have another noun phrase, *the cuddly cat*,
and we can put them together and build a bigger noun phrase, *the cuddly cat by the door*.
And so to represent this you can start to write a phrase structure grammar, or a context-free
grammar, that represents what are the possibilities for building up sentences. Here in
English — those similar kinds of phrase structure grammars can be written for other
languages. So this is sort of starting to give you possible structures for a noun phrase. So
you can have a noun phrase just goes to a determiner followed by a noun. But then, as well as
*the cat* and *a dog*, you can have *the large*

**[6:19]** *cat*, so you might say that okay, rather than that, I might want to have as a
better rule that a noun phrase goes to a determiner, an optional adjective, and then a noun.
If you think about it, you can sort of have multiple adjectives, so you can have *the large
green cat* or something like that. So you can really get multiple adjectives that are heaped
up, and that sort of star, the Kleene star, says you can have lots of them — *the large
cuddly green cat*. But then you can stick things after the noun phrase, so you can put these
prepositional phrases like *in a crate*. So

**[7:04]** we might also want to say that a noun phrase can be rewritten as a noun phrase
followed by a prepositional phrase, where a prepositional phrase can be represented by a
preposition followed by a noun phrase. And somewhere we're also going to want to represent
our parts of speech membership. So a determiner can go to words like *a*, and an adjective
can go to words like *large*, *cuddly*, or many other words that I'm not going to write
down, and a preposition can go to words like *in*, *on*, *under*, etc. After that, okay, so
now I've

**[7:52]** got a little grammar here, and this little grammar here could sort of make
everything I've got in these sentences. Well, actually, it can do this one too, it can do
*the large barking* one, where there are multiple ones. But then if I start going beyond
these noun phrases and say, think of sentences like *talk to the cat* or *talk to the large
cuddly dog by the door* — well, now I've got here a verb, *talk*, and I've again got a
preposition. So I might then have more rules that say I can have a verb phrase, and the verb
phrase can go to a verb and then a prepositional phrase, and then

**[8:40]** that could explain these two sentences as well. And in this kind of way I can
start to build up a grammar of the structure of English sentences as a context-free grammar.
Make sense? Yeah, okay. And so that's what is being quite commonly done in linguistics and
elsewhere. Okay, yeah, so let me just do that once more, but behind, in this one. So one
thing I can do here is say, oh, I'm going to look at this with its phrase

**[9:26]** structure. And if I write it upside down, to give myself some space for later, I
could start making a phrase structure that is of this sentence. I'll start to run out of
space, but I can sort of start to make this phrase structure of the sentence. So that's
phrase structure. But there's another form of representation that has been fairly widely
used in linguistics, and has been commonly used in NLP — and we're going to use it for the
parsers we build — and that's *dependency structure*. So dependency structure represents
things in a slightly different way. It thinks

**[10:11]** about words that are the main word, or head, of something, and then which words
they take as modifiers or arguments. So for *look in the large crate in the kitchen by the
door* — well, this is describing a looking command, so the head of the whole thing is
*look*. And then *look* is taking one or more arguments or modifiers, and what the looking
is saying here is, well, what you want to do is look in the large crate. So we are looking
in something, and then what we're looking in is a crate, and then the crate has some
modifiers:

**[10:57]** it's a *large* crate, it's *the* large crate. And then the crate is also placed
somewhere, it's placed *in the kitchen*, so that *in the kitchen* is also modifying *crate*.
And then we've got over here the *by the door* — well, the *by the door* is also modifying
*crate*, so we've also got a link down over to here. And that gives us our piece of structure
here. Which, having filled that in, makes me realize I actually got it wrong when I was doing
the constituency representation. Whoopsie. So I should get my — in the

**[11:43]** constituency representation I made *the kitchen by the door* into a phrase. That
was actually wrong, whoops, bad me. So what I should have actually had was, we had another
prepositional phrase that went to a noun phrase of *in the kitchen*, and then both of those
were coming off a bigger noun phrase, like that. Whoopsie. Okay, I get it right most of the
time. Okay. But so this idea of dependency structure is that we're sort of finding what is
the head word, and then we're saying which things modify the head word. And either of these
representations we can be using

**[12:31]** to sort of work out what the structure of sentences is, in terms of which words go
together and which words modify other words. And so the basic idea is: when humans
communicate, we communicate in a linear stream. So that if it's conventional writing systems,
it's a linear stream of words that you're reading. If it's spoken language, like you're
understanding me speaking right now, it's not a linear stream of words, it's a linear sound
stream. And when people speak there aren't any — there isn't white space between words when
people speak. Occasionally people pause at the end of a clause or

**[13:17]** sentence or something, but in general I'm just sort of speaking continuous words
that run one into each other, so that there's a linear sequence of sounds coming out of my
mouth, and you have to do all a bit like that. But if you're then thinking, oh gee, I can
actually understand Chris talking, then somehow you're taking that linear stream and you're
turning it into a meaning, where certain words are modifying other words, and you have these
bigger units like constituents that are understanding the meaning of the sentence. And so
human listeners need to work out what modifies what to be able to understand sentences

**[14:04]** correctly. And so similarly our models need to be able to understand sentence
structure in order to be able to interpret language correctly. And so what we're going to be
doing for building dependency parsers is, we're going to be explicitly building a neural
network model that says, let's find the structure of these sentences. In a way we actually
move away from that later on, because when we move into Transformer language models, they
just take in the sequence of words, but actually inside the parameters of the neural network
they're recognizing and building the same kind of structural units, and we'll talk about that
later in the class. To give you

**[14:50]** more of a sense of how understanding what modifies what is important for
interpretation, here are a few funny examples from newspaper headlines. And they're funny
examples because you get — their sentences don't just have one way of interpreting them. When
you have a sequence of words, commonly in human languages sequences of words are ambiguous,
and it's relying on human interpretation of what makes sense and what goes together to work
out how to read them. So here's a first example: *Scientists count whales from space*. Now
that's ambiguous, and you can give

**[15:36]** this two possible readings. So how can you give this headline two possible
readings? *[Student: One is that they're scientists in space counting whales, and the other
one is that they're whales from in space.]* Yeah. So one possibility is — so we've got this
prepositional phrase, this is a prepositional phrase here — one possibility is that this
prepositional phrase is modifying, or is the object — yeah, it's modifying *whales*, so
they're whales from space. And the other possibility is that it's the counting that's
happening from space, so the scientists are counting it from space. Okay, so that corresponds
to my two

**[16:23]** pictures here. So in one picture it's the counting that is happening from space,
which is actually the right interpretation of what the article is about. But in the other
interpretation we have space whales, and the scientists are counting the space whales that
are arriving or something like that, and so then we have the *from space* that are modifying
the whales. Okay, so what we have here is a prepositional phrase which comes after a noun
phrase — it's just a one-word noun phrase here, *whales*, that's fine — and then before that
is a verb. And so one place in English where

**[17:09]** you get a lot of ambiguities is from these prepositional phrases. Because whenever
you get prepositional phrases — and prepositional phrases are really common in English, if
you think about it — whenever you get them like this, it's always ambiguous as to — oops —
it's always ambiguous as to what earlier thing in the sentence they're a dependent of. And so
you can sort of put in another prepositional phrase, *in the morning* or something like that,
and so then the ambiguities just multiply. And so the important thing to notice here about
human language is:

**[17:56]** human language is, in syntactic terms, *globally* ambiguous. Right, so in
programming languages you have local ambiguities of interpretation. How many people have done
a compilers class? I think very few these days. Anyone done a compilers class? Okay, it looks
like fewer people have done a compilers class than a linguistics class, that's interesting.
Okay, well, I won't make too many analogies to compilers classes then. You know, when I was
young, that was still the old-days kind of CS curriculum where writing interpreters and
compilers were seen as the mainstay of computer science education, but no more, I guess.
Yeah, so in programming languages you can have a

**[18:42]** local ambiguity, but ambiguities are always resolved. So we have simple rules in
programming languages that *else* is construed with the nearest *if*. It's a bit different in
Python because it's indentation, but there are rules, so there's never global ambiguity in a
programming language. But human languages just aren't like that — there's nothing that
resolves which of these two readings is correct. If I made it a bigger sentence, that'd still
be ambiguous. You're just sort of meant to read it and use context and your intelligence to
decide what's going on. And so to take a bigger but real

**[19:27]** example, this is the kind of boring sentence that you can read in the *Wall Street
Journal* most mornings: *The board approved its acquisition by Royal Trustco Ltd. of Toronto
for \$27 a share at its monthly meeting.* So what you can see in this sentence is, we've got a
verb, and then we've got a noun phrase, and then after that we have four prepositional phrases
in a row. Okay, so what do these prepositional phrases modify? So what does *by Royal Trustco
Ltd.* modify? The acquisition, right, so it's the acquisition by Royal Trustco. Then *of*

**[20:13]** *Toronto* modifies — so it's Royal Trustco Ltd. of Toronto. So yeah, later on
prepositional phrases can also modify earlier prepositional phrases, or at least the noun
phrase inside them, Royal Trustco Ltd. Okay, *for \$27 a share* is back to modifying the
acquisition. Okay, *at its monthly meeting* is — yeah, it's the approval, so it's gone way
back up to there. So if you start having sentences with a whole bunch of prepositional phrases

**[20:58]** like this, you can start getting more and more ambiguities of attachment. I mean,
you don't get the sort of full free choice, factorial number of attachment points, because
there is a restriction that these dependencies don't cross. So once you've gone back further,
you have to stay equally far back or go even further back again. But nevertheless, the number
of readings you get is a Catalan series, which is a series you see in a whole bunch of other
places. If you've done any graph theory or anything like that — if you're doing
triangulations, you get Catalan numbers, because you get the same property that things don't
cross. So it's an

**[21:43]** exponentially growing sequence of possible readings, and so it quickly gets very
big. So I think when you've got four prepositional phrases you get 13 readings, and if you
have five you [get] 27, and it grows up from there. [Ed: the Catalan formula on slide 12,
C_n = (2n)!/[(n+1)!n!], gives C₄ = 14 and C₅ = 42; the captions read "13" and "27" and the
deck prints no table of values, so what Manning said aloud is not recoverable here.] So you
get a lot of ambiguities. But the crucial thing to notice is, human beings read sentences like
this every morning — or at least people who work in banking do — while having their corn
flakes, and their brain doesn't explode trying to think about the 13 different readings and
which one is correct. We just sort of do this as we go along, and it's sort of obvious. Okay,
let's just do a couple more

**[22:29]** examples of where we get ambiguities in human language. So a different one you get
is *coordination scope ambiguity*. So: *Shuttle veteran and longtime NASA executive Fred
Gregory appointed to board.* How is this sentence ambiguous? *[Student: Does it mean two
people or one person?]* Yeah. So there can either be one person, Fred Gregory, and they're
both a shuttle veteran and a NASA executive; or it can be that there are two people, there's a
shuttle veteran, and there's a longtime NASA executive, Fred Gregory. Okay, yeah. So we'd be
kind of

**[23:14]** capturing those by having extra grammar rules where a noun phrase can go to a noun
phrase, a conjunction and a noun phrase. But then another thing that you get in English is
*apposition*, so you can have a noun phrase that's a descriptive noun phrase of another noun
phrase, like a name — you know, *the author Fred Gregory* or something like that. Saying the
word "English" again — I meant to comment: I'm only going to give English examples here. In
different languages you don't get all the same ambiguities. So if you're familiar with, say,

**[24:00]** Chinese, you might have thought about the prepositional phrase example of, wait a
minute, we don't have that one, because the prepositional phrase modifying the verb would
appear before the verb and the object noun would be afterward, so it would be completely
unambiguous. And that's true. But that doesn't mean that Chinese is unambiguous — Chinese has
lots of very bad ambiguities. And yeah, it's just that different languages have different
syntactic structures. Okay, here's — so sometimes in English, especially when you're sort of in
a more written form, rather than having an explicit coordination word you can just sort of use
juxtaposition with a

**[24:47]** comma to have the idea of coordination. So here's a fun example from the first Trump
administration of how we can have a coordination scope ambiguity: *Doctor: No heart, cognitive
issues.* Right, so again this is the same kind of coordination scope ambiguity, that it can
either be kind of *no heart* and *cognitive issues* being conjoined together like that, or else
it could be that it's *no heart or cognitive issues* being conjoined together like that. You
make the choice. Okay, let's

**[25:33]** see — oh, this is my risqué one, for a different kind of ambiguity. Trigger warning.
*Students get first hand job experience.* So this one is also an ambiguity as to whether you're
having the *first hand*, and then both the *job* and the *first hand* are modifying
*experience*, or there's this other reading, if you have a smutty mind, that might come to you.
Okay, one more fun one. Okay: *Mutilated body washes up on Rio beach to be used for Olympics
beach volleyball.* Okay, so what are the

**[26:20]** two possible readings of this sentence? These are real examples from quality
newspapers. Okay, what are the two readings of this sentence? Yeah. So we've got — so here we
have one of these infinitival — so, infinitival verb phrase, *to be used for Olympics beach
volleyball* — and for these as well, they kind of have the same effect as prepositional
phrases, that they can modify different things. So it can either be the Rio beach that's going
to be used for the

**[27:06]** Olympics beach volleyball, or it's going to be the mutilated body that gets used for
the beach volleyball. Okay. So these are the kind of ways in which we sort of want to use the
structure of the sentence to understand what they're meaning. We also use it in lots of sort of
just practical ways when we're building various kinds of natural language processing systems.
So a kind of thing that people often in practical systems do is that they want to get out facts
of various kinds. So for people who do stuff with bioinformatics, they commonly want to get out
things like protein-protein interaction facts.

**[27:51]** And so commonly you can get those kinds of facts out by looking for patterns. So you
have a verb *interacts*, that's going to be indicating an interaction pattern, and it's going to
be taking arguments, so it's going to be taking a subject, and *interacts with* the
prepositional argument. And so that will be an interaction: that KaiC, whatever that is,
interacts with SasA. But in this case the SasA is coordinated with the KaiA and the KaiB, so
it's also going to end up interacting with those two other things as well. And so you can use
the sort of sentence structure patterns of a dependency parse to be getting out the kind of
facts and events that you're interested in for

**[28:38]** something like an event understanding system. And people do these kinds of analyses
over biomedical texts to build up the kind of structured databases of known protein-protein
interactions and things of that sort. Okay, so linguistic structure is useful, and it's
syntactically very ambiguous. And so you should think of humans as active interpreters that are
using their contextual knowledge — both of earlier stuff in the text, knowledge of the world
around them, how the world works — to work out the right structure. Yeah, so now I want to go on
and show you a bit more about dependency

**[29:25]** grammars, which is what we're going to be using. So dependency syntax postulates that
you can capture the structure of a sentence by having these sort of asymmetric dependent
relations, which we might just call arrows, which are going from heads to dependents. So here
the sentence is *Bills on ports and immigration were submitted by Senator Brownback, Republican
of Kansas*, and we're sort of picking out heads, and then we've got things that depend on them,
that modify them. Yeah, so if you're in the video audience and you are educated in

**[30:11]** the United States and you're over the age of 50 — or if you happen to go to one of
those kinds of private schools where they also teach Latin — you might have seen sentence
diagramming. So Reed–Kellogg sentence diagramming was something that was actually very
widespread in American education. Which really was a — it was really dependency grammar, was a
sort of a somewhat quirky form of dependency grammar, where you had to write lines at different
angles and stuff like that. But basically you're writing sort of heads and their dependents
underneath them, with different funny-shaped lines. It also was dependency grammar. Okay, so
this is the start of

**[30:59]** a dependency grammar. But just like the funny angled lines of sentence diagramming,
normally people want to add some more information than that. And so most commonly the arrows are
then *typed*, by giving the name of some grammatical relation. So something can be the noun
subject, or an oblique, or an appositional modifier, or a case mark, or things like that. And
I'm just trying to give you the idea of dependency grammars — I'm not expecting you to master
all of these names and ways of doing things. And

**[31:44]** there are different systems of deciding what's heads and dependents, and not all the
details are important. What you should get into your head is just sort of the basic idea of what
one of these does, and some sense of, oh, it should be at the phrase level, it should be
representing what's modifying what. So we do actually ask some questions on the assignment, and
so for the cases like the prepositional phrase — what is it modifying — you should be able to
give the right answer to that. Okay, yeah, okay. So this is just a little bit more vocabulary.
So yeah, we have these dependencies, and so I'm

**[32:30]** going to say that they connect between a *head* and a *dependent*, but sometimes
people use other words like *governor* and *modifier* and things like that. And so dependencies
are generally taken — and we'll be taking them — as forming a tree. So you've got something
that's connected, acyclic, and has a single root to it. So our single root is the top of the
sentence here. So although what you see most often these days, either in a linguistics class or
when you get taught CS103 at Stanford, or computer science — what you see there is normally
context-free grammars or phrase structure grammars — I

**[33:17]** mean, really, it is dependency grammars that have the really long history. So really
the predominant way of representing the structure of human languages throughout human history is
dependency grammar. So who is heralded as the first dependency grammarian, or really the first
person who tried to write the grammar of a human language, period, was Pāṇini. So Pāṇini was
working with Sanskrit. Pāṇini lived so long ago that actually people don't really know when he
lived. I mean, he lived somewhere between about the 4th and 8th century before the Common Era,
but really no one knows when. But he lived sort of up in part of

**[34:04]** actually what's now Afghanistan. And for largely religious reasons he set about
developing a grammar of Sanskrit, and the way he represented the syntax of Sanskrit was using a
dependency grammar. So there was a lot of work on grammar in Arabic in the first millennium —
they used dependency grammars. In contrast, the idea of sort of having context-free grammars,
that's really, really recent. So the first work on phrase structure grammars dates to the '40s,
and then was sort of canonicalized by the work of Chomsky in the 1950s. Yeah, so a fact for the
computer

**[34:51]** science part of the people in the audience. So, dear computer scientists: if you know
about Chomsky, computer scientists normally know two things about Chomsky. One is they hate on
the Chomsky hierarchy that they were forced to learn in CS103 or equivalent classes, and the
second one is he's a very left politician. But if I only deal with the first one of the two now:
the Chomsky hierarchy was not invented either to torture elementary computer scientists, or to
explain fundamental facts about formal language theory. The Chomsky hierarchy was actually
invented in thinking about human languages, because at that time, and

**[35:37]** in stuff that's come since, it was commonly the case that people were modeling human
languages with regular, finite — so finite-state grammar equivalent — mechanisms, and Chomsky
wanted to argue that that was a completely inadequate formalism to represent the complexity of
human language. And so it was in the context of arguments about human language that he developed
the Chomsky hierarchy. Okay, yeah, so anyway, that's enough of the history of that. Here's my
picture of part of Pāṇini's grammar — but actually, or a version of it. Actually this is really,
really

**[36:24]** misleading, because one of the astounding facts about Pāṇini's grammar, and part of
why no one knows what century he lived in, was Pāṇini's grammar was composed *orally*. So this
sort of kind of blows my mind. It seems — some of the famous things in the West, like Homer's
works, the *Odyssey* and the *Iliad*, they were originally oral works that were passed down in
oral form. That seems hard to do, but you can kind of believe, if you did plays in high school
or something, that someone could memorize the *Odyssey*, perhaps. But the idea that people could
memorize a

**[37:10]** grammar of a language, passing it down for hundreds of years, kind of blows my mind.
But that's exactly what happened with Pāṇini's grammar. So really, although this is sort of an
old birch-bark manuscript, really it probably dates from about a millennium after Pāṇini
composed his grammar. Okay, getting back to the modern days. So for things to know — yeah, so I
mean, we don't want you to fixate on the details of dependency grammar structure, providing you
have the rough idea. But just one thing that you can possibly be confused about is, there,
people

**[37:57]** do things in different ways. One way in which they don't agree is even which way to
draw the arrows. So some people draw arrows from the head pointing at the dependents, and there
are other people who draw the arrows starting at the dependent and pointing back at the heads.
So modern dependency grammar largely follows the work of Lucien Tesnière, a French linguist. He
did the arrows pointing from the head to the dependent, and so that's what I'm doing today, but
you'll see both. We sort of said that normally you assume that you have a tree with a single
root. It's kind of common, and it

**[38:43]** works out more easily for the parsing, if you sort of add to a sentence a sort of a
fake root node. So that's going to be the starting point, and it's going to take one dependent,
which is the word that's the head of the sentence, and then you're going to work down from
there. Okay, so before getting more into doing dependency parsing, I just wanted to take a
little detour to tell you about the importance that happened with the rise of annotated data in
natural language processing. And this is sort of an interesting flip-flop that's occurred, but
we're going to sort

**[39:29]** of today go in one direction, and in a later class we'll go in the other direction. So
in early natural language processing, people started to see, oh, human languages have structure,
so what we should do is start writing rules for the structure of human languages. And I started
writing a few context-free grammar rules for the structure of English on that early slide, and
you could also write dependency grammar structure rules. So people tried to do natural language
processing by having rules, grammar rules, dictionaries of parts of speech and things like that,
and that gave you parsers that, in retrospect,

**[40:15]** worked out pretty badly. And it worked out pretty badly for a number of reasons. One
reason is that although there are these sort of very canonical, clear structures in human
languages, there's a very long tail of messy stuff where all kinds of weird usages start to
emerge in human languages, which sort of means it's just really hard to get coverage for
handwritten language. And that's because humans use language creatively. So you can start
thinking of some of the things that you've probably come — I'm probably not very good at young
persons' slang usages of grammar these

**[41:02]** days, but the kind of ones that you might be still familiar with: *Star Wars*, you
have Yoda talk, where you rearrange the sentences but people still understand them. So that's
changing the word order. And earlier on than that there was sort of a bit of a fad for putting
*not* at the end of the sentences — "that's a really great idea — not." And people learn to
understand that, but it's different to regular grammar. So it's really hard to write a full
grammar. But the bigger reason actually is the problem of ambiguity I talked about. That if you
just write a grammar — well, my sentence with the prepositional phrases had 13

**[41:49]** different parses, and you didn't have much reason to choose between them. But if you
had information about how often words modify other words, then you could get some statistics and
start to predict in which order, which things modify other things. And so people wanted to start
to be able to do that prediction that underlies probabilistic or machine learning models. And so
to be able to do that led — you know, sort of earliest antecedents in the '60s, but really
starting in the late '80s and into the '90s — that people decided the way to make progress in
natural language processing, natural language understanding, is to build annotated data

**[42:35]** resources. And so all through the '90s and the 2000s decades, the name of the game for
a lot of natural language processing was people building annotated data resources and then
building machine learning systems on top using those resources. Now, that's kind of gone into
reverse and gone away again with large language models, which we'll get to in another week or so.
But here's an example. So this is the Universal Dependencies treebanks, which I've actually been
heavily involved with, and it's a cool resource for all kinds of purposes, because it's actually
a wide crosslinguistic database where there's over a 100 different languages with sentences
parsed with a uniform dependency formalism. So it's actually

**[43:23]** really good for things like crosslinguistic work and psycholinguistic work. But what
these are is taking sentences — *I think Miramar was a famous goat trainer or something* — and
putting a dependency structure on it. It's sort of all written there, sort of very squished down.
And human beings are producing these dependency structures, and then this is giving us data that
we can learn things from — dependency parsers from. And indeed, for what you do on homework 2,
this is precisely what you'll be using, is data of this sort to build a dependency parser. And
it's going to learn that you have goat trainers and you have famous trainers,

**[44:09]** and so it'll build up sort of statistics and information to predict what kinds of
things are likely. Yeah, so starting off, building a treebank like that feels kind of like, oh,
this is going to be slow, hard work — and it is actually slow, hard work. But it proved to be a
very effective strategy, because it gave wonderful reusable resources, that once people had done
it once, all sorts of people could use it to build parsers, part-of-speech taggers, to do
psycholinguistic models and all kinds of things. You'd get the sort of distributional frequency
information that's good for machine learning. It also

**[44:55]** provided one other thing that's crucial: it gave a method to evaluate systems, to say
how good they are at producing parses. I mean, this may seem kind of comical to you in the modern
era of machine learning, but the fact of the matter is, when people did natural language
processing in the '50s, '60s, '70s, nobody had evaluation methods. The way you showed people you
had a good parser is you ran the program, you said type in a sentence, look at what it — look,
it's worked, it's a really good parser. There was no systematic evaluation of NLP systems
whatsoever. So actually

**[45:43]** saying, look, here's a thousand hand-parsed sentences, let's evaluate how well your
parser does on those — that was actually a revolutionary new development that happened in the,
well, end of the '80s, but especially in the '90s. Okay, so now that we have all of that
knowledge, we're going to want to start building dependency parsers. And so I'm going to show a
particular way of dependency parsing, which is the one you're going to use in the assignment. But
just first off, it's sort of worth just thinking for a moment: what kind of information should a
dependency parser

**[46:30]** have to make decisions? So these are kind of the four factors, the sort of obvious
things that are useful for dependency parsing. I mean, the first one is sort of thinking of the
two words at the ends of the arrow as to whether they are plausible. So that for *discussion of
the outstanding issues was completed* — so to have *discussion* of *issues*, that's a plausible
dependency. To have — what's the silly one — to have something like *the* being a dependent of
*completed*, that makes no sense at all. So, what words there are involved. The second

**[47:18]** one is *dependency distance*. So you can have long-distance dependencies that go a
long way, but most dependencies are short distance. A lot of words are depending on their
neighboring words at a very short distance, so that's a good preference to have. As well as just
the distance, it's somewhat informative knowing what's in between, so it's rare for dependencies
to span verbs or punctuation. And then there's a final one, which is to think of the *valency* of
heads, and that's how many arguments they take. So that if you have sort of something like a
verb *broke*,

**[48:04]** well, it probably has something to the left, because it probably has who did the
breaking, and it probably has something to the right, because there might be the cup or something
like that. But it doesn't have to be that, because it could be *the cup broke*. So you can have
something to the left but nothing to the right, but you sort of have to have something to the
left. And conversely, you can't have any number of things — you can't sort of just say *he broke
the cup, the saucer, the dish*. So it doesn't take just lots of arguments to the left. So we've
got a notion of valency like that. Yeah, there's one other tricky

**[48:53]** little notion in dependency parsing, which is that normally dependencies kind of nest
like this, and nesting dependencies corresponds to a tree structure as you'd have in a
context-free grammar. *[Student: Yeah, because in a sense when I read the sentence, which — I
thought that the most important [word] was "discussion"?]* So — fair enough — I will assert that
this

**[49:38]** is a sentence, and *discussion* is the subject of the verb *completed*. And normally
for a sentence we say the main thing in the sentence is its verb, and so that's why the root is
heading to *completed*. And the subject of the verb is also an important thing, but the arguments
of the verb — like the subject of the verb, the object of the verb if there is one, prepositional
phrase modifiers — they're all taken as dependents of the verb. *[Student: Following up on that,
is it not the [subject] that you start with?]*

**[50:24]** So if you have a sentence with a verb like this, it's always that — that is always the
answer. I mean, some of the details here depend on languages, but there are languages in which
you don't have to have a verb in a sentence. And you can get things like — I mean, you can do it
in sort of very restricted ways in English, right, so if you just sort of say *easy as pie*,
there's no verb, and so then you're saying *easy*, the adjective, which is sort of the predicate
adjective, is then the head of the sentence.

**[51:10]** *[Student: Sorry — like a question like "What is the story?", is the "is" — like, would
we still look at that as the [head]?]* That is complicated. Some people would say it is, and some
people would say it isn't. And in particular, in Universal Dependencies we don't actually say that
*is* is the head of the sentence. But I don't want to get too far into this. If you want, you
could sort of look more at how things are done. But I want to fully admit that dependency grammar
isn't sort of one uniquely defined theory. People have had different ideas of which things to take
as the head in various circumstances, and they argue about it — linguists argue about what the
right structure is to put over all sorts of

**[51:55]** sentences. But the fact that people do things different ways doesn't mean that everybody
doesn't agree that there are units, there are phrases and modifiers and ambiguities and so on
between them. Okay. Yeah, so normally we get this sort of nesting that corresponds to what you can
build with context-free grammar structure. But sometimes in human languages you get dependencies
that don't nest. So you get sentences like *I'll give a talk tomorrow on neural networks*, where
actually the *on neural networks* is modifying the *talk*, where the *yesterday* — um, sorry, the
*tomorrow* — is an argument of *give*. And so

**[52:42]** you get these crossing dependencies, which are referred to as *non-projective*
dependencies. You also get them when you form questions, so *Who did Bill buy the coffee from
yesterday?* — that the *who* is the object of the preposition *from*, but it's been moved out the
front, and so that again gives us non-projective. If you think about it, you can still say that you
have a dependency tree, but it's got the words in different orders. And so one of the things that
you have to cope with for full dependency parsing is dealing with this non-projectivity.

**[53:29]** But actually we're not going to deal with it in our parsers, we're only going to do
projective dependency parsing. Okay, so there are various ways that people do dependency parsing.
People have done it by dynamic programming, people have done it using graph algorithms — if I have
enough time at the end I might mention that again — people have done it with constraint
satisfaction methods, if you saw those in CS221. But the most common way in practice that's emerged
has been this *transition-based parsing*, which is kind of interesting as well, and gives a very
simple machine learning mechanism. So it makes it

**[54:16]** good for Assignment 2, and so that's what we're going to explore here. Okay, so what we
do in greedy decision-based parsing, in transition-based parsing, is — this is where it's
unfortunate that only two people in the class have done a compilers class. So a simple form of
parsing that's also used in compilers classes is something called *shift-reduce parsing*, where you
start sort of bottom-up and you start putting little units together and build bigger constituents.
But if most people haven't seen it, that's not going to be very much help. So I'm going to give

**[55:01]** you a concrete example. So the things to know is, we have two data structures — well,
we have more than two, I guess — for dealing with the sentence we have two data structures. We have
a *buffer*, which has the words of our input sentence, and then we start building pieces of
sentence structure which we put on a *stack*. And a little trick to know is, for the buffer the top
is written to the left, and for the stack the top is written to the right. And so we take actions
which are like shift and reduce actions, and when we take arc-building actions we build up a set of
dependency arcs which are going to be the dependency structure of our sentence. And that's all
incredibly abstract, and so

**[55:49]** I'm going to show an example which hopefully will a bit give the idea. So here's an
example. So I want to do this very simple example of parsing up the sentence *I ate fish*. So the
way I do this is, I have my stack, and so I start by putting the root symbol on my stack, and then
I have in my buffer all the words of the sentence. And so that's the sort of start condition I've
written in very small print there. Then for each step of processing I have a choice of three
operations. I can

**[56:34]** either *shift*, which moves the top word on the buffer onto the stack; or I can do *Left
Arc* or *Right Arc*, and these are my two reduce operations, that build a little bit of syntactic
structure by saying that one word is a dependent of another word in either a left or a right
direction. So here's a sequence of operations I can take. And so starting off, the first thing I
can do is shift, so then I've moved *I* onto the stack. I can decide that I want to shift again,
and so then I'd take *ate* and also move it onto the stack. And so I've now got three things on my

**[57:21]** stack. So at this point, I can do other things. I mean, in particular, a Left Arc is
going to say, well, I can take the top two things on the stack and make the thing on the top the
head, and the thing one down on the stack a dependent of it. So if I do a Left Arc operation, I'm
effectively saying that the *I* is a dependent of *ate*, and then I pop the dependent off the
stack, but I add on that I've built a dependency that made *I* a dependent of *ate*. I could then
do another shift operation, so I shift *fish*

**[58:08]** from the buffer onto the stack. And then I can do a Right Arc, which says, okay, I'm
going to have *fish* as a dependent of *ate*. So then *fish* disappears from the stack, and I add
in this new dependency saying *fish* is a dependent of *ate*. I then do Right Arc again, which is
then saying that *ate* is a dependent of root. So I'm left with just root on my stack, and I've
built a new dependency saying *ate* is a dependent of root. And at this point I've got to the
finishing condition — my finishing condition is that my buffer is empty and my stack contains just
the word root. And so this gives me a

**[58:56]** little set of operations, referred to as the *transitions* of transition-based parsing.
And by making a sequence of these different transitions I can build sentence structure. And I've
got choices of when to shift and when to reduce, and whether to reduce left or reduce right — the
Left Arc, Right Arc. And so by making different ones of those choices I could make any structure
for the sentence that I wanted to. So if I somehow thought that this sentence should have a
different structure, and that *I* should be the head, and *ate* is a dependent of that, and *fish*
is a dependent of that — well, I could achieve this by making some

**[59:43]** different choices, as to I'd now be saying I was doing a Right Arc operation so that
*ate* would become a dependent of *I* rather than the other way around. So the choices of which
operations I take determine the syntactic structure, the set of dependencies that I have built,
which are my set of dependencies down here. Now, the set of dependencies I built were exactly the
right ones, because at each step I took the right operation. And so the essential idea of
transition-based parsing, and where it came to the fore, was — there was a particular guy who's,
I've

**[1:00:30]** got a photo of him somewhere in a bit, I thought. So Joakim Nivre is a Swedish NLP
person, and in the early 2000s he came up with the idea of, rather than doing the kind of dynamic
programming and chart parsing and things that people commonly used to do with parsers — these days
we have machine learning, so maybe we could build a fast, efficient parser, and the way we're going
to build it is with this making a sequence of transitions, and it'll be the job of the machine
learning to predict what is the right transition at each point in time. So if you do that, at each

**[1:01:19]** point you're dealing with one thing, and so the number of operations you're doing to
parse a sentence is linear. So this gives a linear time parsing algorithm. Whereas if you've seen
context-free grammars and stuff like that in CS103, and you want to do anything where you're fully
considering the parses and structures of context-free grammars, you've then got a cubic time
algorithm, which is much less pleasant to be dealing with. So for the simplest form of
transition-based parsing you do no search whatsoever. At each step you're just predicting the next
transition, and so you're doing this sort of sequence of

**[1:02:04]** transition predictions as machine learning operations, and that sequence gives you the
parse structure of the sentence. And the essential result that Nivre was able to show is that
machine learning is good enough that you can do this and get a very accurate parser, despite the
fact that no search whatsoever is [done] — it's just doing predictions in this way. Okay, so when he
did that in 2005, that was before neural networks came to the fore, and so the way he was doing it
was by using a sort of an older-style symbolic feature-based

**[1:02:50]** machine learning system. So he had a big classifier, which might have been a logistic
regression classifier or something else like a support vector machine. And so to power that he was
using *indicator features*. So the kind of features you'd use is that the word on the top of the
stack is the word *good* and its part of speech is an adjective; or the word on the top of the stack
is *good* but the word that's sort of second on the stack is the verb *has*. You'd get these sort of
combinations of matching functions, and they would be used as features in a machine learning system
to predict the parse. But the problem is that once you

**[1:03:37]** started building these features that were conjunctions of multiple terms, you ended up
with millions and millions of features. Because you're putting particular words in features, and
then you're combining choices of multiple words, so there are just millions and millions of
features. So you had to deal with millions and millions of features. And furthermore, individual
features were exceedingly sparse, that you barely ever saw them. You'd have a feature that only
turned up 10 times in a million sentences, because they are matching these very precise systems. So
on the one hand, by making these feature conjunctions, parsing got more accurate — and indeed
people produced pretty accurate parsers in those days — but

**[1:04:24]** they had these unappealing characteristics of this sort. Yeah, so before going on
further, I should just explain how we evaluate dependency parsers. So to evaluate dependency parsers
we're basically assessing, are you getting the dependency arcs, arrows, you're proposing right? So
here is someone's dependency parse: *She saw the video lecture*. Well, actually, sorry, that's the
gold parse, okay, that's a correct parse. Okay, *She saw the video lecture*, that's a correct parse.
So you can write out what are the

**[1:05:11]** different dependencies. So one's head is two, two's head is zero, word three's head is
five, word four's head is five, word five's head is two. So these pairs of numbers represent our
dependencies. Then if someone proposes a parse of the sentence, you can literally say, okay, which
of these did they get right? So they didn't get this one right, they got the rest of them right, so
their accuracy is 80%. And so sometimes people just assess the arcs unlabeled, and so that's
referred to as *unlabeled dependency accuracy*. But sometimes people also want to label them with
subject, determiner, object,

**[1:05:59]** etc., and say, also, are you getting the labels right? So in this case only two of the
five labels are right, so the labeled accuracy of the dependency parse is 40%. Okay, so that was
sort of what people did until the mid-2010s. And I sort of already started saying this — the
problems with indicator features were: they were *sparse*, you didn't see them often; they were
*incomplete*, because there's some words and combinations you've seen and some you just didn't see
in the training data, so you're missing features. But

**[1:06:45]** the final problem is actually just computing all those symbolic features was just
expensive. It turns out that if you did runtime analysis, most of the time in the parsing wasn't in
doing the machine learning decisions, it was just simply in computing the features that you put into
this dependency parser. So as neural nets started to show that they were successful for things, that
suggested that maybe you could build a better dependency parser by using a neural net
transition-based dependency parser, which would benefit from the kind of dense and compact feature
vector representations that we've already started to see. And so that's what

**[1:07:32]** started to be explored. And in particular, [Danqi Chen], who was then a PhD student of
mine and was head TA of 224N twice, actually, in the earlier days — so she built a neural
transition-based dependency parser and showed the success of this method. [Ed: the captions drop the
name entirely here; slides 40 and 45 credit the work to Chen and Manning (2014) and carry Danqi
Chen's photograph.] So this was Nivre's transition-based dependency parser. People had also explored
other methods of dependency parsing, so these were two graph-based dependency parsers. And
essentially, for the kind of symbolic feature machine learning methods, Nivre's parser was really

**[1:08:19]** fast, because it was using this linear transition-based parsing idea. The graph-based
dependency parsers were way, way slower — they're about, what, 50 times slower — but they were
slightly more accurate, you can see here that you're getting a bit better numbers. So essentially
what she was able to show was, you could build something that was basically as accurate as the best
known graph-based dependency parsers, but it was fast like other transition-based parsers. Indeed,
despite the fact that you might think that, oh, now I've got real numbers and matrices and stuff,
surely that should be slowing me down — the reality

**[1:09:05]** was that the symbolic models spent so much time in feature computation that actually you
could make it faster at the same time by using a neural network. Okay, so how did that work? Well, so
we've already seen word embeddings, so it's going to exploit word embeddings, so it can use word
representations. And that has the advantage that even if you haven't seen particular words in
particular configurations, you've seen similar words, and so it can exploit what's likely in terms of
word similarity. But it went a bit further than that, because why only have distributed
representations of words? We also have parts of speech. And although I sort of said just noun, verb,
adjective,

**[1:09:53]** most actual systems in NLP — of parts of speech — are much more fine-grained, so they
have different parts of speech for plural nouns versus singular nouns. So they're sort of different
symbols, but they're very similar to each other, so we might give them distributed representations so
they're also close to each other. And the same for the types of our labels for dependencies — some of
them are pretty closely related as well. So all of these were being given distributed
representations. And so then, to represent the state of the dependency parser for predicting
transitions, what you were doing is you had the same kind of stack and buffer, and you are taking the
key

**[1:10:39]** elements of the stack and the buffer, which are essentially the first thing on the
buffer — the word that you would be shifting if you're going to do a shift — and the two things at
the top of the stack. So these are the things that, if you're either doing a Left Arc or a Right Arc,
they're the things that you're considering combining. So for those you're going to be taking the
distributed representations of the word and their parts of speech, and also, with a bit more
complexity, for dependencies you've already constructed — if maybe something on the stack is already
involved in a dependency. Each of those, we're going to take their distributed representations and
we're going to just concatenate them together to produce a big vector, in the same

**[1:11:24]** way we were concatenating together the five words in the last class for predicting
whether something was a location or not. And then we're going to feed that into our neural network. So
our input layer is our concatenated distributed representations. We're going to put that through a
hidden layer, which is like we were talking about last time, **Wx** + **b**₁, then put through a ReLU
non-linearity. And then we're going to put above that the same kind of another multiply by a matrix,
so we've got a second layer of neural network, plus **b**₂. And we're going to take the output of that
and then we're going to put that

**[1:12:10]** through a softmax that gives a probability distribution over whether to shift or do a
Left Arc or a Right Arc operation. And so the other way that this crucially gave us more power is
that other people's dependency parsers were still using linear classifiers, things like support
vector machines or logistic regressions, where we had a deep neural network that gave us a
non-linear classifier. And so that's why we could be more accurate than other previous
transition-based parsers. And so this essentially showed that you could build this very accurate
neural

**[1:12:56]** dependency parser, and that it outperformed symbolic, probabilistic representations, and
basically was as good as any other dependency parser that was known. So, back a decade ago, this was
a big hit. People got very excited about it. The people at Google got very excited about it, because
this gave a scalable way — remember, it's linear time — in which you could efficiently go off and
parse the entire web. So they did some further work on taking that model and improving it, so that
they made a deeper neural network version with bigger vectors and better tuned hyperparameters, and
they

**[1:13:43]** added on a beam search. I've just presented the greedy version, where you always just
immediately make the best choice, but you can improve these parsers by doing some amount of search —
that does help. And so they pushed that up, and so rather than our kind of 92 UAS here, they got it
to, you know, 94, 94.6. And I mean, you're probably all too young to remember this, but really at the
time of 2016, Google did their kind of typical big PR splash for dependency

**[1:14:29]** parsing, which kind of blew my mind, since I didn't ever think that anyone was really
going to be writing articles in *Wired* [Ed: the captions have only "W" here] and VentureBeat and
those kinds of tech blogs. But Google had it all over the place, of the world's most accurate parser,
and they gave it a silly name, Parsey McParseFace, which really worked well for getting lots of media
pickup. And so that was then a very successful parser. I've still got a couple of minutes left, so
let me just do the last three slides to show you sort of another way of doing things, which is also
actually a powerful parsing method that is commonly used. So that

**[1:15:16]** was transition-based parsing, and that's what you'll use in Assignment 2. Another way of
doing things with dependencies and parsing, and it can be done neural, is what's referred to as
*graph-based dependency parsers*. And in graph-based dependency parsers, what you do is, for each word
you sort of ask, for each word, what am I a dependent of? So if the sentence is *The big cat sat*,
each word — for example *big* — has to be a dependent of one of the other four words in this sentence,
including this possibility of root. So we ask: am I a dependent of that, am I a dependent of root, am
I a dependent of *cat*, am I a dependent of *sat*? And we want to

**[1:16:03]** score each of those possibilities. And so hopefully we decide the most likely one is the
*big* as a dependent of *cat*. And then we're going to do the same for every other word. So *sat*
could be a dependent of any of these words, and so we could start asking, okay, which of these words
is it most likely a dependent of? Uh, *big* to *sat*, *cat* to *sat* — sorry, that's unreadable now.
But hopefully we decide that *sat*, most likely, as the verb, is a dependent of root. So we're sort of
scoring the *n* squared possible dependencies of the sentence, and each one is given a score. And then
once

**[1:16:50]** we've done that, our job is — let me go to this one, cleaner — okay, we've decided the
good one there. And so we're going to do this using some of the same features we talked about: looking
at the words at each end, looking at what occurs between them, looking at what occurs around them,
thinking about things. And then once we've done that, the only other thing that's a constraint is,
well, we want the dependencies to form a tree, so that we need to do something like a minimum spanning
tree algorithm, to sort of find the minimum cost tree, because we don't want to find a solution where
there are cycles or the parts of the sentence end up disconnected with

**[1:17:35]** each other. And so that's graph-based dependency parsers. And so just as in the older
symbolic parsing days, where the graph-based dependency parsers were more accurate than the
transition-based parsers, we then started doing some work on neural graph-based dependency parsing.
And so here's our neural graph-based dependency parsing, which was then a bit over a percent more
accurate than Parsey McParseFace, the world's best dependency parser. So that got us to 2017. I mean,
obviously this is still a few years ago, but to get further into the latest parsing stories we then
need to sort of get into the era of large

**[1:18:22]** language models, which I'm not doing today. But it's this neural graph-based dependency
parser that's in Stanza, our open-source parsing software that's available, and you can see it's using
this algorithm as the more accurate one. Okay, so now you hopefully know everything about syntactic
structure, constituency and dependency parsing, and are fully qualified to do Assignment 2. So good
luck with that. Thanks.
