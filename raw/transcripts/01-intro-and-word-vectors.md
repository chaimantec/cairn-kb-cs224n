---
title: Lecture 1 — Intro and Word Vectors
lecture: 1
video: https://www.youtube.com/watch?v=DzpHeXVSC5I
source: YouTube auto-captions, copy-edited for readability
verbatim: false
original: original/01-intro-and-word-vectors.md
slides: ../slides/01-intro-and-word-vectors.md
---

# Lecture 1 — Intro and Word Vectors — transcript

Timestamps mark the start of each paragraph and can be cited directly ("Manning
covers this around 52:03"). All 103 paragraph timestamps from the source captions
are preserved.

**This is an edited transcript.** The auto-generated captions had no punctuation and
mangled most technical vocabulary — *word2vec* arrived as "word Tove" and "word DEC",
*PyTorch* as "py talk", *t-SNE* as "tne", *J.R. Firth* as "Jr F". They have been
copy-edited into sentences: punctuation added, false starts and filler removed,
mis-heard terms restored from context and checked against the slides. **No content was
added, removed, or reordered**, and nothing is paraphrased where the wording carries
meaning. The verbatim captions are kept at
[`original/01-intro-and-word-vectors.md`](original/01-intro-and-word-vectors.md) for
reference — use this file unless you specifically need the raw output.

**Where the source is still unreliable:** the mathematics Manning dictates while
writing on his iPad, roughly 1:06:03 to 1:17:05, came through with wrong subscripts and
stray "[Music]" markers. He says himself that the handwriting is going badly and points
at the version on the website (1:06:03). That version is **slides 33–36**, transcribed
in full at [`../slides/01-intro-and-word-vectors.md`](../slides/01-intro-and-word-vectors.md);
prefer it over the equations below.

---

**[0:05]** The thing that seems kind of amazing to me and to us is the fact that — well,
actually, this course was taught just last quarter, and here we are with an enormous
number of people again taking this class. I guess that says something. Maybe
approximately what it says is ChatGPT. But anyway, it's great to have you all. Lots of
exciting content, and I hope you'll all enjoy it. So let me get started and start
telling you a bit about the course before diving straight into today's content. For
people still coming in, there are oodles of seats still, right on

**[0:52]** either side — especially down near the front, there are tons of seats. So do
feel empowered to go out and seek those seats. If people on the corridors are really
nice they could even move towards the edges to make it easier for people. But one way or
another, feel free to find a seat. Okay. So this is the plan for what I want to get
through today. First of all I'm going to tell you about the course for a few minutes,
then have a few remarks about human language and word meaning. Then the main technical
thing we want to get into today is to start learning about the word2vec algorithm. The
word2vec algorithm is slightly over a decade old now — it was introduced in

**[1:41]** 2013 — but it was a wildly successful, simple way of learning vector
representations of words. So I want to show you that as a sort of first, easy, baby
system for the kind of neural representations that we're going to talk about in class.
We're then going to get more concrete with that, looking at its objective function,
gradients and optimization. And then, hopefully, if all goes well and I stick to
schedule, spend a few minutes just playing around in an IPython notebook — I'm going to
have to change computers for that — seeing some of the things you can do with this.
Okay. So this is the course logistics in brief. I'm Christopher Manning. Hi again,
everyone.

**[2:29]** The head TA unfortunately has a bit of a health problem, so he's not actually
here today. We've got a course manager for the course, who is up the back there. And
then we've got a whole lot of TAs. If you're a TA who's here you could stand up and wave
or something like that, so people can see a few of the TAs and see some friendly faces.
Okay, we've got some TAs and some other ones, and so you can look at them on the
website. If you're here, you know what time the class is. There's an email list, but
preferably don't use it — use the Ed site that you can find on the course website. So the
main place to go and look

**[3:16]** for information is the course website, which we've got up here, and that then
links in to Ed, which is what we're going to use as the main discussion board. Please use
that rather than sending emails. The first assignment for this class is a sort of easy
one, it's the warm-up assignment, but we want to get people busy and doing stuff straight
away. So the first assignment is already live on the web page and it's due next Tuesday
before class, so you've slightly less than seven days left to do it. Do get started on
that. And to help with that, we're going to be immediately starting office hours
tomorrow, and they're also described on the website. We also do a

**[4:03]** few tutorials on Friday. The first of these tutorials is a tutorial on Python
and numpy. Many people don't need that, because they've done other classes and done this,
but for some people — we try and make this class accessible to everybody, so if you'd
like to brush up a bit on Python or how to use numpy, it's a great thing to go along to.
And it's going to be taught on Friday. Okay, what do we hope to teach? At the end of the
quarter, when you get the evaluations, you'll be asked to rate whether this class met its
learning goals. These are my learning goals. What are they? So the first one is to

**[4:50]** teach you about the foundations and current methods for using deep learning
applied to natural language processing. So this class tries to build up from the bottom
up. We start off doing simple things like word vectors and feed-forward neural networks,
recurrent networks and attention. We then fairly quickly move into the kind of key
methods that are used for NLP in 2024 — I wrote down here transformers and
encoder-decoder models; I probably should have written large language models somewhere in
this list as well — but then pre-training and post-training of large language models,
adaptation, model interpretability, agents, etc. But that's not the only thing that we
want to do. There are a couple of other things that

**[5:36]** we crucially want to achieve. The second is to give you some understanding of
human languages and the difficulties in understanding and producing them on computers.
Now, there are a few of you in this class who are linguistics majors, or perhaps symbolic
systems majors — yay to the symbolic systems majors — but for quite a few of the rest of
you, you'll never see any linguistics, in the sense of understanding how language works,
apart from this class. So we do want to try and convey a little bit of a sense of what
some of the issues are in language structure, and why it's proven to be quite difficult to
get computers to

**[6:21]** understand human languages, even though humans seem very good at learning to
understand each other. And then the final thing that we want to make it on to is actually
concretely building systems, so that this isn't just a theory class — we actually want you
to leave this class thinking, oh yeah, in my first job, wherever you go, whether it's a
startup or big tech or some nonprofit, oh, there's something they want to do that would be
useful if we had a text classification system, or if we did information extraction to get
some kind of facts out of documents. I know how to build that. I can build that system,
because I did CS224N. Okay. Here's how you get graded.

**[7:08]** So we have four assignments, mainly one and a half weeks long apart from the
first one. They make up almost half the grade. The other half of the grade is made up of a
final project, of which there are two variants, a custom or default final project, which
we'll get on to in a minute. And then there's a few percent that go for participation. Six
late days. Collaboration policy: like all other CS classes, we've had issues with people
not doing their own work. We really do want you to learn things in this class, and the way
you do that is by doing your own work, so make sure you understand that. And so for the

**[7:56]** assignments, everyone is expected to do their own assignments. You can talk to
your friends, but you're expected to do your own assignment. For the final project you can
do that as a group. Then we have the issue of AI tools. Now, of course, in this class we
love large language models, but nevertheless we don't want you to do your assignments by
saying, hey ChatGPT, could you answer question three for me. That is not the way to learn
things. If you want to make use of AI as a tool to assist you, such as for coding
assistance, go for it, but we're wanting you to be working out how to answer assignment
questions by yourself. Okay, so this is what the

**[8:42]** assignments look like. Assignment one is meant to be an easy on-ramp and it's
done as a Jupyter notebook. Assignment two then has people — you know, what can I say?
Here we are at this fine liberal arts and engineering institution. We're not at a coding
boot camp, so we hope that people have some deep understanding of how things work. So in
assignment two we actually want you to do some math and understand how things work in
neural networks. So for some people assignment two is the scariest assignment in the whole
class. But then it's also the place where we

**[9:28]** introduce PyTorch, which is the software package we use for building neural
networks, and we build a dependency parser, which we'll get to later, as something more
linguistic. Then for assignment three and four we move on to larger projects using PyTorch
with GPUs, and we'll be making use of Google Cloud. For those two assignments we look at
doing machine translation and getting information out with transformers. And then these
are the two final project options. Essentially we have a default final project where we
give you a lot of scaffolding and an outline of what to do, but it's still an open-ended
project — there are lots of different things you can try to make

**[10:14]** this system work better, and we encourage you to explore. But nevertheless
you're given a leg up from quite a lot of scaffolding. We'll talk about this more, but you
can either do that option, or you can just come up with totally your own project and do
that. Okay, that's the course. Any questions on the course? Yes.

*[Question: for the final project, how are mentors assigned?]*

So if you can find your own mentor — you're interested in something and there's someone
that's happy to mentor you — that person can be your mentor. Otherwise one of the course
TAs will be your mentor, and how that person is assigned is, one of the TAs

**[11:02]** who is in charge of final projects assigns people, and they do the best they can
in terms of finding people with some expertise, and having to divide all the students
across the mentors roughly equally. Any other questions? Okay, I'll power ahead. Human
language and word meaning. So let me say a little bit about the big picture here. We're in
the area of artificial intelligence, and we've got this idea that humans are intelligent,
and then there's the question of how does language fit into that. And

**[11:47]** this is something that there is some argument about — if you want to, you can run
off onto social media and read some of the arguments about these things, and contribute to
it if you wish. But here is my perhaps biased take as a linguist. Well, you can compare
human beings to some of our nearest neighbours, like chimpanzees, bonobos and things like
that, and one big distinguishing thing is we have language and they don't. But in most
other respects chimps are very similar to human beings, right? They can use

**[12:33]** tools, they can plan how to solve things, they've got really good memory —
chimps have better short-term memory than human beings do. So in most respects it's hard
to show an intelligence difference between chimps and people, except for the fact that we
have language. But us having language has been this enormous differentiator. If you look
around at what happened on the planet, there are creatures that are stronger than us,
faster than us, more venomous than us, that have every possible advantage, but human
beings took over the whole place. And how did that happen? We had language, so we could
communicate. And that

**[13:20]** communication allowed us to have human ascendency. So one big role of language is
the fact that it allows communication. But I'd like to suggest it's actually not the only
role of language — that language has also allowed humans, I would argue, to achieve a
higher level of thought. There are various kinds of thoughts that you can have without any
language involved. You can think about a scene, you can move some bits of furniture around
in your mind, and there's no language. And obviously emotional responses of feeling scared
or excited, they happen and there's no language involved. But I think most of the time when
we're doing higher level

**[14:07]** cognition — if you're thinking to yourself, oh gee, my friend seemed upset about
what I said last night, I should probably work out how to fix that, or maybe I could do
such and such — I think we think in language and plan out things, and so it's given us a
scaffolding to do much more detailed thought and planning. Most recently of all, of course,
human beings invented ways to write. Writing is really, really recent. No one really knows
how old human languages are; most people think a few hundred thousand years, not very long
by evolutionary time scales. But writing — we do know writing is really, really recent. So
writing is about 5,000

**[14:55]** years old. And writing proved to be this amazing cognitive tool that just gave
humanity an enormous leg up, because suddenly it's not only that you could share
information and learn from the people that were standing within 50 feet of you — you could
then share knowledge across time and space. So really, having writing was enough to take us
from the Bronze Age, very simple metalworking, to the kind of mobile phones and all the
other technology that we walk around with today, in just a very short amount of time. So
language is pretty

**[15:41]** cool. But one shouldn't only fixate on the knowledge side of language and how
that's made human beings great. There's this other side of language, where language is this
very flexible system which is used as a social tool by human beings, so that we can speak
with a lot of imprecision and nuance and emotion in language and we can get people to
understand. We can set up new ways of thinking about things by using words for them. And
languages aren't static — languages change as human beings use them. Languages aren't
something

**[16:28]** that were delivered down on tablets by God. Languages are things that humans
constructed, and humans changed them with each successive generation. And indeed, most of
the innovation in language happens among young people — people that are either a few years
younger than you, or most of you are now, in their earlier teens going into their 20s.
That's a big period of linguistic innovation, where people think up cool new phrases and
ways of saying things, and some of those get embedded and extended, and that then becomes
the future of language. Herb Clark used to be a psychologist at Stanford — he's now retired
— but he had this rather nice quote: the common

**[17:15]** misconception is that language use has primarily to do with words and what they
mean. It doesn't. It has primarily to do with people and what they mean. Okay, so that's
language, in two slides for you. So now we'll skip ahead to deep learning. In the last
decade or so we've been able to make fantastic progress in doing more with computers
understanding human languages, using deep learning. We'll say a bit more about the history
later on, but work on trying to do things with human language started in the 1950s, so it
had been going for 60 years or so, and there was some stuff — it's not that nobody

**[18:00]** could do anything — but the ability to understand and produce language had always
been kind of questionable. It's really in the last decade, with neural networks, that
enormous strides of progress have been made, and that's led into the world that we have
today. So one of the first big breakthroughs came in the area of using neural NLP systems
for machine translation. This started about 2014 and was already deployed live on services
like Google by 2016. It was so good that it saw really, really rapid commercial deployment.
And overall this kind of facility with machine

**[18:48]** translation just means that you're growing up in such a different world to people
a few generations back. People a few generations back — unless you actually knew different
languages of different people, you sort of had no chance to communicate with them. Whereas
now we're very close to having something like the Babel fish from *Hitchhiker's Guide to the
Galaxy* for understanding all languages. It's just, it's not a Babel fish, it's a cell
phone. But you can have it out between two people and have it do simultaneous translation.
And it's not perfect, people keep on doing research on this, but by and

**[19:33]** large it means you can pick anything up from different areas of the world. As you
can see, this example is from a couple of years ago, since it's still from the COVID pandemic
era, but I can see this Swahili from Kenya and say, oh gee, I wonder what that means. Stick
it into Google Translate, and I can learn that Malawi lost two ministers due to COVID
infections and they died. So we're just in this different era of being able to understand
stuff. And then there are lots of other things that we can do with modern NLP. Until a few
years ago we had web search engines, and you put in some text — you could write it as a

**[20:20]** sentence if you wanted to, but it didn't really matter whether you wrote a
sentence or not, because what you got was some keywords that were then matched against an
index, and you were shown some pages that might have the answers to your questions. But
these days you can put an actual question into a modern search engine, like *when did
Kendrick Lamar's first album come out*. It can go and find documents that have relevant
information, it can read those documents, and it can give you an answer. So it actually can
become an answer engine, rather than just something that finds documents that might be
relevant to what you're interested in. And the way that that's done is with big neural
networks, so that you might commonly have, for

**[21:05]** your query, a retrieval neural network which can find passages that are similar to
the query. They might then be reranked by a second neural network, and then there'll be a
third, reading neural network that'll read those passages and synthesize information from
them, which it then returns as the answer. Okay, that gets to about 2018. But then things
got more advanced again. It was really around 2019 that people started to see the power of
large language models. Back in 2019 those of us in NLP were really excited about GPT-2. It
didn't make much of an impact on the nightly news, but it was really

**[21:52]** exciting in NLP land, because GPT-2 already, for the first time, meant here was a
large language model that could just generate fluent text. Until then, NLP systems had done
a decent job at understanding certain facts out of text, but we'd just never been able to
generate fluent text that was at all good. Whereas here, what you could do with GPT-2 is you
could write something like the start of a story — *a train carriage containing controlled
nuclear materials was stolen in Cincinnati today. Its whereabouts are unknown* — and then
GPT-2 would just write a continuation: *the incident occurred on the downtown train line
which runs from*

**[22:39]** *Covington and Ashland stations. In an email to Ohio news outlets, the US
Department of Energy said it is working with the Federal Railroad Administration to find the
thief …* And so the way this is working is, this is conditioning on all the past material,
and as I show at the very bottom line down here, it's generating one word at a time as to
what word it thinks would be likely to come next after that. And so from that simple method
of generating words one after another, it's able to produce excellent text. And the thing to
notice is, this text is not only formally correct — the spelling's correct and the sentences
are real sentences, not

**[23:26]** disconnected garbage — but it actually understands a lot. The prompt that was
written said there were stolen nuclear materials in Cincinnati, but GPT-2 knows a lot of
stuff. It knows that Cincinnati is in Ohio. It knows that in the United States it's the
Department of Energy that regulates nuclear materials. It knows if something is stolen it's a
theft, and that it would make sense that people are getting involved with that. It talks
about a train carriage, so it's talking about the train line, where it goes. It really knows
a lot, and can write coherent discourse like a real story. So that's

**[24:13]** kind of amazing. But things moved on from there, and so now we're in the world of
ChatGPT and GPT-4. One of the things that we will talk about later is that this was a huge,
huge user success, because now you could ask questions or give it commands and it would do
what you wanted, and that was further amazing. So here I'm saying, hey, please draft a polite
email to my boss Jeremy that I would not be able to come into the office for the next two
days because my nine-year-old song — that's a misspelling for son, but the system works fine
despite it — Peter is angry with me that I'm not giving him much time. And it writes a nice
email. It

**[25:01]** corrects the spelling mistake, because it knows people make spelling mistakes; it
doesn't talk about songs, and everything works out beautifully. You can get it to do other
things. You can ask it, what is unusual about this image? In thinking about meaning, one of
the things that's interesting with these recent models is that they're multimodal and can
operate across modes. A favourite term that we coined at Stanford is the term **foundation
models**, which we use as a generalization of large language models, to have the same kind of
technology used across different modalities: images, sound, various kinds of bioinformatic
things,

**[25:48]** DNA, RNA, things like that, seismic waves, any kind of signal — building these same
kind of large models. Another place that you can see that is going from text to images. So if
I ask for a picture of a train going over the Golden Gate Bridge — this is now DALL-E 2 — it
gives me a picture of a train going over the Golden Gate Bridge. This is a perfect time to
welcome anyone who's watching this on Stanford Online. If you're on Stanford Online and are
not in the Bay Area, the important thing to know is no trains go over the Golden Gate Bridge.
But you

**[26:34]** might not be completely happy with this picture, because it shows the Golden Gate
Bridge and a train going over it, but it doesn't show the bay. So maybe I'd like to get it
with the bay in the background, and if I ask for that — well, look, now I've got a train going
over the Golden Gate Bridge with the bay in the background. But you still might think this is
not exactly what you want. Maybe you'd prefer something that's a pencil drawing, so I can say
*a train going over the Golden Gate Bridge, detailed pencil drawing*, and I can get a pencil
drawing. Or maybe it's unrealistic that the Golden Gate Bridge only has trains going over it,
so maybe it'd be good to have some cars as well. So I could ask for a train

**[27:20]** and cars, and we can get a train and cars going over it. Now, I actually made these
ones all by myself, so you should be impressed with my generative AI artwork. But these
examples are actually a bit old now, because they're done with DALL-E 2, and if you keep up
with these things, that's a few years ago — there's now DALL-E 3 and so on. So we can now get
much fancier things again: *an illustration from a graphic novel. A bustling city street under
the shine of a full moon. The sidewalks bustling with pedestrians enjoying the nightlife. At
the corner stall, a young woman with fiery red hair, dressed in a signature velvet cloak, is
haggling with the grumpy old vendor. The grumpy vendor, a tall, sophisticated man, is wearing
a sharp suit, sports a noteworthy moustache, and is animatedly*

**[28:05]** *conversing on his steampunk telephone.* And pretty much we're getting all of that.
Okay, so let's now get on to starting to think more about meaning. So what can we do for
meaning? If you think of words and their meaning, and you look up a dictionary and say, what
does *meaning* mean — meaning is defined as the idea that is represented by a word or phrase;
the idea that a person wants to express by using words; the idea that is expressed. And in
linguistics, if you go and do a semantics class or something, the

**[28:50]** commonest way of thinking of meaning is somewhat like what's presented up above
there: that meaning is thought of as a pairing between what's sometimes called signifier and
signified, but it's perhaps easier to think of as a symbol — a word — and then an idea or
thing. And so this notion is referred to as **denotational semantics**: the idea or thing is
the denotation of the symbol. This same idea of denotational semantics has also been used for
programming languages, because in programming languages you have symbols like `while` and
`if`, and variables, and they have a meaning, and that could be their denotation. So we would
say that the meaning of *tree* is all the

**[29:38]** trees you can find out around the world. That's an okay notion of meaning. It's a
popular one. It's never been very obvious — or at least traditionally it wasn't very obvious —
what we could do with that to get it into computers. So if you looked in the pre-neural world,
when people tried to look at meanings inside computers they had to do something much more
primitive, of looking at words and their relationships. So a very common traditional solution
was to make use of **WordNet**. WordNet was a kind of fancy thesaurus that showed word
relations, so it'd tell you about synonyms and *is-a-kind-of* things.

**[30:23]** So a panda is a kind of carnivore, which is a placental mammal, and things like
that. *Good* has various meanings — it's a trade good, or the sense of goodness — and you could
explore with that. But systems like WordNet were never very good for computational meaning.
They missed a lot of nuance. WordNet would tell you that *proficient* is a synonym for *good*,
but if you think about all the things that you would say were good — you know, that was a good
shot — would you say that was a proficient shot? Sounds kind of weird to me. There's a lot of
colour and nuance in how words are used. WordNet is very incomplete; it's missing anything
that's cooler, more modern slang. This

**[31:10]** maybe isn't very modern slang now, but you won't find more modern slang either in
it. It's very human-made, etc. It's got a lot of issues. So this led into the idea of, can we
represent meaning differently? And this leads us into word vectors. When we have words —
*wicked*, *badass*, *nifty*, *wizard* — what do they turn into when we have computers? Well,
effectively, words are these discrete symbols. They're just some kind of atom or symbol. And if
we then turn those into something that's closer to math, how symbols are

**[31:57]** normally represented is, you have a vocabulary, and your word is some item in that
vocabulary. So *motel* is that word in the vocabulary and *hotel* is this word in the
vocabulary. And commonly this is what computational systems do: you take all your strings and
you index them to numbers, and that's the position in a vector that they belong in. And we have
huge numbers of words, so we might have a huge vocabulary, so we'll have very big and long
vectors. And so these get referred to as **one-hot vectors** for representing the meaning of
words. But representing words by one-hot vectors

**[32:42]** turns out to not be a very good way of computing with them. It was used for decades,
but it turns out to be kind of problematic. And part of why it's problematic is it doesn't have
any natural, inherent sense of the meanings of words. You just have different words. You have
*hotel* and *motel* and *house* and *chair*. And so if you think about these vector
representations, if you have *motel* and *hotel*, there's no indication that they're similar.
They're just two different symbols which have ones in different positions in the vector. Or
formally, in math terms, if you think about taking the dot product of these two vectors, it's
zero. The two vectors are

**[33:30]** **orthogonal**. They have nothing to do with each other. Now, there are things that
you can do with that. You can start saying, oh, let me start building up some other resource of
word similarity, and I'll consult that resource of word similarity and it'll tell me that motels
and hotels are similar to each other. And people did things like that — in web search it was
referred to as query expansion techniques. But still, the point is that there's no natural
notion of similarity in one-hot vectors. And so the idea was that maybe we could do better than
that, that we could learn to include similarity in the vectors themselves. And so that leads

**[34:16]** into the idea of word vectors. But it also leads into a different way of thinking
about semantics. I just realized I forgot to say one thing back two slides: these kinds of
representations are referred to as **localist representations**, meaning that there's one point
at which something is represented, so that here is the representation of *motel* and here is the
representation of *hotel*. It's in one place in the vector that each word is represented, and
they'll be different to what we do next. So there's an alternative idea of semantics which goes
back quite a long way.

**[35:01]** People commonly quote this quote of J.R. Firth, who was a British linguist, who said
in 1957, *you shall know a word by the company it keeps*. But it also goes back to philosophical
work by Wittgenstein and others: that what you should do is represent a word's meaning by the
context in which it appears. So the words that appear around the word give information about its
meaning, and so that's the idea of what's called **distributional semantics**, in contrast to
denotational semantics. So if I want to know about the word *banking*, I say, give me some
sentences that use the word *banking*. Here are some sentences using the word *banking*:
*government debt problems turning*

**[35:47]** *into banking crises as happened in 2009*, etc, etc. And knowing about that context —
words that occur around *banking* — those will become the meaning of *banking*. And so we're
going to use those statistics about words and what other words appear around them in order to
learn a new kind of representation of a word. So our new representation of words is, we're going
to represent them now as a dense, sort of shorter vector that gives the meaning of the words. Now,
my vectors are very short here. These are

**[36:32]** only eight dimensional, if I counted right, so I could fit them on my slide. They're
not that short in practice — they might be 200, 2,000 — but reasonably short. They're not going
to be like the half a million different words in our vocabulary. And the idea is, if words have
stuff to do with each other, they'll have similar vectors, which corresponds to their dot product
being large. So for *banking* and *monetary* in my example here, both of them are positive in the
first dimension, positive in the second dimension, negative on the third; on the fourth they've
got opposite signs. So if we want to work out the dot product, we're taking the product of the
corresponding terms, and it'll get bigger to the extent that

**[37:19]** both of the corresponding ones have the same signs, and bigger if they have large
magnitude. Okay, so these are what we call **word vectors**, which are also known as
**embeddings**, or neural word representations, or phrases like that. And so the first thing we
want to do is learn good word vectors for different words. And our word vectors will be good word
vectors if they give us a good sense of the meanings of words — they know which words are similar
to other words in meaning. We refer to them as embeddings because we can think of this as a vector
in a high dimensional

**[38:05]** space, and so we're embedding each word as a position in that high-dimensional space.
And the dimensionality of the space will be the length of the vector, so it might be something
like a 300 dimensional space. Now, that kind of gets problematic, because human beings can't look
at 300 dimensional spaces and aren't very good at understanding or visualizing what goes on in
them. So the only thing that I can show you is two-dimensional spaces. But a thing that is good
to have somewhat in your head is that really high-dimensional spaces behave

**[38:50]** extremely differently to two-dimensional spaces. In a two-dimensional space you're
only near to something else if you've got similar x and y coordinates. In a high dimensional
space, things can be very near to all sorts of things on different dimensions in the space, and so
we can capture different senses of words, and ways that words are similar to each other. But
here's the kind of picture we end up with. What we're going to do is learn a way to represent all
words as vectors based on the other words that they occur with in context, and we can embed them
into this vector space. And of course you can't read anything

**[39:37]** there, but we can zoom into this space further. And if we zoom into this space and just
show a bit of it — well, here's a part of the space where it's showing country words and some
other location words. So we've got countries up the top there, we've got some nationality terms,
*British*, *Australian*, *American*, *European*. Further down, or we can go to another piece of
the space, and here's a bit of the space where we have verbs. And not only have we got verbs, but
there's actually quite a lot of fine structure here of what's similar, that represents things
about verbs. So you've got verbs of communication, statements — *saying*, *thinking*,

**[40:22]** *expecting* — grouping together. *Come* and *go* group together. Down the bottom
you've got forms of the verb *have*, then you've got forms of the verb *to be*. Above them you've
got *become* and *remain*, which are actually similar to the verb *to be*, because they take these
sort of complements of state — so just as you can say *I am angry*, you can say *he remained
angry* or *he became angry*. So those verbs are, more so than most verbs, similar to the verb *to
be*. So we get these kinds of interesting semantic spaces where things that have similar meaning
are close by to each other. And so the question is, how do we get to those things? And how we get

**[41:09]** to those things is — there are various ways of doing it, but the one I want to get
through today is showing you about word2vec. Okay, I'll pause for 30 seconds for breath. Anyone
have a question, or anything they want to know? Yes.

*[Question: but doesn't this fail to solve the problem where similar meanings might depend on
context? To take your example about* proficient *versus* good *— those two words have their own
vectors, and we understand similarity from the vectors, but it's contextual, right?*

**[41:56]** *Because if you have a different context those two aren't similar, and this also does
not capture that.]*

Yes, correct. So that's a good thought — you can keep it for a few weeks. To some extent, yeah. So
for the first thing we're going to do, we're just going to learn one word vector for a string. So
we're going to have a word, let's say it's *star*, and we're going to learn one word vector for it.
So that absolutely doesn't capture the meaning of a word in context. It won't be saying whether
it's meaning a Hollywood star or an astronomical star or something like that. And so later on we're
going to get on to contextual meaning representation, so wait for that. But the thing I would like

**[42:43]** to say, going along with what I said about high dimensional spaces being weird, the cool
thing that we will already find is our representation for *star* will be very close to the
representations for astronomical words like *nebula*, and whatever other astronomical words, and
simultaneously it'll be very close to words that mean something like a Hollywood star. Help me out
— any words that mean something similar? *Celebrity*, that's a good one. Okay, yeah.

*[Question: how are you visualizing the embedding into a lower dimensional space?]*

**[43:29]** So those pictures I was showing you used a particular method called t-SNE, which is a
nonlinear dimensionality reduction that tends to work better for high dimensional neural
representations than PCA, which you might know. But I'm not going to go into that now. Yes.

*[Question: how do you know how many dimensions to use — not too many, not too few?]*

I mean, that's something that people have worked on. It depends on how much data you've got to make
your representations over. So normally it's worked out either empirically, for what works best,

**[44:16]** or practically, based on how big vectors you want to work on. I mean, to give you some
idea, things start to work well when you get to 100 dimensional space. For a long time people used
300 dimensions, because that seemed to work pretty well. But as people have started building huger
and huger models with way, way more data, it's now become increasingly common to use numbers like
1,000 or even 2,000 dimensional vectors. Yeah.

*[Question: you mentioned there's hidden structure in the small areas as well as the large areas of
the embedding, and different structures come up. But generally we*

**[45:01]** *seem to use distance as the single metric for closeness, which doesn't seem — like the
distance between this and that in the space would be the same, right? So how would that work?]*

We don't only use distance. We also use directions in the spaces having semantic meanings, and I'll
show an example of that soon. Yeah.

*[Question: the entries seem to be between −1 and 1. Is there a reason for that, or do we have
bounds that we set?]*

So, good question. They don't have to be, and the way we're going to learn them, they're not
bounded. But you can bound things — sometimes people length-normalize, so that the

**[45:48]** vectors are of length one. But at any rate, normally in this work we use some method
called regularization that tries to keep coefficients small, so they're generally not getting huge.
Yeah.

*[Question: for a specific word, for example* bank *as we used before in the previous slides — for
the word representation, is there a single embedding for each word, or do we have multiple
embeddings for each word?]*

What we're doing at the moment, each word, each string of letters, has a single embedding. And what
you can think of that embedding as is kind of as an average over all its

**[46:35]** senses. So for example, like *bank* — it can mean the financial institution, or it can
also mean the river bank. And then what I said before about *star* applies. The interesting thing is
you'll find that we're able to come up with a representation where our learned representation,
because it's kind of an average of those, will end up similar to words that are semantically evoked
by both senses. I think I should probably go on at this point. Okay. word2vec. So word2vec was this
method of learning word vectors that was thought up by Tomas Mikolov and colleagues at Google in
2013. It wasn't the first method —

**[47:23]** there are other people that did methods of learning word vectors that go back to about
the turn of the millennium. It wasn't the last, there are ones that come after it as well. But it
was a particularly simple one, and a particularly fast running one, and so it really caught
people's attention. So the idea of it is that we start off with a large amount of text, and that can
just be thought of as a long list of words. And in NLP we refer to that as a **corpus**. *Corpus* is
just Latin for body, so it's exactly the same as if you have a dead person on the floor — right,
that's a corpus. No.

**[48:10]** So it's just a body, but we mean a body of text, not a live person. Oh, sorry, a dead
person. If you want to know more about Latin, since there isn't very good classical education these
days: *corpus*, despite the *-us* ending, is a fourth declension neuter noun, and that means the
plural of *corpus* is not *corpi*, the plural of *corpus* is **corpora**. So I'm sure sometime later
in this class I will read a project or assignment that refers to *corpi*, and I will know that that
person was not paying attention in the first lecture,

**[48:55]** or else they should have said *corpora*. C-o-r-p-o-r-a is the correct form for that.
Okay, I should move on. So we have our text. Then we know that we're going to represent each word —
so this is each word *type*, so *star* or *bank* etc, so for wherever it occurs — by a single
vector. And so what we're going to do in this algorithm is we're going to go through each position
in the text, and at each position in the text, which is a list of words, we're going to have a
**center word** and words outside it. And then what we're going to do is use the similarity of the
word vectors

**[49:42]** for *c* and the outside words to calculate the probability that they should have occurred
or not. And then we just keep fiddling, and we learn word vectors. Now, at first sight — I'll show
this more concretely. Maybe I'll just show it more concretely first. So here's the idea. We're going
to have a vector for each word type. A word type means the word *problems* wherever it occurs, which
is differentiated from a word *token*, which is this instance of the word *problems*. So we're going
to have a vector for each word type. And so I'm going to want to know, look in this text, the word
*turning* occurred before the word *into*. How likely

**[50:29]** should that have been to happen? And what I'm going to do is calculate a probability of
the word *turning* occurring close to the word *into*, and I'm going to do that for each word in a
narrow context. In the example here I'm using two words to the left and two words to the right. And
what I want to do is make those probability estimates as good as possible. So in particular I want
the probability of co-occurrence to be high for words that actually do occur within the nearby
context of each other. And so then the question is, how am I going to — oh, and once I've done it
for that word, I'm going to go along and do exactly the same thing for the next

**[51:15]** word, and so I'll continue through the text in that way. And so what we want to do is
come up with vector representations of words that will let us predict these probabilities, quote
unquote, well. Now, there's a huge limit to how well we can do it, because we've got a simple model.
Obviously, when you see the word *banking*, I can't tell you that the word *into* is going to occur
before *banking*. But I want to do it as well as possible. So what I want my model to say is, after
the word *banking*, *crisis* is pretty likely, but the word *skillet* is not very

**[52:03]** likely. And if I can do that, I'm doing a good job. And so we turn that into a piece of
math. Here's how we do it, turn it into a piece of math. So we're going to go through our corpus,
every position in the corpus, and we're going to have a fixed window size *m*, which was two in my
example. And then what we're going to want to do is have the probability of words in the context
being as high as possible. So we want to maximize this likelihood, where we're going through every
position in the text, and then we're going through every word in the context, and wanting to make
this big. Okay, so conceptually that's what

**[52:51]** we're doing. But in practice we never quite do that. We use two little tricks here. The
first one is, for completely arbitrary reasons — it really makes no difference — everyone got into
minimizing things rather than maximizing things, and so the algorithms that we use get referred to
as gradient *descent*, as you'll see in a moment. So the first thing we do is put a minus sign in
front, so that we can minimize it rather than maximize it. That part's pretty trivial. But the
second part is, here we have this enormous product, and working with enormous products is more
difficult for the math. So the second thing that we do

**[53:37]** is introduce a logarithm. And so once we take the log of the likelihood, then, when we
take logs of products, they turn into sums. And so now we can sum over each word position in the
text, sum over each word in the context window, and then sum these log probabilities. And then we've
still got the minus sign in front, so we want to minimize the sum of log probabilities. So what
we're doing is then wanting to look at the negative log likelihood. And then the final thing that we
do is, since this will get bigger depending on the

**[54:23]** number of words in the corpus, we divide through by the number of words in the corpus.
And so our objective function is the **average negative log likelihood**. So by minimizing this
objective function we're maximizing the probability of words in the context. Okay, we're almost
there. That's what we want to do. We've got a couple more tricks that we want to get through. The
next one is — well, I've said we want to maximize this probability. How do we maximize this
probability? What is this probability? We haven't defined how we're going to calculate this
probability, and this is where the word vectors come in. So we're going to define

**[55:11]** this probability in terms of the word vectors. So we're going to say each word type is
represented by a vector of real numbers — these are 100 real numbers — and we're going to have a
formula that works out the probability simply in terms of the vectors for each word. There are no
other parameters in this model. So over here I've shown this theta, which are the parameters of our
model, and all and only the parameters of our model are these word vectors for each word in the
vocabulary. That's a lot of parameters, because we have a lot of words, and we've got fairly big word
vectors, but they are the only parameters.

**[55:57]** And how we do that is by using this little trick here. We're going to say the probability
of an outside word given a center word is going to be defined in terms of the dot product of the two
word vectors. So if things have a high dot product they'll be similar, and therefore they'll have a
high probability of co-occurrence — where I mean similar in a kind of a weird sense. It is the case
that we're going to want to say *hotel* and *motel* are similar, but it's also the case that we're
going to want to have the word *the* be able to appear easily before the word *student*. So in some
weird sense *the* also has to be similar to *student*. That has to be

**[56:44]** similar to basically any noun. Okay, so we're going to work with dot products, and then
we do this funky little bit of math here, and that will give us our probabilities. Okay, so let's
just go through the funky bit of math. So here's our formula for the probabilities. So what we're
doing here is we're starting off with this dot product. The dot product is, you take the two
vectors, you multiply each component together, and you sum them. So if they're both the same sign
that increases your dot product, and if they're both big it increases it a lot. Okay, so that gives
us a similarity

**[57:30]** between two vectors, and that's unbounded. That's just a real number, it can be either
negative or positive. Okay, but what we'd like to get out is a probability. So for our next tricks
we first of all exponentiate, because if we take *e* to the *x* for any *x*, we now have to get
something positive out. That's what exponentiation does. Okay, and then, since it's meant to be a
probability, we'd like it to be between 0 and 1, and so we turn it into numbers between 0 and 1 in
the dumbest way possible, which is we just normalize. So we work out the quantity in the numerator
for every possible context word, and so we get the total of all

**[58:18]** of those numbers and divide through by it. And then we're getting a probability
distribution of how likely different words are in this context. Okay, so this little trick that we're
doing here is referred to as the **softmax function**. So for the softmax function you can take
unbounded real numbers, put them through this little softmax trick that we just went through the
steps of, and what you'll get out is a probability distribution. So I'm now getting, in this example,
a probability distribution over context words. My probability estimates over all the context words in
my vocabulary will sum up to one, by

**[59:05]** definition, by the way that I've constructed this. By the way, it's called the softmax
function because it amplifies the probabilities of the largest things — that's because of the exp
function — but it's soft because it still assigns some probability to smaller items. But it's sort of
a funny name, because when you think about max, max normally picks out just one thing, whereas the
softmax is turning a bunch of real numbers into a probability distribution. So this softmax is used
everywhere in deep learning. Any time

**[59:50]** that we're wanting to turn things that are just vectors in ℝⁿ into probabilities, we shove
them through a softmax function. Okay. So in some sense this part, I think, still seems very
abstract. And the reason it seems very abstract is because I've said we have vectors for each word,
and using these vectors we can then calculate probabilities. But where do the vectors come from? And
the answer to where the vectors are going to come from

**[1:00:37]** is, we're going to turn this into an optimization problem. We have a large amount of
text, and so therefore we can hope to find word vectors that make the contexts of the words in our
observed text as big as possible. So literally what we're going to do is, we're going to start off
with random vectors for every word, and then we want to fiddle those vectors so that the calculated
probabilities of words in a context go up. And we're going to keep fiddling until they stop going up
any more and we're getting the highest probability estimates that we can. And the way that we do
that fiddling is we use

**[1:01:23]** calculus. So what we're going to do is conceptually exactly what you do if you're in
something like a two-dimensional space, like the picture on the right. If you want to find the
minimum in this two-dimensional space and you start off at the top left, what you can do is say, let
me work out the derivatives of the function at the top left, and they point sort of down and a bit to
the right, and so you can walk down and a bit to the right. And you can say, oh gee, given where I am
now, let me work out the derivatives — what direction do they point? And they're still pointing down,
but a bit more to the right, so you can walk a bit further that way. And you can keep on walking, and
eventually you'll make it to the minimum

**[1:02:11]** of the space. In our case we've got a lot more than two dimensions, so our parameters
for our model are the concatenation of all the word vectors. But it's even slightly worse than I've
explained up until now, because actually for each word we assume two vectors: we assume one vector
when they're the center word, and one vector when they're the outside word. Doing that just makes the
math a bit simpler, which I can explain later. So if we say we had 100 dimensional vectors, we'll have
100 parameters for *aardvark* as an outside word, 100 parameters for *a* as an outside

**[1:02:56]** word, all the way through to 100 parameters for *zebra* as an outside word. Then we'd
have 100 parameters for *aardvark* as a center word, continuing down. So if we had a vocabulary of
400,000 words and 100 dimensional word vectors, that means we'd have 400,000 × 2 is 800,000, times 100
— we'd have 80 million parameters. So that's a lot of parameters in our space to try and fiddle to
optimize things. But luckily we have big computers, and that's the kind of thing that we do. So we
simply say, gee, this is our optimization problem. We're

**[1:03:41]** going to compute the gradients of all of these parameters, and that will give us the
answer of what we have. And this feels like magic. I mean, it doesn't really seem like we could just
start with nothing — we could start with random word vectors and a pile of text and say, uh, do some
math, and we will get something useful out. But the miracle of what happens in these deep learning
spaces is we do get something useful out. We can just minimize over all of the parameters, and then
we'll get something useful out.

**[1:04:29]** So what I wanted to — I guess I'm not going to quite get to the end of what I hoped to
today. But what I wanted to do is get through some of what we do here. I wanted to take a few minutes
to go through concretely how we do the math of minimization. Now, lots of different people take
CS224N, and some of you know way more math than I do, and so this next 10 minutes might be extremely
boring — and if that's the case you can either catch up on Discord or Instagram or something, or else
you can leave. But it turns out

**[1:05:16]** there are other people that do CS224N that can't quite remember when they last did a
math course, and we'd like everybody to be able to learn something about this. So I do actually like,
in the first two weeks, to go through it a bit concretely. So let's try to do this. So this was our
likelihood. And then we'd already covered the fact that what we were going to do is have an objective
function, in terms of our parameters, that was the average negative log likelihood across all the
words. If I remember the notation for this, the sum

**[1:06:03]** — oops. I'll probably have a hard time writing this. The sum of position *m*. I've got a
more neatly written out version of it that appears on the version of the slides that's on the web. And
then we're going to be taking this log of the probability of the word at position *t + j*,

**[1:06:50]** *w_t*. Okay. And so then we had the form of what we wanted to use for the probability.
And the probability of an outside word given a context word was then this softmaxed equation, where
we're taking the exp of the outside vector and the center vector over the normalization term, where we
sum over the

**[1:07:38]** vocabulary. Okay. So to work out how to change our parameters — so our parameters are
all of these word vectors that we summarize inside theta — what we're then going to want to do is work
out the partial derivative of this objective function with respect to all the parameters theta. But in
particular, I'm just going to start doing here the partial derivatives with respect to the center
word, and we can work through the outside words separately. Well, this partial

**[1:08:26]** derivative is a big sum, and it's a big sum of terms like this. And so when I have a
partial derivative of a big sum of terms, I can work out the partial derivatives of each term
independently and then sum them. So what I want to be doing is working out the partial derivative of
the log of this probability, which equals the log of that, with respect to the center vector. And so at
this point I have a log of two things being divided, and so that means I can separate that out to the
log

**[1:09:13]** of the numerator minus the log of the denominator. And so what I'll be doing is working
out the partial derivative with respect to the center vector of the log of the numerator, log exp of
*u_oᵀ v_c*, minus the partial derivative with respect to the denominator, which is then the log of the
sum of *w* = 1 to *V* of exp of *u_w*. Okay, I'm having real trouble here

**[1:10:02]** writing. Look at the slides, where I wrote it neatly at home. Okay, so I want to work
with these two terms. Now, at this point part of it is easy, because here I just have a log of an
exponential, and so those two functions just cancel out and go away. And so then I want to get the
partial derivative of *u* outside transpose *v* center, with respect to *v* center. And what you get
for the answer to that is that that just comes out as

**[1:10:49]** *u*. And maybe you remember that, but if you don't remember that, the thing to think
about is, okay, this is a whole vector, right? And so we've got a vector here and a vector here, so
what this is going to look like is *u₁v₁* plus *u₂v₂* plus *u₃v₃*, etc, along. And so what we're going
to want to do is work out the partial derivative with respect to each element *v_i*. And so if you just
think of a single element derivative, with respect to

**[1:11:34]** *v₁* — well, it's going to be just *u₁*, because every other term would go to zero. And
then if you worked it out with respect to *v₂*, then it would be just *u₂*, and every other term goes to
zero. And so since you keep on doing that along the whole vector, what you're going to get out is the
vector *u₁*, *u₂*, *u₃*, down the vocab, for the whole list of vocab items. Okay, so that part is easy.
But then we also want to work out the partial derivatives of that one, and at that point I maybe have to
go to

**[1:12:20]** another slide. So we then want to have the partial derivative with respect to *v_c* of the
log of the sum *w* = 1 to *V* of the exp *u_wᵀ v_c*. Right, so at this point things aren't quite so easy,
and we have to remember a little bit more calculus. So in particular, what we have to remember is the
chain rule. So here we have this inside function, so that we've got a function *g*

**[1:13:10]** of *v_c*, which we might say the output of that is *z*, and then we put outside that an
extra function *f*. And so when we have something like that, what we get is, the derivative of *f* with
respect to *v_c*, we can take the derivative of *f* with respect to *z* times the derivative of *z* with
respect to *v_c*. That's the chain rule. So we're going to then apply that here. So first of all we're
going to take the derivative of log, and so the derivative of log is 1 over *x* —

**[1:13:55]** you have to remember that, or look it up, or get Mathematica to do it for you or something
like that. And so we're going to have 1 over the inside *z* part, the sum of *w* = 1 to *V* of the exp
*u_wᵀ v_c*, and then that's going to be multiplied by the derivative of the inside part. So then we're
going to have the derivative with respect to *v_c* of the sum of *w* = 1 to *V* of the exp

**[1:14:47]** of — okay. So that's made us a little bit of progress, but we've still got something to do
here. And so what we're going to do here is we're going to notice, oh wait, we're again in the space to
run the chain rule again. So now we've got this function. Well, so first of all we can move the sum to
the outside, right, because we've got a sum of terms *w* = 1 to *V*. And so we want to work out the
derivatives of the inside piece with respect to it — sorry, I'm doing this kind of informally, of just
doing this piece now. Okay, so this again gives us an *f*

**[1:15:33]** over a function *g*, and so we're going to again want to split the pieces up, and so use
the chain rule one more time. So we're going to have the sum of *w* = 1 to *V*, and now we have to know
what the derivative of exp is, and the derivative of exp is exp. So that will be exp of *u_xᵀ v_c*, and
then we're taking the derivative of the inside part with respect to *v_c* of *u_xᵀ v_c*. Well, luckily
this was the bit that we already knew how to do, because we worked it out before. And so this is going
to be

**[1:16:19]** the sum of *w* = 1 to *V* of this exp times *u_x*. Okay, so then at this point we want to
combine these two forms together, so that we want to combine this part that we worked out and this
piece here that we've worked out. And if we combine them together with what we worked out on the first
slide for the numerator — since this is, we have the *u*, which was the derivative of the

**[1:17:05]** numerator, and then for the derivative of the denominator we're going to have on top this
part, and then on the bottom we're going to have that part. And so we can rewrite that as the sum from
*w* = 1 to *V* of the exp of *u_xᵀ v_c* times *u_x*, over the sum over *w* = 1 to *V* of the exp of this
part, of *u_w*.

**[1:17:56]** Okay, so we can rearrange things in that form, and then lo and behold, we find that we've
recreated here this form of the softmax equation. So we end up with *u* minus the sum *x* = 1 to *V* of
the probability of *x* given *c* times *u* of *x*. So what this is saying is, we're wanting to have this
quantity which takes the actual observed *u* vector and compares it to the weighted prediction. So we're
taking the weighted sum of

**[1:18:42]** our current *u_x* vectors based on how likely we thought they were to occur. And so this is
a form that you see quite a bit in these kinds of derivations: you get **observed minus expected**, the
weighted average. And so what you'd like to have is your expectation, the weighted average, be the same
as what was observed, because then you'll get a derivative of zero, which means that you've hit a
maximum. And so that gives us the form of the derivative that we're having with respect to the center
vector parameters. To finish it

**[1:19:30]** off, you'd have to then work it out also for the outside vector parameters. But hey, it's
officially the end of class time, so I'd better wrap up quickly now. But so the deal is, we're going to
work out all of these derivatives for each parameter, and then these derivatives will give a direction
to change numbers, which will let us find good word vectors automatically. I do want you to understand
how this works, but fortunately you'll find out very quickly that computers will do this for you, and on
a regular basis you don't actually have to do it yourself. More about that on Thursday. Okay, see you
everyone.
