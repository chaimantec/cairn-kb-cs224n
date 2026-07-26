---
title: Pretraining
lecture: 9
video: https://www.youtube.com/watch?v=DGfCRXuNA2w
source: edited from auto-generated YouTube transcript
verbatim: false
original: original/09-pretraining.md
slides: ../slides/09-pretraining.md
---

# Pretraining — transcript

Copy-edited for punctuation, sentence boundaries, and mis-heard terms; cross-checked
against `raw/slides/09-pretraining.md`. Mathematical expressions dictated aloud are
written in symbols (bold for vectors/matrices, Unicode subscripts), matching the
convention used for lecture 3 — LaTeX is reserved for the wiki. The verbatim
auto-generated captions are kept at `raw/transcripts/original/09-pretraining.md`.
Lecturer is John Hewitt (slides adapted from Anna Goldie and John Hewitt). Student
questions and comments from the floor are set in *italics*. Timestamps mark the start of
each paragraph; all 102 are preserved in order.

**This is an edited transcript.** The captions had no punctuation and destroyed most of
the technical vocabulary: *word2vec* arrived as "word to VEC", "where to back" and "word
to back"; *UNK* as "ankh", "Unk" and "UNC"; *BERT* as "Bert", "Birch", "bird" and "burp
model"; *RoBERTa* as "Brita"; *GPT-2* and *GPT-3* as "gpt2", "gpd2", "gbd3", "gpd3" and
"cpt3"; *ChatGPT* as "chat gbt"; *Iroh* as "Ira", "iro" and "IRL"; *Adam* as "atom";
*AdaGrad* as "add a grad"; *masked language modeling* as "mass language modeling";
*salient span masking* as "salience band masking"; *prepend* as "propend"; and *Quora* as
"quora". Terms and citations were restored from context and checked against the slides.
**No content was added, removed, or reordered.**

**Where the source is still unreliable**, the text carries an inline `[Ed:` note rather
than a silent guess. There are four, at 11:39, 13:12, 34:03 and 52:30 — two heavily
garbled student questions, a garbled word inside a third, and the corpus sizes for BERT,
where the lecturer corrects himself mid-sentence and the captions mangle the numbers he
lands on. Slide 27 is named in that last case as the figure of record.

One caption reading was resolved by date rather than by slide: at 1:14:49 the captions give
"people using GPT uh three four simple programming things". This lecture was delivered in
February 2023 and GPT-4 was not released until the following month, so this is "GPT-3 **for**
simple programming things", not a reference to GPT-4.

---

**[0:05]** Hello, welcome to CS224N. Today we'll be talking about pretraining, which is
another exciting topic on the road to modern natural language processing. Okay, how is
everyone doing? Thumbs up, some side, thumbs down — wow, no response bias there, all
thumbs up. Oh, sorry — nice, I like that honesty, that's good. Well, okay. So we're now
— what is this, week five? Yes, it's week five. And we have a couple — so this lecture,
the Transformers lecture, and then to a lesser extent Thursday's lecture on natural
language generation

**[0:51]** will be sort of the some of lectures for the assignments you have to do. Right?
So Assignment 5 is coming out on Thursday, and the topics covered in this lecture and the
self-attention and Transformers, and again a little bit of natural language generation,
will be tested in Assignment 5. And then the rest of the course will go through some
really fascinating topics in sort of modern natural language processing that should be
useful for your final projects, and future jobs, and interviews, and intellectual
curiosity. But I think that today's lecture is significantly less technical in detail
than last Thursday's on self-attention and Transformers, but should give you an idea

**[1:37]** of this sort of world of pretraining and how it helps define natural language
processing today. So, a reminder about Assignment 5: your project proposals also are due
on Tuesday, next Tuesday. Please do get those in, try to get them in on time so that we
can give you prompt feedback about your project proposals. And yeah, so let's jump into
it. Okay, so what we're going to start with today is a bit of a technical detail on word
structure and sort of how we model the input sequence of words that we get. So, when we
were teaching word2vec and sort of all the methods that

**[2:26]** we've talked about so far, we assumed a finite vocabulary. Right? So we had a
vocabulary V that you define via — whatever, you've looked at some data, you've decided
what the words are in that data. And so you have some words, like *hat* and *learn*, and
you have this embedding — it's in red because you've learned it properly. Actually,
let's replace *hat* and *learn* with *pizza* and *tasty*, those are better. And so that's
all well and good: you see these words in your model and you have an embedding that's
been learned on your data, to sort of know what to do when you see those words. But when
you see some sort of variations — maybe you see like *tasty*, and maybe a typo like
*learn* — or maybe novel items, where it's

**[3:13]** like a word that you as a human can understand as sort of this combination —
this is called derivational morphology — of like this word *Transformer*, that you know,
and *-ify*, which means take this noun and give me back a verb that means to make more
like that noun. To *Transformerify* NLP might mean to make NLP more like using
Transformers, and such. And for each of these, right, this maybe didn't show up in your
training corpus, and language is always doing this. People are always coming up with new
words, and there's new domains, and young people are always making new words — it's
great. And so it's a problem for your model, though, right? Because you've defined this
finite vocabulary, and there's sort of no mapping

**[3:59]** in that vocabulary for each of these things, even though their meanings should be
relatively well defined based on the data you've seen so far. It's just that the sort of
string of characters that define them aren't quite what you've seen. And so what do you
do? Well, maybe you map them to this sort of universal unknown token — this is UNK,
right. So it's like, oh, I see something, I don't know what — I've never seen it before,
I'm going to say it's always represented by the same token, UNK. And so that's been done
in the past, and that's sort of bad, right, because it's totally losing tons of
information. But you need to map it to something. And so this is like a clear problem —
especially, I mean, in English it's a problem; in many of the world's

**[4:45]** languages it's a substantially larger problem. Right? So English has relatively
simple word structure — there's a couple of conjugations for each verb, like *eat*,
*eats*, *eaten*, *ate*. But in a language with much more complex morphology, or word
structure, you'll have a considerably more complex set of things that you could see in
the world. So here is a conjugation table for a Swahili verb, and it has over 300
conjugations. And if I define the vocabulary to be "every unique string of characters
maps to its own word," then every one of the 300 conjugations would get an independent
vector under my model — which

**[5:32]** makes no sense, because the 300 conjugations obviously have a lot in common and
differ by sort of meaningful extents. So you don't want to do this. I'd have to have a
huge vocabulary if I wanted all conjugations to show up, and that's a mistake for
efficiency reasons and for learning reasons. Any questions so far? Cool. Okay. And so
what we end up doing is, we'll look at subword structure — subword modeling. So what
we're going to do is, we're going to say: I'm not going to even try to define what the
set of all words is. I'm going to define my vocabulary to include parts of

**[6:17]** words. There — where am I? Oh, right. So I'm going to split words into sequences
of known subwords. And so there's a simple sort of algorithm for this, where you start
with all characters — right, so if I only had a vocabulary of all characters and maybe
like an end-of-word symbol, for a finite data set, then no matter what word I saw in the
future, as long as I had seen all possible characters, I could take the word and say, I
don't know what this word is, I'm going to split it into all of its individual
characters. So you won't have this UNK problem, you can sort of represent any word. And
then you're going to find common adjacent characters and say, okay, *a* and *b* co-occur
next to

**[7:02]** each other quite a bit, so I'm going to add a new word to my vocabulary — now
it's all characters plus this new word *ab*, which is a subword. And likewise, so now I'm
going to replace the character pair with the new subword, and repeat until you add a lot,
a lot, a lot of vocabulary items through this process of what things tend to co-occur
next to each other. And so what you'll end up with is a vocabulary of very commonly
co-occurring sort of substrings, by which you can build up words. And this was originally
developed for machine translation, but then has been used considerably in pretty much all
modern language models. So now we have *hat* and *learn* — *hat* and *learn*. So in our
subword vocabulary, *hat* and *learn* showed up enough that they're their own

**[7:48]** individual words. So that's sort of good, right? Simple common words show up as a
word in your vocabulary, just like you'd like them to. But now *tasty* maybe gets split
into `taa` and then — in some cases this hash-hash (`##`) means don't add a space next,
right — so `taa`, and then `aaa`, and then `sty`. Right? So I've actually taken one sort
of thing that seems like a word, and in my vocabulary it's now split into three subword
tokens. So when I pass this to my Transformer, or to my recurrent neural network — the
recurrent neural network would take `taa` as just a single element, do the RNN update, and
then take `aaa`, do the RNN update, and then `sty`. So it could learn

**[8:36]** to process constructions like this, and maybe I can even add more *a*s in the
middle and have it do something similar, instead of just seeing the entire word *tasty*
and not knowing what it means. Is that — that's feedback? Yeah. How loud is that feedback,
we good? Okay, I think we're fixed, great. And so, same with *Transformer*: I'd have maybe
*Transformer* as its own word, and then *-ify*. And so you can see that you have sort of
three learned embeddings instead of one sort of useless UNK embedding. This is just wildly
useful, and is used pretty much everywhere — variants of this

**[9:22]** algorithm are used pretty much everywhere in modern NLP. Questions? Yes.
*[Student:] "If we have three embeddings for tasty, do we just add them together?"* So the
question is, if we have three embeddings for *tasty*, do we just add them together if we
want to represent — so, when we're actually processing the sequence, I'd see something
like "I learned about the `taa` `aaa` `sty`," so it'd actually be totally separate tokens.
But if I wanted to then say, what's my representation of this thing — depends on what you
want to do. Sometimes you average the contextual representations of the three, or look at
the last one. Maybe at that point it's unclear what to do, but

**[10:08]** everything sort of works okay. *[Student:] "How do you know where to split?"*
Yeah, so you know where to split based on the algorithm that I specified earlier for
learning the vocabulary. So you've learned this vocabulary by just combining commonly
co-occurring adjacent strings of letters — right, so like *ab* co-occurred a lot, so now
I've got a new word that's *ab*. And then, when I'm actually walking through and
tokenizing, I try to split as little as possible. So I split words into the maximal
subword that takes up the most characters. There are algorithms for this. Yeah, so like,
okay, if I want to split this up, there's many ways I could split it up, and you try to
find some approximate — like, what the best way to

**[10:54]** split it into the fewest words is. Yeah. *[Student:] "Do people make use of
punctuation in the character set? How do people do it?"* The question is, do people make
use of punctuation in the character set, how do people do it. Yes, absolutely. So, sort of
from this point on, just assume that what text is given to these models is as unprocessed
as possible. You take it, you try to make it sort of clean-looking text where you've
removed HTML tags maybe, if it's from the internet, or whatever, but then beyond that you
process it as little as possible, so that it reflects as well as possible what people
might actually be using this for. So maybe earlier in the course, when we were looking at
word2vec, maybe we

**[11:39]** might have thought about, oh, we don't want word vectors of punctuation, or
something like that. Now everything is just as close as possible to what the text you'd
get with people trying to use your system would be. So yes, in practice punctuation — and
like dot-dot-dot — might be its own word, and maybe a sequence of like hyphens, because
people make big bars across tables. Yeah. *[Student, opening words unintelligible: [Ed:
the captions render the start of this question as "foreign [Music]" and it is
unrecoverable; the audible remainder is] "…could be multiple embeddings versus a single
embedding — does the system treat those any differently?"]* The question is, does the
system treat any differently words that are

**[12:24]** really themselves the whole word, versus words that are sort of pieces? You
know, the system has no idea — they're all just indices into your embedding vocabulary
matrix. So they're all treated equally. *[Student:] "What about really long ones, that are
I guess relatively common — because if you're building up from the same character all the
way up, what happens then?"* Yeah, the question is what happens to very long words if
you're building up from character pairs and portions of characters. You know, in practice
the statistics speak really well for themselves. So if a long word is very common, it will
end up in the vocabulary, and if it's not very common, it won't. There are algorithms that
aren't this that do slightly better in various ways,

**[13:12]** but the intuition that you sort of figure out what the common co-occurring
substrings are, sort of independent of length almost, is the right intuition to have. And
so yeah, you can actually just look at the learned vocabularies of a lot of these models
and you see some long words, just because they showed up a lot. *[Student:] "I'm curious
how it weighs the frequency. So let's say there's like — if `ify`, or [Ed: one word here
is unrecoverable; the captions give "at the in your next slide it was like goodbye"] — at
the very last one. So `if` could be really common. So how does it weigh the frequency of a
subword versus the length of it? Like, it tries to split it up into the smallest number —
but what if it split it up into three, but one of*

**[13:59]** *them was super common?"* Yeah. So the question is, if *Transformer* is a
subword in my vocabulary, and *if* is a subword, and *y* is a subword, and *ify* as a
three-letter tuple is also a subword, how does it choose to take the — you know, if *ify*
maybe is not very common, as opposed to splitting it into more subwords. It's just a
choice. We choose to try to take the smallest number of subwords, because that tends to be
more of the bottleneck, as opposed to having a bunch of very common, very short subwords.
Sequence length is a big problem in Transformers, and this seems to be sort of what works
— although trying to split things into multiple options of a sequence and running the
Transformer on all of them

**[14:44]** is the thing that people have done, to see which one will work better. But
yeah, having fewer, bigger subwords tends to be the best sort of idea. I'm going to start
moving on, though — feel free to ask me more questions about this afterward. Okay. So
let's talk about pretraining from the context of the course so far. So, at the very
beginning of the course we gave you this quote, which was, "You shall know a word by the
company it keeps." This was the sort of thesis of the distributional hypothesis, right —
that the meaning of the word is defined by, or at least reflected by, what words it tends
to co-occur around. And we implemented this via word2vec. The same person who made that
quote had a separate quote, actually earlier,

**[15:29]** that continues this sort of notion of meaning as defined by context, which is
something along the lines of: well, since the word shows up in context when we actually
use it, when we speak to each other, the meaning of the word should be defined in the
context that it actually shows up in. And so, "the complete meaning of a word is always
contextual, and no study of meaning apart from a complete context can be taken seriously."
And so right, the big difference here is like — at word2vec training time, if I have the
word *record*, r-e-c-o-r-d, when I'm training word2vec I get one vector, or two, but one
vector meaning *record*, the string.

**[16:15]** And it has to learn, by what context it shows up in, that sometimes it can mean
*record*, i.e. the verb, or *record*, i.e. the noun. Right? But I only have one vector to
represent it. And so when I use the word embedding of *record*, it sort of has this
mixture meaning of both of its senses, right — it doesn't get to specialize and say, oh,
this part means *record* and this part means *record*. And so word2vec is going to just
sort of fail. And so I can build better representations of language through these
contextual representations that are going to take things like recurrent neural networks or
Transformers that we used before, to build up sort of contextual meaning.

**[17:02]** So what we had before were pretrained word embeddings, and then we had sort of
a big box on top of it, like a Transformer or an LSTM, that was not pretrained. Right? So
you learn via context your word embeddings here, and then you have a task like sentiment
analysis, or machine translation, or parsing, or whatever, and you initialize all the
parameters of this randomly, and then you train to predict your label. And the big
difference in today's work is that we're going to try to pretrain all the parameters. So I
have my big Transformer, and instead of just pretraining my word embeddings with word2vec,
I'm going to train all of the parameters of the

**[17:48]** network, trying to teach it much more about language that I could use in my
downstream tasks. So now, the labeled data that I have for, say, machine translation might
need to be smaller — I might not need as much of it, because I've already trained much
more of the network than I otherwise would have if I had just gotten sort of word2vec
embeddings. Okay. So here, right, I've pretrained this entire sort of structure — the word
embeddings, the Transformer on top, everything's been trained via methods that we'll talk
about today. And so what does this give you? I mean, it gives you very strong
representations of language. So the meaning of *record* and *record* will

**[18:35]** be different in the sort of contextual representations that know where in the
sequence it is and what words are co-occurring with it in the specific input, than
word2vec, which only has one representation for *record* independent of where it shows up.
It'll also be used as strong parameter initializations for NLP models. So in all of your
homework so far, you've worked with building out a natural language processing system sort
of from scratch, right? Like, how do I initialize this weight matrix? And we always say,
oh, small normally distributed noise, like little values close to zero. And here we're
going to say, well, just like we were going to use the word2vec embeddings and those sort
of encoded structure, I'm going to start maybe my machine

**[19:20]** translation system from a parameter initialization that's given to me via
pretraining. And then also it's going to give us probability distributions over language
that we can use to generate, and otherwise — and we'll talk about this. Okay. So whole
models are going to be pretrained. So all of pretraining is effectively going to be
centered around this idea of reconstructing the input. So you have an input, it's a
sequence of text that some human has generated, and the sort of hypothesis is that by
masking out part of it and tasking a neural network with reconstructing the original
input, that neural network has to learn a lot about language, about the world, in order to
do a good job of reconstructing the

**[20:07]** input. Right? So this is now a supervised learning problem, just like machine
translation. Right? I've taken this sentence that just existed — "Stanford University is
located in," say, "Palo Alto, California," or "Stanford, California," I guess — and I have,
by removing this part of the sentence, made a label for myself, right? The input is this
sort of broken, masked sentence, and the label is *Stanford* or *Palo Alto*. So if I give
this example to a network and ask it to predict the center thing, as it's doing its
gradient step on this input it's going to encode information about the co-occurrence
between this context "Stanford University is located in" and "Palo Alto." So by tasking it
with

**[20:54]** this, it might learn, say, where Stanford is. What else might it learn? Well,
it can learn things about maybe syntax. So, "I put blank fork down on the table." Here
there's only a certain set of words that could go here — "I put *the* fork down on the
table," "I put *a* fork down on the table." These are syntactic constraints, right? So the
context shows me sort of what kinds of words can appear in what kinds of contexts. "The
woman walked across the street, checking for traffic over blank shoulder" — any ideas on
what could go here? Right, so this sort of coreference between this entity who is being
discussed in the world, this woman, and her shoulder. Now, when I discuss — this is sort
of a

**[21:41]** linguistic concept — the word *her* here is a coreferent to *woman*, right,
it's referring to the same entity in the discourse. And so the network might be able to
learn things about what entities are doing what where. It can learn things about sort of
semantics: so if I have "I went to the ocean to see the fish, turtles, seals, and blank,"
then the word that's in the blank should be sort of a member of the class that I'm
thinking of, as a person writing this sentence, of stuff that I see when I go to the ocean
and see these other things as well. Right? So in order to do this prediction task, maybe I
learn about the semantics of aquatic creatures. Okay, so what else could I learn? I've got
"Overall, the value I got from the two

**[22:26]** hours watching it was the sum total of the popcorn and the drink. The movie was
blank." What kind of task could I be learning from doing this sort of prediction problem?
*[Student:] "Sentiment."* Sentiment, exactly. So this is just a naturalistic sort of text
that I naturally wrote myself, but by saying "oh, the movie was bad," I'm learning about
sort of the latent sentiment of the person who wrote this, what they were feeling about the
movie at the time. So maybe if I see a new review later on, I can just paste in the review,
say "the movie was blank," and if the model generates *bad* or *good*, that could be
implicitly solving the task of sentiment analysis.

**[23:13]** So here's another one. "Iroh went into the kitchen to make some tea. Standing
next to Iroh, Zuko pondered his destiny. Zuko left the blank." Okay, so in this scenario
we've got a world implicitly that's been designed by the person who is creating this text,
right? I've got physical locations in the discourse, like the kitchen, and I've got Zuko.
We've got Iroh's in the kitchen, Zuko's next to Iroh, so Zuko must be in the kitchen — so
what could Zuko leave but the kitchen? Right? And so in terms of latent notions of
embodiment and physical location, the way that people talk about people being next to
something and then leaving something could tell you stuff about sort of — yeah, a little

**[24:00]** bit about how the world works, even. So here's the secret sequence: "I was
thinking about the sequence that goes 1, 1, 2, 3, 5, 8, 13, 21, blank." And this is a
pretty tough one, right? This is the Fibonacci sequence, right? Create a model by looking
at a bunch of numbers from the Fibonacci sequence, learn to in general predict the next
one — a question you should be thinking about throughout the lecture. Okay, any questions
on these sort of examples of what you might learn from predicting the context? Okay, okay,
cool. So —

**[24:45]** a very simple way to think about pretraining is: pretraining is language
modeling. So we saw language modeling earlier in the course, and now we're just going to
say, instead of using my language model just to provide probabilities over the next word, I
am going to train it on that task. Right? I'm going to actually model the distribution
p_θ(w_t | w_{1:t−1}), the word *t* given all the words previous. And there's a ton of data
for this, right? There's just an amazing amount of data for this in a lot of languages,
especially English. There's very little data for this in actually most of the world's
languages, which is a separate problem. But you can pretrain just through language
modeling. Right? So I'm going to sort of do the teacher-forcing thing, so I have *Iroh*, I
predict *goes*; I have *goes*, I predict *to*. And I'm going to train my

**[25:32]** LSTM or my Transformer to do this task, and then I'm just going to keep all the
weights. Okay, I'm going to save all the network parameters. And then, once I have these
parameters, right, instead of generating from my language model, I'm just going to use them
as an initialization for my parameters. So I have this pretraining/fine-tuning paradigm,
two steps. Most of you, I think — in your, well, maybe not this year — let's say a large
portion of you this year in your final projects will be doing the pretraining/fine-tuning
sort of paradigm, where someone has done the pretraining for you. Right? So you have a ton
of text, you learn very general things about the distribution of words and sort of the
latent things that that tells you about the world and about language. And then in step two
you've got

**[26:19]** some task, maybe sentiment analysis, and you have maybe not very many labels,
you have a little bit of labeled data, and you adapt the pretrained model to the task that
you care about by further doing gradient steps on this task. So you give it "the movie
was," you predict happy or sad, and then you sort of continue to update the parameters
based on the initialization from the pretraining. And this just works exceptionally well. I
mean, unbelievably well, compared to training from scratch — intuitively, because you've
taken a lot of the burden of learning about language, learning about the world, off of the
data that you've labeled for sentiment analysis, and you're sort of giving that task of
learning all this sort of very general stuff to the much more general

**[27:05]** task of language modeling. Yes. *[Student:] "You said we didn't have much data
in other languages — what do you mean by that? Was it just text in that language, or
labeled in some way?"* So the question is, you said we have a lot of data in English but
not in other languages — what do you mean by data that we don't have a lot of in other
languages? Is it just text? It's literally just text, no annotations, because you don't
need annotations to do language model pretraining. Right? The existence of that sequence
of words that someone has written provides you with all these pairs of input and output —
input *Iroh*, output *goes*; input *Iroh goes*, output *to*. Those are all labels sort of
that you've constructed from the input just existing. But in most languages, even on the

**[27:52]** entire internet — I mean, there's about 7,000-ish languages on Earth, and most
of them don't have the sort of billions of words that you might want to train these
systems on. Yeah. *[Student:] "If you're pre-training the entire thing, are you still only
learning one vector representation per word?"* The question is, if you're pretraining the
entire thing, do you still learn one vector representation per word? You learn one vector
representation that is the non-contextual input vector, right. So you have your vocabulary
matrix, you've got your embedding matrix that is vocabulary size by model dimensionality.
And so yeah, *Iroh* has one vector, *goes* has one vector, but then the Transformer that
you're learning on top of it takes in the sequence so far and sort of gives a

**[28:38]** vector to each of them that's dependent on the context in that case. But still,
at the input, you only have one embedding per word. *[Student:] "Yeah — so what sort of
metric would you use to evaluate? It's supposed to be so general, right, but things like
application-specific metrics — which one do you use?"* Yeah, so the question is, what
metric do you use to evaluate pretrained models, since it's supposed to be so general? But
there are lots of sort of very specific evaluations you could use — we'll get into a lot of
that in the rest of the lecture. While you're training it you can use simple metrics that
sort of correlate with what you want but aren't actually what you want, just like the
probability quality, right? So you can evaluate the perplexity of your language model, just
like you would have when you cared about language modeling. And it turns out to be the case

**[29:25]** that better perplexity correlates with all the stuff that's much harder to
evaluate, like lots and lots of different tasks. But also the natural language processing
community has built very large sort of benchmark suites of varying tasks, to try to get at
sort of a notion of generality — although that's very, very difficult, it's sort of
ill-defined even. And so when you develop new pretraining methods, what you often do is you
try to pick a whole bunch of evaluations and show that you do better on all of them, and
that's your argument for generality. Okay. So, why should this sort of
pretraining/fine-tuning two-part paradigm help? This is still an open area of research, but
the intuitions are all you're going to take

**[30:11]** from this course. So right, pretraining provides some sort of starting
parameters θ̂ — so this is like all the parameters in your network, right — from trying to
do this minimum over all possible settings of your parameters of the pretraining loss. And
then the fine-tuning process takes your data for fine-tuning, you've got some labels, and it
tries to approximate the minimum, through gradient descent, of the loss of the fine-tuning
task of θ. But you start at θ̂ — right, so you start gradient descent at θ̂, which your
pretraining process gave you. And then, if you could actually solve this min, and wanted
to, it sort of feels like the starting point shouldn't matter. But it really, really,
really does. It

**[30:58]** really does. So that's — and we'll talk a bit more about this later — but the
process of gradient descent, maybe it sticks relatively close to the θ̂ during fine-tuning.
Right? So you start at θ̂ and then you sort of walk downhill with gradient descent until
you hit sort of a valley, and that valley ends up being really good because it's close to
the pretraining parameters, which were really good for a lot of things. This is a cool
place where sort of practice and theory are sort of meeting, where optimization people want
to understand why this is so useful, NLP people sort of just want to build better systems.
So yeah, maybe the stuff around θ̂ tends to generalize well. If you want to

**[31:44]** work on this kind of thing, you should talk about it. Yeah. *[Student:] "You
said gradient descent sticks relatively close — but what if we were to use a different
optimizer? How would that change the results?"* The question is, if stochastic gradient
descent sticks relatively close, what if we use a different optimizer? I mean, if we use
sort of any common variant of gradient descent, like any first-order method — like Adam,
which we use in this course, or AdaGrad — they all have these very, very similar
properties. Other types of optimization we just tend to not use, so who knows. Ah, yeah.
*[Student:] "Why does fine-tuning work better than just fine-tuning but making the model
bigger — like adding more layers, more data?"* Yeah, the

**[32:30]** question is, why does the pretrain/fine-tune paradigm work better than just
making the model more powerful, adding more layers, adding more data to just the
fine-tuning? That's a — you know, the simple answer is that you have orders of magnitude
more data that's unlabeled, that's just text that you found, than you do carefully labeled
data in the tasks that you care about. Right? Because that's expensive to get — it has to
be examples of your movie reviews or whatever, that you've had someone label carefully. So
you have something like, on the internet, at least five trillion, maybe 10 trillion words
of this, and you have maybe a million words of your labeled data

**[33:16]** or whatever over here. So it's just — the scale is way off. But there's also an
intuition that learning to do a very, very simple thing like sentiment analysis is not
going to get you a very generally able agent in a wide range of settings, compared to
language modeling. So, it's hard to — how to put it — even if you have a lot of labeled
data of movie reviews of the kind that people are writing today, maybe tomorrow they start
writing slightly different kinds of movie reviews and your system doesn't perform as well.
Whereas if you pretrained on a really diverse set of text from a wide range of sources and
people, it might be more adaptable to seeing stuff that

**[34:03]** doesn't quite look like the training data you showed it, even if you showed it a
ton of training data. So one of the sort of big takeaways of pretraining is that you get
this huge amount of sort of variety of text on the internet. You have to be very careful —
I mean, you should be very careful about what kind of text you're showing it and what kind
of text you're not, because the internet is full of awful text as well. But some of that
generality just comes from how hard this problem is and how much data you can show it.
*[Student:] "So much data — how do you then train it so that it considers the stuff that
you're fine-tuning it with as more important, more salient [Ed: the captions give "more
sale into a passive Trend", which is unrecoverable; the sense of the question is clear from
the answer], rather than just one in a billion*

**[34:48]** *articles?"* Yeah, it's a good question. So the question is, given that the
amount of data on the pretraining side is orders of magnitude more than the amount of data
on the fine-tuning side, how do you sort of get across to the model that, okay, actually
the fine-tuning task is what I care about, so focus on that? It's about the fact that I did
this first — the pretraining first — and then I do the fine-tuning second. Right? So I've
gotten my parameter initialization from this, I've set it somewhere, and then I fine-tune,
I move to where the parameters are doing well for this task afterward. And so, well, it
might just forget a lot about how to do this, because now I'm just asking it to do this at
this point. I should move on, I think, but we're going to keep talking about

**[35:34]** this in much more detail with more concrete elements. So, okay. So let's talk
about model pretraining — oh wait, that did not advance the slides. Nice. Okay, let's talk
about model pretraining three ways. In our Transformers lecture Tuesday we talked about
encoders, encoder-decoders and decoders, and we'll do decoders last because actually many
of the largest models that are being used today are all decoders, and so we'll have a bit
more to say about them. Right, so let's recall these three. So encoders get bidirectional
context: you

**[36:21]** have a single sequence and you're able to see the whole thing, kind of like an
encoder in machine translation. Encoder-decoders have one portion of the network that gets
bidirectional context — so that's like the source sentence of my machine translation system
— and then they're sort of paired with a decoder that gets unidirectional context, so that
I have this sort of informational masking where I can't see the future, so that I can do
things like language modeling, I can generate the next token of my translation, whatever.
So you could think of it as, I've got my source sentence here and my partial translation
here, and I'm sort of decoding out the translation. And then decoders only are things like
language models — we've seen a lot of this so far. And there's pretraining for all three
sort of large classes of models,

**[37:08]** and how you pretrain them and then how you use them depends on the properties
and the productivities of the specific architecture. So let's look at encoders first. So
we've looked at language modeling quite a bit, but we can't do language modeling with an
encoder, because they get bidirectional context. Right? So if I'm down here at *I* and I
want to predict the next word, it's a trivial task at this level here to predict the next
word, because in the middle I was able to look at the next word. And so I should just know
— there's nothing hard about learning to predict the next word here, because I could just
look at it, see what it is, and then copy it over. So when I'm training an encoder for

**[37:54]** pretraining, I have to be a little bit more clever. In practice what I do is
something like this: I take the input and I modify it somewhat. I mask out words, sort of
like I did in the examples I gave at the beginning of class. "I blank to the blank." Right?
And then I have the network predict — with this whole, you know, I haven't built contextual
representations — so now this vector representation of the blank sees the entire context
around it here, and then I predict the word *went*, and then here the word *store*. Any
questions? Okay. And you can see how this is doing something quite a bit like language
modeling, but with

**[38:39]** bidirectional context. I've removed the network's information about the words
that go in the blanks, and I'm training it to reconstruct that. So I only have loss terms —
right, I only ask it to actually do the prediction, compute the loss, backpropagate the
gradients, for the words that I've masked out. And you can think of this as, instead of
learning probability of *x*, where *x* is like a sentence or a document, this is learning
the probability of *x*, the real document, given *x̃*, which is this sort of corrupted
document with some of the information missing. Okay. And so maybe we get the sequence of
vectors here, one per word, which is the output of my encoder in blue. And then I'd say
that for the words that I want to predict, y_i, I draw them — this is the

**[39:26]** ∼, means the probability is proportional to my embedding matrix times my
representation of it. So it's a linear transformation of that last thing here, so this
*A* plus *b* is this red portion here, and then do the prediction. And I train the entire
network to do this. Yes. *[Student:] "So far, do we just do it as we are, or is there
something you can do?"* The question is, do we just choose words randomly to mask out, or
is there a scheme? Mostly randomly — we'll talk about a slightly smarter scheme in a couple
of slides, but yeah, just mostly randomly. Yeah. *[Student:] "What was that last part on
the bottom?"* Um, the masked version of, like, if it's

**[40:12]** the first or the very last sentence — yeah. So, so I'm saying that I'm defining
*x̃* to be this input part where I've got the masked version of the sentence with these sort
of words missing, and then I'm defining a probability distribution that's the probability of
a sequence conditioned on the input being the sort of corrupted sequence, the masked
sequence. Okay. So this brings us to a very, very popular sort of NLP model that you need to
know about. It's called BERT, and it was the first one to popularize this masked language
modeling objective. And they released the weights of this

**[40:57]** pretrained Transformer that they pretrained via something that looks a lot like
masked language modeling. And so these you can download, you can use them via code that's
released by the company Hugging Face, that we continue to bring up. Many of you will use a
model like BERT in your final project, because it's such a useful builder of
representations of language and context. So let's talk a little bit about the details of
masked language modeling in BERT. First, we'd take 15% of the subword tokens — so remember,
all of our inputs now are subword tokens. I've made them all look like words, but just like
we saw at the very beginning of class, each of these tokens could just be some portion, some
subword. And I'm going to do a couple of things with it. Sometimes I am going to just mask
out the word

**[41:45]** and then predict the true word. Sometimes I'm going to replace the word with
some random sample of another word from my vocabulary and predict the real word that was
supposed to go there. And sometimes I'm going to not change the word at all and still
predict it. The intuition of this is the following. If I just had to build good
representations, in the sort of middle of this network, for words that are masked out, then
when I actually use the model at test time on some real review to do sentiment analysis on,
well, there are never going to be any tokens like this, so maybe the model won't do a very
good job — because it's like, oh, I have no job to do here, because I only need to deal with
the mask tokens.

**[42:33]** By giving it sequences of words where sometimes it's the real word that needs to
be predicted, sometimes you have to detect if the word is wrong, the idea is that now, when I
give it a sentence that doesn't have any masks, it actually sort of does a good job of
representing all the words in context, because it has this chance that it could be asked to
predict anything at any time. Okay. So the folks at Google who were defining this had a
separate additional task that is sort of interesting to think about. So this was their BERT
model from their paper. They had their position embeddings, just like we saw from our
Transformers lecture; token embeddings

**[43:19]** just like we saw from the Transformers lecture; but then also they had this
thing called a segment embedding, where they had two possible segments, segment A and
segment B. And they had this additional task where they would get a big chunk of text for
segment A and a big chunk of text for segment B, and then they would ask the model, is
segment B a real continuation of segment A — is it the text that actually came next — or did
I just pick this big segment randomly from somewhere else? And the idea is that this should
teach the network some notion of sort of long-distance coherence, right, about the
connection between a bunch of text over here and a bunch of text over there. Turns out it's
not really necessary, but it's an interesting idea,

**[44:04]** and sort of similar things have continued to have some sort of influence since
then. But again, you should get this intuition that we're trying to come up with hard
problems for the network to solve, such that by solving them it has to learn a lot about
language, and we're defining those problems by making simple transformations, or removing
information from text that just happens to occur. Questions? Yeah. *[Student:] "For the plus
signs, do we concatenate the vectors, or do we do an element-wise addition?"* The question
is, for these plus signs, do we concatenate the vectors or do element-wise addition? We do
element-wise addition. You could have concatenated them. However, one of the big sort of

**[44:50]** conventions of all these networks is that you always have exactly the same number
of dimensions everywhere, at every layer of the network — it just makes everything very
simple. So just saying everything's the same dimension and then doing addition just ends up
being simpler. Yeah. *[Student:] "Why was the next sentence prediction not necessary?"* Yeah,
why was the next sentence prediction not necessary. I mean, one thing that it does that's a
negative is that now the effective context length for a lot of your examples is halved. So
one of the things that's useful about pretraining, seemingly, is that you get to build
representations of very long sequences of text. So this is very short, but in practice
segment A was going to

**[45:37]** be something like 250 words and segment B was going to be 250 words. And in the
paper that sort of let us know that this wasn't necessary, they always had a long segment of
500 words, and it seemed to be useful to always have this very long context, because longer
contexts help give you more information about the role that each word is playing in that
specific context. Right? If I see one word, it's hard to know if it's just *the record*,
it's hard to know what it's supposed to mean, but if I see a thousand words around it it's
much clearer what its role in that context is. So yeah, it cuts the effective context size,
as one answer. Another thing is that this is actually much more difficult — this is a much
more recent paper that I don't

**[46:22]** have in the slides, but it's been shown since then that these models are really,
really bad at the next sentence prediction task. So it could be that maybe it just was too
hard at the time, and so it just wasn't useful, because the model was failing to do it at
all. So I'll give the link for that paper later. *[Student:] "Why do we need to do next
sentence prediction? What about just masking and predicting? I missed that jump."* So yeah,
the question is, why do we need to do next sentence prediction, why not just do the masking?
We saw before that the thing you seem to not need to do is next sentence prediction. But,
sort of like, as history of the research, it was thought that this was useful, and the idea
is that it required you to

**[47:08]** develop this sort of pairwise, like, do these two segments of text interact, how
do they interact, are they related — the sort of longer-distance notion. And many NLP tasks
are defined on pairs of things, and they thought that might be useful. And so they published
it with this, and then someone else came through, published a new model that didn't do that,
and it sort of did better. So, you know, this is just — yeah. So yeah, there are intuitions
as to why it could work, it just didn't. It was doing both — it was doing both this next
sentence prediction training as well as this masking training, all at the same time. And so
you had to have a separate

**[47:53]** predictor head on top of BERT, a separate predictor sort of classification thing.
And so one detail there is that there's this special word at the beginning of BERT, in every
sequence, that's [CLS], and you can define a predictor on top of that sort of fake word
embedding that was going to say, is the next sentence real or fake or not. Yeah. Okay, I'm
gonna move on. And so this gets at sort of the question that we had earlier about how do you
evaluate these things. There's a lot of different NLP tasks out there, gosh. And when people
were defining these papers, they would look at a ton of different evaluations that had been
sort of compiled as a set of things that are still hard for today's systems. So, are

**[48:39]** you detecting paraphrases between questions — are two Quora questions actually
the same question? That turns out to be hard. Can you do sentiment analysis on this hard
data set? Can you tell if sentences are linguistically acceptable, are they grammatical or
not? Are two sequences similar semantically, do they mean sort of vaguely the similar thing?
And we'll talk a bit about natural language inference later, but that's the task of defining
— sort of, if I say "I saw the dog," that does not necessarily mean "I saw the little dog,"
but saying "I saw the little dog" does mean "I saw the dog." So that's the natural language
inference task. And, you know, the striking — the difference between sort of pre-pretraining
days,

**[49:26]** where you had this row here, before you had substantial amounts of pretraining,
and BERT — it was just like, the field was taken aback in a way that's hard to describe. You
know, very carefully crafted architectures for each individual task, where everyone was
designing their own neural network and doing things that they thought were sort of clever as
to how to define all the connections and the weights and whatever, to do their tasks
independently. So everyone was doing a different thing for each one of these tasks. Roughly
all of that was blown out of the water by "just build a big Transformer and just teach it to
predict the missing words a whole bunch, and then fine-tune it on each of these tasks." So
this was just a sea change in the field. People were, I mean, amazed —

**[50:12]** it's a little bit less flashy than ChatGPT, I'll admit, but it's really part of
the story that gets us to it. Okay, questions? *[Student:] "So, to get stuff out of the
encoder — during the pretraining stage the encoder usually outputs some sort of hidden
values. How do we correlate those to the words that we are trying to test against?"* So the
question is, the encoder output is a bunch of hidden values — how do we actually correlate
those values to stuff that we want to predict? I'm going to go on to the next slide here, to
bring up this example here. Right, so the encoder gives us, for each

**[50:58]** input word token, a vector of that token that represents the token in context.
And the question is, how do we get these representations and turn them into sort of answers
for the tasks that we care about? And the answer comes back to doing something like this.
Something like this — maybe. Wow. Sure. So, when we were doing pretraining, right, we had the
Transformer that was giving us our representations, and we had

**[51:44]** this little last layer here, this little sort of affine transformation that moved
us from the encoder's hidden-state size to the vocabulary, to do our prediction. And we just
remove this last prediction layer here. And let's say we want to do something that is
classifying the sentiment of the sentence: we just pick, arbitrarily, maybe the last word in
the sentence, and we stick a linear classifier on top and map it to positive or negative, and
then fine-tune the whole thing. Okay. So yeah, the BERT model had two different models — one
was 110 million parameters, one was 340 million. Keep that sort of in the back of your head,
sort of percolating, as we talk about models with many, many more parameters later on.

**[52:30]** It was trained on 800 million words, plus — that is definitely wrong, maybe 2.5,
maybe 25 million words [Ed: the lecturer is correcting himself against slide 27 and the
captions garble the figures he lands on. The slide reads BooksCorpus, 800 million words, and
English Wikipedia, 2,500 million words] — but on the order of less than a billion words of
text, quite a bit still. And it was trained on what was considered at the time to be a whole
lot of compute — it was Google doing this, and they released it, and we were like, oh, who
has that kind of compute but Google. Although nowadays it's not considered to be very much.
But fine-tuning is practical and common on a single GPU, so you could take the BERT model
that they'd spent a lot of time training and fine-tune it yourself on your task, on even a
very, very sort of small GPU.

**[53:16]** Okay. So one question is, like, well, this seems really great, why don't we just
use this for everything? Uh-huh, yeah. And the answer is, well, what is the sort of
pretraining objective, what's the structure of the pretrained model good for? BERT is really
good for sort of filling in the blanks, but it's much less naturally used for actually
generating text. Right? So I wouldn't want to use it to generate a summary of something,
because it's not really built for it — it doesn't have a natural notion of predicting the
next word given all the words that came before it. So maybe I want to use BERT if I want a
good representation of, say, a document, to

**[54:02]** classify it, give it one of a set of topic labels, or say it's toxic or non-toxic
or whatever. But I wouldn't want to use it to generate a whole sequence. Okay, some
extensions of BERT. So we had a question earlier of whether you just mask things out
randomly. One thing that seems to work better is, you mask out sort of whole contiguous
spans, because the difficulty of this problem is much easier than it would otherwise be —
because this is part of *irresistibly*, and you can tell very easily based on the subwords
that came before it. Whereas if I have a much longer sequence — it's a trade-off, but this
might be a harder problem, and it ends up being better to

**[54:49]** do this sort of span-based masking than random masking. And that might be because
subwords make very simple prediction problems when you mask out just one subword of a word
versus all the subwords of a word. Okay, so this ends up doing much better. There's also a
paper called the RoBERTa paper, which showed that the next sentence prediction wasn't
necessary. They also showed that they really should have trained it on a lot more text. So
RoBERTa is a drop-in replacement for BERT, so if you're thinking of using BERT, just use
RoBERTa, it's better. And it gave us this intuition that we really don't know a whole lot
about the best practices for training these things — you sort of train it for as long as
you're willing to, and things do good stuff, and whatever. So this is very — but it's very

**[55:35]** difficult to do sort of iteration on these models, because they're big, it's
expensive to train them. Another thing that you should know for your final projects, and in
the world ahead, is this notion of fine-tuning all parameters of the network versus just a
couple of them. So what we've talked about so far is, you pretrain all the parameters and
then you fine-tune all of them as well, so all the parameter values change. An alternative,
which we call parameter-efficient or lightweight fine-tuning, is you sort of choose little
bits of parameters, or you choose a very smart way of keeping most of the parameters fixed
and only fine-tuning others. And the intuition is that these pretrained parameters were
really good, and you want to make the minimal change from the pretrained model to the model
that does what you want, so that you keep

**[56:22]** some of the generality, some of the goodness, of the pretraining. So one way that
this is done is called prefix tuning — prompt tuning is very similar — where you actually
freeze all the parameters of the network. So I've pretrained my network here, and I've never
changed any of the parameter values. Instead I make a bunch of fake sort of pseudo-word
vectors that I prepend to the very beginning of the sequence, and I train just them. Sort of
unintuitive — it's like, these would have been inputs to the network, but I'm specifying them
as parameters, and I'm training everything to do my sentiment analysis task just by changing
the values of these sort of fake words. And this is nice because I get to keep all the good
pretrained parameters

**[57:08]** and then just specify this sort of diff that ends up generalizing better. This is
a very open field of research. But this is also cheaper, because I don't have to compute the
gradients — or I don't have to store the gradients and all the optimizer state with respect
to all these parameters — I'm only training a very small number of parameters. Yeah.
*[Student:] "It's like fake parameters — and as if like here, but does it make any
difference whether they're at the end or the beginning?"* In a decoder you have to put them
at the beginning, because otherwise you don't see them before you process the whole sequence.
Yeah. *[Student:] "What about a few layers — could I only train new layers?"* The question is,
can we just attach

**[57:55]** new layers on sort of the top of this and only train those? Absolutely, this works
a bit better. Another thing that works well — sorry, we're running out of time — is taking
each weight matrix. So I have a bunch of weight matrices in my Transformer, and I freeze the
weight matrix and learn a very low-rank little diff, and I set the weight matrix's value to
be sort of the original value plus my very low-rank diff from the original one. And this ends
up being a very similarly useful technique, and the overall idea here is that again I'm
learning way fewer parameters than I did via pretraining, and freezing most of the
pretraining parameters. Okay, encoder-decoders. So for encoder-

**[58:42]** decoders, we could do something like language modeling. Right? I've got my input
sequence here, encoder, output sequence here, and I could say this part is my prefix, for
sort of having bidirectional context, and I could then predict all the words that are sort of
in the latter half of the sequence, just like a language model. And that would work fine. And
so this is something that you could do, right — you sort of take a long text, split it into
two, give half of it to the encoder and then generate the second half with the decoder. But
in practice what works much better is this notion of span corruption. Span corruption is
going to show up in your Assignment 5, and the idea here is a lot like BERT, but in a sort of

**[59:28]** generative sense, where I'm going to mask out a bunch of words in the input:
"Thank you `<X>` me to your party `<Y>` week." And then at the output I generate the mask
token and then what was supposed to be there, that the mask token replaced. Right? So "thank
you," then predict "for inviting" at the output; "me to your party," "last," "week." And what
this does is that it allows you to have bidirectional context — right, I get to see the whole
sequence, except I can generate the parts that were missing. So this feels a little bit like
you mask out parts of the input, but you actually generate the output as a sequence like you
would in language modeling. So this

**[1:00:15]** might be good for something like machine translation, where I have an input that
I want bidirectional context in, but then I want to generate an output, and I want to pretrain
the whole thing. So this was shown to work better than language modeling at the scales that
these folks at Google were able to test back in 2018. This is still quite, quite popular.
Yeah, there's a lot of numbers — it works better than the other stuff, I'm not going to worry
about it. There's a fascinating property of these models also. So T5 was the model that was
originally introduced with salient span masking, and you can think of, at pretraining time you
saw a bunch of things like "Franklin D. Roosevelt was born in" blank, and you generated

**[1:01:01]** out the blank. And there's this task called open-domain question answering, which
has a bunch of trivia questions like "when was Franklin D. Roosevelt born," and then you're
supposed to generate out the answer as a string, just from your parameters. Right? So you did
a bunch of pretraining, you saw a bunch of text, and then you're supposed to generate these
answers. And what's fascinating is that this sort of salient span masking method allowed you
to pretrain and then fine-tune on some examples of trivia questions, and then when you tested
on new trivia questions, the model would sort of implicitly extract from its pretraining data,
somehow, the answer to that new

**[1:01:46]** question that it never saw explicitly at fine-tuning time. So it learned this sort
of implicit retrieval — sometimes; sometimes less than 50% of the time or whatever, but much
more than random chance. Yeah. And that's just sort of fascinating, right? So you've sort of
learned to access this sort of latent knowledge that you stored up by pretraining. And so
yeah, you just pass it the text "when was Roosevelt born" and it would pass out an answer. And
one thing to know is that the answers always look very fluent, they always look very
reasonable, but they're frequently wrong. And that's still true of things like ChatGPT. Yeah.
Okay, so that's encoder-decoder models. Next up we've got decoders, and we'll

**[1:02:32]** spend a long time on decoders. So this is just our normal language model. So I
get a sequence of hidden states for my decoder — the models where words can only look at
themselves, not the future — and then I predict the next word in the sentence. And then here
again I can, to do sentiment analysis, maybe take the last state for the last word and then
predict happy or sad based on that last embedding, backpropagate the gradients through the
whole network, train the whole thing, or do some kind of lightweight or parameter-efficient
fine-tuning like we mentioned earlier. So this is our pretraining a decoder. And I can just
pretrain it on language modeling. So again, you might want to do this if you are wanting to
generate

**[1:03:18]** — generate text, generate things. You sort of can use this like you use an
encoder-decoder, but in practice, as we'll see, a lot of the sort of biggest, most powerful
pretrained models tend to be decoder-only. It's not really clear exactly why, except they seem
a little bit simpler than encoder-decoders, and you get to share all the parameters in one big
network for the decoder, whereas in an encoder-decoder you have to split them sort of some into
the encoder, some into the decoder. So for the rest of this lecture we'll talk only about
decoders. So even in modern things, the biggest networks do tend to be decoders. So we're
coming all the way back again to 2018, and the GPT model from OpenAI

**[1:04:06]** was a big success. It had 117 million parameters. It had 768-dimensional hidden
states, and it had this vocabulary that was 40,000-ish words, that was defined via a method
like what we showed at the beginning of class. Trained on BooksCorpus. And actually, "GPT"
never actually showed up in the original paper — it's unclear what exactly it's supposed to
refer to. But this model was a precursor to all the things that you're hearing about nowadays.
If you move forward — oh yeah, so if you

**[1:04:52]** — so if we wanted to do something like natural language inference, right, which
says, take these pairs of sentences, "the man is in the doorway," "the person is near the
door," and say that these mean that one entails the other, the sort of premise entails the
hypothesis — that I can believe the hypothesis if I believe the premise. I just sort of
concatenate them together, right? So give it maybe a [START] token, pass in one sentence, pass
in some [DELIM] delimiter token, pass in the other, and then predict sort of yes/no,
entailment/not entailment. Fine-tuning GPT on this, it worked really well. And then BERT came
after GPT — BERT did a bit better, it had bidirectional context.

**[1:05:39]** But it did an excellent job. And then came GPT-2, where they focused more on the
generative abilities of the network. So right, we looked at now a much larger network — we've
gone from 117 million to 1.5 billion — and given some sort of prompt, it could generate, at the
time, a quite surprisingly coherent continuation to the prompt. So it's telling this sort of
story about scientists and unicorns here. And this size of model is still sort of small enough
that you can use it on a small GPU and fine-tune and whatever, and its capabilities of
generating long, coherent texts was just sort of

**[1:06:24]** exceptional at the time. It was also trained on more data, although I don't —
something like 9 billion words of text. And then, so after GPT-2 we come to GPT-3 — sort of
walking through these models — and then we come with a different way of interacting with the
models. So we've interacted with pretrained models in two ways so far: we've sort of sampled
from the distribution that they define, we've generated text via like a machine translation
system or whatever; or you fine-tuned them on a task that we care about and then we take their
predictions. But GPT-3 seems to have an interesting new ability. It's much larger, and it

**[1:07:12]** can do some tasks without any sort of fine-tuning whatsoever. GPT-3 is much
larger than GPT-2 — right, so we went from GPT, 100-ish million parameters, GPT-2 1.5 billion,
GPT-3 175 billion, much larger. Trained on 300 billion words of text. And this sort of notion
of in-context learning — that it could define, or figure out, patterns in the example that
it's currently seeing and continue the pattern — is called in-context learning. So you've got
the word *thanks* and I pass in this little arrow and say, okay, *thanks* goes to *merci*, and
then *hello* goes to *bonjour*, and then they give it all of these examples and ask it what
*otter* should go to. And

**[1:07:57]** it's learned to sort of continue the pattern and say that this is the translation
of *otter*. So now, remember, this is a single sort of input that I've given to my model, and I
haven't said "oh, do translation," or fine-tuned it on translation or whatever. I've just
passed in the input, given it some examples, and then it is able to, to some extent, do this
seemingly complex task. That's in-context learning. And here are more examples: maybe you give
it examples of addition and then it can do some simple addition afterward; you give it — in
this case this is sort of rewriting typos — it can figure out how to rewrite typos; in-context
learning for machine translation. And this was the

**[1:08:42]** start of this idea that there were these emergent properties that showed up in
much larger models, and it wasn't clear, when looking at the smaller models, that you'd get
this qualitatively new behavior out of them. Right? Like, it's not obvious from just the
language modeling signal — right, GPT-3 is just trained on that decoder-only, just predict the
next word — that it would, as a result of that training, learn to perform seemingly quite
complex things as a function of its context. Yeah. Okay, one or two questions about that. This
should be quite surprising, I

**[1:09:28]** think, right? Like, so far we've talked about good representations, contextual
representations, meanings of words in context. This is some very, very high-level pattern
matching, right? It's coming up with patterns in just the input data, in that one sequence of
text that you've passed it so far, and it's able to sort of identify how to complete the
pattern. And as you think, what kinds of things can this solve, what are its capabilities,
what are its limitations — this ends up being an open area of research. Sort of, what are the
kinds of problems that you maybe saw in the training data? Like, maybe GPT-3 saw a ton of pairs
of words, right? It saw a bunch of dictionaries, bilingual dictionaries, in its training data,
so it learned to do something like this. Or is it doing something much more general, where it's
really learning the task in

**[1:10:13]** context? You know, the actual story, we're not totally sure — something in the
middle, it seems like. It has to be tied to your training data in ways that we don't quite
understand, but there's also a non-trivial ability to learn new sort of — at least types of
patterns — just from the context. So this is a very interesting thing to work on. Now, we've
talked a lot about the size of these models so far, and as models have gotten larger they've
always gotten better; we train them on more data. So right, GPT-3 was trained on 300 billion
words of text and it was 175 billion parameters. And at that scale it costs a lot of money to
build these things, and it's very unclear whether you're getting the best use out of your
money. Like, is bigger

**[1:10:59]** really what you should have been doing, in terms of the number of parameters? So
the cost of training one of these is roughly, you take the number of parameters, you multiply
it by the number of tokens that you're going to train it on, the number of words. And some
folks at DeepMind — I forgot the citation on this — some folks at DeepMind realized, through
some experimentation, that actually GPT-3 was just comically oversized. Right? So Chinchilla,
the model they trained, is less than half the size and works better, but they just trained it
on way more data. And this is sort of an interesting trade-off about how do you best spend your
compute. I mean, you can't do this more than a handful of times, even if you're Google. So,
open questions there

**[1:11:46]** as well. Another sort of way of interacting with these networks that has come out
recently is called chain-of-thought. So the prefix — we saw in the in-context learning slide
that the prefix can help sort of specify what task you're trying to solve right now — and it
can do even more. So here's standard sort of prompting: we have a prefix of examples of
questions and answers, so you have a question and then an example answer, so that's your prompt
that's specifying the task, and then you have a new question and you're having the model
generate an answer, and it generates it wrong. And chain-of-thought prompting says, well, how
about, in the demonstration we give, we give the question and then we give this sort of

**[1:12:31]** decomposition of steps towards how to get an answer. Right? So I'm actually
writing this out as part of the input — I'm giving annotations, as a human, to say, oh, to
solve this sort of word problem, here's how you could think it through-ish. And then I give it
a new question, and the model says, oh, I know what I'm supposed to do, I'm supposed to first
generate a sequence of intermediate steps and then next say "the answer is" and then say what
the answer is. And it turns out — and this should again be very surprising — that the model can
tend to generate plausible sequences of steps and then much more frequently generates the
correct answer after doing so, relative to trying to generate the answer by itself.

**[1:13:17]** So you can think of this as a scratch pad. You can think of this as increasing the
amount of computation that you're putting into trying to solve the problem — sort of writing out
your thoughts, right? As I generate each word of this continuation here, I'm able to condition
on all the past words so far, and so maybe it just — yeah, allows the network to sort of
decompose the problem into smaller, simpler problems, which it's more able to solve, each. No
one's really sure why this works exactly either. At this point, with networks that are this
large, their emergent properties are both very powerful and exceptionally hard to understand,
and very hard, you should think, to trust,

**[1:14:03]** because it's unclear what its capabilities are and what its limitations are, where
it will fail. So, what do we think pretraining is teaching? Gosh, a wide range of things, even
beyond what I've written in this slide, which I mostly wrote two years ago. Right? So it can
teach you trivia, and syntax, and coreference, and maybe some lexical semantics, and sentiment,
and some reasoning — like way more reasoning than we would have thought even three years ago.
And yet they also learn and exacerbate racism and sexism, all manner of biases. There's more on
this later. But the generality of this is really, I think, what's taken many people aback. And
so increasingly these objects are not just

**[1:14:49]** studied for the sake of using them but studied for the sake of understanding
anything about how they work and how they fail. Yeah, any questions? *[Student:] "Has anyone
tried benchmarking GPT for programming tasks — how accurate it is, etc.?"* Yeah, the question is,
has anyone tried benchmarking GPT for programming tasks, anyone seen how well it does. Yes, so
there's definitely examples of people using GPT-3 for simple programming things. And
then the modern state-of-the-art competitive programming bots are all based on ideas

**[1:15:34]** from language modeling, and I think they're all also based on pretrained language
models themselves. Like, if you just take all of these ideas and apply it to GitHub, then you
get some very interesting emergent behaviors relating to code. And so yeah, I think all of the
best systems use this, more or less, so lots of benchmarking there for sure. *[Student:] "Is
that the basis for what GitHub Copilot is doing?"* The question is, is this — is that what we
just mentioned — the basis for the GitHub Copilot system? Yes, absolutely. We don't know exactly
what it is in terms of details, but it's all these ideas. *[Student:] "What if you have a
situation where you have still a large amount of data for general data, and then*

**[1:16:21]** *you have also a large amount of data for your fine-tuning task — at what point is
it better to train a new model for that, versus getting data from both?"* Yeah, the question is,
what if you have a large amount of data for pretraining and a large amount of data for
fine-tuning — when is it better to do sort of a separate training on just the fine-tuning data?
Almost never. If you have a bunch of data for the task that you care about, what's frequently
done instead is three-part training, where you pretrain on a very broad corpus, then you sort of
continue to pretrain using something like language modeling on an unlabeled version of the
labeled data that you have — you just strip the labels off and just treat it all as text and do
language modeling

**[1:17:06]** on that, adapt the parameters a little bit — and then do the final stage of
fine-tuning with the labels that you want. And that works even better. There's an interesting
paper called "Don't Stop Pretraining." Nice. Final question — that's a lot of questions — anyone
new, someone new? The question, yeah. *[Student:] "I was wondering, do you know if there's a lot
of instances where a pretrained model can do some task it's not seen before?"* Yeah, so, are
there any instances where a pretrained model can do a task that it hasn't seen before, without
fine-tuning? The question is, what does "hasn't seen before" mean, right? Like, these models,
especially GPT-3 and similar

**[1:17:53]** very large models — during pretraining, did it ever see something exactly like this
sort of word-problem arithmetic? Maybe, maybe not, it's actually sort of unclear. It's clearly
able to recombine sort of bits and pieces of tasks that it saw implicitly during pretraining. We
saw the same thing with trivia, right? Like, language modeling looks a lot like trivia
sometimes, where you just read the first paragraph of a Wikipedia page and it's kind of like
answering a bunch of little trivia questions about where someone was born and when. But it's
never seen something quite like this, and it's actually still kind of astounding how much it's
able to do things that don't seem like they should have shown up all that directly in the
pretraining data. Quantifying that extent is an open research problem. Okay, that's it, let's
call it.
