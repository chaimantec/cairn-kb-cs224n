---
title: NLP, Linguistics, Philosophy
lecture: 18
video: https://www.youtube.com/watch?v=NxH0Y78xcF4
source: edited from auto-generated YouTube transcript
verbatim: false
original: original/18-nlp-linguistics-philosophy.md
slides: ../slides/18-nlp-linguistics-philosophy.md
---

# NLP, Linguistics, Philosophy — transcript

Copy-edited for punctuation, sentence boundaries, and mis-heard terms; cross-checked against
`raw/slides/18-nlp-linguistics-philosophy.md`. The verbatim auto-generated captions are kept at
`raw/transcripts/original/18-nlp-linguistics-philosophy.md`. Lecturer is Christopher Manning. This
is his last regular lecture of the course, delivered as a closing monologue; the captions contain
no audience interjections anywhere in the hour, so no floor questions or comments needed marking —
per this KB's convention, any such comments from students would be set in *italics*. Timestamps
mark the start of each paragraph; all 98 are preserved in order.

**This is an edited transcript.** The captions had no punctuation and badly mangled the
philosophers, linguists, and citations this lecture is built on. Restorations, checked against the
slides: *Andrej Karpathy* as "Andre Kathy"; *Horace He* (the tweet about Codeforces contamination)
as "Horus Hurst"; *Dell'Acqua* (and colleagues) as "dequa" and *Ethan Mollick* as "Ethan mik";
*automaton* as "autometer"; *Latvian* as "lvan"; *Punjabi, Marathi, Telugu* as "Punjabi morati and
telu"; *McKinsey* as "McKenzie"; *Altman* as "alman"; the Financial Times headline pun *"hypely
intelligent"* as "hyly intelligent"; *Jon Barwise* as "John barwise"/"John barise"; *Norbert
Wiener* as "Norbert weer"; *Marvin Minsky* as "Marvin miny the Teeny"; *Newell and Simon* (Allen
Newell, Herbert Simon) as "new and Simon"; *Frank Rosenblatt*/*Rosenblatt's Perceptron* as "Frank
Rosen blats"/"Rosen blat"; *Wilhelm von Humboldt* as "vilhelm Von Hot"/"V Von humber's"/"hbal";
*Kahneman and Tversky* as "caraman and tersi" (not named on the slides, but the System 1/System 2
framing that follows is unambiguous); *Daniel Dennett* and *From Bacteria to Bach and Back* as
"Daniel dennet" and "from bacteria to bark and back"; Dennett's four grades *Darwinian, Skinnerian,
Popperian, Gregorian* as "dar wian," "scaran," "paparian," "Gregorian"; *Alfred Tarski* as "Alfred
tasi"/"taski"/"tasy" and *logician* as "legis"; *Richard Montague* as "Richard
montigue"/"montue"/"monu's"; *Luke Zettlemoyer* as "Luke zettel moer," *Percy Liang* as "Percy
leang," and *parser* as "paraa"; *J. R. Firth* as "Jr fth" and "the company it keeps" as "the
company at kees"; *Wittgenstein* and *Philosophical Investigations* as "viken Stein" and
"philosophical invest vation"; *Bender and Koller* as "Bender and cola"; *shehnai* throughout as
"Shai"/"shenai," *oboe* as "OBO," *reeds* as "reads," *holes* as "whole," and, in the quoted
passage from Anuradha Roy that Manning reads aloud, *Bikash Babu*, *machans*, *wail*, and *groom's
family* as "bash Babu," "ma CH," "whale," and "Grooms family"; *François Chollet* and *Keras* as
"franois charal" and "caras"; *Joelle Pineau* as "Joel Pino"; *Geoffrey Hinton* as "Jeffrey
Hinton"; *Timnit Gebru* as "Tim n jeu"; *Carl Sagan* as "Carl San"/"K San"/"Carl Sean," *Neil
deGrasse Tyson* as "Neil degrass Tyson," *The Demon-Haunted World* as "the demon Haunted World,"
and, in the closing Sagan quotation, *foreboding* as "for boing"; *RegLab* and *Daniel Ho* as "reg
lab" and "danho" (Ho isn't named on the slides, but the sentence is unmistakably about Stanford's
RegLab and its legal-AI hallucination research). Generic ASR fixes not tied to a proper noun: "NP
systems" → "NLP systems" throughout, "newal" → "neural" throughout, "pause it"/"pass it" → "parse
it," "sealing" → "ceiling" (an LSTM hitting the performance ceiling), "predal calculus" →
"predicate calculus," "iAmic pentameter" → "iambic pentameter," "Sonet" → "sonnet," "Transformer
neet architecture" → "Transformer neural net architecture," "Palo Alo"/"pal" → "Palo Alto,"
"bridging and AFA" → "bridging anaphora," and "Clos models" → "closed models." Wherever the deck
reproduces a passage Manning reads aloud verbatim — the ChatGPT-4o sonnet, the Financial Times/Time
Magazine/New York Times clippings, the shehnai use-quotation, and the closing Carl Sagan quotation
— the wording here follows the slide's exact printed text, since captions garble on-screen reading
more than they garble free speech.

Two passages are genuinely unclear and are marked inline rather than guessed at:

- At **23:25** the captions read "the jewel screen picture that we have at the moment" — no
  confident reading recovers a real phrase here.
- At **49:07** the captions read "this one's a bit different to allow in thus where" — this likely
  refers to the iota (ι) / definite-description operator used in the logical formula on the
  adjacent slide, but that's a guess, not a confident reading, so it's flagged rather than silently
  substituted.

**No content was added, removed, or reordered.**

---

**[0:05]** Okay, hi everyone, I'll get started — the last class. Okay, well, welcome,
congratulations, and thank you for making it to the last real lecture of CS224N. So, this is the
plan for today: the lecture's titled "NLP, Linguistics, and Philosophy," which I took as meaning
that I could talk about anything I wanted to, and so that is what I'm going to do. So this is sort
of what we're going to go through: talk a bit about the major ideas of CS224N and open problems,
some of the more foundational questions of where we are with LLMs, symbolic versus neural systems,
meaning, and linguistics and NLP.

**[0:54]** And then I'll close with some slides on the future risks of AI in the world. Okay, so
here is an attempt to lay out the most major things that we looked at in CS224N. We started with
word vectors, then we developed the idea of neural NLP systems. We expanded from a simple
feed-forward network into doing sequence models — language models, RNNs, LSTMs — and then we
introduced this powerful new model that's been very influential, the Transformer. And then we
built from there to the kind of — it's not exactly an architecture, but a model — that's been
built up in recent years to produce high-performance NLP systems, where we're

**[1:40]** first doing pre-training, and then a post-training phase of various techniques that we
talked about, to produce these general foundation models that understand language so well. And
then we went on from there and talked about various particular topics, like benchmarking and
reasoning. So a few of the major ideas that we looked at were: this idea that you could get a long
way by having dense representations — those are our hidden representations in neural networks —
and then looking at distributional semantics, representing words by their context, the first
slogan of "you shall know a word by the company it keeps," and I'll come back to that a bit later,
and talking about ideas of meaning. But, you know, that's essentially

**[2:27]** been the idea that has driven most of the successful ideas of modern NLP, whether it's
the earlier statistical NLP phase or the more modern neural NLP phase. And in this world we start
instantiating that as these models of word vectors, but the same contextual idea is then used in
all the models up through Transformers. We looked at both the challenges and opportunities of
training large, deep neural networks, and how gradually people developed ideas and tricks such as
having residual connections, which made it much more possible and stable to do successfully —
which took us from a place where a lot of this seemed like black magic that was hard to get right,
to

**[3:14]** people being able to very reliably train high-performance Transformer models. We talked
about sequence models, what's good about them, and some of their problems, and how those problems
have been addressed in large measure by adopting this different architecture of Transformers,
which gives a form of parallelization. And then we moved into the modern form of pre-training by
language modeling, where language modeling seems a simple thing — predicting words in context —
but it emerges as what we think of as a universal pre-training task, where all kinds of both
linguistic and world knowledge help you do this task of predicting words better. And so

**[4:01]** this has ended up as just a general method to produce the kind of powerful, knowledgeable
models that we have today. And up until now there's been this amazing property — this empirical
fact — that we seem to just get, well, not basically, it's extremely linear improvements of
performance as we continue to scale data and compute and model size up by orders of magnitude.
That doesn't mean that all problems in NLP are solved; there are lots of things that people still
work on and see opportunities to try to make things better, and a few of these are

**[4:47]** mentioned on the next few slides. So there's a real question of how much these models
are good at actually learning to do things generally, rather than just being very good at
memorization — that a lot of the benefit of what we're getting from these large pre-trained
language models is that they've seen a huge amount of stuff, and therefore they know everything,
they've seen every pattern before, and they know how to use things. I've occasionally used the
analogy that large language models are sort of like a talking encyclopedia, that they're really,
in many ways, more like a huge

**[5:34]** knowledge store than necessarily something that is intelligent, in the sense of being
able to work out how to solve new problems and generalize as human beings do. A kind of interesting
fact, actually, is that in some ways Transformer models are actually worse at generalizing than the
older LSTMs that preceded them. So here's just one little graph — I'm not going to spend a lot of
time on it — but this was looking at data that's being generated by a finite automaton, and then
trying to learn it from a limited amount of data, with either an LSTM or a Transformer, and the

**[6:20]** observation is that, at the scales they're working at, even having seen quite limited
exemplification, the LSTM is basically ceiling the entire graph — it's just at the one line —
because it generalizes in good ways because of its LSTM architecture, whereas the Transformer needs
to see a ton more data before it actually learns the patterns well. And so, if we think one of the
prime attributes of human intelligence is actually that we're amazing at figuring out and learning
things from very limited exposure — there's something you don't know how to do, and a friend shows
you once, what you

**[7:07]** do to make it work, and by and large you'll improve a few times with practice, but you
can learn effectively new skills from these kinds of single-shot examples. And that's not always
what we seem to be seeing in our models. There's a lot of interest in what's going on inside neural
networks, because a lot of the time neural networks still appear as black boxes, where we have no
real idea of how they're doing what they're doing — and, as perhaps for your final projects, the
main thing you're doing is measuring the final performance number and seeing if it goes up or not.
So there's a lot of interest in better understanding: what do

**[7:53]** they learn, how did they learn it, why do they succeed and fail. And a lot of that work
has started to look more closely into what's happening inside neural network computations. There
is some work of that sort that actually goes back quite a fair way — so here's an old blog post by
Andrej Karpathy, while he was a grad student here, in 2016, and he was looking at LSTMs and how
they learn, and he found that one of the neurons in an LSTM cell was effectively measuring position
along a line of text, and as the line of text got long, its value started to change, because the
model was learning

**[8:40]** that there was sort of a line length to this text, and that the line was likely to be
ending at that point. And in recent times, there's started to be, with Transformers as well, a lot
of work looking at mechanistic interpretability or causal abstraction, trying to understand the
internals of models. A problem that's far from solved, and in many respects probably unsolvable, is
the multilingual question of dealing with all the other languages of the world — you do have to
keep in your head that whatever you see for English, it's worse for every other language, in terms
of what people are getting out of modern language models. Now, there is a good news story here — I
don't

**[9:26]** want to claim that everything is terrible. So in this graph, which is kind of small, the
blue line was the performance of GPT-3.5 English, and then all of the green bars are the
performance of GPT-4. And so there's a genuine good-news story here, which is: look, not just for
English, but for a lot of other languages — Greek, Latvian, Arabic, Turkish — all of them are
better in GPT-4 than English was in GPT-3.5. So that's the good-news argument, that building these

**[10:11]** models big is, in some sense, raising all boats. But these are still all huge languages,
and things are starting to drop off at the bottom of this table, for languages where the
performance is worse than English and GPT-3.5 was. But even for those languages, they're languages
for which much less written data is available, but they're still large languages — so the three at
the bottom are actually all Indian languages, Punjabi, Marathi, and Telugu, which are languages
that are each spoken by millions of people; they're not small languages. So the real question is:
what happens when you actually get to

**[10:58]** the small-resource languages. The vast majority of languages around the world don't have
millions of speakers — they vary from having hundreds of speakers to hundreds of thousands of
speakers, and there are thousands of such languages. A lot of those languages are primarily oral
and have very limited amounts of written text. Now, some of those languages — or many of those
languages — are likely to go extinct in the coming decades, but many of those language communities
would like to preserve their languages, and it's very unclear how the kind of language technologies
that we've been talking about in the later parts of the course can be extended to those

**[11:44]** languages, because there just isn't sufficient data to build the kind of models that
we've been looking at. So, I imagine you've gotten some idea in this course of how evaluation is a
huge part of what we do — that effectively, a lot of the way that progress is being driven is by
defining evaluations of what models should be able to achieve, and then people working to measure
systems and improve systems so they do better on what we see as good language understanding, or
other properties. One of the concerns that many people have about what's happened with the large,
recent, closed models from large

**[12:31]** companies, is a concern that all of the benchmarks are being sullied and are not to be
trusted. So here's one example, that comes from a tweet of Horace He, and he's noting: "I suspect
GPT-4's performance is influenced by data contamination, at least on Codeforces," one of the coding
benchmarks — "of the easiest problems on Codeforces, it solved 10 out of 10 pre-2021 problems, but
zero out of 10 recent problems. This strongly points to contamination." And the worry is that every
time you're seeing these fantastic results of how well the latest, best language model is
performing,

**[13:16]** that at this point so much data is on the web that gets included in the pre-training
data for these large language models, that essentially they're memorizing at least a good share of
the questions that are appearing in these challenges. So they're not actually solving them in a
fair way, as an independent test set at all — they're just memorizing them. And so there are issues
then as to what kind of thoroughly hidden test sets we can have, or dynamic evaluation mechanisms,
so we can actually have benchmark integrity. Another huge area a number of us involved at Stanford
and elsewhere work on is making NLP work in different technical domains — so domains including
biomedical

**[14:03]** or clinical medical NLP have a lot of differences of vocabulary and usage. They have a
lot of potential good uses, but they also have a lot of potential risks of doing harm if the
language understanding is incomplete. I myself have been more involved doing things in legal NLP,
working with other people at the RegLab, with Daniel Ho, building foundation models for law. And
there are all kinds of ways, again, in which this kind of technology could be really useful — the
biggest problem, in most countries — it's bad in the United States, but it's way worse in a place
like India — is that most people

**[14:50]** can't get access to the kind of legal help that they need to solve their problems,
because of the cost of it and the lack of trained lawyers. So if more could be done to help people
via NLP tools, in principle that would be great, but in practice the tools still don't have good
enough language understanding. So, at the RegLab there's a just-completed study out at the moment
looking at legal NLP systems, and we were finding that the hallucination rate — the rate at which
there was made-up stuff in their legal answers — was, effectively, for one question in six, which
isn't a very good accuracy rate if

**[15:35]** you're someone who's wanting to rely on these systems for legal advice. There are lots
of things also to work out dealing with the social and cultural aspects of NLP — NLP systems remain
very biased against various cultures and religions. They have certain social norms, you could say,
that they pick up from somewhere, but those social norms are very biased against certain groups.
And, related to there being small languages that I mentioned before, there are lots of issues with
underrepresented groups in having the kind of NLP that they'd like to have. Okay, so that's sort of
the

**[16:22]** summary of that bit. So for the next bit, I thought I'd just give one more bit of
perspective on where we are with the best language models, like GPT-4. I think it's really
interesting, this moment of where we are, because, on the one hand, the performance of these models
is just amazing, and, even as someone who works in NLP and has worked in it for many, many years
now, I can tell a sort of story — that these models, we do this training to predict the next word,
and it's

**[17:08]** conditioning on a lot of text, and it knows about things, and it does — but, in some
sense, these things still seem like magic, right? It's just kind of hard to believe how this could
possibly work. So, in this example, I asked ChatGPT-4o — I did that this morning — I asked it to
write a sonnet explaining the Transformer neural net architecture, in which every line begins with
the letter T. And it, sort of, still, frankly, blows my mind, and I don't actually feel I can
really explain, even to myself, in a way that's convincing, how this large Transformer is able to
take all its

**[17:54]** pre-training text, and reading that instruction, and, as a next-token prediction
machine, it successfully produces something that is a sonnet, and every line begins with the letter
T. I hope you remember from your high-school English classes, sonnets are meant to have an AB
rhyming scheme — it's a little bit imperfect here, "net"/"set," some people will complain about
that, but it's basically a rhyme, "behold"/"unfold," "grasp"/"clasp," right, so it's basically doing
the rhyming right. And the lines are meant to be in iambic pentameter, and it's basically produced
lines in iambic pentameter, yeah, it's

**[18:42]** incredible. So here's the sonnet it came up with: "To delve into the world of neural
nets, Transformers rise, a marvel to behold. Through layers deep, the network's path is set, to
learn from data, patterns to unfold. The tokens feed into attention's grasp, to weigh their import,
context to align. The queries, keys, and values, in a clasp, to process sequences with power fine.
Through heads of many, multi-faceted sight, to parallelize, capturing the span. Transforming each,
a matrix math delight, to synthesize, with elegance and plan. The model's might, in every task it
shows, to turn raw data into wisdom's prose." Now, you could

**[19:31]** object that I'm not sure this exactly explained the Transformer neural net architecture
— it's a little bit abstract, I'll give it that — but, in another sense, it did, in one place or
another, evoke quite a bit of stuff about Transformers, with queries, keys, and values, and
multi-headed stuff, parallelized, with matrix math and whatever else. Yeah, still kind of blows my
mind how well that works. And, indeed, as natural-language-understanding and world-understanding
devices, I

**[20:17]** mean, these devices have clearly crossed the threshold where they're very usable in many
contexts. So, there's now started to be some fairly good studies that have been done on how much
value people can get out of using LLMs like GPT-4. There's a study by Dell'Acqua and a whole lot of
colleagues, including Ethan Mollick — they took a bunch of consultants from the Boston Consulting
Group, and, you know what that's like, that means 23-year-olds graduating from universities like
this one, but more on the East Coast — they become, you know, Boston Consultants, not exactly

**[21:05]** dummies. And so, they found, in this study — it was a controlled task, there are
actually three groups, but the big contrast is that two of the groups were using GPT-4 to do
consulting tasks, and one of the groups wasn't using GPT-4 to do tasks. The difference between the
two that were was that one of them was given more training on how to use GPT-4, but that didn't
seem to make much of a difference. But their result was that the groups using GPT-4, in their
study, completed 12% more tasks on average, they did the tasks 25% more quickly, and the results
were

**[21:54]** judged 40% higher quality than those not using AI, which I think is a pretty stunning
success, of how GPT-4, or similar LLMs, are good enough to actually help people get real work done —
with whatever asterisks you want to put about the quality of management-consultant work in various
instances. An interesting result is that using these LLMs seems to be a big leveler, and, actually,
you see exactly the same thing for people using coding LLMs: that they're a huge assistance for
people whose own skills are weaker, and they're much less of an

**[22:40]** assistance for people whose own skills are strong. Okay, so that's the good-news story.
But, on the other hand, if you're more interested in the good-news story for human beings, here's a
study that goes in the other direction: can GPT-4 write fiction that matches the quality of New
Yorker fiction writers? And the result of that study was: not even close. GPT-4 was measured as
three to ten times worse at creative writing than a New Yorker fiction writer, so there's still
hope for human beings — hang on there. And so, I think that's

**[23:25]** kind of the [Ed: unclear — captions read "jewel screen"] picture that we have at the
moment. In some ways, these things are great and useful; in other ways, they're not so great. And I
think that's something we're still going to be seeing play out in future years. I think, living in
Silicon Valley, we see a lot of the positive hype, so, if you just want to see a little bit of the
negative on the other side — late last year there was a piece in the Financial Times, which was
titled "Generative AI: Hypely Intelligent." And I won't read all of this, but basically they wanted
to express considerable

**[24:13]** skepticism of the current AI boom: "Investors should keep their heads. Expectations for
generative AI are running way ahead of the limitations that apply to it. As investment in
generative AI grows, so does pressure to create new use cases. By 2027, IDC thinks enterprise
spending on generative AI will reach $143 billion, up from $16 billion this year" — so, ten times up
— "OpenAI hopes for more funding to pursue human-like AI. It is worth remembering that, when
examining Altman's plan for superintelligence, models predict, they do not comprehend. That
limitation casts doubt on AI achieving even human-like intelligence." And then they start talking
about some of the problems,

**[25:00]** with limited gains for low-skilled workers, inaccuracies in the work they produce, and
suggests that the limitations will become more obvious as generative AI tools roll out. "That will
put pressure on providers to address costs. AI could add $4 trillion to profits, says McKinsey. But
pricing clarity is lacking. Without it, companies cannot predict what financial gains AI can
accomplish, and AI cannot predict that either." Okay, that's that topic, I'm chugging through my
topics. The next topic — I wanted to return and say a bit more about symbolic methods that

**[25:46]** dominated AI from the '60s until about 2010, versus what I've termed here "cybernetics,"
because the original alternative, going back to the '50s and '60s, was called cybernetics. And, in
a very real sense, neural networks is a continuation of the cybernetics tradition, rather than the
AI tradition that started in the '50s and '60s. In this context, Stanford is the home of the
Symbolic Systems program — so, at the moment, we are unique in having a Symbolic Systems program.
So the name

**[26:32]** Symbolic Systems came about because, at the time it was started, philosophy was an
active part of the Symbolic Systems program, and Jon Barwise — shown in this picture, he actually
died young, he died in 2000 — Jon Barwise had a very strong belief that you needed to be dealing
with meaning in the world, and the connection between people's thinking and the world. And so he
refused to allow the program to be called "cognitive science," as it's called at most other places,
and it ended up being called

**[27:19]** Symbolic Systems. Now, at one point there were two universities that had Symbolic
Systems, because Jon Barwise actually moved away from Stanford and went to Indiana, which is where
he originally was from, and so Indiana also had a Symbolic Systems program for a number of years,
but they've actually changed theirs to cognitive science now, since he died. So we are unique in
having Symbolic Systems. And so the idea of Symbolic Systems — this is sort of what's on the
website, with a bit of interpretation — "symbolic systems" studies systems of meaningful symbols
that represent the world about us, like human languages, logics, and programming languages, and the
systems that work with

**[28:05]** these symbols, like brains, computers, and complex social systems — contrasting that to
the sort of typical view of cognitive science, which is focusing on the mind and intelligence as a
naturally occurring phenomenon. Symbolic Systems gives equal focus to human-constructed systems
that use symbols to communicate and to represent information. So, in AI terms, AI as a field, and
the name "AI," arose around arguing for a symbolic approach — that John, John McCarthy, who's the
color photo there, and who founded Stanford's artificial

**[28:51]** intelligence lab — the original, famous Stanford AI Lab. So John McCarthy came up with
the name "artificial intelligence," and he very explicitly chose a new name to disassociate what he
was doing from the cybernetics approach, which had been pursued by people including Norbert Wiener,
at MIT, who's shown on the right side. So Marvin Minsky — the tiny photo down here — sort of founded
artificial intelligence at MIT; McCarthy worked with him for a few years, and then McCarthy came to
Stanford. And two of the other most

**[29:37]** prominent early AI people, Newell and Simon, who were at CMU, and the other two people
on the right side. And, in particular, Newell and Simon developed — well, actually, let me say a
sentence first: McCarthy's own background was as a mathematician and logician, so he wanted to
construct an artificial intelligence that looked like math and logic, effectively, and that sort of
was AI as a symbolic system. And that was developed as a position in the philosophy of artificial
intelligence by Newell and Simon, and so they developed what they called the physical symbol system

**[30:25]** hypothesis, so that said a physical symbol system has the necessary and sufficient means
for general intelligent action. And that's a super strong claim — it's not only claiming that having
a symbol system allows you to produce artificial general intelligence, but, through the "necessary"
clause, that you can't have artificial general intelligence without having a symbol system. So that
was sort of the basis of classical AI. And that kind of contrasts a bit with — so, cybernetics had

**[31:11]** its origins in sort of control and communication, so it's much nearer to an
electrical-engineering kind of background, and was wanting to unify ideas of control and
communication between animals — maybe perhaps more than humans — and machines. So, cybernetics
comes from a Greek word, "kybernetes," which has some interesting uses — it's exactly the same root
that occurs both in Kubernetes, if you're familiar with that, the distributed-containers software on
modern systems, but also

**[31:58]** it's actually the same root that the word "government" comes from — of course, it's a
control system as well. So, under the cybernetics tradition was when neural nets first started
being explored — the very earliest neural nets, the most famous ones are Frank Rosenblatt's, which
were used for vision, and the neural net was actually wired. Just to say a teeny bit about this, in
case you think that AI hype is only a thing of the 2020s, there was just as much AI hype in the
1950s, when Rosenblatt unveiled his Perceptron. So, in the New York Times

**[32:45]** article about it: "New Navy Device Learns by Doing — Psychologist Shows Embryo of
Computer Designed to Read and Grow Wiser. The Navy revealed the embryo of an electronic computer
today that it expects will be able to walk, talk, see, write, reproduce itself, and be conscious of
its existence." And this hype is all the more incredible when you get to the later paragraph of the
article, and you find out what the demonstration was actually of — and the demonstration that
people were shown was that this device learned to differentiate between right-arrow and

**[33:31]** left-arrow pictures, after fifty [Laughter] exposures. But there you go. Okay, so what do
we make of this in the case of NLP and language? The position I would like to suggest is: there's
just no doubt that language is a symbolic system — that humans developed language as a symbolic
system. It's perhaps most obvious that if you

**[34:16]** think about it in writing, we have symbols — the letters and words that we use. But even
if there's no writing — and, you know, the majority of human language use over time has been verbal
human language use — even though the substrate it's carried on, whether the sound waves, or, in
sign languages, movements of hands, even though that's a continuous substrate, the structure of
human languages is a symbol system. We have symbols, which are the sounds of human languages — for
"cat" we have a "c," an "a," and a "t" — those are symbols, and they're recognized in a symbolic way
by language users. And, indeed, all the pioneering work in categorical

**[35:02]** perception in cognitive psychology is done with the sounds of human languages, the
"phones," as linguists call them — so spoken language also has a symbolic structure. But, going
against Newell and Simon, the fact that humans use a symbol system for communication doesn't mean
that the process that processes the symbols — the human brain — has to be a physical symbol system,
and so, similarly, we don't have to design our NLP computer processors as physical symbol systems
either. The brain is clearly much more like a neural network model, and probably neural

**[35:48]** models will scale better and capture language processing better than something that is a
symbolic processor. In the same way, that sort of leaves behind the question of: well, why did
humans come up with a symbol system for communication? After all, we could have just hummed at
different frequencies, and that could have been used as our system of communication. I think the
dominant idea, which seems reasonable to me, but who knows, is that having a symbolic system gives
signaling reliability — that if you have discrete target points that are separated, then that gives
you an ability, when there's degradation of the

**[36:34]** signal, to recover it well. So, where does that leave linguistics, which has mainly been
developed in terms of describing a symbolic system? I think the right way to think about it is:
linguistics is good for giving us questions, concepts, and distinctions when thinking about language
acquisition, processing, and understanding. And, indeed, one of the interesting things that's come
about is that, as NLP and AI have been developed further, and are able to do a lot of the low-level
stuff, the sort of higher-level concepts that linguists often talk about

**[37:20]** — a lot, things like compositionality and systematic generalization, which I'll come back
to in a few minutes, the mapping of stable meanings for symbols, the reference of linguistic
expressions in the world — they get talked about more and more in artificial-intelligence contexts,
building neural systems. And I think one way to think about it is that a lot of the early
neural-network work, most notably visual processing, but also other kinds of sensory stuff like
sounds — doing that is sort of what gets you to insect-level intelligence. And if you

**[38:05]** want to get higher up the chain than insect-level intelligence, then a lot of the
questions and properties of linguistic systems become increasingly relevant. At a slightly more
prosaic level, I don't think one necessarily wants to believe all the fine details of different
linguistic theories, but, for how human languages are structured and how they behave, I think most
of our broad understanding of linguistics is right. And so, therefore, when we're thinking about
NLP systems, and we're thinking about understanding how they behave, wanting to know whether they
have certain properties, thinking up ways to

**[38:52]** evaluate them, a lot of that is done in terms of linguistic understanding, wanting to see
whether they capture facts about sentence structure, discourse structure, semantic properties like
natural language inference, whether you can do things like bridging anaphora — which I did not cover
this year's class, because we skipped the coreference lecture when we sliced one lecture off the
class — metaphors, presuppositions — all of these things are linguistic notions that we try to get
our NLP models to capture. So, I just want to say a couple more remarks about the role of human
language in human intelligence — I think this is kind of interesting.

**[39:37]** So an interesting person in the history of linguistics is this guy, Wilhelm von
Humboldt, who was a prominent German academic. Really, the American education system was borrowed
from Germany — up until the Second World War, the preeminent place of science and learning was
Germany, and Germany, essentially via von Humboldt's work, developed the idea of having graduate
education, and the US copied graduate education from Germany and started doing its own. But, in
that context, it was still the

**[40:24]** case that, for people in the United States prior to the 1930s, generally people would go
to Germany to finish their education, either to get their PhD or to do a postdoc, or something like
that. So, if you trace back my own academic tree, or most other academic trees of people who got
PhDs in the US, they actually go back a few generations, and then they go back to Germany — we
don't think of that as much in the modern world. So, von Humboldt was influential in developing the
university system, but he also worked a lot on language, and he's

**[41:13]** someone that Chomsky always cites, because he's known for this famous statement, that
human language must "make infinite use of finite means" — the fact that we have a limited supply of
words and sentence structures, but out of those we can recursively build up an infinite number of
sentences. And that's, in Chomsky's view, supporting the kind of symbolic, structured view of
language that he's been advocating. But I think there's another interesting take of von Humboldt's,
which — we can argue whether it's right or not, but I think is kind of interesting — and one of the
things he wants to stress is that

**[41:59]** language isn't just something used for the purpose of communication. I should actually
introduce something here: Kahneman and Tversky are two well-known cognitive psychologists, and they
introduced this idea that there are two kinds of thinking, System 1 cognition and System 2
cognition. System 1 is the kind of subconscious thinking that you're not really aware of, we just
process stuff as it comes into our heads, whether visual signals or speech, and System 2 thinking is
the conscious, "let me think about this and try to figure out what's going on"

**[42:46]** kind of "I'm solving a math problem" style of thinking. And I think you can see, in von
Humboldt's writings, essentially the same kind of distinction between System 1 and System 2
cognition, although he refers to System 1 cognition as "acts of the spirit," and System 2 cognition
as "thinking." And so, basically, he argues for a version of the philosophical position of the
language of thought, suggesting that effective System 2 thinking requires an extension of the mind
through the symbols of language. And so he argued that having

**[43:34]** language is absolutely a necessary foundation for the progress of the human mind, and I
think that's actually an interesting perspective, which I have some sympathy with. Obviously, we
can think without language — we can feel afraid, we can think visually about how things fit
together — but I think it's fairly plausible that, for the sort of more abstract, larger-scale
thinking that humans engage in, and that has led them to higher levels of thought than a chimpanzee
gets to, language gives a scaffolding inside the mind that makes that possible. Another version of
that

**[44:20]** is from the philosopher Daniel Dennett, who actually just died a couple of months ago.
Dennett wrote this book called From Bacteria to Bach and Back, and the main thing this book was
about was the origin of human consciousness — I'm not going to talk about human consciousness
today — but he introduced this model of four grades of progressively more competent intelligences.
And the four levels he outlined were: the bottom one was Darwinian — Darwinian intelligence was
something that was predesigned and fixed, it doesn't improve during its lifetime, improvement only
happens by

**[45:09]** evolution through genetic selection, so things like bacteria and viruses are Darwinian
intelligences. Then, after that, was Skinnerian intelligences — they improve behavior by learning to
respond to reinforcement, so something like a lizard, or perhaps a dog — we could argue about how
intelligent dogs are — has Skinnerian intelligence. And then the third level up, Popperian
intelligence, is things that learn models of the environment, so they can improve performance by
thinking through plans and then executing them

**[45:57]** and seeing how they behave. So, in a computational sense, Popperian intelligence kind of
means that you can do model-based reinforcement learning, and primates like chimpanzees can
definitely do the kind of planning and model-based reinforcement learning that gives you Popperian
intelligence. But, actually, a lot of recent evidence shows that a lot of simpler creatures can also
do it — I'm not sure of the facts here, but all these studies you see are about crows, from the
South Pacific, Australia, and

**[46:44]** Fiji, and places like that, so I'm not sure if Northern Hemisphere crows are dumber, but
at least Southern Hemisphere crows can learn plans, so that they can do multi-stage planning to work
out ways to get a piece of meat that's down a hole, by learning to pick up a stick and poke it in.
So, even crows can be Popperian intelligences. But what Dennett suggests is that there's a stage
beyond Popperian intelligence, which he calls Gregorian intelligence, and the idea of Gregorian
intelligence is that you can build thinking tools, which allow you to do a higher level of control
of mental

**[47:32]** searches. And so he suggests that things like — well, mathematics is a thinking tool,
but, well, also democracy is a thinking tool — but, nevertheless, out of the space of thinking
tools, human language is the preeminent thinking tool that we have. And so he suggests that the only
biological example we have of a Gregorian intelligence is human beings. And so, I think, in that
kind of sense, you can say that there's a very important role for language. Okay, two parts to go in
my summary. Okay, so the next one is: what kind of semantics should we use for

**[48:19]** language. So, this is getting back to the question I mentioned for word vectors, and
this is kind of interesting: the semantics that's been dominant in philosophy of language, or in
linguistic semantics, is a notion of model-theoretic semantics, where the meaning of words is their
denotation, what they represent in the world. I mentioned this, I think, in an early lecture: if you
have a word like "computer," the meaning of "computer" is the set of computers — this one, that one,
that one, all the other computers out there. So it's a denotational relationship between a word and
its denotation in the world, or in a model of the world, and that was the notion that was used in
most of the

**[49:07]** history of AI for doing symbolic AI, and that then contrasts with this sort of
distributional semantics, that the meaning of a word is understanding the context in which it's
used, which is effectively what we're using for our neural models. So, if you look at the
traditional view of interpreting the meaning of human language — and this is what you'll have seen
if you did an Intro Logic class at some point — we have a sentence, "the red apple is on the table,"
and you get to write it in some logical representation, first-order predicate calculus or whatever —
this one's a bit different [Ed: unclear — captions read "to allow in thus"] — where

**[49:53]** normally, for first-order predicate calculus, you only do "for all" and "there exists,"
but you have a sort of formal logic. And, in the early weeks — weeks one and two of the logic
class — you have some English sentences which you translate into formal logic, and then, after
that, you forget about human languages, and you just start proving stuff about formal logical
systems. And so, to some extent, what you get in a philosophy class represents the tradition of
Alfred Tarski — Tarski believed that you couldn't talk about meaning in terms of human languages,
because human languages were, quote, "impossibly

**[50:40]** incoherent." And so, from about the 1940s until 1980, Tarski was the preeminent logician
in the US — he was at Berkeley — and so that was very much the view of the logicians of the world.
But, during that period, one of his students was this guy, Richard Montague. Richard Montague sort
of rebelled against that picture, saying, "I reject the contention that an important theoretical
difference exists between formal and natural languages." And so he then set about showing that

**[51:27]** well, you could start building up a formal semantics for describing the meaning of
natural-language sentences. And so Richard Montague's work became the foundation of the work that's
used in semantics in linguistics as well — for anyone who's done Linguistics 130 or 230, the picture
you saw is sort of a Montague picture of semantics. And so that was the semantics that was taken
over and essentially used as the model of doing natural-language understanding for most of the
history of NLP, roughly 1960 to

**[52:14]** 2015, 2017. And so the picture, essentially, was that if we wanted to interpret a
sentence like "the red apple is on the table," what we would do is first produce a syntactic
structure for the sentence — so we would parse it — and then, using ideas roughly along the lines
that Montague suggested, we would construct its meaning by looking up meanings of words in a
lexicon, and then using the compositionality of human languages to work out the meanings of
progressively larger phrases and clauses, in terms of the meanings of those words and the way

**[53:01]** that they are combined — slightly reminiscent of my discussion of tree structures to
meanings in the last lecture I gave. And so you would build up a meaning representation of a
sentence, and this could then give you a semantic meaning of a sentence that you could use in a
system. This is approximately a slide — an actual slide that I used to use in CS224N in the 2000s
decade. So we have — we have, or part of a sentence — I get, oh no, it's a whole sentence, here it
is, how many red cars — what can I get, this sentence, I

**[53:47]** think there's a sentence here — how many, oh, how many red cars in Palo Alto does Kathy
like, how many red cars in Palo Alto does Kathy like — sorry, "cars" got hidden underneath here. So
we have a sentence: we parse it, we look up meanings of words in a lexicon, we start composing them
up, we get a semantic form for the whole sentence, which we can then convert into SQL, and we can
run it against a database and get the answer. And this was, in outline, the kind of technology that
was widely used for natural-language-understanding systems that were built anywhere from the 1960s
to the 2000s and 2010s, and

**[54:33]** in particular, they were used not only in a purely rule-based grammar-and-lexicon way —
this same basic technology was incorporated into a machine-learning context, where your goal was to
start to learn various of these parts. You could not only learn the parser, but you could also learn
semantic meanings of words and learn composition rules. And so the acme of that work was then what
was called semantic parsing, that was pioneered by Luke Zettlemoyer and Mike Collins in the 2000s
decade, and then taken up by others, including Percy Liang — Liang's PhD thesis, but also, actually,
his early work at Stanford, before he was convinced to do neural

**[55:19]** networks — was doing semantic-parsing work. So, these systems could actually work, and
were used in limited domains, but they're always extremely brittle. And, the interesting thing is,
sort of, what about humans? There is some evidence that humans do something like this, that they
work out the structure of sentences and compute meanings in a bottom-up, mostly projective way.
There's a lot of controversy as to exactly how human understanding of sentences works, but there are
certainly people who have argued in support of human brains doing something similar,

**[56:07]** that's obviously not what we're getting with current-day Transformers. And so, the
question is: do our current-day neural language models provide suitable meaning functions? And
that's a complex question, because, in many ways, they seem to — they do an amazing job at
understanding whatever sentences you put into them — but there are still some genuine concerns as to
whether they're taking shortcuts, or working, to a certain extent, and don't actually have the same
kind of compositional understanding, with systematic generalization, that human beings do. Okay, so
that's the traditional

**[56:54]** denotational semantics view, and that contrasts with the kind of "use theory of
meaning" — and, in the first or second lecture, and at the beginning of this one, I attributed that
to the British linguist J. R. Firth: "you shall know a word by the company it keeps." But it's not
only a position of Firth's — it's also been a minority position of philosophers, in particular it
was advanced by Wittgenstein in his later work, in his work Philosophical Investigations. So, in
that work, he writes: "When I talk about language, words, sentences, etc., I must speak the language
of every day. Is this language somehow too coarse and material for what we want to say? Then how is
another one to be

**[57:41]** constructed? And how strange that we should be able to do anything at all with the one we
have!" Philosophical Investigations is written in this sort of vaguely poetical, literary style, but
the point of it is meant to be saying: look, these logician people are claiming you can't use
natural human languages to express meaning, and you have to translate into this symbol system — but
isn't that a weird concept, that one symbol system is no good, but this other symbol system somehow
fixes things? And then, about denotational semantics, he writes: "You say the point isn't the word,
but its meaning, and you think of the meaning as a thing of the same kind as the word, though also
different from the word. Here, the

**[58:28]** word, there the meaning." So that's the symbol and its denotation, the money and the cow
that you can buy with it. "But contrast: money, and its use." And he goes on from there to argue
that the meaning of money is the way that money can be used in the world — the meaning of money
isn't pointing at pieces of money. Okay, so this is what's referred to as a "use theory of meaning."
And so the question is: is that a good theory of meaning? Some people just don't accept this kind of
distributional-semantics use theory of meaning as a theory

**[59:14]** of meaning, or semantics — most prominently, in recent NLP work, that's the position of
Bender and Koller, that they just take as axiomatic that the only thing that counts as having a
meaning is that you've got form over here and meaning over there. But I think that's too narrow — I
think we have to argue that meaning arises from connecting — meaning of words arises from connecting
words to other things. And, although in some sense you could say connecting words to things in the
real world is privileged, it's not the only way that you can ground meanings — you can have meanings
in a virtual world, but you can also have

**[1:00:01]** meanings by connecting one word to other things in human language. And the other thing
that I think you need to say is: meaning isn't a sort of zero-one thing, that you either have the
denotation of a word or you don't — I think meaning is a gradient thing, and you can understand the
meanings of words and phrases either more or less. So, this is an example I gave in a piece that I
wrote a couple of years ago: okay, what is the meaning of the word "shehnai"? Well, maybe a few of
you know it, but, if you don't, well, what could I do? Well, if you'd seen or held one, you'd have a
classic grounded meaning,

**[1:00:49]** you'd know something about the denotation. Well, if that's not the case, well, I could
at least show you a picture of one — here's a picture of one — so that gives you some information
about what a shehnai is. But is that the only thing I can do? I mean, suppose — well, sorry, I left
out a bullet point — so this gives you a partial meaning of a shehnai, but surely you have a richer
meaning if you'd heard one being played. And, well, is showing you a picture of one the only thing I
can do? Suppose you'd never seen, felt, or heard one, but I told you it's a traditional Indian
instrument, a bit like an oboe — well, I think you'd

**[1:01:37]** understand something about the meaning of the word at that point — that it's sort of
connected to India, it's a wind instrument using reeds, that's used for playing music. I could tell
you some other things about it — I could say it has holes, sort of like a recorder, but has multiple
reeds and a flared end, more like an oboe — then maybe you know a bit more about a shehnai, even
though you've never seen one. And, if you then extend to what we do more in our sort of corpus-based
linguistic learning, you could imagine it's not that I tried to define one for you — instead, I've
just shown you a textual-use example, so here, or several

**[1:02:25]** of those. So here's one textual-use example: "From a week before, shehnai players sat
in bamboo machans at the entrance to the house, playing their pipes. Bikash Babu disliked the
shehnai's wail, but was determined to fulfil every conventional expectation the groom's family might
have." So, if that's all you know about a shehnai, in some ways you understand less of the meaning
of the word than if you'd seen one, but, actually, in other ways, you understand more of the meaning
of the word than if you'd just seen one, because, from that one textual example, you know some
things — you

**[1:03:12]** have heard a characterization of the sound as wailing, and that it's connected with
weddings, which you don't get from just having held or looked at one, or even having had someone
stand in front of you and play it. And that's an important part of the meaning of "shehnai" to
people. And so that's the sense in which I think meaning comes from various kinds of connections.
Okay, last topic, our AI future. So, there are different senses of "our AI future," and lots of
things that we can be worried about. One thing we can be worried about is whether we're all going to
lose our jobs — interesting question. Here's a

**[1:04:03]** newspaper article from The New York Times: "March of the Machine Makes Idle Hands —
Prevalence of Unemployment with Greatly Increased Industrial Output Points to the Influence of
Labor-Saving Devices as an Underlying Cause." This was published in the New York Times in 1928. But,
it turns out that quite a few people like labor-saving machines, like washing machines, and
dishwashers, and sewing machines — lots of useful labor-saving machines. And, well, this was
published in 1928, just before — at a time when a small group of immensely powerful

**[1:04:49]** and rich men dominated the United States, just before the Great Depression. But what
happened in the decades after that, greatly changed policies in the United States led to boom years
that distributed wealth and work much more evenly across the country, and the country boomed.
Here's another one: "In the past, new industries hired far more people than those they put out of
business. But this is not true of many of today's new industries. Today's new industries have
comparatively few jobs for the unskilled or semiskilled, just the class of workers whose jobs are
being eliminated by automation."

**[1:05:37]** This was Time Magazine, in 1961. So, this is a longstanding fear which, at least so
far, has not been realized — here we are, in a country in which not everyone might have the work
that they wish they had, but, overall, almost everybody has a job, and many people are working a lot
of hours a week — whereas, once upon a time, the claim was that, before the end of the 20th century,
we would only have to do a three-day work week, because there wouldn't be much work to go around.
Imagine! So, another fear is: will almost all the money go to five to ten enormous technology
giants? I actually

**[1:06:24]** think this is a more serious worry — this seems to be the direction that we're headed
in at the moment. I think there's no doubt that modern networks, and a concentration of AI talent,
tend to encourage this outcome — but, essentially, this is the modern analog of what happened in the
early decades of the 20th century. The equivalent then was transportation networks, and it was
domination of the new transportation networks, like railways, that led to a few people dominating
the economic system. But what happens there would essentially come down to a political and social
question. So, as I was mentioning before, after the Great

**[1:07:11]** Depression, countries successfully dealt with the monopolistic power of a small number
of companies, and, with political leadership, we could do that again. The problem is that there's
not much sign of political leadership right at the moment, but that's a political problem to solve,
rather than actually being a technological problem to solve. So, the next problem is: should we be
afraid of an imminent "singularity," i.e., when machines have artificial general intelligence beyond
the human level? In particular, would such an event threaten human survival? So, this is a concern
that has increasingly

**[1:08:00]** exploded into the mainstream, with discussions of AI existential risk, and quite a few
of the discussions that have been leading to the setting up of things like AI safety institutes in
the US and UK are motivated by these worries of out-of-control artificial intelligence taking over
and deciding to eliminate humanity. So, we get these sort of article headlines, like "Pausing AI
Developments Isn't Enough, We Need to Shut It All Down," "How Rogue AIs May Arise," "AI 'Godfather'
Geoffrey Hinton Warns of Dangers as He Quits Google," "We Must Slow Down the Race to God-Like AI,"

**[1:08:47]** I don't personally give these concerns too much credence, and I think there's started
to be increasing pushback against them. So, in the other direction, François Chollet, who is the
architect of Keras, argues: "There does not exist any AI model or technique that could represent an
extinction risk for humanity, not even if you extrapolate capabilities far into the future via
scaling laws. Most arguments boil down to: this is a new type of technology, it could happen."
Joelle Pineau, a senior Meta AI leader, refers to the existential-risk discourse as

**[1:09:34]** "unhinged," and points out the flaw of a lot of the utilitarian argumentation that goes
along with discussions of these risks, which is: if you say the elimination of humanity is
infinitely bad, that means any nonzero chance, multiplied by infinity, will be bigger than the
badness of anything else that could happen in the world — but that isn't actually a sensible way to
have rational discussion about the outcomes. And many people, including Timnit Gebru, have argued
that a lot of — well, a lot of what — a lot of the

**[1:10:19]** outcome of this focus on existential risk — and, if you're more cynical, a lot of the
purpose of this focus on existential risk — is to distract away from the immediate harms that are
arising from companies deploying automated systems, including their biases, worker exploitation,
copyright violation, disinformation, growing concentration of power, and regulatory capture by
leading AI companies. And that's something that is worth thinking about — that behind all the
discussions of our amazing AIs, and all the things we can do with them, like get our homework done,
or generate wonderful images, there are lots of things underneath, about

**[1:11:05]** disinformation, deception, hallucinations, problems of homogeneity of decision-making,
violation of copyrights and people's creativity, lots of carbon emissions, erosion of rich human
practices. So we need to be conscious of the sort of present-day harms that can come about from AI.
And, for NLP as well, there are various kinds of harms that we've touched on, which include
generating offensive content, generating untruthful content, and enabling disinformation. So the
disinformation one is an interesting one — if models can reason well about text, can they also be
persuasive in

**[1:11:51]** communicating incorrect information or opinions to users — perhaps there are new
possibilities for doing very personalized misinformation propagation that more easily persuades
human beings than traditional methods of political advertising. And there's starting to be evidence
that that's true — it's still being debated in the literature, but there are now multiple studies
suggesting that humans can be influenced by disinformation generated by AIs, and it seems reasonable
to think that we're going to start to see more use of that in political systems and elsewhere, which
is potentially quite scary. And, perhaps the worst of

**[1:12:36]** it isn't going to be text-based — it's likely that visual fakes are going to be even
more compelling in a political context. And this sort of seems like, whether it happens in the US,
for this election, or in other countries, in their elections, that we're likely to see some major
incidents where AI-generated fakes can be seen to have a major impact on political systems. So, I
sort of think, really, what we should be doing is worrying not about existential risks, but worrying
about what people and organizations with power will use AI to do. This is a pattern that we've
noticed multiple

**[1:13:24]** times — also with social media, right, in the early days of social media there was the
idea that this was meant to lead to new freedoms for people across the globe, bringing the positives
of free political thought and improved human lives in large measure. That isn't what's happened —
new technologies get captured by powerful people and organizations who master the new technological
options, and AI and machine learning is being increasingly used for surveillance and control, and
we're seeing that around the world at the moment. So, my final thought to end with is a thought
about Carl Sagan. So, when I was young, many decades ago,

**[1:14:12]** Carl Sagan did the series Cosmos on television, explaining the miracles of the
universe, and, at the time, when I was a teenager, I loved Cosmos. Now, this was a long time ago —
so, much more recently, there's now a new generation of Cosmos, and the book is advertised on the
basis of — with a new foreword by Neil deGrasse Tyson. I think Sagan was a good guy, and he didn't
only write Cosmos — he wrote a number of other books, and another of the books he wrote was The
Demon-Haunted World, which has a theme that's a little bit closer to some of the things

**[1:14:58]** that connect with what we're dealing with here. So, in that book, he writes: "I have a
foreboding of a world in my children's or grandchildren's time — when awesome technological powers
are in the hands of a very few, and no one representing the public interest can even grasp the
issues; when the people have lost the ability to set their own agendas or knowledgeably question
those in authority; when, clutching our crystals and nervously consulting our horoscopes, our
critical faculties in decline, unable to distinguish between what feels good and what's true, we
slide, almost without noticing, back into superstition and darkness." I think, if you look around

**[1:15:44]** the US and many other parts of the world today, this is actually much more the risk
that humanity is facing, and why education, which we try to provide at Stanford and other places, is
an important thing that should be valued, and all the other things that go along with this, of
having things like open source, that supports the broad dissemination of learning. Thank you.
[Applause]
