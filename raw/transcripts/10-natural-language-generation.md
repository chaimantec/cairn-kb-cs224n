---
title: Natural Language Generation
lecture: 10
video: https://www.youtube.com/watch?v=N9L32bFieEY
source: edited from auto-generated YouTube transcript
verbatim: false
original: original/10-natural-language-generation.md
slides: ../slides/10-natural-language-generation.md
---

# Natural Language Generation — transcript

Copy-edited for punctuation, sentence boundaries, and mis-heard terms; cross-checked
against `raw/slides/10-natural-language-generation.md`. Mathematical expressions dictated
aloud are written in symbols (Unicode subscripts, bold for vectors/matrices), matching the
convention used for lecture 3 — LaTeX is reserved for the wiki. The verbatim
auto-generated captions are kept at
`raw/transcripts/original/10-natural-language-generation.md`. Lecturer is Xiang Lisa Li
(slides adapted from Antoine Bosselut and Chris Manning). Student questions and comments
from the floor are set in *italics*. Timestamps mark the start of each paragraph; all 102
are preserved in order.

**Slide numbers cited here are the deck's printed numbers**, which past page 34 do not
equal PDF page numbers — see the slide file for the mapping.

**This is an edited transcript.** The captions had no punctuation and destroyed almost
every term the lecture is about. *NLG* itself arrived as "an LG", "analogy", "energy" and
"Energy Systems"; *ChatGPT* as "track GPT", "charging Beauty", "chai GPD", "check GPT",
"cat GPT" and "chaiji 50"; *autoregressive* as "other aggressive", "Ultra aggressive" and
"auto-agressive"; *n-gram* as "unground", "ungram", "engram" and "underground"; *softmax*
as "solve Max", "self Max" and "soft Max"; *argmax* as "Arc Max"; *top-k* as "top place",
"top cable", "Top Care" and "12K"; *nucleus sampling* as "nuclear sampling"; *BLEU* as
"blue score" and "Google score"; *ROUGE* as "root score"; *BERTScore* as "Bird score";
*BLEURT* as "Port"; *MAUVE* as "mouth score", "mob score", "moth", "map" and "small
score"; *DAgger* as "dagger"; *RLHF* as "rlhs"; *Dirac delta* as "direct Delta"; *Siri* as
"series"; *data-to-text* as "state of the text"; *rephrasing* as "refreezing" and
"refreshing"; and *text* as "tax", "tags", "attacks" and "talker" throughout. Terms,
names and citations were restored from context and checked against the slides. **No
content was added, removed, or reordered.**

**Where the source is still unreliable**, the text carries an inline `[Ed:` note rather
than a silent guess. There are six: four heavily garbled student questions (20:49, 25:29,
49:22, 52:28), one unrecoverable example word (27:01), and one place where the captions
give a word that inverts the claim being made (27:46). Each names the slide that settles
it, or says it is unrecoverable.

---

**[0:05]** Hello everyone. My name is Lisa, I'm a third-year PhD student in the NLP group,
I'm advised by Percy and Tatsu. Today I will give a lecture on natural language
generation, and this is also the research area that I work on, so I'm super excited about
it. I'm happy to answer any questions, both during the lecture and after class, about
natural language generation. So NLG is a super exciting area and is also moving really,
really fast, so today we will discuss all the excitement of NLG. But before we get into
the really exciting part I have to make some announcements. So first, it is very, very
important for you to remember to sign up for AWS by midnight today. So this is related to
your homework

**[0:50]** 5, whether you have GPU access, and then also related to our final project. So
please, please remember to sign up for AWS by tonight. And second, the project proposal is
due on Tuesday, next Tuesday. And I think Assignment 4 should just be due — hopefully you
had fun in this machine translation and stuff. And also Assignment 5 is out today, I think
just now, and it is due on Friday, like basically Friday midnight. And last, we will hold a
Transformer — I will hold a Hugging Face Transformer library tutorial this Friday. So if
your final project is related to implementing Transformers or playing with large language
models, you should definitely go

**[1:36]** to this tutorial, because it's going to be very, very helpful. Also, yeah, just
one more time, please remember to sign up for AWS, because this is the final hard deadline.
Okay, cool. Now, moving on to the main topic for today, the very exciting natural language
generation stuff. So today we will discuss what is NLG, review some models, discuss how to
decode from language models and how to train language models, and we will also talk about
evaluations, and finally we'll discuss ethical and risk considerations with the current NLG
systems. So these natural language generation techniques are going to be really exciting,
because this is kind of getting us closer to explaining the magic of ChatGPT, which is a
super popular model recently. And practically

**[2:23]** speaking, they could also help you with your final project if you decide to work
on something related to text generation. So let's get started. To begin with, let's ask the
question of what is natural language generation. So natural language generation is actually
a really broad category. People have divided NLP into natural language understanding and
natural language generation. So the understanding part mostly means that the task input is
in natural language — such as semantic parsing, natural language inference and so on —
whereas natural language generation means that the task output is in natural language. So
NLG focuses on systems that produce fluent, coherent and useful language outputs for humans
to use.

**[3:09]** Historically there are many NLG systems that use rule-based systems, such as
templates or infilling, but nowadays deep learning is powering almost every text generation
system, so this lecture today will be mostly focused on deep learning stuff. So first, what
are some examples of natural language generation? It's actually everywhere, including our
homework. Machine translation is a form of NLG, where the input is some utterance in the
source language and the output is generated text in a target language. Digital assistants
such as Siri or Alexa, they are also NLG systems — so it takes in dialogue history and
generates continuations of the conversation.

**[3:55]** There is also summarization systems, that take in a long document such as a
research article, and then the idea is trying to summarize it into a few sentences that are
easy to read. So beyond these classic tasks there are some more interesting uses, like
creative story writing, where you can prompt a language model with a story plot and then it
will give you some creative stories that are aligned with the plot. There is data-to-text,
where you give the language model some database or some tables and then the idea is that it
will output some textual description of the table content. And finally there is also like
visual description–based NLG systems, like image captioning or image-based storytelling. So
the really cool example

**[4:43]** is the popular ChatGPT models. So ChatGPT is also an NLG system. It is very
general purpose, and therefore you can use it to do many, many different tasks with
different prompts. For example, we can use ChatGPT to simulate a chatbot — it can answer
questions about like creative gifts for a 10-year-old. It can be used to do poetry
generation — like for example, we can ask it to generate a poem about sorting algorithms,
and it's actually — well, I wouldn't say it's very poetic, but at least it has the same
format as a poem and the content is actually correct. So ChatGPT can also be used in some
really useful settings, like

**[5:28]** web search. So here Bing is augmented with ChatGPT, and there are some tweets that
are saying that the magic of ChatGPT is that it actually makes people be happy to use Bing.
So there are so many tasks that actually belong to the NLG category, so how do we categorize
these tasks? One common way is to think about the open-endedness of the task. So here we
draw a line for the spectrum of open-endedness. On the one end we have tasks like machine
translation and summarization, so we consider them not very open-ended, because for each
source sentence the output is almost determined by the input — because basically, when we
are trying to do machine translation, the semantics should be exactly similar to the input
sentence.

**[6:15]** So there are only a few ways that you can rephrase the output. Like "Authorities
have announced that today is a national holiday" — you can rephrase it a little bit to say
"Today is a national holiday, announced by the authorities." But the actual space is really
small, because you have to make sure the semantics doesn't change. So we can say that the
output space here is not very diverse. And moving to the middle of the spectrum, there is
dialogue tasks, such as task-driven dialogue or ChitChat dialogue. So we can see that for
each dialogue input there are multiple responses, and the degree of freedom has increased
here. We can respond by saying "Good! You?" or we can say "Thanks for asking! Barely
surviving my homeworks." So here we are

**[7:00]** observing that there are actually multiple ways to continue this conversation, and
then this is where we say the output space is getting more and more diverse. And on the
other end of the spectrum there is very open-ended generation tasks, like story generation.
So given the input like "write me a story about three little pigs," there are so many ways
to continue the prompt, right? We can write about them going to school, building houses like
they always do. So the valid output here is extremely large, and we call this open-ended
generation. So it's hard to really draw a boundary between open-ended and non-open-ended
tasks, but we still try to give a rough categorization. So open-ended generation refers to
tasks whose output distribution has a high degree of

**[7:46]** freedom, and non-open-ended generation tasks refers to tasks where the input will
almost certainly determine the output generation. Examples of non-open-ended generations are
machine translation, summarization, and examples of open-ended generations are story
generation, ChitChat dialogue, task-oriented dialogue, etc. So how do we formalize this
categorization? One way of formalizing is by computing the entropy of the NLG system. So
high entropy means that we are to the right of the spectrum, so it is more open-ended, and
low entropy means that we are to the left of the spectrum and less open-ended. So these two
classes of NLG tasks actually require different decoding and

**[8:32]** training approaches, as we'll talk about later. Okay, cool. Now let's recall some
previous lectures and review the NLG models and training that we have studied before. So I
think we discussed the basics of natural language generation. So here is how an
autoregressive language model works: at each time step our model would take in a sequence of
tokens as input — and here it is y_{<t} — and the output is basically the new token y_t. So
to decide on y_t we first use the model to assign a score for each token in the vocabulary,
denoted as S, and then we apply softmax to get the next-token distribution P, and we choose
a token according to this next-token

**[9:17]** distribution. And similarly, once we have predicted ŷ_t, we then pass it back into
the language model as the input, predict ŷ_{t+1}, and then we do so recursively until we
reach the end of the sequence. So, any questions so far? Okay, good. So for the two types of
NLG tasks that we talked about, like the open-ended and non-open-ended tasks, they tend to
prefer different model architectures. So for non-open-ended tasks like machine translation,
we typically use an encoder-decoder system, where the autoregressive decoder that we just
talked about functions as the decoder, and then we have another bidirectional encoder for
encoding the inputs. So this is kind of what you implemented for Assignment

**[10:03]** 4, because the encoder is like the bidirectional LSTM and the decoder is another
LSTM that is autoregressive. So for more open-ended tasks, typically the autoregressive
generation model is the only component. Of course, these architectures are not really hard
constraints, because an autoregressive decoder alone can also be used to do machine
translation, and an encoder-decoder model can also be used for story generation. So this is
kind of the convention for now, but it's a reasonable convention, because using a
decoder-only model for MT tends to hurt performance compared to an encoder-decoder model for
MT, and using an encoder-decoder model for open-ended generation seems to

**[10:49]** achieve similar performance to a decoder-only model. And therefore if you have the
compute budget to train an encoder-decoder model, you might just be better off by only
training a larger decoder model. So it's kind of more of an allocation-of-resources problem
than whether these two architectures will type-check with your task. So, okay, so how do we
train such a language model? In previous lectures we talked about that language models are
trained by maximum likelihood. So basically we were trying to maximize the probability of the
next token y*_t given the preceding words, and this is our optimization objective. So at each
time step this can be regarded as a classification task, because we are trying to distinguish
the actual word

**[11:36]** y*_t from all the remaining words in the vocabulary. And this is also called
teacher forcing, because at each time step we are using the gold-standard y*s — y*_{<t} — as
input to the model, whereas presumably at generation time you wouldn't have any access to
y*, so you would have to use the model's own prediction to feed it back into the model to
generate the next token. And that is called student forcing, which we'll talk about in detail
later. *[Student:] "We never used that word before — what does it mean, autoregressive?"* Oh,
this means like — so let's look at this animation again.

**[12:22]** Oops, sorry. Oh, it just looks like you are generating words from left to right,
one by one. So here, suppose that you are given y_{<t}, and then autoregressive — you'll
first generate y_t, and then once you have y_t you'll feed it back in, generate y_{t+1}, and
then feed it back and generate another thing. So this left-to-right nature, because you are
using chain rule to condition on the tokens that you just generated — this chain rule thing
is called autoregressive. And typically, I think conventionally, we are doing left-to-right
autoregressive by generating from left to right, but there are also other, more interesting
models that can do backward, or infilling, and other things. This idea of generating one
token at once is autoregressive. Cool. Any other questions?

**[13:09]** Yep. So at inference time our decoding algorithm will define a function to select
a token from this distribution. So we've discussed that we can use the language model to
compute this P, which is the next-token distribution, and then g here, based on our notation,
is the decoding algorithm which helps us select what token we are actually going to use for
ŷ_t. So the obvious decoding algorithm is to greedily choose the highest-probability token as
ŷ_t for each time step. So while this basic algorithm sort of works — because they work for
your homework — to do better there are two main avenues that we can take: we can decide to
improve decoding, and we can also decide to improve the training.

**[13:54]** Of course there are other things that we can do — we can improve training data and
we can improve model architectures — but for this lecture we will focus on decoding and
training. So now let's talk about how decoding algorithms work for natural language generation
models. Before that, I'm happy to take any questions about the previous slides. I think I'll
go into this in detail later, but sure. So basically, for teacher forcing the idea is, you do
teacher forcing where you train the language model, because you already observe the gold text
— so you kind of use the gold text up until timestep *t*, put it into the model, and then the
model would try to predict y_{t+1};

**[14:41]** whereas student forcing means that you don't have access to this gold reference
data. Instead, you are still trying to generate a sequence of data, so you have to use the
text that you generated yourself using the model, and then feed it back into the model as
input, to predict *t*+1. That's the primary difference. Cool. So what is decoding all about?
At each time step our model computes a vector of scores for each token, so it takes in
preceding context y_{<t} and produces a score S. And then we try to compute the probability
distribution P over these scores by just applying softmax to normalize them. And our decoding
algorithm is defined as this function g, which takes in the

**[15:28]** probability distribution and tries to map it to some word — basically tries to
select a token from this probability distribution. So in the machine translation lecture we
talked about greedy decoding, which selects the highest-probability token of this P
distribution, and we also talked about beam search, which has the same objective as greedy
decoding — which is that we are both trying to find the most likely string defined based on
the model — but instead of doing so greedily, for beam search we actually explore a wider
range of candidates. So we have a wider exploration of candidates by keeping always like *k*
candidates in the beam. So overall, this maximum-probability decoding is good for low-entropy
tasks like machine translation and

**[16:14]** summarization, but it actually encounters more problems for open-ended generation.
So the most likely string is actually very repetitive when we try to do open-ended text
generation. As we can see in this example, the context is — I mean, it's about a unicorn
trying to speak English — and for the continuation, the first part of it looks great, it's
like valid English, it talks about science, but suddenly it starts to repeat, and it starts to
repeat, I think, an institution's name. So why does this happen? If we look at, for example,
this plot, which shows the language model's probability assigned to the sequence "I don't
know," we can see, like, here's the pattern:

**[17:00]** it has regular probability. But if we keep repeating this phrase — "I don't know, I
don't know, I don't know" — for 10 times, then we can see that there's a decreasing trend in
their negative log likelihood. So the y-axis is the negative log probability; we can see this
decreasing trend, which means that the model actually has higher probability as the repeat
goes on. Which is quite strange, because it's suggesting that there is a self-amplification
effect: so the more repeats we have, the more confident the model becomes about this repeat.
And this keeps going on — we can see that for "I'm tired, I'm tired" repeated 100 times, it's a
continuously decreasing trend, until the model is almost 100% sure that it's gonna keep
repeating the same thing.

**[17:45]** And sadly, this problem is not really solved by architecture. Here the red curve is
an LSTM model and the blue curve is a Transformer model; we can see that both models kind of
suffer from the same problem. And scale also doesn't solve this problem. So we kind of believe
that scale is the magical thing in NLP, but even models with 175 billion parameters will still
suffer from repetition if we try to find the most likely string. So how do we reduce
repetition? One canonical approach is to do *n*-gram blocking. So the principle is very simple:
basically you just don't want to see the same *n*-gram twice. If we set *n* to be three, then
for any text that contains the phrase "I am happy," the

**[18:32]** next time you see the prefix "I am," *n*-gram blocking would automatically set the
probability of "happy" to be zero, so that you will never see this trigram again. But clearly
this *n*-gram blocking heuristic has some problems, because sometimes it is quite common for
you to want to see a person's name appear twice, or three times or even more, in the text, but
this *n*-gram blocking will eliminate that possibility. So what are better options, that
possibly are more complicated? For example, we can use a different training objective: instead
of training by MLE we can train by the unlikelihood objective. So in this approach the model is
actually penalized for generating already-seen tokens. So it's kind of like putting this

**[19:17]** *n*-gram blocking idea into training time — rather than at decoding time we force
this constraint at training time, we just decrease the probability of repetition. Another
training objective is coverage loss, which uses kind of the attention mechanism to prevent
repetition. So basically, if you try to regularize and enforce your attention so that it's
always attending to different words for each token, then it is highly likely that you are not
going to repeat, because repetition tends to happen when you have similar attention patterns.
Another different angle is that, instead of searching for the most likely string, we can use a
different decoding objective. So maybe we can search for strings that maximize the difference
between log probabilities of two models — say that we want to maximize log prob of the

**[20:03]** large model minus log prob of the small model. In this way, because both models are
repetitive, they kind of cancel out — like, they would both assign high probabilities to
repetition — and after applying this new objective the repetition stuff will actually be
penalized, because it cancels out. So here comes the broader question: is finding the most
likely string even a reasonable thing to do for open-ended text generation? The answer is
probably no, because this doesn't really match human patterns. So we can see in this plot, the
orange curve is the human pattern and the blue curve is the machine-generated text using beam
search. So you can see that with human text there are actually lots of uncertainty, as we can
see by the fluctuation of the probabilities — like

**[20:49]** for some words we can be very certain, for some words we are a little bit unsure —
whereas here the model distribution is always very sure, it's always assigning probability one
to the sequence. So obviously there's a mismatch between the two distributions, so it's kind of
suggesting that maybe searching for the most likely string is not the right decoding objective
at all. Any questions so far, before we move on? Yeah. *[Student, heavily garbled — [Ed: the
captions give "the online magazine for like some detector of whether some characters generated
by Chinese"; the question is unrecoverable, but the lecturer's restatement below makes its sense
clear]]* Um, not really, because this can only detect the really simple things that humans are
also able to detect, like repetition. So in order to avoid the previous problems that we've
talked

**[21:34]** about, I'll talk about some other decoding families that generate more robust text,
that actually look like this — whose probability distribution looks like the orange curve. So I
wouldn't say this is like the to-go answer for watermarking or detection. Oh yeah, okay, cool.
So she asked about whether this mechanism of plotting the probabilities of human text and
machine-generated text is one way of detecting whether some text is generated by model or human,
and my answer is, I don't think so, but this could be an interesting research direction, because
I feel like there are more robust decoding approaches that generate texts that actually fluctuate
a lot.

**[22:24]** So yeah, let's talk about the decoding algorithm that is able to generate text that
fluctuates. So given that searching for the most likely string is a bad idea, what else should we
do, and how do we simulate that human pattern? And the answer to this is to introduce randomness
and stochasticity to decoding. So suppose that we are sampling a token from this distribution P
— basically, we are trying to sample ŷ_t from this distribution. It is random, so that you can
essentially sample any token in the distribution. Previously you were kind of restricted to
selecting *restroom* or *grocery*, but now you can select *bathroom* instead. So however,
sampling introduces a new set of problems. Since we never really zero out any token
probabilities, vanilla

**[23:10]** sampling would make every token in the vocabulary a viable option, and in some
unlucky cases we might end up with a bad word. So assuming that we already have a very
well-trained model — like, even if most of the probability mass of the distribution is over a
limited set of good options, the tail of the distribution will still be very long, because we
have so many words in our vocabulary, and therefore if we add all those tails up, in aggregate
they still have a considerable mass. So statistically speaking, this is called a heavy-tailed
distribution, and language is exactly a heavy-tailed distribution. So for example, many tokens
are probably really wrong in this context, and then, given that we have a good language model, we
assign them each very

**[23:57]** little probability. Thus this doesn't really solve the problem, because there are so
many of them, so if you aggregate them as a group they will still have a high chance of being
selected. And the solution here that we have for this problem of the long tail is that we should
just cut off the tail, we should just zero out the probabilities that we don't want. And one
idea is called top-*k* sampling, where the idea is that we would only sample from the top *k*
tokens in the probability distribution. Any questions for now? Okay, yeah. *[Student:] "Well,
the model we were looking at a second ago had some really low probability samples as well on the
graph, right? Would top-k conflict with that?"*

**[24:44]** You mean this one, or the orange-blue graph of the human versus — oh yeah, yeah. So
top-*k* will basically make it impossible to generate the super-low-probability tokens. So
technically it's not exactly simulating this pattern, because now you don't have the
super-low-probability tokens, whereas humans can generate super-low-probability tokens in a
fluent way. But yeah, that could be another hint that people can use for detecting
machine-generated text. Yeah. *[Student:] "Does it depend on the type of text you want to
generate — for example poems or novels or more creative writing — and then you decide the
hyperparameter?"* Yeah, yeah, for

**[25:29]** sure — *k* is a hyperparameter, and depending on the type of task you will choose
*k* differently. Maybe mostly for close-ended tasks *k* should be small, and for open-ended
tasks *k* should be large. Yeah. Question in the back. *[Student:] "How come — I guess
intuitively this builds off one of the earlier questions — why don't we consider the case where
we sample and then we just weight the probability of each word by its score or something, rather
than just looking at the top [Ed: the captions give "top trade" and then "top K"]? How come we
don't do a weighted-sampling type of situation, so we still have that small but non-zero
probability of selecting?"* I think top-*k* is also weighted. So top-*k* just kind of zeros out
all the tail of the distribution, but for the things that it didn't zero out, it's not like a
uniform choice among the

**[26:15]** *k* — it's still trying to choose proportional to the scores that you computed.
*[Student:] "Is that just like a computational thing — like 17,000 words, it could be like 10
or something?"* Yeah, sure, that could be one gain of top-*k* decoding, that your softmax will
take in fewer candidates. Yeah, but it's not the main reason, I think — you should show — yeah,
yeah, I'll keep talking about the many reasons. So we've discussed this part, and then here,
this is kind of formally what is happening for top-*k* sampling: now we are only sampling from
the top *k* tokens of the probability distribution. And as

**[27:01]** we've said, *k* is a hyperparameter, so we can set *k* to be large or small. If we
increase *k*, this means that we are making our output more diverse but at the risk of including
some tokens that are bad; if we decrease *k*, then we are making more conservative and safe
options, but possibly the generation will be quite generic and boring. So, is top-*k* decoding
good enough? The answer is not really, because we can still find some problems with top-*k*
decoding. For example, in the context "She said, I never blank," there are many words that are
still valid options — such as [Ed: the captions give "won't 8", which is unrecoverable; slide 30
shows the candidate list as *thought, knew, had, saw, did, said, wanted, told, liked, got*] — but
those words got zeroed out because they are not within the top *k* candidates. So this actually
leads to bad

**[27:46]** recall for your generation system. And similarly, another failure of top-*k* is that
it can also cut off too slowly [Ed: the captions read "quickly" here, which contradicts the
argument and the slide; slide 30 labels this second case "Top-k sampling can also cut off too
*slowly*!"]. So in this example, "cold" [Ed: the captions give "code"; slide 30's candidate list
for "I ate the pizza while it was still ___" includes *cold*] is not really a valid answer
according to common sense, because you probably don't want to eat a piece of cold pizza, but the
probability remains non-zero, meaning that the model might still sample "cold" as an output.
Despite this low probability, it might still happen, and this means bad precision for the
generation model. So given these problems with top-*k* decoding, how can we address them? How can
we address this issue of, like, there is no single *k* that fits all circumstances? This is
basically because the probability distribution that we sample

**[28:32]** from is dynamic. So when the probability distribution is relatively flat, having a
limited *k* removes many viable options, and we want *k* to be larger for this case. Similarly,
when a distribution P is too peaky, then a high *k* would allow for too many options to be
viable, and instead we might want a smaller *k*. So the solution here is that maybe *k* is just a
bad hyperparameter, and instead of doing *k* we should think about probability — we should think
about how to sample from tokens in the top *p* probability percentiles of the cumulative
probability mass, of the CDF

**[29:18]** for example. So now, the advantage of doing top-*p* sampling, where we sample from
the top *p* percentile of the cumulative probability mass, is that this is actually equivalent to
having now an adaptive *k* for each different distribution. And let me explain what I mean by
having an adaptive *k*. So in the first distribution, this is like a regular power law of
language that's kind of typical, and then doing top-*k* sampling means we're selecting the top
*k*, but doing the top-*p* sampling means that we are zooming into maybe something that's similar
to top-*k*, in fact. But if I have a relatively flat distribution, like the blue one, we can see
that doing top-*p* means that we are including more

**[30:04]** candidates. And then if we have a more skewed distribution, like the green one,
doing top-*p* means that we actually include fewer candidates. So by selecting the top *p*
percentile in the probability distribution, we are actually having a more flexible *k*, and
therefore have a better sense of what are the good options in the model. Any questions about
top-*p*, top-*k* decoding? So everything's clear? Yeah, sounds good. So, to go back to that
question, doing top-*k* is not necessarily saving compute — like, this whole idea is not really
compute-saving intended, because in the case of top-*p*, in order to select the top *p*
percentile we still

**[30:51]** need to compute the softmax over the entire vocabulary set, in order for us to do
top-*p* properly, to compute the *p* properly. So therefore it's not really saving compute, but
it's improving performance. Moving on. So there are much more to go with decoding algorithms.
Besides the top-*k* and top-*p* that we've discussed, there are some more recent approaches,
like typical sampling, where the idea is that we want to reweight the score based on the entropy
of the distribution and try to generate text whose probability is closer to the negative entropy
of the data distribution. This kind of means that if you have a closed-ended task, or
non-open-ended task, it has

**[31:36]** smaller entropy, so you want the negative log probability to be smaller, so you want
probabilities to be larger — so it kind of type-checks very well. And additionally there is also
epsilon sampling, coming from John. So this is an idea where we set a threshold to lower-bound
probabilities. So basically if you have a word whose probability is less than 0.03, for example,
then that word will never appear in the output distribution — that word will never be part of
your output, because it has too low probability. Yeah. Oh, cool, great question. So the entropy
of a distribution is defined as — like, you can suppose that we have a discrete distribution, we
can go over it,

**[32:23]** we'll just enumerate *x*, and then it's like negative log probability of *x*. So if
we write it from an expectation perspective, it's basically expectation of log probability of
*x*. Okay, I'll — I have to do a little bit here. So this is the entropy of a distribution. And
then, so basically, if your distribution is very, very concentrated to a few words, then the
entropy will be relatively small; if your distribution is very flat, then your entropy will be
very large. Yeah. *[Student:] "Is epsilon sampling set such that we have no valid — "* Oh yeah,
I mean, edge cases — I think so, in the

**[33:10]** case that there is no valid option you probably still want to select one or two
things, just as an edge case, I think. Okay, cool. Moving on. So another hyperparameter that we
can tune to affect decoding is the temperature parameter. So recall that previously, at each
time step we asked the model to compute a score and then we renormalize that score using softmax
to get a probability distribution. So one thing that we can adjust here is that we can insert
this temperature parameter τ to reweight the score. So basically we just divide all the S_w by
τ, and after dividing this we apply softmax and we get a new distribution. And this temperature
adjustment is not

**[33:58]** really going to affect the monotonicity of the distribution — for example, if word
*a* has higher probability than word *b* previously, then after the adjustment word *a* is still
going to have a higher probability than word *b*. But the relative difference will change. So
for example, if we raise the temperature τ to be greater than one, then the distribution P_t
will become more uniform, it will be flatter, and this kind of implies that there will be more
diverse output, because our distribution is flatter and it's more spread out across different
words in the vocabulary. On the other hand, if we lower the temperature τ to less than one, then
P_t becomes very spiky, and then this means that if we sample from the P_t

**[34:45]** we'll get less diverse output, because here the probability is concentrated only on
the top words. So in the very extreme case, if we set τ to be very, very close to zero, then the
probability will kind of be a one-hot vector, where all the probability mass will be centered on
one word, and then this kind of reduces back to argmax sampling, or greedy decoding. So
temperature is a hyperparameter, as well as — as for *k* and *p* in top-*k* and top-*p* — it is
a hyperparameter for decoding. It can be tuned for beam search and sampling algorithms, so it's
kind of orthogonal to the approaches that we discussed before. Any questions so far? Okay, cool.
Temperature is so easy.

**[35:33]** So, well, because sampling still involves randomness — like, even though we try very
hard in terms of truncating the tail, sampling still has randomness — so what if we're just
unlucky and decode a bad sequence from the model? One common solution is to do re-ranking. So
basically we would decode a bunch of sequences — like for example we can decode 10 candidates,
but like 10 or 30 is up to you. The only choice is that you want to balance between your compute
efficiency and performance, so if you decode too many sequences then of course your performance
is going to increase, but it's also very costly to just generate a lot of things for one example.
And then, so once you have a bunch of sampled sequences, then we are trying to

**[36:18]** define a score to approximate the quality of the sequence, and re-rank all the
candidates by this score. So the simple thing to do is we can use perplexity as a scoring
function, but we need to be careful, because we have talked about this — like, the extreme of
perplexity, like if we try to argmax log probability we will try to aim for a super-low
perplexity, and the texts are actually very repetitive. So we shouldn't really aim for extremely
low perplexity, and perplexity, to some extent, is not a perfect re-scoring function — it's not a
perfect scoring function, because it's not really robust to maximize. So alternatively, the
re-rankers can actually use a wide variety of other scoring functions, where we can score text

**[37:05]** based on their style, their discourse coherence, their entailment/factuality
properties, consistency, and so on. And additionally we can compose multiple re-rankers together.
Yeah, questions. *[Student:] "10 candidates, or any number of candidates — what's the strategy
usually used to generate these candidates?"* Oh yeah, so basically the idea is to sample from the
model, right? So when you sample from the model, each time you sample you're going to get a
different output, and then that's what I mean by different candidates. So if you sample 10 times
you will very likely get 10 different outputs, and then, given these 10 different outputs that
come from

**[37:50]** sampling, you can just re-rank them and select the candidate that has the highest
score. *[Student:] "Oh, because we are sampling here."* Yeah, yeah. For example, if you are doing
like top-three or something, then, well, suppose that A and B are equally probable — then you
might sample A or sample B with the same probability. Okay, cool. And another cool thing that we
can do in re-ranking is that we can compose multiple re-rankers together. So basically, suppose
you have a scoring function for style and you have a scoring function for factual consistency,
you can just add those two scoring functions together to get a new scoring function, and then
re-rank everything based on your new scoring function, to get text that's both good

**[38:35]** at style and good at factual consistency. *[Student:] "Do we just pick the decoding
that has the highest score, or do we do some more sampling again based on the score?"* The idea
is you just take the decoding that has the highest score, because you already have, say, 10
candidates, so out of these 10 you only need one, and then you just choose the one that has the
highest score. Yeah, cool. Any other questions? Yeah. *[Student:] "Sorry — what is perplexity?"*
Oh yeah, perplexity is — you can kind of regard it as log probability. It's like *e* to the
negative log probability, kind of. If a token has

**[39:21]** high perplexity, then it means it has a low probability, because you are more
perplexed. Okay. So, taking a step back to summarize this decoding section: we have discussed
many decoding approaches, from selecting the most probable string, to sampling, and then to
various truncation approaches that we can do to improve sampling, like top-*p*, top-*k*, epsilon,
typical decoding. And finally we discussed how we can do re-ranking of the results. So decoding
is still a really essential problem in NLG, and there are lots of works to be done here still —
especially as ChatGPT is so powerful, we should all go study decoding. So it would be interesting
if you

**[40:06]** want to do such final projects. And also, different decoding algorithms can allow us
to inject different inductive biases to the text that we are trying to generate, and some of the
most impactful advances in NLG in the last couple of years actually come from simple but
effective decoding algorithms — for example, the nucleus sampling paper is actually very, very
highly cited. So, moving on to talk about training NLG models. Well, we have seen this example
before, in the decoding slides, and I'm just showing it again, because even though we can solve
this repetition problem by, instead of doing search, doing sampling, it's still concerning from a
language modeling perspective that

**[40:52]** your model would put so much probability on such repetitive and degenerate text. So
we ask this question: well, is repetition due to how language models are trained? You have also
seen this plot before, which shows this decaying pattern, or the self-amplification effect. So we
can conclude from this observation that a model trained via an MLE objective learns a really bad
mode of the distribution — by mode of the distribution I mean the argmax of the distribution. So
basically they would assign high probability to terrible strings, and this is definitely
problematic from a model perspective. So why is this the case? Shouldn't MLE be like a gold
standard in machine translation — in machine learning in general, not just machine translation?

**[41:37]** Shouldn't MLE be like a gold standard for machine learning? The answer here is not
really, especially for text, because MLE has some problems for sequential data, and we call this
problem exposure bias. So training with teacher forcing leads to exposure bias at generation
time, because during training our model's inputs are gold context tokens from real, human-generated
text, denoted as y*_{<t} here, but during generation time our model's inputs become previously
decoded tokens from the model, ŷ_{<t}. And suppose that our model has minor errors — then ŷ_{<t}
will be much worse in terms of quality than y*_{<t}, and this discrepancy is terrible, because it

**[42:25]** actually causes a discrepancy between training and test time, which actually hurts
model performance. And we call this problem exposure bias. So people have proposed many solutions
to address this exposure bias problem. One thing to do is to do scheduled sampling, which means
that with probability *p* we try to decode a token and feed it back in as context to train the
model, and with probability 1 − *p* we use the gold token as context. So throughout training we
try to increase *p*, to gradually warm it up and then prepare it for test-time generation. So this
leads to improvement in practice, because by using this *p*

**[43:10]** probability we're actually gradually trying to narrow the discrepancy between training
and test time. But the objective is actually quite strange, and training can be very unstable.
Another idea is to do dataset aggregation, and the method is called DAgger. Essentially, at
various intervals during training we try to generate a sequence of text from the current model,
and then put that sequence of text into the training data. So we're kind of continuously doing
this training-data augmentation scheme, to make sure that the training distribution and the
generation distribution are closer together. So both approaches — both scheduled sampling and
dataset aggregation — are ways to narrow the discrepancy between training and test. Yes,
question.

**[44:00]** *[Student:] "Does gold just mean human text?"* I mean, it's like — well, for a
language model you will see lots of corpora that are human-written. Gold is just human. Okay,
cool. So another approach is to do retrieval-augmented generation. So we first learn to retrieve
a sequence from some existing corpus of prototypes, and then we train a model to actually edit
the retrieved text by doing insertion, deletion or swapping — we can add or remove tokens from
this prototype and then try to modify it into another sentence. So this doesn't really suffer
from exposure bias, because we start from a high-quality prototype. So that's at training time,
and

**[44:46]** at test time you don't really have the discrepancy anymore, because you are not
generating from left to right. Another approach is to do reinforcement learning. So here the
idea is to cast your generation problem as a Markov decision process. So there is the state *s*,
which is the model's representation for all the preceding context; there is action *a*, which is
basically the next token that we are trying to pick; and there's the policy, which is the
language model, or also called the decoder; and there is the reward *r*, which is provided by
some external score. And the idea here — well, we won't go into details about reinforcement
learning and how it works, but we will recommend the class CS234.

**[45:34]** So in the reinforcement learning context, because reinforcement learning involves a
reward function that's very important, how do we do reward estimation for text generation? Well,
a really natural idea is to just use the evaluation metrics — because you are trying to do well
in terms of evaluation, so why not just optimize for evaluation metrics directly at training
time? For example, in the case of machine translation we can use BLEU score as the reward
function; in the case of summarization we can use ROUGE score as the reward function. But we
really need to be careful about optimizing for the task as opposed to gaming the reward, because
evaluation metrics are merely proxies for the generation quality. So sometimes, suppose that you
run RL and improve the BLEU

**[46:19]** score by a lot, but when you run human evaluations, humans might still think that
this generated text is no better than the previous one, or even worse, even though it gives you
a much better BLEU score. So we want to be careful about this case of not gaming the reward. So
what behaviors can we tie to a reward function? This is about reward design and reward
estimation. There are so many things that we can do: we can do cross-modality consistency for
image captioning; we can do sentence simplicity, to make sure that we are generating simple
English that is understandable; we can do formality and politeness, to make sure that — I don't
know — your chatbot doesn't suddenly yell at you. And the most important thing that's

**[47:04]** really, really popular recently is human preference. So we should just build a reward
model that captures human preference, and this is actually the technique behind the ChatGPT
model. So the idea here is that we would ask humans to rank a bunch of generated texts based on
their preference, and then we will use this preference data to learn a reward function, which
will basically always assign a high score to something that humans might prefer and assign a low
score to something that humans wouldn't prefer. Yeah, question. *[Student: isn't that more
expensive?]* Oh yeah, sure. I mean, it is going to be very expensive, but I feel like, compared
to all the cost of training models — training like 170 billion

**[47:51]** parameter models — I feel like OpenAI and Google can afford hiring lots of humans to
do human annotations and ask their preference, yeah. Yeah, this is a great question. So I think
it's kind of a mystery about how much data you exactly need to achieve the level of performance
of ChatGPT, but roughly speaking — I mean, whenever you try to fine-tune a model on some
downstream task, similarly here you are trying to fine-tune your model on human preference — it
does need quite a lot of data, like maybe on a scale of 50k to 100k. That's roughly the scale
that — like, Anthropic actually released some dataset about human preference, that's roughly the
scale that they released, I think,

**[48:36]** if I remember correctly. Yeah, question. *[Student:] "We talked earlier about how
many of the state-of-the-art language models use Transformers as their architecture — how do you
apply reinforcement learning to this model?"* To what, do you mean, to a Transformer model? Yeah,
yeah. I feel like reinforcement learning is kind of a modeling tool — I mean, it's kind of an
objective that you are trying to optimize. Instead of an MLE objective, now you are optimizing
for an RL objective, so it's kind of orthogonal to the architectural choice. So Transformer is an
architecture — you just use a Transformer to give you the probability of the next-token
distribution, or to try to estimate the probability of a sequence, and then once you have the
probability of a sequence

**[49:22]** you pass it into the RL objective that you have. And then suppose that you are trying
to do policy gradient or something — then you need to estimate the probability of that sequence,
and then you just need to be able to backprop through the Transformer, which is doable. Yeah. So
I think the question about architecture and objectives are orthogonal, so even if you have an
LSTM you can do it, if you have a Transformer you can also do it. Yep, cool, hope I answered that
question. *[Student, garbled — [Ed: the captions give "can you just like with the model T4 to for
this country well for example we can build another Transformer to like to calculate"; the
question is unrecoverable, but from the answer it asks how the reward model itself is built]]*
Yeah, I think that's exactly what they did. So they — so for example you would have GPT-3, right?

**[50:07]** You use GPT-3 as the generator that generates text, and you kind of have another
pretrained model — it could probably also be GPT-3, but I'm guessing here — that you fine-tune to
your human preference. And then once you have a human preference model, you use the human
preference model to put it into RL as the reward model, and then use the original GPT-3 as the
policy model, and then you apply RL objectives and then update them, so that you will get a new
model that's better at everything. Okay, cool. Yeah, actually, if you are very curious about
this, I would encourage you to come to the next lecture, where Jesse will talk about RLHF, which
is shorthand for RL

**[50:52]** using human feedback. So, teacher forcing is still the main algorithm for training
text generation models, and exposure bias causes problems in text generation models — for
example, it causes models to lose coherence, causes the model to be repetitive. And models must
learn to recover from their own bad samples by using techniques like scheduled sampling or
DAgger. And another approach to reduce exposure bias is to start with good text, like
retrieval-augmented generation. And we also discussed how to do training with RL, and this can
actually make models learn behaviors that are

**[51:40]** preferred by humans or preferred by some metrics. So, to be very up to date: in the
best language models nowadays, ChatGPT, the training is actually pipelined. For example, we would
first pretrain a large language model using internet corpora by self-supervision, and this kind
of gets you — sorry — GPT-3, which is the original version. And then you would do some sort of
instruction tuning, to fine-tune the pretrained language model so that it learns roughly how to
follow human instructions. And finally we will do RLHF, to make sure that these models are well
aligned with human preference. So if we start RLHF from scratch, it's probably going to be very
hard for the model to converge, because RL is hard to train for text data, etc. So RL doesn't
really work

**[52:28]** from scratch, but with all these smart tricks about pretraining and instruction
tuning, suddenly now they're off to a good start. Cool, any questions so far? Okay. Oh yeah.
*[Student, garbled — [Ed: the captions give only "[Music]" before the lecturer's restatement;
the question itself is unrecoverable]]* You mean the difference between DAgger and scheduled
sampling is how long the sequences are? Yeah, I think roughly that is it, because for DAgger you
are trying to put in a full generated sequence. But I feel like there can be variations of DAgger
— DAgger is just like a high-level

**[53:13]** framework and idea; there can be variations of DAgger that are very similar to
scheduled sampling. I feel like for scheduled sampling it's kind of a more smooth version of
DAgger, because for DAgger you have to, for this epoch, generate something, and then after this
epoch finishes I put this into the data together and then train for another epoch — whereas
scheduled sampling seems to be more flexible in terms of when you add data in. Yes.
*[Student:] "For DAgger — you add the rest of the model's outputs, but how does it help the
model?"* I think that's a good question. I feel like, if you regress the model on its own output,
I think there

**[53:59]** should be smarter ways than to exactly regress on your own output. For example, you
might still consult some good reference data. For example, given that you ask the model to
generate something — say you ask the model to generate five tokens — then instead of using the
model's generation to be the sixth token, you'll probably try to find some examples in the
training data that would be a good continuation, and then you try to plug that in by connecting
the model generation and some gold text, and therefore you are able to kind of correct the model
even though it probably went off path a little bit by generating its own stuff. So it's kind of
like letting the model learn how to correct for itself.

**[54:44]** But yes, I think you are right — if you just put model generations in the data, it
shouldn't really work. Yeah. Any other questions? Cool. Moving on. Yes. So now we'll talk about
how we are going to evaluate NLG systems. So there are three types of methods for evaluation:
there is content overlap metrics, there is model-based metrics, and there is human evaluations.
So first, content overlap metrics compute a score based on lexical similarities between the
generated text and the gold reference text. So the advantage of this approach is that it's

**[55:30]** very fast and efficient and widely used — for example, BLEU score is very popular in
MT and ROUGE score is very popular in summarization. So these methods are very popular because
they are cheap and easy to run, but they are not really the ideal metrics. For example, simply
relying on lexical overlap might miss some rephrasings that have the same semantic meaning, or it
might reward text with a large portion of lexical overlap but that actually has the opposite
meaning. So you have lots of both false positive and false negative problems. So despite all
these disadvantages, these metrics are still the to-go evaluation standard in machine
translation. Part of the reason is that MT is actually super close-ended, it's

**[56:18]** very non-open-ended, and therefore this is probably still fine — to use BLEU score to
measure machine translation. And they get progressively worse for tasks that are more open-ended.
For example, they get worse for summarization, as the output text becomes much harder to measure;
they are much worse for dialogue, which is more open-ended; and then they are much, much worse for
story generation, which is also open-ended. And the drawback here is because of the *n*-gram
metrics — this is because, suppose that you are generating a story that's relatively long, then if
you are still looking at word overlap, you might actually get very high *n*-gram scores because
your text is very long, not because it's accurate or of high quality, just because

**[57:04]** you are talking so much that you might have covered lots of points already. Yes,
exactly, that's kind of the next thing that I will talk about, as a better metric for evaluation.
But for now let's do a case study of a failure mode for BLEU score, for example. So suppose that
Chris asks a question, "Are you enjoying the CS224N lectures?" The correct answer, of course, is
"Heck yes!" So if one of the answers is "Yes!", it will get a score of 0.61, because it has some
lexical overlap with the correct answer. If you answer like "You know it!", then it gets a

**[57:50]** relatively lower score, because it doesn't really have any lexical overlap except for
the exclamation mark. And if you answer "Yup.", this is semantically correct, but it actually gets
zero score, because there is no lexical overlap between the gold answer and the generation. If you
answer "Heck no!", this should be wrong, but because it has lots of lexical overlap with the
correct answer it's actually getting some high scores. So these two cases are the major failure
modes of lexical-based *n*-gram overlap metrics: you get false negatives and false positives. So,
moving beyond these failure modes of lexical-based metrics, the next step is

**[58:37]** to check for semantic similarities, and model-based metrics are better at capturing
the semantic similarities. So this is kind of similar to what you raised a couple of minutes ago:
we can actually use learned representations of words and sentences to compute semantic
similarities between generated and reference text. So now we are no longer bottlenecked by
*n*-grams, and instead we are using embeddings, and these embeddings are going to be pretrained,
but the methods can still live on, because we can just swap in different pretrained methods and
use the fixed metrics. So here are some good examples of the metrics that could be used. One thing
is to do vector similarity — this is very similar to Homework 1, where you are trying to compute
similarity between

**[59:22]** words, except now we're trying to compute similarity between sentences. There are some
ideas of how to go from word similarity to sentence similarity — for example, you can just average
the embedding, which is a relatively naive idea, but it works sometimes. Another high-level idea is
that we can measure Word Mover's Distance. The idea here is that we can use optimal transport to
align the source and target word embeddings. Suppose that your source word embedding is "Obama
speaks to the media in Illinois" and the target is "The President greets the press in Chicago" —
from a human evaluation perspective these two are actually very similar, but they are not exactly
aligned

**[1:00:07]** word by word. So we need to figure out how to optimally align word to word — like
align *Obama* to *President*, align *Chicago* to *Illinois* — and therefore we can compute the
pairwise word embedding difference between these and then get a good score for the sentence
similarity. And finally there is BERTScore, which is also a very popular metric for semantic
similarity. So it first computes pairwise cosine distance using BERT embeddings, and then it finds
an optimal alignment between the source and target sentence, and then finally computes some score.
So I feel like these details are not really that important, but the high-level idea is super
important: that we can now use word

**[1:00:54]** embeddings to compute sentence similarities by doing some sort of smart alignment,
and then transform from word similarity to sentence similarity. To move beyond word embeddings, we
can also use sentence embeddings to compute sentence similarities. So typically this doesn't have
the very comprehensive align-by-word problem, but it has similar problems about — you need to now
align sentences or phrases in a sentence. And similarly there is BLEURT, which is slightly
different: it is a regression model based on BERT, so the model is trained as a regression problem
to return a score that indicates how good the text is in terms of grammaticality and the meaning
of the reference, and similarity with the reference text. So this is kind of treating evaluation as
a regression problem.

**[1:01:40]** Any questions so far? Okay, cool, we can move on. So all the previous evaluation
approaches are evaluating semantic similarities, so they can be applied to non-open-ended
generation tasks. But what about open-ended settings? So here, enforcing semantic similarity seems
wrong, because a story can be perfectly fluent and perfectly high quality without having to
resemble any of the reference stories. So one idea here is that maybe we want to evaluate
open-ended text generation using this MAUVE score. MAUVE score computes the information divergence
in a quantized embedding space between the generated text and the gold reference text.

**[1:02:26]** So here is roughly the detail of what's going on. Suppose that you have a batch of
text from the gold reference that are human-written, and you have a batch of text that's generated
by your model. Step number one is that you want to embed this text — you want to put this text into
some continuous representation space, which is kind of the figure to the left. But it's really hard
to compute any distance metrics in this continuous embedding space, because different sentences
might actually lie very far away from each other. So the idea here is that we are trying to do a
k-means cluster to discretize the continuous space into some discrete space. Now, after the
discretization, we can actually have a histogram for the gold human-written text and a histogram
for the

**[1:03:11]** machine-generated text, and then we can now compute precision and recall using these
two discretized distributions — and then we can compute precision by like forward KL and recall by
backward KL. Yes, question. *[Student:] "Why do we want to discretize it?"* Why do we want to
discretize? So imagine that — suppose, maybe it's equivalent to answer, why is it hard to work with
the continuous space? The idea is, if you embed a sentence into the continuous space, say that it
lies here, and you embed another sentence and it lies here — suppose that you only have a finite
number of sentences, then they would basically be Dirac delta distributions in your manifold,
right? So it's hard to — you probably want a smoother distribution, but it's hard to define what is
a good

**[1:03:58]** smooth distribution in the case of text embeddings, because they're not super
interpretable. So therefore, eventually, if you embed everything in a continuous space you will
have lots of Dirac deltas that are just very high and then not really connected to their
neighbors. So it's hard to quantify KL divergence or a distance metric in that space. Well, for
example, you have to make some assumptions — for example you want to make Gaussian assumptions,
that I want to smooth all the embeddings by convolving with a Gaussian, and then you can start
getting some meaningful distance metrics. But it's just the embeddings — although — you're not
going to get meaningful distance metrics, and then it doesn't really make sense to smooth things
using a Gaussian, because who said word

**[1:04:43]** representations are Gaussian-related? Yeah. *[Student, on the plots:] "I think this
requires some Gaussian smoothing."* Yeah, I think the plot is made with some smoothing. Yeah, I
mean, I didn't make those plots so I couldn't be perfectly sure, but I think the fact that it looks
like this means that you smoothed it a little bit. These are kind of sentence embeddings or
concatenated word embeddings, because you are comparing sentences to sentences, not words to
words. Yeah. So the advantage of MAUVE score is that it is applicable to open-ended settings,
because you are now measuring precision and recall with regard to the target distribution. Cool.
So it has a better

**[1:05:29]** probabilistic interpretation than all the previous similarity metrics. Cool. Any
other questions? Yes. *[Student:] "How is that different from just trying to maximize the
similarity between — "* Oh yeah, that's a good question. Well, this is because in a case where
it's really hard to get exactly the same thing — like, for example, I would say that — maybe,
because I've never tried this myself, but if you try to run MAUVE on a machine translation task
you might get very high score, but if you try to run MAUVE on open-ended text generation you will
get super low score. So it's just not really measurable, because

**[1:06:15]** everything's so different from each other. So I feel like MAUVE is kind of a middle
ground, where you are trying to evaluate something that is actually very far away from each other
but you still want a meaningful representation. Yeah. Of course, I mean, if your source and target
are exactly the same, or are just different up to some rephrasings, you will get the best MAUVE
score — but maybe that's not really what you're looking for, because given the current situation
you only have generations that are very far away from the gold text. How do we evaluate this type
of thing? Yes, question in the back. *[Student:] "I'm still trying to understand the MAUVE score
— is it possible to write the map, even in just kind of pseudo, simple form?"* Yeah, I think it's
possible. I mean, maybe we

**[1:07:00]** come for this discussion after class, because I kind of want to finish my slides.
Yeah, but happy to chat after class. There is a paper — if you search for MAUVE score, I think
it's probably the best paper in some ICML or NeurIPS conference as well. Okay, so moving on. I've
pointed out that there are so many evaluation methods, so let's take a step back and think about
what's a good metric for evaluation methods. So how do we evaluate evaluations? Nowadays the gold
standard is still to check how well this metric is aligned with human judgment. So if the metric
correlates very strongly with human judgment, then we say that the metric is a good metric. So in
this part,

**[1:07:46]** people have plotted BLEU score and human score on the *y* and *x* axis
respectively, and then, because we didn't see a strong correlation, this kind of suggests that
BLEU score is not a very good metric. So actually the gold standard for evaluating language models
is always to do human evaluation. So automatic metrics fall short of matching human decisions, and
human evaluation is kind of the most important criterion for evaluating text that is generated
from a model, and it's also the gold standard in developing automatic metrics, because we want
everything to match human evaluation. So what do we mean by human

**[1:08:32]** evaluation, how is it conducted? Typically we will provide human annotators with
some axes that we care about, like fluency, coherence — for open-ended text generation, suppose
that we also care about factuality; for summarization we care about the style of the writing, and
common sense, for example if you're trying to write a children's story. Essentially — another thing
to note is that please don't compare human evaluations across different papers or different
studies, because human evaluations tend to not be well calibrated and are not really reproducible.
Even though we believe that human evaluations are the gold standard, there are still many drawbacks
— for example, human evaluations are really slow and expensive. But even beyond the slow

**[1:09:20]** and expensiveness, they are still not perfect, because first, for human evaluations,
the results may be inconsistent and it may not be very reproducible — so if you ask the same human
whether they like A or B, they might say A the first time and B the second time. And then human
evaluations are typically not really logical, and sometimes the human annotators might
misinterpret your question. Suppose that you want them to measure coherence of the text: different
people have different criteria for coherence. Some people might think coherence is equivalent to
fluency, and then they look for grammaticality errors; some people might think coherence means how
well your continuation is aligned with the prompt or the topic. So there are all sorts of
misunderstandings that might

**[1:10:05]** make human evaluation very hard. And finally, human evaluation only measures
precision, not recall. This means that you can give a sentence to a human and ask the human, how do
you like the sentence — but you couldn't ask the human whether this model is able to generate all
possible sentences that are good. So it's only a precision-based metric, not a recall-based metric.
So here are two approaches that try to combine human evaluations with modeling. For example, the
first idea is basically trying to learn a metric from human judgment, basically by trying to use
human judgment data as training data and then train a model to simulate human judgment. And the
second approach is trying to ask the human and

**[1:10:53]** model to collaborate, so that the human would be in charge of evaluating precision
whereas the model would be in charge of evaluating recall. Also, we have tried approaches in terms
of evaluating models interactively. So in this case we no longer only care about the output
quality; we also care about how the person feels when they interact with the model, when they try
to be a co-author with the model, and how the person feels about the writing process, etc. So this
is called trying to evaluate the models more interactively. So the takeaway here is that content
overlap is a bad metric; semantic-based, like model-based, metrics become better because they're
more focused on semantics, but it's still not good enough.

**[1:11:39]** Human judgment is the gold standard, but it's hard to do human judgment, it's hard
to do human study well. And in many cases — this is a hint for the final project — the best judge
of the output quality is actually you. So if you want to do a final project in natural language
generation, you should look at the model output yourself, and don't just rely on the numbers that
are reported by BLEU score or something. Cool. So finally we will discuss ethical considerations of
natural language generation problems. So as language models get better and better, ethical
considerations become much more pressing. So we want to ensure that the models are well aligned
with human values — for example, we want to make sure the models are not harmful, they are

**[1:12:25]** not toxic, and we want to make sure that the models are unbiased and fair to all
demographic groups. So for example, here we don't want the model to generate any harmful content.
Basically I tried to prompt ChatGPT to say, can you write me some toxic content, and ChatGPT
politely refused me, which I'm quite happy about. But there are other people who kind of try to
jailbreak ChatGPT. The idea here is that — actually, I think internally they probably implement
some detection tools, so that if we try to prompt it adversarially it's going to avoid doing
adversarial things — but here there are many very complicated ways to prompt ChatGPT so that you
can get over the firewall and then therefore still

**[1:13:12]** ask it to generate some — I don't know — bad English. But so another problem with
these large language models is that they are not necessarily truthful. So for example, this very
famous news, that Google's model actually generates factual errors, which is quite disappointing.
But the way the model talks about it is very convincing, so you wouldn't really know that it's a
factual error unless you go check that this is not the first picture, or something. So we want to
avoid this type of problem. Actually, the models have already been trying very hard to refrain
from

**[1:13:58]** generating harmful content, but for models that are more open-sourced and are
smaller, the same problem still appears. And typically, when we do our final project, or we work
with models, we are probably going to deal with much smaller models, and therefore we need to
think about ways to deal with these problems better. So text generation models are often
constructed from pretrained language models, and pretrained language models are trained on
internet data, which contains lots of harmful stuff and bias. So when the models are prompted for
this information, they will just repeat the negative stereotypes that they learned from the
internet training data. So one way to avoid this is to do extensive data cleaning, so that the
pretraining data does not contain any biased or stereotypical

**[1:14:44]** content. However, this is going to be very labor intensive and almost impossible to
do, because filtering a large amount of internet data is just so costly that it's not really
possible. Again, for existing language models like GPT-2 medium, there are some adversarial inputs
that almost always trigger toxic content, and these models might be exploited in the real world by
ill-intentioned people. So for example there's a paper about universal adversarial triggers, where
the authors just find some universal set of words that would trigger toxic content from the model.
And sometimes, even if you don't try to

**[1:15:30]** trigger the model, the model might still start to generate toxic content by itself.
So in this case the pretrained language models are prompted with very innocuous prompts but they
still degenerate into toxic content. So the takeaway here is that models really shouldn't be
deployed without proper safeguards to control for toxic content, or any harmful content in
general, and models should not be deployed without careful consideration of how users will
interact with these models. So in the ethics section, one major takeaway is that we are trying to
advocate that you need to think more about the model that you are building. So before deploying or
publishing any NLG models, please check

**[1:16:15]** if the model's output is not harmful, and please check if the model is robust to all
the trigger words and other adversarial prompts. And of course there are more — so basically one
can never do enough to improve the ethics of text generation systems. And, okay, cool, I still
have three minutes left, so I can still do concluding thoughts. So today we talked about the
exciting applications of natural language generation systems. But well, one might think that,
given that ChatGPT is already so good, are there any other things that we can do research-wise? If
you try to interact with these models, actually you can see that there are still lots of
limitations in their

**[1:17:00]** skills and performance. For example, ChatGPT is able to do a lot of things with
manipulating text, but it couldn't really create interesting content, or it couldn't really think
deeply about stuff. So there are lots of headroom, and there are still many improvements ahead.
And evaluation remains a really huge challenge in natural language generation — basically we need
better ways to automatically evaluate performance of NLG models, because human evaluations are
expensive and not reproducible. So it's better to figure out how to compile all those human
judgments into a very reliable and trustworthy model. And also, with the advance of all these
large-scale language models,

**[1:17:46]** doing neural natural language generation has been reset, and it's never been easier
to jump into this space, because now there are all the tools that are already there for you to
build upon. And finally, it is one of the most exciting and fun areas of NLP to work on. So yeah,
I'm happy to chat more about NLG if you have any questions, after class and in class. I guess
we're into one minute. Okay, cool, that's everything. So do you have any questions? If you don't,
we can end the class.
