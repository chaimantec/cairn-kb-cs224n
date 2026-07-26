---
title: Post-training
lecture: 11
video: https://www.youtube.com/watch?v=35X6zlhoCy4
source: edited from auto-generated YouTube transcript
verbatim: false
original: original/11-post-training.md
slides: ../slides/11-post-training.md
---

# Post-training — transcript

Copy-edited for punctuation, sentence boundaries, and mis-heard terms; cross-checked
against `raw/slides/11-post-training.md`. Mathematical expressions dictated aloud are
written in symbols, matching the convention used from lecture 3 onwards — LaTeX is
reserved for the wiki. The verbatim auto-generated captions are kept at
`raw/transcripts/original/11-post-training.md`. Lecturer is Archit Sharma (slides based on
Jesse Mu's). Student questions and comments from the floor are set in *italics*.
Timestamps mark the start of each paragraph; all 104 are preserved in order.

**This is an edited transcript.** The captions had no punctuation and destroyed nearly
every model name and technical term in the lecture: *ChatGPT* arrived as "CH GPD", "chart
GPD", "chargy GPD", "chbd" and "chbt"; *GPT-2/3/4* as "gpd2", "gp22", "gbd3", "gpd3",
"GPD 3" and "gbd4"; *InstructGPT* as "instruct gbt"; *RLHF* as "rlf", "R LF", "rhf" and
"rft"; *DPO* and *PPO* as "DP", "BP" and "po"; *Bradley–Terry* as "Brad lary" and "broadly
ter"; *Kullback–Leibler* as "cbak LI Li"; *Boltzmann* as "boltzman"; *MMLU* as "mlu";
*Flan-T5* as "FL T5"; *PaLM* as "power models"; *LIMA* as "LMA" and its title *Less Is More
for Alignment* as "paperless as more for alignment"; *Mistral* as "mistol"; *Claude* as
"Claud"; *Copilot* as "co-pilot"; *TL;DR* as "tldr" and "TLD drr"; *scaling laws* as
"scaling loss"; *entailment* as "entainment"; and *upvotes* as "upwords". Terms and
citations were restored from context and checked against the slides. **No content was
added, removed, or reordered.**

Two readings are worth naming because they change a number or a name rather than a
spelling. At 22:20 the captions have GPT-3 following up with "more questions about what a
60-year-old might want" — the prompt on slide 36 is "Explain the moon landing to a **6 year
old**", so this is *six-year-old*. At 1:08:40 the captions render a student's question as
"why good and why win and why lose" — these are the symbols *y_good*, *y^w* and *y^l*.

**Every number was compared against the original**, and the ten differences are all
deliberate: ASR damage repaired (11:35 "gp22"→GPT-2, 19:17 "5 40 billion"→540 billion,
1:12:32 "th000 tasks"→a thousand tasks), spelled-out numbers written as digits (37:51 "PR
two"→Problem 2, 47:10 "y two"→y_2, 57:13 "GPT 4 versus three"→GPT-4 versus 3), the
six-year-old correction above, 26:14 "type 1 a supernova"→type-Ia supernova, 1:10:11
"199"→the student's "plus one versus plus 99" (they are quoting the lecturer's own +1/−1
example), and one false start dropped at 47:57, where "where y_1 is a winning completion"
is immediately restarted as "we have a winning completion y^w".

**Where the source is still unreliable**, the text carries an inline `[Ed:` note rather
than a silent guess. There are five, at 12:22, 13:08, 27:00, 57:13 and 1:07:07 — four
heavily garbled student questions and one benchmark name a student mentions in passing.

---

**[0:05]** Good evening, people. How are you guys doing? All right. My name is Archit
Sharma, I'm a PhD student at Stanford, and I'm very, very excited to talk about
post-training, generally speaking, for large language models. I hope you guys are ready to
learn some stuff, because the last few years in machine learning have been very, very
exciting with the advent of large language models, ChatGPT and everything to that extent.
Hopefully after today's lecture you will be more comfortable understanding how we go from
pretrained models to models like ChatGPT, and we'll take a whole journey through prompting,
instruction finetuning, and DPO and RLHF. So let's get

**[0:51]** started. All right. So something that has been very fundamental to our entire
field is this idea of scaling laws: models are increasingly becoming larger and larger and
they're expending more and more compute. This is a graph of models starting all the way
back in the 1950s to somewhere around — this is still an outdated graph, so this shows up
to 10^24 flops, or floating point operations, that go into pretraining these models, but
the number is well above 10^26 now. You can see the graph and the way it's trending. And
more and more compute requires more and more data, because you need to train on something
meaningful. This is roughly the trend on the amount of language tokens that are going

**[1:37]** into the language models in pretraining — and again this plot is outdated. Does
anybody want to guess? We're in 2024. In 2022 we were at 1.4 trillion tokens, or words
roughly speaking, in language model pretraining. Does anyone want to guess where we are in
2024? *(A student guesses.)* That's a pretty good guess, yeah. So we're close to 15 trillion
tokens — the recent Llama 3 models were roughly trained on 15 trillion tokens. So yeah,
just for a second, appreciate that these are a lot of words. I don't think anybody of us
listens to trillions of tokens in our lifetime. So this is where we are right now, and I

**[2:24]** hope you guys were here for the pretraining lectures. Cool. So what do we do?
Broadly speaking, we are really just learning to predict text tokens, or language tokens.
But what do we learn in the process of pretraining? Why are people spending so much money,
so much compute? Because this compute and these tokens take dollars, and we're on the order
of spending hundreds of millions of dollars on these runs. So why are we doing this? This
is basically a recall from whatever you have probably learned till now, but we're learning
things like — oh, we are learning knowledge: "Stanford University is located in Santa Clara,
California", or wherever you want to say. You're learning syntax, you're learning semantics
of the sentences. These are things that you would expect

**[3:10]** to learn when you're training on language data broadly. You're probably learning
a lot about different languages as well, so depending on your text data distribution you're
learning a lot of things. But the models we interact with are very intelligent, so where is
that coming from? Simply learning about very factual things, and it's a very simple loss
function we're optimizing — where is that intelligence coming from? And this perhaps is the
interesting bit. Recently people have started accumulating evidence that when you optimize
the next-token-prediction losses, you're not just learning about syntax, you're not just
learning knowledge, but you're starting to form models of agents, beliefs

**[3:55]** and actions as well. So how do we know this? Again, a lot of this is speculative
evidence, but it may be enough to form an understanding that the losses we're optimizing
are not just about fitting the data — you start learning something maybe more meaningful as
well. For example, in this specific case we change the last sentence and the prediction of
the next text that is predicted changes as well. So here it starts with "Pat watches a
demonstration of a bowling ball and a leaf being dropped at the same time. Pat, who is a
physicist, predicts that the bowling ball and the leaf will land at the same rate." We all
know gravity, the way it works. But when you change the last sentence to "Pat, who has never
seen this demonstration before", Pat predicts that the bowling ball

**[4:40]** will fall to the ground first. Maybe somebody who's never seen this experiment
before might intuitively believe that, correct? So the language model was able to predict
this. How do you predict this? You have to have some notion of understanding of how humans
work to even be able to predict this, and that's maybe something that is not obvious when
you're simply optimizing to predict the text. We're going to run through some examples to
communicate that when you're pretraining these models you're learning much more than just
language tokens. You're also learning about math — you're able to understand what a graph of
a circle means, and what the center is, and how to understand equations. Probably my
favourite example, something I use pretty much every day, is

**[5:26]** you're learning how to write code. I don't know how many of you have interacted
with Copilot before, but if you have, you probably know: if you write down a few comments,
write down a function template, it will automatically complete code for you. Again it's not
perfect, but it has to have some deeper understanding of what your intent is for something
like that to emerge. Similarly we have examples from medicine as well. I don't know about
you guys, but whenever I have some issue I probably go to ChatGPT or Claude or something to
that effect and ask them for a diagnosis for those things as well. I don't recommend that —
please don't take medical advice from me. But yeah, broadly, the way we're seeing language
models at this point is that they're sort of emerging as

**[6:12]** this general-purpose multitask assistant, and it's very strange, right? We
started off with text token prediction and we're reaching the stage where you can rely on
them to do many, many different things. So how are we getting there? I'm sure you all are
aware of what these models are. Today's lecture is largely going to be about how we go from
"Stanford University is located in" — this very simple pretraining task, a very simple
procedure; well, it's more complicated, but in abstract terms it's not very complicated —
to something as powerful as ChatGPT. Cool. I recommend you guys stopping me, asking me a lot
of questions, because there's a lot of fun examples and a lot of fun techniques, so I want
you guys to learn everything

**[6:58]** about here. So the overall plan: we're going to talk about zero-shot and few-shot
in-context learning; next we're going to follow up with instruction finetuning; and then
we're going to talk about optimizing for preferences, and this is where roughly things are
right now in the industry. And then we're going to talk about what's next, what the
limitations are, and how we move on from here. Cool. So we're going to start off with
zero-shot and few-shot in-context learning. We're going to take the example of GPT, or the
Generative Pretrained Transformer, and this is a whole series of models that started off in
roughly 2018 and up to 2020 — they were building GPT, GPT-2, GPT-3. So we're going to start
off with this example. It's a decoder-only model that is

**[7:45]** trained on roughly 4.6GB of text, and it has 12 Transformer layers, and it's
trained with the next-token-prediction loss. The first model obviously was not extremely
good, but it started showing that hey, this technique for pretraining can be very effective
for general-purpose tasks. And we're going to see some examples — for example, here it's able
to do the task for entailment. GPT-1 itself was not very strong as a model, so they took the
same recipe and tried to increase the model size. So they went from 117 million parameters
to about 1.5 billion parameters, and we're

**[8:32]** now scaling up the data alongside as well, so we went from 4GB of data to
approximately 40GB of data. Pretraining is a whole different melting pot of techniques and
there's a lot that goes into it, but roughly — for example, here they filter data by the
number of upvotes on the Reddit data. So this is roughly where we are, and I think one of
the things that started emerging with GPT-2 is zero-shot learning. What do we mean by
zero-shot learning? Conventionally in the field, when we pretrain models there was the idea
that you take a few examples, you update the model, and then you are able to adapt to a
specific task. But as you pretrain on more and more data and more and more tasks, you sort
of start seeing

**[9:17]** this phenomenon where they're able to do the task basically zero-shot: they're
shown no examples of how to do the task. And you can start thinking of, oh, how you can do
summarization, you can follow some instructions, you can do maybe a little bit of math as
well. So this is where the idea of zero-shot learning started to emerge. So how do we do
zero-shot learning, or task-specific learning, from these pretrained models? Really the idea
is that we have to be creative here. We know that these are text prediction models: if you
put in a text they will complete whatever follows. So if we can sort of coerce these models
into completing the task we care about — maybe it's question answering — we can start getting
them to solve tasks here. So for

**[10:02]** example, if you want to ask questions about Tom Brady, you set it up: you put
information about Tom Brady, and then you put a question that you want it to answer, and
then it will autocomplete in some sense. So this is one early perspective on these models:
these are very advanced autocomplete models. And similarly, if you want to figure out which
answer is true and which is not, something that is very useful to measure is log
probabilities. So for example, we want to figure out what the word "it" is referring to here
in this sentence: "The cat couldn't fit into the hat because it was too big." What we can do
is take the sentence, replace "it" with either the cat or the hat, and then you can measure
the probability of which one

**[10:49]** the model thinks is higher, and you can get the idea what the reference here is.
So none of this is in the training data; it's simply learning to predict text. But you can
start seeing how we can leverage these models to do other tasks as well besides prediction.
So this is just more evidence about how GPT-2 — no task-specific finetuning, no task-specific
training, it simply is learning to predict text — establishes the state of the art on many,
many different tasks, simply by scaling up the model parameters and scaling up the amount of
data it's trained on. So this is a fun example: if you want to do summarization, or you

**[11:35]** have a news article that you want to summarize, how do you get a zero-shot model
to do it? The answer is you put the document into the context and you simply put `TL;DR` in
front of it. Because if most of the data on the internet, whenever you see TL;DR, you'll
naturally summarize it. So yeah, you can get zero-shot performance on summarization here as
well. And again, this is not trained to do summarization in any specific way, and it's still
doing really well simply because of its pretraining data. So GPT-2 `TL;DR` is somewhere there,
and some of the very task-specific trained models are up here. I think you will see the
trend: if you were Alec Radford or somebody, and you see these

**[12:22]** cool things emerging, your next step would obviously be, I'm going to scale this
up a little more, I'm going to make an even bigger model, I'm going to train it on even more
data and we'll see how things go, right? So that's how we got GPT-3. We went from 1.5 billion
parameters to 175 billion parameters, we went from 40GB of data to 600GB of data. Of course,
now we're in terabytes of data, and text is a very compressed representation, so terabytes of
data is a lot. And we talked about zero-shot learning; the cool thing that emerged in GPT-3
is — go ahead. *[Ed: the student's question is garbled in the captions; from the answer it
asks whether the `TL;DR` marker goes before the passage.]* No, you typically put the passage
first. If you've interacted with Reddit or something like that, typically somebody will write

**[13:08]** an entire post and then end with "TL;DR: here's a summary of the thing, too long
didn't read." *[Ed: garbled — the student says something like "if you have used [it], it comes
first".]* Oh yeah, there are situations where it also comes first. But one reason is that
these are decoder-only models, so they are often causal attention models, so they typically
need to see the context before. *I understand. I'm just curious — from my experience it comes
first.* *[Ed: the rest of the exchange is unrecoverable.]* Okay, there's probably a lot of
data where the TL;DR comes first, but there's probably a lot of data where TL;DR comes after
as well. Cool. So we saw zero-shot learning emerging in GPT-2. Few-shot learning maybe

**[13:54]** seems slightly easier, but this is where things started getting really funny:
you're starting to beat the state of the art simply by just putting examples in context. So
what does few-shot learning mean here, what are we talking about? As I mentioned, the typical
idea is that you want to solve translation, so you would put some examples of translation into
context — this is a correction task here, or maybe you're interested in translation — and no
gradient updates, no learning in any conventional sense whatsoever. You put a few examples in,
and that's it. You know how to solve the task. Isn't that crazy? You guys did the assignment
on translation, right? But this is what modern NLP

**[14:42]** looks like. So you put in some examples and you have the entire system. And this
is where things got really interesting: all these task-specific models that were created to
be really, really good at translation or really good at summarization — you can just… let's
look at this graph. So we start with a zero-shot performance in a similar fashion to what I
described earlier and you start somewhere there. You put one example in of translation from
English to French, you get to somewhere like already at a fine level. A few examples in,
you're already starting to be close to the state-of-the-art models. *Wait, but in that graph
the state of the art is really high, isn't it?* The fine-tuned version of the BERT++ here, I
think, is the one I'm

**[15:27]** referring to. The fine-tuned one, which is trained exclusively on a lot of
translation data, might be slightly better, yes. And I think that's the relevant comparison
here: the in-context learning starts to emerge at scale. And this I think is the key point —
some of this is contested, just to be very upfront, but there's this idea of emergence of
this property as you train on more compute and more scale. There's more recent research which
suggests that if we plot the axes correctly it feels less emergent. But the general idea is,
as you increase the number of parameters and increase the amount of compute that is going
into the models, the ability to go from a few examples to really strong performance is very

**[16:14]** compelling. Cool. And as I explained earlier, the general idea is that this is
very different from the conventional idea of finetuning that we typically go for. Instead of
iterating over examples, putting them in and doing gradient updates, we are actually just
going for few-shot prompting: we're going to put in a few examples and that's going to give
us the system. *(A student asks about the format.)* I mean, the exact details roughly can
depend on the prompt template that you use, but typically you would just put in these
examples, and then whatever your task is you can just let the model complete from there,
because it can infer the task

**[16:59]** based on the examples you've given. Any other questions? Cool. So we have gotten
from zero-shot prompting, and we've been seeing that few-shot prompting is becoming really
competitive with good models. But there's still limitations to this: you cannot solve every
task that you see here, and particularly things that involve richer multi-step reasoning is
something that can actually be pretty challenging. And just to be fair, humans struggle at
these tasks as well. So things like addition and so on — these are probably still hard to do
when you keep increasing the number of digits. But one thing that you have to start being
creative with — I alluded to

**[17:45]** this earlier — is that you can get these models to do the task if you're creative
in how you prompt the model, and this is what we're going to see next. So this technique
called chain-of-thought prompting emerged here. The idea that we have explored thus far is
that we put in examples of the kind of tasks we want to do and we expect the model to
zero-shot learn what the task is and go from there. The idea now is that instead of just
showing what the task is, you show them examples where they reason through the task, so
they're not just learning to do the task but also learning how the reasoning is working. So
in this example, initially we have to solve a simple math problem and we are just shown
exactly the answer directly. And if you do that

**[18:32]** directly, you'll observe that the model gets the answer wrong. Instead of that,
what if you show the model how to reason about the task, show it a chain of thought, and
include that in the prompt as well? And then you ask it a new question. The idea is that now
the model is not just going to output an answer, it's going to reason about the task, and
it's going to do actually a lot better. And this has been shown to be very effective.
Chain-of-thought is also, as you can see, something that improves a lot with model scale. But
what you can probably start seeing is it's nearly better than the supervised best models
here. So PaLM models roughly were

**[19:17]** about 540 billion parameters, and simply with this chain-of-thought kind of a
skill you're already beating the state of the art. Cool. So I showed you examples of
chain-of-thought reasoning up to this point, where you go through a reasoning chain. But you
can be even slightly smarter than that: you might not even need to show them any examples,
you just need to trick them into thinking about what to do next. So this idea emerged in this
paper where you say "let's think step by step" — instead of even showing an example, you just
start your answer with "let's think step by step", and that's it. The model will

**[20:03]** start reasoning about the answer itself instead of just autocompleting to an
answer, and you get something like this. So maybe you don't even need to show any examples —
you can probably induce the reasoning behaviour through zero-shot behaviour as well. And
again, what the final numbers look like is: compared to the zero-shot performance that we got
from essentially autocompleting, this zero-shot chain-of-thought substantially improves the
performance, so you go from 17.7 to 78.7. It's still worse than putting in examples of
reasoning — multi-shot, few-shot chain-of-thought — as well, but you can see how much it
improves the performance simply by asking it to think step by step. And maybe

**[20:48]** this is a lesson about interacting with these models. When you interact with these
models you might not get the exact desired behaviour from them up front, but often these
models are capable of doing the behaviour that you want, and often you have to think about
how to induce that behaviour. And the right way to think, perhaps, is: what is the pretraining
data, what is the data on the internet it might have seen which induces a similar behaviour to
the kind I want? You probably want to think about that and then induce these kinds of
behaviours from those models. And we hand-designed some of these prompts; you can also get an
LLM to design these prompts as well. There are

**[21:34]** recursive self-improving ideas that happen here, and you can bump up the
performance a little bit more. Cool. So what we have seen so far is that as models get
stronger and stronger you can start forcing them to do your task zero-shot or with few-shot
examples, and you can trick them into thinking about what task you want them to solve. But
the downside is that there's only so much you can fit into context. This might not be very
true any more, models are getting increasingly larger context, but it's still somewhat
unsatisfactory to think you have to trick the model into doing your task rather than it just
doing the task you wanted it to do. And potentially, going forward,

**[22:20]** you probably still want to finetune these models for more and more complex tasks,
and that's where we're going to go forward in this next section, which we're going to cover:
instruction finetuning. The general idea we have right now, as we talked about, is that
pretraining is not about assisting users, it is about predicting the next token. Now, you can
trick it into assisting users and following the instruction you wanted to, but in general
that's not what it's pretrained for. And this is an example where if you ask GPT-3 — a pretty
strong model — to "explain the moon landing to a six-year-old in a few sentences", it will
follow up with more questions about what a six-year-old might want. This is not what you
wanted the model to do, right? So the general

**[23:07]** term that people use these days is that they're not aligned with user intent, and
the next sections that we're going to cover are going to talk about how to align it with the
user intent so that we don't have to trick the model into doing whatever we wanted to do. And
this is the kind of desired completion we want at the end of instruction tuning. So how do we
get from those pretrained models to models which can respond to user intent? Again, I hope
this was covered somewhere in the class, the general idea of pretraining and finetuning. But
what you have probably seen thus far is that you pretrain on a lot of different language data,
but then you finetune on your specific task. So you're taking the

**[23:53]** same decoder-only models and you're finetuning to some task with a very little
amount of data. The thing that is going to be different now is not that we're no longer
finetuning on a little amount of data — we're going to finetune on many, many different tasks,
and we're going to just try to put them into a single usable UX for users. And this is where
instruction finetuning comes in. Cool. So again, the recipe is not very, very complicated
here. We're going to collect a lot of examples of instruction and output pairs, and the
instructions are going to range over several tasks, different forms. There's going to be
question answering, there's going to be summarization, translation, code, reasoning and so on,
and we're going to collect a lot of examples related

**[24:41]** to all those tasks. And the idea is that we'll train on instruction and output
pairs exactly with them, and then we're going to evaluate on some unseen tasks as well. So
this is the general paradigm of instruction finetuning. And again, it's the same idea which we
explored in pretraining: data plus scale is really important. These days you start off with
one task, you're now extending it over thousands and thousands and thousands of tasks with
like three million plus examples. This is generally the broad range of tasks that you might
see in instruction finetuning data sets. And you might even think, why are we calling it
finetuning any more, it's almost starting to look like pretraining? But yeah, these are just
terms, so you

**[25:28]** can decide whatever you are comfortable with. So we get this huge instruction data
set, we finetune our model, and the next question is, how do we evaluate these data sets? Now
I think you guys will see another lecture on evaluation, so I don't want to dive too deep into
this, but generally evaluation of these language models is an extremely tricky topic. There's
a lot of biases that you need to deal with and a lot of this will be covered. But some more
recent progress on this is that we are starting to curate these really large benchmarks, like
MMLU, where the models are tested on a broad range of diverse knowledge. This is just one
example, and these are the topics that you will see. And just to

**[26:14]** give some intuition of what the examples in these evaluations look like: under
astronomy you might be asked what is true for a type-Ia supernova, or you might be asked some
questions about biology, and there's a huge host of tasks for this. These are typically
multiple-choice questions and you can ask the model to answer the question — if they're
instruction finetuned already, hopefully they can simply answer the question. But you can also
chain-of-thought prompt these questions or few-shot prompt these questions too. And recently
there's been a huge amount of progress on this benchmark. What people have observed is that
more and more pretraining on more and more data and larger models is simply just climbing up
the number on this. So 90% is often seen as a benchmark number that these models wanted to
cross,

**[27:00]** because it's roughly human-level knowledge or understanding, and recently the
Gemini models purportedly crossed this number. So yeah, go ahead. *Isn't this like the entire
sort of thing all over again? Like [Ed: the benchmark named here is garbled; from context it is
almost certainly ImageNet] — at some point you're like, okay, maybe my methods are too
implicitly finetuned on it. Isn't something like that happening here as well?* Yes, I think
this is a tricky topic, because with a lot of the models there's this idea about whether your
test sets are leaking into your training data set, and there are huge concerns about that.
It's a perfectly valid question to ask. How do we even evaluate this — this is why evaluation
is actually very tricky. But one general thing to be

**[27:48]** careful about is: at some point it doesn't matter what your train/test split is, if
the models are generally useful. If the models are doing useful stuff — if you train on
everything you care about and it does well on it — does it matter? So yeah, again, we still
need better ways to evaluate the models and to understand what methods are doing and whether
they're improving the model or not, but at some point those boundaries start to be less
important. Cool. So, massive progress on this benchmark starting with GPT-2, and we're roughly
at 90%, to the point where these benchmarks are starting to become unclear as to whether
improvements on these are actually

**[28:33]** meaningful or not. In fact, most of the times when the models are wrong, you might
often find that the question itself was unclear or ambiguous. So all evaluation benchmarks have
a certain limited utility to them. So yeah, I'm going to go over another evaluation example of
how this recipe changes things. T5 models were instruction finetuned on a huge number of tasks,
and another trend — which I think will be the theme across this lecture — is that as your
models become larger, as they're trained on more data, they become more and more responsive to
your task information as well. So what you will observe here is, as the number of parameters
increases — we have T5-Small, Flan-T5-Small, and we go up to 11 billion

**[29:20]** parameters where we have T5-XXL — you'll see that the improvement actually improves.
Going from a pretrained model into an instruction model, the instruction model is all the more
better at following instructions. So the difference is plus 6.1 and goes to plus 26.6 as the
models become larger. So this is another very encouraging trend: that you probably should train
on a lot of data with a lot of compute, and pretraining just keeps on giving. So yeah, I hope
you guys get a chance to play with a lot of these models — I think you already hopefully are.
But before instruction finetuning, when you're asked a question related to disambiguation QA,

**[30:07]** you get something like this, and it doesn't actually follow the "let's think step by
step" instruction very clearly. But after instruction finetuning it is able to answer the
question here. And more recently people have been researching into what the instruction tuning
data set should look like. There's a huge plethora of instruction tuning data sets now
available; this is just a representative diagram, and there's a huge open-source community
developing around these as well. Some high-level lessons that we have learned from this: one
lesson that I think might be interesting is that we can actually use really large, strong
models to generate some of the instruction tuning data to train some of our smaller models. So
take your favourite model right now, GPT-4 maybe,

**[30:54]** or maybe Claude or whatever, and you can get it to answer some questions and
generate instruction-output pairs for training your open-source or smaller model, and that
actually is a very successful recipe. So instead of getting a human to collect all the
instruction-output pairs, or getting humans to generate the answers, you can get bigger models
to generate the answers as well. So that's the number one thing that has recently emerged.
Another thing that is emerging, or is being discussed, is how much data do we need. I talked
about millions of examples, but people have often found that if you have really high-quality
examples you can get away with a thousand examples as well. So this is the paper *Less Is More
for Alignment*, and this is still an active area of research: how data scaling in instruction
tuning affects the final model performance. And yeah, crowdsourcing these

**[31:42]** models can be effective as well, so there are very cool benchmarks that are
emerging like OpenAssistant. A lot of activity in the field, and hopefully a lot more progress
as we go on. *Yes — a question sort of in the spirit of this LIMA paper. Don't code or math word
problems have this desired structure? So shouldn't we just be training code models and doing
some English stuff and then be like, okay, this is the best reasoning we can get at some point?
Because code has the structure where you're going step by step and you're sort of thinking in
some way, in like*

**[32:27]** *a — so, breaking down a concept into a smaller [one]. So you can consider that code
has very high-value tokens, so maybe just doing [that].* So I think — again, pretraining is a
whole dark art that I am not completely familiar with — but code actually ends up being really
useful in pretraining mixtures, and people do up-weight code data quite a lot. Similarly, but
it depends upon what the users are going to use the models for, right? Some people might use it
for code, some people might use it for reasoning, but that's not the only task we care about. As
you might see later on in the next step, we'll discuss this as well: people often use these
models for creative tasks. They want to write a story, they want to generate a movie script and
so on, and I don't know if

**[33:13]** necessarily training on reasoning-only tasks would help with that. So go ahead. *Yeah,
would you explain — there exists some data distribution which is high-value for creative tasks?*
Yes. A lot of people write about stories and everything on the internet all the time, which is
not code. And sometimes there's this idea of hallucinations as well in this field, but you can
often think, hey, creativity might be a byproduct of hallucinations as well. So I don't know
what exact data would lead to more creative models, but generally there's a lot of data, a lot
of stories that are written on the internet, which allows the model to be

**[33:59]** creative. But I don't know if I have a specific answer to the question. Cool. So we
discussed instruction finetuning — very simple and very straightforward, there's no complicated
algorithms here, just collect a lot of data — and then you can start leveraging the performance
at scale as well: as models become better, these models also become more easily specifiable and
they become more responsive to tasks as well. We're going to discuss some limitations, and I
think this is really important to understand why we are going to optimize for human preferences.
Cool. So, instruction finetuning is necessarily contingent on humans labelling the data. Now,
it's expensive to collect this

**[34:46]** data, especially as the questions become more and more complex. You want to answer
questions which may be at physics PhD level or things to that effect; these things become
increasingly expensive to collect. So this is maybe perhaps obvious: collecting data for
pretraining does not require any specific data, you scrape data off the web, but for instruction
finetuning you probably need to recruit some people to write down answers to your instructions.
So this can become very expensive very quickly. But there are more limitations to this as well,
and we were just discussing this: there are open-ended tasks related to creativity that don't
really have an exact correct answer to begin with. So how do you generate the right answer to
that kind of a question? And language modelling

**[35:34]** inherently penalizes all token-level mistakes equally — this is what supervised
finetuning does as well — but often not all mistakes are the same. So this is an example where
you're trying to do this prediction task, "Avatar is a fantasy TV show", and perhaps you can see
that calling it an "adventure" TV show is perhaps okay, but calling it a "musical" may be a much
worse mistake, and both these mistakes are penalized equally. And I think one general aspect
which is becoming increasingly relevant is that the humans that you might ask might not generate
the right or the highest-quality answer. Your models are becoming increasingly competitive, and
in some sense

**[36:19]** you're going to be limited by how high-quality the answer is that humans can generate.
But often I find that the models are generating better and better answers, so do we really want
to keep relying on humans to write down the answers, or do we want to somehow go over that? So
these are the three problems we have talked about with instruction finetuning, and we made a lot
of progress with this, but this is not how we got ChatGPT. And one high-level problem here is
that even when we are instruction finetuning, there is still a huge mismatch: the end goal is to
optimize for human preferences, generate an output

**[37:04]** that a human might like, and we're still doing prediction kind of tasks where we're
predicting the next token, but now in a more curated data set. So that's still a bit of a
mismatch going on here, and it's not exactly what we want to do. I'm going to take a second here
to pause, because this is important to understand the next section, and if there's any questions
feel free to ask. *So is this step still taken as a first step, or do we discard this?* It's a
good question. I think this is still one of the more important steps that you take before taking
the next step, but people are trying to remove the step altogether and jump directly to the next
step. So there's work emerging on that. But yeah, this is still a very

**[37:51]** important step before we do the next step. Go ahead. *Is Problem 2 also present in
pretraining, and if so, how do you avoid that — just by having a lot of data?* Yeah, that's a
great question. There's one major difference in pretraining: pretraining covers a lot more text.
So just for context, as we talked about, pretraining is roughly 15 trillion tokens, whereas
supervised instruction finetuning might be somewhere on the order of millions to billions of
tokens, so it's a few orders of magnitude lower. Typically you'd only see one answer for a
specific instruction, but during pretraining you'll see multiple texts and multiple completions
for the same kind of a prompt.

**[38:39]** Now that's good, because when you see multiple answers or completions during
pretraining you sort of start to weigh different answers, you start to put probability mass on
different kinds of answers or completions, but instruction finetuning might force you to put
weight on only one answer. But generally, yeah, this is a problem with both the stages, you're
right. Anything else? Cool. So, as this whole thing alludes to, we're going to start to attempt
to satisfy human preferences directly. We're no longer going to try to get humans to generate
some data and try to do some kind of a token-level prediction loss; we're going to try to
optimize for

**[39:24]** human preferences directly, and that is the general field of RLHF, and that's the
final step in typically getting a model like ChatGPT. So we talked about how collecting
demonstrations is expensive, and there's still a broad mismatch between the LM objective and
human preferences, and now we're going to try and optimize for human preferences directly. So
what does optimizing for human preferences even mean? To concretely establish that, let's go
through a specific example, which is summarization. We want to train a model to be better at
summarization and we want to satisfy human preferences. So let's imagine that a human is able to
prescribe a reward for a specific summary. Let's just pretend there is a reward function — you
and I can assign, say, a reward, this is

**[40:11]** plus one, this is minus one, or something to that effect. So in this specific case we
have this input x, which is about an earthquake in San Francisco — so this is a news article that
we want to summarize — and let's pretend that we get these rewards and we want to optimize this.
So we get one summary y_1 which gives us "an earthquake hit" and so on, and we assign a reward of
8.0, and another summary which gives us a reward of 1.2. Generally speaking, the objective that
we want to set up is something of the following form, where we want to take our language model
p_θ, which generates a completion y given

**[40:57]** an input x, and we want to maximize the reward R(x, y), where x is the input and y is
the output summary in this specific task. And maybe just to really concretely point out something
here: this is different from everything that we have done in one very specific way. We are
sampling from the model itself. In the bottom term, if you see, we're using y from p_θ. In
everything we've seen so far, the data is sampled from some other source — either during
pretraining or in supervised finetuning — and we're maximizing the log likelihood of those tokens.
But now we're explicitly sampling from our model and optimizing a potentially non-differentiable
objective. Cool. So broadly the RLHF pipeline looks

**[41:46]** something like this. The first step is still instruction tuning, something we have seen
up until now, where we take our pretrained model, we instruction tune on a large collection of
tasks, and we get something which starts responding to our desired intent, or not. But there are
two more steps after this which are typically followed in creating something like InstructGPT.
The first step is estimating some kind of a reward model, something which tells us, given an
instruction, how much would a human like this answer or how much would a human hate this answer.
So we looked at something like this earlier, but I didn't talk about how we even get something
like that — that's the second step. And then we take this reward model and we optimize it through
the optimization that I suggested earlier, so maximizing the expected reward under your language

**[42:32]** model. And we're going to go over a lot in the second and third steps. So the first
question we want to answer is, how do we even get a reward model? What are humans going to like?
This is a very ill-defined problem generally speaking. So there are two problems here that we're
going to address. First is, a human in the loop is expensive. Let's say if I ask a model to
generate an answer and then I get a human to label it with some kind of a score, and I'm doing
this over millions of completions — that is not very scalable. I don't want to sit around and
label millions of examples. So this is very easy — we're in a machine learning class, so what are
we going to do? What we're going to do is we're going to train something which

**[43:18]** predicts what a human would like or what a human might not like. And this is
essentially a machine learning problem where we take these reward scores and we try to train a
reward model to predict, given an input and output, what the reward score would look like. Simple
machine learning regression-style problem, you might have seen this earlier. Cool. Now there's a
bigger problem here — sorry, go ahead. *Do we use, like, I don't know, just embeddings, or do we
use a real language model to do that?* That's a good question. Generally what we do is, we still
typically need reward models where they need to be able to understand the

**[44:04]** text really well, so bigger models, and they're typically initialized from the language
model that you pretrained as well. So you typically start with the pretrained language model and
do some kind of prediction that we'll talk about, and they'll give you a score. *If you're doing
that, how do you separate x and y? How does the language model know which part it doesn't need to
—* It can put the x and y — it only sees x and y as an input, so it doesn't need to typically see
it separated, it's just going to predict a score at the end. *Okay.* Yeah, the x and y is more for
notational convenience here, because for us x and y are different: x is a question the user asked
and y is something the model generated. *But you shove the whole thing in—* You shove the whole
thing in, yes. Cool. Now, this is the bigger problem here:

**[44:51]** human judgments are very noisy. We have talked about how we want to assign a score to a
completion. This is something that's extremely non-trivial to do. If I give you a summary like
this, what score are you going to assign on a scale of 10? If you ask me on different days I'll
give a different answer, first of all. But across humans itself this number is not calibrated in
any meaningful way, so you could assign a number of 4.1, 6.6, and different humans would just
simply assign different scores. And there are ways to address this — you can calibrate humans, you
can give them a specific rubric, you can talk to them — but it's a very complicated process, and
still there's a lot of room for judgment, which is not typically very nice for training a model
like this. If your labels can

**[45:37]** vary a lot, it's just hard to predict. So the way this is addressed is that instead of
trying to predict the reward label directly, you actually want to set up the problem in a slightly
different way. Something that's much easier for humans to do is to give them two answers, or maybe
many answers, and ask them which one is better. So this is where the idea of asking humans to rank
answers comes in. If I give you a whole news article and ask you which summary is better, you might
be able to give me a ranking: oh, this second summary is the worst, but the first one is better,
and the third one is somewhere in the middle between those two. So you get a ranking which gives
you a preference over summaries. And hopefully you can see the idea that is important

**[46:24]** here: even when we have some kind of a consistent utility function, it's much easier to
compare something and know which is better than to ascribe it an arbitrary number on a scale, and
that's why the signal from something like this is a lot better. Now, we talked about how we get
this kind of preference data, and now we need some kind of a reward score out of this. We shove in
our input, we shove in a summary as well, and we still need to get a score out of this. But it's
not clearly obvious to me how we take this data and convert it into that kind of score. In come our
pretty good friends named Bradley and Terry. Essentially

**[47:10]** there's a lot of study in economics and psychology which basically tries to model how
humans make decisions. In this specific case, the Bradley–Terry model essentially says that the
probability that a human chooses answer y_1 over y_2 is based on the difference between the rewards
that humans assign internally, and then you take a sigmoid around it. So if you have looked at
binary classification before, the logit is simply the difference between the reward of y_1 minus
y_2, or the difference between the winning completion and the losing completion. Is everybody with
me till this point?

**[47:57]** So the idea is that if you have a data set where, given y_1 and y_2 — where we have a
winning completion y^w and a losing completion y^l — the winning completion should score higher
than the losing completion. Go ahead. *Sorry, what is J? Is that a log, or — sorry, what is the
type of J? Like this number here that we're getting as the expectation, is it a log prob or what is
it?* It's a log probability, so it will be a scalar at the end. The sigmoid — so you're taking, let's
say you have a reward model which gives a score R_1 to y^w and R_2 to y^l; you subtract that number,
you get another number, you put it into the sigmoid and you

**[48:42]** get a probability, because the sigmoid will convert a logit into a probability. And then
you take a logarithm of that, and you take the expectation of everything, and you get this final
number which tells you how good your reward model is doing on the entire data set. So a good model
of humans should behave like this: a good model of humans would score very low here, so it would
generally assign a higher reward to the winning completion and generally assign a lower reward to
the losing completion. Cool. The math is just beginning, so hold on to your seats. Cool. So now
let's see where we are. We have a pretrained model p^PT(y | x),

**[49:28]** and we've got this fancy reward model which tells us that hey, we have a model of humans
and it can tell us which answer they liked and which answer they did not like. Now, to do RLHF —
generally we have discussed what this will look like — we will copy our pretrained model or
instruction-tuned model and we'll optimize the parameters for those models. And I suggested that the
objective that we want to optimize is the expected reward when we sample completions from p_θ, and
we're going to optimize our learned reward model instead of the true reward model which humans would
have typically assigned. Do you guys see any problem with this? Is there something that's wrong here

**[50:14]** or that might go wrong if we do something along these lines? Go for it. *It might
collapse.* Yes. Okay. Generally, at least from my intuition, if you're ever doing something and
you're optimizing some learned metric, I'd be very careful, because typically our loss functions are
very clearly defined, but here my reward model is learned. When it's learned it means it will have
errors. It's going to be trained on some distribution, it will generalize as well, but it will have
errors, and when you're optimizing against a learned model it will tend to hack the reward model. So
it might exploit the reward model — it might

**[51:02]** erroneously assign a really high score to a really bad completion, and if your policy
learns, or if your language model learns to do that, it will completely hack that and start
generating those gibberish completions. So just as a general machine learning tip as well: if you're
optimizing a learned metric, be careful about what you're optimizing and make sure that it's actually
reliable. And this is obviously not desirable — if you start optimizing this objective you're going to
converge to gibberish language models very, very quickly. So typically what people do is that you want
to add some kind of a penalty that avoids it drifting too far from its initialization. And why do we
want to do that? If it cannot drift too far from its initialization, we know the initialization of the
model is a decent

**[51:47]** language model, and we know it is not necessarily satisfying this reward model too much,
and we also know that the reward model is trained on a distribution of completions where the initial
model is. So typically when we talk about training this reward model we have trained on certain
completions which are sampled from this initial distribution, so we know the reward model will be
somewhat reliable in that distribution. So we're simply going to add a penalty which tells us that you
should not drift too far away from the initial distribution. And just to go over this: we want to
maximize the objective where we have RM_φ, which is our learned reward model, but we're going to add
this term, beta log ratio, and the ratio is the model we're optimizing, p^RL_θ,

**[52:33]** and our initial model p^PT. And what this says is that if we assign a much higher
probability to a certain completion as compared to our pretrained model, you're going to add an
increasingly large penalty to it, and simply, you're paying a price for drifting too far from the
initial distribution. If you guys have taken machine learning, the expectation of this quantity is
exactly the Kullback–Leibler divergence, or KL divergence, between p^RL_θ and p^PT. So you're
penalizing drifting between two distributions. Go ahead, question. *Shouldn't you also add a penalty
like this in the previous version, where you did finetuning, or is this only relevant for the RLHF?*
That's a good question. So I think

**[53:21]** people do add some kinds of regularization in finetuning; it's just not nearly as
critical as when you're doing this with RL, where the incentive is to exploit this reward model as
much as possible. And we'll see examples where the learned reward predicts that it's doing really
well but the true reward models are completely garbage, so it's much more important in this
optimization. Cool. So now — this course does not assume background on reinforcement learning, so
we're not going to go into reinforcement learning, but I just want to give a very high-level
intuition about how this works. Reinforcement learning is not typically just used for language
models, it's been applied to several domains of interest: game-

**[54:07]** playing agents, robotics, developing chip designs and so on. And the interest in RL and
LMs also dates back to roughly 2016 as well, but it's been really successful recently, and especially
with the success of RLHF. The general idea is that we're going to use our model that we're optimizing
to generate several completions for an instruction, we're going to compute the reward under our
learned reward model, and then we're going to simply try and update our model to increase the
probability on the high-reward completions. So when we sample a model we'll see completions of varying
quality — we'll see some good completions, good summaries for our task, some bad summaries for our
task — and

**[54:53]** we'll try to update our log probabilities in a way such that the reward when you use an
updated model is typically in the higher-reward region. Does the high-level summary make sense? Cool.
And RLHF is incredibly successful. I think this is a very good example — this is the same
summarization example — and I think the key point here is that the performance improves by increasing
the model size, for sure, we have seen this in many different examples. What you can actually see is
that even very small models can outperform human completions if you train it with RLHF, and this is
exactly the result you see here. The reference summaries are human-generated,

**[55:39]** and when you evaluate, when you ask humans which ones they prefer, they often prefer the
model-generated summary over the human-generated summary. And this is something you only observe with
RLHF, even at small scales. And again, the same scaling phenomenon still holds here — bigger models do
become more responsive — but RLHF itself is very impactful here. Cool. The problem with RLHF is that
it's just incredibly complex. I gave you a very high-level summary; there are whole courses on this for
a reason. And this image is not for you to understand, it's just completely to intimidate you. So, you
want to fit a value function to something, you have to sample

**[56:25]** the model a lot, it can be sensitive to a lot of hyperparameters, so there's a lot that
goes on here. And if you start implementing an RLHF pipeline it can be very hard, and this is the
reason why a lot of RLHF was restricted to very, very high-compute, high-resource places and it was not
very accessible. So what we're going to talk about and cover in this course is something called direct
preference optimization, which is a much simpler alternative to RLHF and hopefully much more
accessible. But please bear with me, there will be a lot of math here, but the end goal of the math is
to come up with a very simple algorithm. So hopefully — and feel free to stop me and ask me questions
if you need. *In terms of GPT-*

**[57:13]** *4 versus 3, how much do the number of parameters in the base model help with — do you need
to reduce the number of… sorry, the number of examples from humans for RLHF to work well?* *[Ed: the
question is heavily garbled in the captions; the reading above follows the answer.]* Yeah, that's a
really good question. So generally speaking, if you hold the data set size constant and simply increase
the model size, it will improve quite a lot, sure. But the nice thing is that you can reuse the data and
you can keep adding data as you keep scaling models up. So typically nobody tries to reduce the amount
of data collection, right — you just keep increasing both things. Cool. So we talked about RLHF and the
current pipeline is something like: we train a reward model on the

**[57:59]** comparison data that we have seen so far, and we're going to start with our pretrained or
instruction-tuned model and convert it into an RLHF model using the reinforcement learning techniques.
Now, really the key idea in direct preference optimization is: what if we could simply write a reward
model in terms of our language model itself? Now, to intuitively understand what is going on: a
language model is assigning probabilities to whatever is the most plausible completion next, but those
plausible completions might not be what we intended. But you could restrict the probability simply to
the completions that a human might like, and then the log probabilities of your model might represent
something which the humans might like and not just some arbitrary completion on the internet. So there
is a direct correspondence between the log

**[58:46]** probability that a language model assigns and how much a human might like the answer — they
can have a direct correspondence between them. And this is not some arbitrary intuition that I'm trying
to come up with, we will derive this mathematically. So the general idea with direct preference
optimization is: we're going to write down the reward model in terms of our language model, and now
that we can write our reward model in terms of our language model we can simply directly fit our reward
model to the preference data we have and we don't need to do the RL step at all. So we started off with
some preference data and we simply fit our reward model to it, which directly optimizes the language
model parameters. And maybe at a high level, why

**[59:31]** is this even possible? We did this really cumbersome process of fitting a reward model and
optimizing it, but in the whole process the only external information that was being added to the
system was human labels — labels on the preference data. When we optimize a learned reward model
there's no new information being added into the system. So this is why something like this is even
possible. For quite a few years this was not obvious, but as you will see, some of these results start
to make sense. So we're going to derive direct preference optimization. I'll be here after the class as
well if you have questions, but hopefully this is clear. So, we discussed that we wanted to solve this
expected reward problem where

**[1:00:17]** we want to maximize the expected reward but we subtract this term, which is the beta log
ratio, which essentially penalizes the distance between where our current model is and where we started
off, so we don't want to drift too far away from where we started. Now, it turns out that for this
specific problem, instead of doing an iterative routine, there's actually a closed-form solution to
this problem. So the closed-form solution looks something like this. Again, if you have seen the
Boltzmann distribution or something to that effect before, this is basically the same idea. But the
idea is this: we're going to take a pretrained distribution p^PT(y | x) and we're going to reweight the
distribution by

**[1:01:02]** the expected reward. So if a completion has a very high reward it's going to have a
higher probability mass, and if it has a lower reward it's going to have a lower probability mass, and
it's determined by the expected reward. And beta is a hyperparameter which essentially governs what the
trade-off is between the reward model and the constraint, and as beta becomes lower and lower you're
going to start paying more and more attention to the reward model. So the probabilities look something
like this, and there's this really annoying term, this Z(x). The reason why it exists is that the
numerator by itself is not normalized, it's not a probability distribution, so to construct an actual
probability distribution you have to normalize it,

**[1:01:48]** and Z(x) is simply just this normalization. So if we write Z(x) out, it's the sum over all
y. *(A student comments.)* Okay, yeah, and that's exactly it — it's a sum over all ys for a given
instruction, and that's exactly why this is very pesky: it's intractable. If I take an instruction and
try to sum over every possible completion — and not just syntactically correct ones, every single
possible one — we have 50,000 tokens, maybe even more, and the completions can go arbitrarily long, so
this space is completely intractable. This quantity is not easy to approximate, even. So the main point
here is that if you're given a reward model, there does exist at least a closed-form solution which
tells us what the optimal policy will look like, or optimal language model will look

**[1:02:33]** like. But if you do a little bit of algebra — just move some terms around, take a
logarithm here or there, I promise this is not very complicated — you can actually express the reward
model in terms of the language model itself. And I think this term is reasonably intuitive as well.
What it says is that a completion ŷ has a high reward if the model, my optimal policy, assigns a higher
probability to it relative to my initialized model, and this is scaled by beta. So the beta log ratio is
what we're looking at here, and the partition function — let's just ignore it for now, but it's
intractable — but the beta log ratio is the key part here. Is everyone following

**[1:03:18]** along? Awesome. Okay, so right now I'm talking about optimal policies, but really every
policy is probably optimal for some kind of a reward, right? This is mathematically true as well. So
the important bit here is that you can actually take a current policy, take your initialized model, and
you can get some kind of a reward model out of it, and this is the exact identity which leads to this.
So a reward model can be expressed in terms of your language model, barring the log partition term,
which we'll see what happens to. Go for it. *Sorry, I don't know how you got — why is it that we can
swap? Because there is a thing that we're trying to optimize, and how does p\* turn into p?* Yeah.

**[1:04:04]** For now, we're not optimizing any reward model, okay? All I'm saying is that if I take my
current language model, it probably represents some kind of a reward model implicitly, because of this
relationship, because this holds for every p\* and every reward model. What I'm saying is that if I plug
in my current language model, it also represents some kind of a reward model. I'm not saying it's
optimal. *Okay, but I want to say — because at the beginning p^RL is p^PT—* Yes. *—and so we just get
that the reward is basically zero, and so what do we do?* Initially it's zero, but we can optimize the
parameters. *Okay, okay.* Yeah, but that's a good observation, that it's basically zero in the beginning.
But how do we start optimizing

**[1:04:50]** it? I'll get to that. Okay. Any other questions? *So the idea is, given the language
model, you have a [reward] model such that — that makes the language model op[timal]?* *[Ed: the end of
the question is cut off in the captions.]* Yes, that's the next step, yes. But the key idea is that my
language model's probabilities already implicitly define a reward model. I think that's really the main
point here, and this mathematical relationship is exact. Cool. Now, I'm obviously ignoring the elephant
in the room here, which is the partition function. It's not going to magically vanish away. If this was
just the beta log ratio that would be really nice — I can

**[1:05:35]** compute all these quantities. I know how to compute the log probability under my language
model, I know how to compute the log probability under my pretrained model, and I can compute the reward
score and I can optimize this. But I don't know what to do about my log partition function. This is
where something fun happens. So recall what the reward modelling objective was when we started off. We
started off with our friends Bradley and Terry again, and what we really wanted to optimize was the
reward difference between the winning completion and the losing completion. And really, that's all we
care about. We don't care about the exact reward itself; what we care about is maximizing the difference
between the winning and losing completion, and that's

**[1:06:21]** actually really key here. Because if you plug in the definition of RM_θ there, what you'll
observe is that the partition function actually just cancels out. Now, why does it cancel out? The
input is exactly the same, the x is exactly the same in the difference, so the partition function Z(x)
will just cancel out — it's the same in both the terms. So what you get is that the reward difference
between the winning and losing completion is the difference between the beta log ratio for the winning
and losing completion. You can plug in the terms, you can work it out, it's fairly simple. So the
partition function, which was something we could not address, we could not compute, actually simply
vanished away. *I'm sorry — Z*

**[1:07:07]** *doesn't appear in the reward model, but it appears here in this equation. So how does [it]
plug in [to the] model?* *[Ed: the question is partly garbled.]* So we're going to take this equation, the
last line that you see, and we're going to plug it in in place of RM_φ. *Okay, so in the first loss
equation?* Oh I see, got it. Yeah, so the first loss equation is the Bradley–Terry loss model. Cool. So
this really is it. The key observation is we could express our reward model in terms of the language
model, and our problems with the partition function actually go away because we were optimizing the
Bradley–Terry model. And what you get is something like this: we're going to

**[1:07:55]** express the loss function directly in terms of our language model parameters θ, and we're
going to be able to directly optimize on our data without doing any RL steps. And this is simply a
binary classification problem, so we're really just trying to classify whether an answer is good or bad,
and that's really what we're doing. Before I go on — do people want to absorb this, do they feel they're
okay with it? *I don't get where the y-good and the y-win and the y-lose come from. Are they human [or
model-generated]?* Good question. It's the same data set we started with in RLHF as well, but the way the
process works is that you take a set of instructions and

**[1:08:40]** get the model to generate some answers, and then you get humans to label which answer they
prefer. So they're model-generated typically — they can be human-generated as well, but they're typically
model-generated — and then you get some preference labels. Okay? All you need is a label saying which is
a better answer. *What do you lose here? Like, you must be losing some information because of the lack of
information about the partition function — you're cancelling out your… because of the lack of any
information about the partition function, yeah, you are bound to lose information about other possible
completions which you would have taken into account in standard RLHF, right?*

**[1:09:26]** That's a really good question. I don't think I'll be able to completely answer this question
in time, but the partition function is almost kind of a free variable. So I think the point here is that
there are many reward models that satisfy this optimization, so there's a free variable here that you can
actually completely remove, and that's what this optimization benefits from. So think of it this way: if
I assign something a reward of plus one and assign something a reward of minus one, that's basically the
same as saying it's a reward of plus 99 — it will give you the same loss, right? So that scale, that
shift invariance in a way— *Is that somehow like not what you*

**[1:10:11]** *want, though? Like, okay, if you're actually training a reward model, plus one versus plus
99 — you should pay much less attention to that as compared to like one versus zero or something.* What
we're assuming, our choice model here, is that if a human prefers something over the other, the
probability is governed only by the difference between the rewards. So that's an assumption that every
RLHF also makes, and DPO also makes. Now, is that assumption true? Not completely true, but it holds to a
fairly large degree. But that's a good question, yeah. Cool. I'll move on, in the rest of the time, and
really the goal

**[1:10:58]** of this plot is that we actually get fairly performant models when we optimize things with
DPO. In this plot I think the main thing that you should look at is PPO, which is the typical RLHF
pipeline, and we are evaluating the models for summarization and we're comparing to human summaries. And
what we find is that DPO and PPO sort of do similarly — you're really not losing much by just doing the
DPO procedure instead of RLHF, and that's really compelling, because DPO is simply a classification loss
instead of a whole reinforcement learning procedure. So I want to quickly summarize. What we have seen
thus far is that we want to optimize for human preferences, and the way we do this is that instead of
relying on uncalibrated scores we're getting comparison data and feedback on that, and we use this ranking

**[1:11:46]** data to either do something like RLHF, where we first fit a reward model and optimize using
reinforcement learning, or we do something like direct preference optimization, where we simply take the
data set and do a classification loss on that. And there are trade-offs in these algorithms. When people
have a lot of computational budget they typically maybe go for RLHF or some routine like that, but if
you're really looking to get the bang for your buck, you might want to go for DPO, and that's probably
going to work out of the box. It's still an active area of research, people are still trying to understand
how to best work with these algorithms, so I'm not making any strong claims here, but both of these
algorithms are very effective. DPO is just much simpler to work with.

**[1:12:32]** Cool. So let's see, we went through all this instruction tuning, RLHF — what do we get?
InstructGPT is the first model which sort of followed this pipeline, it defined this pipeline. So we got
models which did 30,000 or so tasks. Remember when we were doing only one task, and now we have scaled it
up from a thousand tasks to 30,000 different tasks with many, many different examples. So that's where we
are with InstructGPT, and it follows this pipeline that we just described. In this case they're following
a specific RLHF pipeline where we explicitly fit a reward model and then do some kind of a reinforcement
learning routine on top of it. And the tasks collected from labellers look something like this — I leave
it to

**[1:13:18]** your imagination, or you can look at the details. But how we started off with this model was
something like the completions we see from GPT-3, which, you know, explained the moon landing to a
six-year-old, and it is not really following the instructions, but InstructGPT will give you something
which is meaningful. So it's inferring what a user wanted from the specific instruction and it's
converting it to a realistic answer that a user might like. And these are just more examples of what an
InstructGPT-like model would do, whereas your base model might not follow the instructions to your desired
intentions. And we went from InstructGPT to ChatGPT, and it was essentially this pipeline. The key
difference here is

**[1:14:04]** that it is still doing the instruction tuning, but it is more optimized for dialogue, more
optimized for interacting with users. So the core algorithmic techniques that we discussed today are what
give us ChatGPT, but you have to be really careful about the kind of data you're training on, and that's
really the whole game. But this is the foundation for ChatGPT, and it follows the same pipeline as well.
And you might interact with ChatGPT — I'm sure you all have interacted with it in some form or another —
but this is an example of what a ChatGPT interaction might look like. You want to make a Gen Z… you know,
the idea here is that it's very good at responding to instructions and intent. This is not

**[1:14:49]** something that we could even few-shot in very easily; these kinds of instructions are hard to
come up with examples for, and this is probably not something it's trained on either, but it's able to
infer the intent and generalize very, very nicely, and that's something I find personally very remarkable.
Cool. And there's been a lot of progress on the open-source front as well. DPO is much simpler and much
more efficient, and essentially all the open-source models these days are using DPO. So this is a
leaderboard that is maintained by Hugging Face, and 9 out of 10 models here are trained with DPO. So
that's something that has enabled the open-source community to instruction tune their models better as
well, and the same is being used in many production models now

**[1:15:36]** as well. Mistral is using DPO, Llama 3 used DPO — so these are very, very strong models which
are nearly GPT-4 level and they're also starting to use these algorithms as well. And something that's very
cool to see is — we went through all this optimization and math and stuff, but what is really fundamentally
changing in the behaviour? I think this is a really good example: if you simply ask an instruction and ask
for an SFT output from an instruction-tuned model you'll get something like this, but when you RLHF the
model you actually get a lot more detail in your answer, and it'll probably organize the answers a little
better. And that's something that maybe humans prefer, that's why it's a property that is emerging in these
models, but it's something that's a very

**[1:16:21]** clear difference between simply instruction-tuned models and models which are RLHF-trained.
So yeah, we discussed this whole RLHF routine where we are directly modelling the preferences and we are
generalizing beyond labelled data, and we also discussed that RL can be very tricky to correctly implement,
though DPO sort of avoids some of these issues. And we briefly also touched upon the idea of reward models
and reward hacking. When you're optimizing for learned reward models you will often see this — this example
is that there's a way for it to just simply crash into the objects, keep repetitively

**[1:17:08]** crashing the boat to get more and more points. That wasn't the goal of this game. So this is a
very common example that is shown for reward hacking: if you do not specify rewards well the models can
learn weird behaviours which are not your desired intent, and that's something a lot of people worry about
as well. Part of the reason is reinforcement learning is a very strong optimization algorithm, it's at the
heart of AlphaGo and AlphaZero, which result in superhuman models, so you have to be careful about how you
specify things. And the other thing is, even optimizing for human preferences is often not the right thing,
because humans do not always like things which are in their best interest. So something that emerges is that
they like authoritative and helpful answers, but they often don't necessarily like

**[1:17:54]** truthful answers. So one property that happens is that they'll prefer authoritativeness more
than correctness, which is maybe not something nice. Please go ahead. *On those lines, I'm curious if maybe
ChatGPT being so widely used by the public will maybe change how people made the rewards. Because I at
least feel like now when I go to ChatGPT it gives me five detailed paragraphs of information, and sometimes
I'm just annoyed by that, that's not what I wanted. But maybe in the original reward function the original
people actually preferred that, and now they prefer it less.* Yeah, that's a great point, because as these
models integrate more and more into our system they're going to collect more and more data and they will
pick up on things, maybe undesirable things as well.

**[1:18:40]** As far as I understand, ChatGPT is really cutting down on the verbosity, which is a huge issue
that all of these models are trying to cut down on, and they are dealing with that. Part of the reason why
that emerges is that when you collect preference data at scale, people are not necessarily reading the
answers — the Turkers might just simply choose the longer answer, and that's a property that actually goes
into these models. But hopefully these things will improve over time as they get more feedback. And yeah,
hallucinations are not a problem that is going to go away with RL, and we talked a bit about reward hacking
as well, and biases from things and so on. But hopefully what I want to conclude with is: we started with
pretrained models, we had these things which could predict text, and we got ChatGPT, and hopefully it's a
little more

**[1:19:26]** clear how we go from something like that to ChatGPT. And that's — I'll end here. Thanks.
