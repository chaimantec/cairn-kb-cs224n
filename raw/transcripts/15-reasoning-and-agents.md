---
title: Reasoning and Agents
lecture: 15
video: https://www.youtube.com/watch?v=I0tj4Y7xaOQ
source: edited from auto-generated YouTube transcript
verbatim: false
original: original/15-reasoning-and-agents.md
slides: ../slides/15-reasoning-and-agents.md
---

# Reasoning and Agents — transcript

Copy-edited for punctuation, sentence boundaries, and mis-heard terms; cross-checked against
`raw/slides/15-reasoning-and-agents.md`. The verbatim auto-generated captions are kept at
`raw/transcripts/original/15-reasoning-and-agents.md`. Lecturer is Shikhar Murty; the deck and
video call this lecture 14, while the Cairn catalog lists it at position 15, which is why this
file is named `15-`. Student questions and comments from the floor are set in *italics*.
Timestamps mark the start of each paragraph; all 82 are preserved in order.

**This is an edited transcript.** The captions had no punctuation and mangled a lot of the
technical vocabulary this lecture is built on: *rationale(s)* arrived throughout as "rational(s)";
*chain-of-thought* as "Chain of Thought"; "Let's think step by step" (the zero-shot-CoT trigger
phrase) as "let's things step by step" / "lets things step by step"; *variety* as "VAR"; *Orca* as
"orai"; *GPT-4* as "gbd4", "gp4" and "gpd4", and *GPT-4V* as "gbd4 V"; *ChatGPT* as "chat GPT" and
"chat GPD"; *FLAN-v2* as "flan V2"; *BigBench-hard* as "big bench hard", "Big bench heart" and
plain "Big bench"; *sub-tasks* as "subas"; *GSM8K* as "gsm 8K"; *grade-school math* as "great
School math"; the *MATH* benchmark as plain "math"; *Llama* as "llama"/"Lama"; *Vicuna* as "wuna";
*PaLM* as "Palm"; *ReST* (reinforced self-training) as "rest"; *counterfactual* as "counteract" and
"contactual"; *axioms* as "aums"; *deductive* as "dedu Ive"; *postconditions* as "post conditions";
*MiniWoB++* as "mini wob"; *WebArena* as "web Arina"; *WebLINX* as "web links"; *relabeler* as "Rel
laaber"; *coarse filter* as "course filter"; *bespoke* as "thepoke"; *few-shot* as "few short"
throughout; *LLaVA* as "lava"; *CLIP* as "clip"; *Pix2Struct* as "pix to struct" and "pix to
struck"; *text decoder* as "Texton decoder"; *DOM* as "Dom"; *masked-out* as "marked outout";
*gears* as "gar" ("switch gar"); *pre-LLMs* as "pre-ms"; *understood* as "under OD"; *"what states
border Texas"* as "what state botherers Texas"; and the worked letter-sequence examples `ABCDE`
and `ABCDF` as "AB bcde e" and "ABCD F". Terms were restored from context and checked against the
slides. **No content was added, removed, or reordered** — including one place (the inductive-
reasoning example at 1:38–2:25) where the captions drop the word "bird" from "the creature is
likely to be [bird]"; slide 6 gives the missing word ("Conclusion: The creature is likely to be a
bird"), so it is restored rather than left as a dangling clause.

**No numbers were corrected against the slides.** Every count the lecturer gives that the slides
can check — 23 BigBench-hard sub-tasks (slide 24), Llama/Orca/Vicuna at 13B (slides 23, 28),
lectures 9/10/11 cross-referenced (slide 9), the 2011 and 2009 systems (slides 47–48), the 13-point
MiniWoB++ improvement (slide 67) — matches the deck as spoken. The only content-level fixes of
this kind are the two garbled letter sequences noted above, restored to `ABCDE`/`ABCDF` (slide 35's
"Extend sequence: abcd → abcde" and slide 36's modified "abcd → abcdf"), which are letter strings
rather than numbers.

**Where the source is still unreliable**, the text carries an inline `[Ed:` note rather than a
silent guess. There are two, both audience interjections picked up away from the microphone: at
25:46–26:33, a student's question about why base-9 addition (rather than base-10) counts as the
counterfactual condition, reconstructed from badly garbled captions; and at 40:32–41:19, a second
student's paraphrase of the Decision-Transformer input/output tokens ("we resolve three input
tokens into one output token, and turn it off?"), where "turn it off" does not clearly parse.

---

**[0:06]** Okay, let's just get started. Welcome to lecture 14, everyone. Hope you've been doing
well, and, you know, managing all of the various deadlines. So today we'll be looking at two
interesting applications of language models. The first half I'll be talking about using language
models to reason in domains like math, geometry — doing things like spatial reasoning. And then in
the second half of the lecture I'll be talking about how you can use language models to take
actions in grounded environments. Okay, so a little bit of a disclaimer: a lot of the content
today is research that was done in the

**[0:52]** last three, four years, so there's plenty of questions, plenty of unanswered questions,
and not a lot of — so, let's — maybe we can have more of a discussion around these topics. Okay,
so let's get started with reasoning. So experts like to start a lecture on reasoning by really
talking about what are the various kinds of reasoning, so I'm going to do that here. Okay, but at
a high level it's really about using facts and logic to arrive at an answer. But more concretely,
there are three distinct categories of reasoning that we can talk about. The first one, which is
probably the one that most of you are familiar with, is deductive reasoning, where we go

**[1:38]** from rules of logic, along with a premise, to come up with a firm conclusion. So an
example of that could be that we have the sentence: all mammals have kidneys, and all whales are
mammals, and then we can come up with the conclusion: all whales have kidneys. And we could do
multiple such steps of reasoning. Okay, a second form of reasoning is inductive, where, given
observations, we derive conclusions. So maybe we've learned from experience that every time we see
a creature with wings it is usually a bird, and let's say we observe a state where we see a
creature with wings, and using our

**[2:25]** experience, we can come up with this conclusion that the creature is likely to be a
bird. So that form of reasoning is inductive. Okay, and finally we have abductive reasoning, where
we're given an observation and then we start drawing possible explanations. Okay, so maybe you see
a car that cannot start, and there's a puddle of liquid under the engine, and then you start
drawing inferences about the situation. So one of them could be that the car has a leak in the
radiator. Okay, all right. And apart from that taxonomy, we can also think of reasoning in formal
and informal terms, where formal reasoning involves using axioms and rules of formal logic to
derive

**[3:11]** truth conditions. Okay, there's also informal reasoning, which is what you and I
probably do every day, and here we just reason about everyday situations and use common sense to
derive conclusions. For most of the lecture, when I say reasoning I will mean informal deductive
reasoning, and it's often going to involve multiple steps. Okay, so let's come back to language
models. Okay, so we've learned in lectures 9, 10, 11 that language models — or large language
models — are really, really good at coming up with plausible continuations of text that reflect
human preferences and constraints. Today we'll try to answer if they can also reason.

**[4:00]** Okay, so one of the most basic ways we can try to answer this question is via
prompting. Okay, and we've probably already seen this: there's this popular method called
chain-of-thought prompting, where you get a language model to produce a reasoning step before
producing an answer. And we could do this by providing some in-context examples with explicit
reasoning steps that the language model can then mimic at test time. Okay, so that's
chain-of-thought prompting. Another, rather surprising, property of language models is that
sometimes you don't even have to show them these in-context examples, and you could just prompt
them with the sentence, "Let's think step by step," and you can

**[4:47]** get these reasoning rationales before they produce an answer. Okay, so that's pretty
simple, but let's keep going. Okay, so another popular way to prompt language models to do
reasoning is via self-consistency. So here, what we do is, instead of greedily sampling a rationale
followed by an answer, we're going to sample multiple reasoning steps and correspondingly multiple
answers. Okay, so what we see in the figure on the right: we have a question, and then what you
would normally do with chain-of-thought prompting is you would greedily decode a rationale and
then, conditioned on the rationale, generate an answer. With

**[5:33]** self-consistency we're going to sample multiple times — sample multiple rationales, they
are all going to lead to multiple answers, and then we're going to pick the one that is the most
common. Okay, with the idea being that if an answer keeps appearing for multiple rationales — as
the majority of the rationales agree on — then it's more likely to be correct. And the authors of
self-consistency find that on a variety of mathematical reasoning tasks, if you add this simple
idea of self-consistency — where you sample multiple times and sort of do majority voting — that
improves performance pretty drastically over standard chain-of-thought. And interestingly, you
know, when I saw this result the first time I

**[6:19]** thought, okay, this is just like ensembling, which is, you know, we learned this in
CS229: the idea is if you want to boost the performance of your system, I'm going to produce like
10 classifiers with different random seeds, and I'm going to produce a classification decision, and
I'm going to do a majority voting. But it turns out that it's doing maybe a little bit more than
just simple ensembling. So the authors also compared an ensembling approach where it's the same
language model with multiple different prompts, and then you do majority voting there. And it
turns out that self-consistency is better than just simple ensembling. Okay, okay, so earlier
today I said that I'll be talking about multi-step reasoning. So far we've

**[7:06]** looked at sort of math problems, but not — like — prompting, but not necessarily
multi-step reasoning. One of the main aspects of multi-step reasoning is it involves breaking down
a large problem into several subparts, answering each of the subparts, and then combining
everything into a solution. Okay, so there's this kind of decomposition strategy that was
integrated into another prompting method called least-to-most prompting. And the idea behind
least-to-most prompting is, like I said, given a question we're going to first break it down into
sub-questions, as shown here. And then, given these sub-questions, the language model will sort of

**[7:53]** answer each of the sub-questions, and then, conditioned on its answers to the
sub-questions, is going to generate the final answer. And this is kind of how it looks for a math
reasoning problem. So in standard chain-of-thought prompting you would have a question followed by
a rationale and the answer. With least-to-most prompting, which is this decomposition strategy, you
take the question and, instead of directly producing a rationale, you ask the language model to
break it down into sub-problems, and then you have these two different sub-problems, and then you
start answering both of those sub-problems, and then condition your final answer on the answers to

**[8:39]** those sub-problems. So, okay, so that's just a prompting method, right? One interesting
experiment from least-to-most prompting was showing that you can sometimes generalize from a small
number of reasoning steps to a much larger number of reasoning steps. So here, in this math word
problem, there's two reasoning steps, and if we show this prompt to the language model as an
in-context example, we see that it continues to generalize even on examples that required more
than five steps of reasoning. And, in a way, that's much better than standard chain-of-thought.
But it's not entirely clear if structuring inference in this manner

**[9:26]** is really fundamental. One of the other results they reported was that, with enough
prompt engineering, the row corresponding to best-prompted chain-of-thought is on par with
least-to-most prompting. But it's kind of an interesting idea, trying to break down problems into
sub-problems, solving the sub-problems, and then building up a solution based on your answers to
the sub-problems. Okay, so all this was different sort of prompting methods to get reasoning
behavior out of language models. Can we do something more? So one of the things that we might be
interested in is, instead of trying to get really large language models to do reasoning, maybe we
want to somehow get

**[10:13]** this kind of reasoning behavior in a smaller language model. And one popular approach
for doing that is distillation, where maybe you want to fine-tune a smaller Llama model by teaching
it to imitate a larger language model. And so that's what we're going to look at now. Okay, so this
model is called Orca, and at a high level Orca is going to fine-tune a smaller 13-billion-parameter
Llama language model on explanations produced by GPT-4. And to construct this data, it's pretty
simple. It has three steps. So the first step is we get a

**[10:58]** wide variety of instructions from the FLAN-v2 collection. Okay, so FLAN-v2 is basically
a dataset — it kind of accumulates multiple datasets into one collection — and it consists of
instructions paired with questions and answers, and I'll show an example of this in a moment. And
then we're going to prompt GPT-4 or ChatGPT with these instructions along with a system message.
And the objective of the system message is to get ChatGPT or GPT-4 to produce an informative
explanation along with the answer. So here we have a question about, you know, simple data
processing, about

**[11:45]** calculating the median. And there's a system instruction that says, please justify your
steps, and answer step by step. And, in producing its output, the model provides a fairly detailed
explanation of how it got to the answer. And what Orca is going to do is use precisely this
explanation to fine-tune a much smaller model. Okay, so that's what's going to happen. Once we
have these explanations, we're going to fine-tune a much smaller, 13-billion-parameter Llama model
on these explanations. Okay, so, so far we've looked at sort of math reasoning, and sort of
grade-school-math

**[12:34]** problems. Let's turn to a different benchmark for reasoning: we're going to look at
BigBench-hard. And this is another dataset for multi-step reasoning. Okay, and let's look at some
examples from BigBench-hard. It consists of multiple different sub-tasks — there's a total of 23
different sub-tasks. I'm going to show a few examples. One of them is evaluating Boolean
expressions, so the question is: "True and False and not True and True is" — okay, so that's
basically, you know, evaluate this Boolean expression, and, with chain-of-thought, the model can
evaluate each of the sub-expressions and

**[13:19]** get to the final answer. And another example of a task from BigBench-hard is — sorry,
this is date understanding, not data understanding — so the question is: tomorrow is a given date,
what is the date one year ago from today, in a given format. And it's paired with some options, and
again the model can think step by step, following basic chain-of-thought, and then come up with an
answer. So this is kind of the flavor of tasks in BigBench-hard: most of these involve multi-step
reasoning, they're fairly synthetic, but also reasonably hard for language models.

**[14:06]** Okay, another example is geometric shapes, and this one — you know, it's pretty
surprising that language models can do anything here — so you're given the SVG path element, and,
you know, I have no idea what this renders as, but the question is, given the SVG, what shape are
you going to get? Okay, and there's a bunch of options, and then, again, the model, prompted with
"let's think step by step," will produce some answer. We don't know if it's correct, but it's
going to produce some answer. Okay, and so this is basically a dataset covering different kinds of
reasoning: spatial reasoning, date understanding, evaluating Booleans. And it's sort of

**[14:53]** multi-choice, so it's easier to get an accuracy number. And so, yeah, so it covers a
wide variety of different tasks. On the left we have performance from really large language
models — this is zero-shot chain-of-thought, with just the prompt "let's think step by step." So
GPT-4 has some potential contamination issues with BigBench-hard, so maybe we can ignore that
column. Vicuna — I think, a few months ago, it was state-of-the-art as an instruction-tuned
Llama-13B model —

**[15:39]** and Orca is, again, a Llama-13B that's fine-tuned specifically on this explanation
data — where, you know, you have instructions and then you have explanations from ChatGPT or
GPT-4, and you fine-tune on that. And we see that, overall, it outperforms ChatGPT, maybe because
it's specialized to just these reasoning problems, and it outperforms Vicuna, which was not trained
on these really extensive explanations. So that's one way you can get a smaller language model to
display some kind of reasoning behavior. Okay, so, you know, this was all great, and we're very
happy that

**[16:26]** you can just generate rationales from a big LM and then fine-tune a smaller language
model on that. But then someone could ask, why not just fine-tune the big language model on its
own rationales, right? So that's also been explored, and there's a bunch of different methods that
do this. I'm going to talk about one of them, called reinforced self-training, or ReST, and that's
going to alternate between two stages. The first stage: given a reasoning problem, and perhaps the
prompt "let's think step by step," I'm going to have the language model generate multiple
rationales. Okay, and then I'm going to filter these rationales based on whether they give me the
correct answer or not. Okay, so, think about — you know — algebra word problems: someone has

**[17:14]** three apples, someone else has four apples. You generate a rationale, and the answer
comes out to be seven — you keep that rationale. The answer is 12 — you leave that rationale out.
And then I'm going to do an update step, where I'm going to take these rationales that I filtered
in my first stage, and I'm going to fine-tune the language model on that. And then I can do this
iteratively: now I have an updated language model, I can get hopefully better rationales, and then
I can update the language model on better rationales to get an even better language model, and I
can keep doing that. Okay, and the results are promising, but, you know, what we find is, on
GSM8K, which is this grade-school-math

**[18:02]** dataset of algebraic word problems, as you increase the number of iterations of
self-training, we see a slight improvement in performance, and then it starts degrading. MATH is
another dataset that again focuses on multi-step reasoning, covering math problems. And again, on
this dataset, we see that as we do more iterations of this reinforced self-training paradigm we see
an improvement in the accuracy. And the numbers in orange here are a much larger PaLM model, the
numbers in blue are a smaller model, and the dashed lines represent what you get

**[18:48]** sort of if you did supervised fine-tuning on human-provided rationales. So one of the
promising things about this approach is, when you do multiple iterations of training on your own
rationales, you can outperform sort of human-generated rationales. And that is exemplified again in
this graph, where what we find is: the blue bar represents accuracy when you take the PaLM model
and you do supervised fine-tuning on human-provided rationales. Okay, so blue is if you fine-tune
on all human-provided

**[19:34]** rationales, orange is if you fine-tune on one rationale per training example — okay,
and these are written by humans — and green is what you get if you fine-tune on one rationale,
chosen at random per question, which is generated by the model. So it's controlling for the number
of rationales, and we see that it outperforms human-provided rationales. And then, if you do the
full multi-step iterative procedure where you keep improving the model, we see again a boost in
performance. So that's super promising. But let's revisit the question that we asked in the
beginning about reasoning in

**[20:21]** language models. Okay, so one way of answering that question is we can apply all these
methods and look at benchmarks. But maybe the way to answer the question correctly is to be more
systematic, come up with counterfactual tasks, and be very careful about possible data
contamination. And I'm going to show some results around that. So we started the lecture with
chain-of-thought, and maybe the first question to ask is: are the rationales that the model
produces — the chain-of-thought — faithful? What I mean by faithful is, maybe the model produces
some rationale and then it produces an answer — maybe the

**[21:07]** answer does not even depend on the rationale that it produced, right? So maybe the
question was, you know, Tom has three apples and Jerry has four apples, and the rationale it
produced was, okay, Tom has three apples, Jerry has four, 3 + 4 is seven, so the answer is 25. You
know, so in a case like that you'd say that the model was not faithful to its rationale. And so
what we see in this plot is a very careful experiment where, on the x-axis, we have the number of
reasoning samples. Okay, so the setup is something like this: for every question the model
produces a rationale, and a rationale here is multiple

**[21:53]** sentences. And what we're going to do is we're going to force the model to early exit
from its rationalization, and just force it to produce an answer. Okay, so if it produced four
rationales, I can early-exit right after the first rationale and ask it to produce an answer. I can
exit after the second rationale, ask it to produce an answer, and so on. And what I'm going to plot
on the y-axis is the model's accuracy after early-exiting in this procedure. So let's say that I
early-exited after just one rationale, and the model produced exactly the same answer that it
would if it had seen all four sentences in its rationale — then maybe we can conclude that the
kind of

**[22:39]** reasoning is not faithful — it doesn't matter if the model sees the full rationale or
just the first sentence. And if you take that to the extreme, you know, maybe you terminate it
even without any rationale and it produces the same answer. So the results here are somewhat mixed,
but we see that there are enough datasets where it doesn't matter if the model sees the full
rationale before answering or if you early-exit — you kind of get the same answer, which means
that sometimes these rationales may be post-hoc explanations of the model's answer. Okay, another
experiment that tries to answer this exact same question is: you can take these rationales and

**[23:27]** then start corrupting them. So maybe your rationale was length four, and then I
generate the first rationale, the second rationale, and for the third rationale I just corrupt
it — okay — and then the fourth rationale, and then I ask the model to generate my answer. If it
turns out that, no matter how much I corrupt my rationale, the model produces the same answer,
then I can conclude that, again, the answer did not depend on my rationale. So on the x-axis we're
looking at the percentage of reasoning steps before I add a mistake in the rationale. Okay, so what
you should see is a strictly increasing trend, where if I add a mistake after

**[24:14]** the very first step, then that's probably going to change the answer a lot, and then if
I add a mistake after the last step, that maybe doesn't change the answer all that much. But again,
we find that for some datasets it so happens that you can add a mistake in the first sentence in
your rationale and the answer is not going to change all that much. And so that's also kind of an
indicator that maybe these rationales are sort of post-hoc explanations of the model's behavior.
Um, yeah, so there's a lot of lines here — so if anyone has questions — I see a few blank faces in
the audience. Okay, so let's keep moving.

**[25:01]** Okay, so that was about whether — where sort of chain-of-thought expresses a kind of
reasoning that the model is faithful to. Another question you could ask is: what if I change my
setting a little bit? So my model — let's say I observe that it's able to do arithmetic in base
10, so it's able to answer something like 12 + 14 — does that mean that my model knows how to do
arithmetic, or maybe this exact same example was present in the training data? So one way you
could test for this is by creating counterfactuals, which, based on your understanding of the
data, you expect to not be present that frequently in the training

**[25:46]** data. So instead of doing base-10 addition, you could do addition in base 9, and then
if the model has the same accuracy in base 9, then you can conclude that maybe this model has
understood how to do addition. Similarly, for logic: maybe the reason why the model is so good at
solving logic problems is because it's seen something very similar in its training data. So what
if I construct a world where — I don't know — corgis are reptiles? Can it still do this logic
problem? Okay, and so what we find is there is, you know, sometimes a pretty significant drop when
you move from — there's a question, sorry.

**[26:33]** *[Ed: student question badly garbled in the captions and reconstructed here — likely
something like:] You said "counterfactual" — why is base 9 the counterfactual, and not base 10?*
So it's a counterfactual — excuse me — in the sense that the authors comment that base-10 addition
is frequently observed in the training data, but very few people do base-9 addition, and so there's
going to be many fewer examples of this in the training data. *So it's more so
out-of-distribution, right?* Yeah, yeah, so you can also call it out-of-distribution, for sure.
And, yeah, so, from results — what we see is there's this drop in performance, even for very
simple logic problems that don't involve multiple steps of reasoning — a, you know, kind of a

**[27:18]** significant drop in performance, which maybe suggests that there's not that much
reasoning, there's more memorization. Yeah, so we could keep going with this paradigm of changing
the problem setting so that it starts looking sort of out-of-distribution to the training corpus.
And this is exactly what was done in this paper that looked at analogical reasoning. So, basically,
the setup is something like this: I'm going to show certain examples of string transformations, and
I'm going to ask the model to generalize to new examples. Okay, so in this extend-sequence problem
I

**[28:04]** have ABCD, and the output is ABCDE. And then, given IJKL, the model has to produce
IJKLM. Okay, and so, now, the way you can make this into a counterfactual, or something that is
out-of-distribution, is — maybe you can change what the extend-sequence task is. So now, instead
of outputting ABCDE, maybe the model has to output ABCDF. So, instead of outputting the next
character, it has to output, um, sort of one more — so the second character after the next — and so
on. The other kind of counterfactual you could add is: instead of operating on the

**[28:51]** standard alphabet, you could modify the alphabet completely. So instead of the alphabet
being ABC, maybe you start at X, Y, and so on. So what we find is — so we find two things. The
first thing that we find is that there's a significant drop in performance as we go from the
standard analogical reasoning problem to one of these counterfactuals, where we either change the
alphabet or we change the description of the task so that it becomes slightly unnatural. On the
other hand, the authors also did this exact same experiment on human subjects, where they find very
little

**[29:37]** decrease in performance. Okay, so overall, what this result suggests is: maybe there's
some reasoning, maybe there's some memorization, but there's nothing systematic. Okay, so, you
know, again, this is all emerging, so maybe someone will find that if you change your prompt a
little bit, now models can do reasoning. But this is kind of the current lay of the land. Okay, so
that was the reasoning module of the lecture. I'm going to now switch gears and talk about language
model agents. So, and this is kind of related to reasoning, in the sense that

**[30:22]** reasoning involves this multi-step inference where, given some facts, you have to
arrive at completely new conclusions. With agents, what we'll see is that there's some high-level
objective the model has to accomplish, and it has to reason about postconditions, object
affordances, a kind of uncertainty in the world, to carry out a sequence of steps. So let's start
with some terminology. Okay, so we have our agent on the right — that's going to be some neural
network — and then we have an environment, and I'll give some examples of what these environments
could be. The agent receives an observation

**[31:10]** from its environment, and based on the observation it issues an action. Okay, and along
with that it receives this second variable, g, and g represents a language instruction. Okay, so
there's many names for this setting, and these models — digital agent, language-conditioned
policy, or an instruction-following agent. Some examples of environments are, maybe, a web
browser — and in a browsing environment where the objective is to book a flight from San Francisco
to New York, and the observation could

**[31:56]** either be raw pixels that the model sees, or it could be the HTML DOM representation,
and the action space, if you're looking at these web environments, could be typing on specific web
elements, clicking on web elements, moving your mouse to a certain web element to interact with it,
and so on. And, yeah, I mean, there's a vast number of applications — I don't think we can cover
all applications, but we can look at some. So there's obviously digital assistants, like — you
know, I'm not going to say the names, because I know people's phones might start popping up — but
you know, you can give

**[32:44]** them natural language commands, and, you know, set an alarm, set reminders, and so on.
You could also do natural language programming, where, given natural language descriptions, you
get a model to write Python code. Another example of this could be UI automation, where maybe you
want to do automated testing of UI elements, and so instead of having a human verify whether a UI
element works, maybe you can get a model to execute actions corresponding to a given instruction.
Or it could be something more user-facing, where, given some kind of complex environment like
Spotify, you could ask

**[33:31]** an agent to play some songs. And then, finally, there is this emerging application
where we want to add additional tools, or plugins, to language models so that they can control
various different applications. Okay, so before we look at how we can use language models to do
instruction following, I think it's very helpful to look at how this was done before language
models. So there were basically three main ideas. Sometimes the right thing to do was to collect
examples of utterances paired with logical forms.

**[34:20]** So logical forms could be some kind of executable representation that you could just
execute against either a knowledge graph or a database to get an answer. So maybe you have a query
like, what states border Texas, and then there exists some sort of program description that you
could execute against a knowledge graph to get an answer, or a list, here. And idea number one that
people looked at was to treat this as almost like machine translation, right? So you have a source
language, which is sort of English commands, and then you have a target language, which is sort of
these

**[35:06]** meaning representations, or logical forms. And then you could apply the same machinery
from Assignment 3 to build a natural language interface here. Okay, so you directly maximize the
probability of a sequence of actions given a goal or a command. Idea number two was something a
little bit more complex. So here you have instructions paired with actions. Instead of directly
mapping instructions to actions, I'm going to infer an executable plan from these instructions and
action sequences, and I'm going to train a model to go from

**[35:53]** instructions to these plans, and then define a very rich execution model that's going
to directly execute these plans. The advantage of this is, maybe there is more high-level decisions
you could encode in your plan, which would be harder to get into the model if you were to just
train it to produce the action trajectories directly. And I have an example of a system like that
from 2011, which was basically an agent that could navigate in a grounded environment. And, yeah,
the idea was something like this: that you took an instruction and obtained a plan, and then you
would

**[36:39]** train a semantic parser, which is basically this kind of machine translation system
that would convert commands into sequences — into this plan — and then, once that's trained, at
test time, given a completely new instruction, you would run the semantic parser, get this plan,
and then execute it in this execution model. Okay, and I have an example of an instruction and a
plan from this 2011 system. The third idea, which is probably — you know, maybe the first one that
comes to mind if you see a setting like that — is to use reinforcement learning directly. And what
people did there was to use RL to directly map instructions

**[37:25]** into actions. So I'm going to learn a policy that outputs actions that maximize some
reward, okay, which is conditioned on my natural language instruction and the observation. And this
reward could be both sparse — which is, I carry out the entire task and then my environment tells
me if I achieved the task or not — or it could be something that I obtain after each step: I take
an action, and then the environment tells me if this action completed some percentage of my task
or not. And, on top, I've included an example of a system from 2009 that did this for automated
Windows debugging. And so, you have some natural language instruction to click some

**[38:13]** UI elements, and that gets mapped into an API command that the model executes, one
after the other. Okay, so these were basically the three main ideas that people had before language
models: you would either train semantic parsers, or you would infer these plans from
instruction–trajectory pairs and then learn to directly model plans and have an execution model
that can execute plans, or you would do reinforcement learning if you had a reward signal. So how
do we do things in 2024? So there are a few ways to think about this. I think maybe the most
instructive is to think about what we're

**[38:59]** trying to achieve, right? So we are trying to model trajectories — sequences of actions
conditioned on some goal. Okay, so I want my model to book a flight from San Francisco to New
York, and I want it to produce a trajectory of, maybe, typing and clicking actions. So let's look
at how that factorizes. So the probability of a trajectory conditioned on a goal or an instruction
is just the probability of the state, action, next state, and so on, conditioned on the goal. And
you could factorize that into two terms: the first term is the transition dynamics of the
environment, and that's just — what happens if I take a certain

**[39:44]** action in a given state, how is my state going to change. And the second object is the
agent policy, which is: given my goal and the trajectory so far, what is the next action I should
be taking? Okay, and then people quickly realized that you could just treat this as a generative
problem. So you could treat the problem of decision-making in environments as sort of a generative
trajectory-modeling problem. And what I have in the top right is an example of a transformer that
just takes the history of actions it's taken so far, the current state, and some indication of

**[40:32]** what task it should achieve — here, based on reward, but it could be a natural language
string — and it's just trained to predict what's the next action. Okay, and you could just train
an autoregressive language model to do this, and it turned out that this worked very well in sort
of an offline-RL case. *Question — sorry, in the figure — why are we predicting one action?* So we
are predicting one action, before — and the current action — uh, oh, so, no, no. So you predict an
action, execute that, right? Append that to your trajectory, and then you predict the next action,
and so on. *[Ed: exchange partly garbled in the captions] So we resolve three input tokens into
one output token, and turn it off?* Yeah, okay, sounds

**[41:19]** good. And it turned out that this worked really well. And so, instead of getting these
latent plans and training semantic parsers, or trying to do reinforcement learning, we started
using language models as policies. And so a simple way to do all of that is to prompt a language
model in a loop. Okay, so we're going to specify the action space in text. So this is a simple
language model agent — this is not going to work at all, but it's illustrative of how an agent can
be built. Now you provide an action space in text — so maybe it's a digital environment, and

**[42:06]** maybe it can type, maybe it can click, maybe it can type characters, maybe it can move
the mouse somewhere. You provide it an instruction, and you provide it the sequence of actions and
observations it's received so far. Okay, and then, conditioned on all that, you ask it to predict
the next action. And there's nothing deep going on here — this is just chain-of-thought prompting
in a loop. Okay, but the hope is that, because we've reduced the problem of decision-making into
just autoregressive modeling, this could work. Okay, and indeed, a slightly more complex version of
this can work in

**[42:52]** some environments. Okay, so now I'm going to give a little flavor of what different
environments look like now, for evaluating language models as agents. So the simplest environment
that people consider is MiniWoB++. So this is a sandbox environment that evaluates basic browser
interactions — like, maybe on a mini-Twitter environment, can you get a language model to retweet
a given tweet; given a simulated email client, can the model forward someone's email, can it
compose an email, can it click on certain buttons or not. It's not at all real-world, so

**[43:38]** it's not real websites, and it's relatively short-horizon: given any instruction, most
tasks can be accomplished in under three actions. But zero-shot performance of even the best
language models is still far from perfect, even on this very simple benchmark. A second, slightly
more real-world benchmark is WebArena. And this is also a sandbox environment, but it's a pretty
close approximation of real websites, that spans e-commerce — so there is a website in WebArena
that resembles Amazon — social media, something that resembles Twitter, and additionally there are
utility tools like maps. So an

**[44:23]** instruction could require a model to open up a map application, find the shortest path
from point A to point B, and use that in its later sequence of actions. And there's multi-tab
browsing, like we commonly do — so with MiniWoB++ there's only a single tab, and with WebArena, I
think this was the first environment that introduced this idea, where you have multiple tabs and
the agent can switch between apps, tabs. And, again, we are going to evaluate functional
correctness, which is whether the model gave the correct answer at the end, whether the sequence
of steps it took

**[45:10]** gave the intended behavior, as opposed to whether it took a sequence of steps that
maybe a user had pre-programmed. So another popular environment, or dataset, is WebLINX. WebLINX
also has multi-tab browsing, and it has web interactions on real websites — so this is not
sandboxed approximations of real websites, it's not sandboxed — just, like, browser
interactions — these are actual real websites. And it also introduced a new action, where the
agent could communicate with the user. So maybe there's some instruction which is to, like,
reserve —

**[45:57]** I don't know — like, a movie, or buy a movie ticket, or something, and then, at some
point, the model has to request credit card information. And so there is this additional action
where a human could be involved in communicating with the agent. And this is not an environment,
but just a collection of interactions, so you can't, for example, do any kind of exploration or
online learning here, but you could definitely use this for evaluating. Okay, so this was just a
taste of what some benchmarks look like for language model agents. So how are we going to train
these models, right? So, given that we're going to train — we're going to treat decision-making as

**[46:44]** causal language modeling. We're not going to use any of the ideas from pre-LLMs. The
standard practice is to do in-context learning with few-shot examples.
And in the few-shot examples, typically, for any new website or any new use case, you're going to
get humans to perform those tasks and feed that into the language model's prompt as in-context
demonstrations, which it could then use to solve similar-looking tasks on very similar websites.
So, obviously, this is not scalable — there's thousands of environments, and, on some
environments, lots of different interactions that are possible. And so maybe there's

**[47:31]** something better that we can do than just getting humans to provide demonstrations for
every new use case. And so we're going to use something we saw early on in the lecture, which was
to use the language model to generate rationales and then fine-tune on that. And here we don't
have rationales, but we could produce action trajectories and then use that as supervision. Okay,
so the way that looks like is something like this: so let's say I have some environment — you
know, let's say it's some MiniWoB++ environment — and I'm going to just get an agent that can
randomly explore the environment. So it'll just execute a random sequence of clicks and types, and

**[48:19]** scrolling operations, and let's say it produces some trajectories. Okay, and now I'm
going to use these trajectories, and somehow filter them — so that was the idea from earlier. So
you're going to get a bunch of different outputs, and then you're going to filter it somehow. So
here we're going to use a second language model, because we don't know what a good trajectory
looks like — not like a math problem, where you know the correct answer. We just had a language
model interact with the website and generate trajectories, and we want to somehow filter out what
a good trajectory is. And so we're going to use a second model that will produce a description of
these trajectories, and the idea here is that if you can get a model to produce a description of

**[49:05]** what the sequence of actions corresponds to, then maybe that's a good enough signal for
a good trajectory. Okay, and so, maybe, given the first trajectory, it guesses that the
instruction was to book a flight from San Francisco to New York. For the second trajectory, it
said, set the date to some given date. And maybe it wasn't able to come up with any good
instruction for the third trajectory. And then we're going to do something, again, that we saw
earlier on, which is to do this iteratively. So now we have a goal that we got for a trajectory,
and now I'm going to get the language model to condition its behavior on this

**[49:52]** goal. So the goal is to set the date as some given date, and now, instead of doing
random exploration, the model is going to produce a sequence of actions that have a better
correspondence with some natural language instruction. So it produced a trajectory based on that
instruction, and then I'm going to use some coarse filter that's just going to look at
correspondences between the instruction and the sequence of actions and the states the language
model visited, and use that to decide if the trajectory was a good trajectory for the instruction.
And, in this case, given the instruction, this seems like a pretty good trajectory for completing

**[50:39]** this task. And so, then, we added it to a set of examples. Okay, but maybe sometimes
things are not so good. So for that second instruction, the generated label was to book a flight
from San Francisco to New York, and let's say we run that again through the language model, and it
produced a second trajectory. Okay, and clearly this does not seem like a successful trajectory
corresponding to booking a flight. And so what do we do here? Maybe we can throw away this
interaction, but interactions are pretty costly — specifically, you know, if you're looking at
real websites, and each interaction could take a few milliseconds, and so maybe we don't want

**[51:24]** to throw away this interaction. So what we're going to do here is, again, invoke the
relabeler to take the trajectory and assign it a new label. So the model was not successful at
accomplishing the task it set out to do, but it accomplished something, and we're going to come up
with the best guess of what that was with a second language model. And it's going to say that,
okay, maybe the instruction you accomplished instead was to set the origin to SFO and the
destination to New York City. Okay, and so that's going to get fed back into the language model,
and we're going to keep doing this iteratively till our filter says that this is a good
instruction–trajectory pair. Okay, so we have the same idea of using a language model to generate
outputs, and

**[52:10]** some iterative procedure that will give us a good set of training examples. So,
overall, the method looks something like this: you know, you have some environment, we're going to
use an unconditioned language model to just randomly explore the environment and generate a
sequence of trajectories, and then we're going to convert these trajectories into synthetic
training data by iteratively converting trajectories into natural language descriptions, and then
taking natural language descriptions and converting them into even better trajectories, and so on.
And once we have this collection of synthetic examples,

**[52:55]** there are two things we could do: one, you could fine-tune using this data — but the
simplest thing you could do is repeat the paradigm from earlier, replacing human-provided
demonstrations in-context with these synthetic demonstrations. And we find a reasonable boost in
performance, or a 13-point improvement, on the MiniWoB++ benchmark. And again, you know, even
though MiniWoB++ is very, very simple, zero-shot performance for even the best language models is
far from perfect. And we also see an improvement on a second, multi-step, tool-use environment. But
so far we've only looked at text, right? But maybe for real-world

**[53:40]** applications, it's kind of intractable to, for every environment, obtain the HTML and
feed that into the language model's context. Sometimes there can be tens of thousands of DOM
elements, and then corresponding JavaScript, and inputting all that into the language model's
context could be, you know, intractable. And maybe that's also not the best way to show the state
of the environment — maybe the best way is to directly show the pixels corresponding to the
environment. And so now we're going to look at some examples of vision-language models that people
have used for building these agents. Okay, so the first one that we're

**[54:26]** going to look at is LLaVA. And the idea here is, again, kind of similar to Orca, that
we looked at in the reasoning half of the lecture: we're going to use GPT-4 to generate, this time,
both instructions and responses for textual descriptions of images. So maybe there's an image, and
we're going to use metadata corresponding to that image to come up with a textual description,
feed that into GPT-4, and ask it to generate possible questions and responses. And then we're going
to jointly fine-tune an image encoder — here, CLIP — along with a

**[55:12]** text decoder — here, Vicuna, which is a Llama model that is instruction-tuned. And,
through this joint fine-tuning, at the end, we get this image encoder that can output language
responses, and now we can ask questions about images, maybe use that to directly input screenshots
instead of HTML DOM elements. So a second approach that looked at building joint image-language
models, that people later adapted to agents, was Pix2Struct. And the idea is, again, very similar:
there's an image encoder and

**[55:59]** a text decoder. The image encoder will take the image, convert it into patches, and
assign each patch a position ID, run that through a transformer, and then there's a decoder that
will decode out some text. Okay, one of the new things that Pix2Struct introduced was a new
pre-training task. So, for LLaVA, the pre-training was fairly simple: we're going to use GPT-4 to
just generate synthetic questions and responses based on textual descriptions of images. But
there's only so far you can go with textual descriptions of images. What Pix2Struct did was look
at screenshots

**[56:45]** from websites, and mask out screenshots, and then ask the transformer decoder to
produce HTML corresponding to the masked-out elements. So here there is this list that has a
corresponding HTML. One of the data points in Pix2Struct looks something like this: so you might
mask out, let's say, the first answer corresponding to Python, and ask the model to produce the
HTML corresponding to just the patch that was masked out. And so this seems like a more natural
pre-training objective that can maybe have better interactions between image

**[57:30]** and text. And then this was also adapted for building these multimodal agents. Okay,
so at this point I just want to highlight that this is really an emerging application. There's
kind of this huge prompting gap — is what I like to call it — so if you do not do extensive
prompting, and if you do not use bespoke few-shot examples where, for every different environment,
you have a different set of few-shot examples, even the best language models are very, very far
from perfect, even on very simple tasks like MiniWoB++ — where, you know, the goal was just to
click on certain elements, or respond to someone's email, where in MiniWoB++ that just takes like
five

**[58:16]** actions. And then, even for something as simple as MiniWoB++, even after doing
extensive prompting and few-shot examples, there's this drop in performance as you go from the
simplest task, that involve mapping an instruction into a single action, to mapping an instruction
into maybe five or 10 actions. So long-horizon planning is still very, very hard, even on these
very simple benchmarks. And then, if you look at something more complex, like WebArena, which
tries to approximate real websites, has multi-tab browsing, has external tools that the model can
use, there's just a huge difference between sort of human-

**[59:02]** level task success rate and what the best models get, even after prompting, even with
few-shot examples. And then the kinds of errors models make are also pretty weird. So one of the
examples, from WebLINX, was: the task was to just open Google Translate and sign in using
credentials — there was an email and a password — and then what GPT-4V did was, instead of typing
in the password, it just typed the email into the password field. And it just couldn't recover from
this error. So, you know, it tried to sign in, there was an error, it tried to insert — tried to

**[59:48]** type in the email again, and so on. And I'm sure with extensive prompting you can fix
this, and maybe that's beside the point, right? And then, again, you know, there was a different
example where the model had to issue a search, and then, instead of issuing the search with the
correct term, it sort of repeated the same term like three times. And obviously that's not going
to return any — return any results. So there's a lot of room for improvement, as you can see, and
there's lots to be done in this space. Okay, so I'm going to recap and take any questions. So we
looked

**[1:00:34]** at two different things today. We looked at reasoning in language models: we saw
that there's a few ways that you can get reasoning-like behavior in language models. You can
prompt them in various ways, so the simplest example of that is chain-of-thought prompting. You
can do chain-of-thought prompting but generate multiple rationales and try to reconcile them and
pick the answer that was most frequent. You can do problem decomposition in your prompt, asking
the model to explicitly decompose a problem into multiple steps before answering. So that was all
prompting. You could also try to train specialized small language models for reasoning, by
generating rationales from a big language model and then fine-tuning a

**[1:01:20]** smaller language model on these rationales. Instead of fine-tuning a smaller language
model on rationales from a big language model, you could just fine-tune the big language model on
its own rationales, and keep doing this iteratively. And we saw that, sometimes, if you do multiple
iterations of that, performance can keep improving, and can even outperform human-provided
rationales. But, on the flip side, we saw that, while there are some initial reasons to be
optimistic, if we go and do counterfactual evaluation, we see that it's not clear if the models are
good because they're reasoning, or if models are good because, you know, all of these problems
were in some shape or

**[1:02:07]** form already in the training data. And we saw that with counterfactual evaluation. In
the second part, we looked at language model agents. We talked about the historical perspective
through which people built grounded agents, and then we saw that you could recast the problem of
decision-making as causal language modeling. And then we looked at various ways through which
people have modeled decision-making with language models — most of it involves prompting and
in-context learning. And then we looked at a method — similar to what we saw in the first
module — for generating synthetic demonstrations, and here we looked at doing exploration and the
same kind of

**[1:02:53]** iterative relabeling. You know, most of the language models we looked at today were
text-only. We saw some examples of language models that can take both text and visual input. And,
you know, we saw that benchmarks are very, very challenging, models make trivial mistakes, there's
a huge gap between human performance and what we get with models — so there's a huge difference
between human performance and where models are, and a lot of room for driving further improvement.
And maybe some of you are doing it for your projects. Thank you. [Applause]
