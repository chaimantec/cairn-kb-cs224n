---
title: After DPO
lecture: 16
video: https://www.youtube.com/watch?v=dnF463_Ar9I
source: edited from auto-generated YouTube transcript
verbatim: false
original: original/16-after-dpo.md
slides: ../slides/16-after-dpo.md
---

# After DPO — transcript

Copy-edited for punctuation, sentence boundaries, and mis-heard terms; cross-checked against
`raw/slides/16-after-dpo.md`. The verbatim auto-generated captions are kept at
`raw/transcripts/original/16-after-dpo.md`. Lecturer is Nathan Lambert (Allen Institute for AI /
AI2). Student questions and comments from the floor are set in *italics*. Timestamps mark the
start of each paragraph; all 90 are preserved in order.

**This is an edited transcript.** The captions had no punctuation and mangled a lot of the
vocabulary this talk is built on. Most pervasive: *RLHF* almost always arrived as "rhf", and
*PPO* almost always arrived as "Po"/"PO"/"pop" (once spelled out correctly around 47:45 and
again at 54:40). Other restorations, checked against the slides: *Claude Shannon* as "Claude
Channon"; *GPT-1, ELMo, BERT* as "gpt1 Elmo and Bert"; *ChatGPT* as "chat gbt", "chat 2bt",
"chbt" and "chpt"; *Bradley-Terry model* as "Bradley teror Terry"; *John Schulman* as "John
scholman"; *REINFORCE* as "reinforce"; *Alpaca, Vicuna, Koala, Dolly* as "vuna" and "alpaca Von
kuna"; *ShareGPT* as "share GPT", "share gbt", "Shar GPT" and "tra gbt"; *LMSYS* as "lmis" and
"lmc's"; *WildChat* as "wild chat"; *OpenAssistant* as "open Assistant"; *CarperAI* as "Carper
AI"; *AlpacaEval* as "alpaca Val"; *MT-Bench* as "Mt Ben"/"Mt bench"; *XSTest* as "EXs test" and
"excess T Test"; *LLMBar* as "llm bar"; *GPT-4* as "gb4"/"gp4"; *GPT-4o* as "gbt 40"; *OpenBMB*
as "open BMB"; *Cohere* as "coher"/"co here"/"Co here"; *RLHFlow* as "alignment lab ... RL rhf
flow"; *RewardBench* as "reward bench"; *Tulu 2* as "tul2", "2u" and "2 through2"; *JAX* as
"Jacks"; *D2PO* as "d2p"; *Self-Rewarding Language Models* (Meta) as "self-rewarding language
models for meta"; *SteerLM* (NVIDIA) as "steer LM"; *KTO* as "kto"; *University of Washington*
as "udub"; *Hugging Face* as "hugging face"/"hugging phase"/"hugging P's"; *AI2 / Allen
Institute for AI* as "ai2"; best-of-*n* sampling as "best event sampling"; and *ablate* as
"oblate". Two further restorations are lower-confidence and flagged as such rather than silently
fixed: *Q\** (the OpenAI rumor) for "the qar rumors," and *Llama 2* — whose paper documents a
margin term in its reward-model loss — for "lamud did this weird margin loss." **No content was
added, removed, or reordered.**

**Two passages are genuinely unclear and are marked inline rather than smoothed over:**

- At **2:26** the captions read "...collected on chatbot arena for mmis..." — no reading of
  "mmis" (possibly "months") is confident enough to commit to, so it is left as heard with a
  note.
- At **1:00:56** Nathan stumbles over a researcher's name attached to the KTO method ("kto name
  like csky, I always mess it up with these names"). KTO stands for Kahneman-Tversky
  Optimization; the slides don't spell out an author name, so "csky" is left as heard rather than
  guessed at.

The reward-hacking question starting at **1:05:45** is cut off mid-sentence by the moderator
handling the queue ("we get one at the front first") and only fully asked once the student's
turn comes around at **1:06:31** — both fragments are kept in place rather than merged.

---

**[0:05]** Okay, well, welcome back to CS224N — welcome back for me to CS224N too, since I was
traveling for a couple of weeks. I hope everything went smoothly in the meantime. So today I'm
delighted to introduce our first invited speaker, Nathan Lambert. Nathan did his PhD at UC
Berkeley, so you're allowed to boo and hiss for that, but since then he worked first for a couple
of years at Hugging Face, and now he's working at AI2, the Allen Institute for Artificial
Intelligence, in Seattle. Nathan comes from a background in

**[0:54]** reinforcement learning, like quite a few other people who are now applying
reinforcement learning to language models. He had an early background applying reinforcement
learning to robots, but it turns out it's more fun to do it with language models — no, it's not,
okay. But anyway, he's been very influential in both developing ideas as to how to do
post-training with RLHF and other ideas that come since then, including DPO, that he'll
definitely mention in today's talk. So he's one of the best experts on the post-training phase of
language model development, which has just proven, as time has passed, that more and more of

**[1:39]** the action of the large language model companies is happening not in the initial
pre-training language model training phase, but in this subsequent post-training phase, and
Nathan will have a lot to say about that today. Thanks a lot for coming to do this. — Yeah, thanks
for the wonderful intro. You can see my talk is "Life after DPO," which is a little bit of an
unclear title, so I apologize about this, but it's trying to capture what is the moment that we're
at in alignment and alignment research. Really, DPO is the paper — the story of last year — this
paper that came out, and I'll get to the math, and now a lot more people are interested in and
able to do alignment, and it's building on from there. So it's like, what are we going to be
interested in after DPO? And a tidbit — talking with Chris, that isn't explicitly in my slides —

**[2:26]** is what we're trying to close — the gap between labs like Meta and people, with the
amount of data that they're using for this kind of post-training fine-tuning. There's all these
words, all defined — is so big that the amount of data points that Meta bought for Llama 2 from
one of these providers is much more data than all of the data that's been collected on Chatbot
Arena, for [Ed: unclear — "mmis"]. So Chatbot Arena has like 800,000 data points that have been
collected, and Llama 2's paper says they bought about 1.5 million comparisons, and these are
years outdated, and Chatbot Arena's data — that's as of a few weeks ago. So you can only imagine
what OpenAI, Anthropic, etc. are buying at this scale, and this is the kind of reality that we
need to adapt to — what is different, like, we don't have

**[3:11]** that type of resource doing research, and what are we going to do. So this lecture is
some history on things that led up to DPO that I saw, that I think are important to remember, and
then really we'll go zero to a hundred and talk about recent research that we're doing to try to
answer this question and define what is happening. So I'll start with a heavily abbreviated
history of language models — I won't go through all of this, there's a bunch of this in the class
already, this is late in the lecture. I like to start with Claude Shannon, and then you skip a
whole bunch of stuff where this autoregressive loss function shows a lot of promise, and this was
not fast — you can see how many years it took to build language modeling as a field here, and
deep learning is brewing in the background as one of many

**[3:58]** things that went into this. And then you have these years, like 2017, the Transformer
paper that you hear about; 2018, with GPT-1, ELMo, and BERT — kind of these foundational topics in
language processing and how embeddings are created. And then with GPT-2, scaling laws become this
kind of key idea that people are looking at and tracking, and how these models are improving. And
then in 2020 is when people really started to wake up to how useful these large-scale trained
language models were — at this time I wasn't even a language modeling person, but for a lot of
people in AI this is when the gravity of the situation was starting to suck people in. And there's
a lot of cadence to these things — in 2021 we had the Stochastic Parrots paper, which, before

**[4:43]** ChatGPT, is raising the warnings of what we're actually putting into these models and
what are they learning — are they actually learning something meaningful from language, or are
they repeating the language that we have. This is a kind of philosophical debate, depending on
where you land on what language is, what these language models are doing today. But it's
important that it came out before ChatGPT — these foundations of debates of what language models
are doing. In late 2022 is when ChatGPT actually came out, which was supposed to be this kind of
quiet launch — a demo from OpenAI — and it has since captured the attention of the world that we
have seen. And the simple question is: can ChatGPT exist without RLHF? I think it's important

**[5:30]** to acknowledge that so much of this is from pre-training, but at every point of the
line — ChatGPT, and then a lot of these popular models since then — RLHF and these human-related
or other fine-tuning technologies seem to be necessary but not sufficient. Like, you need the
pre-training, but you also need this kind of RLHF, or this post-training, to really shift the
needle on what the most important models are at that certain moment. Some examples you can list —
so many of them — where RLHF has been relied upon. I like to look at these plots from the
Anthropic Constitutional AI paper, where they show this iterative improvement of their different
RLHF methods. It kind of shows how you have these multiple model versions that are evolving over
time as you add more fine-tuning data. This is a dense paper,

**[6:15]** but one of the most representative figures of what RLHF can do — there's a lot of
information in here that you don't need to follow right now. And then Meta's Llama 2 paper is
pretty funny, where they have this quote: "Reinforcement learning, known for its instability,
seemed a somewhat shadowy field for those in the NLP research community. However, reinforcement
learning proved highly effective, particularly given its cost and time effectiveness." This is
from the technical report directly, which I find really entertaining — this is back in the day
when we were like, oh, we don't know if RLHF is really going to take off. This is July of 2023, in
this building period, and it's just directly from the report, and that's aged really well, where
people are still using this today. But there's just a lot of interesting hints of the history and
culture of RLHF in the releases of these

**[7:01]** models where these companies like to talk about it and give us these cultural details
on what's going on. So I'm going to go through some definitions — I don't spend too much time
doing RLHF 101 and exactly what is happening with these mathematical terms, but it's important to
get on the same page of what some of these things do and don't mean. There's a lot of definitions
— I think some of the interesting ones, if they don't make sense right now, to come back to, is
like: what's the difference between instruction fine-tuning and supervised fine-tuning? I think
instruction fine-tuning is what's become really popular, where you're training a model to follow
instructions — I have another slide on this after — and supervised fine-tuning is this
domain-specific thing. We want to do both of them. I think instruction fine-tuning is more linked
to RLHF, it's about making

**[7:48]** these models really useful and really engaging and easy to work with. And then there's
other things like alignment, which is super vague, but it's in the word — it's align — it's
training a model to be mirrored to what a user wants, and there's a lot of things you can align
to. RLHF is a mouthful, which is one specific tool for doing alignment, where you have this kind
of human feedback data — feedback is a really loaded word there, where there can be preferences,
and learning to rank is related to actually putting feedback on preferences. There's a lot of
little things — I tried to make "preference fine-tuning" a phrase at one point, but didn't really
double down on it. I think it's a little bit clearer than RLHF, especially in the context of DPO,
but there's just a lot of spheres that are overlapping in this kind of post-training or
fine-tuning space of models

**[8:34]** these days. Instruction tuning, instruction fine-tuning, is still the foundation of a
lot of this — this is where things called system prompts are added, where we're making the model
ready for a specific style of input. OpenAI is still innovating on this — they have this model
spec document they released a few weeks ago, where they said they're going to have a second-level
system prompt. This just adds some structure to how the models can take in data, so that you can
do a lot more of this fine-tuning down the line, and how user data actually gets passed to the
model, or how the developer passes information that the user doesn't see. So what this can often
look like is Stack Overflow, Reddit data, where you have a question at the top and then an answer,
and this is still, I think, a lot

**[9:21]** of what is happening behind the scenes. There's a lot of data sets of Stack Overflow
out there, Reddit has these data partnerships, and this still uses the autoregressive loss
function that we started with — we haven't branched out into different loss functions yet, but
it's still super important. A lot of academic research shows that this is all you need, in some
ways, which I think is a much more mixed bag, but it's the simple method, and it's the right place
to start. And where we go from there is this RLHF objective, which looks really familiar to
people trained in reinforcement learning — I think this is a little different from the NLP loss
function. On the left side is the standard reinforcement learning objective, which is you're
learning a policy pi to maximize some reward, which is a function of something, depending on how

**[10:07]** you set up the problem. And then on the right side is going to be this kind of KL
constraint — it's a distance, so that the policy doesn't change too much. It's related to this
whole idea of over-optimization that I don't go into too much in this talk, but the key idea is
that we want to optimize a reward but not over-optimize it. And the primary questions when doing
RLHF are: how do we implement a reward function — what is our reward actually going to be — and
then how do we optimize it. You'll see this abstracted later as: we train a specific reward model,
and then we have specific policy updates. And DPO, direct preference optimization, handles this a
little bit differently. So before we get there, the actual preference model that people use for
RLHF is — well, I find this interesting —

**[10:53]** it's from this Bradley-Terry model, which is from economics, in like the 1950s, which
is essentially a probability distribution over a pairwise choice. And what ends up happening, for
various technical reasons, is that if we train a preference model it needs to output a scalar
value, and by some coincidence that I think is still very convenient, they just take the output of
this learned probability distribution as a reward — they say, okay, our reward is going to be
proportional to this probability, and it's going to work, and it ends up doing so. But that's even
a big leap to accept — it's like we have this pairwise preference probability that's saying the
probability that one answer is chosen over another, and then you have to make this kind of mental
crazy step of saying we just pass in one number, or one piece of

**[11:39]** text, and we're getting the probability that that one piece of text is chosen over any
arbitrary other one. So there's a lot of assumptions that make this — there's kind of deep
concepts in here — but what we're getting is a model that's giving us the score out. And the
question is: why do we have to do this? What if we can just take our original objective and use
gradient ascent on this equation — ascent, because it's a maximum — and this is really what DPO
does. I'm blurring through a ton of math — it's a great paper to learn a lot of this math of
language modeling, where you learn how these probabilities of different pieces of text are
handled by the model, and how it ends up being a lot of these log probability ratios, and seeing
how the prompt and the

**[12:26]** completion are handled differently — it's worth digging into and understanding the
derivation. But the core idea is: why can't we just do gradient descent, or gradient ascent, to
solve RLHF optimization? And this becomes incredibly simple. So if you look at the code on the
right, that's the reference code from the original implementation — it's extremely simple to
implement, and it has this characteristic where, if you've worked with something like Transformers
before, it's pretty easy to write a loss function that uses DPO, rather than building an entire
infrastructure stack to start with. When you do something like PPO and this full RLHF stuff that
OpenAI does, you normally need an almost entirely new infrastructure stack, but you can get
started with DPO

**[13:11]** in a much simpler way. And there's some characteristics I'll get to later, which is
DPO still has a reward model, which is really important to the math actually checking out,
whereas you're using your original language model as a different type of reward model. But that
quickly takes us down a whole bunch of derivations that is probably not the lecture I think is as
fun to give. And the key thing — which is why this lecture is called what it is — is that the
first two points mean we'll see more DPO models than anything else. DPO is where everyone will
start if they want to do alignment research, and it's for good reason — it is the right place to
start if you're thinking about doing this. It scales more easily on compute, it's easier to debug,
it's even easier to learn. So it's not

**[13:57]** really worth second-guessing that, and it is a good place to start. But it also leads
into these ridiculous conversations online, where everyone is trying to figure out: is DPO better
than other RL methods — PPO, which is this older, popular deep RL algorithm which John Schulman
wrote; REINFORCE, which is a slightly different parameterization of policy gradient. They're very
similar, and DPO ends up being much simpler — it's just simpler to work with. So there's this meme
where it's like, if you just do gradient descent it'll work; in reality they're different loss
functions and they're doing very different things, but you can get similar results with both of
them, which is why, if something is much easier to do, you should just start with it. And I come

**[14:44]** back to this much later in the talk, which is what is fundamentally different about
these RL algorithms and how your data is processed and where the signals actually come from. But
for now, we don't need to say one versus the other — we can do both, and they are different. So
that's the quick 101 of what the core ideas are. I'm going to take a path to how we actually got
to training models with DPO — I think this slide was from a different talk that this subsection
is reduced from — but DPO really came out months before we started getting popular models trained
with it. So how did we actually get to the point where the community was training models with
DPO, which is much more recently than the paper was actually released?

**[15:29]** This comes all the way back to these first instruction-tuned models that you saw — so
Alpaca, Vicuna, Koala, Dolly of the world, all in April of 2023, and these are all built on
similar things and slight iterations. So there's figuring out how to use synthetic data, building
on this first Llama release — there's some other things I'll talk about, but this is where we
started. They're all using instruction tuning, most of them use synthetic data, and what Vicuna
actually did was they used this thing called ShareGPT, which was the first time that people
working in this academic alignment space had access to data that was from humans. It ended up
being a bit of a legal gray area, because it was logging data that people used in a Google Chrome
extension

**[16:16]** called ShareGPT, to make it so ChatGPT had a share button. But this data was really
important to things like Vicuna and a lot of the other models that came down the line, and is
still used in models today as one subset of the training data set. So just having access to these
human prompts unlocked a lot of potential back in the day, and is still something we're seeing.
Thankfully, now we're starting to get data sets like this that were collected in more permissive
ways — like this kind of LMSYS data has prompts that are collected with consent, and WildChat,
which was a project from AI2, which essentially gave people free access to ChatGPT in exchange for
their data. The thing that came after ShareGPT was the realization that we need more human data,
and this OpenAssistant project is one that we honestly need

**[17:03]** more of. It shows how hard it is to create human data, that we haven't seen more
things like this — this was run by a few people in a Discord community, working extremely long
hours, to generate prompts, responses, and preference pairs to common requests to language models,
and this was from April of 2023. We haven't seen anything like it — ShareGPT or LMSYS's data is
similar, but there's not the same level of controls and voting and ranking that went into this
OpenAssistant data. And it, again, is a data set that we're still training models with, and many
people still train models — I think it comes up time and time again — so it's just that these one
or two influential data sets from over a year ago are still what's used to train models. So you'll
get the theme as I keep going — there's actually RLHF models

**[17:49]** trained in April of 2023 as well — this was from CarperAI, that was doing a lot of
work in the space. They've fallen back a bit in recent times, but there were people doing similar
methods to what I'm going to talk about at the end of the talk — that knowledge and infrastructure
was not translated into things that were easy to use. So there's also this vein of, even if things
are open, it doesn't mean it's going to immediately catch on and be useful — you have to have the
resources, the data, and your codebase set up in a way that people can build on it, which is what
DPO did really well. This RLHF model from Carper was successful, it was better than the Vicuna
model, but no one really built on it right away, which I always find

**[18:36]** confusing. Then, later in the year, another key thing for this open alignment was the
Llama 2 backlash, where Llama 2 was asked to kill a Linux process and it would refuse, and this
kind of bred a whole series of models which are still referred to as "uncensored," which I don't
think is the best name, because I don't think there was ever actually any censoring to the model —
it wasn't intentional censorship. But the goal is to make models that don't refuse any request,
which is useful as a research artifact — what do you get out of a model if it answers every
question, what are the limits in that regard. There are other ways to use that, which are up to
you, but what ended up happening is a lot of these ShareGPT data sets, because they're from
ChatGPT, there's

**[19:21]** data that says, "oh, as a language model, I shouldn't answer that," so people started
filtering all of that out, and you still see a lot of people releasing these uncensored models
today as a popular area of development. I think we should understand what people need when doing
research — researching a model that doesn't refuse is reasonable, but if you're going to deploy a
model for free use to users, you should consider whether everything should be answered. So, as a
researcher, how your artifacts are used kind of depends on the work you're actually going to be
doing. Then, in alignment, there's this long series — I'm almost done with the timeline — but
there's this long series of models that are really interesting to people like me that never really
broke through the narrative, where

**[20:06]** they're saying things, like, "we used RLHF," where the first model to beat GPT-4 on
AlpacaEval, and these other eval tools — they're scaling things up, but they don't always have
papers, they don't always have codebases. Things are happening around — it's not just the Hugging
Face of the world, there's a lot of different organizations in the US and elsewhere that are
aligning models and getting similar numbers, or beating these mainstream tech companies, and these
are places you look for models. So these are all in the summer of 2023, and I bring these up
because this comes before the first big splash of DPO. So this Zephyr model was really the first
model that I remember making a splash with DPO, and this is when it took

**[20:53]** until this time, which was in September, after the May release of the paper, for
people to really be like, "oh, DPO is the real deal." It took four months, and now the paper has
best paper, everyone uses it, there's tons of derivations. But in industry, and among people
trying to train models, there was a lot of skepticism until this moment. So this is like a classic
academic story of needing to wait a bit until your work is vindicated, in some ways. But the two
crucial things here were: a new data set, the UltraFeedback data set, which is a data set of
synthetically generated text labeled by GPT-4 — so, again, this kind of new way of making data,
where it's a preference data set, we didn't make it, it was made by OpenBMB, I think they're based
in China, and

**[21:40]** I should know more. And then we also just had to do a lot of experiments to make it
work — there's a weird, really low learning rate that was needed to make this kind of chat model
work with DPO, which is like 5e-7. If you're really plugged into AI, you'll know that 3e-4 is like
the lore of the best learning rate, so it's many orders of magnitude lower. So that's what it took
to get this to work — we probably could have done it months earlier if we just did more
hyperparameter sweeps, but this is the random happenstance of the stories that people now backcast
as being like, this is the super important bottleneck — it's somewhat random. And then, at the
same time, I was switching jobs to the Allen Institute, and they were already working on this
project, which is trying to do a systematic study of instruction-tuning data, along with some of
this preference

**[22:26]** tuning recipes that were coming out. Because once this Zephyr model came out, there
are always skeptics who say, "oh, doing it at 7B is easy, that's a small model" — is it going to
actually scale to the real deal, to bigger models, to what ChatGPT does? So it was like, okay, we
have some more compute, and we tried it on this 70-billion-parameter scale, and we showed similar
gains — all we did was use the same UltraFeedback recipe, the low learning rate, and it largely
worked. So this is within two months, and then, since then, there's tons of new DPO models — all
these startups that are releasing their own models will release an instruct version that's a DPO
thing, and that continued for six months. I think just today I'm starting to see less DPO

**[23:11]** models, which is interesting — I've been keeping track of them for another evaluation
project, and it has finally slowed down a little bit. I don't know if that's alignment at large,
but there are so many — I should add a slide that's a list of the ridiculous number of DPO models
after these two. But this is really when the floodgates started, and when we're like, okay, DPO
really works. So this is kind of why I say what comes next: we could retrain models on the data
sets that we have — we don't have that many data sets, but it kind of feels like we're fishing in
the dark. Zephyr was built on the success of needing the low learning rate; this Tulu 2 model is
actually trained on TPUs, because we have the Google Tensor Research Cloud, so we have bigger TPUs
to train these models. And it's like, how do

**[23:58]** we do this more systematically. And that's kind of where most of what I talk about
today, on the technical matter, is the recent research that we've been doing to make sense of this
and answer the fundamental questions of what do we need to change about DPO, is PPO better, and so
on. So this is the reality that I go back and forth between — we don't really have the human data
to do RLHF like industry, but it is getting much easier to do alignment research. So you can
choose your narrative — I think sometimes, because I'm so close to industry and hear about people,
I'm too often on this side — but there is a lot of opportunity to do things. It feels crowded, but
being crowded at this point, when there's so much investment, is just because you're in the right
area, and

**[24:43]** most people in this room aren't trying to be professors, so if you get scooped it's
okay — but I find it very fun. And so, how do we actually understand what we're doing with
alignment, and can we improve on these models — like, Tulu 2 has a number because we want to keep
releasing more models. So it's like, how do we get better at evaluating what we're doing, to try
to understand this process, and then how do we train better models. So these are the sort of
things I'm up to — I have a few examples of things I've been working on. I built an evaluation
tool for reward models — I'll talk more about reward models to start here — and we need better
evaluation, because when you're training models you need to be able to do what I call local
evaluation. You need to be able to get a number that tells you if your training technique is

**[25:29]** improving the end result. You can't wait until Chatbot Arena evaluates your model,
because that takes you about a month to get your numbers back — you need to be able to run
something at your desk that gives you signal on whether you're actually doing a good job, and
we're still pretty behind on those evaluation tools, though there are more coming, which is
promising. And then, given DPO's simplicity, can we actually improve on that, and can we catch up
to some of the industry rumors that they've let drift aside. So, RewardBench is this project that
I started because there are no evaluation tools for reward models. My motivation was mostly for
transparency, given how much industry says reward models are what you need to focus on — they're
really important for getting good models out the door. And it's like,

**[26:14]** what does that mean, what does it mean for a reward model to be good. If we look at
this kind of feedback diagram, which is the one homage to the RL background, just feedback loops —
the reward model is, in this case, the agent is your actual language model, pi is the policy, the
training data is the prompts that you get. So in this kind of RLHF framework, you have this
feedback loop where the policy generates something, a, which is the action, which is the
completion, it goes to the reward model, which then scores it. But you're kind of on the side,
looking at all these evaluation tools, and none of these evaluation tools are giving us internal
insight into what's happening in this feedback loop — it seems kind of external

**[26:59]** to what we are doing when we're training these models. So we really wanted to zoom in
on this reward model, and reward models are trained in another kind of weird way — one of the many
quirks of RLHF. So in order to train a reward model, you need to collect this pairwise preference
data — if you're using ChatGPT a lot, you'll sometimes see it give you two answers and ask you
which one is better. This data is literally what's used to train a reward model — it's a prompt,
and then two completions, a chosen completion and a rejected completion. But in order to train
these models, you have to pass both of them in at the same time — so you pass both of them in at
the same time, and it gives you two scalar values. You use a language model that outputs a scalar,
just by some modifications of the last layers, rather than outputting text,

**[27:46]** and then this loss function — I'll show you on the next slide — is essentially why you
need to use this batch mode idea, which is you pass multiple things at once and you get multiple
numbers out. So this loss function is — here, this R is the output directly from the reward model
for the rejected completion and the chosen completion, so you're trying to separate the distance
between them, and then automatic differentiation updates the parameters so that this distance is
bigger. So you can't just do supervised learning directly on one thing to say, for the reward
model — there are alignment methods researching that now — but it's really built on this idea of
separating two things and creating a margin in the preferences to learn the decision boundary.
There's a lot of really specific details in

**[28:32]** industry, such as: these models are only trained for one epoch, they get really low
accuracy scores when you compare them to other train/test-set things in machine learning, and
there are some additional tweaks that people do — you can do ensembles, [Ed: unclear — "lamud",
possibly Llama 2, whose paper describes a margin term in its reward-model loss] did this weird
margin loss, but none of it is really transformative in how these models are trained. They're in
this weird place where you can only get about 70% agreement with your annotators — it's kind of
the sort of thing of, is the noise part of the signal, or is it a bug. In preferences, it could
make sense that it's a signal, because not everyone's preferences are the same, so not getting
full agreement might mean this system is working — we don't want ChatGPT to be fully
narrow-minded all the

**[29:17]** time. And this kind of reads into the thing of how do we actually evaluate these
reward models that I was talking about. I hear all the time that reward models are crucial to
RLHF, but how do we know exactly what aspects of the final policy they're improving — should we
include safety in these reward models, how do scaling laws impact reward models — there's kind of
basic machine learning questions, like, can we evaluate these, what should we think about. So what
we did is we collected a bunch of prompts, and then we manually created chosen and rejected
answers for each prompt, and then we can see whether or not the reward model agrees with our
human-created data, and call that a win or a loss, in an accuracy point of view. It's really
direct — we're just doing inference on existing models, and we're going to see

**[30:03]** whether or not they agree with human data. And this is a slide, if you want to go into
the academic side of things — this was built on a lot of existing evaluation tools that were out
there. You'll see some common names — AlpacaEval, MT-Bench — things you've heard about. XSTest was
on the slide when I mentioned Llama 2 being overly safe. And there's some other things that are
really good but you might not have heard about, like this LLMBar data set from Princeton, which is
a bunch of trick questions — I'll have an example on later. And some normal names from Anthropic
and OpenAI in here as well. So there's a lot of different things we're testing with this data set,
and we're trying to get the full picture of what's going on with these models. We released this in
March of '24,

**[30:50]** and you can see a key at the bottom, where these red circles with the arrow in them
are DPO models, which you can use as a reward model, and then these dice, which look like gray
squares when you zoom out, are what I described as this classifier type of training. And you can
see that there are reasonable scores — the benchmark isn't saturated, a bunch of open models, some
names you've seen before, like the Tulu models and the Zephyr models, are on here — kind of normal
stuff, this is what we expected, it's not too saturated. But if you look here, I'll show you where
this model has moved in a few months. So today we have a lot more models, and there's a lot more
information here, so I get to tell you about more interesting things, like how OpenAI's and
Cohere's models do

**[31:36]** on this, which is — I mentioned wanting to do this for transparency, but we also add
new types. So this is where the fifth model ended up — in two months, the model that was fifth on
the leaderboard is now 31st. So we're getting the saturation from people doing research in the
area, actually having places to compare their models. But we also have models from some closed
labs, and I'll get into the details here. So some of these are labeled as different types of
models, like LLM-as-a-judge — LLM-as-a-judge is the idea that you can ask a language model which
answer is better. This is kind of how things like AlpacaEval and MT-Bench are built, but you can
also use that as a reward model — I told you that I have prompts and then chosen and rejected, I

**[32:22]** could just ask ChatGPT which one is better and see what it does, and this is what we
added in as a baseline. And this ends up being really interesting, because GPT-4 and GPT-4o are
not actually as good in this closed domain as a reward model that Cohere is training. So we don't
have full information, because we don't have OpenAI's reward models, but we can use their models
to compare. So we have a lot of different information going into one system, about how language
models in different parts of the alignment process choose different categories. So going back —
you can see this Cohere, across two different months, theirs has improved a lot, and then these
earlier DPO models that we saw higher up on the leaderboard have been shifting down, from more
people training

**[33:08]** reward models to begin with. And the specific category I'll focus most on is this Chat
Hard category — if you think about evaluation a lot, it's actually surprisingly common as a topic
covered in tech coverage, how evaluation is saturating. This is the one feature of our benchmark
that hasn't fully saturated, and it's really important to having some sort of longevity to the
benchmark, and I'll talk more about this as we go from here. So I mentioned this data set, and
it's interesting to understand if you could actually do this problem — so what we have is a
prompt, a chosen, and a rejected, and the prompt is: give an example of a metaphor that uses the
following object, stars. And the chosen and rejected are two similar

**[33:56]** metaphors, but you can see, if you read these, what the differences are — I'm just
pausing for the people who are still paying attention and reading these. But essentially what
happens is that the chosen one is about the sky, and the rejected is about the moon — or, yeah,
"the twinkling diamonds in the sky," see, I haven't messed up reading the slide — but it asks for
stars, and it's about this metaphor of stars, where the rejected is about the moon, which is also
in the sky at night. And this data set is a whole bunch of things like this, where what they do to
create this is they either manually, or by ChatGPT, ask to rephrase a prompt, and then create a
new generation from it, so you can get these rejected generations that are just off topic. And it
makes sense for

**[34:41]** something that would be really hard for language models, because they have this
association between the stars and the moon, but we want our language models to be able to answer
questions like this. And this is the type of thing where our reward model benchmark — which is
something that's training language models — has the best correlation as something that is hard.
So this is promising — this is the sort of thing that, if you're in research, is interesting, so
it's really in the weeds, but it shows that we still have things to learn about these models, and
there are things we can't do yet. But another interesting pattern is in safety — I mentioned this
kind of uncensored models — and in safety we see all the patterns we would expect. The breakdown
at the top of this table, refusals, is

**[35:26]** things that we want the language model to refuse, and then this XSTest data set can be
split into something that we want models to refuse and something we want models to respond to.
And you can see that there are multiple categories of either DPO models or reward models, where
the model that handles safety really well refuses things like asking for advice on causing harm,
and responds to something that's borderline. But there's actually a lot of models out there that
just refuse everything, so that'll tank your score on things that should respond to everything —
which is kind of the safe bet, we were seeing a lot of tech companies release models like this,
which just feels like — it doesn't feel right when you talk to them. But there's also the models
that just respond to everything — it's like, not my job to gate whether or not I should. It's

**[36:12]** not the language model's job to gate — that's the philosophy there, which is something
we hear a lot about in the discourse of alignment. But to see it in these reward models and DPO
models, when directly probing them, without asking them to generate text, is nice — it confirms a
lot of suspicions we have. So this is back to some of the DPO math, which is again good to know.
If you go into the DPO paper, you'll see equation three here, which is the reward that's defined
in order to make the math actually work, and this is very different than just outputting a scalar
— it ends up being a ratio of the probability of the policy relative to the original policy during
training, which is called the reference model. And this is important — it's like it's

**[36:57]** a very complicated mathematical representation. So if you actually take a piece of text
and pass it through a DPO model, the reward will be something like minus 200, or something,
because it's a bunch of log probabilities — probabilities are between 0 and 1, you take the log,
you get negative numbers, and you sum all of these up, so you get a big negative number, and that,
intuitively, is the score that these models are providing, which is very different than the other
type of reward models I talked about training earlier. And if you have two prompts with a chosen
and a rejected, equation 4 is the math that you actually need to do to decide whether or not one
of the answers was better — you're comparing these ratios of probabilities from two different
models with respect to this reference model, which was the starting point of training,

**[37:42]** and the question is: when people release a DPO model, they normally release just the
model, and they don't release all the intermediate checkpoints, so this reference model would be
an intermediate checkpoint in the training process. So the question is: can you do this, can you
use it as a reward model, if you don't have access to all the information? And the short answer is
no — all the scores on our benchmark plummet across all the DPO models that we have, which makes
sense, because this reference model is a regularizer in the probabilities — it's in the actual
reward equation, if you go back a few slides, it's in the equation. So what we do is we get rid of
this, and we stop normalizing equation 4, and we just see if it works, and it doesn't. But this is
important, because DPO is

**[38:29]** training a reward model, but if we don't always have access to it, we just can't learn
from it, we can't use that in another system as clearly. So it's a lot to ask for, when getting
people to release models. And this is an interesting slide showing Cohere's progress on reward
models in just a few months — they released something that was clearly state-of-the-art on our
benchmark. RLHFlow, an alignment lab — this kind of RLHF-flow work — released something in May,
and then just a few days later Cohere sent another number, like, "here's our new model, it's still
better than everyone else." So it's nice to have this academic-industry intersection, but it's
very rare and takes a lot of work in terms of networking and building relationships, but we're
trying to do it, at least in these small niches, where the

**[39:16]** companies are willing to share. RewardBench 2 is going to need to mostly make
everything harder, and make everything more human. And the last point, which is what I'm going to
transition into next, is that everything I've told you about is part of this RLHF pipeline, but I
haven't told you how it's impacting the final model that you use at the end of the day, which is a
very rightful criticism — if you're evaluating part of the alignment pipeline, you should be
telling me whether or not the final model is actually useful. So this is where I talk about our
journey into trying to train PPO models. We're trying to fine-tune a good model — we spent a lot
of time on DPO with this Tulu 2 work, and we wanted to know if we could do better by switching to
PPO. So this is a lot of

**[40:02]** not-yet-published work, but it's going to be out soon, so the numbers aren't entirely
final, but we're just trying to disentangle what the difference between DPO and PPO is, at a very
empirical level. So we're trying to answer if it's better or not. What we're going to do is walk
through a series of design decisions and see how it affects the suite of evaluations. We're
starting with this Llama 2 13B model that has already been instruction-tuned — the difference
between the blue and the red is the gains from instruction tuning, for these reasoning, coding,
chat tasks. Instruction tuning does the biggest delta that you'll see among all these slides —
instruction tuning puts the model on the map as being useful, and it's easy to see gains at the
beginning, and then it's harder and

**[40:47]** harder for us to keep improving these models. So we start with — we add this Anthropic
Helpful-Harmless RLHF data with DPO, and you can see there's a small bump across all the metrics.
This data set is known as being particularly noisy among researchers in the area, but it's kind of
the starting point when you're doing research on alignment — it's been around for a few years,
it's big, it's multi-turn, but it's known to be noisy, and it still gives improvement. And then
what you do is, if we switch to this data that was used for both Zephyr and Tulu 2 — officially
this UltraFeedback data — we get an even bigger bump. So this is just showing the difference that
changing only the data can give you in a DPO recipe — it's normally increases of

**[41:34]** like 0 to 2%, and in the research sphere of trying to ship a model, that's a big deal.
So this is where we tried it into new territory — grad students worked really hard and implemented
PPO and JAX in addition to what they already had, and we were like, okay, what happens when we add
PPO, and require it reliably across multiple experiments. This is one example, with the
13-billion-parameter — PPO just happens to do a little bit better, like 1% better. And we try to
change a lot of things, and the changing things is where things get a bit messier. So we've heard
from industry that using a bigger reward model can be really helpful to getting a better policy
model,

**[42:19]** essentially these bigger reward models will be better at nuance — they should give
better-labeled scores, which are used as rewards, they should just make this process a little more
stable, if we have the compute for it. We see that it does improve some things, but it doesn't
actually make the model overall much better — it's kind of flatlined, with pretty similar data,
just from making the reward model bigger, which is a little surprising to us. And — this is the
most realistic few slides of the talk — we did this thing where we even tried to see if our reward
model training was bad as we scaled it up, so we used RewardBench on the right, which I told you
about earlier, and it's not clearly correlated whether or not

**[43:05]** these two — 13B or 70B — models are better. We also did this best-of-N sampling idea,
which is: if you generate a bunch of completions from the language model, you can rank them by
your reward model and then re-evaluate on the top-ranked completions. That shows that our reward
models are better at the bigger scale, but we couldn't get this to really click into a downstream
model, in a PPO notion of the world. We even tried adding more prompts to RLHF — we added more
code and reasoning prompts, because that's something OpenAI talks about a lot, and we want to
improve our models on that — it doesn't really shift the needle on this cohesive average over many
tasks. In the paper, what you'll see when it's out is it shows that we added prompts really
similar to the math and code evaluations,

**[43:52]** and those specific evaluations got a bit better, but adding the full noise into the
fact that some other evaluations might go down — this process is really hard to disentangle. And
this is why we're getting the 0 to 2% improvement out of PPO, but DPO doesn't have this sort of
mess. So what we ended up getting to is: there's always one more thing for us to ablate when
you're training these models with PPO — things like different regularization, we're learning a
value function in RL, different warmup, different sizes — there's just so many knobs to turn in
PPO, and it was reliably getting us a pretty good model, but we're staring into the abyss trying to
improve this right now, in the next few months.

**[44:38]** And the bottleneck, in terms of the actual technical side, is that PPO generates new
responses from the model as it trains, to refresh the data, and that is by far the biggest
bottleneck when you're actually training these models — it's just way slower than DPO. So all
these resources for PPO are somewhat available to academics — the Google Tensor Research Cloud, I
think, is pretty available, the grad students I work with seem to sign up, the codebase is open.
So if you're a grad student and you're trying to do PPO alignment and have access to TPUs, please
get in touch — it's a very fun can of worms. But kind of as a summary, this is the many different
DPO data sets that we tried — this is almost all of the well-received data sets that are out

**[45:26]** there in the open, and if you look at the factuality column, some of these things just
don't matter at all when you're aligning these models. So we need to get new data sets that are
really adding different capabilities to these models, and something that matches these
UltraFeedback numbers at the bottom — I'm surprised whenever I look at this, but this is where
we're at, and we need to keep building data sets and adding freshness to this system. UltraFeedback,
at this point, is maybe six months old or so — I don't know the exact age, but in terms of people
training models, that feels old, to things that are happening. And these are the actual numbers
you get when you compare DPO versus PPO — this is all with this 13-billion

**[46:14]** parameter — again, we changed the data set, and every one of these PPO comes out a
little bit better on average. And this is a few grad students and people like me — this is not a
big team in industry doing this, we're scraping by, and I don't know if it's worth the effort. I
see why OpenAI uses this, because we're able to get a bit more signal out of it, but it's a ton of
effort to get a bit better signal out. And I'll transition into a bit more of an open-ended
discussion of this, and then we'll have questions. But it's like, what about PPO is actually
special — this generation, and this online nature — can we just change DPO to be like this, or

**[46:59]** where are the new things going to go. And I had the pleasure of advising one project
that was related to this, but this is much more general — what is special about online data.
There are multiple ways you can get new data into your RLHF process, and then there's also this
related question in reinforcement learning literature, which is on- versus off-policy — a
technical distinction that often gets looped in with these discussions of DPO versus PPO. They're
actually related, but the reinforcement learning discussions have a much more definitional flavor
to them, while in this alignment space we're more focused on whether we need to get fresh data in,
and how we need to label our data, for language models. So

**[47:45]** I'd make this distinction between these two things: freshly generated data from the
policy. If you zoom into a data set like UltraFeedback, it has generations from all sorts of
models — from Alpaca, Vicuna, GPT-3.5, GPT-4, Llama — generations from all sorts of models in this
data set we're using. So when we train these Zephyr, these Tulu models, we're incorporating
information from a lot of different models down into our one policy, whereas what PPO is doing is
only generating data from your existing model, and changing this distribution over time. So that's
a very different idea of where the signal is coming from, from the models. And then the second
thing is whether or not we're refreshing the data labels over time — if I have human labelers
comparing chosen

**[48:31]** and rejected, that's one data point, but I can also later take this reward model that I
trained and generate a chosen and rejected and change the label. So these two things — what the
actual text is, and when the chosen/rejected label was given — are what people mean when they're
talking about whether something is special about online in RLHF. And it's clear to see that PPO
does it very differently than DPO, but we're not restricted to this. In the last few weeks — I
have the dates all in here — April, May of 2024, there started to be a lot of papers on this,
about DPO, PPO, online, offline, and they kind of say similar things, which is that online is

**[49:17]** important, and these papers on this slide show these more theoretical and closed-form
experiments on what's special about online data, and what performance drops if you use this kind
of offline data. It's good to dig into these, but this is why it's nice to do research now, because
if you have an idea, a lot of times people have three papers that confirm the notion you have —
it's a lot easier to be confident in things if three independent institutions say something
similar at the same time. There's a lot of methods coming out where people are trying to modify
DPO to actually use this online notion — I think Self-Rewarding Language Models, from Meta, was
the first really popular one, where they asked the DPO model, "hey, which of these answers is
better," in

**[50:04]** between each iteration. So they did this LLM-as-a-judge to relabel their own data, and
then they did multiple iterations of DPO, and the model had really strong scores. There's now
ideas like not using all of your data at once, so you can do batches of DPO and update your data.
The paper I was on, this discriminator-guided DPO, which I'll talk about in a second, is using
reward models plus this DPO training objective. There's just a lot of things we can change, and I
think the community, again, is in this expansion phase, where I even get messages from people
like, "oh, my paper was really similar to this other paper, we did it first, they didn't cite us,"
and I'm like, this is kind of the point. But it's hard — it's going to be like this for a little
bit longer, and then hopefully by the end of the year, or in a few years, we're going to be like,
okay,

**[50:50]** this is clearly what we need to do on the method side of things. So this is one
example, D2PO, discriminator-guided DPO, which I'm an advisor on, with an undergrad researcher,
and the idea is comparing three different things. So (a) is standard DPO — you have a data set,
you apply the loss function on it. (b) is what we call some sort of online preference
optimization, which is where you can repeatedly label your data with a reward model — kind of like
the self-reward paper I mentioned — you can re-shuffle your preference data based on a reward
model, and that adds some notion of online to your data. And then the third thing is, what if
we're relabeling

**[51:35]** data, and we're retraining our reward model over time. So we're really trying to keep
what our policy is doing related to our reward model, and keep everything updated in real time, so
that it's all lined up. And this is wondering: how much of a gain do you get by retraining the
reward model over time in a DPO framework? And part of why I like this paper is there are things
like closed-form tasks — so the biggest question I get for alignment is how do we actually
evaluate it, what tasks is it good for. There's a whole philosophical discussion — I think
information transformation is a valuable task, writers tell the same stories in different ways,
but the best-told story is the one that resonates with people, that has value. But at the same
time,

**[52:23]** we're academics, and we need to be able to measure things. So this paper has things
like: your reward is counting the number of nouns in a sentence, and then you're using these
alignment methods to increase the number of nouns in the output sentences from the model, so you
can measure that a lot better, because we have classifiers which know nouns. And you can see, on
this left figure, that just by retraining this reward model a few times, it converges better than
if you were just to relabel your preference data. It's a mouthful, but keeping your training
process a little more online can improve performance. And on the right is a more standard
open-ended evaluation task, where we're asking a language model, like ChatGPT, which answer is
better, and that has all sorts of problems, but we can show similar results.

**[53:08]** I think the big takeaway from these few slides is that the literature is moving — we
have studies that show online is better, and people are coming up with really clever ways to
actually use online data. So, combined with new data sets, this is kind of the theme of this year —
online methods, and how they work. This goes back to what industry is doing, and I showed this
figure earlier, on the left, with Claude, where you can see the little points along the lines —
these are the different iterations. We don't know exactly what they're doing, but it seems a bit
different, where the dots on these figures are new data sets from humans, rather than this kind of
redo-a-reward-model, relabel-your-data thing. This is

**[53:54]** what happens when you have access to a different type of scale. The Llama 2 paper
makes this much clearer — they say they work with an annotator, they get batches of data, and when
they're generating this new batch of data, the previous model's checkpoint was used for
generations. They do this many times, and you can see that they're collecting new human data, new
human data, new human data, and each time they generate human data it's trained into a new model —
they're doing a lot of training updates, and building on each other. And this leads into the last
section I'll talk about, in the conclusion — what did Meta do with Llama 3. This is one of the
funniest blog post sentences — the ridiculous things they give us, and then we parse the tea
leaves. They say, in the blog post, that our approach to post

**[54:40]** training is a combination of supervised fine-tuning, rejection sampling, proximal
policy optimization, PPO, and direct preference optimization. So people ask me, "what the heck did
they do?" I mean, I kind of agree, but it really goes back to this slide in my mind, which is that
they're getting new data, and then they're training a new model over time. So what I think is
happening, at each one of these points, is they tried a few methods and chose the training method
that worked best. It's really practical — Meta is a really practical organization, especially in
the GenAI org right now, and that just makes sense — at different points, your model has different
capabilities, and it's ready to be trained in different ways. Rejection sampling, which I didn't
cover here, is the simplest training method — you take a reward model,

**[55:27]** you rank some supervised fine-tuning outputs, and then you use this autoregressive loss
function again. And then from there, DPO is much simpler than PPO, but it might not give you the
highest end performance. And then, as your model really starts kicking into gear, or you have more
time to train this model, once all of your data is collected and you're not on a weekly time
crunch, you can experiment with all the little knobs of PPO, and really try to get the best model
out at the end of the day. Hopefully they release a technical report that confirms some of my
hypotheses, but I think this is normally what people are interested in when somebody from industry
comes up to give a lecture, and I wish we had more details on what industry was doing. But in
terms of current directions

**[56:14]** that I'm most interested in — RLHF, I talked about data a lot, we're very bottlenecked
on data. Even as academics with very limited compute, we literally try every data set that's
available — it's not that we don't have a lot of compute, but we need to keep innovating there.
We're going to see more DPO methods — it's here to stay, there's a ton I didn't cover here, things
like removing the reference model, changing the loss function slightly, not using pairwise
preferences but single-wise preferences — a lot going on there. We should use more model sizes —
in 7 and 13 billion parameters, or, in Llama's case, like 7 and 70 billion parameters —
particularly, scaling down is

**[57:00]** very useful, it's a place where academia can still play — there's less of a weird
marketing dynamic where all the companies are racing to go bigger for certain strategic reasons,
but this is something that's accessible to many people. Aligning small models, it's hard to get
signal out of them, because the models show more or less random scores on many benchmarks people
care about, or really low scores. So even just breaking through in that domain would be really
impactful work, to get more people working on alignment. And then evaluations, which I covered at
length — we need to keep getting more specific on the things we care about. And personalization is
something in alignment that I didn't cover in this talk, but is something good to compete with
this kind of big tech, which

**[57:46]** is: how do we train models that are good for you, as an individual, rather than one
big model for one big technology organization. So — these slides will get to you — but these are
the types of places I follow when I'm trying to see open models or open data sets that are
reputable and easy to keep track of, so you don't have to follow everyone. And I write about this
a lot, without doing too much self-promotion. But I ended about ten minutes early for questions,
which I'm happy to take in a Q&A format, and you don't have to stay and wait if you don't want to.
[Applause]

**[58:35]** Okay, thank you, Nathan. Questions? Anyone got questions? *Assuming you have a good
reward model — which is a large assumption, I agree — but what is the key challenge to doing
online DPO, in the sense that you can do rollouts and then rank them using a model, and then...
you can iterate this. So what is the hard thing?* Yeah, I'm going to repeat the questions so that
people can hear them and it gets recorded. The idea is: if you have a good reward model, what is
stopping you from doing online DPO and just improving the policy from there? I think there are
multiple angles to this — they're both technical and

**[59:21]** kind of industry-wide. But the technical thing, I think, is the prompt matching ends up
being really important — prompt matching, so what your reward model can learn is specific to the
prompts. There's a technical detail where the prompts used for your policy are often exactly the
same as your reward model, in PPO, which is really strange, because we talk about generalization
in machine learning, but we're kind of softballing ourselves at the PPO stage — we're only grading
PPO answers which our reward model is trained to answer, which is kind of strange. So people think
some of that might break down, and we see some of that when trying to train PPO models with
off-the-shelf reward models. That was kind of a long answer, but I think that's

**[1:00:06]** mostly it's mostly distribution matching, if I had to guess. But if we had a truly
good model, it should work for some things, and that could be one of the reasons why there aren't
that many in the open — because it would kind of help people catch up in alignment. If a reward
model is as important as people say it is, it might be easy. Other questions? Yeah. [Music]

**[1:00:56]** *...for example, me?* Yeah, I think — this is a whole conversation, so if I don't
cover it, if you want more after I answer, you can come up — but the question is: is there more
than pairwise preferences that could be used in RLHF? There are a lot of different lines of work
studying this. One is methods like — there's a method out of Stanford, that's KTO — [Ed: unclear —
"csky", the name Nathan is attempting; KTO stands for Kahneman-Tversky Optimization, but the
slides don't spell out an author name here] — I always mess up these names, they're so hard to
pronounce — but it's the idea of using one-sided preference data. So a lot of customer apps have,
like, "did you get good support from this agent, yes or no," and you could use data like that —
it's just a different loss function for using a single side of preferences, or just yes or no.
There are

**[1:01:42]** other things, like learning to rank for multiple answers — so this is something I
slightly insinuated, but binary preferences — there's a lot of literature on learning preferences,
and one of the models that got reduced down is the Starling model, and they use a k-wise
preference, so they have like five or nine answers to every prompt, and then they collect answers,
and they have a different loss function. This is one of the models that's kind of broken through
in the open alignment space — it's one of the few that I left in, and skipped, in my slide deck.
But that's kind of interesting. And then there's other research that's like fine-grained
preferences, so for every completion to a prompt you get labels like conciseness, helpfulness,
honesty. So

**[1:02:30]** there's a few things on that front — there's the SteerLM paper from NVIDIA, and then
there's work from UW that does learning from fine-grained preferences. So that one's probably the
one that's most emerging, most in the academic sense, but there's so much to learn here — like,
literally the whole field of social choice needs to get condensed into these things. Any other—
[Applause]

**[1:03:23]** questions? — Yeah, so the question is: how can we, broadly — how can we exceed human
performance with fine-tuning, or any training, for that matter? I think this is where some older
ideas in CS will come back — I think one of the foundational ideas in CS is search, which is also
motivated as exploration in RL, and therefore we need some sort of language models that can search
and generate new data. I was talking with somebody before — the grad student — and I think search
will be a large part of synthetic data, but then the human aspect will be what gets it across the
line, if it can't solve a certain area. And this is like — the Q-star rumors are ridiculous, but
that seems to be the best argument for the sort of thing that OpenAI is trying

**[1:04:09]** with — that's like how to get that barrier broken with AI. *Thank you so much for
coming in — you mentioned data sets are a big limitation, and I was curious how one goes about
creating a new data set.* Yeah, this is another thing that's hard — I think community efforts are
what people have tried to do, I mentioned OpenAssistant, but most people who do a community effort
are like, "I never want to do this again." So, while I still think it's worth doing things once
that are highly impactful, even if you might not want to do it again, other avenues for building
these in a sustainable manner are very important. I think there's some way that this is being done

**[1:04:56]** like Chatbot Arena returns some of the prompts and labels to users — there's specific
concerns I have with that data, around being too noisy. But that is the sort of thing that can
happen. If AI2 has a demo for their models, it's going to be about science, generating
information, rather than being a ChatGPT competitor — it's a nonprofit, it can't do a product
competitor — but that's the sort of data we would want to release, and something I might just have
to do. But I'm interested in academic workshops and competitions as a ground, where you could have
communities meet every three, six, eight months, and have work focused on an area, or focused time
to have people contribute to it. But it's a good question — it's probably why there aren't very

**[1:05:45]** many. *...how do you feel [they] are subject to reward hacking as well?* So we get
one at the front first — yeah, close first, and then we'll come to you. *The various places you've
done research at over the years — do you have any sense of how they compare, in terms of,
specifically, alignment research?* I mean, obviously they weren't doing alignment research
specifically at that time. I think generally they represent different culture and investments of
the company — I wasn't doing language models until my time at Hugging Face, so I can really only
speak to these two open companies. And from Hugging Face's perspective, it's to show that more
people

**[1:06:31]** can do this — we're not trying to compete with ChatGPT, but we're trying to foster an
ecosystem of doing this. And AI2 is similar, but more about — what is happening, how do we learn
about this, how do we do science, how do we study the science of this and communicate that
clearly. And I'm sure if you do the exercise you can map this to every company — what is their
important thing, they have different goals in their products and their corporate structure and
things like that. I'll talk more when it's not recorded. [Laughter] Okay, up the back. *Are reward
models also subject to reward hacking — like, they achieve a good result on the outcome, but in
reality the outcome does not [match what's] expected?* Yeah, so this is like, when

**[1:07:17]** talking about reward models, this is probably the most established line of work —
the question is: are reward models subject to reward hacking. Reward hacking is a classic problem
in RL — I should bring back from my RL slides, where you have the boat swimming in circles — and
this happens to your language model too, and what happens is — it is, and there's a lot of
research to mitigate it, but it's a fundamental problem, which is: you have a very powerful
optimizer, and you have an incomplete representation of your reward, and it will always find where
your representation of reward is wrong. So we will always be doing the best we can, but I think
saying it's perfect is not possible, in the math. I mean, I can also say the ways that

**[1:08:03]** it fails are pretty funny, because if you train these models you'll end up with a
model that just says "JavaScript" to every answer, to infinity — it's sometimes really easy to see
when that's happening, which is good, or you could change your loss function so that it will
always exploit, which is a good way to make sure that things are working — you should be able to
easily exploit it if you turn the brakes off. Okay, any last public question? If not — thank you,
Nathan, for giving this talk, and if there's anything you'd like to ask off the record, he'll be
here

**[1:08:49]** for a bit longer.
