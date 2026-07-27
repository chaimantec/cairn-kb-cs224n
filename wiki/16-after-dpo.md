# Lecture 16 — After DPO

A guest lecture from **Nathan Lambert** of the Allen Institute for AI, and the course's most
current snapshot of what post-training research actually looks like from inside it. The deck is
titled *Life after DPO*, and Lambert admits the title is "a little bit of an unclear title, so I
apologize about this" (≈1:39) — what it means is that
[Direct Preference Optimization](direct-preference-optimization.md) was the story of the previous
year, it made alignment research accessible to people without an RL infrastructure team, and the
open question is what comes next.

Three things make this lecture worth reading even though the field has moved. First, it is
explicit about the **resource gap** between academic and industrial alignment, with numbers.
Second, it walks through two large empirical projects — RewardBench and a PPO-versus-DPO ablation
— including the parts that did not work, which papers usually bury. Third, much of the
"unpublished, coming soon" work it describes is presented with its uncertainty intact: "the
numbers aren't entirely final" (≈40:02).

Chris Manning's introduction places Lambert's background in reinforcement learning, notes he was
at Hugging Face before AI2, and makes the framing point for the whole lecture: "more and more of
the action of the large language model companies is happening not in the initial pre-training
language model training phase, but in this subsequent post-training phase" (≈1:39).

**Slide-by-slide text of this deck: [86 slides](../raw/slides/16-after-dpo.md)** — every page
from 2 onward prints its number in the bottom-right corner, matching the PDF page 1:1.

Slides PDF: [Life after DPO](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture15-life-after-dpo-lambert.pdf) ·
[Full transcript](../raw/transcripts/16-after-dpo.md)

> **A note on numbering.** This lecture sits at **position 16** in the playlist this knowledge
> base follows; the video title calls it "Lecture 15". Repo files use the position.

## The gap the talk is about

Lambert opens with a comparison he says is not in the slides, from a conversation with Manning
just beforehand (≈2:26). The amount of preference data Meta bought from a single vendor for
Llama 2 exceeds everything collected on Chatbot Arena:

| Source | Scale |
| --- | --- |
| Llama 2 purchased comparisons (per the paper) | ~1.5 million |
| Chatbot Arena, total collected (as of ≈May 2024) | ~800,000 |

And the Llama 2 figure is "years outdated" — "you can only imagine what OpenAI, Anthropic, etc.
are buying at this scale, and this is the kind of reality that we need to adapt to" (≈2:26).

That constraint shapes everything after it. Academic alignment research cannot out-collect
industry, so it competes on evaluation, on method, and on scaling *down*.

## Can ChatGPT exist without RLHF?

Slide 6's question, and the answer is a careful no (≈5:30). Most capability comes from
pre-training, but at every landmark model, "RLHF and these human-related or other fine-tuning
technologies seem to be **necessary but not sufficient**." You need the pre-training, and you
also need post-training "to really shift the needle on what the most important models are at that
certain moment."

Two pieces of evidence (slides 7–9). Anthropic's Constitutional AI paper contains what Lambert
calls "one of the most representative figures of what RLHF can do" — a helpfulness-versus-
harmlessness plot in which successive RLHF variants trace a Pareto improvement over the base
model (≈5:30). And Meta's Llama 2 technical report contains a passage he finds funny:

> Reinforcement learning, known for its instability, seemed a somewhat shadowy field for those in
> the NLP research community. However, reinforcement learning proved highly effective,
> particularly given its cost and time effectiveness.

Written in July 2023 "back in the day when we were like, oh, we don't know if RLHF is really
going to take off," and it "aged really well" (≈6:15).

## Vocabulary

Slide 11 is a definitions slide, and the distinctions matter because the literature uses them
loosely (≈7:01):

- **Instruction fine-tuning (IFT)** — training a model to follow instructions. The popular one,
  and the one most linked to RLHF, because it is "about making these models really useful and
  really engaging and easy to work with."
- **Supervised fine-tuning (SFT)** — the domain-specific version. You want both.
- **Alignment** — "super vague, but it's in the word — it's align — it's training a model to be
  mirrored to what a user wants, and there's a lot of things you can align to" (≈7:48).
- **RLHF** — one specific tool for doing alignment, using human feedback data. Lambert flags
  *feedback* as "a really loaded word."
- **Preference fine-tuning** — a phrase he "tried to make … at one point, but didn't really double
  down on it," offered as clearer than RLHF in the DPO era.

Instruction tuning remains the foundation, and it is where **system prompts** live — structure
that makes a model "ready for a specific style of input," and that lets a developer pass
information the user does not see (≈8:34). OpenAI was still innovating here, having just released
a model spec document proposing a second-level system prompt. The data behind it often looks like
Stack Overflow or Reddit — a question then an answer — and it still uses the plain autoregressive
loss: "we haven't branched out into different loss functions yet" (≈9:21). Some academic work
argues instruction tuning is all you need, which Lambert calls "a much more mixed bag, but it's
the simple method, and it's the right place to start."

## The RLHF objective

Slide 14 (≈9:21). Familiar to anyone with RL training, less so from an NLP loss-function
perspective:

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(y \mid x)}\Big[r_\phi(x,y) - \beta \mathbb{D}_{\mathrm{KL}}\big[\pi_\theta(y \mid x) \,\|\, \pi_{\mathrm{ref}}(y \mid x)\big]\Big]$$

Here $\pi_\theta$ is the policy being trained, $x$ a prompt drawn from dataset $\mathcal{D}$, $y$
a completion sampled from the policy, $r_\phi$ the learned reward model, $\pi_{\mathrm{ref}}$ the
frozen reference policy the run started from, and $\beta$ the strength of the constraint.

The left term is the standard RL objective — learn a policy maximising a reward. The right term is
a **KL constraint**, "a distance, so that the policy doesn't change too much," and it exists
because of over-optimisation: "the key idea is that we want to optimize a reward but not
over-optimize it" (≈10:07). Two questions follow from this equation and organise everything
after: *how do we implement a reward function*, and *how do we optimise it*.

### Where the reward comes from

The preference model is the **Bradley-Terry** model, from economics in the 1950s — a probability
distribution over a pairwise choice (slide 17, ≈10:53):

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\big(r^*(x,y_1)\big)}{\exp\big(r^*(x,y_1)\big) + \exp\big(r^*(x,y_2)\big)}$$

Lambert lingers on how odd the standard move is. A reward model must emit a scalar, so "by some
coincidence that I think is still very convenient," people take the learned quantity inside that
probability as a reward and it works. But accepting it is "a big leap": the model was fit to say
which of two answers is preferred, and it is then asked, given *one* piece of text, for the
probability it would be chosen over "any arbitrary other one" (≈11:39).

Reward-model training is correspondingly odd (slides 40–41, ≈26:59). You collect **pairwise
preference data** — a prompt, a chosen completion, a rejected completion, the thing ChatGPT asks
you when it shows two answers. Both completions must be passed through together, because the loss
works on the *difference* between the two scalars: the model "is trying to separate the distance
between them," and gradient descent widens that gap (≈27:46). You cannot supervise directly on a
single example; the method "is really built on this idea of separating two things and creating a
margin in the preferences to learn the decision boundary."

Industry details Lambert flags (≈28:32): these models are trained for **one epoch**; their
accuracy numbers look terrible next to ordinary train/test machine learning; ensembles and margin
losses exist but "none of it is really transformative." Agreement with annotators tops out at
about **70%**, which raises the question "is the noise part of the signal, or is it a bug." His
answer is that for preferences it may genuinely be signal — "not everyone's preferences are the
same … we don't want ChatGPT to be fully narrow-minded all the time."

## What DPO changed

Slides 18–21. The question is: why train a separate reward model at all — "what if we can just
take our original objective and use gradient ascent on this equation" (≈11:39)? That is what
[DPO](direct-preference-optimization.md) does. Lambert deliberately skips the derivation ("I'm
blurring through a ton of math") while recommending the paper as a way to learn how probabilities
over text behave, "how it ends up being a lot of these log probability ratios, and seeing how the
prompt and the completion are handled differently" (≈12:26).

What he emphasises instead is *engineering* consequence (slide 20, ≈12:26):

- The reference implementation is "extremely simple." If you have used Transformers, "it's pretty
  easy to write a loss function that uses DPO, rather than building an entire infrastructure
  stack to start with."
- PPO-style RLHF "normally needs an almost entirely new infrastructure stack."
- **DPO still has a reward model** — "which is really important to the math actually checking
  out" — except that the language model itself plays that role (≈13:11).

Hence the lecture's thesis: "we'll see more DPO models than anything else. DPO is where everyone
will start if they want to do alignment research, and it's for good reason … It scales more
easily on compute, it's easier to debug, it's even easier to learn" (≈13:11).

Slide 21 is the "midwit" meme about DPO versus PPO and REINFORCE — PPO being the older deep-RL
algorithm John Schulman wrote, REINFORCE a slightly different parameterisation of policy gradient.
Lambert's read is deflationary in both directions: "in reality they're different loss functions
and they're doing very different things, but you can get similar results with both of them, which
is why, if something is much easier to do, you should just start with it" (≈13:57). And: "we
don't need to say one versus the other — we can do both, and they are different" (≈14:44).

## How the community actually got to DPO models

Slides 22–32, and the section that is hardest to reconstruct from papers, because it is about
release dynamics rather than results. DPO's paper appeared in May 2023; models trained with it
did not make an impact until September (≈14:44).

**April 2023 — the first open instruction-tuned models.** Alpaca, Vicuna, Koala, Dolly, all built
on the first Llama release, mostly using synthetic data (≈15:29).

**ShareGPT** was the unlock. Vicuna used it, and it was "the first time that people working in
this academic alignment space had access to data that was from humans." It came from a Chrome
extension that added a share button to ChatGPT and logged the results — "a bit of a legal gray
area" — and it is still a subset of training data today (≈15:29). More permissive successors
followed: **LMSYS** data collected with consent, and **WildChat**, an AI2 project that "gave
people free access to ChatGPT in exchange for their data" (≈16:16).

**OpenAssistant** (April 2023) is the one Lambert singles out as under-replicated: run by "a few
people in a Discord community, working extremely long hours," producing prompts, responses and
preference pairs with a level of "controls and voting and ranking" that ShareGPT and LMSYS data
lack. "We haven't seen anything like it," and models are still trained on it — "these one or two
influential data sets from over a year ago are still what's used to train models" (≈17:03).

**CarperAI** trained RLHF models in April 2023, using methods close to what the rest of the talk
covers, and it went nowhere — "that knowledge and infrastructure was not translated into things
that were easy to use." The model beat Vicuna and "no one really built on it right away, which I
always find confusing" (≈17:49). His lesson: openness alone is insufficient; "you have to have
the resources, the data, and your codebase set up in a way that people can build on it, which is
what DPO did really well."

**The Llama 2 backlash** (≈18:36): Llama 2 chat refused to kill a Linux process, which "bred a
whole series of models which are still referred to as 'uncensored'." Lambert dislikes the name —
"I don't think there was ever actually any censoring to the model, it wasn't intentional
censorship" — while defending the research use: knowing "what do you get out of a model if it
answers every question" is legitimate. Mechanically, these models came from filtering "as a
language model, I shouldn't answer that" strings out of ShareGPT data (≈19:21). His caveat is
about deployment rather than research: "if you're going to deploy a model for free use to users,
you should consider whether everything should be answered."

**Zephyr β** (September 2023) is where DPO landed (≈20:53). Two things made it work: the
**UltraFeedback** dataset — synthetically generated text labelled by GPT-4, built by OpenBMB, not
by the Zephyr team — and a very low learning rate of **5e-7**. "If you're really plugged into AI,
you'll know that 3e-4 is like the lore of the best learning rate, so it's many orders of magnitude
lower" (≈21:40). Lambert is pointed about what this says about research narratives: "we probably
could have done it months earlier if we just did more hyperparameter sweeps, but this is the
random happenstance of the stories that people now backcast as being like, this is the super
important bottleneck — it's somewhat random."

**Tulu 2** answered the scale objection (≈22:26). Sceptics said 7B is easy; AI2 applied the same
UltraFeedback recipe and low learning rate at **70B** on TPUs from the Google TPU Research Cloud
and "showed similar gains." After that, "the floodgates started" — every startup shipping an
instruct model shipped a DPO one, for roughly six months, and by the time of the talk that had
"finally slowed down a little bit" (≈23:11). **SteerLM** (NVIDIA) and **Starling** represent the
parallel RLHF/PPO thread.

The section closes on the honest summary (slides 33–37, ≈23:58): "we don't really have the human
data to do RLHF like industry, but it is getting much easier to do alignment research." Two
questions structure the rest — can we evaluate better, and can we improve on DPO.

## RewardBench

Slides 38–61. Lambert built [RewardBench](rewardbench.md) "because there are no evaluation tools
for reward models," motivated mainly by transparency: industry insists reward models are what
matters, so "what does it mean for a reward model to be good" (≈25:29).

The practical argument is **local evaluation** (≈24:43). When training models you need a number
at your desk that tells you whether a technique helped. "You can't wait until Chatbot Arena
evaluates your model, because that takes you about a month to get your numbers back."

The construction is deliberately plain (≈29:17): collect prompts, manually create a chosen and a
rejected answer for each, and check whether the reward model agrees with the human-made pair —
a win or a loss, scored as accuracy. "It's really direct — we're just doing inference on existing
models." The dataset (slide 44) draws on AlpacaEval, MT-Bench, XSTest, Princeton's **LLMBar**
("a bunch of trick questions"), and prior sets from Anthropic and OpenAI, grouped into **Chat,
Chat Hard, Safety, Reasoning** and **Prior Sets**.

Released March 2024. By May, "the model that was fifth on the leaderboard is now 31st" (≈31:36) —
saturation driven by people finally having somewhere to compare. Findings Lambert draws out:

- **LLM-as-a-judge is not state of the art.** Asking GPT-4 or GPT-4o which answer is better is
  included as a baseline, and both "are not actually as good in this closed domain as a reward
  model that Cohere is training" (≈32:22). See [LLM as a judge](llm-as-a-judge.md).
- **DPO models drift down.** Early DPO models scored well when few dedicated reward models
  existed; as more people trained actual reward models, the DPO entries fell (≈32:22).
- **Chat Hard is the category that has not saturated**, which is what gives the benchmark
  longevity (≈33:08).

### The Chat Hard example

Slide 53, and worth doing yourself (≈33:08). The prompt asks for *a metaphor that uses the
following object: stars*. Both candidate answers are plausible metaphors; the chosen one is about
stars ("the twinkling diamonds in the sky") and the rejected one is about the **moon** — also in
the night sky, and strongly associated with stars (≈33:56). LLMBar builds these by asking for a
rephrased prompt and generating from it, producing rejected answers that are fluent and
off-topic. It is hard "because they have this association between the stars and the moon, but we
want our language models to be able to answer questions like this" (≈34:41).

### Safety patterns

Slide 54 splits refusal behaviour into what a model *should* refuse and what it should answer —
XSTest supplies both halves. Three patterns appear (≈35:26):

1. Models that handle safety well: refuse requests for harmful advice, respond to borderline ones.
2. Models that **refuse everything**, which tanks the should-respond score — "the safe bet, we
   were seeing a lot of tech companies release models like this, which … doesn't feel right when
   you talk to them."
3. Models that **respond to everything** — "it's not the language model's job to gate, that's the
   philosophy there."

The value is methodological: seeing these patterns "in these reward models and DPO models, when
directly probing them, without asking them to generate text, is nice — it confirms a lot of
suspicions we have" (≈36:12).

### Can a DPO model be used as a reward model?

Yes, and the maths is why DPO works at all (slides 55–57, ≈36:12). The DPO paper's equation 3
defines the implicit reward as a log-ratio against the reference policy:

$$r(x,y) = \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

and comparing two completions reduces to comparing those ratios:

$$\log \frac{\pi(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} > \log \frac{\pi(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)}$$

This is "very different than just outputting a scalar." Pass text through a DPO model and "the
reward will be something like minus 200," because it is a sum of log-probabilities, each negative
(≈36:57).

The practical problem is that $\pi_{\text{ref}}$ is an *intermediate checkpoint* which almost
nobody publishes — "when people release a DPO model, they normally release just the model"
(≈37:42). So: can you use a DPO model as a reward model without it? "The short answer is no — all
the scores on our benchmark plummet across all the DPO models that we have," which makes sense
because the reference model acts as a regulariser and is in the reward equation itself. Slide 56
shows exactly this by crossing out every $\pi_{\text{ref}}$ term. The conclusion is a release-
practices one: "DPO is training a reward model, but if we don't always have access to it, we just
can't learn from it" (≈38:29).

Slides 58–60 track **Cohere's** closed reward models improving over months, with RLHFlow briefly
taking the lead in May before Cohere replied days later. Lambert values the exchange while noting
its cost: "it's very rare and takes a lot of work in terms of networking and building
relationships" (≈38:29). RewardBench 2 "is going to need to mostly make everything harder, and
make everything more human."

## Fine-tuning a good model: PPO versus DPO

Slides 62–74, and the section carries a caveat Lambert states up front: "this is a lot of
not-yet-published work … the numbers aren't entirely final" (≈40:02). See
[PPO vs DPO](ppo-vs-dpo.md) for this ablation on its own.

The method is a build-up: start from Llama 2 13B, add one design decision at a time, and watch a
fixed evaluation suite (factuality, reasoning, coding, chat, safety, truthfulness). Slides 64–70
add one bar series per slide. The results, in order:

| Step | Effect |
| --- | --- |
| Instruction tuning (SFT) | **The biggest delta in the whole talk** (≈40:02) |
| + DPO on Anthropic HH-RLHF data | a small bump across all metrics |
| + DPO on UltraFeedback instead | a bigger bump — from changing *only* the data |
| Switch DPO → PPO | ~1% better |
| Scale the reward model 13B → 70B | "kind of flatlined" |
| Add more code/reasoning RLHF prompts | those evals improve; the average does not move |

"Instruction tuning puts the model on the map as being useful, and it's easy to see gains at the
beginning, and then it's harder and harder for us to keep improving these models" (≈40:02). The
HH-RLHF data "is known as being particularly noisy among researchers in the area, but it's kind
of the starting point when you're doing research on alignment," and still helps (≈40:47).
Swapping in UltraFeedback gives more — "just showing the difference that changing only the data
can give you in a DPO recipe" — with typical gains of 0–2%, which "in the research sphere of
trying to ship a model … is a big deal" (≈41:34).

The reward-model scaling result is the interesting negative. Industry says bigger reward models
give better-labelled scores and a more stable process; here it "does improve some things, but it
doesn't actually make the model overall much better" (≈42:19). They checked whether their own
larger reward model was simply badly trained, two ways: RewardBench scores were "not clearly
correlated" between the 13B and 70B versions, while **best-of-$n$ sampling** — generate many
completions, rank with the reward model, evaluate the top-ranked ones — did show the bigger reward
model to be better (≈43:05). So the reward model improved and the downstream policy did not.

The overall verdict is deflationary and unusually frank (≈45:26, ≈46:14): across datasets "every
one of these PPO comes out a little bit better on average" — slide 74's table quantifies it as
**PPO outperforming DPO by 1.2% on average** across five fixed datasets at 13B — but "this is a
few grad students and people like me — this is not a big team in industry doing this, we're
scraping by, and I don't know if it's worth the effort. I see why OpenAI uses this, because we're
able to get a bit more signal out of it, but it's a ton of effort to get a bit better signal out."

Two structural reasons PPO is harder (≈43:52, ≈44:38): "there's always one more thing for us to
ablate" — regularisation, the value function, warmup, sizes — and, technically, **PPO generates
new responses from the model as it trains**, which "is by far the biggest bottleneck … it's just
way slower than DPO."

Slide 73's data ablation across nearly every well-received open preference dataset produces its
own conclusion: look at the factuality column and "some of these things just don't matter at all
when you're aligning these models." UltraFeedback still leads, and "at this point is maybe six
months old or so … in terms of people training models, that feels old" (≈45:26). See
[preference data](preference-data.md).

## What "online" means, and why it might be the real difference

Slides 75–81. If PPO's advantage is small but real, what is it *from*? Lambert's answer is that
PPO generates fresh data as it trains, and separates that into two distinct axes (≈47:45):

1. **Freshly generated data from the policy.** UltraFeedback contains generations from Alpaca,
   Vicuna, GPT-3.5, GPT-4 and Llama, so DPO training on it "incorporat[es] information from a lot
   of different models down into our one policy," whereas PPO only ever generates from the current
   model, shifting that distribution over time.
2. **Freshness of the labels.** A human comparison is one fixed data point, but a trained reward
   model can re-label chosen and rejected later (≈48:31).

Those two — what the text is, and when the preference label was assigned — "are what people mean
when they're talking about whether something is special about online in RLHF." He notes this
overlaps with, but is not identical to, the RL literature's **on-policy versus off-policy**
distinction, which "has a much more definitional flavor" (≈46:59).

By April–May 2024 several papers had converged on the same conclusion — online matters — and
Lambert makes a methodological point about that: "it's a lot easier to be confident in things if
three independent institutions say something similar at the same time" (≈49:17).

Methods trying to give DPO an online character (slide 78, ≈49:17): **Self-Rewarding Language
Models** from Meta, which asks the DPO model itself which answer is better between iterations —
LLM-as-a-judge used to relabel its own data — then runs several DPO rounds; batched DPO that
updates data between batches; and **D2PO**.

**D2PO** (discriminator-guided DPO, slides 79–80), a paper Lambert advised on, compares three
regimes (≈50:50):

- (a) standard DPO — a fixed dataset and the loss function;
- (b) online preference optimisation — repeatedly re-label the data with a reward model;
- (c) additionally **retrain the reward model over time**, keeping policy and reward model in step.

Its evaluation includes a deliberately synthetic closed-form task — the reward is *the number of
nouns in a sentence*, so progress is measurable with a classifier rather than a judge (≈52:23).
The result: retraining the reward model a few times converges better than only re-labelling
preference data. "Keeping your training process a little more online can improve performance."
The right-hand panel repeats it on a standard open-ended evaluation with a language-model judge,
"which has all sorts of problems, but we can show similar results."

Lambert grounds the synthetic-task choice in the field's real difficulty: "the biggest question I
get for alignment is how do we actually evaluate it" — he thinks information transformation is a
genuinely valuable task, since "writers tell the same stories in different ways, but the
best-told story is the one that resonates with people" — "but at the same time, we're academics,
and we need to be able to measure things" (≈51:35).

## What industry appears to be doing

Slide 81 puts the Constitutional AI chart next to Llama 2's reward-model accuracy across
data-collection batch stages. The distinction he draws (≈53:08): the dots on Anthropic's curves
"are new data sets from humans, rather than this kind of redo-a-reward-model, relabel-your-data
thing." Llama 2's paper makes the loop explicit — work with an annotator, collect a batch,
generate the next batch from the *previous checkpoint*, train again, repeat (≈53:54). That is
online, but powered by repeated human collection rather than by algorithmic re-labelling.

On **Llama 3** (slide 83), Lambert reads the blog post's sentence — post-training is "a
combination of supervised fine-tuning, rejection sampling, proximal policy optimization (PPO),
and direct preference optimization" — as evidence of practicality rather than of a single grand
method: "at each one of these points, they tried a few methods and chose the training method that
worked best" (≈54:40). His ordering of the toolkit is useful on its own:

- **Rejection sampling** is the simplest — rank SFT outputs with a reward model, then train on the
  winners with the ordinary autoregressive loss.
- **DPO** is much simpler than PPO "but it might not give you the highest end performance."
- **PPO** is what you reach for "once all of your data is collected and you're not on a weekly
  time crunch," when you can afford to tune its many knobs (≈55:27).

## Current directions

Slide 84 (≈56:14):

- **Data is the bottleneck.** "Even as academics with very limited compute, we literally try every
  data set that's available."
- **More DPO variants** — removing the reference model, altering the loss, moving from pairwise to
  single-sided preferences.
- **More model sizes**, and specifically **scaling down**: "it's a place where academia can still
  play," free of the marketing dynamic pushing companies bigger. The hard part is that small
  models "show more or less random scores on many benchmarks people care about," so "even just
  breaking through in that domain would be really impactful work" (≈57:00).
- **Evaluations** — "we need to keep getting more specific on the things we care about."
- **Personalization** — "how do we train models that are good for you, as an individual, rather
  than one big model for one big technology organization" (≈57:46).

## Questions from the floor

The Q&A runs about ten minutes and is substantive.

**Why is online DPO hard, if you have a good reward model?** (≈58:35) Lambert's answer is
**distribution matching**. "What your reward model can learn is specific to the prompts," and
there is a detail he finds strange: in PPO the prompts used for the policy are often exactly the
prompts the reward model was trained on. "We talk about generalization in machine learning, but
we're kind of softballing ourselves at the PPO stage." That breakdown shows up when training PPO
models with off-the-shelf reward models (≈59:21).

**Is there more than pairwise preference?** (≈1:00:56) Three lines: **KTO** uses one-sided
preference data — "a lot of customer apps have, like, 'did you get good support from this agent,
yes or no'"; **Starling** uses *k*-wise preferences with five or nine answers per prompt and a
different loss; and **fine-grained preferences** label each completion for conciseness,
helpfulness, honesty — NVIDIA's SteerLM and work from the University of Washington. "Literally the
whole field of social choice needs to get condensed into these things" (≈1:02:30).

**How do we exceed human performance by fine-tuning?** (≈1:03:23) Via **search** — "one of the
foundational ideas in CS is search, which is also motivated as exploration in RL, and therefore
we need some sort of language models that can search and generate new data," with humans supplying
what search cannot reach. He references the OpenAI rumour of the time as the best public argument
for this direction. (The transcript flags the restoration of that project name as
lower-confidence.)

**How do you create a new dataset?** (≈1:04:09) Community efforts work but burn people out — "most
people who do a community effort are like, 'I never want to do this again'." Alternatives:
returning prompts and labels to users as Chatbot Arena does (with noise concerns), demo-driven
collection at a nonprofit like AI2, and academic workshops or competitions "where you could have
communities meet every three, six, eight months" (≈1:04:56).

**Are reward models subject to reward hacking?** (≈1:06:31) Yes, and structurally so: "you have a
very powerful optimizer, and you have an incomplete representation of your reward, and it will
always find where your representation of reward is wrong." Mitigations exist but "saying it's
perfect is not possible, in the math" (≈1:07:17). The failures are at least visible — "you'll end
up with a model that just says 'JavaScript' to every answer, to infinity" — and he suggests
deliberately removing the constraint as a test: "you should be able to easily exploit it if you
turn the brakes off" (≈1:08:03).

## Related pages

- [RewardBench](rewardbench.md) — the benchmark, its categories, and what it revealed.
- [PPO vs DPO](ppo-vs-dpo.md) — the ablation, the online-data question, and how to choose.
- [Preference data](preference-data.md) — ShareGPT, OpenAssistant, UltraFeedback and the rest.
- [Direct Preference Optimization](direct-preference-optimization.md) — the method this lecture
  is "after".
- [RLHF](rlhf.md) — the objective and the pipeline.
- [Reward modeling](reward-modeling.md) — how reward models are trained and why it is strange.
- [LLM as a judge](llm-as-a-judge.md) — used here as a RewardBench baseline, and inside
  self-rewarding methods.
- [Lecture 11 — Post-training](11-post-training.md) — the course's own RLHF and DPO lecture.
- [Instruction fine-tuning](instruction-finetuning.md) — the foundation everything here builds on.
- [Evaluating LLMs](evaluating-llms.md) — and why local evaluation matters more than a leaderboard.
