# PPO vs DPO

Whether policy-gradient RLHF (PPO) beats [Direct Preference Optimization](direct-preference-optimization.md)
in practice, and what the difference actually consists of. The empirical answer in
[lecture 16](16-after-dpo.md) is: PPO is slightly better, by about **1.2% on average**, and it
costs far more to obtain.

## The two positions

DPO's appeal is engineering, not accuracy (≈12:26). Its reference implementation is "extremely
simple" — if you have used Transformers, "it's pretty easy to write a loss function that uses
DPO, rather than building an entire infrastructure stack to start with." PPO-style RLHF "normally
needs an almost entirely new infrastructure stack." So DPO "scales more easily on compute, it's
easier to debug, it's even easier to learn" (≈13:11).

Nathan Lambert's framing is that the online argument about which is *better* is mostly noise:
"in reality they're different loss functions and they're doing very different things, but you can
get similar results with both of them, which is why, if something is much easier to do, you should
just start with it" (≈13:57). And: "we don't need to say one versus the other — we can do both,
and they are different" (≈14:44). Slide 21 is a meme to this effect.

Note also that **DPO has not eliminated the reward model** — "DPO still has a reward model, which
is really important to the math actually checking out," except the language model itself plays the
role (≈13:11). See [RewardBench](rewardbench.md) for what follows from that.

## The ablation

AI2 tried to settle it empirically (slides 62–74). The work was unpublished at the time: "the
numbers aren't entirely final" (≈40:02). Method: start from an instruction-tuned Llama 2 13B,
change one design decision at a time, and watch a fixed suite — factuality, reasoning, coding,
chat, safety, truthfulness, instruction following.

| Step | Effect on the average |
| --- | --- |
| Instruction tuning (SFT) over the base model | **The largest single gain in the entire talk** |
| + DPO on Anthropic HH-RLHF | a small bump across all metrics |
| + DPO on UltraFeedback instead | a larger bump — from changing *only the data* |
| DPO → PPO | ~1% better |
| Reward model 13B → 70B | flat |
| More code/reasoning prompts in RLHF | those evals rise; the average does not |

**Instruction tuning dominates.** "Instruction tuning puts the model on the map as being useful,
and it's easy to see gains at the beginning, and then it's harder and harder for us to keep
improving these models" (≈40:02). Everything the rest of the lecture argues about is fighting over
a much smaller margin than [instruction fine-tuning](instruction-finetuning.md) already delivered.

**Data beats algorithm, at this scale of effect.** Swapping HH-RLHF for UltraFeedback — same
method, same model — gives a bigger jump than switching from DPO to PPO does (≈40:47). Typical
gains here are 0–2%, which "in the research sphere of trying to ship a model … is a big deal"
(≈41:34).

**Bigger reward models did not help downstream.** Industry holds that larger reward models are
better at nuance and stabilise training. Here it "does improve some things, but it doesn't
actually make the model overall much better — it's kind of flatlined" (≈42:19). The team checked
whether their own 70B reward model was simply bad, two ways (≈43:05):

- RewardBench: "not clearly correlated" between the 13B and 70B versions.
- **Best-of-$n$ sampling** — generate many completions, rank with the reward model, evaluate the
  top-ranked ones — *did* show the bigger reward model to be better.

So the reward model genuinely improved and the policy did not follow. That gap is the interesting
result, and it is unresolved in the lecture.

**Adding targeted prompts is a wash on the average.** More code and reasoning prompts made the
math and code evaluations better, "but adding the full noise into the fact that some other
evaluations might go down — this process is really hard to disentangle" (≈43:52).

### The headline table

Slide 74 compares DPO and PPO at 13B across five fixed preference datasets, all downsampled to
60,908 examples (ChatArena excepted). The caption: **"PPO outperforms DPO by an average of
1.2%."** Per dataset, PPO's advantage on the average column runs +0.1 (HH-RLHF) to +1.2
(UltraFeedback FG), and individual columns move in both directions — for example on Nectar, PPO
gains +6.6 on coding and loses 8.0 on truthfulness.

## The cost side

Lambert's verdict is unusually candid (≈46:14):

> this is a few grad students and people like me — this is not a big team in industry doing this,
> we're scraping by, and I don't know if it's worth the effort. I see why OpenAI uses this,
> because we're able to get a bit more signal out of it, but it's a ton of effort to get a bit
> better signal out.

Two reasons PPO is expensive:

1. **Unbounded tuning surface.** "There's always one more thing for us to ablate when you're
   training these models with PPO — things like different regularization, we're learning a value
   function in RL, different warmup, different sizes — there's just so many knobs to turn"
   (≈43:52).
2. **Generation during training.** "PPO generates new responses from the model as it trains, to
   refresh the data, and that is by far the biggest bottleneck when you're actually training these
   models — it's just way slower than DPO" (≈44:38).

Point 2 is also, as the next section argues, most of the reason PPO wins at all.

## What "online" means

If PPO's edge is small but real, where does it come from? Lambert separates two axes people
conflate under "online" (≈47:45):

**1. Freshly generated data from the policy.** UltraFeedback contains generations from Alpaca,
Vicuna, GPT-3.5, GPT-4 and Llama, so DPO on it "incorporat[es] information from a lot of different
models down into our one policy." PPO only generates from the *current* model, and that
distribution shifts as training proceeds. "That's a very different idea of where the signal is
coming from."

**2. Freshness of the labels.** A human comparison is a fixed data point. But "I can also later
take this reward model that I trained and generate a chosen and rejected and change the label"
(≈48:31).

"These two things — what the actual text is, and when the chosen/rejected label was given — are
what people mean when they're talking about whether something is special about online in RLHF."

This overlaps with, but is not the same as, the RL literature's **on-policy versus off-policy**
distinction, which "has a much more definitional flavor to them, while in this alignment space
we're more focused on whether we need to get fresh data in, and how we need to label our data"
(≈46:59).

Crucially, PPO has no monopoly on either axis — "it's clear to see that PPO does it very
differently than DPO, but we're not restricted to this" (≈48:31). That is the opening for online
DPO variants.

## Making DPO online

By April–May 2024 several independent papers had converged on "online is important," which
Lambert treats as evidence in itself: "it's a lot easier to be confident in things if three
independent institutions say something similar at the same time" (≈49:17).

Methods (slide 78):

- **Self-Rewarding Language Models** (Meta) — between DPO iterations, ask the model itself which
  answer is better, i.e. [LLM-as-a-judge](llm-as-a-judge.md) relabelling its own data, then run
  several DPO rounds (≈49:17).
- **Batched DPO** — do not consume all the data at once; update it between batches.
- **D2PO** (discriminator-guided DPO), which Lambert advised on (slides 79–80, ≈50:50). It
  compares three regimes: (a) standard DPO on a fixed dataset; (b) online preference optimisation,
  re-labelling data with a reward model; (c) additionally **retraining the reward model over
  time**, keeping policy and reward model in step.

D2PO's evaluation includes a synthetic closed-form task — the reward is *the number of nouns in a
sentence*, measurable with a classifier instead of a judge (≈52:23). Result: retraining the reward
model converges better than merely re-labelling. "Keeping your training process a little more
online can improve performance." A standard open-ended evaluation with a model judge shows the
same, "which has all sorts of problems, but we can show similar results."

### Why online DPO is not simply free

Asked in the Q&A why, given a good reward model, one cannot just do rollouts, rank them and
iterate, Lambert's answer is **distribution matching** (≈58:35). "What your reward model can learn
is specific to the prompts." And a detail he finds odd: in PPO, the prompts used for the policy
are often exactly those the reward model was trained on. "We talk about generalization in machine
learning, but we're kind of softballing ourselves at the PPO stage — we're only grading PPO
answers which our reward model is trained to answer." The breakdown shows up when training PPO
with off-the-shelf reward models (≈59:21).

## How industry appears to choose

Llama 2's paper describes repeated rounds: work with an annotator, collect a batch of human
preferences, generate the next batch from the previous checkpoint, retrain, repeat (≈53:54).
That is online — powered by fresh *human* collection rather than algorithmic relabelling.

Llama 3's blog post lists "supervised fine-tuning, rejection sampling, proximal policy
optimization (PPO), and direct preference optimization" together, which Lambert reads as
practicality rather than a grand method: "at each one of these points, they tried a few methods
and chose the training method that worked best" (≈54:40). His ordering (≈55:27):

- **Rejection sampling** — simplest: rank SFT outputs with a reward model, train on the winners
  with the ordinary autoregressive loss.
- **DPO** — much simpler than PPO, "but it might not give you the highest end performance."
- **PPO** — for when the data is collected and you are not on a weekly deadline, and can afford
  to tune it.

## Practical summary

Start with DPO. It is simpler, cheaper and close in quality. Reach for PPO when you have the
infrastructure, the time, and a reason to want the last one or two points — and expect the effort
to be disproportionate to the gain. Before either, check that your **data** is not the binding
constraint, because in this ablation it repeatedly was; see
[preference data](preference-data.md).

## Related pages

- [Lecture 16 — After DPO](16-after-dpo.md) — the source.
- [Direct Preference Optimization](direct-preference-optimization.md) — the method and its
  derivation.
- [RLHF](rlhf.md) — the objective both methods optimise.
- [Reward modeling](reward-modeling.md) — the component PPO depends on.
- [RewardBench](rewardbench.md) — evaluating that component, and its imperfect correlation with
  downstream quality.
- [Preference data](preference-data.md) — the datasets in the ablation.
- [Instruction fine-tuning](instruction-finetuning.md) — the step that dominates both.
