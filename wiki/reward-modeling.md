# Reward modeling

Turning "which answer would a human prefer?" into a differentiable scalar function you can
optimize. Covered in [lecture 11](11-post-training.md), slides 56–63, and the prerequisite for
both [RLHF](rlhf.md) and [DPO](direct-preference-optimization.md).
[Lecture 16](16-after-dpo.md) returns to it at length, since it is the component
[RewardBench](rewardbench.md) was built to evaluate.

## The target

Slide 56 sets up the ideal object. For an instruction $x$ and a language model sample $y$, suppose
a human could give you a reward $R(x, y) \in \mathbb{R}$, higher being better. On the lecture's
running summarization example — a CNN article about a magnitude 4.2 earthquake in San Francisco:

| Summary | $R(x, y)$ |
| --- | --- |
| $y_1$: "An earthquake hit San Francisco. There was minor property damage, but no injuries." | 8.0 |
| $y_2$: "The Bay Area has good weather but is prone to earthquakes and wildfires." | 1.2 |

Then the objective is simply

$$\mathbb{E}_{\hat{y} \sim p_\theta(y \mid x)}\left[R(x, \hat{y})\right]$$

Two problems stand in the way, and the lecture takes them in order.

## Problem 1: humans in the loop are expensive

Slide 61. You cannot ask a person to score millions of completions. "I don't want to sit around
and label millions of examples. So this is very easy — we're in a machine learning class, so what
are we going to do?" (≈42:32). Train a model $RM_\phi(x, y)$ to predict the human reward from an
annotated dataset, and optimize against $RM_\phi$ instead (Knox and Stone, 2009).

Asked what architecture the reward model should be, the answer (≈43:18) is that it needs to
understand text well, so it is typically **initialized from the pretrained language model
itself**, with a scalar head. And $x$ and $y$ are not separated in the input — "it only sees $x$
and $y$ as an input… it's just going to predict a score at the end"; the notation distinguishes
them only for our benefit.

## Problem 2: human numbers are noisy and uncalibrated

Slide 62 is the objection to the setup on slide 56. Shown a third summary — "A 4.2 magnitude
earthquake hit San Francisco, resulting in massive damage." — what is $R(x, y_3)$? 4.1? 6.6? 3.2?

> "If you ask me on different days I'll give a different answer, first of all. But across humans
> itself this number is not calibrated in any meaningful way." (≈44:51)

You can mitigate this with rubrics and annotator training, but it stays a judgment call, and
"if your labels can vary a lot, it's just hard to predict".

**The fix is to change the question.** Instead of asking for a rating, ask for a **pairwise
comparison** — which of these two is better? (Phelps et al., 2015; Clark et al., 2018). This is
much easier for a person to answer consistently: "it's much easier to compare something and know
which is better than to ascribe it an arbitrary number on a scale, and that's why the signal from
something like this is a lot better" (≈46:24).

Slide 63 orders the three summaries $y_1 > y_3 > y_2$ — a ranking, not scores.

## Bradley–Terry: from comparisons back to scores

The remaining gap is that a preference dataset gives orderings while the objective needs a scalar.
The bridge is the **Bradley–Terry paired comparison model** (1952), which comes from the
literature in economics and psychology on how people make choices. It posits that the probability
a human picks $y_1$ over $y_2$ depends only on the *difference* between their internal rewards,
passed through a sigmoid:

$$P(y_1 \succ y_2 \mid x) = \sigma\!\left(R(x, y_1) - R(x, y_2)\right)$$

which is exactly binary logistic regression with the reward difference as the logit. Fitting
$RM_\phi$ by maximum likelihood over a dataset $D$ of triples (instruction, winning completion
$y^w$, losing completion $y^l$) gives the loss on slide 63:

$$J_{RM}(\phi) = -\mathbb{E}_{(x,\,y^w,\,y^l) \sim D}\left[\log \sigma\!\left(RM_\phi(x, y^w) - RM_\phi(x, y^l)\right)\right]$$

Read left to right: score both completions, subtract, squash to a probability, take the log, and
average with a minus sign. Minimizing it means $y^w$ should score higher than $y^l$. The lecture
walks the types explicitly for a student (≈47:57): the subtraction gives a real number, the
sigmoid makes it a probability, the log makes it a log-likelihood, and the expectation gives one
scalar summarizing how well the reward model matches human choices.

Where the comparison data comes from, per ≈1:07:55: take a set of instructions, sample several
answers **from the model**, and have humans label which they prefer. "They're model-generated
typically — they can be human-generated as well… All you need is a label saying which is a better
answer."

## The shift-invariance this buys, and costs

Because only differences enter, the reward scale is only identified up to an additive constant.
A student pushes on this in the [DPO](direct-preference-optimization.md) discussion (≈1:10:11):
surely +1 versus +99 should mean something different from +1 versus −1? The answer names the
assumption squarely:

> "What we're assuming, our choice model here, is that if a human prefers something over the
> other, the probability is governed **only** by the difference between the rewards. So that's an
> assumption that every RLHF also makes, and DPO also makes. Now, is that assumption true? Not
> completely true, but it holds to a fairly large degree."

That unidentified constant is not merely tolerable — it is what lets DPO cancel the intractable
partition function.

## Why a learned reward is dangerous to optimize

A reward model is fit on a distribution of completions and will have errors off it. Optimizing
hard against it therefore invites **reward hacking**: the policy finds inputs where the model
erroneously assigns a high score. The lecture's general rule (≈50:14):

> "If you're ever doing something and you're optimizing some learned metric, I'd be very careful,
> because typically our loss functions are very clearly defined, but here my reward model is
> learned. When it's learned it means it will have errors."

Empirically this is slide 88's over-optimization curve, where the reward model's predicted win
rate climbs to 1.0 while true human preference peaks and then collapses. The mitigation — a KL
penalty tying the policy to its initialization, so that it stays in the region where the reward
model was actually trained — is developed in [RLHF](rlhf.md).

## Who writes the labels

Slides 91–92 close the loop: RLHF comparison labels are frequently collected from overseas,
low-wage annotators, and InstructGPT's published labeler demographics (52.6% Southeast Asian, 22%
Filipino, 22% Bangladeshi, 89% with an undergraduate degree or higher) are not a neutral sample.
Whatever this pool prefers is what $RM_\phi$ learns to score highly.
[Lecture 12](12-benchmarking.md) finds the same fingerprint on the *evaluation* side — see
[LLM-as-a-judge](llm-as-a-judge.md) and the OpinionQA discussion in
[evaluating LLMs](evaluating-llms.md).

## How these models are actually trained

[Lecture 16](16-after-dpo.md) describes the practice, and it is stranger than the derivation
suggests (≈26:59):

- **Both completions go through together.** The loss acts on the *difference* between the two
  scalars — the model "is trying to separate the distance between them" and gradient descent
  widens that gap — so you cannot supervise on a single example. It "is really built on this idea
  of separating two things and creating a margin in the preferences to learn the decision
  boundary" (≈27:46).
- **The architecture is a language model with the last layers modified** to emit a scalar instead
  of text (≈26:59).
- **One epoch.** These models are trained for a single pass (≈28:32).
- **The accuracy numbers look bad**, next to ordinary train/test machine learning, and the tweaks
  people add — ensembles, margin losses — are not transformative.
- **~70% agreement with annotators** is the ceiling, which raises "is the noise part of the
  signal, or is it a bug." For preferences it may be signal: "not everyone's preferences are the
  same … we don't want ChatGPT to be fully narrow-minded all the time" (≈28:32).

## Does a better reward model give a better policy?

Not reliably, on the evidence in that lecture. Scaling the reward model from 13B to 70B "does
improve some things, but it doesn't actually make the model overall much better — it's kind of
flatlined" (≈42:19). Two checks confirmed the larger reward model was not simply broken:
[RewardBench](rewardbench.md) scores were "not clearly correlated" between the two sizes, while
**best-of-$n$ sampling** did show the bigger one to be better (≈43:05).

So the component improved and the downstream policy did not — an open problem, and the strongest
caveat on evaluating reward models in isolation. See [PPO vs DPO](ppo-vs-dpo.md).
