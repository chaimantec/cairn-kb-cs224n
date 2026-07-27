# RewardBench

An evaluation benchmark for **reward models** — the scoring models that sit inside
[RLHF](rlhf.md) — built by Nathan Lambert at AI2 and presented in
[lecture 16](16-after-dpo.md), slides 38–61. Released March 2024.

Its premise is that reward models were the least-examined component of the alignment pipeline
relative to how much importance industry placed on them: "I hear all the time that reward models
are crucial to RLHF, but how do we know exactly what aspects of the final policy they're
improving" (≈29:17).

## Why it exists

Two motivations, and the second is the practical one.

**Transparency.** Industry says reward models are what you must get right, so the obvious
follow-up is "what does it mean for a reward model to be good" (≈26:14). Before RewardBench there
was no shared way to answer that. Lambert notes the diagnostic gap directly: the available
evaluation tools all sit *outside* the RLHF feedback loop — policy generates, reward model
scores — and "none of these evaluation tools are giving us internal insight into what's happening
in this feedback loop" (≈26:14).

**Local evaluation.** When you are training models you need a number you can compute at your desk
that tells you whether a change helped. "You can't wait until Chatbot Arena evaluates your model,
because that takes you about a month to get your numbers back" (≈25:29). This is the same
argument that makes any fast proxy metric valuable, and it is why the benchmark is built to be
cheap: it is pure inference, no training.

## How it works

Deliberately simple (≈29:17):

1. Collect a set of prompts.
2. For each, *manually* construct a **chosen** and a **rejected** answer.
3. Run the reward model on both.
4. Score a win if it ranks the chosen answer higher. Report accuracy.

"It's really direct — we're just doing inference on existing models, and we're going to see
whether or not they agree with human data" (≈29:17). Because the comparison is pairwise, the same
harness can score three quite different kinds of model: a classifier-style reward model, a
[DPO](direct-preference-optimization.md) model used as an implicit reward model, and an
[LLM-as-a-judge](llm-as-a-judge.md) simply asked which answer is better.

## The categories

Slide 44's Table 1. The dataset is assembled from existing evaluation resources — AlpacaEval,
MT-Bench, XSTest, Princeton's LLMBar, and prior sets from Anthropic and OpenAI (≈30:03):

| Category | N | What it contains |
| --- | --- | --- |
| **Chat** | 358 | AlpacaEval and MT-Bench comparisons with clear quality gaps |
| **Chat Hard** | 456 | MT-Bench 7–8s vs 5–6s, plus five LLMBar subsets including adversarial ones |
| **Safety** | 740 | Refusals (dangerous, offensive), XSTest should-refuse and should-respond, Do Not Answer |
| **Reasoning** | 1431 | PRM Math (human vs buggy LLM answers) and HumanEvalPack correct-vs-buggy code in six languages |
| **Prior Sets** | 17.2k | Anthropic Helpful, Anthropic HHH, SHP, Summarize |

The **Reasoning** construction is worth noticing: correct code versus buggy code in C++, Go,
JavaScript, Java, Python and Rust gives a preference pair with an unambiguous ground truth, which
is rare in preference data.

## What it found

**The leaderboard saturated fast.** Between the March 2024 launch and May 2024, "the model that
was fifth on the leaderboard is now 31st" (≈31:36) — the expected consequence of giving a research
community somewhere to compare.

**LLM-as-a-judge is not state of the art.** Asking GPT-4 or GPT-4o which answer is better was
added as a baseline, and it loses: both "are not actually as good in this closed domain as a
reward model that Cohere is training" (≈32:22). A useful corrective, since LLM-as-a-judge is how
AlpacaEval and MT-Bench themselves are built.

**DPO models slid down the rankings.** Early on, DPO models used as implicit reward models scored
well — partly because few dedicated reward models existed. As more people trained actual reward
models, the DPO entries fell (≈32:22).

**Chat Hard is the only category that has not saturated**, which is what gives the benchmark
longevity (≈33:08).

### The Chat Hard example

Slide 53 (≈33:08). The prompt: *give an example of a metaphor that uses the following object,
stars*. Both candidates are fluent metaphors. The chosen one is about stars — "the twinkling
diamonds in the sky" — and the rejected one is about the **moon**, which is also in the night sky
and strongly associated with stars (≈33:56).

LLMBar generates these by rephrasing the prompt and sampling from the rephrased version, so the
rejected answer is well written and simply off-topic. That is hard for a reward model "because
they have this association between the stars and the moon, but we want our language models to be
able to answer questions like this" (≈34:41). Chat Hard also has "the best correlation as
something that is hard" with what the benchmark is trying to measure.

### Safety patterns

Slide 54 splits refusal behaviour using XSTest's two halves — prompts that *should* be refused and
prompts that merely contain trigger words and should be answered. Three patterns fall out
(≈35:26):

1. **Handles safety well** — refuses requests for harmful advice, responds to borderline ones.
2. **Refuses everything** — scores well on should-refuse, tanks should-respond. Lambert saw "a lot
   of tech companies release models like this, which … doesn't feel right when you talk to them."
3. **Responds to everything** — "it's not the language model's job to gate, that's the philosophy
   there."

The methodological value is that these are read off the reward model *directly*, "without asking
them to generate text" (≈36:12).

## Using a DPO model as a reward model

RewardBench can score DPO models because DPO's maths defines an implicit reward — the paper's
equation 3, a log-ratio against the reference policy (slide 55, ≈36:12):

$$r(x,y) = \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

so ranking two completions reduces to comparing their ratios:

$$\log \frac{\pi(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} > \log \frac{\pi(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)}$$

These scores look nothing like a classifier's. Pass text through a DPO model and "the reward will
be something like minus 200," since it sums log-probabilities, all negative (≈36:57).

**The catch is $\pi_{\text{ref}}$.** The reference model is an intermediate training checkpoint,
and "when people release a DPO model, they normally release just the model" (≈37:42). Can you drop
the reference term? Slide 56 tries exactly that, crossing out every $\pi_{\text{ref}}$, and the
answer is no: "all the scores on our benchmark plummet across all the DPO models that we have."
That is the expected result, since the reference model is a regulariser sitting inside the reward
definition itself.

The conclusion is about **release practice**, not about DPO: "DPO is training a reward model, but
if we don't always have access to it, we just can't learn from it, we can't use that in another
system as clearly" (≈38:29).

## Limits, stated by its author

- **It measures a component, not an outcome.** "Everything I've told you about is part of this
  RLHF pipeline, but I haven't told you how it's impacting the final model that you use at the end
  of the day, which is a very rightful criticism" (≈39:16). The
  [PPO vs DPO](ppo-vs-dpo.md) ablation is partly an attempt to close that loop — and there,
  RewardBench scores were "not clearly correlated" with whether a 13B or 70B reward model produced
  a better policy (≈42:19).
- **Most categories saturate.** Only Chat Hard resisted.
- **RewardBench 2** "is going to need to mostly make everything harder, and make everything more
  human" (≈39:16).

## Related pages

- [Lecture 16 — After DPO](16-after-dpo.md) — the source lecture.
- [Reward modeling](reward-modeling.md) — how the models being evaluated are trained.
- [PPO vs DPO](ppo-vs-dpo.md) — where these reward models are put to work.
- [LLM as a judge](llm-as-a-judge.md) — a baseline here, and not a winning one.
- [Direct Preference Optimization](direct-preference-optimization.md) — and its implicit reward.
- [Evaluating LLMs](evaluating-llms.md) — benchmark design in general.
- [Benchmark contamination](benchmark-contamination.md) — saturation's companion problem.
