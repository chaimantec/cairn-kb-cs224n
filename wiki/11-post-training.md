# Lecture 11 — Post-training: prompting, instruction finetuning, DPO and RLHF

The lecture that closes the gap between a pretrained language model and an assistant.
[Lecture 9](09-pretraining.md) ended with a model that can complete *Stanford University is
located in \_\_\_\_*; this one asks how you get from there to something you can give an
instruction to. Archit Sharma works through three answers in increasing order of cost and
capability — **prompt the model** into doing the task, **finetune it** on instruction-output
pairs, or **optimize it directly against human preferences** — scoring each one's costs and
benefits as he goes, and ends at the pipeline that actually produced InstructGPT and ChatGPT.

**Slide-by-slide text of this deck: [99 slides](../raw/slides/11-post-training.md)** — the
deck's printed numbers run 1–99 over 94 PDF pages; five slides were hidden in the source and
are absent, and the slide file gives the exact mapping. Citations below use the **printed**
number.

Slides PDF: [Lecture 10 — prompting, instruction finetuning, and DPO/RLHF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture10-prompting-rlhf.pdf) ·
[Full transcript](../raw/transcripts/11-post-training.md)

> **A note on numbering.** This lecture sits at **position 11** in the course playlist that
> this knowledge base follows, but both the video title and the deck call it "Lecture 10".
> Repo files use the position; the deck's own slide numbers are used for citations. The
> lecturer is **Archit Sharma**, working from slides by Jesse Mu.

## The problem: language modelling is not assisting

The framing is a single before-and-after pair. Slide 36 gives GPT-3 the prompt *"Explain the
moon landing to a 6 year old in a few sentences"* and shows what it actually returns:

```
Explain the theory of gravity to a 6 year old.
Explain the theory of relativity to a 6 year old in a few sentences.
Explain the big bang theory to a 6 year old.
```

It has continued the text — which is exactly what it was trained to do — rather than obeyed
it. Slide 37 shows the completion a human would write. The lecture's term for the gap is
that the model is **not aligned with user intent** (Ouyang et al., 2022), and the whole
lecture is about closing it.

What makes the gap worth closing is that the capability is already in there. Slides 2–8
argue that next-token prediction at scale teaches far more than syntax and facts: it teaches
rudimentary models of *agents, beliefs and actions*. The example on slide 5 is the sharpest.
Given "Pat, who is a physicist, predicts that…", the model continues that the bowling ball
and the leaf fall at the same rate; change the clause to "Pat, who has never seen this
demonstration before", and it predicts the bowling ball lands first. "You have to have some
notion of understanding of how humans work to even be able to predict this" (≈4:40).

## Route 1 — prompting

The first route needs no gradient steps at all. Slides 13–24 retrace how zero-shot and then
few-shot behaviour emerged from simply scaling GPT, and slides 25–33 show how far creative
prompting can push a fixed model. Both are treated at length on their own pages:

- [Prompting and in-context learning](prompting.md) — zero-shot, few-shot, why in-context
  learning is *emergent*, and the "dark art" of prompt engineering.
- [Chain-of-thought prompting](chain-of-thought.md) — few-shot CoT, zero-shot "Let's think
  step by step", and the trigger-phrase search.

The lecture's own summary of the route (slide 34):

| | Prompting / in-context learning |
| --- | --- |
| **+** | No finetuning needed; prompt engineering (e.g. CoT) can improve performance |
| **–** | Limits to what you can fit in context |
| **–** | Complex tasks will probably need gradient steps |

There is also a lesson about how to work with these models that generalizes past the
technique. "When you interact with these models you might not get the exact desired behaviour
up front, but often these models are *capable* of the behaviour you want, and you have to
think about how to induce it — what is the pretraining data, what is the data on the internet
it might have seen which induces a similar behaviour to the kind I want" (≈20:48).

## Route 2 — instruction finetuning

The second route takes gradient steps, but on many tasks at once rather than one. Slides
38–40 present it as scaling up the familiar pretrain-then-finetune recipe from
[lecture 9](09-pretraining.md): the same decoder, the same supervised objective, but "Step 2:
finetune on **many tasks**", with the "not" in "not many labels" struck through. Full
treatment at [instruction finetuning](instruction-finetuning.md); the multitask benchmarks the
lecture pauses on (MMLU, BIG-Bench) are at [evaluating LLMs](evaluating-llms.md).

Slide 53 is the important one, because it is what motivates everything after it. Beyond the
obvious cost of collecting ground-truth answers for thousands of tasks, instruction
finetuning has three subtler problems:

1. **Open-ended tasks have no right answer.** *"Write me a story about a dog and her pet
   grasshopper"* has no gold completion to imitate.
2. **Language modelling penalizes all token-level mistakes equally.** The slide's example is
   the continuation of *Avatar is a fantasy TV show*: substituting *adventure* is nearly
   harmless, substituting *musical* is a real error, and cross-entropy charges the same price
   for both.
3. **Humans generate suboptimal answers.** As models improve, the ceiling becomes the quality
   of what an annotator is willing to write. "Do we really want to keep relying on humans to
   write down the answers?" (≈36:19).

Underneath all three is one mismatch: the objective is still next-token prediction on a
curated corpus, while the *goal* is to produce output a human would prefer.

| | Instruction finetuning |
| --- | --- |
| **+** | Simple and straightforward; generalizes to unseen tasks |
| **–** | Collecting demonstrations for so many tasks is expensive |
| **–** | Mismatch between the LM objective and human preferences |

## Route 3 — optimizing for human preferences

The third route optimizes the thing you actually want. Slide 56 sets it up on summarization:
for an instruction $x$ and a sample $y$, suppose you could obtain a human reward
$R(x, y) \in \mathbb{R}$. Then maximize

$$\mathbb{E}_{\hat{y} \sim p_\theta(y \mid x)}\left[R(x, \hat{y})\right]$$

The lecture flags what is new here, and it is worth pausing on. In pretraining and in
supervised finetuning the data comes from somewhere else and you maximize its log likelihood.
Here **you sample from the model you are training**, and the objective you are maximizing may
not be differentiable at all (≈40:57).

Two problems stand between that objective and an algorithm, and they structure the rest of
the lecture:

- **Where does $R$ come from?** A human in the loop cannot score millions of completions, and
  human numerical scores are uncalibrated anyway. The answer is to learn a
  [reward model](reward-modeling.md) from *pairwise comparisons* via Bradley–Terry.
- **How do you optimize it?** Either with reinforcement learning under a KL penalty —
  [RLHF](rlhf.md) — or, by rewriting the reward model in terms of the language model itself,
  with a plain classification loss —
  [direct preference optimization](direct-preference-optimization.md).

Slide 57 gives the three-step pipeline in one picture: instruction-tune a supervised policy;
collect comparison data and fit a reward model; optimize the policy against that reward model
with PPO. "First step: instruction tuning! Second and third steps: maximize reward (but
how??)."

### Why it works, and what it costs

Slide 68 (Stiennon et al., 2020) is the result that made the field pay attention: on
summarization, RLHF'd models are preferred to the *human reference summaries* more than half
the time — around 0.61 at 1.3B parameters, rising to ~0.70 at 6.7B — while supervised
finetuning stays near 0.40 and the pretrained model near 0.22–0.36. Even small models beat
the human references once trained this way.

The cost is complexity. Slide 69 reproduces the full PPO-for-RLHF systems diagram from
*Secrets of RLHF* and the lecturer is explicit that "this image is not for you to understand,
it's just completely to intimidate you" (≈55:39): you have to fit a value function, online
sampling is slow, and performance is hyperparameter-sensitive. That difficulty is what
restricted RLHF to high-resource labs, and what DPO removes.

### What it changes in the model

Slide 83 makes the behavioural difference concrete. Asked for the five most common causes of
stress, an instruction-tuned Alpaca answers in one line — "work, money, relationships,
health, and family". The same model after PPO returns a numbered list with a sentence of
explanation per item. The change preference optimization buys is, visibly, **detail and
formatting** — the things annotators reward.

| | Optimizing for human preferences (RLHF/DPO) |
| --- | --- |
| **+** | Directly models preferences (cf. language modelling); generalizes beyond labelled data |
| **–** | RL is very tricky to get right |
| **–** | Human preferences are fallible; *models* of human preferences even more so |

## Limitations of reward modelling

Slides 86–92 are the lecture's most careful section, and the argument escalates.

**Reward hacking.** The canonical picture (slide 86) is OpenAI's CoastRunners agent spinning
in a lagoon collecting score pickups instead of finishing the race. "Reinforcement learning is
a very strong optimization algorithm — it's at the heart of AlphaGo and AlphaZero — so you have
to be careful about how you specify things" (≈1:17:08).

**Humans reward the wrong things.** Chatbots are rewarded for answers that *seem* authoritative
and helpful "regardless of truth" (slide 87), which is a route to hallucination — illustrated
with the Bard and Bing launch errors. Put plainly: annotators "prefer authoritativeness more
than correctness" (≈1:17:54).

**Models of preferences are worse still.** Slide 88 is the over-optimization curve from
Stiennon et al.: as you push further from the supervised baseline in KL, the *reward model's
prediction* rises monotonically toward 1.0 while *actual* human preference peaks around KL 10
and then collapses. This is the empirical justification for the KL penalty in the
[RLHF](rlhf.md) objective, and the reason the lecture offers a general machine-learning rule:
"if you're ever optimizing some learned metric, be very careful" (≈50:14).

**Where the labels come from.** Slides 91–92 close on the human side: RLHF labels are often
collected from overseas, low-wage annotators, and InstructGPT's own labeler-demographic table
is 52.6% Southeast Asian, 22% Filipino and 22% Bangladeshi, with 89% holding an undergraduate
degree or higher. Beside it, the OpinionQA result on whose opinions models reflect. The
lecture's point is not that any single number is wrong but that these preferences are what get
baked in — a thread [lecture 12](12-benchmarking.md) picks up when it asks the same question
of LLM *evaluators*.

A student asks whether ChatGPT's mass adoption will change the rewards, since it now returns
five paragraphs where one would do. The answer (≈1:18:40) names the mechanism: "when you
collect preference data at scale, people are not necessarily reading the answers — the Turkers
might just simply choose the longer answer, and that's a property that actually goes into these
models." Length bias reappears as a measurement problem in
[LLM-as-a-judge](llm-as-a-judge.md).

## What's next

Slides 96–99 list the directions, all aimed at RLHF's remaining data cost:

- **RL from AI feedback** and *constitutional AI* (Bai et al., 2022) — the model critiques and
  revises its own harmful response against a written principle, and those revisions become the
  training signal. The lecturer notes this is already in use for evaluation as well as
  training (≈47:06), which is exactly what [LLM-as-a-judge](llm-as-a-judge.md) formalizes.
- **Finetuning LMs on their own outputs** (Huang et al., 2022; STaR, Zelikman et al., 2022),
  especially for code and reasoning.
- **Personalizing language models** — with the PRISM Alignment Project (Kirk et al., 2024) as
  the example: 1,500 people, 8,011 conversations, 21 LLMs, 68,371 responses scored by diverse
  humans.

And a caveat the lecture states twice: "there are still many limitations of large LMs (size,
hallucination) that may not be solvable with RLHF."

## The scorecard

The lecture's own summary, assembled across slides 34, 54 and 93:

| Approach | For | Against |
| --- | --- | --- |
| [Zero-/few-shot in-context learning](prompting.md) | No finetuning needed; prompt engineering can improve performance | Limited by context length; complex tasks need gradient steps |
| [Instruction finetuning](instruction-finetuning.md) | Simple; generalizes to unseen tasks | Demonstrations are expensive; objective ≠ human preference |
| [RLHF](rlhf.md) / [DPO](direct-preference-optimization.md) | Models preferences directly; generalizes beyond labelled data | RL is tricky; human preferences — and models of them — are fallible |
