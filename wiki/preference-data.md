# Preference data

The datasets that post-training actually runs on: prompts paired with a **chosen** and a
**rejected** completion. [Lecture 16](16-after-dpo.md) argues that these — not the choice of
algorithm — are the binding constraint on open alignment research, and it is the lecture's most
repeated point: "we're very bottlenecked on data. Even as academics with very limited compute, we
literally try every data set that's available" (≈56:14).

## The scale gap

The opening comparison of the lecture, offered as context that is not on the slides (≈2:26):

| Source | Scale |
| --- | --- |
| Comparisons Meta purchased from one vendor for Llama 2 | ~1.5 million |
| Everything collected on Chatbot Arena, as of ≈May 2024 | ~800,000 |

The Llama 2 number is "years outdated," and "you can only imagine what OpenAI, Anthropic, etc. are
buying at this scale" (≈2:26). This is why the lecture's answer to industry is not to match its
data collection but to compete on evaluation, on method, and on scaling down.

## What the data looks like

A preference data point is a prompt plus two completions with a label saying which is better —
"if you're using ChatGPT a lot, you'll sometimes see it give you two answers and ask you which one
is better. This data is literally what's used to train a reward model" (≈26:59).

Both completions must be fed through together, because the reward-model loss operates on the
*difference* between the two scalar scores rather than on either alone (≈27:46). See
[reward modeling](reward-modeling.md).

Annotator agreement is around **70%** (≈28:32), which raises the question "is the noise part of
the signal, or is it a bug." Lambert's view is that for preferences it plausibly is signal — "not
everyone's preferences are the same, so not getting full agreement might mean this system is
working — we don't want ChatGPT to be fully narrow-minded all the time."

## The datasets that mattered

Slides 22–32 tell this as a history, and the striking part is how few datasets carry the field.

**ShareGPT** (2023). The first human data available to academic alignment work. It came from a
Chrome extension that added a share button to ChatGPT and logged the results — "a bit of a legal
gray area" — and Vicuna's use of it is what made Vicuna work. "Just having access to these human
prompts unlocked a lot of potential back in the day," and it remains a subset of training data
today (≈15:29).

**LMSYS data** and **WildChat** are the more permissive successors: LMSYS prompts "are collected
with consent," and WildChat, an AI2 project, "gave people free access to ChatGPT in exchange for
their data" (≈16:16).

**OpenAssistant** (April 2023) is the one Lambert wishes had successors. Run by "a few people in a
Discord community, working extremely long hours," it produced prompts, responses *and* preference
pairs with a level of "controls and voting and ranking" that ShareGPT and LMSYS lack. "We haven't
seen anything like it," and models are still trained on it — "these one or two influential data
sets from over a year ago are still what's used to train models" (≈17:03). His account of why is
blunt: creating human data is hard, and "most people who do a community effort are like, 'I never
want to do this again'" (≈1:04:09).

**UltraFeedback** is the workhorse of the DPO era: synthetically generated text labelled by GPT-4,
built by OpenBMB rather than by the teams that made it famous (≈20:53). It is half the reason
Zephyr β worked — the other half being a learning rate of 5e-7, "many orders of magnitude lower"
than the folk-wisdom 3e-4 (≈21:40).

**Anthropic HH-RLHF** is the default starting point despite being "known as being particularly
noisy among researchers in the area … it's been around for a few years, it's big, it's
multi-turn" — and it still improves models (≈40:47).

**Nectar** is the k-wise dataset behind Starling, with five or nine answers per prompt rather than
a pair (≈1:01:42).

## Synthetic versus human, and why it is not the main axis

UltraFeedback is GPT-4-labelled; OpenAssistant is human-made; both are in daily use. The axis the
lecture treats as more important is **freshness** — of the generations and of the labels.

UltraFeedback "has generations from all sorts of models — from Alpaca, Vicuna, GPT-3.5, GPT-4,
Llama," so training on it "incorporat[es] information from a lot of different models down into our
one policy," whereas PPO generates only from the current model (≈47:45). And labels can be
refreshed too: "I can also later take this reward model that I trained and generate a chosen and
rejected and change the label" (≈48:31). Those two axes — what the text is, and when the label was
assigned — are what "online" means in this literature. See [PPO vs DPO](ppo-vs-dpo.md).

Freshness is also a shelf-life problem. "UltraFeedback, at this point, is maybe six months old or
so — I don't know the exact age, but in terms of people training models, that feels old" (≈45:26).

## What the ablations showed about data

Slide 73 runs Tulu 2 + DPO across "almost all of the well-received data sets that are out there in
the open," and the conclusion is unflattering (≈45:26):

- Looking at the factuality column, "some of these things just don't matter at all when you're
  aligning these models."
- UltraFeedback's numbers still lead, which surprises him each time he looks.
- "We need to get new data sets that are really adding different capabilities to these models."

Meanwhile in the [PPO vs DPO](ppo-vs-dpo.md) build-up, swapping HH-RLHF for UltraFeedback — same
algorithm, same model — produced a larger gain than switching algorithm did (≈40:47). At the
margins this field currently operates in, the dataset is the more powerful lever.

## Beyond pairwise preferences

From the Q&A (≈1:00:56), three directions:

- **One-sided preferences.** KTO uses a single-sided signal — "a lot of customer apps have, like,
  'did you get good support from this agent, yes or no'." Different loss function, much more
  plentiful data.
- ***k*-wise preferences.** Starling collects five or nine answers per prompt and uses a
  learning-to-rank style loss (≈1:01:42).
- **Fine-grained preferences.** Label each completion for conciseness, helpfulness, honesty —
  NVIDIA's SteerLM and work from the University of Washington. "That one's probably the one that's
  most emerging, most in the academic sense, but there's so much to learn here — like, literally
  the whole field of social choice needs to get condensed into these things" (≈1:02:30).

## How new datasets get made

Lambert's own list of viable routes (≈1:04:09):

1. **Community efforts** like OpenAssistant — high impact, rarely repeated by the same people.
2. **Returning data to users**, as Chatbot Arena does with some prompts and labels, though he has
   "specific concerns … around being too noisy."
3. **Demo-driven collection** at an organisation with no competing product incentive: an AI2 demo
   "is going to be about science, generating information, rather than being a ChatGPT competitor."
4. **Workshops and competitions** — "you could have communities meet every three, six, eight
   months, and have work focused on an area" (≈1:04:56).

## Related pages

- [Lecture 16 — After DPO](16-after-dpo.md) — the source lecture.
- [Reward modeling](reward-modeling.md) — what this data trains.
- [PPO vs DPO](ppo-vs-dpo.md) — the ablations, and the online-data question.
- [RewardBench](rewardbench.md) — evaluation built from manually constructed preference pairs.
- [RLHF](rlhf.md) — the pipeline this data feeds.
- [Direct Preference Optimization](direct-preference-optimization.md) — which consumes the same
  pairs without a separate reward model.
- [Instruction fine-tuning](instruction-finetuning.md) — the earlier stage, and its own data.
