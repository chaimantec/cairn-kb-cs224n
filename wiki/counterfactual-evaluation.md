# Counterfactual evaluation

A way of asking whether a model is reasoning or remembering. Take a task the model performs well,
change it in a way that leaves the *reasoning* required identical but makes the surface form
unlikely to appear in pretraining data, and measure the drop. If the model was reasoning,
accuracy should hold; if it had memorised, accuracy should fall.

The method is introduced in [lecture 15](15-reasoning-and-agents.md) (slides 33–37, ≈25:01) as
the corrective to benchmark-driven claims about reasoning. Murty's framing: benchmark numbers
alone cannot settle the question, so you must "be more systematic, come up with counterfactual
tasks, and be very careful about possible data contamination" (≈20:21).

## Why benchmark accuracy is not enough

A model answers `12 + 14` correctly. Two explanations fit: it has learned addition, or that exact
example was in its training data (≈25:01). Ordinary accuracy cannot separate them, because the
test set is drawn from the same distribution as the training data. This is the same worry that
motivates [benchmark contamination](benchmark-contamination.md) — and in the same lecture, the
GPT-4 column of a BigBench-hard results table is waved off for exactly that reason (≈14:53).

Counterfactual evaluation attacks it from the other side: instead of trying to prove the test
data is clean, deliberately construct a variant the model cannot have seen much of.

## What "counterfactual" means here

Not counterfactual in the causal-inference sense. The criterion is **distributional**, and a
student in the lecture pushes on precisely this point, asking why base 9 is the counterfactual
rather than base 10 (≈26:33). The answer:

> base-10 addition is frequently observed in the training data, but very few people do base-9
> addition, and so there's going to be many fewer examples of this in the training data (≈26:33)

The student then offers the cleaner term — "so it's more so out-of-distribution, right?" — and
Murty accepts it: "yeah, you can also call it out-of-distribution, for sure." That is the honest
description of what the method manipulates.

## The three constructions in the lecture

**Change the number base.** A model that does base-10 arithmetic is asked to do base-9 arithmetic.
Same algorithm, far rarer in text. If accuracy survives, "maybe this model has understood how to
do addition" (≈25:46).

**Change the world's facts.** For logic, the concern is that the model has seen very similar
syllogisms. So build a world in which — Murty's example — corgis are reptiles, and see whether it
can still run the inference (≈25:46). The logical form is untouched; only the premises are
unfamiliar.

**Change the task or the symbol inventory.** The analogical-reasoning study (slides 35–37,
≈27:18) is the most carefully controlled of the three. The base task is string transformation
shown by example: `ABCD → ABCDE`, so `IJKL → IJKLM` (≈28:04). Two variants:

- *Task counterfactual*: the output must be `ABCDF` — not the next character, but the one after
  it (≈28:04).
- *Alphabet counterfactual*: replace the standard alphabet with one starting at X, Y, and so on
  (≈28:51).

## What it found

There is "a significant drop" on the arithmetic and logic counterfactuals — and notably it
appears "even for very simple logic problems that don't involve multiple steps of reasoning,"
which suggests "there's not that much reasoning, there's more memorization" (≈27:18). The
single-step result matters: a drop on long chains could be blamed on error accumulation, but a
drop on one-step problems cannot.

The analogical-reasoning study adds the control that makes the argument stick: **the same
experiments were run on human subjects**, who show "very little" decrease in performance (≈29:37).
Humans are the existence proof that the counterfactual tasks are solvable by reasoning alone, so
the model's drop is about the model, not the task's difficulty.

Murty's conclusion is deliberately unheroic — "maybe there's some reasoning, maybe there's some
memorization, but there's nothing systematic" — with the hedge that this is a fast-moving area and
"maybe someone will find that if you change your prompt a little bit, now models can do
reasoning" (≈29:37).

## Relationship to rationale faithfulness

Counterfactual evaluation asks whether the *answer* is earned. A separate family of experiments in
the same lecture asks whether the *stated reasoning* is what produced the answer — by truncating
rationales early, or corrupting a step and seeing whether the answer moves (≈21:53, ≈23:27).
Both are on [chain-of-thought](chain-of-thought.md). The two are complementary: a model could
produce faithful rationales for memorised answers, or unfaithful rationales for correct ones.

## Using it

The transferable recipe:

1. Identify what the model is claimed to have learned (an algorithm, a rule, a relation).
2. Find a variant of the task that requires the same competence but has a rare surface form.
3. Check that the variant is genuinely solvable — ideally with a human control group.
4. Compare accuracy on original versus variant. Interpret the *gap*, not either number alone.

Step 3 is the one most often skipped and the one that carries the argument.

## Related pages

- [Lecture 15 — Reasoning and agents](15-reasoning-and-agents.md) — where this is presented.
- [Chain-of-thought](chain-of-thought.md) — the prompting family being interrogated, and the
  faithfulness experiments.
- [Benchmark contamination](benchmark-contamination.md) — the problem counterfactuals route
  around.
- [Evaluating LLMs](evaluating-llms.md) — benchmark design generally.
- [Self-training and rationale distillation](self-training-and-rationale-distillation.md) — the
  methods whose gains this evaluation puts in question.
