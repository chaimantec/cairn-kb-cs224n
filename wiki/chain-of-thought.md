# Chain-of-thought prompting

Getting a model to write out its reasoning before its answer, and thereby to reason better.
Introduced in [lecture 11](11-post-training.md), slides 25–31, and taken further — then taken
apart — in [lecture 15](15-reasoning-and-agents.md), slides 10–19 and 31–37.

## The problem it solves

Slide 25: some tasks are too hard for a large model to learn from examples alone, "especially
tasks involving **richer, multi-step reasoning**". The example is multi-digit addition shown
few-shot:

```
19583 + 29534 = 49117
98394 + 49384 = 147778
29382 + 12347 = 41729
93847 + 39299 = ?
```

The slide's answer is not a bigger model or a finetune. It is: **change the prompt.**

## Few-shot chain-of-thought

The change (Wei et al., 2022; slide 26) is to make the *exemplar's answer* show its working.
Standard prompting gives the model an exemplar of the form "Q: … A: The answer is 11." and it
answers the new question wrong. Chain-of-thought prompting gives it

> A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11.
> The answer is 11.

and the model's own output now follows the same shape — "The cafeteria had 23 apples originally.
They used 20 to make lunch. So they had 23 - 20 = 3. They bought 6 more apples, so they have
3 + 6 = 9. The answer is 9." — and is correct.

The mechanism, in the lecture's phrasing: "instead of just showing what the task is, you show
them examples where they reason *through* the task, so they're not just learning to do the task
but also learning how the reasoning is working" (≈17:45).

## It is emergent too

Slide 27 plots GSM8K solve rate against model scale for three families. At small scales
chain-of-thought is *worse* than standard prompting; the curves cross and then diverge sharply
at the top end:

| Family | Standard prompting (largest) | Chain-of-thought (largest) |
| --- | --- | --- |
| LaMDA (137B) | ~7% | ~15% |
| GPT (175B) | ~16% | ~47% |
| PaLM (540B) | ~18% | ~57% |

The dashed line at ~55% is the prior *supervised* best. PaLM with chain-of-thought crosses it.
"So PaLM models were roughly about 540 billion parameters, and simply with this chain-of-thought
kind of a skill you're already beating the state of the art" (≈19:17).

This is the same scale dependence [prompting](prompting.md) documents for in-context learning:
the technique does nothing until the model is large enough.

## Zero-shot chain-of-thought

Slide 28 asks the obvious follow-up: "Do we even need examples of reasoning? Can we just ask the
model to reason through things?"

Slide 29's answer (Kojima et al., 2022) is to prepend a single trigger phrase to the answer
rather than supplying any exemplars:

> Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls
> are blue. How many blue golf balls are there?
> A: ***Let's think step by step.*** There are 16 balls in total. Half of the balls are golf
> balls. That means there are 8 golf balls. Half of the golf balls are blue. That means there are
> 4 blue golf balls. ✓

Slide 30 quantifies it:

| | MultiArith | GSM8K |
| --- | --- | --- |
| Zero-Shot | 17.7 | 10.4 |
| Few-Shot (8 samples) | 33.8 | 15.6 |
| **Zero-Shot-CoT** | **78.7** | **40.7** |
| Few-Shot-CoT (8 samples) | 93.0 | 48.7 |

Two readings, both stated on the slide: zero-shot CoT "greatly outperforms zero-shot" — 17.7 to
78.7 on MultiArith from one sentence — and "manual CoT still better", so exemplars are not
obsolete.

## The trigger phrase matters

Slide 31 is a table of trigger phrases ranked by accuracy, which is either encouraging or
alarming depending on temperament:

| Category | Trigger prompt | Accuracy |
| --- | --- | --- |
| LM-designed | Let's work this out in a step by step way to be sure we have the right answer. | **82.0** |
| Human-designed | Let's think step by step. | 78.7 |
| | First, | 77.3 |
| | Let's think about this logically. | 74.5 |
| | Let's solve this problem by splitting it into steps. | 72.2 |
| | Let's be realistic and think step by step. | 70.8 |
| | Let's think like a detective step by step. | 70.3 |
| | Let's think | 57.5 |
| | Before we dive into the answer, | 55.7 |
| | The answer is after the proof. | 45.7 |
| — | (Zero-shot) | 17.7 |

The best phrase was **designed by a language model**, not a human (Zhou et al., 2022), and beats
the best human-written one by 3.3 points. "You can also get an LLM to design these prompts as
well. There are recursive self-improving ideas that happen here" (≈21:34).

The spread from 45.7 to 82.0 across phrases that all mean roughly the same thing is also, read
the other way, a measurement problem: it is the same brittleness
[lecture 12](12-benchmarking.md) documents when relabelling MMLU's options changes the model
ranking. See [benchmark contamination and overfitting](benchmark-contamination.md).

## Self-consistency: sample many rationales, vote

[Lecture 15](15-reasoning-and-agents.md) (slides 12–14, ≈4:47) changes the *decoding* rather than
the prompt. Instead of greedily decoding one rationale and then an answer conditioned on it,
sample many rationales, collect the answer each leads to, and return the most common one. The
intuition is that an answer reached by many independent routes is more likely to be right
(≈5:33), and across a range of mathematical reasoning tasks this improves on standard
chain-of-thought "pretty drastically."

The obvious objection is that this is just ensembling — the CS229 trick of training several
classifiers and taking a majority vote — and Murty says that was his own first reaction (≈6:19).
The authors tested it: against one model with several *different prompts* and a majority vote,
self-consistency still won. Sampling multiple reasoning paths from a fixed prompt does something
beyond averaging away prompt variance.

## Least-to-most: decompose before answering

Multi-step reasoning means "breaking down a large problem into several subparts, answering each
of the subparts, and then combining everything into a solution" (≈7:06), so **least-to-most
prompting** (slides 15–19) makes each of those an explicit generation: break the question into
sub-questions, answer each, then condition the final answer on those answers (≈7:53).

Its notable result is **length generalisation** — prompted with a two-step math word problem, the
model keeps working on test items needing more than five steps (≈8:39). But the same paper's
ablation shows that "with enough prompt engineering, the row corresponding to best-prompted
chain-of-thought is on par with least-to-most prompting" (≈9:26). Useful, not fundamental.

## Are the rationales faithful?

The uncomfortable question (slides 31–32, ≈20:21): does the answer actually *depend* on the
rationale? The failure mode is a model that writes "Tom has three apples, Jerry has four, 3 + 4 is
seven" and then answers 25 (≈21:07) — reasoning as decoration.

Two experiments probe it:

- **Early exit** (≈21:53). Force the model to stop after the first sentence of its rationale and
  answer; then after the second; and so on, plotting accuracy against how much rationale it was
  allowed. If one sentence yields the same answer as four, the rest was not doing work — and in
  the limit you can cut the rationale entirely (≈22:39).
- **Corruption** (≈23:27). Introduce a mistake after some fraction of the reasoning steps. The
  expected curve is strictly increasing: an error in step one should wreck the answer, an error
  in the last step should barely matter (≈24:14).

Results are mixed across datasets, but on enough of them the answer is unmoved by truncation or
early corruption, which "means that sometimes these rationales may be post-hoc explanations of
the model's answer" (≈22:39).

That is a distinct worry from whether the answer itself was memorised — see
[counterfactual evaluation](counterfactual-evaluation.md), which attacks the other half of the
question. A model can produce faithful reasoning toward a memorised answer, or unfaithful
reasoning toward a correct one.

## Where it goes next

Chain-of-thought reappears as *training* signal rather than prompt: FLAN's instruction mixture
includes "answer the following question by reasoning step-by-step" examples
([instruction finetuning](instruction-finetuning.md), slide 40), and the Self-Taught Reasoner
(STaR) loop finetunes a model on its own chains of thought (slide 98) — a pattern
[lecture 15](15-reasoning-and-agents.md) generalises as
[self-training and rationale distillation](self-training-and-rationale-distillation.md). The
Llama 3 note on slide
82 gives the sharpest statement of why preference training helps here: "the model knows how to
produce the right answer, but it does not know how to select it. Training on preference rankings
enables the model to learn how to select it."
