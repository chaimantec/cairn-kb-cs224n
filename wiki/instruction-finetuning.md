# Instruction finetuning

Finetuning a pretrained language model on (instruction, output) pairs drawn from *many* tasks,
so that it follows instructions on tasks it has never seen. Covered in
[lecture 11](11-post-training.md), slides 38–53.

## The recipe

It is the [pretrain-then-finetune](pretraining-and-finetuning.md) paradigm from
[lecture 9](09-pretraining.md) with one word changed. Slide 38 restates the old picture —
pretrain a decoder on language modelling, then finetune it on *your task* with not many labels.
Slide 39 replaces that second step with **"finetune on many tasks"**, and strikes out the "not"
in "not many labels".

Concretely (slide 40):

1. **Collect examples** of (instruction, output) pairs across many tasks and finetune an LM.
2. **Evaluate on unseen tasks.**

The FLAN figure gives the flavour of the training pairs — "Please answer the following question.
What is the boiling point of Nitrogen?" → "-320.4F"; "Answer the following question by reasoning
step-by-step. The cafeteria had 23 apples…" → the full worked
[chain of thought](chain-of-thought.md) — and of the held-out evaluation: "Can Geoffrey Hinton
have a conversation with George Washington? Give the rationale before answering." → "Geoffrey
Hinton is a British-Canadian computer scientist born in 1947. George Washington died in 1799.
Thus, they could not have had a conversation together. So the answer is 'no'."

There are no new algorithms here. "The recipe is not very complicated — we're going to collect a
lot of examples of instruction and output pairs… and then evaluate on some unseen tasks"
(≈23:53).

## Scale, again

Slide 41 crosses out "finetuning" in its own title and writes "pretraining?" instead, because the
data volumes stopped looking like finetuning: Super-NaturalInstructions has **over 1.6K tasks and
3M+ examples**, spanning classification, sequence tagging, rewriting, translation and QA. The
lecture's aside: "you might even think, why are we calling it finetuning any more? It's almost
starting to look like pretraining. But yeah, these are just terms" (≈24:41).

Slide 47 is the scaling result, on Flan-T5 — T5 models finetuned on 1.8K additional tasks,
evaluated on the normalized BIG-Bench + MMLU average:

| Params | Base | Flan | Δ |
| --- | --- | --- | --- |
| 80M | T5-Small −9.2 | −3.1 | **+6.1** |
| 250M | T5-Base −5.1 | 6.5 | **+11.6** |
| 780M | T5-Large −5.0 | 13.8 | **+18.8** |
| 3B | T5-XL −4.1 | 19.1 | **+23.2** |
| 11B | T5-XXL −2.9 | 23.7 | **+26.6** |

**Bigger model = bigger Δ.** Instruction finetuning does not merely add a constant; larger models
get more out of the same procedure. This is the theme the lecture says runs through the whole
hour: "as your models become larger, as they're trained on more data, they become more and more
responsive to your task information as well" (≈28:33).

Slides 48–49 show the qualitative change on one Disambiguation QA item. Before instruction
finetuning the model paraphrases the input sentence four times and never answers. After, it
answers: "The reporter and the chef will discuss their favorite dishes does not indicate whose
favorite dishes they will discuss. So, the answer is (C)."

## How the data gets made

Slide 50 is a dense lineage graph of the open-source instruction-tuning ecosystem that followed
the LLaMA release — Alpaca, Self-Instruct, Koala, Baize, OpenAssistant, databricks-dolly-15k,
GPT4All, MPT-7B-Instruct, and dozens more. Slide 51 extracts three lessons from it:

- **You can generate the data synthetically, from a bigger model.** Alpaca's pipeline: 175
  Self-Instruct seed tasks → modified Self-Instruct generation with `text-davinci-003` → 52K
  instruction-following examples → supervised finetuning of LLaMA 7B. "Instead of getting a human
  to collect all the instruction-output pairs, or getting humans to generate the answers, you can
  get bigger models to generate the answers" (≈30:54).
- **You may not need much of it.** LIMA — *Less Is More for Alignment* — finds that a thousand
  really high-quality examples can be enough. The lecture is careful to call this an active area
  rather than a settled result.
- **Crowdsourcing works.** OpenAssistant is the example.

A student asks whether, given that code and math word problems already have the step-by-step
structure you want, one should just train on code. The answer (≈32:27) is that code *is*
up-weighted heavily in pretraining mixtures and does help, but that it does not cover what people
actually use these models for — "people often use these models for creative tasks. They want to
write a story, they want to generate a movie script" — and reasoning-only training would not help
there.

## Evaluating a model that does everything

Slide 41 ends with a question the lecture then has to answer: **how do you evaluate such a model?**
Slides 42–46 are its aside on multitask benchmarks — MMLU's 57 knowledge-intensive tasks, the
progress curve from GPT-2 at ~32% to Gemini Ultra at ~90%, and BIG-Bench's 200+ tasks (including
"Kanji ASCII Art to Meaning"). This is the point where the course hands off to
[lecture 12](12-benchmarking.md); see [evaluating LLMs](evaluating-llms.md).

Two cautions are raised on the spot. Asked whether benchmark saturation is "the whole ImageNet
thing all over again", the lecturer grants the concern about test-set leakage and offers a
pragmatic counterweight: "at some point it doesn't matter what your train/test split is, if the
models are generally useful" (≈27:48) — a position [lecture 12](12-benchmarking.md) treats far
less charitably under [contamination](benchmark-contamination.md). And on saturation itself:
"most of the times when the models are wrong, you might often find that the question itself was
unclear or ambiguous" (≈28:33).

## Why it is not enough

Slide 53 lists the limitations that motivate preference optimization. Beyond the obvious expense
of collecting ground-truth answers — which grows as the questions get harder, up to "physics PhD
level" — there are three:

**Problem 1: open-ended tasks have no right answer.** *"Write me a story about a dog and her pet
grasshopper."* There is nothing to imitate.

**Problem 2: language modelling penalizes all token-level mistakes equally.** The slide's figure
takes *Avatar is a fantasy TV show* and crosses out two alternatives at the same position:
*adventure*, which would be nearly fine, and *musical*, which would be wrong. Cross-entropy
charges the same for both.

**Problem 3: humans generate suboptimal answers.** The demonstrations cap the model at the
quality of what an annotator will write, and models are getting competitive with that. "Do we
really want to keep relying on humans to write down the answers?" (≈36:19).

Underneath all three sits one mismatch: even with instruction finetuning, the objective is
next-token prediction on a curated corpus while the goal is *satisfying human preferences*. That
is what [RLHF](rlhf.md) and [DPO](direct-preference-optimization.md) attack.

Instruction finetuning does not go away, though — it is step 1 of the RLHF pipeline (slide 57),
and ChatGPT's own method description begins with it (slide 79). Asked whether the step is still
taken or discarded, the lecturer says it remains "one of the more important steps", though
"people are trying to remove the step altogether and jump directly to the next step" (≈37:04).
