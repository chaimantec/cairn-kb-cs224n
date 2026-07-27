# Self-training and rationale distillation

One recipe, three appearances. Generate candidate outputs with a language model, **filter** them
by some signal of quality, **fine-tune** on what survives, and optionally **iterate**. In
[lecture 15](15-reasoning-and-agents.md) this pattern shows up as Orca (a big model teaching a
small one), ReST (a model teaching itself), and BAGEL (an agent generating its own
demonstrations) — and Murty draws the connection explicitly when he reaches the agents half:
"we're going to use something we saw early on in the lecture" (≈47:31).

What differs between the three is only *where the filter's signal comes from*. That is the axis
worth holding onto.

| Method | Generator | Filter signal | Trained model |
| --- | --- | --- | --- |
| **Orca** | GPT-4 / ChatGPT | none — teacher output is taken as-is | a smaller Llama-13B |
| **ReST** | the model itself | the known correct answer | the same (large) model, iteratively |
| **BAGEL** | the model exploring an environment | a second model's judgment $R(g, \tau)$ | used as in-context demos, or fine-tuned |

## Orca: distilling rationales into a small model

Prompting-based reasoning needs a very large model. Distillation is the alternative: "maybe you
want to fine-tune a smaller Llama model by teaching it to imitate a larger language model"
(≈10:13).

Orca fine-tunes a **13-billion-parameter Llama** on explanations produced by GPT-4, in three
steps (slides 20–28, ≈10:58):

1. Draw a wide variety of instructions from the **FLAN-v2** collection, itself an aggregation of
   many datasets pairing instructions with questions and answers.
2. Prompt GPT-4 or ChatGPT with each instruction **plus a system message** designed to elicit an
   informative explanation alongside the answer — "please justify your steps, and answer step by
   step" (≈11:45). The worked example on the slide asks for a median and gets back a full
   derivation.
3. Fine-tune the small model on those explanations.

The system message is the active ingredient. Without it the teacher emits answers; with it the
teacher emits the [chain-of-thought](chain-of-thought.md) that the student needs in order to
learn the *procedure* rather than the answer key.

Evaluated on **BigBench-hard** (23 multi-step sub-tasks), Orca beats ChatGPT — plausibly because
it is specialised to reasoning problems — and beats Vicuna, an
[instruction-tuned](instruction-finetuning.md) Llama-13B trained without extensive explanations
(≈15:39). The GPT-4 column is set aside for possible
[contamination](benchmark-contamination.md) (≈14:53).

## ReST: training a model on its own rationales

The obvious follow-up question is why the *large* model should not learn from itself (≈16:26).
**ReST** — reinforced self-training (slides 29–30) — alternates two stages:

1. **Generate and filter.** Given a reasoning problem, and perhaps "let's think step by step,"
   sample multiple rationales. Keep those that reach the correct final answer, discard the rest.
   Murty's illustration: three apples plus four apples, a rationale ending in seven is kept, one
   ending in 12 is dropped (≈16:26).
2. **Update.** Fine-tune on the surviving rationales.

Then repeat: a better model gives better rationales, which give a better model (≈17:14).

The filter here is the strongest of the three — a verifiable ground-truth answer — which is
exactly why the method is confined to domains where such an answer exists. It also does not check
that the rationale is *valid*, only that it arrived somewhere correct; a rationale can be lucky.

**Results are mixed, and the lecture says so.** On **GSM8K** (grade-school algebraic word
problems) accuracy improves slightly with more iterations "and then it starts degrading" (≈18:02).
On **MATH** it keeps improving. In those plots orange is a larger PaLM model, blue a smaller one,
and dashed lines mark supervised fine-tuning on human-written rationales (≈18:02).

The headline finding is that self-generated rationales can beat human ones, shown with a
quantity-controlled comparison (≈18:48):

| Bar | Training data |
| --- | --- |
| Blue | **all** human-provided rationales |
| Orange | **one human** rationale per training example |
| Green | **one model-generated** rationale per question, chosen at random |

Green over orange is the like-for-like comparison, and green wins; running the full iterative
procedure adds more. The reading is not that models write better explanations than people, but
that they write explanations better matched to their own distribution.

## BAGEL: synthetic demonstrations for agents

For [language model agents](language-model-agents.md) the bottleneck is demonstrations. Standard
practice puts human-performed task traces into the prompt as few-shot examples, which must be
redone for every new website — hopeless across "thousands of environments" (≈46:44).

The difficulty in reusing ReST here is that **there is no known correct answer for a trajectory**
(≈48:19). A math problem has a checkable result; "click, type, scroll" does not. BAGEL —
Bootstrapping Agents by Guiding Exploration with Language (slides 57–67) — solves that by making
a second language model supply the signal:

1. **Explore randomly.** An unconditioned agent fires off random clicks, types and scrolls,
   producing trajectories (≈47:31).
2. **Guess the instruction.** A second model describes what each trajectory accomplished —
   "book a flight from San Francisco to New York," "set the date to …" — and sometimes fails to
   produce anything sensible (≈48:19). The premise: if a trajectory can be described, it was
   probably coherent.
3. **Re-roll on the guessed goal.** Feed that instruction back so the model acts *toward* it
   rather than at random (≈49:52).
4. **Filter.** A coarse filter, written in the deck as $R(g, \tau)$, scores the correspondence
   between instruction, action sequence and visited states (≈49:52).
5. **Relabel instead of discarding.** A failed trajectory still did *something*, and real-website
   interactions are expensive, so the relabeler assigns a new instruction — perhaps "set the
   origin to SFO and the destination to New York City" — and the loop repeats until the filter
   passes (≈51:24).

The output can be fine-tuned on, but the simplest use is to substitute it for human
demonstrations in the prompt (≈52:55): a **13-point gain on MiniWoB++** (46.8 → 60.5 on slide 67),
plus a smaller and less uniform gain on a multi-step tool-use benchmark, where the average moves
40.9 → 43.3 and some categories do not improve.

Step 5 is the genuinely new idea. ReST throws away everything that fails its filter; BAGEL treats
a failure as a correctly-executed trajectory for a *different, unknown* instruction, and recovers
the data by inferring what that instruction was.

## The caveat that applies to all three

These methods are evaluated by benchmark accuracy, and the same lecture spends its middle section
arguing that benchmark accuracy may reflect memorisation rather than reasoning
([counterfactual evaluation](counterfactual-evaluation.md), ≈25:01). The gains above are real
gains on the benchmarks as constituted; whether they represent improved reasoning is the question
the lecture leaves open (≈1:01:20).

## Related pages

- [Lecture 15 — Reasoning and agents](15-reasoning-and-agents.md) — the source.
- [Chain-of-thought](chain-of-thought.md) — what a "rationale" is.
- [Language model agents](language-model-agents.md) — the setting BAGEL serves.
- [Counterfactual evaluation](counterfactual-evaluation.md) — the check on these results.
- [Instruction fine-tuning](instruction-finetuning.md) — the training regime being extended.
- [Pretraining and fine-tuning](pretraining-and-finetuning.md) — the general two-stage picture.
- [Post-training](11-post-training.md) — where preference-based alternatives to this recipe live.
