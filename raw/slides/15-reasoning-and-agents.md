---
title: Lecture 15 — Reasoning and Agents (slide deck)
lecture: 15
slides: 75 printed / 75 pages in the PDF
source_pdf: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture14-agents-shikhar-updated.pdf
note: |
  Lecturer is Shikhar Murty. The deck's own title is "Lecture 14: Reasoning and Agents"; the
  Cairn catalog lists it at **position 15**, while the video title and deck filename call it
  lecture 14 — this repo names files by the catalog position. Printed slide numbers match PDF
  page numbers 1:1 with no gaps and no offset: page 1 (the title page) prints no number but
  occupies position 1, and slide N is page N throughout.
---

# Lecture 15 — Reasoning and Agents: slide-by-slide

Text and figures of
[`cs224n-spr2024-lecture14-agents-shikhar-updated.pdf`](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture14-agents-shikhar-updated.pdf),
transcribed from the deck. Diagrams and plots are described in prose since the KB is
read as text.

Companion pages: [wiki page for this lecture](../../wiki/15-reasoning-and-agents.md) ·
[transcript](../transcripts/15-reasoning-and-agents.md)

## Contents

The two section-title slides are 3 and 38.

| Slides | Section |
| ------ | ------- |
| 1 | Title |
| 2 | Lecture plan and disclaimer |
| 3 | §1 Section title: Reasoning (with Large Language Models) |
| 4–8 | What is reasoning: deductive, inductive, abductive reasoning; formal vs. informal reasoning |
| 9 | Can current LLMs reason? |
| 10–14 | Reasoning via prompting: chain-of-thought, zero-shot CoT, self-consistency, and a benchmark results table |
| 15–19 | Problem decomposition with least-to-most prompting, including math reasoning |
| 20–28 | Reasoning via distillation: Orca, instruction-tuning small LMs with CoT rationales |
| 29–30 | Reasoning by finetuning LMs on their own outputs |
| 31–37 | Can language models reason? Faithfulness of CoT rationales; reasoning vs. memorization via counterfactuals (arithmetic and analogical reasoning) |
| 38 | §2 Section title: Language Model Agents |
| 39–42 | Some terminology for LM agents |
| 43–45 | Applications: natural language interfaces, UI automation, multi-step tool use |
| 46–48 | Instruction-following agents [Pre LLMs] |
| 49–51 | Instruction-following agents [in 2024] |
| 52 | A simple language model agent with ReACT |
| 53–55 | Benchmarks for LM agents: MiniWoB++, WebArena, WebLINX |
| 56 | Training data for language model agents |
| 57–64 | Use exploration + model-generated data: an exploring robot, a labelling robot, the reward function $R(g,\tau)$, and iterative re-labeling |
| 65–67 | BAGEL (Bootstrapping Agents by Guiding Exploration with Language): the three-step pipeline, and MiniWoB++/ToolQA results |
| 68–70 | Multimodality: operating over pixels instead of HTML; LLaVA; Pix2Struct |
| 71–74 | LM agents is an emerging application: the "prompting gap," long-horizon planning difficulty, human-vs-model success rates, and a grounding/action-failure example |
| 75 | Lecture-14 Recap |

---

## Slide 1 — Title

**Natural Language Processing with Deep Learning — CS224N/Ling284.** Below the Stanford arch
logo (a dark-red arch over three tan/beige smaller arches): **Shikhar Murty**, *Lecture 14:
Reasoning and Agents*.

## Slide 2 — Lecture Plan

**Lecture 14: Reasoning and Agents**
1. Reasoning in Language Models [35 mins]
2. Mini-break [5 mins]
3. Language Model Agents [40 mins]

- Announcements
  - Project Milestone due on Wed May 22nd at 4:30 pm
  - Your Project Mentors have already reached out to you (If not, let us know via Ed!)
  - Guest lectures on May 21st and May 28th: Students get 0.75% per guest lecture for attending
    live or writing a reaction paragraph (More details will be on Ed)

Boxed disclaimer: "Disclaimer: Content for today is an active area of research and still
emerging."

## Slide 3 — Section title: Reasoning (with Large Language Models)

## Slide 4 — What is Reasoning?

"Using *facts* and *logic* to arrive at an answer" (with "facts" and "logic" in purple italics).

Footer credit: "Slide credit: Graham Neubig (11-711 ANLP)"

## Slide 5 — What is Reasoning?

Same opening line as slide 4, with a new definition added:

**Deductive Reasoning:** Use logic to go from premise to firm conclusion

Boxed example:
```
Premise: All mammals have kidneys
Premise: All whales are mammals
Conclusion: All whales have kidneys
```

## Slide 6 — What is Reasoning?

Same content as slide 5, with a second definition added:

**Inductive Reasoning:** From observation, predict a likely conclusion

Boxed example:
```
Observation: When we see a creature with wings, it is usually a bird
Observation: We see a creature with wings.
Conclusion: The creature is likely to be a bird
```

## Slide 7 — What is Reasoning?

Same content as slide 6, with a third definition added:

**Abductive Reasoning:** From observation, predict the most likely explanation

Boxed example:
```
Observation: The car cannot start and there is a puddle of liquid under the engine.
Likely Explanation: The car has a leak in the radiator
```

## Slide 8 — Reasoning: Formal vs Informal

**Formal Reasoning:** Follows formal rules of logic along with axiomatic knowledge to derive
conclusions.

**Informal Reasoning:** Uses intuition, experience, common sense to arrive at answers.

Below, offset to the right: "For most of this lecture, by 'reasoning' we mean informal deductive
reasoning, often involving multiple steps"

## Slide 9 — Reasoning in Large Language Models

"Large Language models are **REALLY GOOD** at predicting **plausible continuations of text
(Lecture-9)**, that respect **constraints in the input (Lecture 10,11)**, and align well with
**human preferences (Lecture-10, 11)**." (the three parenthetical cross-references are in green).

**Question**: Can *current* LLMs reason? (the word "current" is in purple).

## Slide 10 — Reasoning in Large Language Models: prompting

"Chain-of-thought prompting:"

A reproduced figure (Figure 1) comparing **Standard Prompting** and **Chain-of-Thought
Prompting**, each with a two-shot example about tennis balls followed by a new apple-counting
question:
- **Standard Prompting**: Model Input gives a worked example ("Q: Roger has 5 tennis balls...
  A: The answer is 11.") then a new question about a cafeteria's apples; Model Output is "A: The
  answer is 27." marked wrong with a red X.
- **Chain-of-Thought Prompting**: the same worked example's answer is expanded into an explicit
  reasoning chain (highlighted in blue): "Roger started with 5 balls. 2 cans of 3 tennis balls
  each is 6 tennis balls. 5 + 6 = 11. The answer is 11."; given the same new apple question, the
  Model Output (highlighted in green) reasons "The cafeteria had 23 apples originally. They used
  20 to make lunch. So they had 23 - 20 = 3. They bought 6 more, so they have 3 + 6 = 9. The
  answer is 9." marked correct with a green check.

Caption: "Figure 1: Chain-of-thought prompting enables large language models to tackle complex
arithmetic, commonsense, and symbolic reasoning tasks. Chain-of-thought reasoning processes are
highlighted."

Source: Wei et al. 2023

## Slide 11 — Reasoning in Large Language Models: prompting

"Zero-shot CoT prompting:"

A reproduced four-panel figure using the same juggler word problem ("A juggler can juggle 16
balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf
balls are there?"):
- **(a) Few-shot**: a tennis-ball worked example followed by the juggler question with a bare
  "A:" prompt; Output "The answer is 8." marked wrong (red X).
- **(b) Few-shot-CoT**: the tennis-ball example's answer is spelled out as a reasoning chain;
  given the juggler question, the model's output (italic) reasons "The juggler can juggle 16
  balls. Half of the balls are golf balls. So there are 16 / 2 = 8 golf balls. Half of the golf
  balls are blue. So there are 8 / 2 = 4 blue golf balls. The answer is 4." marked correct
  (green check).
- **(c) Zero-shot**: just the juggler question with prompt "A: The answer (arabic numerals) is";
  Output "8" marked wrong (red X).
- **(d) Zero-shot-CoT (Ours)**: the juggler question with prompt "A: **Let's think step by
  step.**" (highlighted pink); Output (italic) "There are 16 balls in total. Half of the balls
  are golf balls. That means that there are 8 golf balls. Half of the golf balls are blue. That
  means that there are 4 blue golf balls." marked correct (green check).

Source: Kojima et al. 2023

## Slide 12 — Reasoning in Large Language Models: prompting

'CoT with "Self-consistency": Replace greedy decoding with an ensemble of samples…'

**Main idea:** correct reasoning processes have greater agreement than incorrect processes.

A reproduced diagram with two rows:
- Top row, **Chain-of-thought prompting**: a Prompt box feeds a Language model box, which
  produces one **Greedy decode** reasoning chain about a woman selling eggs ("This means she uses
  3 + 4 = 7 eggs every day. She sells the remainder for \$2 per egg, so in total she sells 7 * \$2
  = \$14 per day. **The answer is \$14.**") — a single, incorrect answer box.
- Bottom row, **Self-consistency**: the same prompt (with two example Q/A pairs about parking-lot
  cars and Janet's ducks/eggs) feeds a Language model box, which now **"Sample[s] a diverse set
  of reasoning paths"** — three separate reasoning chains are drawn: (1) "She has 16 - 3 - 4 = 9
  eggs left. So she makes \$2 * 9 = \$18 per day." → "The answer is \$18."; (2) "This means she
  sells the remainder for \$2 * (16 - 4 - 3) = \$26 per day." → "The answer is \$26."; (3) "She
  eats 3 for breakfast, so she has 16 - 3 = 13 left. Then she bakes muffins, so she has 13 - 4 = 9
  eggs left. So she has 9 eggs * \$2 = \$18." → "The answer is \$18." These three candidate
  answers are then **"Marginalize[d] out … to aggregate final answers"**, converging on a single
  final green box: "The answer is \$18."

Source: Wang et al. 2023

## Slide 13 — Reasoning in Large Language Models: prompting

A results table comparing CoT-prompting against Self-consistency across six math/reasoning
benchmarks, for three base models:

| | Method | AddSub | MultiArith | ASDiv | AQuA | SVAMP | GSM8K |
|---|---|---|---|---|---|---|---|
| | Previous SoTA | **94.9**ᵃ | 60.5ᵃ | 75.3ᵇ | 37.9ᶜ | 57.4ᵈ | 35ᵉ / 55ᵍ |
| UL2-20B | CoT-prompting | 18.2 | 10.7 | 16.9 | 23.6 | 12.6 | 4.1 |
| UL2-20B | Self-consistency | 24.8 (+6.6) | 15.0 (+4.3) | 21.5 (+4.6) | 26.9 (+3.3) | 19.4 (+6.8) | 7.3 (+3.2) |
| LaMDA-137B | CoT-prompting | 52.9 | 51.8 | 49.0 | 17.7 | 38.9 | 17.1 |
| LaMDA-137B | Self-consistency | 63.5 (+10.6) | 75.7 (+23.9) | 58.2 (+9.2) | 26.8 (+9.1) | 53.3 (+14.4) | 27.7 (+10.6) |
| PaLM-540B | CoT-prompting | 91.9 | 94.7 | 74.0 | 35.8 | 79.0 | 56.5 |
| PaLM-540B | Self-consistency | 93.7 (+1.8) | 99.3 (+4.6) | 81.9 (+7.9) | 48.3 (+12.5) | 86.6 (+7.6) | 74.4 (+17.9) |

Right: "**Out-performs regular CoT on a variety of benchmarks**"

Source: Wang et al. 2023

## Slide 14 — Reasoning in Large Language Models: prompting

Same table as slide 13, with a second table added below comparing self-consistency against
simple ensembling:

| | GSM8K | MultiArith | SVAMP | ARC-e | ARC-c |
|---|---|---|---|---|---|
| CoT (Wei et al., 2022) | 17.1 | 51.8 | 38.9 | 75.3 | 55.1 |
| Ensemble (3 sets of prompts) | 18.6 ± 0.5 | 57.1 ± 0.7 | 42.1 ± 0.6 | 76.6 ± 0.1 | 57.0 ± 0.2 |
| Ensemble (40 prompt permutations) | 19.2 ± 0.1 | 60.9 ± 0.2 | 42.7 ± 0.1 | 76.9 ± 0.1 | 57.0 ± 0.1 |
| Self-Consistency (40 sampled paths) | **27.7 ± 0.2** | **75.7 ± 0.3** | **53.3 ± 0.2** | **79.3 ± 0.3** | **59.8 ± 0.2** |

Right: "**Self-consistency is doing more than simple ensembling**"

Source: Wang et al. 2023

## Slide 15 — Reasoning in Large Language Models: prompting — Problem decomposition with Least-to-Most prompting

A reproduced diagram, **"Stage 1: Decompose Question into Subquestions"**: a Q box ("It takes
Amy 4 minutes to climb to the top of a slide. It takes her 1 minute to slide down. The water
slide closes in 15 minutes. How many times can she slide before it closes?") feeds a Language
Model box, whose output (A) explains: 'To solve "How many times can she slide before it closes?"
(highlighted orange), we need to first solve: "How long does each trip take?" (highlighted
green).'

## Slide 16 — Reasoning in Large Language Models: prompting — Problem decomposition with Least-to-Most prompting

Same Stage 1 diagram as slide 15, with **"Stage 2: Sequentially Solve Subquestions"** added
below: the original problem statement plus **Subquestion 1** ("How long does each trip take?",
highlighted green) feeds a Language Model box, whose answer is "It takes Amy 4 minutes to climb
and 1 minute to slide down. 4 + 1 = 5. So each trip takes 5 minutes." (the last clause highlighted
green). This answer is then appended to the context ("Append model answer to Subquestion 1"),
together with **Subquestion 2** ("How many times can she slide before it closes?", highlighted
orange), and fed to a second Language Model box, whose answer is "The water slide closes in 15
minutes. Each trip takes 5 minutes. So Amy can slide 15 ÷ 5 = 3 times before it closes." (with
"Each trip takes 5 minutes" highlighted green and "Amy can slide 15 ÷ 5 = 3 times" highlighted
orange).

Source: Zhou et al. 2023

## Slide 17 — Reasoning in Large Language Models: prompting — Least-to-Most prompting for Math reasoning

Two boxed prompt examples, both using an apples word problem ("Elsa has 5 apples. Anna has 2 more
apples than Elsa. How many apples do they have together?"):
- **Standard CoT prompting**: "A: Anna has 2 more apples than Elsa, so Anna has 2 + 5 = 7 apples.
  Elsa and Anna have 5 + 7 = 12 apples together. The answer is 12."
- **Least-to-most prompting**: "A: Let's break down this problem: 1. How many apples does Anna
  have? 2. How many apples do Elsa and Anna have together? 1. Anna has 2 more apples than Elsa.
  So Anna has 2 + 5 = 7 apples. 2. Elsa and Anna have 5 + 7 = 12 apples together." followed by a
  template for a new question: "Q: {question} A: Let's break down this problem: — The answer
  is:"

Source: Zhou et al. 2023

## Slide 18 — Reasoning in Large Language Models: prompting — Least-to-Most prompting for Math reasoning

A table, **"Accuracy by Steps (GSM8K)"**, comparing Least-to-Most and Chain-of-Thought broken
down by number of reasoning steps in the problem:

| Accuracy by Steps (GSM8K) | All | 2 Steps | 3 Steps | 4 steps | ≥ 5 steps |
|---|---|---|---|---|---|
| Least-to-Most | **62.39** | 74.53 | **68.91** | **59.73** | **45.23** |
| Chain-of-Thought | 60.87 | **76.68** | 67.29 | 59.39 | 39.07 |

Right: "**Generalizes to more #steps than in-context example!**"

Source: Zhou et al. 2023

## Slide 19 — Reasoning in Large Language Models: prompting — Least-to-Most prompting for Math reasoning

Same "Accuracy by Steps (GSM8K)" table as slide 18, with a second table added below:

| Prompting method | Accuracy |
|---|---|
| Zero-Shot | 16.38 |
| Standard prompting | 17.06³ |
| Chain-of-Thought (original) | 61.18 |
| Chain-of-Thought (1-shot) | 60.88 |
| Least-to-Most (1-shot) | 62.39 |
| Chain-of-Thought (best) | **68.61³** |
| Least-to-Most (best) | 68.01 |

Right: "**But with enough prompt engineering, CoT ≈ Least-to-Most**"

Source: Zhou et al. 2023

## Slide 20 — Reasoning in ~~Large~~ Language Models via distillation

"So far, we've only looked at prompting >100B parameter models for reasoning"

"Can we get reasoning-like behavior with smaller LMs by teaching them to imitate larger models?"

Below, a decorative illustration of a small llama/alpaca standing beside a larger adult
llama/alpaca on a light hexagon-patterned background — a visual pun on "distillation" from a
larger "teacher" model into a smaller "student" model.

## Slide 21 — Orca: Instruction-tuning small LMs with CoT Rationales

"1. Collect a wide variety of instructions from the FLAN-v2 collection"

A reproduced table, **"Table 3: Construction of our training data with 5 million samples."**:

| Mixture Name | Sampling Algorithm | Original Size | Sampled |
|---|---|---|---|
| CoT | No Sampling | 150K | 150K |
| NIV2 | Stratified Sampling | 5M | 440K |
| FLAN2021 | Stratified Sampling | >28.9M | 2.5M |
| T0 | Stratified Sampling | 85.7M | 2M |
| Dialog | Skipped | 22.5M | 0 |

Source: Mukherjee et al. 2023

## Slide 22 — Orca: Instruction-tuning small LMs with CoT Rationales

"1. Collect a wide variety of instructions from the FLAN-v2 collection."
"2. Prompt GPT4 or ChatGPT with these instructions along with a **system message**" (the phrase
"system message" is in purple).

Below, two stacked boxes reproducing an example interaction:
- Top box: "**System Instruction:** You are an AI assistant. User will you give you a task. Your
  goal is to complete the task as faithfully as you can. While performing the task think
  step-by-step and justify your steps." / "**User Instruction:** Use the given data to calculate
  the median." / "**Input:** [7, 3, 8, 2, 10]"
- Bottom box: "**Output:** To calculate the median, I will follow these steps: Step 1: Arrange
  the data in ascending order [2, 3, 7, 8, 10] Step 2: Determine if the data set has an odd or
  even number of values The data set has 5 values, which is an odd number. Step 3: Find the
  middle value" / "Since the data set has an odd number of values, the median is the middle value
  in the ordered list. The middle value in the list [2, 3, 7, 8, 10] is 7."

Source: Mukherjee et al. 2023

## Slide 23 — Orca: Instruction-tuning small LMs with CoT Rationales

Same two bullets as slide 22, with a third added:
"3. Finetune Llama-13b on outputs generated via ChatGPT + GPT4"

## Slide 24 — Orca: Instruction-tuning small LMs with CoT Rationales

"Evaluation: *BigBench-hard*, a collection of 23 Big-Bench tasks with a special focus on
multi-step reasoning" (the term "BigBench-hard" is in purple italics).

## Slide 25 — Orca: Instruction-tuning small LMs with CoT Rationales

Same opening line as slide 24. A boxed sub-heading **"Boolean Expressions"**, below it a worked
example reproduced verbatim:

"Q: True and False and not True and True is
A: Let's think step by step.
Remember that (i) expressions inside brackets are always evaluated first and that (ii) the order
of operations from highest priority to lowest priority is "not", "and", "or", respectively.
We first simplify this expression "Z" as follows: "Z = True and False and not True and True = A
and B" where "A = True and False" and "B = not True and True".
Let's evaluate A: A = True and False = False.
Let's evaluate B: B = not True and True = not (True and True) = not (True) = False.
Plugging in A and B, we get: Z = A and B = False and False = False. So the answer is False."

Source: Suzgun et al. 2022

## Slide 26 — Orca: Instruction-tuning small LMs with CoT Rationales

Same opening line as slide 24. A boxed sub-heading **"Data Understanding"**, below it a worked
example reproduced verbatim:

"Q: Tomorrow is 11/12/2019. What is the date one year ago from today in MM/DD/YYYY?
Options:
(A) 09/04/2018
(B) 11/11/2018
(C) 08/25/2018
(D) 11/02/2018
(E) 11/04/2018
A: Let's think step by step.
If tomorrow is 11/12/2019, then today is 11/11/2019. The date one year ago from today is
11/11/2018. So the answer is (B)."

Source: Suzgun et al. 2022

## Slide 27 — Orca: Instruction-tuning small LMs with CoT Rationales

Same opening line as slide 24. A boxed sub-heading **"Geometric Shapes"**, below it a worked
example reproduced verbatim:

'Q: This SVG path element `<path d="M 14.19,26.04 L 51.43,39.21 L 58.44,36.69 L 56.63,30.17 L
48.53,26.66 L 14.19,26.04"/>` draws a
Options:
(A) circle (B) heptagon (C) hexagon (D) kite (E) line (F) octagon (G) pentagon (H) rectangle (I)
sector (J) triangle
A: Let's think step by step.
This SVG path element contains "M" and "L" commands. M takes two parameters (x,y) and moves the
current point to the coordinates (x,y). L takes two parameters (x,y) and draws a line from the
previous coordinate to the new coordinate (x,y). This path can be decomposed into 6 separate
commands.
(1) M 14.19,26.04: Move the current point to 14.19,26.04.
(2) L 51.43,39.21: Create a line from 14.19,26.04 to 51.43,39.21.
(3) L 58.44,36.69: Create a line from 51.43,39.21 to 58.44,36.69.
(4) L 56.63,30.17: Create a line from 58.44,36.69 to 56.63,30.17.
(5) L 48.53,26.66: Create a line from 56.63,30.17 to 48.53,26.66.
(6) L 14.19,26.04: Create a line from 48.53,26.66 to 14.19,26.04.
This SVG path starts at point 14.19,26.04, creates five consecutive and touching lines, and then
returns back its starting point, thereby creating a five-sided shape. It does not have any curves
or arches. "pentagon" is the only five-sided polygon on the list. So the answer is (G).'

Source: Suzgun et al. 2022

## Slide 28 — Orca: Instruction-tuning small LMs with CoT Rationales

A results table comparing four systems across the 19 BigBench-hard tasks named on slides 25–27
(Orca-13B cells also show a percentage relative improvement over Vicuna-13B in parentheses):

| Task | ChatGPT | GPT-4 | Vicuna-13B | Orca-13B |
|---|---|---|---|---|
| Boolean Expressions | 82.8 | 77.6 | 40.8 | **72.0** (76.5%) |
| Causal Judgement | 57.2 | 59.9 | 42.2 | **59.9** (41.8%) |
| Date Understanding | 42.8 | 74.8 | 10.0 | **50.0** (400.0%) |
| Disambiguation QA | 57.2 | 69.2 | 18.4 | **63.6** (245.7%) |
| Formal Fallacies | 53.6 | 64.4 | 47.2 | **56.0** (18.6%) |
| Geometric Shapes | 25.6 | 40.8 | 3.6 | **20.8** (477.8%) |
| Hyperbaton | 69.2 | 62.8 | 44.0 | **64.0** (45.5%) |
| Logical Deduction (5 objects) | 38.8 | 66.8 | 4.8 | **39.6** (725.0%) |
| Logical Deduction (7 objects) | 39.6 | 66.0 | 1.2 | **36.0** (2900.0%) |
| Logical Deduction (3 objects) | 60.4 | 94.0 | 16.8 | **57.6** (242.9%) |
| Movie Recommendation | 55.4 | 79.5 | 43.4 | **78.3** (80.6%) |
| Navigate | 55.6 | 68.8 | 46.4 | **57.6** (24.1%) |
| Penguins in a Table | 45.9 | 76.7 | 15.1 | **42.5** (181.8%) |
| Reasoning about Colored Objects | 47.6 | 84.8 | 12.0 | **48.4** (303.3%) |
| Ruin Names | 56.0 | 89.1 | 15.7 | **39.5** (151.2%) |
| Salient Translation Error Detection | 40.8 | 62.4 | 2.0 | **40.8** (1940.0%) |
| Snarks | 59.0 | 87.6 | 28.1 | **62.4** (122.0%) |
| Sports Understanding | 79.6 | 84.4 | 48.4 | **67.2** (38.8%) |
| Temporal Sequences | 35.6 | 98.0 | 16.0 | **72.0** (350.0%) |
| Tracking Shuffled Objects (5 objects) | 18.4 | 25.2 | 9.2 | **15.6** (69.6%) |
| Tracking Shuffled Objects (7 objects) | 15.2 | 25.2 | 5.6 | **14.0** (150.0%) |
| Tracking Shuffled Objects (3 objects) | 31.6 | 42.4 | 23.2 | **34.8** (50.0%) |
| Web of Lies | 56.0 | 49.6 | 41.2 | **51.2** (24.3%) |
| **Average** | 48.9 | 67.4 | 23.3 | **49.7** (113.7%) |

Right-hand bullets:
- Outperforms Vicuna-13B
- Outperforms ChatGPT!
- GPT-4 has potential data contamination issues with Bigbench-hard (in red)

Source: Mukherjee et al. 2023

## Slide 29 — Reasoning by Finetuning LMs on their own outputs?

"ReSTᴱᴹ alternates between the following two steps:
1. `Generate` **(E-Step)**: Given reasoning problem, sample multiple solutions from language
   model. Filter based on some (problem specific) function [answer correctness for math
   problems] (the bracketed phrase in purple)
2. `Improve` **(M-Step)**: Update the language model to maximize probability of filtered
   solutions, using supervised finetuning"

Two scatter charts share the x-axis **Num iterations** (0, 1, 2, 3). Four series in each,
per the shared legend: **Palm-2-S** (solid blue dot), **Palm-2-L** (solid orange dot),
**Palm-2-L-SFT** (orange dotted horizontal reference line), **Palm-2-S-SFT** (blue dotted
horizontal reference line).

- Left, **Hendrycks MATH** (y-axis **Pass@1 Test Accuracy (%)**, ≈15–42):
  - Palm-2-S: rises from ≈17 at iteration 0 to ≈20.5 (iter 1), ≈22.5 (iter 2), ≈22 (iter 3).
  - Palm-2-L: rises from ≈35.5 at iteration 0 to ≈39 (iter 1), ≈41 (iter 2), ≈42 (iter 3).
  - Palm-2-S-SFT: flat dotted reference line at ≈18.
  - Palm-2-L-SFT: flat dotted reference line at ≈36.
- Right, **Transfer to GSM8K** (y-axis **Pass@1 Test Accuracy (%)**, ≈50–90):
  - Palm-2-S: rises from ≈53 at iteration 0 to ≈59 (iter 1), ≈59 (iter 2), ≈60 (iter 3).
  - Palm-2-L: rises from ≈80 at iteration 0 to ≈83 (iter 1), ≈86 (iter 2), ≈84 (iter 3).
  - Palm-2-S-SFT: flat dotted reference line at ≈53.
  - Palm-2-L-SFT: flat dotted reference line at ≈83.

Source: Singh et al. 2024

## Slide 30 — Reasoning by Finetuning LMs on their own outputs?

Same opening bullets as slide 29. Below, a bar chart, **"Hendrycks MATH (Test)"**, y-axis
**Pass@1 Performance (%)** (≈34–42), x-axis **Method (Num questions)**. Four bars, one data
series (method compared), colours per bar:
- **SFT (7K)** (blue) ≈ 36.2
- **SFT (5K)** (orange) ≈ 35.6
- **ReST\* (5K)** (green) ≈ 38.8
- **ReSTᴱᴹ (5K)** (red) ≈ 42.0

Source: Singh et al. 2024

## Slide 31 — Can Language Models Reason?

A collage of four reproduced news-headline clippings:
- *The Economist* (Science and technology | Generative AI): "Large language models' ability to
  generate text also lets them plan and reason"
- *TechTalks*: "Large language models have a reasoning problem"
- *The New York Times*: "Microsoft Says New A.I. Shows Signs of Human Reasoning"
- *WIRED*: "Some Glimpse AGI in ChatGPT. Others Call It a Mirage"

Two large red question marks flank the New York Times headline. Below: "Let's look at some more
careful evaluation to see if reasoning in LMs is **systematic**" (the word "systematic" in
purple).

Source: <https://aiguide.substack.com/p/can-large-language-models-reason>

## Slide 32 — Can Language Models Reason? — CoT Rationales are often not faithful

Two groups of four small line charts (eight data series each, legend shared across all panels:
**AQuA** navy/dark-purple, **TruthfulQA** light blue, **MMLU** teal, **OpenBookQA** dark green,
**ARC (Challenge)** olive/dark-yellow, **LogiQA** light yellow/tan, **ARC (Easy)** pink/magenta,
**HellaSwag** maroon).

- Left group, four panels labelled **3-Step CoTs**, **4-Step CoTs**, **5-Step CoTs**, **6-Step
  CoTs**: y-axis **% Same Answer as Complete CoT** (0–100), x-axis **% of Reasoning Sample
  Provided** (0, 25, 50, 75, 100). In every panel, all eight series rise from left to right and
  converge to 100% at x=100. Seven of the eight series (TruthfulQA, MMLU, OpenBookQA, ARC
  (Challenge), LogiQA, ARC (Easy), HellaSwag) start relatively high, roughly 65–95% at x=0, and
  rise gently; **AQuA is the clear outlier, starting much lower (roughly 35–38%) and staying
  well below the rest of the pack until a late, steep rise between x=50 and x=100.** Caption
  below: "Models do not always need the full rationale to answer correctly → rationale may be
  post-hoc?"
- Right group, the same four step-counts: y-axis **% Same Answer as Original** (0–100), x-axis
  **% of Reasoning Sample Before Mistake** (0, 20, 40, 60, 80). Same eight series and colours;
  again seven series sit roughly 65–95% throughout, rising only slightly, while **AQuA (navy) is
  again the clear outlier, sitting lowest throughout at roughly 40–60% and never catching up to
  the other seven by x=80.** Caption below: "Sometimes, models answer correctly even with an
  incorrect rationale.."

*(At this resolution the eight overlapping series in each of the eight small panels cannot be
read out to precise per-point values; the description above gives the reliable qualitative
pattern — AQuA well below the pack, the other seven bunched together and rising toward 100% —
rather than per-series numbers.)*

Source: Lanham et al. 2023

## Slide 33 — Can Language Models Reason? — Reasoning vs Memorization: Using Counterfactuals

Header **"GPT-4 Performance"** beside two small preview bar-charts, **Arithmetic** and
**Logic**, each showing two bars (blue, taller, and orange, shorter) on a 0–100 axis with a
dotted **"random"** reference mark near the low end for Arithmetic; these are miniature previews
of the fuller results charted on the next slide, illustrating that accuracy is high on one
condition (blue) and lower on another (orange). Below the Arithmetic mini-chart, an example
question box: **"27+62"**. Below the Logic mini-chart, an example template box: **"If X are Y, Y
are Z. Are X Z?"**

Below that, two rows illustrate the counterfactual-evaluation method, each with a globe-and-sun
icon:
- **Default** row (blue globe icon, sun in its normal position): the Arithmetic example is
  worked **"in base-10"**, giving the answer **89** (in blue); the Logic example instantiates
  "X = corgi, Y = mammals, Z = animals" and answers **"Yes"** (in blue).
- **Counterfactual** row (orange globe icon, sun repositioned to signal an altered premise): the
  same Arithmetic example is now worked **"in base-9"**, giving the answer **100** (in orange);
  the same Logic template now instantiates "X = corgi, Y = reptiles, Z = plants" and still
  answers **"Yes"** (in orange).

This illustrates the paper's method: pairing each standard ("default") question with a
counterfactual variant that keeps the same reasoning structure but changes a premise the model
is likely to have memorized (the usual base-10 arithmetic; the usual mammal/animal category
relation), to test whether performance is driven by genuine reasoning or memorized defaults.

Source: Wu et al., 2024

## Slide 34 — Can Language Models Reason? — Reasoning vs Memorization: Using Counterfactuals

Four column panels, one per model — **GPT-4**, **GPT-3.5**, **Claude**, **PaLM-2** — each with
two rows of bar charts, **Arithmetic** (subtitled "Two-digit addition") on top and **Logic**
(subtitled "First-order logic deduction in natural language") below. Shared legend: **w/o 0-CoT**
(unfilled/outline bars), **w/ 0-CoT** (filled bars), **CCC** (small arrow markers above bars),
**Random** (dotted horizontal reference line).

- **Arithmetic rows** (y-axis **Accuracy (%)**, 0–100; x-axis **Base**, categories 8, 9, 10, 11,
  16, shown once for "w/o 0-CoT" and again for "w/ 0-CoT"): in every model panel, the bar for
  base 10 (the real-world base, shown in blue) is the tallest, close to 100%, while the bars for
  the counterfactual bases 8, 9, 11, 16 (shown in orange) are markedly lower; adding zero-shot
  CoT ("w/ 0-CoT") raises the non-base-10 bars somewhat but base 10 still dominates. GPT-4 and
  PaLM-2 show the counterfactual-base bars recovering more under 0-CoT than GPT-3.5 and Claude,
  whose non-base-10 bars stay low in both conditions.
- **Logic rows** (y-axis **Accuracy (%)**, 0–100; x-axis **Follow Common Sense?**, categories Y
  and N, shown once for "w/o 0-CoT" and again for "w/ 0-CoT"): a dotted **Random** line sits
  around 30%. In each model panel the "Y" bars (premises that follow common sense/real-world
  category structure) sit clearly above the "N" bars (counterfactual premises that contradict
  common sense) in both conditions; GPT-4 and PaLM-2 show the highest overall Logic accuracy
  (roughly 80–100% on Y), while Claude's Logic accuracy is markedly lower across the board (its
  "w/o 0-CoT" bars sit close to or even below the Random line).

*(With four models × two conditions × up to five categories per panel, this is a dense grid of
small bars; the description above captures the reliable qualitative pattern — base-10 and
common-sense-following premises score highest, 0-CoT helps the counterfactual conditions
partially recover — rather than claiming exact bar heights, which are not reliably legible at
this resolution.)*

Source: Wu et al., 2024

## Slide 35 — Can Language Models Reason? — Reasoning vs Memorization: Counterfactuals for Analogical Reasoning

Two stacked tables of letter-sequence transformation examples, each with six categories arranged
in a 2×3 grid: **Extend sequence**, **Successor**, **Predecessor**, **Remove redundant letter**,
**Fix alphabetic sequence**, **Sort**. Each category shows a first example in black establishing
the rule, then a second example whose answer (shown in blue/teal) is the one being tested.

**Original transformation types:**
- Extend sequence: `abcd → abcde`; `ijkl → ijklm`
- Successor: `abcd → abce`; `ijkl → ijkm`
- Predecessor: `bcde → acde`; `ijkl → hjkl`
- Remove redundant letter: `abbcde → abcde`; `ijkklm → ijklm`
- Fix alphabetic sequence: `abcwe → abcde`; `ijkxm → ijklm`
- Sort: `adcbe → abcde`; `kjmli → ijklm`

**Modified transformation types** (same six categories, letters shifted to test whether the
model is pattern-matching memorized letter sequences rather than applying the rule):
- Extend sequence: `abcd → abcdf`; `ijkl → ijkln`
- Successor: `abcd → abcf`; `ijkl → ijkn`
- Predecessor: `cdef → adef`; `jklm → hklm`
- Remove redundant letter: `acegii → acegi`; `ikkmoq → ikmoq`
- Fix alphabetic sequence: `acego → acegi`; `ikxoq → ikmoq`
- Sort: `kfapu → afkpu`; `imkoq → ikmoq`

Source: Hodel et al. 2024

## Slide 36 — Can Language Models Reason? — Reasoning vs Memorization: Counterfactuals for Analogical Reasoning

Same **"Original transformation types"** table as slide 35, unchanged. Below it, a third table,
**"Modified transformation types with synthetic alphabet"**, using a shuffled 26-letter synthetic
alphabet in place of a–z — a full permutation of the alphabet, not a subset:
**"x y l k w b f z t n j r q a h v g m u o p d i c s e"**. The same six
categories are re-run using letters drawn from this synthetic ordering (again, second example's
answer in blue/teal):
- Extend sequence: `xylk → xylkb`; `tnjr → tnjra`
- Successor: `xylk → xylb`; `tnjr → tnja`
- Predecessor: `lkwb → xkwb`; `njrq → zjrq`
- Remove redundant letter: `xlwwft → xlwft`; `ttjqhg → tjqhg`
- Fix alphabetic sequence: `xlwrt → xlwft`; `tjphg → tjqhg`
- Sort: `xlfwt → xlwft`; `jtqhg → tjqhg`

This tests whether performance depends on the model's memorized a–z ordering rather than the
abstract transformation rule.

Source: Hodel et al. 2024

## Slide 37 — Can Language Models Reason? — Reasoning vs Memorization: Counterfactuals for Analogical Reasoning

Two bar charts, both with y-axis **Generative accuracy** (0–1) and x-axis **Transformation type**
(Extend sequence, Successor, Predecessor, Remove redundant letter, Fix alphabetic sequence,
Sort), each bar with a vertical error-bar line. Three data series in the left chart's legend:
**Original** (blue/lavender), **Interval** (green), **Interval & synthetic alphabet** (orange).

- Left panel (evaluating GPT-4, per the caption below it):
  - Original: Extend sequence ≈0.96, Successor ≈0.94, Predecessor ≈0.78, Remove redundant letter
    ≈0.86, Fix alphabetic sequence ≈0.52, Sort ≈0.22.
  - Interval: Extend sequence ≈0.32, Successor ≈0.60, Predecessor ≈0.16, Remove redundant letter
    ≈0.78, Fix alphabetic sequence ≈0.26, Sort ≈0.08.
  - Interval & synthetic alphabet: Extend sequence ≈0.02, Successor ≈0.06, Predecessor ≈0.02,
    Remove redundant letter ≈0.76, Fix alphabetic sequence ≈0.02, Sort ≈0.14.
  - Caption: "Significant drop in performance for GPT-4 → evidence of spurious reasoning?"
- Right panel (evaluating humans, per the caption below it), same three series:
  - Original: Extend sequence ≈0.85, Successor ≈0.86, Predecessor ≈0.82, Remove redundant letter
    ≈0.87, Fix alphabetic sequence ≈0.42, Sort ≈0.36.
  - Interval: Extend sequence ≈0.68, Successor ≈0.64, Predecessor ≈0.70, Remove redundant letter
    ≈0.82, Fix alphabetic sequence ≈0.21, Sort ≈0.26.
  - Interval & synthetic alphabet: Extend sequence ≈0.78, Successor ≈0.74, Predecessor ≈0.79,
    Remove redundant letter ≈0.86, Fix alphabetic sequence ≈0.31, Sort ≈0.29.
  - Caption: "No drop in performance for humans"

The contrast between panels is the point: GPT-4's accuracy collapses on "Extend sequence,"
"Successor," and "Predecessor" once the task is shifted away from the memorized a–z alphabet
(Interval, then Interval & synthetic alphabet), while humans show comparatively little drop
across the same manipulations, on the same six transformation types.

Source: Hodel et al. 2024

## Slide 38 — Section title: Language Model Agents

Subtitle: "with some slides borrowed from Frank Xu (CMU)"

## Slide 39 — Some Terminology

Two labelled icons, not yet connected: on the left, a globe icon labelled **Environment** (in
purple); on the right, a small robot icon next to a boxed formula **π(·|g)**.

## Slide 40 — Some Terminology

Same globe (**Environment**) and robot (**π(·|g)**) icons as slide 39, now joined into a loop by
two curved arrows: an **Action** arrow curving from the robot to the environment (top), and an
**Observation** arrow curving from the environment back to the robot (bottom) — the standard
agent–environment interaction loop, with the robot's policy π conditioned on a goal g.

## Slide 41 — Some Terminology

Same Environment/robot loop diagram as slide 40. To the right of the robot, three phrases stacked
vertically: '"Instruction-following agent"', '"Language conditioned policy"', '"digital agent"'.
Below the robot, a purple arrow points up into it from a new icon: a person standing at a
whiteboard with a checkmark, facing a small group of three people, labelled **"Language
Instruction."**

## Slide 42 — Some Terminology

Same loop diagram, robot, and "Language Instruction" icon/arrow as slide 41. Above the loop, a
line of placeholder actions in monospace: `Type … on …, Click on …, Choose … from dropdown, …`.
Beside the Language Instruction icon, a purple quote: **"Book a flight from San Francisco to New
York."** Below the diagram, two screenshots side by side:
- Left, an Expedia flight-search results page (flight listings, prices, "Choose Your Departing
  Flight" panel), captioned **"Raw pixels as observation?"**
- Right, a snippet of HTML/DOM source from `webarena.onestopshop.com` (`<li>`, `<div>`, `<a
  href="...">`, a "Rating:" span showing "82%", a "12 Reviews" link), captioned **"HTML DOM as
  observation?"**

## Slide 43 — Applications: Natural Language Interfaces

Left, a row of virtual-assistant logos: **Cortana**, the four-colored **Google Assistant** dot
logo, **"Hey Siri,"** and **alexa**. Below them, a pink box, **"Virtual Assistants,"** with three
example commands (each preceded by a small person/speech-bubble icon): *"Set an alarm at 7 AM,"*
*"Remind me for the meeting at 5pm,"* *"Play Jay Chou's latest album."*

Right, a screenshot of a code editor window (title bar "Untitled-1," a Python file tab, line
numbers 1–5, line 3 highlighted green, status bar reading "master*," "Python 3.6.5 64-bit").
Below it, a second pink box, **"Natural Language Programming,"** with three example commands:
*"Sort my_list in descending order,"* *"Copy my_file to home folder,"* *"Dump my_dict as a csv
file output.csv."*

## Slide 44 — Applications: UI automation

Left, a yellow highlighted instruction: **"Click the 'Menu' button, and then find and click on
the item with the ▶| icon."** Below it, a small mock application menu: a **Save** item, a
highlighted **Playback ▶** item expanding into a submenu (**Prev**, **Stop**, **Play**, **Next**,
the last marked with a ▶| icon), a greyed-out **Print...** item, **Zoom In**, and **Zoom Out**,
sitting above a **Menu** button — illustrating the agent needing to open the Menu, then the
Playback submenu, then click the icon matching "▶|".

Right, a Spotify-style screenshot (sidebar with Home/Search/Your Library/Create Playlist/Liked
Songs, a "BORN PINK" album banner, a "Good afternoon" row of recently-played tiles, a "Great
first audiobooks" row), captioned below: **"Play some synthwave songs."**

## Slide 45 — Applications: Multi-step "Tool use"

A screenshot of the OpenAI "ChatGPT plugins" announcement page (heading **"ChatGPT plugins"**,
partly overlapped by the plugin grid, reading "We've implemented initial support for plugins in
ChatGPT. Plugin[s] ... language model[s] ... help ChatGPT a[ccess] ... computations, o[r] ..."),
overlaid with a grid of plugin cards, each with an icon, name and one-line description: **Expedia**
("Bring your trip plans to life—get there, stay there, find things to see and do"), **FiscalNote**
("Provides and enables access to select market-leading, real-time data sets for legal, political,
and regulatory data and information"), **Instacart** ("Order from your favorite local grocery
stores"), **KAYAK** ("Search for flights, stays and rental cars. Get recommendations for all the
places you can go within your budget"), **Klarna Shopping** ("Search and compare prices from
thousands of online shops"), **Milo Family AI** ("Giving parents superpowers to turn the manic to
magic, 20 minutes each day. Ask: Hey Milo, what's magic today?"), **OpenTable** ("Provides
restaurant recommendations, with a direct link to book"), **Shop** ("Search for millions of
products from the world's greatest brands"), **Speak** ("Learn how to say anything in another
language with Speak, your AI-powered language tutor"), **Wolfram** ("Access computation, math,
curated knowledge & real-time data through Wolfram|Alpha and Wolfram Language"), **Zapier**
("Interact with over 5,000+ apps like Google Sheets, Trello, Gmail, HubSpot, Salesforce, and
more"). Below, a link: "ChatGPT plugins."

## Slide 46 — Instruction following agents [Pre LLMs]

Same Environment/robot/π(·|g) loop diagram as earlier slides, in plain black icons (no logos).
Right, three worked semantic-parsing examples reproduced from the source paper:

a) What states border Texas
$$\lambda x. state(x) \land borders(x, texas)$$

b) What is the largest state
$$\arg\max(\lambda x. state(x), \lambda x. size(x))$$

c) What states border the state that borders the most states
$$\lambda x. state(x) \land borders\left(x, \arg\max(\lambda y. state(y), \lambda y. count(\lambda z. state(z) \land borders(y,z)))\right)$$

Below: "Idea #1: Directly map from instructions to action sequences like Machine Translation
[works well for simple grounded environments like text2sql, knowledge graph querying]"

$$\max_\theta p_\theta(\{a_1, a_2, \ldots\} \mid g)$$

Source: Zettlemoyer et al. 2012

## Slide 47 — Instruction following agents [Pre LLMs]

Same loop diagram as slide 46. Right, a reproduced system diagram, **"Learning system for parsing
navigation instructions"**: during **Training**, Observation (**World State**, **Action Trace**,
**Instruction**) feeds a **Navigation Plan Constructor**, which feeds **Plan Refinement**, which
feeds a **Semantic Parser Learner**; during **Testing**, **Instruction**, **World State**, and
**Action Trace** feed a **Semantic Parser**, which feeds an **Execution Module (MARCO)**.

Below, a worked example:

**Instruction:** "Place your back against the wall of the 'T' intersection. Turn left. Go forward
along the pink-flowered carpet hall two segments to the intersection with the brick hall. This
intersection contains a hatrack. Turn left. Go forward three segments to an intersection with a
bare concrete hall, passing a lamp. This is Position 5."

**Parse:** `Turn(), Verify(back: WALL), Turn(LEFT), Travel(), Verify(side: BRICK HALLWAY),
Turn(LEFT), Travel(steps: 3), Verify(side: CONCRETE HALLWAY)`

Below the diagram: "Idea #2: Infer executable, structured plans from (instruction, trajectory)
pairs and train a model to go from instructions to plans" (the word "executable" in purple).

Source: Chen and Mooney 2011

## Slide 48 — Instruction following agents [Pre LLMs]

Three stacked rows, each showing a growing action trace for the same instruction, alongside a
matching Windows screenshot (labelled $\mathcal{E}$):

- Row 1 — **u**: "click Run, and press OK after typing secpol.msc in the open box." **ā**:
  `left-click [Run...]` (with "left-click" labelled **C** and "[Run...]" labelled **R**).
  $\mathcal{E}$: a Start-menu screenshot with "Run..." highlighted.
- Row 2 — same **u**. **ā**: `left-click [Run...]` then `type-into [open "secpol.msc"]` (the new
  segment labelled **C**: type-into, **R**: open "secpol.msc"). $\mathcal{E}$: the "Run" dialog
  with "secpol.msc" typed into the Open field.
- Row 3 — same **u**. **ā**: `left-click [Run...]`, `type-into [open "secpol.msc"]`, then
  `left-click [OK]` (the new segment labelled **C**: left-click, **R**: OK).
  $\mathcal{E}$: the same Run dialog with the OK button highlighted.

Below: "Idea #3: Use RL to directly map instructions to actions"

$$\max_\theta \mathbb{E}_{a \sim \pi_\theta} R(a; \text{instruction}, \text{observation})$$

Source: Branavan et al. 2009

## Slide 49 — Instruction following agents [in 2024]

Same Environment/robot loop diagram, but the robot icon is now replaced by a stack of three
modern LLM logos: the OpenAI (ChatGPT) mark, the **Gemini** wordmark, and a blue circular
llama/alpaca-head logo (representing an open LLM such as Vicuna/Llama). The "Language
Instruction" person icon still points an arrow up into this stack, and the boxed **π(·|g)**
label remains beside it.

Below:

$$p(\tau \mid g) = p(s_1, a_1, s_2, a_2, \ldots \mid g) = \prod_t p(s_t \mid s_{t-1}, a_t) \times \pi(a_t \mid \tau_{\le t}, g)$$

## Slide 50 — Instruction following agents [in 2024]

Same diagram and equation as slide 49, now with two upward labels added beneath the equation: an
arrow from the transition term $p(s_t \mid s_{t-1}, a_t)$ to **"Transition dynamics,"** and an
arrow from the policy term $\pi(a_t \mid \tau_{\le t}, g)$ to **"Agent policy."**

## Slide 51 — Instruction following agents [in 2024]

Same diagram and labelled equation as slide 50 (the "Agent policy" label now reads "Agent policy
(with a transformer!)"). Right, "**Main Idea: Generative trajectory modeling with causal
transformers!**" above a reproduced architecture diagram: a sequence of tokens — return
$\hat{R}_{t-1}$, state $s_{t-1}$, action $a_{t-1}$, return $\hat{R}_t$, state $s_t$, action $a_t$
— each embedded (with positional encoding) and fed into a **causal transformer**, which feeds a
**linear decoder** that predicts the next actions $a_{t-1}$ and $a_t$. Below the token sequence,
three small icons illustrate a concrete return/state/action triple: **21** (return), a small
brown block (state), and a grey plus/cross icon (action) — a Decision-Transformer-style setup.

Source: Chen et al. 2021

## Slide 52 — A Simple Language Model Agent with ReACT

A boxed prompt template, reproduced with its original colour-coding:

```
You are an agent capable of the following actions:
1. Type X on Y
2. Move mouse to
3. Click on X
4. Type Char x on Y

Your objective is to follow user instructions, by mapping them into a sequence of
actions.
Instruction: {g}

So far, you have taken the following actions and observed the following environment
states:

Previous Actions and Observations:
o1:
a1:

o2:
a2:
…

After executing these actions, you observe the following HTML state: <HTML state>

Now, think about your next action:
Thought: [model-pred]

Now, take an action:
Action: [model-pred]
```

In the original, "1.–4." (the action list) is teal, "Your objective ... Instruction: {g}" is
orange, and everything from "So far, you have taken..." onward is red. Below the box:
$\pi_{\text{LM}}(\cdot \mid \tau_{\le t}, g)$.

Right, a numbered list matching those colours: 1. **Action space in text** (teal) 2.
**Instruction in text** (orange) 3. **Previous observations and actions** (red) 4. **Provide
current observation [as text]** (red). Below: "Model generates next action (sequence prediction
task), use that action to update environment and repeat!" and "Mostly, just CoT prompting in a
loop."

Source: Yao et al. 2023

## Slide 53 — Some popular benchmarks for LM agents: MiniWoB++

A grid of roughly 18 small browser-task thumbnails, each with a yellow instruction bar above a
miniature interactive widget. Legible examples include: "Move the cube around so that '5' is the
active side facing the user" (a 3-D cube widget); "Set the sliders to the combination [13,20,13]
and submit" (three sliders); "Draw the number '2' in the checkboxes using the example on the
right and press Submit when finished" (a checkbox grid); "Drag Ree to the 4th position" (a
draggable name list); "Keep your mouse inside the circle as it moves around"; "Enter the value of
Country into the text field and press Submit" (a form with Gender/First name/Country/Year of
Birth/Religion fields); "Drag all triangles into the black box"; "Select 09/23/2016 as the date
and hit submit" (a calendar widget); "Sort the numbers in increasing order, starting with the
lowest number at the top of the list"; "Copy the text from the 1st text area below and paste it
into the text input"; "Select all the shades of blue and press Submit" (a colour-swatch grid);
"Find the 4th word in the paragraph, type that into the textbox and press 'Submit'"; "Click the
button in the dialog box labeled 'Cancel'"; "Highlight the text in the paragraph below and click
submit"; "Find the 11th word in the paragraph, type that into the textbox and press 'Submit'";
"Move the cube around so that '2' is the active side facing the user"; "Drag the smaller box so
that it is completely inside the larger box." *(A couple of the smaller thumbnails are not
legible at this resolution.)*

Right, bullets: "Sandboxed environment evaluating basic browser interactions across a range of
applications from social media to email clients"; "Evaluates functional correctness"; "Not real
world (limited functionality)" (red); "Relatively short-horizon" (red); "**Zero-shot performance
far from perfect!**" (teal, bold).

Source: Shi et al. 2017

## Slide 54 — Some popular benchmarks for LM agents: WebArena

A screenshot of three side-by-side browser panels — a Wikipedia-style page ("List of museums in
Pittsburgh"), an OpenStreetMap-style directions page ("Travel in Northeast US," a route from
Schenley Park), and a GitLab-style repo page ("Update README.md") — annotated with a speech
bubble from a person icon: **"Create a plan to visit Pittsburgh's art museums with minimal
driving distance starting from Schenley Park. Log the order in my 'awesome-northeast-us-travel'
repository."** Below the three panels, three small captions with robot icons: "Search for museums
in Pittsburgh," "Search for each art museum on the Map," "Record the optimized results to the
repo" — illustrating the three sub-tasks the agent must chain together across the three sites.

Right, bullets: "Environment with sandboxed approximations of real websites spanning e-commerce,
social media!"; "Additional utility tools: Maps, calculators, scratchpads, Wikipedia…"; "Multi-tab
browsing"; "Long-horizon tasks"; "Evaluates functional correctness."

Source: Zhou et al. 2024

## Slide 55 — Some popular benchmarks for LM agents: WebLINX

A reproduced conversational trace, alternating a person icon's turns and the agent's actions and
screenshots: person: **"Create a task for a Career Fair on Google calendar"** → agent
`say("Sure!")` → agent `load("calendar.google.com")` → [Google Calendar screenshot] → agent
`click(<div>)` → [calendar screenshot with an event slot highlighted] → agent
`input(<input>, "Career Fair")` → [event-creation popup showing "Career Fair"] → agent
`say("Do you want to add any description?")` → person: **"Yes, please add 'Bring multiple copies
of my resume' as the note."** → agent `input(<div>, "Bring multiple copies of my resume")` →
[task form screenshot] → agent `click(<span>)` → [task form with the note saved] → agent
`say("Task created. Anything else I can assist you with?")` → person: **"No. That's all for
now."**

Right, bullets: "Web-interactions on real websites"; "Conversational: includes a new 'say' action
to communicate with human to gather information"; "Multi-tab browsing"; "Turn-level metrics for
evaluation"; "Not an environment, but a collection of interactions" (red).

Source: Lù et al. 2024

## Slide 56 — Training data for Language Model Agents

- Standard practice: In-context learning with few-shot demonstrations of humans performing
  following similar instructions.
- This is still not scalable / reliable

Below, four icons (a calculator, a database/SQL icon, a crossed-wrench "tools" icon) beside three
website screenshots: a Twitter-like social feed, a DoorDash-like "Browse" food-delivery app, and
an e-commerce "One Stop Market" storefront. Caption: "1000s of environments, many kinds of
interactions possible…" followed by "Can agents autonomously **explore** their environments to
construct **high quality synthetic demonstrations**?" (the two phrases in purple).

## Slide 57 — Use Exploration + Model Generated Data!

A robot icon labelled $\pi_e(\cdot)$ next to a speech bubble reading: "Prompt: Given a website,
take actions of the following format to explore…. Action: [[pred]]" — pointing at a "Book Your
One-Way Flight" form (From/To fields, Departure Date, Search button).

## Slide 58 — Use Exploration + Model Generated Data!

Same exploring robot $\pi_e(\cdot)$ and flight-search form as slide 57, now with three arrows
branching from the robot into three rows of resulting interaction traces (each row a sequence of
form/results screenshots showing the agent typing into fields, selecting airports, opening a
calendar, and reaching a flight-results screen, e.g. one trace ends on a results screen showing
an SFO→NYC flight for December 13, 2016, another ends with a booking price of "$241" for October
20, 2016). To the right, a second robot icon labelled $p_{\text{label}}(\cdot \mid \tau)$ next to
a speech bubble: "Prompt: You are given a sequence of actions and corresponding HTML states on a
website… Label: [[pred]]" Below: "How can we decide if a sequence of interactions is meaningful?
*Use Natural Language!*"

## Slide 59 — Use Exploration + Model Generated Data!

Same three-row diagram as slide 58. The first row of screenshots (the SFO→NYC trace ending in
flight results for 12/13/2016) is now highlighted with a green box, and the labelling robot's
speech bubble is filled in with a thought-bubble output: **"Book a flight from SFO to NYC."**

## Slide 60 — Use Exploration + Model Generated Data!

Same diagram again. Now the second (middle) row of screenshots — the calendar and
From/To-with-"twy" trace ending on Departure Date 12/26/2016 — is highlighted in green, and the
labelling robot's thought bubble reads: **"Set the date as 12/26/2016."**

## Slide 61 — Use Exploration + Model Generated Data!

A new diagram formalizes the idea from slides 57–60. A speech bubble reading "Prompt: Map the
given instruction to a sequence of actions, one at a time. Thought: [[pred]] Action: [[pred]]"
points down at a robot icon labelled $\pi_{\text{LM}}(\cdot \mid g)$. To its left is the blank
"Book Your One-Way Flight" form (From/To fields, Departure Date, Search button); an arrow runs
from the robot to two resulting screenshots: an open December-2016 calendar picker with a date
cell highlighted, and the same flight form now showing Departure Date filled in as "12/26/2016".
Below the robot, an oval reads "Instruction: Set the date as 12/26/2016"; to its right, a second
oval labelled "Trajectory". A brace spans both ovals and points down to $R(g, \tau)$, glossed by a
speech bubble: "Prompt: Output "1" if the trajectory is correct for the given instruction… Label:
[[pred]]" — introducing $R(g,\tau)$ as an LM-based reward/verifier that judges whether a rolled-out
trajectory $\tau$ actually satisfies instruction $g$.

## Slide 62 — Use Exploration + Model Generated Data!

Same diagram as slide 61, now resolved: next to $R(g, \tau)$ is a green checkmark, and below it a
green cylindrical database icon appears. This shows the outcome when the reward model judges the
trajectory correct: it gets written into a growing database of synthetic demonstrations.

## Slide 63 — Use Exploration + Model Generated Data!

A new example instruction, "Book a flight from SFO to NYC," drives the same pipeline.
$\pi_{\text{LM}}(\cdot \mid g)$ acts on the blank "Book Your One-Way Flight" form, producing two
resulting screenshots: one with "sfo" typed into the From field and an autocomplete suggestion
"San Francisco, CA (SFO)" below it, and one with the From field filled as "San Francisco, CA ("
and the To field filled as "New York, NY - All a[irports]". To the right, a second robot icon
labelled $p_{\text{label}}(\cdot \mid \tau)$ has a thought bubble reading "Set origin to SFO and
dest to NYC" — its own guess at what instruction the trajectory accomplishes. Below, the brace over
the instruction/trajectory ovals points to $R(g, \tau)$ marked with a red X: the reward model judges
this trajectory as not matching the original instruction (the trajectory only sets origin and
destination; it never finishes the booking).

## Slide 64 — Use Exploration + Model Generated Data!

Same scene as slide 63, with one addition: a bracket-shaped arrow now runs from near the
$p_{\text{label}}$ robot's thought bubble, left across the top of the slide, and down onto the
$\pi_{\text{LM}}$ robot. This depicts the re-labeling step: the labelling robot's inferred
sub-instruction ("Set origin to SFO and dest to NYC") is fed back in as a new, corrected goal
paired with the same trajectory — turning a mismatched rollout into a valid demonstration for a
different (relabeled) instruction, instead of discarding it.

## Slide 65 — BAGEL: Use Exploration + Model Generated Data!

The title now names the method: **BAGEL**. Diagram: a speech bubble "Prompt: Map the given
instruction to a sequence of actions, one at a time. Thought: [[pred]] Action: [[pred]]" points to
a $\pi_{\text{LM}}(\cdot \mid g)$ robot icon; a double-dashed arrow (pointing both ways) connects it
to a $p_{\text{label}}(\cdot \mid \tau)$ robot icon, which is in turn pointed to by a second speech
bubble: "Prompt: You are given a sequence of actions and corresponding HTML states on a website…
Label: [[pred]]". Below both robots, a yellow oval reads "BAGEL". Caption beneath: "(Bootstrapping
Agents by Guiding Exploration with Language)".

## Slide 66 — BAGEL: Use Exploration + Model Generated Data!

A three-step pipeline diagram:
1. **"Explore Environment to collect trajectories"** — a rounded box labelled $p_0(\tau)$ sits above
   three explicit rows of trajectories, each row four gray boxes long connected by arrows; a row of
   vertical dots (⋮) sits between the second and third rows under each of the four columns,
   indicating that many more such trajectory rows lie between them (only a sample is drawn).
2. **"Create Synthetic demonstrations via iterative re-labeling"** — three trajectory groups,
   labelled $\tau^0$, $\tau^1$, … $\tau^T$ (each again a row of four gray boxes with arrows), feed
   into a shared $R(g, \tau)$ box that also draws from a green database-cylinder icon above it, via
   a brace spanning all three groups. Under each trajectory group sits a repeating pair of small
   rounded boxes — a green $p_{\text{label}}(\cdot \mid \tau)$ box and a blue
   $\pi_{\text{LM}}(\cdot \mid g)$ box — linked by two curved arrows forming a small cycle, with the
   refined goal at each round labelled $g^0$, $g^1$, … $g^T$ on a small tag beneath the connecting
   arrows: the labelling and acting robots iteratively refine the instruction attached to a
   trajectory.
3. **"Instruction-Following (Inference Time): Retrieve Relevant Demonstration via retrieval to use
   as in-context exemplars"** — given a new instruction, quoted as "Book the cheapest flight from
   Denver to LA," a database-plus-robot icon retrieves matching prior demonstrations and produces a
   new trajectory (four gray boxes in a row). Above this, two retrieved example demonstrations are
   shown mapping instruction to action sequence: "*Book the cheapest …*" → `{type on …, select …,
   click …}` and "Buy a flight from Denver …" → `{type …, click …, select …}`.

Bottom right, in red: "Finetuning possible too!" — noting the synthetic demonstrations can also be
used to finetune the policy directly, not only as retrieval exemplars.

## Slide 67 — BAGEL: Use Exploration + Model Generated Data!

Two side-by-side bar charts, both with a y-axis from 0–100 (gridlines at 0, 20, 40, 60, 80, 100),
each comparing exactly two data series per a shared legend: **Zero-Shot** (light teal-green bars)
and **+ BAGEL** (dark blue bars).

- **Left chart, "MiniWob++"** — x-axis categories (rotated labels): book-flight, choose-date,
  social-media, email-inbox-all, click-checkboxes-soft, click-tab-2-hard, social-media-some,
  tic-tac-toe, use-autocomplete, search-engine, Average. Only the Average pair carries printed
  numeric labels: **46.8** (Zero-Shot) and **60.5** (+BAGEL). The other ten category pairs are
  unlabelled; reading approximate heights off the gridlines: book-flight (~5 → ~15), choose-date
  (~20 → ~40), social-media (~60 → ~70), email-inbox-all (~88 → ~100, the tallest pair),
  click-checkboxes-soft (~85 → ~90), click-tab-2-hard (~75 → ~100), social-media-some (~75 → ~80),
  tic-tac-toe (~20 → ~40), use-autocomplete (~25 → ~45), search-engine (~20 → ~25). In every
  category +BAGEL is at or above Zero-Shot. **[These per-category numbers are visual estimates off
  an axis with no per-bar labels except Average, and could be off by several points at this
  resolution.]**
- **Right chart, "ToolQA"** — x-axis categories: Agenda, AirBnB, Coffee, DBLP, Flights, GSM8K,
  Scirex, Yelp, Average. Average is labelled **40.9** (Zero-Shot) and **43.3** (+BAGEL).
  Approximate unlabelled heights: Agenda (~30 → ~28, Zero-Shot slightly ahead), AirBnB (~78 → ~70,
  Zero-Shot ahead), Coffee (~83 → ~92), DBLP (~18 → ~22), Flights (~15 → ~33), GSM8K (~40 → ~38),
  Scirex (both bars at or near zero — no visibly rendered bar), Yelp (~65 → ~62, Zero-Shot ahead).
  Unlike MiniWoB++, +BAGEL is not uniformly ahead here — Agenda, AirBnB, and Yelp show Zero-Shot at
  or above +BAGEL. **[Same caveat: unlabelled bars are visual estimates.]**

Caption below both charts: "**13% point improvement** on MiniWoB++ and **2.5% improvement** on
multi-step tool use, using PALM-2 as the base language model, with no human supervision" (the two
improvement figures highlighted, one in purple, one in a rust/orange color).

## Slide 68 — Multimodality?

Three screenshots of realistic web UIs: a Twitter/X-style social feed (posts attributed to "Colin
Wright," "Cloudspace," "HuffPost Politics," "BI: Tech," "Bryce Roberts"), a "Browse" /
DoorDash-style food-delivery home screen (a search bar; category icons for Grocery, Convenience,
Alcohol, Offers, Pets, Packages, Snacks, Beauty, Flowers, Gifts, GreenMart, Drugstore, Apparel,
Pantry, Home Goods, Retail; and an "Explore Food Near You" row with Chicken/Pho tiles), and a "One
Stop Market" e-commerce storefront (product tiles with star ratings, prices, and "Add to Cart"
buttons). Below:
- So far, we've looked at using text-only language models for agents
- This is intractable for real-world UIs with very long HTML
- Can we instead operate directly over pixel space?

## Slide 69 — Multimodality: LLaVA

A reproduced architecture diagram: an image $\mathbf{X}_\text{v}$ passes through a **Vision
Encoder** and a **Projection $\mathbf{W}$**, producing embedding $\mathbf{Z}_\text{v}$ and visual
tokens $\mathbf{H}_\text{v}$ (drawn as white house-shaped icons); a language instruction
$\mathbf{X}_\text{q}$ yields token embeddings $\mathbf{H}_\text{q}$ (gray house-shaped icons); both
token sets feed into a shared **Language Model $f_\phi$** box, which outputs the **Language
Response $\mathbf{X}_\text{a}$** (green house-shaped icons).

Below: "Prompt GPT-4 to generate instructions and responses given textual descriptions of images"
and "Finetune a CLIP encoder jointly with a Vicuna-13B decoder on this data" (the phrase
"Vicuna-13B decoder" in purple).

Source: Liu et al. 2023

## Slide 70 — Multimodality: Pix2Struct

"Finetune a ViT encoder and a transformer decoder on a new HTML screenshot parsing task" (the task
name in purple).

A reproduced diagram: a row of small landscape/building photos is chopped into patches (each patch
drawn as a small colored square), passed through "Patch + Position Embedding" numbered 0*–9 (0*
marked with a footnote "Extra learnable [class] embedding") and a "Linear Projection of Flattened
Patches" box, which feeds a **Transformer Encoder** box; a separate **Transformer Decoder** box is
drawn to its right. Below, an input/output example: a screenshot of a "Programming Survey" form
(radio buttons for Python / C++ / Java, a Submit button) next to a second copy of the same
screenshot with the Java radio button and Submit button boxed/highlighted, mapping (via "→" through
a colon-separated pair) to a structured, HTML-like output:
```
<<<Python>
  <img_src=py_logo.png img_alt=Python>>
<<C++>
  <img_src=cpp_logo.png img_alt=C++>>
<<Java>
  <img_src=java_logo.png img_alt=Java>>
<<Submit>>
```

Source: Lee et al. 2023

## Slide 71 — LM Agents is an emerging application!

The same MiniWoB++ bar chart as slide 67's left panel (Zero-Shot vs. +BAGEL, identical categories,
identical Average values 46.8/60.5), shown alone with its legend now at top right — not a new
result, a restatement of the slide-67 chart. Caption below: "The 'prompting gap': without extensive
prompting / bespoke few-shot examples, competitive LMs are far from perfect on even the simplest
environments."

## Slide 72 — LM Agents is an emerging application!

Three small bar-chart panels side by side, titled **Easy**, **Medium**, and **Hard**, each with its
own y-axis "Success rate" from 0.00 to 1.00 (gridlines every 0.25) and each comparing the same three
bars: **InstructGPT-3+RLHF (Baseline)** (blue), **InstructGPT-3** (orange), and **GPT-3** (green).
- **Easy**: InstructGPT-3+RLHF(Baseline) ≈0.97, InstructGPT-3 ≈0.48, GPT-3 ≈0.22.
- **Medium**: InstructGPT-3+RLHF(Baseline) ≈0.68, InstructGPT-3 ≈0.38, GPT-3 has no visibly
  rendered bar (≈0).
- **Hard**: InstructGPT-3+RLHF(Baseline) ≈0.45, InstructGPT-3 ≈0.22, GPT-3 ≈0.08.

**[No panel carries printed numeric labels; the values above are visual estimates off the 0.25
gridlines.]**

Caption: "Long-horizon planning is hard: Even on simple benchmarks, performance drops drastically
on tasks that require longer horizon planning."

## Slide 73 — LM Agents is an emerging application!

A single bar chart — one data series, one bar per model, not a two-condition comparison — with
y-axis 0–80 (gridlines at 0, 20, 40, 60, 80) and the exact value printed inside each bar: **Human
78.2**, **GPT-3.5 6.4**, **GPT-3.5-CoT 8.8**, **GPT-4-CoT 11.7**, **GPT-4-Prompt Eng. 14.4**. An
italic annotation above the Human bar reads "(~2mins/task)" — a margin note on how long a human
took per task, not a second data series. An arrow at the right points at the chart with the text:
"Latest Work: BrowserGym 25% / More prompt engineering / More observation/action interface
engineering."

## Slide 74 — LM Agents is an emerging application!

Two side-by-side failure-case examples (no source citation is visible on the slide):
- **Left** — task text "S2: Open Google translate and sign in using the following credentials:
  [email] [password]" above a screenshot of a Google sign-in "Welcome" screen (account
  "webtasks.navigator@gmail.com", a password field boxed in purple, a "Show password" checkbox, a
  "Forgot password?" link, a "Next" button). Below it: "**Reference (B):** [password]" (printed in
  blue), "**GPT-4V (R):** [email]" (printed in red), "**LLaMA (B):** [password]" (printed in blue).
  The reference and LLaMA agree the highlighted field is the password field; GPT-4V answers "email"
  instead. **[The exact meaning of the "(B)"/"(R)" suffixes is not spelled out on the slide itself;
  read in context they appear to flag each answer as matching (B) or contradicting (R) the
  reference, consistent with the colors used.]**
- **Right** — a "Search" page screenshot (top nav with home/search icons, Log in, Sign up, and a
  menu icon; red banner) showing a "Search query" field containing "DMV area" and a red Search
  button; below, "50 results for *DMV area*:" with a highlighted accessibility-tree readout
  `[2430] searchbox 'Search query'` / `[5172] StaticText 'DMV area'`, and an arrow pointing back up
  from the results text to the search field; below that, a second "Search query" field now
  containing the corrupted, repeated text "DMV areaDMV areaDMV areaDMV area" followed by another
  Search button — illustrating an agent that keeps re-appending to the query box instead of
  clearing it first, compounding the same text on every step.

## Slide 75 — Lecture-14 Recap

- **Reasoning in Language Models:**
  - Via prompting
  - By distilling rationales from big LMs into small LMs
  - By finetuning LMs on their own rationales, iteratively
  - Counterfactual evaluation reveals reasoning may not be systematic
- **Language Model Agents:**
  - Prompting and in-context learning
  - BAGEL for synthetic demonstrations: exploration and iterative relabeling
  - Multimodality
  - Benchmarks still challenging

Boxed callout at bottom right: "💡Lots to be done to drive further improvements!"
