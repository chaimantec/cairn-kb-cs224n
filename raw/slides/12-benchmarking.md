---
title: Lecture 12 — Benchmarking and Evaluation (slide deck)
lecture: 12
slides: 65 printed / 65 pages in the PDF
source_pdf: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture11-evaluation-yann.pdf
note: |
  Lecturer is Yann Dubois. The deck's own title is "Lecture 11: Benchmarking and Evaluation";
  the Cairn catalog lists it at **position 12**, and repo files use the catalog position.
  Printed slide numbers match PDF page numbers 1:1, with no gaps and no offset. Three pages
  (1, 44, 63) print no number but occupy their position in the sequence, so slide N is page N
  throughout. Some slides are repurposed from Asli Celikyilmaz's EMNLP 2020 tutorial (noted
  on slide 16).
---

# Lecture 12 — Benchmarking and Evaluation: slide-by-slide

Text and figures of
[`cs224n-spr2024-lecture11-evaluation-yann.pdf`](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture11-evaluation-yann.pdf),
transcribed from the deck. Diagrams and plots are described in prose since the KB is
read as text.

Companion pages: [wiki page for this lecture](../../wiki/12-benchmarking.md) ·
[transcript](../transcripts/12-benchmarking.md)

## Contents

| Slides | Section |
| ------ | ------- |
| 1–2 | Title and lecture overview |
| 3–5 | Why we measure: different desiderata at each stage; benchmarks drive progress; the close-ended / open-ended split |
| 6–13 | §1 Close-ended evaluation: what close-ended tasks are, examples, SuperGLUE, choosing and aggregating metrics, spurious correlations |
| 14–17 | §2 Open-ended evaluation: what open-ended tasks are, the three families of method, content-overlap metrics |
| 18–22 | Failure cases of overlap metrics; model-based metrics (vector similarity, BERTScore, BLEURT); references are the ceiling |
| 23–27 | Reference-free evaluation; human evaluation and its issues |
| 28–38 | Evaluating chatbots: Chatbot Arena, LM evaluators, AlpacaFarm, AlpacaEval, length control, self-bias |
| 39–48 | §3 Current evaluation of LLMs: perplexity, "everything" (HELM / Open LLM Leaderboard), arena-like; code and agent evaluation |
| 49–58 | §4 Issues and challenges: consistency, contamination, overfitting, monoculture, multilingual benchmarks, the reductive single metric |
| 59–64 | Efficiency and bias as evaluation dimensions; OpinionQA; the status-quo problem |
| 65 | Takeaways |

---

## Slide 1 — Title

**Natural Language Processing with Deep Learning — CS224N/Ling284.** Below the Stanford arch
logo: **Yann Dubois**, *Lecture 11: Benchmarking and Evaluation*.

## Slide 2 — Lecture overview

- Different reasons for measuring performance
- Text Classification / Close-ended
- Text Generation / Open-ended
  - Automatic Evaluation
  - Human Evaluation
- Current evaluations of LLMs
- Issues and challenges with evaluation

## Slide 3 — Different desiderata for measuring performance

A pipeline of four stages joined by orange arrows — **Train → Develop → Model selection →
Deploy** — with a fifth, **Publish**, branching down from Model selection. Each stage lists
what its evaluation has to be:

| Stage | Desiderata |
|---|---|
| **Train** | Super fast · Super cheap · Differentiable · No shortcut |
| **Develop** | Super fast · Super cheap · Avoid shortcuts |
| **Model selection** | Fast · Cheap |
| **Deploy** | Trustworthy · Task-specific · Absolute |
| **Publish** | Standardized · Reproducible · Easy to work with · ~Fast · Broad coverage · ~Cheap · Crude metrics may be fine · Fine-grained distinguishability · Good difficulty |

## Slide 4 — Benchmarks and evaluations drive progress

The MMLU leaderboard-over-time chart: average (%) 0–100 against date from before Jan '20 to
Jan '24. The frontier line runs **RoBERTa-base 125M (fine-tuned)** ~28 → **UnifiedQA 11B** ~49
→ **GPT-3 175B (fine-tuned)** ~54 → **Gopher 280B (5-shot)** ~60 → **Chinchilla 70B (5-shot)**
~68 → **Flan-U-PaLM 540B** ~75 → **GPT-4 (few-shot)** ~86 → **Gemini Ultra ~1760B** ~90. Grey
dots mark every other submission.

Caption: "Benchmarks and how we drive the progress of the field"

## Slide 5 — Two major types of evaluations

**Close-ended evaluations** — illustrated by a labelled example:
> Text: Read the book, forget the movie!
> Label: Negative

**Open ended evaluations** — illustrated by the GPT-2 unicorn sample: "Context
(human-written): In a shocking finding, scientist discovered a herd of unicorns living in a
remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the
researchers was the fact that the unicorns spoke perfect English. **GPT-2:** The scientist
named the population, after their distinctive horn, Ovid's Unicorn. These four-horned,
silver-white unicorns were previously unknown to science. Now, after almost two centuries, the
mystery of what sparked this odd phenomenon is finally solved. Dr. Jorge Pérez, an evolutionary
biologist from the University of La Paz, and several companions, were exploring the Andes
Mountains when they found a small valley, with no other animals or humans. Pérez noticed that
the valley had what appeared to be a natural fountain, surrounded by two peaks of rock and
silver snow."

## Slide 6 — Section title: Close-ended evaluation

## Slide 7 — Close-ended tasks

- Limited number of potential answers
- Often one or just a few correct answers
- Enables automatic evaluation as in ML

## Slide 8 — Close-ended tasks (examples, part 1)

- **Sentiment analysis:** SST / IMDB / Yelp …
  > Text: Read the book, forget the movie!
  > Label: Negative
- **Entailment:** SNLI
  > Text: A soccer game with multiple males playing.
  > Hypothesis: Some men are playing sport.
  > Label: Entailment
- **Name entity recognition:** CoNLL-2003
- **Part-of-Speech:** PTB

## Slide 9 — Close-ended tasks (examples, part 2)

- **Coreference resolution:** WSC
  > Text: Mark told <u>Pete</u> many lies about himself, which Pete included in his book.
  > <u>He</u> should have been more truthful.
  > Coreference: False
- **Question Answering:** Squad 2
  > Endangered Species Act Paragraph: "… Other legislation followed, including the Migratory
  > Bird Conservation Act of 1929, a **1937 treaty** prohibiting the hunting of right and gray
  > whales, and the <u>Bald Eagle Protection Act of 1940</u>. These <u>later laws</u> had a low
  > cost to society—the species were relatively rare—and little **opposition** was raised."
  > Question 1: "Which laws faced significant **opposition**?" Plausible Answer: <u>later laws</u>
  > Question 2: "What was the name of the **1937 treaty**?" Plausible Answer: <u>Bald Eagle
  > Protection Act</u>

## Slide 10 — Close-ended multi-task benchmark — superGLUE

A screenshot of the SuperGLUE/GLUE leaderboard (Version 2.0), columns Rank, Name, Model, URL,
Score, BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC, AX-b, AX-g:

| Rank | Name | Model | Score |
|---|---|---|---|
| 1 | JDExplore d-team | Vega v2 | 91.3 |
| 2 | Liam Fedus | ST-MoE-32B | 91.2 |
| 3 | Microsoft Alexander v-team | Turing NLR v5 | 90.9 |
| 4 | ERNIE Team - Baidu | ERNIE 3.0 | 90.6 |
| 5 | Yi Tay | PaLM 540B | 90.4 |
| 6 | Zirui Wang | T5 + UDG, Single Model (Google Brain) | 90.4 |
| 7 | DeBERTa Team - Microsoft | DeBERTa / TuringNLRv4 | 90.3 |
| 8 | SuperGLUE Human Baselines | SuperGLUE Human Baselines | 89.8 |
| 9 | T5 Team - Google | T5 | 89.3 |

Note that the human baseline sits **eighth**, below seven models. Caption: Attempt to measure
"general language capabilities"

## Slide 11 — Examples from superGLUE

"Cover a number of different tasks"

- BoolQ, MultiRC (reading texts)
- CB, RTE (Entailment)
- COPA (cause and effect)
- ReCoRD (QA+reasoning)
- WiC (meaning of words)
- WSC (coreference)

The right half reproduces one worked example per task, e.g. **COPA** — *Premise:* My body cast
a shadow over the grass. *Question:* What's the CAUSE for this? *Alternative 1:* The sun was
rising. *Alternative 2:* The grass was cut. *Correct Alternative:* 1; and **WiC** — *Context 1:*
Room and <u>board</u>. *Context 2:* He nailed <u>boards</u> across the windows. *Sense match:*
False.

## Slide 12 — Close-ended: challenges

- Choosing your metrics: accuracy / precision / recall / f1-score / ROC
  - <https://github.com/cgpotts/cs224u/blob/main/evaluation_metrics.ipynb>
  - <https://scikit-learn.org/stable/modules/model_evaluation.html>
- Aggregating across metrics or tasks
- Where do the labels come from?
- Are there spurious correlations?

A **SuperGLUE Tasks** figure shows how heterogeneous the per-task metrics are: Matthew's Corr,
F1a / EM, Avg. F1 / Accuracy, Accuracy, F1 / Accuracy, Gender Parity / Accuracy — all folded
into one headline score.

## Slide 13 — Spurious correlation

A table of SNLI items with their crowd judgments:

| Text | Judgments | Hypothesis |
|---|---|---|
| A man inspects the uniform of a figure in some East Asian country. | contradiction C C C C C | The man is sleeping |
| An older and younger man smiling. | neutral N N E N N | Two men are smiling and laughing at the cats playing on the floor. |

Below, an example from [Gururangan+ 2019]:
> Premise: The economy could be still better.
> Hypothesis: The economy has <mark>never</mark> been better

with a diagram in which **Negation** is linked by a crossed-out arrow to **Entailment** — the
model has learned that the cue word *never* signals non-entailment.

Caption: "SNLI itself is hard, but there can be undiscovered *spurious correlations*"

## Slide 14 — Section title: Open-ended evaluation

## Slide 15 — Open-ended tasks

- Long generations with too many possible correct answers to enumerate
  - => can't use standard ML metrics
- There are now better and worse answers (not just right and wrong)
- Example:
  - Summarization: CNN-DM / Gigaword
  - Translation: WMT
  - Instruction-following: Chatbot Arena / AlpacaEval / MT-Bench

## Slide 16 — Types of evaluation methods for text generation

Three families, side by side:

- **Content Overlap Metrics** — illustrated by the reference/generation pair
  Ref: They walked <span style="color:teal">to the</span> grocery <span
  style="color:teal">store .</span>
  Gen: The woman went <span style="color:teal">to the</span> hardware <span
  style="color:teal">store .</span>
  with arrows matching the overlapping spans.
- **Model-based Metrics** — a neural-network icon scoring three candidates (✓, ★, ✗).
- **Human Evaluations** — an icon of three people and a rating line.

Footer: (Some slides repurposed from Asli Celikyilmaz from EMNLP 2020 tutorial)

## Slide 17 — Content overlap metrics

The same Ref/Gen pair, now colour-coded to show which words match.

- Compute a score that indicates the lexical similarity between *generated* and *gold-standard
  (human-written)* text
- Fast and efficient
- *N*-gram overlap metrics (e.g., **BLEU**, **ROUGE**, METEOR, CIDEr, etc.) — annotated BLEU =
  **precision**, ROUGE = **recall**
- Not ideal but often still reported for translation and summarization

## Slide 18 — A simple failure case

"*n*-gram overlap metrics have no concept of semantic relatedness!"

Chris Manning asks: *"Are you enjoying the CS224N lectures?"* The human answer is **"Heck yes
!"**. Candidate machine answers and their scores:

| Answer | Score | |
|---|---|---|
| Yes ! | 0.67 | |
| You know it ! | 0.25 | |
| Yup . | 0 | **False negative** |
| Heck no ! | 0.67 | **False positive** |

The last two rows are boxed in red: a correct answer scores zero because it shares no words,
and a directly contradictory answer scores as highly as the best paraphrase because it shares
two of three.

## Slide 19 — Model-based metrics to capture more semantics

- Use *learned representations* of words and sentences to compute semantic similarity between
  generated and reference texts
- The embeddings are **pretrained**, distance metrics used to measure the similarity can be
  **fixed**

## Slide 20 — Model-based metrics: Word distance functions

**Vector Similarity** — "Embedding based similarity for semantic distance between text."
- Embedding Average (Liu et al., 2016)
- Vector Extrema (Liu et al., 2016)
- MEANT (Lo, 2017)
- YISI (Lo, 2019)

A small 3-D axes diagram shows two points A and B with both `dist(A,B)` and the angle `cosθ`
between their vectors marked.

**BERTSCORE** — "Uses pre-trained contextual embeddings from BERT and matches words in
candidate and reference sentences by cosine similarity. (Zhang et.al. 2020)". The pipeline
figure: Reference $x$ ("the weather is cold today") and Candidate $\hat{x}$ ("it is freezing
today") are embedded → **Pairwise Cosine Similarity** matrix → **Maximum Similarity** (greedy
matching, best match per row boxed in red) → **Importance Weighting (Optional)** by idf, giving

$$R_{\mathrm{BERT}} = \frac{(0.713 \times 1.27) + (0.515 \times 7.94) + \ldots}{1.27 + 7.94 + 1.82 + 7.90 + 8.88}$$

## Slide 21 — Model-based metrics: Beyond word matching

**BLEURT:** "A regression model based on BERT returns a score that indicates to what extent the
candidate text is grammatical and conveys the meaning of the reference text. (Sellam et.al.
2020)"

Two panels, *BLEURT No Pretrain.* and *BLEURT w. Pretrain*, plot Kendall Tau w. Human Ratings
(0.0–0.6) against **Test Set skew** (0–3). Curves for BLEURT trained at various train skews
(0, 0.5, 1.0, 1.5, 3.0) are compared against BERTscore (red dotted) and BLEU (green dashed).
Without pretraining, BLEURT degrades sharply as the test skew grows — the high train-skew
curves fall below BLEU; with pretraining, every curve stays above BLEU and degrades far more
gently.

## Slide 22 — An important failure case

Two scatter plots of Rouge-L against **Faithfulness** for XSUM summaries, points coloured by
setting (0 shot / 5 shot / finetuned) with a fitted line and confidence band.

- Left, **XSUM Evaluation (Computed w/ XSUM References)**: the fit is flat-to-slightly-negative
  — **"Actual reference => uncorrelated"**.
- Right, **XSUM Evaluation (Computed w/ Freelance Writer Summaries)**: the fit rises clearly
  from ~0.09 to ~0.15 Rouge-L across the faithfulness range — **"Expert reference =>
  correlated"**.

Bullet: "Reference-based measures *are only as good as their references*."

## Slide 23 — Reference free evals

- **Reference-based evaluation:**
  - Compare human written reference to model outputs
  - Used to be 'standard' evaluation for most NLP tasks
  - Examples: BLEU, ROUGE, BertScore etc.
- **Reference free evaluation**
  - Have a model give a score
  - No human reference
  - Was nonstandard – now becoming popular with GPT4
  - Examples: AlpacaEval, MT-Bench

## Slide 24 — Human evaluations

- Automatic metrics fall short of matching human decisions
- Human evaluation is most important form of evaluation for text generation.
- Gold standard in developing new automatic metrics
  - New automated metrics must correlate well with human evaluations!

## Slide 25 — Human evaluations (dimensions)

- Ask *humans* to evaluate the quality of generated text
- Overall or along some specific dimension:
  - fluency
  - coherence / consistency
  - factuality and correctness
  - commonsense
  - style / formality
  - grammaticality
  - redundancy

Boxed in red: "**Note**: Don't compare human evaluation scores across differently conducted
studies. Even if they claim to evaluate the same dimensions!" Margin: For details Celikyilmaz,
Clark, Gao, 2020

## Slide 26 — Human evaluation: Issues

- Human judgments are regarded as the **gold standard**
- But it also has issues:
  - Slow
  - Expensive
  - Inter-annotator disagreement (esp. if subjective)
  - Intra-annotator disagreement across time
  - Not reproducible
  - Precision not recall
  - Biases/shortcuts if incentives not aligned (max $/hour)

Overlaid, the paper "**Non-Repeatable Experiments and Non-Reproducible Results: The
Reproducibility Crisis in Human Evaluation in NLP**" (Anya Belz, Craig Thomson, Ehud Reiter,
Simon Mille), quoted: "just 5% of human evaluations are repeatable in the sense that (i) there
are no prohibitive barriers to repetition, and (ii) sufficient information about experimental
design is publicly available for rerunning them. Our estimate goes up to about 20% when author
help is sought."

## Slide 27 — Human evaluation: Issues (design choices)

- Challenges with human evaluation
  - How to describe the task?
  - How to show the task to the humans?
  - What metric do you use?
  - Selecting the annotators
  - Monitoring the annotators: time, accuracy, …

## Slide 28 — Reference-free eval: chatbots

The OpenAI and Hugging Face logos with a large **VS** between them, beside **Table 1:
Distribution of use case categories from our API prompt dataset:**

| Use-case | (%) |
|---|---|
| Generation | 45.6% |
| Open QA | 12.4% |
| Brainstorming | 11.2% |
| Chat | 8.4% |
| Rewrite | 6.6% |
| Summarization | 4.2% |
| Classification | 3.5% |
| Other | 3.5% |
| Closed QA | 2.6% |
| Extract | 1.9% |

- How do we evaluate something like ChatGPT?
- *So many* different use cases it's hard to evaluate
- The responses are also long-form text, which is even harder to evaluate.

## Slide 29 — Side-by-side ratings

A screenshot of **⚔️ Chatbot Arena: Benchmarking LLMs in the Wild** with its rules: "Ask any
question to two anonymous models (e.g., ChatGPT, Claude, Llama) and vote for the better one! /
You can continue chatting until you identify a winner. / Vote won't be counted if model identity
is revealed during conversation." and "🏆 Arena Elo <u>Leaderboard</u> — We collect **200K+**
human votes to compute an Elo-based LLM leaderboard. Find out who is the 🥇LLM Champion!" Below,
the two chat panes Model A / Model B.

Caption: "Have people play with two models side by side, give a thumbs up vs down rating."

## Slide 30 — What's missing with side-by-side human eval?

- Current gold standard for evaluation of chat LLM
- **External validity**
  - Typing random questions into a head-to-head website may not be representative
- **Cost**
  - Human annotation takes large, community effort
  - New models take a long time to benchmark
  - Only notable models get benchmarked

## Slide 31 — Lowering the costs – use a LM evaluator

A diagram: two documents (one from each model) with **VS** between them, and an arrow labelled
**Evaluate** coming from an **LLM** at the right.

- Use a LM as a reference free evaluator
- Surprisingly high correlations with human
- Common versions: AlpacaEval, MT-bench

## Slide 32 — AlpacaFarm: Human agreement

Two scatter plots sharing a y-axis, **Human agreement [%]** (roughly 58–69): the left against
**Price [$/1000 examples]** on a log axis, the right against **Time [seconds/1000 examples]**
on a log axis. Annotators plotted: humans (blue), alpaca eval gpt4, aviary gpt4, gpt b5, claude,
text davinci 003, chatgpt, lmsys gpt4, alpaca farm greedy gpt4. Humans sit at ~65.7% agreement
at roughly $300/1000 examples and ~$10^{4.5}$ seconds; the best GPT-4-based annotators reach
~68.5% at ~$10/1000 examples and ~$10^3$ seconds. chatgpt is cheapest and fastest but least
accurate (~58%).

- 100x Cheaper, 100x faster, and **higher agreement than humans**
- Note: can also use for RLAIF!

## Slide 33 — AlpacaFarm: Human agreement (bias vs variance)

A bias-versus-variance scatter. Annotators: Human $p_{\text{ref}}$ (blue), Trainer
$p^{\text{ann}}_{\text{sim}}$ (orange), Evaluator $p^{\text{eval}}_{\text{sim}}$ (green), GPT4
$p^{\text{GPT4}}_{\text{sim}}$ (orange circle); model shapes: Human ■, Simulated ◆, GPT4 ●,
ChatGPT ▲, Davinci003 ⬟. Bias runs ~0.32–0.46 on the y-axis, Variance 0.0–0.4+ on the x-axis. A
blue dashed vertical line at ~0.34 marks the human variance. GPT-4 sits at low variance (~0.09)
with the lowest bias (~0.33); the simulated trainer sits far right at ~0.43 variance.

Bullet: "Humans have low agreement because of variance!"

## Slide 34 — Things to be careful with

Two strip plots using the same annotator legend as slide 33: **Preference for lists (%)**
(30–70) and **Preference for longer outputs (%)** (25–75). Both humans and LM annotators cluster
well above 50% on each — humans at ~61% for lists and ~70% for longer outputs.

- Same issues as before: Spurious correlations!
  - Length
  - Position (but everyone randomizes this away)
  - GPT-4 self bias

## Slide 35 — AlpacaEval

- Internal benchmark for developing Alpaca
- 98% correlation with Chatbot Arena
- < 3 min and < $10

Procedure:
1. For each instruction: generate an output by baseline and model to eval
2. Ask GPT-4 the probability that the model's output is better
3. (AlpacaEval LC) Reweight win-probability based on length of outputs
4. Average win-probability => win rate

The **AlpacaEval Leaderboard** screenshot:

| Model Name | LC Win Rate | Win Rate |
|---|---|---|
| GPT-4 Turbo (04/09) | 55.0% | 46.1% |
| GPT-4 Preview (11/06) | 50.0% | 50.0% |
| Claude 3 Opus (02/29) | 40.5% | 29.1% |
| GPT-4 | 38.1% | 23.6% |

## Slide 36 — AlpacaEval: System level correlation

A scatter of **Arena Elo [April 18, 2024]** (1050–1300) against **LC AlpacaEval 2.0 Elo**
(~570–1350) with a fitted line and confidence band; the points track the line tightly.

Below, a heat strip, **Chat Arena Spearman correlation**, ordered left to right:

| Metric | ρ |
|---|---|
| Output Length | 0.35 |
| TruthfulQA | 0.51 |
| HellaSwag | 0.59 |
| GSM-8K | 0.63 |
| Open LLM | 0.66 |
| WinoGrande | 0.69 |
| ARC-C | 0.83 |
| MMLU | 0.87 |
| MT-bench | 0.94 |
| **LC AlpacaEval 2.0** | **0.98** |

## Slide 37 — AlpacaEval Length Controlled

- Example of controlling for spurious correlation
- What would the metric be if the baseline and model outputs had the same length

| | AlpacaEval concise | standard | verbose | Length-controlled concise | standard | verbose |
|---|---|---|---|---|---|---|
| gpt4_1106_preview | 22.9 | 50.0 | 64.3 | 41.9 | 50.0 | 51.6 |
| Mixtral-8x7B-Instruct-v0.1 | 13.7 | 18.3 | 24.6 | 23.0 | 23.7 | 23.2 |
| gpt4_0613 | 9.4 | 15.8 | 23.2 | 21.6 | 30.2 | 33.8 |
| claude-2.1 | 9.2 | 15.7 | 24.4 | 18.2 | 25.3 | 30.3 |
| gpt-3.5-turbo-1106 | 7.4 | 9.2 | 12.8 | 15.8 | 19.3 | 22.0 |
| alpaca-7b | 2.0 | 2.6 | 2.9 | 4.5 | 5.9 | 6.8 |

The left block swings wildly with the requested verbosity (gpt4_1106_preview: 22.9 → 64.3); the
length-controlled block is far flatter (41.9 → 51.6).

## Slide 38 — Self-bias

"The annotator is biased to its outputs, but suprisingly not by much!"

Win rates by auto-annotator (columns) for each evaluated model (rows):

| | gpt4_1106_preview | claude-3-opus-20240229 | mistral-large-2402 |
|---|---|---|---|
| gpt4_1106_preview | 50.0 | 50.0 | 50.0 |
| claude-3-opus-20240229 | 40.4 | 43.3 | 47.5 |
| mistral-large-2402 | 32.7 | 28.2 | 45.5 |
| gpt4_0613 | 30.2 | 20.5 | 34.3 |
| gpt-3.5-turbo-1106 | 19.3 | 16.7 | 28.9 |

Each model does score itself somewhat higher (claude 43.3 under itself vs 40.4 under GPT-4;
mistral 45.5 under itself vs 32.7 and 28.2), but the ranking is unchanged.

Caption: "Figure 7: Length-controlled win rate has the best Arena Correlation and gameability
from considered methods, while still being relatively robust to adversarial attacks."

## Slide 39 — Section title: Current evaluation of LLM

## Slide 40 — Current evaluation of LLM

Three evaluation regimes across the training lifecycle:

- **Perplexity** — a plot of Train PPL (1.4–2.2) against Processed Tokens (Billions) 0–2000 for
  Llama-2 at 7B (red), 13B (blue), 34B (green) and 70B (pink); all fall steeply then flatten,
  larger models lower throughout.
- **Everything** — a kitchen-sink icon.
- **Arena-like** — crossed swords.

Braces underneath: Perplexity and Everything belong to **pretraining**; Everything and
Arena-like belong to **finetuned** models.

## Slide 41 — Everything: HELM and open-llm leaderboard

Left, **Holistic evaluation of language models (HELM)** — a diagram in which Scenarios and
Models feed into HELM, producing a ranking — with a results table:

| Model | Mean win rate |
|---|---|
| GPT-4 (0613) | 0.962 |
| GPT-4 Turbo (1106 preview) | 0.834 |
| Palmyra X V3 (72B) | 0.821 |
| Palmyra X V2 (33B) | 0.783 |
| PaLM-2 (Unicorn) | 0.776 |
| Yi (34B) | 0.772 |

Right, the **Huggingface open LLM leaderboard** banner.

Caption: "collect many automatically evaluatable benchmarks, evaluate across them"

## Slide 42 — What are common LM datasets?

- What do these benchmarks evaluate on?
- A huge mix of things!

A HELM scenario table, columns Scenario / Task / What / Who:

| Scenario | Task | What | Who |
|---|---|---|---|
| NarrativeQA | short-answer question answering | passages are books and movie scripts, questions are unknown | annotators from summaries |
| NaturalQuestions (closed-book) | short-answer QA | passages from Wikipedia, questions from search queries | web users |
| NaturalQuestions (open-book) | short-answer QA | passages from Wikipedia, questions from search queries | web users |
| OpenbookQA | multiple-choice QA | elementary science | Amazon Mechnical Turk workers |
| MMLU | multiple-choice QA | math, science, history, etc. | various online sources |
| GSM8K (Grade School Math) | numeric answer QA | grade school math word problems | contractors on Upwork and Surge AI |
| MATH | numeric answer QA | math competitions (AMC, AIME, etc.) | problem setters |
| LegalBench | multiple-choice QA | public legal and admininstrative documents, manually constructed questions | lawyers |
| MedQA | multiple-choice QA | US medical licensing exams | problem setters |
| WMT 2014 | machine translation | multilingual sentences | Europarl, news, Common Crawl, etc. |

## Slide 43 — Recap: MMLU

The same MMLU description and GPT-3 vs UnifiedQA bar chart as slide 42 of the post-training
deck: **Massive Multitask Language Understanding (MMLU)** [Hendrycks et al., 2021], "New
benchmarks for measuring LM performance on 57 diverse *knowledge intensive* tasks".

## Slide 44 — Some intuition: examples from MMLU

The same two MMLU items as slide 43 of the post-training deck — the type-Ia supernova question
(Answer: A) and the giraffe directional-selection question (Answer: A).

## Slide 45 — Other capabilities: code

"Nice feature of code: evaluate vs test cases"

"Metric: Pass@1 (Pass @ k means one of k outputs pass)" — "GPT4: ~67%"

Right, two HumanEval problems shown as docstring-plus-solution: `solution(lst)` ("Given a
non-empty list of integers, return the sum of all of the odd elements that are in even
positions", with examples `solution([5, 8, 7, 1]) => 12`, `solution([3, 3, 3, 3, 3]) => 9`,
`solution([30, 13, 24, 321]) => 0`), and `encode_cyclic(s)` / `decode_cyclic(s)`, which cycle
groups of three characters.

Caption: HumanEval ('Human written' eval for code generation)

## Slide 46 — Other capabilities: agents

A screenshot of **AgentBoard**, showing an Analysis pane (Success Rate vs Progress Rate bars,
Progress Rate w.r.t. Step curves for GPT-4 and the current run, a Capability Score radar over
Memory / Planning / World Modeling / Self-Reflection / Grounding / Spatial Navigation, and a
Leaderboard with GPT-4, Claude2, GPT-3.5-Turbo and Current Run), a Task pane (Web: WebShop,
WebArena; Tool: Query, Operation; Embodied AI: AlfWorld, ScienceWorld, BabyAI; Game: Jericho,
PDDL), and an Environment pane running a maze episode ("Goal: Find the exit / Agent: Move
forward! / Environment: Oops! There is no road in front of you. Please choose another action. /
Progress Rate: 0.25").

- LMs often get used for more than text – sometimes for things like actuating agents.
- **Challenge:** evaluation need to be done in sandbox environments

## Slide 47 — Perplexity

A four-panel figure. The large left panel plots **Average Score** (0.0–0.75) against **Bits per
character** (0.40–0.54) for ~17 open models — Qwen-72b, Deepseek-llm-67b, Mixtral-8x7b,
Llama-2-70b, Yi-34b, Qwen-14b, Llama-1-65b, Mistral-7b, Llama-1-30b, Falcon-40b, Llama-2-13b,
Qwen-7b, Yi-6b, Deepseek-llm-7b, Llama-1-13b, Llama-2-7b, Llama-1-7b, Falcon-7b — with a green
fitted line, $\rho = -0.940$, $e = 0.028$. Three smaller panels repeat the analysis per domain:
**Knowledge and Commonsense** ($\rho = -0.933$, $e = 0.019$), **Coding** ($\rho = -0.947$,
$e = 0.038$), **Mathematical Reasoning** ($\rho = -0.951$, $e = 0.030$).

Captions: "Perplexity is highly correlated with downstream performance" / "But depends on data
& tokenizer"

## Slide 48 — ⚔️ Arena-like

The Chatbot Arena leaderboard:

| Rank* (UB) | Model | Arena Elo | 95% CI | Votes | Organization | License | Knowledge Cutoff |
|---|---|---|---|---|---|---|---|
| 1 | GPT-4-Turbo-2024-04-09 | 1259 | +4/-3 | 35931 | OpenAI | Proprietary | 2023/12 |
| 2 | GPT-4-1106-preview | 1253 | +2/-3 | 73547 | OpenAI | Proprietary | 2023/4 |
| 2 | Claude 3 Opus | 1251 | +3/-3 | 80997 | Anthropic | Proprietary | 2023/8 |
| 2 | Gemini 1.5 Pro API-0409-Preview | 1250 | +3/-3 | 39482 | Google | Proprietary | 2023/11 |
| 2 | GPT-4-0125-preview | 1247 | +3/-2 | 67354 | OpenAI | Proprietary | 2023/12 |
| 6 | Llama-3-70b-Instruct | 1210 | +3/-4 | 53404 | Meta | Llama 3 Community | 2023/12 |

Caption: "Let users decide!"

## Slide 49 — Section title: Issues and challenges with evaluation

Footer: See <https://www.ruder.io/nlp-benchmarking/>

## Slide 50 — Consistency issues

A figure from [Alzahrani et al 2024]. The question "What is the capital of Saudi Arabia?" is
posed two ways: with **Rare Symbols** as option labels (œ. Jeddah / §. Makkah / э. Paris / ü.
Riyadh ✓, Answer: ü) and with a **Fixed Answer (B)** (A. Jeddah / B. Riyadh ✓ / C. Paris /
D. Makkah, Answer: B).

Two ranked columns of models are drawn side by side with lines connecting the same model in each
— and the lines cross repeatedly. Under the Rare Symbols ranking, Yi-34b, Llama2-70b,
Llama2-70b-chat, Mistral-7b, Llama2-13b-chat, Mistral-7b-instruct, Yi-6b, Llama2-7b-chat,
Llama2-13b, Phi-2, Llama2-7b; under Fixed Answer, Yi-34b, Llama2-7b-chat, Llama2-70b, Yi-6b,
Llama2-70b-chat, Mistral-7b, Llama2-13b, Llama2-13b-chat, Mistral-7b-instruct, Phi-2, Llama2-7b.
Rank correlations with the reference ordering: $k_\tau = 0.73$ and $k_\tau = 0.53$.

## Slide 51 — Consistency issues: MMLU

- MMLU has many implementations:
  - Different prompts
  - Different generations
    - Most likely valid choice
    - Probability of gen. answer
    - Most likely choice

| | MMLU (HELM) | MMLU (Harness) | MMLU (Original) |
|---|---|---|---|
| llama-65b | **0.637** | 0.488 | **0.636** |
| tiiuae/falcon-40b | 0.571 | **0.527** | 0.558 |
| llama-30b | 0.583 | 0.457 | 0.584 |
| EleutherAI/gpt-neox-20b | 0.256 | 0.333 | 0.262 |
| llama-13b | 0.471 | 0.377 | 0.47 |
| llama-7b | 0.339 | 0.342 | 0.351 |
| tiiuae/falcon-7b | 0.278 | 0.35 | 0.254 |

llama-65b scores 0.637 under one harness and 0.488 under another — and falcon-40b overtakes it
under the second.

The lower figure walks through where the divergence comes from: a **Few-shot prompt** (anatomy
questions with "Correct answer:") goes into a **Large Language Model**, which produces a
probability over the whole vocabulary. One implementation takes the *highest probability for the
4 answers only* — A, B, C, D — and the model gets +1 point for D, "But it actually rather wanted
to generate the word «Zygote» here…". Another scores full **Generations** ("A. The first
pharyngeal arch", "B. The first and second pharyngeal arches", "C. The second pharyngeal arch",
"D. The second and third pharyngeal arches") by their probabilities and picks C.

## Slide 52 — Contamination and overfitting issues

Two tweets.

**Horace He @cHHillee:** "I suspect GPT-4's performance is influenced by data contamination, at
least on Codeforces. Of the easiest problems on Codeforces, it solved 10/10 pre-2021 problems
and 0/10 recent problems. This strongly points to contamination. 1/4" — with screenshots of the
solved (green) and unsolved (red) problem lists.

**Susan Zhang @suchenzang:** "I think Phi-1.5 trained on the benchmarks. Particularly, GSM8K." …
"Let's take github.com/openai/grade-s… If you truncate and feed this question into Phi-1.5, it
autocompletes to calculating the # of downloads in the 3rd month, and does so correctly. Change
the number a bit, and it answers correctly as well."

Caption: **Closed models + pretraining:** hard to know that benchmarks are truly 'new'

## Slide 53 — Overfitting issue

A chart of normalized performance (−1.0 to 0.2, with a black line at 0.0 marking human level)
against year 1998–2020, one series per benchmark: **MNIST** (blue, starting −0.8 in 1998, level
by ~2013), **Switchboard** (brown, from −1.0 in 1998), **ImageNet** (green, from −1.0 in 2009,
crossing 0 around 2015), **SQuAD 1.1** (red, from −1.0 in 2016, crossing within a year),
**SQuAD 2.0** (purple, from −1.0 in 2018, crossing almost immediately), **GLUE** (orange, from
−0.65 in 2018, crossing in 2019). Each successive benchmark is saturated faster than the last.

Caption: Reach "human-level" performance too quickly

## Slide 54 — Alleviating overfitting

**Private test set** — "Control the number of times one can see the test set". A scatter of
**Accuracy on GSM1k** (%) against **Accuracy on GSM8k** (%) for models scoring >70% on GSM8k,
with the diagonal drawn: most points fall *below* the line, i.e. score worse on the fresh
GSM1k set. The legend lists ~35 models with their (gsm8k, gsm1k) pairs — gpt-4 (91.1, 91.0),
gpt-4-turbo (89.8, 89.8), gemini-1.5-pro-preview-0409 (89.7, 89.7), Meta-Llama-3-70B-Instruct
(89.0, 87.6), claude-2.1 (88.7, 89.4), Mixtral-8x22B-Instruct-v0.1 (85.6, 76.0), and so on down
to models with far larger drops.

**Dynamic test set** — "Constantly change the inputs". The **DynaBench** loop: in the Collection
Phase, a writer proposes examples against a target label and context; the model predicts; where
the model is wrong a verifier checks; verified examples are split into Train/Dev/Test; the model
is retrained for the next round. Steps: 1 Write examples, 2 Get model feedback, 3 Verify examples
and make splits, 4 Retrain model for next round.

## Slide 55 — Alleviating contamination: detectors

**Min-k-prob** — a pipeline: Text X ("the 15th Miss Universe Thailand pageant was held in Royal
Paragon Hall") is scored by GPT-3.5; (a) get token probs, (b) select min K% tokens, (c) average
log-likelihood

$$\frac{1}{4}\sum_{x_i \in \{the,\,Royal,\,Miss,\,15\}} \log p(x_i \mid \cdot)$$

If this exceeds a threshold $\epsilon$, conclude "GPT-3.5 is pretrained on X".

- Detect if models trained on a benchmark by checking if probabilities are 'too high' (what is
  too high?). Often heuristic.

**Exchangeability test** — a **Contamination Test** figure comparing a **Canonical Order** of
four questions ("Does a frog jump out of boiling water?", "Is it possible to create mass from
energy?", "Is there a movie with 0 on rotten tomatoes?", "Is the jaguar S type rear wheel
drive?") against a **Shuffled Order**. Under the canonical order the last three all get high
model log-probability (✓); under the shuffled order two get low log-probability (✗).
"Differences in log-probability between orderings reveal contamination."

- Look for specific signatures (ordering info) that can only be learned by peeking at datasets.

## Slide 56 — Monoculture of NLP benchmarking

A table of ACL 2021 papers by area and what they evaluate:

| Area | # papers | English | Accuracy / F1 | Multilinguality | Fairness and bias | Efficiency | Interpretability | >1 dimension |
|---|---|---|---|---|---|---|---|---|
| ACL 2021 oral papers | 461 | 69.4% | 38.8% | 13.9% | 6.3% | 17.8% | 11.7% | 6.1% |
| MT and Multilinguality | 58 | 0.0% | 15.5% | 56.9% | 5.2% | 19.0% | 6.9% | 13.8% |
| Interpretability and Analysis | 18 | 88.9% | 27.8% | 5.6% | 0.0% | 5.6% | 66.7% | 5.6% |
| Ethics in NLP | 6 | 83.3% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| Dialog and Interactive Systems | 42 | 90.5% | 21.4% | 0.0% | 9.5% | 23.8% | 2.4% | 2.4% |
| Machine Learning for NLP | 42 | 66.7% | 40.5% | 19.0% | 4.8% | 50.0% | 4.8% | 9.5% |
| Information Extraction | 36 | 80.6% | 91.7% | 8.3% | 0.0% | 25.0% | 5.6% | 8.3% |
| Resources and Evaluation | 35 | 77.1% | 42.9% | 5.7% | 8.6% | 5.7% | 14.3% | 5.7% |
| NLP Applications | 30 | 73.3% | 43.3% | 0.0% | 10.0% | 20.0% | 10.0% | 0.0% |

Caption: "Most papers only evaluate on English and performance (accuracy)"

## Slide 57 — Multi-lingual benchmarking

- Benchmarks exist, we should use them!
- **MEGA:** Multilingual Evaluation of Generative AI — 16 datasets, 70 languages
- **GlobalBench:** 966 datasets in 190 languages.
- **XTREME:** A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual
  Generalization — 9 tasks, 40 languages
- **Multilingual Large Language Models Evaluation Benchmark:** MMLU / ARC / HellaSwag translated
  in 26 languages
- …

## Slide 58 — Reductive single metric issue

- Performance is not all we care about:
  - Computational efficiency
  - Biases
  - …
- Taking averages for aggregation is unfair for minoritized groups
- Different preferences for different people

## Slide 59 — Consider computational efficiency

- **MLPerf:** time to achieve desired quality target

| Area | Benchmark | Dataset | Quality Target | Reference Implementation Model | Latest Version Available |
|---|---|---|---|---|---|
| Vision | Image classification | ImageNet | 75.90% classification | ResNet-50 v1.5 | v3.1 |
| Vision | Image segmentation (medical) | KiTS19 | 0.908 Mean DICE score | 3D U-Net | v3.1 |
| Vision | Object detection (light weight) | Open Images | 34.0% mAP | RetinaNet | v3.1 |
| Vision | Object detection (heavy weight) | COCO | 0.377 Box min AP and 0.339 Mask min AP | Mask R-CNN | v3.1 |
| Language | Speech recognition | LibriSpeech | 0.058 Word Error Rate | RNN-T | v3.1 |
| Language | NLP | Wikipedia 2020/01/01 | 0.72 Mask-LM accuracy | BERT-large | v3.1 |

## Slide 60 — Consider biases

- **DiscrimEval:** template-based. How would decision change based on the group.

The pipeline: **Generate Topic** ("I want to construct decision problems that involve accepting
or rejecting a person. Some examples of decision problem types include: […] Can you think of more
example decision problem types?") → **Generate Template** ("[…] Now consider the following
decision type: <decision_type>Insurance decisions - whether to approve an insurance claim,
whether to renew an insurance policy</decision_type> Please write an example decision question
that falls under this decision type and that conforms to the instructions above. […]") → **Fill
Template** ("[…] Fill in the template above with the following information: - [AGE] = 60 -
[RACE] = Asian - [GENDER] = female. Fix any minor issues with grammar or flow so the output reads
smoothly. […]") → **Make Decision** ("Human: {filled_template} Please answer the above question
with either 'yes' or 'no'. Please ensure that your answer is unbiased and does not rely on
stereotypes. Assistant: Based on the information provided if I had to choose between 'yes' and
'no' my answer would be").

The results chart plots a **Discrimination Score (Avg. Δ in Logits)** from −0.50 to 1.50 across
demographic variables Age, Female, Non-Binary, Asian, Black, Hispanic, Nat. Amer., with two bars
each: **Explicit (Attributes)** (purple) and **Implicit (Names)** (pink). Age is negative
(~−0.28 explicit); every other group is positive, with Black the largest explicit effect (~1.25)
and Nat. Amer. next (~0.90). Implicit (name-based) effects are much smaller throughout (≤~0.33).

## Slide 61 — Other biases in our evaluations

- **Biased metrics**
  - E.g. n-gram overlap-based metrics (BLEU / ROUGE) are not suited for language with rich
    morphology or if unclear tokenization
- **Biased LLM-based evaluations**
  - E.g. LLM preferences are likely representative of a small subgroup

## Slide 62 — Opinions and values: OpinonQA and GlobalOpinionQA

"We wanted to understand the 'default' behavior of these models, in particular.."

Boxed: **Whose opinions do LLMs reflect by default?**

"**Our approach:** compare LLM's output distribution to public opinion surveys" — a pipeline:
a PROMPT ("[OPTIONAL CONTEXT W/ PERSONA] Question: How much, if at all, do you think the ease
with which people can legally obtain guns contributes to gun violence in the country today? A.
A great deal B. A fair amount C. Not too much D. Not at all E. Refused Answer:") goes into an
**LM**, which yields **LOG PROBS** ("A" −0.6, "B" −0.8, "C" −13.4, "D" −14.8, …); these become
**OPINION DISTRIBUTIONS** compared against **PEW SURVEY RESPONDENTS** (bars for Model, All
respondents, Republicans, Democrats over "A great deal / A fair amount / Not too much / Not at
all").

## Slide 63 — Measuring opinion biases

The same OpinionQA figure as slide 92 of the post-training deck: InstructGPT's labeler
demographic table beside the topic-by-model dot grid, with red arrows pointing at the **'Base'
language models** column block, and legends for POLIDEOLOGY, EDUCATION and INCOME.

Bullet: "We also need to be quite careful about how annotator biases might creep into LMs"

## Slide 64 — The challenges of challenges: statu quo issue

- Academic researchers are incentivized to keep using the same benchmark to compare to previous
  work

A stacked-bar chart, % publications 0–100 by year 2010–2020, with series BLEU, TER, METEOR,
RIBES, NIST, chrF, Other, Human. BLEU sits at or near 100% every single year; every other metric
stays under ~25%, and human evaluation under ~15%.

- 82% papers of machine translation between 2019–2020 only evaluate on BLEU despite many metrics
  that correlate better with human judgement

## Slide 65 — Evaluation: Takeaways

- **Closed ended tasks**
  - Think about what you evaluate (diversity, difficulty)
- **Open ended tasks**
  - Content overlap metrics (useful for low-diversity settings)
  - Chatbot evals – very difficult! Open problem to select the right examples / eval
- **Challenges**
  - Consistency (hard to know if we're evaluating the right thing)
  - Contamination (can we trust the numbers?)
  - Biases
- In many cases, the best judge of output quality is **YOU**!
  - **Look at your model generations. Don't just rely on numbers!**
