# Evaluating LLMs

What the field actually measures, and with what. Covered in [lecture 12](12-benchmarking.md),
slides 3–23 and 39–48, with the multitask-benchmark aside in
[lecture 11](11-post-training.md), slides 42–46. For open-ended generation metrics see
[evaluating NLG](evaluating-nlg.md); for chatbot evaluation see
[LLM-as-a-judge](llm-as-a-judge.md); for what breaks see
[benchmark contamination and overfitting](benchmark-contamination.md).

## Evaluation is stage-dependent

There is no single good metric, because five different stages of a project want different things
(lecture 12, slide 3):

| Stage | Requirements |
| --- | --- |
| Train | super fast, super cheap, **differentiable**, no shortcut |
| Develop | super fast, super cheap, avoids shortcuts |
| Model selection | fast, cheap |
| Deploy | **trustworthy**, task-specific, **absolute** |
| Publish | standardized, reproducible, easy to work with, broad coverage, ~fast, ~cheap; crude metrics may be fine; fine-grained distinguishability; good difficulty |

Two entries carry most of the argument. **Absolute** is a deployment-only requirement: you need a
threshold ("if I have less than 95% accuracy I'm not putting my model in production"), whereas
every other stage only ever compares (≈3:56). And **crude metrics may be fine** applies to
publishing specifically — over a decade what matters is that the metric points where the field is
genuinely improving, not that adjacent numbers are separable (≈5:28).

Academic benchmarks have a further constraint: too hard and everything scores at random, too easy
and nothing can beat the baseline. Either way nobody uses the benchmark (slide 6).

## Close-ended tasks

A close-ended task has a limited answer set — "think, like, less than 10" — usually with one
correct answer, which makes it ordinary supervised machine learning (lecture 12, slides 7–9):

| Task | Benchmarks |
| --- | --- |
| Sentiment analysis | SST, IMDB, Yelp |
| Entailment | SNLI |
| Named entity recognition | CoNLL-2003 |
| Part of speech | Penn Treebank |
| Coreference resolution | WSC |
| Question answering | SQuAD 2 |

**SuperGLUE** (slides 10–11) bundles eight of them — BoolQ and MultiRC (reading), CB and RTE
(entailment), COPA (cause and effect), ReCoRD (QA + reasoning), WiC (word senses), WSC
(coreference) — and averages into one "general language capabilities" score. Worth noticing on the
leaderboard screenshot: the **SuperGLUE Human Baselines** row sits *eighth*, below seven models.

Three ways to get this wrong (slides 12–13), all of which recur later:

- **The metric choice determines the winner.** With 90% of email legitimate, always predicting
  "not spam" scores 90% accuracy and classifies nothing; hence precision, recall, F1, ROC/AUC.
- **Aggregation across heterogeneous metrics.** SuperGLUE averages Matthews correlation, F1a/EM,
  accuracy and gender-parity/accuracy together. The lecturer recalls a benchmark that averaged in
  a column where *lower was better*, uncorrected, for some time (≈14:45).
- **Spurious correlations.** On SNLI a hypothesis-only classifier does well, because annotators
  writing non-entailed hypotheses reach for negation (Gururangan et al., 2019). The dataset is
  hard; the shortcut was undiscovered.

## Multitask benchmarks

**MMLU** — Massive Multitask Language Understanding (Hendrycks et al., 2021) — is the one the field
converged on: 57 knowledge-intensive multiple-choice tasks spanning abstract algebra, anatomy,
astronomy, business ethics, econometrics, formal logic, medicine, law and more. A sample item:

> **Astronomy.** What is true for a type-Ia supernova?
> A. This type occurs in binary systems. B. This type occurs in young galaxies. C. This type
> produces gamma-ray bursts. D. This type produces high amounts of X-rays. *Answer: A*

The progress curve (lecture 11 slide 44; lecture 12 slide 4) runs GPT-2 ~32% → UnifiedQA and GPT-3
~50% → Gopher 5-shot ~60% → Chinchilla 5-shot ~68% → Flan-PaLM ~75% → GPT-4 ~86% → Gemini Ultra
~90%, against a 25% random floor. 90% is treated as roughly human level. "When Mark Zuckerberg said
that Llama 3 was out, he talked about MMLU scores, which I find kind of crazy" (≈57:46) — the
crazy part being that MMLU turns out not to be a single well-defined number at all; see
[benchmark contamination and overfitting](benchmark-contamination.md).

**BIG-Bench** (Srivastava et al., 2022) is the breadth-first alternative: 200+ tasks covering
common sense, logical reasoning, reading comprehension, mathematics, multilingual, non-language,
theory of mind and self-play — including "Kanji ASCII Art to Meaning", which asks a model to read a
kanji rendered in `#` characters (lecture 11, slides 45–46).

## Open-ended tasks

An open-ended task has too many correct answers to enumerate, *and* a quality continuum rather than
right/wrong (lecture 12, slide 15): summarization (CNN/DailyMail, Gigaword), translation (WMT),
instruction following (Chatbot Arena, AlpacaEval, MT-Bench).

Instruction following is "the mother of all tasks" — any earlier task can be posed to a chatbot, so
evaluating a chatbot subsumes evaluating everything (≈19:22).

The three families of method (slide 16), inherited from
[lecture 10](10-natural-language-generation.md):

- **Content overlap metrics** — BLEU (precision-flavoured), ROUGE (recall-flavoured), METEOR,
  CIDEr. Fast, and still what translation and summarization papers report. Full treatment at
  [evaluating NLG](evaluating-nlg.md) and [evaluating machine translation](evaluating-machine-translation.md).
- **Model-based metrics** — vector similarity, **BERTScore** (greedy cosine matching between BERT
  embeddings of candidate and reference, optionally idf-weighted), **BLEURT** (a BERT regression
  model continually pretrained to predict BLEU and BERTScore, then finetuned on human ratings).
- **Human evaluation** — the gold standard, and the yardstick every automatic metric is validated
  against.

Two additions this lecture makes to that picture:

**References are the ceiling.** Slide 22: on XSUM, ROUGE-L against the *dataset's* references is
essentially uncorrelated with human faithfulness ratings; against summaries written by expert
freelance writers it correlates clearly. Since news-summarization references are typically the
article's own bullet points, "reference-based measures are only as good as their references"
(≈29:24).

**Reference-free evaluation now works.** Slide 23: asking a model for a score with no human
reference used to fail; with GPT-4 it works well enough that AlpacaEval and MT-Bench are built on
it. See [LLM-as-a-judge](llm-as-a-judge.md).

## Human evaluation and its failure modes

Slides 24–27. Ask humans to rate generated text — overall, or on fluency, coherence/consistency,
factuality, commonsense, style, grammaticality, redundancy. The rule printed in red on slide 25:
**never compare human evaluation scores across differently conducted studies, even if they claim to
evaluate the same dimensions.**

The issues, several with numbers attached:

- **Slow** and **expensive**.
- **Inter-annotator disagreement.** In AlpacaFarm, five researchers who spent two or three hours
  writing detailed rubrics agreed on only **67%** of pairwise preferences, against a 50% floor
  (≈36:19).
- **Intra-annotator disagreement** — the same person differs before and after dinner.
- **Not reproducible.** Belz et al.: "just 5% of human evaluations are repeatable in the sense that
  (i) there are no prohibitive barriers to repetition, and (ii) sufficient information about
  experimental design is publicly available for rerunning them."
- **Precision, not recall.** You can judge the generation shown; you cannot judge the ones not
  sampled.
- **Misaligned incentives.** AlpacaFarm budgeted 1.5× California minimum wage from researcher
  timings; annotators worked 2–3× faster and earned 2–2.5×, via shortcuts. The visible shortcut is
  preferring longer answers (≈40:10).

Slide 27 lists the design decisions that all of this rests on: how to describe the task, how to
show it, what metric, how to select annotators, and how to monitor them over time (gold-labelled
items salted into each batch; per-item annotation times).

## The three regimes in current practice

Slide 40 sorts what people actually do into three, with pretrained models getting the first two and
finetuned models the last two.

### Perplexity

Slide 47: bits-per-character correlates with average downstream score at $\rho = -0.940$ across
~17 open models (Llama 1/2, Qwen, Yi, Mistral, Mixtral, Falcon, DeepSeek), and holds separately for
knowledge and commonsense ($-0.933$), coding ($-0.947$) and mathematical reasoning ($-0.951$).

So a lot of teams develop against perplexity alone. The lecturer would not recommend it, "but if you
have to have something quick and dirty it usually works pretty well" (≈1:02:20). The caveats:
perplexity is not comparable **across datasets** and not comparable **across tokenizers**, since
entropy is upper-bounded by the log of the vocabulary size (≈1:03:07).

### "Everything"

Slides 41–46: **HELM** and the **Hugging Face Open LLM Leaderboard** collect many automatically
scorable benchmarks and average across them. The HELM scenario table gives a sense of the mix —
NarrativeQA, NaturalQuestions (open and closed book), OpenbookQA, MMLU, GSM8K, MATH, LegalBench,
MedQA, WMT 2014 — along with who wrote each one (annotators, web users, Mechanical Turk workers,
Upwork and Surge contractors, lawyers, problem setters).

Two capability areas get called out:

- **Code** is unusually well-suited to automatic evaluation, because you can run test cases.
  HumanEval scores **Pass@1** (Pass@k = one of k samples passes); GPT-4 is around 67%. It is also
  worth measuring because coding ability correlates with reasoning (≈58:33).
- **Agents** are unusually badly suited. AgentBoard evaluates web, tool, embodied-AI and game
  environments, but "evaluation needs to be done in sandbox environments" — and once your model
  pings people on Slack or writes emails, you need a sandbox per application (≈1:00:48).

### Arena-like

Slide 48: the Chatbot Arena leaderboard, with Elo, 95% confidence intervals, vote counts, licence
and knowledge cutoff. "Let users decide!" Mechanics and limits at
[LLM-as-a-judge](llm-as-a-judge.md).

## Dimensions other than accuracy

Slides 58–61. Performance is not all that matters, and averaging over examples silently asserts
that every example is worth the same — unfair to minoritized groups, and wrong in general, since
production code and a burger recommendation are not equally valuable (≈1:13:05).

- **Computational efficiency.** MLPerf inverts the usual framing: fix a quality target (75.90%
  ImageNet classification, 0.058 LibriSpeech WER, 0.72 masked-LM accuracy for BERT-large) and
  measure time to reach it.
- **Bias.** DiscrimEval (Anthropic) is template-based: generate a decision scenario, fill in
  `[AGE]`, `[RACE]`, `[GENDER]`, and measure how the decision shifts. The results show positive
  discrimination scores for every group except age, with the largest explicit effects for Black and
  Native American subjects; name-based (implicit) effects are much smaller.
- **Metric bias.** BLEU and ROUGE assume you can find word boundaries, which fails for languages
  without spaces — the lecturer's example is Thai versus Vietnamese (≈1:14:37).
- **Judge bias.** If GPT-4 labels everything, its preferences scale across the field. OpinionQA and
  GlobalOpinionQA (slides 62–63) compare LLM output distributions to Pew survey responses to ask
  *whose* opinions a model reflects by default; base models are not strongly tied to one group,
  finetuned models are — plausibly reflecting where the SFT and RLHF annotators were (≈1:16:58).
  This is the evaluation-side view of the labeler demographics in
  [reward modeling](reward-modeling.md).

## Multilingual coverage

Slides 56–57. Of 461 ACL 2021 oral papers, 69.4% evaluate only on English and 38.8% only on
accuracy/F1; 6.1% evaluate on more than one dimension at all. The benchmarks exist —

| Benchmark | Coverage |
| --- | --- |
| MEGA | 16 datasets, 70 languages |
| GlobalBench | 966 datasets, 190 languages |
| XTREME | 9 tasks, 40 languages |
| Multilingual LLM Evaluation Benchmark | MMLU / ARC / HellaSwag in 26 languages |

— so "it's not that we don't have the benchmarks, it's that there are no incentives, unfortunately,
in academia, to actually evaluate on those benchmarks. So if you have the chance, use those
benchmarks" (≈1:12:18).

## The takeaway

Slide 65: think about what you evaluate (diversity, difficulty) for closed tasks; use overlap
metrics only in low-diversity settings for open ones; watch for consistency, contamination and
biases. And:

> **In many cases, the best judge of output quality is YOU. Look at your model generations. Don't
> just rely on numbers!**
