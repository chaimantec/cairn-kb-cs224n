# Lecture 12 — Benchmarking and evaluation

The lecture that asks how you know any of the preceding ten lectures worked. Yann Dubois
argues that evaluation is "something that I think not enough people look at in academia, but
if you really want to put something in production… evaluation is really key" (≈0:05), then
works through the whole space: why different stages of a project need *different* evaluations,
how close-ended tasks are evaluated (standard machine learning, done carelessly), how
open-ended generation is evaluated (badly), how chatbots are evaluated now (with other
chatbots), and what is wrong with all of it — consistency, contamination, overfitting,
monoculture and bias.

**Slide-by-slide text of this deck: [65 slides](../raw/slides/12-benchmarking.md)** — printed
slide numbers match PDF pages 1:1.

Slides PDF: [Lecture 11 — benchmarking and evaluation](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture11-evaluation-yann.pdf) ·
[Full transcript](../raw/transcripts/12-benchmarking.md)

> **A note on numbering.** This lecture sits at **position 12** in the playlist this knowledge
> base follows; the video title and deck both call it "Lecture 11". Repo files use the
> position. The lecturer is **Yann Dubois**, an author of AlpacaFarm and AlpacaEval, both of
> which appear as worked examples.

## There is no single "good" evaluation

The opening move (slide 3) is the one that organizes everything else. A machine learning
project passes through five stages, and each one wants something different from its metric:

| Stage | What the evaluation must be |
| --- | --- |
| **Train** | Super fast, super cheap, **differentiable**, no shortcut |
| **Develop** | Super fast, super cheap, avoids shortcuts |
| **Model selection** | Fast, cheap |
| **Deploy** | **Trustworthy**, task-specific, **absolute** |
| **Publish** | Standardized, reproducible, easy to work with, broad coverage; crude metrics may be fine |

Two of those entries carry most of the weight. *Absolute* matters because at deployment you
need a threshold — "if I have less than 95% accuracy I'm not putting my model in production" —
whereas at every other stage you only ever compare (≈3:56). And *crude metrics may be fine*
is a claim specifically about academia: what matters over a decade is that the metric points
in the direction the field is actually improving, not that any two adjacent numbers are
meaningfully different (≈5:28).

Academic benchmarks also have a constraint that production metrics do not: they must be
neither too hard nor too easy. Too complicated and every method scores at random; too simple
and nobody can beat the baseline. Either way the benchmark goes unused (slide 6).

Slide 4 makes the case that this matters: MMLU went from ~25% (random, for four-way multiple
choice) to ~90% in four years, and that curve *is* how the field measures its own progress.

## Close-ended evaluation

A close-ended task has a limited number of candidate answers — "think, like, less than 10" —
and usually one correct one (slide 7). Sentiment analysis (SST, IMDB, Yelp), entailment
(SNLI), NER (CoNLL-2003), part-of-speech (Penn Treebank), coreference (WSC) and extractive QA
(SQuAD 2) are the examples on slides 8–9; SuperGLUE (slides 10–11) aggregates eight of them
into a single "general language capabilities" number.

The lecture's position is that this is ordinary machine learning — "there's nothing special
about NLP here" — but that this is not the same as easy (≈9:16). Three traps, from slides
12–13:

- **Metric choice changes which algorithm wins.** The worked example is spam: if 90% of email
  is legitimate, a classifier that always predicts "not spam" scores 90% accuracy while
  classifying nothing. Hence precision, recall and F1 (≈13:59).
- **Aggregation across tasks is mostly indefensible.** SuperGLUE's columns are Matthews
  correlation, F1a/EM, average F1, accuracy and gender-parity/accuracy — and the headline score
  averages them. The lecturer recalls a benchmark where one column was *better when lower* and
  was averaged in anyway "until someone realized, like, maybe we should put a minus there"
  (≈14:45).
- **Spurious correlations may be undiscovered.** Slide 13: on SNLI, a model that looks *only*
  at the hypothesis and ignores the premise does well, because annotators asked to write a
  non-entailed hypothesis reach for negation. The cue word *never* is doing the work
  (Gururangan et al., 2019).

The moral is stated flatly: "don't always think that what people do in academia is correct"
(≈15:31).

Full detail on the metrics themselves is at [evaluating LLMs](evaluating-llms.md); the
generation-side story continues on [evaluating NLG](evaluating-nlg.md) from
[lecture 10](10-natural-language-generation.md).

## Open-ended evaluation

An open-ended task has too many correct answers to enumerate, *and* a continuum of quality
rather than a right/wrong split (slide 15). Summarization (CNN/DailyMail), translation (WMT)
and — now the dominant case — instruction following (Chatbot Arena, AlpacaEval, MT-Bench).

Instruction following is singled out as "the mother of all tasks": any earlier task can be
posed to a chatbot as a question, so a chatbot evaluation subsumes all of them (≈19:22).

Slide 16 gives the three families of method — **content overlap metrics**, **model-based
metrics**, **human evaluations** — which is the same taxonomy
[lecture 10](10-natural-language-generation.md) used. This lecture adds two things to it: a
much harder look at *references*, and the LLM-judge family that did not exist when the earlier
lecture was recorded.

### Overlap metrics and the reference problem

BLEU (precision-flavoured) and ROUGE (recall-flavoured) are $n$-gram overlap metrics, fast and
still standard for translation and summarization. Slide 18's failure case is the one to
remember. Chris Manning asks *"Are you enjoying the CS224N lectures?"*; the gold answer is
*"Heck yes!"*:

| Candidate | Score | |
| --- | --- | --- |
| Yes ! | 0.67 | |
| You know it ! | 0.25 | |
| Yup . | 0 | **false negative** — right meaning, no shared words |
| Heck no ! | 0.67 | **false positive** — opposite meaning, two shared words |

Model-based metrics (slides 19–21) replace string matching with learned representations:
embedding-average and vector-extrema similarity, then **BERTScore** (greedy cosine matching
between BERT embeddings of candidate and reference, optionally idf-weighted), then **BLEURT**,
a BERT regression model finetuned on human ratings. A student immediately spots the tension —
if BLEURT's continual pretraining target is BLEU, does it inherit BLEU's blindness? The answer
(≈27:50) is that BLEU is used only as an unsupervised auxiliary target alongside BERTScore
before the human-rating finetune.

Slide 22 is the finding that reframes the whole family. On XSUM summarization, ROUGE-L computed
against the **dataset's own references** is essentially uncorrelated with human faithfulness
judgments; computed against **summaries written by expert freelance writers**, it correlates
clearly. The metric was not the problem. "Reference-based measures are only as good as their
references" — and news-summarization references are the article's own bullet points (≈29:24).

### Human evaluation, and why it is not a solution

Human judgment is the gold standard for open-ended tasks and the yardstick against which every
new automatic metric is validated (slide 24). It is also slow, expensive, and — slides 25–27 —
unreliable in specific, documented ways:

- **Inter-annotator disagreement.** In AlpacaFarm, *five researchers* who spent two or three
  hours writing detailed rubrics still agreed only **67%** of the time on pairwise preferences,
  against a 50% chance floor (≈36:19).
- **Intra-annotator disagreement.** The same person scores differently before and after dinner.
- **Not reproducible.** Belz et al. find that "just 5% of human evaluations are repeatable in
  the sense that (i) there are no prohibitive barriers to repetition, and (ii) sufficient
  information about experimental design is publicly available for rerunning them."
- **Precision, not recall.** A human can judge the generation you show them; they cannot judge
  the generations you did not sample.
- **Misaligned incentives.** AlpacaFarm paid 1.5× California minimum wage per example, budgeted
  from how long the researchers took; annotators worked two to three times faster and so earned
  2–2.5×, by finding shortcuts (≈40:10). The shortcut that shows up in the data is preferring
  longer answers.

And a rule with no exceptions: **never compare human evaluation scores across studies**, even
when they claim to measure the same dimension (slide 25).

## Evaluating chatbots

Slides 28–38 are the lecture's core contribution, and have their own page:
[LLM-as-a-judge](llm-as-a-judge.md). In outline:

**Chatbot Arena** (slide 29) is the current human gold standard — anonymous side-by-side chats,
a vote, an Elo rating, 200,000+ votes. Its limits are external validity (people typing whatever
occurs to them into a website) and, decisively, **cost**: only notable models ever get
benchmarked, and never in time to guide development (slide 30).

**LLM evaluators** replace the annotator with GPT-4. The AlpacaFarm measurements (slides 32–33)
are the surprise: ~100× cheaper, ~100× faster, and *higher agreement with human majority than
individual humans achieve*. The explanation is a bias–variance decomposition — humans have zero
bias by definition but very high variance, while GPT-4 has ~32% bias and far less variance.

**AlpacaEval** (slides 35–37) operationalizes it: generate baseline and candidate outputs, ask
GPT-4 for the probability the candidate is better, length-correct, average into a win rate. It
correlates 0.98 with Chatbot Arena at ~3 minutes and <$10 per model.

**The biases are measurable and partly fixable.** Length is the big one — prompting GPT-4 to be
verbose moves its own AlpacaEval win rate from 50.0 to 64.3, and prompting it to be concise
drops it to 22.9; length control compresses that spread to 51.6/50.0/41.9 (slide 37). Position
bias is randomized away. Self-bias exists but is smaller than expected: Claude scores itself
43.3 versus 40.4 under GPT-4, and the *ranking* is unchanged whichever judge you use (slide 38).

## How LLMs are actually evaluated today

Slide 40 sorts current practice into three regimes: **perplexity**, **everything**, and
**arena-like**. Pretrained models get the first two; finetuned models get the last two.

- **Perplexity** (slide 47) is startlingly predictive: bits-per-character correlates with
  average downstream score at $\rho = -0.940$ across ~17 open models, and holds within knowledge,
  coding and mathematical reasoning separately. The caveats are that perplexity is not
  comparable across datasets or across tokenizers, since entropy is upper-bounded by the log of
  the vocabulary size (≈1:03:07).
- **"Everything"** (slides 41–46) is HELM and the Hugging Face Open LLM Leaderboard: collect
  many automatically-scorable benchmarks — MMLU, GSM8K, MATH, LegalBench, MedQA, HumanEval — and
  average. Code is singled out as unusually well-suited to evaluation, because you can run test
  cases; agents as unusually badly suited, because you need a sandbox for every application the
  model can touch.
- **Arena-like** (slide 48) is the Chatbot Arena leaderboard. "Let users decide!"

Detail at [evaluating LLMs](evaluating-llms.md).

## What is broken

Slides 50–64 are the catalogue, and the page that develops them is
[benchmark contamination and overfitting](benchmark-contamination.md).

**Consistency.** Replacing A/B/C/D with rare symbols reorders the model ranking (Alzahrani et
al., 2024; slide 50). Worse, slide 51 shows that MMLU — the benchmark Mark Zuckerberg quoted
when Llama 3 launched — had *three* mutually incompatible implementations being compared against
each other for about a year: llama-65b scores 0.637 under HELM, 0.636 under the original, and
0.488 under the Harness, which is the one Hugging Face uses.

**Contamination.** Horace He's observation that GPT-4 solved 10/10 easy pre-2021 Codeforces
problems and 0/10 recent ones; Susan Zhang's on Phi-1.5 and GSM8K. With closed models you cannot
even inspect the pretraining data (slide 52).

**Overfitting.** Slide 53 tracks how quickly benchmarks reach "human level": MNIST took over a
decade, GLUE and SQuAD 2.0 took roughly a year. Mitigations on slide 54 — private test sets
(GSM1k, on which open-source models drop noticeably while Claude and GPT-4 do not) and dynamic
test sets (DynaBench, and Chatbot Arena by construction).

**Monoculture.** Of 461 ACL 2021 oral papers, ~70% evaluate only on English and ~40% only on
accuracy; a paper analysing an earlier conference found the same (slide 56). Multilingual
benchmarks exist — MEGA (16 datasets, 70 languages), GlobalBench (966 datasets, 190 languages),
XTREME (9 tasks, 40 languages) — so "it's not that we don't have the benchmarks, it's that there
are no incentives" (≈1:12:18). And the problem runs deeper than data: BLEU and ROUGE presuppose
that you can find word boundaries at all, which fails for Thai (≈1:14:37).

**Reductive single metrics.** Performance is not all we care about — computational efficiency
(MLPerf: time to reach a quality target) and bias (DiscrimEval: template-based, varying the
subject's race, gender and age) are measurable too, and averaging over examples treats a
production code generation and a burger recommendation as equally valuable (slides 58–60).

**Biased judges.** If GPT-4 labels everything and GPT-4 has a bias, that bias is scaled across
the whole field. OpinionQA and GlobalOpinionQA (slides 62–63) ask *whose* opinions LLMs reflect
by default, and find that base models are not strongly aligned to any single group while
finetuned models are — plausibly, the lecturer suggests, because the SFT and RLHF annotators
were largely in Southeast Asia and highly educated (≈1:16:58). That is the same labeler-pool
observation [lecture 11](11-post-training.md) makes about training data, arriving now from the
evaluation side.

**The status quo.** Slide 64: BLEU appears in ~100% of machine translation papers every year
from 2010 to 2020, and 82% of 2019–2020 MT papers evaluate on *nothing else*, despite better
metrics existing. Reviewers ask for it. The incentive is to keep comparing to work from three
years ago — "but this is really specific to academia; in reality, if you know that your metric is
bad, just switch" (≈1:18:30).

## Takeaways

Slide 65, plus the exchange that follows it:

- **Closed-ended tasks** — think about what you are evaluating: diversity, difficulty.
- **Open-ended tasks** — content overlap metrics are useful in low-diversity settings; chatbot
  evaluation is very difficult, and selecting the right examples is an open problem.
- **Challenges** — consistency, contamination, biases.
- **In many cases the best judge of output quality is you.** "Look at your model generations.
  Don't just rely on numbers!" The lecturer's own account: the team believed AlpacaEval's
  numbers, but it was *playing with the model* that convinced them Alpaca was worth pursuing,
  even though it looked poor on standard academic benchmarks (≈1:19:16).

A student presses on the tension in that advice — inspecting outputs yourself injects your bias,
but not inspecting them leaves GPT-4's bias unchecked. The answer (≈1:21:35) is that a single
person's bias is *less* worrying than a judge's, because GPT-4's bias is identical across every
lab that uses it: "things really scale up and it becomes a monoculture." The constructive
version, offered at the very end, is that we under-specify what we ask the judge for — instead of
"how good is this summary, out of five", write the detailed rubric you would hand a TA, and trust
the model to apply it. "This is exactly what professors do… and this is not how we currently do
it" (≈1:23:58).
