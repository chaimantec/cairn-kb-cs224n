# Evaluating NLG

How to tell whether generated text is any good — and why, as of this lecture, nobody has a
satisfying answer. Covered in [lecture 10](10-natural-language-generation.md), slides 49–67.

Slide 50 divides the field into three families, in increasing order of both cost and
trustworthiness: **content overlap metrics**, **model-based metrics**, and **human
evaluations**.

## Content overlap metrics

These score the lexical similarity between the generated text and a gold, human-written
reference. Slide 51's running example:

> Ref: They walked **to the** grocery **store .**
> Gen: **The woman went** to the **hardware** store .

with the shared spans matched up. The family is the $n$-gram overlap metrics —
[BLEU](evaluating-machine-translation.md), ROUGE, METEOR, CIDEr. They are fast, efficient and
widely used.

### They degrade with open-endedness

Slide 52's ladder tracks the
[open-endedness spectrum](natural-language-generation.md#the-open-endedness-spectrum) exactly:

- **Not ideal even for machine translation.**
- **Worse for summarization**, as longer output texts are harder to measure.
- **Much worse for dialogue**, which is more open-ended than summarization.
- **Much, much worse for story generation**, "which is also open-ended, but whose sequence
  length can make it seem you're getting decent scores!"

That last clause is the trap the lecturer expands on (≈56:18): a long generated story shares a
lot of vocabulary with any reference story simply by being long, so the $n$-gram score can look
respectable "not because it's accurate or of high quality, just because you are talking so much
that you might have covered lots of points already."

They survive as the standard in MT because MT sits at the constrained end of the spectrum,
where lexical overlap is a defensible proxy.

### The failure case

Slide 53 is the clearest single demonstration in the lecture. Chris Manning asks "Are you
enjoying the CS224N lectures?" and the reference answer is **"Heck yes!"** Four candidate
responses are scored:

| Score | Candidate | |
| --- | --- | --- |
| 0.61 | Yes ! | |
| 0.25 | You know it ! | |
| **0** | Yup . | **false negative** |
| **0.67** | Heck no ! | **false positive** |

"Yup." is a correct answer and scores **zero**, because it shares no words with the reference.
"Heck no!" means the opposite and scores **highest of all**, because it shares "Heck" and the
exclamation mark. As slide 53 puts it: *$n$-gram overlap metrics have no concept of semantic
relatedness.*

## Model-based metrics

The response is to replace $n$-gram matching with learned representations, so that semantic
similarity can be measured directly. Slide 55: no more $n$-gram bottleneck, because text units
are embeddings; the embeddings are **pretrained** and the distance function can be **fixed**.

Slide 56 lists the word-level distance metrics:

- **Vector similarity** — embedding-based semantic distance, via cosine or Euclidean distance.
  Variants: Embedding Average and Vector Extrema (Liu et al., 2016), MEANT (Lo, 2017), YISI (Lo,
  2019). This is the same operation as CS224N's Homework 1, lifted from words to sentences
  (≈58:37).
- **Word Mover's Distance** (Kusner et al., 2015; Zhao et al., 2019) — uses optimal transport to
  find the best alignment between the two sentences' word embeddings. The classic figure aligns
  "Obama speaks to the media in Illinois" with "The President greets the press in Chicago":
  *Obama* → *President*, *speaks* → *greets*, *media* → *press*, *Illinois* → *Chicago*. Two
  sentences with almost no lexical overlap, correctly judged similar.
- **BERTScore** (Zhang et al., 2020) — computes pairwise cosine similarity between the
  contextual [BERT](bert.md) embeddings of candidate and reference tokens, takes the maximum
  similarity per reference token, and optionally weights by idf.

Slide 57 moves past word alignment:

- **Sentence Mover's Similarity** (Clark et al., 2019) — Word Mover's Distance applied to
  sentence embeddings from RNN representations, so alignment happens at the sentence level.
- **BLEURT** (Sellam et al., 2020) — a *regression* model on top of BERT, trained to return a
  score indicating how grammatical the candidate is and how well it conveys the reference's
  meaning. Evaluation itself becomes a supervised learning problem (≈1:00:54). Slide 57's plots
  show pretrained BLEURT holding up far better than BLEU as the test set's skew increases.

## MAUVE: evaluating open-ended generation

All of the above measure similarity to a reference, which is the wrong question for open-ended
tasks — "a story can be perfectly fluent and perfectly high quality without having to resemble
any of the reference stories" (≈1:01:40).

**MAUVE** (slides 58–59; Pillutla et al., 2022) compares *distributions* instead. It computes
the information divergence between the generated text and the gold reference text in a
**quantized embedding space**. The procedure as the lecture describes it (≈1:02:26):

1. Embed a batch of human-written text and a batch of model-generated text.
2. Cluster the embedding space with k-means to **discretize** it.
3. Build a histogram over the discrete cells for each batch.
4. Compute precision as a forward KL divergence and recall as a backward KL.

The two error types have interpretations, shown on slide 58: **Type I** is the model putting
mass where humans don't — degenerate repetition, "The time is the time is the time…" — and
**Type II** is the model failing to cover regions humans do use. Slide 58's right-hand panel
plots the divergence frontier for GPT-2 large under three decoding algorithms, with **nucleus
sampling** bowing furthest out, then plain sampling, then greedy — which is MAUVE reproducing
the [decoding](decoding-algorithms.md) results as a distributional statement.

**Why discretize?** A finite set of embedded sentences is a set of Dirac deltas in a continuous
manifold, and no divergence can sensibly be computed over that. Smoothing would work, but only
under assumptions — Gaussian, say — and "who said word representations are Gaussian-related?"
(≈1:03:58).

MAUVE's advantage is that it is applicable to open-ended settings and measures **both precision
and recall** against the target distribution, which no similarity metric does. Asked how it
differs from maximizing similarity, the answer is that it occupies a middle ground: on MT it
would score very high, on open-ended generation a similarity metric scores near zero and stops
being informative, "so it's just not really measurable, because everything's so different from
each other" (≈1:05:29).

## How to evaluate an evaluation metric

Slide 60 asks the recursive question, and the standard answer is **correlation with human
judgment**. The figure (Liu et al., EMNLP 2016) plots BLEU-2, embedding average, and
human-versus-human agreement against human score, on the Twitter and Ubuntu dialogue corpora.
The BLEU-2 panels are near-formless clouds; the human-versus-human panels are a tight diagonal.
Humans agree with each other; the automatic metric does not agree with humans. This is the
evidence that BLEU is not a good metric for dialogue.

## Human evaluation

Slide 61: automatic metrics fall short of matching human decisions, so human evaluation is both
the most important form of evaluation *and* the gold standard against which new automatic
metrics must be validated.

Slide 62 lists the axes annotators are typically asked about — fluency, coherence/consistency,
factuality and correctness, commonsense, style/formality, grammaticality, typicality,
redundancy — overall or one dimension at a time. It also carries a rule in a red box:

> **Don't compare human evaluation scores across differently conducted studies. Even if they
> claim to evaluate the same dimensions!**

Slide 63 and the surrounding discussion enumerate why human evaluation is nonetheless far from
perfect (≈1:09:20):

- **Slow and expensive**, which is the obvious cost.
- **Inconsistent and not reproducible.** "If you ask the same human whether they like A or B,
  they might say A the first time and B the second time."
- **Can be illogical.**
- **Annotators misinterpret the question.** Asked for "coherence," some people check
  grammaticality, others check whether the continuation stays on topic — two different
  measurements reported under one name.
- **Precision, not recall.** You can show a person a sentence and ask if it is good. You cannot
  ask whether the model is capable of producing *every* good sentence.

## Combining humans and models

Slide 64 shows two hybrid approaches:

- **ADEM** (Lowe et al., 2017) — learn a metric *from* human judgments, training a model on
  human-scored dialogue data to simulate human judgment.
- **HUSE** (Hashimoto et al., 2019) — Human Unified with Statistical Evaluation, comparing the
  model's output distribution to a human reference distribution. The division of labour the
  lecture draws out (≈1:10:53): the human evaluates precision, the model evaluates recall.

Slide 65 goes further, toward evaluating models **interactively** (Lee et al., 2022; the HALIE
framework). Prior work has a third party judge the quality of the output; this work expands
along three axes — capturing the whole *process* rather than only the final *output*, capturing
the *first-person* experience of the user rather than only a *third-party* view, and considering
*preference* alongside *quality*. The motivation is that when a person co-writes with a model,
how the interaction felt is part of what you are evaluating (≈1:10:53).

## Takeaways

Slide 67:

- **Content overlap metrics** are a reasonable starting point and **not good enough on their
  own**.
- **Model-based metrics** correlate better with human judgment, but their behaviour is **not
  interpretable**.
- **Human judgments** are critical — and humans are inconsistent.
- **In many cases the best judge of output quality is you.** Look at your model's generations;
  don't just rely on numbers. Publicly release large samples of your system's output.

The lecturer flags the third bullet as direct advice for CS224N final projects: "if you want to
do a final project in natural language generation, you should look at the model output yourself,
and don't just rely on the numbers that are reported by BLEU score or something" (≈1:11:39). See
[final project guidance](final-project-guidance.md).

Slide 76 closes on the state of the problem: "Evaluation remains a huge challenge. We need
better ways of automatically evaluating performance of NLG systems."

## Related pages

- [Lecture 10 — Natural Language Generation](10-natural-language-generation.md) — the lecture.
- [Evaluating machine translation](evaluating-machine-translation.md) — BLEU in detail, from
  lecture 7.
- [Natural language generation](natural-language-generation.md) — the open-endedness spectrum
  that predicts where these metrics break.
- [Decoding algorithms](decoding-algorithms.md) — what MAUVE is comparing when it ranks nucleus
  sampling above greedy.
- [Exposure bias and teacher forcing](exposure-bias-and-teacher-forcing.md) — what goes wrong
  when a metric becomes a training reward.
- [Perplexity](perplexity.md) — the intrinsic metric these extrinsic ones supplement.
- [Evaluating word vectors](evaluating-word-vectors.md) — the course's earlier treatment of
  intrinsic versus extrinsic evaluation.
- [BERT and masked language modeling](bert.md) — the embeddings BERTScore and BLEURT are built
  on.
- [Evaluating LLMs](evaluating-llms.md) — how this taxonomy is used in practice, from
  [lecture 12](12-benchmarking.md), including the finding that reference-based metrics are only as
  good as their references.
- [LLM-as-a-judge](llm-as-a-judge.md) — the reference-free family that replaced these metrics for
  chatbot evaluation.
- [Benchmark contamination and overfitting](benchmark-contamination.md) — why a correctly computed
  number can still be meaningless.
- [Final project guidance](final-project-guidance.md).
