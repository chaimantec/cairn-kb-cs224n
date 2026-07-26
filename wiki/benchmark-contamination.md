# Benchmark contamination, overfitting and inconsistency

The three ways a benchmark number can be meaningless even when it is correctly computed. Covered
in [lecture 12](12-benchmarking.md), slides 49–55 and 64.

## Consistency: the number depends on how you ask

Slide 50 (Alzahrani et al., 2024). Take a multiple-choice question — "What is the capital of Saudi
Arabia?" — and pose it two ways: once with the usual A/B/C/D labels and a fixed correct answer,
once with rare symbols (`œ`, `§`, `э`, `ü`) as the option labels. Rank eleven models under each.
The two rankings are drawn side by side with lines connecting each model, and **the lines cross
repeatedly**: rank correlation with the reference ordering is $k_\tau = 0.73$ under one format and
$0.53$ under the other. Mistral-7b, Yi-6b and Llama2-7b-chat all move several places.

So even the simplest possible task — pick one of four — "will be very dependent on exactly how you
format these choices" (≈1:03:54).

### The MMLU case

Slide 51 is the consequential version, because MMLU is the benchmark the field quotes. For roughly
a year there were **three incompatible implementations** in circulation and people were comparing
numbers across them:

| Model | MMLU (HELM) | MMLU (Harness) | MMLU (Original) |
| --- | --- | --- | --- |
| llama-65b | **0.637** | 0.488 | **0.636** |
| tiiuae/falcon-40b | 0.571 | **0.527** | 0.558 |
| llama-30b | 0.583 | 0.457 | 0.584 |
| EleutherAI/gpt-neox-20b | 0.256 | 0.333 | 0.262 |
| llama-13b | 0.471 | 0.377 | 0.47 |
| llama-7b | 0.339 | 0.342 | 0.351 |
| tiiuae/falcon-7b | 0.278 | 0.35 | 0.254 |

llama-65b scores 0.637 or 0.488 depending on the harness — and under the Harness column, which is
the implementation Hugging Face's leaderboard used, falcon-40b **overtakes** it.

The divergence has two sources:

1. **Different prompts**, which trivially change the answer.
2. **Different ways of extracting the prediction**, which do not look like they should matter but
   do. The slide walks three:
   - **Most likely valid choice.** Restrict attention to the tokens `A`, `B`, `C`, `D` and take the
     argmax — constrained decoding. The model gets the point for `D` even though its actual
     highest-probability continuation was the word "Zygote".
   - **Most likely token.** Don't constrain. Now the same model answers "Zygote" and scores zero.
   - **Probability of the generated answer.** Score the full option strings ("The first pharyngeal
     arch", "The second pharyngeal arch", …) by log likelihood and pick the likeliest — which can
     select a different option again.

The lecturer notes the columns have since converged, "but at that time they weren't" (≈1:06:13).
The general lesson is the same one [chain-of-thought](chain-of-thought.md) illustrates from the
other side, where trigger phrases meaning the same thing span 45.7 to 82.0 accuracy: these models
are far more sensitive to surface form than a benchmark number suggests.

## Contamination: the test set was in the training data

Slide 52 gives two public examples:

- **Horace He on GPT-4 and Codeforces.** Of the easiest Codeforces problems, it solved **10/10
  pre-2021** problems and **0/10 recent** ones. "This strongly points to contamination."
- **Susan Zhang on Phi-1.5 and GSM8K.** Truncate a GSM8K question, feed it to Phi-1.5, and it
  autocompletes the rest of the problem correctly — then answers correctly when the numbers are
  changed too.

Two structural reasons this is hard to rule out, both stated on the slide: models are pretrained on
so much data that even *with* access you could not easily check, and the frontier models are
closed-source, so you do not have access. "Closed models + pretraining: hard to know that benchmarks
are truly 'new'."

### Detectors

Slide 55 gives two heuristics for testing contamination from the outside:

- **Min-k% prob.** Take the text $X$, get per-token probabilities under the model, select the $k\%$
  lowest-probability tokens, and average their log-likelihoods:
  $\frac{1}{4}\sum_{x_i \in \text{min-}k\%} \log p(x_i \mid \cdot)$. If even the *least* likely
  tokens are unusually probable, the model has probably seen the text. The lecture is candid that
  "what is too high?" has no principled answer — "often heuristic."
- **Exchangeability test.** Score a test set in its canonical order and in a shuffled order. If the
  model has memorized the set, it expects example two to follow example one, so log-probabilities
  drop under shuffling. "Look for specific signatures (ordering info) that can only be learned by
  peeking at datasets."

## Overfitting: benchmarks saturate faster and faster

Slide 53 plots normalized performance against year for six benchmarks, with human level at 0.0.
MNIST (1998) took roughly fifteen years to cross; Switchboard similar; ImageNet (2009) about six;
SQuAD 1.1 (2016), SQuAD 2.0 (2018) and GLUE (2018) each about a year. "Reach 'human-level'
performance too quickly."

The lecture is careful not to attribute this solely to contamination — it may just be that a lot of
people are doing hyperparameter tuning against the test set (≈1:08:30). Either way it is
overfitting.

### Mitigations

Slide 54 gives two:

**Private test sets** — control how many times anyone can see the test data. The example is
**GSM1k**, a fresh resampling of GSM8K released about two weeks before the lecture. Plotting GSM1k
accuracy against GSM8k accuracy for models scoring above 70%, most points fall *below* the diagonal:
Mixtral-8x22B-Instruct drops 85.6 → 76.0, Meta-Llama-3-70B-Instruct 89.0 → 87.6. GPT-4 (91.1 → 91.0),
GPT-4-Turbo (89.8 → 89.8) and Claude-2.1 (88.7 → 89.4) essentially do not move. The gap is
concentrated in open-source models.

**Dynamic test sets** — constantly change the inputs. **DynaBench** does this as a loop: a writer
proposes examples, the model predicts, a verifier checks the cases where the model is wrong, verified
examples are split into train/dev/test, and the model is retrained for the next round. Chatbot Arena
gets this property for free, since users keep typing new prompts.

## The status quo problem

Slide 64 is the meta-issue, and the lecture calls it "the challenge of all challenges": even where
better metrics exist, nobody is incentivized to switch. The chart shows BLEU appearing in essentially
100% of machine translation publications every year from 2010 to 2020, with every other metric —
TER, METEOR, RIBES, NIST, chrF — under ~25% and human evaluation under ~15%. **82% of 2019–2020
machine translation papers evaluate on BLEU and nothing else**, despite metrics that correlate better
with human judgment.

The incentive is circular: you want to compare against work from three years ago, and reviewers ask
for the number they know. "So it's not even that you're incentivized not to look at something else,
you're also incentivized to continue" (≈1:17:44).

The escape hatch is that this is an academic problem, not a universal one:

> "This is really specific to academia — like in reality, if you know that your metric is bad, just
> switch." (≈1:18:30)

## Related

- [Evaluating LLMs](evaluating-llms.md) — what the benchmarks are and what the three current
  evaluation regimes measure.
- [LLM-as-a-judge](llm-as-a-judge.md) — the length, position and self-bias problems in model-based
  evaluation, and how AlpacaEval controls for one of them.
- [Evaluating NLG](evaluating-nlg.md) — the metric-level view from
  [lecture 10](10-natural-language-generation.md).
