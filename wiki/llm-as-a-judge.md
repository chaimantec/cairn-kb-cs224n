# LLM-as-a-judge

Using a strong language model in place of a human annotator to score open-ended output. Covered
in [lecture 12](12-benchmarking.md), slides 28–38 — the lecture's own research area, so the
numbers come with unusual detail.

## The problem: evaluating a chatbot

Slide 28. ChatGPT-style assistants are hard to evaluate for two compounding reasons: the use-case
distribution is enormous, and the responses are long-form text. The InstructGPT API prompt
distribution gives the spread:

| Use case | Share |
| --- | --- |
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

No reference-based metric survives this. There is no gold answer to "list five ideas for how to
regain enthusiasm for my career".

## The human gold standard: Chatbot Arena

Slide 29. Anyone can go to the site, ask two anonymous models the same question, keep chatting
until they can pick a winner, and vote. Votes are discarded if a model reveals its identity.
200,000+ human votes are pooled into an **Elo** rating — chess-style, so not every model has to
play every other model.

Slide 30 lists what is missing:

- **External validity.** "Typing random questions into a head-to-head website may not be
  representative." The lecturer's own hedge is that at this sample size it becomes reasonably
  representative — "it's probably better than whatever we have, but it is still not ideal"
  (≈44:46).
- **Cost.** Human annotation at this scale takes a large community effort, new models take a long
  time to appear, and only notable models get benchmarked at all. "You will never have, for your
  random model, 200,000 people who are willing to annotate it for free" (≈45:33).

The cost objection is decisive for the stage-dependent argument in
[evaluating LLMs](evaluating-llms.md): even the largest labs cannot use the Arena during
development. It is a model-selection instrument, used once at the end.

## Replacing the annotator with a model

Slide 31: give the two outputs to an LLM and ask which is better. "Use a LM as a reference-free
evaluator. Surprisingly high correlations with human." Common instantiations: **AlpacaEval**,
**MT-Bench**.

### It is cheaper, faster and — surprisingly — more accurate

Slide 32, from AlpacaFarm. Plotting human agreement against price and against time, with humans
and nine LLM annotator configurations:

- Humans: ~65.7% agreement, at roughly $300 per 1000 examples and ~10⁴·⁵ seconds.
- Best GPT-4-based annotators: ~68.5% agreement, at ~$10 per 1000 examples and ~10³ seconds.

That is **~100× cheaper and ~100× faster with *higher* agreement than humans** (≈46:19). The slide
also notes the same annotator can be used for RLAIF — see [RLHF](rlhf.md).

The measurement being reported is a leave-one-out one: take a pool of four humans, remove one,
and compare that person's preferences against the mode of the other three. That agreement is
*lower* than a model's agreement with the same mode.

### Why: bias versus variance

Slide 33 explains the surprise. Decomposed into bias and variance:

- Humans have **zero bias by definition** (they are the reference) but **very high variance** —
  the dashed line sits at ~0.34.
- GPT-4 has substantial bias — around 0.33 — but far lower variance, ~0.09.

"Humans have low agreement because of variance!" A model is consistent; a person is not. The
lecturer's read (≈48:37): "which is actually a good sign, because that makes it much easier for
research. The bad sign is that the bias is still high."

## AlpacaEval

Slide 35. Built as an internal development benchmark for Alpaca, because the team "did not trust
many of the benchmarks out there at this point for instruction following", and then became its own
thing (≈50:54). In numbers: **98% correlation with Chatbot Arena, under 3 minutes and under $10**
per model.

The procedure:

1. For each instruction, generate an output from the baseline and from the model being evaluated.
2. Ask GPT-4 for the **probability** that the model's output is better.
3. *(AlpacaEval LC)* Reweight the win probability based on output length.
4. Average the win probabilities to get a **win rate**.

Slide 36 gives the system-level correlations against Chatbot Arena Elo:

| Metric | Spearman ρ |
| --- | --- |
| Output Length | 0.35 |
| TruthfulQA | 0.51 |
| HellaSwag | 0.59 |
| GSM-8K | 0.63 |
| Open LLM | 0.66 |
| WinoGrande | 0.69 |
| ARC-C | 0.83 |
| MMLU | 0.87 |
| MT-Bench | 0.94 |
| **LC AlpacaEval 2.0** | **0.98** |

Note the top of the table as well as the bottom: **raw output length alone correlates 0.35 with
Arena Elo**, which is the whole reason step 3 exists.

## The biases, and what can be done about them

Slide 34 lists the spurious correlations, which afflict human and LLM annotators alike:

- **Length.** Both humans (~70%) and models prefer longer outputs at similar rates.
- **Lists.** Both prefer list-formatted answers, ~60% of the time.
- **Position.** Real, but "everyone randomizes this away".
- **GPT-4 self-bias.**

### Length control

Slide 37 is the demonstration that makes length a first-order problem, not a nuisance. Prompt the
*same* GPT-4 to be concise, standard or verbose and its AlpacaEval win rate moves 22.9 → 50.0 →
64.3. "It really doesn't fit our mental model of what benchmarks should be doing — if I just tweak
the prompt a little bit, I don't want my model to change completely its ranking" (≈53:11).

Length-controlled AlpacaEval asks what the metric *would* be if the baseline and model outputs had
the same length, and compresses the spread:

| Model | AlpacaEval concise / standard / verbose | Length-controlled |
| --- | --- | --- |
| gpt4_1106_preview | 22.9 / 50.0 / 64.3 | 41.9 / 50.0 / 51.6 |
| Mixtral-8x7B-Instruct-v0.1 | 13.7 / 18.3 / 24.6 | 23.0 / 23.7 / 23.2 |
| gpt-3.5-turbo-1106 | 7.4 / 9.2 / 12.8 | 15.8 / 19.3 / 22.0 |
| alpaca-7b | 2.0 / 2.6 / 2.9 | 4.5 / 5.9 / 6.8 |

This is the lecture's model of what to do about a known spurious correlation in general: don't
hope it averages out, estimate the counterfactual and control for it.

### Self-bias

Slide 38 measures whether GPT-4 flatters itself. Win rates by auto-annotator (columns) for each
evaluated model (rows):

| | gpt4_1106_preview | claude-3-opus | mistral-large-2402 |
| --- | --- | --- | --- |
| gpt4_1106_preview | 50.0 | 50.0 | 50.0 |
| claude-3-opus | 40.4 | **43.3** | 47.5 |
| mistral-large-2402 | 32.7 | 28.2 | **45.5** |
| gpt4_0613 | 30.2 | 20.5 | 34.3 |
| gpt-3.5-turbo-1106 | 19.3 | 16.7 | 28.9 |

Self-bias exists — Claude scores itself 43.3 versus 40.4 under GPT-4, Mistral 45.5 versus 32.7 —
but the **ranking is unchanged** whichever judge you use. "Even though it's true that if I look at
Mistral evaluated by Mistral it gives itself a much higher accuracy, it still prefers Claude and
GPT-4. So it's not as bad as what you may think — it's still bad though" (≈54:43).

## The monoculture objection

The deeper worry is not any single bias but that everyone shares the same judge. If GPT-4 labels
the field's outputs and GPT-4 has a preference, that preference is scaled uniformly across every
lab (slide 61).

A student asks the converse: if you inspect a subset of GPT-4's judgments yourself to check them,
aren't you injecting *your* bias into a process a controlled experiment would blind you from? The
answer (≈1:21:35):

> "I actually feel less concerned about biases of a single person. My issue with the GPT-4 biases
> is that it's the same across every model, so things really scale up and it becomes a monoculture,
> and I think that's much worse than if everyone incorporates a little bit of the biases that they
> have in their direction."

## The constructive version

The lecture's closing suggestion (≈1:23:11) is that we currently under-specify the judging task. We
ask GPT-4 "how good is the summary, out of five", and it fills the gap with preferences inherited
from its own training data. The better instrument is a **detailed rubric** listing what has to be
in a good answer:

> "This is exactly what professors do when they evaluate for a class: they basically say, okay,
> Yann is a TA but I cannot trust him blindly, so what I will do is write a very detailed rubric
> and I trust that he can apply that rubric. And I think that's also how we should be thinking
> about GPT-4 — and this is not how we currently do it."

And the standing advice regardless of judge (slide 65): **look at your model generations, don't
just rely on numbers.** The lecturer's own case is that AlpacaEval's numbers were believed, but it
was playing with Alpaca that convinced the team it was worth pursuing, "even though on maybe
standard academic benchmarks it was pretty bad" (≈1:19:16).
