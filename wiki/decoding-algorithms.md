# Decoding algorithms

Given a trained language model, decoding is the step that turns its probability distribution
into actual tokens. It is not a detail: the lecture's claim is that "some of the most impactful
advances in NLG of the last few years have come from simple but effective modifications to
decoding algorithms" (slide 37). Covered in
[lecture 10](10-natural-language-generation.md), slides 19–37.

## The setup

At each time step $t$ the model computes a score for every token in the vocabulary,
$S \in \mathbb{R}^{V}$, from the preceding context:

$$S = f(\{y_{<t}\})$$

A [softmax](softmax-and-cross-entropy.md) turns those scores into a distribution:

$$P(y_t = w \mid \{y_{<t}\}) = \frac{\exp(S_w)}{\sum_{w' \in V} \exp(S_{w'})}$$

and the **decoding algorithm** is a function $g$ that selects a token from it:

$$\hat{y}_t = g\big(P(y_t \mid \{y_{<t}\})\big)$$

Everything below is a choice of $g$ (slide 20).

## Maximum-probability decoding, and where it fails

**Greedy decoding** takes the argmax at every step:

$$\hat{y}_t = \arg\max_{w \in V} P(y_t = w \mid y_{<t})$$

**Beam search** (covered in [lecture 6](06-sequence-to-sequence-models.md)) has the same
objective — find the string maximizing log-probability — but explores more candidates, keeping
$k$ partial hypotheses in the beam. Slide 21's summary: **maximum-probability decoding is good
for low-entropy tasks like MT and summarization.** For open-ended generation it fails in two
distinct ways.

### Failure 1: degenerate repetition

Slide 22 shows GPT-2 continuing the unicorn prompt fluently for a sentence and a half, then
looping "Universidad Nacional Autónoma de México" without end.

Slides 23–24 explain the mechanism. Plot negative log likelihood as a phrase repeats — "I don't
know. I don't know. I don't know." — and the loss assigned to each repetition *falls*. The
model becomes more confident the more it has already repeated. Slide 23 names this a
**self-amplification effect** (≈17:00), and slide 24 rules out the two obvious escapes:

- **Not an architecture problem.** Both an LSTM and a Transformer show the same decaying curve.
- **Not a scale problem.** "Scale doesn't solve this problem: even a 175 billion parameter LM
  still repeats when we decode for the most likely string."

### Failure 2: it doesn't look like human text

Slide 26 plots per-token probability over 100 timesteps for beam-search output and for human
writing. Beam search sits pinned near 1.0 for essentially the whole span, dipping only three or
four times. Human text oscillates violently across the full range, repeatedly touching both 0
and 1. Human writing is full of locally surprising choices; an algorithm whose objective is to
maximize probability cannot produce them by construction. As slide 26 puts it, **it fails to
match the uncertainty distribution for human generated text.**

This is the argument that motivates sampling. It also suggests a detection idea, which a
student raises and the lecturer partly deflates — the pattern is only a signal against naive
decoding, since more robust decoding algorithms produce text that fluctuates too (≈21:34).

## Reducing repetition directly

Slide 25 lists the responses, from crude to principled.

**$n$-gram blocking** is the simple heuristic: never emit the same $n$-gram twice. With
$n = 3$, once the text contains "I am happy," any later occurrence of the prefix "I am" has
$P(\text{happy})$ forced to zero. It works, and it is obviously too blunt — it is perfectly
normal for a person's name to appear three times in a passage, and blocking eliminates that
(≈18:32).

More principled options change the objective rather than the output:

- **Unlikelihood objective** (Welleck et al., 2020) — a *training* objective that penalizes
  generating already-seen tokens. In effect, $n$-gram blocking moved from decoding time to
  training time.
- **Coverage loss** (See et al., 2017) — regularizes the [attention](attention.md) mechanism to
  attend to different words at each step, on the reasoning that repetition co-occurs with
  repeated attention patterns.
- **Contrastive decoding** (Li et al., 2022) — a *decoding* objective that searches for strings
  maximizing $\log p_{\text{large}}(x) - \log p_{\text{small}}(x)$. Both models are repetitive,
  so the repetition-loving component cancels and what survives is what the large model knows
  that the small one doesn't (≈20:03).

## Sampling

The alternative to searching for the most likely string is to sample from the distribution:

$$\hat{y}_t \sim P(y_t = w \mid \{y\}_{<t})$$

Slide 27's figure makes the difference concrete: given "He wanted to go to the", greedy
decoding is stuck with *restroom*, but sampling can select *bathroom* well down the ranked
list.

This buys the fluctuation that beam search lacks, at a cost: nothing is ever zeroed out, so
every token in the vocabulary is a viable option, and "in some unlucky cases we might end up
with a bad word." Slide 28 makes the argument carefully. Even with a well-trained model, most
of the probability mass sits on a small set of good options, but the **tail** is enormous —
language is a **heavy-tailed** distribution. Individually each wrong token has tiny
probability; *as a group* they carry considerable mass, and are therefore selected often
(≈23:10).

The fix is to truncate the tail before sampling.

### Top-$k$ sampling

Sample only from the $k$ highest-probability tokens, renormalized (slides 28–29; Fan et al.,
2018; Holtzman et al., 2018). Common values are $k = 50$, "but it's up to you."

- **Increase $k$**: more **diverse** but **risky** output.
- **Decrease $k$**: more **safe** but **generic** output.

Note that top-$k$ is still weighted sampling — it zeroes the tail, then samples proportionally
to the surviving scores, not uniformly among the $k$ (≈25:29). And it does not save compute:
you still evaluate the softmax over the whole vocabulary to find the cutoff, so "it's not
really saving compute, but it's improving performance" (≈30:04).

### Why a fixed $k$ is wrong

Slide 30 gives the two failure cases, which pull in opposite directions:

- **Cuts off too quickly.** After "She said, 'I never'", the distribution is nearly flat —
  *thought, knew, had, saw, did, said, wanted, told, liked, got* are all plausible. A small $k$
  discards viable continuations, hurting **recall**.
- **Cuts off too slowly.** After "I ate the pizza while it was still", the distribution is
  sharply peaked on *hot*. A large $k$ keeps implausible options such as *cold* in play, and
  because their probability is non-zero the model will occasionally emit one, hurting
  **precision**.

No single $k$ is right for both, because the distributions are dynamic (slide 31).

### Top-$p$ (nucleus) sampling

Sample from the smallest set of tokens whose cumulative probability mass reaches $p$ (slides
31–32; Holtzman et al., 2020). Equivalently: **an adaptive $k$**, which grows when the
distribution is flat and shrinks when it is peaked, so the same $p$ keeps roughly the same
amount of *probability* rather than the same *number* of candidates. Slide 32 illustrates this
with three distributions — a moderately peaked one where the nucleus is about five tokens, a
flat one where it is about ten, and an extremely peaked one where it is essentially one.

Nucleus sampling is the algorithm the lecture singles out as an example of a simple decoding
change with outsized impact — "the nucleus sampling paper is actually very, very highly cited"
(≈40:06).

### Further variants

Slide 33 names two more:

- **Typical sampling** (Meister et al., 2022) reweights by the **entropy** of the distribution,
  aiming for tokens whose log-probability is close to the negative entropy of the data
  distribution. The lecturer's sanity check: a closed-ended task has smaller entropy, so you
  want smaller negative log probability, so you want higher-probability tokens — "it kind of
  type-checks very well" (≈31:36).
- **Epsilon sampling** (Hewitt et al., 2022) simply lower-bounds valid probabilities: a token
  below the threshold — 0.03, say — can never be emitted.

## Temperature

Temperature is orthogonal to all of the above: it reshapes the distribution before any
selection happens. Insert a hyperparameter $\tau$ into the softmax (slide 34):

$$P_t(y_t = w) = \frac{\exp(S_w / \tau)}{\sum_{w' \in V} \exp(S_{w'} / \tau)}$$

- **$\tau > 1$** — $P_t$ becomes more **uniform**: more diverse output, mass spread across the
  vocabulary.
- **$\tau < 1$** — $P_t$ becomes more **spiky**: less diverse output, mass concentrated on the
  top words.

Temperature never changes the *ranking* of tokens — if $a$ was more probable than $b$ it still
is — only the relative gaps between them. In the limit $\tau \to 0$ the distribution collapses
to a one-hot vector and sampling reduces to argmax, i.e. greedy decoding (≈34:45).

Slide 34's boxed note: **temperature is a hyperparameter for decoding, and can be tuned for
both beam search and sampling.**

## Re-ranking

Even with truncation, sampling is random, so a single draw can be bad. Slide 36's answer is to
decode several candidates and choose among them:

1. Decode a batch of sequences — "10 candidates is a common number, but it's up to you." The
   trade-off is compute against quality (≈35:33).
2. Score them with a function approximating quality.
3. Return the highest-scoring one.

The obvious score is low [perplexity](perplexity.md), and slide 36 attaches the warning
immediately: **repetitive utterances generally get low perplexity**, which is exactly the
degenerate text this whole section is trying to avoid. Perplexity "is not really robust to
maximize" (≈36:18).

Better re-rankers score properties the model was not optimizing: style (Holtzman et al., 2018),
discourse coherence (Gabriel et al., 2021), entailment and factuality (Goyal et al., 2020),
logical consistency (Lu et al., 2020). Slide 36 adds two practical notes — beware
poorly-calibrated re-rankers, and re-rankers **compose**: add a style score to a factual-consistency
score and re-rank on the sum to get text that is good at both (≈37:50).

## Choosing

The organizing principle is the open-endedness of the task (see
[natural language generation](natural-language-generation.md)):

| Task type | Sensible decoding |
| --- | --- |
| Low entropy — MT, summarization | greedy or beam search; maximum probability is appropriate |
| High entropy — dialogue, story generation | sampling with truncation (top-$p$, top-$k$), tuned temperature, optional re-ranking |

Slide 37's closing position is that decoding remains open: "there's a lot more work to be
done," and different algorithms are a way to inject different inductive biases into the
generated text.

## Related pages

- [Lecture 10 — Natural Language Generation](10-natural-language-generation.md) — the lecture.
- [Natural language generation](natural-language-generation.md) — the open-endedness spectrum
  that decides which algorithm applies.
- [Seq2seq and encoder-decoder models](seq2seq-and-encoder-decoder.md) — where beam search is
  introduced.
- [Perplexity](perplexity.md) — the first-choice re-ranking score, and why it misleads.
- [Softmax and cross-entropy](softmax-and-cross-entropy.md) — the distribution temperature
  reshapes.
- [Exposure bias and teacher forcing](exposure-bias-and-teacher-forcing.md) — the same
  repetition problem, attacked from the training side.
- [Evaluating NLG](evaluating-nlg.md) — MAUVE, which measures whether a decoding algorithm's
  output distribution matches the human one.
- [Language models in decoding](language-models-in-decoding.md) — beam search with a language
  model fused into the objective, and the word insertion bonus that fixes its length bias.
- [Connectionist Temporal Classification](connectionist-temporal-classification.md) — beam
  search over a collapsing alphabet, where different extensions can merge into one prefix.
