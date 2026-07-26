# Pretraining and fine-tuning

The two-step recipe that defines modern NLP: train a network on a self-supervised objective
over a very large unlabeled corpus, then adapt those parameters to the task you actually care
about using a much smaller labeled dataset. Covered in [lecture 9](09-pretraining.md), slides
7–20 (the paradigm), 21–39 (the three architecture classes) and 32–34
(parameter-efficient adaptation).

## What changed

Circa 2017 (slide 9), the standard setup was: start from pretrained
[word embeddings](word2vec.md), stack a randomly initialized
[LSTM](lstm.md) or [Transformer](transformer.md) on top, and train the whole thing on your
task's labels. Two problems follow directly. The labeled data has to teach *every* contextual
aspect of language by itself, and most of the network's parameters begin as small random
noise.

Slide 10 states the modern position: **all, or almost all, parameters in an NLP network are
initialized via pretraining.** The pretraining methods all work the same way — hide part of
the input and train the model to reconstruct it — and the payoff is threefold:

- strong **representations of language** (contextual, so *record* the verb and *record* the
  noun get different vectors — see [word senses and polysemy](word-senses-and-polysemy.md));
- strong **parameter initializations** for downstream models;
- **probability distributions over language** that you can sample from, which is what
  [natural language generation](natural-language-generation.md) needs.

## Why it works

Slide 20 is unusually candid about the state of the explanation. Write $\hat{\theta}$ for the
parameters pretraining hands you:

$$\hat{\theta} \approx \arg\min_\theta \mathcal{L}_{\text{pretrain}}(\theta)$$

Fine-tuning then approximates

$$\arg\min_\theta \mathcal{L}_{\text{finetune}}(\theta)$$

by [gradient descent](gradient-descent.md) **starting from $\hat{\theta}$**. Here
$\mathcal{L}_{\text{pretrain}}$ is the pretraining loss (typically
[language modeling](language-modeling.md) or masked language modeling) and
$\mathcal{L}_{\text{finetune}}$ is the loss on the downstream task.

If that second minimization were actually solved, the starting point would be irrelevant. It
is not solved — it is approximated by SGD, which stays relatively close to where it started —
and so $\hat{\theta}$ matters "enormously" (≈30:11). Two intuitions are offered, both
hedged: the local minima *near* $\hat{\theta}$ may tend to generalize well, and/or the
gradients of the fine-tuning loss near $\hat{\theta}$ may propagate nicely. The lecturer
describes this as a live meeting point between optimization theory and practice.

The complementary argument is about data, not optimization (≈32:30). Unlabeled text runs to
roughly 5–10 trillion words on the internet; a carefully labeled task dataset might be a
million words. Beyond raw scale there is a robustness claim: a system trained only on the
movie reviews people write today degrades when people start writing them slightly differently
tomorrow, whereas one pretrained on a broad and varied corpus adapts better to text that does
not look like its fine-tuning data.

The corollary about ordering is worth keeping: with pretraining data outnumbering fine-tuning
data by orders of magnitude, what makes the model prioritize the task is simply that
fine-tuning comes **second** — "I've set [the parameters] somewhere and then I fine-tune, I
move to where the parameters are doing well for this task afterward" (≈34:48). And when you
*do* have a lot of task data, the answer is not to train from scratch — it is three-stage
training: pretrain broadly, continue pretraining with language modeling on your task's text
with the labels stripped off, then fine-tune with labels. This is the "Don't Stop Pretraining"
result (≈1:16:21).

## The three architecture classes

Slide 22 organizes the rest of the lecture. Which objective you can pretrain with is
determined by what the architecture is allowed to see.

| Class | Context | Pretraining objective | Natural use |
| --- | --- | --- | --- |
| **Encoders** | bidirectional | masked language modeling — can't do LM, the answer is visible | representations, classification |
| **Encoder-decoders** | bidirectional input, unidirectional output | span corruption (denoising) | conditional generation, e.g. MT |
| **Decoders** | unidirectional | language modeling | generation; "all the biggest pretrained models are decoders" |

**Encoders.** Language modeling is unavailable: with bidirectional context, "predicting the
next word is a trivial task … because I could just look at it, see what it is, and then copy
it over" (≈37:08). Masked language modeling replaces some tokens with `[MASK]`, computes
loss only at those positions, and so learns $p_\theta(x \mid \tilde{x})$ where $\tilde{x}$ is
the corrupted input. See [BERT](bert.md).

**Encoder-decoders.** The obvious objective is a *prefix* language model: give the encoder
$w_1, \dots, w_T$ with bidirectional context, and have the decoder generate the rest (slide
36). What Raffel et al. found works better is **span corruption**, the T5 objective — replace
variable-length spans with unique sentinel tokens and have the decoder emit the removed spans
in order:

> Original: `Thank you for inviting me to your party last week.`
> Inputs: `Thank you <X> me to your party <Y> week.`
> Targets: `<X> for inviting <Y> last <Z>`

It is implemented purely as text preprocessing, so from the decoder's point of view it is
still language modeling (slide 37). Slide 38's controlled comparison has the encoder-decoder
with a denoising objective on top across GLUE (83.28), CNN/DailyMail, SQuAD, SuperGLUE and
three translation directions, beating both decoder-only architectures and language-modeling
objectives.

Slide 39 adds **salient span masking**, which biases the corruption toward named entities and
facts. Fine-tuned on trivia questions, T5 then answers *new* trivia questions by retrieving
facts absorbed at pretraining time — implicit retrieval from parameters, at rates "much more
than random chance" though often below 50% (≈1:01:46). The caveat attached is permanent:
"the answers always look very fluent, they always look very reasonable, but they're frequently
wrong."

**Decoders.** Ordinary [language modeling](language-modeling.md), used two ways. Discard the
LM head and train a classifier $y \sim A h_T + b$ on the last hidden state, where $A$ and $b$
are randomly initialized and gradients flow through the whole network (slide 41); or keep the
pretrained output layer and fine-tune the model as a generator, $w_t \sim A h_{t-1} + b$,
which suits any task whose output is a sequence over the pretraining vocabulary — dialogue,
summarization (slide 42). See [GPT and in-context learning](gpt-and-in-context-learning.md).

## Full versus parameter-efficient fine-tuning

Fine-tuning every parameter works well but is memory-intensive, and it moves the model further
from the pretrained solution than necessary. **Lightweight** or **parameter-efficient**
fine-tuning keeps most parameters frozen and adapts the model in a constrained way, which
"leads to less overfitting and/or more efficient finetuning and inference" (slide 32). The
motivating intuition is that "these pretrained parameters were really good, and you want to
make the minimal change from the pretrained model to the model that does what you want, so
that you keep some of the generality" (≈55:35).

**Prefix-tuning / prompt tuning** (slide 33; Li and Liang, 2021; Lester et al., 2021) freezes
*all* pretrained parameters and prepends a block of learned pseudo-word vectors to the
sequence, training only those. The model processes them exactly as it would real word
embeddings. The lecturer notes this is "sort of unintuitive — these would have been inputs to
the network, but I'm specifying them as parameters" (≈56:22). They must go at the *beginning*
in a decoder, because otherwise the model does not see them until it has already processed the
sequence (≈57:08). One advantage the slide highlights: because the prefix is data rather than
weights, each element of an inference batch can carry a different tuned prefix.

**LoRA — low-rank adaptation** (slide 34; Hu et al., 2021) freezes each weight matrix
$W \in \mathbb{R}^{d \times d}$ and learns a low-rank correction, using
$A \in \mathbb{R}^{d \times k}$ and $B \in \mathbb{R}^{k \times d}$ with $k \ll d$:

$$W + AB$$

Since $A$ and $B$ have $2dk$ parameters against $W$'s $d^2$, only a small fraction of the
network is trained. The lecture calls it "easier to learn than prefix-tuning" (slide 34) and
"a very similarly useful technique" (≈57:55).

A third option raised from the floor — freeze the pretrained network and train only new layers
stacked on top — gets a one-word endorsement: "absolutely, this works a bit better" (≈57:55).

Beyond memory, the saving is in optimizer state: you avoid storing gradients and momentum for
the frozen parameters, not just their updates (≈57:08).

> **Lecture 13 develops all of this properly.** See
> [parameter-efficient finetuning](parameter-efficient-finetuning.md) for the wider family and
> the case for it, and [LoRA](lora.md) for the method in full — the rank and $\alpha$
> hyperparameters, which matrices to adapt, the results tables, and the practical defaults.
> [GPU memory for training](gpu-memory-for-training.md) quantifies the optimizer-state saving
> noted above: a trainable parameter costs 16 bytes and a frozen one costs 2.
>
> **Note the change of notation between the two decks.** Lecture 9's slide 34 writes the
> correction as $W + AB$ with $A \in \mathbb{R}^{d \times k}$ and $B \in \mathbb{R}^{k \times d}$;
> lecture 13's slide 59 writes it as $W_0 + \alpha BA$ with $B \in \mathbb{R}^{d \times r}$ and
> $A \in \mathbb{R}^{r \times k}$ — the letters are swapped and the rank is called $r$. The
> method is the same; each page uses its own lecture's symbols.

## Evaluating a general-purpose model

Pretrained models are meant to be general, which makes evaluation awkward. The lecture gives
two practical answers (≈28:38). During training, track [perplexity](perplexity.md) — it is
cheap and "better perplexity correlates with all the stuff that's much harder to evaluate."
Afterwards, the field uses large benchmark suites of varied tasks, and a new pretraining
method argues for generality by improving across all of them. The lecturer's own assessment of
this practice: notions of generality are "very, very difficult, ill-defined even."

## Related pages

- [Lecture 9 — Pretraining](09-pretraining.md) — the lecture this comes from.
- [BERT and masked language modeling](bert.md) — the encoder branch.
- [GPT and in-context learning](gpt-and-in-context-learning.md) — the decoder branch, and what
  happens when models get large enough that fine-tuning becomes optional.
- [Language modeling](language-modeling.md) — the objective pretraining reuses.
- [Gradient descent](gradient-descent.md) — why the initialization matters at all.
- [Transformer](transformer.md) — the architecture the three classes are variants of.
- [Subword modeling](subword-modeling.md) — how the inputs are tokenized.
- [Regularization and dropout](regularization-and-dropout.md) — the overfitting picture
  lightweight fine-tuning improves on.
- [Instruction finetuning](instruction-finetuning.md) — the same recipe scaled to thousands of
  tasks at once, from [lecture 11](11-post-training.md).
