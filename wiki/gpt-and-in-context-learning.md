# GPT and in-context learning

The decoder branch of pretraining, and the point at which scaling it changed what a language
model is for. Covered in [lecture 9](09-pretraining.md), slides 40–51.

## Pretraining a decoder

A decoder-only [Transformer](transformer.md) is an ordinary
[language model](language-modeling.md): each position may attend only to itself and the past,
and the model is trained to predict the next token. Slide 40 records the fact that ends up
dominating the field — **all the biggest pretrained models are decoders.** Asked why, the
lecture is honest that the reason is unclear, and offers two practical ones: they are simpler
than encoder-decoders, and all the parameters sit in one network rather than being split
between an encoder and a decoder (≈1:03:18).

Once pretrained, a decoder can be used two ways.

**As a classifier** (slide 41). Ignore that it was trained to model
$p(w_t \mid w_{1:t-1})$, run the sequence through, and train a classifier on the *last* word's
hidden state:

$$h_1, \dots, h_T = \mathrm{Decoder}(w_1, \dots, w_T)$$

$$y \sim A h_T + b$$

where $A$ and $b$ are randomly initialized and determined by the downstream task's label set.
Gradients backpropagate through the whole network. Slide 41's margin note is the thing to
notice: this linear layer was never pretrained and must be learned from scratch.

**As a generator** (slide 42). Keep the pretrained output layer and fine-tune the
distribution itself:

$$h_1, \dots, h_T = \mathrm{Decoder}(w_1, \dots, w_T)$$

$$w_t \sim A h_{t-1} + b$$

Here $A$ and $b$ *were* pretrained, as part of the language model. This suits any task whose
output is a sequence over the pretraining vocabulary — dialogue (context = dialogue history),
summarization (context = document).

## The GPT line

**GPT** (Radford et al., 2018), slide 43: a 12-layer Transformer decoder, 117M parameters,
768-dimensional hidden states, 3072-dimensional feed-forward layers,
[byte-pair encoding](subword-modeling.md) with 40,000 merges, trained on BooksCorpus (over
7,000 unique books, chosen partly because books contain long spans of contiguous text, which
is what teaches long-distance dependencies). The slide notes that the acronym never appears in
the original paper; it could stand for "Generative PreTraining" or "Generative Pretrained
Transformer."

Slide 44 shows the input format that made a decoder work on a *pair*-of-sentences task.
Natural language inference asks you to label a premise/hypothesis pair as
entailing/contradictory/neutral — "The man is in the doorway" entails "The person is near the
door." Since a decoder consumes one sequence, the pair is simply concatenated with special
tokens:

> `[START]` *The man is in the doorway* `[DELIM]` *The person is near the door* `[EXTRACT]`

and the linear classifier is applied to the representation of the `[EXTRACT]` token. Slide 45's
results table shows the fine-tuned Transformer LM taking MNLI (82.1/81.4), SNLI (89.9), SciTail
(88.3) and QNLI (88.1), losing only on the small RTE dataset. [BERT](bert.md) arrived shortly
after and did better still, with bidirectional context (≈1:04:52).

**GPT-2** (slide 46) scaled to 1.5B parameters on more data, and the emphasis shifted to
generation. The unicorn sample on the slide — a fluent, coherent, entirely fabricated news
story about English-speaking unicorns in the Andes — was the demonstration. The lecture adds
that this size is still small enough to fine-tune on a modest GPU (≈1:05:39).

**GPT-3** (slide 47) is 175B parameters trained on 300B tokens, against the largest T5's 11B.
At that size, something new appears.

## In-context learning

Slides 47–49. Up to this point there have been two ways to interact with a pretrained model:
sample from the distribution it defines, or fine-tune it and take its predictions. GPT-3 does
something else — it performs tasks **without any gradient steps**, purely from examples placed
in its context window.

Slide 48's example is a single decoder input:

```
thanks -> merci
hello -> bonjour
mint -> menthe
otter ->
```

and the conditional generation is `loutre`. Nobody said "do translation"; nobody fine-tuned
anything. "The in-context examples seem to specify the task to be performed, and the
conditional distribution mocks performing the task to a certain extent" (slide 48). Slide 49
shows the same behaviour on arithmetic (`5 + 8 = 13`, …), on unscrambling anagrams
(`gaot => goat`, …) and on translation, all as sequences inside one context, under a long arrow
labelled "Learning via SGD during unsupervised pre-training."

The lecture flags how surprising this ought to be. "So far we've talked about good
representations, contextual representations, meanings of words in context. This is some very,
very high-level pattern matching" (≈1:09:28). And it explicitly leaves the mechanism open:
did GPT-3 learn to do this because it saw many bilingual dictionaries during pretraining, or is
it genuinely learning the task from the context?

> "The actual story, we're not totally sure — something in the middle, it seems like. It has
> to be tied to your training data in ways that we don't quite understand, but there's also a
> non-trivial ability to learn new … at least types of patterns … just from the context."
> (≈1:10:13)

This is the first appearance in the course of **emergent properties** — qualitatively new
behaviour that shows up at large scale and is not predictable from smaller models.

## Chain-of-thought prompting

Slide 51 (Wei et al., 2023) extends the idea: the prefix specifies not just *what* task to do
but *how* to work through it. In standard prompting the demonstration is a question and its
answer; the model produces an answer to a new question directly, and on a two-step arithmetic
word problem it gets 27 instead of 9. In chain-of-thought prompting the demonstration includes
the intermediate reasoning —

> A: *Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11.*
> The answer is 11.

— and the model imitates the *shape* of that answer, generating its own steps before
committing to a number, and gets it right.

Two ways to think about why (≈1:13:17). It is a **scratch pad**: each generated token is
conditioned on all the tokens before it, so writing the reasoning down makes it available as
context for the rest of the computation. And it **increases the amount of computation** spent
on the problem, decomposing it into smaller problems the model can each solve. "No one's really
sure why this works exactly either."

## Scaling: bigger was the wrong lever

Slide 50 is the correction to "just make it larger." The cost of training a large Transformer
scales roughly as **parameters × tokens**, so a fixed compute budget buys a trade-off between
the two — and GPT-3 chose badly:

| Model | Parameters | Training tokens |
| --- | --- | --- |
| LaMDA (Thoppilan et al., 2022) | 137 B | 168 B |
| GPT-3 (Brown et al., 2020) | 175 B | 300 B |
| Jurassic (Lieber et al., 2021) | 178 B | 300 B |
| Gopher (Rae et al., 2021) | 280 B | 300 B |
| MT-NLG 530B (Smith et al., 2022) | 530 B | 270 B |
| **Chinchilla** | **70 B** | **1.4 T** |

Chinchilla is less than half the size of GPT-3 and better, because DeepMind spent the budget on
data instead. The lecture's framing: GPT-3 "was just comically oversized," and getting this
allocation right matters because "you can't do this more than a handful of times, even if
you're Google" (≈1:10:59).

## What to take away

The lecture's parting position (slide 54, and ≈1:13:17) is deliberately unresolved on both
sides. These models are not well understood; "small" models like BERT have become general tools
for a wide range of settings; and the emergent behaviours are

> "both very powerful and exceptionally hard to understand, and very hard, you should think, to
> trust, because it's unclear what its capabilities are and what its limitations are, where it
> will fail."

Asked whether a pretrained model can do a task it has genuinely never seen, the answer given is
that the question is not well posed — the models clearly recombine fragments of tasks absorbed
during pretraining, and "quantifying that extent is an open research problem" (≈1:17:53).

## Related pages

- [Lecture 9 — Pretraining](09-pretraining.md) — where this is taught.
- [Pretraining and fine-tuning](pretraining-and-finetuning.md) — the paradigm and the other two
  architecture classes.
- [BERT and masked language modeling](bert.md) — the encoder alternative.
- [Language modeling](language-modeling.md) — the pretraining objective.
- [Transformer](transformer.md) — the decoder architecture, including future-masking.
- [Subword modeling](subword-modeling.md) — GPT's byte-pair encoding.
- [Decoding algorithms](decoding-algorithms.md) — how you get text out of one of these once you
  have it.
- [Lecture 10 — Natural Language Generation](10-natural-language-generation.md).
