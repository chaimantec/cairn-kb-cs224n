# Natural language generation

The half of NLP whose *output* is language. Covered in
[lecture 10](10-natural-language-generation.md), slides 4–18.

## The definition, and the split

Slide 4 divides the field:

$$\text{NLP} = \text{Natural Language Understanding (NLU)} + \text{Natural Language Generation (NLG)}$$

In NLU the task **input** is natural language — semantic parsing,
[natural language inference](gpt-and-in-context-learning.md), and so on. In NLG the task
**output** is natural language. The working definition given is that NLG covers "systems that
produce fluent, coherent and useful language output for human consumption."

Historically many NLG systems were rule-based — templates and slot-filling — but "nowadays deep
learning is powering almost every text generation system" (≈3:09).

The examples on slides 5–6 span most of the course and beyond it:

| Task | Input | Output |
| --- | --- | --- |
| [Machine translation](machine-translation.md) | utterance in the source language | text in the target language |
| Dialogue / digital assistants | dialogue history | a response continuing the conversation |
| Summarization | a long document | a few readable sentences |
| Creative story generation | a plot outline | a story aligned with the outline |
| Data-to-text | a table or database | prose describing its contents |
| Visual description | an image | a caption or a story |

Slides 7–9 add the state of the art at the time of the lecture: ChatGPT, which is a single
general-purpose NLG system reachable through different prompts (chatbot, poetry generation),
and Bing augmented with it for web search.

## The open-endedness spectrum

Slides 10–13 build the taxonomy that organizes everything else in the lecture. Tasks are
placed on a spectrum by how constrained their output is:

**Machine Translation → Summarization → Task-driven Dialog → ChitChat Dialog → Story Generation**
(less open-ended → more open-ended)

At the left end, a source sentence essentially determines its translation. Slide 10's example
gives one Chinese sentence and three English renderings that differ only in word order and
voice, because "you have to make sure the semantics doesn't change" (≈6:15). **The output
space is not very diverse.**

In the middle, "Hey, how are you?" admits genuinely different responses — "Good! You?", "I just
heard some exciting news, do you want to hear it?", "Thanks for asking! Barely surviving my
homeworks." **The output space is getting more diverse.**

At the right end, "Write a story about three little pigs?" has no bounded answer set at all.
**The output space is extremely diverse.**

Slide 13 formalizes the axis by **entropy** — high entropy toward the open-ended end, low
entropy toward the constrained end — and states the consequence that the rest of the lecture
depends on:

> These two classes of NLG tasks require different decoding and/or training approaches.

This is not a decorative taxonomy. It is the reason maximum-probability decoding is right for
MT and wrong for story generation (see [decoding algorithms](decoding-algorithms.md)), and the
reason BLEU is tolerable for MT and useless for dialogue (see
[evaluating NLG](evaluating-nlg.md)).

## The model, and where decoding fits

Slide 15 restates the autoregressive model from lecture 5. At each time step the model takes
the tokens so far, $\{y\}_{<t}$, and emits a new token $\hat{y}_t$. For a model $f$ and
vocabulary $V$ it computes a score vector

$$S = f(\{y_{<t}\}, \theta) \in \mathbb{R}^{V}$$

and normalizes it with a [softmax](softmax-and-cross-entropy.md):

$$P(y_t \mid \{y_{<t}\}) = \frac{\exp(S_w)}{\sum_{w' \in V} \exp(S_{w'})}$$

Training is one token at a time by maximum likelihood (slide 17), minimizing

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(y^*_t \mid \{y^*\}_{<t})$$

where $y^*$ denotes the gold tokens. Because the model's inputs during training are the gold
tokens rather than its own outputs, this is **teacher forcing** — see
[exposure bias and teacher forcing](exposure-bias-and-teacher-forcing.md).

The piece that is *not* determined by the model is the last step. Slide 18 introduces the
decoding algorithm $g$:

$$\hat{y}_t = g\big(P(y_t \mid \{y_{<t}\})\big)$$

The "obvious" choice is to take the argmax greedily at each step. Slide 18 then frames the
lecture: given a working model, the two main avenues to better output are to **improve
decoding** and to **improve the training**. (A pink margin note concedes the obvious third and
fourth: improve the training data or the model architecture.)

## Which architecture for which end of the spectrum

Slide 16 gives the convention:

- **Non-open-ended tasks** (e.g. MT) typically use an
  [encoder-decoder](seq2seq-and-encoder-decoder.md) — a bidirectional encoder over the input,
  an autoregressive decoder generating the output. This is what CS224N's Assignment 4 builds.
- **Open-ended tasks** (e.g. story generation) typically use the autoregressive decoder alone,
  conditioned on a prompt.

The lecturer is explicit that these are conventions, not constraints: a decoder alone can do
machine translation, and an encoder-decoder can generate stories. The reason the convention
holds is a budget argument (≈10:03): decoder-only *hurts* on MT relative to an
encoder-decoder, while an encoder-decoder gives roughly *no benefit* on open-ended generation
— so given a fixed compute budget you are better off spending it on a larger decoder. "It's
kind of more of an allocation-of-resources problem than whether these two architectures will
type-check with your task."

## Related pages

- [Lecture 10 — Natural Language Generation](10-natural-language-generation.md) — the lecture.
- [Decoding algorithms](decoding-algorithms.md) — how the distribution becomes tokens, and why
  the open-endedness of the task decides which algorithm to use.
- [Exposure bias and teacher forcing](exposure-bias-and-teacher-forcing.md) — the training
  side.
- [Evaluating NLG](evaluating-nlg.md) — why measuring this is unsolved.
- [Language modeling](language-modeling.md) — the underlying model.
- [Seq2seq and encoder-decoder models](seq2seq-and-encoder-decoder.md) — the architecture used
  at the constrained end of the spectrum.
- [GPT and in-context learning](gpt-and-in-context-learning.md) — the pretrained decoders these
  systems are built from.
