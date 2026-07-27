# Sequence-to-sequence and encoder-decoder models

The architecture that made neural [machine translation](machine-translation.md) work, and a
pattern that outlives the RNNs it was first built from. Introduced in
[lecture 6](06-sequence-to-sequence-models.md), slides 49–54.

## The model

**Neural Machine Translation** does translation with a *single end-to-end neural network*.
That network is a **sequence-to-sequence** model — **seq2seq** — and it involves **two** RNNs
(slide 49), in practice two [LSTMs](lstm.md) (lecture 6, ≈1:06:08).

- The **encoder RNN** reads the source sentence and **outputs nothing**. Its job is only to
  build up a hidden state that knows what is in the source. Its final hidden state is the
  **encoding of the source sentence**.
- The **decoder RNN** is a [language model](language-modeling.md) that generates the target
  sentence, **conditioned on that encoding**: the encoding is fed in as its initial previous
  hidden state, and generation starts from a `<START>` token.

The two have **separate parameters** — one LSTM whose weights encode the source language, one
whose weights know the target language (≈1:06:56).

Slide 50's worked example translates the French *il m' a entarté* into *he hit me with a pie*.
At each decoder step an **argmax** picks the output word, and at test time that word is fed
back in as the next step's input.

## It is a conditional language model

Slide 52 gives the formal description, and both halves of the name earn their place:

- **Language model**, because the decoder is predicting the next word of the target
  sentence $y$.
- **Conditional**, because its predictions are *also* conditioned on the source sentence $x$.

So NMT directly calculates $P(y \mid x)$, where $y_1, \dots, y_T$ are the target words:

$$P(y \mid x) = P(y_1 \mid x) \cdot P(y_2 \mid y_1, x) \cdot P(y_3 \mid y_1, y_2, x) \cdots P(y_T \mid y_1, \dots, y_{T-1}, x)$$

Each factor is "the probability of the next target word, given the target words so far and
the source sentence $x$". Compare this with
[statistical machine translation](machine-translation.md), which had to factor
$P(y \mid x)$ with Bayes rule into a translation model and a language model learned
separately; the neural model learns the whole thing at once.

## Training end-to-end

Slide 53. The diagram is the same as slide 50's, with one important difference: **at training
time the decoder is fed the actual target words from the corpus**, not its own output. That
is [teacher forcing](recurrent-neural-networks.md), applied to the decoder.

The procedure (≈1:08:29):

1. Take a sentence and its translation from parallel text.
2. Run the encoder over the source, then the decoder over the target.
3. At each decoder position, ask what probability was assigned to the actual next word; that
   negative log probability is the loss $J_t$.
4. Average the losses over the $T$ decoder positions:
   $J = \frac{1}{T} \sum_{t=1}^{T} J_t$.
5. Backpropagate through the **entire** network — decoder *and* encoder — and update all the
   parameters.

The slide's summary is the point: **"Seq2seq is optimized as a single system. Backpropagation
operates end-to-end."** Manning's framing of why this matters generally (≈1:05:23): putting a
single loss at the end and backpropagating right down through the system aligns all the
learning with the task you actually want, which pipeline models of separately-designed
components could not do.

## The multi-layer version, and the bottleneck

Slide 54 (Sutskever et al. 2014; Luong et al. 2015) shows a three-layer stacked
encoder-decoder translating *Die Proteste waren am Wochenende eskaliert* into *The protests
escalated over the weekend*. The hidden states of RNN layer $i$ are the inputs to layer $i+1$,
and each generated word is fed back in as the next input. Machine translation is the clearest
case where [stacked RNNs](recurrent-neural-networks.md) earned their keep (≈1:13:08).

Written by hand at the bottom of that slide, under the single column of state that joins
encoder to decoder:

> **Conditioning = Bottleneck**

That is the honest weakness of the design. The entire source sentence has to pass through one
fixed-size vector. It is the problem **[attention](attention.md)** solves in the next
lecture, and slide 25's remark about "more direct and linear pass-through connections"
points at the same fix. See [vanishing and exploding gradients](vanishing-and-exploding-gradients.md).

## Two design questions from the lecture

Both come from students, and both sharpen what the architecture is for.

**Why not just build a deeper network on top of the source?** (≈1:10:03). People have tried
it occasionally and it has never been very successful, for two reasons Manning gives. Word
order changes a lot between languages, so positions do not correspond. And the **length does
not even stay the same**: one of the big ways languages vary is in what little words they
have — English uses auxiliary verbs and articles that Chinese does not — so depending on
direction you must add or delete a lot of words, which is very hard when you are building on
top of fixed source positions (≈1:10:49).

**Is the encoder bidirectional?** (≈1:11:37). It could be, and that might well be better. But
the famous original instantiation at Google was **not** bidirectional — it simply took the
final hidden state. Note that the *decoder* cannot be bidirectional, because it is generating
and only has left context; this is the general constraint on
[bidirectional RNNs](recurrent-neural-networks.md) (slide 37).

## The pattern generalizes

Slide 51's point, and the reason this page exists separately from
[machine translation](machine-translation.md).

The general notion is an **encoder-decoder** model: one network takes input and produces a
neural representation; another network produces output based on that representation. If both
the input and the output are sequences, it is a seq2seq model.

Many NLP tasks can be phrased this way:

- **Summarization** — long text → short text
- **Dialogue** — previous utterances → next utterance
- **Parsing** — input text → output parse as a sequence
- **Code generation** — natural language → Python code

And the pattern survives the architectural change that comes later (≈1:09:15): "even when we
go on to do other things like use Transformers rather than LSTMs, we're still commonly going
to use these kinds of encoder-decoder models." See [Transformer](transformer.md) for the
encoder-decoder shape that architecture takes.

## When to reach for one, and when not to

Lecture 10 revisits the choice now that decoder-only models are available, and states it as a
convention rather than a constraint (slide 16). Non-open-ended tasks such as machine
translation typically use an encoder-decoder; open-ended tasks such as story generation
typically use the autoregressive decoder alone. But a decoder alone *can* do MT, and an
encoder-decoder *can* generate stories — the convention holds for a budget reason rather than
a type-checking one: decoder-only hurts performance on MT, while an encoder-decoder gives
roughly no benefit on open-ended generation, so with a fixed compute budget "you might just be
better off by only training a larger decoder model" (lecture 10, ≈10:03). See
[natural language generation](natural-language-generation.md).

Lecture 9 adds the pretraining side. An encoder-decoder can be pretrained as a prefix language
model — encode the first half of a text with bidirectional context, generate the second half —
but **span corruption** works better: replace variable-length spans of the input with sentinel
tokens and have the decoder emit the removed spans. That is T5, and it is what CS224N's
Assignment 5 implements. See
[pretraining and fine-tuning](pretraining-and-finetuning.md#the-three-architecture-classes).

## Related pages

- [Machine translation](machine-translation.md) — the task, and the statistical era this
  replaced.
- [Attention](attention.md) — the fix for this architecture's bottleneck, introduced in
  [lecture 7](07-attention-final-projects-and-llm-intro.md).
- [Transformer](transformer.md) — the encoder-decoder architecture that later replaces this
  one.
- [LSTM](lstm.md) — what the encoder and decoder are in practice.
- [Recurrent neural networks](recurrent-neural-networks.md) — teacher forcing,
  bidirectionality and stacking, all of which appear here.
- [Language modeling](language-modeling.md) — conditional language models generally.
- [Pretraining and fine-tuning](pretraining-and-finetuning.md) — span corruption, the T5
  objective for pretraining this architecture.
- [Natural language generation](natural-language-generation.md) — when an encoder-decoder is
  the right shape and when a bare decoder is.
- [Decoding algorithms](decoding-algorithms.md) — beam search generalized, and the sampling
  methods that replace it for open-ended tasks.
- [Connectionist Temporal Classification](connectionist-temporal-classification.md) — what to
  use instead when the alignment is **monotonic**, and why arbitrary alignment is a cost rather
  than a free gift when training data is scarce.
- [Lecture 6 — Sequence to Sequence Models](06-sequence-to-sequence-models.md)
- [Lecture 10 — Natural Language Generation](10-natural-language-generation.md)
- [Lecture 14 — Brain-computer interfaces](14-brain-computer-interfaces.md)
