---
title: Lecture 9 — Pretraining (slide deck)
lecture: 9
slides: 54 pages in the PDF; printed numbers match pages 1:1
source_pdf: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture9-pretraining.pdf
note: |
  Lecturer is John Hewitt; the deck credits "Adapted from slides by Anna Goldie, John Hewitt".
  Printed slide numbers match PDF page numbers 1:1. Pages 38, 39 and 46 print no number
  (they are full-bleed table/figure pages) but sit in sequence, so "slide N" = "page N"
  throughout.
provenance: |
  This deck is from the **Winter 2023** offering (cs224n.1234), not the Spring 2024 site the
  rest of this KB was crawled from. The lecture video in the Cairn catalog is the Winter 2023
  recording — slide 2's "Assignment 5 is out on Thursday! It covers lecture 8 and lecture
  9 (Today)!" matches the lecturer's spoken announcement word for word, and Spring 2024 ran
  only A1–A4. A separate, later deck exists on the Spring 2024 site
  (cs224n-spr2024-lecture09-pretraining-updated.pdf, 64 pages); its slide numbers do **not**
  line up with this video, so cite this file.
---

# Lecture 9 — Pretraining: slide-by-slide

Text and figures of
[`cs224n-2023-lecture9-pretraining.pdf`](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture9-pretraining.pdf),
transcribed from the deck. Diagrams and plots are described in prose since the KB is
read as text.

Companion pages: [wiki page for this lecture](../../wiki/09-pretraining.md) ·
[transcript](../transcripts/09-pretraining.md)

## Contents

| Slides | Section |
| ------ | ------- |
| 1–2 | Title and lecture plan |
| 3–6 | §1 A brief note on subword modeling: the fixed-vocabulary assumption, morphology, byte-pair encoding |
| 7–20 | §2 Motivating model pretraining from word embeddings: Firth on context, pretrained word embeddings vs. pretraining whole models, the "reconstruct the input" examples, pretraining through language modeling, the pretrain/finetune paradigm and why SGD makes it work |
| 21–34 | §3.1 Encoders: masked language modeling, BERT, its results, limitations, RoBERTa and SpanBERT, full vs. parameter-efficient finetuning (prefix-tuning, LoRA) |
| 35–39 | §3.2 Encoder-decoders: prefix language modeling, T5's span corruption, salient-span masking and closed-book QA |
| 40–51 | §3.3 Decoders: finetuning a classifier on the last hidden state, generative finetuning, GPT, GPT-2, GPT-3 and in-context learning, Chinchilla scaling, chain-of-thought |
| 52–53 | §4 What do we think pretraining is teaching? — the opening examples labelled by what they test |
| 54 | Parting remarks |

Note on §3 ordering: **slide 2's lecture plan lists decoders, encoders, encoder-decoders
in that order, but the deck itself presents encoders → encoder-decoders → decoders**, which
is the order slide 7 and every later section slide use. The lecture follows the deck.

---

## Slide 1 — Title

**Natural Language Processing with Deep Learning — CS224N/Ling284**

John Hewitt. Lecture 9: Pretraining. *Adapted from slides by Anna Goldie, John Hewitt.*

## Slide 2 — Lecture Plan

1. A brief note on subword modeling
2. Motivating model pretraining from word embeddings
3. Model pretraining three ways
   1. Decoders
   2. Encoders
   3. Encoder-Decoders
4. Interlude: what do we think pretraining is teaching?
5. Very large models and in-context learning

**Reminders:** Assignment 5 is out on Thursday! It covers lecture 8 and lecture 9 (Today)!
It has ~pedagogically relevant math~

## Slide 3 — Word structure and subword models

Let's take a look at the assumptions we've made about a language's vocabulary.

We assume a fixed vocab of tens of thousands of words, built from the training set. All
*novel* words seen at test time are mapped to a single UNK.

A three-column table headed **word** / **vocab mapping** / **embedding** shows the
failure. Rows are bracketed on the left by category:

| Category | word | vocab mapping | embedding |
| --- | --- | --- | --- |
| Common words | hat | pizza (index) | a distinct (red) vector |
| Common words | learn | tasty (index) | a distinct (red) vector |
| Variations | taaaaasty | UNK (index) | the same grey vector |
| misspellings | laern | UNK (index) | the same grey vector |
| novel items | Transformerify | UNK (index) | the same grey vector |

(The "vocab mapping" column deliberately shows *hat* → pizza and *learn* → tasty: the
mapping is to an arbitrary index, and the point of the figure is that the last three
rows all collapse onto one identical embedding.)

## Slide 4 — Word structure and subword models

Finite vocabulary assumptions make even *less* sense in many languages.

- Many languages exhibit complex **morphology**, or word structure.
  - The effect is more word types, each occurring fewer times.

Example: Swahili verbs can have hundreds of conjugations, each encoding a wide variety of
information. (Tense, mood, definiteness, negation, information about the object, ++)

Here's a small fraction of the conjugations for *ambia* – to tell.

The right half of the slide is a screenshot of a Wiktionary conjugation table for the
Swahili verb *-ambia*. It is a dense grid: rows are Polarity (Positive/Negative) crossed
with tense/mood blocks — Past, Present, Future, Subjunctive, Present Conditional, Past
Conditional, Conditional Contrary to Fact, Gnomic, Perfect — and columns are Persons
(1st/2nd singular and plural, 3rd sg/pl) plus a long series of noun **Classes** (M-mi,
Ma, Ki-vi, N, U, Ku, Pa, Mu, numbered 3–18). Every cell holds a distinct word form
(*niliambia*, *tuliambia*, *uliambia*, *mliambia*, *aliambia*, *waliambia*, …), which is
the point: hundreds of surface forms for one verb. Credited **[Wiktionary]**.

## Slide 5 — The byte-pair encoding algorithm

Subword modeling in NLP encompasses a wide range of methods for reasoning about structure
below the word level. (Parts of words, characters, bytes.)

- The dominant modern paradigm is to learn a vocabulary of **parts of words (subword
  tokens).**
- At training and testing time, each word is split into a sequence of known subwords.

**Byte-pair encoding** is a simple, effective strategy for defining a subword vocabulary.

1. Start with a vocabulary containing only characters and an "end-of-word" symbol.
2. Using a corpus of text, find the most common adjacent characters "a,b"; add "ab" as a
   subword.
3. Replace instances of the character pair with the new subword; repeat until desired
   vocab size.

Originally used in NLP for machine translation; now a similar method (WordPiece) is used
in pretrained models.

Credited **[Sennrich et al., 2016, Wu et al., 2016]**.

## Slide 6 — Word structure and subword models

Common words end up being a part of the subword vocabulary, while rarer words are split
into (sometimes intuitive, sometimes not) components.

In the worst case, words are split into as many subwords as they have characters.

The same table as slide 3, now with subword mappings — every row gets real (red)
embeddings rather than a shared UNK:

| Category | word | vocab mapping | embedding |
| --- | --- | --- | --- |
| Common words | hat | hat | one vector |
| Common words | learn | learn | one vector |
| Variations | taaaaasty | taa## aaa## sty | three vectors |
| misspellings | laern | la## ern## | two vectors |
| novel items | Transformerify | Transformer## ify | two vectors |

## Slide 7 — Outline

1. A brief note on subword modeling *(greyed out — done)*
2. **Motivating model pretraining from word embeddings** *(current)*
3. Model pretraining three ways *(greyed out)*
   1. Encoders
   2. Encoder-Decoders
   3. Decoders
4. What do we think pretraining is teaching? *(greyed out)*

## Slide 8 — Motivating word meaning and context

Recall the adage we mentioned at the beginning of the course:

> *"You shall know a word by the company it keeps"* (J. R. Firth 1957: 11)

This quote is a summary of **distributional semantics**, and motivated **word2vec**. But:

> "… the complete meaning of a word is always contextual, and no study of meaning apart
> from a complete context can be taken seriously." (J. R. Firth 1935)

Consider *I **record** the **record***: the two instances of ***record*** mean different
things.

Margin note: [Thanks to Yoav Goldberg on Twitter for pointing out the 1935 Firth quote.]

## Slide 9 — Where we were: pretrained word embeddings

Circa 2017:

- Start with pretrained word embeddings (no context!)
- Learn how to incorporate context in an LSTM or Transformer while training on the task.

**Some issues to think about:**

- The training data we have for our **downstream task** (like question answering) must be
  sufficient to teach all contextual aspects of language.
- Most of the parameters in our network are randomly initialized!

The diagram on the right shows a stack over the input *… the movie was …*: a bottom row
of red blocks labelled **pretrained (word embeddings)**, and above it two rows of grey
blocks with bidirectional arrows, bracketed as **Not pretrained**, feeding a grey
triangle output ŷ.

Margin note: [Recall, *movie* gets the same word embedding, no matter what sentence it
shows up in]

## Slide 10 — Where we're going: pretraining whole models

In modern NLP:

- All (or almost all) parameters in NLP networks are initialized via **pretraining**.
- Pretraining methods hide parts of the input from the model, and train the model to
  reconstruct those parts.

- This has been exceptionally effective at building strong:
  - **representations of language**
  - **parameter initializations** for strong NLP models.
  - **Probability distributions** over language that we can sample from

The diagram is the same stack as slide 9 over *… the movie was …*, but now every block —
embeddings, both recurrent layers, and the output triangle producing ŷ — is red and the
whole thing is bracketed **Pretrained jointly**.

Margin note: [This model has learned how to represent entire sentences through
pretraining]

## Slide 11 — What can we learn from reconstructing the input?

> Stanford University is located in __________, California.

*(A sequence of seven slides, 11–17, each showing one cloze example and nothing else.
Slide 53 comes back and labels each one with the capability it probes.)*

## Slide 12 — What can we learn from reconstructing the input?

> I put ___ fork down on the table.

## Slide 13 — What can we learn from reconstructing the input?

> The woman walked across the street, checking for traffic over ___ shoulder.

## Slide 14 — What can we learn from reconstructing the input?

> I went to the ocean to see the fish, turtles, seals, and _____.

## Slide 15 — What can we learn from reconstructing the input?

> Overall, the value I got from the two hours watching it was the sum total of the popcorn
> and the drink. The movie was ___.

## Slide 16 — What can we learn from reconstructing the input?

> Iroh went into the kitchen to make some tea. Standing next to Iroh, Zuko pondered his
> destiny. Zuko left the ______.

## Slide 17 — What can we learn from reconstructing the input?

> I was thinking about the sequence that goes 1, 1, 2, 3, 5, 8, 13, 21, ____

## Slide 18 — Pretraining through language modeling [Dai and Le, 2015]

Recall the **language modeling** task:

- Model p_θ(w_t | w_{1:t−1}), the probability distribution over words given their past
  contexts.
- There's lots of data for this! (In English.)

**Pretraining through language modeling:**

- Train a neural network to perform language modeling on a large amount of text.
- Save the network parameters.

The diagram shows a red **Decoder (Transformer, LSTM, ++)** box. Inputs along the bottom
are *Iroh, goes, to, make, tasty, tea*; outputs along the top are *goes, to, make, tasty,
tea, END* — each output shifted one position left of its input.

## Slide 19 — The Pretraining / Finetuning Paradigm

Pretraining can improve NLP applications by serving as parameter initialization.

**Step 1: Pretrain (on language modeling)** — Lots of text; learn general things!

The left diagram repeats slide 18: a red **(Transformer, LSTM, ++)** box over inputs
*Iroh, goes, to, make, tasty, tea* predicting *goes, to, make, tasty, tea, END*.

**Step 2: Finetune (on your task)** — Not many labels; adapt to the task!

The right diagram shows the same box, now shaded pink (i.e. adapted), over the input
*… the movie was …*, with a single ☺/☹ sentiment output above the last position.

## Slide 20 — Stochastic gradient descent and pretrain/finetune

Why should pretraining and finetuning help, from a "training neural nets" perspective?

- Consider, provides parameters θ̂ by approximating min_θ 𝓛_pretrain(θ).
  - (The pretraining loss.)
- Then, finetuning approximates min_θ 𝓛_finetune(θ), starting at θ̂.
  - (The finetuning loss)
- The pretraining may matter because stochastic gradient descent sticks (relatively) close
  to θ̂ during finetuning.
  - So, maybe the finetuning local minima near θ̂ tend to generalize well!
  - And/or, maybe the gradients of finetuning loss near θ̂ propagate nicely!

*(The first bullet is elliptical as printed — it means "pretraining provides parameters θ̂
by approximating the minimizer of the pretraining loss".)*

## Slide 21 — Lecture Plan

1. A brief note on subword modeling *(greyed)*
2. Motivating model pretraining from word embeddings *(greyed)*
3. **Model pretraining three ways** *(current)*
   1. Encoders
   2. Encoder-Decoders
   3. Decoders
4. What do we think pretraining is teaching? *(greyed)*

## Slide 22 — Pretraining for three types of architectures

The neural architecture influences the type of pretraining, and natural use cases.

**Encoders** — small diagram of one row of blue blocks with arrows crossing in both
directions between all positions.
- Gets bidirectional context – can condition on future!
- How do we train them to build strong representations?

**Encoder-Decoders** — diagram of a blue block row (encoder) whose arrows feed a red block
row (decoder) with left-to-right arrows.
- Good parts of decoders and encoders?
- What's the best way to pretrain them?

**Decoders** — one row of red blocks with arrows going only left-to-right.
- Language models! What we've seen so far.
- Nice to generate from; can't condition on future words

## Slide 23 — Pretraining for three types of architectures

Same slide as 22, with **Encoders** highlighted and the other two greyed out — the section
marker for §3.1.

## Slide 24 — Pretraining encoders: what pretraining objective to use?

So far, we've looked at language model pretraining. But **encoders get bidirectional
context,** so we can't do language modeling!

Idea: replace some fraction of words in the input with a special [MASK] token; predict
these words.

> **h₁, …, h_T = Encoder(w₁, …, w_T)**
> **y_i ∼ A w_i + b**

Only add loss terms from words that are "masked out." If x̃ is the masked version of x,
we're learning p_θ(x | x̃). Called **Masked LM**.

The diagram shows a blue encoder over the input *I [M] to the [M]*, with red output blocks
above the two masked positions predicting *went* and *store*; the hidden states are
labelled h₁, …, h_T and the output layer *A, b*.

Credited **[Devlin et al., 2018]**.

## Slide 25 — BERT: Bidirectional Encoder Representations from Transformers

Devlin et al., 2018 proposed the "Masked LM" objective and **released the weights of a
pretrained Transformer**, a model they labeled BERT.

Some more details about Masked LM for BERT:

- Predict a random 15% of (sub)word tokens.
  - Replace input word with [MASK] 80% of the time
  - Replace input word with a random token 10% of the time
  - Leave input word unchanged 10% of the time (but still predict it!)
- Why? Doesn't let the model get complacent and not build strong representations of
  non-masked words. (No masks are seen at fine-tuning time!)

The diagram shows a blue **Transformer Encoder** over the input *I pizza to the [M]*,
where *pizza* is printed in red. Annotations point at the input positions: *pizza* is
**[Replaced]**, *to* is **[Not replaced]**, and *[M]* is **[Masked]**. Above, marked
**[Predict these!]**, the targets *went*, *to*, *store*.

Credited **[Devlin et al., 2018]**.

## Slide 26 — BERT: Bidirectional Encoder Representations from Transformers

- The pretraining input to BERT was two separate contiguous chunks of text:

A figure (from the BERT paper) shows the input `[CLS] my dog is cute [SEP] he likes play
##ing [SEP]` and three rows of embeddings summed position by position:

| Row | Values |
| --- | --- |
| Token Embeddings | E_[CLS], E_my, E_dog, E_is, E_cute, E_[SEP], E_he, E_likes, E_play, E_##ing, E_[SEP] |
| Segment Embeddings | E_A for the first six positions, E_B for the last five |
| Position Embeddings | E₀, E₁, E₂, E₃, E₄, E₅, E₆, E₇, E₈, E₉, E₁₀ |

with **+** signs between the rows.

- BERT was trained to predict whether one chunk follows the other or is randomly sampled.
  - Later work has argued this "next sentence prediction" is not necessary.

Credited **[Devlin et al., 2018, Liu et al., 2019]**.

## Slide 27 — BERT: Bidirectional Encoder Representations from Transformers

Details about BERT

- Two models were released:
  - BERT-base: 12 layers, 768-dim hidden states, 12 attention heads, 110 million params.
  - BERT-large: 24 layers, 1024-dim hidden states, 16 attention heads, 340 million params.
- Trained on:
  - BooksCorpus (800 million words)
  - English Wikipedia (2,500 million words)
- Pretraining is expensive and impractical on a single GPU.
  - BERT was pretrained with 64 TPU chips for a total of 4 days.
  - (TPUs are special tensor operation acceleration hardware)
- Finetuning is practical and common on a single GPU
  - "Pretrain once, finetune many times."

Credited **[Devlin et al., 2018]**.

## Slide 28 — BERT: Bidirectional Encoder Representations from Transformers

BERT was massively popular and hugely versatile; finetuning BERT led to new
state-of-the-art results on a broad range of tasks.

- **QQP:** Quora Question Pairs (detect paraphrase questions)
- **QNLI**: natural language inference over question answering data
- **SST-2**: sentiment analysis
- **CoLA**: corpus of linguistic acceptability (detect whether sentences are grammatical.)
- **STS-B**: semantic textual similarity
- **MRPC**: microsoft paraphrase corpus
- **RTE**: a small natural language inference corpus

The GLUE results table (dataset sizes on the second header row):

| System | MNLI-(m/mm) 392k | QQP 363k | QNLI 108k | SST-2 67k | CoLA 8.5k | STS-B 5.7k | MRPC 3.5k | RTE 2.5k | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pre-OpenAI SOTA | 80.6/80.1 | 66.1 | 82.3 | 93.2 | 35.0 | 81.0 | 86.0 | 61.7 | 74.0 |
| BiLSTM+ELMo+Attn | 76.4/76.1 | 64.8 | 79.8 | 90.4 | 36.0 | 73.3 | 84.9 | 56.8 | 71.0 |
| OpenAI GPT | 82.1/81.4 | 70.3 | 87.4 | 91.3 | 45.4 | 80.0 | 82.3 | 56.0 | 75.1 |
| BERT_BASE | 84.6/83.4 | 71.2 | 90.5 | 93.5 | 52.1 | 85.8 | 88.9 | 66.4 | 79.6 |
| **BERT_LARGE** | **86.7/85.9** | **72.1** | **92.7** | **94.9** | **60.5** | **86.5** | **89.3** | **70.1** | **82.1** |

Credited **[Devlin et al., 2018]**.

## Slide 29 — Limitations of pretrained encoders

Those results looked great! Why not used pretrained encoders for everything?

If your task involves generating sequences, consider using a pretrained decoder; BERT and
other pretrained encoders don't naturally lead to nice autoregressive (1-word-at-a-time)
generation methods.

Two diagrams side by side. **Left**: a red *Pretrained Encoder* over *Iroh goes to [MASK]
tasty tea*, producing a single output *make/brew/craft* above the masked slot — it fills a
blank, it does not continue the sequence. **Right**: a blue *Pretrained Decoder* over
*Iroh goes to make tasty tea*, producing *goes to make tasty tea END* one position at a
time.

## Slide 30 — Extensions of BERT

You'll see a lot of BERT variants like RoBERTa, SpanBERT, +++

Some generally accepted improvements to the BERT pretraining formula:

- RoBERTa: mainly just train BERT for longer and remove next sentence prediction!
- SpanBERT: masking contiguous spans of words makes a harder, more useful pretraining task

Two diagrams contrast the masking. **BERT** takes `[MASK] irr## esi## sti## [MASK] good`
and predicts the scattered tokens *It's* and *bly*. **SpanBERT** takes `It' [MASK] [MASK]
[MASK] [MASK] good` — one contiguous run of masks — and must predict *irr## esi## sti##
bly* together.

Credited **[Liu et al., 2019; Joshi et al., 2020]**.

## Slide 31 — Extensions of BERT

A takeaway from the RoBERTa paper: more compute, more data can improve pretraining even
when not changing the underlying Transformer encoder.

| Model | data | bsz | steps | SQuAD (v1.1/2.0) | MNLI-m | SST-2 |
| --- | --- | --- | --- | --- | --- | --- |
| **RoBERTa** with BOOKS + WIKI | 16GB | 8K | 100K | 93.6/87.3 | 89.0 | 95.3 |
| + additional data (§3.2) | 160GB | 8K | 100K | 94.0/87.7 | 89.3 | 95.6 |
| + pretrain longer | 160GB | 8K | 300K | 94.4/88.7 | 90.0 | 96.1 |
| + pretrain even longer | 160GB | 8K | 500K | **94.6/89.4** | **90.2** | **96.4** |
| **BERT_LARGE** with BOOKS + WIKI | 13GB | 256 | 1M | 90.9/81.8 | 86.6 | 93.7 |

Credited **[Liu et al., 2019; Joshi et al., 2020]**.

## Slide 32 — Full Finetuning vs. Parameter-Efficient Finetuning

Finetuning every parameter in a pretrained model works well, but is memory-intensive.
But **lightweight** finetuning methods adapt pretrained models in a constrained way.
Leads to **less overfitting** and/or **more efficient finetuning and inference.**

**Full Finetuning** — Adapt all parameters. Diagram: the whole *(Transformer, LSTM, ++)*
box is shaded pink over *… the movie was …*, with a ☺/☹ output.

**Lightweight Finetuning** — Train a few existing or new parameters. Diagram: the same box
stays red (frozen) except for a few small pink bars inside it, again with a ☺/☹ output.

Credited **[Liu et al., 2019; Joshi et al., 2020]**.

## Slide 33 — Parameter-Efficient Finetuning: Prefix-Tuning, Prompt tuning

Prefix-Tuning adds a **prefix** of parameters, and **freezes all pretrained parameters.**
The prefix is processed by the model just like real words would be.
Advantage: each element of a batch at inference could run a different tuned model.

The diagram shows the red *(Transformer, LSTM, ++)* box over *… the movie was …*; to the
left of the real input sit three columns of pink blocks, labelled by an arrow **Learnable
prefix parameters**, extending into the model's first layers. Output ☺/☹.

Credited **[Li and Liang, 2021; Lester et al., 2021]**.

## Slide 34 — Parameter-Efficient Finetuning: Low-Rank Adaptation

Low-Rank Adaptation Learns a low-rank "diff" between the pretrained and finetuned weight
matrices.
Easier to learn than prefix-tuning.

The diagram points from the model box (over *… the movie was …*, output ☺/☹) to **Each
weight matrix**: a large red square **W ∈ ℝ^(d×d)** beside two pink trapezoids
**B ∈ ℝ^(k×d)** and **A ∈ ℝ^(d×k)**, with the composed result written below as

> **W + AB**

Credited **[Hu et al., 2021]**.

## Slide 35 — Pretraining for three types of architectures

Same three-architecture slide as 22, now with **Encoder-Decoders** highlighted and the
other two greyed out — the section marker for §3.2.

## Slide 36 — Pretraining encoder-decoders: what pretraining objective to use?

For **encoder-decoders**, we could do something like **language modeling**, but where a
prefix of every input is provided to the encoder and is not predicted.

> **h₁, …, h_T = Encoder(w₁, …, w_T)**
> **h_{T+1}, …, h₂ = Decoder(w₁, …, w_T, h₁, …, h_T)**
> **y_i ∼ A h_i + b, i > T**

*(The subscript on the second line is printed as `h_2` in the deck; from the figure and
the surrounding text it should read h_{2T} — the decoder produces states T+1 through 2T.)*

The **encoder** portion benefits from bidirectional context; the **decoder** portion is
used to train the whole model through language modeling.

The diagram shows blue encoder blocks over **w₁, …, w_T** with arrows crossing in both
directions, feeding red decoder blocks over **w_{T+1}, …, w_{2T}** with left-to-right
arrows, emitting **w_{T+2}, …** at the top.

Credited **[Raffel et al., 2018]**.

## Slide 37 — Pretraining encoder-decoders: what pretraining objective to use?

What [Raffel et al., 2018] found to work best was **span corruption.** Their model: **T5**.

Replace different-length spans from the input with unique placeholders; decode out the
spans that were removed!

This is implemented in text preprocessing: it's still an objective that looks like
**language modeling** at the decoder side.

The worked example, in three highlighted strips:

- **Original text**: `Thank you for inviting me to your party last week.` (with *for
  inviting* and *last* struck through)
- **Inputs**: `Thank you <X> me to your party <Y> week.`
- **Targets**: `<X> for inviting <Y> last <Z>`

The same encoder-decoder block diagram as slide 36 sits on the right.

## Slide 38 — Pretraining encoder-decoders: what pretraining objective to use?

*(Page prints no slide number.)*

[Raffel et al., 2018] found encoder-decoders to work better than decoders for their tasks,
and span corruption (denoising) to work better than language modeling.

| Architecture | Objective | Params | Cost | GLUE | CNNDM | SQuAD | SGLUE | EnDe | EnFr | EnRo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ★ Encoder-decoder | Denoising | 2P | M | **83.28** | **19.24** | **80.88** | **71.36** | **26.98** | **39.82** | **27.65** |
| Enc-dec, shared | Denoising | P | M | 82.81 | 18.78 | **80.63** | 70.73 | 26.72 | 39.03 | **27.46** |
| Enc-dec, 6 layers | Denoising | P | M/2 | 80.88 | 18.97 | 77.59 | 68.42 | 26.38 | 38.40 | 26.95 |
| Language model | Denoising | P | M | 74.70 | 17.93 | 61.14 | 55.02 | 25.09 | 35.28 | 25.86 |
| Prefix LM | Denoising | P | M | 81.82 | 18.61 | 78.94 | 68.11 | 26.43 | 37.98 | 27.39 |
| Encoder-decoder | LM | 2P | M | 79.56 | 18.59 | 76.02 | 64.29 | 26.27 | 39.17 | 26.86 |
| Enc-dec, shared | LM | P | M | 79.60 | 18.13 | 76.35 | 63.50 | 26.62 | 39.17 | 27.05 |
| Enc-dec, 6 layers | LM | P | M/2 | 78.67 | 18.26 | 75.32 | 64.06 | 26.13 | 38.42 | 26.89 |
| Language model | LM | P | M | 73.78 | 17.54 | 53.81 | 56.51 | 25.23 | 34.31 | 25.38 |
| Prefix LM | LM | P | M | 79.68 | 17.84 | 76.87 | 64.86 | 26.28 | 37.51 | 26.76 |

## Slide 39 — Pretraining encoder-decoders: what pretraining objective to use?

*(Page prints no slide number.)*

A fascinating property of T5: it can be finetuned to answer a wide range of questions,
retrieving knowledge from its parameters.

NQ: Natural Questions
WQ: WebQuestions
TQA: Trivia QA

All "open-domain" versions

The figure at the top shows the salient-span-masking idea: above the *Pre-training* line,
a thought bubble reads "President Franklin D. Roosevelt was born in January 1882."; below
the *Fine-tuning* line, the input "When was Franklin D. Roosevelt born?" goes into a **T5**
box and out comes "1882".

| | NQ | WQ | TQA dev | TQA test | |
| --- | --- | --- | --- | --- | --- |
| Karpukhin et al. (2020) | **41.5** | 42.4 | **57.9** | – | |
| T5.1.1-Base | 25.7 | 28.2 | 24.2 | 30.6 | 220 million params |
| T5.1.1-Large | 27.3 | 29.5 | 28.5 | 37.2 | 770 million params |
| T5.1.1-XL | 29.5 | 32.4 | 36.0 | 45.1 | 3 billion params |
| T5.1.1-XXL | 32.8 | 35.6 | 42.9 | 52.5 | 11 billion params |
| T5.1.1-XXL + SSM | 35.2 | **42.8** | 51.9 | **61.6** | |

Credited **[Raffel et al., 2018]**.

## Slide 40 — Pretraining for three types of architectures

Same three-architecture slide as 22, now with **Decoders** highlighted and the other two
greyed out — the section marker for §3.3. The Decoders entry gains a third bullet:

- Language models! What we've seen so far.
- Nice to generate from; can't condition on future words
- **All the biggest pretrained models are Decoders.**

## Slide 41 — Pretraining decoders

When using language model pretrained decoders, we can ignore that they were trained to
model p(w_t | w_{1:t−1}).

We can finetune them by training a classifier on the last word's hidden state.

> **h₁, …, h_T = Decoder(w₁, …, w_T)**
> **y ∼ A h_T + b**

Where A and b are randomly initialized and specified by the downstream task.

Gradients backpropagate through the whole network.

The diagram shows a row of red decoder blocks over **w₁, …, w_T** with left-to-right
arrows; only the last hidden state feeds a grey **Linear** box labelled *A, b*, which
emits ☺/☹.

Margin note: [Note how the linear layer hasn't been pretrained and must be learned from
scratch.]

## Slide 42 — Pretraining decoders

It's natural to pretrain decoders as language models and then use them as generators,
finetuning their p_θ(w_t | w_{1:t−1})!

This is helpful in tasks **where the output is a sequence** with a vocabulary like that at
pretraining time!

- Dialogue (context=dialogue history)
- Summarization (context=document)

> **h₁, …, h_T = Decoder(w₁, …, w_T)**
> **w_t ∼ A h_{t−1} + b**

Where A, b were pretrained in the language model!

The diagram shows the decoder over **w₁ … w₅** emitting **w₂ … w₆**, with the *A, b* layer
drawn as small red blocks above every position rather than only the last.

Margin note: [Note how the linear layer has been pretrained.]

## Slide 43 — Generative Pretrained Transformer (GPT) [Radford et al., 2018]

2018's GPT was a big success in pretraining a decoder!

- Transformer decoder with 12 layers, 117M parameters.
- 768-dimensional hidden states, 3072-dimensional feed-forward hidden layers.
- Byte-pair encoding with 40,000 merges
- Trained on BooksCorpus: over 7000 unique books.
  - Contains long spans of contiguous text, for learning long-distance dependencies.
- The acronym "GPT" never showed up in the original paper; it could stand for "Generative
  PreTraining" or "Generative Pretrained Transformer"

*(The slide's bottom-right citation reads [Devlin et al., 2018]; the title citation
[Radford et al., 2018] is the one that matches the content.)*

## Slide 44 — Generative Pretrained Transformer (GPT) [Radford et al., 2018]

How do we format inputs to our decoder for **finetuning tasks?**

**Natural Language Inference:** Label pairs of sentences as *entailing/contradictory/neutral*

Premise: *The man is in the doorway*
Hypothesis: *The person is near the door*
→ **entailment**

Radford et al., 2018 evaluate on natural language inference.
Here's roughly how the input was formatted, as a sequence of tokens for the decoder.

> [START] *The man is in the doorway* [DELIM] *The person is near the door* [EXTRACT]

The linear classifier is applied to the representation of the [EXTRACT] token.

## Slide 45 — Generative Pretrained Transformer (GPT) [Radford et al., 2018]

GPT results on various *natural language inference* datasets.

| Method | MNLI-m | MNLI-mm | SNLI | SciTail | QNLI | RTE |
| --- | --- | --- | --- | --- | --- | --- |
| ESIM + ELMo [44] (5x) | – | – | <u>89.3</u> | – | – | – |
| CAFE [58] (5x) | 80.2 | 79.0 | <u>89.3</u> | – | – | – |
| Stochastic Answer Network [35] (3x) | <u>80.6</u> | <u>80.1</u> | – | – | – | – |
| CAFE [58] | 78.7 | 77.9 | 88.5 | <u>83.3</u> | | |
| GenSen [64] | 71.4 | 71.3 | – | – | <u>82.3</u> | 59.2 |
| Multi-task BiLSTM + Attn [64] | 72.2 | 72.1 | – | – | 82.1 | **61.7** |
| **Finetuned Transformer LM (ours)** | **82.1** | **81.4** | **89.9** | **88.3** | **88.1** | 56.0 |

## Slide 46 — Increasingly convincing generations (GPT2) [Radford et al., 2018]

*(Page prints no slide number.)*

We mentioned how pretrained decoders can be used **in their capacities as language
models.**
**GPT-2,** a larger version (1.5B) of GPT trained on more data, was shown to produce
relatively convincing samples of natural language.

The boxed sample:

> **Context (human-written):** In a shocking finding, scientist discovered a herd of
> unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even
> more surprising to the researchers was the fact that the unicorns spoke perfect English.
>
> **GPT-2:** The scientist named the population, after their distinctive horn, Ovid's
> Unicorn. These four-horned, silver-white unicorns were previously unknown to science.
>
> Now, after almost two centuries, the mystery of what sparked this odd phenomenon is
> finally solved.
>
> Dr. Jorge Pérez, an evolutionary biologist from the University of La Paz, and several
> companions, were exploring the Andes Mountains when they found a small valley, with no
> other animals or humans. Pérez noticed that the valley had what appeared to be a natural
> fountain, surrounded by two peaks of rock and silver snow.

## Slide 47 — GPT-3, In-context learning, and very large models

So far, we've interacted with pretrained models in two ways:

- Sample from the distributions they define (maybe providing a prompt)
- Fine-tune them on a task we care about, and take their predictions.

Very large language models seem to perform some kind of learning **without gradient steps**
simply from examples you provide within their contexts.

GPT-3 is the canonical example of this. The largest T5 model had 11 billion parameters.
**GPT-3 has 175 billion parameters.**

## Slide 48 — GPT-3, In-context learning, and very large models

Very large language models seem to perform some kind of learning **without gradient steps**
simply from examples you provide within their contexts.

The in-context examples seem to specify the task to be performed, and the conditional
distribution mocks performing the task to a certain extent.

**Input (prefix within a single Transformer decoder context):**

> "  thanks -> merci
>    hello -> bonjour
>    mint -> menthe
>    otter ->            "

**Output (conditional generations):**

>    loutre…"

## Slide 49 — GPT-3, In-context learning, and very large models

Very large language models seem to perform some kind of learning **without gradient steps**
simply from examples you provide within their contexts.

The figure (from the GPT-3 paper) shows three stacked example sequences under a long
horizontal arrow labelled **Learning via SGD during unsupervised pre-training**; each
sequence has a downward arrow labelled **In-context learning**.

- *sequence #1* — arithmetic: `5 + 8 = 13`, `7 + 2 = 9`, `1 + 0 = 1`, `3 + 4 = 7`,
  `5 + 9 = 14`, `9 + 8 = 17`
- *sequence #2* — unscrambling: `gaot => goat`, `sakne => snake`, `brid => bird`,
  `fsih => fish`, `dcuk => duck`, `cmihp => chimp`
- *sequence #3* — translation: `thanks => merci`, `hello => bonjour`, `mint => menthe`,
  `wall => mur`, `otter => loutre`, `bread => pain`

## Slide 50 — Scaling Efficiency: how do we best use our compute

GPT-3 was **175B parameters** and trained on **300B** tokens of text.
Roughly, the cost of training a large transformer scales as **parameters\*tokens**
Did OpenAI strike the right parameter-token data to get the best model? No.

| Model | Size (# Parameters) | Training Tokens |
| --- | --- | --- |
| LaMDA (Thoppilan et al., 2022) | 137 Billion | 168 Billion |
| GPT-3 (Brown et al., 2020) | 175 Billion | 300 Billion |
| Jurassic (Lieber et al., 2021) | 178 Billion | 300 Billion |
| *Gopher* (Rae et al., 2021) | 280 Billion | 300 Billion |
| MT-NLG 530B (Smith et al., 2022) | 530 Billion | 270 Billion |
| *Chinchilla* | 70 Billion | 1.4 Trillion |

An arrow points at the Chinchilla row: **This 70B parameter model is better than the much
larger other models!**

## Slide 51 — The prefix as task specification and scratch pad: chain-of-thought

The slide is the chain-of-thought figure, two columns.

**Standard Prompting** — Model Input:

> Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis
> balls. How many tennis balls does he have now?
> A: The answer is 11.
>
> Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many
> apples do they have?

Model Output: `A: The answer is 27.` ❌

**Chain-of-Thought Prompting** — Model Input:

> Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis
> balls. How many tennis balls does he have now?
> A: *Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 =
> 11.* The answer is 11.
>
> Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many
> apples do they have?

Model Output: `A: The cafeteria had 23 apples originally. They used 20 to make lunch. So
they had 23 - 20 = 3. They bought 6 more apples, so they have 3 + 6 = 9. The answer is 9.` ✔

(In the figure, the reasoning inserted into the prompt is highlighted blue and the
generated reasoning is highlighted green.)

Credited **[Wei et al., 2023]**.

## Slide 52 — Outline

1. A brief note on subword modeling *(greyed)*
2. Motivating model pretraining from word embeddings *(greyed)*
3. Model pretraining three ways *(greyed)*
   1. Encoders
   2. Encoder-Decoders
   3. Decoders
4. **What do we think pretraining is teaching?** *(current)*

## Slide 53 — What kinds of things does pretraining teach?

There's increasing evidence that pretrained models learn a wide variety of things about
the statistical properties of language. Taking our examples from the start of class:

- *Stanford University is located in __________, California.* **[Trivia]**
- *I put ___ fork down on the table.* **[syntax]**
- *The woman walked across the street, checking for traffic over ___ shoulder.*
  **[coreference]**
- *I went to the ocean to see the fish, turtles, seals, and _____.* **[lexical
  semantics/topic]**
- *Overall, the value I got from the two hours watching it was the sum total of the popcorn
  and the drink. The movie was ___.* **[sentiment]**
- Iroh went into the kitchen to make some tea. Standing next to Iroh, Zuko pondered his
  destiny. Zuko left the ______. **[some reasoning – this is harder]**
- I was thinking about the sequence that goes 1, 1, 2, 3, 5, 8, 13, 21, ____ **[some basic
  arithmetic; they don't learn the Fibonnaci sequence]**
- Models also learn – and can exacerbate racism, sexism, all manner of bad biases.
- More on all this in the interpretability lecture!

*("Fibonnaci" is the deck's spelling.)*

## Slide 54 — Parting remarks

These models are still not well-understood.
"Small" models like BERT have become general tools in a wide range of settings.
More on this in later lectures!
Assignment 5 out Thursday! Tuesday's and today's lectures in its subject matter.
