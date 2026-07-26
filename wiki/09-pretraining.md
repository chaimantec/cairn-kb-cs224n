# Lecture 9 — Pretraining

The lecture where the course stops building task-specific networks and starts adapting
general ones. Having established the Transformer in
[lecture 8](08-self-attention-and-transformers.md), John Hewitt asks what to *train* it on
when you have far more raw text than labels. The answer — hide part of the input and make
the network reconstruct it — turns out to be enough to teach a model syntax, coreference,
lexical semantics, sentiment, world facts and a little reasoning, all without a single
annotation. The lecture works through the resulting **pretrain-then-finetune** paradigm for
all three Transformer shapes, and ends at the point where models became large enough that
fine-tuning stopped being necessary at all.

**Slide-by-slide text of this deck: [54 slides](../raw/slides/09-pretraining.md)** —
printed slide numbers match PDF pages 1:1.

Slides PDF: [Lecture 9 — pretraining](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture9-pretraining.pdf) ·
[Full transcript](../raw/transcripts/09-pretraining.md)

> **A note on provenance.** This lecture is a **Winter 2023** recording by John Hewitt, not
> a Spring 2024 one, so its deck comes from the Winter 2023 course archive rather than the
> Spring 2024 site the rest of this knowledge base was built from. Slide 2's reminder
> ("Assignment 5 is out on Thursday! It covers lecture 8 and lecture 9") matches the
> lecturer's spoken announcement word for word. A separate, later 64-page deck exists on the
> Spring 2024 site whose slide numbers do **not** line up with this video; cite the file
> above.

## The finite-vocabulary problem, and subwords

The lecture opens on a piece of unfinished business (slides 3–6). Everything up to this
point — [word2vec](word2vec.md), the RNNs, the Transformer — assumed a fixed vocabulary $V$
built from the training set, with every unseen word at test time mapped to a single `UNK`
token. Slide 3 shows why that is worse than it sounds: *taaaaasty*, the misspelling *laern*
and the coinage *Transformerify* all collapse onto the same vector, even though a human
reads all three without difficulty. English makes this look like a minor annoyance; slide 4
puts up a Swahili conjugation table with hundreds of forms of a single verb to show that in
a morphologically rich language, treating each surface form as its own vocabulary entry is
hopeless (≈4:45).

The fix is to stop trying to enumerate words at all and build the vocabulary out of *pieces*
of words. The full treatment of byte-pair encoding, the `##` continuation convention, and
why "fewest subwords" is the right tokenization objective is at
[subword modeling](subword-modeling.md).

## From word embeddings to whole pretrained models

Slides 8–10 frame the shift historically. The course opened with Firth's "You shall know a
word by the company it keeps" (1957), which motivated
[distributional semantics](distributional-semantics.md) and word2vec. Slide 8 produces an
*earlier* Firth quote that word2vec does not satisfy:

> "… the complete meaning of a word is always contextual, and no study of meaning apart from
> a complete context can be taken seriously." (Firth 1935)

The example is *I **record** the **record***: word2vec assigns one vector to the string
`record`, so that vector has to carry a blend of the verb and noun senses and cannot
specialize (≈16:15). This is the same limitation
[word senses and polysemy](word-senses-and-polysemy.md) circles in lecture 2, now stated as
the reason to pretrain contextual models.

Slide 9 shows the circa-2017 setup: pretrained word embeddings at the bottom, everything
above them randomly initialized and learned from the task's own labels. Two problems follow
— the downstream task's labeled data has to teach *all* contextual aspects of language by
itself, and most of the network's parameters start from noise. Slide 10 is the alternative:
pretrain the whole stack. What that buys you, in the lecture's own three-part framing, is
strong **representations of language**, strong **parameter initializations**, and
**probability distributions over language you can sample from**.

## What reconstructing the input actually teaches

Slides 11–17 are seven cloze examples shown one at a time with no commentary, and slide 53
returns at the end of the lecture to label each with the capability it probes. Together they
are the lecture's argument that a single fill-in-the-blank objective is a proxy for most of
NLP:

| Cloze prompt | What predicting the blank requires |
| --- | --- |
| *Stanford University is located in ____, California.* | trivia / world knowledge |
| *I put ___ fork down on the table.* | syntax |
| *The woman walked across the street, checking for traffic over ___ shoulder.* | coreference |
| *I went to the ocean to see the fish, turtles, seals, and _____.* | lexical semantics / topic |
| *…the value I got from the two hours watching it was the sum total of the popcorn and the drink. The movie was ___.* | sentiment |
| *Iroh went into the kitchen to make some tea. Standing next to Iroh, Zuko pondered his destiny. Zuko left the ______.* | reasoning about entities and locations — "harder" |
| *I was thinking about the sequence that goes 1, 1, 2, 3, 5, 8, 13, 21, ____* | arithmetic — and slide 53's verdict is that models **don't** learn the Fibonacci sequence |

The sentiment example makes the practical point sharpest (≈22:26): the text is ordinary
prose that nobody labeled, but predicting *bad* rather than *good* after "the movie was" is
implicitly solving sentiment analysis. Slide 53 also records the other side — models "learn,
and can exacerbate, racism, sexism, all manner of bad biases."

## The pretrain / finetune paradigm

Slides 18–20 give the two-step recipe and an honest account of why it works. Pretraining
runs [language modeling](language-modeling.md) — model $p_\theta(w_t \mid w_{1:t-1})$ — over
a large corpus and saves the parameters; fine-tuning starts gradient descent from those
parameters on the task you actually care about. Slide 20's explanation is deliberately
hedged: pretraining approximates a minimizer $\hat{\theta}$ of the pretraining loss,
fine-tuning then approximates a minimizer of the fine-tuning loss *starting from*
$\hat{\theta}$, and if you could genuinely solve the second minimization the starting point
should not matter — but it does, enormously. The offered intuitions are that stochastic
gradient descent stays relatively close to $\hat{\theta}$, so the minima it reaches are ones
near $\hat{\theta}$, which "tend to generalize well," and/or that gradients propagate
nicely there (≈30:11).

Asked why this beats just adding layers and labeled data, the lecture gives the scale
argument — on the order of 5–10 trillion words of unlabeled internet text against perhaps a
million words of labeled data — plus a robustness argument: a system trained only on today's
movie reviews degrades when people start writing tomorrow's slightly differently, where a
broadly pretrained model adapts (≈32:30). Full treatment, including
parameter-efficient fine-tuning, is at
[pretraining and fine-tuning](pretraining-and-finetuning.md).

## Pretraining three ways

Slides 21–51 work through the three Transformer shapes in turn. **Note the ordering**: slide
2's lecture plan lists decoders first, but the deck and the lecture present encoders →
encoder-decoders → decoders, saving decoders for last because "all the biggest pretrained
models are decoders" (slide 40).

**Encoders** (slides 24–34) cannot be trained by language modeling at all — with
bidirectional context, predicting the next word is trivial because the model can already see
it (≈37:08). The fix is **masked language modeling**: corrupt the input, predict the
corrupted positions, take loss only there. This is BERT, and it is covered in full at
[BERT and masked language modeling](bert.md), along with RoBERTa, SpanBERT, and why a
pretrained encoder is the wrong tool when you need to *generate* text (slide 29).

**Encoder-decoders** (slides 35–39) could be trained as a prefix language model — give the
encoder the first half of a text, generate the second half with the decoder — but Raffel et
al. found **span corruption** works better. Replace variable-length spans of the input with
unique placeholder tokens and have the decoder emit the removed spans:

> Original: `Thank you for inviting me to your party last week.`
> Inputs: `Thank you <X> me to your party <Y> week.`
> Targets: `<X> for inviting <Y> last <Z>`

This is T5, and slide 38's table shows encoder-decoders with a denoising objective beating
both decoder-only models and language-modeling objectives across GLUE, SQuAD, SuperGLUE and
three translation pairs. Span corruption is what Assignment 5 implements (≈58:42). Slide 39
adds the result the lecturer calls "fascinating": with **salient span masking**, T5
fine-tuned on trivia questions answers *new* trivia questions by implicitly retrieving facts
stored in its parameters at pretraining time — "sometimes less than 50% of the time or
whatever, but much more than random chance." The caveat he attaches is the one that still
applies: "the answers always look very fluent, they always look very reasonable, but they're
frequently wrong. And that's still true of things like ChatGPT" (≈1:01:46).

**Decoders** (slides 40–51) are ordinary language models, and they can be used two ways:
throw away the language-model head and put a classifier on the last hidden state (slide 41),
or keep the pretrained output layer and fine-tune the model as a generator (slide 42). The
GPT line, in-context learning, chain-of-thought prompting and the Chinchilla scaling result
are covered at [GPT and in-context learning](gpt-and-in-context-learning.md).

## The point where the paradigm changes

The last third of the lecture is about a discontinuity. GPT-3 (175 billion parameters,
300 billion training tokens) can perform tasks with no gradient steps at all, purely from
examples placed in its context window (slides 47–49). The lecturer flags how strange this
should be: "it's not obvious from just the language modeling signal … that it would, as a
result of that training, learn to perform seemingly quite complex things as a function of its
context" (≈1:08:42). Whether the model is genuinely learning the task in context, or
recombining task fragments it saw during pretraining, is left explicitly open — "the actual
story, we're not totally sure — something in the middle, it seems like" (≈1:10:13).

Two closing cautions are worth carrying forward. Slide 50's Chinchilla comparison shows a
70-billion-parameter model beating models two to seven times its size by training on far more
data, so bigger was not what OpenAI should have been buying with its compute. And the
emergent behaviours are, in the lecturer's words, "both very powerful and exceptionally hard
to understand, and very hard, you should think, to trust, because it's unclear what its
capabilities are and what its limitations are, where it will fail" (≈1:13:17).

## Answers to questions from the floor

- **Do we add the three subword embeddings for one word together?** No — they stay separate
  tokens through the model. If you want a single representation afterwards you might average
  the contextual vectors or take the last, but at that point "it's unclear what to do"
  (≈9:22).
- **Do we still learn one vector per word if we pretrain everything?** Yes, one
  *non-contextual* input embedding per vocabulary item; the contextual vectors are what the
  Transformer computes on top of them (≈27:52).
- **How do you evaluate something that's supposed to be general?**
  [Perplexity](perplexity.md) correlates surprisingly well with downstream ability and is
  cheap to track during training; beyond that, the field builds large benchmark suites and a
  new method argues for generality by winning across all of them — which the lecturer
  concedes is "very, very difficult, it's sort of ill-defined even" (≈28:38).
- **Does the choice of optimizer change the "stays near $\hat{\theta}$" story?** Any common
  first-order method — Adam, AdaGrad — behaves similarly; other families are simply not used,
  "so who knows" (≈31:44).
- **What if you have lots of fine-tuning data too?** Then training from scratch is "almost
  never" right. The usual recipe is three stages: pretrain on a broad corpus, continue
  pretraining with language modeling on your task's text with the labels stripped off, then
  fine-tune with labels — the "Don't Stop Pretraining" result (≈1:16:21).

## Related pages

- [Subword modeling](subword-modeling.md) — byte-pair encoding and the end of the fixed
  vocabulary.
- [Pretraining and fine-tuning](pretraining-and-finetuning.md) — the paradigm, the three
  architecture classes, and parameter-efficient adaptation.
- [BERT and masked language modeling](bert.md) — the encoder branch in detail.
- [GPT and in-context learning](gpt-and-in-context-learning.md) — the decoder branch, and
  what changed at scale.
- [Language modeling](language-modeling.md) — the objective pretraining reuses.
- [Transformer](transformer.md) — the architecture all three variants are built from.
- [Lecture 10 — Natural Language Generation](10-natural-language-generation.md) — what to do
  with a pretrained decoder once you have one.
