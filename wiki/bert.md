# BERT and masked language modeling

The pretrained encoder that reorganized NLP in 2018. **BERT** — Bidirectional Encoder
Representations from Transformers (Devlin et al., 2018) — popularized the masked language
modeling objective and, more importantly, released the weights, so that anyone could
fine-tune a strong language model instead of designing an architecture from scratch. Covered
in [lecture 9](09-pretraining.md), slides 24–34.

## Why encoders need a different objective

A [Transformer](transformer.md) encoder has bidirectional context: every position attends to
every other position, in both directions. That makes ordinary
[language modeling](language-modeling.md) useless as a pretraining task, because the target is
already visible in the input — "predicting the next word is a trivial task … because I could
just look at it, see what it is, and then copy it over" (≈37:08).

**Masked language modeling** (slide 24) restores the difficulty by corrupting the input.
Replace some fraction of tokens with a special `[MASK]` symbol, run the encoder, and predict
the original tokens at the masked positions:

$$h_1, \dots, h_T = \mathrm{Encoder}(w_1, \dots, w_T)$$

$$y_i \sim A w_i + b$$

where $h_1, \dots, h_T$ are the encoder's contextual hidden states, and $A$ and $b$ are the
output projection from hidden size to vocabulary size. Loss terms come **only** from masked
positions. Writing $\tilde{x}$ for the corrupted version of the input $x$, the model is
learning

$$p_\theta(x \mid \tilde{x})$$

This is the same reconstruct-the-input idea as language modeling, but with the whole
surrounding context available rather than only the left side (≈38:39).

## BERT's masking recipe

Slide 25 gives the details, all in [subword tokens](subword-modeling.md). BERT predicts a
random **15%** of tokens, and treats them in three different ways:

| Treatment | Share | Model sees | Model must predict |
| --- | --- | --- | --- |
| Masked | 80% | `[MASK]` | the original token |
| Replaced | 10% | a random token from the vocabulary | the original token |
| Unchanged | 10% | the original token | the original token |

The 80/10/10 split exists to stop the model from learning "only masked positions matter." If
every prediction target were a `[MASK]`, the encoder could build weak representations for
everything else — and at fine-tuning time there are no mask tokens at all, so the model would
be poorly prepared for the input it actually receives. Mixing in replaced and unchanged tokens
means any position might be a target, so the model has to represent all of them well: "it
doesn't let the model get complacent" (slide 25, ≈42:33).

## Next sentence prediction, and why it was dropped

BERT's pretraining input was two contiguous chunks of text, distinguished by a learned
**segment embedding** ($E_A$ or $E_B$) added element-wise to the token and position
embeddings (slide 26). Alongside masked LM, BERT was trained to predict whether segment B
genuinely followed segment A or was sampled at random from elsewhere, using a classifier head
on the special `[CLS]` token at the start of every sequence. The intent was to teach
long-distance coherence and to prepare the model for the many NLP tasks defined over *pairs*
of texts (≈47:08).

Later work (Liu et al., 2019) showed it was unnecessary, and the lecture offers two reasons
(≈44:50):

1. **It halves the effective context.** Each segment was around 250 words; the paper that
   removed NSP used a single 500-word segment instead. Longer context is one of the main things
   pretraining buys — "if I see one word it's hard to know what it's supposed to mean, but if I
   see a thousand words around it it's much clearer what its role in that context is."
2. **Models are very bad at the task anyway.** Later work found NSP accuracy poor enough that
   the objective may simply have been too hard to provide useful signal.

Note the addition is element-wise, not concatenation. Token, segment and position embeddings
all share one dimensionality, which is the convention throughout these networks: "you always
have exactly the same number of dimensions everywhere, at every layer … it just makes
everything very simple" (≈44:04).

## Scale, cost, and the asymmetry that made it useful

Slide 27's numbers:

| | Layers | Hidden | Heads | Parameters |
| --- | --- | --- | --- | --- |
| BERT-base | 12 | 768 | 12 | 110 million |
| BERT-large | 24 | 1024 | 16 | 340 million |

Trained on BooksCorpus (800 million words) and English Wikipedia (2,500 million words), using
64 TPU chips for four days.

The slide's own summary is the important part: **"Pretrain once, finetune many times."**
Pretraining was impractical outside a large company — "it was Google doing this, and they
released it, and we were like, oh, who has that kind of compute but Google" — while
fine-tuning is practical on a single, even small, GPU (≈52:30). That asymmetry, plus the
released weights and the Hugging Face library, is what put BERT in everyone's hands.

> The transcript at ≈52:30 garbles these corpus sizes; the lecturer starts to quote 800
> million, corrects himself mid-sentence, and the captions mangle the figures he lands on.
> Slide 27 is the figure of record and is what is quoted above.

## How you actually use it

Slide 28's GLUE table is the argument. Before BERT, each task had its own carefully crafted
architecture; after, a single pretrained encoder fine-tuned per task beat all of them:

| System | MNLI-(m/mm) | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pre-OpenAI SOTA | 80.6/80.1 | 66.1 | 82.3 | 93.2 | 35.0 | 81.0 | 86.0 | 61.7 | 74.0 |
| BiLSTM+ELMo+Attn | 76.4/76.1 | 64.8 | 79.8 | 90.4 | 36.0 | 73.3 | 84.9 | 56.8 | 71.0 |
| OpenAI GPT | 82.1/81.4 | 70.3 | 87.4 | 91.3 | 45.4 | 80.0 | 82.3 | 56.0 | 75.1 |
| BERT-base | 84.6/83.4 | 71.2 | 90.5 | 93.5 | 52.1 | 85.8 | 88.9 | 66.4 | 79.6 |
| **BERT-large** | **86.7/85.9** | **72.1** | **92.7** | **94.9** | **60.5** | **86.5** | **89.3** | **70.1** | **82.1** |

The tasks span paraphrase detection (QQP, MRPC), natural language inference (QNLI, RTE,
MNLI), sentiment ([SST-2](softmax-and-cross-entropy.md)), grammatical acceptability (CoLA) and
semantic similarity (STS-B). The lecturer's account of the reception: "the field was taken
aback in a way that's hard to describe … roughly all of that was blown out of the water by
'just build a big Transformer and just teach it to predict the missing words a whole bunch,
and then fine-tune it on each of these tasks'" (≈49:26).

Mechanically, fine-tuning is simple (≈51:44). The encoder produces one contextual vector per
input token. Remove the affine layer that mapped hidden states to the vocabulary for masked-LM
prediction, pick a position — for sentence classification, "arbitrarily, maybe the last word in
the sentence" — attach a linear classifier mapping that vector to your label set, and fine-tune
the whole network.

## Limitations

Slide 29 is the reason BERT is not the answer to everything. A pretrained encoder fills in
blanks; it has no natural notion of predicting the next word given everything before it, so it
"doesn't naturally lead to nice autoregressive (1-word-at-a-time) generation methods." Use BERT
when you want a good representation of a document to classify — topic labels, toxic/non-toxic —
and a pretrained decoder when you need to produce a sequence (≈53:16). See
[natural language generation](natural-language-generation.md).

## Extensions of BERT

Slides 30–31 name the two changes that stuck.

**RoBERTa** (Liu et al., 2019) is, in the lecture's summary, "mainly just train BERT for longer
and remove next sentence prediction." Slide 31's table shows the payoff from compute and data
alone, with the Transformer encoder unchanged: BERT-large on 13GB reaches 90.9/81.8 on SQuAD
and 86.6 on MNLI-m, while RoBERTa on 160GB trained for 500K steps reaches 94.6/89.4 and 90.2.
The practical advice is blunt — "if you're thinking of using BERT, just use RoBERTa, it's
better" (≈54:49) — and the broader lesson is that "we really don't know a whole lot about the
best practices for training these things."

**SpanBERT** (Joshi et al., 2020) masks *contiguous spans* rather than scattered tokens. The
motivation is a direct consequence of [subword tokenization](subword-modeling.md): masking one
subword of `irr## esi## sti## bly` is nearly free, because the neighbouring subwords give the
word away. Masking the whole run makes the task genuinely harder and the representations more
useful (≈54:02).

## Related pages

- [Lecture 9 — Pretraining](09-pretraining.md) — where this is taught.
- [Pretraining and fine-tuning](pretraining-and-finetuning.md) — the paradigm, the other two
  architecture classes, and parameter-efficient adaptation.
- [Transformer](transformer.md) — the encoder BERT is built from.
- [Self-attention](self-attention.md) — why bidirectional context makes language modeling
  unusable.
- [Subword modeling](subword-modeling.md) — the tokenization BERT's 15% is measured in.
- [Softmax and cross-entropy](softmax-and-cross-entropy.md) — the output layer and loss.
- [GPT and in-context learning](gpt-and-in-context-learning.md) — the decoder alternative.
