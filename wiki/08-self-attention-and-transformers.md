# Lecture 8 — Self-Attention and Transformers

The lecture that replaces recurrence outright. Having motivated attention as a fix bolted
onto an RNN encoder-decoder in [lecture 7](07-attention-final-projects-and-llm-intro.md),
Anna Goldie asks the question the field asked in 2017: is attention *all* we need? The
answer, worked out across the lecture, is "not quite" — attention within a sequence
(**self-attention**) genuinely can replace recurrence, but it needs three fixes before it
does (a way to represent word order, nonlinearities, and future-masking), plus several more
engineering pieces (multi-head attention, scaled dot-product attention, residual
connections, layer normalization) before it's the architecture actually used in practice:
the **Transformer**.

**Slide-by-slide text of this deck: [62 slides](../raw/slides/08-self-attention-and-transformers.md)**
— printed slide numbers match PDF pages 1:1.

Slides PDF: [Lecture 8 — transformers](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture08-transformers.pdf) ·
[Full transcript](../raw/transcripts/08-self-attention-and-transformers.md)

## Why Transformers, and why now

Slides 4–14 open with the empirical case before any architecture is shown: Transformer
results on machine translation BLEU, on SuperGLUE, on large-language-model leaderboards,
and — via Vision Transformers and AlphaFold2 — well outside NLP entirely, into image
classification and protein folding. The scaling-laws plots on slide 14 (Kaplan et al.,
2020) make the stakes concrete: Transformer language-modeling loss falls smoothly, as a
clean power law, as model size, data, and compute all grow together, with no sign of
flattening out at the scales tested — the promise (and the open question) that motivated
the field's shift to ever-larger Transformer-based models. Full discussion of *why*
recurrence loses on the architectural merits — linear interaction distance and
non-parallelizability — is at
[self-attention § why move beyond recurrence](self-attention.md#why-move-beyond-recurrence)
(lecture slides 15–24).

## From query-key-value to a working building block

Slides 25–50 build self-attention up from its bare mechanism — attention as a "fuzzy
hashtable" lookup, where a query matches every key to some degree and returns a weighted
sum of values (slide 28) — to something that can actually stand in for an RNN. The lecture
frames this as three problems that plain self-attention has, fixed one at a time: no
sense of sequence order (fixed with position embeddings), no nonlinearities (fixed with a
position-wise feed-forward layer), and the ability to illegally peek at future tokens
(fixed with masking, needed in decoders but not encoders). The full mechanism, the
query-key-value recipe, and all three fixes are covered at
[self-attention](self-attention.md).

## The Transformer proper

Slides 33–56 assemble the actual architecture, adding the pieces that make the "minimal"
self-attention building block into something that trains well at scale: **multi-head
attention** (running several attention operations in parallel so a word can attend to
different other words for different reasons at once), **scaled dot-product attention** (a
$1/\sqrt{d_k}$ correction that keeps softmax inputs well-behaved once
[layer normalization](transformer.md#residual-connections-and-layer-normalization) is in
play), and the **residual-connection-plus-layer-norm** pairing that appears around every
sublayer. These combine into three architecture shapes — encoder-only (no masking,
bidirectional), decoder-only (masked, used for language modeling), and the original
encoder-decoder (with an added cross-attention sublayer in the decoder, queries from the
decoder and keys/values from the encoder) — the last of which is how the "Attention Is All
You Need" paper itself presents the Transformer. Full architectural detail, all the
equations, and the encoder/decoder/encoder-decoder distinction are at
[Transformer](transformer.md).

## What's still unresolved

Slides 57–61 close the lecture by pushing back against treating the Transformer as a
finished answer: self-attention's $O(n^2)$ compute cost is a real step backwards from
recurrence's $O(n)$ cost as sequences get long, position representations remain an open
design choice, and a large controlled study found that most *proposed* architectural
modifications don't actually transfer or meaningfully help once tested consistently. See
[Transformer § drawbacks and variants](transformer.md#drawbacks-and-variants) for Linformer,
BigBird, and the rest.

## Related pages

- [Self-attention](self-attention.md) — the query-key-value mechanism, why recurrence lost,
  and the three fixes (position, nonlinearity, masking).
- [Transformer](transformer.md) — multi-head attention, scaled dot-product attention,
  residual connections and layer norm, the encoder/decoder/encoder-decoder shapes, and the
  drawbacks-and-variants discussion.
- [Attention](attention.md) — the encoder-decoder attention mechanism this lecture
  generalizes into self-attention.
- [Vanishing and exploding gradients](vanishing-and-exploding-gradients.md) — residual
  connections, covered there in general.
- [Lecture 7 — Attention, Final Projects and LLM Intro](07-attention-final-projects-and-llm-intro.md) —
  the lecture this one continues from.
