# LSTM — Long Short-Term Memory

The gated [recurrent neural network](recurrent-neural-networks.md) that solves the
[vanishing gradient problem](vanishing-and-exploding-gradients.md), and the architecture
Assignment 3 uses. Taught in [lecture 6](06-sequence-to-sequence-models.md), slides 19–25.

## Parsing the name

Worth doing, because Manning says people often do not (≈23:23). The thing being modelled is
**short-term memory** — the distinction humans draw between what they heard recently and
what they have permanently stored. Human short-term memory holds things for quite a while:
in a conversation you can still recall what someone said several turns ago and bring it back
up. Simple RNNs had a short-term memory of about **seven tokens**. The goal was to make
short-term memory *long*. Hence **long short-term memory** — not "long-short term memory".

## History

Slide 20, and Manning's expansion at ≈24:55–27:59.

- **Hochreiter and Schmidhuber, 1997** proposed it as a solution to vanishing gradients.
  This is the paper everyone cites.
- **Gers, Schmidhuber and Cummins, 2000** introduced the **forget gate** — which Manning
  stresses is "a crucial part of the modern LSTM" that was *not* in the original paper. The
  first LSTM was purely additive, and that turned out to be imperfect: if you keep adding
  over a long sequence it becomes dysfunctional after a point (≈42:06).
- **Alex Graves**, Schmidhuber's student in the mid-2000s, did the work that got LSTMs
  noticed — and in the same work invented **CTC** (connectionist temporal classification) for
  speech recognition.
- Graves went to Toronto as a **postdoc with Geoff Hinton**, which brought attention to
  LSTMs; Hinton went to **Google in 2013**, and LSTMs became the dominant framework in the
  **2014–16** period.

Manning adds the human side (≈25:40): Schmidhuber's group did crucial foundational work in
the late 90s when almost everyone else had given up on neural networks, at a time when that
was very much *not* the route to a well-compensated job at Google, Meta or OpenAI. Gers left
AI for multimedia; Sepp Hochreiter spent years publishing in bioinformatics before returning
to neural networks recently. His summary of the startup analogy: "this is what you call being
too early."

## Hidden state and cell state

Slide 21. On each step there are now **two** vectors, both of length *n*:

- the **hidden state** h⁽ᵗ⁾, as before, and
- the **cell state** c⁽ᵗ⁾, which stores long-term information.

The LSTM can **read**, **erase** and **write** the cell — the slide's phrasing is that the
cell "becomes conceptually rather like RAM in a computer".

Which information gets erased, written and read is controlled by three **gates**, also
vectors of length *n*. Gates are **calculated things whose values are probabilities**: each
element lies between 0 and 1 and can be open (1), closed (0) or in between. Crucially they
are **dynamic** — computed from the current context, not fixed parameters.

## The equations

Slide 22. Each gate is a sigmoid of the same shape of expression a vanilla RNN step uses;
the sigmoid is what forces values into [0, 1].

    f⁽ᵗ⁾ = σ( W_f h⁽ᵗ⁻¹⁾ + U_f x⁽ᵗ⁾ + b_f )     forget gate:  what to keep from the last cell state
    i⁽ᵗ⁾ = σ( W_i h⁽ᵗ⁻¹⁾ + U_i x⁽ᵗ⁾ + b_i )     input gate:   what of the new content to write
    o⁽ᵗ⁾ = σ( W_o h⁽ᵗ⁻¹⁾ + U_o x⁽ᵗ⁾ + b_o )     output gate:  what of the cell to expose to h

    c̃⁽ᵗ⁾ = tanh( W_c h⁽ᵗ⁻¹⁾ + U_c x⁽ᵗ⁾ + b_c )  candidate new cell content
    c⁽ᵗ⁾ = f⁽ᵗ⁾ ⊙ c⁽ᵗ⁻¹⁾ + i⁽ᵗ⁾ ⊙ c̃⁽ᵗ⁾          erase, then write
    h⁽ᵗ⁾ = o⁽ᵗ⁾ ⊙ tanh c⁽ᵗ⁾                     read

⊙ is the **element-wise (Hadamard) product** — that is how a gate is applied. The candidate
content c̃ uses tanh rather than a sigmoid because it is content, not a gate.

Two remarks from the lecture:

- **The forget gate is misnamed.** It computes how much you *remember*, so "remember gate"
  would make more sense (≈29:32).
- **All four of f, i, o and c̃ have the same shape**, so in practice you stack the weight
  matrices into one big matrix and compute all four in a single multiply (≈34:16).

The output layer on top is unchanged from a vanilla RNN-LM: ŷ = softmax(Uh + b₂) over the
vocabulary (slide 24). See
[softmax, the logistic function, and cross-entropy](softmax-and-cross-entropy.md) and
[activation functions](activation-functions.md).

## Why the cell and the hidden state are separate

This is the part students push back on in lecture, and the answer is the clearest statement
of what the architecture is for (≈32:43–33:30).

The hidden state of a plain RNN does **double duty**. One job is to feed the output layer and
predict the next token. The other is to carry information about the past that might be useful
later. Those are different jobs, and only some of what you want to remember is relevant to
the word you are producing right now.

Manning's example: if the previous words were *sat in*, then for predicting the next word all
you need to know is that you are in a "sat in" context, where *the* or *a* will come next. But
if the sentence earlier said *the King of Prussia*, you want that fact kept somewhere, because
it may matter for future words — without it interfering with the current prediction.

So the **cell is the memory**, and the **output gate controls how much of it is exposed** for
generating the current word. When a student asks whether the output gate is redundant given
the forget and input gates, that is the answer: you want to keep information in c⁽ᵗ⁾ for the
future while masking it from the current output (≈35:02–35:49).

Manning also admits the part he finds hardest to justify: why there is a tanh on the cell
state in the h update. The argument he offers is that the cell can hold unbounded real
numbers while the hidden state wants a bounded range — "I guess they did it that way, it
seemed to work well" (≈35:49).

What the gates actually learn is unspecified and up to training (≈38:58). A student asks
about thresholds; Manning's answer is that it is not a threshold at all, because the gate is
a whole *vector* — it can keep dimensions 1 to 17 and throw away 18 to 22, or do so
probabilistically to different extents. The hope is that it learns cues: seeing the word
"next" might mean a change of topic, and a good time to forget and reset (≈39:45).

## The picture

Slides 23–24 use the well-known diagrams from **Chris Olah**'s "Understanding LSTMs" post
(colah.github.io) — Manning notes Olah now works at Anthropic (≈37:23). The cell state runs
straight across the top of the cell, crossed first by a ⊗ (forget) and then a ⊕ (write); the
gates are computed by yellow sigmoid layers below it from h⁽ᵗ⁻¹⁾ and x⁽ᵗ⁾.

Slide 24 annotates every part and adds the callout that matters: pointing at the ⊕,
**"The + sign is the secret!"**

## Why it fixes vanishing gradients

Slide 25 and ≈41:19. The essence:

> In a simple recurrent neural network the next hidden state is the result of
> **multiplicative** operations, and therefore it is very hard to preserve information.
> The essence of the LSTM is that you have a past memory and you **add** new information to it.

Manning notes this seems fundamentally right for how human memory works — memory is basically
additive.

The concrete consequence on slide 25: **set the forget gate to 1 for a cell dimension and the
input gate to 0, and the information in that dimension is preserved indefinitely.** A vanilla
RNN would have to learn a recurrent weight matrix W_h that happens to preserve information,
which is much harder.

**In practice you get about 100 timesteps rather than about 7.**

The additive path also helps with exploding gradients, for the same reason — you are not
performing a long chain of multiplies (≈43:39).

Slide 25 closes with a forward pointer: there are alternative ways of creating direct and
linear pass-through connections for long-distance dependencies. Those are attention and
residual connections in later lectures; the vision-side versions — ResNet, DenseNet,
HighwayNet — are on slides 26–27 and covered under
[vanishing and exploding gradients](vanishing-and-exploding-gradients.md).

## Where LSTMs stand

Slide 41. **2013–2015**: state of the art in handwriting recognition, speech recognition,
machine translation, parsing, image captioning and language modeling; the dominant approach
for most NLP tasks. **2019–2024**: Transformers dominate everything.

Manning's practical framing (≈40:33): when he taught this in 2016–17, LSTMs were the best
language-modeling architecture available and the class spent hours on them and their variants
— you can vary the gating, use fewer or more gates. In 2024 it is "probably not the most
important thing to know", but it is a thing to be aware of, and it is used in Assignment 3 —
"you can just ask PyTorch for an LSTM and it'll give you one that does all of this stuff"
(≈41:19).

Lecture 6's first named takeaway (slide 56): **LSTMs are powerful** — if you are doing
something with a recurrent neural network, you probably want to use an LSTM.

## Related pages

- [Recurrent neural networks](recurrent-neural-networks.md) — the architecture this refines,
  including the bidirectional and stacked variants that can be built from LSTM steps.
- [Vanishing and exploding gradients](vanishing-and-exploding-gradients.md) — the problem it
  solves, and the non-recurrent analogues of the fix.
- [Activation functions](activation-functions.md) — the sigmoid that makes a gate a gate, and
  the tanh on cell content.
- [Sequence-to-sequence and encoder-decoder models](seq2seq-and-encoder-decoder.md) — where
  two LSTMs get wired together.
- [Perplexity](perplexity.md) — the numbers LSTMs moved.
- [Lecture 6 — Sequence to Sequence Models](06-sequence-to-sequence-models.md)
