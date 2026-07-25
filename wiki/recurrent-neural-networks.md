# Recurrent neural networks

The third neural architecture family the course covers, after word2vec's simple
encoder-decoder and the feed-forward / fully-connected networks of lectures 2–4, and before
Transformers (lecture 5, ≈51:07). Introduced in
[lecture 5](05-recurrent-neural-networks.md) to solve a specific problem with fixed-window
[language models](language-modeling.md), and extended in
[lecture 6](06-sequence-to-sequence-models.md) with the [LSTM](lstm.md), bidirectionality
and stacking.

## The problem it solves

A fixed-window neural language model concatenates the embeddings of a fixed number of
preceding words, runs them through a hidden layer, and softmaxes over the vocabulary
(lecture 5, slides 27–30). Two things are wrong with it:

- **The window is fixed**, enlarging it enlarges *W*, and no window is ever big enough.
- **There is no symmetry across positions.** x⁽¹⁾ and x⁽²⁾ are multiplied by completely
  different sub-parts of *W*. What the model learns about *student* in position 1 is learned
  separately from what it learns about *student* in position 2, even though the evidence is
  the same. Manning's example: *the students opened their* and *the students slowly opened
  their* differ only in linguistic structure, but the parameters used are entirely different
  (≈49:36).

So we need an architecture that processes any-length input and uses the **same** parameters
to register "I saw the word *student*" regardless of where it occurred.

## The architecture

**Core idea** (lecture 5, slide 31): apply the same weights *W* repeatedly, across successive
positions in the text. A simple RNN language model (slide 32):

    e⁽ᵗ⁾ = E x⁽ᵗ⁾                                    word embeddings
    h⁽ᵗ⁾ = σ( W_h h⁽ᵗ⁻¹⁾ + W_e e⁽ᵗ⁾ + b₁ )           hidden state
    ŷ⁽ᵗ⁾ = softmax( U h⁽ᵗ⁾ + b₂ ) ∈ ℝ^|V|            output distribution

h⁽⁰⁾ is the initial hidden state and can simply be zeros. The hidden state accumulates a
memory of everything seen so far. Outputs at each step are **optional** — an RNN produces
them only if the task needs them.

The nonlinearity σ has most commonly been **tanh** for RNNs, chosen because it is balanced
across positive and negative (lecture 5, ≈54:14). See
[activation functions](activation-functions.md).

## Advantages and disadvantages

Slide 33 lists both, and both matter for the rest of the course.

**Advantages:**
- Processes **any length** of input.
- Computation at step *t* can, in theory, use information from many steps back.
- **Model size does not increase** with context length — the representation of arbitrarily
  long context stays a fixed-size vector h, so there is no exponential blow-up as with
  [*n*-grams](n-gram-language-models.md).
- Same weights at every timestep, so there is symmetry in how inputs are processed.

**Disadvantages:**
- **Recurrent computation is slow.** You must compute one hidden vector at a time. Manning
  notes the irony: this is literally the for loop he told you never to write, and it is one
  of the reasons RNNs fell out of favour (≈56:33).
- **Accessing information from many steps back is hard in practice.** Distant words fade and
  recent words dominate the hidden state. He allows that this is partly correct — recent
  context *is* usually most relevant, and humans forget too — but simple RNNs forget "rather
  too quickly" (≈57:21). Lecture 6 puts a number on it: effective conditioning of about
  **7 tokens** (slide 15). This is the
  [vanishing gradient problem](vanishing-and-exploding-gradients.md).

## Training an RNN language model

**The loss** (lecture 5, slide 34) at step *t* is the cross-entropy between the predicted
distribution ŷ⁽ᵗ⁾ and the true next word, which for a one-hot target reduces to a negative
log probability; the overall loss is the average across positions:

    J⁽ᵗ⁾(θ) = − log ŷ⁽ᵗ⁾_{x_{t+1}}          J(θ) = (1/T) Σₜ J⁽ᵗ⁾(θ)

See [softmax, the logistic function, and cross-entropy](softmax-and-cross-entropy.md).

**Teacher forcing** (slide 39). This is the procedural fact that makes training tractable. At
each step the model is asked for a distribution and scored against the actual next word — and
then fed **the actual next word**, not its own prediction. Manning is explicit that this is
not free generation: "we're only doing 'tell me the next word' from some human-generated
piece of text", and its imperfection is that you never explore what the model would have
produced on its own (≈1:01:59).

**Batching** (slide 40). Computing the loss over a whole corpus at once is impossible
memory-wise, so the data is cut into segments. Linguistically, sentences or documents would
be tidy; in practice, for GPU efficiency, people just cut every 100 words so a batch of
equal-length segments packs into a matrix (≈1:04:15). A student points out that this
reintroduces a context limit, and Manning agrees: **cutting into segments is making a Markov
assumption again**, which is one of the ways modern LLMs differ — they use thousands of words
of prior context (≈1:13:35).

## Backpropagation through time

W_h appears at every timestep, so the question is how to differentiate with respect to a
**repeated** weight matrix (lecture 5, slides 41–43). The answer:

> The gradient with respect to a repeated weight is the **sum** of the gradient with respect
> to each time it appears.

    ∂J⁽ᵗ⁾/∂W_h = Σ_{i=1..t} ∂J⁽ᵗ⁾/∂W_h |₍ᵢ₎

The justification is lecture 3's rule that **gradients sum at outward branches**, applied
here by treating W_h as being copied by identity into W_h⁽¹⁾, W_h⁽²⁾, … at each timestep;
identity copies have partial derivative 1 with respect to each other, so the multivariable
chain rule leaves a plain sum (≈1:06:36). The algorithm is called **backpropagation through
time** (Werbos 1988).

In practice it is often **truncated** after around 20 timesteps for training efficiency
(slide 43). Manning notes the asymmetry: the forward pass still uses the full context; only
the backward pass is cut short (≈1:08:09). See [backpropagation](backpropagation.md) and
[gradient descent](gradient-descent.md).

## Generating text

**Roll-out** (lecture 5, slide 44): sample from ŷ⁽ᵗ⁾, feed the sampled word back in as the
next input, repeat. Start from a special `<s>` pseudo-word — which has its own embedding —
rather than from a real word, and stop when `</s>` is generated (≈1:08:57). Manning notes
this is exactly what ChatGPT does with a more complicated model, and that because it is
probabilistic, running it twice gives different answers (≈1:10:31).

The demonstrations (slides 45–48) train the same simple architecture on Obama speeches,
*Harry Potter*, recipes, and paint colour names. The paint model illustrates two variations
at once: it is **character-level** (the time steps are letters, not words), and it is
conditional — the initial hidden state is set from the RGB values of a colour rather than
zeros, so the model names a given colour (≈1:16:41). RNNs are also used over non-language
sequences such as DNA (≈1:15:55).

## Other uses

Lecture 5's slide 64 makes the point explicitly: **an RNN is not a language model**. It is a
family of architectures; language modeling is one task among many. Slides 66–71 and lecture
6's slides 28–32:

- **Sequence tagging** — a label per position: part-of-speech tagging, named entity
  recognition. Lecture 6 connects this back to lecture 2's window classifier: rather than
  labelling the middle of a window, label at every position (≈49:01).
- **Sentence classification / sentence encoding** — sentiment, for instance. The simplest
  approach uses the **final hidden state**, since it has seen the whole sentence; in practice
  taking the **element-wise max or mean of all hidden states** is usually better (lecture 5,
  slides 68–69; lecture 6, ≈49:49).
- **As an encoder module** inside a larger system — the question in a question-answering
  system, for example (lecture 5, slide 70).
- **Conditional language models** — generation conditioned on some other input: audio for
  speech recognition, a source sentence for [machine translation](machine-translation.md),
  a document for summarization (lecture 5, slide 71).

## Bidirectional RNNs

Lecture 6, slides 33–37. The motivation is a limitation of contextual representations: in
*the movie was terribly exciting !*, the hidden state above *terribly* encodes only the left
context "the movie was terribly", and knows nothing about *exciting* — which is exactly what
flips *terribly* from negative to positive.

So run a second RNN backwards and concatenate the two states at each position:

    h→⁽ᵗ⁾ = RNN_FW( h→⁽ᵗ⁻¹⁾, x⁽ᵗ⁾ )
    h←⁽ᵗ⁾ = RNN_BW( h←⁽ᵗ⁺¹⁾, x⁽ᵗ⁾ )
    h⁽ᵗ⁾  = [ h→⁽ᵗ⁾ ; h←⁽ᵗ⁾ ]

The two directions generally have **separate weights**. RNN_FW is deliberately generic
notation — it can be a simple RNN step or an [LSTM](lstm.md) step. The concatenation is what
counts as "the hidden state" and gets passed to the rest of the network.

**The constraint** (slide 37) is important and easy to forget: bidirectional RNNs need the
**entire input sequence**, so they are **not applicable to language modeling**, where you
only have left context. Where you do have the whole sequence — any encoding task —
bidirectionality is powerful and should be the default. **BERT** (Bidirectional Encoder
Representations from Transformers) is built on exactly this idea.

## Multi-layer (stacked) RNNs

Lecture 6, slides 38–40. RNNs are already deep along the time dimension; you can also stack
them, with layer *i*'s hidden states as layer *i*+1's inputs. Lower layers compute
lower-level features, higher layers higher-level ones. Manning's answer to "does this
actually do anything, or are they just big vectors above the words?" is that the extra layer
buys the same successive-feature-extraction advantage depth buys any neural network (≈53:44).

The empirical picture (≈54:29): **two layers is reliably much better than one; three or four
is iffier.** Britz et al. (2017) found 2–4 layers best for an NMT encoder and 4 for the
decoder; getting to 8 needs skip or dense connections. RNNs stayed shallow compared to vision
networks. Transformers changed this completely — 12 or 24 layers, with many skip-like
connections.

Multi-layer LSTMs earned their keep most clearly in neural machine translation (lecture 6,
≈1:13:08 and slide 54).

## Where RNNs stand now

Lecture 6's slide 41 is the honest summary. **2013–2015**: LSTMs achieved state of the art in
handwriting recognition, speech recognition, machine translation, parsing, image captioning
and language modeling, and became the dominant approach for most NLP tasks. **2019–2024**:
Transformers dominate everything. The WMT machine translation competition makes it concrete —
WMT 2014 had zero neural systems, WMT 2016's report mentions "RNN" 44 times and those systems
won, and WMT 2019 mentions "RNN" 7 times against "Transformer" 105 times.

Lecture 5's slide 72 names what the course built: the **simple / vanilla / Elman RNN**.

## Related pages

- [LSTM](lstm.md) — the gated variant that makes long-range memory work.
- [Vanishing and exploding gradients](vanishing-and-exploding-gradients.md) — the reason
  vanilla RNNs need fixing.
- [Language modeling](language-modeling.md) — the task the RNN-LM performs.
- [Backpropagation](backpropagation.md) — the general algorithm BPTT specializes.
- [Sequence-to-sequence and encoder-decoder models](seq2seq-and-encoder-decoder.md) — two RNNs
  wired together.
- [Lecture 5 — Recurrent Neural Networks](05-recurrent-neural-networks.md)
- [Lecture 6 — Sequence to Sequence Models](06-sequence-to-sequence-models.md)
