# Lecture 6 — Sequence to Sequence Models

The deck calls this lecture *LSTM RNNs and Neural Machine Translation*, which describes its
two halves. The first finishes the argument [lecture 5](05-recurrent-neural-networks.md)
started: vanilla RNNs cannot preserve information over many timesteps, and the **LSTM**
fixes that by giving the network a separate memory that is *added to* rather than
overwritten. Manning is emphatic about where the fix lives — "the secret" is a single plus
sign in the cell-state update. The second half introduces **machine translation**, tells the
story of its statistical era, and arrives at the **sequence-to-sequence** encoder-decoder
model that made neural MT work in 2014.

Along the way the lecture covers evaluation by perplexity properly, the bidirectional and
multi-layer RNN variants, and the observation that vanishing gradients are not an RNN
problem at all but a deep-network problem, with ResNet, DenseNet and HighwayNet as the
vision-side answers.

**Slide-by-slide text of this deck: [56 slides](../raw/slides/06-sequence-to-sequence-models.md)**
— printed slide numbers match PDF pages 1:1. **Note:** slides 4–18 re-run lecture 5's slides
49–63 as a recap; the only new material there is slide 15's "~7 tokens back" rule of thumb.

Slides PDF: [Lecture 6 — fancy-rnn](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture06-fancy-rnn.pdf) ·
Notes: [2019 notes 05 — LM and RNN](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/readings/cs224n-2019-notes05-LM_RNN.pdf) ·
[Full transcript](../raw/transcripts/06-sequence-to-sequence-models.md)

## Perplexity, properly

Slide 4 is the same formula as lecture 5's slide 49, but here Manning explains where it came
from and why it survives.

The evaluation logic first (≈2:26): a language model scores a piece of text and says how
likely it is; our standard for text in a language is text produced by human beings; so show
the model **fresh** text it was not trained on and ask how well it predicts each successive
word. Perplexity takes the model's probability at each position, inverts it (0.002 becomes
500), takes the product across positions, and takes the geometric average — which is exactly
the exponential of the per-word cross-entropy loss.

The historical story (≈4:48). In the era of symbolic AI — John McCarthy, Ed Feigenbaum —
people at IBM including **Fred Jelinek** started applying probabilistic methods to speech
recognition. Jelinek's account was that none of the AI people he was talking to in the late
70s and early 80s understood information theory, so he needed something simpler than
cross-entropy rate. Exponentiating gives a number you can interpret concretely: **a
perplexity of 64 is like rolling a 64-sided die** each time you guess the next word (≈5:36).
Manning's own view is that from a modern perspective it "kind of makes no sense" to use
perplexity rather than cross-entropy — but it stuck.

One practical warning: perplexity depends on the **base** of your logarithm. Base 2 was
traditional; natural logs are now common; numbers are not comparable across the two
(≈4:02).

Slide 5's table read as history (≈7:07): the cleverest *n*-gram smoothing known,
**interpolated Kneser-Ney**, gave about 67. Early RNNs could not beat *n*-grams alone —
they only won when combined with something else, like a maximum-entropy model, giving around
51. Real progress came with **LSTMs**, at 43.7 and 30. Halving perplexity from 60 to 30
corresponds to reducing cross-entropy by about one bit. Manning adds two pieces of context:
when he started in NLP, perplexities were three-figure numbers; and by modern standards even
30 is very high, since the best current models reach **single digits**. See
[perplexity](perplexity.md).

## The vanishing gradient, and the seven-token limit

Slides 6–17 recap lecture 5's derivation. One addition matters. Slide 15 ends with a bullet
lecture 5's version did not have:

> In practice, a simple RNN will only condition ~7 tokens back [vague rule-of-thumb]

Manning frames this as a back-of-the-envelope number, but it is the one that makes the case
against vanilla RNNs (≈16:26). Compare it with *n*-grams, where the practical maximum was
5-grams because of sparsity and storage: "although in theory we've now got a much better
solution, in practice, because of vanishing gradients, we're only kind of getting the
equivalent of 8-grams. So we haven't made that much progress, it feels like" (≈17:13).

He also makes the diagnosis architectural (≈20:20). In a vanilla RNN the hidden state is
**completely rewritten** at every step: you take the previous hidden vector, multiply it by
a matrix that changes it entirely, and add in the input. If you want to say "there is useful
stuff in h⁽ᵗ⁻¹⁾, please keep it around for a while", learning weights that mostly preserve
what was there is not an obvious thing for gradient descent to find.

## LSTMs

**The history** (slide 20, ≈24:55). Proposed by **Hochreiter and Schmidhuber in 1997**, but
Manning points out that the paper everyone cites is not the whole story: a second paper,
**Gers and Schmidhuber (2000)**, introduced the forget gate, which is a crucial part of the
LSTM as actually used. Schmidhuber's group did foundational work through the late 90s when
almost everyone else had given up on neural networks — and, Manning notes, at a time when
that was not the route to a well-compensated industry job it is today. Recognition came
through Schmidhuber's later student **Alex Graves** (who also invented CTC for speech
recognition), who went to Toronto as a postdoc with **Geoff Hinton**; Hinton went to Google
in 2013, and LSTMs became the dominant framework in the 2014–16 period (≈27:14). Sepp
Hochreiter spent much of the intervening period publishing in bioinformatics.

**Parsing the name** (≈23:23). The point is *long* **short-term memory**. Humans distinguish
short-term memory from permanently stored memory, and human short-term memory holds things
for quite a while — you can still recall what someone said a few conversational turns ago.
Simple RNNs had a short-term memory of about seven tokens. The goal was to make short-term
memory long.

**The architecture** (slide 22). There are now two vectors per step: a **hidden state**
h⁽ᵗ⁾ and a **cell state** c⁽ᵗ⁾, both length *n*. The cell stores long-term information and
behaves conceptually like RAM — it can be read, erased and written. Three **gates**, also
length *n*, control which. Gates are computed, not fixed: each is a sigmoid of the same
shape of expression as a vanilla RNN step, so its elements lie between 0 and 1 and can be
open, closed or in between, and their values depend on the current context.

    f⁽ᵗ⁾ = σ( W_f h⁽ᵗ⁻¹⁾ + U_f x⁽ᵗ⁾ + b_f )      forget gate
    i⁽ᵗ⁾ = σ( W_i h⁽ᵗ⁻¹⁾ + U_i x⁽ᵗ⁾ + b_i )      input gate
    o⁽ᵗ⁾ = σ( W_o h⁽ᵗ⁻¹⁾ + U_o x⁽ᵗ⁾ + b_o )      output gate

    c̃⁽ᵗ⁾ = tanh( W_c h⁽ᵗ⁻¹⁾ + U_c x⁽ᵗ⁾ + b_c )   new cell content
    c⁽ᵗ⁾ = f⁽ᵗ⁾ ⊙ c⁽ᵗ⁻¹⁾ + i⁽ᵗ⁾ ⊙ c̃⁽ᵗ⁾           erase, then write
    h⁽ᵗ⁾ = o⁽ᵗ⁾ ⊙ tanh c⁽ᵗ⁾                      read

⊙ is the element-wise (Hadamard) product. Manning's aside on naming: the **forget gate is
wrongly named** — it computes how much you *remember*, so "remember gate" would make more
sense (≈29:32).

**Why the hidden state and the cell state are separate** is worth understanding, because it
is the part students push back on. The hidden state does double duty: it feeds the output
layer to predict the next token, *and* it carries information forward that may be useful
later. Those are different jobs. Manning's example (≈33:30): if the previous words were
*sat in*, all you need for the next word is that you are in a "sat in" context where *the*
or *a* comes next — but if the sentence earlier said *the King of Prussia*, you want to keep
that around for future predictions without it interfering now. So the cell is the memory,
and the output gate controls how much of it is exposed for generating the current word. When
a student asks whether the output gate is redundant given the forget and input gates, that
is the answer: you want to keep information in c⁽ᵗ⁾ for the future while masking it from the
current output (≈35:02–35:49).

An implementation note (≈34:16): all four of f, i, o and c̃ have exactly the same shape, so
you can stack the four weight matrices into one big matrix and compute them in a single
multiply.

**Why it fixes vanishing gradients** (slide 25, ≈41:19). The essence: in a simple RNN the
next hidden state comes out of *multiplicative* operations, which makes preserving
information hard. In an LSTM you have a past memory and you **add** new information to it —
which, Manning notes, seems fundamentally right for how human memory works. **The plus sign
is the secret** (slide 24's callout). Set the forget gate to 1 and the input gate to 0 for
some cell dimension, and that dimension is preserved indefinitely. The practical result: you
get about **100 timesteps** rather than about 7.

A nice historical wrinkle: the first LSTM was *purely* additive with no forget gate, and
that turned out to be imperfect — if you keep adding over a long sequence it becomes
dysfunctional after a point — which is why the Gers paper's forget gate was the big
improvement (≈42:06). Asked whether the additive path also helps with *exploding* gradients,
Manning says yes, because you are not doing a long sequence of multiplies (≈43:39).

See [LSTM](lstm.md).

## Vanishing gradients are not an RNN problem

Slides 26–27 generalize. Any very deep network suffers this: little gradient signal reaches
the lower layers, so their parameters barely update, so they learn nothing, so the network
does not work — which is part of why deep networks were stuck in the early 2000s (≈45:10).

The fix is the same idea in a vertical direction: **add more direct connections**.

- **Residual connections / ResNet** (He et al. 2015) — carry the input around a block with
  an identity function and add it back. The identity connection preserves information by
  default, and this is what made deep computer vision models learnable (≈46:44).
- **Dense connections / DenseNet** (Huang et al. 2017) — connect each layer directly to all
  later layers.
- **Highway connections / HighwayNet** (Srivastava et al. 2015) — as ResNet, but the balance
  between the identity path and the transformed path is controlled by a **dynamic gate**,
  explicitly inspired by LSTMs.

Slide 27's conclusion: vanishing and exploding gradients are a general problem, but **RNNs
are particularly unstable** because of the repeated multiplication by the *same* weight
matrix (Bengio et al. 1994). See
[vanishing and exploding gradients](vanishing-and-exploding-gradients.md).

## Bidirectional and multi-layer RNNs

**Bidirectional RNNs** (slides 33–37). The motivation is a contextual representation
problem: in *the movie was terribly exciting !*, the hidden state above *terribly* encodes
"the movie was terribly" and knows nothing about what follows — but *exciting* is exactly
what flips *terribly* from negative to positive. So run a second RNN backwards and
concatenate:

    h→⁽ᵗ⁾ = RNN_FW( h→⁽ᵗ⁻¹⁾, x⁽ᵗ⁾ )
    h←⁽ᵗ⁾ = RNN_BW( h←⁽ᵗ⁺¹⁾, x⁽ᵗ⁾ )
    h⁽ᵗ⁾  = [ h→⁽ᵗ⁾ ; h←⁽ᵗ⁾ ]

The two directions generally have separate weights, and RNN_FW is generic notation — it
could be a simple RNN or an LSTM step. The concatenation is what gets passed onward.

The constraint on slide 37 is the important one: bidirectional RNNs need the **entire input
sequence**, so they are **not applicable to language modeling**, where you only have left
context. Where you do have the whole sequence — any encoding task — bidirectionality is
powerful and should be the default. **BERT** (Bidirectional Encoder Representations from
Transformers) is built on exactly this idea.

**Multi-layer / stacked RNNs** (slides 38–40). RNNs are already deep along time; you can
also stack them, feeding layer *i*'s hidden states as layer *i*+1's inputs, so lower layers
compute lower-level features and higher layers higher-level ones. Manning's answer to "does
this actually do anything" is that the extra layer buys exactly the same feature-extraction
advantage that depth buys any neural network (≈53:44).

The empirical picture (≈54:29, slide 40): two layers is reliably much better than one; three
or four is iffier. Britz et al. (2017) found 2–4 layers best for an NMT encoder and 4 for the
decoder, and getting to 8 needs skip or dense connections. Transformers changed this
completely — they are routinely 12 or 24 layers, with plenty of skip-like connections.

Slide 41 places LSTMs in time. 2013–2015: state of the art in handwriting recognition, speech
recognition, machine translation, parsing, image captioning; the dominant approach for most
NLP tasks. Now (2019–2024): Transformers. The WMT counts make it vivid — WMT 2014 had **zero**
neural MT systems; WMT 2016's summary report says "RNN" **44** times and those systems won;
WMT 2019 says "RNN" **7** times and "Transformer" **105** times.

## Machine translation, before neural networks

**MT is where NLP started** (≈55:15). In the early 1950s there was no field of AI and no
field of NLP, but there was machine translation. The reason is wartime: computers were
developed during the Second World War for two purposes — artillery tables and **code
breaking** — and as the Cold War began, someone had the idea that translation between
languages might be like code breaking. That idea reached funding agencies, and a great deal
of money went into it (≈56:48).

It was a flop, for two reasons Manning names (≈57:35). Nobody knew anything about the
structure of human languages — the Chomsky hierarchy had not been invented, formal
properties of language were unexplored. And the machines were hopeless: "the little power
brick for your laptop has more computing power inside it than the big mainframe computers
they used to be using". So the systems were simple lexicons and rule-based substitution.

**Statistical machine translation** (1990s–2010s, slides 45–47). The core move is to learn a
probabilistic model from data, and to break argmax_y P(y|x) with **Bayes rule** into

    argmax_y P(x|y) · P(y)

- P(x|y) is the **translation model** — how words and phrases get translated (*fidelity*) —
  learned from parallel data.
- P(y) is the **language model** — how to write good English (*fluency*) — learned from
  monolingual data.

Manning explains why this factorization made progress possible even though it looks like it
just swapped *x* and *y* (≈1:00:41): the translation model can then be very simple — see
*homme*, emit "man" or "person" with some probabilities — and needs to know nothing about
word order or grammar in the target language, because **all the cleverness moves into the
language model**. This is a direct payoff of the language-modeling machinery from lecture 5.

Parallel data came from the European Union (many European language pairs), Hong Kong
(English–Chinese), and the UN (≈59:07). And the systems were enormous: hundreds of important
details, many separately-designed subcomponents, heavy feature engineering, maintained
resource tables, and the whole effort repeated for each language pair (slide 47).

**Why it stayed hard** (slide 46). Word order does not correspond across languages — the
German/English alignment on the slide has crossing arrows — and the running example is
Manning's favourite failing sentence, from the Chinese translation of Jared Diamond's *Guns,
Germs, and Steel*. The correct reading is *In 1519, six hundred Spaniards landed in Mexico
to conquer the Aztec Empire with a population of a few million. They lost two thirds of
their soldiers in the initial clash.* Google Translate in 2009 gave "1519 600 Spaniards
landed in Mexico, millions of people to conquer the Aztec empire, the first two-thirds of
soldiers against their loss." The word choices are poor, but the real failure is structural:
Chinese marks with the particle *de* that "a few million people" modifies "the Aztec Empire",
and the system loses that, so the millions of people become the ones doing the conquering
(≈1:03:48). Manning re-tested it over the years: 2013 looked like progress, 2015 had gone
back downhill, so 2013 was luck rather than improvement (≈1:04:35).

See [machine translation](machine-translation.md).

## Neural machine translation and seq2seq

**2014** (slide 48 is a meteor hitting MT research). Neural MT does the whole task with a
**single end-to-end neural network** — a general and powerful idea, because putting one loss
function at the end and backpropagating through everything aligns all the learning with the
task you actually want (≈1:05:23).

The architecture is **sequence-to-sequence** (slide 50) and uses two RNNs, in practice LSTMs:

- The **encoder RNN** reads the source sentence and outputs nothing; it just builds a hidden
  state that represents the source. Its final hidden state is the **encoding**.
- The **decoder RNN** is a language model that generates the target sentence, initialized
  with that encoding as its previous hidden state, and started with `<START>`.

The two have **separate parameters** — one LSTM that knows the source language, one that
knows the target (≈1:06:56). The slide's worked example translates *il m' a entarté* to
*he hit me with a pie*.

At test time the decoder feeds its own argmax output back in as the next input; at training
time you have parallel text, so you feed in the **actual** target words, score each
prediction, average the losses, and backpropagate through the decoder *and* the encoder,
updating everything (slide 53). That is what "trained end-to-end" means here (≈1:08:29).

Formally (slide 52) seq2seq is a **conditional language model**: a language model because the
decoder predicts the next target word, conditional because it is also conditioned on the
source. It calculates P(y|x) directly:

    P(y|x) = P(y₁|x) · P(y₂|y₁,x) · P(y₃|y₁,y₂,x) ⋯ P(y_T|y₁,…,y_{T−1},x)

Slide 51's point is that the **encoder-decoder** idea generalizes well beyond MT:
summarization (long text → short text), dialogue (previous utterances → next utterance),
parsing (text → parse as a sequence), code generation (natural language → Python). And the
pattern survives the move from LSTMs to Transformers (≈1:09:15).

Two student questions sharpen the design. *Why not just build a deeper network on top of the
source?* — because word order changes a lot between languages, and because the length does
not even stay the same: English has auxiliaries and articles that Chinese does not, so
depending on direction you must add or delete words, which is very hard when you are
building on top of the source positions (≈1:10:03). *Is the encoder bidirectional?* — it
could be, and that might be better, but the famous original Google instantiation was not; it
simply took the final hidden state (≈1:11:37).

Slide 54 shows the multi-layer version (Sutskever et al. 2014; Luong et al. 2015): a stacked
encoder-decoder, with **Conditioning = Bottleneck** written by hand under the single column
of state that joins the two halves. That bottleneck is the problem attention solves in a
later lecture. In practice, this was one of the places where stacked LSTMs clearly earned
their keep (≈1:13:08).

See [sequence-to-sequence and encoder-decoder models](seq2seq-and-encoder-decoder.md).

## Why NMT was the first big success story

Slide 55: NMT went from a fringe research attempt in **2014** to the leading standard method
in **2016**, when Google Translate switched from SMT to NMT; by 2018 everyone had —
Microsoft, SYSTRAN, Facebook, Baidu, NetEase, Tencent, Sogou.

Manning places it third in the sequence of deep learning's wins: speech recognition first,
object recognition in vision second, machine translation third (≈1:13:54). And he underlines
what made it striking: the SMT systems being displaced had been worked on for about a decade
by hundreds of people, with millions of lines of code and per-language-pair hacks — and a
comparatively simple, small neural system beat them (≈1:15:26). The quality difference was
obvious enough that people noticed before Google announced it, which is what the *New York
Times* piece linked on the slide is about.

## The four takeaways

Slide 56 ends with them, and they are a fair summary of the lecture:

1. **LSTMs are powerful** — if you are using a recurrent network, use an LSTM.
2. **Clip your gradients.**
3. **Use bidirectionality when possible** — but not for generation.
4. **Encoder-decoder NMT systems work very well.**

## Related pages

- [LSTM](lstm.md) — the gates, the equations, the history, and why the plus sign matters.
- [Machine translation](machine-translation.md) — the 1950s, statistical MT, and the shift to
  neural MT.
- [Sequence-to-sequence and encoder-decoder models](seq2seq-and-encoder-decoder.md) — the
  architecture, conditional language models, training end-to-end, and the bottleneck.
- [Vanishing and exploding gradients](vanishing-and-exploding-gradients.md) — the problem
  this lecture's first half exists to solve.
- [Recurrent neural networks](recurrent-neural-networks.md) — the bidirectional and stacked
  variants introduced here.
- [Perplexity](perplexity.md) — Jelinek's dice, and the table of numbers.
- [Lecture 5 — Recurrent Neural Networks](05-recurrent-neural-networks.md) — the lecture this
  one continues.
