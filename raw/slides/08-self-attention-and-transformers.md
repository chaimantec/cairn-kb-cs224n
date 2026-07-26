---
title: Lecture 8 — Self-Attention and Transformers (slide deck)
lecture: 8
slides: 62 printed / 62 pages in the PDF
source_pdf: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture08-transformers.pdf
note: Printed slide numbers match PDF page numbers 1:1, no gaps or offset. Lecturer is Anna Goldie; slides adapted from Anna Goldie and John Hewitt.
---

# Lecture 8 — Self-Attention and Transformers: slide-by-slide

Text and figures of
[`cs224n-spr2024-lecture08-transformers.pdf`](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture08-transformers.pdf),
transcribed from the deck. Diagrams and plots are described in prose since the KB is
read as text.

Companion pages: [wiki page for this lecture](../../wiki/08-self-attention-and-transformers.md) ·
[transcript](../transcripts/08-self-attention-and-transformers.md)

## Contents

| Slides | Section |
| ------ | ------- |
| 1–3 | Title and lecture plan |
| 4–14 | §1 Impact of Transformers on NLP: "Attention Is All You Need", MT/SuperGLUE/LLM-leaderboard results, protein folding and vision, scaling laws |
| 15–24 | §2 From recurrence to attention-based models: RNN recap, the 3 desiderata, complexity/interaction-distance/parallelizability, self-attention as query-key-value lookup |
| 25–58 | §3 Understanding the Transformer: self-attention recipe, position representations, nonlinearities, masking, multi-head attention, scaled dot-product attention, residual connections, layer norm, encoder/decoder/encoder-decoder, results |
| 57–61 | §4 Drawbacks and variants: quadratic compute, position representations, Linformer, BigBird, "do modifications transfer?" |
| 62 | Closing: parting remarks |

---

## Slide 1 — Title

**Natural Language Processing with Deep Learning — CS224N/Ling284**

Anna Goldie. Lecture 8: Transformers. *Adapted from slides by Anna Goldie, John
Hewitt.*

## Slide 2 — Lecture Plan

1. Impact of Transformers on NLP (and ML more broadly)
2. From Recurrence (RNNs) to Attention-Based NLP Models
3. Understanding the Transformer Model
4. Drawbacks and Variants of Transformers

## Slide 3 — Outline (§1 highlighted)

Same four-item outline as slide 2, with "Impact of Transformers on NLP" highlighted as
the current section.

## Slide 4 — Transformers: Is Attention All We Need?

Last lecture, we learned that attention dramatically improves the performance of
recurrent neural networks. Today, we will take this one step further and ask **Is
Attention All We Need?** Reproduction of the "Attention Is All You Need" paper's title
block and author list: Ashish Vaswani*, Noam Shazeer*, Niki Parmar*, Jakob Uszkoreit*
(Google Brain/Research), Llion Jones* (Google Research), Aidan N. Gomez*† (University
of Toronto), Łukasz Kaiser* (Google Brain), Illia Polosukhin*‡.

## Slide 5 — Transformers: Is Attention All We Need? (spoiler)

Same as slide 4, with an added line: **Spoiler: Not Quite!**

## Slide 6 — Transformers Have Revolutionized the Field of NLP

By the end of this lecture, you will deeply understand the neural architecture that
underpins virtually every state-of-the-art NLP model today! Full architecture diagram
(the canonical Transformer figure from Vaswani et al., 2017): stacked Encoder (repeat
6×: Multi-Head Attention → Add & Norm → Feed Forward → Add & Norm) and Decoder (repeat
6×: Masked Multi-Head Attention → Add & Norm → Multi-Head Attention → Add & Norm → Feed
Forward → Add & Norm), with Input/Output Embeddings plus Positional Encoding at the
bottom and Linear → Softmax → Output Probabilities at the top. A movie-still image
("Courtesy of Paramount Pictures", showing robots from the *Transformers* film
franchise) plays on the name.

## Slide 7 — Great Results with Transformers: Machine Translation

First, Machine Translation results from the original Transformers paper! Table of
BLEU (EN-DE / EN-FR) and training cost in FLOPs for ByteNet, Deep-Att + PosUnk, GNMT +
RL, ConvS2S, MoE, and their ensembles, versus the Transformer (base model): 27.3 EN-DE
/ 38.1 EN-FR at 3.3×10¹⁸ training FLOPs, and Transformer (big): **28.4 EN-DE / 41.8
EN-FR** at 2.3×10¹⁹ FLOPs — best BLEU on EN-DE and near-best on EN-FR, at a fraction of
the training cost of the ensembled prior systems (e.g. ConvS2S Ensemble needed
7.7×10¹⁹–1.2×10²¹ FLOPs). Test sets: WMT 2014 English-German and English-French.
[Vaswani et al., 2017]

## Slide 8 — Great Results with Transformers: SuperGLUE

SuperGLUE is a suite of challenging NLP tasks, including question-answering, word
sense disambiguation, coreference resolution, and natural language inference.
Leaderboard table (SuperGLUE Leaderboard v2.0) [Wang et al., 2019] shows top-10 systems
by overall score, all in the high-80s to low-90s: JDExplore d-team (Vega v2, 91.3),
Liam Fedus (ST-MoE-32B, 91.2), Microsoft Alexander v-team (Turing NLR v5, 90.9), ERNIE
Team–Baidu (ERNIE 3.0, 90.6), Yi Tay (PaLM 540B, 90.4), Ziru Wang (T5 + UDG, 90.4),
DeBERTa Team–Microsoft (DeBERTa/TuringNLRv4, 90.3), SuperGLUE Human Baselines (89.8),
T5 Team–Google (T5, 89.3), SPoT Team–Google (Frozen T5 1.1 + SPoT, 89.2), with
per-task scores on BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC, AX-b, AX-g.

## Slide 9 — Great Results with Transformers: Rise of Large Language Models!

Today, Transformer-based models dominate the LMSYS Chatbot Arena Leaderboard! Table
of top-ranked models by Arena Elo (with 95% CI and vote counts) as of the deck's
snapshot: GPT-4-Turbo-2024-04-09 (1258, OpenAI), GPT-4-1106-preview (1253, OpenAI),
Claude 3 Opus (1251, Anthropic), Gemini 1.5 Pro API-0409-Preview (1249, Google),
GPT-4-0125-preview (1248, OpenAI), Meta Llama 3 70b Instruct (1213, Meta, Llama 3
Community license), Bard (Gemini Pro) (1208, Google), Claude 3 Sonnet (1201,
Anthropic). Logos shown for Gemini/Bard (Google), ChatGPT/GPT-4 (OpenAI), Claude 3
(Anthropic), Llama 3 (Meta). [Chiang et al., 2024]

## Slide 10 — Transformers Even Show Promise Outside of NLP

Section-header slide (text repeated and expanded on slide 11).

## Slide 11 — Transformers Even Show Promise Outside of NLP: Protein Folding

*Nature* magazine cover, "PROTEIN POWER: AI network predicts highly accurate 3D
structures for the human proteome" — [Jumper et al. 2021], aka **AlphaFold2**.

## Slide 12 — Transformers Even Show Promise Outside of NLP: Image Classification

Same protein-folding cover, plus: a corgi photo and its attention-map visualization
(Original vs. Attention Map). **Image Classification**: [Dosovitskiy et al. 2020]
Vision Transformer (ViT) outperforms ResNet-based baselines with substantially less
compute. Table comparing Ours-JFT (ViT-H/14, ViT-L/16), Ours-I21k (ViT-L/16), BiT-L
(ResNet152x4), and Noisy Student (EfficientNet-L2) on ImageNet (88.55 for ViT-H/14),
ImageNet ReaL, CIFAR-10 (99.50), CIFAR-100, Oxford-IIIT Pets, Oxford Flowers-102,
VTAB (19 tasks), and TPUv3-core-days of training compute (2.5k for ViT-H/14 vs. 9.9k
for BiT-L, 12.3k for Noisy Student).

## Slide 13 — Transformers Even Show Promise Outside of NLP: ML for Systems

Same protein-folding and ViT content, plus **ML for Systems**: [Zhou et al. 2020] A
Transformer-based compiler model (GO-one) speeds up a Transformer model! Table
comparing GO-one against HP, METIS, and HDP hardware-placement heuristics across
2/4/8-layer RNNLM, GNMT, and Transformer-XL models, and Inception/AmoebaNet/WaveNet —
GO-one achieves the lowest runtime with speedups up to 27.8× over HP/HDP and 20.5%/
18.2% geomean speedup, and its search is 15× faster.

## Slide 14 — Scaling Laws: Are Transformers All We Need?

With Transformers, language modeling performance improves smoothly as we increase
model size, training data, and compute resources in tandem. This power-law
relationship has been observed over multiple orders of magnitude with no sign of
slowing! If we keep scaling up these models (with no change to the architecture),
could they eventually match or exceed human-level performance? Three log-log plots
(Kaplan et al., 2020): Test Loss vs. Compute (PF-days, non-embedding), fit
L = (C_min/2.3·10⁸)^−0.050; vs. Dataset Size (tokens), fit L = (D/5.4·10¹³)^−0.095; vs.
Parameters (non-embedding), fit L = (N/8.8·10¹³)^−0.076 — all three show smooth,
near-linear (in log-log space) decreases in loss across many orders of magnitude.

## Slide 15 — Outline (§2 highlighted)

Same four-item outline, with "From Recurrence (RNNs) to Attention-Based NLP Models"
highlighted.

## Slide 16 — As of last lecture: recurrent models for (most) NLP!

Circa 2016, the de facto strategy in NLP is to **encode** sentences with a
bidirectional LSTM (for example, the source sentence in a translation). Define your
output (parse, sentence, summary) as a sequence, and use an LSTM to generate it. Use
attention to allow flexible access to memory. Three small diagrams: a bidirectional
LSTM encoder; a bidirectional LSTM decoder generating a sequence; and an
encoder-plus-decoder-with-attention diagram, with arrows fanning from the encoder
states into the first decoder step.

## Slide 17 — Why Move Beyond Recurrence? Motivation for Transformer Architecture

The Transformers authors had 3 desiderata when designing this architecture:

1. Minimize (or at least not increase) computational complexity per layer.
2. Minimize path length between any pair of words to facilitate learning of
   long-range dependencies.
3. Maximize the amount of computation that can be parallelized.

[Vaswani et al., 2017]

## Slide 18 — 1. Transformer Motivation: Computational Complexity Per Layer

When sequence length (*n*) ≪ representation dimension (*d*), complexity per layer is
lower for a Transformer compared to the recurrent models we've learned about so far.
Table 1 of the Transformer paper (maximum path lengths, per-layer complexity, and
minimum number of sequential operations; *n* = sequence length, *d* = representation
dimension, *k* = kernel size of convolutions, *r* = neighborhood size in restricted
self-attention):

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
| --- | --- | --- | --- |
| Self-Attention | O(n²·d) | O(1) | O(1) |
| Recurrent | O(n·d²) | O(n) | O(n) |
| Convolutional | O(k·n·d²) | O(1) | O(log_k(n)) |
| Self-Attention (restricted) | O(r·n·d) | O(1) | O(n/r) |

## Slide 19 — 2. Transformer Motivation: Minimize Linear Interaction Distance (1)

RNNs are unrolled "left-to-right". It encodes linear locality: a useful heuristic!
Nearby words often affect each other's meanings ("tasty pizza" diagram, two boxes with
a bidirectional arrow). **Problem**: RNNs take O(sequence length) steps for distant
word pairs to interact. Diagram: "The **chef** who … **ate**" with an O(sequence
length) bracket spanning the intervening (grayed-out) RNN steps.

## Slide 20 — 2. Transformer Motivation: Minimize Linear Interaction Distance (2)

O(sequence length) steps for distant word pairs to interact means: hard to learn
long-distance dependencies (because gradient problems!); linear order of words is
"baked in" — we already know sequential structure doesn't tell the whole story… Same
"The **chef** who … **ate**" diagram, with a callout: "Info of *chef* has gone through
O(sequence length) many layers!"

## Slide 21 — 3. Transformer Motivation: Maximize Parallelizability

Forward and backward passes have O(seq length) unparallelizable operations. GPUs (and
TPUs) can perform many independent computations at once! But future RNN hidden states
can't be computed in full before past RNN hidden states have been computed — inhibits
training on very large datasets! Particularly problematic as sequence length
increases, as we can no longer batch many examples together due to memory
limitations. Diagram numbering the minimum steps before each hidden state h₁ … h_T can
be computed (0, 1, 2, 3, …, growing with position).

## Slide 22 — High-Level Architecture: Transformer is all about (Self) Attention

To recap, **attention** treats each word's representation as a **query** to access and
incorporate information from **a set of values**. Last lecture, we saw attention from
the **decoder** to the **encoder** in a recurrent sequence-to-sequence model.
**Self-attention** is **encoder**-**encoder** (or **decoder**-**decoder**) attention
where each word attends to each other word **within the input (or output)**. Diagram:
three layers (embedding → attention → attention), with a callout: "All words attend to
all words in previous layer; most arrows here are omitted."

## Slide 23 — Computational Dependencies for Recurrence vs. Attention (1)

Side-by-side dependency diagrams: an RNN-based encoder-decoder model with attention
(sequential dependencies within each row, plus attention arrows from encoder to the
first decoder step) versus a Transformer-based encoder-decoder model (dense all-pairs
connections within each layer, plus attention arrows from encoder to decoder).

## Slide 24 — Computational Dependencies for Recurrence vs. Attention (2)

Same diagram as slide 23, with an added callout: **Transformer Advantages**: number of
unparallelizable operations does not increase with sequence length; each "word"
interacts with each other, so maximum interaction distance is O(1).

## Slide 25 — Outline (§3 highlighted)

Same four-item outline, with "Understanding the Transformer Model" highlighted.

## Slide 26 — The Transformer Encoder-Decoder [Vaswani et al., 2017]

In this section, you will learn exactly how the Transformer architecture works: first
the Encoder, then the Decoder (which is quite similar). Full architecture diagram
(same as slide 6).

## Slide 27 — Encoder: Self-Attention

Self-Attention is the core building block of Transformer, so let's first focus on
that! Simplified diagram: Encoder box containing only a "Self-Attention" block, fed by
Input Embedding; Decoder box (contents not yet shown) fed by Output Embedding.

## Slide 28 — Intuition for Attention Mechanism

Let's think of attention as a "fuzzy" or approximate hashtable: to look up a **value**,
we compare a **query** against **keys** in a table. In a hashtable (shown on the
bottom left): each **query** (hash) maps to exactly one **key**-**value** pair. In
(self-)attention (shown on the bottom right): each **query** matches each **key** to
varying degrees; we return a sum of **values** weighted by the **query**-**key**
match. Two small diagrams: a hashtable with one query pointing to one key/value pair
(k₂→v₂), versus attention with one query fanning out to multiple keys with varying
weights (shown as varying color intensity on v₀…v₇).

## Slide 29 — Recipe for Self-Attention in the Transformer Encoder

- Step 1: For each word *x_i*, calculate its **query**, **key**, and **value**:
  q_i = **W**^Q x_i, k_i = **W**^K x_i, v_i = **W**^V x_i
- Step 2: Calculate attention score between **query** and **keys**: e_ij = q_i · k_j
- Step 3: Take the softmax to normalize attention scores: α_ij = softmax(e_ij) =
  exp(e_ij) / Σ_k exp(e_ik)
- Step 4: Take a weighted sum of **values**: Output_i = Σ_j α_ij v_j

## Slide 30 — Recipe for (Vectorized) Self-Attention in the Transformer Encoder

- Step 1: With embeddings stacked in **X**, calculate **queries**, **keys**, and
  **values**: **Q** = **X****W**^Q, **K** = **X****W**^K, **V** = **X****W**^V
- Step 2: Calculate attention scores between **query** and **keys**: **E** = **QK**ᵀ
- Step 3: Take the softmax to normalize attention scores: **A** = softmax(**E**)
- Step 4: Take a weighted sum of **values**: Output = **AV**

Boxed summary: **Output = softmax(QKᵀ)V**

## Slide 31 — What We Have So Far: (Encoder) Self-Attention!

Same simplified encoder/decoder diagram as slide 27 (Self-Attention block only).

## Slide 32 — But attention isn't quite all you need!

**Problem**: Since there are no element-wise non-linearities, self-attention is simply
performing a re-averaging of the value vectors. **Easy fix**: Apply a feedforward
layer to the output of attention, providing non-linear activation (and additional
expressive power). Equation for Feed Forward Layer: m_i = MLP(output_i) =
**W**₂·ReLU(**W**₁ × output_i + b₁) + b₂. Diagram: self-attention layer stacked twice,
each followed by an independent "FF" box per word (w₁ = "The", w₂ = "chef", w₃ =
"who", …, w_T = "food"), feeding into the encoder/decoder architecture diagram (Feed
Forward on top of Self-Attention).

## Slide 33 — But how do we make this work for deep networks?

Cartoon (xkcd-style): a figure labeled "NEURAL NETWORKS" holding a "STACK MORE LAYERS"
sign in front of an ever-rising "LAYERS vs LAYERS" graph. **Training Trick #1**:
Residual Connections. **Training Trick #2**: LayerNorm. **Training Trick #3**: Scaled
Dot Product Attention. Encoder/decoder diagram now shows Feed Forward stacked on
Self-Attention, repeated 6× for both encoder and decoder.

## Slide 34 — Training Trick #1: Residual Connections [He et al., 2016]

Residual connections are a simple but powerful technique from computer vision. Deep
networks are surprisingly bad at learning the identity function! Therefore, directly
passing "raw" embeddings to the next layer can actually be very helpful: x_ℓ =
F(x_{ℓ−1}) + x_{ℓ−1}. This prevents the network from "forgetting" or distorting
important information as it is processed by many layers. Callout: residual
connections are also thought to smooth the loss landscape and make training easier —
side-by-side 3D loss-landscape visualizations [no residuals] (jagged, mountainous) vs.
[residuals] (smooth bowl) [Li et al., 2018, on a ResNet].

## Slide 35 — Training Trick #2: Layer Normalization [Ba et al., 2016] (1)

**Problem**: Difficult to train the parameters of a given layer because its input from
the layer beneath keeps shifting. **Solution**: Reduce variation by **normalizing** to
zero mean and standard deviation of one within each **layer**. Mean: μ^ℓ = (1/H)
Σ_{i=1}^H a_i^ℓ; Standard Deviation: σ^ℓ = √((1/H) Σ_{i=1}^H (a_i^ℓ − μ^ℓ)²).
Standardization: x^ℓ′ = (x^ℓ − μ^ℓ) / (σ^ℓ + ε).

## Slide 36 — Training Trick #2: Layer Normalization [Ba et al., 2016] (2)

Same equations as slide 35, plus a worked illustration ("An Example of How LayerNorm
Works", credited to Bala Priya C., Pinecone): a 1-batch, 3-sample, 4-feature table
(x_1…x_4) with per-sample mean and std_dev computed — normalization across features,
independently for each sample.

## Slide 37 — Training Trick #3: Scaled Dot Product Attention

After LayerNorm, the mean and variance of vector elements is 0 and 1, respectively.
(Yay!) However, the dot product still tends to take on extreme values, as its variance
scales with dimensionality d_k. Quick Statistics Review: Mean of sum = sum of means =
d_k·0 = 0; Variance of sum = sum of variances = d_k·1 = d_k; To set the variance to 1,
simply divide by √d_k! Updated Self-Attention Equation: **Output = softmax(QKᵀ/√d_k)V**.

## Slide 38 — Major issue!

We're almost done with the Encoder, but we have a major problem! Has anyone spotted
it? Consider this sentence: "Man eats small dinosaur." Same Transformer-based
encoder-decoder dependency diagram as slide 23/24.

## Slide 39 — Major issue! (explanation)

Same as slide 38, plus: Wait a minute, order doesn't impact the network at all! This
seems wrong given that word order does have meaning in many languages, including
English!

## Slide 40 — Solution: Inject Order Information through Positional Encodings!

Encoder/decoder diagram now shows a "Positional Encoding" sinusoid symbol added
(⊕) to the Input/Output Embeddings before Scaled Attention.

## Slide 41 — Fixing the first self-attention problem: sequence order

Since self-attention doesn't build in order information, we need to encode the order
of the sentence in our keys, queries, and values. Consider representing each
**sequence index** as a **vector**: p_i ∈ ℝ^d, for i ∈ {1,2,…,T} are position vectors.
Don't worry about what the p_i are made of yet! Easy to incorporate this info into our
self-attention block: just add the p_i to our inputs! Let ṽ_i, k̃_i, q̃_i be our old
values, keys, and queries: v_i = ṽ_i + p_i, q_i = q̃_i + p_i, k_i = k̃_i + p_i. Callout:
"In deep self-attention networks, we do this at the first layer! You could concatenate
them as well, but people mostly just add…"

## Slide 42 — Position representation vectors through sinusoids (original)

**Sinusoidal position representations**: concatenate sinusoidal functions of varying
periods: p_i = (sin(i/10000^(2·1/d)), cos(i/10000^(2·1/d)), …, sin(i/10000^(2·(d/2)/d)),
cos(i/10000^(2·(d/2)/d))). Heatmap visualization (Dimension × Index in the sequence)
shows a wavy interference pattern. **Pros**: periodicity indicates that maybe "absolute
position" isn't as important; maybe can extrapolate to longer sequences as periods
restart. **Cons**: not learnable; also the extrapolation doesn't really work.

## Slide 43 — Extension: Self-Attention w/ Relative Position Encodings

**Key Insight**: The most salient position information is the relationship (e.g. "cat"
is the word before "eat") between words, rather than their absolute position (e.g.
"cat" is word 2"). Original Self-Attention Output: z_i = Σ_j α_ij(x_j**W**^V). where
α_ij = exp(e_ij)/Σ_k exp(e_ik); e_ij = (x_i**W**^Q)(x_j**W**^K)ᵀ/√d_z. Relation-Aware
Self-Attention Output adds a relative term: z_i = Σ_j α_ij(x_j**W**^V + a^V_ij), where
e_ij = x_i**W**^Q(x_j**W**^K + a^K_ij)ᵀ/√d_z, a^K_ij = w^K_clip(j−i,k),
a^V_ij = w^V_clip(j−i,k), clip(x,k) = max(−k, min(k,x)) — learned relative position
representations w^K = (w^K_{−k}, …, w^K_k) and w^V = (w^V_{−k}, …, w^V_k). Side table
shows EN-DE BLEU rising from 12.5 at clipping window k=0 to 25.9 at k=16–64, then
plateauing. Table and equations from [Shaw et al., 2018].

## Slide 44 — Multi-Headed Self-Attention: k heads are better than 1!

**High-Level Idea**: Let's perform self-attention multiple times in parallel and
combine the results. Canonical multi-head-attention diagram (V, K, Q → three Linear
projections → Scaled Dot-Product Attention ×h → Concat → Linear) [Vaswani et al.
2017], alongside a multi-headed hydra illustration (Wizards of the Coast, artist Todd
Lockwood) playing on "heads".

## Slide 45 — The Transformer Encoder: Multi-headed Self-Attention

What if we want to look in multiple places in the sentence at once? For word *i*,
self-attention "looks" where x_iᵀQᵀKx_j is high, but maybe we want to focus on
different *j* for different reasons? We'll define **multiple attention "heads"**
through multiple Q,K,V matrices. Let Q_ℓ, K_ℓ, V_ℓ ∈ ℝ^(d×d/h), where h is the number
of attention heads, and ℓ ranges from 1 to h. Each attention head performs attention
independently: output_ℓ = softmax(XQ_ℓK_ℓᵀXᵀ)·XV_ℓ, where output_ℓ ∈ ℝ^(d/h). Then the
outputs of all the heads are combined: output = Y[output₁;…;output_h], where
Y ∈ ℝ^(d×d). Each head gets to "look" at different things, and construct value
vectors differently. Side image: a BERTviz-style attention visualization (layer 5,
"it" attending across "The animal didn't cross the street because it was too tire d",
credit jalammar.github.io/illustrated-transformer/), with multiple colored attention
heads shown fanning from "it_" to earlier tokens.

## Slide 46 — Yay, we've completed the Encoder! Time for the Decoder…

Encoder/decoder diagram with the completed Encoder stack (Multi-Head Attention → Add
& Norm → Feed Forward → Add & Norm, with positional encoding) shown in full color; the
Decoder box still empty.

## Slide 47 — Decoder: Masked Multi-Head Self-Attention (1)

**Problem**: How do we keep the decoder from "cheating"? If we have a language
modeling objective, can't the network just look ahead and "see" the answer? Same
dependency diagram as slides 23/24/38/39.

## Slide 48 — Decoder: Masked Multi-Head Self-Attention (2)

Same as slide 47, plus: **Solution**: Masked Multi-Head Attention. At a high level, we
hide (mask) information about future tokens from the model.

## Slide 49 — Masking the future in self-attention

To use self-attention in **decoders**, we need to ensure we can't peek at the future.
At every timestep, we could change the set of **keys and queries** to include only
past words. (Inefficient!) To enable parallelization, we **mask out attention** to
future words by setting attention scores to −∞: e_ij = q_iᵀk_j if j < i, else −∞ if
j ≥ i. Triangular matrix diagram over [START]/The/chef/who, shaded gray above the
diagonal ("We can look at these (not greyed out) words" / "For encoding these words").

## Slide 50 — Decoder: Masked Multi-Headed Self-Attention (3)

Encoder/decoder diagram with the Decoder's first sub-block now labeled "Masked
Multi-Head Attention".

## Slide 51 — Encoder-Decoder Attention

We saw that self-attention is when keys, queries, and values come from the same
source. In the decoder, we have attention that looks more like what we saw last week.
Let h₁, …, h_T be **output** vectors from the Transformer **encoder**; x_i ∈ ℝ^d. Let
z₁, …, z_T be input vectors from the Transformer **decoder**, z_i ∈ ℝ^d. Then keys and
values are drawn from the **encoder** (like a memory): k_i = **K**h_i, v_i = **V**h_i.
And the queries are drawn from the **decoder**, q_i = **Q**z_i. Diagram now labels the
decoder's second sub-block "Multi-Head Cross-Attention".

## Slide 52 — Decoder: Finishing touches! (1)

Encoder/decoder diagram, decoder side now shown with Masked Multi-Head Attention →
Add & Norm → Multi-Head Attention → Add & Norm (Feed Forward not yet added).

## Slide 53 — Decoder: Finishing touches! (2)

Add a feed forward layer (with residual connections and layer norm). Diagram now
completes the decoder stack with Feed Forward → Add & Norm on top.

## Slide 54 — Decoder: Finishing touches! (3)

Add a feed forward layer (with residual connections and layer norm). Add a final
linear layer to project the embeddings into a much longer vector of length vocab size
(logits). A "Linear" box now sits atop the decoder stack.

## Slide 55 — Decoder: Finishing touches! (4)

Add a final softmax to generate a probability distribution of possible next words! A
"Softmax" box now sits atop "Linear", producing "Output Probabilities" — the complete
decoder stack.

## Slide 56 — Recap of Transformer Architecture

Full canonical architecture diagram (same as slide 6), now fully assembled and
labeled: Encoder (Multi-Head Attention → Add & Norm → Feed Forward → Add & Norm,
repeat 6×) and Decoder (Masked Multi-Head Attention → Add & Norm → Multi-Head
Attention → Add & Norm → Feed Forward → Add & Norm, repeat 6×), with Positional
Encoding, Input/Output Embedding, and Linear → Softmax → Output Probabilities.

## Slide 57 — Outline (§4 highlighted)

Same four-item outline, with "Drawbacks and Variants of Transformers" highlighted.

## Slide 58 — What would we like to fix about the Transformer?

**Quadratic compute in self-attention (today)**: Computing all pairs of interactions
means our computation grows **quadratically** with the sequence length! For recurrent
models, it only grew linearly! **Position representations**: Are simple absolute
indices the best we can do to represent position? As we learned: Relative linear
position attention [Shaw et al., 2018]; Dependency syntax-based position [Wang et al.,
2019]; Rotary Embeddings [Su et al., 2021].

## Slide 59 — Recent work on improving on quadratic self-attention cost (1): Linformer

Considerable recent work has gone into the question, *Can we build models like
Transformers without paying the O(T²) all-pairs self-attention cost?* For example,
**Linformer** [Wang et al., 2020]. Key idea: map the sequence length dimension to a
lower-dimensional space for values, keys (diagram: V/K pass through an extra
"Projection" layer before Scaled Dot-Product Attention). Inference-time line chart
(seconds vs. sequence length/batch size, from 512/128 to 65536/1): the plain
Transformer's inference time rises steeply past 120s, while Linformer variants (k=128
to k=2048) stay nearly flat, under ~20s even at the longest sequence lengths.

## Slide 60 — Recent work on improving on quadratic self-attention cost (2): BigBird

For example, **BigBird** [Zaheer et al., 2021]. Key idea: replace all-pairs
interactions with a family of other interactions, like **local windows**, **looking at
everything**, and **random interactions**. Four grid diagrams: (a) Random attention
(scattered dots), (b) Window attention (a banded diagonal), (c) Global Attention
(full first row/column), (d) BIGBIRD (a combination of all three patterns).

## Slide 61 — Do Transformer Modifications Transfer?

*"Surprisingly, we find that most modifications do not meaningfully improve
performance."* Large results table (Vanilla Transformer and ~25 architectural
variants — GeGLU, Switch, ELU, GLU, DeGLU, ReGLU, SeGLU, LiGLU, Sigmoid, Softplus, RMS
Norm, Rezero, Rezero + LayerNorm, Fixup, Factorized embedding, Factorized & shared
embeddings, Tied encoder/decoder input-output embeddings, Adaptive softmax, Adaptive
softmax without projection, Mixture of softmaxes, Transparent attention, Dynamic
convolution (lightweight/expert/synthesizer variants), Universal Transformer, Mixture
of experts, Switch Transformer, Product key memory, Weighted transformer) reporting
Params, Ops, Step/s, Early loss, Final loss, and SGLUE/XSum/WebQ/WMT EnDe scores — the
best-performing variants (e.g. Mixture of softmaxes, Switch Transformer) edge out
Vanilla Transformer only modestly, and several score worse. From "Do Transformer
Modifications Transfer Across Implementations and Applications?" — Sharan Narang,
Hyung Won Chung, Yi Tay, William Fedus, Thibault Fevry, Michael Matena, Karishma
Malkan, Noah Fiedel, Noam Shazeer, Zhenzhong Lan, Yanqi Zhou, Wei Li, Nan Ding, Jake
Marcus, Adam Roberts, Colin Raffel.

## Slide 62 — Parting remarks

Yay, you now understand Transformers! Next class, we will see how pre-training can
take performance to the next level! Good luck on assignment 4! Remember to work on
your project proposal!
