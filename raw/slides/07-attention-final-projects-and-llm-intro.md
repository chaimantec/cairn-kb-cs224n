---
title: Lecture 7 — Attention, Final Projects and LLM Intro (slide deck)
lecture: 7
slides: 73 printed / 73 pages in the PDF
source_pdf: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture07-final-project.pdf
note: Printed slide numbers match PDF page numbers 1:1, no gaps or offset.
---

# Lecture 7 — Attention, Final Projects and LLM Intro: slide-by-slide

Text and figures of
[`cs224n-spr2024-lecture07-final-project.pdf`](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture07-final-project.pdf),
transcribed from the deck. Diagrams and plots are described in prose since the KB is
read as text.

The title slide calls this *Lecture 7: Attention — final Projects — practical tips!*;
the course catalog lists it as *Attention, Final Projects and LLM Intro*.

Companion pages: [wiki page for this lecture](../../wiki/07-attention-final-projects-and-llm-intro.md) ·
[transcript](../transcripts/07-attention-final-projects-and-llm-intro.md)

## Contents

| Slides | Section |
| ------ | ------- |
| 1–2 | Title and lecture plan |
| 3 | Recap: multi-layer deep encoder-decoder MT net (the conditioning bottleneck) |
| 4–6 | §1 Evaluation of MT — BLEU, worked example, MT progress over time |
| 7–28 | §2 Attention — the bottleneck problem, the mechanism, equations, why it's great, variants, general definition |
| 29 | §3 Course work and grading policy |
| 30–35 | The Final Project: default vs. custom, staff/mentor info, gamesmanship |
| 36–37 | Computing: how to get GPUs and API credit |
| 38–41 | Project proposal, milestone, and writeup requirements |
| 42–50 | §4 finding research topics — project types, and eight past-project examples |
| 51–58 | Where to find ideas, state-of-the-art leaderboards, the old-vs-new-DL-NLP shift, using ChatGPT/GPT-4 |
| 59–66 | §5 Finding data — Hugging Face, Papers with Code, LDC, train/dev/test discipline |
| 67–72 | Training models and pots of data; getting a network to train; experimental strategy |
| 73 | Closing: good luck with your projects |

---

## Slide 1 — Title

**Natural Language Processing with Deep Learning — CS224N/Ling284**

Christopher Manning. Lecture 7: Attention and Final Projects; Practical Tips.

## Slide 2 — Lecture Plan

**Lecture 7: Attention – final Projects – practical tips!**

1. Evaluation of MT [5 mins]
2. Attention [30 mins] — Mini Break —
3. Final projects types and details; assessment revisited [20 mins]
4. Finding research topics and sources of data [25 mins]

**Announcements**
- Final project guide and project proposal details is out today! You get 9 days for it,
  due Thu; a significant part of it is writing a paper review. Talk to TAs about it.
- Also make sure that you're underway on Assignment 3, due Tue — get started early,
  it's bigger and harder coding-wise than the previous assignments. **Starting with
  Ass 3, the TAs will no longer look at and debug your code for you!**

## Slide 3 — 1. Multi-layer deep encoder-decoder machine translation net

[Sutskever et al. 2014; Luong et al. 2015]

Diagram of a three-layer LSTM encoder-decoder for German→English translation ("Die
Proteste waren am Wochenende eskaliert" → "The protests escalated over the weekend").
Callout: "The hidden states from RNN layer *i* are the inputs to RNN layer *i* + 1."
Handwritten annotation: **Conditioning = Bottleneck**.

## Slide 4 — How do we evaluate Machine Translation?

Commonest way: **BLEU** (**Bi**lingual **E**valuation **U**nderstudy). *"You'll see
BLEU in detail in Assignment 3!"*

- BLEU compares the machine-written translation to one or several human-written
  translation(s), and computes a similarity score based on:
  - Geometric mean of *n*-gram precision (usually for 1, 2, 3 and 4-grams)
  - Plus a penalty for too-short system translations
- BLEU is useful but imperfect
  - There are many valid ways to translate a sentence
  - Therefore, a good translation can get a poor BLEU score because it has low *n*-gram
    overlap with the human translation

Source: "BLEU: a Method for Automatic Evaluation of Machine Translation", Papineni et
al, 2002.

## Slide 5 — BLEU score against 4 reference translations

[Papineni et al. 2002]. *"But commonly now there is only one reference and so the
results are more 'in expectation'."*

Worked example: a machine translation of a Guam bio-attack news item ("The American
[?] international airport and its the office all receives one calls self the sand Arab
rich business [?] and so on electronic mail, which sends out; The threat will be able
after public place and so on the airport to start the biochemistry attack, [?] highly
alerts after the maintenance.") is matched against four independently human-written
reference translations, with overlapping spans circled and connected by lines — e.g.
"International Airport and its" matches reference 2 as a 4-gram, "biochemistry" matches
reference 4.

## Slide 6 — MT progress over time

[Edinburgh En-De WMT newstest2013 Cased BLEU; NMT 2015 from U. Montréal; NMT 2019 FAIR
on newstest2019]

Bar/line chart, 2013–2019, BLEU score on the y-axis (0–45): phrase-based SMT and
syntax-based SMT both climb slowly from ~20 to the mid-20s over the whole period; Neural
MT starts *below* both in 2015 (~19) but crosses them by 2016 and reaches ~42 by 2019,
far outpacing the other two.

## Slide 7 — 2. Why attention? Sequence-to-sequence: the bottleneck problem

Diagram: source sentence "il a m'entarté" (French for "he pied me") encoded word-by-word
by an encoder RNN, with the final encoder hidden state (labeled "Encoding of the source
sentence") feeding a decoder RNN that generates "he hit me with a pie <END>" one token
at a time, starting from `<START>`. Callout box: **"Problems with this architecture?"**

## Slide 8 — Why attention? Sequence-to-sequence: the bottleneck problem (cont.)

Same diagram as slide 7, with the callout expanded: *"Encoding of the source sentence.
This needs to capture* **all information** *about the source sentence. Information
bottleneck!"*

## Slide 9 — Attention

- **Attention** provides a solution to the bottleneck problem.
- **Core idea**: on each step of the decoder, use *direct connection to the encoder* to
  *focus on a particular part* of the source sequence
- First, we will show via diagram (no equations), then we will show with equations

## Slide 10 — Sequence-to-sequence with attention (1)

**Core idea**: on each step of the decoder, use direct connection to the encoder to
focus on a particular part of the source sequence. Diagram: the decoder's first hidden
state (after `<START>`) computes a dot product against the first encoder hidden state
(over "il"), producing one attention score.

## Slide 11 — Sequence-to-sequence with attention (2)

Same diagram, now showing dot products against the first two encoder hidden states
("il", "a"), producing two attention scores.

## Slide 12 — Sequence-to-sequence with attention (3)

Dot products extended to three encoder hidden states ("il", "a", "m'").

## Slide 13 — Sequence-to-sequence with attention (4)

Dot products computed against all four encoder hidden states ("il", "a", "m'",
"entarté") — the full set of attention scores for this decoder step.

## Slide 14 — Sequence-to-sequence with attention (5): attention distribution

Callout: *"Take softmax to turn the scores into a probability distribution."* The
resulting attention distribution is shown as a bar chart, with almost all the weight
on the first position — callout: *"On this decoder timestep, we're mostly focusing on
the first encoder hidden state ('he')."*

## Slide 15 — Sequence-to-sequence with attention (6): attention output

Callout: *"Use the attention distribution to take a* **weighted sum** *of the* encoder
hidden states*. The attention output mostly contains information from the hidden
states that received high attention."* The weighted sum produces a single "Attention
output" vector.

## Slide 16 — Sequence-to-sequence with attention (7): first output word

Callout: *"Concatenate attention output with decoder hidden state, then use to compute
ŷ₁ as before."* The model outputs "he" as the first translated word.

## Slide 17 — Sequence-to-sequence with attention (8): second step

The decoder advances one step (having generated "he"), and the attention distribution
now peaks on the second encoder position ("a"). Callout: *"Sometimes we take the
attention output from the previous step, and also feed it into the decoder (along with
the usual decoder input). We do this in Assignment 4."* Output: "hit".

## Slide 18 — Sequence-to-sequence with attention (9): third step

Attention distribution now peaks on the third position ("m'"). Output: "me".

## Slide 19 — Sequence-to-sequence with attention (10): fourth step

Attention distribution peaks sharply on the fourth position ("entarté"). Output: "with".

## Slide 20 — Sequence-to-sequence with attention (11): fifth step

Attention again peaks on "entarté". Output: "a".

## Slide 21 — Sequence-to-sequence with attention (12): sixth step

Attention again peaks on "entarté". Output: "pie" — completing "he hit me with a pie".

## Slide 22 — Attention: in equations

- We have encoder hidden states h₁, …, h_N ∈ ℝʰ
- On timestep *t*, we have decoder hidden state s_t ∈ ℝʰ
- We get the attention scores e^t for this step: **e^t = [s_tᵀh₁, …, s_tᵀh_N] ∈ ℝᴺ**
- We take softmax to get the attention distribution α^t for this step (a probability
  distribution, sums to 1): **α^t = softmax(e^t) ∈ ℝᴺ**
- We use α^t to take a weighted sum of the encoder hidden states to get the attention
  output a_t: **a_t = Σᵢ₌₁ᴺ α^t_i h_i ∈ ℝʰ**
- Finally we concatenate the attention output a_t with the decoder hidden state s_t and
  proceed as in the non-attention seq2seq model: **[a_t; s_t] ∈ ℝ²ʰ**

## Slide 23 — Attention is great!

- Attention significantly **improves NMT performance** — it's very useful to allow the
  decoder to focus on certain parts of the source
- Attention provides a more **"human-like" model** of the MT process — you can look
  back at the source sentence while translating, rather than needing to remember it all
- Attention **solves the bottleneck problem** — allows decoder to look directly at
  source; bypass bottleneck
- Attention **helps with the vanishing gradient problem** — provides shortcut to
  faraway states
- Attention provides **some interpretability**
  - By inspecting attention distribution, we see what the decoder was focusing on
  - We get (soft) **alignment for free**! This is cool because we never explicitly
    trained an alignment system — the network just learned alignment by itself

A small alignment-matrix graphic shows "il"/"a"/"m'"/"entarté" (rows) against
"he"/"hit"/"me"/"with"/"a"/"pie" (columns), darkest where "entarté" aligns with "hit",
"with", "a", and "pie" — the "pie" verb's alignment spreads across the whole English
verb phrase, as demonstrated in the transcript's worked example (≈24:21).

## Slide 24 — There are several attention variants

- We have some *values* h₁, …, h_N ∈ ℝ^d1 and a *query* s ∈ ℝ^d2
- Attention always involves: (1) Computing the *attention scores* e ∈ ℝᴺ (*"There are
  multiple ways to do this"*); (2) Taking softmax to get *attention distribution* α:
  **α = softmax(e) ∈ ℝᴺ**; (3) Using attention distribution to take weighted sum of
  values: **a = Σᵢ₌₁ᴺ αᵢhᵢ ∈ ℝ^d1**, thus obtaining the *attention output* **a**
  (sometimes called the *context vector*)

## Slide 25 — Attention variants

*"You'll think about the relative advantages/disadvantages of these in Assignment 3!"*

- **Basic dot-product attention**: e_i = sᵀh_i ∈ ℝ. This assumes d1 = d2. This is the
  version we saw earlier.
- **Multiplicative attention**: e_i = sᵀ**W**h_i ∈ ℝ [Luong, Pham, and Manning 2015].
  Where **W** ∈ ℝ^(d2×d1) is a weight matrix. Perhaps better called "bilinear attention".

## Slide 26 — Attention variants (continued)

*"You'll think about the relative advantages/disadvantages of these in Assignment 3!"*

- **Reduced-rank multiplicative attention**: e_i = sᵀ(**U**ᵀ**V**)h_i = (**U**s)ᵀ(**V**h_i).
  For low-rank matrices **U** ∈ ℝ^(k×d2), **V** ∈ ℝ^(k×d1), k ≪ d1, d2
- **Additive attention**: e_i = **v**ᵀtanh(**W**₁h_i + **W**₂s) ∈ ℝ [Bahdanau, Cho, and
  Bengio 2014]. Where **W**₁ ∈ ℝ^(d3×d1), **W**₂ ∈ ℝ^(d3×d2) are weight matrices and
  **v** ∈ ℝ^d3 is a weight vector. d3 (the attention dimensionality) is a
  hyperparameter. "Additive" is a weird/bad name — it's really using a feed-forward
  neural net layer. Callout: *"Remember this when we look at Transformers next week!"*

More information: "Deep Learning for NLP Best Practices", Ruder, 2017; "Massive
Exploration of Neural Machine Translation Architectures", Britz et al, 2017.

## Slide 27 — Attention is a general Deep Learning technique

- We've seen that attention is a great way to improve the sequence-to-sequence model
  for Machine Translation.
- **However**: You can use attention in *many architectures* (not just seq2seq) and
  *many tasks* (not just MT)
- **More general definition of attention**: Given a set of vector *values*, and a
  vector *query*, **attention** is a technique to compute a weighted sum of the values,
  dependent on the query.
- We sometimes say that the query *attends to* the values.
- For example, in the seq2seq + attention model, each decoder hidden state (query)
  *attends to* all the encoder hidden states (values).

## Slide 28 — Attention is a general Deep Learning technique (intuition)

**More general definition of attention** (repeated from slide 27).

**Intuition**:
- The weighted sum is a *selective summary* of the information contained in the values,
  where the query determines which values to focus on.
- Attention is a way to obtain a *fixed-size representation of an arbitrary set of
  representations* (the values), dependent on some other representation (the query).

**Upshot**: Attention has become the powerful, flexible, general way pointer and
memory manipulation in all deep learning models. A new idea from after 2010! From NMT!

## Slide 29 — 3. Course work and grading policy

- 4 × ~9 day Assignments: 6% + 3 × 14%: 48%
- Final Default or Custom Course Project (1–3 people): 49%. Project proposal: 8%;
  milestone: 6%; poster: 3%; report: 32%
- Participation: 3% — Attending guest speaker lectures, Ed, our course evals, karma —
  see website!
- **Late day policy**: 6 free late days; then 1% of total off per day after that; max 3
  late days per assignment
- **Collaboration policy**: Read the website and the Honor Code! For projects: it's
  okay to use existing code/resources, but you must document it, and **you will be
  graded on your value-add**. If multi-person: include a brief statement on the work of
  each team-mate (almost always everyone gets the same score, but we reserve the right
  to differentiate if needed). If using project for other classes or as RAship, PhD
  rotation, etc., you must indicate this in the proposal.

## Slide 30 — The Final Project

- For FP, you either: do the default project, which is **minBERT and Downstream
  Tasks** (open-ended but an easier start; a good choice for most), or propose a custom
  final project, which we must approve (you will receive feedback from a **mentor** —
  TA/prof/postdoc/PhD)
- You can work in teams of 1–3. Being in a team is encouraged. A larger team project,
  or a project used for multiple classes, should be larger and often involves exploring
  more models or tasks
- You can use any language/framework for your project, though we expect nearly all of
  you to keep using PyTorch

## Slide 31 — Custom Final Project

*"I'm very happy to talk to people about final projects, but the slight problem is that
there's only one of me…"* Look at TA expertise for custom final projects:
http://web.stanford.edu/class/cs224n/office_hours.html#staff

Staff-expertise table lists office hours by day/time/room for: Chris Manning (most
areas of NLP), Rashon Poole (chatbot, human-LLM interaction), Shijia Yang (multimodal
LLMs), Ryan Li (NLP, human-AI interaction), Shikhar Murty (language models,
compositionality, reasoning, grounding), Neil Nie (CV, LM, VLM, RAG, robotics), Kaylee
Burns (robotics, RL), Zhoujie Ding (trustworthy ML), Jingwen Wu (NLP, healthcare), Josh
Singh (QA, conversational assistants, GNNs), Chaofei Fan (brain speech to text), Moussa
Doumbouya (NLP, translation, under-served languages, computational education), Archit
Sharma (reinforcement learning, RLHF, robotics), Olivia Lee (language models, RL,
VLMs/multimodal, robotics), Yann Dubois (LLM fine-tuning and evaluation,
representation learning), Kamyar Salahi (vision and language), Johnny Chang (NLP,
RAGs), Soumya Chatterjee (NLP, translation, QA), Aditya Agrawal (multimodal learning,
explainability, GNNs), Yuan Gao (pretraining, multimodal LLMs, generative models), Anna
Goldie (language models, RL, retrieval-augmented generation), Timothy Dai (AI for
sustainability), Sonia Chu (vision, time series), Arvind Mahankali (language models, AI
for math, deep learning theory).

## Slide 32 — The Default Final Project

- The 2024 final project handouts are on the website now!
- **This year**: Building and experimenting with a minBERT implementation. Provided
  starter code in PyTorch.
- We will discuss transformer models and BERT next in the class (next 2+ lectures)
- What you do: finish writing an implementation of BERT; fine-tune it for sentiment
  analysis (example: Rotten Tomatoes review "Light, silly, photographed with colour and
  depth, and rather a good time." → Sentiment: 4, Positive); extend and improve it in
  various ways of your choice: contrastive learning, paraphrasing, regularized
  optimization.

## Slide 33 — Why Choose The Default Final Project?

If you: have limited experience with research, don't have any clear idea of what you
want to do, or want guidance and a goal, … and a leaderboard, even. Then: do the
default final project! Many people should do it! (Past statistics: about half of
people do DFP.)

## Slide 34 — Why Choose The Custom Final Project?

If you: have some research idea that you're excited about (and are possibly already
working on it); want to try to do something different on your own; want to explore
more of the process of defining a research goal, finding data and tools, and working
out something you could do that is interesting (and perhaps novel), and how to
evaluate it. Then: do the custom final project! **Note: The final project for CS224N
must substantively involve both human language and neural networks** (the topics of
this class)!

## Slide 35 — Gamesmanship

The default final project is more guided, but it should be the same amount of work.
It's just that you can focus on a given problem rather than coming up with your own.
The default final project is also an open-ended project where you can explore
different approaches, but to a given problem. Strong default final projects do new
things. There are great default final projects and great custom final projects … and
there are weak default final projects and weak custom final projects. The path to
success is not to do something that looks kinda weak/ill-considered compared to what
you could have done with the DFP. Neither option is the easy way to a good grade; we
give Best Project Awards to both.

## Slide 36 — Computing: How to get GPUs (which you will need), part 1

Confession: We're not in as good a position for providing free compute as previous
years.

- GCP: Thanks to Google, \$50 credit per person, which can be used for Ass 3, 4, FP
- All clouds (GCP, AWS, Azure, …): if you haven't used a cloud with an account before,
  you can usually get some free credit when you begin [Fine point: does it enable GPU
  use?]
- You can often also usefully use Google Colab, which allows limited GPU use. You'll
  get better GPU availability, especially late in the quarter, if you pay \$10/month
  for Colab Pro. Alternatively, similar Jupyter notebooks in the cloud: AWS Sagemaker
  Studio Lab; Kaggle Notebooks (perhaps with better GPU availability for longer)
- Modal.com: low price GPU provider; allows a limited amount of free compute per month.
  You can also try other providers like Vast.ai, Lambda Labs, …

## Slide 37 — Computing: How to get GPUs (which you will need), part 2

- Together.ai is providing \$50 **API access to LLMs** per FP team if you register with
  us
- You can use other APIs like OpenAI, Anthropic, etc., but you'll need to pay to use
  them. There are some options for free use of OpenAI models on Microsoft, Perplexity,
  Gemini on Google, etc., but you usually don't get API use

See the various guides posted on Ed: GCP, Colab, Modal, and Together guide documents
(linked as Google Docs / Ed discussion threads on the slide).

## Slide 38 — Project Proposal — one from every team, 8%

1. Find a relevant (key) research paper for your topic (for DFP, we provide some
   suggestions, but you might look elsewhere for interesting work)
2. Write a summary of that research paper and what you took away from it as key ideas
   that you hope to use
3. Write what you plan to work on and how you can innovate in your final project work
   (suggest a good milestone to have achieved as a halfway point)
4. Describe as needed, **especially for Custom projects**: a project plan, relevant
   existing literature, the kind(s) of models you will use/explore; the **data** you
   will use (and how it is obtained), and how you will **evaluate** success
5. Write an ethical considerations paragraph, outlining potential ethical challenges of
   your work if deployed in the real world and practical risk mitigation strategies

3–4 pages, due Thu May 2, 4:30pm on Gradescope (separate CFP and DFP submissions).

## Slide 39 — Project Proposal (grading guidance)

2. How to think critically about a research paper for the summary: what were the main
   novel contributions or points? Is what makes it work something general and reusable
   or a special case? Are there flaws or neat details in what they did? How does it fit
   with other papers on similar topics? Does it provoke good questions on further or
   different things to try? *Grading of research paper review is primarily summative
   and is most of the points.*
3. How to do a good job on your project plan: you need to have an overall sensible idea
   (!). But most project plans that are lacking are lacking in nuts-and-bolts ways: do
   you have appropriate data or a realistic plan to be able to collect it in a short
   period of time; do you have a realistic way to evaluate your work; do you have
   appropriate baselines or proposed ablation studies for comparisons. *Grading of
   project proposal is primarily formative.*

## Slide 40 — Project Milestone — from everyone, 6%

This is a progress report. You should be more than halfway done! Describe the
experiments you have run; describe the preliminary results you have obtained; describe
how you plan to spend the rest of your time. You are expected to **have implemented
some system** and to **have some initial experimental results** to show by this date
(except for certain unusual kinds of projects). Due Tue May 21, 4:30pm on Gradescope.

## Slide 41 — Project writeup

Writeup quality is very important to your grade!!! Look at recent years' prize winners
for good examples. A typical paper structure is shown as eight boxes: Abstract/
Introduction, Prior related work, Model, Model, Data, Experiments, Results, Analysis &
Conclusion.

## Slide 42 — Finding Research Topics

Two basic starting points, for all of science:
- [Nails] Start with a (domain) problem of interest and try to find good/better ways to
  address it than are currently known/used
- [Hammers] Start with a technical method/approach of interest, and work out good ways
  to extend it, improve it, understand it, or find new ways to apply it

## Slide 43 — Project types

This is not an exhaustive list, but most projects are one of:

1. Find an application/task of interest and explore how to approach/solve it
   effectively, often with an existing model (could be a task in the wild or, more
   commonly, some existing Kaggle/bake-off/shared task)
2. Implement a complex neural architecture (perhaps novel) and explore its performance
   on some data
3. Explore using prompting (in-context learning) or language model programs (the above
   types are expected to have experiments with numbers and ablations)
4. Analysis/Interpretability project. Analyze the behavior of a model: how it
   represents linguistic or world knowledge, or what kinds of phenomena it can handle
   or errors that it makes
5. Rare theoretical or linguistic project: Show some interesting, non-trivial
   properties of a model type, data, or a data representation

## Slide 44 — Example: Deep Poetry

**"Deep Poetry: Word-Level and Character-Level Language Models for Shakespearean
Sonnet Generation"** — Stanley Xie, Ruchir Rastogi and Max Chang.

A **Gated LSTM** architecture generates sonnet-style text, e.g.: *"Thy youth 's time and
face his form shall cover? / Now all fresh beauty, my love there / Will ever Time to
greet, forget each, like ever decease, / But in a best at worship his glory die."* The
gated LSTM combines word-level and character-level representations via a learned gate
g_wt: (1 − g_wt)·x^word_wt + g_wt·x^char_wt, feeding a Word LSTM and softmax to predict
w_{t+1}.

## Slide 45 — Example: Implementation and Optimization of Differentiable Neural Computers

**Carol Hsin**, Graduate Student in Computational & Mathematical Engineering.

Abstract: implemented and optimized Differentiable Neural Computers (DNCs) as
described in the Oct. 2016 DNC paper on the bAbI dataset and on copy tasks described in
the Neural Turing Machine paper — documenting the approach and the challenges of
optimizing DNCs. Diagram: a DNC cell with an NN Controller taking x_t, producing ξ_t
and ν_t, interacting with a Memory Module to produce r^i_t, feeding a Final Output y_t.

## Slide 46 — Example: Improved Learning through Augmenting the Loss

**Hakan Inan** and **Khashayar Khosravi** (inanh@stanford.edu, khosravi@stanford.edu).

Two improvements to Recurrent Neural Network Language Models (RNNLM): (1) use the word
embedding matrix to project the RNN output onto the output space, achieving a large
reduction in free parameters while still improving performance; (2) instead of merely
minimizing standard cross-entropy loss between the prediction distribution and the
"one-hot" target distribution, minimize an additional loss term that accounts for the
inherent metric similarity between the target word and other words. Experiments on the
Penn Treebank Dataset show (1) significantly lower average word perplexity than
previous models of the same network size and (2) a new state of the art using far
fewer parameters. Published as a conference paper at ICLR 2017. (This is the paper
behind the "share word-vector and output-matrix" idea discussed in the transcript,
≈1:06:11.)

## Slide 47 — Example: ConCoRD (Enhancing Self-Consistency and Performance of Pre-Trained Language Models through Natural Language Inference)

Eric Mitchell, Joseph J. Noh, Siyan Li, William S. Armstrong, Ananth Agarwal, Patrick
Liu, Chelsea Finn, Christopher D. Manning, Stanford University.

Large pre-trained LMs often lack logical consistency across test inputs (e.g. a QA
model answers "Yes" to "Is a sparrow a bird?" and "Does a bird have feet?" but "No" to
"Does a sparrow have feet?"). ConCoRD (Consistency Correction through Relation
Detection) boosts consistency and accuracy of pre-trained NLP models using pre-trained
natural language inference (NLI) models, without fine-tuning or re-training: it samples
candidate outputs, estimates pairwise constraints via an NLI model, and finds optimal
assignments with a weighted MaxSAT solver. Merge of the work of two CS224N project
teams, published at EMNLP 2022.

## Slide 48 — Example: Word2Bits — Quantized Word Vectors

**Maximilian Lam** (maxlam@stanford.edu).

Word vectors require significant memory/storage, an issue for resource-limited devices
like mobile phones and GPUs. Shows that high-quality quantized word vectors using 1–2
bits per parameter can be learned by introducing a quantization function into
Word2Vec, which also acts as a regularizer. Word vectors trained on English Wikipedia
(2017), evaluated on word similarity/analogy tasks and question answering (SQuAD):
8–16× less space than full precision (32-bit) word vectors, while also outperforming
them on word similarity tasks and question answering.

## Slide 49 — Example: Fine-tuning CodeLlama-7B on Synthetic Training Data for Fortran Code Generation using PEFT

Stanford CS224N Custom Project — Soham Govande, Taeuk Kang, Andrew Shi (Winter 2024
project).

Table comparing the fine-tuned model against CodeLlama-7B-Instruct on functions
written/compiled/executed/output-generated/timed-out/runtime-error/incorrect/partially-
correct/correct: the fine-tuned model wrote 522 functions (vs. 448), compiled 185 (vs.
131), executed 125 (vs. 50), and was judged correct on 56 (vs. 32) — a clear
improvement from parameter-efficient fine-tuning (PEFT) on Fortran generation.

## Slide 50 — Example: AI-Driven Fashion Cataloging

Stanford CS224N Custom Project — SiYi Ma, Nishant Gopinath (Winter 2024 project).

Transforming product images into textual descriptions (category, silhouette, fitting,
pattern, shoulder-style, neckline, length, sleeve-length) using a vision-language
model, CogVLM. A worked example shows the pre-trained CogVLM mis-predicting
"silhouette" and "fitting" (in red) against ground truth, while the fine-tuned CogVLM
matches ground truth exactly — "Enhanced Accuracy through Fine-Tuning."

## Slide 51 — 4. How to find an interesting place to start?

- Look at ACL anthology for NLP papers: https://aclanthology.org/
- Also look at the online proceedings of major ML conferences: NeurIPS, ICML, ICLR
- Look at past CS224N projects: http://cs224n.stanford.edu/
- Look at online preprint servers, especially https://arxiv.org
- Even better: look for an interesting problem in the world! Hal Varian: "How to Build
  an Economic Model in Your Spare Time"

## Slide 52 — Want to try to beat the state of the art on something?

Sites that collate state-of-the-art info (not always correct, though):
paperswithcode.com/sota, nlpprogress.com. Specific tasks/topics, e.g.
gluebenchmark.com/leaderboard, conll.org/previous-tasks. Screenshot example: Papers
With Code's Machine Translation leaderboard, showing WMT2014 English-French/German and
IWSLT2015 German-English state-of-the-art results dominated by Transformer variants
(including "Attention Is All You Need" itself on IWSLT2015).

## Slide 53 — Finding a topic

Turing award winner and Stanford CS emeritus professor Ed Feigenbaum says to follow the
advice of his advisor, AI pioneer and Turing and Nobel prize winner Herb Simon: *"If
you see a research area where many people are working, go somewhere else."* But where
to go? Wayne Gretzky: *"I skate to where the puck is going, not where it has been."*

## Slide 54 — Old Deep Learning (NLP), new Deep Learning NLP

In the early days of the Deep Learning revival (2010–2018), most of the work was in
defining and exploring better deep learning architectures. Typical paper: "I can
improve a summarization system by not only using attention standardly, but allowing
copying attention — where you use additional attention calculations and an additional
probabilistic gate to simply copy a word from the input to the output." That's what a
lot of good CS 224N projects did too. In 2019–2024, that approach is dead (well, that's
too strong, but it's difficult and much rarer). Most work downloads a big pre-trained
model (which fixes the architecture) — action is in fine-tuning, or domain adaptation
followed by fine-tuning, etc., etc.

## Slide 55 — 2024 NLP … recommended for most of your practical projects

Code sketch (not quite runnable, gives the general idea):

```
pip install transformers  # By Huggingface — Tutorial: Fri May 3, TBD
from transformers import BertForSequenceClassification, AutoTokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.train()
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
fine_tuner = Trainer(model=model, args=training_args, train_dataset=train_dataset,
    eval_dataset=test_dataset)
fine_tuner.train()
eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
results = evaluate(model, tokenizer, eval_dataset, args)
```

## Slide 56 — Exciting areas 2024 (1)

A lot of what is exciting now is problems that work within or around this world:
evaluating and improving models for something other than accuracy (evaluating
robustness; adaptation under domain shift); doing empirical work on what large
pre-trained models have learned and how; getting knowledge and good task performance
from large models for particular tasks without much data (in-context learning,
transfer learning, etc.); looking at the bias, trustworthiness, and explainability of
large models; working on how to augment data to improve performance; looking at
low-resource languages or problems; improving performance on the tail of rare stuff,
addressing bias; building larger language model programs — look at RAG (Retrieval
Augmented Generation), LangChain, DSPy from Stanford!

## Slide 57 — Exciting areas 2024 (2): Scaling models up and down

- Building big models is BIG … **but just not possible for a CS224N project** — do
  think and be realistic about the scale of compute you can do!
- Building small, performant models is also BIG. This could be a great project: model
  pruning or model quantization; how well can you do QA in 6GB or 500MB?
  (efficientqa.github.io)
- Doing parameter-efficient fine tuning and low-rank adaptation, etc.
- Looking to achieve more advanced functionalities: compositionality, systematic
  generalization, fast learning (e.g. meta-learning) on smaller problems and amounts of
  data, and more quickly. References: COGS/ReCOGS, gSCAN.

## Slide 58 — Can I use ChatGPT (GPT-4) or Gemini etc. to do my final project?

- You need to be very cognizant of how large a model you can train — you just don't
  have the resources to train your own GPT-2 model, let alone Llama. You probably
  don't have the resources to even **load** T5 11B
- You are welcome to use ChatGPT, Claude, Gemini, etc. in your final project — we can
  make some API access available for students via gift from Together.ai
- There are almost certainly interesting projects you can do using them: analysis
  projects, learning good prompts, chain of thought reasoning, larger language model
  systems
- But be careful to remember that you will be evaluated based on what you have done —
  and not on the amazing ChatGPT output that you show us ("look, it works zero shot")

## Slide 59 — 5. Finding data for your projects

A few people collect their own data — **we like that, but it's tricky to do quickly
enough!** You may have a project that uses "unsupervised" data; you can annotate a
small amount of data; you can find a website that effectively provides annotations,
such as likes, stars, ratings, responses, etc. (this lets you learn about real-world
challenges of applying ML/NLP — **but be careful on scoping so this doesn't take most
of your time!!!**). Some people have existing data from a research project or company
(fine to use, providing you can provide data samples for submission, report, etc.).
**Most people make use of an existing, curated dataset built by previous
researchers** — you get a fast start and there is obvious prior work and baselines.

## Slide 60 — Traditional Linguistic Data Provider: Linguistic Data Consortium

https://catalog.ldc.upenn.edu/. Stanford licenses this data; you can get access — sign
up/ask questions at https://linguistics.stanford.edu/resources/resources-corpora.
Treebanks, named entities, coreference data, lots of clean newswire text, lots of
speech with transcription, parallel MT data, etc. **Don't use for non-Stanford
purposes!** Screenshot shows the LDC's "Top Ten LDC Corpora": TIMIT, OntoNotes Release
5.0, Web 1T 5-gram, CELEX2, Treebank-3, NYT Annotated Corpus, TIDIGITS, Switchboard-1
Release 2, English Gigaword Fifth Edition, TIPSTER Complete.

## Slide 61 — Huggingface Datasets

https://huggingface.co/datasets — 638 datasets browsable by task category
(text-classification, sequence-modeling, question-answering, etc.), task,
language, multilinguality, size, and license. Example datasets shown: acronym_identification,
ade_corpus_v2 (adverse drug reaction data), adversarial_qa (crowdsourced reading
comprehension using BiDAF/BERT-Large/RoBERTa-Large in the annotation loop).

## Slide 62 — Paperswithcode Datasets

https://www.paperswithcode.com/datasets?mod=texts&page=1 — 835 dataset results for
"Texts", including Penn Treebank, SQuAD, Visual Genome, GLUE, SNLI, CLEVR, Visual
Question Answering (VQA), Billion Word Benchmark.

## Slide 63 — Machine translation

http://statmt.org — look in particular at the various WMT shared tasks. Screenshot of
the Statistical Machine Translation site's sitemap: SMT Book, Research Survey Wiki,
Moses MT System, Europarl Corpus, News Commentary Corpus, Online Evaluation, Online
Moses Demo, Translation Tool, WMT Workshops 2006–2014, plus tutorial links (Brown/
Della Pietra/Della Pietra/Mercer; Kevin Knight; Callison-Burch/Koehn) and the MT
Archive by John Hutchins.

## Slide 64 — Dependency parsing: Universal Dependencies

https://universaldependencies.org — a framework for cross-linguistically consistent
grammatical annotation and an open community effort with over 200 contributors
producing more than 100 treebanks in over 70 languages. Links to UD annotation
guidelines, contribution guides, treebank-search tools (SETS, PML Tree Query, Kontext,
Grew-match) maintained by the Universities of Turku, Charles University Prague, and
Inria Nancy, and a mailing list / GitHub issue tracker.

## Slide 65 — Many, many more

Look at Kaggle; look at research papers to see what data they use; look at lists of
datasets (machinelearningmastery.com, github.com/niderhoff/nlp-datasets); lots of
particular things (gluebenchmark.com/tasks, nlp.stanford.edu/sentiment/,
research.fb.com/downloads/babi/ for Facebook bAbI). Ask on Ed or talk to course staff.

## Slide 66 — 5. Care with datasets and in model development

Many publicly available datasets are released with a **train/dev/test** structure.
**We're all on the honor system to do test-set runs only when development is
complete. Do development testing on the dev set only!** Splits like this presuppose a
fairly large dataset. If there is no dev set or you want a separate tune set, then you
create one by splitting the training data — weigh the usefulness of it being a certain
size against the reduction in train-set size. Cross-validation is a technique for
maximizing data when you don't have much. Having a fixed test set ensures all systems
are assessed against the same gold data — generally good, but problematic when the
test set turns out to have unusual properties that distort progress on the task.

## Slide 67 — Training models and pots of data (1)

We are always interested in **generalization performance** on an **independent test
set**, not seen or used during training. You build (estimate/train) a model on a
**training set**. Often, you then set further hyperparameters on another, independent
set of data, the **tuning set** — the tuning set is the training set for the
hyperparameters! You measure progress as you go on a **dev set** (development test set
or validation set) — if you do that a lot you overfit to the dev set, so it can be
good to have a second dev set, the **dev2** set. **Only at the end**, you evaluate and
present final numbers on a **test set** — use the final test set **extremely** few
times … ideally only once.

## Slide 68 — Training models and pots of data (2)

The **train**, **tune**, **dev**, and **test** sets need to be completely distinct. It
is invalid to give results testing on material you have trained on — you **will** get
a falsely good performance; we almost always overfit on train. You need an independent
tuning set — the hyperparameters won't be set right if tune is the same as train. If
you keep running on the same evaluation set, you begin to overfit to that evaluation
set — effectively you are "training" on the evaluation set, learning things that do
and don't work on that particular eval set and using the info. To get a valid measure
of system performance you need another untrained-on, independent test set … hence
dev2 and final test.

## Slide 69 — Getting your neural network to train

Start with a positive attitude! **Neural networks want to learn!** If the network
isn't learning, you're doing something to prevent it from learning successfully.
Realize the grim reality: **there are lots of things that can cause neural nets to not
learn at all or to not learn very well** — finding and fixing them ("debugging and
tuning") can often take more time than implementing your model. It's hard to work out
what these things are, but experience, experimental care, and rules of thumb help!

## Slide 70 — Experimental strategy (1)

Work incrementally! Start with a very simple model and get it to work! (It's hard to
fix a complex but broken model.) Add bells and whistles one-by-one and get the model
working with each of them (or abandon them). Initially run on a tiny amount of data —
you will see bugs much more easily on a tiny dataset, and they train really quickly.
Something like 4–8 examples is good. Often synthetic data is useful for this. Make
sure you can get 100% on this data (testing on train) — otherwise your model is
definitely either not powerful enough or it is broken.

## Slide 71 — Experimental strategy (2)

Train and run your model on a large dataset — it should still score close to 100% on
the training data after optimization (otherwise, you probably want to consider a more
powerful model!). Overfitting to training data is **not** something to fear when doing
deep learning — these models are usually good at generalizing because of the way
distributed representations share statistical strength regardless of overfitting to
training data. But, still, you now want good generalization performance: regularize
your model until it doesn't overfit on dev data — strategies like L2 regularization
can be useful, but normally **generous dropout** is the secret to success.

## Slide 72 — Details matter!

Look at your data, collect summary statistics. Look at your model's outputs, do error
analysis. Tuning hyperparameters, learning rates, getting initialization right, etc. is
**often** important to the successes of neural nets.

## Slide 73 — Closing

**Good luck with your projects!**
