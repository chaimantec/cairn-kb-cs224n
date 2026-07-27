---
title: "Lecture 16: ConvNets for NLP and Tree Recursive Neural Networks (slide deck)"
lecture: 17
slides: 60 pages in the PDF; printed numbers run 1–48 and match the PDF page number 1:1; 49–60 are inferred (see note)
source_pdf: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture16-CNN-TreeRNN.pdf
note: |
  Lecturer is Christopher Manning. The deck's own title page reads "Lecture 16: ConvNets for
  NLP and Tree Recursive Neural Networks"; the Cairn catalog lists it at **position 17**, one
  ahead of the deck's own numbering — this is the same systematic off-by-one seen from catalog
  position 11 onward in this course (see `AGENTS.md`), so this file is named
  `17-convnets-and-treernns.md` after the catalog position. The deck filename itself
  (`cs224n-spr2024-lecture16-CNN-TreeRNN.pdf`) has no `lecture17` counterpart in the crawled set
  — the deck-filename numbering and the catalog numbering diverge below `lecture16`, and this
  deck was matched to catalog position 17 ("ConvNets and TreeRNNs") by title, not by filename
  number, per the note in `TODO.md`.

  On slide numbering: this deck prints a small page number in the bottom-left corner of every
  content slide, matching the PDF page number exactly, **with four exceptions inside the
  numbered run that print no number at all**: page 1 (the title page, consistent with every
  other deck in this KB), and pages 16, 18 and 35, each of which sits between two normally
  numbered pages (15/17, 17/19, 34/36) so its position — and therefore its slide number — is
  unambiguous from context even though nothing is printed on the page itself. The PDF has 60
  pages total, but printed numbers stop at 48 (the last page to carry one). **Pages 49–60 print
  no number at all** — they come after the last numbered page and are presumed backup/appendix
  material. Their labels "Slide 49" through "Slide 60" below are **inferred by continuing the
  page sequence**, not read off the page; this is stated explicitly here and again in the
  "Slide numbers vs PDF pages" section below, per this KB's rule that a null numbering result
  must never be reported as a clean 1:1 mapping.
---

# Lecture 16 — ConvNets for NLP and Tree Recursive Neural Networks: slide-by-slide

Text and figures of
[`cs224n-spr2024-lecture16-CNN-TreeRNN.pdf`](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture16-CNN-TreeRNN.pdf),
transcribed from the deck. Diagrams and plots are described in prose since the KB is
read as text.

Companion pages: [wiki page for this lecture](../../wiki/17-convnets-and-treernns.md) ·
[transcript](../transcripts/17-convnets-and-treernns.md)

## Slide numbers vs PDF pages

Printed numbers run 1–48 and equal the PDF page number throughout that range, except that pages
1, 16, 18 and 35 print no number (their slide numbers are inferred from their position between
numbered neighbors). PDF pages 49–60 (12 pages) print **no number at all** — there is nothing on
the page to read. The "Slide 49" … "Slide 60" labels used below for that tail are constructed
purely by continuing the page count; they are not printed slide numbers, and any citation to
them should be understood as a PDF page position, not a number the instructor wrote.

## Contents

| Slides | Section |
| ------ | ------- |
| 1 | Title |
| 2 | Lecture Plan |
| 3–5 | §1 Course organization updates: mid-quarter survey, final project logistics, GPU/cloud-compute options |
| 6–18 | §2 From RNNs to CNNs: motivation, what a convolution is, 1D convolution for text, padding, multi-channel/multi-filter convolution, max/average pooling, PyTorch `Conv1d` example, stride, local max pooling, k-max pooling, dilation |
| 19–21 | §3 Single-layer CNN for sentence classification (Yoon Kim 2014): setup, equations, pooling/channels/classification |
| 22–23 | A pitfall when fine-tuning word vectors, and the channel-doubling fix |
| 24–29 | §4 CNN potpourri: Kim (2014) architecture picture, results table, comparison caveats, model-comparison toolkit recap, BatchNorm, size-1 convolutions |
| 30–34 | §5 Very deep CNNs for text classification (Conneau et al. 2017): VD-CNN and conv-block architecture, dataset table, results tables |
| 35–37 | §6 TreeRNNs: recursion in human language (Hauser/Chomsky/Fitch excerpt), PP-attachment tree examples, a full Penn Treebank tree |
| 38–48 | Mapping phrases into vector space, constituency parsing as structure+representation learning, recursive vs. recurrent nets, greedy RNN parsing walkthrough (slides 42–47), discussion of the simple TreeRNN's limits |
| 49–59 | §7 Recursive Neural Tensor Networks and sentiment analysis: RNTN formula build-up, Stanford Sentiment Treebank, results and negation experiments (no printed page numbers — see note) |
| 60 | Title slide, reprised (no printed page number) |

---

## Slide 1 — Title

**Natural Language Processing with Deep Learning / CS224N/Ling284.** Below the title, a small
decorative logo: a dark-red angular arch (like a roofline) over three tan/beige semicircular
arches, evoking a bridge or aqueduct. Below that: "Christopher Manning" / "Lecture 16: ConvNets
for NLP and Tree Recursive Neural Networks". No footer page number (title page).

## Slide 2 — Lecture Plan

A numbered agenda, each item with a time estimate:

1. Course organization updates (5 mins)
2. Intro to CNNs (25 mins)
3. Simple CNN for Sentence Classification: Yoon (2014) (10 mins)
4. CNN potpourri (5 mins)
5. Deep CNN for Sentence Classification: Conneau et al. (2017) (10 mins)
6. Tree Recursive Neural Nets, briefly (15 mins)
7. Recursive Neural Tensor Networks and Sentiment Analysis (15 mins)

## Slide 3 — 1. Course Organization Updates

**Mid-quarter survey** — quoted student comments:

- "Fantastic lectures and really interesting content"
- "I enjoy the lectures a lot. Also I like the coding part of the problem sets."
- "I love how the lectures focus on theory and assignments on implementation - I feel learning
  both is hugely beneficial."
- "Maybe an exam to help reinforce concepts from class"
- "There could be more timely responses from course staff"
- "I think that some of the more math-heavy lectures are better explained with a
  whiteboard/pen and paper. The PowerPoints are well made but it can sometimes be hard to
  understand the individual steps in a process."
- "I would love to see more content about more recent models like state space models"

Right, two pie charts from the survey:

- **"Do you feel your questions are adequately addressed on Ed?"** — two slices: Yes 89.3%
  (blue, the large majority), No 10.7% (red, a small wedge).
- **"Are your concerns are adequately addressed in office hours?"** — a many-slice pie
  dominated by Yes 69.0% (blue); the remaining ~31% is split into many small, differently
  colored wedges with partly-truncated labels, including "Sometimes" 11.1% (orange), "N/A"
  1.8%, "sometimes yes, som…" 0.4%, "I haven't been to offi…" 0.4%, "Not applicable" 0.4%,
  "No" 0.7%, and "I haven't gone to offi…" 0.2%, each too small a wedge to read precisely at
  this resolution.

## Slide 4 — Course Organization Updates

**Final Project: The key remaining thing to do**

- Final Project Milestone was due yesterday, Wed May 22!
- Make an effort to get feedback in person from your mentor as well in office hours!
- **Final project poster session: Mon Jun 10, 11:00am–3:00pm: You need to be there***
  - Alumni Center, McCaw Hall and Ford Gardens
  - Groundbreaking research! Prizes! **Food!**

**Invited speakers**

- We had Nathan Lambert in the previous lecture
- Next Tuesday, May 28 is Adina Williams on Safety
- **Attendance is expected for on-campus students**; otherwise: "reaction paragraph"

## Slide 5 — Course Organization Updates

**GPUs: Cloud Compute for projects**

- You're welcome to use **Google Colab**, but it provides limited, inconsistent GPU access
  - We recommend paying \$10/month for Colab Pro, which gives better GPU access
  - We can't reimburse you for that.
- We encourage you to use the **GCP** credits we got for the class and/or API access through
  **Together AI**, if appropriate for your project
- You're also welcome to try Kaggle Notebooks
  - A vanilla Jupyter notebook, not as fancy as Colab, but better GPU access
- Some groups have done well using Modal
  - Some free hours, then need to pay

Right, a reproduced news-article snippet headlined "AI chip shortages continue, but there may be
an end in sight" (News Analysis, May 07, 2024, tagged "CPUs and Processors," "Generative AI,"
"Technology Industry," with a small AI-chip photo), with body text about GPU/memory-chip supply
and demand pressure from generative-AI adoption. Below it, a pie chart titled **"What do you
plan to use for compute for the final project?"** with roughly a dozen slices, the largest being
GCP credits at 33.9% (blue); other legible slices include "GCP credits, Modal" 14.5%, "GCP
credits, Togeth…" 10.7%, Modal 6.2%, "Together API" 3.1%, "Modal, Together API" 2.7%, Colab
0.9%, AWS 0.7%, "Not sure yet" 0.7%, "Direct APIs (e.g., Op…)" 0.2%, "Other GPU" 0.2%, "lab
compute" 0.2%, and "GCP credits, Modal,…" 0.4% — several labels are truncated by the chart's own
layout and cannot be read further.

## Slide 6 — 2. From RNNs to Convolutional Neural Nets

- Recurrent neural nets cannot capture phrases without prefix context
- Often capture too much of last words in final vector

Below, a reproduced RNN chain diagram over the sentence "Monáe walked into the ceremony": each
word has a small input vector (Monáe [0.4, 0.3], walked [2.1, 3.3], into [7, 7], the [4, 4.5],
ceremony [2.3, 3.6]) shown as an upward arrow feeding into a chain of hidden-state vectors
(Monáe's state [1, 3.5] → walked's state [1, 5] → into's state [5.5, 6.1] → the's state
[4.5, 3.8] → ceremony's state [2.5, 3.8]), connected left-to-right by horizontal arrows, with a
final upward arrow out of the last hidden state representing the softmax/output step.

- E.g., softmax for word prediction is usually calculated based on the last step

## Slide 7 — From RNNs to Convolutional Neural Nets

- Main Convolutional Neural Net (CNN/ConvNet) idea:
  - What if we compute vectors for every possible word subsequence of a certain length?
- Example: "tentative deal reached to keep government open" computes vectors for:
  - *tentative deal reached*, *deal reached to*, *reached to keep*, *to keep government*,
    *keep government open* (each shown in teal as a distinct 3-word window)
- Regardless of whether subsequence is grammatical or a natural linguistic constituent
  - Not very linguistically or cognitively plausible
- Then group them afterwards (more soon)

## Slide 8 — What is a convolution anyway?

- 1d discrete convolution generally: $(f * g)[n] = \sum_{m=-M}^{M} f[n-m]g[m]$
- Convolution is classically used to extract features from images
  - Models position-invariant identification
  - Longer version in cs231n!
- 2d example →
- Yellow color and red numbers show filter (=kernel) weights
- Green shows input
- Pink shows output

Right, a reproduced 2D-convolution illustration (credited "From Stanford UFLDL wiki"): a 5×5
binary "Image" grid (values 1,1,1,0,0 / 0,1,1,1,0 / 0,0,1,1,1 / 0,0,1,1,0 / 0,1,1,0,0), with its
top-left 3×3 block highlighted yellow and each cell carrying a small red subscript (×1, ×0, ×1 /
×0, ×1, ×0 / ×1, ×0, ×1) showing the 3×3 kernel weights overlaid on that patch, and the
remaining cells of the grid shown in green. To the right, a 3×3 "Convolved Feature" output grid
with its top-left cell shaded pink and containing the computed value "4" (the dot product of the
highlighted patch and the kernel), the other eight output cells left blank to indicate the
kernel would slide across the rest of the image next.

## Slide 9 — A 1D convolution for text

Left, a word-embedding table (columns are 4-dimensional vectors) for the seven words of
"tentative deal reached to keep government open":

| Word | dim 1 | dim 2 | dim 3 | dim 4 |
| --- | --- | --- | --- | --- |
| tentative | 0.2 | 0.1 | −0.3 | 0.4 |
| deal | 0.5 | 0.2 | −0.3 | −0.1 |
| reached | −0.1 | −0.3 | −0.2 | 0.4 |
| to | 0.3 | −0.3 | 0.1 | 0.1 |
| keep | 0.2 | −0.3 | 0.4 | 0.2 |
| government | 0.1 | 0.2 | −0.1 | −0.1 |
| open | −0.4 | −0.4 | 0.2 | 0.3 |

A bold box outlines a window of rows at the top of the table — the first filter application,
over "tentative deal reached." (The outline is drawn loosely: it reads as a clean rectangle
across all four columns for those three rows, and then continues down over the "to" row with a
ragged right edge, so it is probably two overlapping window boxes rather than one four-row box.
Either way the filter width is 3, stated in the caption and fixed by the 3-row filter matrix.)
Below: "Apply a **filter** (or **kernel**) of size 3", with the 3×4
filter matrix shown: row 1 = [3, 1, 2, −3], row 2 = [−1, 2, 1, −3], row 3 = [1, 1, −1, 1].

Right, a small results table whose row labels are single-letter shorthand for the sliding
3-word windows (t = tentative, d = deal, r = reached, t = to, k = keep, g = government,
o = open, so "t,d,r" means the window "tentative deal reached," "d,r,t" means "deal reached to,"
and so on): t,d,r → −1.0; d,r,t → −0.5; r,t,k → −3.6; t,k,g → −0.2; k,g,o → 0.3.

Next to that, a second small table with two numbers per row for the same five windows — 0.0 /
0.50 (t,d,r), 0.5 / 0.38 (d,r,t), −2.6 / 0.93 (r,t,k), 0.8 / 0.31 (t,k,g), 1.3 / 0.21 (k,g,o) —
and below it, "+ bias → non-linearity". **[The slide does not label these two columns
explicitly; given the "+ bias → non-linearity" caption directly beneath them, they most likely
show a raw filter value and the value after adding a bias and applying a non-linearity, but this
reading is inferred from the layout rather than stated on the slide.]**

## Slide 10 — 1D convolution for text with padding

Same embedding table as slide 9, now with a zero-padding row (∅: 0.0, 0.0, 0.0, 0.0) added both
above "tentative" and below "open". Same 3×4 filter as slide 9. The results table on the right
now has seven rows, one per padded 3-word window: ∅,t,d → −0.6; t,d,r → −1.0; d,r,t → −0.5;
r,t,k → −3.6; t,k,g → −0.2; k,g,o → 0.3; g,o,∅ → −0.5.

- Could also use (zero) padding = 2
- Also called "wide convolution"

## Slide 11 — 3 channel 1D convolution with padding = 1 and 3 filters

Same padded embedding table as slide 10. "Apply 3 **filters** of size 3", with three 3×4
filter matrices shown side by side:

- Filter A: [3, 1, 2, −3] / [−1, 2, 1, −3] / [1, 1, −1, 1] (same as slides 9–10)
- Filter B: [1, 0, 0, 1] / [1, 0, −1, −1] / [0, 1, 0, 1]
- Filter C: [1, −1, 2, −1] / [1, 0, −1, 3] / [0, 2, 2, 1]

The results table now has three value columns, one per filter, for the same seven padded
windows:

| Window | Filter A | Filter B | Filter C |
| --- | --- | --- | --- |
| ∅,t,d | −0.6 | 0.2 | 1.4 |
| t,d,r | −1.0 | 1.6 | −1.0 |
| d,r,t | −0.5 | −0.1 | 0.8 |
| r,t,k | −3.6 | 0.3 | 0.3 |
| t,k,g | −0.2 | 0.1 | 1.2 |
| k,g,o | 0.3 | 0.6 | 0.9 |
| g,o,∅ | −0.5 | −0.9 | 0.1 |

## Slide 12 — conv1d, padded with max pooling over time

Same padded embedding table, same 3-filter results table as slide 11. A new "max p" row is
added below the table, giving the column-wise (over-time) maximum for each filter: Filter A →
0.3, Filter B → 1.6, Filter C → 1.4 — i.e. max pooling collapses each filter's seven window
outputs down to a single scalar.

## Slide 13 — conv1d, padded with ave pooling over time

Same tables as slide 12, but the summary row is now "ave p" (average instead of max): Filter A
→ −0.87, Filter B → 0.26, Filter C → 0.53.

## Slide 14 — In PyTorch

A code listing:

```python
batch_size = 16
word_embed_size = 4
seq_len = 7
input = torch.randn(batch_size, word_embed_size, seq_len)
conv1 = Conv1d(in_channels=word_embed_size, out_channels=3,
               kernel_size=3)  # can add: padding=1
hidden1 = conv1(input)
hidden2 = torch.max(hidden1, dim=2)  # max pool
```

## Slide 15 — Other (maybe less useful) notions: stride = 2

Same padded embedding table and same three filters (A, B, C) as slides 11–13. Because the
convolution now steps by 2 instead of 1, the results table has only four rows instead of seven
— every other window is skipped: ∅,t,d → (−0.6, 0.2, 1.4); d,r,t → (−0.5, −0.1, 0.8); t,k,g →
(−0.2, 0.1, 1.2); g,o,∅ → (−0.5, −0.9, 0.1).

## Slide 16 — Local max pool, stride = 2

*(No printed page number; inferred from position between slides 15 and 17.)*

Same padded embedding table and the full seven-row, three-filter results table from slide 11,
now with an eighth row appended, ∅ → (−Inf, −Inf, −Inf), representing padding added before
pooling. A box groups the first two rows (∅,t,d and t,d,r) together, illustrating a local
max-pooling window of size 2 with stride 2. Below, a second table shows the pooled results, one
row per pair of adjacent windows, labelled with the concatenated word-window pairs:

| Pooled window | Filter A | Filter B | Filter C |
| --- | --- | --- | --- |
| ∅,t,d,r | −0.6 | 1.6 | 1.4 |
| d,r,t,k | −0.5 | 0.3 | 0.8 |
| t,k,g,o | 0.3 | 0.6 | 1.2 |
| g,o,∅,∅ | −0.5 | −0.9 | 0.1 |

Each pooled value is the max of the corresponding pair of rows above it (e.g. max(−0.6, −1.0) =
−0.6 for Filter A on the first pooled row).

## Slide 17 — conv1d, k-max pooling over time, k = 2

Same padded embedding table, same seven-row three-filter results table as slide 11, plus a
"2-max p" block giving, for each filter, its top-2 values across all seven time steps (not
necessarily adjacent): Filter A → 0.3, −0.2; Filter B → 1.6, 0.6; Filter C → 1.4, 1.2 — i.e. the
two largest activations per filter are kept instead of only the single maximum.

## Slide 18 — Other somewhat useful notions: dilation = 2

*(No printed page number; inferred from position between slides 17 and 19.)*

Same padded embedding table and seven-row three-filter results table as slide 11. Boxes now
highlight every other row — ∅,t,d, d,r,t, and t,k,g (rows 1, 3, 5) — illustrating a dilated
filter that skips one window at a time. Below, a small table lists three dilated groupings by
row index, "1,3,5", "2,4,6", and "3,5,7", with values filled in only for the first group (0.3,
0.0) and the other two rows left blank on the slide. Below that, two further 3×3 filter
matrices are shown — [2, 3, 1] / [1, −1, −1] / [3, 1, 0] and [1, 3, 1] / [1, −1, −1] / [3, 1, −1]
— presumably the dilated filters used to produce the "1,3,5" grouping's values, though the slide
does not label them explicitly and the incomplete "2,4,6" / "3,5,7" rows suggest this table may
be a partially-built illustration rather than a fully worked example.

## Slide 19 — 3. Single Layer CNN for Sentence Classification

- Yoon Kim (2014): *Convolutional Neural Networks for Sentence Classification*. EMNLP 2014.
  https://arxiv.org/pdf/1408.5882.pdf
- Goal: Sentence classification:
  - Mainly positive or negative sentiment of a sentence
  - Other tasks like:
    - Subjective or objective language sentence
    - Question classification: about person, location, number, …

## Slide 20 — Single Layer CNN for Sentence Classification

- A simple use of one convolutional layer and **max pooling**
- Word vectors: $\mathbf{x}_i \in \mathbb{R}^k$
- Sentence: $\mathbf{x}_{1:n} = \mathbf{x}_1 \oplus x_2 \oplus \cdots \oplus \mathbf{x}_n$
  (vectors are concatenated)
- Filter applied to concatenation of words in range: $\mathbf{x}_{i:i+j}$ (symmetric more
  common)
- Convolutional filter $\mathbf{w} \in \mathbb{R}^{hk}$ applied to all possible windows
  $\{\mathbf{x}_{1:h}, \mathbf{x}_{2:h+1}, \ldots, \mathbf{x}_{n-h+1:n}\}$
  - Filter is done as a long vector over window of $h$ words
  - Filter could be of size $h$ = 2, 3, or 4 words
- To compute feature (one *channel*) for CNN layer:
  $$c_i = f(\mathbf{w}^T \mathbf{x}_{i:i+h-1} + b)$$
- Result is a feature map: $\mathbf{c} = [c_1, c_2, \ldots, c_{n-h+1}] \in \mathbb{R}^{n-h+1}$

Below, two small reproductions of the same RNN-chain-style vector diagram as slide 6, now over
"the country of my birth" (the [0.4, 0.3], country [2.1, 3.3], of [7, 7], my [4, 4.5],
birth [2.3, 3.6]):

- **Left diagram**: arrows from "the", "country" and "of" converge on a single output vector
  [1.1] above them (one filter window, $h=3$, applied to "the country of"); the arrow from
  "country" is highlighted orange, distinguishing it from the two dark-red arrows from "the"
  and "of".
- **Right diagram**: the same five words now feed three overlapping windows, each producing its
  own scalar output — "the country of" → 1.1, "country of my" → 3.5, "of my birth" → 2.4 — shown
  as three separate small stacks of convergent arrows, illustrating the full feature map
  $\mathbf{c} = [1.1, 3.5, 2.4]$ for this five-word sentence with a size-3 filter.

## Slide 21 — Pooling, channels, and classification

- Pooling: max-over-time pooling layer
- Idea: capture most important activation (maximum over time)
- Use multiple filter weights **w** (i.e., multiple channels)
- From feature map $\mathbf{c} = [c_1, c_2, \ldots, c_{n-h+1}] \in \mathbb{R}^{n-h+1}$
- Pooled single number: $\hat{c} = \max\{\mathbf{c}\}$
  - Because of max pooling $\hat{c} = \max\{\mathbf{c}\}$, length of **c** can be variable
- One convolution layer, followed by one max-pooling
  - To obtain final feature vector: (assuming $m$ filters **w**) $\mathbf{z} = [\hat{c}_1,
    \ldots, \hat{c}_m]$
  - Used 100 feature maps each of sizes 3, 4, 5
- Simple final softmax layer: $y = softmax\big(W^{(S)}z + b\big)$

## Slide 22 — A pitfall when fine-tuning word vectors

- **Setting:** We are training a model for movie review sentiment building on word vectors
- In the training data we have "tedious", "dull"; in the testing data we have "plodding"
- The pre-trained word vectors have all three similar:
- **Question: What happens when we update the word vectors?**
- **Answer:** Words in the training data move around; other words stay where they were

Below, two reproduced 2D scatter/decision-region plots over an unlabeled word-vector space
(axes and scale not printed on the slide), each split into a green region and a red decision
region by a boundary line, with scattered word-vector points as dots:

- **Left plot ("before"):** the decision boundary is roughly a vertical line a bit left of
  center, with green filling the majority of the plot (left and most of the area) and a red
  wedge on the right. A cluster of three red dots sits in the middle-right of the plot, with
  labelled arrows from "tedious", "dull", and "plodding" all pointing into that same small
  cluster — i.e. all three words start out close together, correctly on the "red" side.
- **Right plot ("after fine-tuning"):** the boundary has rotated to run diagonally from
  upper-left to lower-right, so red now occupies the upper-right and green the lower-left. Two
  of the three dots — labelled "tedious" and "dull" — have moved up into the red region (they
  were in the training data, so they moved during fine-tuning). The third dot, labelled
  "plodding" (only in the test data, so its vector never moved), is left behind in what is now
  the green region, i.e. now on the wrong side of the boundary. A pink callout box to the right
  reads "**This can be bad!**"

## Slide 23 — A solution: Channel doubling multi-channel input idea

- Initialize model with pre-trained word vectors (e.g., word2vec or Glove)
- Start with two copies
- Backprop into only one set, keep other "static"
  - Fine-tuning should be useful for improving word vectors for task
  - But there is a problem that words in pre-training (and maybe runtime data) but not in
    training data will not move. So, it also makes sense to leave all word vectors where they
    are and to only update the parameters above the word vectors
  - Having two copies is an attempt to get the best of both worlds
- Both channel sets are added to $c_i$ before max-pooling

## Slide 24 — Kim (2014)

"From: Zhang and Wallace (2015) *A Sensitivity Analysis of (and Practitioners' Guide to)
Convolutional Neural Networks for Sentence Classification*. https://arxiv.org/pdf/1510.03820.pdf
(follow on paper, not famous, but a nice picture)"

Right, a reproduced architecture diagram (from Zhang & Wallace 2015) for the example sentence
"I like this movie very much !" (7 words, embedding dimension $d=5$, so a 7×5 sentence matrix
shown on the left as a grid): arrows fan out from the sentence matrix into six colored blocks
labelled "3 region sizes: (2,3,4), 2 filters for each region size, totally 6 filters" — dark
red, red, dark green, light green, olive, and yellow blocks, two per region size (2, 3, and 4).
Each colored block's convolution output feeds into a same-colored, shorter "1-max pooling"
vector. The six 1-max-pooled vectors (one per filter, each a single number, shown as small
colored segments) are then stacked/concatenated into one combined multi-colored column, labelled
"6 univariate vectors concatenated together to form a single feature vector," which feeds into a
final blue/purple two-unit output box labelled "2 classes" via a "softmax function
(regularization in this layer)" step at the top of the diagram.

## Slide 25 — Experiments on text classification

A reproduced results table comparing CNN variants (from Kim 2014) against a long list of prior
methods, columns **Model | MR | SST-1 | SST-2 | Subj | TREC | CR | MPQA** (accuracy percentages;
"—" marks cells the original paper does not report):

| Model | MR | SST-1 | SST-2 | Subj | TREC | CR | MPQA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CNN-rand | 76.1 | 45.0 | 82.7 | 89.6 | 91.2 | 79.8 | 83.4 |
| CNN-static | 81.0 | 45.5 | 86.8 | 93.0 | 92.8 | 84.7 | **89.6** |
| CNN-non-static | **81.5** | 48.0 | 87.2 | 93.4 | 93.6 | 84.3 | 89.5 |
| CNN-multichannel | 81.1 | 47.4 | **88.1** | 93.2 | 92.2 | **85.0** | 89.4 |
| RAE (Socher et al., 2011) | 77.7 | 43.2 | 82.4 | — | — | — | 86.4 |
| MV-RNN (Socher et al., 2012) | 79.0 | 44.4 | 82.9 | — | — | — | — |
| RNTN (Socher et al., 2013) | — | 45.7 | 85.4 | — | — | — | — |
| DCNN (Kalchbrenner et al., 2014) | — | 48.5 | 86.8 | — | 93.0 | — | — |
| Paragraph-Vec (Le and Mikolov, 2014) | — | **48.7** | 87.8 | — | — | — | — |
| CCAE (Hermann and Blunsom, 2013) | 77.8 | — | — | — | — | — | 87.2 |
| Sent-Parser (Dong et al., 2014) | 79.5 | — | — | — | — | — | 86.3 |
| NBSVM (Wang and Manning, 2012) | 79.4 | — | — | 93.2 | — | 81.8 | 86.3 |
| MNB (Wang and Manning, 2012) | 79.0 | — | — | **93.6** | — | 80.0 | 86.3 |
| G-Dropout (Wang and Manning, 2013) | 79.0 | — | — | 93.4 | — | 82.1 | 86.1 |
| F-Dropout (Wang and Manning, 2013) | 79.1 | — | — | **93.6** | — | 81.9 | 86.3 |
| Tree-CRF (Nakagawa et al., 2010) | 77.3 | — | — | — | — | 81.4 | 86.1 |
| CRF-PR (Yang and Cardie, 2014) | — | — | — | — | — | 82.7 | — |
| SVM$_S$ (Silva et al., 2011) | — | — | — | — | **95.0** | — | — |

## Slide 26 — Be careful of fine-points in comparisons!

- Kim (2014) uses dropout, reporting that it gives 2–4% accuracy improvement!
- But several compared-to systems came earlier and hence didn't use dropout (from 2012/2014)
  and would probably gain equally from it
- Still seen as remarkable results from a simple architecture!
- Differences from window architecture we described in an early lecture:
  - Many filters and pooling

## Slide 27 — 4. Model comparison: Our growing toolkit

- **Bag of Vectors**: Surprisingly good baseline for simple classification problems
  - Especially if followed by a few ReLU layers! (See paper: Deep Averaging Networks)
- **Window Model**: Good for single word classification for problems that do not need wide
  context. E.g., POS, NER
- **CNNs:** good for classification, need zero padding for shorter phrases, somewhat
  implausible/hard to interpret, **easy to parallelize on GPUs;** efficient and versatile
- **Recurrent Neural Networks**: Cognitively plausible (reading from left to right), not best
  for classification (if just use last state), much slower than CNNs, good for sequence
  tagging and classification, good for language models, better with attention
- **Transformers:** Great for language models, great for sentence calculations; in general,
  still the best thing since sliced bread for all NLP problems
  - "Vision Transformers" are taking over in vision but some papers argue that CNNs and
    transformers have complementary advantages, and you can usefully use both

## Slide 28 — Batch Normalization (BatchNorm)

"[Ioffe and Szegedy. 2015. Batch normalization: Accelerating deep network training by reducing
internal covariate shift. arXiv:1502.03167.]"

- Often used in CNNs
- Transform the convolution output of **a batch** by scaling the activations to have zero mean
  and unit variance
  - Again, like the familiar Z-transform of statistics
  - Related to LayerNorm, which is standard in Transformers, but crucially different:
    - LayerNorm calculates statistics across all feature dimensions for each instance
      independently
    - BatchNorm normalizes across all elements and items in a batch for each feature
      independently
- Use of BatchNorm also makes models **much** less sensitive to parameter initialization, since
  outputs are automatically rescaled
  - It also tends to make tuning of learning rates simpler
- PyTorch: `nn.BatchNorm1d`

## Slide 29 — Size 1 Convolutions

"[Lin, Chen, and Yan. 2013. Network in network. arXiv:1312.4400.]"

- **Does this concept make sense?!? Yes.**
- Size 1 convolutions ("1x1"), a.k.a. Network-in-network (NiN) connections, are convolutional
  kernels with kernel_size=1
- A size 1 convolution gives you a fully connected linear layer across channels!
- It can be used to map from many channels to fewer channels
- Size 1 convolutions add additional neural network layers with very few additional parameters
  - Unlike Fully Connected (FC) layer across data item which adds **tons** of parameters
  - This is similar to the per-position feed-forward layers in transformers

## Slide 30 — 5. Very Deep Convolutional Networks for Text Classification

- Conneau, Schwenk, Lecun, Barrault. EACL 2017.
- Starting point: sequence models (LSTMs) had been very dominant in NLP
  - Also CNNs, Attention, etc., but all the models were basically not very deep – not like the
    deep models in Vision
- What happens when we build a vision-like system for NLP?
- Model works up from the character level
  - Desire for "NLP from scratch" [raw signal]

## Slide 31 — VD-CNN architecture

"The system very much looks like a vision system in its design, similar to VGGnet or ResNet"

"It looks unlike then typical Deep Learning NLP systems"

- It looks a bit more like a Transformer?

"Local pooling at each stage halves temporal resolution and doubles number of features"

"s = 1024 chars; 16d embed"

"Result is constant size, since text is truncated or padded"

Right, a reproduced VD-CNN architecture diagram, read bottom to top: "Text" → "Lookup table, 16"
(input: 1×s) → "3, Temp Conv, 64" (output: 16×s) → a pair of "Convolutional Block, 3, 64" boxes
(output: 64×s) → "pool/2" → a pair of "Convolutional Block, 3, 128" boxes (output: 128×s/2) →
"pool/2" → a pair of "Convolutional Block, 3, 256" boxes (output: 256×s/4) → "pool/2" → a pair
of "Convolutional Block, 3, 512" boxes (output: 512×s/8) → "k-max pooling, k=8" (output: 512×k)
→ "fc(4096, 2048), ReLU" → "fc(2048, 2048), ReLU" → "fc(2048, nClasses)". Each of the four
stages (64, 128, 256, 512 channels) is drawn in its own color, and each convolutional-block pair
has a curved "optional shortcut" arrow drawn beside it, bypassing the pair of blocks — a
ResNet-style residual/skip connection at each stage.

## Slide 32 — Convolutional block in VD-CNN

- Each convolutional block is two convolutional layers, each followed by batch norm and a ReLU
  nonlinearity
- Convolutions of size 3
- Pad to preserve (or halve when local pooling) dimension

Right, a reproduced block diagram, bottom to top: "3, Temp Conv, 256" → "Temporal Batch Norm" →
"ReLU" → "3, Temp Conv, 256" → "Temporal Batch Norm" → "ReLU", each stage connected to the next
by an upward arrow — i.e. two [conv → batch-norm → ReLU] sub-layers stacked to form one block.

## Slide 33 — Experiments

- Use large text classification datasets
  - Much bigger than the small datasets used in the Yoon Kim (2014) paper

A reproduced dataset table, columns **Data set | #Train | #Test | #Classes | Classification
Task**:

| Data set | #Train | #Test | #Classes | Classification Task |
| --- | --- | --- | --- | --- |
| AG's news | 120k | 7.6k | 4 | English news categorization |
| Sogou news | 450k | 60k | 5 | Chinese news categorization |
| DBPedia | 560k | 70k | 14 | Ontology classification |
| Yelp Review Polarity | 560k | 38k | 2 | Sentiment analysis |
| Yelp Review Full | 650k | 50k | 5 | Sentiment analysis |
| Yahoo! Answers | 1,400k | 60k | 10 | Topic classification |
| Amazon Review Full | 3,000k | 650k | 5 | Sentiment analysis |
| Amazon Review Polarity | 3,600k | 400k | 2 | Sentiment analysis |

## Slide 34 — Experiments

Two reproduced results tables. **Table 4** ("Best published results from previous work"),
columns are per-corpus (AG, Sogou, DBP., Yelp P., Yelp F., Yah. A., Amz. F., Amz. P.), with rows
for Method, Author, Error, and a comparison row from Yang et al.:

| | AG | Sogou | DBP. | Yelp P. | Yelp F. | Yah. A. | Amz. F. | Amz. P. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | n-TFIDF | n-TFIDF | n-TFIDF | ngrams | Conv | Conv+RNN | Conv | Conv |
| Author | [Zhang] | [Zhang] | [Zhang] | [Zhang] | [Zhang] | [Xiao] | [Zhang] | [Zhang] |
| Error | 7.64 | 2.81 | 1.31 | 4.36 | 37.95* | 28.26 | 40.43* | 4.93* |
| [Yang] | – | – | – | – | – | 24.2 | 36.4 | – |

Caption: "Table 4: Best published results from previous work. Zhang et al. (2015) best results
use a Thesaurus data augmentation technique (marked with an *). Yang et al. (2016)'s
hierarchical methods is particularly [text cut off at bottom of slide]."

**Table 5** ("Testing error of our models on the 8 data sets. No data preprocessing or
augmentation is used."), columns **Depth | Pooling | AG | Sogou | DBP. | Yelp P. | Yelp F. |
Yah. A. | Amz. F. | Amz. P.**, comparing three depths (9, 17, 29 layers) crossed with three
pooling strategies (Convolution/no pooling, KMaxPooling, MaxPooling):

| Depth | Pooling | AG | Sogou | DBP. | Yelp P. | Yelp F. | Yah. A. | Amz. F. | Amz. P. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | Convolution | 10.17 | 4.22 | 1.64 | 5.01 | 37.63 | 28.10 | 38.52 | 4.94 |
| 9 | KMaxPooling | 9.83 | 3.58 | 1.56 | 5.27 | 38.04 | 28.24 | 39.19 | 5.69 |
| 9 | MaxPooling | 9.17 | 3.70 | 1.35 | 4.88 | 36.73 | 27.60 | 37.95 | 4.70 |
| 17 | Convolution | 9.29 | 3.94 | 1.42 | 4.96 | 36.10 | 27.35 | 37.50 | 4.53 |
| 17 | KMaxPooling | 9.39 | 3.51 | 1.61 | 5.05 | 37.41 | 28.25 | 38.81 | 5.43 |
| 17 | MaxPooling | 8.88 | 3.54 | 1.40 | 4.50 | 36.07 | 27.51 | 37.39 | 4.41 |
| 29 | Convolution | 9.36 | 3.61 | 1.36 | 4.35 | **35.28** | 27.17 | 37.58 | **4.28** |
| 29 | KMaxPooling | **8.67** | **3.18** | 1.41 | 4.63 | 37.00 | 27.16 | 38.39 | 4.94 |
| 29 | MaxPooling | 8.73 | 3.36 | **1.29** | **4.28** | 35.74 | **26.57** | **37.00** | 4.31 |

## Slide 35 — 6. TreeRNNs: Recursion in human language

*(No printed page number; inferred from position between slides 34 and 36.)*

Left, the section heading. Right, a reproduced magazine-style page from *Science*'s "Compass ·
Review" section, headed "REVIEW: NEUROSCIENCE," titled "**The Faculty of Language: What Is It,
Who Has It, and How Did It Evolve?**" by Marc D. Hauser, Noam Chomsky, and W. Tecumseh Fitch. A
boxed abstract reads: "We argue that an understanding of the faculty of language requires
substantial interdisciplinary cooperation. We suggest how current developments in linguistics
can be profitably wedded to work in evolutionary biology, anthropology, psychology, and
neuroscience. We submit that a distinction should be made between the faculty of language in
the broad sense (FLB) and in the narrow sense (FLN). FLB includes a sensory-motor system, a
conceptual-intentional system, and the computational mechanisms for recursion, providing the
capacity to generate an infinite range of expressions from a finite set of elements. We
hypothesize that FLN only includes recursion and is the only uniquely human component of the
faculty of language. We further argue that FLN may have evolved for reasons other than
language, hence comparative studies might look for evidence of such computations outside of the
domain of communication (for example, number, navigation, and social relations)." Below the
abstract, the article's opening paragraph begins: "If a martian graced our planet, it would be
struck by one remarkable similarity among Earth's living creatures and a key difference.
Concerning similarity, it would note that all..." (cut off at the bottom of the slide). To the
right of the text, a vertical column of six small line-drawing illustrations of a human hand
making a sign, a shouting chimpanzee, a fish, a dolphin/orca, a grasshopper, a small bird, and a
howling wolf — a set of species used elsewhere in the article to compare communication and
cognition across species.

## Slide 36 — Are languages recursive?

- Cognitively somewhat debatable (need to head to infinity)
- But: recursive structure is natural/right for describing language
  - *[The person standing next to [the man from [the company that purchased [the firm that you
    used to work at]]]]*
  - noun phrase containing a noun phrase containing a noun phrase
- It's a very powerful prior for language structure

Below, two reproduced constituency parse trees illustrating classic PP-attachment ambiguity,
both starting "He eats spaghetti with...":

- **Left tree** ("with a spoon" = instrument reading): S → NP(PRP "He") + VP; VP has three
  children — VBZ "eats", NP → NNS "spaghetti", and PP → IN "with" + NP → DT "a" + NN "spoon".
  The PP attaches directly under VP, i.e. the spoon is the instrument of eating.
- **Right tree** ("with meat" = modifier reading): S → NP(PRP "He") + VP; VP has two children —
  VBZ "eats" and a single NP, which itself splits into NP → NNS "spaghetti" and PP → IN "with" +
  NP → NN "meat". Here the PP attaches inside the object NP, i.e. "with meat" describes the
  spaghetti rather than the manner of eating.

## Slide 37 — Penn Treebank tree

A reproduced full syntactic parse tree, labelled "Penn Treebank tree," for the sentence
"Analysts said [that] Mr. Stronach wants to resume a more influential role in running the
company," including Penn Treebank empty-category traces: top node S, with NP-SBJ (NNS
"Analysts") + VP (VBD "said" + SBAR); SBAR → -NONE- "0" + S; that inner S → NP-SBJ-1 (NNP NNP
"Mr. Stronach") + VP (VBZ "wants" + S); that S → NP-SBJ (-NONE- "*-1") + VP (TO "to" + VP (VB
"resume" + NP)); the object NP splits into NP (DT "a" + ADJP (RBR "more" + JJ "influential") +
NN "role") and PP-LOC (IN "in" + S-NOM); S-NOM → NP-SBJ (-NONE- "*") + VP (VBG "running" + NP
(DT "the" + NN "company")). The tree is wide and shallow-branching at the top, growing
increasingly deep toward the right as each embedded clause introduces another layer of
subordination — a visual example of the recursive embedding described on slide 36.

## Slide 38 — How should we map phrases into a vector space?

"Socher, Manning, and Ng. ICML, 2011"

Boxed text: "Use principle of compositionality" / "The meaning (vector) of a phrase or sentence
is determined by (1) the meanings of its words and (2) the rules that combine them."

Below-left, a reproduced binary compositional tree over "the country of my birth," building
vectors bottom-up: leaves the=[0.4, 0.3], country=[2.1, 3.3], of=[7, 7], my=[4, 4.5],
birth=[2.3, 3.6]; "the"+"country" compose to [1, 3.5]; "my"+"birth" compose to [2.5, 3.8]; "of"
+[2.5, 3.8] compose to [5.5, 6.1]; finally [1, 3.5]+[5.5, 6.1] compose to the root vector
[1, 5].

Right, a 2D scatter plot with axes $x_1$ (0 to 10) and $x_2$ (1 to 5), showing six labelled
points (single series, all plotted as red ✕ markers, no color-coded groups): "the country of my
birth" at roughly (1, 5); "the place where I was born" at roughly (1.3, 4.5); "Germany" at
roughly (1, 3); "France" at roughly (1.7, 2.8); "Monday" at roughly (10, 2); "Tuesday" at
roughly (10, 1.6). A purple curved arrow runs from the root of the compositional tree (the [1,5]
vector) up to the "the country of my birth" point on the scatter plot, illustrating that the
phrase's composed vector is meant to land near its semantic neighbors (the paraphrase "the place
where I was born" sits closest to it, while unrelated words "Monday"/"Tuesday" sit far away on
the right).

## Slide 39 — Constituency Sentence Parsing: What we want

A reproduced tree diagram over "The cat sat on the mat," with six leaf word vectors (each
2-dimensional): The=[9,1], cat=[5,3], sat=[7,1], on=[8,5], the=[9,1], mat=[4,3]. Above the
leaves, unfilled gray circles mark the internal nodes to be computed, labelled by their
constituent type only (no vectors filled in yet): an NP node over "The cat," an S node at the
root joining that NP to a VP; the VP joins "sat" to a PP; the PP joins "on" to a second NP over
"the mat." This slide shows the target tree *shape* the model should produce, before any
composed vectors are filled in (contrast with slide 40).

## Slide 40 — Learn Structure and Representation

"Models in this section can jointly learn parse trees and compositional vector representations"

The same tree and leaf vectors as slide 39 ("The cat sat on the mat"), now with every internal
node's composed vector filled in: NP ("The cat") = [5,2]; PP ("on the mat") = [8,3]; VP ("sat on
the mat") = [7,3]; S (root) = [5,4]; and the inner NP ("the mat") = [3,3]. This illustrates the
finished version of the same structure left blank on slide 39 — both the tree structure and the
vector at each node are learned jointly.

## Slide 41 — Recursive vs. recurrent neural networks

- Recursive neural nets provide representations for linguistic phrases
- But they require a tree structure
- Recurrent neural nets cannot capture phrases without prefix context
- They often capture too much of last words in "phrase" vector

Right, two reproduced diagrams over "the country of my birth," contrasting the two
architectures:

- **Top (tree):** the same compositional binary tree as slide 38 — the=[0.4,0.3] +
  country=[2.1,3.3] → [1,3.5]; my=[4,4.5] + birth=[2.3,3.6] → [2.5,3.8]; of=[7,7] +
  [2.5,3.8] → [5.5,6.1]; [1,3.5] + [5.5,6.1] → root [1,5]. Every internal node's vector comes
  only from its two children, regardless of linear word order.
- **Bottom (chain):** the same left-to-right RNN chain layout as slide 6, now over "the country
  of my birth": the[0.4,0.3] → state [1,3.5]; country[2.1,3.3] → state [1,5]; of[7,7] → state
  [5.5,6.1]; my[4,4.5] → state [4.5,3.8]; birth[2.3,3.6] → state [2.5,3.8], each hidden state
  feeding into the next left-to-right. The final state depends on the entire prefix in a fixed
  left-to-right order rather than on tree structure.

## Slide 42 — Recursive Neural Networks for Structure Prediction

"Inputs: two candidate children's representations"

"Outputs:"

1. The semantic representation if the two nodes are merged.
2. Score of how plausible the new node would be.

Below, two linked diagrams over a fragment of "… on the mat": on the right, a partial tree
shows leaves on=[8,5], the=[9,1], mat=[4,3]; "the" and "mat" have already merged into a gray
circle node [3,3], and that node is shown as a *candidate* to merge with "on" into a higher gray
circle node [8,3] (not yet confirmed — outlined but unfilled, like the un-scored nodes on slide
39). On the left, a standalone "Neural Network" box takes the two candidate children's vectors
[8,5] and [3,3] as input and outputs the parent vector [8,3] together with a score, "**1.3**"
(in blue), evaluating how plausible that merge is. Two curved arrows (purple and blue) trace
from the neural network's two input vectors back down to their source positions in the tree (the
"on" leaf and the "the, mat" node), tying the abstract scoring computation to the concrete tree
fragment it is being applied to.

## Slide 43 — Simple Tree Recursive Neural Network Definition

Left, the same "Neural Network" diagram as slide 42: inputs $c_1=[8,5]$ and $c_2=[3,3]$ feed a
neural network box, producing "score = **1.3**" and "= parent" pointing at the output vector
$[8,3]$.

Right, the underlying equations:

$$\text{score} = U^T p$$
$$p = \tanh\!\Big(W\begin{bmatrix} c_1 \\ c_2 \end{bmatrix} + b\Big)$$

"**Same** $W$ parameters at all nodes of the tree"

Bottom right, a small thumbnail reproduction of the completed "The cat sat on the mat" tree
(the same structure built up across slides 39–40 and completed on slide 47).

## Slide 44 — Parsing a sentence with an RNN (greedily)

A reproduced diagram over the six leaf words of "The cat sat on the mat" (The=[9,1], cat=[5,3],
sat=[7,1], on=[8,5], the=[9,1], mat=[4,3]): five separate "Neural Network" boxes, one per
adjacent word pair, each taking two neighboring leaf vectors as input and producing a candidate
parent vector (shown as an unfilled gray circle above it) with a score printed above that:

- (The, cat) → [5,2], score **3.1** (green)
- (cat, sat) → [0,1], score **0.3** (red)
- (sat, on) → [2,0], score **0.1** (red)
- (on, the) → [1,0], score **0.4** (red)
- (the, mat) → [3,3], score **2.3** (green)

The two green scores (3.1 and 2.3) are the highest, so greedy parsing will merge one of these
pairs first; no title bullets appear on this slide beyond the heading.

## Slide 45 — Parsing a sentence

The (The, cat) merge from slide 44 has been carried out, producing node [5,2]. The diagram now
shows the candidate pairs re-scored over the resulting five-unit sequence ([The cat], sat, on,
the, mat):

- ([The cat], sat) → [2,1], score **1.1** (red)
- (sat, on) → [2,0], score **0.1** (red)
- (on, the) → [1,0], score **0.4** (red)
- (the, mat) → [3,3], score **2.3** (green) — unchanged from slide 44, still the highest score,
  so this pair merges next.

## Slide 46 — Parsing a sentence

The (the, mat) merge is now carried out, producing node [3,3] (drawn without a score above it,
since it is already merged rather than a candidate). The sequence is now ([The cat], sat, on,
[the mat]). Remaining candidate scores shown:

- ([The cat], sat) → [2,1], score **1.1** (red, unchanged)
- (sat, on) → [2,0], score **0.1** (red, unchanged)
- (on, [the mat]) → [8,3], score **3.6** (green) — the new highest-scoring candidate, so "on"
  merges with the "the mat" node next.

## Slide 47 — Parsing a sentence

The completed parse tree for "The cat sat on the mat," built by the greedy process shown on
slides 44–46: leaves The=[9,1], cat=[5,3], sat=[7,1], on=[8,5], the=[9,1], mat=[4,3]; "The"+"cat"
→ [5,2]; "the"+"mat" → [3,3]; "on"+[3,3] → [8,3]; "sat"+[8,3] → [7,3]; finally [5,2]+[7,3] →
root [5,4]. This mirrors the same tree shape and node labels (NP/VP/PP/NP) shown unfilled on
slide 39 and filled on slide 40.

- The score of a tree is computed by the sum of the parsing decision scores at each node:
  $$s(x,y) = \sum_{n \in nodes(y)} s_n$$
- $x$ is sentence; $y$ is parse tree

Bottom right, a small recap diagram: inputs $[8,5]$ and $[3,3]$ feed a box now labelled "**RNN**"
(rather than "Neural Network"), producing output $[8,3]$ with score **1.3** (green) — the same
example computation as slides 42–43, relabeled to the more general "RNN" term used going
forward.

## Slide 48 — Discussion: Simple TreeRNN

- We got some decent results with a single layer TreeRNN like this!
  - [Socher, Manning, and Ng. ICML, 2011] got a best paper award!
- A single weight matrix TreeRNN could capture some things but not more complex, higher order
  composition and parsing long sentences
- There is no real interaction between the input words
- And the composition function is the same for all syntactic categories, punctuation, etc.

Bottom right, a small abstract diagram: two rows of blue dots labelled $c_1$ and $c_2$ feed
upward through a matrix labelled $W$ into a row of dots labelled $p$ (the parent), which in turn
feeds through a matrix labelled $W^{\text{score}}$ up to a single output labelled $s$ (the
score) — a generic, unlabeled-numbers version of the score/parent computation shown concretely
on slides 42–47, emphasizing that the same $W$ and $W^{\text{score}}$ are reused at every node.

## Slide 49 — 7. Recursive Neural Tensor Networks

*(No printed page number; printed numbering stops at slide 48. This and all following slides'
numbers are inferred by continuing the page count — see the "Slide numbers vs PDF pages"
section above.)*

"Socher, Perelygin, Wu, Chuang, Manning, Ng, and Potts 2013"

- Allows two word or phrase vectors to interact multiplicatively

Left, a small tree over "… not very good …": leaves a="not", b="very", c="good"; "very"+"good"
compose into $p_1=g(b,c)$, marked with a blue "+" (positive sentiment); "not"+$p_1$ compose into
$p_2=g(a,p_1)$ at the root, marked with an orange "−" (negative) — illustrating that "not"
negates the positive sentiment of "very good" into an overall negative.

Right, a "Neural Tensor Layer" diagram: two stacked "Slices of Tensor Layer" blocks, each
showing a row of blue dots (the vector $\begin{bmatrix}b\\c\end{bmatrix}^T$), a small colored
square matrix (~4×4 grid, one slice reddish-brown, the other orange), and a column of blue dots
(the vector $\begin{bmatrix}b\\c\end{bmatrix}$) — a bilinear form per tensor slice — added (+) to
a "Standard Layer" term (a tan/yellow row-vector matrix times $\begin{bmatrix}b\\c\end{bmatrix}$).
Below, the corresponding equation:

$$p = f\!\left(\begin{bmatrix} b \\ c \end{bmatrix}^{\!T} V^{[1:2]} \begin{bmatrix} b \\ c
\end{bmatrix} + W\begin{bmatrix} b \\ c \end{bmatrix}\right)$$

- Not today, but see also Tai, Socher, Manning [2015]: TreeLSTMs
  - Work even better

## Slide 50 — Beyond the bag of words: Sentiment detection

*(No printed page number; inferred — see slide 49's note.)*

"Is the tone of a piece of text positive, negative, or neutral?"

- Sentiment is that sentiment is "easy"
- Detection accuracy for longer documents ~90%, BUT

Below, a line of red text scattered with ellipses, illustrating naive keyword-spotting: "… …
loved … … … … … great … … … … … … impressed … … … … … … marvelous … … … …"

Below that, a green flower/asterisk icon next to a quoted review: "With this cast, and this
subject matter, the movie should have been funnier and more entertaining," next to a small
black-and-white photo of a person (a film critic). This is the slide's punchline example: the
sentence contains no scattered "positive" keywords at all, yet a naive bag-of-words detector
tuned to spot words like "loved," "great," "impressed," "marvelous" would still miss that this
is actually a negative review of the film, since its negativity is expressed through what's
absent ("should have been funnier") rather than through explicit negative words.

## Slide 51 — Stanford Sentiment Treebank

*(No printed page number; inferred — see slide 49's note.)*

- 215,154 phrases labeled in 11,855 sentences
- Can actually train and test compositions

Below, a montage of roughly 32 small thumbnail parse-tree diagrams arranged in a grid, each
depicting one sentence's binary parse tree with every node colored by its labeled sentiment
(shades of blue for positive, shades of orange/red for negative, white/uncolored for neutral) —
a visual sampler of the treebank's dense, per-phrase sentiment annotations; individual sentences
are too small to read at this resolution. Caption: "http://nlp.stanford.edu:8080/sentiment/".

## Slide 52 — Better Dataset Helped All Models

*(No printed page number; inferred — see slide 49's note.)*

A bar chart, y-axis "accuracy" ranging 75–84 (gridlines every 1 point), x-axis two groups:
"Training with Sentence Labels" and "Training with Treebank." Three data series, one bar per
group per series:

- **Bi NB** (green): ~79 (Sentence Labels), ~83.2 (Treebank)
- **RNN** (brown/orange): ~78.2 (Sentence Labels), ~82.4 (Treebank)
- **MV-RNN** (blue): ~80 (Sentence Labels), ~82.9 (Treebank)

All three models score higher when trained with the fine-grained Treebank labels than with
whole-sentence labels alone.

- Hard negation cases are still mostly incorrect
- We also need a more powerful model!

## Slide 53 — Recursive Neural Tensor Network

*(No printed page number; inferred — see slide 49's note.)*

"Idea: Allow both additive and mediated multiplicative interactions of vectors"

Right, the same small "not very good" tree as slide 49. Below, a partial build-up of the Neural
Tensor Layer diagram showing only the *first* tensor slice: a dashed box containing a row of
blue dots, a reddish-brown ~4×4 matrix, and a column of blue dots, with the equation fragment
$\begin{bmatrix}b\\c\end{bmatrix}^{\!T} V \begin{bmatrix}b\\c\end{bmatrix}$ beneath it — the
first step in constructing the full formula completed on slide 56.

## Slide 54 — Recursive Neural Tensor Network

*(No printed page number; inferred — see slide 49's note.)*

Same layout as slide 53, now with a *second* tensor slice added below the first (a stacked
dashed box with an orange ~4×4 matrix), and the equation updated to
$\begin{bmatrix}b\\c\end{bmatrix}^{\!T} V^{[1:2]} \begin{bmatrix}b\\c\end{bmatrix}$, i.e. summing
over both slices of the tensor $V$.

## Slide 55 — Recursive Neural Tensor Network

*(No printed page number; inferred — see slide 49's note.)*

Same two-slice diagram as slide 54, now with a "+" and a "Standard Layer" term appended (a
tan/yellow row-vector matrix times $\begin{bmatrix}b\\c\end{bmatrix}$), and the equation updated
to $\begin{bmatrix}b\\c\end{bmatrix}^{\!T} V^{[1:2]} \begin{bmatrix}b\\c\end{bmatrix} +
W\begin{bmatrix}b\\c\end{bmatrix}$ — the full pre-nonlinearity expression.

## Slide 56 — Recursive Neural Tensor Network

*(No printed page number; inferred — see slide 49's note.)*

- Use resulting vectors in tree as input to a classifier like logistic regression
- Train all weights jointly with gradient descent

The complete "Neural Tensor Layer" diagram (same as slide 49's right panel) with the full
equation:

$$p = f\!\left(\begin{bmatrix} b \\ c \end{bmatrix}^{\!T} V^{[1:2]} \begin{bmatrix} b \\ c
\end{bmatrix} + W\begin{bmatrix} b \\ c \end{bmatrix}\right)$$

Right, the small "not very good" tree thumbnail again, and a small standalone plot of a classic
S-shaped sigmoid curve (axes unlabeled) illustrating the nonlinearity $f$.

## Slide 57 — Positive/Negative Results on Treebank

*(No printed page number; inferred — see slide 49's note.)*

"Classifying Sentences: Accuracy improves to 85.4"

A bar chart, y-axis "accuracy" ranging 74–86, x-axis two groups: "Training with Sentence
Labels" and "Training with Treebank." Four data series, one bar per group per series:

- **Bi NB** (green): ~79, ~83.2
- **RNN** (brown/orange): ~78.2, ~82.4
- **MV-RNN** (blue): ~80, ~82.9
- **RNTN** (red): ~79.8, **85.4** — the new model, and the only one to pull clearly ahead of the
  others when trained with Treebank labels.

## Slide 58 — Experimental Results on Treebank

*(No printed page number; inferred — see slide 49's note.)*

- RNTN can capture constructions like *X but Y*
- RNTN accuracy of 72%, compared to MV-RNN (65%), biword NB (58%) and RNN (54%)

Below, a reproduced sentiment-annotated parse tree for "There are slow and repetitive parts,
but it has just enough spice to keep it interesting." Every node is marked with a sentiment sign
(−, +, or 0/neutral) and colored accordingly (orange/red for negative, blue for positive, white
for neutral). The left branch builds up negativity from the bottom: "slow" and "and" combine
to a "−" node, which combines with "repetitive" to a more negative "−" node, then with "parts"
to another "−" node, then with the comma to "−", finally combining with "but" (0) at a "−" node
under the root. The right branch builds up positivity: "just" and "enough" combine to 0, then
with "spice" (+) to a "+" node, then up through "to," "keep," "it," and "interesting" (+) to a
chain of "+" nodes. At the very top, the left ("−") and right ("+") branches combine, and the
**root node is "+"** (positive) — despite the sentence opening with a negative clause, the
sentiment after "but" dominates the overall sentence-level judgment, which is exactly the *X but
Y* pattern the bullets describe.

## Slide 59 — Negation Results

*(No printed page number; inferred — see slide 49's note.)*

"When negating negatives, positive activation should increase!"

Left, two small sentiment-annotated trees:

- **Top tree**, over roughly "It's just incredibly dull.": the root is marked "−−" (strongly
  negative, dark red), built up from a chain of "−" nodes down to a strongly negative "−−" leaf
  at "dull" — a plain negative sentence with no negation.
- **Bottom tree**, over roughly "It's not definitely dull.": the root is marked "+" (blue,
  positive), built up through "+"-marked internal nodes even though the leaf "dull" is still
  itself marked strongly negative ("−−") — illustrating that inserting "not" in front of the
  clause flips the composed sentiment at the root from negative to positive, even though the
  word "dull" alone is unchanged.

**[The individual leaf-by-leaf word labels in these two small trees are difficult to read with
full confidence at this resolution beyond "It," "'s," "dull," and "not"/"definitely"; the
overall root-level sentiment flip (− − at top vs. + at bottom) is clear and is the point the
slide is making.]**

Right, two horizontal bar charts, x-axis "change in activation" ranging −0.6 to 0.4 on both,
each with the same four series (biNB green, RRN orange, MV-RNN blue, RNTN red):

- **"Negated Positive Sentences: Change in Activation"** — biNB −0.16, RRN −0.34, MV-RNN −0.5,
  RNTN −0.54. All four models correctly show a *decrease* in positive activation when a positive
  sentence is negated, with RNTN showing the largest swing.
- **"Negated Negative Sentences: Change in Activation"** — biNB −0.01, RRN −0.01, MV-RNN +0.01,
  RNTN +0.25. Here only RNTN shows a clear, correctly-signed *increase* in activation (sentiment
  moving toward positive) when a negative sentence is negated; the other three models show only
  negligible or essentially zero change, failing to capture the effect of negating a negative.

## Slide 60 — (title slide, reprised)

*(No printed page number; inferred — see slide 49's note.)*

An exact repeat of slide 1's title slide: "Natural Language Processing with Deep Learning /
CS224N/Ling284," the same red-arch-over-three-arches logo, "Christopher Manning" / "Lecture 16:
ConvNets for NLP and Tree Recursive Neural Networks." No new content is added; this appears to
be a closing/reprised title slide rather than a numbered content slide, consistent with pages
49–60 being backup/appendix material appended after the deck's last printed page.
