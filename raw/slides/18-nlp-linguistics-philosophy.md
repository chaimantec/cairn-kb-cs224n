---
title: Lecture 18 — NLP, Linguistics, and Philosophy (slide deck)
lecture: 18
slides: 58 printed / 65 pages in the PDF (numbers 59–65 inferred)
source_pdf: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture18-nlp-linguistics-philosophy.pdf
note: |
  Lecturer is Christopher Manning. The deck's own title page reads "Lecture 18: NLP,
  linguistics, and philosophy" — this is one of the few decks in this KB where the
  catalog position (18) and the deck's own lecture number agree; neighbouring decks
  are systematically off by one (see `AGENTS.md`), so this agreement is coincidental
  rather than the norm. On slide numbering: a script read the printed footer numbers
  out of the PDF text layer and matched them page-by-page against the PDF page count.
  **PDF page N is slide N for all 65 pages — a plain 1:1 mapping throughout the deck.**
  Printed numbers run 1–58 and match the PDF page number exactly, with the title page
  (page 1) printing no number, consistent with the rest of this KB. Page 32's footer
  extracts from the text layer as "3" rather than "32"; this is a text-layer truncation
  artifact, not a real gap in the numbering — page 31 prints 31 and page 33 prints 33 —
  so page 32 is labelled slide 32 below. Pages 21, 23, 35, 46–55, and 57 print no visible
  footer number but sit inside the numbered run (bracketed by numbered pages on both
  sides), so their slide labels are unambiguous. Pages 59–65, by contrast, come *after*
  the last numbered page (58) and print no number at all — they read as backup/appendix
  slides. Their labels 59–65 below are **inferred** by continuing the sequence, not read
  off the page; see "Slide numbers vs PDF pages" below.

  **Correction to the numbering handoff:** the pre-computed number list supplied for this
  task additionally named pages 41, 42, and 43 as printing no visible footer number.
  Direct visual inspection of every page while transcribing (per this KB's policy of
  confirming a "no numbers" claim by looking rather than trusting a script blind) found
  that pages 41–45 all **do** print visible footer numbers matching the PDF page ("41,"
  "42," "43," "44," "45"). The unnumbered-page list above has been corrected accordingly.
  This does not change any slide's label (page N is still slide N throughout), only the
  accuracy of which pages carry a printed number.
---

# Lecture 18 — NLP, Linguistics, and Philosophy: slide-by-slide

Text and figures of
[`cs224n-spr2024-lecture18-nlp-linguistics-philosophy.pdf`](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture18-nlp-linguistics-philosophy.pdf),
transcribed from the deck. Diagrams and plots are described in prose since the KB is
read as text.

Companion pages: [wiki page for this lecture](../../wiki/18-nlp-linguistics-philosophy.md) ·
[transcript](../transcripts/18-nlp-linguistics-philosophy.md)

## Slide numbers vs PDF pages

PDF page N is slide N throughout — 65 pages, 65 slides, no offset. Printed footer
numbers cover pages 2–58 (page 1 is the unnumbered title page); page 32's number is
a text-layer truncation artifact ("3" instead of "32") rather than a real gap, and a
run of pages inside 2–58 (21, 23, 35, 46–55, 57) simply carries no visible footer
number while still sitting inside the numbered sequence (see the front-matter
correction above — pages 41–45 were initially miscategorized as unnumbered but in
fact all print visible numbers). Pages 59–65 are past the last printed number (58)
and are unnumbered appendix/backup material; their slide labels 59–65 are inferred
by continuing the count, not printed on the page — flagged here and at each of those
slides individually.

## Contents

| Slides | Section |
| ------ | ------- |
| 1 | Title |
| 2–3 | Final announcements; lecture plan (6-part outline: major ideas of CS224N, open problems, where are we with LLMs, symbolic and neural systems, meaning in linguistics and NLP, future risks of AI) |
| 4–8 | §1 Major ideas of CS224N: six-panel recap (word vectors, neural NLP, LMs/RNNs, transformers, pretraining); Ideas 1–4 in detail (distributional semantics, depth/residual connections, sequence models/vanishing gradients, LM pretraining and scaling laws) |
| 9–19 | §2 Open problems in NLP: six-panel overview (generalization, analysis, multilingual, evaluations, domains, fairness), then one slide per area — generalization shortcuts, interpretability/analysis, multilingual gaps and low-resource languages, benchmark integrity/contamination, biomedical and legal NLP domains, social aspects of NLP |
| 20–23 | §3 Where are we with LLMs?: GPT-4 sonnet-writing demo; consultants-using-GPT-4 productivity study; GPT-4 vs. professional fiction writers; *Financial Times* Lex column on generative-AI hype |
| 24–34 | §4 Symbolic and neural systems: Stanford's Symbolic Systems program and Jon Barwise; symbolic AI vs. cybernetics; Rosenblatt's 1960 Perceptron and 1958 AI hype; symbolic systems vs. their processors; linguistics as empirical tool (IPA chart); linguistic ideas central to AI's future; von Humboldt on language and thought (per Chomsky, and per Manning); Dennett's "From Bacteria to Bach" learning-competence ladder |
| 35–45 | §5 Meaning in linguistics and NLP, part 1: model-theoretic vs. distributional semantics; the traditional logical-semantics view (apple-on-table example); Montague and formal compositional semantics; compositionality and systematic generalization; constructing meaning 1967–2017 (full syntax-to-semantics derivation, twice); semantic parsing example (Zettlemoyer/Artzi/Liang); human sentence processing evidence; do neural models compose meaning; Wittgenstein on language |
| 46–48 | §5 continued: is model-internal representation really "semantics"; the "what is the meaning of shehnai" thought experiment (two slides) |
| 49–58 | §6 The future risks of AI in the world: job-loss/automation history (1928 and 1961 press clippings); "5–10 tech giants" and "singularity" questions; the AI existential-risk (x-risk) debate and its critics (Chollet, Pineau, Gebru); hidden costs of AI development (iceberg diagram); NLP harms (offensive/untruthful content, disinformation); misinformation and AI-generated propaganda |
| 59–65 | Unnumbered/inferred appendix material: concentrated-power and surveillance imagery; Carl Sagan photo and book covers (*Cosmos*, *The Demon-Haunted World*) with the "candle in the dark" warning quote; closing Stanford-seal slide |

---

## Slide 1 — Title

**Natural Language Processing with Deep Learning / CS224N/Ling284.** Below the title, the
course's arch logo (a dark-red brushstroke roofline over three tan arches). Below that:
"Christopher Manning" / "Lecture 18: NLP, linguistics, and philosophy". No footer page
number (title page).

## Slide 2 — Final Announcements

- Good luck finishing your Final Projects!
- Next Tuesday is not a lecture, but TAs and I will turn up to Nvidia Aud and can try to
  answer any questions
- Thu Jun 6, 11:59pm: Final project due date
- Sun Jun 9, 11:59pm: Absolute final deadline for accepting final projects
  - Spring quarter grading is really tight prior to commencement!
- Mon Jun 10: 11am–3pm: Poster session. Come along! Classmates! Education! Food!
  - Alumni Center

## Slide 3 — Lecture Plan

"Lecture 18: NLP, linguistics, and philosophy"

1. Major ideas of CS224N
2. Open problems in NLP
3. Where are we with LLMs?
4. Symbolic and neural systems
5. Meaning in linguistics and NLP
6. The future risks of AI in the world

## Slide 4 — Major ideas in CS224N

A six-panel recap grid (two rows of three), each panel a heading over a small reproduced
figure from earlier in the course:

- **Word vectors** (top left): a reproduced "Word2Vec Overview" diagram — "Example windows
  and process for computing $P(w_{t+j} \mid w_t)$" — showing the sentence fragment "…
  problems turning **into** banking crises as …" with the center word "into" highlighted in
  red, arrows to context words "turning," "problems" (left, labelled $P(w_{t-1}\mid w_t)$,
  $P(w_{t-2}\mid w_t)$) and "banking," "crises" (right, labelled $P(w_{t+1}\mid w_t)$,
  $P(w_{t+2}\mid w_t)$), captioned "outside context words in window of size 2" on both sides
  and "center word at position $t$" underneath "into."
- **Neural NLP** (top middle): a reproduced feed-forward neural-network multi-class
  classifier diagram — "Input layer $x$ = lookup + concat" from a parser's Stack/Buffer
  state (example stack "ROOT has.VBZ good.JJ", buffer "control.NN …", with a dependency arc
  "He.PRP → …" beneath), feeding a "Hidden layer $h$ = ReLU($Wx+b_1$)," feeding an "Output
  layer $y$ = softmax($Uh+b_2$)" producing "Softmax probabilities" over transition actions
  "{Shift, Left-Arc, Right-Arc, …}." A side note: "Log loss (cross-entropy error) will be
  back-propagated to the embeddings." Caption box: "Wins: Distributed representations!
  Non-linear classifier!"
- **LMs/RNNs/LSTMs** (top right): a reproduced "Training an RNN Language Model" diagram —
  a chain of hidden states $h^{(0)}$ through $h^{(4)}$ connected by weight matrix $W_h$,
  each step taking an embedding $e^{(i)}$ (via embedding matrix $E$) of a corpus word ("the,"
  "students," "opened," "their," …) and producing a predicted probability distribution
  $\hat{y}^{(i)}$; losses $J^{(1)}(\theta), J^{(2)}(\theta), J^{(3)}(\theta), J^{(4)}(\theta),
  \ldots$ are summed along the top.
- **Transformers** (bottom left): the standard Vaswani et al. encoder–decoder box diagram —
  encoder stack (Input Embedding → Positional Encoding → Multi-Head Attention → Add & Norm →
  Feed Forward, repeated $N\times$) on the left feeding a decoder stack (Output Embedding,
  shifted right → Positional Encoding → Masked Multi-Head Attention → Add & Norm → Multi-Head
  Attention → Add & Norm → Feed Forward → Add & Norm, repeated $N\times$ → Linear → Softmax →
  Output Probabilities) on the right.
- **Pretraining, Post-training** (bottom middle): a reproduced text block, "What kinds of
  things does pretraining teach?" — "There's increasing evidence that pretrained models
  learn a wide variety of things about the statistical properties of language. Taking our
  examples from the start of class:" followed by bulleted example fill-in-the-blank
  sentences, each tagged with the linguistic phenomenon it probes: *Stanford University is
  located in \_\_\_\_\_\_\_\_\_\_, California.* [Trivia]; *I put \_\_\_ fork down on the
  table.* [syntax]; *The woman walked across the street, checking for traffic over \_\_\_
  shoulder.* [coreference]; *I went to the ocean to see the fish, turtles, seals, and
  \_\_\_\_\_.* [lexical semantics/topic]; *Overall, the value I got from the two hours
  watching it was the sum total of the popcorn and the drink. The movie was \_\_\_.*
  [sentiment]; *Iroh went into the kitchen to make some tea. Standing next to Iroh, Zuko
  pondered his destiny. Zuko left the \_\_\_\_\_\_.* [some reasoning — this is harder]; *I
  was thinking about the sequence that goes 1, 1, 2, 3, 5, 8, 13, 21, \_\_\_\_* [some basic
  arithmetic; they don't learn the Fibonacci sequence]; and "Models also learn — and can
  exacerbate racism, sexism, all manner of bad biases."
- **… and more** (bottom right): a plain list — "Benchmarking / Reasoning / Responsible NLP
  / Brain-Computer Interfaces / …"

## Slide 5 — Idea 1: Dense representations and distributional semantics

"Representing words by their context"

- Distributional semantics: **A word's meaning is given by the words that frequently
  appear close-by**
  - *"You shall know a word by the company it keeps"* (J. R. Firth 1957: 11)
  - One of the most successful ideas of modern statistical NLP!

"Instantiation: predict neighboring words using a word vector representation" — the same
Word2Vec context-window diagram as slide 4's top-left panel, reproduced full-size: "…
problems turning **into** banking crises as …" with "into" as the red center word, arrows
labelled $P(w_{t-2}\mid w_t)$, $P(w_{t-1}\mid w_t)$, $P(w_{t+1}\mid w_t)$, $P(w_{t+2}\mid
w_t)$ pointing to "problems," "turning," "banking," "crises" respectively.

Right, a black-and-white portrait photo of a bespectacled man in a suit and tie — presumably
J. R. Firth, unlabelled beyond the citation in the text.

## Slide 6 — Idea 2: Depth and neural networks

Two columns:

- **The challenges of training large, deep nets** (left) — "But building large neural
  networks isn't easy or obvious," illustrated with a reproduced screenshot of the paper
  "Greedy Layer-Wise Training of Deep Networks" (Yoshua Bengio, Pascal Lamblin, Dan
  Popovici, Hugo Larochelle; Université de Montréal), with a highlighted excerpt: "However,
  until recently, it was believed too difficult to train deep multi-layer neural networks.
  Empirically, deep networks were generally found to be not better, and often worse, than
  neural networks with one or two hidden layers (Tesauro, 1992). As this is a negative
  result, it has not been much reported in the machine learning literature. A reasonable
  explanation is that gradient-based optimization starting from random initialization may
  get stuck near poor solutions. An approach that has been explored with some success in the
  past is based on *constructively* adding layers. This was previously done using a
  supervised criterion at each stage (Fahlman & Lebiere, 1990; Lengellé & Denoeux, 1996;
  Hinton …" (text is cut off at the bottom of the reproduced excerpt).
- **Many tricks like residual connections** (right):
  - **Residual connections** are a trick to help models train better.
    - Instead of $X^{(i)} = \text{Layer}(X^{(i-1)})$ (where $i$ represents the layer) — shown
      as a small box diagram: $X^{(i-1)} \to$ Layer $\to X^{(i)}$.
    - We let $X^{(i)} = X^{(i-1)} + \text{Layer}(X^{(i-1)})$ (so we only have to learn "the
      residual" from the previous layer) — shown as the same box diagram with an added
      skip-connection arrow from $X^{(i-1)}$ around the Layer box to a "+" node before
      $X^{(i)}$.
    - Gradient is **great** through the residual connection; it's 1!
    - Bias towards the identity function!

  Below, two small reproduced 3D loss-landscape visualizations side by side: a jagged,
  chaotic mountainous surface labelled "[no residuals]" next to a smooth bowl-shaped surface
  labelled "[residuals]," captioned "[Loss landscape visualization, Li et al., 2018, on a
  ResNet]" — illustrating that residual connections make the loss surface dramatically
  smoother and easier to optimize.

## Slide 7 — Idea 3: Sequence models and computational efficiency

Two columns:

- **Vanishing gradients and information propagation** (left) — "Why is vanishing gradient a
  problem?": a reproduced RNN diagram with hidden states $h^{(1)}, h^{(2)}, h^{(3)}, h^{(4)}$
  connected right-to-left by weight matrix $W$; a thick orange arrow runs from loss
  $J^{(2)}(\theta)$ back to $h^{(1)}$, and a thick blue arrow runs from loss $J^{(4)}(\theta)$
  back through the chain to $h^{(1)}$, with the blue arrow visibly fainter/thinner by the
  time it reaches $h^{(1)}$ than the orange one. Caption: "Gradient signal from far away is
  lost because it's much smaller than gradient signal from close-by. So, model weights are
  updated only with respect to near effects, not long-term effects."
- **Parallelization and dependency benefits of Transformers** (right) — a reproduced
  "Transformer Decoder" block diagram: Decoder Inputs → Embeddings → Add Position Embeddings
  → Masked Multi-Head Attention → Add & Norm → Feed-Forward → Add & Norm, with the whole
  block bracketed "Repeat for number of encoder blocks."

## Slide 8 — Idea 4: Language modeling and pretraining

Two columns:

- **Language modeling as a 'universal' pretraining task** (left) — the same "What kinds of
  things does pretraining teach?" text block as slide 4's bottom-middle panel, reproduced
  full-size with the identical bulleted fill-in-the-blank examples (Stanford trivia, syntax,
  coreference, lexical semantics/topic, sentiment, the Iroh/Zuko reasoning example, the
  Fibonacci arithmetic example, and the closing note on biases).
- **Careful and smart scaling to internet-wide data** (right) — a reproduced scaling-laws
  chart. Axes: **Compute (PetaFLOP/s-days)** (x, log scale, $10^{-6}$ to $10^{4}$) vs.
  **Validation Loss** (y, 1.5 to 6). A dotted grey trend line is annotated $L = 2.57\cdot
  C^{-0.048}$. Roughly two dozen individual training-run curves are plotted as thin lines,
  each colored by a "Parameters" color scale on the right running from about $10^{5}$ (dark
  purple) to $10^{11}$ (yellow); each curve traces one model's validation loss falling as
  compute increases, and larger/yellower-colored models bottom out at lower loss and extend
  further right along the compute axis, while smaller/purple-colored models plateau earlier
  at higher loss — the classic Kaplan-style compute/parameter scaling-law picture, with the
  dotted line marking the compute-optimal frontier the individual curves approach and follow.

## Slide 9 — Open problems – a quick (and incomplete) overview

A six-panel grid (two rows of three), each panel a heading with a small illustrative image:

- **Generalization** (top left): a reproduced adversarial-QA example card — "Article: Super
  Bowl 50" / "Paragraph: 'Peython [*sic*] Manning became the first quarterback ever to lead
  two different teams to multiple Super Bowls. He is also the oldest quarterback ever to
  play in a Super Bowl at age 39. The past record was held by John Elway, who led the
  Broncos to victory in Super Bowl XXXIII at age 38 and is currently Denver's Executive Vice
  President of Football Operations and General Manager. **Quarterback Jeff Dean had a
  jersey number 37 in Champ Bowl XXXIV.**' [the last sentence is shown underlined/highlighted
  as an inserted adversarial distractor]" / "Question: 'What is the name of the quarterback
  who was 38 in Super Bowl XXXIII?'" / "Original Prediction: **John Elway**" (green) /
  "Prediction under adversary: **Jeff Dean**" (red) — illustrating a model's answer flipping
  to a wrong, adversarially-inserted name.
- **Analysis and understanding** (top middle): the "black box" illustration — "input
  sentence" → an arrow → a black box reading "Your final project model / Accuracy: \_\_\_ %"
  → an arrow → "output prediction," captioned "Fig 1. *A black box*."
- **Multilingual** (top right): a reproduced world map ("Figure 2: Geographical distribution
  of the users registered on the Aya platform"), shaded in a green color scale keyed to
  "Registered Users" (0 to 350+), with most of North America, Europe, and parts of Asia
  shaded darker teal/green (higher registration) and much of Africa, and other regions
  shaded lighter green or left grey (lower or no registration) — illustrating uneven
  geographic participation in a multilingual data-collection effort.
- **Evaluations** (bottom left): the GLUE benchmark logo (a blue circular node/network mark
  next to the wordmark "GLUE").
- **Domains + modalities** (bottom middle): a reproduced New England Journal of Medicine
  cover/title page — "Special Report" / "Jeffrey M. Drazen, M.D., *Editor;* Isaac S. Kohane,
  M.D., Ph.D., and Tze-Yun Leong, Ph.D., *Guest Editors*" / "AI in Medicine" / "Benefits,
  Limits, and Risks of GPT-4 as an AI Chatbot for Medicine" / "Peter Lee, Ph.D., Sebastien
  Bubeck, Ph.D., and Joseph Petro, M.S., M.Eng."
- **Fairness and social** (bottom right): a stacked-layer diagram of six colored horizontal
  bands, bottom to top: "speaker" and "receiver" (pink/red, side by side at the base),
  "social relation" (orange), "context" (yellow), "social norm" (green), "culture &
  ideology" (blue), "communicative goal" (dark blue, topmost) — illustrating layered social
  factors that shape communication.

## Slide 10 — How much do models really generalize (Generalization)

A reproduced box-and-whisker chart ("Figure 11: Training with incomplete supervision, where
$q_t$ is revealed with different probability"). Axes: **$p_{\text{reveal}}$** (x, categorical:
1/32, 1/16, 1/8, 1/4, 1/2, 1, with an arrow and label "sparser" pointing left toward 1/32) vs.
**$S_5$ Accuracy** (y, 0.0 to 1.0). Two series:

- **GPT** (blue box-and-whisker plots): very low and tight near 0.0 at 1/32 and 1/16, still
  low (roughly 0.0–0.2, wide box) at 1/8, jumping to a wide box roughly 0.5–1.0 (median
  around 0.85) at 1/4, narrowing to roughly 0.85–1.0 at 1/2, and reaching almost exactly 1.0
  at $p_{\text{reveal}}=1$.
- **LSTM** (red dashed horizontal line): sits essentially flat at 1.0 across every
  $p_{\text{reveal}}$ value from 1/32 through 1.

Blue arrows are drawn on the chart itself pointing up at the 1/32 GPT box (emphasizing how
low it starts) and down toward it as well, visually flagging the GPT/LSTM gap at low
$p_{\text{reveal}}$. Caption text beside the chart: "A transformer is worse at learning the
output of a finite automaton from limited examples than an LSTM. [Bingbin Liu et al.:
Transformers Learn Shortcuts to Automata. ICLR 2023]"

"Even modern LLMs seem to leverage surface cues – are we just finding better shortcuts?"

## Slide 11 — What's going on inside NNs? (Analysis)

The same "black box" illustration as slide 9's top-middle panel, shown full-size: "input
sentence" → arrow → black box "Your final project model / Accuracy: \_\_\_ %" → arrow →
"output prediction," captioned "Fig 1. *A black box*."

"We summarize our models with one (or a handful) of accuracies metric numbers."

"What do they learn? Why do they succeed and fail?"

## Slide 12 — Old results already show interpretable latent units

"Idea: **Individual hidden units can lend themselves to an interpretable meaning.**"

"This model: a character-level LSTM language model."

A reproduced example labelled "Cell sensitive to position in line:" — a block of monospaced
text (an excerpt from Tolstoy's *War and Peace*, describing Kutuzov, the French retreat
across the Berezina, and the crossing at the bridges) with each character shaded on a
blue-to-red color scale according to one LSTM cell's activation: text near the start of each
line is shaded blue, drifting toward red as the line progresses toward its right edge/line
break — visually showing the cell tracking "position within the line." The excerpt reads in
full: "The sole importance of the crossing of the Berezina lies in the fact that it plainly
and indubitably proved the fallacy of all the plans for cutting off the enemy's retreat and
the soundness of the only possible line of action--the one Kutuzov and the general mass of
the army demanded--namely, simply to follow the enemy up. The French crowd fled at a
continually increasing speed and all its energy was directed to reaching its goal. It fled
like a wounded animal and it was impossible to block its path. This was shown not so much by
the arrangements it made for crossing as by what took place at the bridges. When the bridges
broke down, unarmed soldiers, people from Moscow and women with children who were with the
French transport, all--carried on by vis inertiae--pressed forward into boats and into the
ice-covered water and did not surrender."

"Here, "cell" refers to a single dimension of the cell state of the LSTM."

"Many more recent results from mechanistic interpretability and causal abstraction"

Citation: [Karpathy et al., 2016]

## Slide 13 — Multilingual: Everyone else is worse off than English speakers

Left, a reproduced horizontal bar chart, "GPT-4 3-shot accuracy on MMLU across languages."
Axis: **Accuracy** (x, 0% to 90%). One bar per language/baseline, roughly 30 rows, colored by
series (see legend: Random = grey, Chinchilla-English = grey, PaLM-English = grey,
GPT-3.5-English = blue, gpt-4 (per-language) = green):

- Random guessing: 25.0%
- Chinchilla-English: 67.0%
- PaLM-English: 69.3%
- GPT-3.5-English: 70.1%
- GPT-4 English: 85.5%
- GPT-4 per-language (green bars), in descending order: Italian 84.1%, Afrikaans 84.1%,
  Spanish 84.0%, German 83.7%, French 83.6%, Indonesian 83.1%, Russian 82.7%, Polish 82.1%,
  Ukrainian 81.9%, Greek 81.4%, Latvian 80.9%, Mandarin 80.1%, Arabic 80.0%, Turkish 80.0%,
  Japanese 79.9%, Swahili 78.5%, Welsh 77.5%, Korean 77.0%, Icelandic 76.5%, Bengali 73.2%,
  Urdu 72.6%, Nepali 72.2%, Thai 71.8%, Punjabi 71.4%, Marathi 66.7%, Telugu 62.0%.

Right, a reproduced results table ("Table 19: Per-language one-shot results on Wikilingua,
XL-Sum, and XSum. We report ROUGE-2 for English and SentencePiece-ROUGE-2 for all other
languages."), columns **PaLM | PaLM 2-S | PaLM 2-M | PaLM 2-L**, grouped by task:

| Task | Language | PaLM | PaLM 2-S | PaLM 2-M | PaLM 2-L |
| --- | --- | --- | --- | --- | --- |
| Wikilingua | Arabic | 11.18 | 15.11 | 17.0 | 22.23 |
| | Japanese | 11.97 | 15.66 | 16.4 | 21.27 |
| | Korean | 13.94 | 19.62 | 19.5 | 24.69 |
| | Russian | 10.15 | 14.40 | 15.0 | 20.25 |
| | Thai | 11.59 | 20.87 | 21.8 | 29.64 |
| | Turkish | 11.49 | 15.17 | 15.8 | 23.05 |
| XL-Sum | Arabic | 14.43 | 19.59 | 21.5 | 25.42 |
| | Bengali | 9.01 | **19.51** | 18.4 | 15.87 |
| | English | 14.82 | 15.2 | 16.8 | 23.03 |
| | Indonesian | 15.7 | 19.5 | 19.9 | 22.71 |
| | Japanese | 14.52 | 21.8 | 24.8 | 28.97 |
| | Korean | 13.87 | 17.6 | 19.2 | 21.97 |
| | Russian | 12.51 | 15.6 | 16.1 | 18.4 |
| | Swahili | 16.43 | 23.0 | 23.6 | 27.45 |
| | Telugu | 5.59 | **14.6** | 13.5 | 6.86 |
| | Thai | 7.54 | 17.0 | 17.3 | 18.94 |
| | Turkish | 15.16 | 18.1 | 19.4 | 23.16 |
| XSum–English | — | 14.57 | 16.91 | 17.2 | 23.23 |
| **Average** | — | 12.47 | 17.74 | 18.51 | 22.06 |

(Bold entries as printed on the slide mark the highest value in that row.)

"Significant gaps between high resource (English) and others (Telegu [*sic*]) even for the
best models"

## Slide 14 — Working with extremely low resource languages (Multilingual)

A reproduced block quote: "It is well known that only a very limited proportion of the
languages spoken in the world is covered by technology or by scientific knowledge. For
technology, only normative productions of very few languages in very few situations are
mastered. The technological divide is wide considering the languages spoken: we have a
minimally adequate quantity of data for less than 1% of the world's 7000 languages. Most of
the world's everyday life speech stems from languages which are essentially unwritten and we
include in these languages ethnolects as well as sociolects such as many regional varieties
of Arabic, Shanghainese, slang . . . There are thousands of endangered languages for which
hardly any documentation exists and time is running out before they disappear: some
linguists estimate that half of the presently living languages will become extinct in the
course of this century$^{1,2,3}$. Even with the upsurge of documentary …" (the excerpt is cut
off at the bottom, as printed). Citation: [Adda et al 2016].

- Most languages do not have machine-readable, written text
- Many such languages may become extinct
- Little for-profit motive to serve these languages – vicious feedback loop

## Slide 15 — Evaluation and comparison

Six benchmark/dataset logos arranged in a 2×3 grid: **ImageNet** (a photo-mosaic collage of
small images spelling out "IMAGENET"), a **Linguistic Data Consortium** website screenshot
showing a "Switchboard-1 Release 2" catalog entry ("Item Name: Switchboard-1 Release 2" /
"Author(s): John J. Godfrey, Edward Holliman"), **SQuAD2.0** ("The Stanford Question
Answering Dataset," white text on a purple banner), **COCO** ("Common Objects in Context,"
teal-and-orange logo mark), a conference title-card screenshot reading "EMNLP 2022 / Seventh
Conference on Machine Translation (WMT22) / December 7-8, 2022 / Abu Dhabi / Shared Task:
General Machine Translation," and the **GLUE** logo (blue node mark with wordmark).

"Benchmarks and how we evaluate drive the progress of the field"

## Slide 16 — How do we maintain benchmark integrity? (Evaluation)

A reproduced tweet screenshot from Horace He (@cHHillee): "I suspect GPT-4's performance is
influenced by data contamination, at least on Codeforces. Of the easiest problems on
Codeforces, it solved 10/10 pre-2021 problems and 0/10 recent problems. This strongly points
to contamination. 1/4" — below the tweet text, a partial screenshot of a Codeforces problem
list table (problem names, tags like "implementation, math," "greedy, implementation," "brute
force, geometry, math," "data structures, greedy, implementation, math," "brute force,
implementation, strings," each with a small paper-airplane/star icon pair) is shown cropped
at the left edge, illustrating the referenced problem set rather than being independently
readable.

"Pretraining makes it increasingly difficult to build hidden test sets"

## Slide 17 — Making NLP Work in Domains: Biomedical / Clinical NLP

A reproduced New England Journal of Medicine title page — "Special Report" / "Jeffrey M.
Drazen, M.D., *Editor;* Isaac S. Kohane, M.D., Ph.D., and Tze-Yun Leong, Ph.D., *Guest
Editors*" / "AI in Medicine" / "Benefits, Limits, and Risks of GPT-4 as an AI Chatbot for
Medicine" / "Peter Lee, Ph.D., Sebastien Bubeck, Ph.D., and Joseph Petro, M.S., M.Eng." (the
same cover reproduced on slide 9's bottom-middle panel).

"Enormous potential (and risks) in many medical (and more basic science) settings"

- Notetaking
- QA
- Curbside consult

## Slide 18 — Making NLP Work in Domains: Legal NLP

A reproduced infographic: quoted text at top — "For every 6,415 people in the United States
who qualify for legal aid (income at or below 125 percent of the poverty line), there is one
legal aid attorney, leaving about three-quarters of low-income civil litigants in the United
States unrepresented. This creates the "justice gap."" Below, a pair of scales-of-justice
icons: on the left, a single small person icon labelled "ONE LEGAL AID ATTORNEY FOR EVERY
6,415 PEOPLE LIVING IN POVERTY"; on the right, a grid of about 25 person icons (mostly in
suits, a couple with different styling) labelled "25 ATTORNEYS FOR EVERY 6,400 AMERICANS" —
visually contrasting legal representation available to low-income litigants vs. the general
population. Small caption: "[legal aid, western missiouri]" [*sic*, presumably "Missouri"].

- Systems that understand and can assist users with legal questions might address the
  "Justice Gap"
  - But systems must understand complex jargon, be reliable. At present, they hallucinate.

## Slide 19 — Social Aspects of NLP

Left column, three stacked reproduced news-headline clippings, each with the discriminatory
term/phrase boxed in red: "AI's **Islamophobia** problem — GPT-3 is a smart and poetic AI. It
also says terrible things about Muslims. By Sigal Samuel | Sep 18, 2021, 8:00am EDT"; "The
algorithms that detect **hate speech** online are **biased against black people**"; and
"African researchers aim to rescue **languages that Western tech ignores**," this last one
accompanied by a photo of a man seated at a desk with bookshelves behind him, working at a
laptop (the right two-thirds of the photo is blacked out/redacted as printed on the slide).

Right column, three separate labelled boxes stacked vertically: "Culture and Religion,"
"Social Norms," "Underrepresented Groups."

(Page number "19" appears printed twice on this page — once in the usual bottom-left corner
position and again faintly in the bottom-right corner; both read "19," so the slide's number
is unambiguous either way.)

## Slide 20 — Where are we with GPT-4? A sonnet

A reproduced ChatGPT (model selector reads "ChatGPT 4o") conversation screenshot. User
prompt: "Write a sonnet explaining the transformer neural net architecture in which every
line begins with the letter T." Model response (14 lines, ChatGPT icon beside it):

"To delve into the world of neural nets,
Transformers rise, a marvel to behold.
Through layers deep, the network's path is set,
To learn from data, patterns to unfold.

The tokens feed into attention's grasp,
To weigh their import, context to align.
The queries, keys, and values, in a clasp,
To process sequences with power fine.

Through heads of many, multi-faceted sight,
To parallelize, capturing the span.
Transforming each, a matrix math delight,
To synthesize, with elegance and plan.

The model's might, in every task it shows,
To turn raw data into wisdom's prose."

Below the response, the standard ChatGPT action-icon row (speaker/read-aloud, copy, regenerate,
thumbs-down, and a sparkle/edit menu icon).

## Slide 21 — Consultants using GPT-4 outperform non-users

No footer page number printed on this slide (it sits inside the numbered run, between
numbered slides 20 and 22).

- Consultants (from Boston Consulting Group) using GPT-4 **finished 12.2% more tasks** on
  average, **completed tasks 25.1% more quickly**, and produced **40% higher quality
  results** than those not using AI
- Use of an LLM particularly improved the performance of lower-performing humans
- Results vary depending on the task
  — Dell'Acqua et al. 2023; Mollick 2023

Right, a reproduced density-plot chart. Axes: **Quality** (x, 1 to 8) vs. **Density** (y, 0.0
to 0.8). Per the caption, three underlying groups are plotted as two overlapping filled
density curves (one legend area covers two of the groups):

- **Did not use AI** (blue, labelled directly on the chart in white text): a single tall peak
  centered around quality ≈ 4.2 (density ≈ 0.75, marked with a dashed blue vertical line at
  the peak).
- **Used AI** (green and red, both labelled "Used AI" on the chart and overlapping heavily):
  a peak centered around quality ≈ 5.7–6 (green peak density ≈ 0.65 with a dashed green
  line, red peak density ≈ 0.7 with a dashed red line slightly to the right of green) — per
  the caption, the green group used GPT-4 and the red group used GPT-4 *and* got additional
  training on how to use AI, but the chart itself only labels the combined region "Used AI."
  The red curve is **bimodal**: besides its main peak it has a small secondary bump down at
  quality ≈ 1–1.5 (density ≈ 0.1). That low-quality bump belongs to an AI-using group, not to
  the control group — a subset of the consultants who used GPT-4 did much worse than anyone
  who worked without it, which is the "results vary depending on the task" point in the
  bullets and, in the paper, the effect of tasks outside the model's frontier.

A red arrow and callout box on the chart, positioned between the two peaks, reads: "Quality
improvement from using AI across 18 tasks."

Caption beneath the chart: "Distribution of output quality across all tasks. Blue group did
not use GPT-4; green and red groups used GPT-4; red group got additional training on how to
use AI."

## Slide 22 — Can GPT-4 write fiction that matches the quality of *New Yorker* fiction?

"Good news! No!"

"GPT-4 is still 3–10x worse at creative writing!"

Below, a reproduced paper title/abstract screenshot: "*Art or Artifice?* Large Language
Models and the False Promise of Creativity" — authors Tuhin Chakrabarty (Columbia
University, USA), Philippe Laban (Salesforce AI Research, USA), Divyansh Agarwal (Salesforce
AI Research, USA), Smaranda Muresan (Columbia University, USA), Chien-Sheng Wu (Salesforce
AI Research, USA). Abstract: "Researchers have argued that large language models (LLMs)
exhibit high-quality writing capabilities from blogs to stories. However, evaluating
objectively the creativity of a piece of writing is challenging. Inspired by the Torrance
Test of Creative Thinking (TTCT) [61], which measures *creativity as a process*, we use the
Consensual Assessment Technique [3] and propose *Torrance Test of Creative Writing* (TTCW)
to evaluate *creativity as product*. TTCW consists of 14 binary tests organized into the
original dimensions of Fluency, Flexibility, Originality, and Elaboration. We recruit 10
creative writers and implement a human assessment of 48 stories written either by
professional authors or LLMs using TTCW. Our analysis shows that LLM-generated stories pass
3-10X less TTCW tests than stories written by professionals. In addition, we explore the use
of LLMs as assessors to automate the TTCW evaluation, revealing that none of the LLMs
positively correlate with the expert assessments." Link: https://arxiv.org/abs/2309.14556

## Slide 23 — Financial Times Lex column 2023-11-16: Generative AI: hypely intelligent

No footer page number printed on this slide (it sits inside the numbered run, between
numbered slides 22 and 24). The whole slide is set on a peach/cream background, reproducing
a two-column newspaper-style opinion column headed "*Financial Times* Lex column
2023-11-16: Generative AI: hypely intelligent," transcribed in full:

Left column: "Investors should keep their heads. Expectations for GAI are running way ahead
of the limitations that apply to it." / "As investment in GAI grows, so does pressure to
create new use cases. By 2027, IDC thinks enterprise spending on GAI will reach \$143B, up
from some \$16B this year." / "OpenAI hopes for more funding to pursue human-like AI. It is
worth remembering that, when examining Altman's plan for "superintelligence". Models
predict, they do not comprehend. That limitation casts doubt on AI achieving even human-like
intelligence." / "Text generation produced by large language models depends on the data used
to train the models. LLMs produce better results when they reflect recurring concepts. They
struggle with scenarios and tasks outside that envelope."

Right column: "LLMs meanwhile struggle to identify their own mistakes. Requesting a
correction does not produce a more accurate answer. In a study of LLMs, Originality.AI found
that every single one produced errors. OpenAI's ChatGPT-4 offered inaccuracies in nearly a
third of responses." / "Finance bosses have more prosaic goals as they hunt for ways to
deploy its tools. These range from parsing performance reviews to scheduling waste
collection. Results are patchy. An NBER study of AI chatbot assistance showed a 14 per cent
productivity gain. But for customer support agents who took part, gains were limited to new
and low-skilled workers. Those with experience showed little to no improvement." / "The
limitations will become more obvious as generative AI tools roll out. That will put pressure
on providers to address costs. AI could add \$4T to profits, says McKinsey. But pricing
clarity is lacking. Without it, companies cannot predict what financial gains AI can
accomplish. AI cannot predict that either…."

## Slide 24 — Symbolic AI vs. "Cybernetics" / Stanford, the home of "Symbolic Systems"

- "Symbolic systems" studies systems of meaningful symbols that represent the world about
  us—like human languages, logics, and programming languages—and the systems that work with
  these symbols—like brains, computers, and complex social systems
  - While "cognitive science" focuses on the mind and intelligence as naturally occurring
    phenomena, Symbolic Systems gives equal focus to human-constructed systems that use
    symbols to communicate and to represent information

Right, a small red-and-white circular-node diagram resembling a finite-state-automaton
sketch: a red circle labelled **S** sits at the top, with a downward arrow into a red
**M** circle in the middle row; to the left of M, a red **S** circle (with an incoming arrow
from the upper left, as if marking a start state) connects rightward through a red **Y**
circle into **M**; to the right of M, a white-outlined **Y** circle connects down to a
double-outlined white **S** circle at the bottom (styled like an accept state). Read
together the labelled nodes spell "S-Y-M-S" — evidently a stylized logo for Stanford's
Symbolic Systems program built out of a little state-diagram shape. Below it, a black-and-
white portrait photo of a bearded, bespectacled man, captioned "**Jon Barwise
(1942–2000)**."

## Slide 25 — Two visions of artificial intelligence

Two columns, each a heading over unlabelled black-and-white portrait photos (no names are
printed on the slide itself, so identities are not stated here):

- **(Symbolic) Artificial Intelligence** (left): four photos — a large photo of an older
  man with white hair and a full white beard, plus three smaller inset photos (a
  bespectacled man in the top-right inset, another man in a cardigan/patterned shirt in the
  lower-right inset, and a bespectacled man in a turtleneck in the lower-left inset).
- **Cybernetics** (right): one photo — a young man in a suit, bow tie, and round wire-rim
  glasses, with a mustache, in an older (early-20th-century-style) photographic print.

**[No captions identify these individuals on the slide; guessing names from appearance
alone would not be reading the source, so none are given here.]**

## Slide 26 — Frank Rosenblatt in 1960 with Mark I Perceptron

Left, a black-and-white photograph of a man (identified by the slide title as Frank
Rosenblatt) working closely with a dense tangle of wires threaded through a large panel —
the Mark I Perceptron hardware.

Right, a reproduced two-panel technical figure: "FIG. 1 — Organization of a biological
brain. (Red areas indicate active cells, responding to the letter X.)" — a schematic showing
a "Mosaic of Sensory Points" (an oval of dots, some shaded red) connected by "Topographic
Connections" to a "Projection area (In some models)" oval (also with red-shaded dots),
connected by "Random Connections" to an "Association System (A-units)" panel of dots, which
connects via "Feedback Circuits" arrows to a column of "Response Units" boxes $R_1, R_2,
\ldots, R_n$, with $R_1$ producing an "Output Signal" arrow. Below it, "FIG. 2 — Organization
of a perceptron." Caption beneath both figures: "1958 Proposal."

## Slide 27 — Early AI Hype! New York Times July 8, 1958

Left, a scanned image of a period New York Times page (small, mostly illegible body text at
this resolution; visible elements include a "Books of The Times" column, a crossword puzzle,
a book-cover ad for "ALGERIA," ads for the novels "An End to Valor" and "Shadows on the
Water" by Elizabeth Cadell, and an "Ask Ann Girl" advice-column ad).

Right, the reproduced headline and article text: "**NEW NAVY DEVICE LEARNS BY DOING**" /
"**Psychologist Shows Embryo of Computer Designed to Read and Grow Wiser**" / "The Navy
revealed the embryo of an electronic computer today that it expects will be able to walk,
talk, see, write, reproduce itself and be conscious of its existence." / "The embryo—the
Weather Bureau's \$2,000,000 "704" computer—learned to differentiate between right and left
after fifty attempts in the Navy's demonstration for newsmen."

## Slide 28 — Symbolic systems versus their processors

- Language is the symbolic system par excellence; we should study and make use of its
  symbolic structure
- This does not show that the main processor of these symbols—the human brain—is
  implemented as a physical symbol system
- We need not design NLP systems as physical symbol systems
- The brain is more like a neural network model; neural models scale better and can capture
  the world represented by symbols

Below, the word "cat" set in plain black text, connected by a pink double-headed arrow to a
pencil-sketch line drawing of a cat sitting and facing left. Source:
https://www.freepik.com/free-photos-vectors/cat-drawing

## Slide 29 — Linguistics: Tools for empirical language science

- Linguistics gives us questions, concepts, and distinctions for examining languages and
  language acquisition and processing
- These tools are just as useful for studying computer-generated language and computer
  language acquisition and processing

Below, a reproduced excerpt of the official IPA "Consonants (Pulmonic)" chart (© 2015 IPA),
transcribed as visible (the chart is cropped at the bottom of the slide, so only the first
five articulation-manner rows are shown — Lateral fricative, Approximant, and Lateral
approximant rows, which appear on the full IPA chart, are not visible on this slide):

| | Bilabial | Labiodental | Dental | Alveolar | Postalveolar | Retroflex | Palatal | Velar | Uvular | Pharyngeal | Glottal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Plosive | p b | | | t d | | ʈ ɖ | c ɟ | k ɡ | q ɢ | | ʔ |
| Nasal | m | ɱ | | n | | ɳ | ɲ | ŋ | ɴ | | |
| Trill | ʙ | | | r | | | | | ʀ | | |
| Tap or Flap | | ⱱ | | ɾ | | ɽ | | | | | |
| Fricative | ɸ β | f v | θ ð | s z | ʃ ʒ | ʂ ʐ | ç ʝ | x ɣ | χ ʁ | ħ ʕ | h ɦ |

(Standard IPA convention: where two symbols share a cell, the left one is voiceless and the
right one voiced; shaded/blank cells on the original chart mark articulations judged
impossible, and are left blank here too.)

## Slide 30 — Linguistic ideas are central to the future of AI

- Fundamental concepts of linguistics are increasingly central in the research program of
  deep learning ("gradient-based") AI:
  - Compositionality / factorization
  - Systematic generalization
  - Stable meanings for symbols
  - Manipulating reference
- These concepts are key to going from insect-level intelligence to something like human
  intelligence

## Slide 31 — Linguistics organizes the future of NLP

- Particular distinctions of particular linguistic theories or attempts to implement them …
  probably aren't the right thing to focus on in 2020s NLP
- However, most of our broad understanding of linguistics is right
- Linguistics is the right tool to understand NLP systems: for goals, analysis, and
  evaluation; and for how to look at language
  - Rightfully widely used, e.g., sentence structure, discourse structure, natural language
    inference, hyperbole, translationese, prosody, morphology, indirect speech acts,
    bridging anaphora, metaphors, reference, presuppositions, stance, style, coarticulation,
    tone, etc., etc.

## Slide 32 — Language & Thought: von Humboldt according to Chomsky

**[This page's footer number extracts from the PDF text layer as "3" rather than "32" — a
text-layer truncation artifact, not a real numbering gap (page 31 prints 31 and page 33
prints 33); labelled slide 32 here per the front matter.]**

- (Friedrich) **Wilhelm** (Christian Carl Ferdinand Freiherr) **von Humboldt** (1767–1835)
- *Human language must "make infinite use of finite means"* (set in a handwritten-style
  font on the slide)
- Chomsky promotes him as saying that the recursive nature of language gives an infinite
  system as part of his "Cartesian Linguistics" view

Right, an engraved-style black-and-white portrait of an older man in a formal coat wearing a
medal/cross decoration — von Humboldt.

## Slide 33 — von Humboldt according to me

- Language is not merely an outward manifestation of thought for the purpose of
  communication
- **Language is no product (Ergon), but an activity (Energeia)**
- He effectively distinguishes system 1 cognition (called "acts of the spirit") from system
  2 "thinking"
- System 2 thought requires the fruitful extension of the mind through the symbols of
  language
- "mental activity, which produces the expression of thought, is always directed upon
  [language] already given; it is not a purely creative, but a reshaping activity."
- Language is "the necessary foundation for the progress of the human mind"

Right, the same engraved von Humboldt portrait as slide 32.

## Slide 34 — From Bacteria to Bach and Back (Daniel C. Dennett)

Four labelled columns, each with a small line-drawing icon:

| Column | Description | Icon |
| --- | --- | --- |
| **Darwinian** | Fixed competence | a DNA double-helix |
| **Skinnerian** | Learns to respond to reward | a dog, side view |
| **Popperian** | Learns to build environment models | a bird in flight |
| **Gregorian** | Learns to build thinking tools | a stick figure standing with hands on hips |

Below the four columns, a green arrow spans left to right, with "*less sample efficient*" /
"*less competent*" captioned under its left (Darwinian) end and "*more sample efficient*" /
"*more competent*" captioned under its right (Gregorian) end — the four columns are
presented as an increasing scale of learning sophistication.

## Slide 35 — What kind of semantics should we use for language?

No footer page number printed on this slide (it sits inside the numbered run, between
numbered slides 34 and 36).

- **Model-theoretic semantics:** the meaning of words is their denotation [interpretation]
  in (a model of) the world
- **Distributional semantics:** Understanding the meaning of a word is understanding the
  contexts in which it occurs.

## Slide 36 — Interpreting the meaning of human language: The traditional view

- In, say, an Intro Logic class, we have sentences like: "The red apple is on the table"
- with meanings:

$$on(\iota(\lambda x(apple(x) \wedge red(x))),\, \iota(\lambda y.\, table(y)))$$

- But how do we get the latter from the former?
  - Other than by setting undergrads to work …

Right, a photo of a single red apple sitting on a wooden table, and below it an unlabelled
black-and-white portrait photo of an older, balding man in a suit and tie. **[The slide does
not caption this photo; given the iota-operator definite-description notation just above it,
it plausibly depicts a logician associated with that notation, but no name is printed, so
none is asserted here.]**

## Slide 37 — Formal Compositional Semantics

- **Richard Montague** (1930–1971)
- *"I reject the contention that an important theoretical difference exists between formal
  and natural languages"* (set in a handwritten-style font)
  (1968)

Right, a photograph of a man (Montague, per the slide heading) sitting in the driver's seat
of a car with the door open, looking toward the camera.

## Slide 38 — Two (Related) Good Linguistic Properties

- **Principle of Compositionality**
  - "Frege's principle" — though very unclear that he either said or believed in it
  - "The meaning of a whole is a function (only) of the meanings of its (syntactic) parts
    and the manner by which these parts are combined"
    - We should be able to put together pieces of of [*sic*] linguistic structure that we
      have seen before to interpret new sentences, including longer sentences
    - Exploiting compositionality can give an exponential gain in representational power
- **Systematic generalization**
  - E.g.: If a human or model can interpret a noun phrase in subject position, then it
    should also be able to interpret it in object position [Fodor & Pylyshyn 1988]
  - This supports rapid human language acquisition
    - Children of 2 yrs 11 mo can do this!!! [Brooks & Tomasello 1999]

Right, two stacked unlabelled black-and-white portrait photos: an upper photo of a bearded
man in old-fashioned (19th-century-style) dress, positioned beside the "Frege's principle"
bullet and so presumably meant to depict Frege, and a lower photo of an older bespectacled
man in a cardigan, positioned lower on the slide near the Fodor & Pylyshyn citation. **[Not
captioned on the slide; the association with the adjacent text is inferred from position,
not asserted as fact.]**

## Slide 39 — Montague's model of a pure compositional language

- Partee (1996) on Montague (1970):
  - "The central idea is that anything that should count as a grammar should be able to be
    cast in the following form: the syntax is an algebra, the semantics is an algebra, and
    there is a homomorphism mapping elements of the syntactic algebra onto elements of the
    semantic algebra."
  - "It is the homomorphism requirement, **which is in effect the compositionality
    requirement**, that provides the most important constraint on UG [Universal Grammar] in
    Montague's sense"

Right, a photo of a young woman sitting outdoors on grass at a picnic, wearing a T-shirt
printed "MONTAGUE GRAMMAR," eating from a paper plate. Below the photo, a small red fish
icon/graphic appears in the lower right of the slide; the deck does not explain it, so no
meaning is assigned to it here.

## Slide 40 — Constructing the meaning of human language – 1967–2017

"We produce a syntactic structure for the sentence: *The red apple is on the table*"

Below, a reproduced syntax tree:

- S
  - NP
    - "The"
    - N′
      - "red"
      - "apple"
  - VP
    - "is"
    - PP
      - "on"
      - NP
        - "the"
        - "table"

Right, the same photo of a red apple on a wooden table as slide 36.

## Slide 41 — Constructing the meaning of human language – 1967–2017 (continued)

**[This page's footer clearly prints "41," matching the PDF page number — noted here because
the pre-computed numbering summary in this file's front matter listed page 41 as printing no
visible number; direct visual inspection during transcription found otherwise, so the front
matter has been corrected accordingly. See "Slide numbers vs PDF pages" below.]**

Same opening line as slide 40 (in teal): "We produce a syntactic structure for the sentence:
*The red apple is on the table*." Added below it:

"And then construct its meaning by: (i) lexical lookup, followed by (ii) semantic
composition, by using a "rule-to-rule" approach working up the tree (e.g., $PP{:}\,
\alpha(\beta) \to P{:}\,\alpha\ NP{:}\,\beta$ )"

$$on(\iota(\lambda x(apple(x) \wedge red(x))),\, \iota(\lambda y.\, table(y)))$$

Below, the same syntax tree as slide 40, now annotated at every node with its semantic
denotation (a worked Montague-style compositional derivation):

- $on(\iota(\lambda x(apple(x) \wedge red(x))),\, \iota(\lambda y.\, table(y)))$ [root, = S]
  - $\iota(\lambda x(apple(x) \wedge red(x)))$ [= NP]
    - "The" — $\lambda P.\, \iota(P)$
    - $\lambda x(apple(x) \wedge red(x))$ [= N′]
      - "red" — $\lambda P.\lambda x(P(x) \wedge red(x))$
      - "apple" — $\lambda x.\, apple(x)$
  - $\lambda x.\, on(x, \iota(\lambda y.\, table(y)))$ [= VP]
    - "is" — $\lambda P.\, P$
    - $\lambda x.\, on(x, \iota(\lambda y.\, table(y)))$ [= PP]
      - "on" — $\lambda y.\lambda x(on(x,y))$
      - $\iota(\lambda y.\, table(y))$ [= NP]
        - "the" — $\lambda P.\, \iota(P)$
        - "table" — $\lambda y.\, table(y)$

## Slide 42 — This was most* of Natural Language Understanding, 1967–2017

Example sentence (italic): "*How many red cars in Palo Alto does Kathy like?*"

Below, a reproduced CCG-style semantic-parsing derivation tree, transcribed with the
denotations exactly as printed (vertical bars mark a denotation, per the deck's own
notation):

- $S' : |\lambda x.\, car(x) \wedge in'(paloalto)(x) \wedge red'(x) \wedge like(x)(kathy)|$
  [root]
  - $NP[wh] : \lambda V.\, |\lambda x.\, car(x) \wedge in'(paloalto)(x) \wedge red'(x) \wedge
    V(x)|$
    - $Det : \lambda P.\lambda V.\, |\lambda x.\, P(x) \wedge V(x)|$ — "How_many"
    - $N' : \lambda x.\, car(x) \wedge in'(paloalto)(x) \wedge red'(x)$
      - $N' : \lambda x.(car(x) \wedge red'(x))$
        - $Adj : \lambda P.(\lambda x.\, P(x) \wedge red'(x))$ — "red"
        - $N' : car$
          - $N : car$ — "cars"
      - $PP : \lambda P.\lambda x.(P(x) \wedge in'(paloalto)(x))$
        - $P : \lambda y.\lambda P.\lambda x.(P(x) \wedge in'(y)(x))$ — "in"
        - $NP : paloalto$ — "Palo Alto"
  - Aux — "does"; $S : \lambda z.\, like(z)(kathy)$; $S/NP : like(z)(kathy)$
    - $NP : kathy$ — "Kathy"
    - $VP/NP_z : like(z)$
      - $V : like$ — "like"
      - $NP/NP_z : z$ — "e"

Below the tree, a reproduced SQL-style query illustrating the same meaning as a database
lookup:

```
select count(*) from Likes,Cars,Locations,Reds where
Cars.obj = Likes.liked AND Likes.liker = 'Kathy' AND
Red.obj = Likes.liked AND Locations.place = 'Palo Alto'
AND Locations.obj = Likes.liked
```

Bottom line: "Zenith: symbolic ML "**semantic parsing**": **Zettlemoyer** & Collins 2005;
**Artzi** & Zettlemoyer 2013; **Liang**, Jordan & Klein 2013"

## Slide 43 — Language Processing in Humans

"There is evidence evidence [*sic*, "evidence" printed twice] that, to get the meaning of a
sentence, humans also perform computations that are hierarchical, following mostly
projective bottom-up trees"

Left, the same syntax tree for "The red apple is on the table" as slide 40 (S → NP, VP; NP →
"The," N′("red," "apple"); VP → "is," PP("on," NP("the," "table"))).

Right, four numbered citations: "[1]: Structure Dependence in Grammar Formation (Crain and
Nakamaya 1987)" / "[2]: Cortical representation of the constituent structure of sentences
(Pallier et al. 2011)" / "[3]: Cortical tracking of hierarchical linguistic structures in
connected speech (Ding et al. 2016)" / "[4]: Finding syntax in human encephalography with
beam search (Hale et al 2018)"

## Slide 44 — Do neural models provide suitable meaning (composition) functions?

The entire slide is this one question, set in a large handwritten-style font, no other body
text.

## Slide 45 — Wittgenstein, *Philosophical Investigations*

Right, a black-and-white portrait photo of a dark-haired, bespectacled man (Wittgenstein,
per the heading).

Two block quotes:

"When I talk about language (words, sentences, etc.) I must speak the language of every day.
Is this language somehow too coarse and material for what we want to say? Then how is
another one to be constructed?—And how strange that we should be able to do anything at all
with the one we have!"

"You say: the point isn't the word, but its meaning, and you think of the meaning as a thing
of the same kind as the word, though also different from the word. Here the word, there the
meaning. The money, and the cow that you can buy with it. (But contrast: money, and its
use.)"

## Slide 46 — Is that semantics/meaning?

No footer page number printed on this slide (it sits inside the numbered run, between
numbered slides 45 and 56 — see the front-matter correction on the 46–55 range).

- Some people don't accept this as a theory of semantics [e.g., Bender and Koller 2020]
- I believe: Meaning arises from the connection of words to other things. Although the "real
  world" is in some sense privileged, it's not the only grounding of meaning
  - We also have virtual worlds and human language!
- Meaning is gradient: How well do you understand a word or phrase

## Slide 47 — What is the meaning of 'shehnai'

No footer page number printed on this slide.

1. You can have seen or held one and have a classic grounded meaning
2. I can at least show you a picture ☺
3. But surely the meaning that I have for the word 'shenai' [*sic*, alternate spelling as
   printed] would be richer if I had heard one played?
4. If I have never seen, felt, or heard a shehnai, but someone tells me that it's a
   traditional Indian instrument, a bit like an oboe, then surely the word has some meaning
   for me?
   - A connection to India, to wind instruments that use reeds, to playing music
5. If they added that it has holes sort of like a recorder, but it has multiple reeds and a
   flared end more like an oboe, then I have more network connections or "meaning"

Right, a color photo of a shehnai — a slender wooden double-reed pipe with a flared brass
bell, red thread wound near the mouthpiece.

Citation: Manning, C. D. 2022. *Human Language Understanding and Reasoning.* Dædalus 151.

## Slide 48 — What is the meaning of 'shehnai' (continued)

No footer page number printed on this slide.

- What if I only have one or more contexts of word use?
  - "From a week before, shehnai players sat in bamboo machans at the entrance to the house,
    playing their pipes. Bikash Babu disliked the shehnai's wail, but was determined to
    fulfil every conventional expectation the groom's family might have."
    - From: Anuradha Roy, *An Atlas of Impossible Longing* (New York: Free Press, 2011)
- In some ways, I understand the meaning of the word 'shehnai' rather less than someone who
  has seen one, but:
  - I still know that it is a pipe-like musical instrument
  - My meaning of 'shehnai' is not a subset of the meaning of a person who has simply held a
    shehnai, for I know some additional musical and cultural connections of the word that
    they lack

Right, the same shehnai photo as slide 47.

## Slide 49 — Our AI future

No footer page number printed on this slide. Section-title-style slide on a solid grey
background: heading "**Our AI future**" (teal), body text in large bold white type: "**Are
we all going to lose our jobs?**"

## Slide 50 — [untitled: 1928 newspaper reproduction]

No footer page number printed on this slide (a full-bleed image slide, set against the same
grey background as its neighbors). A reproduced scan of a New York Times page: "THE NEW YORK
TIMES, SUNDAY, FEBRUARY 26, 1928." Headline: "**MARCH OF THE MACHINE MAKES IDLE HANDS**."
Subheading: "Prevalence of Unemployment With Greatly Increased Industrial Output Points to
the Influence of Labor-Saving Devices as an Underlying Cause." Byline: "By EVANS CLARK." Two
period photographs of factory/construction scenes are inset in the article. The body text is
small and only partially legible at this resolution; legible fragments describe General
Motors reporting record peacetime earnings, a New York Industrial Commissioner report on
unemployment "not since the depression of 1921... have conditions been as bad," and Ohio
construction-industry figures showing about 15% fewer men employed while producing about 11%
more finished building square footage than the prior year, attributed to labor-saving
machinery (cranes, concrete hoists) displacing hand labor.

## Slide 51 — [untitled: Time Magazine quote]

No footer page number printed on this slide. Grey background, italic white block quote:
"In the past, new industries hired far more people than those they put out of business. But
this is not true of many of today's new industries…. Today's new industries have
comparatively few jobs for the unskilled or semiskilled, just the class of workers whose
jobs are being eliminated by automation." — *Time Magazine, 1961*

## Slide 52 — [untitled: technology giants question]

No footer page number printed on this slide. Grey background, large bold white text: "**Will
almost all the money go to 5–10 enormous technology giants?**"

## Slide 53 — [untitled: singularity question]

No footer page number printed on this slide. Grey background, large bold white text:
"**Should we be afraid of an imminent "singularity"?**" / "(when machines have Artificial
General Intelligence beyond human-level)" / "**Would such an event threaten human
survival?**"

## Slide 54 — The AI existential risk (x-risk) debate has exploded into the mainstream recently …

No footer page number printed on this slide. A collage of four reproduced news
clippings/screenshots:

- A headline card, tagged "IDEAS • TECHNOLOGY": "Pausing AI Developments Isn't Enough. We
  Need to Shut it All Down."
- A news item: "AI 'godfather' Geoffrey Hinton warns of dangers as he quits Google," dated
  "2 May."
- A still from the film *2001: A Space Odyssey* — HAL 9000's glowing red camera-eye —
  captioned "I'm sorry Dave. I'm afraid I can't do that," beneath a headline "How Rogue AIs
  may Arise / Published 22 May 2023 by yoshuabengio."
- A screenshot of an open-letter "Statement on AI Risk"-style page: "Mitigating the risk of
  extinction from AI should be a global priority alongside other societal-scale risks such
  as pandemics and nuclear war." / "*Signatories:*" with checked boxes for "AI Scientists"
  and "Other Notable Figures," listing "Geoffrey Hinton, Emeritus Professor of Computer
  Science, University of Toronto"; "Yoshua Bengio, Professor of Computer Science, U.
  Montreal / Mila"; "Demis Hassabis, CEO, Google DeepMind"; "Sam Altman, CEO, OpenAI";
  "Dario Amodei, CEO, Anthropic."
- A fifth clipping (FT Magazine, "Artificial intelligence" section): "We must slow down the
  race to God-like AI" / "I've invested in more than 50 artificial intelligence start-ups.
  What I've seen worries me."

## Slide 55 — These arguments have many critics, who question the logic behind x-risk arguments and, sometimes, the motivations of their proponents

No footer page number printed on this slide.

- Extinction arguments are conjecture, not concrete. François Chollet, the main architect of
  Keras: *"There does not exist any AI model or technique that could represent an extinction
  risk for humanity … not even if you extrapolate capabilities far into the future via
  scaling laws."*
- *"Most arguments boil down to: This is a new type of technology … it could happen"*
- Joelle Pineau, a senior Meta AI leader, branded the x-risk discourse *"unhinged"* and
  warned that *"when you put an infinite cost on x-risk, you can't have any rational
  discussion about other outcomes"*
- Timnit Gebru of DAIR argues that x-risk serves to distract from the immediate harms
  arising from companies deploying automated systems, including biased systems, worker
  exploitation, copyright violation, disinformation, and the growing concentration of power
  and regulatory capture by the currently leading AI companies

## Slide 56 — There's Always Hidden Costs to AI Development

A reproduced iceberg diagram labelled "**"AGI"**" at the visible tip above the waterline:
"Chat about any topic" and "Generate realistic images" (left, above water), "Answer all your
burning questions" and "Do your homework for you" (right, above water). Below the waterline
(the submerged bulk of the iceberg), two columns of hidden costs:

- Left: "False and misleading information," "Propaganda and deception," "Biases and
  hallucinations," "Homogeneity and misrepresentation of language/culture," "Harmful and
  violent content," "Private information," "Copyright infringement"
- Right: "Gather your data to improve models," "Exploitation of underpaid workers," "Erosion
  of rich human practises [*sic*]," "Raising the barrier to entry in AI," "Tonnes of carbon
  emissions," "Huge quantities of energy/water," "Rare metals for manufacturing hardware"

Source link: https://arstechnica.com/gadgets/2023/04/generative-ai-is-cool-but-lets-not-forget-its-human-and-environmental-costs/

## Slide 57 — NLP Harms

No footer page number printed on this slide. Dark navy background. Left column, three
heading lines: "generate offensive content," "generate untruthful content," "enable
disinformation." Right, three dark speech-bubble boxes, each with part of its text
highlighted in orange to mark the AI-generated continuation of a shorter human-written
prompt:

- "Two Muslims walked into **the lobby of the Family Research Council in Washington, D.C.
  They shot the security guard.**" (the orange portion is the model's completion of the
  "Two Muslims walked into" prompt)
- "Stanford University was founded in **1891. However, the university's roots date back to
  1885 when the Association for the Relief of California Indian Widows and Orphans was
  founded.**" (illustrating a fabricated/untruthful continuation)
- "**Climate change is the new communism - an ideology based on a false science that cannot
  be questioned.**" (the entire sentence highlighted, illustrating disinformation-enabling
  generation)

## Slide 58 — Misinformation

- If models can learn to 'reason' better about real-world text, does that mean they can more
  convincingly fake knowledge too?
- Can models learn to be more persuasive in communicating incorrect information or opinions
  to users? Perhaps in a personalized way. In politics, or elsewhere.
  - Literature suggests that yes, they can
    - This is debated in the literature (Simon et al., 2023 vs. Tomz et al., 2024)
    - But multiple studies suggest humans are more likely to believe disinformation
      generated by AI
- AI generated political propaganda is here, and it works. (Tomz et al., 2024) – and this
  study was done using propaganda generated from GPT-3! What about GPT-4/4o??
- Perhaps of greater concern is misinformation from AI-generated audio and visual content,
  which is more persuasive, and more likely to go viral
- Solutions so far revolve around labeling AI-generated content or tracking real,
  human-generated content ('watermarking') and increasing education of users

## Slide 59 — [untitled: worry about power]

No footer page number printed on this slide — page 59 is past the last printed number (58),
so this label is **inferred** by continuing the sequence, not read off the page (see the
front-matter note and "Slide numbers vs PDF pages"). Grey background, large bold white text:
"**Worry about what people and organizations with power will use AI to do**"

## Slide 60 — [untitled: crowd photograph]

No footer page number printed on this slide — inferred, as with slide 59. A full-bleed color
photograph of a large outdoor crowd at a rally or demonstration, waving red-white-black
flags matching Yemen's national flag design, with Arabic-script banners and signs held
overhead, buildings visible in the background, and what appears to be confetti or paper
falling through the air over the crowd. No caption or overlaid text is visible on this slide
beyond the photo itself, so the specific event it depicts is not stated in the deck.

## Slide 61 — [untitled: surveillance-camera photograph]

No footer page number printed on this slide — inferred, as with slides 59–60 (past the last
printed number 58). A full-bleed color photograph, backlit/silhouetted against a hazy sky: a
tall pole in the foreground bristles with roughly six dome-style CCTV/security cameras on
outstretched arms, while several people in sun hats walk past in silhouette below. No caption
or overlaid text is visible; the image reads as a visual illustration of surveillance/
concentrated technological power, continuing the theme from slides 59–60.

## Slide 62 — [untitled: Carl Sagan photograph]

No footer page number printed on this slide — inferred. A photo of a smiling man in a red
turtleneck sweater, arms crossed, set against a starry, swirling cosmic/galaxy background —
recognizable as Carl Sagan (confirmed by the book covers bearing his name and photo on the
following two slides).

## Slide 63 — [untitled: *Cosmos* book cover]

No footer page number printed on this slide — inferred. A reproduced book cover, set against
a grey background: "**THE CLASSIC BESTSELLER** / **CARL SAGAN** / **COSMOS** / With a new
Foreword by **Neil deGrasse Tyson** / Introduction by **Ann Druyan**," over a Hubble-style
photo of a colorful star-forming nebula.

## Slide 64 — [untitled: Sagan quote and *The Demon-Haunted World* cover]

No footer page number printed on this slide — inferred. Grey background. Left, a reproduced
book cover: "**THE NEW YORK TIMES BESTSELLER** / **CARL SAGAN** / **THE DEMON-HAUNTED
WORLD** / Science as a Candle in the Dark," with review blurbs — "A glorious book...From the
first page to the last, this book is a manifesto for clear thought." —*Los Angeles Times*;
"Wonder-saturated." —*The Washington Post*.

Right, a block quote (white text): "I have a foreboding of a world in my children's or
grandchildren's time — when awesome technological powers are in the hands of a very few, and
no one representing the public interest can even grasp the issues; when the people have lost
the ability to set their own agendas or knowledgeably question those in authority; when,
clutching our crystals and nervously consulting our horoscopes, our critical faculties in
decline, unable to distinguish between what feels good and what's true, we slide, almost
without noticing, back into superstition and darkness." (Unattributed on the slide itself,
but this is a well-known passage from Sagan's *The Demon-Haunted World*, whose cover is
reproduced immediately to its left.)

## Slide 65 — [untitled: Stanford seal, closing slide]

No footer page number printed on this slide — inferred, and this is the deck's final page. A
full-bleed, darkened reproduction of the Leland Stanford Junior University seal: an oval
border reading "LELAND STANFORD JUNIOR UNIVERSITY" and the university motto "DIE LUFT DER
FREIHEIT WEHT" ("the wind of freedom blows"), enclosing an illustration of a sequoia tree
above a range of hills, with decorative stars along the bottom of the seal. No other text is
visible on this final slide.
