# CS224N — Natural Language Processing with Deep Learning (Stanford, Spring 2024)

CS224N is Stanford's course on deep learning for natural language processing,
taught here by **Christopher Manning** in Spring 2024, with guest lectures from
several researchers. It builds from the bottom up: word vectors, feed-forward
networks, recurrent networks and attention, then the methods that define the field
today — transformers, encoder-decoder models, pretraining and post-training of
large language models, adaptation, interpretability, and agents. Manning's stated
goals are that students learn the methods, gain some real understanding of human
language and why it is hard for computers, and come out able to build working
systems.

> **Coverage note.** This knowledge base covers **lectures 1 to 18**: wiki pages,
> timestamped transcripts, and full slide-by-slide text for each. Slide *URLs* for
> lectures 1–18 are listed in [sources.md](sources.md), so questions about later
> lectures can be answered by pointing at the right PDF, but there are no transcripts,
> wiki pages, or slide text for them yet. See [TODO.md](TODO.md) for what remains.
>
> **Lectures 9 and 10 are Winter 2023 recordings**, unlike lectures 1–8 and 11–16. Spring 2024 had no
> Natural Language Generation lecture at all, and its Pretraining lecture is not the one in
> the playlist — so both decks come from the Winter 2023 course archive
> (`cs224n.1234`), not the Spring 2024 site. Lecture 9 is taught by **John Hewitt** and
> lecture 10 by **Xiang Lisa Li**, not by Manning. Lecture 10 also carries four different
> numbers (catalog position 10, video title "Lecture 11", deck title "Lecture 12", filename
> `lecture10-nlg`); this KB uses the catalog position throughout, and the slide file records
> all four. **Lectures 11 and 12 are back on the Spring 2024 track** and use the Spring 2024
> decks, but their titles are off by one against this KB's numbering: catalog position 11 is
> titled "Lecture 10 - Post-training" (Archit Sharma) and position 12 "Lecture 11 -
> Benchmarking" (Yann Dubois), and position 13 "Lecture 12 - Efficient Training" (Shikhar
> Murty). Repo files use the position; slide citations use each deck's own printed numbers.
> **Lecture 14** (Brain-Computer Interfaces, Chaofei Fan) continues the off-by-one — the video
> calls it "Lecture 13" — and is a guest research talk whose deck carries no lecture number at
> all. That deck also prints **no slide numbers on any page**, so its slide citations are PDF
> page positions; the slide file says so in its header. **Lecture 15** (Reasoning and Agents,
> Shikhar Murty) continues the same pattern — video and deck both title it "Lecture 14" — but
> its deck numbers cleanly, 1:1 with the PDF pages. **Lecture 16** (After DPO, Nathan Lambert,
> AI2) goes one step further — catalog and video call it "Lecture 15" while the deck's own title
> is simply "Life after DPO" — and also maps 1:1, though its footer prints the number in the
> **bottom-right** corner rather than the bottom-left this KB's other decks use.
> **Lecture 17** (ConvNets and TreeRNNs) is the last of the off-by-one lectures — video and deck
> both call it "Lecture 16" — and **lecture 18** (NLP, Linguistics, Philosophy) is where the
> offset closes, because the course's own lecture 17 appears neither in this playlist nor in the
> course site's slide directory, so position 18 and the deck's "Lecture 18" agree. Both are Manning's. Their printed numbers match the PDF pages
> 1:1, but each deck simply *stops* numbering before the end: lecture 17 prints 1–48 of 60 pages
> and lecture 18 prints 1–58 of 65, so the trailing slides are labelled by continuing the count
> and both slide files flag those labels as inferred.
>
> **Citing sources.** Prefer citing a **slide number** for anything on a slide
> (equations, tables, definitions) and a **timestamp** for anything Manning says aloud
> (asides, worked reasoning, answers to student questions). The slide files in
> `raw/slides/` carry the numbers; the transcripts carry the timestamps.
>
> **Mathematics.** Every equation on a wiki page is written in **LaTeX** — `$...$`
> inline, `$$...$$` displayed — in the course's own notation, and can be quoted to a
> learner as-is. The transcripts are the exception: they are a verbatim record of speech
> and spell notation out in words ("theta sub j minus alpha times…"), so quote the wiki
> page, not the transcript, when a learner wants the formula.

## Lecture pages

- [Lecture 1 — Intro and Word Vectors](wiki/01-intro-and-word-vectors.md) — what
  the course covers and how it is graded; why language matters and how neural NLP
  progressed from 2014 machine translation to GPT-2 and ChatGPT; denotational vs
  distributional theories of meaning; why WordNet and one-hot vectors fail; the
  word2vec setup, objective function, and the full hand-derivation of its gradient.
- [Lecture 2 — Word Vectors and Language Models](wiki/02-word-vectors-and-language-models.md)
  — stochastic gradient descent and random initialization; why word2vec uses two
  vectors per word; skip-gram vs CBOW; negative sampling and the unigram^(3/4)
  sampling trick; the gensim/GloVe notebook demo and analogies; co-occurrence
  counts, SVD and GloVe; intrinsic vs extrinsic evaluation; word senses; and the
  first neural classifier, a window classifier for named entities.
- [Lecture 3 — Backpropagation and Neural Networks](wiki/03-backpropagation-and-neural-networks.md)
  — how neural networks are trained, in two halves. The non-linearities lecture 2 left
  unexplained and why a network without them collapses to one layer; gradients, Jacobians
  and the chain rule; the worked hand-derivation of ∂s/∂**b** and ∂s/∂**W** and the δ
  error signal; the Jacobian form vs shape convention collision; then computation graphs,
  the upstream × local = downstream rule, the (*x*+*y*)·max(*y*,*z*) worked example,
  automatic differentiation and numeric gradient checking.
- [Lecture 4 — Dependency Parsing](wiki/04-dependency-parsing.md) — constituency vs
  dependency views of sentence structure; why human language is *globally* ambiguous and
  the four headline ambiguity types; heads, dependents, typed relations, projectivity;
  Pāṇini to Tesnière and the rise of treebanks and Universal Dependencies; the
  arc-standard transition system worked through on *I ate fish*; UAS and LAS; and why
  Chen & Manning's neural parser beat symbolic feature parsers on accuracy *and* speed.
- [Lecture 5 — Recurrent Neural Networks](wiki/05-recurrent-neural-networks.md) — the
  lecture Manning calls the most important in the class. Regularization and the modern
  view that overfitting no longer matters, dropout, vectorization, Xavier initialization
  and the adaptive optimizers; what a language model *is*, in two equivalent definitions;
  *n*-gram models, their sparsity and storage failures, and the "today the price of gold"
  generation demo; the fixed-window neural LM and why it lacks symmetry across positions;
  the RNN, its training by teacher forcing, backpropagation through time, and roll-out
  generation (including the paint-colour names); perplexity; and the vanishing/exploding
  gradient problem with its eigenvalue proof sketch.
- [Lecture 6 — Sequence to Sequence Models](wiki/06-sequence-to-sequence-models.md) —
  perplexity explained properly, with Jelinek's 64-sided die; the ~7-token limit of a
  simple RNN; the LSTM in full — cell state, three gates, the equations, the history from
  Hochreiter and Schmidhuber through Graves to Hinton at Google, and why "the + sign is
  the secret"; vanishing gradients as a general deep-network problem, with ResNet,
  DenseNet and HighwayNet; bidirectional and stacked RNNs; machine translation from the
  1950s code-breaking analogy through statistical MT to the 2014 neural breakthrough; and
  the seq2seq encoder-decoder model, trained end-to-end.
- [Lecture 7 — Attention, Final Projects and LLM Intro](wiki/07-attention-final-projects-and-llm-intro.md)
  — BLEU and how machine translation is evaluated; attention as the fix for the seq2seq
  bottleneck, worked through visually with *il a m'entarté* → "he hit me with a pie"; the
  history from Bahdanau's additive attention (2014) through Luong and Manning's
  multiplicative form (2015); the general definition of attention beyond MT; then a
  practical brief on choosing and scoping the CS224N final project, finding data and
  research topics, and getting a neural network to actually train.
- [Lecture 8 — Self-Attention and Transformers](wiki/08-self-attention-and-transformers.md)
  — is attention all you need? Why recurrence loses on linear interaction distance and
  parallelizability; self-attention as query-key-value lookup, and the three fixes
  (position representations, nonlinearities, future-masking) that make it a working
  building block; multi-head attention, scaled dot-product attention, residual
  connections and layer norm; the encoder, decoder, and encoder-decoder Transformer
  shapes; and the still-open drawbacks — quadratic compute, position representations,
  and how little most proposed modifications actually help.
- [Lecture 9 — Pretraining](wiki/09-pretraining.md) (John Hewitt) — where the course stops
  building task-specific networks and starts adapting general ones. The fixed-vocabulary
  problem and byte-pair encoding; Firth's *earlier* quote and why word2vec cannot satisfy it;
  the seven cloze examples and what each one teaches (trivia, syntax, coreference, lexical
  semantics, sentiment, reasoning — and *not* Fibonacci); the pretrain/finetune paradigm and
  the hedged explanation of why it works; pretraining for encoders, encoder-decoders and
  decoders in turn; parameter-efficient fine-tuning; and GPT-3, in-context learning,
  chain-of-thought and the Chinchilla scaling correction.
- [Lecture 10 — Natural Language Generation](wiki/10-natural-language-generation.md)
  (Xiang Lisa Li) — how you actually get good text out of a trained model. The open-endedness
  spectrum from machine translation to story generation; why maximum-probability decoding
  degenerates into repetition and fails to match human uncertainty; sampling, top-*k*, top-*p*
  (nucleus), typical and epsilon sampling, temperature and re-ranking; exposure bias and the
  four responses to it, ending at RLHF; the three families of evaluation metric and why all of
  them fall short; and the ethics of deploying generation systems.
- [Lecture 11 — Post-training](wiki/11-post-training.md) (Archit Sharma) — how a pretrained
  model becomes an assistant, in three escalating routes. The alignment gap, shown by GPT-3
  answering "explain the moon landing to a 6 year old" with four more questions; zero-shot and
  few-shot in-context learning and why they are emergent; chain-of-thought and the "let's think
  step by step" trigger; instruction finetuning, Flan-T5's bigger-model-bigger-gain table, and
  its three subtler limitations; then reward modelling from pairwise comparisons, the
  KL-penalized RLHF objective, the full DPO derivation, and what all of this costs — reward
  hacking, over-optimization, and whose preferences get encoded.
- [Lecture 12 — Benchmarking and Evaluation](wiki/12-benchmarking.md) (Yann Dubois) — how you
  know any of it worked. Why train, develop, select, deploy and publish each need a *different*
  metric; close-ended evaluation and its three traps; open-ended evaluation, the "Heck yes!"
  failure case, and the finding that reference-based metrics are only as good as their
  references; human evaluation's 67% inter-annotator agreement and 5% reproducibility; Chatbot
  Arena, LLM judges, AlpacaEval and length control; the three current regimes (perplexity,
  "everything", arena-like); and what is broken — MMLU's three incompatible implementations,
  contamination, benchmark saturation, English monoculture and judge bias.
- [Lecture 13 — Efficient Training](wiki/13-efficient-training.md) (Shikhar Murty) — the systems
  lecture, aimed explicitly at final projects: how to train a model that does not fit on the GPUs
  you have. Floating-point layouts and why pure FP16 collapses DistilBERT to 50.08% accuracy;
  mixed precision with master weights and loss scaling, then bfloat16, which removes the scaling;
  the per-parameter memory budget where Adam's optimizer state costs three times the model;
  DDP and its wasteful replication; the four MPI collectives and the identity that makes ZeRO
  stages 1 and 2 **free**; ZeRO stage 3 / FSDP taking 120 GB per GPU down to 1.9 GB; and LoRA —
  matching full finetuning of GPT-3's 175B parameters while training 4.7M.
- [Lecture 14 — Brain-Computer Interfaces](wiki/14-brain-computer-interfaces.md) (Chaofei Fan) —
  a guest research talk, and the only lecture where the model's input is neurons firing rather
  than text. The history from Caton's 1875 galvanometer through Berger's EEG to intracortical
  arrays; spike trains, cosine tuning curves and population decoding; the words-per-minute scale
  that all assistive communication is measured against; why speech is decoded as **phonemes**
  rather than articulator movements; participant T12's four arrays and the surprise that
  Broca's-area recordings carried almost nothing; and then a full NLP pipeline — why an
  encoder-decoder is *too powerful*, CTC, a GRU rather than a Transformer, beam search, an
  n-gram LM inside a 20 ms budget with a Transformer rescoring on top — reaching ~25% word error
  rate at 60–70 WPM, and under 1% in the 2024 follow-up. Closes on inner-speech decoding and the
  neuroethics it forces.
- [Lecture 15 — Reasoning and Agents](wiki/15-reasoning-and-agents.md) (Shikhar Murty) — two
  halves sharing one question: can a model trained to continue text do something over multiple
  steps? Reasoning first — the deductive/inductive/abductive taxonomy and the narrowing to
  *informal deductive* reasoning; chain-of-thought and zero-shot CoT; self-consistency, and the
  control showing it beats plain prompt-ensembling; least-to-most decomposition and its length
  generalization; Orca distilling GPT-4 explanations into a 13B Llama; and ReST fine-tuning a
  model on its own answer-filtered rationales until it beats human-written ones. Then the
  demolition: early-exit and corruption experiments suggesting rationales are often post-hoc, and
  counterfactual tasks — base-9 arithmetic, a world where corgis are reptiles, `ABCD → ABCDF` —
  where models collapse and human subjects barely move. The second half is agents: the pre-LLM
  history of semantic parsers, latent plans and RL; the factorization of a trajectory into
  transition dynamics and agent policy that recasts decision-making as causal language modeling;
  MiniWoB++, WebArena and WebLINX; BAGEL generating synthetic demonstrations by exploring and
  relabeling; LLaVA and Pix2Struct for acting on pixels instead of HTML; and the "prompting gap,"
  where GPT-4V types an email into a password field and cannot recover.
- [Lecture 16 — After DPO](wiki/16-after-dpo.md) (Nathan Lambert, AI2) — a guest lecture on
  post-training as it is actually practised, and the course's frankest account of research that
  did not work. The resource gap first: Meta bought ~1.5M preference comparisons for Llama 2, more
  than the ~800k ever collected on Chatbot Arena. Then RLHF's objective and its KL term; the
  Bradley-Terry leap from a pairwise probability to a scalar reward; why DPO won on
  *infrastructure* rather than accuracy, and the four months and one 5e-7 learning rate it took
  before Zephyr β convinced anyone. RewardBench — built because reward models had no evaluation —
  finds LLM-as-a-judge is not state of the art, that only its Chat Hard split resists saturation,
  and that a DPO model cannot serve as a reward model once its reference checkpoint is
  unavailable. Then an ablation where instruction tuning dwarfs everything after it, changing the
  *data* beats changing the algorithm, a 70B reward model improves on best-of-n yet flatlines
  downstream, and PPO beats DPO by 1.2% for an effort Lambert doubts is worth it. Closes on what
  "online" data means, D2PO, what Llama 3's blog post implies, and a Q&A on reward hacking.
- [Lecture 17 — ConvNets and Tree Recursive Neural Networks](wiki/17-convnets-and-treernns.md) —
  two architectures that lost, taught because "people find new ways to reinvent things." First
  half: the 1D convolution over $n$-grams worked through numerically, padding and wide
  convolutions, why max pooling is the right summary if a filter is a feature detector, stride,
  local and $k$-max pooling, dilation, `Conv1d`; then Yoon Kim's 2014 single-layer CNN with its
  softmax classifier and its dropout confound, the **word-vector fine-tuning pitfall** (words
  absent from your training set never move, so *plodding* is stranded on the wrong side of the
  boundary) and the channel-doubling fix; BatchNorm vs LayerNorm; size-1 convolutions as the
  ancestor of the Transformer's per-position feed-forward layer; and VD-CNN, 29 layers deep from
  raw characters. Second half: recursion in language via Hauser, Chomsky and Fitch, the TreeRNN's
  shared composition matrix and greedy parser, why one matrix cannot serve every syntactic
  relation, the RNTN's multiplicative tensor layer, the Stanford Sentiment Treebank, and the
  negation result Manning thinks Transformers still do not match.
- [Lecture 18 — NLP, Linguistics, Philosophy](wiki/18-nlp-linguistics-philosophy.md) — Manning's
  closing lecture, and the most argumentative page in this KB. The course's four big ideas; the
  open problems (memorization vs generalization — with an LSTM out-generalizing a Transformer on
  limited data — interpretability, the multilingual long tail, benchmark contamination, legal and
  clinical NLP, bias). Then an honest read on GPT-4: a passable T-alliterative sonnet, BCG
  consultants 40% higher quality with it, "not even close" to *New Yorker* fiction, and the FT's
  "models predict, they do not comprehend." Then the history — symbolic AI vs cybernetics, the
  physical symbol system hypothesis, 1958's Perceptron hype — and Manning's own position:
  **language is a symbolic system, but its processor need not be**. Then meaning: model-theoretic
  vs distributional semantics, Tarski and Montague, the full lambda-calculus derivation, semantic
  parsing to SQL, Wittgenstein against denotation, and the *shehnai* argument that meaning is
  gradient and comes from connections of many kinds. Closes on AI risk, arguing existential risk
  is overweighted and concentration of power under-weighted.

## Topic pages

- [word2vec](wiki/word2vec.md) — the algorithm in full: center and outside words,
  the average-negative-log-likelihood objective, why there are two vectors per
  word, the 80-million-parameter count, skip-gram vs CBOW, naive softmax vs
  negative sampling, and how negative samples are drawn. **Start here for anything
  about how word vectors are learned.**
- [Distributional semantics](wiki/distributional-semantics.md) — the idea the whole
  course rests on: meaning as context. Denotational semantics and its limits,
  WordNet's failures, why one-hot vectors are orthogonal and useless, what dense
  embeddings buy you, how high-dimensional spaces behave differently from 2-D, and
  the count-based lineage through SVD, latent semantic analysis and Rohde's COALS.
- [GloVe](wiki/glove.md) — the Stanford model, and the one whose vectors are used
  in the class demo. Why *ratios* of co-occurrence probabilities encode meaning
  components, the ice/steam/solid/gas worked example, and how taking logs turns
  that ratio into a log-bilinear model with linear meaning directions.
- [Gradient descent](wiki/gradient-descent.md) — the update rule and why the
  learning rate must be small; why plain gradient descent is never used and
  stochastic/mini-batch gradient descent is both faster *and* a better optimizer;
  why zero initialization breaks learning; and the calculus used in the lecture-1
  derivation, including the "observed minus expected" gradient form.
- [Softmax, the logistic function, and cross-entropy](wiki/softmax-and-cross-entropy.md)
  — exponentiate-then-normalize and why the name "softmax" is slightly misleading;
  the logistic function and why "sigmoid" is the looser term; and why
  cross-entropy loss with one-hot labels is exactly the negative log likelihood
  you have been minimizing all along. **Read this if PyTorch's
  `CrossEntropyLoss` confuses you.**
- [Word senses and polysemy](wiki/word-senses-and-polysemy.md) — the cost of one
  vector per word. The *pike* and *field* examples, per-sense clustering and the
  four senses of *jaguar*, why a single vector is a frequency-weighted
  superposition of senses, Manning's argument that discrete senses are the wrong
  model anyway, and the sparse-coding result that recovers senses from one vector.
- [Evaluating word vectors](wiki/evaluating-word-vectors.md) — intrinsic vs
  extrinsic evaluation, a distinction that recurs all course. Analogy scoring (and
  Manning admitting the demo was cherry-picked), similarity against human
  judgments with the actual score table, the model comparison from plain SVD up to
  GloVe, and named entity recognition as the downstream task.
- [Backpropagation](wiki/backpropagation.md) — the algorithm, reduced to its two ideas:
  the chain rule, and caching intermediate results. Computation graphs, the
  upstream × local = downstream rule at a single node, why gradients **sum** at outward
  branches, what `+`, `max` and `*` each do to a gradient, the same-big-O invariant,
  what frameworks automate and what they leave you to write, and numeric gradient
  checking. **Start here for anything about how a network is trained.**
- [Matrix calculus](wiki/matrix-calculus.md) — the by-hand version: gradient → Jacobian,
  multiplying Jacobians for the chain rule, the four Jacobians that do all the work, the
  δ error signal and why ∂s/∂**W** = δᵀ**x**ᵀ is an outer product, and the **shape
  convention** vs Jacobian form — which is the thing that most often confuses people, and
  what Assignment 2 expects.
- [Activation functions](wiki/activation-functions.md) — the *f* in *f*(**Wx** + **b**).
  Why a network without one collapses to a single linear transform, why the 1943 threshold
  unit could not learn, and the whole family: logistic, tanh (a rescaled logistic), hard
  tanh, ReLU and why its dead zone works anyway, Leaky/Parametric ReLU, Swish and GELU.
- [Dependency grammar](wiki/dependency-grammar.md) — heads and dependents, typed
  relations, trees and the fake ROOT, projectivity and the five ways to cope with
  crossing arcs, the history from Pāṇini through Tesnière, and why treebanks beat
  hand-written grammars — including the arrival of evaluation itself.
- [Transition-based parsing](wiki/transition-based-parsing.md) — the arc-standard system
  (stack, buffer, Shift/Left-Arc/Right-Arc) worked through step by step on *I ate fish*;
  UAS and LAS with the scoring example; the three problems with indicator features
  (sparse, incomplete, and 95% of runtime); and the Chen & Manning neural parser, its
  results table, and the graph-based alternative up to Dozat & Manning.
- [Syntactic ambiguity](wiki/syntactic-ambiguity.md) — why parsing is a prediction
  problem. The global-vs-local ambiguity argument against programming languages, the four
  headline types with their newspaper examples, Catalan growth in the four-PP *Wall Street
  Journal* sentence, and why this is what defeated hand-written grammars.
- [Language modeling](wiki/language-modeling.md) — the concept Manning calls the most
  important in the class. The two equivalent definitions and why the chain rule makes them
  the same, why it is not a 2022 invention, the old and new answers to why it matters, the
  four ways the course builds one, conditional language models, and the distinction that a
  recurrent neural network is *not* a language model. **Start here for anything about what
  an LM is.**
- [*n*-gram language models](wiki/n-gram-language-models.md) — how language models worked
  from 1975 to 2012. The Markov assumption and counting, the "students opened their"
  example and what discarding the proctor context costs, both sparsity problems with
  smoothing and backoff, storage, why *n* stopped at 5, the generation demo, and why
  "scale will solve everything" is an old story.
- [Recurrent neural networks](wiki/recurrent-neural-networks.md) — the architecture and the
  fixed-window problem it solves; the equations; advantages and the two disadvantages that
  shape the rest of the course; training with teacher forcing and 100-word segments;
  backpropagation through time and truncation; roll-out generation; sequence tagging,
  sentence encoding and encoder uses; and the bidirectional and stacked variants with the
  rule that bidirectionality cannot be used for generation.
- [LSTM](wiki/lstm.md) — parsing the name, the history from Hochreiter and Schmidhuber 1997
  through the Gers forget gate to Hinton at Google, hidden state vs cell state, the full
  gate equations, why the cell and hidden state are separate (with the *King of Prussia*
  example), and why the additive update gives ~100 timesteps of memory instead of ~7.
  **The + sign is the secret.**
- [Vanishing and exploding gradients](wiki/vanishing-and-exploding-gradients.md) — the
  chain-rule intuition, the eigenvalue proof sketch for W_h^ℓ, the printer/tickets example,
  the ~7-token rule of thumb, gradient clipping, and the generalization beyond RNNs to
  ResNet, DenseNet and HighwayNet — plus why RNNs are worse than deep feed-forward networks
  (the repeated matrix is the *same* one).
- [Perplexity](wiki/perplexity.md) — the standard LM metric. The evaluation logic, the
  formula, the identity perplexity = exp(cross-entropy), the warning about logarithm base,
  Fred Jelinek's 64-sided-die explanation of why it exists at all, and the full results
  table from Kneser-Ney at 67.6 down to LSTMs at 30.
- [Machine translation](wiki/machine-translation.md) — where NLP started. The 1950s
  code-breaking analogy and why it flopped, statistical MT and the Bayes-rule split into a
  translation model and a language model, why that split let the translation model stay
  dumb, the Aztec Empire sentence Google Translate kept failing, and the 2014 neural
  breakthrough.
- [Sequence-to-sequence and encoder-decoder models](wiki/seq2seq-and-encoder-decoder.md) —
  the two-RNN architecture, seq2seq as a conditional language model, training end-to-end
  with teacher forcing in the decoder, the "Conditioning = Bottleneck" note that attention
  later solves, why you cannot just stack layers on the source, and the tasks beyond MT
  the pattern covers.
- [Regularization and dropout](wiki/regularization-and-dropout.md) — L2 and the classic
  overfitting picture, and Manning's claim that modern practitioners do not believe it;
  why models are now trained to memorize the training set; dropout's mechanics at train and
  test time; and the three explanations of why it works, including the naïve-Bayes /
  logistic-regression middle ground.
- [Evaluating machine translation: BLEU](wiki/evaluating-machine-translation.md) — why
  translation can't be scored like classification, the *n*-gram-overlap-against-references
  recipe and its too-short-translation penalty, what BLEU numbers mean in practice, and the
  2013–2019 chart where neural MT overtakes statistical MT.
- [Attention](wiki/attention.md) — the fix for the seq2seq bottleneck: the query/key/value
  mechanism, worked through with *il a m'entarté* → "he hit me with a pie"; the three ways
  to score attention (Bahdanau's additive form, Luong-Pham-Manning's multiplicative form,
  and basic dot-product); and the general definition — a query attending to a set of
  values — that self-attention specializes next lecture. **Start here for anything about
  how attention works, before Transformers.**
- [Self-attention](wiki/self-attention.md) — attention within a single sequence: why
  recurrence loses on interaction distance and parallelizability, the query-key-value
  recipe, and the three fixes (position representations, a feed-forward nonlinearity, and
  future-masking) needed before it can replace an RNN.
- [Transformer](wiki/transformer.md) — the architecture built on self-attention: why it
  displaced recurrence, multi-head attention, scaled dot-product attention, residual
  connections and layer normalization, the encoder/decoder/encoder-decoder shapes
  (including cross-attention), and the still-open drawbacks (quadratic compute, position
  representations, and how little most proposed modifications actually help). **Start
  here for anything about how a Transformer is built.**
- [Final project guidance](wiki/final-project-guidance.md) — choosing the default vs. a
  custom CS224N final project, the five common project shapes, where to find a research
  topic and where to find data, the train/tune/dev/test discipline that keeps results
  trustworthy, and a checklist for getting a neural network to actually train.
- [Subword modeling](wiki/subword-modeling.md) — how modern NLP represents words it has never
  seen. Why a fixed vocabulary and a single `UNK` token throw away too much (with the Swahili
  conjugation table that makes the point), the byte-pair encoding algorithm, the `##`
  continuation convention, and the practical consequences — everything is a subword token,
  including punctuation.
- [Pretraining and fine-tuning](wiki/pretraining-and-finetuning.md) — the two-step recipe
  that defines modern NLP. What changed from pretrained word embeddings to pretrained whole
  models, the hedged account of why starting at $\hat{\theta}$ matters, which pretraining
  objective each of the three Transformer shapes can use (masked LM, span corruption/T5,
  language modeling), and parameter-efficient fine-tuning — prefix tuning and LoRA. **Start
  here for anything about how modern models are trained.**
- [BERT and masked language modeling](wiki/bert.md) — the pretrained encoder. Why
  bidirectional context rules out language modeling, the masked-LM objective, BERT's
  80/10/10 masking recipe and the reason for it, next sentence prediction and why it was
  dropped, the GLUE results table that changed the field, how fine-tuning actually attaches a
  classifier, the generation limitation, and RoBERTa and SpanBERT.
- [GPT and in-context learning](wiki/gpt-and-in-context-learning.md) — the pretrained
  decoder. Using one as a classifier versus as a generator, the GPT/GPT-2/GPT-3 line with
  their sizes, the `[START]`/`[DELIM]`/`[EXTRACT]` input format, in-context learning and how
  surprising it should be, chain-of-thought prompting as a scratch pad, and the Chinchilla
  table showing GPT-3 was "comically oversized."
- [Natural language generation](wiki/natural-language-generation.md) — the half of NLP whose
  output is language. The NLU/NLG split, the example tasks, and the **open-endedness
  spectrum** from machine translation to story generation that decides which decoding
  algorithm and which evaluation metric apply — plus which architecture goes with which end,
  and why that is a compute-budget argument rather than a hard constraint.
- [Decoding algorithms](wiki/decoding-algorithms.md) — turning a distribution into tokens.
  Why greedy and beam search degenerate into repetition (the self-amplification effect) and
  fail to match human uncertainty; *n*-gram blocking, unlikelihood training, coverage loss and
  contrastive decoding; sampling and the heavy tail; top-*k*, its two opposite failure modes,
  top-*p* (nucleus) as an adaptive *k*, typical and epsilon sampling; temperature; and
  re-ranking, including why low perplexity is the wrong thing to maximize. **Start here for
  anything about how text is generated.**
- [Exposure bias and teacher forcing](wiki/exposure-bias-and-teacher-forcing.md) — the
  train/test mismatch built into maximum-likelihood training of generation models, and the
  four responses: scheduled sampling, DAgger, retrieval augmentation, and reinforcement
  learning. Reward estimation and the danger of optimizing a metric, then RLHF and the
  pretrain → instruction-tune → RLHF pipeline behind ChatGPT.
- [Evaluating NLG](wiki/evaluating-nlg.md) — why measuring generated text is unsolved.
  Content overlap metrics and the "Heck yes!" failure case where a correct paraphrase scores
  0 and the opposite meaning scores 0.67; model-based metrics (Word Mover's Distance,
  BERTScore, BLEURT); MAUVE for open-ended settings and why it discretizes the embedding
  space; and human evaluation — the gold standard, and slow, expensive, inconsistent and
  precision-only.
- [Prompting and in-context learning](wiki/prompting.md) — getting a task out of a fixed model
  by writing the input carefully. Zero-shot learning as it emerged in GPT-2 (including the
  `TL;DR:` summarization trick), few-shot prompting and why the first example buys most of the
  gain, the word-unscrambling curves that make in-context learning look emergent, and the
  "dark art" of prompt engineering — jailbreaks, style incantations and all.
- [Chain-of-thought prompting](wiki/chain-of-thought.md) — making a model write out its
  reasoning before its answer. Few-shot CoT and the tennis-balls exemplar, the GSM8K scaling
  curves where CoT only helps at the top end, zero-shot "Let's think step by step" (17.7 → 78.7
  on MultiArith), and the trigger-phrase table whose winner was written by a language model.
- [Instruction finetuning](wiki/instruction-finetuning.md) — finetuning on (instruction, output)
  pairs across many tasks so a model generalizes to unseen ones. The Flan-T5 table where the
  gain grows with model size (+6.1 at 80M to +26.6 at 11B), Super-NaturalInstructions' 1.6K
  tasks, synthetic data from bigger models (Alpaca) and LIMA's less-is-more result — and the
  three limitations that motivate preference optimization.
- [Reward modeling](wiki/reward-modeling.md) — turning "which answer would a human prefer?" into
  a scalar you can optimize. Why direct ratings are uncalibrated, why pairwise comparisons are
  the fix, the Bradley–Terry loss in full, and the shift-invariance it introduces — which DPO
  later exploits. **Start here before RLHF or DPO.**
- [RLHF](wiki/rlhf.md) — the three-step pipeline behind InstructGPT and ChatGPT. The objective,
  why optimizing a *learned* reward invites hacking, the KL penalty and what justifies it, PPO
  at an intuitive level, the Stiennon summarization result where RLHF'd models beat human
  reference summaries at every size, and the over-optimization curve where predicted reward
  rises while actual preference collapses.
- [Direct preference optimization](wiki/direct-preference-optimization.md) — RLHF's effect from
  a binary classification loss. The closed-form solution to the KL-constrained objective, the
  inversion that expresses the reward via the policy, why the intractable partition function
  cancels in a Bradley–Terry *difference*, the final loss, and what the cancellation costs.
  **Start here for anything about DPO.**
- [Evaluating LLMs](wiki/evaluating-llms.md) — what the field measures and with what. The
  stage-by-stage requirements table, close- vs open-ended tasks, SuperGLUE, MMLU and BIG-Bench,
  human evaluation's documented failure modes, and the three current regimes: perplexity
  (ρ = −0.940 with downstream score), "everything" (HELM, Open LLM Leaderboard, HumanEval,
  agents) and arena-like. Plus efficiency, bias and multilingual coverage as dimensions.
- [LLM-as-a-judge](wiki/llm-as-a-judge.md) — using GPT-4 in place of an annotator. Chatbot
  Arena and why cost rules it out during development; the AlpacaFarm result that a model agrees
  with human majority *better than individual humans do*, explained by bias vs variance;
  AlpacaEval's 0.98 correlation at <$10; and the biases — length (win rate 22.9 → 64.3 from
  prompting alone), position, and self-bias.
- [Benchmark contamination and overfitting](wiki/benchmark-contamination.md) — the three ways a
  correctly computed number can be meaningless. Consistency (MMLU had three incompatible
  implementations scoring llama-65b at 0.637 and 0.488), contamination (GPT-4 solving 10/10
  pre-2021 Codeforces problems and 0/10 recent ones) with min-k% and exchangeability detectors,
  overfitting and the GSM1k/DynaBench mitigations, and the BLEU status-quo trap.
- [Mixed precision training](wiki/mixed-precision-training.md) — running the forward and backward
  passes in half precision while keeping the weight update exact. The bit layouts of FP32, FP16
  and bfloat16 and what range vs. precision buys; the two ways naive FP16 fails (more than half
  of all activation gradients underflow to zero; 1.0001 rounds to 1); master weights and loss
  scaling; the PyTorch `GradScaler`/`autocast` code and the bfloat16 version that drops it; and
  the DistilBERT table where pure FP16 collapses to chance. **Start here — the lecture's advice
  is to always use it.**
- [GPU memory for training](wiki/gpu-memory-for-training.md) — the arithmetic behind every CUDA
  out-of-memory error. The 16-bytes-per-parameter budget, why Adam's optimizer state costs 12 of
  them, the $(2+2+K)\Psi$ formula giving 120 GB for a 7.5B model, and the term the first version
  leaves out — activations, which scale with the batch size and which **no** ZeRO stage shards.
- [Collective communication](wiki/collective-communication.md) — the four MPI operations that
  multi-GPU training is built from: all-reduce, reduce-scatter, all-gather and reduce, each with
  what it moves and which scheme uses it. Includes the identity **all-reduce = reduce-scatter +
  all-gather**, which is the single fact that makes ZeRO stages 1 and 2 cost nothing.
- [Distributed data parallel](wiki/distributed-data-parallel.md) — the baseline multi-GPU setup:
  split the data, replicate the model, all-reduce the gradients. Why it parallelizes computation
  well and scales badly, with the 120 GB figure that motivates everything after it.
- [ZeRO and FSDP](wiki/zero-and-fsdp.md) — sharding what DDP replicates, in three stages:
  optimizer state (120 → 31.4 GB), gradients (→ 16.6 GB), parameters (→ 1.9 GB). Why the first
  two are **free**, how stage 2 avoids ever instantiating a full gradient, FSDP units and the
  FlatParameter, the prefetching timeline that hides the communication, and the two wrinkles —
  unit 0 is never freed, and the sharding policy is architecture-specific.
- [Parameter-efficient finetuning](wiki/parameter-efficient-finetuning.md) — freezing almost
  everything and training a small set of parameters instead. Why a frozen parameter costs 2 bytes
  against a trainable one's 16; the general $\Delta\phi(\Theta)$ formulation; the family compared
  on GPT-3 (BitFit, prefix methods, adapters, LoRA); and the efficiency-as-a-value case —
  compute doubling every 3.4 months, the accuracy-vs-efficiency paper counts, and CS234's 880
  kilowatt-hours.
- [LoRA](wiki/lora.md) — low-rank adaptation in full. The intrinsic-rank observation, the
  $W_0 + \alpha BA$ decomposition with every symbol defined, what $\alpha$ does, the code (and why
  $B$ is initialized to zero), no-added-inference-latency task switching, and the ablations that
  give the defaults: adapt the **query and value** matrices, **rank 8**, **alpha 1**. Rank 1 is
  already competitive. **Start here for anything about LoRA.**
- [Brain-computer interfaces](wiki/brain-computer-interfaces.md) — what a BCI is and the four
  parts every one of them has; the history from Caton (1875) through Berger's EEG (1924) and
  Lucier's brain-wave music (1965) to intracortical arrays; a table of what has actually been
  demonstrated, from 2D cursor typing to 0.99% word error rate; and the two problems that never
  go away — neurons are noisy and the recording drifts. **Start here for anything about BCIs.**
- [Neural recording technologies](wiki/neural-recording-technologies.md) — EEG on the scalp, ECoG
  on the cortical surface, and penetrating microelectrode arrays, ordered by how far in they go.
  The space-versus-time chart that organizes every method, why fMRI's ~1 s resolution destroys a
  1 ms code, what a single electrode actually reports, and why decoders must be recalibrated
  every session.
- [Neural population decoding](wiki/neural-population-decoding.md) — turning spikes into intent.
  Spike trains and firing rates, the cosine tuning curve, why one neuron cannot distinguish 120°
  from 240°, why two neurons still cannot once noise is admitted, and how the problem becomes a
  classification problem with decision boundaries over firing rates.
- [Connectionist Temporal Classification](wiki/connectionist-temporal-classification.md) — the
  loss for monotonic sequence problems with unknown alignment and a large length mismatch. The
  blank symbol and the merge-then-delete collapsing rule, why the blank is what makes "hello"
  spellable, the marginalization over all valid alignments, the CTC-specific wrinkle in beam
  search, and when *not* to use it.
- [Language models in decoding](wiki/language-models-in-decoding.md) — how an LM gets folded into
  a decoder that is reading a noisy signal. The fused $P(\mathbf{Y}\mid\mathbf{X})^{\alpha}
  P(\mathbf{Y}) L(\mathbf{Y})^{\gamma}$ objective, why a word insertion bonus is *necessary*
  rather than a tweak, and the two-speed arrangement — an n-gram inside a 20 ms budget because
  it is a memory lookup, a Transformer rescoring the n-best list afterwards.
- [Neuroethics](wiki/neuroethics.md) — the questions inner-speech decoding makes practical rather
  than hypothetical. Reading thoughts you chose not to say versus recovering memories lost to
  Alzheimer's; enhancement beyond natural function, and the observation that steroids and
  stimulants already pose the same question; and the partnership stance the lecture recommends
  instead of an answer.
- [Language model agents](wiki/language-model-agents.md) — the agent setting on one page: agent,
  environment, observation, action and the language instruction $g$; the three pre-LLM approaches
  (semantic parsing to logical forms, latent plans, direct RL); the trajectory factorization that
  makes a policy a next-token problem; ReACT as chain-of-thought in a loop; MiniWoB++, WebArena
  and WebLINX compared; where training data comes from; operating over pixels; and the prompting
  gap that still separates models from humans.
- [Counterfactual evaluation](wiki/counterfactual-evaluation.md) — how to ask whether a model is
  reasoning or remembering. Why benchmark accuracy cannot settle it, what "counterfactual" means
  here (distributional, not causal — base-9 addition is rare in training data), the three
  constructions used, why the human control group is what makes the argument stick, and the
  four-step recipe for applying it yourself.
- [Self-training and rationale distillation](wiki/self-training-and-rationale-distillation.md) —
  one recipe wearing three hats: generate with a model, filter, fine-tune, iterate. Orca (teacher
  rationales into a small model), ReST (a model on its own correct rationales), and BAGEL (an
  agent inventing its own demonstrations). The axis that distinguishes them is where the filter's
  signal comes from, and BAGEL's contribution is relabeling failures rather than discarding them.
- [RewardBench](wiki/rewardbench.md) — the benchmark for reward models: why local evaluation
  matters when Chatbot Arena takes a month, how it is built from manually written chosen/rejected
  pairs, its five categories, the Chat Hard metaphor example that trips models on the stars/moon
  association, the three safety patterns, and why a DPO model stops working as a reward model once
  its reference checkpoint is gone.
- [PPO vs DPO](wiki/ppo-vs-dpo.md) — the empirical comparison and what actually separates them.
  DPO's advantage is infrastructural; PPO's is about 1.2% on average at 13B, bought with an
  unbounded tuning surface and generation-during-training. Covers the full ablation, the two
  senses of "online" data (fresh generations, fresh labels), the DPO variants chasing it, and why
  online DPO is not free — prompt distribution matching.
- [Preference data](wiki/preference-data.md) — the datasets post-training runs on, and the binding
  constraint on open alignment research. ShareGPT's legal grey area, OpenAssistant and why nothing
  replaced it, UltraFeedback as the workhorse, HH-RLHF's useful noise, ~70% annotator agreement as
  possible signal rather than bug, and the routes beyond pairwise preferences (KTO's one-sided
  labels, Starling's k-wise, fine-grained attributes).
- [Convolutional neural networks for NLP](wiki/convolutional-neural-networks.md) — the mechanics
  in one place: the feature-map equation, narrow vs wide convolutions, multiple channels, max vs
  average pooling and the feature-detector intuition that justifies max, stride, local and
  $k$-max pooling, dilation, `Conv1d`, BatchNorm vs LayerNorm, size-1 convolutions, and the two
  landmark systems (Kim 2014, VD-CNN).
- [Tree recursive neural networks](wiki/tree-recursive-neural-networks.md) — recursive vs
  recurrent, why linguists think recursion is the defining property of language, the
  $p = \tanh(W[c_1;c_2] + b)$ composition with a score head, greedy bottom-up parsing worked
  through on "The cat sat on the mat," why a single shared matrix cannot model different
  syntactic relations, the RNTN's tensor layer, and the negation experiments — the one result
  where these models still beat Transformers.
- [Compositionality](wiki/compositionality.md) — "the meaning of a whole is a function of the
  meanings of its parts and how they combine," in both its logical form (Montague's homomorphism
  requirement, the lambda-calculus derivation) and its vector form (TreeRNN, RNTN). Covers
  systematic generalization and the Fodor & Pylyshyn test, why negation is the sharp empirical
  case, and the open question of whether Transformers compose at all. **Spans lectures 17 and 18.**
- [Formal semantics](wiki/formal-semantics.md) — the denotational theory of meaning that ran NLP
  from 1967 to 2017: Tarski's rejection of natural language, Montague's break from it, the
  parse → lexical lookup → rule-to-rule composition pipeline, semantic parsing compiled to SQL,
  why it was brittle, and what survives. The counterpart to
  [distributional semantics](wiki/distributional-semantics.md).
- [Symbolic and neural AI](wiki/symbolic-and-neural-ai.md) — the two research traditions and why
  the distinction is not decorative: McCarthy coining "artificial intelligence" to break from
  cybernetics, Newell and Simon's physical symbol system hypothesis and the strength of its
  "necessary" clause, the 1958 Perceptron hype, Manning's separation of the symbol system from
  its processor, signalling reliability as the reason language is discrete, what linguistics is
  for now, and language as a thinking tool (von Humboldt, Dennett's four grades).
- [Sentiment analysis](wiki/sentiment-analysis.md) — the course's default sentence-classification
  task. Why keyword matching reaches ~90% and then stops, the Stanford Sentiment Treebank's
  215,154 phrase labels and the four-point lift they give even a naive Bayes baseline, how each
  model in the course scores, and SST-2's later life as a GLUE task.
- [AI risks and harms](wiki/ai-risks-and-harms.md) — lecture 18's closing argument, laid out:
  the 1928 and 1961 automation panics, concentration of power as the risk Manning does weight,
  the x-risk debate with Chollet's, Pineau's and Gebru's objections (including the
  infinity-times-any-probability flaw), the present-harms list, and the Sagan quotation the
  course ends on.

## Raw materials

- [`raw/slides/`](raw/slides/) — **the full text of every slide, with slide numbers**,
  for lecture 1 ([40 slides](raw/slides/01-intro-and-word-vectors.md)), lecture 2
  ([47 slides](raw/slides/02-word-vectors-and-language-models.md)), lecture 3
  ([85 slides](raw/slides/03-backpropagation-and-neural-networks.md)), lecture 4
  ([49 slides](raw/slides/04-dependency-parsing.md)), lecture 5
  ([72 slides](raw/slides/05-recurrent-neural-networks.md)), lecture 6
  ([56 slides](raw/slides/06-sequence-to-sequence-models.md)), lecture 7
  ([73 slides](raw/slides/07-attention-final-projects-and-llm-intro.md)), lecture 8
  ([62 slides](raw/slides/08-self-attention-and-transformers.md)), lecture 9
  ([54 slides](raw/slides/09-pretraining.md)), lecture 10
  ([76 printed slides](raw/slides/10-natural-language-generation.md)), lecture 11
  ([99 printed slides](raw/slides/11-post-training.md)), lecture 12
  ([65 slides](raw/slides/12-benchmarking.md)), lecture 13
  ([65 slides](raw/slides/13-efficient-training.md)), lecture 14
  ([75 slides](raw/slides/14-brain-computer-interfaces.md)), lecture 15
  ([75 slides](raw/slides/15-reasoning-and-agents.md)), lecture 16
  ([86 slides](raw/slides/16-after-dpo.md)), lecture 17
  ([60 slides](raw/slides/17-convnets-and-treernns.md)) and lecture 18
  ([65 slides](raw/slides/18-nlp-linguistics-philosophy.md)). Each file opens
  with a section-to-slide-range table, then transcribes every slide in order — including the
  equations, the tables of numbers, the margin annotations, and prose descriptions of the diagrams
  and plots. For lectures 1–3, 5–9, 12, 13 and 15 **the printed slide number equals the PDF page
  number**, so "slide 28" is page 28. **Three decks are exceptions.** Lecture 4's printed numbers
  run 1–49 but the PDF has only 45 pages, because printed slides 4, 5, 8 and 13 were hidden in the
  source deck and never exported. Lecture 10 is worse: its printed numbers run 1–76 against 71 PDF
  pages, with printed 35, 41, 47, 54 and 66 absent, so the offset **accumulates** rather than
  being constant. Lecture 11 is the same shape: printed 1–99 against 94 PDF pages, with five
  numbers among 58–60, 64 and 84 absent. Both of those files carry a full page-to-slide mapping
  table, and everything in this KB cites their printed numbers. **Lecture 14 is a fourth kind of
  exception**: its deck prints no slide number on any page at all, so slide *N* there means PDF
  page *N* by construction, and the file's header says so rather than implying the deck numbered
  them. **Lecture 15's printed numbers match PDF pages 1:1 with no gaps**, same as the clean decks
  above — only page 1, the title, prints no number. **Lecture 16 also maps 1:1, but is worth a
  specific note**: it prints its slide number in the **bottom-right** corner ("Life after DPO |
  Lambert: N" on content slides, a bare N on section dividers) rather than the bottom-left this
  KB's other decks use, which is why the repo's `slide_number_map.py` — whose corner heuristic
  scans the bottom-left — initially reported the deck as carrying no printed numbers at all,
  before the mapping was checked by eye against the PDF text layer; page 1 again prints none.
  **Lectures 17 and 18 are a fifth kind of exception**: both map 1:1, but each deck stops printing
  numbers before it ends — lecture 17 prints 1–48 across 60 pages and lecture 18 prints 1–58
  across 65 — so the trailing slides carry labels inferred by continuing the count, and both files
  say which those are rather than presenting them as printed. Lecture 18 has one further quirk
  worth knowing if you re-check it with a script: page 32's footer splits into two text spans with
  the "2" positioned below the page edge, so an extractor reads it as "3" and reports a numbering
  gap that is not there. One further caveat: **lecture 6's slides 4–18 repeat lecture 5's slides 49–63** as a recap, and are
  transcribed in brief with pointers rather than in full — the exception is slide 15, which adds
  the "~7 tokens back" rule of thumb that slide 25 then contrasts LSTMs against. Use these files
  when a learner asks where something is in the slides, wants an equation exactly as written, or
  when the transcript is unclear — the slides are the authority. Note that three slides in lecture
  10's ethics section (70, 73, 74) reproduce hate speech, sexual violence and profanity in full on
  the deck; the slide file states what each figure shows and cites the source paper but does not
  reproduce the passages.
- [`raw/transcripts/`](raw/transcripts/) — lecture transcripts for lectures 1 to 18,
  grouped into paragraphs each prefixed with an `[MM:SS]` timestamp. Use these to
  point a learner at the exact moment something is explained ("Manning covers this
  around 42:00"), or to quote him directly — they read as sentences, so they quote
  cleanly. These are auto-generated captions that have been **copy-edited**:
  punctuation and sentence boundaries added, filler and false starts removed, and
  mis-heard vocabulary restored (*word2vec* arrived as "word Tove" and "word DEC",
  *CBOW* as "sibo", *COALS* as "Kohl's", *ReLU* as "value", *tanh* as "10 H", and in
  lecture 4 *parsing* itself as "paing" and *parser* as "paa"; in lectures 5 and 6
  *n-gram* as "engram", *Xavier* as "harier", *Hadamard* as "hadam mod", *eigenvalue* as
  "ion value", and the names *Bengio*, *Jelinek*, *Feigenbaum*, *Hochreiter*, *Gers* and
  *Olah* all beyond recognition; in lectures 7 and 8, *BLEU* arrived as "blue", *bake-off*
  as "boff", *Bahdanau, Cho, and Bengio* as "B hour Al"/"dimma bad now K huno and Yoshua
  Benjo", *Luong* as "tanglong", the French toy sentence *il a m'entarté* as "the a is a
  sort of perfect past um exiler", and *minBERT* as "bir"; in lectures 9 and 10, *word2vec*
  again as "word to VEC" and "where to back", *UNK* as "ankh", *BERT* as "Birch" and "burp
  model", *RoBERTa* as "Brita", *Iroh* as "Ira" and "IRL", *NLG itself* as "an LG", "analogy"
  and "energy", *autoregressive* as "other aggressive", *n-gram* as "unground" and "engram",
  *nucleus sampling* as "nuclear sampling", *MAUVE* as "mouth score" and "small score",
  *BLEURT* as "Port", *RLHF* as "rlhs", and *ChatGPT* as "chaiji 50"; in lectures 11 and 12,
  *ChatGPT* again as "CH GPD" and "chargy GPD", *InstructGPT* as "instruct gbt", *DPO* and
  *PPO* as "DP", "BP" and "po", *Bradley–Terry* as "Brad lary", *Kullback–Leibler* as "cbak LI
  Li", *MMLU* as "mlu", *PaLM* as "power models", *SuperGLUE* as "super clue", *BLEU* as
  "blur", *BERTScore* as "bir", *AlpacaEval* as "Paka eval" and "back a EV", *Chatbot Arena* as
  "chadbad Arena", *MLPerf* as "ml puff", *DiscrimEval* as "dis remal" and *Anthropic* as
  "entropic"; and in lecture 13 the systems vocabulary throughout — *Adam* as "adom", *bfloat16*
  as "B float 16" and "b flat 16", *FSDP* as "Fs DP", *shard/sharded/sharding* as "shot",
  "shoted" and "Shing", *LoRA* as "Laura" and "Lura", *batch size* as "bat size", *DistilBERT* as
  "dist bir", *GradScaler* as "grad scalar", *reduce-scatter* as "reduce CER", *gradients* as
  "Radiance", and *PyTorch* as "pyos"; and in lecture 14 the entire clinical and neuroscience
  vocabulary — *brain-computer interface* as "bre computer interface" and "green computer
  interface", *BCI* as "PCI", *ALS* as "AOS", *motor cortex* as "modor cortex" and "model
  cortex", *orofacial* as "artificial", *phoneme* as "fum", "PHS" and "volume", *CTC* as "CDC",
  *GRU* as "Gru Gator recurring unit", *n-gram* as "angram", *ECoG* as "EOG", *fMRI* as "Mi",
  *Broca's area* as "broadcast area", *Hans Berger* as "hansburg", *Richard Caton* as "Richard
  Kon", and *Frank Willett* as "Frank wet"; in lecture 17 *the attention function* as "this
  tension function", *Yoon Kim* as "Yun Kim", *Conneau et al.* as "Cano Al", the whole $n$-gram
  vocabulary as "engram"/"byr"/"Tri"/"forr", and the sentiment examples *plodding* and *spice* as
  "plotting" and "space"; and in lecture 18 the philosophers and linguists the lecture is built
  on — *Montague* as "montigue", *von Humboldt* as "vilhelm Von Hot", *Tarski* as "Alfred tasi",
  *Wittgenstein* as "viken Stein", *Dennett* as "Daniel dennet" with his four grades as "dar
  wian"/"scaran"/"paparian", *Barwise* as "John barwise", *Wiener* as "Norbert weer",
  *Minsky* as "Marvin miny the Teeny", *Chollet* as "franois charal", *Gebru* as "Tim n jeu",
  *Sagan* as "Carl San", and *shehnai* throughout as "Shai"). No content was added,
  removed or reordered, and every timestamp is preserved. Mathematical notation in
  lectures 7–11's and 13's transcripts stays spelled out (bold for vectors/matrices, Unicode
  subscripts), matching lecture 3's convention — LaTeX is wiki-only. Student questions
  are marked in italics. Where a garble could not be resolved from the slides, the text
  carries an inline `[Ed: …]` note saying so instead of guessing — treat those as known
  gaps, and prefer the slide. Each header notes what remains unreliable in that lecture.
- [`raw/transcripts/original/`](raw/transcripts/original/) — the untouched verbatim
  captions, kept only for reference. **Prefer the edited transcripts above**; reach for
  these only to check exactly what the speech recognizer produced.
- [`sources.md`](sources.md) — the full inventory of course documents, with a
  canonical URL for each: lecture slides for Spring 2024's lectures 1–18, the two
  **Winter 2023** decks that go with catalog lectures 9 and 10 (catalog lectures 11, 12, 13, 14,
  15 and 16 use the Spring 2024 `lecture10-prompting-rlhf`, `lecture11-evaluation-yann`,
  `lecture12-training-shikhar`, `lecture13-speech-bci`, `lecture14-agents-shikhar-updated` and
  `lecture15-life-after-dpo-lambert` decks, catalog lecture 17 the `lecture16-CNN-TreeRNN` deck,
  and catalog lecture 18 the `lecture18-nlp-linguistics-philosophy` deck — the course's own
  lecture 17 is absent from both the playlist and the course site's slide directory, which is
  why the numbering offset closes at position 18),
  supplementary readings
  (the 2019 course notes, the gradient and differential-calculus reviews, the
  self-attention and transformers notes, the Python review), the assignment 2–4
  handouts, the final-project handouts, and 43 further papers the syllabus links
  to on arxiv and the ACL Anthology. **The slides are the authority wherever the
  transcript is garbled**, especially for mathematical notation — cite the URL and
  let the learner open it.

  The PDF binaries are deliberately not committed here. This knowledge base is
  read by an agent that navigates markdown and cannot extract anything from a
  PDF blob in the repo, so the canonical course URL is the useful artifact and
  committing the decks would have added over 100MB to every clone. Wiki pages link
  slides directly at `web.stanford.edu`.

## How this KB is organized

See [AGENTS.md](AGENTS.md) for the conventions — relative links only, every claim
traceable to a transcript timestamp or a slide, LaTeX for all mathematics, and never
inventing course content.
Where a transcript is genuinely unclear, the pages say so rather than filling the
gap.
