# Final project guidance

Practical advice on choosing, scoping, and executing a CS224N final project, and on
getting a neural network to actually train — covered in the second half of
[lecture 7](07-attention-final-projects-and-llm-intro.md) (slides 29–72), after the
technical content on [attention](attention.md). Grading weights: assignments 48%, the
final project 49% (proposal 8%, milestone 6%, poster 3%, report 32%), participation 3%
(slide 29).

## Default vs. custom

Every student picks one of two paths (slides 30–35). The **default final project** —
building and extending a minimal BERT implementation ("minBERT"), fine-tuned for sentiment
analysis — is open-ended within a guided scope: you get a leaderboard and a clear starting
point, and can extend it with ideas like contrastive learning, paraphrasing, or low-rank
adaptation. It's the right choice if you have limited research experience, don't have a
clear project idea of your own, or want a concrete goal to work against. Historically,
about half of students choose it, including some who start on a custom project and switch
after a few weeks once their original idea stops working out (≈45:19).

The **custom final project** is the right choice if you already have a research idea
you're excited about, want the experience of defining your own research goal end-to-end —
finding data, finding tools, and working out how to evaluate success — or want to try
something no one has guided you through before. It has to substantively involve **both**
human language and neural networks, since this is the NLP class; it doesn't have to be
*only* about human language (visual-language or music-and-language combinations are fine),
but human language can't be trivial to the project (≈46:05).

**Neither path is the easier route to a good grade.** The default project is more guided,
but graded to the same standard of effort; there's more variance in custom-project quality
(both the best and the worst projects tend to be custom), so the failure mode to avoid is a
custom project that ends up looking weaker than what a default project would have produced.
Best Project Awards go to both kinds every year (slide 35).

**Team size** is 1–3, and grading scales expectations with team size — a bigger team is
expected to do proportionately more work, though quality and scope still matter more than
raw headcount. In practice, some of the very best projects each year are solo efforts by
someone with a clear, well-scoped idea; the failure mode for a solo project is trying to
do *too much* and only getting partway through, rather than doing something narrower but
complete. Projects shared with another class, or done as part of an RA-ship or PhD
rotation, are expected to represent proportionately more work too, and must be disclosed
in the proposal (≈39:53–42:12).

## Project types

Not exhaustive, but most projects fall into one of five shapes (slide 43):

1. Take an application or task of interest and explore how to approach or solve it
   effectively, often with an existing model — this could be a task "in the wild," or,
   more commonly, an existing Kaggle competition, bake-off, or shared task.
2. Implement a complex (perhaps novel) neural architecture and explore its performance on
   some data.
3. Explore prompting (in-context learning) or building a program that orchestrates a
   language model. Types 1–3 are all expected to run real experiments with numbers and
   ablations.
4. An analysis or interpretability project — study how a model represents linguistic or
   world knowledge, or what kinds of errors or phenomena it handles well or badly.
5. A rare theoretical or linguistic project, showing some non-trivial property of a model
   type, dataset, or data representation.

## Finding a topic

Two starting points cover essentially all of science (slide 42): the **[nails]**
approach — start from a problem you want to make progress on, and find good ways to
address it — or the **[hammers]** approach — start from a technique you're interested in,
and find good ways to extend, improve, understand, or apply it.

Manning relays two often-quoted pieces of advice on *where* to look (slide 53): Turing
Award winner Ed Feigenbaum, quoting his own advisor Herbert Simon — *"If you see a research
area where many people are working, go somewhere else"* — and, on where to go instead,
Wayne Gretzky's *"I skate to where the puck is going, not where it has been."* Concretely,
worth checking: the [ACL Anthology](https://aclanthology.org/) and major ML conference
proceedings (NeurIPS, ICML, ICLR); past CS224N project reports; and arXiv. But some of the
best and most fun projects come from finding your *own* interesting problem in the world —
a website with unusual text you'd like to extract information from, for instance — rather
than chasing an existing leaderboard, since leaderboard-chasing tends to mean doing
marginally better on a problem someone else already solved, rather than something more
original (≈1:10:04–1:10:51).

**Past example projects**, referenced by name in the lecture (slides 44–50): "Deep
Poetry" (Xie, Rastogi, Chang) — a gated LSTM generating Shakespearean-sonnet-style text; a
reimplementation of DeepMind's then-unreleased Differentiable Neural Computer (Carol Hsin)
— finished and working just before the deadline; "Word2Bits" (Lam) — 1–2-bit quantized
word vectors that *outperform* full-precision ones on some tasks; an RNNLM improvement
tying the input embedding and output-projection matrices together (Inan and Khosravi,
published at ICLR 2017); fine-tuning CodeLlama-7B on synthetic data for Fortran code
generation with parameter-efficient fine-tuning (Govande, Kang, Shi); and AI-driven fashion
cataloging, fine-tuning a vision-language model to describe product images (Ma, Gopinath).

## The world has changed: pre-trained models by default

For most of the 2010s, a typical good CS224N project proposed a new architectural idea —
"add attention in a new place," a new layer — and could plausibly get close to the
state of the art from scratch. In the last several years that's become much harder: most
practical work, including professional research, now starts from an existing large
pre-trained model, which fixes most of your architectural choices for you. For nearly any
practical project, the sensible default is to load a pre-trained model via **Hugging Face
Transformers** and build your contribution on top of it — building your own architecture
from scratch is really only worthwhile as a small, targeted exploration (e.g. testing one
new nonlinearity), not as the basis of a full project (lecture 7, ≈1:12:22–1:13:55).

That includes using large language models directly, via API: you're welcome to use
GPT-4, Gemini, or Claude in your project, but only through API access — you cannot
realistically train your own model at that scale, and in many cases can't even load one
locally (you can typically load a 7B-parameter open model, not a 70B one, on the GPUs
available to you). Together AI provided \$50 of API credit per team as of this run of the
course. Whatever you use it for — in-context learning, prompting, chain-of-thought,
building a larger program around an LLM component — remember you'll be evaluated on what
*you* contributed on top of the model, not on the model's own output: "I ran this through
GPT-4 and it produced great summaries" is not, by itself, an interesting research project
(slides 54–58).

## Finding data

Collecting your own data is welcomed but risky on a course timeline — it's easy for it to
eat most of your available time, so scope it carefully if you go that route. Most projects
instead use an existing, curated dataset, which gives a faster start and built-in prior
work and baselines to compare against (slide 59). Places to look: **Hugging Face
Datasets**, **Papers with Code Datasets**, the **Linguistic Data Consortium** (Stanford has
an institutional license — LDC data is for Stanford purposes only), **Universal
Dependencies** for parsing data, and statmt.org for the WMT shared-task machine translation
data (slides 60–65).

## Train / tune / dev / test discipline

A methodological point the lecture treats as important enough to dwell on (slides 66–68):
the **train**, **tune**, **dev**, and **test** sets need to stay completely distinct
throughout a project.

- **Train** — what the model's parameters are fit on.
- **Tune** — an independent set for setting hyperparameters. Using the training set for
  this won't set hyperparameters correctly.
- **Dev (development / validation)** — what you check progress against as you iterate. If
  you check against it a lot, you effectively start "training" on it too, in the sense of
  learning what does and doesn't work for that particular set — hence sometimes keeping a
  second dev set, **dev2**, once the first has been leaned on heavily.
- **Test** — evaluated **only at the end**, and ideally only once. Reporting results on
  material the model was trained on gives a falsely good number — models almost always
  overfit their training data to some degree, and a fixed, held-out test set is the only
  way to get a trustworthy measure of generalization.

Always have an appropriate **baseline** to compare against: either a prior system that did
the same thing, or — for something more novel — the simplest reasonable approach you can
think of (e.g. averaging word vectors and taking a dot product, as a baseline for a
learned text-similarity model). Without one, a good-looking number has no way to be judged
good (slide 38, ≈58:28).

## Getting a network to train

A short, practical checklist, offered in the spirit of "neural networks want to learn — if
one isn't, you're doing something to stop it" (slides 69–72):

- **Work incrementally.** Get a very simple model working first; it's much harder to debug
  a complex model that's broken than to add complexity to one that already works.
- **Start tiny.** Run on 4–8 examples before running on real data — bugs show up faster,
  and training is nearly instant. You should be able to get to ~100% accuracy on this tiny
  set; if you can't, the model is either broken or not powerful enough.
- **Then scale up.** On the full training set, you should still get close to 100% training
  accuracy after optimization. Overfitting the training set is *not* itself something to
  fear in deep learning — these models tend to generalize despite it, because of how
  distributed representations share statistical structure. Once training accuracy looks
  right, use regularization — generous dropout is usually the biggest lever, more so than
  L2 — to bring dev performance in line.
- **Look at the actual data and the actual outputs.** Collect summary statistics on your
  data; do error analysis on your model's predictions. Hyperparameter tuning, learning
  rate, and initialization often matter more than they seem like they should.

## Two later lectures add project advice

**Read your model's output.** Lecture 10 flags this explicitly as a hint for final projects
(≈1:11:39): every automatic generation metric is a flawed proxy, so "the best judge of the
output quality is actually you." Look at what your system generates rather than reporting a
BLEU number and stopping. Slide 67 of that lecture adds a second habit worth adopting — publicly
release large samples of your system's output. See [evaluating NLG](evaluating-nlg.md).

**Adapt cheaply.** Lecture 9 covers parameter-efficient fine-tuning — prefix tuning and LoRA —
as things "you should know for your final projects, and in the world ahead" (≈55:35). They let
you adapt a large pretrained model while training a small fraction of its parameters, which
matters when you are working within a course GPU budget. See
[pretraining and fine-tuning](pretraining-and-finetuning.md#full-versus-parameter-efficient-fine-tuning).

Lecture 10's ethics section applies here too: the smaller open models you are likely to use
carry fewer safeguards than the commercial APIs, so toxic degeneration is more likely, not less
(≈1:13:58).

## Related pages

- [Attention](attention.md) — the technical content lecture 7 covers before this material.
- [Regularization and dropout](regularization-and-dropout.md) — the "generous dropout"
  recommendation above, covered in full.
- [Pretraining and fine-tuning](pretraining-and-finetuning.md) — the pretrained models this
  advice assumes, and how to adapt them cheaply.
- [Evaluating NLG](evaluating-nlg.md) — why you should look at your generations yourself.
- [Lecture 7 — Attention, Final Projects and LLM Intro](07-attention-final-projects-and-llm-intro.md)
- [Lecture 9 — Pretraining](09-pretraining.md)
- [Lecture 10 — Natural Language Generation](10-natural-language-generation.md)
