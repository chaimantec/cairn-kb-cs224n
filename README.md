# CS224N Knowledge Base

A machine-readable knowledge base for **Stanford CS224N — Natural Language Processing
with Deep Learning** (Spring 2024, taught by Christopher Manning): interlinked wiki
pages, timestamped lecture transcripts, and the full text of every slide.

It was built for [**Cairn**](https://cairnstudy.com), whose in-extension AI chat reads it
to answer questions about the course with citations — but it is plain markdown in a public
repo, so **any agent can use it**. There is no API, no build step, no authentication and
no JavaScript.

👉 **Start at [INDEX.md](INDEX.md).**

## Any agent can read this

The repo is deliberately boring, which is what makes it portable:

- **Plain UTF-8 markdown**, readable as-is. Nothing is generated at request time.
- **[INDEX.md](INDEX.md) is the fixed entry point** — a course summary plus an annotated
  table of contents, with a one-line description of every page so an agent can pick the
  right file without reading all of them.
- **Relative links only** between pages, so the whole graph can be walked from `INDEX.md`
  by following links. Course PDFs are the one exception and are linked at their canonical
  `web.stanford.edu` URLs.
- **Fetchable unauthenticated** over `raw.githubusercontent.com`, or just `git clone` it —
  the whole repo is a few megabytes because the PDFs are not committed.
- **Stable citation anchors.** Every claim is traceable to either a **slide number** or an
  **`[MM:SS]` timestamp**, so an agent can tell a learner *where* to look, not just what
  the answer is.

[AGENTS.md](AGENTS.md) is the schema: how the wiki is organized and the conventions every
page follows. Read it before writing to the repo, and it is a useful orientation before
reading from it too.

## What's covered

**Lectures 1–18**, in full — wiki pages, edited transcripts and slide-by-slide text for
each:

| # | Lecture | Slides | Transcript |
| --- | --- | --- | --- |
| 1 | [Intro and Word Vectors](wiki/01-intro-and-word-vectors.md) | [40](raw/slides/01-intro-and-word-vectors.md) | [103 ¶](raw/transcripts/01-intro-and-word-vectors.md) |
| 2 | [Word Vectors and Language Models](wiki/02-word-vectors-and-language-models.md) | [47](raw/slides/02-word-vectors-and-language-models.md) | [102 ¶](raw/transcripts/02-word-vectors-and-language-models.md) |
| 3 | [Backpropagation and Neural Networks](wiki/03-backpropagation-and-neural-networks.md) | [85](raw/slides/03-backpropagation-and-neural-networks.md) | [95 ¶](raw/transcripts/03-backpropagation-and-neural-networks.md) |
| 4 | [Dependency Parsing](wiki/04-dependency-parsing.md) | [49](raw/slides/04-dependency-parsing.md) | [102 ¶](raw/transcripts/04-dependency-parsing.md) |
| 5 | [Recurrent Neural Networks](wiki/05-recurrent-neural-networks.md) | [72](raw/slides/05-recurrent-neural-networks.md) | [102 ¶](raw/transcripts/05-recurrent-neural-networks.md) |
| 6 | [Sequence to Sequence Models](wiki/06-sequence-to-sequence-models.md) | [56](raw/slides/06-sequence-to-sequence-models.md) | [100 ¶](raw/transcripts/06-sequence-to-sequence-models.md) |
| 7 | [Attention, Final Projects and LLM Intro](wiki/07-attention-final-projects-and-llm-intro.md) | [73](raw/slides/07-attention-final-projects-and-llm-intro.md) | [100 ¶](raw/transcripts/07-attention-final-projects-and-llm-intro.md) |
| 8 | [Self-Attention and Transformers](wiki/08-self-attention-and-transformers.md) | [62](raw/slides/08-self-attention-and-transformers.md) | [100 ¶](raw/transcripts/08-self-attention-and-transformers.md) |
| 9 | [Pretraining](wiki/09-pretraining.md) | [54](raw/slides/09-pretraining.md) | [102 ¶](raw/transcripts/09-pretraining.md) |
| 10 | [Natural Language Generation](wiki/10-natural-language-generation.md) | [76](raw/slides/10-natural-language-generation.md) | [102 ¶](raw/transcripts/10-natural-language-generation.md) |
| 11 | [Post-training](wiki/11-post-training.md) | [99](raw/slides/11-post-training.md) | [104 ¶](raw/transcripts/11-post-training.md) |
| 12 | [Benchmarking and Evaluation](wiki/12-benchmarking.md) | [65](raw/slides/12-benchmarking.md) | [110 ¶](raw/transcripts/12-benchmarking.md) |
| 13 | [Efficient Training](wiki/13-efficient-training.md) | [65](raw/slides/13-efficient-training.md) | [81 ¶](raw/transcripts/13-efficient-training.md) |
| 14 | [Brain-Computer Interfaces](wiki/14-brain-computer-interfaces.md) | [75](raw/slides/14-brain-computer-interfaces.md) | [94 ¶](raw/transcripts/14-brain-computer-interfaces.md) |
| 15 | [Reasoning and Agents](wiki/15-reasoning-and-agents.md) | [75](raw/slides/15-reasoning-and-agents.md) | [82 ¶](raw/transcripts/15-reasoning-and-agents.md) |
| 16 | [After DPO](wiki/16-after-dpo.md) | [86](raw/slides/16-after-dpo.md) | [90 ¶](raw/transcripts/16-after-dpo.md) |
| 17 | [ConvNets and TreeRNNs](wiki/17-convnets-and-treernns.md) | [60](raw/slides/17-convnets-and-treernns.md) | [93 ¶](raw/transcripts/17-convnets-and-treernns.md) |
| 18 | [NLP, Linguistics, Philosophy](wiki/18-nlp-linguistics-philosophy.md) | [65](raw/slides/18-nlp-linguistics-philosophy.md) | [98 ¶](raw/transcripts/18-nlp-linguistics-philosophy.md) |

Lectures 9 and 10 are **Winter 2023** recordings by John Hewitt and Xiang Lisa Li; the rest
are Spring 2024. Lectures 11–16 are guest lectures — Archit Sharma, Yann Dubois, Shikhar Murty
(twice, lectures 13 and 15), Chaofei Fan, and Nathan Lambert (AI2) — and their video titles run
one behind this table, as does lecture 17's; lecture 18 is where the offset closes, because the
course's own lecture 17 is in neither the playlist nor the site's slide directory. Lectures 17
and 18 are Manning's again. See [INDEX.md](INDEX.md) for the mapping. Lecture 14's deck prints
**no slide numbers at all**, so its slide citations are PDF page positions; its slide file says
so in the header. Lecture 16's deck numbers every page too, but in the bottom-right corner
rather than the bottom-left every other deck uses.

Plus **63 topic pages** covering concepts that span lectures — word2vec, distributional
semantics, GloVe, gradient descent, backpropagation, matrix calculus, activation
functions, dependency grammar, transition-based parsing, language modeling, *n*-gram
models, recurrent neural networks, LSTMs, perplexity, vanishing and exploding gradients,
machine translation, sequence-to-sequence models, attention, self-attention, Transformers,
pretraining and fine-tuning, BERT, decoding algorithms, prompting, chain-of-thought,
instruction finetuning, reward modeling, RLHF, DPO, how any of it gets evaluated, and — from
lecture 13 — mixed precision training, GPU memory accounting, collective communication, DDP,
ZeRO/FSDP, parameter-efficient finetuning and LoRA, and from lecture 14 brain-computer
interfaces, neural recording technologies, neural population decoding, Connectionist Temporal
Classification, language models in decoding, and neuroethics — and from lectures 15–16
language model agents, counterfactual evaluation, self-training and rationale distillation,
RewardBench, PPO vs DPO, and preference data. These are what make it a wiki rather than a pile
of lecture summaries.

**Lectures 17–23 are not built yet.** This is deliberate — the KB is built incrementally, a
couple of lectures per run, and [TODO.md](TODO.md) is the authoritative record of what is
done and what remains. Slide *URLs* for lectures 1–18 are already inventoried in
[sources.md](sources.md), so questions about later lectures can at least be pointed at the
right PDF.

## Layout

| Path | Contents |
| --- | --- |
| [`INDEX.md`](INDEX.md) | **Entry point.** Course summary and annotated table of contents. |
| [`AGENTS.md`](AGENTS.md) | Schema: how the wiki is organized, and the conventions. |
| [`wiki/`](wiki/) | Durable pages — one per lecture, plus cross-lecture topic pages. |
| [`raw/slides/`](raw/slides/) | Full text of every slide, numbered, with diagrams described in prose. |
| [`raw/transcripts/`](raw/transcripts/) | Lecture transcripts with `[MM:SS]` paragraph markers. |
| [`raw/transcripts/original/`](raw/transcripts/original/) | Verbatim auto-captions, for reference only. |
| [`sources.md`](sources.md) | Every course document with its canonical URL and fetch date. |
| [`TODO.md`](TODO.md) | Build tracker. Unchecked boxes are outstanding work. |

## A note on the transcripts

The transcripts are YouTube auto-captions that have been **read and copy-edited by hand**.
This matters more than it sounds. Raw captions arrive as an unpunctuated wall of text that
mangles exactly the vocabulary the course is about — *word2vec* came through as "word
Tove", *ReLU* as "value", *parsing* as "paing", *n-gram* as "engram", *Hadamard* as "hadam
mod", and names like Bengio, Jelinek, Hochreiter and Olah were unrecognizable. An agent
asked "how does word2vec work?" cannot match a transcript in which the string never
appears.

So each transcript has had punctuation and sentence boundaries added, filler and false
starts removed, mis-heard terms restored from context and **cross-checked against the
slides**, and student questions marked. What was *not* done matters just as much:

- **No content was added, removed or reordered.** This is copy-editing, not summarizing.
- **Every `[MM:SS]` marker is preserved**, in order — verified by diff against the
  original, along with a comparison of every number in the text.
- **Nothing was guessed.** Where a garble leaves genuine ambiguity — a number, a name, some
  dictated notation — the text carries an inline `[Ed: …]` note saying so. Treat those as
  known gaps and prefer the slide.

The untouched captions are kept in [`raw/transcripts/original/`](raw/transcripts/original/)
so the edited versions can be checked against what was actually said.

**Where the transcript and the slides disagree, the slides win.** The transcripts are
corrected speech recognition; the slides are what the instructor wrote.

## Provenance

This is an **unofficial** study resource. CS224N is Stanford's course, and the lectures,
slides and handouts are the work of Christopher Manning, the course staff and the guest
lecturers. Everything here is derived from publicly available course materials at
[web.stanford.edu/class/archive/cs/cs224n/cs224n.1246](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/)
and the public lecture recordings.

**The course PDFs are not committed to this repo.** [sources.md](sources.md) records the
canonical URL for all 76 documents instead. Two reasons: the decks total over 160MB, which
does not belong in every clone; and this KB is read by agents that navigate markdown and
cannot extract anything from a PDF blob anyway, so the URL is the more useful artifact.
Past student final-project reports were deliberately excluded from the crawl — that is
student work, not course material.

If you are studying CS224N, go read the real thing. This is a navigation aid, not a
replacement.

## Built with Cairn

[Cairn](https://cairnstudy.com) tracks lecture courses on YouTube and puts an AI chat
alongside the video. When a knowledge base like this one is linked to a course, the chat
reads it to ground its answers in what the course actually taught — citing slide numbers
and timestamps rather than improvising from general knowledge.

This repo was compiled by Cairn's `cairn-kb` build process. To extend it, append entries to
[TODO.md](TODO.md) and re-run — it only does unchecked work.
