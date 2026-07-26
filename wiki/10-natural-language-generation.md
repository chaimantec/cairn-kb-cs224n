# Lecture 10 — Natural Language Generation

The lecture that takes a trained language model and asks how you actually get good text out
of it. Xiang Lisa Li's argument is that the model's probability distribution is only half
the system: the **decoding algorithm** that turns that distribution into tokens matters
enormously, and the right choice depends on how open-ended the task is. Along the way the
lecture explains why maximum-probability decoding produces degenerate repetition, why
teacher forcing leaves a train/test mismatch, and why almost every automatic evaluation
metric for generation is worse than it looks.

**Slide-by-slide text of this deck: [76 printed slides / 71 PDF pages](../raw/slides/10-natural-language-generation.md)**
— printed slide numbers, which are what the lecturer refers to, **do not** equal PDF page
numbers past page 34; the slide file carries the mapping.

Slides PDF: [Lecture 10 — NLG](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture10-nlg.pdf) ·
[Full transcript](../raw/transcripts/10-natural-language-generation.md)

> **A note on provenance and numbering.** This lecture is a **Winter 2023** recording; the
> Spring 2024 offering of CS224N had no NLG lecture at all. The one lecture carries four
> different numbers — the Cairn catalog has it at position 10, the video title says "Lecture
> 11", the deck's title slide says "Lecture 12: Neural Language Generation", and the file on
> the Winter 2023 site is `lecture10-nlg.pdf`. This knowledge base uses the catalog position
> throughout.

## What NLG is, and the open-endedness spectrum

Slides 4–13 set up the taxonomy the rest of the lecture depends on. NLP splits into natural
language *understanding* (task input is language) and natural language *generation* (task
output is language); NLG is the systems that "produce fluent, coherent and useful language
output for human consumption." Machine translation, dialogue, summarization, data-to-text,
image captioning and ChatGPT are all NLG.

What distinguishes them is how constrained the output is. Slides 10–13 build a spectrum:

**Machine Translation → Summarization → Task-driven Dialog → ChitChat Dialog → Story Generation**

with a worked example at each end. For the Chinese source sentence on slide 10 there are
only a handful of valid English renderings, because the semantics cannot change — "the output
space is not very diverse." For "write me a story about three little pigs" on slide 12, "the
output space is extremely diverse." Slide 13 formalizes the distinction by **entropy**, and
states the consequence that organizes the whole lecture: these two classes "require different
decoding and/or training approaches." Full treatment at
[natural language generation](natural-language-generation.md).

The architecture convention follows from the same split (slide 16): non-open-ended tasks
typically use an [encoder-decoder](seq2seq-and-encoder-decoder.md), open-ended tasks usually
just an autoregressive decoder. The lecturer is careful that this is convention, not
constraint — a decoder alone can do MT, an encoder-decoder can write stories — and gives a
budget argument for why it holds: decoder-only hurts on MT, encoder-decoder gives no benefit
on open-ended generation, so if you have the compute you are better off spending it on a
larger decoder (≈10:03).

## Why the most likely string is the wrong thing to search for

This is the lecture's central result, built across slides 21–26. Greedy decoding and beam
search both try to maximize the model's probability of the output string, which is
appropriate for low-entropy tasks like MT and summarization. On open-ended generation, it
breaks in two distinct ways.

**It degenerates into repetition.** Slide 22 shows GPT-2 continuing the unicorn prompt
fluently for a sentence and a half and then looping "Universidad Nacional Autónoma de México"
indefinitely. Slides 23–24 diagnose why: plotting negative log likelihood as a phrase repeats,
the loss assigned to each repetition *falls*. The model gets more confident the more it has
already repeated — a **self-amplification effect** (≈17:00). Slide 24's caption records that
this is not an architecture problem (both an LSTM and a Transformer show it) and not a scale
problem: "even a 175 billion parameter LM still repeats when we decode for the most likely
string."

**It doesn't look like human text.** Slide 26 plots per-token probability over 100 timesteps
for beam-search output and for human writing. Beam search sits pinned near 1.0 almost
throughout; human text oscillates across the whole range. Human writing is full of locally
surprising choices, and an algorithm that maximizes probability cannot produce them by
construction.

The repetition fixes and the sampling algorithms that follow from this — top-$k$, top-$p$
(nucleus), typical and epsilon sampling, temperature, and re-ranking — are covered at
[decoding algorithms](decoding-algorithms.md).

## Is repetition a training problem?

Slides 38–48 re-ask the question from the training side, and the answer is that MLE training
"learns a bad mode of the distribution" — the argmax of a well-trained model is terrible text
(≈40:52). The named culprit is **exposure bias**: with teacher forcing, the model's inputs at
training time are gold tokens $y^*_{<t}$; at generation time they are its own previous
outputs $\hat{y}_{<t}$, which are worse, and get worse as errors compound (slide 42).

The lecture surveys four families of response — scheduled sampling, DAgger, retrieval-augmented
generation, and reinforcement learning — all covered at
[exposure bias and teacher forcing](exposure-bias-and-teacher-forcing.md). The RL discussion is
where RLHF enters: slide 46 names human preference as the reward signal "behind the ChatGPT
model," and slide 45 carries the warning that goes with optimizing any metric —

> "even though RL refinement can achieve better BLEU scores, it barely improves the human
> impression of the translation quality" — Wu et al., 2016

The lecture closes this section by laying out ChatGPT's training as a pipeline: pretrain a
large model on internet text by self-supervision (giving GPT-3), instruction-tune it to follow
instructions, then apply RLHF to align it with human preference. The ordering matters — "RL
doesn't really work from scratch" (≈51:40).

## Evaluating generation, and why it is unsolved

Slides 49–67 cover three families, in increasing order of both cost and trustworthiness.

**Content overlap metrics** (BLEU, ROUGE, METEOR, CIDEr) score lexical similarity to a
reference. They are fast, widely used, and degrade predictably as the task gets more
open-ended: "worse for summarization … much worse for dialogue … much, much worse for story
generation, which is also open-ended, but whose sequence length can make it seem you're
getting decent scores!" (slide 52). Slide 53's small failure case is the clearest statement of
the underlying problem — given the reference "Heck yes!" to Chris Manning's question about the
course, "Yup." scores **0** despite being correct, and "Heck no!" scores **0.67** despite
meaning the opposite. False negative and false positive, from one metric, on one four-word
exchange.

**Model-based metrics** replace $n$-gram matching with embeddings — vector similarity, Word
Mover's Distance, BERTScore, Sentence Mover's Similarity, BLEURT (slides 55–57) — and MAUVE
(slides 58–59) extends evaluation to open-ended settings by comparing the model's output
*distribution* against the human distribution in a quantized embedding space, so that both
precision and recall are measurable.

**Human evaluation** remains the gold standard and is itself unreliable: slow, expensive,
inconsistent between annotators and between sessions, prone to annotators silently
reinterpreting the question ("coherence" read as fluency by some and as topicality by
others), and — structurally — a precision-only measure, since you can ask a person whether a
sentence is good but not whether the model could produce every good sentence (≈1:09:20). All
of this, plus slide 62's rule that human evaluation scores must never be compared across
differently conducted studies, is at [evaluating NLG](evaluating-nlg.md).

The section's practical advice is blunt, and slide 67 puts it in bold: **look at your model
generations, don't just rely on numbers**, and publicly release large samples of your system's
output. The lecturer flags this explicitly as final-project guidance (≈1:11:39).

## Ethical considerations

Slides 68–75 close the lecture, prefaced by the deck's own content warning. The chain of
argument is short. Text generation models are built from pretrained language models;
pretrained language models are trained on internet text; internet text carries bias, so the
models repeat negative stereotypes when prompted (slide 72's table of GPT-2 completions
differing by the demographic in the prompt). Filtering the pretraining data is the obvious
fix and is "almost impossible to do" at internet scale (≈1:14:44).

Two failure modes are worse than ordinary bias. **Universal adversarial triggers** (slide 73)
are short, meaningless token sequences that reliably steer a model into extreme toxicity and
transfer across prompts — an exploit available to any ill-intentioned user. And slide 74 shows
models degenerating into toxic output from prompts that are entirely innocuous, so the risk
does not require an attacker at all. Slides 69–71 show the other side: ChatGPT's filtering
works on a direct request and fails against a structured jailbreak, and Bard's launch demo
contained a confident factual error.

The stated takeaway is that models should not be deployed without safeguards for toxic content
*and* without careful consideration of how users will actually interact with them — and, for
this audience specifically, that the smaller open models you use in a final project have fewer
of these protections than ChatGPT does (≈1:13:58).

## Answers to questions from the floor

- **What does "autoregressive" mean?** Generating one token at a time, left to right, each
  conditioned on the tokens already generated — the chain rule applied to a sequence. Other
  orders (backward, infilling) exist (≈12:22).
- **Doesn't top-$k$ conflict with the observation that humans use low-probability words?**
  Yes, precisely. Top-$k$ makes very low-probability tokens impossible, so it does not
  reproduce the human pattern — "that could be another hint that people can use for detecting
  machine-generated text" (≈24:44).
- **Why not weight by score instead of truncating?** Top-$k$ *is* weighted; it zeroes the
  tail, then samples proportionally to score among what remains (≈25:29).
- **Is truncation a compute saving?** No. You still compute the softmax over the full
  vocabulary to find the cutoff, so it "improves performance" rather than saving compute
  (≈30:04).
- **What is perplexity?** Roughly $e$ to the negative log probability — high perplexity means
  low probability, "because you are more perplexed" (≈38:35). See [perplexity](perplexity.md).
- **How much preference data does RLHF need?** Unknown for ChatGPT specifically; on the order
  of 50k–100k comparisons, judging by the scale of the preference dataset Anthropic released
  (≈47:51).
- **How do you apply RL to a Transformer?** The two are orthogonal — the architecture supplies
  the sequence probability, the RL objective consumes it, and you backpropagate through the
  Transformer as usual. It works the same with an LSTM (≈48:36).
- **Why discretize the embedding space for MAUVE?** A finite set of embedded sentences is a
  set of Dirac deltas in a continuous manifold, which no divergence can sensibly be computed
  over; smoothing would require assumptions (Gaussian, say) that text embeddings do not
  justify (≈1:03:11).

## Related pages

- [Natural language generation](natural-language-generation.md) — the task family and the
  open-endedness spectrum.
- [Decoding algorithms](decoding-algorithms.md) — greedy, beam, sampling, top-$k$, top-$p$,
  temperature, re-ranking.
- [Exposure bias and teacher forcing](exposure-bias-and-teacher-forcing.md) — the training
  side, through to RLHF.
- [Evaluating NLG](evaluating-nlg.md) — content overlap, model-based metrics, MAUVE, human
  evaluation.
- [Perplexity](perplexity.md) — the metric re-ranking reaches for first, and why it misleads.
- [Evaluating machine translation](evaluating-machine-translation.md) — BLEU, introduced in
  lecture 7 and stress-tested here.
- [Lecture 9 — Pretraining](09-pretraining.md) — where the models being decoded from come
  from.
