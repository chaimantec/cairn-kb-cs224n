# Exposure bias and teacher forcing

The mismatch built into the standard way of training text generation models, and the family of
responses to it — ending at RLHF. Covered in
[lecture 10](10-natural-language-generation.md), slides 38–48.

## Teacher forcing

Text generation models are trained one token at a time by maximum likelihood (slide 17),
minimizing

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(y^*_t \mid \{y^*\}_{<t})$$

where $y^*_t$ is the gold token at position $t$ and $\{y^*\}_{<t}$ are the gold tokens before
it. At each step this is a classification problem over the vocabulary — distinguish the actual
next word from all the others — and it is optimized with
[cross-entropy](softmax-and-cross-entropy.md).

The name comes from what goes in at the bottom: at every step the model is fed the **gold**
prefix, not what it actually predicted. "You reset at each time step to the ground truth"
(slide 17). Slide 48 records that teacher forcing "is still the main algorithm for training
text generation models."

## Is repetition a training problem?

Slides 39–40 re-open the degeneration problem from
[decoding](decoding-algorithms.md) on the training side. Even granting that sampling avoids
repetitive output at decode time, it remains alarming that the model *assigns so much
probability* to degenerate text in the first place. Slide 40's conclusion: **an MLE-trained
model learns a bad mode of the text distribution** — where "mode" means the argmax. "They would
assign high probability to terrible strings, and this is definitely problematic from a model
perspective" (≈40:52).

So should MLE not be the gold standard? "The answer here is not really, especially for text,
because MLE has some problems for sequential data" (≈41:37). The named problem is exposure
bias.

## Exposure bias

Slide 42 states it as a difference between two losses. During training, the model's inputs are
gold context tokens from real, human-written text:

$$\mathcal{L}_{\text{MLE}} = -\log P(y^*_t \mid \{y^*\}_{<t})$$

At generation time, the model's inputs are its own previously decoded tokens:

$$\mathcal{L}_{\text{dec}} = -\log P(\hat{y}_t \mid \{\hat{y}\}_{<t})$$

If the model makes even minor errors, $\{\hat{y}\}_{<t}$ is worse in quality than
$\{y^*\}_{<t}$ — and the model has never been trained on inputs of that quality. The
discrepancy compounds as generation proceeds, which is why models "lose coherence easily"
(slide 48). The lecture calls the alternative regime, where the model consumes its own outputs,
**student forcing** (≈11:36).

## Four responses

### Scheduled sampling

(Slide 43; Bengio et al., 2015.) With probability $p$, decode a token and feed *that* back as
the next input instead of the gold token; with probability $1 - p$, use the gold token. Increase
$p$ over the course of training, warming the model up for test-time conditions.

It improves results in practice, but slide 43 flags the cost: it "can lead to strange training
objectives" — you are no longer optimizing a clean likelihood — and training can be unstable
(≈43:10).

### DAgger — dataset aggregation

(Slide 43; Ross et al., 2011.) At intervals during training, generate sequences from the current
model and **add them to the training set** as additional examples. The training distribution is
continuously pulled toward the generation distribution.

Slide 43's margin note calls scheduled sampling and DAgger "basically variants of the same
approach." The lecturer's own distinction is granularity (≈52:28): DAgger adds whole generated
sequences between epochs, while scheduled sampling interleaves at the token level — "a more
smooth version of DAgger."

Asked how training a model on its own output can possibly help, the answer is that naively it
does not: "if you just put model generations in the data, it shouldn't really work" (≈54:44).
What helps is *correcting* them — for example, generate five tokens, then rather than accepting
the model's sixth, find a good continuation in the training data and splice it on, so the model
learns to recover from a path it has already gone slightly wrong on (≈53:59).

### Retrieval augmentation

(Slide 44; Guu\*, Hashimoto\*, et al., 2018.) Rather than generating from scratch, learn to
**retrieve** a sequence from an existing corpus of human-written prototypes — dialogue responses,
say — and then learn to **edit** it by inserting, deleting and modifying tokens.

This sidesteps exposure bias rather than mitigating it: you start from high-quality human text,
and at test time you are not generating left-to-right from nothing, so the train/test
discrepancy largely disappears (≈44:46).

### Reinforcement learning

(Slide 44.) Cast generation as a Markov decision process:

| MDP component | In text generation |
| --- | --- |
| **State** $s$ | the model's representation of the preceding context |
| **Action** $a$ | the next token to emit |
| **Policy** $\pi$ | the decoder |
| **Reward** $r$ | an external score |

The slide's own advice for the theory is to take CS234. Architecturally, RL is orthogonal to
the model: a [Transformer](transformer.md) (or an [LSTM](lstm.md)) supplies the sequence
probability, the RL objective consumes it, and you backpropagate through the network as usual
(≈48:36).

## Reward estimation

If you are optimizing a reward, you have to choose one. Slide 45's first idea is the obvious
one — use the evaluation metric you are being judged on:

- **BLEU** for machine translation (Ranzato et al., ICLR 2016; Wu et al., 2016)
- **ROUGE** for summarization (Paulus et al., ICLR 2018; Celikyilmaz et al., NAACL 2018)
- **CIDEr**, **SPIDEr** for image captioning (Rennie et al., CVPR 2017; Liu et al., ICCV 2017)

And slide 45's warning is the reason this is not the end of the story. Evaluation metrics are
**proxies** for quality, so optimizing them hard means optimizing the proxy:

> "even though RL refinement can achieve better BLEU scores, it barely improves the human
> impression of the translation quality" — Wu et al., 2016

You can move the number a long way and find that human evaluators think the output is no better,
or worse (≈46:19). See [evaluating machine translation](evaluating-machine-translation.md) for
what BLEU does and does not measure.

Slide 46 lists behaviours people have attached rewards to instead — cross-modality consistency
in captioning (Ren et al., 2017), sentence simplicity (Zhang and Lapata, 2017), temporal
consistency (Bosselut et al., 2018), utterance politeness (Tan et al., 2018), formality (Gong et
al., 2019).

## RLHF

The reward that mattered most is **human preference**, and slide 46 names it as "the technique
behind ChatGPT" (Ziegler et al., 2019; Stiennon et al., 2020). The procedure: ask humans to rank
generated texts by preference, use that comparison data to learn a **reward model** that scores
text the way humans would, then optimize the language model against that reward with RL.

Concretely, as the lecture sketches it (≈50:07): GPT-3 is the generator; a second pretrained
model — possibly also GPT-3 — is fine-tuned on the human preference data to become the reward
model; the original model becomes the policy, and RL updates it against the learned reward.

Two practical points from the floor:

- **It is expensive**, but not relative to the cost of pretraining. "Compared to all the cost of
  training … 170 billion parameter models, I feel like OpenAI and Google can afford hiring lots
  of humans" (≈47:04).
- **How much data?** Unknown for ChatGPT specifically, but on the order of **50k–100k**
  comparisons, judging by the scale of the preference dataset Anthropic released (≈47:51).

### The pipeline

Slide 48's discussion and ≈51:40 lay out why RLHF is the *last* stage rather than the only one:

1. **Pretrain** a large language model on internet text by self-supervision — this gives GPT-3.
   See [pretraining and fine-tuning](pretraining-and-finetuning.md).
2. **Instruction-tune** it, so it learns roughly how to follow human instructions.
3. **RLHF**, to align it with human preference.

The ordering is not incidental. "If we start RLHF from scratch it's probably going to be very
hard for the model to converge, because RL is hard to train for text data … but with all these
smart tricks about pretraining and instruction tuning, suddenly now they're off to a good
start."

## Takeaways

Slide 48's summary:

- Teacher forcing is still the main training algorithm.
- Exposure bias causes generation models to lose coherence easily. Either the model must learn
  to recover from its own bad samples (scheduled sampling, DAgger), or it must not be allowed to
  generate bad text in the first place (retrieval + generation).
- Training with RL lets models learn behaviours preferred by humans or by metrics — with the
  proxy caveat above attached.

## Related pages

- [Lecture 10 — Natural Language Generation](10-natural-language-generation.md) — the lecture.
- [Decoding algorithms](decoding-algorithms.md) — the same degeneration problem attacked at
  inference time.
- [Natural language generation](natural-language-generation.md) — the model and the training
  objective this page starts from.
- [Language modeling](language-modeling.md) — where teacher forcing is first introduced.
- [Evaluating machine translation](evaluating-machine-translation.md) — BLEU, used here as a
  reward and found wanting.
- [Evaluating NLG](evaluating-nlg.md) — why metric-as-reward is risky in general.
- [Pretraining and fine-tuning](pretraining-and-finetuning.md) — stage one of the pipeline.
