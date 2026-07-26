# RLHF — reinforcement learning from human feedback

Optimizing a language model against a learned model of human preference, under a penalty for
drifting away from where it started. Covered in [lecture 11](11-post-training.md), slides 57 and
65–69, with results at 68 and limitations at 86–92.

## The pipeline

Slide 57 gives the three steps that produced InstructGPT, and every RLHF'd model since:

1. **Collect demonstration data and train a supervised policy.** Sample a prompt, have a labeler
   write the desired output, finetune with supervised learning. This is
   [instruction finetuning](instruction-finetuning.md).
2. **Collect comparison data and train a reward model.** Sample several outputs for a prompt, have
   a labeler rank them best to worst, fit $RM_\phi$. This is
   [reward modeling](reward-modeling.md).
3. **Optimize a policy against the reward model using reinforcement learning.** Sample a new
   prompt, generate, score with $RM_\phi$, update the policy with PPO.

"First step: instruction tuning! Second and third steps: maximize reward (but how??)."

## The objective

You have a pretrained, possibly instruction-finetuned LM $p^{PT}(y \mid x)$ and a reward model
$RM_\phi(x, y)$ fit on human comparisons. Copy the LM into a policy $p^{RL}_\theta(y \mid x)$ with
trainable parameters $\theta$, and maximize the expected reward of its own samples (slide 65):

$$\mathbb{E}_{\hat{y} \sim p^{RL}_\theta(\hat{y} \mid x)}\left[RM_\phi(x, \hat{y})\right]$$

Note what is different from everything earlier in the course. In pretraining and supervised
finetuning the data comes from an external source and you maximize its log likelihood; here you
**sample from the model you are training**, and the objective need not be differentiable
(≈40:57).

## Why you cannot optimize that as written

Slide 66 asks the class to spot the problem, and a student gets it in one word: it might collapse.

$RM_\phi$ is *learned*. It will be accurate on the distribution it was fit on and wrong off it, and
a sufficiently strong optimizer will find the places where it is wrong and go there. "It might
erroneously assign a really high score to a really bad completion, and if your language model
learns to do that, it will completely hack that and start generating those gibberish completions"
(≈51:02).

The fix is a penalty for drifting from the initialization:

$$\mathbb{E}_{\hat{y} \sim p^{RL}_\theta(\hat{y}|x)}\left[RM_\phi(x, \hat{y}) \;-\; \beta \log\!\left(\frac{p^{RL}_\theta(\hat{y} \mid x)}{p^{PT}(\hat{y} \mid x)}\right)\right]$$

where $\beta$ trades off reward against faithfulness to the starting model. You pay a price
whenever $p^{RL}_\theta(\hat{y} \mid x) > p^{PT}(\hat{y} \mid x)$ — that is, whenever the policy has
become *more* confident in a completion than the pretrained model was. In expectation the penalty
term is exactly the **Kullback–Leibler divergence** between $p^{RL}_\theta$ and $p^{PT}$.

The justification is not just "stay close to something sane". It is that the reward model was
trained on completions sampled from around $p^{PT}$, so that is precisely the region where it can
be trusted (≈51:47).

Asked whether supervised finetuning needs the same penalty, the answer (≈53:21) is that
regularization exists there too but matters far less, because "with RL the incentive is to exploit
this reward model as much as possible".

## How the optimization is done

The lecture explicitly does not teach reinforcement learning — "this course does not assume
background on reinforcement learning" — and gives the intuition instead (slide 67):

- Generate completions from $p^{RL}_\theta$ for several instructions.
- Compute their rewards under $RM_\phi(x, y)$.
- Update $p^{RL}_\theta(y \mid x)$ to increase the probability of the high-reward completions.

In practice this is **PPO** (proximal policy optimization). The relevant history: RL has studied
these problems since Williams (1992) and Sutton and Barto (1998); interest revived around 2013 with
deep RL for game playing (Mnih et al.); applying it to modern LMs is newer still (Ziegler et al.,
2019; Stiennon et al., 2020; Ouyang et al., 2022).

## It works

Slide 68 (Stiennon et al., 2020) is the summarization result, plotting the fraction of the time
humans prefer a model's summary over the dataset's *reference* summary:

| Model size | Pretrain only $p^{PT}$ | Supervised $p^{IFT}$ | RLHF $p^{RL}$ |
| --- | --- | --- | --- |
| 1.3B | ~0.22 | ~0.39 | **~0.61** |
| 2.7B | ~0.31 | ~0.40 | ~0.64 |
| 6.7B | ~0.29 | ~0.42 | **~0.70** |
| 12.9B | ~0.36 | ~0.44 | — |

The dashed line at 0.5 is the reference summaries themselves. Only the RLHF curve is above it, and
it is above it *at every size measured* — "even very small models can outperform human completions
if you train them with RLHF" (≈54:53). Scaling still helps, but RLHF is the effect that moves the
model across the line.

Slide 83 shows what changes behaviourally: after PPO, Alpaca answers "what are the five most common
causes of stress" with a numbered list and a sentence of explanation per item, instead of a
five-word enumeration. "Significantly more detailed, nicer/clearer list-like formatting."

## It is also very hard

Slide 69 reproduces the complete PPO-for-RLHF systems diagram from *Secrets of RLHF* (Zheng et al.,
2023) — SFT model, reward model, value model, GAE advantage estimation, experience buffer, PPO-clip
loss, LM loss, MSE loss — and the lecturer's comment is that "this image is not for you to
understand, it's just completely to intimidate you" (≈55:39). The listed difficulties:

- you have to fit a value function;
- online sampling is slow;
- performance is sensitive to hyperparameters.

That difficulty is why "a lot of RLHF was restricted to very high-compute, high-resource places and
it was not very accessible" (≈56:25) — and the motivation for
[direct preference optimization](direct-preference-optimization.md), which reaches comparable
quality with a plain classification loss.

## Where it shows up

- **InstructGPT** (slides 75–78) scaled the pipeline to ~30,000 tasks, collected from labelers three
  ways: *plain* (invent a task), *few-shot* (invent an instruction plus query/response pairs), and
  *user-based* (write prompts for use cases from OpenAI API waitlist applications). The moon-landing
  and wise-frog completions on slides 77–78 are the before/after.
- **ChatGPT** (slides 79–81) is the same recipe optimized for dialogue: SFT on conversations where
  trainers played both user and assistant, then comparison data ranked by trainers, then PPO, "several
  iterations of this process". The lecture's summary: "the core algorithmic techniques that we
  discussed today are what give us ChatGPT, but you have to be really careful about the kind of data
  you're training on, and that's really the whole game" (≈1:14:04).
- **Llama 3** (slide 82) combines rejection sampling, PPO *and* DPO.

## Limitations

Slides 86–92, escalating:

**Reward hacking.** The CoastRunners agent spinning in a lagoon to farm score pickups instead of
finishing the race. RL is a strong enough optimizer — "it's at the heart of AlphaGo and AlphaZero" —
that mis-specified rewards get fully exploited (≈1:17:08).

**Humans reward the wrong things.** Models are rewarded for answers that *seem* authoritative and
helpful "regardless of truth", so hallucination is incentivized. Bluntly: annotators "prefer
authoritativeness more than correctness" (≈1:17:54). The lecture's other example is verbosity — at
scale, preference labelers "might just simply choose the longer answer, and that's a property that
actually goes into these models" (≈1:18:40).

**Over-optimization.** Slide 88 is the decisive plot. As the policy moves further from the supervised
baseline in KL, the reward model's *predicted* win rate rises monotonically to 1.0 while *actual*
human preference peaks around KL 10 at ~0.47 and then collapses to near zero by KL 250. The KL
penalty is not a nicety; without it you optimize straight past the point where the reward model
still means anything.

**Misalignment.** Slide 89 quotes Percy Liang: "Given reward hacking and the fallibility of humans,
this strategy seems bound to produce agents that merely appear to be aligned, but are bad/wrong in
subtle, inconspicuous ways."

**Whose preferences.** Slides 91–92 on annotation labour and labeler demographics — see
[reward modeling](reward-modeling.md).

## What comes after

Slides 96–99: **RL from AI feedback** and constitutional AI (Bai et al., 2022), where the model
critiques and revises its own harmful answer and the revision becomes training data; finetuning LMs
on their own outputs (Huang et al., 2022; STaR); and personalization, with the PRISM Alignment
Project as the example. The lecture's closing caveat is that hallucination and size "may not be
solvable with RLHF".
