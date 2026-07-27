# Direct preference optimization (DPO)

A way to get the effect of [RLHF](rlhf.md) with a plain binary classification loss and no
reinforcement learning, by writing the reward model in terms of the language model. Covered in
[lecture 11](11-post-training.md), slides 70–74, with results at 73 and practice at 82, and
revisited a year later by a practitioner in [lecture 16](16-after-dpo.md) — "life after DPO."

## The idea

The RLHF pipeline has two learned objects: a reward model $RM_\phi(x, y)$ fit to human
comparisons, and a policy $p^{RL}_\theta$ optimized against it. Slide 70 asks: **what if you could
write $RM_\phi$ in terms of $p^{RL}_\theta$?** Then fitting the reward model to the preference data
*is* training the language model, and the RL step disappears.

The intuition offered first (≈57:59): a language model already assigns probability to plausible
completions; if you could restrict that probability mass to the completions humans like, then its
log-probabilities would themselves encode human preference. "So there is a direct correspondence
between the log probability that a language model assigns and how much a human might like the
answer."

And an argument for why this ought to be possible at all (≈59:31): the only external information
entering the RLHF pipeline is the human preference labels. Optimizing a learned reward model adds
no new information to the system. So nothing is lost by collapsing the two stages into one.

## The derivation

### Step 1 — the KL-constrained objective has a closed form

Recall the RLHF objective (slide 71):

$$\mathbb{E}_{\hat{y} \sim p^{RL}_\theta(\hat{y}|x)}\left[RM(x, \hat{y}) - \beta \log\!\left(\frac{p^{RL}_\theta(\hat{y} \mid x)}{p^{PT}(\hat{y} \mid x)}\right)\right]$$

Rather than optimizing it iteratively, solve it. The maximizer is

$$p^*(\hat{y} \mid x) = \frac{1}{Z(x)}\, p^{PT}(\hat{y} \mid x)\, \exp\!\left(\frac{1}{\beta} RM(x, \hat{y})\right)$$

which is the pretrained distribution **reweighted by the exponentiated reward**: high-reward
completions get more mass, low-reward ones less, with $\beta$ setting how aggressively. The lecture
notes it is the Boltzmann distribution in disguise (≈1:00:17). As $\beta$ falls you pay more
attention to the reward model.

$Z(x)$ is the normalizer that makes this a distribution:

$$Z(x) = \sum_{y} p^{PT}(y \mid x)\exp\!\left(\frac{1}{\beta}RM(x, y)\right)$$

and it is the villain of the derivation. Summing over *every* completion — not just the
syntactically valid ones, with a 50,000-token vocabulary and unbounded length — is intractable, and
not easy to approximate either (≈1:01:48).

### Step 2 — invert it to express the reward via the policy

Take logs and rearrange:

$$RM(x, \hat{y}) = \beta \log \frac{p^*(\hat{y} \mid x)}{p^{PT}(\hat{y} \mid x)} + \beta \log Z(x)$$

This says something readable: a completion has high reward when the optimal policy assigns it more
probability than the initialization did. The identity holds for *any* $p^*$ and its corresponding
reward, so substituting an arbitrary policy $p^{RL}_\theta$ defines a reward model
$RM_\theta$ implicitly:

$$RM_\theta(x, \hat{y}) = \beta \log \frac{p^{RL}_\theta(\hat{y} \mid x)}{p^{PT}(\hat{y} \mid x)} + \beta \log Z(x)$$

This is the step students push back on hardest, and the answer is worth keeping (≈1:04:04):

> "For now, we're not optimizing any reward model. All I'm saying is that if I take my current
> language model, it probably represents some kind of a reward model implicitly, because this holds
> for every $p^*$ and every reward model… I'm not saying it's optimal."

A second student notices that at initialization $p^{RL}_\theta = p^{PT}$, so the implied reward is
identically zero. Correct — "but we can optimize the parameters."

### Step 3 — the partition function cancels

$Z(x)$ is still intractable and will not vanish on its own. What saves the derivation is that the
[Bradley–Terry](reward-modeling.md) loss never uses a reward, only a **reward difference**:

$$J_{RM}(\phi) = -\mathbb{E}_{(x,\,y^w,\,y^l) \sim D}\left[\log \sigma\!\left(RM_\phi(x, y^w) - RM_\phi(x, y^l)\right)\right]$$

Both completions share the same instruction $x$, so both carry the same $\beta \log Z(x)$ term, and
subtracting kills it:

$$RM_\theta(x, y^w) - RM_\theta(x, y^l) = \beta \log \frac{p^{RL}_\theta(y^w \mid x)}{p^{PT}(y^w \mid x)} - \beta \log \frac{p^{RL}_\theta(y^l \mid x)}{p^{PT}(y^l \mid x)}$$

### Step 4 — the loss

Substituting into Bradley–Terry gives the DPO objective (slide 72):

$$J_{\mathrm{DPO}}(\theta) = -\mathbb{E}_{(x,\,y^w,\,y^l) \sim D}\left[\log \sigma\!\left(RM_\theta(x, y^w) - RM_\theta(x, y^l)\right)\right]$$

with $RM_\theta$ expanded as above. Every quantity in it is computable: four forward passes give the
log-probabilities of the winning and losing completions under the current policy and under the
frozen reference model.

> "We have a *simple classification loss* function that connects **preference data** to **language
> model parameters** directly!"

The training data is unchanged from RLHF: instructions, two completions (usually model-generated),
and a human label saying which is better.

## What the cancellation costs

A student asks the right question (≈1:08:40): by cancelling $Z(x)$, aren't you throwing away
information about all the other completions that standard RLHF would have accounted for?

The answer (≈1:09:26) is that the partition function is essentially a free variable — many reward
models satisfy the same optimization, differing by an additive constant per instruction, and that
degree of freedom is what gets removed. The underlying assumption, shared with RLHF, is the
Bradley–Terry one: preference probability depends **only** on the reward difference, so assigning
$+1$ versus $-1$ and $+99$ versus $+97$ are the same model. "Is that assumption true? Not completely
true, but it holds to a fairly large degree" (≈1:10:11).

The lecture also names one real difference in its summary (slide 74): DPO **does not leverage online
data**. RLHF samples fresh completions from the current policy and scores them; DPO trains on a fixed
preference dataset.

## Results

Slide 73 compares win rates against ground-truth completions:

| | Summarization | Dialogue |
| --- | --- | --- |
| **DPO** | **0.615** | **0.617** |
| Best of 128 | 0.607 | 0.598 |
| PPO | 0.598 | — |
| PFT | 0.408 | 0.447 |
| SFT / Base | 0.408 | 0.238 |

DPO and PPO land within a point of each other. "You're really not losing much by just doing the DPO
procedure instead of RLHF, and that's really compelling, because DPO is simply a classification loss
instead of a whole reinforcement learning procedure" (≈1:10:58).

## In practice

Slide 82 is the field's verdict. The Hugging Face Open LLM Leaderboard's top entries are annotated by
hand in red — DPO, DPO, DPO, "Merge (of DPO models)", "No info but prob DPO, given Merge (incl. DPO)"
— and the caption reads "Open source LLMs now almost all just use DPO (and it works well!)". The
lecture's count: "nine out of ten models here are trained with DPO" (≈1:14:49). Mistral uses it;
Llama 3 combines rejection sampling, PPO and DPO.

The slide-74 summary of when to choose which: "when people have a lot of computational budget they
typically maybe go for RLHF or some routine like that, but if you're really looking to get the bang
for your buck, you might want to go for DPO, and that's probably going to work out of the box"
(≈1:11:46) — with the caveat that this is an active research area and no strong claims are being
made.

DPO does not escape the limitations that come from the *preferences themselves* — reward hacking is
attenuated but the fallibility of human comparisons, length bias and annotator demographics all carry
over unchanged. Those are treated in [RLHF](rlhf.md) and [reward modeling](reward-modeling.md).

## A year later

[Lecture 16](16-after-dpo.md) is a guest lecture by Nathan Lambert (AI2) devoted to what came
after. Four things it adds:

**Why DPO won was infrastructural, not statistical.** The reference implementation is "extremely
simple," so "it's pretty easy to write a loss function that uses DPO, rather than building an
entire infrastructure stack to start with," where PPO "normally needs an almost entirely new
infrastructure stack" (≈12:26). That, more than accuracy, is why "we'll see more DPO models than
anything else" (≈13:11).

**Adoption took four months and one hyperparameter.** The paper appeared in May 2023; the first
model that made people believe it — Zephyr β — arrived in September, and needed the UltraFeedback
dataset plus a learning rate of 5e-7, "many orders of magnitude lower" than the usual 3e-4
(≈21:40). Lambert's reflection: "we probably could have done it months earlier if we just did more
hyperparameter sweeps … it's somewhat random" (≈21:40).

**DPO still has a reward model, and you usually cannot use it.** The implicit reward is a
log-ratio against the reference policy, $r(x,y) = \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$,
so scoring text with a DPO model yields something like $-200$ (≈36:57). But $\pi_{\text{ref}}$ is
an intermediate checkpoint almost nobody releases, and dropping it makes benchmark scores
"plummet across all the DPO models that we have" (≈37:42). See [RewardBench](rewardbench.md).

**PPO is slightly better and much more expensive** — about 1.2% on average at 13B, at a cost
Lambert is not sure is worth paying. See [PPO vs DPO](ppo-vs-dpo.md), which also covers the
online-data question and the DPO variants (self-rewarding, batched, D2PO) trying to answer it.
