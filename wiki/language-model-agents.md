# Language model agents

An **agent**, in this course's sense, is a model that receives observations from an environment
and issues actions into it, in order to carry out an instruction given in natural language. The
canonical example is a web browser: the goal is "book a flight from San Francisco to New York,"
the observations are pixels or HTML, and the actions are clicks and keystrokes.

The whole treatment in CS224N is [lecture 15](15-reasoning-and-agents.md), second half, from
Shikhar Murty. This page collects the setting, its history, its benchmarks and its failure modes.

## The setting

Standard reinforcement-learning terminology with one addition (lecture 15 slides 39–42, ≈30:22).
An agent — a neural network — faces an **environment**. It receives an **observation**, issues an
**action**, and is additionally conditioned on $g$, a **language instruction**.

That conditioning variable is the whole difference, and it is why the same object appears in the
literature under several names: *digital agent*, *language-conditioned policy*, or
*instruction-following agent* (≈31:10).

The connection to the reasoning half of the lecture is that both require multi-step inference.
Given a high-level objective, an agent "has to reason about postconditions, object affordances, a
kind of uncertainty in the world, to carry out a sequence of steps" (≈30:22).

For a browsing environment, the observation is **either raw pixels or the HTML DOM**, and the
action space is typing into web elements, clicking them, or moving the mouse to interact with
them (≈31:56). That choice of representation matters enough to have driven its own line of work —
see [pixels instead of HTML](#operating-over-pixels) below.

Applications the lecture lists (slides 43–45, ≈32:44): digital assistants taking spoken commands,
natural-language programming that emits Python, UI automation for software testing, user-facing
control of applications such as a music player, and tools or plugins attached to a language model
so it can drive other software.

## Three pre-LLM approaches

Worth knowing because the modern framing collapses all three (slides 46–48, ≈33:31).

**Semantic parsing to logical forms.** Collect utterances paired with executable representations.
For "what states border Texas," that is a program you can run against a knowledge graph or
database. Treat the mapping as translation — English commands as source language, meaning
representations as target — and apply ordinary
[seq2seq](seq2seq-and-encoder-decoder.md) machinery to maximise the probability of the action
sequence given the command (≈34:20).

**Latent plans.** Rather than mapping instructions straight to actions, infer an executable
**plan** from instruction/trajectory pairs, train a model to emit plans, and write a rich
execution model that runs them (≈35:06). The payoff is that high-level decisions can live in the
plan, where they would be hard to induce in a model trained on raw action sequences. The
lecture's example is a 2011 grounded-navigation system.

**Reinforcement learning.** Learn a policy that outputs actions maximising a reward, conditioned
on instruction and observation (≈37:25). Rewards are either **sparse** — the environment reports
success only at the end — or per-step. The example is a 2009 automated-Windows-debugging system
that mapped instructions to UI clicks and then to API calls.

## Agents in 2024: decision-making as language modelling

What is being modelled is a **trajectory**: a sequence of actions conditioned on a goal. It
factorises as (slide 49, ≈38:59)

$$p(\tau \mid g) = p(s_1, a_1, s_2, a_2, \ldots \mid g) = \prod_t p(s_t \mid s_{t-1}, a_t) \times \pi(a_t \mid \tau_{\le t}, g)$$

with $\tau$ the trajectory, $g$ the goal, $s_t$ the state at step $t$ and $a_t$ the action.

- $p(s_t \mid s_{t-1}, a_t)$ is the **transition dynamics** — a property of the environment, not
  something the agent learns.
- $\pi(a_t \mid \tau_{\le t}, g)$ is the **agent policy** — given the goal and everything so far,
  what to do next (≈39:44).

Only the second term is yours to model, and it is a next-token problem. That is the pivot: "you
could treat the problem of decision-making in environments as sort of a generative
trajectory-modeling problem" (≈39:44). Slide 51 shows the Decision-Transformer form — a
transformer over the history of actions, the current state and a task indicator (a return, though
"it could be a natural language string"), with a linear decoder predicting the next action.
Trained autoregressively, it works well in offline RL (≈40:32).

Hence: language models as policies. The simplest implementation is to **prompt in a loop** (slide
52, labelled **ReACT**, policy written $\pi_{\text{LM}}(\cdot \mid \tau_{\le t}, g)$): describe
the action space in text, supply the instruction plus the observations and actions so far, ask
for the next action, execute it, repeat. There is "nothing deep going on here — this is just
[chain-of-thought](chain-of-thought.md) prompting in a loop" (≈42:06), and Murty is explicit that
his minimal version "is not going to work at all" but that more elaborate versions do work in
some environments (≈42:52).

## Benchmarks

| Benchmark | Real sites? | Horizon | Distinguishing feature |
| --- | --- | --- | --- |
| **MiniWoB++** | No — sandbox | Most tasks under 3 actions | Basic browser interactions: retweet, forward an email, click a button |
| **WebArena** | No — close sandbox approximations | Longer, multi-step | Multi-tab browsing; e-commerce, social media and map tools; functional-correctness evaluation |
| **WebLINX** | Yes — actual websites | Longer | An action for the agent to *communicate with the user*; a static collection, not a live environment |

**MiniWoB++** (≈42:52) is a sandbox of basic browser interactions on mini versions of familiar
sites. Not real-world, and short-horizon — yet "zero-shot performance of even the best language
models is still far from perfect."

**WebArena** (≈43:38) approximates real sites closely, spanning an Amazon-like store, a
Twitter-like social site, and utility tools such as maps, so an instruction can require opening a
map, finding a shortest path, and using the result downstream (≈44:23). It introduced multi-tab
browsing. Its evaluation is **functional correctness** — whether the steps achieved the intended
outcome, not whether they matched a pre-programmed sequence (≈45:10).

**WebLINX** (≈45:10) runs on genuinely real websites and adds a *communicate with the user*
action, needed when a task such as buying a movie ticket requires credit-card details (≈45:57).
Because it is a recorded collection rather than a live environment, it supports evaluation but
not exploration or online learning.

## Where training data comes from

Standard practice is in-context learning with few-shot demonstrations, which requires paying
humans to demonstrate every new site (≈46:44). That does not scale across thousands of
environments, so the lecture's answer is synthetic data via exploration — the **BAGEL** pipeline
(Bootstrapping Agents by Guiding Exploration with Language), described in full on
[self-training and rationale distillation](self-training-and-rationale-distillation.md).

The short version: explore randomly, have a second model *guess what instruction* each trajectory
accomplished, re-roll conditioned on that guessed goal, filter with a reward $R(g, \tau)$ over
goal and trajectory, and **relabel rather than discard** failures — a trajectory that missed its
goal still accomplished something, and real-site interactions are too costly to throw away
(≈50:39). Used as in-context demonstrations, this gives a 13-point gain on MiniWoB++ (46.8 → 60.5
on slide 67).

## Operating over pixels

Feeding HTML into the context does not scale — a page may carry tens of thousands of DOM elements
plus JavaScript — and may not even be the best state representation (≈53:40). Two vision-language
models later adapted for agents:

- **LLaVA** (≈54:26) applies the Orca recipe to images: GPT-4 generates instructions and responses
  from *textual descriptions* of images built from metadata, then a CLIP image encoder is jointly
  fine-tuned with a Vicuna text decoder.
- **Pix2Struct** (≈55:12) has the same encoder/decoder shape — patches with position IDs through
  a transformer — but contributes a better pre-training task: mask regions of website screenshots
  and predict the **HTML of the masked element** (≈56:45), which forces real image-text
  interaction rather than relying on what a caption can express.

## The prompting gap

Murty's own term for the state of the field (slides 71–74, ≈57:30): without extensive prompting
and per-environment bespoke few-shot examples, even the best models are far from perfect on even
the simplest tasks. Three findings:

1. **Long horizons break it.** On MiniWoB++, accuracy falls off going from single-action tasks to
   five- or ten-action ones. "Long-horizon planning is still very, very hard, even on these very
   simple benchmarks" (≈58:16).
2. **Realism widens the gap.** On WebArena there is a large human-versus-model gap even with
   prompting and few-shot examples (≈59:02).
3. **The failures are trivial.** GPT-4V, asked to sign into Google Translate, typed the email into
   the *password* field and could not recover, looping on retries (≈59:02). Another model issued
   a search with the same term repeated three times (≈59:48).

## Related pages

- [Lecture 15 — Reasoning and agents](15-reasoning-and-agents.md) — the source lecture.
- [Self-training and rationale distillation](self-training-and-rationale-distillation.md) — BAGEL
  and the generate/filter/train recipe it shares with the reasoning half.
- [Chain-of-thought](chain-of-thought.md) — what prompting an agent in a loop actually is.
- [Prompting](prompting.md) and [GPT and in-context learning](gpt-and-in-context-learning.md).
- [Sequence-to-sequence and encoder-decoder models](seq2seq-and-encoder-decoder.md) — the
  pre-LLM semantic parsers' machinery.
- [Evaluating LLMs](evaluating-llms.md) — benchmark design and its limits.
