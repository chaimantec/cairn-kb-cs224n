# Lecture 15 — Reasoning and agents

Two applications of language models that both come down to the same question: can a model that
was trained to continue text be made to *do* something over multiple steps? Shikhar Murty — back
after [lecture 13](13-efficient-training.md) — spends the first half on reasoning, meaning
multi-step inference in domains like arithmetic, logic and analogy, and the second half on
agents, meaning models that issue actions into a web browser or an operating system to carry out
an instruction.

The lecture is unusually honest about its own subject. The first half builds up four ways of
extracting reasoning from a model and then spends nine minutes taking them apart: the rationales
a model prints are often not what produced its answer, and performance collapses as soon as a
problem is nudged out of the training distribution. The second half does the same for agents —
the benchmarks are simple, the models are far from solving them, and the failures are trivial
rather than deep. Murty flags this in his opening: most of the content is research from "the last
three, four years, so there's plenty of questions, plenty of unanswered questions" (≈0:52).

**Slide-by-slide text of this deck: [75 slides](../raw/slides/15-reasoning-and-agents.md)** —
printed slide numbers match PDF pages 1:1 throughout.

Slides PDF: [Reasoning and Agents](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture14-agents-shikhar-updated.pdf) ·
[Full transcript](../raw/transcripts/15-reasoning-and-agents.md)

> **A note on numbering.** This lecture sits at **position 15** in the playlist this knowledge
> base follows; the video title and the deck both call it "Lecture 14". Repo files use the
> position. The recap slide is headed "Lecture-14 Recap" for the same reason.

## What counts as reasoning

Murty opens with the textbook taxonomy (slides 4–7), because the word is used loosely and the
rest of the lecture only tests one branch of it. At a high level reasoning is "using facts and
logic to arrive at an answer" (≈0:52), and it splits three ways:

- **Deductive** — from rules of logic plus a premise to a firm conclusion. *All mammals have
  kidneys; all whales are mammals; therefore all whales have kidneys* (≈1:38). Multiple such
  steps can be chained.
- **Inductive** — from observations to a probable conclusion. Every winged creature you have seen
  was a bird; you see a winged creature; it is *likely* to be a bird (≈2:25).
- **Abductive** — from an observation to a possible explanation. A car will not start and there
  is a puddle under the engine, so perhaps the radiator is leaking (≈2:25).

Cutting across that is the **formal / informal** distinction (slide 8). Formal reasoning uses
axioms and the rules of formal logic to derive truth conditions; informal reasoning is the
everyday, common-sense kind. Murty then narrows the whole lecture: "when I say reasoning I will
mean informal deductive reasoning, and it's often going to involve multiple steps" (≈3:11).

That last clause is what makes it an NLP problem rather than a theorem-proving one. Everything
tested below is a chain of ordinary-language inferences, not a proof.

The setup question (slide 9) leans on [lectures 9](09-pretraining.md),
[10](10-natural-language-generation.md) and [11](11-post-training.md): those established that
large language models are "really, really good at coming up with plausible continuations of text
that reflect human preferences and constraints" (≈3:11). Plausible continuation is not the same
thing as correct inference. The rest of the lecture is about the gap.

## Getting reasoning out of a model by prompting

Four prompting methods, in increasing order of structure (slides 10–19). All of them are
elaborations on [chain-of-thought](chain-of-thought.md).

**Chain-of-thought prompting** gets the model to produce a reasoning step before producing an
answer, by showing in-context examples that contain explicit reasoning for it to mimic (≈4:00).

**Zero-shot chain-of-thought** is the surprising version: you do not need the examples at all.
Appending "Let's think step by step" is enough to make the model emit a rationale before its
answer (≈4:00). Slide 11 carries the trigger phrase verbatim.

**Self-consistency** changes the decoding rather than the prompt (≈4:47). Instead of greedily
decoding one rationale and then an answer conditioned on it, sample *many* rationales, take the
answer each one leads to, and return the most common answer. The intuition is that if an answer
survives many independent routes to it, it is more likely to be right (≈5:33). Across a variety
of mathematical reasoning tasks this improves on standard chain-of-thought "pretty drastically."

Murty's own first reaction to that result is worth keeping, because it is the obvious objection:
this is just ensembling, the trick from CS229 where you train ten classifiers with different
seeds and take a majority vote (≈6:19). The authors checked. They compared against exactly that
baseline — one language model, several *different prompts*, majority vote — and self-consistency
still won. So sampling multiple reasoning paths from a fixed prompt is doing something more than
variance reduction over prompts.

**Least-to-most prompting** adds explicit decomposition (slides 15–19, ≈7:06). Multi-step
reasoning means "breaking down a large problem into several subparts, answering each of the
subparts, and then combining everything into a solution," so least-to-most makes each of those a
separate generation: first ask the model to break the question into sub-questions, then answer
each sub-question, then condition the final answer on those answers (≈7:53).

Its most interesting result is **length generalization**: prompted with a math word problem
needing only two reasoning steps, the model continues to work on test examples requiring more
than five (≈8:39). But the paper's own ablation blunts the claim — "with enough prompt
engineering, the row corresponding to best-prompted chain-of-thought is on par with
least-to-most prompting" (≈9:26). Structuring inference this way helps, but it is not clearly
*fundamental*.

## Putting reasoning into a smaller model: Orca

Prompting only helps if you already have a very large model. The alternative is distillation:
teach a small model to imitate a large one (slides 20–28, ≈10:13).

**Orca** fine-tunes a 13-billion-parameter Llama on explanations generated by GPT-4. Three steps
(≈10:58):

1. Take a wide variety of instructions from the **FLAN-v2** collection — itself an aggregation of
   many datasets, pairing instructions with questions and answers.
2. Prompt GPT-4 or ChatGPT with each instruction *plus a system message* whose job is to elicit
   an informative explanation alongside the answer — "please justify your steps, and answer step
   by step" (≈11:45). The slide's worked example is a question about computing a median, and the
   teacher's output is a detailed derivation rather than a bare number.
3. Fine-tune the small Llama on those explanations.

The evaluation switches benchmark to **BigBench-hard** (slides 24–27, ≈12:34), a multi-step
reasoning suite of **23 sub-tasks**. Murty shows three, and they are deliberately varied:

- *Boolean expressions* — "True and False and not True and True is", which chain-of-thought
  solves by evaluating sub-expressions one at a time (≈12:34).
- *Date understanding* — given that tomorrow is some date, what was the date a year ago, in a
  specified format (≈13:19). He corrects himself on the slide: date understanding, not data
  understanding.
- *Geometric shapes* — given an SVG path element, what shape does it render as (≈14:06). "It's
  pretty surprising that language models can do anything here," and he is candid that he cannot
  tell from the prompt whether the answer shown is correct.

The tasks are multiple-choice, which is what makes a clean accuracy number possible (≈14:53).

The results table (≈14:53) needs one caveat read with it: **GPT-4 has potential contamination
issues on BigBench-hard**, so Murty tells the room to ignore that column — an instance of the
problem [benchmark contamination](benchmark-contamination.md) covers in general. Among the
comparable models, Orca (Llama-13B fine-tuned on explanation data) beats both ChatGPT — plausibly
because it is specialised to these reasoning problems — and Vicuna, an instruction-tuned Llama-13B
that was until recently state of the art but was never trained on extensive explanations
(≈15:39).

## Training a model on its own rationales

The obvious follow-up: if rationales from a big model can train a small one, why not use them to
train the big model itself (≈16:26)? Several methods do this; the lecture covers **ReST**,
reinforced self-training (slides 29–30), which alternates two stages:

1. **Generate and filter.** Given a reasoning problem, and perhaps "let's think step by step,"
   sample many rationales, then keep only those that reach the correct answer. Murty's example:
   someone has three apples and someone else has four; a rationale ending in seven is kept, one
   ending in 12 is discarded (≈16:26).
2. **Update.** Fine-tune the model on the surviving rationales.

Then iterate — a better model produces better rationales, which produce a better model (≈17:14).

The results are genuinely mixed, and the lecture says so. On **GSM8K**, the grade-school algebraic
word-problem set, accuracy improves slightly with more self-training iterations "and then it
starts degrading" (≈18:02). On **MATH**, which also targets multi-step problems, accuracy keeps
improving. In those plots the orange numbers are a much larger PaLM model, blue is a smaller one,
and the dashed lines mark supervised fine-tuning on *human*-written rationales (≈18:02).

The most striking finding is that self-generated rationales can beat human ones, and the bar
chart is carefully controlled for quantity (≈18:48):

| Bar | What it is |
| --- | --- |
| Blue | PaLM fine-tuned on **all** human-provided rationales |
| Orange | fine-tuned on **one human** rationale per training example |
| Green | fine-tuned on **one model-generated** rationale, chosen at random per question |

Green beats orange at matched counts, and running the full iterative procedure adds a further
boost. See [self-training and rationale distillation](self-training-and-rationale-distillation.md)
for this recipe as a general pattern — it recurs in the agents half of the lecture.

## Does any of this count as reasoning?

Slides 31–37, and the part of the lecture worth reading twice. Benchmark numbers alone cannot
answer the question, so the move is to "be more systematic, come up with counterfactual tasks,
and be very careful about possible data contamination" (≈20:21).

### Are chain-of-thought rationales faithful?

**Faithful** here means the answer actually depends on the rationale. The failure mode is a model
that writes "Tom has three apples, Jerry has four, 3 + 4 is seven" and then answers 25 (≈21:07) —
the rationale is decoration.

Two experiments test it. The first is **early exit** (≈21:53): a rationale is several sentences,
so force the model to stop after sentence one and answer, then after sentence two, and so on,
plotting accuracy against the number of reasoning sentences it was allowed. If truncating after
one sentence yields the same answer as the full four, the extra reasoning was not doing work —
and in the limit you can cut the rationale entirely and get the same answer (≈22:39).

The second **corrupts** rationales instead of truncating them (≈23:27). Introduce a mistake after
some percentage of the reasoning steps and see whether the answer changes. The expected shape is
a strictly increasing curve: a mistake in step one should wreck the answer, a mistake in the last
step should barely matter (≈24:14).

Results on both are mixed, but on enough datasets the answer is unmoved by truncation or by an
early corruption — which "means that sometimes these rationales may be post-hoc explanations of
the model's answer" (≈22:39). Murty notices the room glazing over here: "there's a lot of lines
here — so if anyone has questions — I see a few blank faces in the audience" (≈24:14).

### Reasoning or memorisation? Counterfactual tasks

The second attack changes the task so that it cannot have been memorised (≈25:01). A model does
base-10 arithmetic; is that arithmetic, or is `12 + 14` in the training data? Test it in **base
9** instead. If accuracy holds, the model has something like addition; if it collapses, it had
the answers.

A student asks the right question — why is base 9 the counterfactual rather than base 10? — and
the answer is distributional: base-10 addition is frequently observed in training data and very
few people do base-9 arithmetic, so there will be far fewer examples of it. The student offers
"so it's more so out-of-distribution, right?", and Murty accepts the reframing (≈26:33). The same
trick works for logic: construct a world in which corgis are reptiles and see whether the model
can still run the syllogism (≈25:46).

There is a significant drop, "even for very simple logic problems that don't involve multiple
steps of reasoning," suggesting "there's not that much reasoning, there's more memorization"
(≈27:18).

The **analogical reasoning** study (slides 35–37, ≈27:18) is the cleanest version because it adds
a human control group. The base task is string transformation shown by example: `ABCD → ABCDE`,
so `IJKL → IJKLM` (≈28:04). Two counterfactual variants:

- Change the *task*: the output must be `ABCDF` instead of `ABCDE` — not the next character but
  the one after it (≈28:04).
- Change the *alphabet*: instead of starting at A, B, C, start at X, Y (≈28:51).

Models drop significantly on both. Humans, run on the identical experiment, show "very little"
decrease (≈29:37). Murty's summary is the lecture's thesis for the first half: "maybe there's
some reasoning, maybe there's some memorization, but there's nothing systematic" — with the
appropriate hedge that the field is moving and "maybe someone will find that if you change your
prompt a little bit, now models can do reasoning" (≈29:37).

See [counterfactual evaluation](counterfactual-evaluation.md) for the method on its own.

## Language model agents: the setup

The bridge between the two halves is that both need multi-step inference. With agents, "there's
some high-level objective the model has to accomplish, and it has to reason about postconditions,
object affordances, a kind of uncertainty in the world, to carry out a sequence of steps"
(≈30:22).

The terminology (slides 39–42, ≈30:22) is standard RL with one addition. An **agent** — a neural
network — sits opposite an **environment**. It receives an **observation**, issues an **action**,
and additionally receives $g$, a **language instruction**. That extra conditioning variable is
what makes it an NLP problem, and it is why the same object goes by several names: *digital
agent*, *language-conditioned policy*, or *instruction-following agent* (≈31:10).

For a web browser with the goal "book a flight from San Francisco to New York," the observation
is either raw pixels or the HTML DOM, and the action space is typing into elements, clicking
elements, or moving the mouse to interact with them (≈31:56).

Applications (slides 43–45, ≈32:44): digital assistants taking spoken commands — Murty pointedly
declines to say the product names aloud so as not to set off the room's phones — natural-language
programming that emits Python, UI automation for testing (a model executing the actions a human
tester would), user-facing control of an application like Spotify, and the emerging pattern of
attaching tools or plugins so a model can drive other software (≈33:31).

## How this was done before LLMs

Slides 46–48. Murty argues the pre-LLM history is worth knowing before looking at 2024, and there
were three main ideas (≈33:31).

**1. Semantic parsing to logical forms.** Collect utterances paired with executable
representations — for "what states border Texas," a program you can run against a knowledge graph
or database to get the list (≈34:20). Then treat it as translation: source language is English
commands, target language is meaning representations, and you can apply "the same machinery from
Assignment 3" — the [seq2seq](seq2seq-and-encoder-decoder.md) stack from
[lecture 6](06-sequence-to-sequence-models.md) — to directly maximise the probability of an
action sequence given the command (≈35:06).

**2. Infer latent plans.** Instead of mapping instructions straight to actions, infer an
executable **plan** from instruction/action-sequence pairs, train a model to produce plans, and
define a rich execution model that runs them (≈35:06). The advantage is that high-level decisions
can be encoded in the plan that would be hard to get into a model trained on raw trajectories.
The example is a 2011 system that navigated a grounded environment: train a semantic parser to
turn commands into plans, then at test time parse a new instruction and execute the resulting
plan (≈36:39).

**3. Reinforcement learning.** The idea that comes to mind first — learn a policy that outputs
actions maximising a reward, conditioned on the instruction and the observation (≈37:25). The
reward may be **sparse** (the environment says at the end whether the task was achieved) or
per-step (after each action the environment reports what fraction of the task is done). The
example is a 2009 system for automated Windows debugging, mapping natural-language instructions
to UI clicks and thence to API commands executed one after another (≈38:13).

## Agents in 2024: decision-making as language modelling

The modern framing (slides 49–51) starts from what is actually being modelled: trajectories,
meaning sequences of actions conditioned on a goal (≈38:59). That factorises as

$$p(\tau \mid g) = p(s_1, a_1, s_2, a_2, \ldots \mid g) = \prod_t p(s_t \mid s_{t-1}, a_t) \times \pi(a_t \mid \tau_{\le t}, g)$$

where $\tau$ is the trajectory, $g$ the goal or instruction, $s_t$ the state at step $t$ and
$a_t$ the action. The first factor $p(s_t \mid s_{t-1}, a_t)$ is the **transition dynamics** of
the environment — what happens to the state when you take an action — and belongs to the world,
not the model. The second, $\pi(a_t \mid \tau_{\le t}, g)$, is the **agent policy**: given the
goal and the trajectory so far, what should the next action be (≈39:44).

Once you write it that way, "people quickly realized that you could just treat this as a
generative problem" — decision-making in environments becomes generative trajectory modelling
(≈39:44). Slide 51 shows the Decision-Transformer arrangement: a transformer consuming the
history of actions, the current state, and an indication of the task (there expressed as a
return, though it "could be a natural language string"), trained with a linear decoder to predict
the next action. Trained autoregressively, this worked well in the offline-RL setting (≈40:32).

A student asks why only one action is predicted; the answer is the standard autoregressive loop —
predict an action, execute it, append it to the trajectory, predict the next (≈40:32). (The
transcript marks a second, partly garbled student turn here.)

So instead of latent plans and semantic parsers, "we started using language models as policies"
(≈41:19). The simplest realisation is to **prompt a language model in a loop** (slide 52): specify
the action space in text — type, click, type characters, move the mouse — supply the instruction
and the actions and observations so far, and ask for the next action (≈42:06). The deck labels
this **ReACT** and writes the policy as $\pi_{\text{LM}}(\cdot \mid \tau_{\le t}, g)$. Murty is
blunt that his stripped-down version "is not going to work at all, but it's illustrative": there
is "nothing deep going on here — this is just chain-of-thought prompting in a loop." The hope is
that having reduced decision-making to autoregressive modelling, it works anyway — and a slightly
more complex version does, in some environments (≈42:52).

## Benchmarks for agents

Three environments, in increasing order of realism (slides 53–55).

**MiniWoB++** (≈42:52) is the simplest: a sandbox of basic browser interactions — retweet a given
tweet in a mini-Twitter, forward or compose an email in a simulated client, click particular
buttons. Not real websites, and short-horizon: most tasks finish in under three actions. Even so,
"zero-shot performance of even the best language models is still far from perfect."

**WebArena** (≈43:38) is still a sandbox but closely approximates real sites, spanning e-commerce
(an Amazon-like store), social media (a Twitter-like site) and utility tools such as maps — so an
instruction can require opening the map app, finding the shortest path from A to B, and using
that result in later actions (≈44:23). It introduced **multi-tab browsing**, where the agent
switches between tabs and apps. Evaluation is **functional correctness**: did the sequence of
steps produce the intended outcome, rather than matching a pre-programmed sequence (≈45:10).

**WebLINX** (≈45:10) uses *actual* real websites rather than sandboxed approximations, also with
multi-tab browsing, and adds an action for the agent to **communicate with the user** — needed
when a task like buying a movie ticket reaches the point of requesting credit-card information
(≈45:57). The catch: it is a collection of recorded interactions, not a live environment, so
there is no exploration or online learning, only evaluation.

## Where the training data comes from: BAGEL

Standard practice is in-context learning with few-shot examples, which means paying humans to
demonstrate tasks on every new website and pasting those demonstrations into the prompt (≈46:44).
That does not scale — "there's thousands of environments, and, on some environments, lots of
different interactions that are possible" (≈46:44).

So the agents half reuses the recipe from the reasoning half: generate data with the model,
filter it, train on what survives. Here there are no rationales to filter by correctness, because
unlike a math problem there is no known correct answer for a trajectory (≈48:19). The pipeline
(slides 57–67), which the deck names **BAGEL** — Bootstrapping Agents by Guiding Exploration with
Language:

1. **Explore randomly.** An unconditioned agent executes random clicks, types and scrolls in the
   environment, producing trajectories (≈47:31).
2. **Label them.** A second language model looks at each trajectory and produces a description of
   what it accomplished — a guessed instruction. Trajectory one might be labelled "book a flight
   from San Francisco to New York," trajectory two "set the date to …," and for trajectory three
   it may fail to come up with anything (≈48:19). The premise is that if a model can describe a
   sequence of actions, that is signal the sequence was coherent.
3. **Re-roll conditioned on the label.** Now feed the guessed goal back in, so instead of random
   exploration the model produces actions that correspond to a natural-language instruction
   (≈49:52).
4. **Filter.** A coarse filter — written in the deck as a reward $R(g, \tau)$ over goal and
   trajectory — checks the correspondence between the instruction, the action sequence and the
   states visited, and decides whether this is a good instruction/trajectory pair (≈49:52).
5. **Relabel rather than discard.** When the trajectory clearly does not accomplish the stated
   goal, throwing it away wastes a costly interaction, "specifically, you know, if you're looking
   at real websites" (≈50:39). Instead invoke the relabeler: the model did accomplish *something*,
   so guess what — perhaps "set the origin to SFO and the destination to New York City" — and feed
   that back. Iterate until the filter passes (≈51:24).

The output is synthetic instruction/trajectory data. You can fine-tune on it, but the simplest
use is to drop it into the prompt in place of human demonstrations (≈52:55). That yields a
**13-point improvement on MiniWoB++** — slide 67 labels the bars 46.8 zero-shot against 60.5 with
BAGEL — plus a smaller gain on a second multi-step tool-use benchmark, where the average moves
40.9 → 43.3 and BAGEL is *not* uniformly ahead across categories.

## Pixels instead of HTML

Feeding HTML into the context does not scale either: a page can carry tens of thousands of DOM
elements plus JavaScript, and it may not even be the best representation of the state — the
pixels might be (≈53:40). Two vision-language models later adapted for agents (slides 68–70):

**LLaVA** (≈54:26) mirrors Orca exactly. Use GPT-4 to generate both instructions and responses
from *textual descriptions* of images (built from image metadata), then jointly fine-tune an image
encoder — CLIP — with a text decoder — Vicuna, the instruction-tuned Llama. The result takes
images and emits language, so screenshots can be fed in place of DOM elements (≈55:12).

**Pix2Struct** (≈55:12) has the same shape — image encoder, text decoder — with the encoder
splitting the image into patches, assigning each a position ID and running them through a
transformer. Its contribution is a better **pre-training task**. LLaVA's pre-training is limited
by what a textual description of an image can capture; Pix2Struct instead takes website
screenshots, masks out regions, and asks the decoder to produce the **HTML for the masked-out
element** (≈56:45) — for instance masking the first answer in a list and predicting its markup.
That objective forces genuine interaction between image and text.

## The prompting gap

The closing assessment (slides 71–74, ≈57:30). Murty's term for it is the **prompting gap**:
without extensive prompting and bespoke few-shot examples tailored per environment, even the best
models are "very, very far from perfect, even on very simple tasks like MiniWoB++."

Three specific findings:

- **Horizon length hurts.** Even on MiniWoB++, even with extensive prompting, accuracy falls off
  as tasks go from one action to five or ten. "So long-horizon planning is still very, very hard,
  even on these very simple benchmarks" (≈58:16).
- **Realism hurts more.** On WebArena, with its multi-tab browsing and external tools, there is a
  large gap between human task success rate and the best models even after prompting and few-shot
  examples (≈59:02).
- **The errors are trivial, not subtle.** In a WebLINX example the task was to open Google
  Translate and sign in with a supplied email and password; GPT-4V typed the *email* into the
  password field and then could not recover — it retried the sign-in, tried to type the email
  again, and looped (≈59:02). In another case the model issued a search with the same term
  repeated three times, which returns nothing (≈59:48). Murty concedes these are probably fixable
  with prompting "and maybe that's beside the point."

## Recap

The lecture's own summary (≈1:00:34): reasoning-like behaviour can be elicited by prompting
(chain-of-thought; sampling multiple rationales and reconciling them; explicit decomposition), by
training a small model on a big model's rationales, or by training a big model on its own
rationales iteratively — which can beat human-written rationales. But counterfactual evaluation
leaves it unclear "if the models are good because they're reasoning, or if models are good
because … all of these problems were in some shape or form already in the training data"
(≈1:01:20).

For agents: a historical route through semantic parsers, latent plans and RL; the recasting of
decision-making as causal language modelling; mostly prompting and in-context learning in
practice; and the same synthetic-data recipe from the first half, via exploration and iterative
relabelling. Benchmarks are challenging, models make trivial mistakes, and the human-model gap is
large — "a lot of room for driving further improvement. And maybe some of you are doing it for
your projects" (≈1:02:53).

## Related pages

- [Chain-of-thought](chain-of-thought.md) — the prompting family this lecture extends and then
  interrogates.
- [Language model agents](language-model-agents.md) — the agent setting, its environments and
  benchmarks, gathered on one page.
- [Counterfactual evaluation](counterfactual-evaluation.md) — testing reasoning by moving the
  task out of distribution.
- [Self-training and rationale distillation](self-training-and-rationale-distillation.md) — Orca,
  ReST and BAGEL as one recipe.
- [Benchmark contamination](benchmark-contamination.md) — why the GPT-4 column on BigBench-hard
  gets ignored.
- [Prompting](prompting.md) and [GPT and in-context learning](gpt-and-in-context-learning.md) —
  the mechanisms all of this is built on.
- [Instruction fine-tuning](instruction-finetuning.md) — what Vicuna and Orca are doing to Llama.
- [Sequence-to-sequence and encoder-decoder models](seq2seq-and-encoder-decoder.md) — the
  Assignment 3 machinery the pre-LLM semantic parsers reused.
- [Evaluating LLMs](evaluating-llms.md) — benchmark design, and its limits.
