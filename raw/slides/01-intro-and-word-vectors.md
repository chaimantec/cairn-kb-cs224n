---
title: Lecture 1 — Intro and Word Vectors (slide deck)
lecture: 1
slides: 40
source_pdf: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture01-wordvecs1.pdf
note: Slide numbers printed on the slides match the PDF page numbers exactly.
---

# Lecture 1 — Intro and Word Vectors: slide-by-slide

Text and figures of all 40 slides of
[`cs224n-spr2024-lecture01-wordvecs1.pdf`](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture01-wordvecs1.pdf),
transcribed from the deck. Cite these as "slide N" — the printed number equals the
PDF page number, so "slide 28" is page 28 of that PDF. Diagrams and plots are
described in prose since the KB is read as text.

Companion pages: [wiki page for this lecture](../../wiki/01-intro-and-word-vectors.md) ·
[transcript](../transcripts/01-intro-and-word-vectors.md)

## Contents

| Slides | Section |
| ------ | ------- |
| 1 | Title |
| 2 | Lecture plan and the key takeaway |
| 3–8 | §1 The course: logistics, staff, learning goals, grading, assignments |
| 9 | Lecture plan (§1 done) |
| 10–16 | §2 Human language and word meaning: why language matters, what neural NLP can now do |
| 17–21 | How to represent meaning: denotational semantics, WordNet, one-hot vectors and their failure |
| 22–24 | Distributional semantics and word vectors |
| 25–27 | §3 Word2vec overview: skip-gram, center/outside words, sliding window |
| 28–30 | §4 Word2vec objective function, two vectors per word, softmax prediction function |
| 31–32 | Parameters θ, training as optimization; interactive session |
| 33–36 | The gradient derivation, worked by hand (neat handwritten slides) |
| 37–39 | §5 Optimization: gradient descent, the update rule, stochastic gradient descent |
| 40 | Lecture plan (§6 = Jupyter notebook demo) |

---

## Slide 1 — Title

"Natural Language Processing with Deep Learning — CS224N/Ling284". Christopher
Manning. "Lecture 1: Introduction and Word Vectors". Stanford arch logo.

## Slide 2 — Lecture Plan

The six sections with their time budgets:

1. The course (10 mins)
2. Human language and word meaning (15 mins)
3. Word2vec introduction (15 mins)
4. Word2vec objective function gradients (25 mins)
5. Optimization basics (5 mins)
6. Looking at word vectors (10 mins or less)

> **Key learning today: The (astounding!) result that word meaning can be
> represented rather well by a (high-dimensional) vector of real numbers**

## Slide 3 — Course logistics in brief

- Instructor: Christopher Manning
- Head TA: Shikhar Murty
- Course Manager: John Cho
- TAs: many — see website
- Time: Tu/Th 4:30–5:50 Pacific, Nvidia Aud. (livestreamed to PanOpto video)
- Email list: cs224n-spr2324-staff@lists.stanford.edu
- Class webpage: http://cs224n.stanford.edu/ a.k.a. http://www.stanford.edu/class/cs224n/
  — TAs, syllabus, help sessions/office hours, Ed for all course questions
- Office hours start **Wednesday**
- Python/numpy then PyTorch tutorials: first two Fridays (4/5, 4/12), 3:30–4:20, Gates B01
- Slide PDFs uploaded before each lecture

## Slide 4 — Course staff photos

A photo grid: instructor Chris Manning; head TA Shikhar Murty; course manager
John Cho; and TAs Aditya Agrawal, Anna Goldie, Archit Sharma, Arvind Mahankali,
Chaofei Fan, Jingwen Wu, Johnny Chang, Kamyar Salahi, Kaylee Burns, Moussa
Doumbouya, Neil Nie, Olivia Lee, Rashon Poole, Ryan Li, Shijia Yang, Soumya
Chatterjee, Timothy Dai, Yuan Gao, Zhoujie Ding.

## Slide 5 — What do we hope to teach? (a.k.a. "learning goals")

1. **The foundations of the effective modern methods for deep learning applied to
   NLP.** Basics first: word vectors, feed-forward networks, recurrent networks,
   attention. Then key methods used in NLP in 2024: transformers,
   encoder-decoder models, pretraining, post-training (RLHF, SFT), efficient
   adaptation, model interpretability, language model agents, etc.
2. **A big picture understanding of human languages** and the difficulties in
   understanding and producing them via computers.
3. **An understanding of and ability to build systems** (in PyTorch) for some of
   the major problems in NLP: word meaning, dependency parsing, machine
   translation, question answering.

## Slide 6 — Course work and grading policy

- 4 × mainly 1.5-week assignments: 6% + 3 × 14% = **48%**
  - HW1 released today, due next Tuesday at 4:30 p.m.
  - Submitted to Gradescope in Canvas (use @stanford.edu email)
- Final Default or Custom Course Project (1–3 people): **49%**
  - Project proposal 8%, milestone 6%, poster 3%, report 32%
- Participation: **3%** (guest lecture reactions, Ed, course evals, karma)
- Late day policy: 6 free late days; afterwards 1% off total course grade per day
  late. Assignments not accepted more than 3 days late per assignment without
  advance permission.

## Slide 7 — Course work and grading policy (collaboration and AI)

- **Collaboration policy:** read the website and the Honor Code. Don't take code
  off the web; acknowledge working with other students; write your own assignment
  solutions. Students must independently submit their solutions to CS224N
  homeworks.
- **AI tools policy:** "Large language models are great (!), but we don't want
  ChatGPT's solutions to our assignments." Collaborative coding with AI tools is
  allowed; asking it to answer questions is strictly prohibited. Employing AI
  tools to substantially complete assignments is an Honor Code violation.

## Slide 8 — High-Level Plan for Assignments (to be completed individually!)

- Ass 1: an easy on-ramp — a Jupyter/IPython notebook
- Ass 2: expects (multivariate) calculus so you really understand the basics,
  introduces PyTorch, and you build a feed-forward network for dependency parsing
- Ass 3 and Ass 4: PyTorch on a GPU (Google Cloud). "Libraries like PyTorch,
  Tensorflow, and Jax are now the standard tools of DL"
- Final Project: either the **default project** — implement a BERT LLM then
  fine-tune and adapt it for downstream tasks, open-ended but an easier start —
  or **propose a custom final project**, which the staff approve, with feedback
  from a mentor (TA/prof/postdoc/PhD). Teams of 1–3; any language/packages.

## Slide 9 — Lecture Plan

Same six-item list as slide 2, with item 1 greyed out to mark it complete.

## Slide 10 — Photograph (no text)

Full-bleed photograph of the Long Room of Trinity College Library, Dublin — a
barrel-vaulted hall lined floor to ceiling with old books and marble busts. Used
as the visual opening for the discussion of human language and accumulated
knowledge.

## Slide 11 — xkcd comic (no title)

Stick-figure comic. One figure says: "I don't mean to go all language nerd on
you, but I just legit adverbed 'legit,' verbed 'adverb,' and adjectived 'language
nerd.'" Illustrates that language is a flexible, changing system that its speakers
constantly reinvent.

## Slide 12 — Neural machine translation was an early big success of Neural NLP

Screenshot from the Kenyan news site TUKO (kiswahili.tuko.co.ke), showing a
Swahili headline "Malawi yawapoteza mawaziri 2 kutokana na maafa ya COVID-19"
passed through Google Translate (Swahili → English) to produce "Malawi loses 2
ministers due to COVID-19 disaster", with the body text likewise translated:
"TUKO.co.ke has been informed that local government minister Lingson Belekanyama
and his transport counterpart Sidik Mia died within two separate hours."

## Slide 13 — Free-text question answering: Next gen search

Example query and answer:

> `when did Kendrick lamar's first album come out?` → `July 2, 2011`

alongside a Wikipedia passage containing the answer. Cites **YONO** (Lee et al.
2021, https://arxiv.org/pdf/2112.07381.pdf), which "uses a T5-Large model
fine-tuned for QA 3 times to run entire QA pipeline".

The diagram shows a three-stage architecture over a query and passages 1…n:
a **Retrieval Layer** (per-passage encoders feeding passage-wise hard attention),
a **Reranking Layer** (concatenating query and passage, re-encoding, passage-wise
hard attention), and a **Reading Layer** (encoders → concat all → decoder →
answer), grouped as Encoder and Fused Decoder.

## Slide 14 — GPT-2 (2019): An early step towards foundation models

Marked with the OpenAI logo. A prompt/completion table.

**Prompt (human-written):** "A train carriage containing controlled nuclear
materials was stolen in Cincinnati today. Its whereabouts are unknown."

**Model completion:** "The incident occurred on the downtown train line, which
runs from Covington and Ashland stations. / In an email to Ohio news outlets, the
U.S. Department of Energy said it is working with the Federal Railroad
Administration to find the thief. / 'The theft of this nuclear material will have
significant negative consequences on public and environmental health, our
workforce and the economy of our nation,' said Tom Hicks, the U.S. Energy
Secretary, in a statement. 'Our top priority is to secure the theft and ensure it
doesn't happen again.' / The stolen material was taken from the University of
Cincinnati's Research Triangle Park nuclear research site, according to a news
release from Department officials."

## Slide 15 — ChatGPT, GPT-4, and more

Two screenshots. Left: a ChatGPT exchange where the user writes "Hey please draft
a polite mail to explain my boss Jeremy that I would not be able to come to office
for next 2 days because my 9 year song Peter is angry with me that I am not giving
him much time…" and the model returns a well-formed email to Jeremy that silently
corrects "song" to "son". Right: a GPT-4 vision example — "What is unusual about
this image?" over a photo of a man ironing clothes on an ironing board attached to
the roof of a moving taxi; GPT-4 answers exactly that.

## Slide 16 — DALL·E image generation (full-bleed image)

A generated illustration and its long prompt: "A illustration from a graphic
novel. A bustling city street under the shine of a full moon. The sidewalks
bustling with pedestrians enjoying the nightlife. At the corner stall, a young
woman with fiery red hair, dressed in a signature velvet cloak, is haggling with
the grumpy old vendor. the grumpy vendor, a tall, sophisticated man is wearing a
sharp suit, sports a noteworthy moustache is animatedly conversing on his
steampunk telephone." The image renders essentially every element of the prompt.

## Slide 17 — How do we represent the meaning of a word?

Definition of **meaning** (Webster dictionary):

- the idea that is represented by a word, phrase, etc.
- the idea that a person wants to express by using words, signs, etc.
- the idea that is expressed in a work of writing, art, etc.

**Commonest linguistic way of thinking of meaning:**

> signifier (symbol) ⟺ signified (idea or thing)
>
> = **denotational semantics**
>
> tree ⟺ {🌳, 🌲, 🌴, …}

## Slide 18 — How do we have usable meaning in a computer?

**Previously commonest NLP solution:** use e.g. **WordNet**, a thesaurus
containing lists of **synonym sets** and **hypernyms** ("is a" relationships).

Two NLTK code examples with their output. Synonym sets containing "good" produce
noun: good; noun: good, goodness; noun: commodity, trade_good, good; adj: good;
adj (sat): full, good; adj (sat): estimable, good, honorable, respectable; adj
(sat): beneficial, good; adj (sat): good, just, upright; adverb: well, good;
adverb: thoroughly, soundly, good. Hypernyms of "panda" produce the chain
procyonid → carnivore → placental → mammal → vertebrate → chordate → animal →
organism → living_thing → whole → object → physical_entity → entity.

## Slide 19 — Problems with resources like WordNet

- A useful resource but **missing nuance**: "proficient" is listed as a synonym
  for "good" — only correct in some contexts. Also WordNet lists offensive
  synonyms in some synonym sets without any coverage of connotations or
  appropriateness.
- **Missing new meanings of words**: wicked, badass, nifty, wizard, genius, ninja,
  bombest. Impossible to keep up-to-date.
- Subjective.
- Requires human labor to create and adapt.
- Can't be used to accurately compute word similarity.

## Slide 20 — Representing words as discrete symbols

In traditional NLP we regard words as discrete symbols: *hotel*, *conference*,
*motel* — a **localist** representation. Such symbols can be represented by
**one-hot** vectors (annotated "Means one 1, the rest 0s"):

```
motel = [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
hotel = [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
```

Vector dimension = number of words in vocabulary (e.g. 500,000+).

## Slide 21 — Problem with words as discrete symbols

**Example:** in web search, if a user searches for "Seattle motel", we would like
to match documents containing "Seattle hotel". But with the one-hot vectors above,
**these two vectors are orthogonal** — there is no natural notion of **similarity**
for one-hot vectors.

**Solution:**

- Could try to rely on WordNet's list of synonyms to get similarity? But it is
  well-known to fail badly: incompleteness, etc.
- **Instead: learn to encode similarity in the vectors themselves.**

## Slide 22 — Representing words by their context

Photograph of J.R. Firth in the corner.

- **Distributional semantics: A word's meaning is given by the words that
  frequently appear close-by.**
  - *"You shall know a word by the company it keeps"* (J. R. Firth 1957: 11)
  - One of the most successful ideas of modern statistical NLP!
- When a word *w* appears in a text, its **context** is the set of words that
  appear nearby (within a fixed-size window).
- We use the many contexts of *w* to build up a representation of *w*.

Three example lines with *banking* highlighted: "…government debt problems turning
into **banking** crises as happened in 2009…", "…saying that Europe needs unified
**banking** regulation to replace the hodgepodge…", "…India has just given its
**banking** system a shot in the arm…". Caption: "These **context words** will
represent *banking*".

## Slide 23 — Word vectors

We will build a dense vector for each word, chosen so that it is similar to
vectors of words that appear in similar contexts, measuring similarity as the
vector **dot (scalar) product**. Two 8-dimensional example vectors are shown
side by side:

| | banking | monetary |
| - | ------- | -------- |
| | 0.286 | 0.413 |
| | 0.792 | 0.582 |
| | −0.177 | −0.007 |
| | −0.107 | 0.247 |
| | 0.109 | 0.216 |
| | −0.542 | −0.718 |
| | 0.349 | 0.147 |
| | 0.271 | 0.051 |

Note: word vectors are also called **(word) embeddings** or **(neural) word
representations**. They are a **distributed** representation.

## Slide 24 — Word meaning as a neural word vector – visualization

A 9-dimensional vector for *expect* (0.286, 0.792, −0.177, −0.107, 0.109, −0.542,
0.349, 0.271, 0.487) beside a 2-D scatter plot of verbs. Visible structure: *need*
and *help* top right; *come*/*go* together; *give*, *keep*, *make*, *get*, *take*,
*meet*, *see*, *continue* in a cluster; *expect*, *want*, *think*, *say* to the
left; *become* and *remain* near *be*, *are*, *is*, *was*, *were*; and
*being*/*been*/*had*/*has*/*have* lower down.

## Slide 25 — 3. Word2vec: Overview

**Word2vec is a framework for learning word vectors** (Mikolov et al. 2013).

Idea:

- We have a large corpus ("body") of text: a long list of words
- Every word in a fixed vocabulary is represented by a **vector**
- Go through each position *t* in the text, which has a center word *c* and
  context ("outside") words *o*
- Use the **similarity of the word vectors** for *c* and *o* to **calculate the
  probability** of *o* given *c* (or vice versa)
- **Keep adjusting the word vectors** to maximize this probability

Diagram, labelled "Skip-gram model (Mikolov et al. 2013)": a single input `w(t)`
feeding a projection layer, which fans out to four outputs `w(t−2)`, `w(t−1)`,
`w(t+1)`, `w(t+2)`.

## Slide 26 — Word2Vec Overview (window at *into*)

"Example windows and process for computing P(w_{t+j} | w_t)".

The sentence strip `… problems turning into banking crises as …` with **into**
highlighted as the center word at position *t*. Arrows fan out from it labelled
P(w_{t−2} | w_t), P(w_{t−1} | w_t), P(w_{t+1} | w_t), P(w_{t+2} | w_t). Braces
mark "outside context words in window of size 2" on each side (*problems
turning* and *banking crises*) and "center word at position t" in the middle.

## Slide 27 — Word2Vec Overview (window advanced to *banking*)

Identical to slide 26 with the window slid one position right: **banking** is now
the center word, the left context is *turning into* and the right context is
*crises as*. The pair of slides animates the window moving through the text.

## Slide 28 — Word2vec: objective function

For each position *t* = 1, …, *T*, predict context words within a window of fixed
size *m*, given center word *w_t*. Data likelihood:

> Likelihood = L(θ) = ∏_{t=1}^{T} ∏_{−m ≤ j ≤ m, j≠0} P(w_{t+j} | w_t ; θ)

annotated "θ is all variables to be optimized".

The **objective function** J(θ) is the (average) negative log likelihood
(annotated "sometimes called a *cost* or *loss* function"):

> J(θ) = −(1/T) log L(θ) = −(1/T) ∑_{t=1}^{T} ∑_{−m ≤ j ≤ m, j≠0} log P(w_{t+j} | w_t ; θ)

> **Minimizing objective function ⟺ Maximizing predictive accuracy**

## Slide 29 — Word2vec: objective function (two vectors per word)

Restates J(θ), then:

- **Question:** How to calculate P(w_{t+j} | w_t ; θ)?
- **Answer:** We will *use two* vectors per word *w*:
  - `v_w` when *w* is a **center** word
  - `u_w` when *w* is a **context** word
  - (annotated: "These word vectors are subparts of the big vector of all
    parameters θ")
- Then for a center word *c* and a context word *o*:

> P(o|c) = exp(u_oᵀ v_c) / ∑_{w ∈ V} exp(u_wᵀ v_c)

## Slide 30 — Word2vec: prediction function

The same equation, annotated in three numbered steps:

1. **Dot product compares similarity of *o* and *c*.** uᵀv = u·v = ∑_{i=1}^{n} u_i v_i.
   Larger dot product = larger probability.
2. **Exponentiation makes anything positive** (points at the numerator).
3. **Normalize over entire vocabulary to give probability distribution** (points
   at the denominator).

- This is an example of the **softmax function** ℝⁿ → (0,1)ⁿ (annotated "Open
  region"):

> softmax(x_i) = exp(x_i) / ∑_{j=1}^{n} exp(x_j) = p_i

- The softmax function maps arbitrary values x_i to a probability distribution p_i
  - "max" because it amplifies probability of largest x_i
  - "soft" because it still assigns some probability to smaller x_i
  - Frequently used in Deep Learning
  - (annotated: "But sort of a weird name because it returns a distribution!")

## Slide 31 — To train the model: Optimize value of parameters to minimize loss

To train a model, we gradually adjust parameters to minimize a loss.

- Recall: θ represents **all** the model parameters, in one long vector
- With *d*-dimensional vectors and *V*-many words, θ is the stacked column
  `[v_aardvark, v_a, …, v_zebra, u_aardvark, u_a, …, u_zebra] ∈ ℝ^{2dV}`
- **Remember: every word has two vectors**
- We optimize these parameters by walking down the gradient (see right figure)
- We compute **all** vector gradients!

The right figure is a contour plot of `z = x² + 2y²` with a path of small arrows
spiralling from the upper left down into the minimum at the centre.

## Slide 32 — Interactive Session!

Just the two equations, for working through live:

> L(θ) = ∏_{t=1}^{T} ∏_{−m ≤ j ≤ m, j≠0} P(w_{t+j} | w_t ; θ)
>
> For a center word *c* and a context word *o*: P(o|c) = exp(u_oᵀ v_c) / ∑_{w ∈ V} exp(u_wᵀ v_c)

## Slide 33 — (Handwritten) 4. Objective Function

Handwritten slide, neatly prepared. Maximize J′(θ) = ∏_{t=1}^{T} ∏_{−m≤j≤m, j≠0}
p(w′_{t+j} | w_t ; θ). "Or minimize ave. neg. log likelihood":

> J(θ) = −(1/T) ∑_{t=1}^{T} ∑_{−m≤j≤m, j≠0} log p(w′_{t+j} | w_t)

Margin annotations: "[negate to minimize; log is monotone]", "Text length" pointing
at *T*, "window size" pointing at *m*, "word IDs". And:

> where p(o|c) = exp(u_oᵀ V_c) / ∑_{w=1}^{V} exp(u_wᵀ V_c)

with the note "Each word type (vocab entry) has **two** word representations: as
center word and context word" and "We now take derivatives to work out minimum".

## Slide 34 — (Handwritten) Splitting the derivative

Handwritten. Takes ∂/∂v_c of log[ exp(u_oᵀv_c) / ∑_{w=1}^{V} exp(u_w v_c) ] and
splits it into term ① ∂/∂v_c log exp(u_oᵀv_c) minus term ② ∂/∂v_c log ∑_{w=1}^{V}
exp(u_wᵀv_c).

Term ① is worked out: log and exp are "inverses" and cancel, so it becomes
∂/∂v_c u_oᵀv_c = **u_o**. Margin notes: "Vector! Not high school single variable
calculus"; "You can do things one variable at a time, and this may be helpful when
things get gnarly", with the componentwise version ∀j ∂/∂(v_c)_j u_oᵀv_c =
∂/∂(v_c)_j ∑_{i=1}^{d} (u_o)_i (v_c)_i = (u_o)_j, "Each term is zero except when
i = j".

## Slide 35 — (Handwritten) The denominator, via the chain rule

Handwritten. Term ② ∂/∂v_c log ∑_{w=1}^{V} exp(u_wᵀv_c), with *f* and
*z = g(v_c)* labelled, and the chain rule written out as ∂/∂v_c f(g(v_c)) =
∂f/∂z · ∂z/∂v_c.

Result: 1 / ∑_{w=1}^{V} exp(u_wᵀv_c) · ∂/∂v_c ∑_{x=1}^{V} exp(u_xᵀv_c), with the
margin note "**Important to change index**". Then "Move deriv inside sum" and a
second application of the chain rule gives ∑_{x=1}^{V} exp(u_xᵀv_c) · ∂/∂v_c
u_xᵀv_c = ∑_{x=1}^{V} exp(u_xᵀv_c) u_x.

## Slide 36 — (Handwritten) observed − expected

Handwritten, the payoff slide. Combining the two terms:

> ∂/∂v_c log p(o|c) = u_o − [1 / ∑_{w=1}^{V} exp(u_wᵀv_c)] · (∑_{x=1}^{V} exp(u_xᵀv_c) u_x)

"Distribute term across sum" gives

> = u_o − ∑_{x=1}^{V} [exp(u_xᵀv_c) / ∑_{w=1}^{V} exp(u_wᵀv_c)] u_x
> = u_o − ∑_{x=1}^{V} p(x|c) u_x

with the annotation "This an expectation: average over all context vectors
weighted by their probability" and, boxed as the conclusion:

> **= observed − expected**

Closing notes in blue: "This is just the derivatives for the center vector
parameters. Also need derivatives for output vector parameters (they're similar).
Then we have derivative w.r.t. all parameters and can minimize."

## Slide 37 — 5. Optimization: Gradient Descent

- We have a cost function J(θ) we want to minimize
- **Gradient Descent** is an algorithm to minimize J(θ)
- **Idea:** for current value of θ, calculate gradient of J(θ), then take a small
  step in direction of negative gradient. Repeat.

Figure: a convex curve of Cost against θ, with a "Random initial value" on the
left, a series of shrinking "Learning step" arrows walking down the slope, and the
"Minimum" marked at θ̂. Side notes: "Note: Our objectives may not be convex like
this ☹" and "But life turns out to be okay ☺".

## Slide 38 — Gradient Descent (the update rule)

Update equation in matrix notation:

> θ^new = θ^old − α ∇_θ J(θ)

annotated "α = *step size* or *learning rate*".

Update equation for a single parameter:

> θ_j^new = θ_j^old − α ∂/∂θ_j^old J(θ)

Algorithm:

```python
while True:
    theta_grad = evaluate_gradient(J,corpus,theta)
    theta = theta - alpha * theta_grad
```

## Slide 39 — Stochastic Gradient Descent

- **Problem:** J(θ) is a function of **all** windows in the corpus (potentially
  billions!). So ∇_θ J(θ) is very expensive to compute.
- You would wait a very long time before making a single update!
- **Very** bad idea for pretty much all neural nets!
- **Solution: Stochastic gradient descent (SGD)** — repeatedly sample windows, and
  update after each one.
- Labelled in large type at the right: **Mini Batch Gradient Descent**

Algorithm:

```python
while True:
    window = sample_window(corpus)
    theta_grad = evaluate_gradient(J,window,theta)
    theta = theta - alpha * theta_grad
```

Note: the deck does **not** state a mini-batch size. The transcript garbles the
figure Manning quotes aloud (≈6:21 of lecture 2), so the specific number is not
recoverable from the course materials in this KB.

## Slide 40 — Lecture Plan

The six-item list again, items 1–5 greyed out, item 6 "Looking at word vectors (10
mins or less)" active with the sub-bullet "See Jupyter Notebook". The notebook demo
was actually given at the start of lecture 2.
