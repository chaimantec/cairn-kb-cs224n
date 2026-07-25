---
title: Lecture 5 — Language Models and Recurrent Neural Networks (slide deck)
lecture: 5
slides: 72 printed / 72 pages in the PDF
source_pdf: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture05-rnnlm.pdf
note: Printed slide numbers match PDF page numbers 1:1. A few pages carry no printed number (3–6, 11, 12) but occupy their position in the sequence, so slide N is page N throughout.
---

# Lecture 5 — Language Models and Recurrent Neural Networks: slide-by-slide

Text and figures of
[`cs224n-spr2024-lecture05-rnnlm.pdf`](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture05-rnnlm.pdf),
transcribed from the deck. Diagrams and plots are described in prose since the KB is
read as text.

**Slide numbers vs PDF pages.** The deck has 72 pages and the printed numbers run 1–72
with no gaps and no offset: **slide N is PDF page N**. Six pages (3, 4, 5, 6, 11, 12)
print no number in the corner, but they sit in sequence — page 10 prints "10" and
page 13 prints "13" — so nothing is missing.

The title slide calls this *Lecture 5: Language Models and Recurrent Neural Networks*;
the course catalog lists it as *Recurrent Neural Networks*.

Companion pages: [wiki page for this lecture](../../wiki/05-recurrent-neural-networks.md) ·
[transcript](../transcripts/05-recurrent-neural-networks.md)

## Contents

| Slides | Section |
| ------ | ------- |
| 1–2 | Title and lecture plan |
| 3–4 | Who is in CS224N — enrollment pie charts |
| 5–6 | Modern networks are enormous; deep nets were once thought untrainable |
| 7 | §1 Regularization and the modern view of overfitting |
| 8–9 | Dropout |
| 10 | Vectorization — matrices instead of for loops |
| 11 | Parameter initialization; Xavier initialization |
| 12 | Optimizers — SGD, Adagrad, RMSprop, Adam, AdamW, NAdamW |
| 13–14 | §2 Language Modeling defined; probability of a piece of text |
| 15–16 | You use language models every day — phone keyboard, Google search |
| 17–18 | n-gram language models; the Markov assumption and counting |
| 19 | 4-gram worked example: "the students opened their ___" |
| 20 | Sparsity problems 1 and 2; smoothing and backoff |
| 21 | Storage problems |
| 22 | A trigram LM over 1.7M words of Reuters in practice |
| 23–26 | Generating text with an n-gram LM; the "today the price of gold" sample |
| 27–30 | A fixed-window neural language model, its improvements and its limits |
| 31 | §3 Recurrent Neural Networks — the core idea |
| 32–33 | A simple RNN language model; RNN advantages and disadvantages |
| 34–40 | Training an RNN-LM: cross-entropy loss, teacher forcing, SGD over sentences |
| 41–43 | Backpropagation through time; the multivariable chain rule |
| 44–48 | Generating with an RNN-LM: Obama speeches, Harry Potter, recipes, paint colors |
| 49–50 | Evaluation: perplexity, and the perplexity table showing RNNs winning |
| 51–56 | §4 Vanishing and exploding gradients — the intuition, step by step |
| 57–58 | Vanishing gradient proof sketch (linear case), eigenvalue argument |
| 59–60 | Why vanishing gradient is a problem; the "printer/tickets" example |
| 61–62 | Exploding gradients and gradient clipping |
| 63 | How to fix vanishing gradients — a look ahead to LSTMs and attention |
| 64–65 | §5 Recap; why we should care about language modeling |
| 66–71 | Other RNN uses: tagging, sentence classification, encoder module, conditional LM |
| 72 | Terminology and a look forward — Elman RNN, LSTM, GRU |

---

## Slide 1 — Title

**Natural Language Processing with Deep Learning — CS224N/Ling284**

Christopher Manning. Lecture 5: Language Models and Recurrent Neural Networks.

The Stanford logo (a red arch over three sandstone-coloured arches) sits in the middle
of the slide.

## Slide 2 — Lecture Plan

1. A bit more about neural networks (10 mins)

Language modeling + RNNs

- 2. A new NLP task: **Language Modeling** (20 mins)
  — annotated in a magenta box: *"This is the most important concept in the class! It leads to BERT, GPT-3 and ChatGPT!"* A magenta arrow labelled **motivates** runs from item 2 down to item 3.
- 3. A new family of neural networks: **Recurrent Neural Networks (RNNs)** (25 mins)
  — annotated: *"Important and used in Ass4, but not the only way to build LMs"*
- 4. Problems with RNNs (15 mins)
- 5. Recap on RNNs/LMs (10 mins)

Reminder: Thursday: Assignment 2 is due; Assignment 3, using RNNs for machine
translation, out.

## Slide 3 — Who is in CS224N?

A pie chart titled "CS224N Spring 2024 Enrollment - Coarse Grouping". The largest
wedges are Computer Science (Grad), Computer Science (Undergrad), NDO Grad (SCPD) and
Undeclared (Undergrad) — together roughly two-thirds of the class. The remaining
wedges are thin slivers, labelled with leader lines: SymSys (Undergrad and Grad),
Sustainability (Undergrad and Grad), Statistics (Grad), Social Sci (Undergrad and
Grad), Nat Sci (Undergrad and Grad), MS&E (Grad), Medicine, Law, GSB, Humanities,
Engineering (Grad and Undergrad), EE (Grad), Education (Grad), Data Sci (Undergrad),
CS+Soc Sci, CS+Nat Sci, CS+Humanities and CS+Engineering (all Undergrad).

## Slide 4 — Enrollment by level

A pie chart titled "CS224N 2024 Enrollment by Level". GradYear1 is the largest wedge
(about a third), followed by Senior and Junior at roughly a fifth each, then
GradYear2, GradYear4+, Sophomore, GradYear3, and slivers for ClinicYr1 and Freshman.

## Slide 5 — Modern neural networks (esp. language models) are *enormous*

A log-scale line plot of model size in billions of parameters (0.01 to 1000) against
year (2018 to 2022), with a dashed red trend line running straight through the points
— i.e. roughly exponential growth. Labelled points: ELMo (94M), BERT-Large (340M),
GPT-2 (1.5B), Megatron-LM (8.3B), T5 (11B), Turing-NLG (17.2B), GPT-3 (175B),
Megatron-Turing NLG (530B).

- Large, deep neural nets are a cornerstone of modern NLP systems

Source: https://huggingface.co/blog/large-language-models

## Slide 6 — But building large neural networks wasn't easy or obvious

A screenshot of the paper **"Greedy Layer-Wise Training of Deep Networks"** [Bengio et
al 2006] — Yoshua Bengio, Pascal Lamblin, Dan Popovici, Hugo Larochelle, Université de
Montréal. A highlighted passage from its introduction reads:

> However, until recently, it was believed too difficult to train deep multi-layer
> neural networks. Empirically, deep networks were generally found to be not better,
> and often worse, than neural networks with one or two hidden layers (Tesauro, 1992).
> As this is a negative result, it has not been much reported in the machine learning
> literature. A reasonable explanation is that gradient-based optimization starting
> from random initialization may get stuck near poor solutions.

- It took a long time and much work to make deep neural networks practical!

## Slide 7 — We have models with many parameters! Regularization!

- A full loss function includes **regularization** over all parameters θ, e.g., L2
  regularization:

  J(θ) = (1/N) Σ_{i=1..N} −log( e^{f_{y_i}} / Σ_{c=1..C} e^{f_c} ) **+ λ Σ_k θ_k²**

  (the boxed term on the right is the regularizer)

- Classic view: Regularization works to prevent **overfitting** when we have a lot of
  features (or later a very powerful/deep model, etc.)
- Now: Regularization **produces models that generalize well** when we have a "big" model
  - We do not care that our models overfit on the training data, even though they are
    **hugely** overfit

Two plots sit at the bottom. On the left, the textbook picture: error against model
"power", with training error falling monotonically and test error falling then rising
again, the gap between them labelled **overfitting**. On the right, a real measured
curve of zero-one loss (%) against training, where the test curve rises to a bump at a
dashed vertical line and then *falls* again, continuing down to near the training
curve — the empirical picture that motivates the "now" view.

## Slide 8 — Dropout (Srivastava, Hinton, Krizhevsky, Sutskever, & Salakhutdinov 2012/JMLR 2014)

- During training
  - For each data point each time:
    - Randomly set input to 0 with probability *p* "dropout ratio" (often *p* = 0.5
      except *p* – 0.15 for input layer) via dropout mask
- During testing
  - Multiply all weights by 1 − *p*
  - No other dropout

Three small network diagrams illustrate this. Each has inputs x₁…x₄ plus a bias unit 1
feeding a single output through weights w₁…w₄ and b.

- **Train 1**: x₂ and x₄ are zeroed. y = w₁x₁ + w₃x₃ + b
- **Train 2**: x₄ is zeroed. y = w₁x₁ + w₂x₂ + w₃x₃
- **Test**: all inputs present. y = (1 − p)(w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄)

## Slide 9 — Dropout, why it works

Why does it work?

> Prevents Feature Co-adaptation = Good Regularization! **Use it everywhere!**

Let's talk through an example..

- Training time: at each instance of evaluation (in online SGD-training), randomly set
  ~50% (*p*%) of the inputs to each neuron to 0 (less for the first layer)
- Test time: halve the model weights (now twice as many)
- No co-adaptation: A feature cannot only be useful in the presence of particular
  other features

In a single layer: A kind of middle-ground between Naïve Bayes (all feature weights
set independently) and logistic regression models (weights are set in the context of
all others)

- Can be thought of as a form of model bagging (i.e., like an ensemble model)
- Nowadays usually thought of as strong, feature-dependent regularizer
  [Wager, Wang, & Liang 2013]

## Slide 10 — "Vectorization"

- E.g., looping over word vectors versus concatenating them all into one large matrix
  and then multiplying the softmax weights with that matrix:

```python
from numpy import random
N = 500 # number of windows to classify
d = 300 # dimensionality of each window
C = 5   # number of classes
W = random.rand(C,d)
wordvectors_list = [random.rand(d,1) for i in range(N)]
wordvectors_one_matrix = random.rand(d,N)

%timeit [W.dot(wordvectors_list[i]) for i in range(N)]
%timeit W.dot(wordvectors_one_matrix)
```

- for loop: 1000 loops, best of 3: **639 µs** per loop
  Using a single C x N matrix: 10000 loops, best of 3: **53.8 µs** per loop
- Matrices are awesome!!! Always try to use vectors and matrices rather than for loops!
- The speed gain goes from 1 to 2 orders of magnitude with GPUs!

## Slide 11 — Parameter Initialization

- You normally must initialize weights to small random values (i.e., not zero matrices!)
  - To avoid symmetries that prevent learning/specialization

A 3-D surface plot shows a saddle-shaped loss landscape — two valleys separated by a
ridge, with a red dot marked at the saddle point in the middle.

- Initialize hidden layer biases to 0 and output (or reconstruction) biases to optimal
  value if weights were 0 (e.g., mean target or inverse sigmoid of mean target)
- Initialize **all other weights** ~ Uniform(–*r*, *r*), with *r* chosen so numbers get
  neither too big or too small
  [later, the need for this is removed with use of layer normalization]
- Xavier initialization has variance inversely proportional to fan-in *n_in* (previous
  layer size) and fan-out *n_out* (next layer size):

  Var(W_i) = 2 / (n_in + n_out)

## Slide 12 — Optimizers

- Usually, plain SGD will work just fine!
  - However, getting good results will often require hand-tuning the learning rate
    - E.g., start it higher and halve it every *k* epochs (passes through full data,
      **shuffled** or sampled)
- For more complex nets, or to avoid worry, try more sophisticated "adaptive"
  optimizers that scale the adjustment to individual parameters by an accumulated
  gradient
  - These models give differential per-parameter learning rates
    - Adagrad ← Simplest member of family, but tends to "stall early"
    - RMSprop
    - Adam ← **A fairly good, safe place to begin in many cases**
    - AdamW
    - NAdamW ← Can be better with word vectors (W) and for speed (Nesterov acceleration)
    - …
  - Start them with an initial learning rate, around 0.001 ← Many have other hyperparameters

## Slide 13 — 2. Language Modeling

- **Language Modeling** is the task of predicting what word comes next

  *the students opened their ______*

  Magenta arrows fan out from the blank to four candidate continuations in bubbles:
  *books*, *laptops*, *exams*, *minds*.

- More formally: given a sequence of words **x**⁽¹⁾, **x**⁽²⁾, …, **x**⁽ᵗ⁾, compute the
  probability distribution of the next word **x**⁽ᵗ⁺¹⁾:

  P(**x**⁽ᵗ⁺¹⁾ | **x**⁽ᵗ⁾, …, **x**⁽¹⁾)

  where **x**⁽ᵗ⁺¹⁾ can be any word in the vocabulary V = {**w**₁, …, **w**_{|V|}}

- A system that does this is called a **Language Model**

## Slide 14 — Language Modeling (as a probability of text)

- You can also think of a Language Model as a system that **assigns a probability to a
  piece of text**

- For example, if we have some text **x**⁽¹⁾, …, **x**⁽ᵀ⁾, then the probability of this
  text (according to the Language Model) is:

  P(**x**⁽¹⁾, …, **x**⁽ᵀ⁾) = P(**x**⁽¹⁾) × P(**x**⁽²⁾ | **x**⁽¹⁾) × ⋯ × P(**x**⁽ᵀ⁾ | **x**⁽ᵀ⁻¹⁾, …, **x**⁽¹⁾)

  = ∏_{t=1..T} P(**x**⁽ᵗ⁾ | **x**⁽ᵗ⁻¹⁾, …, **x**⁽¹⁾)

  A brace under the product term is annotated *"This is what our LM provides"*.

## Slide 15 — You use Language Models every day!

A screenshot of a phone messaging app. The composed text reads "I'll meet you at the",
and the keyboard's predictive bar above the keys offers three completions: *cafe*,
**airport** (highlighted), *office*.

## Slide 16 — You use Language Models every day!

A screenshot of Google search. The query box contains "what is the " and the
autocomplete dropdown lists: what is the **weather**, what is the **meaning of life**,
what is the **dark web**, what is the **xfl**, what is the **doomsday clock**, what is
the **weather today**, what is the **keto diet**, what is the **american dream**, what
is the **speed of light**, what is the **bill of rights**.

## Slide 17 — n-gram Language Models

*the students opened their ______*

- **Question**: How to learn a Language Model?
- **Answer** (pre-Deep Learning): learn an *n*-gram Language Model!

- **Definition:** An *n*-gram is a chunk of *n* consecutive words.
  - **uni**grams: "the", "students", "opened", "their"
  - **bi**grams: "the students", "students opened", "opened their"
  - **tri**grams: "the students opened", "students opened their"
  - **four**-grams: "the students opened their"

- **Idea:** Collect statistics about how frequent different n-grams are and use these
  to predict next word.

## Slide 18 — n-gram Language Models (the Markov assumption)

- First we make a **Markov assumption**: *x*⁽ᵗ⁺¹⁾ depends only on the preceding *n*-1 words

  P(**x**⁽ᵗ⁺¹⁾ | **x**⁽ᵗ⁾, …, **x**⁽¹⁾) = P(**x**⁽ᵗ⁺¹⁾ | **x**⁽ᵗ⁾, …, **x**⁽ᵗ⁻ⁿ⁺²⁾)   (assumption)

  The bracket over the conditioning terms is labelled *n*-1 words.

  = P(**x**⁽ᵗ⁺¹⁾, **x**⁽ᵗ⁾, …, **x**⁽ᵗ⁻ⁿ⁺²⁾) / P(**x**⁽ᵗ⁾, …, **x**⁽ᵗ⁻ⁿ⁺²⁾)   (definition of conditional prob)

  The numerator is annotated *prob of an n-gram*, the denominator *prob of a (n-1)-gram*.

- **Question:** How do we get these *n*-gram and (*n*-1)-gram probabilities?
- **Answer:** By **counting** them in some large corpus of text!

  ≈ count(**x**⁽ᵗ⁺¹⁾, **x**⁽ᵗ⁾, …, **x**⁽ᵗ⁻ⁿ⁺²⁾) / count(**x**⁽ᵗ⁾, …, **x**⁽ᵗ⁻ⁿ⁺²⁾)   (statistical approximation)

## Slide 19 — n-gram Language Models: Example

Suppose we are learning a **4-gram** Language Model.

*~~as the proctor started the clock, the~~ students opened their _____*

The struck-through portion is labelled **discard**; the retained "students opened
their" is braced and labelled **condition on this**.

P(**w** | students opened their) = count(students opened their **w**) / count(students opened their)

For example, suppose that in the corpus:

- "students opened their" occurred 1000 times
- "students opened their **books**" occurred **400** times
  - → P(books | students opened their) = **0.4**
- "students opened their **exams**" occurred **100** times
  - → P(exams | students opened their) = **0.1**

A brace beside the last two lines asks: *Should we have discarded the "proctor" context?*

## Slide 20 — Sparsity Problems with n-gram Language Models

Centred on the counting equation P(**w** | students opened their) = count(students
opened their **w**) / count(students opened their), with the numerator and denominator
each called out:

**Sparsity Problem 1** (numerator):
- **Problem:** What if *"students opened their w"* never occurred in data? Then *w* has probability 0!
- **(Partial) Solution:** Add small δ to the count for every *w* ∈ *V*. This is called *smoothing*.

**Sparsity Problem 2** (denominator):
- **Problem:** What if *"students opened their"* never occurred in data? Then we can't calculate probability for *any w*!
- **(Partial) Solution:** Just condition on *"opened their"* instead. This is called *backoff*.

**Note:** Increasing *n* makes sparsity problems *worse*. Typically, we can't have *n* bigger than 5.

## Slide 21 — Storage Problems with n-gram Language Models

The same counting equation, with the numerator called out:

**Storage**: Need to store count for all *n*-grams you saw in the corpus.

> Increasing *n* or increasing corpus increases model size!

## Slide 22 — n-gram Language Models in practice

- You can build a simple trigram Language Model over a 1.7 million word corpus
  (Reuters) in a few seconds on your laptop*
  — Reuters is annotated *Business and financial news*.

*today the _______*

An arrow labelled *get probability distribution* points to the model's output:

```
company  0.153
bank     0.153
price    0.077
italian  0.039
emirate  0.039
         …
```

The top two entries are boxed and labelled **Sparsity problem**: *not much granularity
in the probability distribution*. Beneath: *Otherwise, seems reasonable!*

\* Try for yourself: https://nlpforhackers.io/language-models/

## Slide 23 — Generating text with a n-gram Language Model (step 1)

You can also use a Language Model to **generate text**

*today the _______* — "today the" is braced and labelled **condition on this**; an
arrow labelled *get probability distribution* points to:

```
company  0.153
bank     0.153
price    0.077   ← sample
italian  0.039
emirate  0.039
```

`price` is boxed and labelled **sample**.

## Slide 24 — Generating text with a n-gram Language Model (step 2)

*today the price _______* — "the price" is the conditioning context; the distribution is:

```
of   0.308   ← sample
for  0.050
it   0.046
to   0.046
is   0.031
```

`of` is sampled.

## Slide 25 — Generating text with a n-gram Language Model (step 3)

*today the price of _______* — "price of" is the conditioning context; the distribution is:

```
the   0.072
18    0.043
oil   0.043
its   0.036
gold  0.018   ← sample
```

`gold` is sampled.

## Slide 26 — Generating text with a n-gram Language Model (the result)

The full sample:

> *today the price of gold per ton , while production of shoe lasts and shoe industry ,
> the bank intervened just after it considered and rejected an imf demand to rebuild
> depleted european stocks , sept 30 end primary 76 cts a share .*

**Surprisingly grammatical!**

…but **incoherent.** We need to consider more than three words at a time if we want to
model language well.

But increasing *n* worsens sparsity problem, and increases model size…

## Slide 27 — How to build a *neural* language model?

- Recall the Language Modeling task:
  - Input: sequence of words **x**⁽¹⁾, **x**⁽²⁾, …, **x**⁽ᵗ⁾
  - Output: prob. dist. of the next word P(**x**⁽ᵗ⁺¹⁾ | **x**⁽ᵗ⁾, …, **x**⁽¹⁾)

- How about a **window-based neural model**?
  - We saw this applied to Named Entity Recognition in Lecture 2:

A diagram of the lecture-2 window classifier: five word vectors (*museums*, *in*,
*Paris* underlined, *are*, *amazing*) concatenated into one input row, multiplied by
**W** into a hidden layer, then by **U** to the output label **LOCATION**.

## Slide 28 — A fixed-window neural Language Model (the window)

*~~as the proctor started the clock~~ the students opened their ______*

The first six words are struck through and labelled **discard**; "the students opened
their" is braced and labelled **fixed window**.

## Slide 29 — A fixed-window neural Language Model (the architecture)

Bottom to top:

- words / one-hot vectors **x**⁽¹⁾, **x**⁽²⁾, **x**⁽³⁾, **x**⁽⁴⁾ — *the*, *students*, *opened*, *their*
- concatenated word embeddings **e** = [**e**⁽¹⁾; **e**⁽²⁾; **e**⁽³⁾; **e**⁽⁴⁾]
- hidden layer **h** = f(**We** + **b**₁)
- output distribution **ŷ** = softmax(**Uh** + **b**₂) ∈ ℝ^{|V|}

The output is drawn as a bar chart over the vocabulary from *a* to *zoo*, with arrows
picking out two tall bars labelled *books* and *laptops*.

## Slide 30 — A fixed-window neural Language Model (assessment)

Approximately: Y. Bengio, et al. (2000/2003): A Neural Probabilistic Language Model

**Improvements** over *n*-gram LM:
- No sparsity problem
- Don't need to store all observed *n*-grams

Remaining **problems**:
- Fixed window is too small
- Enlarging window enlarges *W*
- Window can never be large enough!
- *x*⁽¹⁾ and *x*⁽²⁾ are multiplied by completely different weights in *W*. **No symmetry** in how the inputs are processed.

> We need a neural architecture that can process *any length input*

## Slide 31 — 3. Recurrent Neural Networks (RNN): A family of neural architectures

**Core idea:** Apply the same weights *W* *repeatedly*

The diagram, bottom to top: an input sequence (any length) **x**⁽¹⁾, **x**⁽²⁾, **x**⁽³⁾,
**x**⁽⁴⁾, … feeds a row of hidden states **h**⁽¹⁾, **h**⁽²⁾, **h**⁽³⁾, **h**⁽⁴⁾, … each
connected to the next by an arrow labelled **W** — the same **W** every time. Outputs
**ŷ**⁽¹⁾ … **ŷ**⁽⁴⁾ rise from each hidden state and are marked *(optional)*. A magenta
box highlights the single timestep **x**⁽²⁾ → **h**⁽²⁾ → **ŷ**⁽²⁾ as the repeating unit.

## Slide 32 — A Simple RNN Language Model

Bottom to top:

- words / one-hot vectors **x**⁽ᵗ⁾ ∈ ℝ^{|V|} — *the*, *students*, *opened*, *their*
- word embeddings **e**⁽ᵗ⁾ = **Ex**⁽ᵗ⁾ (each obtained through the embedding matrix **E**)
- hidden states **h**⁽ᵗ⁾ = σ( **W**_h **h**⁽ᵗ⁻¹⁾ + **W**_e **e**⁽ᵗ⁾ + **b**₁ )
  — **h**⁽⁰⁾ is the initial hidden state
- output distribution **ŷ**⁽ᵗ⁾ = softmax( **U h**⁽ᵗ⁾ + **b**₂ ) ∈ ℝ^{|V|}

At the top right: **ŷ**⁽⁴⁾ = P(**x**⁽⁵⁾ | the students opened their), drawn as the same
bar chart over *a* … *zoo* with *books* and *laptops* picked out.

*Note*: this input sequence could be much longer now!

## Slide 33 — RNN Language Models: advantages and disadvantages

Same architecture diagram as slide 32, annotated:

RNN **Advantages**:
- Can process **any length** input
- Computation for step *t* can (in theory) use information from **many steps back**
- **Model size doesn't increase** for longer input context
- Same weights applied on every timestep, so there is **symmetry** in how inputs are processed.

RNN **Disadvantages**:
- Recurrent computation is **slow**
- In practice, difficult to access information from **many steps back**

(both marked *More on these later*)

## Slide 34 — Training an RNN Language Model

- Get a **big corpus of text** which is a sequence of words **x**⁽¹⁾, …, **x**⁽ᵀ⁾
- Feed into RNN-LM; compute output distribution **ŷ**⁽ᵗ⁾ for *every step t*.
  - i.e., predict probability dist of *every word*, given words so far

- **Loss function** on step *t* is **cross-entropy** between predicted probability
  distribution **ŷ**⁽ᵗ⁾, and the true next word **y**⁽ᵗ⁾ (one-hot for **x**⁽ᵗ⁺¹⁾):

  J⁽ᵗ⁾(θ) = CE(**y**⁽ᵗ⁾, **ŷ**⁽ᵗ⁾) = − Σ_{w∈V} **y**⁽ᵗ⁾_w log **ŷ**⁽ᵗ⁾_w = − log **ŷ**⁽ᵗ⁾_{**x**_{t+1}}

- Average this to get **overall loss** for entire training set:

  J(θ) = (1/T) Σ_{t=1..T} J⁽ᵗ⁾(θ) = (1/T) Σ_{t=1..T} − log **ŷ**⁽ᵗ⁾_{**x**_{t+1}}

## Slide 35 — Training an RNN Language Model (step 1)

The unrolled RNN over the corpus *the students opened their exams …*, with the
predicted distributions **ŷ**⁽¹⁾…**ŷ**⁽⁴⁾ and losses J⁽¹⁾(θ)…J⁽⁴⁾(θ) drawn above.
J⁽¹⁾(θ) is boxed and annotated *= negative log prob of "students"*.

## Slide 36 — Training an RNN Language Model (step 2)

Same diagram; J⁽²⁾(θ) is boxed and annotated *= negative log prob of "opened"*.

## Slide 37 — Training an RNN Language Model (step 3)

Same diagram; J⁽³⁾(θ) is boxed and annotated *= negative log prob of "their"*.

## Slide 38 — Training an RNN Language Model (step 4)

Same diagram; J⁽⁴⁾(θ) is boxed and annotated *= negative log prob of "exams"*.

## Slide 39 — Training an RNN Language Model ("Teacher forcing")

Same diagram, with the per-step losses now summed:

J⁽¹⁾(θ) + J⁽²⁾(θ) + J⁽³⁾(θ) + J⁽⁴⁾(θ) + … = J(θ) = (1/T) Σ_{t=1..T} J⁽ᵗ⁾(θ)

A magenta box in the top right names the scheme: **"Teacher forcing"**.

## Slide 40 — Training a RNN Language Model (batching)

- However: Computing loss and gradients across **entire corpus** **x**⁽¹⁾, …, **x**⁽ᵀ⁾
  at once is **too expensive** (memory-wise)!

  J(θ) = (1/T) Σ_{t=1..T} J⁽ᵗ⁾(θ)

- In practice, consider **x**⁽¹⁾, …, **x**⁽ᵀ⁾ as a **sentence** (or a **document**)

- Recall: **Stochastic Gradient Descent** allows us to compute loss and gradients for
  small chunk of data, and update.

- Compute loss J(θ) for a sentence (actually, a batch of sentences), compute gradients
  and update weights. Repeat on a new batch of sentences.

## Slide 41 — Backpropagation for RNNs

A chain of hidden states **h**⁽⁰⁾ … **h**⁽ᵗ⁻³⁾, **h**⁽ᵗ⁻²⁾, **h**⁽ᵗ⁻¹⁾, **h**⁽ᵗ⁾, each
linked by the same **W**_h, with the loss J⁽ᵗ⁾(θ) rising from **h**⁽ᵗ⁾.

**Question:** What's the derivative of J⁽ᵗ⁾(θ) w.r.t. the **repeated** weight matrix **W**_h?

**Answer:** ∂J⁽ᵗ⁾/∂**W**_h = Σ_{i=1..t} ∂J⁽ᵗ⁾/∂**W**_h |₍ᵢ₎

> "The gradient w.r.t. a repeated weight is the sum of the gradient w.r.t. each time it appears"

*Why?*

## Slide 42 — Multivariable Chain Rule

- Given a multivariable function f(x, y), and two single variable functions x(t) and
  y(t), here's what the multivariable chain rule says:

  (d/dt) f(x(t), y(t)) = (∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt)

  The left-hand side is braced and labelled *Derivative of composition function*.

Left box: a small diagram with **One input** *t* at the bottom, **Two intermediate
outputs** x(t) and y(t) above it, and **One final output** f(x(t), y(t)) at the top,
arrows running upward.

Right box, headed **Gradients sum at outward branches**: a computation graph where one
node fans out into two, with a **+** marking where the two backward arrows recombine.
Beside it:

  a = x + y
  b = max(y, z)
  f = ab
  ∂f/∂y = (∂f/∂a)(∂a/∂y) + (∂f/∂b)(∂b/∂y)

Source: https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/differentiating-vector-valued-functions/a/multivariable-chain-rule-simple-version

## Slide 43 — Training the parameters of RNNs: Backpropagation for RNNs

The same chain of hidden states, now with fat blue arrows running *backwards* from
J⁽ᵗ⁾(θ) through every **W**_h. Every one of those arrows is labelled *equals* and
points down to a single **W**_h — the point being that they are all the same matrix.

∂J⁽ᵗ⁾/∂**W**_h = Σ_{i=1..t} ∂J⁽ᵗ⁾/∂**W**_h |₍ᵢ₎

**Question:** How do we calculate this?

**Answer:** Backpropagate over timesteps *i* = *t*, …, 0, summing gradients as you go.
This algorithm is called **"backpropagation through time"** [Werbos, P.G., 1988,
*Neural Networks* **1**, and others]

Apply the multivariable chain rule:

  ∂J⁽ᵗ⁾/∂**W**_h = Σ_{i=1..t} ( ∂J⁽ᵗ⁾/∂**W**_h |₍ᵢ₎ ) · ( ∂**W**_h|₍ᵢ₎ / ∂**W**_h )

  where the boxed right-hand factor **= 1**, giving

  = Σ_{i=1..t} ∂J⁽ᵗ⁾/∂**W**_h |₍ᵢ₎

A blue note at the right: *In practice, often "truncated" after ~20 timesteps for
training efficiency reasons*.

## Slide 44 — Generating with an RNN Language Model ("Generating roll outs")

Just like an n-gram Language Model, you can use a RNN Language Model to **generate
text** by **repeated sampling**. Sampled output becomes next step's input.

The diagram starts from `<s>` and runs: sampled *my* → fed back in → *favorite* →
*season* → *is* → *spring* → `</s>`. Each step is marked *sample* on the arrow from
**ŷ**⁽ᵗ⁾ to the emitted word, and the emitted word is fed into the next timestep
through the embedding matrix **E**.

## Slide 45 — Generating text with an RNN Language Model (Obama)

Let's have some fun!
- You can train an RNN-LM on any kind of text, then generate text in that style.
- RNN-LM trained on **Obama speeches** (with a photo of Obama speaking at a podium):

> *The United States will step up to the cost of a new challenges of the American
> people that will share the fact that we created the problem. They were attacked and
> so that they have to say that all the task of the final days of war that I will not
> be able to get this done.*

Source: https://medium.com/@samim/obama-rnn-machine-generated-political-speeches-c8abd18a2ea0

## Slide 46 — Generating text with an RNN Language Model (Harry Potter)

- RNN-LM trained on *Harry Potter* (with a still of Harry holding a wand):

> "Sorry," Harry shouted, panicking—"I'll leave those brooms in London, are they?"
>
> "No idea," said Nearly Headless Nick, casting low close by Cedric, carrying the last
> bit of treacle Charms, from Harry's shoulder, and to answer him the common room
> perched upon it, four arms held a shining knob from when the spider hadn't felt it
> seemed. He reached the teams too.

Source: https://medium.com/deep-writing/harry-potter-written-by-artificial-intelligence-8a9431803da6

## Slide 47 — Generating text with an RNN Language Model (recipes)

- RNN-LM trained on **recipes** (with a photo of mushrooms, celery and peppers):

```
        Title: CHOCOLATE RANCH BARBECUE
   Categories: Game, Casseroles, Cookies, Cookies
        Yield: 6 Servings

   2 tb Parmesan cheese -- chopped
   1  c  Coconut milk
   3     Eggs, beaten

Place each pasta over layers of lumps. Shape mixture into the moderate oven and simmer
until firm. Serve hot in bodied fresh, mustard, orange and cheese.

Combine the cheese and salt together the dough in a large skillet; add the ingredients
and stir in the chocolate and pepper.
```

Source: https://gist.github.com/nylki/1efbaa36635956d35bcc

## Slide 48 — Generating text with a RNN Language Model (paint colors)

- RNN-LM trained on **paint color names**. A swatch chart of invented colors and RGB
  values:

| | |
| --- | --- |
| Ghasty Pink 231 137 165 | Sand Dan 201 172 143 |
| Power Gray 151 124 112 | Grade Bat 48 94 83 |
| Navel Tan 199 173 140 | Light Of Blast 175 150 147 |
| Bock Coe White 221 215 236 | Grass Bat 176 99 108 |
| Horble Gray 178 181 196 | Sindis Poop 204 205 194 |
| Homestar Brown 133 104 85 | Dope 219 209 179 |
| Snader Brown 144 106 74 | Testing 156 101 106 |
| Golder Craam 237 217 177 | Stoner Blue 152 165 159 |
| Hurky White 232 223 215 | Burble Simp 226 181 132 |
| Burf Pink 223 173 179 | Stanky Bean 197 162 171 |
| Rose Hork 230 215 198 | Turdly 190 164 116 |

> This is an example of a **character-level RNN-LM** (predicts what **character** comes next)

Source: http://aiweirdness.com/post/160776374467/new-paint-colors-invented-by-neural-network

## Slide 49 — Evaluating Language Models

- The standard **evaluation metric** for Language Models is **perplexity**.

  perplexity = ∏_{t=1..T} ( 1 / P_LM(**x**⁽ᵗ⁺¹⁾ | **x**⁽ᵗ⁾, …, **x**⁽¹⁾) )^{1/T}

  The product term is braced and labelled *Inverse probability of corpus, according to
  Language Model*; the exponent 1/T is labelled *Normalized by number of words*.

- This is equal to the exponential of the cross-entropy loss J(θ):

  = ∏_{t=1..T} ( 1 / **ŷ**⁽ᵗ⁾_{**x**_{t+1}} )^{1/T} = exp( (1/T) Σ_{t=1..T} − log **ŷ**⁽ᵗ⁾_{**x**_{t+1}} ) = exp(J(θ))

> **Lower** perplexity is better!

## Slide 50 — RNNs greatly improved perplexity over what came before

A table of results, with the *n*-gram model at the top and increasingly complex RNNs
below it, perplexity improving (lower) down the column:

| Model | Perplexity |
| --- | --- |
| Interpolated Kneser-Ney 5-gram (Chelba et al., 2013) | 67.6 |
| RNN-1024 + MaxEnt 9-gram (Chelba et al., 2013) | 51.3 |
| RNN-2048 + BlackOut sampling (Ji et al., 2015) | 68.3 |
| Sparse Non-negative Matrix factorization (Shazeer et al., 2015) | 52.9 |
| LSTM-2048 (Jozefowicz et al., 2016) | 43.7 |
| 2-layer LSTM-8192 (Jozefowicz et al., 2016) | 30 |
| **Ours small** (LSTM-2048) | 43.9 |
| **Ours large** (2-layer LSTM-2048) | 39.8 |

Source: https://research.fb.com/building-an-efficient-neural-language-model-over-a-billion-words/

## Slide 51 — 4. Problems with RNNs: Vanishing and Exploding Gradients

A four-step chain **h**⁽¹⁾ → **h**⁽²⁾ → **h**⁽³⁾ → **h**⁽⁴⁾, each link labelled **W**,
with the loss J⁽⁴⁾(θ) rising from **h**⁽⁴⁾. (Section title slide; the derivation
follows.)

## Slide 52 — Vanishing gradient intuition (1)

The same chain, with a blue backward arrow from J⁽⁴⁾(θ) all the way to **h**⁽¹⁾:

∂J⁽⁴⁾ / ∂**h**⁽¹⁾ = **?**

## Slide 53 — Vanishing gradient intuition (2)

∂J⁽⁴⁾/∂**h**⁽¹⁾ = (∂**h**⁽²⁾/∂**h**⁽¹⁾) × (∂J⁽⁴⁾/∂**h**⁽²⁾)   *chain rule!*

## Slide 54 — Vanishing gradient intuition (3)

∂J⁽⁴⁾/∂**h**⁽¹⁾ = (∂**h**⁽²⁾/∂**h**⁽¹⁾) × (∂**h**⁽³⁾/∂**h**⁽²⁾) × (∂J⁽⁴⁾/∂**h**⁽³⁾)   *chain rule!*

## Slide 55 — Vanishing gradient intuition (4)

∂J⁽⁴⁾/∂**h**⁽¹⁾ = (∂**h**⁽²⁾/∂**h**⁽¹⁾) × (∂**h**⁽³⁾/∂**h**⁽²⁾) × (∂**h**⁽⁴⁾/∂**h**⁽³⁾) × (∂J⁽⁴⁾/∂**h**⁽⁴⁾)   *chain rule!*

## Slide 56 — Vanishing gradient intuition (the problem)

The full product is shown with each of the three Jacobian factors boxed, and an arrow
from below asking: *What happens if these are small?*

> **Vanishing gradient problem:** When these are small, the gradient signal gets
> smaller and smaller as it backpropagates further

## Slide 57 — Vanishing gradient proof sketch (linear case)

- Recall: **h**⁽ᵗ⁾ = σ( **W**_h **h**⁽ᵗ⁻¹⁾ + **W**_x **x**⁽ᵗ⁾ + **b**₁ )
- What if σ were the identity function, σ(x) = x?

  ∂**h**⁽ᵗ⁾/∂**h**⁽ᵗ⁻¹⁾ = diag( σ′( **W**_h **h**⁽ᵗ⁻¹⁾ + **W**_x **x**⁽ᵗ⁾ + **b**₁ ) ) **W**_h   (chain rule)
  = **I W**_h = **W**_h

- Consider the gradient of the loss J⁽ⁱ⁾(θ) on step *i*, with respect to the hidden
  state **h**⁽ʲ⁾ on some previous step *j*. Let ℓ = *i* − *j*

  ∂J⁽ⁱ⁾(θ)/∂**h**⁽ʲ⁾ = (∂J⁽ⁱ⁾(θ)/∂**h**⁽ⁱ⁾) ∏_{j<t≤i} (∂**h**⁽ᵗ⁾/∂**h**⁽ᵗ⁻¹⁾)   (chain rule)

  = (∂J⁽ⁱ⁾(θ)/∂**h**⁽ⁱ⁾) ∏_{j<t≤i} **W**_h = (∂J⁽ⁱ⁾(θ)/∂**h**⁽ⁱ⁾) **W**_h^ℓ   (value of ∂**h**⁽ᵗ⁾/∂**h**⁽ᵗ⁻¹⁾)

  **W**_h^ℓ is boxed, with the note: *If W_h is "small", then this term gets
  exponentially problematic as ℓ becomes large*

Source: "On the difficulty of training recurrent neural networks", Pascanu et al, 2013.
http://proceedings.mlr.press/v28/pascanu13.pdf (and supplemental materials, at
http://proceedings.mlr.press/v28/pascanu13-supp.pdf)

## Slide 58 — Vanishing gradient proof sketch (eigenvalues)

- What's wrong with **W**_h^ℓ?
- Consider if the eigenvalues of **W**_h are all less than 1:

  λ₁, λ₂, …, λ_n < 1   ← annotated *sufficient but not necessary*
  **q**₁, **q**₂, …, **q**_n  (eigenvectors)

- We can write (∂J⁽ⁱ⁾(θ)/∂**h**⁽ⁱ⁾) **W**_h^ℓ using the eigenvectors of **W**_h as a basis:

  (∂J⁽ⁱ⁾(θ)/∂**h**⁽ⁱ⁾) **W**_h^ℓ = Σ_{i=1..n} c_i λ_i^ℓ **q**_i ≈ **0** (for large ℓ)

  λ_i^ℓ is boxed: *Approaches 0 as ℓ grows, so gradient vanishes*

- What about nonlinear activations σ (i.e., what we use?)
  - Pretty much the same thing, except the proof requires λ_i < γ for some γ dependent
    on dimensionality and σ

Source: Pascanu et al, 2013 (as above).

## Slide 59 — Why is vanishing gradient a problem?

The four-step chain again, now with *two* backward flows drawn: an orange one from a
nearby loss J⁽²⁾(θ) back to **h**⁽¹⁾, and a blue one from the distant loss J⁽⁴⁾(θ) back
along the whole chain. The blue arrows shrink as they travel left.

> Gradient signal from far away is lost because it's much smaller than gradient signal
> from close-by.
>
> So, model weights are updated only with respect to **near effects**, not **long-term effects**.

## Slide 60 — Effect of vanishing gradient on RNN-LM

- **LM task:** *When she tried to print her tickets, she found that the printer was out
  of toner. She went to the stationery store to buy more toner. It was very overpriced.
  After installing the toner into the printer, she finally printed her ________*

- To learn from this training example, the RNN-LM needs to **model the dependency**
  between *"tickets"* on the 7th step and the target word *"tickets"* at the end.

- But if the gradient is small, the model **can't learn this dependency**
  - So, the model is **unable to predict similar long-distance dependencies** at test time

## Slide 61 — Why is exploding gradient a problem?

- If the gradient becomes too big, then the SGD update step becomes too big:

  θ^new = θ^old − α ∇_θ J(θ)

  with α labelled *learning rate* and ∇_θ J(θ) labelled *gradient*.

- This can cause **bad updates**: we take too large a step and reach a weird and bad
  parameter configuration (with large loss)
  - You think you've found a hill to climb, but suddenly you're in Iowa

- In the worst case, this will result in **Inf** or **NaN** in your network (then you
  have to restart training from an earlier checkpoint)

## Slide 62 — Gradient clipping: solution for exploding gradient

- **Gradient clipping**: if the norm of the gradient is greater than some threshold,
  scale it down before applying SGD update

```
Algorithm 1  Pseudo-code for norm clipping
   ĝ ← ∂E/∂θ
   if ‖ĝ‖ ≥ threshold then
       ĝ ← (threshold / ‖ĝ‖) ĝ
   end if
```

- **Intuition**: take a step in the same direction, but a smaller step

- In practice, **remembering to clip gradients is important**, but exploding gradients
  are an easy problem to solve

Source: Pascanu et al, 2013.

## Slide 63 — How to fix the vanishing gradient problem?

- The main problem is that *it's too difficult for the RNN to learn to preserve
  information over many timesteps*.

- In a vanilla RNN, the hidden state is constantly being **rewritten**

  **h**⁽ᵗ⁾ = σ( **W**_h **h**⁽ᵗ⁻¹⁾ + **W**_x **x**⁽ᵗ⁾ + **b** )

- First off next time: How about an RNN with separate **memory** which is added to?
  - LSTMs

- And then: Creating more direct and linear pass-through connections in model
  - Attention, residual connections, etc.

## Slide 64 — 5. Recap

- **Language Model**: A system that **predicts the next word**

- **Recurrent Neural Network**: A family of neural networks that:
  - Take **sequential input of any length**
  - Apply the **same weights on each step**
  - Can optionally produce output on each step

- Recurrent Neural Network ≠ Language Model

- We've shown that RNNs are a great way to build a LM (despite some problems). But:
  - RNNs are also useful for much more!
  - There are other models for building LMs (esp. Transformers!)

## Slide 65 — Why should we care about Language Modeling?

- Old answer:
  - Language Modeling is a **benchmark task** that helps us **measure our progress** on
    predicting language use
  - Language Modeling is a **subcomponent** of many NLP tasks, especially those
    involving **generating text** or **estimating the probability of text**:
    - Predictive typing, Speech recognition, Handwriting recognition, Spelling/grammar correction
    - Authorship identification, Machine translation, Summarization, Dialogue
    - etc.

- New answer:
  - Everything in NLP has now been rebuilt upon Language Modeling!
    - GPT-3 is an LM! GPT-4 is an LM! Claude Opus is an LM! Gemini Ultra is an LM!
    - We can now instruct LMs to do language understanding and reasoning tasks for us

## Slide 66 — Other RNN uses: RNNs can be used for sequence tagging

e.g., **part-of-speech tagging**, named entity recognition

An RNN runs left to right over *the startled cat knocked over the vase*, emitting a tag
above each word: DT, JJ, NN, VBN, IN, DT, NN.

## Slide 67 — RNNs can be used for sentence classification (the question)

e.g., **sentiment classification**

An RNN runs over *overall I enjoyed the movie a lot*; above it sits a **Sentence
encoding** vector feeding the label **positive**. The question posed: *How to compute
sentence encoding?*

## Slide 68 — RNNs can be used for sentence classification (basic way)

Same diagram. **Basic way**: Use final hidden state — a magenta arrow labelled *equals*
runs from the last hidden state up to the sentence encoding.

## Slide 69 — RNNs can be used for sentence classification (better way)

Same diagram. **Usually better**: Take element-wise max or mean of all hidden states —
arrows run from *every* hidden state up into the sentence encoding.

## Slide 70 — RNNs can be used as an encoder module

e.g., **question answering**, machine translation, *many other tasks!*

An RNN runs over the Question *what nationality was Beethoven ?*. Arrows from all its
hidden states feed upward through "lots of neural architecture" into the **Answer:
German**, which also draws on the **Context:** *Ludwig van Beethoven was a German
composer and pianist. A crucial figure …* through more neural architecture.

> Here the RNN acts as an **encoder** for the Question (the hidden states represent the
> Question). The encoder is part of a larger neural system.

## Slide 71 — RNN-LMs can be used to generate text

e.g., **speech recognition**, machine translation, summarization

An audio waveform on the left feeds, via a dotted arrow labelled *conditioning*, into an
RNN-LM on the right that generates *what's the weather*, starting from `<START>` and
feeding each output back as the next input.

> This is an example of a *conditional language model*.
> We'll see Machine Translation in much more detail starting next lecture.

## Slide 72 — Terminology and a look forward

The RNN described in this lecture = **simple**/vanilla/**Elman** RNN

**Next lecture:** You will learn about other RNN flavors like **LSTM** and **GRU** and
multi-layer RNNs

(illustrated with ice-cream cones — one scoop for the simple RNN, different flavors for
LSTM and GRU, a triple-stack for multi-layer RNNs)

**By the end of the course:** You will understand phrases like *"stacked bidirectional
LSTMs with residual connections and self-attention"* (illustrated with an
extravagantly-loaded ice cream sundae).
