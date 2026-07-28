# Lecture 17 — ConvNets and Tree Recursive Neural Networks

Christopher Manning's tour of two architectures that are *not* the state of the art, and he says
so in the first minute: "these two techniques are ones that people aren't using very much these
days, and that's partly why they get sort of stuck towards the end of the course" (≈0:05). The
argument for spending a lecture on them is that "in any scientific field there are different
ideas and techniques that bounce around, and it's good to know a few of the different ideas that
are out there, because often what happens is people find new ways to reinvent things" (≈0:50).

The two halves are almost opposites. **Convolutional neural networks** apply the same filter to
every $n$-gram regardless of structure — Manning is explicit that "there's nothing linguistically
or cognitively especially plausible here" (≈5:28). **Tree recursive neural networks** are the
maximally linguistic alternative: build the meaning of a sentence by composing its syntactic
constituents, one parent from two children, all the way up a parse tree. They were Stanford's own
line of work, and the closing minutes are Manning's assessment of why they lost and what they
still get right that Transformers do not (≈1:10:00).

**Slide-by-slide text of this deck: [60 slides](../raw/slides/17-convnets-and-treernns.md).**

Slides PDF: [ConvNets and TreeRNNs](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture16-CNN-TreeRNN.pdf) ·
[Full transcript](../raw/transcripts/17-convnets-and-treernns.md)

> **A note on numbering.** This lecture sits at **position 17** in the playlist this knowledge
> base follows; the video title and the deck itself both call it "Lecture 16". Repo files use the
> position. Printed slide numbers stop at 48; slides 49–60 are labelled by continuing the count.

---

# Part 1 — Convolutional networks for language

## Why a convolution at all

The framing is CNNs *versus RNNs*, not versus Transformers, and Manning explains why: "that's
partly because that's how the ideas of convolutional neural networks really were explored — it
was in the days when most people were using recurrent neural networks for NLP" (≈3:55). In the
five years Transformers have dominated, "there hasn't been much use of convolutional neural
networks for NLP" (≈4:43).

The limitation being attacked is that a [recurrent network](recurrent-neural-networks.md)
computes forward through the string, so the representation at any position "included everything
that came before you." For *Monáe walked into the ceremony*, you never get a representation of
*the ceremony* on its own — only of the whole prefix ending in it (≈4:43, slide 6).

A convolution takes the opposite approach: compute a representation for **every $n$-gram**,
whether or not it is a linguistic unit. For *tentative deal reached to keep government open*, the
trigrams are *tentative deal reached*, *deal reached to*, *reached to keep*, *to keep
government*, *keep government open* — five of them, each getting its own neural representation
(≈5:28, slide 7). Those get grouped later; see [pooling](#pooling-summarizing-a-feature-map).

In vision, where convolutions were invented, the payoff is translation invariance — "you could
recognize your kangaroo no matter where in the frame it was" (≈6:15, slide 4). The mask slides
over the image, and at each position you take a dot product of the mask weights with the patch
underneath (≈7:03). Text is the same operation in one dimension instead of two.

## The 1D convolution, concretely

Slide 9 works the arithmetic through. Seven words, each with a 4-dimensional word vector, and a
size-3 filter — a $3 \times 4$ matrix. Slide the filter down one position at a time, take the dot
product of filter and window, and you get one number per window: $-1.0$, $-0.5$, $-3.6$, $-0.2$,
$0.3$. Add a bias (here $+1$) and push the result through a nonlinearity such as a sigmoid
(≈8:36). That is one filter's **feature map**.

In Yoon Kim's notation (slide 20), with word vectors $\mathbf{x}_i \in \mathbb{R}^k$ and a
sentence formed by concatenation $\mathbf{x}_{1:n} = \mathbf{x}_1 \oplus \mathbf{x}_2 \oplus
\cdots \oplus \mathbf{x}_n$, a filter is a long vector $\mathbf{w} \in \mathbb{R}^{hk}$ covering a
window of $h$ words. Each output element is

$$c_i = f\big(\mathbf{w}^{T} \mathbf{x}_{i:i+h-1} + b\big)$$

and the feature map for the whole sentence is
$\mathbf{c} = [c_1, c_2, \ldots, c_{n-h+1}] \in \mathbb{R}^{n-h+1}$. Here $h$ is the filter (or
kernel) size, $b$ the bias, and $f$ the nonlinearity.

**Padding.** A size-3 filter over seven words produces only five outputs — "we ended up with
something smaller than our input sentence" (≈9:22). Padding one zero vector at each end restores
seven outputs; padding two at each end gives a *wide convolution* producing nine (≈10:10, slides
10–11).

**Multiple filters.** One filter is "pretty limiting," so in practice you define many, each
producing its own feature map. The output at each position becomes a vector whose length is the
number of filters, which may be shorter, equal to, or longer than the word dimensionality
(≈10:10, slide 11).

**In PyTorch** this is `Conv1d`, where `out_channels` is the number of filters and `kernel_size`
is $h$ — three, in the running example (≈13:16–14:03, slide 14).

## Pooling: summarizing a feature map

**Max pooling** — $\hat{c} = \max\{\mathbf{c}\}$ — is the standard choice, and Manning gives the
intuition that makes it stick (≈10:56–12:29). Think of a filter as a *feature detector*: maybe
one matches first-person language ("I", "my", "we", "our"), another matches speech and thinking
verbs ("think", "say", "said", "told"). What you usually want to know is whether the feature
fires *anywhere* in the text, "regardless of whether it's in the first, second, third, or fourth
position." Max pooling answers exactly that question. It also makes the network robust to
sentence length, since $\max\{\mathbf{c}\}$ is a single number however long $\mathbf{c}$ is
(slide 21).

**Average pooling** suits a different model of what a filter measures — a gradual quality such as
casualness or learnedness, where the average across the text is the meaningful summary. You can
compute both and feed both forward, but if you use only one, "max pooling is the most effective —
that kind of 'does the feature fire' metaphor tends, in general, to be the best way of thinking
about things" (≈13:16).

Three variations, all "less useful and less used in language cases" (≈14:03):

- **Stride.** Adjacent trigrams overlap heavily and so carry near-identical content, "and that
  would be even more so if we weren't using trigrams, if we were using something like
  five-grams." A stride of 2 moves the filter two positions at a time so the windows overlap less
  (≈14:50, slide 15).
- **Local max pooling.** Instead of one max over the whole sentence, pool within windows: if
  first-person language appears at four different points in a long sentence, "maybe you should
  get four points for that, rather than just sort of the one point" (≈15:35, slide 16).
- **$k$-max pooling.** Keep the $k$ largest activations per filter rather than only the largest —
  another way to detect that something matched in more than one place (≈16:21, slide 17). These
  need not be adjacent.
- **Dilation.** Build a trigram from positions 1, 3, 5 rather than 1, 2, 3, so a filter of the
  same width sees a wider span. "That's more commonly used in places like speech than in natural
  language" (≈17:08–17:55, slide 18).

## Yoon Kim (2014): the famous one

[Kim's *Convolutional Neural Networks for Sentence Classification*](https://arxiv.org/pdf/1408.5882.pdf)
(EMNLP 2014) is "the single most famous piece of work that made use of convolutional neural
networks for natural language processing" (≈17:55, slide 19). "In retrospect, it's sort of
actually pretty simple, but he got in early with the idea" (≈18:41).

The architecture, in one sentence: filters of several widths over the word vectors, max-pool each
filter to a single number, concatenate, softmax. Concretely (slides 20–21, ≈19:27–20:13) —
filters of size $h = 3, 4, 5$ with 100 feature maps each; max-over-time pooling reduces each
feature map to one number $\hat{c}_i$; the pooled values are concatenated into
$\mathbf{z} = [\hat{c}_1, \ldots, \hat{c}_m]$ for $m$ filters; and the classifier is a single
softmax layer

$$y = \mathrm{softmax}\big(W^{(S)} \mathbf{z} + b\big)$$

The task is [sentiment analysis](sentiment-analysis.md), with subjectivity and question-type
classification as secondary tasks. Slide 25's results table puts CNN variants against a long list
of contemporary models across seven datasets, and Kim's claim is that this simple architecture
matches or beats all of them — CNN-multichannel reaches 88.1 on SST-2 and 85.0 on CR, for
instance.

Manning flags a real methodological problem with that comparison (≈28:04, slide 26). Kim used
[dropout](regularization-and-dropout.md), which he reports gives a 2–4% accuracy gain. Dropout
appeared in 2012, so several of the compared systems predate it and "would probably gain equally
from it." The honest experiment would have been to re-run the baselines with dropout. The result
still stands as evidence that "you could get strong results using convolutional neural networks
with just a very simple architecture" (≈28:50).

## The word-vector fine-tuning pitfall

This is the most portable idea in the first half, and it applies well beyond CNNs (≈21:01,
slide 22).

You start from pre-trained [word vectors](word2vec.md) — GloVe or word2vec — and train a
sentiment classifier on a small supervised dataset, backpropagating into the word vectors as well
as the classifier. That seems obviously right: generic word vectors "aren't especially tuned to
predicting sentiment correctly."

The failure is that **only words present in the training data move**. Suppose *tedious*, *dull*
and *plodding* start out close together, all negative. *Tedious* and *dull* appear in training and
migrate to the negative side along with the shifting decision boundary; *plodding* appears only at
test time, "so it's just sitting exactly where it was at the start of the process, and now it's
being treated as a positive word, which is completely wrong" (≈22:34–23:22). The net effect is
that fine-tuning word vectors on small datasets gave "kind of ambivalent results" — sometimes
better, sometimes worse (≈24:11).

Kim's fix is **channel doubling** (slide 23): keep two copies of every word-vector channel,
backpropagate into one and hold the other static, and add both to $c_i$ before pooling. You get
task specialization from the fine-tuned copy and intact semantic geometry from the frozen one —
"the best of both worlds" (≈24:11). This is the ancestor of the intuition behind
[parameter-efficient finetuning](parameter-efficient-finetuning.md): leave the pre-trained
representation in place and add capacity beside it.

## Where CNNs sit in the toolkit

Slide 27 is the recap, and it is a compact statement of the course's whole architecture story
(≈28:50–29:36):

| Model | What it is good for |
| --- | --- |
| Bag of vectors | Surprisingly strong baseline for simple classification, especially with a few ReLU layers on top |
| Window model | Single-word classification not needing wide context — POS tagging, NER |
| CNN | Classification; needs zero padding for short phrases; hard to interpret; **easy to parallelize on GPUs** |
| [RNN](recurrent-neural-networks.md) | Cognitively plausible; slower than a CNN; good for sequence tagging and language modelling; better with [attention](attention.md) |
| [Transformer](transformer.md) | "Still the best thing since sliced bread for all NLP problems" |

The parallelism point is the substantive one: a CNN's filters are independent of each other and
of position, so the whole layer computes at once, while an RNN "reads through sentences from left
to right" and cannot. Manning notes the traffic now runs the other way — Vision Transformers are
taking over in vision, "though there's still, I think, more debate in the vision world as between
CNNs and Transformers, with some people arguing that both of them have complementary advantages"
(≈29:36).

## Two asides worth keeping

**BatchNorm versus LayerNorm** (slide 28, ≈30:23–31:10). Both rescale activations to zero mean
and unit variance — "like the familiar Z-transform of statistics" — and they differ only in the
axis they compute statistics over. LayerNorm, standard in [Transformers](transformer.md),
computes across the feature dimensions of each instance independently. BatchNorm, standard in
CNNs, normalizes across all items in a batch for each feature independently. BatchNorm also makes
models much less sensitive to parameter initialization and simplifies learning-rate tuning.
PyTorch: `nn.BatchNorm1d`.

**Size-1 convolutions** (slide 29, ≈31:55). A kernel of size 1 sounds pointless — "you just got
one thing and it's staying just one thing" — but it is a fully connected layer *across channels*
at a single position. It can map many channels to few, and it adds a layer with very few
parameters compared with a fully connected layer over the whole sentence. Manning makes the
connection explicit: "that's sort of actually what we also have with the fully connected layers
in Transformers" — the per-position feed-forward sublayer is a size-1 convolution (≈32:41). The
original reference is Lin, Chen and Yan's *Network in Network* (2013).

## VD-CNN: what happens if you make it deep

Conneau, Schwenk, LeCun and Barrault, EACL 2017 (slides 30–34). The motivation is a
straightforward comparison of fields (≈33:27–35:03). Vision was using ResNets 30, 50, 100 layers
deep, while NLP sequence models were "always a single-digit number of layers" — two, sometimes
three or four, eight if you had a lot of data. Vision also worked from the raw pixel signal while
NLP started from words, so "things were much more grouped before they began."

So: start from characters and build a vision-style network. The architecture (slide 31) embeds
characters in 16 dimensions over a fixed length of $s = 1024$ characters, then alternates
convolutional blocks with local pooling, where each pooling stage "halves temporal resolution and
doubles number of features": 64 channels at length $s$, 128 at $s/2$, 256 at $s/4$, 512 at $s/8$,
followed by $k$-max pooling with $k = 8$ and three fully connected layers. Each convolutional
block is two [conv → BatchNorm → ReLU] sublayers with a size-3 kernel (slide 32), and each stage
carries an optional ResNet-style shortcut.

Manning walks the receptive field up the stack (≈36:36–38:57): after enough pooling, each unit
represents an eight-character span and the trigram filters above it cover 24 characters — "sort
of seeing something like six-word sequences." The $k$-max pooling at the top "makes sense for
something like a text classifier, because you want to count up the amount of evidence" — if the
category is copper mining, you want to know whether several places in the text discuss it.

Results (slide 34): depths of 9, 17 and 29 layers on eight large classification datasets. "In all
cases, they got the best results with their deepest network, which was a 29-layer model" (≈40:31)
— for example 26.57 error on Yahoo! Answers and 37.00 on Amazon Full, both at depth 29 with max
pooling, against previous bests of 24.2 and 36.4 from Yang et al. Which pooling won varied by
dataset. Against the previous state of the art the record is mixed — better on DBpedia and both
Yelp sets, split on Amazon, slightly behind on some news classification — but the achievement is
that a purely character-level convolutional network gets there "with none of the sort of having
learned word vectors in advance" (≈42:01).

---

# Part 2 — Tree recursive neural networks

"Tree recursive neural networks is a framework that me and students developed at Stanford … for
about the first five years, what me and students worked on was doing these tree recursive neural
networks, and so they were sort of the Stanford brand. Ultimately, they didn't prove as
successful as other things that came along, but I think they're linguistically interesting"
(≈42:48).

## Recursion as the defining property of language

Slide 35 reproduces Hauser, Chomsky and Fitch's *The Faculty of Language: What Is It, Who Has It,
and How Did It Evolve?* (*Science*, 2002), whose claim is that the faculty of language in the
narrow sense consists of **recursion** alone, and that this is the only uniquely human component.

The linguistic point is hierarchical nesting — the same structure repeating inside itself
(≈44:21, slide 36):

> [The person standing next to [the man from [the company that purchased [the firm that you used
> to work at]]]]

A noun phrase containing a noun phrase containing a noun phrase. Context-free grammar permits
this nesting to be infinite, "the same kind of nesting that you get in programming languages,
where you can sort of use if statements and nest them as deeply as you want to." Manning is
careful about the empirical status: people can neither understand nor produce infinite recursion,
and in practice "no one's going to go more than eight deep," but the *structure* of the language
appears to have no depth bound (≈45:07–45:52). Recursive structure is, at minimum, "a very
powerful prior for language structure."

Real sentences do nest deeply. Slide 37's Penn Treebank tree for *Analysts said Mr. Stronach
wants to resume a more influential role in running the company* contains four verb phrases inside
one another: *running the company* ⊂ *resume a more influential role in running the company* ⊂
*wants to resume …* ⊂ *said Mr. Stronach wants to resume …* (≈45:52–46:39).

## Compositionality as the design principle

The reason to care, for someone building representations, is
[compositionality](compositionality.md) — "the meaning of a phrase or sentence is determined by
the meanings of its words and the rules that combine them" (slide 38, ≈47:25). The proposal is to
take the phrase-structure tree and combine word vectors along it, producing a vector for each
phrase that lands in the same space as the words: *the country of my birth* should end up near
words denoting locations, and near its paraphrase *the place where I was born* (≈48:14, slide 38).

Manning defuses the terminology in passing: "the difference between 'recursive' and 'recurrent'
is sort of a fake difference, right, they both come from the same 'recur' word" — the real
distinction is that recursion runs *up a tree* rather than *along a sequence* (≈49:03, slide 41).
A recurrent net cannot represent a phrase without its prefix context, and tends to over-weight
the last words of the span.

## The simple TreeRNN

Two jobs, from one small network (slides 42–43, ≈49:49). Given two candidate children $c_1$ and
$c_2$, produce the parent representation and a score for whether merging them is plausible:

$$p = \tanh\!\left(W \begin{bmatrix} c_1 \\ c_2 \end{bmatrix} + b\right)$$

$$\text{score} = U^{T} p$$

where $W$ is a shared composition matrix, $b$ a bias, and $U$ a learned scoring vector. The
**same $W$ is used at every node of the tree**, exactly as a recurrent network reuses its
parameters at every timestep (≈50:35).

That gives a **greedy parser** (slides 44–47, ≈51:22). Score every adjacent pair of words; commit
to the highest-scoring merge; re-score the pairs that the merge created; repeat. On *The cat sat
on the mat*: (*The*, *cat*) scores 3.1 and (*the*, *mat*) scores 2.3, well above the alternatives
around 0.1–0.4, so *The cat* merges first; *the mat* next; then *on* with *the mat* at 3.6; then
*sat*; and finally the two halves join at the root. The score of a whole tree is the sum of its
node decisions,

$$s(x,y) = \sum_{n \in \mathrm{nodes}(y)} s_n$$

for sentence $x$ and parse $y$. The 2011 Socher, Manning and Ng paper won a best-paper award, and
the phrase representations were good enough to reuse for sentence classification (≈52:09–52:56).

**Why one matrix is not enough** (slide 48, ≈52:56–53:42). With a single $W$ at every node "you
sort of can't have different forms of interaction between the different words — you're just
uniformly computing things." But an adjective modifying a noun and a verb taking an object are
different relations: in *the red ball*, *red* contributes attributes of the noun, whereas in *kick
the ball* the object plays a quite different role (≈55:15). There is no real interaction between
the input words, and the composition function ignores syntactic category entirely.

## The Recursive Neural Tensor Network

The fix (Socher, Perelygin, Wu, Chuang, Manning, Ng and Potts, 2013; slides 49, 53–56) is to let
the two children interact **multiplicatively** as well as additively. Instead of concatenating and
applying one linear map, learn a stack of in-between matrices — a three-dimensional tensor — and
compute a bilinear form for each slice (≈56:04):

$$p = f\!\left(\begin{bmatrix} b \\ c \end{bmatrix}^{T} V^{[1:2]} \begin{bmatrix} b \\ c
\end{bmatrix} + W \begin{bmatrix} b \\ c \end{bmatrix}\right)$$

Here $b$ and $c$ are the child vectors, $V^{[1:2]}$ is the tensor (two slices, in this
illustration — one output dimension per slice), $W$ the standard linear layer, and $f$ a
nonlinearity. The first term is what is new: a vector times a matrix times a vector, so each
child's value can scale the other's contribution. Manning notes the line of work continued into
[TreeLSTMs](lstm.md) (Tai, Socher, Manning 2015), which "work even better" but are not covered
this year (≈54:29, slide 49).

## The Stanford Sentiment Treebank, and what it was for

The application is [sentiment analysis](sentiment-analysis.md), and the point of the dataset is
compositional structure (≈56:54–59:14, slides 50–51).

Keyword matching gets you a long way — "if you see 'great,' 'marvelous,' 'wonderful,' positive
sentiment" — and on long documents accuracy is around 90%. The counterexample Manning uses is a
Rotten Tomatoes-style snippet:

> With this cast and this subject matter, the movie should have been funnier and more
> entertaining.

Dictionary matching sees *funnier* and *entertaining*, both positive, and calls it a positive
review. It is a negative review, "because it's buried under 'should have been'" — the positive
qualities are asserted to be *absent* (≈58:27).

So the **Stanford Sentiment Treebank** annotates sentiment at every node of the parse tree, not
just the sentence: 215,154 labelled phrases across 11,855 sentences (slide 51). *With this cast*
is neutral; *entertaining* is positive; *funnier and more entertaining* is very positive; embed it
under *should have been* and it turns negative (≈59:59).

The first result is that the dataset alone helps every model (slide 52, ≈1:01:31). A bigram naive
Bayes classifier — a strong, standard sentiment baseline — scores 79% trained on sentence labels
and 83% trained on every node of the treebank, a four-point lift purely from denser supervision.
The early tree RNNs beat *unigram* naive Bayes but not the bigram version, because "a lot of the
extra information that you want to capture for sentiment analysis, you can get from bigrams" —
bigrams already cover *not good* and *somewhat interesting* (≈1:02:17).

The RNTN, with its multiplicative interactions, reaches **85.4%** on the same comparison (slide
57) and 72% on the *X but Y* construction against 65% for MV-RNN, 58% for bigram naive Bayes and
54% for the simple RNN (slide 58).

## Negation is the result worth remembering

Manning is clear that the aggregate accuracy is not the interesting part (≈1:03:03). What matters
is that the model makes judgments about *parts* of sentences and how they combine.

Slide 58's example: *There are slow and repetitive parts, but it has just enough spice to keep it
interesting.* The tree builds negativity on the left (*slow*, *repetitive*, *parts*), positivity
on the right (*spice*, *interesting*), and correctly resolves the root as **positive** — the *X
but Y* pattern where the second clause dominates (≈1:03:49–1:04:34).

Negation is the sharper test (slide 59, ≈1:05:20–1:09:14), and it addresses a criticism neural
models still attract: "you often find the result that neural network models just don't pay
attention to negation" — *a lot of students are studying for their final exams* versus *a lot of
students aren't studying for their final exams* produce nearly the same representation.

Two cases, and they are not equally hard:

- **Negating a positive** — *it's definitely not good*. Every model handles this, "because 'not'
  is a negative word, and so, therefore, it weakens the positivity of the positive word." All four
  models show a decrease in positive activation, from $-0.16$ (bigram NB) to $-0.54$ (RNTN).
- **Negating a negative** — *it's definitely not dull*. This should make the sentence *more*
  positive, and it requires actual composition rather than word-level polarity. Bigram naive
  Bayes and the earlier recursive models produce essentially zero change ($-0.01$, $-0.01$,
  $+0.01$); only the RNTN moves correctly, $+0.25$.

The aside on *not* is a nice piece of corpus statistics: "'not' occurs much more often in
negative-sentiment sentences than it does in positive-sentiment sentences. So, if you want to be
a more positive person, use negation less" (≈1:07:39). And *incredibly dull* is the mirror case —
*incredible* is positive in isolation, but the composed phrase stays firmly negative, which the
RNTN gets right (≈1:06:06).

Manning's assessment: "to some extent, this result, I think, still isn't captured as well by any
of the current Transformer models, even though they have many other advantages and are much
better than a tree recursive neural network" (≈1:09:14).

## Why they lost, and what they still had

The closing diagnosis (≈1:10:00–1:11:35) is precise and worth quoting nearly in full. TreeRNNs
became uncompetitive because

> these models had a strictly context-free backbone, and the only information flow was
> tree-structured … whereas in the Transformer you've got this attention function, where, at every
> position, you're looking at every other position, and so you can have much more general
> information flow.

[Self-attention](self-attention.md) lets any position read any other; a tree lets a node read only
its children. That is a strictly poorer information channel, and it is why Transformers win.

The counterweight: "to the extent that you actually want to carefully model the sort of semantics
of human language — sort of what modifies what, and how does negation or quantifiers in a
sentence behave — in some sense these models were more right." The lecture ends on the open
question rather than a conclusion: whether one can "combine together some of the benefits of both
of these ways of thinking, and have something that's a bit more tree-structured, while still more
flexible, like a Transformer."

That question is picked up from the linguistic side in
[lecture 18](18-nlp-linguistics-philosophy.md), where Manning asks whether neural models compose
meaning at all.

## Related pages

- [Convolutional neural networks for NLP](convolutional-neural-networks.md) — the mechanics in
  one place: filters, padding, stride, pooling variants, dilation.
- [Tree recursive neural networks](tree-recursive-neural-networks.md) — TreeRNN, RNTN, and the
  parsing algorithm as a standalone topic.
- [Compositionality](compositionality.md) — the principle behind both this lecture's second half
  and lecture 18's semantics discussion.
- [Sentiment analysis](sentiment-analysis.md) — the task both halves of the lecture use, and the
  Stanford Sentiment Treebank.
- [Recurrent neural networks](recurrent-neural-networks.md) and
  [Transformers](transformer.md) — the architectures this lecture is positioned against.
- [Regularization and dropout](regularization-and-dropout.md) — the confound in Kim's comparison.
