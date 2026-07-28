# Sentiment analysis

Deciding whether a piece of text is positive, negative or neutral. It is the course's default
example task for sentence classification, and it recurs from
[RNN sentence encoders](recurrent-neural-networks.md) in lecture 5 through
[BERT fine-tuning](bert.md) in lecture 9 to both halves of
[lecture 17](17-convnets-and-treernns.md).

Its usefulness as a teaching example comes from a specific property: the easy 90% is trivially
easy, and the remaining 10% requires exactly the compositional structure that distinguishes
architectures from one another.

## Why keyword matching almost works

"A lot of the time, doing sentiment analysis is pretty easy — in the 2010s, and probably even
today, quite a few people's sentiment analysis systems are essentially just keyword matching. If
you see 'great,' 'marvelous,' 'wonderful,' positive sentiment; if you see something like 'poor,'
'bad,' negative sentiment" (lecture 17, ≈56:54). On long documents, dictionary matching reaches
roughly 90% accuracy, because a long review contains many polarity-bearing words and the errors
average out.

## Why it then stops working

Manning's counterexample is a Rotten Tomatoes-style snippet (lecture 17, slide 50, ≈57:40):

> With this cast and this subject matter, the movie should have been funnier and more
> entertaining.

A dictionary sees *funnier* and *entertaining* — two positive words, no negative ones — and calls
it positive. It is a negative review. The positive qualities are asserted to be **absent**,
because the phrase is embedded under *should have been*. No amount of word-level polarity
recovers this; you have to know what scopes over what.

Negation is the same problem in its sharpest form. *It's definitely not dull* is positive though
every content word in it is negative. *Incredibly dull* is negative though *incredible* is
positive. And models are known to ignore negation entirely: "you can have some sentence, and you
can compare … a lot of students are studying for their final exams, versus a lot of students
aren't studying for their final exams — and the negation just gets lost" (lecture 17, ≈1:05:20).

## The Stanford Sentiment Treebank

Socher et al.'s response (2013) was to change the supervision rather than only the model.
Ordinary sentiment datasets label whole texts. The **Stanford Sentiment Treebank** parses ~11,855
sentences and labels the sentiment of **every phrase in every tree** — 215,154 labelled phrases
in total (lecture 17, slide 51).

On the running example, the labels change as the phrase grows:

| Phrase | Sentiment |
| --- | --- |
| *with this cast* | neutral |
| *entertaining* | positive |
| *funnier and more entertaining* | very positive |
| *should have been funnier and more entertaining* | negative |
| *the movie should have been funnier and more entertaining* | negative |

Two consequences. First, you can now *train and test composition* directly, since the model is
supervised at every node rather than only at the root. Second, the denser supervision helps
everything: a bigram naive Bayes baseline goes from 79% trained on sentence labels to 83% trained
on treebank labels — a four-point lift with no change of model (lecture 17, slide 52, ≈1:01:31).

The dataset uses five-way labels — very negative, somewhat negative, neutral, somewhat positive,
very positive — though the commonly reported SST-2 figure is the binary collapse of that
(lecture 17, ≈1:06:51).

## How the course's models do on it

- **Bigram naive Bayes** is the baseline to beat, and it is stronger than students expect,
  because bigrams already capture *not good* and *somewhat interesting*.
- **[CNN sentence classifiers](convolutional-neural-networks.md)** (Kim 2014) reach 88.1 on SST-2
  and 48.0 on the fine-grained SST-1 — competitive with everything else of that era, from a
  single convolutional layer plus max pooling.
- **[TreeRNNs and the RNTN](tree-recursive-neural-networks.md)** reach 85.4 on the phrase-labelled
  comparison, and are the only models in the lecture that handle negating a *negative* sentence
  correctly (+0.25 change in activation, against roughly zero for everything else).
- **[BERT](bert.md)** and later pre-trained models fine-tune on SST-2 as one of the GLUE tasks,
  which is where the dataset mostly appears today.
- **Pre-training alone teaches sentiment implicitly**: [lecture 9](09-pretraining.md) uses the
  cloze example *…the value I got from the two hours watching it was the sum total of the popcorn
  and the drink. The movie was ___* to show that predicting the next word forces a language model
  to solve sentiment as a subproblem.

That progression is the point. The task barely changed for a decade; what changed is that the
compositional structure the treebank made explicit is now learned implicitly from
[pre-training](pretraining-and-finetuning.md) — except, per lecture 17, for negating negatives.

## See also

- [Lecture 17 — ConvNets and TreeRNNs](17-convnets-and-treernns.md) — Kim's CNN, the RNTN and the
  treebank.
- [Tree recursive neural networks](tree-recursive-neural-networks.md) — the negation results.
- [Compositionality](compositionality.md) — why *should have been funnier* is hard.
- [BERT](bert.md) — SST-2 as a GLUE fine-tuning task.
- [Evaluating LLMs](evaluating-llms.md) — how classification benchmarks like this fit the modern
  evaluation picture.
