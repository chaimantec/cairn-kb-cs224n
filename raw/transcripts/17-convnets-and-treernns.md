---
title: ConvNets and TreeRNNs
lecture: 17
video: https://www.youtube.com/watch?v=S8d-7v3f5MQ
source: edited from auto-generated YouTube transcript
verbatim: false
original: original/17-convnets-and-treernns.md
slides: ../slides/17-convnets-and-treernns.md
---

# ConvNets and TreeRNNs — transcript

Copy-edited for punctuation, sentence boundaries, and mis-heard terms; cross-checked against
`raw/slides/17-convnets-and-treernns.md`. The verbatim auto-generated captions are kept at
`raw/transcripts/original/17-convnets-and-treernns.md`. Lecturer is Christopher Manning. Student
questions and comments from the floor are set in *italics* — the only clear instance in this
lecture is a one-word "yeah" at 7:03, confirming Manning's aside "does that make sense?" (the
captions had run it straight into his next sentence). Timestamps mark the start of each
paragraph; all 93 are preserved in order.

**This is an edited transcript.** The captions had no punctuation and mangled a lot of the
vocabulary this lecture is built on. The most consequential single fix: at 1:10:48 the captions
read "this tension function," which is *the attention function* — the Transformer mechanism
Manning is contrasting with TreeRNNs' fixed tree backbone. Other restorations, checked against
the slides:

- **People and papers:** *Yoon Kim* as "Yun Kim" (repeated); a reference to *Kim's* use of
  Socher's Stanford Sentiment Treebank data as "the Yim work" (56:54); *Conneau et al.* as "Cano
  Al"; *Hauser, Chomsky[, and Fitch]* as "gome Chomsky"; *Mr. Stronach* (the Penn Treebank example
  sentence) as "Mr stronck."
- **Tools and products:** *Google Colab* / *Colab Pro* as "Google collab" / "collab Pro";
  *QueueStatus* as "Q status"; *Kaggle Notebooks* as "kaggle notebooks"; *Modal* as "modal";
  *Together AI* as "together AP"; *PyTorch* as "pytorch"; *Conv1D* as "con 1D"/"com 1D".
- **Model/architecture terms:** *neural network(s)* consistently arrived as "new network(s)" or
  "newal network(s)" and is silently normalized throughout without separate note; *max pooling* /
  *average pooling* / *k-max pooling* as "Max pull(ing)" / "average pull" / "k Max"; *BatchNorm*
  and *LayerNorm* as "batch Norm" / "layer Norm"; *ResNet*, *VGGNet*, *LSTM*, *TreeLSTM(s)* as
  "resnet," "vggnet," "lstm," "tree lsdm"; *VD-CNN* as "VD CNN"; *DBpedia ontology* as "dbpedia
  onology"; *neural tensor layer/network* as "neural tensil/tenser layer/network."
- **N-gram vocabulary:** "engram(s)" restored to *n-gram(s)*, "byr(s)"/"Tri"/"forr(s)" to
  *bigram(s)*/*trigram(s)*/*four-gram(s)*, "NR" to *n* (filter size), "3 G"/"two G"/"five
  G"/"eight G" to *trigram*/*bigram*/*five-gram*/*eight-gram*, and "INR" (31:55, listing filter
  sizes right before the "size-one convolution" discussion) to *unigram*.
- **Sentiment-analysis examples:** the Yoon Kim fine-tuning pitfall's third word, "plotting," is
  *plodding* (all three of "tedious," "dull," "plodding" are named on slide 22); the Rotten
  Tomatoes-style RNTN example's "just enough space to keep it interesting" is *just enough spice*
  (matches slide 58 exactly); "naive Bas"/"Nave Bas" is *naive Bayes* throughout; "sentiment
  classifiers" at 1:01:31 is restored to *sentence labels*, matching the x-axis label on slide 52's
  bar chart (the 79%→83% figures Manning cites there match that chart's Bi-NB bars exactly).
- At 5:28, the RNN example sentence's subject, "man," is restored to *Monáe* (Janelle Monáe),
  matching the "Monáe walked into the ceremony" example on slide 6.
- At 5:28, the run of overlapping trigrams over "tentative deal reached to keep government open"
  was garbled and partly dropped in the captions; the full five-trigram list is restored against
  slide 7's worked example.

**One passage is left flagged rather than guessed at:** at 3:10, describing Colab Pro's monthly
cost, the captions read "not so many copies worth of money" — no confident reading of what this
phrase was meant to convey. It is left in place as `[Ed: unclear — captions read "not so many
copies worth of money"]`.

**No content was added, removed, or reordered.**

---

**[0:05]** Hi, okay, let me get started for today. I guess I'm now down to the more select week
eight audience of people who actually want to learn, so my welcome and my pleasure for the people
who show up today. Thank you, thank you. Okay, so what I want to do today is principally sort of
talk about a couple of other neural network techniques which can be used for language. I mean, in
some sense, these two techniques are ones that people aren't using very much these days, and
that's partly why they get sort

**[0:50]** of stuck towards the end of the course, because we try to teach people early on in the
course the most essential things that you should definitely know about. But the fact of the
matter is, in any scientific field there are different ideas and techniques that bounce around,
and it's good to know a few of the different ideas that are out there, because often what happens
is people find new ways to reinvent things and put things together and see different insights
from them. So today I'm going to tell you a little bit about using convolutional neural networks
for language, and then a bit about tree recursive neural networks, but before that, just

**[1:36]** course organization. This is a bit after it happened, but I guess I've never been back
to say it, so thanks to everyone who filled in the mid-quarter surveys. Some people said very nice
things about the lecture — fantastic lectures and really interesting content. Some people wished
that we were teaching more about state space models; I guess we haven't added that lecture in
yet. A couple of people thought it'd be good to have an exam in this class — clearly they weren't
people who have friends in CS231N, from what I've heard. But, in general, people are pretty happy
how Ed has been going — a bit less happy on

**[2:23]** how office hours have been going. I mean, honestly, it's a hard problem, I feel, to do
office hours. Some people are saying, oh, you should just use QueueStatus — I sort of remember,
badly, a year where we did everything but QueueStatus, and near the assignment due date the queue
would stretch six hours long, and that didn't seem such a good solution either. But we'll work
along with it. Finally, on cloud compute — I know this is something that people variously do have
issues with — there are quite a few people that are still trying to do things with Google Colab,
which I realize is sort of a very convenient, nice interface, but you sort of do suffer from
access to

**[3:10]** GPUs. On Google Colab, the best way to get better access to GPUs is to pay 10 bucks for
a month of Colab Pro, which perhaps means that you end up paying for two months — for, if it's May
and June, we can't reimburse you for that, but [Ed: unclear — captions read "not so many copies
worth of money"], and it does just give you better access to GPUs. I encourage you to use the GCP
credits and Together AI API access that we've given to you. You're also welcome to try other
things — Kaggle Notebooks can actually give you better GPU access, but not all the nice features
of Colab. Some

**[3:55]** groups have started using Modal, which can also be a good way to get GPU access. Okay,
that was the intro to that. So now I wanted to just sort of talk about convolutional neural
networks for language. I mean, these slides are sort of positioned a bit as convolutional neural
networks versus RNNs, as opposed to versus Transformers. I mean, that's partly, you could say,
because I haven't updated my slides enough, but in another sense that's partly because that's how
the ideas of convolutional neural networks really were explored — it was in the days when most
people were using recurrent neural networks for NLP

**[4:43]** a few people started saying, hey, maybe we should use convolutional neural networks for
language as well, whereas in truth, in the last five years when Transformers have dominated,
there hasn't been much use of convolutional neural networks for NLP. So if we think back to our
RNNs — if you remember those — they kind of gave a way of giving a representation for a sentence
or part of a sentence, but they sort of computed forward through the string, and so you kind of
had to get a representation that included everything that came before you, and then — so, you
didn't really have a representation of the ceremony

**[5:28]** — you had a representation of Monáe walked into the ceremony that you could use. So in
contrast to that, convolutional neural networks basically say, well, kind of like an n-gram model,
that we should be able to take n-grams of words, like bigrams or trigrams. For the example
"tentative deal reached to keep government open," we can take each trigram — tentative deal
reached, deal reached to, reached to keep, to keep government, keep government open — and we can
make some neural representation for each of those. So notice, it's just being done for every
n-gram of a certain n, so there's nothing linguistically or cognitively especially plausible here,
but we're

**[6:15]** just going to sort of form representations of multi-word units, which we'll then group
in some form further way, later on, and the standard way of doing that is with convolutional
neural networks. So the classic case of convolutional neural networks is in vision — the
convolutional neural networks were invented for vision, where they gave you a kind of a
translation-invariant model, so that you could recognize your kangaroo no matter where in the
frame it was. And so this little picture here — I'll just do the lower half of the slide — is
sort of what a convolutional neural network is doing in 2D vision. So the convolution is like a
mask that

**[7:03]** you're sliding over the image, and the mask is defined by weights, which are the little
things shown in red. And so for each place you slide your mask to, you're then calculating a score
by taking — well, what's effectively — a dot product of the mask terms by the elements in that
patch, and that's then filling in the matrix on the right, that's shown in pink. And so that's
then calculating our convolved feature from the image. Does that make sense? *Yeah.* So, well,
what happens if we then want to do that for language? Well, for language we don't have a 2D

**[7:49]** picture, we've got a 1D picture, we've got a sequence of words. So we can have
"tentative deal reached to keep government open." So each of our words will have a word vector —
I'm using four-dimensional in my examples to keep it compact on my slide — and so then we can
apply a filter that applies to an n-gram, so this is going to be a filter for a trigram, and so
then we're going to slide that downwards in exactly the same way as for the vision case, apart
from we're just sliding in one dimension. So I calculate the dot product of the filter and this
trigram, and that gives me a value, minus one, if I did my

**[8:36]** arithmetic right. Then I slide it down to the next position and work it out, and I get
minus 0.5. Slide it down, get the other values, and then typically I can add on a bias term — so
my bias is plus one in this example — and then I'll stick it through a nonlinearity, like a
sigmoid or something like that. And so I'll be calculating a value for each of these trigrams, and
so that is a convolution for a single filter. And then, commonly, what I'm doing after that is
deciding that I'm going to have more than one filter, and I'll show that in a minute, in this
example — and in my vision

**[9:22]** example earlier, we sort of had shrinkage, because we started off with seven words, but
of course we sort of slid this trigram over it, we only sort of had space for five trigrams, and
so we ended up with something smaller than our input sentence. Often people want to keep it the
same size, and the way you can keep it the same size is by having padding. So if I put a zero
padding at each end, well, now I'm going to get seven trigrams coming out, corresponding to my
original seven words, and normally I'll just sort of pad it with zeros like that. You can actually
increase the size of things, because if you add

**[10:10]** padding of two at each end, you can then have a wide convolution, and so seven will
then go to nine different things. Okay, so if we only had one filter, things are pretty limiting,
and so, commonly, as in the vision case, what we're going to do is define multiple filters, and
then we're going to be calculating a value for each of these filters over each of these trigrams.
And so then we're getting out a new representation as a vector, and, depending on how many filters
we have relative to what the word dimensionality is, we might end up with something that's
shorter, as in this example, the same length, or actually longer than what our

**[10:56]** input was in terms of word vectors. But, commonly, when we do that, we then, in some
way, want to summarize all of these filters, and the most common way of doing that is to do
something that's called max pooling, and max pooling is something you see quite a bit in neural
networks in general. And the way to think of max pooling that I think makes sense is — you can
think of max pooling as doing what you want if you want to run something that's like a feature
detector. So, if you imagine that you learn these functions that

**[11:44]** will look at word vectors, and that they will look for evidence of something
particular. So, maybe this filter looks for the person using "I" language — that it matches the
words "I," "my," "we," "our," something like this. And maybe this is a filter that matches speech
verbs, or thinking verbs, like think, say, said, told, etc., like that. And so each of these is
sort of some kind of feature of the text that you might want to detect. Well, if that's your model
of it, when you sort of slide your feature detector down the piece of text, you sort of want to
know: does this match

**[12:29]** anywhere in this piece of text — is somewhere at all using an "I" word, regardless of
whether it's in the first, second, third, or fourth position. And so that's effectively what
you're getting out with max pooling — that a feature is counted as firing to the extent that it
fires strongly in any position in the text that you could match. That's not the only way you can
think of doing it — an alternative way you could do it is you could think of your feature detector
as sort of measuring some quality of the text, like casualness or learnedness or something like
that, and then you might think, oh, well, for overall wanting to know how casual the text is,
maybe I want to know the average of how casual

**[13:16]** it is in different parts of the text, and so then you can do the alternative of
average pooling, and sometimes people do that as well — you can do both, you can work out an
average pool and a max pool and put both of them into the feature detector. In general, for the
kind of features people learn in neural networks, if you're just doing one or the other, the
result does seem to be that max pooling is the most effective — that kind of "does the feature
fire" metaphor tends, in general, to be the best way of thinking about things. Okay, so if you
want to do all of this in PyTorch, Conv1D, right — so, I guess the one-dimensional convolutions
aren't the most common case

**[14:03]** and so you're using Conv1D and all these kinds of things that you can then be
specifying. So the output channels is the number of filters you have, the kernel size is saying —
the size is how big it is, which for my example was three — and then you can sort of just collapse
things with the max pooling. Okay, there's a space of other things that you can also do with
convolutional neural networks, which I think are sort of less useful and less used in language
cases, but I can sort of say them quickly. So one thing you can do is sort of have a stride,
because when we sort of did every

**[14:50]** trigram — of sort of zero, tentative, deal, then tentative deal reached, then deal
reached to — you could feel like, well, they're overlapping each other a lot, so they've actually
got very similar stuff in them, and that would be even more so if we weren't using trigrams, if we
were using something like five-grams. So something that you can do is — the stride is sort of how
much you move along. So if you move along two, you'd have one trigram that's padding, tentative,
deal, and then the next one would then be deal reached to, and then the next one would then be to
keep government, so that they're overlapping by less as you go through it. Another thing that you
can then do

**[15:35]** that's sort of stride-like, is rather than doing max pooling over the entire thing, you
could do more of a local max pool. So you could think that, well, I want to have this feature
detector for something like use of "I" language, but if it's a big, long sentence and there's "I"
language at four different points, maybe you should get four points for that, rather than just
sort of the one point that you're going to get from max pooling. So you could sort of do local max
pooling, sensitive to the stride — so here I could look at the first two of these and max pool
those two, then the next two and max pool those, the next two and max pool those, and the next two
and max pool

**[16:21]** those, and you could sort of then end up with this sort of local max pooling as you go
along. Okay, and then one other idea that's sort of related that you can do — well, another way of
capturing sort of "does something match in multiple places" is, rather than only keeping the one
max in each column, maybe you could just do a k-max, so you could keep the two maximum things in a
column, and that might also be a way of seeing whether something is detected in two places or not.
Okay, then I've got lots of

**[17:08]** notions here. Okay, so dilation is then the notion that what we'd like to do is sort of
form our trigrams not only as adjacent things but as things that are spaced out. So after having
done our first layer of convolutional filters that took trigrams, that got us to the sort of
top-right part here, we could then do a dilated trigram convolution, which means that we're going
to take the first, third, and fifth things and combine them in a convolutional filter, and then
we'll take the second, fourth, and sixth things and

**[17:55]** combine them in a convolutional filter, and so we've then got a trigram filter, but it
sort of has a bigger range of size that it can see. And that's more commonly used in places like
speech than in natural language. Okay, so those are the kind of tools we have for calculating
things with these convolutions over text. And so next, what I want to do is tell you about a
couple of pieces of work that made use of convolutions in natural language processing. I guess
this is a decade old now, because this is from 2014 — this is the single most famous piece of work
that made use of

**[18:41]** convolutional neural networks for natural language processing. And Yoon Kim is now an
assistant professor at MIT. I mean, in retrospect, it's sort of actually pretty simple, but, I
guess, he got in early with the idea of, okay, maybe we could use convolutions for NLP, and did a
kind of clear example of that that worked pretty well, and so this piece of work is very well
known. So this was writing a sentiment classifier — so, looking at a sentence and deciding whether
it's positive or negative. And actually, for both of the kinds of models that I'm going to talk
about today, we're

**[19:27]** going to use examples that are doing sentiment classification. He also considered
other tasks — subjective or objective language, question classification as to what they were
about — but the main application was doing sentiment analysis. So what you're going to be doing —
this paper shows things more in his notation, but it's exactly the same as we've just been talking
about — that you're taking n-grams of word vectors, you're going to be multiplying them by a
convolution and calculating new vectors, and it's going to be done in his model for different
sizes of n, so he's going to have some

**[20:13]** convolutional filters that look at bigrams, some at trigrams, and some of them that
look at four-grams, and then those just slide across the positions in the sentence. Then, having
done that, it does max pooling, as we've been talking about, which gives a single number coming
out of each filter, and those max-pooled numbers from each filter are then going to be used as
input to a classifier — a final, simple softmax layer — to give the final answer. There's one
other thing that came up in this paper, which is kind of just an interesting general idea, to

**[21:01]** be aware of, and it was something he sort of pioneered, which is the following: that
it's a very common case that what you find, when you're sort of doing this — I guess this again
occurs less with huge pre-trained Transformers, but for sort of the classic case of models where
you had word vectors and then you were training some neural network model on some supervised data
— there was this following pitfall of what happened when you fine-tuned word vectors. And so the
setting is: we've started off with our pre-trained word vectors from GloVe or word2vec, or
whatever it

**[21:48]** is, and then we've got a smaller sentiment analysis data set, and that we're going to
train a sentiment classifier, and that will involve not only learning the parameters of our
sentiment classifier, but also we can backprop into the word vector representations. And if you do
that — I mean, it seems like that should be a good idea, because normal word vectors aren't
especially tuned to predicting sentiment correctly, they're sort of more tuned to the meaning of
words, as to sort of just what words are about — and so it

**[22:34]** seems like it should help you if you could backprop into the word vectors and change
them as you go along, but if you do that, there tends to be a problem. And the problem is, what
you'll find is that some words will be in your sentiment training data set, and when you learn,
with backprop, these word vectors will move, but some words just won't be in your training data,
and they're going to stay exactly where they were in the word vectors, because there's nothing to
move them around. So what tends to happen is, you sort of started off like this, where all of
"tedious," "dull," and "plodding" were close by

**[23:22]** each other, as having similar meanings and being indicators of something negative. But
after you've done your training, "tedious" and "dull," as part of backprop, have moved over here,
where they're part of the negative land, and the classification boundary's moved over here. But
"plodding" wasn't in the training set, so it's just sitting exactly where it was at the start of
the process, and now it's being treated as a positive word, which is completely wrong. And so
that's tended to have the result that, when people sort of train a language neural network on a
small supervised data set, you got kind of ambivalent results — that sometimes doing backprop into

**[24:11]** the word vectors would help, because you could specialize your word vectors to your
task, but sometimes it would hurt you, because of this kind of effect, that you sort of messed up
the semantic relations that were captured reasonably well in the initial word vectors. So the way
that Yoon Kim dealt with that was a fairly simple way — he just doubled his number of channels, and
so he made two copies of each channel, each filter, in his convolutional neural network, and for
one of them it used the fine-tuned word vectors, and for one of them it kept the original word
vectors, and then he could have the best of both

**[24:59]** worlds. Okay, so this picture captures the sort of whole of his network. This picture
actually comes from a follow-on paper which produced this nice picture. So we start off with a
sentence, "I like this movie very much," which should be classified positive. So we have words and
their word vectors, and so then you're going to have convolutional filters that are both bigram
filters, trigram filters, and four-gram filters, and at each of those sizes you're going to have
ones that work on the un-fine-tuned word vectors and the fine-tuned word vectors, and

**[25:46]** so you're going to put these filters and slide them over the text and get
representations. And the way he's doing this, the filters are done without padding, so that the
four-gram filters, you're getting smaller vectors coming out, and the bigram filters, you've got
bigger vectors coming out. And so then, for each of these, you're then going to max pool — so
you're just getting the highest value from it — and then you're getting a highest value from the
ones with the fine-tuning of the word vectors, and the ones not. And so you're getting one feature
out of each filter. You're then concatenating all of those max-pooled

**[26:32]** outputs together, so then one vector for the entire sentence, which is of fixed size,
reflecting the number of filters. And then you're just sticking this, as a straightforward linear
classifier, into a softmax, that's then giving you a probability of positive or negative, and that
was the entire model. And the interesting thing was, this actually worked pretty well for natural
language classification tasks. So this is a big table of results from his paper — so there are
sentiment data sets, like the Stanford Sentiment Treebank, two versions of that, movie

**[27:18]** reviews, there's another sentiment data set, there's a subjectivity classifier, the
TREC was the kind of question-type classifier — so various data sets. And various people,
including us at Stanford — I guess all of these, our results were ones we were doing at Stanford —
had built lots of models on various of these data sets, and his argument was that by using this
simple convolutional neural network you could do as well, sometimes better, than any of these
other models that were being considered at the time for

**[28:04]** sentiment analysis. Now, there was at least one way in which maybe that comparison was
too generous to the CNN, because — if you remember, back when we were doing dropout, and we said
dropout is such a good idea — I mean, dropout, I think, came out in 2012, if I'm remembering
correctly. So the reality is, a lot of these other methods were being written before dropout
appeared on the scene, whereas he was using dropout, and that gave him an advantage. A sort of
better experimental technique might have been to redo the other models with dropout as well, which
he didn't

**[28:50]** but nevertheless, it sort of shows that you could get strong results using
convolutional neural networks with just a very simple architecture. So that's one more thing that
you can do. And so, I mean, the thing to think about here is, we have this sort of toolkit of ways
that you can do things. We started off with word vectors and bags of vectors, which you could use
for simple classification. We talked early on about window models, and window models are sort of
like what you get for convolutional neural networks, but sort of more ad hoc. Then we have
convolutional neural networks, which are definitely good for

**[29:36]** classification, and very easy to parallelize, which is good. And we talked about
recurrent neural networks, which seem to be cognitively plausible, reading through sentences from
left to right, but aren't easy to parallelize. And then we've talked about Transformers, which, to
some extent, is our best model for NLP and is being used everywhere. And indeed, what's happening
now is that things are going in reverse, and people are increasingly using Transformers for vision
as well, though there's still, I think, more debate in the vision world as between CNNs and
Transformers, with some people arguing that both of them have complementary advantages. Okay, a
couple of other, just

**[30:23]** facts on the side, and then I'll show you one other, bigger, fancier convolutional
neural network model for language. So, we talked about, for Transformer models, the use of layer
normalization, which sort of keeps the size of the numbers in the middle layers of the neural
network about the same, by giving zero mean and unit variance. There are slightly different ways
that you can do that — for convolutional neural networks, the standard thing to be using is batch
normalization, and indeed batch normalization was the thing that was invented first, and sort of
layer

**[31:10]** normalization and batch normalization are sort of doing the same thing, of sort of
scaling numbers to give them zero mean and unit variance, but the way that they differ is sort of
under what dimensions they're doing their calculations. So LayerNorm is calculating statistics
across the feature dimension, whereas BatchNorm is normalizing all the elements in the batch for
each feature independently. Okay, one other little concept that turns up, which actually sort of
connects a bit to Transformers as well — there's this sort of funny thing, that — all of what I've
presented

**[31:55]** so far was sort of convolutions that are — some unigram, bigram, trigram, four-gram —
and so they're also, um, size-one convolutions. And at first sight that seems to make no sense at
all, 'cause what's the point of doing a size-one convolution, because you just got one thing and
it's staying just one thing. But it actually does make sense, because it corresponds to having a
little fully connected layer that's only looking at the representation in one position. So, in a
language term, it's taking a word vector and putting it through a fully connected neural network
to produce a new

**[32:41]** representation, just of that word. And that's sort of actually what we also have with
the fully connected layers in Transformers, right — that you've got a fully connected layer that's
just at one, well, subword token position, and calculates a new representation for it. And so
that's — that allows you to sort of create new representations with actually many fewer parameters
than if you're allowing a fully connected layer across the entire sentence. Okay, and so this is
then a more recent version of a convolutional neural network, still again used for text
classification, but a much more complex

**[33:27]** one, from Conneau et al. in 2017. And again, this was still at the stage in which LSTM
sequence models were dominant in NLP — I guess in 2017, this is sort of the same year the first
Transformer paper came out. And the motivations were sort of comparing vision and language, and
so, at that point in time, convolutional neural network models in vision were already very deep
models, so people were using things like ResNet models that had 30, 50, 100 layers in them, and
that stood in stark contrast to

**[34:16]** what was happening in the LSTM world for sequence models, where commonly people were
just using two-layer sequence models, and if you're wanting to go further you might be using a
three-layer sequence model, or four-layer sequence model, or, occasionally, if you got really,
really deep, people had used eight-layer sequence models, if they had a lot of data. But
essentially, it was always a single-digit number of layers. And then a second thing was, in some
sense, the vision models were more raw-signal models, because they were operating on the
individual pixel level, whereas in NLP the standard was that we were using word-level models

**[35:03]** still, in the Transformer model. So it sort of seemed like things were much more
grouped before they began. And so the idea of this paper is, well, maybe we could do NLP kind of
like it was vision — so we'll start with the raw characters as our signal, we're going to put them
into a deeper convolutional neural network, and use the same kind of architecture we use for
vision, and use that for language classification tasks. And so that led to this VD-CNN
architecture, which is something that looks very like a vision system in design. And so, what do
we have here?

**[35:51]** At the bottom, we have individual characters, and the individual characters get a 16-d
representation. And then you've got some sort of size of piece of text that you're classifying,
which for them was 1,024. And then, at each stage, we're then going to have convolutional blocks,
and so these convolutional blocks have a whole bunch of filters, but they're also then going to
group stuff together, so that we're kind of sort of starting to collapse into multi-character

**[36:36]** units. So we're starting off, first of all, having 64 size-three convolutional filters,
and so that gives us a representation of 64 times the window size. And then we're going to do that
again, and put it through another set of convolutional filters, of size three, and 64 of them,
which gets us sort of up to here. And then, at each point, we also have residual connections,
which we also saw in Transformers, but were pioneered in the vision space, so that we have a path
that

**[37:22]** things can just go straight through. But then, when we get to here, we're then going to
do local pooling, so each pair of representations here will be pooled together. And so at that
point, we've no longer got a length of the initial length of 1,024, we've now got a length of 512.
So now we're going to be putting it through, again, sort of trigram convolutions, but now we're
going to have 128 of those channels. We're going to repeat that again, and then we're going to,
again, group with pooling, so now we're going to

**[38:09]** have a 256-long sequence, because we've done local pooling of each pair, and we're
going to then have 256 filters at each stage, and we go up. And then we do local pooling again, so
each of them is now representing an eight-gram of characters, and we're putting trigram filters
over those eight-grams. So really, the amount of a sentence that the convolutional filters are
seeing at this point is 24 characters, so sort of seeing something like six-word sequences or
something like that. More convolutional blocks there, then, at there, they then do this

**[38:57]** k-max pooling. So some of the ideas from the beginning of the lecture do show up — so
you're then doing k-max pooling, and finding the eight highest activations in the sequence, and
that sort of makes sense for something like a text classifier, because you want to count up the
amount of evidence, right? If you've got some category like — is this about, I don't know, copper
mining — you want to be seeing whether there's a bunch of places in the text that's talking about
copper mining. And then, right up at the top, they have several fully connected layers, which
again is very typical of what you find in vision networks, such as something like VGGNet, that,
after

**[39:44]** you've done a whole bunch of convolutional layers, you just stick it through multiple
fully connected layers at the top, and so that's what they're doing as well, and this is the
architecture for doing text. Okay, I think I talked through that in a lot of detail, so I'll skip
this slide. So their experiments were done on text classification data sets — various news
classification data sets, DBpedia ontology, then doing sentiment analysis on Yelp reviews and
Amazon reviews. And here, results

**[40:31]** from theirs — so they're taking the previous known best published results, which are
shown here in Table 4, and then they're considering whether they can do better by using their
architecture. And they used architectures of different lengths, in terms of the number of layers —
of nine layers, 17, and 29 layers — and the result of the paper is, in all cases, they got the best
results with their deepest network, which was a 29-layer model, which is sort of then similar to
what people were doing in vision. And then, there's some

**[41:16]** variation as to which was best, using the max pooling or the k-max pooling, but in
general it was always the deep model, and it varied a bit according to the data set. But at least
sometimes they were able to produce the best results that were known. So, I mean, I guess for
these text-classification ones, previous results were slightly better than their results, but for
some of the other ones, like the DBPedia and the Yelp — well, for both of the Yelp data sets their
results were better than the best known previous results. The Amazon ones, one was better, one was
worse. But, to a first approximation, this meant that they could basically reach the

**[42:01]** state of the art of a text classification system with something that was just a deep
convolutional neural network, starting from the character level, with none of the sort of having
learned word vectors in advance or anything like that. And so that was a pretty cool achievement,
which showed that you could go a fair way in doing things with just this sort of raw,
character-level convolutional neural networks, sort of more like a vision system. Okay, so that's
that. And then, for the final piece of the class, I want to tell you about something in the other
extreme, which is about tree recursive neural networks. So, tree

**[42:48]** recursive neural networks is a framework that me and students developed at Stanford.
So, I mean, really, when I first got into neural networks, in 2010, that sort of, for about the
first five years, what me and students worked on was doing these tree recursive neural networks,
and so they were sort of the Stanford brand. Ultimately, they didn't prove as successful as other
things that came along, but I think they're linguistically interesting, and I think there's a
clear idea here, which is still an idea that exists, and I think there may be

**[43:34]** still some things to do with, which I'll come back to. But the starting point is
essentially being motivated by the structure of human language, and so most of this slide is
filled by a paper from Hauser, Chomsky, and colleagues, sort of discussing their views of the
human faculty of language — what it is, who has it, and how did it evolve. And I don't want to
dwell on this in too much detail, but essentially, in this paper, what they argue is that the
defining property of human language, that's not observed in other things that humans do, is that
language has this recursive structure, that you have this

**[44:21]** hierarchical nesting, where the same structure repeats inside itself. So if you have an
example, like "the person standing next to the man from the company that purchased the firm that
you used to work at" — what you have is, the whole of this is a noun phrase, "the person," headed
by "the person," and then it's "standing next to," then the first square brackets here is another
noun phrase, "the man from," then inside that prepositional phrase there's another noun phrase,
"the company that purchased," the firm, and then "the firm" is another noun phrase that has the
relative clause modifier "of the firm

**[45:07]** that you used to work at." So we have these embedded layers of noun phrases with the
same syntactic structure underneath them. And so, for the kind of formalisms that we use in
linguistics, of context-free grammar, it permits the kind of infinite embedding of nesting, which
is the same kind of nesting that you get in programming languages, where you can sort of use if
statements and nest them as deeply as you want to, because you just have the same repeating
recursive structure. Now, of course, human beings can't actually understand infinite recursion,
and people don't actually produce infinite recursion — you could sort of say, oh, in practice no
one's going to

**[45:52]** go more than eight deep when they're saying a sentence, but in terms of the structure
of what the language looks like, it seems like you should be able to do it infinitely deep. And
when you actually start looking at the structures of sentences, they do sort of repeat over the
same structure quite deeply. So this is an example of a Penn Treebank tree, which is sort of the
best-known constituency treebank. And so here's my random sentence: "Analysts said Mr. Stronach
wants to resume a more influential role in running the company." And, well, what we end up with,
sort of, if we have these nested things of verb phrases — so "running the company" is a verb

**[46:39]** phrase — "resume a more influential role in running the company" is a bigger verb
phrase, "wants to resume a bigger role in running the company" is an even bigger verb phrase, and
then "said Mr. Stronach wants to resume a more influential role in running the company" is an even
bigger verb phrase. And so we have sort of one, two, three, four verb phrases, all nested inside
each other. And so the idea was, well, maybe we should be thinking of sentences as having this
kind of tree structure, and computing representations of meanings of sentences in terms of

**[47:25]** this tree structure. So we have words that have representations in word vector space,
like we saw right at the beginning of the class, but then we're going to have a phrase like "the
country of my birth." And the classic linguistic answer that you find, both in linguistic
semantics classes or philosophy of language, is that we should construct representations of
phrases using the principle of compositionality, which says that the meaning of a phrase or
sentence is determined by the meanings of its words, which are our word vectors, and the rules
that combine them. So maybe we could take the phrase structure tree of a sentence and combine the
word vectors

**[48:14]** together, by some means, and then we can construct a representation of the meaning of
phrases in a more linguistic way, giving us a vector representation of the meaning of the phrase,
which we could also put into our vector space, and we'd hope that a phrase like "the country of my
birth" would appear in the vector space in a similar place to where words representing locations
appeared. Okay, so what we want is to be able to start with word vectors and parse up a sentence,
and as we parse the sentence, we're then going to be computing representations for the different
phrases of the sentence. And so the difference here is, now you

**[49:03]** know — the difference between "recursive" and "recurrent" is sort of a fake
difference, right, they both come from the same "recur" word — but rather than having the
recursion just happening along a sequence, as in a recurrent neural network, we're going to have
the recursion happening up a tree structure, so we can compute representations for linguistically
meaningful phrases. And so there — what we're going to do with that is, the easy case is, if we
know the phrase structure tree, we can take the representations of the child nodes, put

**[49:49]** them into a neural network, which could give us the representation of the parent node.
But we'd also like to find the tree structure, and so a way we could do that is, we then get a
second thing out of the neural network — we could get a score for how plausible something is as a
constituent, does it make sense to combine these two nodes together to form a larger constituent —
and then we can use that in parsing. So, formally, the very simplest kind of tree RNN, and the
first one we explored, was: when we had two child vectors, we're going to be representing the
parent vector by

**[50:35]** concatenating the two children, multiplying them by a matrix, adding a bias, putting it
through a nonlinearity to get a parent representation, p, and then we'd score whether it's a good
constituent by taking another vector of learned parameters, which would do a dot product with p,
and that would give us a score as to whether this was a good constituent to include in your parse
tree. And the same W parameters were used at all nodes of the tree, in the same way as a recurrent
neural network kept using the same parameters. Okay, so if we did that, if

**[51:22]** we had that, we could build a greedy parser, 'cause what we could do is, we could start
with all the word vectors, and we could just take every pair of words and put it through this
system, and calculate what the representation of that pair would be as a constituent, and then get
a score as to whether it seemed a good constituent or not. And then we could just greedily decide —
this is the best constituent, "the cat" — and so, if we do a greedy parse, we can commit to that,
and then, well, we still know the possibilities of combining other pairs of words, and we could
just additionally score how good "the cat"

**[52:09]** combined with "sat" is, so that we're producing binary parse structures. So now the
best pair to combine, greedily, is "the mat," so we can combine them together and commit to that.
We can score combining "on" with "the mat," and now that seems the best thing, so we'll commit to
that, and we just sort of keep going on up, and we produce the binary parse of the sentence, and
this gives us our sentence representation. Okay, which is like that. Okay, and so that gives us our
simple RNN, and so, back in 2011, we got some pretty decent results, showing that you could use
this as a

**[52:56]** sentence parser that worked pretty well. But, beyond that, the representations we
calculated for sentences and phrases were good enough representations that you would use it for
tasks like sentence classification, sentiment analysis, and it works reasonably well. It only
worked reasonably well, though, because, if you start thinking about it further, there were strong
limitations of having this single W matrix that's used at all points to combine things — that if
you sort of have that architecture, you sort of can't have

**[53:42]** different forms of interaction between the different words — you're just uniformly
computing things — and that sort of stands in distinction to the fact that different kinds of
things in natural language seem kind of different. You have different properties with verbs and
their objects, versus an adjective modifying a noun, just in terms of what the roles of the
different words were. So we started to see limitations of this architecture, and so, in following
years, we started exploring other ways to build tree recursive neural networks, which had more
flexibility as to how things were combined together, and I'm not going to show you all the details
of all of that, but I will show you

**[54:29]** one more model that we used for building tree recursive neural networks, and that was
used in sort of some of our sentiment analysis work, called the recursive neural tensor network.
It wasn't actually the final version that we did — after that, we sort of started taking LSTM
ideas and extending those to the tree-structured case, and we worked on TreeLSTMs, but I'm not
going to show that this year. But the idea of recursive neural tensor networks is, when pairs of
words or phrases combine together, in linguistic semantics terms, depending on the pairs of words,
they modify each other in

**[55:15]** different ways. So if you have an adjective and a noun, like "a red ball," sort of
"red" is giving attributes of the noun, whereas if you have something like a verb and object, like
"kick the ball," you've got a very different role for the object, as the right-hand side, versus
"the red ball," it's sort of the opposite way around. So we want to have more flexibility in the
way we can calculate meanings of phrases depending on what's in it, and the way we came up with
doing that is to come up with what we call this neural tensor layer. And so the idea in the neural
tensor layer is that we have the representations of the child

**[56:04]** words or phrases. And so, rather than directly concatenating them and then putting it
through a sort of linear transformation, like a regular neural network layer, instead, what we
could do is, we could learn in-between matrices, and if we put several of those together, we're
then getting a three-dimensional tensor, and we could multiply a vector by a tensor times a
vector, and then we'll end up getting out vectors — for each one, we'll have multiple such

**[56:54]** vectors. Okay, and the place that we applied this model is for this task of sentiment
analysis, so let me just tell you a little bit more of what we did here. And this is, in fact,
going back to the Stanford Sentiment Treebank, that was already used in Kim's work. So the goal of
sentiment analysis is to see whether a piece of text is positive, negative, or neutral. So a lot of
the time, doing sentiment analysis is pretty easy — in the 2010s, and probably even today, quite a
few people's sentiment analysis systems are essentially just keyword matching, right — if you see
"great," "marvelous,"

**[57:40]** "wonderful," positive sentiment; if you see something like "poor," "bad," negative
sentiment. And so, lots of the time, you can sort of effectively do a kind of dictionary matching
and get pretty good sentiment, especially on longer documents. But, on the other hand, people use
language in lots of interesting ways, and it's not always that easy. So if you look at something
like movie reviews, such as the snippets you get on Rotten Tomatoes — you get snippets like this
on Rotten Tomatoes: "With this cast and this subject matter, the movie should have been funnier
and more entertaining." And if you just think of it as, okay, we're doing dictionary matching,
there's the word

**[58:27]** "entertaining" — that's definitely positive — and "funnier" — that's positive — so
there are two positive words, so this should be a positive review. But, of course, it's not a
positive review, this is a negative review, because it's saying — well, I'm just reading it out
again — "with this cast and subject matter, the movie should have been funnier and more
entertaining." Right, so the compositional structure of human language goes together to mean that
— because it's buried under "should have been" — the "funnier" and "entertaining" are actually
lacking, and so it's a negative review. And so these were the kind of examples that we were
interested in, and saying

**[59:14]** could we sort of actually understand the structure of sentences more and do a better
job at sentiment analysis. And so, up until this time, people just sort of had pieces of text and
a classification judgment of positive and negative. So we decided we're going to do more than
that, and come up with the Stanford Sentiment Treebank, where what we did was parse up a whole lot
of sentences — almost 12,000 of them — and then what we're going to do is put sentiment judgments
on every linguistic phrase of the sentence. So, for something like this example, "with this cast"
is a

**[59:59]** phrase — no sentiment, so that's just neutral. "Entertaining" is a phrase, a one-word
phrase, its sentiment is positive. "Funnier and more entertaining," that's a phrase, very positive.
But then, by the time we get embedded under "should have been," "funnier and more entertaining" —
that's a bigger phrase, its sentiment is now negative — and "the movie should have been funnier
and more entertaining," that's an even bigger phrase, it's negative. And so we were parsing up
trees like that, and these examples are very small, I'll show you bigger examples later, but you
can sort of just see that

**[1:00:46]** in the trees, there are blue nodes and orange nodes, corresponding to positive and
negative sentiment, reflecting units at the different sizes. And so the interesting thing is, this
gave us a richer annotated data set, because it's not only whole sentences or whole articles that
were annotated for sentiment — we had annotations for different phrases. And simply the fact that
you were annotating phrases meant that you could learn more from the examples — so even if you're
using something very simple, like a naive Bayes classifier, because there are annotations on words
and smaller phrases, you could learn a bit more about which were

**[1:01:31]** positive and which were negative. And so that was the first result — people use a
baseline method of a bigram naive Bayes classifier, which is a very common sentiment classifier:
if you just trained with sentence labels you got 79% on this data set, if you trained using every
node of the treebank you got 83%, so you got a 4% lift, and so that was kind of good. These other
two lines show two of our early tree RNNs, and the negative part of the result is, they weren't
really better than a bigram naive Bayes classifier — they

**[1:02:17]** were better than a unigram naive Bayes classifier, but a lot of the extra information
that you want to capture for sentiment analysis, you can get from bigrams, because that can
already tell you sort of "not good," "somewhat interesting," and things like that. But then, so
the other hope was to have a more powerful model, and so that then led into use of this recursive
neural tensor network, which allowed sort of the mediated, multiplicative interactions between
word or phrase vectors. And so we built that, and so then here are the results of that model,
that's

**[1:03:03]** shown in red. So, by having our recursive neural tensor network, we were able to
build a somewhat better neural network that performed at least reasonably better than a bigram
naive Bayes model — right, we were getting sort of about 22% better than a bigram naive Bayes
model — so that was progress. But I think, perhaps, the more interesting thing isn't the aggregate
results, but the fact that, because we were building up this model — the computed representations
over a constituency tree — it actually made judgments of

**[1:03:49]** different parts of sentences and how they combined. So, here's the movie review
sentence: "There are slow and repetitive parts, but it has just enough spice to keep it
interesting." So I hope you'll agree with the judgment that, overall, that's a positive statement
about the movie. And so the recursive neural tensor network builds the tree structure over this
sentence, and it says — "slow and repetitive," that's negative, "there are slow repetitive
parts," it's all negative here — but for the part over to the right, "interesting," "spice," they're
both positive, and "spice to keep it interesting," that's positive

**[1:04:34]** "it has just enough spice to keep it interesting," positive. And it correctly
predicts that, when you put these two halves of the sentence together, the overall judgment is
that this remains a positive review, and it gives a positive judgment overall. So that was kind of
cool. And, in particular, the fact that we were building these phrase judgments meant that it
seemed like we could actually do a better job of sentence understanding, in the way that sort of
any linguist doing linguistic semantics would like to see sentence understanding. So one of the
things that neural networks, when looking at language, have often been faulted for

**[1:05:20]** and are still faulted for, to this day, using Transformer models — is, you often find
the result that neural network models just don't pay attention to negation. That you can have some
sentence, and you can compare the sentence of — a lot of students are studying for their final
exams, versus a lot of students aren't studying for their final exams — and the negation just gets
lost, that it doesn't produce the differences in representation and meaning that you'd like it to.
So, somewhat interestingly, with this model, it seemed like, because we were modeling the

**[1:06:06]** the recursive building-up of sentence structure, that we actually could do
interesting things with modeling negation. Right, so in particular, the results that you'd like to
get is, if you have something like "it's just incredibly dull" — so "dull" is a very negative
word, "incredible" is a positive word by itself, but when you're sort of saying "incredibly dull,"
it's definitely still negative — and this, our recursive neural tensor network is correctly
modeling: "it's just incredibly dull" is

**[1:06:51]** very negative, despite "incredible" being a sort of positive word. So, actually, in
this model, there was five-way classification, so there was very negative, somewhat negative,
neutral, somewhat positive, very positive. So there's sort of some bouncing around as to whether
it's giving the classification very negative versus somewhat negative — I can't really explain why
in the middle it goes to somewhat negative and then goes back to very negative, but that's the
results that came out of the network. And, at any rate, it all stays negative — the fact that
"incredible" by itself, "incredibly," is a positive word, it's seen in the modification of "dull,"
and that keeps it negative. But, on the other

**[1:07:39]** hand, if you put a negation in here — "it's definitely not dull" — well, then what
happens? Now, interestingly, the word "not" by itself is a negative word. If you just sort of do
the raw statistics of it, "not" occurs much more often in negative-sentiment sentences than it
does in positive-sentiment sentences. So, if you want to be a more positive person, use negation
less. So "not" by itself is negative, but if you then combine it together — "not dull," or, in
this case, "definitely not dull" — well, "not dull" is — you have two negations, so that they

**[1:08:25]** cancel each other out, and you get something that's positive, and so "it's
definitely not dull" comes out as a positive sentence. And so the interesting result here is that,
if you compare what happens between — if you have negated positive sentences, so, "it's definitely
not good" — various models can model that correctly, because "not" is a negative word, and so,
therefore, it weakens the positivity of the positive word, and so putting a "not" in front of a
positive — into a positive sentence — makes it less positive. And even a "not very

**[1:09:14]** well." But even a naive Bayes model can do that, because "not" by itself is seen as a
negative word. But the hard case is, what happens if you negate a negative sentence? Well, the
result that you should get is, it becomes more positive, and neither a bigram naive Bayes model,
nor our earlier attempts at recursive models, can capture that, whereas this RNN structure was
able to correctly capture this sort of semantic modification structure, and say, hey, that's made
the sentence much more positive. So that was a cool result, and, to some extent, this result, I
think, still isn't captured as well by any of the current

**[1:10:00]** Transformer models, even though they have many other advantages and are much better
than a tree recursive neural network. So, I mean, just to say a couple — this is basically the end
— just to say a couple of final remarks about these tree recursive neural networks: the reason
that they became uncompetitive is because they just didn't allow the kind of associations and
information flow that you have in a Transformer, right, that these models had a strictly
context-free backbone, and the only information flow was tree-structured,

**[1:10:48]** following the context-free backbone, whereas in the Transformer you've got this
attention function, where, at every position, you're looking at every other position, and so you
can have much more general information flow, and, in general, that is just good, and Transformers
are much more powerful. But, on the other hand, to the extent that you actually want to carefully
model the sort of semantics of human language — sort of what modifies what, and how does negation
or quantifiers in a sentence behave — in some sense these models were more right. And so one of the
things I'm still kind of interested in is, are there any opportunities to

**[1:11:35]** combine together some of the benefits of both of these ways of thinking, and have
something that's a bit more tree-structured, while still more flexible, like a Transformer. Okay,
that's it for today. Thanks a lot.
