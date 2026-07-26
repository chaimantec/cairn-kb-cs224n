---
title: Benchmarking
lecture: 12
video: https://www.youtube.com/watch?v=TO0CqzqiArM
source: edited from auto-generated YouTube transcript
verbatim: false
original: original/12-benchmarking.md
slides: ../slides/12-benchmarking.md
---

# Benchmarking — transcript

Copy-edited for punctuation, sentence boundaries, and mis-heard terms; cross-checked
against `raw/slides/12-benchmarking.md`. The verbatim auto-generated captions are kept at
`raw/transcripts/original/12-benchmarking.md`. Lecturer is Yann Dubois. Student questions
and comments from the floor are set in *italics*. Timestamps mark the start of each
paragraph; all 110 are preserved in order.

**This is an edited transcript.** The captions had no punctuation and mangled almost every
benchmark and metric name in the lecture: *MMLU* arrived as "mlu", "MML U" and "mmu";
*SuperGLUE* as "super clue"; *BoolQ* as "bull Q"; *BLEU* as "blue" and "blur"; *ROUGE* as
"rou" and "RK"; *BERT* and *BERTScore* as "bird", "bir" and "Bird score"; *BLEURT* as
"blurt"; *n-gram* as "engram" and bigrams as "Iams" and "bams"; *AlpacaEval* as "Paka
eval", "app Eva", "alaka" and "back a EV"; *AlpacaFarm* as "alpaca form"; *MT-Bench* as
"empty bench"; *Chatbot Arena* as "chatboard Arena" and "chadbad Arena"; *ChatGPT* and
*GPT-4* as "Chad GPT", "chpt", "gp4", "GPD for" and "GT4"; *Mistral* as "Moll" and "mol";
*Claude* as "cloud" and "Claud"; *Penn Treebank* as "pentry bank"; *CoNLL* as "caral";
*scikit-learn* as "es learn"; *ROC/AUC curves* as "Rock curves Au curves"; *Codeforces* as
"code Force"; *Phi-1.5* as "F 1.5"; *DynaBench* as "Dynam Ben"; *MLPerf* as "ml puff";
*DiscrimEval* as "dis remal"; *Anthropic* as "entropic"; *hyperparameter tuning* as "highi
tuning", "High Prim tuning", "high PR tuning" and "Hy prar tuning"; *spurious correlations*
as "superious", "Pur", "SP" and "spus"; *Horace He* as "harass he"; *Zygote* as "Z zigot";
and the lecturer's own name as "Yan". Terms and citations were restored from context and
checked against the slides. **No content was added, removed, or reordered.**

**Where the source is still unreliable**, the text carries an inline `[Ed:` note rather
than a silent guess. There are six, at 37:04, 48:37, 59:18, 1:02:20, 1:06:13 and 1:16:58 —
five garbled student questions or interjections, and one phrase in the lecturer's own
account of the OpinionQA result.

**Every number was compared against the original.** Three readings were corrected against
the slides rather than kept as heard, and are flagged here: at 24:43 the captions give the
BLEU score for "Heck no!" as "7%", but slide 18 scores it **0.67**, the same as "Yes!",
which is the whole point of the false-positive example, so it reads 67%; at 1:06:13 the
captions give llama-65b's MMLU under HELM as "63 3.7", which slide 51 records as **0.637**,
so it reads 63.7; and at 7:00 "90 is% accuracy" is 90%. Two further differences are
spellings, not values: 57:01 "type 1 a supernova" is *type-Ia*, and 13:59 "Precision recall
and Fon call" is *precision, recall and F1*. Every other number is as spoken, including the
ones the lecturer rounds off the slide (69.4% read as "70%", 38.8% as "40%", the
human-variance line at "around 31 or 33").

---

**[0:05]** Great. So I think let's get started, because we have a lot to cover today. My
name is Yann. For those who don't know me, I'm a third-year PhD student advised by Tatsu
and Percy, and today I'll be talking about benchmarking and evaluations. So benchmarking
and evaluations are honestly something that I think not enough people look at in academia,
but if you really want to put something in production and you really care about, let's say,
real-world machine learning, evaluation is really key. So let's talk about that. So,
overview of what we'll talk about. First is different reasons for measuring performance;
then I'll talk about text classification and how you measure performance there; then text
generation and how you measure performance there; and finally, how

**[0:52]** do you evaluate current large language models, and some issues and challenges
with the ways that we actually perform evaluations. Okay. So my mental model of how you
actually develop a machine learning model is that first you will be training your model, so
here measuring performance is really key, because you need to have a loss that you need to
know basically how to optimize. Then, once you are optimizing your loss, the second step is
basically development — so usually this is hyperparameter tuning, or for example if you have
early stopping during your models: if you see that your model is not performing that well
you might, or that there's some overfitting happening,

**[1:37]** you might decide to stop, or you might decide to change the learning rate during
the training of your model. So development is kind of the second step, and here you need to
measure performance because you need to know how to actually do hyperparameter tuning and
changing hyperparameters. Then the third step is essentially model selection: so if I have a
task that I really care about, which model performs best for my task? That might be a model
that I have trained, it might be a model that another group has trained. And finally, at
least in the real world, you would decide to deploy your model, and here measuring
performance is really key because you need to know whether your model is good enough to put
in production. In the parallel universe that we live in there's also publishing, so you

**[2:24]** basically need to evaluate a model on standard benchmarks, and the reason why we do
that is essentially for communicating to different groups the quality of our model. So at
every step of this pipeline you really need to measure performance, and that's what we'll
talk about today. But what is key to understand is that at different steps you need to
measure performance in different ways. So there's really not a single ideal way of measuring
performance. For example, on the left, when you train your model, for evaluating performance
you really need to have a way of measuring performance that is super fast, super cheap and
differentiable — because usually, I mean, with neural networks you basically backpropagate
through the loss, so it needs to be differentiable. And finally you really cannot have a way

**[3:11]** for your model to optimize some shortcuts, to optimize the loss even though it's not
really what you wanted to optimize. And as you move more to the right, basically you will
measure performance less often, so it's fine if it's more expensive, but you really need your
evaluation metrics to be higher quality, because the issues if you put a model in production
are higher. So during the development stage you need a way of measuring performance that is
fast, cheap, and also kind of avoiding shortcuts, because when you do hyperparameter tuning
you're essentially also optimizing over a certain objective. Model selection can be a

**[3:56]** little bit less fast and less cheap, but still you will have to do it many, many
times. And most importantly, when you deploy a model you really want the way to evaluate
performance to be trustworthy, because once you put something in production there's kind of
no way to go back for what happened during that time when it was in production. You also want
things to be very task-specific: so if I care about a certain task when I put my model in
production, you really need to evaluate on that specific task, I don't care about other
tasks. And finally you need your metrics to be absolute. The reason why I'm highlighting that
is that in the three other steps you really just care about comparing between things, and that
is very different from if you want to have a threshold which says, if I have less than 95%
accuracy I'm not putting my model in

**[4:42]** production. Okay. And now let's talk about publishing. This is a little bit different
from honest evaluation in the real world. But when you basically do academic benchmarking, and
when you evaluate your models on academic benchmarks, you want the benchmark to be reproducible
and standardized. And the reason why is basically because for the next five or six or 10 years
everyone will be evaluated on that one benchmark, and you want papers in three years to be
comparable to yours. So it's really important that your evaluations are reproducible. Honestly
you don't really care about that in the real world. You also want things to be easy to work
with, because researchers usually are a little bit — they don't want to do additional work that
they need to — and also they usually don't have that much resource, so it needs to be fast and

**[5:28]** cheap. And finally, one thing which I really want to highlight is that for the academic
benchmarks that we usually have, it's fine if the metrics that we use are not perfect, because
really what matters is that over 10 years the direction that your metric is showing you to go
into — basically how the field is moving. Really, if the metric is saying that it's better over
10 years, that in reality the field has made some progress. So at a meta level it's fine if you
use crude metrics in academia. And also you kind of need to balance between difficulty and
simplicity, and what I mean by that is

**[6:14]** that if your benchmark is way too complicated then basically all methods will have
essentially random performance, so no one will use your benchmark; and if your benchmark is too
simple then the baseline will be so good that no one will use your benchmark because no one can
beat the baseline. This is really something that is specific to academia — in the real world
you're not going to be able to change the task that you're performing based on how good your
model is. So that's why I kind of just want to highlight that, because usually people talk about
evaluations, but there's really different ways of evaluating and different reasons why we
evaluate. Does that all make sense? Also, feel free to ask questions. Great. Okay, so benchmarks
in

**[7:00]** academia — this is really the way we drive the field. So this is the MMLU benchmark. I
think Archit briefly mentioned it, but I'll talk about it later again. So this is the most
standard benchmark right now, and you basically see that in the last 4-ish years it has gone
from 25% accuracy — which is essentially random, because it's multiple choice and there are four
choices — to around 90% accuracy. So yeah, benchmarking is really what drives the progress in the
field. And again, what I meant here is that it's not really the differences between small points
that matter, at least in academia. You have to take a step back and you have to think: what
matters is how your models will perform over 10

**[7:45]** years, and making sure that the model on the top right here is better than the model on
the bottom left, even if the benchmark is not perfect — and I think MMLU is a pretty good one in
that sense. Okay. So there are two main types, at least classically, of tasks in NLP.
Close-ended tasks — so I'll talk about it later, but essentially you can think about
classification, where you know exactly the correct label for the task that you're performing. So
here this is the IMDB data set where you're asked to say whether a sentence has positive
sentiment or negative sentiment. So the text is "Read the book, forget the movie!" — so this is
about sentiment classification of the

**[8:31]** movie, so here it's basically negative. And then there's open-ended evaluation. So
think about ChatGPT: how do you evaluate something like that, where really there's no correct
answer, and there are many possible correct answers and they all have different qualities? So
we're going to distinguish between those two. So, close-ended evaluation. As I just said,
close-ended tasks — there's a limited number of potential answers; think, like, less than 10 —
and often there's just one or maybe a few possible correct answers. So this really is standard
machine learning. If you think

**[9:16]** about standard classification you can just do accuracy, you can look at your precision,
your recall. There's nothing special here about NLP. That is not to say that it's simple, it's
just that there's nothing special about NLP here. So, some close-ended tasks I already told you
about: sentiment analysis, so usually this is a binary classification task where you just have to
say whether the sentiment is positive or whether it's negative. Another task is entailment. Also
for sentiment analysis, the typical benchmarks — I always put it next to the task — are IMDB and
SST from Stanford. Entailment is SNLI, also from Stanford, where basically you have some text —
so here, "A soccer game with multiple males playing" — and then you have a hypothesis, "Some men
are playing sport",

**[10:03]** and you have to say whether the hypothesis is implied, or entailed, by the text. So
here it is. Other tasks: part of speech, typical benchmark Penn Treebank; and named entity
recognition, which is a CoNLL benchmark. A few other tasks — you don't need to know all of them,
but just to give you a brief overview. Coreference resolution: so this is actually a pretty
challenging NLP task where you have to say which pronoun refers to what noun. So you have here
the sentence, "Mark told Pete many lies about himself, which Pete included in his book. He should
have been more truthful." And now you have to say, what does "he" refer to,

**[10:51]** whether "he" refers to Pete. And then there's question answering, where you basically
have a long text and the test asks a question and you're supposed to provide an answer based on
the text that you have before. So those are some examples of close-ended tasks. And again, the key
here is that the way we evaluate those is just standard machine learning: you can look at
accuracy, precision, recall, F1 score. Hopefully you all know about these types of metrics, but if
you don't you should look at Chris Potts's class — I think it's CS224U — but his lecture is online
and it's actually really good, on different metrics. So the ways that people evaluate some of these
benchmarks is

**[11:38]** usually by looking at many of them concurrently. So the most common, I would say, super
or multitask benchmark is called SuperGLUE. So here you see on the columns you have all the
different tasks in SuperGLUE — so I think there are eight or nine — and then you really just look
at the average performance in each of these benchmarks and you get a ranking on that, and that is
kind of an attempt to measure general language capabilities. This is what people used to do, I
would say, until maybe two years ago. I will tell you about what people do now around the end of
the lecture. But yeah, SuperGLUE is definitely something you should at least be aware

**[12:25]** of. And examples of tasks that are in SuperGLUE: one is BoolQ, which is simply, you have
some text, you have some question, and you have to say whether the answer is yes or whether it's
no. So that's very easy to evaluate, you just look at accuracies or precision/recall. Entailment we
already talked about. And then the other ones, like coreference resolution, which we also talked
about, and meaning of words, which is something where you have two sentences with the same words
and you have to say whether they actually mean the same thing in this sentence. For example, "bank"
could mean bank like water and bank like money, and you have to say whether in these two sentences
they refer to the same concept. And there are some question answering tasks too. So this is about

**[13:12]** SuperGLUE — are there any questions? No? Cool. So again, although I said many times that
this is essentially just classical machine learning, I want to emphasize that it doesn't mean that
it's simple, and you really have to think carefully about what you do when you use those types of
close-ended tasks. In particular, you're going to have to choose whether you look at accuracies,
precision, recall, F1 score, ROC curves, AUC curves. If you don't know these names you should really
check out the scikit-learn documentation or the lecture from Chris Potts that I linked above, both
of which are really good. But depending on which metric you will choose, you will decide on very
different types of algorithms. And the usual example is

**[13:59]** that if, let's say, you look at spam — you want to do classification of whether an email
is spam or not. Most emails are not spam, thankfully, at least I hope. So let's say that 90% of
emails were actually not spam and only 10% of them are spam. If you look at the accuracy, then just
a random classifier that predicts the most likely label will get 90% accuracy, and that seems — if
you don't really know about your data set, 90% accuracy seems good, but in reality here it means
that you're not classifying anything. So that's why you want to look at precision, recall and F1.
Anyway, I will not talk too much about that, because again this is not specific to NLP, but it
doesn't mean it's easy. Another issue is that once

**[14:45]** you have multiple different tasks there's a question of how do you aggregate these
metrics. So right before, I told you, oh, you just take the average between all of these things.
This honestly is a really terrible thing to do, but that's actually what people do. But these
columns actually mean very different things: some of them are accuracies, others are F1 score,
others are correlations, and you just average everything. I can't remember which benchmark, but I
remember a few years ago there was one benchmark where actually one of the columns was — basically
you had better performance if the value was lower, and you still took an average of these things,
until someone realized, like, maybe we should put a minus there. So yeah, be careful, and

**[15:31]** don't always think that what people do in academia is correct. You should think a little
bit about that. Then there are some other questions I want you to think about. Where do those labels
come from? I said that it is usually a real answer, but how you actually get those labels is unclear.
So I will tell you about some issues in the next slide. And also related to that, there might be some
spurious correlations, and that's what we're going to talk about right now. So we already talked
about SNLI, so, entailment. So here you have again your premise, "The economy could be still better",
and the hypothesis, "The economy has never been better", and you have to say whether the

**[16:17]** hypothesis is implied by the premise. And what this paper from 2019 found is that actually
all the different models were performing really well, but if you just classified based on the
hypothesis you could also perform really well. So even if you did not look at the premise — which
seems like something that you need to take into account, because it's part of the task — you could
perform well. And the reason why is because they realized that when the humans actually wrote the
hypothesis, they will ask, oh, write a hypothesis which is not entailed by the premise, and how
humans usually do that is by adding a negation. So if you only look at the hypothesis and you see a
negation, it's very likely that it's not entailed by the premise.

**[17:03]** So again, even though this is standard machine learning, be really careful about what
metric you use and where the labels come from, and don't just use what people do thinking that if
there was an issue people would have realized. So yeah, so that is spurious correlations. Any
questions on close-ended tasks? Cool. Okay. Open-ended evaluations. I'm going to mostly talk about
that, because this is what is specific to NLP. So an open-ended task is essentially the opposite of
the close-ended task, which is to say that there are many possible correct answers and you cannot
enumerate

**[17:50]** all of them. So you really can't use standard machine learning metrics. And even more
than the fact that you cannot enumerate all the possible answers, usually there are different levels
of correctness. So if I ask you to write a book, or if I ask ChatGPT to write a book, it might be a
decent book, but it might be a better book that it could have written or that another model could
write. So it's not just right and wrong, it's a continuum. So, standard examples for open-ended
tasks. The two most common ones are summarization — so, summarization: you have a long piece of text
and you just ask to summarize it in less than X characters. The standard

**[18:36]** benchmark is the CNN/DailyMail benchmark. The way they actually collected that data set is
that they took a lot of CNN articles, and you know at the top of CNN articles you have bullet points
that kind of say what are the most important things in the article, so they use this as essentially
the gold summary. So this is the classic one for summarization. For translation you basically have
sentences in two different languages and you have to translate from one to the other. So those are
the classical ones. The way that people currently do it — I would say the most standard task right
now is instruction following. Instruction following is kind of the mother of all tasks, in the sense
that you

**[19:22]** can view any previous task as just a chatbot, or some question that you ask to ChatGPT.
You can think classification — I could just ask ChatGPT to do that; you can think summarization — I
could ask ChatGPT to do that. So essentially you could just view a chatbot as the most general type
of task, and you can ask it to perform any possible task and it should just provide the answer for
that task. So this is what we call instruction following. So as you might think, evaluation is very
hard in that domain, and that's what we'll talk about later: how do you evaluate something like
ChatGPT. Okay, so, types of evaluation methods for text generation, or open-ended tasks. The
classical ones are content overlap metrics, which I'll talk about — so

**[20:08]** that's really comparing just the words between a reference answer, a gold answer that
humans wrote, and the actual generation that you got from your model. Then there are model-based
metrics, where you basically turn evaluation into machine learning: you train a model to basically
become an evaluator. And then there's human evaluation, which is usually seen as the gold standard
for open-ended tasks. So, content overlap metrics. As I just said, this is really just comparing word
by word, or group of words, between the generated sequence and some reference. So here I have the
generated sequence being "The woman went

**[20:55]** to the hardware store" and the gold reference, which is the reference written by humans —
I actually don't even know what the task is, but the reference here is "They walked to the grocery
store". And then what you do is that you just compare the two different sentences by looking at the
lexical similarity between those two texts. And this is super fast and efficient, and the way you
usually do that is by using n-gram overlap metrics. So what I mean by this is that the simplest
possible thing is just to say, for every word in the generated sequence, whether it appears in the
reference sequence, and if it does then you kind of increment your performance. So n-grams is
essentially the same thing, but instead of looking at a single word you basically look at bigrams,

**[21:40]** trigrams, and kind of multiple words next to one another. So the usual overlap metrics,
the most common ones, are BLEU and ROUGE. "Bleu" means blue and "rouge" means red — that's not what
they stand for though, and I always forget what they stand for. But basically what BLEU is, is that
it's an n-gram overlap metric that tries to look at precision, while ROUGE is what looks at the
recall. So as I alluded to before, even if you turn everything into a kind of sentence
classification, you have to think about whether you care about precision or recall. So those metrics
are not ideal,

**[22:26]** but until, I would say, two years ago they were the gold standard for translation and
summarization. For translation people use BLEU, because you really — let's say I'm translating from
French to English. I want to look at the generated sequence in English and the actual reference
sequence in English, and I want to know how many of the bigrams that I generated appear in the
reference sequence. There's one additional thing, which is that they don't only look at precision,
because you could get a very high precision by actually predicting something very small. For example
if you always predicted the word "the", if you only

**[23:12]** generated the word "the", you would most likely get very high precision, because "the"
usually appears in every sentence — or like a full stop. So there's also some length penalty. And
ROUGE is kind of the opposite, it just looks at recall. So those are the common content overlap
metrics. And just to illustrate why those are not ideal — well, they have many issues, but one of
them is that they don't really take into account the semantic relatedness between words. So imagine
that Chris asks you, "Are you enjoying the CS224N lectures?" Of course the gold answer is "Heck
yes!" So that's the

**[23:58]** reference answer. So now let's say that the model just generates "Yes!" So here what
you're going to have is, if I look at the BLEU score, I will have 67% essentially BLEU score,
because two of the unigrams I generated are in the gold reference. If I generate "You know it!"
then I will only have a single token in the generated sequence that appears in the reference
sequence, which is the exclamation point, so I get a much lower BLEU score. And if I just say "Yup."
then that doesn't appear at all in the reference sequence, so I get zero BLEU score, which is a
false negative, because it literally means the

**[24:43]** same thing as "Heck yes!" So hopefully you see that these metrics really have issues.
Also, you can have false positives: for example if you say "Heck no!" then most of the words are
the same so you get 67% BLEU score, but it really means something completely different. Does that
make sense? Any questions? Cool. So, very naturally, now that you know everything about embeddings,
what you might ask is: oh, why do we look at words, if what we could do is look at learned
representations which really kind of maintain the semantic similarity between words?

**[25:28]** So this is exactly what people have done around 2019, I think — or even before actually,
2016. They took some word embeddings, they associated every word in the reference sequence to a
word embedding, every word in the generated sequence to the corresponding word embedding, and they
basically started comparing the word embeddings. So a very simple way of comparing word embeddings
is: you just take the average between the word embeddings in the reference sequence and the average
between the word embeddings in the generated sequence, and you maybe look at cosine similarity. I
mean, there are more ways of doing it, but honestly at this point it's not that important. So you
can think about averaging. Another one — as you

**[26:17]** know at this point, word embeddings don't really take into account the context of where
the word basically appears, so a better way of getting good representations for a word is by looking
essentially at BERT. So what you could do is, you could take a BERT model, you could pass the
generated sequence through it, you get some embeddings, and then you can take BERT again — the same
BERT — you pass the reference sequence to it, you get some other embeddings, and then you do again
some comparison. I mean, this BERTScore, pretty famous paper, they do some smart comparison, but
it's not that important to understand what exactly they do. What is important is that they take some
smart averaging

**[27:03]** between those words. Cool. Any questions? Okay. So that was the simplest type of learning
method, which is word matching. Another slightly more complicated one is called BLEURT, also pretty
famous, which is a mix between BLEU and BERT. So the way that they did that is that basically they
took a pretrained BERT, and then they do some continual pretraining by trying to predict the BLEU
score and some other metrics, and then they finetune — that's the important part, that they finetune
the pretrained model to actually do the evaluation that they

**[27:50]** care about. So let's say that I have a lot of different sequences and I have some human
annotations of how I should be evaluating it. I could just treat that as a normal machine learning
task and I just finetune my BERT to do the evaluation. So this is BLEURT. Any questions? Yes.
*Curious — if you pretrain on BLEU, wouldn't it cause the same problems? If your pretraining task is
BLEU, then would it learn the ability to model languages semantically in the first place?* Yeah,
that's a very good point. So actually I also find it kind of surprising. So they did two things:
first they do the real pretraining of BERT, and then they do continual pretraining for predicting
BLEU. And

**[28:38]** the reason why is because usually they say, we have a lot of sequences in our data set
that are unlabelled, so we have some reference sequences and some generated sequences but we don't
have the human annotation of whether this is good or bad, so we will treat that as an unsupervised
learning objective. So what do you use for the supervised learning objective? Well, you have to use
something, and they basically use BLEU, and they also use BERTScore, so they use many different
tasks and they basically do multitask learning. Cool. Okay. So one important issue with all these
methods is that really they can only be as good as the references are, and in reality the

**[29:24]** references are usually not that good. So this is a paper that looks at summarization of
news. So basically, as I said before, most of the news summarization benchmarks usually take the
reference summary as being the bullet points that you find at the top of an article, and this is
usually not that good. So here what you see on the left, this is if you look at the correlation
between the x-axis being the human-evaluated performance of every model, and on the y-axis you see
the ROUGE-L, which is just a variant of ROUGE, and

**[30:09]** you look at whether these two are correlated. And what you see is that it's essentially
not correlated, which means that ROUGE-L on standard references is really not correlated to what
humans would say is a good summary. That is not to say that ROUGE is a bad score, that is to say
that actually the references are bad — because if you look at the exact same thing, but now you ask
experts to write very good summaries, then you see that the correlation actually increases by a
decent amount. Still not perfect, ROUGE is definitely not perfect, but at least it's much better. So
this is to say that the metric itself is not always perfect, but not only this, the references are
usually actually not great. Cool. So that begs a very natural

**[30:58]** question, which is, can we just move away from reference-based evaluation? So as we just
said, reference-based evaluations are the ones that compare human-written references to some model
outputs using different types of metrics, and those used to be the standard benchmark for evaluating
NLP tasks, I would say up to like 2 or 3 years ago. Right now I think papers still have to
always show the BLEU scores, like for example in translation, because reviewers want those. But I
don't think anyone in the real world actually uses them — but I might be wrong

**[31:43]** on that. So yeah, so BLEU, ROUGE, BERTScore. Oh, and I was mostly talking about BLEU and
ROUGE — BERTScore is actually still decently used and actually pretty good. Okay, so, reference-free
evaluation. Reference-free evaluation is basically, you have a model and you ask it to give a score,
but there are no human references. So the way that this used to be done is essentially by taking a
model like BERT again, but instead of comparing between a reference answer and the generated answer
you could just ask it to take the input and just predict the score. That's one simple way of doing
it. That used to really not work well — and I say "used to" because until basically ChatGPT and
GPT-4. Now what

**[32:28]** people do, and honestly that works super well, is that you just ask GPT-4 to do the same
task as you would ask a human. So you give a very long text and then you give the generated summary
and you ask, like, how good is it, essentially. And that works surprisingly well. So common
benchmarks here: AlpacaEval and MT-Bench. There are many others now; honestly most people start using
these types of techniques, but we'll be talking at least about AlpacaEval. Good. Okay, so let's talk
a little bit about human evaluation before looping back to GPT-4. So as we saw, the metrics until now
all have some shortcomings, and they're definitely not

**[33:14]** as good as if you ask directly for human evaluation, because they are based on references.
So human evaluation is really the gold standard for open-ended tasks. And not only is it really the
gold standard for evaluation, it's also the gold standard for developing new automatic evaluations.
So every time you develop a new automatic evaluation you will want to compare to what humans would
have basically predicted. Yeah. Okay. So, doing human evaluation. First it might seem very simple:
you basically ask humans to evaluate the

**[34:01]** quality of some generated text. Seems simple, right? But actually it's super complicated
and it's a real challenge and it has many issues. So first — oh sorry, I'll talk about that before.
Maybe one additional thing is that you should not only ask the human, you usually also ask them to
evaluate across different axes: for example the fluency of the text, or the coherence of the text, or
common sense, or the style, grammaticality, redundancy, and different axes that you might care about.
Another thing to note is that you should absolutely never compare different human evaluations. So if
there's one paper that says, oh, humans have evaluated the fluency of our text to be, I don't know,
four out of five,

**[34:47]** and then another paper that says like three out of five — they use different humans,
different ways of prompting the humans, so it's absolutely not comparable. Okay, so let's go back to
some of the issues. So as I said, human judgments are regarded as the gold standard, but it definitely
has issues. First, it's super slow — as you might expect, humans are definitely not as fast as
automatic metrics. Second, at least in academia, it's still pretty expensive to do, because I mean,
when you pay well your workers it's pretty expensive to do human evaluation well. Another part is
inter-annotator disagreement. So if I take two random

**[35:33]** people in this room and I ask them to evaluate the quality of a generated text, I can
assure you that you will really not agree. So this is bad, especially if it's subjective, but even if
you talk for like one hour beforehand about how you should be evaluating generations, I can most
likely guarantee you that you will still disagree on many of the evaluations. And to give you an
example: when we were doing AlpacaFarm last year — which is something where we basically had to take
some inputs and then take two models, think ChatGPT, Alpaca and these types of models, and you just
have the two models

**[36:19]** predict an answer and then you ask the humans to say which answer they prefer. This is a
very simple task, and this is — but I will talk about it later — this is what a lot of people basically
use right now for evaluating models like ChatGPT. So a natural question is whether humans are good at
doing that. And what we saw is that — so we were five researchers doing that, and the five of us we
talked for like two or three hours, we wrote extremely detailed rubrics about how to do these
evaluations, and still we only agreed 67% of the time. So 50% is random, and if we just label things
independently we only agree 67% of the time. And we really tried to do our best, we were working on
this thing,

**[37:04]** so it's not as if we were trying to do it quickly. So really, people disagree. Of course, if
you then allow discussions between the annotators then agreement actually improves, but then it
becomes even slower and more expensive. Intra-annotator disagreement — this is something that is
extremely annoying, which is that if I ask a human, if I ask myself right now to evaluate something,
or in three hours, after I have dinner or after I went for a run, I will actually give different
annotations. *[Ed: a student asks a question here that the captions do not recover; from the answer
it is about how you validate or check annotator agreement.]* Yeah, so this is a very good question.
Honestly there's no good answer.

**[37:50]** The usual way that people do it is that you look at some statistical metrics, basically,
where you're like, okay, I want to compare between these two models, I'm going to basically perform a
t-test and I want to know that my p-value is less than a certain amount. What people usually do also
when they have human annotations — I unfortunately didn't put a slide on that — but they have a metric
for computing the inter-annotator agreement, and they try to achieve a certain inter-annotator
agreement, and if not they will essentially ask for more humans or for relabellings. Yeah. It's not
reproducible, and this is partly because of the two things that we said before, but also partly

**[38:35]** because — yeah, I mean, mostly because of the two things before. So this is an interesting
paper. I forgot which year, I think it's from 2021 but I'm not sure, where basically they say — and I
read from the abstract here — "just 5% of human evaluations are repeatable in the sense that there are
no prohibitive barriers to repetition, and sufficient information about experimental design is publicly
available for rerunning them". So this is a paper that analysed I think 128 different papers that were
published across like five years, I think between 2015 and 2020, and they found that essentially only
5% of those papers were reproducible. So honestly, working with humans is hard. That's definitely
something to

**[39:21]** remember. Another part is that humans only basically evaluate precision and not recall. So
what I mean by that is that if you show me what the model generated, I can only evaluate that
generation, I cannot evaluate all the other possible generations I could have generated, because then
you really have to sample a lot of things and that will become way too slow and way too expensive. And
finally, usually the incentives are not aligned. So what you want is for the humans to basically do the
best possible evaluations, but what crowd workers usually want is basically to maximize the amount of
money that they get paid per hour. So to give you again a concrete example: when we were doing
AlpacaFarm, I think we were paying

**[40:10]** relatively well, in the sense that we were paying 1.5 times the minimum wage in California,
and then we basically looked at how much time we would spend to evaluate a single example the best we
could, and then we divided by that time to basically know how much we would pay for every example. And
what we realized is that they ended up being paid, I think, two or 2.5 times the minimum wage, because
they were just doing things like two, three times faster than us. And I mean, we could be slow, but I
think what was happening is that they were just trying to maximize the dollars that they were getting
per hour, and as a result they were finding shortcuts for doing their evaluations. And this is

**[40:56]** something that you really see in the papers. For example in our case you saw that humans
really preferred longer answers — and of course, if you give me two very long generations and you ask
me with a minimal amount of work to say which one is better, if I see a longer one I'm like, probably
there are more details, probably it's better. Anyway, it's not to say that everyone is like that, but
definitely the incentives are not aligned, so you have to be careful of this. Other challenges. First,
you have to decide how to describe the task — you really have to give very detailed rubrics for how the
humans have to evaluate the task. Then there's a question of how do you show the task to the humans:
for example, the order in which you give examples is actually really important. In our case, because we

**[41:41]** had two examples side by side, which one is on the left and which one is on the right is
actually also very important. So all these things really matter. Of course you can randomize these
things away, but it adds challenges. What metrics to use — I mean, this is not specific to humans.
Selecting the annotators, this is also very complicated. You might think, okay, I have some money now,
I can go on Amazon Mechanical Turk and I can just ask them to do some annotations. But in reality you
want to have the good annotators. So how it usually works on Amazon Mechanical Turk is that basically
you say, oh, here's a task, I want like 30 different people to do these annotations, and then they
start annotating, and then if they don't achieve the level that you want you

**[42:27]** basically pay for what they annotated until then and you work with someone else afterwards.
So then there's a question of how do you decide whether they achieved the performance that you want.
So you probably have to do some gold labelling before, and then look at accuracies of how well, and
some inter-annotator agreement with you and with the other researchers on your team. So it is very
complicated. And not only this, you have to monitor that over time. So there are many different ways
you can monitor that over time: looking again at the accuracy, so a typical thing is that every batch
of examples that you label, you give a few examples that are actually ones that you already know what
the gold label is, and you see how well they're performing on that. Another way to look at it is the
time that

**[43:12]** they take to annotate. Yeah. Okay, so that was about humans. So human evaluation is hard,
but it is a gold standard. Okay, now let's talk about reference-free evaluation and chatbots. So I
already told you about it before, very briefly: how do you evaluate something like ChatGPT? This is
extremely complicated, because basically you could ask it any task you want and it can answer text
that is arbitrarily long, and that just makes evaluation extremely hard. So as I suggested before,
the usual way that it's done is that you take two models, you put them side by side, you ask the same
question, and you just ask either some humans or some model, as we will see afterwards, which one is
better.

**[43:58]** So this is the most common benchmark right now, I would say, for human evaluation. It's
called Chatbot Arena, where basically anyone can go online and just play for free with some of the
best models out there, and all they ask you is to say whether you prefer the one on the right or
whether you prefer the one on the left, essentially. And then once they reach — I think a crazy amount
of data, 200,000 human votes for example — they basically add it to a leaderboard. And the way they add
it to the leaderboard is that, I don't know if you know how chess works, but they basically look at
the Elo ratings. So they basically put everything as if it was a tournament, such that not every model
has to play against every other model, and then they get Elo

**[44:46]** scores. Okay, so what's missing with this side-by-side human eval? As I said, this is really
the gold standard for evaluation of chat LLMs, but there are still some challenges. First, it's
basically random people online that ask random questions and they provide their preferences, so that
may not be representative. Although arguably, when you have that many examples it becomes actually
pretty representative of what people would want. So it's probably better than whatever we have, but it
is still not ideal. And then really the big issue is cost. This takes a huge community effort and a lot
of people to work on that. Also it takes a lot of time to get new models on the benchmark, and

**[45:33]** only the notable models — so think like the OpenAI models and the Claude and the Google ones
and the Facebook ones — are going to be benchmarked. You will never have, for your random model,
200,000 people who are willing to annotate it for free. So this is an issue. And again, as we talked
about in the first slide, even for those big companies they can definitely not do that for development
of their model; this is something that comes at the end, for maybe model selection. Okay, so how do we
make it faster? So one very natural solution is basically to ask a large language model to do the
evaluation for you. So imagine that I want to compare ChatGPT with Mistral — I basically ask GPT-4 to
evaluate which one is better.

**[46:19]** This is surprisingly good, and I will show you some results afterwards. And some common
versions are AlpacaEval and MT-Bench, probably the two most common ones. So when we started doing that
— that's a problem I told you about, we started that around last year — we found that using GPT-4
essentially for evaluation is, at least if you look at the prices now, 100 times faster and 100 times
cheaper than if you use human evaluations. But — and this is very surprising — the agreement with
humans is actually higher than humans agree with themselves. So what I mean by that is, if I ask — so
this is what we found — if I ask four humans, let's say I have a pool of four humans and I take out one

**[47:06]** human and I look at the agreement between that human's preferences and the mode of the
preferences of the three others, and I do that in a leave-one-out fashion and I look at this agreement,
this will be lower than if I ask the model to predict essentially the preference of the mode of the
humans. So in some ways models are more highly correlated with humans than humans themselves, which is
very surprising, and I will tell you about it in two seconds a little bit more. When we did that, we
actually used that for collecting preferences for RLHF — so that's what we call RLAIF, as I think Archit
told you about these things last week. So going back to this

**[47:51]** surprising result that actually models are more highly correlated with humans than humans
themselves: the reason why this is, is because humans actually have high inter-annotator disagreement
and have high variance essentially. Models, they will always be very consistent — or maybe not
perfectly, there's still some stochasticity, but essentially they will always predict the same label,
so they have very little variance. So here what you see on this plot is, on the x-axis we estimated the
variance, and you see that the human has a variance of like around 31 or 33. Well, if you look at the
red point, this is basically if you just ask GPT-4 to do evaluations: so even though the bias is still
pretty high — so bias by definition for humans is zero, for

**[48:37]** GPT-4 it is like around 32% — the variance is much lower than humans. So this is why you can
see that actually sometimes agreement is higher, but that's really because there's no variance, or very
little variance, in LLMs. Yeah, does that make sense? *[Ed: a student's interjection is garbled here;
it reads roughly "…is higher than a human — sorry, it means the internal consistency is higher".]*
Exactly, which is actually a good sign, because that makes it much easier for research. The bad sign is
that the bias is still high. Yeah. Okay, so, things to be careful with when you work — I mean, this is
both with humans and with LLMs. There

**[49:23]** will be some spurious correlations. So we already talked about spurious correlations, but you
will see a lot of those. One very common example is length. So if you just — as I told you before — if
you ask crowd workers which examples they prefer, they are highly biased towards longer outputs. So
here the blue is humans, it's around I think 70% preference for longer outputs, and models are around
the same bias. And another example is preference for lists: so usually if you see lists in an output,
models and humans prefer these examples. Another bias, or spurious correlation, is position — I told you,
like, which one you put on the left, which one do you put on the right when you ask humans to label.
There's the same

**[50:09]** thing with models, but this is usually pretty easy to control for, you just randomize both.
Another issue is GPT-4 self-bias. So very naturally you might wonder, if I ask GPT-4 to evaluate itself,
it will probably be biased, it will prefer itself over other models. And this is true, but less than what
you might think — I will tell you about it later. Okay, so, AlpacaEval — wait, until what time do I have?
*You have 30 minutes.* Oh, thanks, great. Okay. AlpacaEval. So AlpacaEval is the benchmark that we
developed when we were working on Alpaca. So as I told you before, one thing which is very important is
what you use

**[50:54]** for development, so basically for hyperparameter tuning. So what we did is that we basically
did not trust many of the benchmarks out there at this point for instruction following, so we just
developed a very small benchmark for ourselves, and this is what we were doing for hyperparameter
tuning, and then it kind of became its own thing. So, AlpacaEval in a few numbers. It has very high
correlation with Chatbot Arena: so the ranking, if you look at the correlation between the ranking in
Chatbot Arena and in AlpacaEval, it's 98%, so very high. And it takes around 3 minutes and $10 to
evaluate. And the way it works — I think I already mentioned it — but basically you take an instruction,
you generate an output from one model and then from another model that you're

**[51:39]** comparing it to, and you ask GPT-4 to basically give the probability that it prefers the model
that you're evaluating versus the baseline that you're comparing to. And then you do some reweighting,
and the reason why you do some reweighting is because these models, as I said, are very biased towards
longer outputs, so you want to reweight such that if it's a longer output you give it a slightly less
high preference. And then you average across your entire data set and you get a win rate. So that's how
it works. Any questions? Cool. So, system-level correlation. So here what you see on the x-axis is
basically AlpacaEval — I mean, a

**[52:25]** slight transform of it, but essentially AlpacaEval scores — and on the y-axis is this Chatbot
Arena, which is the gold standard, and you see that things are relatively highly correlated. And on the
lower plot you see basically the correlation between different benchmarks and Chatbot Arena, and you see
that MT-Bench and AlpacaEval, which are the two ones that use LLMs for evaluations, are relatively
highly correlated with Chatbot Arena, and MMLU, which is the automated one that doesn't use an LLM, is
also very highly correlated. So I told you very briefly about the fact that we had to do some
reweighting. So I'm not going to tell you how we do it, but I want to tell you why we do it. One of the
issues that we realized a little bit too late is

**[53:11]** that if you take something like GPT-4 and you just prompt it to basically provide much more
detailed answers, its win rate — so its performance on your benchmark — goes from 50% to 64.3. So that's
this one, 64.3. If you ask it to be more concise it decreases to 22.9. And it really doesn't fit our
mental model of what benchmarks should be doing: if I just tweak a little bit the prompt, I don't want
my model to change completely its ranking. So that's why we have to do some reweighting, and you see
that after the reweighting you basically have that the performance after you ask the model to be more
verbose is

**[53:57]** very close to the performance without any prompt tuning. Cool. So I told you very briefly
before about self-bias. I do want to say that I'm pretty surprised about this result, but actually
self-bias exists but is not as high as you might think. So here you see the different models that you're
evaluating — sorry, that's on the rows — and on the columns you see who is evaluating, which model are
you using for evaluation. And you actually see that regardless of the model that you evaluate with, the
ranking will be the same. So even though it's true that if I look

**[54:43]** at Mistral evaluated by Mistral, it gives itself a much higher accuracy, it still prefers
Claude and GPT-4. So it's not as bad as what you may think — it's still bad though. Cool. Okay, so that
leads me to talking about current evaluation of LLMs. So I would say there are three main ways that
people currently evaluate LLMs. The first one is perplexity, which is essentially just looking at
training losses or validation losses. The second one is basically averaging everything, which is actually
surprisingly more common than what you may think. And the third one is this Arena-like, where you
basically have comparisons between models and

**[55:30]** either use humans or use models to do the evaluation. And usually how it works is that
pretrained models — let's say when Llama 4 comes out, or when GPT-5 comes out — they basically mostly
show perplexity and average over everything; and the finetuned models, they usually tend to show average
over everything and Arena-like performance. And the reason why is because for models that are finetuned,
usually the log likelihood that they predict is not calibrated for your data set. So what do I mean by
"everything"? I would say the two most common benchmarks that basically look at

**[56:15]** everything are HELM and the Hugging Face Open LLM Leaderboard. It's really just a collection
of a lot of different automatically evaluated benchmarks, and you evaluate across all of them. So what
are some of the common benchmarks that we use? One is measuring math performance, so GSM8K, that's a
pretty common one, that's basically grade school math. MMLU is multiple-choice question answering on
math, science, history. LegalBench is on the legal aspect. Then you have MedQA — so this, I believe this
is for HELM — MedQA is medical licensing exams. So you basically ask many, many different questions that
you can automatically evaluate, and you hope that by taking

**[57:01]** averages it will say how well your model performs. So that's kind of like the newer version of
SuperGLUE, I would say. One benchmark which is probably the most widely used and the one that people
believe the most is MMLU, so Massive Multitask Language Understanding. So this is — I think maybe Archit
mentioned it last week — but this is basically multiple-choice questions on 57 different tasks. So you
have tasks like formal logic, conceptual physics, econometrics and these types of tasks. So here's an
example: "What is true for a type-Ia supernova? This type occurs in binary systems. This type occurs in
young

**[57:46]** galaxies." And you basically have to say which answer. So that seems very simple — I mean, the
task is not simple, but the way you evaluate seems simple. And then like high school biology: "In a
population of giraffes, an environmental change…" and then, this is an example of directional selection.
So that seems simple, but actually it's also more complicated than what you might think, and I will tell
you about it later. But that's one of the most common, probably the most common benchmark, and what
people actually look at — for example when Mark Zuckerberg said that Llama 3 was out, he talked about
MMLU scores, which I find kind of crazy. But yeah. Other

**[58:33]** capabilities that people look at: coding. Coding is a very common one that people evaluate on,
for two different reasons. One, because if you perform well on code, these models usually actually
perform well on reasoning, which is actually pretty cool, so that's highly correlated with things that
people care about. Two, I mean a lot of us are coders, so we like to have better models for helping us
code. And three, the other point is that it's actually pretty easy to evaluate, because you can write
test cases. So you basically ask the model to generate very long code, or functions, to do something, and
then you just run the test and you see whether it succeeds or not. Yes. *Sorry, going back to the
previous evaluations — some*

**[59:18]** *of them were short answer. How would you evaluate a short-answer QA type of thing? Multiple
choice makes sense, but if it's short-answer QA, how would you say something is correct as an automatic
metric?* *[Ed: the end of the question is garbled.]* Yeah, I actually don't know. Huh. I actually don't
know, yeah, I should check, sorry. So I don't know specifically for this one, but HotpotQA and BeerQA are
other QA data sets, and there they look at F1 for the true and false, and then they also have an exact
match, which is pretty punitive, because if you say "President Reagan"

**[1:00:03]** and the answer is like "President Ronald Reagan", it will ding you. But anyway, so they use
an exact match on that, yeah. Cool, thanks. Okay. Another thing that people are starting to look at is
agents. I think Shikhar is going to give a lecture on it, so I'm not going to talk too much about it. But
one cool thing that LLMs can do right now is basically call APIs and then take actions in the real world
essentially, or take control of your computer. You should not give it control of your computer. So a
natural question is, how do you evaluate these types of things? This is a real challenge, because I mean
the biggest challenge is that if for example I really wanted to evaluate how

**[1:00:48]** good it is at coding, or how good it is at doing things in my terminal, I need to give it
access to my terminal, and I really don't want to give my LLM access to my terminal. So you really need
sandbox environments. For the specific case of a terminal, I mean it's pretty easy to sandbox, but once
you want to do evaluation of a model that pings people on Slack or writes things in your emails, then you
have to write an entire sandbox environment for all the applications that you want your LLMs to have
access to. So this is actually really complicated, and something that people really have to deal with in
the real world — at least we have to, because right now it's still not in production. Okay. The last part
is — or the penultimate one — perplexities. So

**[1:01:34]** one thing which is very surprising, at least the first time you see it, is that really the
performance that you have on pretraining is extremely highly correlated with basically performance on any
downstream task, at least for the current types of LLMs. So what I mean by this is that if you just look
at your training performance, just predicting the next word, it's extremely highly correlated. So this is
the x-axis, which is essentially perplexities, and the y-axis, which is just the average over many
different tasks. What you will see is that models that perform well on perplexities will actually have
high average scores. And as a result a lot of people actually end up, when they develop, just looking at
perplexities, and they just trust it

**[1:02:20]** enough that they don't need to do the downstream evaluations. I would not recommend doing it,
but if you have to have something quick and dirty it usually works pretty well. One thing to be careful
with, though, is that the perplexities are not going to be comparable across different data sets, so you
really have to be careful with what perplexities you're looking at. And two, it will depend on the
tokenizer. So if you have Llama 3 and you compare it to Gemini, even on the same data set it's going to
give different scores and it's not comparable. *[Ed: a student's question here is not recoverable from the
captions.]* Yes. The easy answer — I mean, it's not the only answer, but the easy answer — is that if the
size of the vocabulary changes, then clearly, I mean, everything

**[1:03:07]** is not on the — the upper bound is different. *[Ed: the student's interjection is garbled.]*
Yeah, but I'm not talking about that, I'm talking about the fact that — I mean, just think about it: if
you have a vocabulary size of one, then I have to always predict the same thing. So your entropy is upper
bounded by the log of the cardinality of your vocabulary size, so you're going to depend on that. Cool.
And the last one is Arena-like: as I already told you, basically you compare different models, you make
them fight essentially against each other, and you have Elo ratings at the end. So a more general way of
saying it is, I really just let the users decide, and that works also pretty

**[1:03:54]** well. Okay. Issues and challenges with current evaluations. First, consistency issues. If you
look at multiple-choice questions — so you see on the top left and top right — if you just change A, B, C,
D to random symbols, the generations that you will give are actually going to be different, and then the
rankings between different models will be different. So even things that are very simple, like multiple
choice, like selecting out of four choices, will be very dependent on exactly how you format these
choices. And one real example — that's what I was alluding to before — is MMLU. So MMLU

**[1:04:40]** seems really simple to evaluate: you just ask it to say which one of the four the model
prefers. But actually, for a very long time — I think for nearly one year — there were three main
implementations of MMLU, and people were comparing between those three having no idea that those three
gave different scores. And the two main differences were: one, people use different prompts, so that
clearly will give different answers; but two, they were using different ways of sampling to get the actual
most likely prediction. So one of them for example was saying, I have the four choices, now to get my most
likely — let's say that the correct answer is D — I will just

**[1:05:26]** look at the most likely answer out of A, B, C, D, even though, like, "Zygote" was another
answer that has a higher likelihood. I will not look at it because I will basically do constrained
decoding. And if I do constrained decoding here I will say that the correct answer is D; but if I actually
just look at the most likely token, I will not get the correct answer. So those were two different
implementations. And a third different implementation, which seems really different, is that instead of
generating the correct token — which is basically the letter A, B, C, D — you can look at, after this
question, what is the likelihood that the model would generate this. So you would look at the log
likelihood, or

**[1:06:13]** the perplexity essentially, of predicting that, and that gives very different answers. So if
you look at the top right, you see that llama-65b's MMLU on HELM was 63.7 and the original MMLU 63.6, but
on Harness — which is the thing that Hugging Face actually uses — is 48.8. So that's a huge difference.
*What is HELM, Harness and Original? What do those three things map to?* *[Ed: the question is partly
garbled.]* Yeah — I can't remember which one does what, but each of them does something different. Actually
now it's not true any more, because the middle column changed what they're doing, so they start matching
the other two ones; but at that time they weren't. I'm not sure which one — my guess

**[1:06:58]** would be that they did the last one, but I'm not sure. Okay. Questions? Cool. Another issue:
contamination. So here you have Horace He — if you don't follow him on Twitter you should — and he basically
said that he was looking at code benchmarks and he was saying that pre-2021, I can't remember which model,
or GPT-4, was getting 10 out of 10 on questions on Codeforces, but after 2021, or more recent problems, it
was getting zero out of 10, which seems very, very strange. So that really strongly points to the fact
that it was

**[1:07:44]** contaminated, and probably the model was pretrained on that data set, or the Codeforces data
set was probably in the pretraining data set. And of course, if you essentially do training on your test
set then you're going to perform really well. And Susan said also — also said something similar for
Phi-1.5, which is a model from Microsoft. So what is challenging here is that with closed models — I mean
there are two things actually that are challenging. One is that those are pretrained on so much data that
even if we had access to the data it would be hard to actually know if they were pretrained on your test
set. But two, those are all closed-source models, so you really don't even have access to the data set, so
you have no idea if they were pretrained on that

**[1:08:30]** data. Overfitting issues — that's also relatively related, but could be slightly different. So
here you see how much time it took for standard data sets to achieve, quote-unquote, human-level
performance. And what you see is that on the recent ones, where you really have this pretraining, in less
than like six months you perform at human-level performance. We don't really know if it's because of the
contamination, or if it's simply that a lot of people are basically developing and trying to do
hyperparameter tuning on these test sets. We don't know why, but it's clearly an issue with overfitting.
So how do you alleviate that? One, you can have private test sets.

**[1:09:15]** So there's a paper from I think two weeks ago that presented GSM1K, which is the same thing as
the GSM8K that we saw before, which is the math data set, but tries to basically regenerate or resample
this data set, or recollect this data set, and then they look at how well different models perform on both
GSM1K and GSM8K. And what you see is that at least the open-source models, they perform much worse on the
new data set than on the one that people are able to tune on. This is not true though for Claude and
GPT-4. Another one is DynaBench, or just dynamic test sets. So ideally, every X number of days you would
basically have

**[1:10:01]** new instructions or new inputs to the models, and your data set would basically be dynamic.
That's essentially also what Chatbot Arena does, so that definitely helps. Another way of alleviating
contamination is that you may try to estimate, or to look at, whether the models were actually trained on
your test set. So one very simple way of doing it, which actually works I think relatively well, is just
looking at the probability of different answers, and you will see that if your model is really sure about
a certain answer then probably it was trained on that answer. Another one, which is also really cool, is
looking at the order of your test set. So if a

**[1:10:46]** model was pretrained on the test set, then most likely it thinks that example two comes after
example one. So if you switch example one and example two and you see drops in log likelihoods, then most
likely the model was actually pretrained on that data set. Cool. Any questions here? Okay. So another issue
is that really there's a monoculture of NLP benchmarking. What I mean by this is mostly the fact that we
all just look at English. And this is a paper from 2021, 2022 I think, but they look at ACL 2021, which is
probably the most common conference in

**[1:11:32]** NLP, and they look at the best papers, so the oral papers, and they saw that out of the 461
papers, 70% of them only look at English and 40% of them only look at accuracy, so essentially just
performance. So there are very few papers that look at multilinguality and even efficiency and
interpretability or fairness. And there's a similar paper that analyses another conference in 2008 and it
was essentially the same finding, so unfortunately it doesn't seem to improve over time. The thing is,
there are actually a lot of benchmarks for multilinguality. I just highlight a few here: MEGA,
GlobalBench, XTREME. Those have at

**[1:12:18]** least 30, 40 languages and many, many different tasks. So it's not that we don't have the
benchmarks, it's that there are no incentives, unfortunately, in academia to actually evaluate on those
benchmarks. So if you have the chance, use those benchmarks. Another issue is that really we reduce
everything to a single metric. So I already told you before, the way we aggregate metrics, this is usually
kind of broken in some of these super benchmarks. But also we only look at performance, and in the real
world we really care about computational efficiency too, we also care about biases, and we care about many
other aspects, and most of these benchmarks don't consider those. Another part is that

**[1:13:05]** we usually average across every example. We just say that every example has the same value,
essentially the same weight. So this is definitely unfair for minoritized groups. But more than this, I
think, for example if you think about agents, where maybe one example will be how well it performs on, I
don't know, writing code that will actually be put in production, versus just answering your daily
question about, I don't know, where to buy the best burger — the value that you will get out of these
examples is very different, and right now when we evaluate stuff we don't actually consider that. So
that's, I think, a real issue. And also, we basically don't take into account that different

**[1:13:50]** people have different preferences. So, a few outs. One, considering computational efficiency:
so MLPerf has a great benchmark, where basically instead of trying to maximize the performance on a
certain benchmark, they say I want to achieve that performance in the least amount of time. So now you
basically consider both accuracies and speed, either for training or for inference. For biases,
DiscrimEval is a good data set from Anthropic, where basically they have some templates, and they try to
ask questions like, knowing whether someone should keep their insurance or not, and they have templates
where they change

**[1:14:37]** the race or the gender of the person in the template, and they see how the decisions made by
the model would change. And I mean, unfortunately but unsurprisingly, you will see that some groups are
much more discriminated against than others. Other biases in our evaluation. So I already told you
slightly about the multilingual issues, but honestly this issue about English is much more prevalent than
you would think. For example, BLEU and ROUGE, they really assume that you basically have access to words,
like you know how to tokenize and how to get words. So I used to work with Thai and Vietnamese. With
Vietnamese you have spaces in between words, and Thai you have no spaces between words, like you have no

**[1:15:24]** idea how to run BLEU or ROUGE. Really, it's much more than just the data — all our algorithms
are really focused on English, or at least Western languages. Biased LLM-based evaluations. So one thing
that I told you about is that it's really cool because now you can use essentially GPT-4 for doing
labelling, but that also means that, given that GPT-4 is very consistent, if it has some biases then
essentially most of the NLP community will have these biases scaled up. So one benchmark which tries to
look at whose opinions LLMs reflect by default — this is actually pretty cool work that looks at

**[1:16:10]** the output distribution of LLMs on public opinion surveys, so just trying to understand
whether LLMs reflect opinions from which groups. And they find that at least when you only do pretraining,
the models actually relatively — they are not too optimized to a single group; but after — so this is in
red — but after finetuning you basically see that the models really start being optimized for certain
preferences, which is unsurprising because that's how we actually train the model. And typically these
models actually mostly reflect — actually they answer as if they

**[1:16:58]** were from, I mean, white and Southeast Asian [groups]. So I think the [Ed: the phrase here is
garbled — the captions give "selfie station"] is actually pretty interesting. I think it's probably because
a lot of these models — the human data that was used for supervised finetuning and for RLHF was actually
labelled by people in Southeast Asia, which would explain why these models have these types of views, and
usually also highly educated. Okay, so this is the main challenge, the challenge of all challenges. We saw
that there are many challenges in evaluation, at least in academic benchmarking, but the biggest one is
that really there are no incentives for us to move to anything else. And this is

**[1:17:44]** actually a pretty interesting paper that looks at machine translation, at many papers from
2019 to 2020 in machine translation, and they found that 82% of papers only evaluated BLEU scores. And as
we said, BLEU scores have many, many issues, and we know that there are many better metrics, but still
people are not incentivized to look at anything else — and actually reviewers will usually ask you to show
performance on BLEU scores. So it's not even that you're incentivized not to look at something else, you're
also incentivized to continue. And it kind of makes sense, because you want to be able to compare to
methods from two, three years ago, but it also means that it's hard for the academic field to change

**[1:18:30]** to other benchmarks. But this is really specific to academia — like in reality, if you know
that your metric is bad, just switch. Okay, evaluation takeaways. So first, I mentioned that there were
different types of evaluation and different desired properties for different types of evaluation. Then I
talked about close-ended tasks and how you evaluate those, the fact that it's basically standard machine
learning, but that you have to think carefully, even though it's standard machine learning, about how you
evaluate them. Then there are open-ended tasks, where you look at content overlap metrics typically — so
things like BLEU and ROUGE and BERTScore — and then you have chatbot evaluations, which is

**[1:19:16]** extremely difficult, but people have started doing with essentially LLM-based evaluations. And
then we talked about challenges, one of them being consistency, the other one contamination, and the third
one biases. In reality, honestly, the best evaluation is just check your outputs. So I think too many people
just believe numbers. In reality, never just believe numbers. I remember when we did Alpaca initially, we
kind of believed our AlpacaEval, but once we saw — playing with it — that's when we were like, okay, this
thing is actually — I mean at that time good; now it would be a pretty bad model, but at that time we were
like, okay, this thing is actually pretty good, we

**[1:20:02]** should do something about it, even though on maybe standard academic benchmarks it was pretty
bad. So yeah, don't rely on numbers. And I'm happy to — what time is it — to take any other questions that
you may have. Yes. *Question about — so there's this whole issue of bias which we're really trying to deal
with, but we're sweeping under the rug here. So if we have a problem in which we're dealing with a very
specialized domain, and yes, we try and run reference-free evals using, let's say, GPT-4 — is it considered
bad practice to be checking a*

**[1:20:49]** *subset of these GPT-4 evals, ranking them ourselves, and then inserting ourselves and our bias
into this process by actually looking at many, many data points?* So, just to make sure I understand your
question: you're saying that if we try to look ourselves at the answers we might be incorporating some
biases there. *Yes, but we should look at the answers to make sure that GPT-4 isn't being biased when it
looks at the answers. There's this tension here and I don't know what — because in a controlled scientific
experiment you would blind yourself to looking at these answers. How do you deal with this?* Yeah,

**[1:21:35]** that's a good question. I actually don't quite know. But one thing — I actually feel less
concerned about biases of a single person. My issue with the GPT-4 biases is that it's the same across
every model, so things really scale up and it becomes a monoculture, and I think that's much worse than if
everyone incorporates a little bit of the biases that they have in their direction. I'm not saying that
that's the best answer, but I think it's slightly better than just going with whatever they have. Yeah.
*Following up on that: how does one avoid a situation if one is trying to solve a problem with a model, and
one evaluates it with ChatGPT/GPT-4, and then one starts to*

**[1:22:23]** *look at it and say, okay, is this good, and then one goes, okay, this is great — and everyone
else in the world and GPT-4 thinks it's a terrible, terrible model, and it's just some academic pressuring
themselves into publishing something that doesn't actually work? How does the field structurally avoid
situations like that?* Well, I think that's one reason why they want standardized benchmarks, and why every
reviewer actually wants standardized benchmarks, because at least, even though everyone knows that they're
wrong, they understand how they are wrong. So I think that's one perspective. Another thing, which doesn't
completely answer your question, but which I think could be a potential solution, is that how I view

**[1:23:11]** GPT-4 is just something that is really good at performing what I want it to perform. Right now
the thing is, I'm not very specific about what I want it to perform, and as a result it will basically come
in with its own biases that come from its pretraining data or finetuning data. A potentially better way of
doing it is that I could write exactly what I want. So right now, when we do the prompting to GPT-4, I
basically ask a simple question like, how good is the summary, out of five. But a much better way would
probably be writing a very detailed rubric of everything that has to be in this answer for it to be a good
answer. And if you think about it, this is exactly what professors do when they evaluate for a class: like,
they basically say, okay, Yann is a TA but I

**[1:23:58]** cannot trust him blindly, so what I will do is that I will write a very detailed rubric and I
trust that he can apply that rubric. And I think that's also how we should be thinking about GPT-4, and this
is not how we currently do it. Any other questions?
