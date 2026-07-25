---
title: Lecture 2 — Word Vectors and Language Models
lecture: 2
video: https://www.youtube.com/watch?v=nBor4jfWetQ
source: YouTube auto-captions, copy-edited for readability
verbatim: false
original: original/02-word-vectors-and-language-models.md
slides: ../slides/02-word-vectors-and-language-models.md
---

# Lecture 2 — Word Vectors and Language Models — transcript

Timestamps mark the start of each paragraph and can be cited directly ("Manning covers
this around 42:00"). All 102 paragraph timestamps from the source captions are preserved.

**This is an edited transcript.** The auto-generated captions had no punctuation and
mangled most technical vocabulary — *word2vec* arrived as "word DEC" and "watch ve",
*CBOW* as "sibo", *GloVe* as "glav", *COALS* as "Kohl's", *Doug Rohde* as "Doug roie",
*Pearson* as "piercon", *brioche, baguette, focaccia* as "Brios bagette fatcha". They
have been copy-edited into sentences: punctuation added, false starts and filler removed,
mis-heard terms and names restored from context and checked against the slides. **No
content was added, removed, or reordered.** Student questions, which the captions run
together with Manning's speech, are marked in italics. The verbatim captions are kept at
[`original/02-word-vectors-and-language-models.md`](original/02-word-vectors-and-language-models.md)
for reference — use this file unless you specifically need the raw output.

**Where the source is still unreliable:** the **mini-batch size** Manning quotes aloud at
6:21 came through as "16 or 2" and is not recoverable — slide 6 gives no number either —
so it is left marked as unclear below rather than guessed at.

---

**[0:05]** Okay, I should try and get started. So what we're going to do today is we're
going to try and do everything else that you need to know about word vectors, and start to
learn a teeny bit about neural nets, and then we'll get much further into the math of
neural nets next week. So this is the general plan. I'm going to sort of finish up from
where I was last time with optimization basics, then look a little bit more about word2vec
and word vectors, and then some of the variants of word2vec. And then I'm going to briefly
consider alternatives — sort of like, what can you get from just counting words in
different ways? Then we're

**[0:52]** going to go on and talk a little bit about the evaluation of word vectors, the
topic of word senses that already came up a couple of times last time when people were
asking questions, and then towards the end start to introduce the idea of classification,
doing neural classification, and what neural networks are about, which is something that
they'll then expand on more in the second week. Before I get into that, just notes on
course organization. So remember, the first assignment is already out and it's due before
class next Tuesday. So then our Python review session is going to be taught this

**[1:39]** Friday, 3:30 to 4:20. It's not going to be taught here, it's going to be taught in
Gates B01, the Gates basement. And I encourage everyone again to come to office hours and
help sessions. They've already started, they're listed on the website. We're having these
office hour help sessions in classrooms with multiple TAs, so just turn up if you're on
campus and you can be helped. And if you are on campus we'd like you to just turn up. We do
also have a Zoom option for Stanford Online students. Finally, I have office hours which I
have not yet opened, but I will open sometime tonight. They're going

**[2:24]** to be on Monday afternoons. Now obviously, given the number of people, not
everyone can make it into my office hours, and I'm going to do these by appointment, so
they're 15-minute appointments on Calendly. But I'm very happy to talk to some people. And
I put this little note at the end saying, don't hog the slots. Some people think it'd be a
really good idea if they work out how to sign up every week for an office hour session with
me, and that's a little bit antisocial, so think about that. Okay. So at the end of last
time I did a bad job of trying to write on slides of

**[3:09]** working out the derivatives of word2vec, and hopefully you could read it much more
clearly in the version that appears on the website, where I was doing it at home more
carefully. So that was saying that we had this log function and our job was to work out its
derivatives, which would tell us which direction to go to walk downhill. And so I didn't
really quite finish the loop here. So we have some cost function that we want to minimize,
and then we work out the gradient of that function to work out which direction is downhill.
And then the simplest algorithm is that we work out the direction downhill,

**[3:58]** we walk a little bit in that direction, and then we repeat: we work out the
gradient at this point, we walk downhill a little bit, and we keep on going, and we'll get
to the minimum. And with a one-dimensional function like this it's very simple, we're just
walking downhill. But when we have a function of many, many dimensions, when we calculate
the gradient at different points we might be starting to walk in different directions. And
so that's why we need to do calculus and have gradients. And so this gives us the basic
algorithm of gradient descent. And so under the gradient descent algorithm, what we're
doing is that we've got our loss function J,

**[4:46]** we're working out its gradient, and then we're taking a little multiple of the
gradient, so that alpha is our step size or learning rate. And normally alpha is a very
small number, something like 10⁻³ or 10⁻⁴, or maybe even 10⁻⁵. So we're taking a really
little bit of the gradient and then we're subtracting it from our parameters to get new
parameters. And as we do that we will walk downhill. And the reason why we want to have a
small learning rate is we don't want to walk too fast. So if from here we worked out the
gradient and said it's in this direction, and we just kept on walking, we might end up way
over here. Or if we had a really big step

**[5:33]** size we might even end up at a worse point than we started with. So we want to take
little steps to walk downhill. And so that's the very basic gradient descent algorithm. Now,
the very basic gradient descent algorithm we never use. What we actually use is the next
thing up, which is called stochastic gradient descent. So the problem is, for the basic
gradient descent algorithm we've worked out, for an entire set of data, what the objective
function is and what the slope at the point of evaluation is. And in general we've got a lot
of data on

**[6:21]** which we're computing models. So simply trying to calculate our objective function
over all of our data — the training data for the model — would take us a very, very long
time. So that's very expensive to compute, and so we'd wait a very long time before we make
even a single step of gradient update. So for neural nets what you're always doing is using
this variant that's called stochastic gradient descent. And so for stochastic gradient
descent, what that means is we pick a very small subset of our data — like maybe we pick
[*the number is unclear in the captions*] data items — and we pretend that's all of our data,
and we evaluate the function J based on

**[7:07]** that small subset and work out the gradient based on that small subset. So it's a
noisy, inaccurate estimate of the gradient, and we use that to be the direction in which we
walk. So that's normally referred to also as having mini batches, or mini batch gradient
descent. And in theory working out the gradient based on this small subset is an
approximation, but one of the interesting things in the way things have emerged in neural
network land is it turns out that neural networks actually often work better when you throw
some noise into the system, that having this noise in the system gives you jiggle and

**[7:53]** moves things around. And so actually stochastic gradient descent not only is way,
way faster, but actually works better as a system for optimization of neural networks. Okay.
So if you remember from last time, for word2vec the idea was we started by just saying each
word has a random vector representing it. So we will literally just get random small numbers
and fill up the vectors with those random small numbers. There's an important point there,
which is you do have to initialize your vectors with random small numbers. If you just leave
all the vectors as zero then nothing works. And that's because

**[8:41]** if everything starts off the same you get these false symmetries, which means that
you can't learn. So you always do want to be initializing your vectors with random numbers.
And then we're going to go through each position in the corpus; using our estimates we're
going to try and predict the probability of words in the context, as we talked about last
time. Then that gives us an objective function from which we can then look at our errors,
look at our gradient, and update the vectors so that they learn to predict surrounding words
better. And so the incredible thing is that we can do no more than that and we end up
learning word vectors which

**[9:27]** actually capture quite a lot of the semantics, the meaning and relationships between
different words. So when this was first discovered for these algorithms, it really feels like
magic that you can just do this simple math over a lot of text and actually learn about the
meanings of words — that it's just surprising that something so simple could work. But as
time has gone on, this same recipe has then been used for all kinds of learning about the
behaviour of language from neural networks. So let's just go through

**[10:12]** a sense of how that is. But before we do that, let me just mention: so for our
word2vec algorithms the only parameters of the model are these word vectors. They're the
outside word vectors and the center word vectors, which we actually treat as disjoint, as I
mentioned last time. And when we do the computations we're considering the dot product between
the various possible outside words with our center word, and using those to get a probability
distribution over how likely the model thinks that different outside words were. And then
we're comparing that to the actual outside word in the context, and that gives us

**[10:58]** our source of error. So as such, this is what's referred to in NLP as a **bag of
words** model — that it doesn't actually know about the structure of sentences, or even what's
to the left and what's to the right. It's predicting exactly the same probabilities at each
position, to the left or right. But it's wanting to know about what kind of words appear in the
context of the center word. So I just wanted to stop this for a minute and — let's see, not
that one — give you some kind of a sense that this really

**[11:45]** does work. So this is a little Jupyter notebook that I've got for this. Okay. And so
this is using a package gensim, which we don't continue to use after that really, but it's one
package that lets you load and play with word vectors. And the word vectors I'm going to use
here are GloVe word vectors — and actually, GloVe was a model we built at Stanford and I'm going
to talk about it a little bit later. So strictly speaking these aren't exactly word2vec word
vectors, but they behave in exactly the same way. And so okay, so now it's loaded up my

**[12:32]** word vectors, because the word vectors are a big data file. And so as we've discussed,
for a word — right, the representation of any word, here *bread*, is just a vector of real
numbers. So I'm using 100 dimensional word vectors to keep things quicker for my class demo. So
this is the word *bread*. And then I can say, well, what's the representation for *croissant*?
And this is *croissant*. And we can sort of get a visual sense of, oh, they're at least a little
bit similar, right? The first components are both negative, the second components are both
positive,

**[13:18]** the third components are both negative and large, the fourth components are both
positive. They seem like they're kind of similar vectors. So that seems kind of hopeful, because
that means that it knows that *bread* and *croissant* are a bit similar to each other. This
package has a nice simple function where, rather than doing that by hand, you can just ask it
about all the word vectors and say which ones are most similar. So I can ask it what words in
its vocabulary are most similar to *USA* — and in this model everything's been lowercased, I
should mention. And so if I do that, it has *Canada*, *America*, *U.S.A*, then *United States*,
*Australia*. Well, those

**[14:06]** seem a fairly reasonable list of most similar words, though you might think it's a
little strange that *Canada* wins out over the *USA* with dots over it. Similarly I can ask
what's most similar to *banana*, and I get *coconut*, *mango*, *bananas*, *potato*, *pineapple*,
*fruit*, etc. Again, pretty sensible — a little bit of a bias to more tropical fruits. Or I can
go to *croissant* and ask what's most similar to *croissant*. The most similar things to
*croissant* isn't *bread*, but it's things like *brioche*, *baguette*, *focaccia*, which
basically makes sense — though here's *pudding* here. And I've got — wait, I'd already done —
oh sorry, yeah, I remember what this is. Right, so with this most

**[14:52]** similar, you've got a positive word vector and you're saying what other words are most
similar in position to that. There's something else you can do, which is to say, let me take the
negative of that word vector and say what's most similar to the negative of it. And you could
possibly think that might be useful to find antonyms or something like that. I mean, the truth is
it isn't. If you ask for the things that are most similar to the negative of the *banana* vector
— and in most other vectors it's the same — you get out these weirdo things that you're not
really sure if they're words at all, or maybe they are in some other language, or some of them are
names, like *Shichi* is a Japanese name. But

**[15:38]** not very useful stuff. They don't really feel like a negative of *banana*. But it turns
out that from there we get this powerful ability that was observed for word2vec, which is that we
could isolate semantic components and then put them together in interesting ways. So looking at
this picture, what we could do is start with a positive vector for *king*, from the origin up to
*king*. Then we could use the negation to subtract out the vector for *man*, and then we could
have another positive vector, add on the vector for *woman*, and then we can ask the model

**[16:25]** is, if you're over here in the space, what is the nearest word to you over there? And
so that's what this next thing does. It sort of says, positive vector for *king*, negative for
*man*, also positive for *woman* — where does that get you to? And that gets you to *queen*, yay.
And so this was the most celebrated property that was discovered with these word vectors: that
they weren't only good for meaning similarity, but that they were good for doing these kinds of
meaning components. And these got referred to as **analogies**, because you can think of them as
*a* is to *b* as *c* is to what. So it's sort

**[17:12]** of like, *woman* is to *king* — no sorry — *man* is to *king*, or *king* is to *man*
as *woman* is to what. I'm saying this the wrong way around. *Man* is to *king* as *woman* is to
what, in the analogies. And so here I've defined a little function that is now saying — this
little function just automates that and will compute analogies. And so now I can ask it in just
this analogy format, *man* is to *king* as *woman* is to *queen*. And that one was sort of the
canonical example. But you can actually have fun with this. And I mean, this is pretty
old-fashioned

**[17:58]** stuff. I feel like I'm maybe, like, now at this point an old guy talking about how much
fun we used to have sitting around the radio listening to radio plays, because basically no one
uses this stuff anymore, and there are much, much better and fancier things like ChatGPT. But back
in the day when I was younger, it was really stunning already just how this very simple model
built on very simple data could just have quite good semantic understanding and do quite good
analogies. So you can actually play with this quite a bit and have a bit of fun. So you can do
something like, analogy: *Australia*, comma,

**[18:48]** *beer*, *France*. Okay, what do people think the answer will be? Close. The answer
gives us *champagne*, but that seems a pretty good answer. I could then put in *Russia* — what do
people think? *Vodka*, yeah. You get back *vodka*. This actually works kind of interestingly. I
could do a different one. I can test something different. I can do something like *pencil* is to
*sketching* as *camera* is to *photographing*. Yeah, that works quite

**[19:34]** well. So we built this model in 2014, so it's a little bit out of date in politics. So
we can't do the last decade of politics, which is maybe unfortunate, but we could try out older
politics questions. So we could try *Obama* is to *Clinton* as *Reagan* is to — if you remember
your US history class, any guesses what it's going to say? There's a Bush one. Any other ideas?
Some people have different

**[20:19]** opinions of Bill Clinton. What it answers is *Nixon*, which I think is actually kind of
fair. But you can also get it to do some just sort of language syntactic facts. So you can do
something like *tall* is to *tallest* as *long* is to — this one's easy — yeah. So with this simple
method of learning, with this simple bag of words model, it's enough to learn a lot about the
semantics of words,

**[21:08]** and stuff that's beyond conventional semantics. Like our examples with *Australia*,
*beer*, as *Russia* is to *vodka* — I mean that's cultural world knowledge which goes a little bit
beyond what people normally think of as word meaning semantics, but it's also in there. Yes?

*[Question: if you subtract the distance from, let's say, man and king, does that also capture a
concept of relationship between the two words? Would that give you back something like ruler?]*

Taking the difference between two vectors does capture some — right, the distance between *man* —
so *man* compared to *king* should be a ruler concept. But isn't that what I'm using? Because then
I'm taking the

**[21:57]** distance between *man* and *king*, which is what I'm adding on to *woman* to get the
*queen*, right? Yeah. So depending on which thing you think of as the analogy, you can think of it
that you've both got a difference vector between words that gives you a gender analogy, and one
that gives you a ruler analogy. Yeah, absolutely. Any other questions? Yeah.

*[Question: in word2vec we get two vectors for each one, a u and a v, but here you only have one*

**[22:42]** *vector. So how do you go from two to one?]*

Yeah, good question. I mean, the commonest way in practice was you just average the two of them.
And really you find out that they end up very close, because if you think of it, since you're
going along every position of the text, you're both going to be in the case where, if the text is
"the octopus has legs", you're going to have *octopus* in the center with *legs* in the context,
and a couple of time steps later it's going to be *legs* in the center with *octopus* in the
context. So although they vary a bit for all the regions of the neural nets, very basically they
end up very similar, and people normally just average them. Yeah.

**[23:28]** *[Question: can you use this process — use the answer of one and then place it into the
analogy function of another, and see how far away you can go before it starts to break down?]*

I think you can. So you're wanting to — how distant a relation between two words can you do before
it starts providing an incorrect relationship? But are you wanting to make two steps from
somewhere? Yeah, many steps going away from it. So it doesn't always work. There are some examples
that fail.

**[24:15]** I'm sort of shy to try that now because I don't have a predefined function that did it,
and that might take me too long. But you could play with it at home and see how it works for you.

*[Question: curious, as a clarification — why is it that we use two separate sets of vectors for
word2vec? Is it just to get more parameters, or is there something else?]*

I'll get back to that. Maybe I should go on at this point. Let me move on and kind of just get
through some more details of the word2vec algorithm. So just a technical point on this class, so
you don't make any big mistakes and waste your weekend: for most instances of 224N we've actually
had people

**[25:03]** implement word2vec from scratch as assignment two. But for this quarter, doing it in
spring quarter — as you probably know, spring quarter is actually a little shorter than the other
two quarters — we decided to skip having people implement word2vec. So don't look at the old
assignment two that says implement word2vec, or else you'll be misspending your time. Wait for the
newer assignment two to come out. But despite that, let me just say a little bit more about some
of the details. Right, yeah, so why two vectors? So the two vectors is, it just makes the math a
little bit easier. So if you think about the math, if you

**[25:49]** have the same vectors for the center word and for the outside words — well, for whatever
the center word is, let's say it's *octopus*, when you're going through trying out every possible
context word for the normalization, at some point you'll hit *octopus* again. And so at that point
you'll have a quadratic term, you'll have the exp squared of the *octopus* vector, and that kind of
messes up — I mean, you're clever people, you could work out the math of it, but it makes the math
more of a mess, because every other term it's something different and it's just like an exp, and
then at one position you've got an exp squared. So it just makes the math messier, and

**[26:34]** so they kept it really simple by just having them be disjoint vectors. But it doesn't
make it better. It actually turns out it works a fraction better if you do it right. But in
practice people have usually just estimated them separately and then averaged them at the end. If
you actually look at the paper — here's more of it out, you can find it, the 2013 paper — there's
actually a family of methods that they describe. So they described two methods, one of which was
that you have an inside word that's predicting the words around it, and then the other one tried to
predict the center word from all the words in the

**[27:20]** context, which was called **continuous bag of words** in their paper. The one that I've
described is **skip-gram**, which is simpler and works just great. But then the other part of it is
working out what loss function to be used for training. And what I've presented so far is **naive
softmax**, where we just consider every possible choice of a context word and just run all the math.
That's totally doable, and with our modern super fast computers it's not even that unreasonable to
do — we do things like this all the time. But at least at the time that they wrote their paper this
seemed kind of

**[28:06]** expensive, and they considered other alternatives, like a hierarchical softmax, which I'm
not going to explain right now. But I do just want to explain **negative sampling**. Okay, so this
is just to see a bit of a different way of doing things. So for what we did last time, we had this
straightforward softmax equation, and so in the denominator you're summing over every word in the
vocabulary. And so if you might have 400,000 words in your vocabulary — a lot of words in human
languages — that's kind of a big sum, especially when for each element of the sum you're taking a
dot product between 100 dimensional or 300 dimensional vectors

**[28:51]** and then exponentiating it. A lot of math going on somewhere in there. So maybe we could
short circuit that. And so the idea of the negative sampling was to say, well, rather than
evaluating it for every single possible word, maybe we could just train some simple logistic
regressions where they're going to say you should like the true word that's in the context, and if
we randomly pick a few other words you shouldn't like them very much. And that's skip-gram negative
sampling. So that's what this looks like as an equation. So we've got our center word and our actual

**[29:37]** context word, and we're saying, well, let's work out the term for the actual center word;
we'd like this to be high probability. So since we're minimizing, we're going to negate that and
have it go down. And then we're going to sample some other words and we'd like this to be the
opposite. But the other thing that we've changed here is now we're not using the softmax any more,
we're using this sigma, which stands for the logistic function, which is often called the sigmoid.
Sigmoid just means s-shaped, but you could actually have an infinity of s-shaped functions, and the
one that we actually use is the logistic function. So the

**[30:25]** logistic function has this form, and maps from any real number to a probability between
zero and one. So what we're wanting to say at that point is, for the real outside word we're hoping
that this dot product is large so its probability is near one, and so that will then help with the
minimization. And for the other words we'd like their probability to be small, we'd like them to
appear sort of over here. And that's what this is calculating. But as written, it's sticking the
minus sign on the inside

**[31:10]** there, which works because this is symmetric, right — so you're wanting to be over here,
which means that if you negate it you'll be on this side, which will be large. Okay, and so then the
final bit of this, which is the asterisk, is: so we're going to pick a few words, it might only be
five or ten, that are our negative samples. But for picking those words, what works well is not just
to pick randomly, uniformly, from all the 400,000 words in our vocab. What you basically want to do
is be

**[31:55]** paying attention to how common the words are. Something like *the* is a really common
word. And so we refer to that as the **unigram distribution** — that means you're just taking
individual words independently, how common they are. So about 10% of the time you'd be choosing
*the*. So that's roughly what you want to do for sampling. But people have found that you can
actually do even a bit better than that. So the standard thing that they presented for word2vec is
you're taking the unigram probability of the word and raising it to the power of 3/4. What does that
end up doing? Question for the audience: if I take probabilities and raise them to the

**[32:41]** three quarters — *[some less frequent words just become more likely]* — correct, yeah. So
raising to the 3/4 means that you're somewhat upping the probability of the less frequent words. So
you're in between having every word uniform and exactly using their relative frequencies in the
text; you're moving a little bit in the direction of uniform. And so you get better results by going
somewhat in the distance of sampling more uniformly, but you don't want to go all the way there,
which would correspond to, I guess, putting a zero in there rather than

**[33:28]** three quarters. Okay, yeah, let's see, I had a slide here, but time rushes along, so let's
not bother with this slide, it's not that important. Okay, so that's the word2vec algorithm, that
we've seen all of, in its different forms. A reasonable wonder that you have at this point is, this
seems a kind of a weird way of doing what we're wanting to do, right? The idea is, look, we have this
text, we have words, and we have words in the context

**[34:14]** of words. It sort of seems like an obvious thing to do would be to say, let's just count
some statistics. We have words and there are other words that occur in their context, so let's just
see how often the word *swim* occurs next to *octopus*, and how often the word *fish* occurs next to
*octopus*. Let's get some counts and see how often words occur in the context of other words, and
maybe we could use that to calculate some form of word vectors. And so that's something that people
have already also considered. So if we use the same kind of idea of a context window, we could just
make a matrix of how often words occur in the context of other words. And so

**[35:01]** here's a baby example. My corpus is "I like deep learning, I like NLP, I enjoy flying".
And my context window I'm using is just one word to the left and the right. And then I can make this
kind of co-occurrence count matrix, where I'm putting in the counts of different words in every
context. And because my corpus is so small, everything in the matrix is a zero or one, except for
right here where I've got the twos, because I have "I like" twice. But in principle I've got a matrix
of counts for all the different counts here. So maybe this gives me a word vector, right? Here's a
word vector for *deep* — is

**[35:49]** this long vector here. And I could just say that is my word vector. And indeed sometimes
people have done that, but they're kind of ungainly word vectors, because if we have 400,000 words in
our vocabulary the size of this matrix is 400,000 by 400,000, which is a lot worse than our word2vec
word vectors, because if we're making them only 100 dimensional we've only got 400,000 by 100, which
is still a big number but it's a lot smaller than 400,000 times 400,000. So that's inconvenient. So
when people have started with these kinds of co-occurrence matrices, the general thing that people
have done is to say, well,

**[36:35]** somehow we want to reduce the dimensionality of that matrix so that we have a smaller
matrix to deal with. And so then, how can we reduce the dimensionality of the matrix? And at this
point, if you remember your linear algebra and stuff like that, you should be thinking of things like
PCA, and in particular, if you want it to work for any matrix of any shape, there's the singular
value decomposition. So there, the classic singular value decomposition: for any matrix you can
rewrite it as a product of three matrices, a U and a V which are both orthonormal — which means that
you get these

**[37:22]** independent vectors, they're orthogonal to each other — and then in the middle we have the
singular values, which are ordered in size, so this is the most important singular vector, and these
are sort of weighting terms on the different dimensions. And so this is the full SVD decomposition.
But part of it is relevant, because if I've got this picture, nothing is happening in the part that's
shown in yellow there. But at the moment this is just a full decomposition, whereas if we're wanting

**[38:08]** to have smaller, low dimensional vectors, well, the next trick we pull is we say, well, we
know where the smallest singular vectors are, so we could just set them to zero. And if we did that,
then more of this goes away and we end up with a two-dimensional representation of our words. And so
that gives us another way of forming low-dimensional word representations. And this had actually been
explored before modern neural word vectors, using algorithms such as **latent semantic analysis**.
And it sort of half worked but it never worked

**[38:53]** very well. But some people, especially in psychology, had kept on working on it. And among
other people in the early 2000s there was this grad student Doug Rohde who kept on working on it, and
he came up with an algorithm that he called **COALS**. And he had known, as other people before him
had known, that just doing an SVD on raw counts didn't seem to give you word vectors that worked very
well. But he had some ideas to do better than that. So one thing that helps a lot is if you log the
frequencies, so you can put log frequencies in the cells. But then

**[39:39]** he used some other ideas, some of which were also picked up in word2vec, one of which is
ramping the windows, so that you count closer words more than further away words. He used Pearson
correlations instead of counts, etc. But he ended up coming up with a low dimensional version of word
vectors that are ultimately still based on an SVD. And he got out these word vectors, and
interestingly, no one really noticed at the time, but Doug Rohde in his dissertation effectively
discovered this same property of having linear semantic components. So look, here we go, here's one
of — so this is actually a picture from his dissertation, and look here, you

**[40:27]** know, we've got this meaning component which is *doer of an event*, and he's essentially
shown, with the way he's processed his word vectors, that the doer of an event is a linear meaning
component that you can use to move between a verb and the doer of the verb. Kind of cool. But he
didn't become famous, because no one was paying attention to what he had come up with. So once
word2vec became popular, that was something that I was kind of interested in. And so, working
together with a postdoc, Jeffrey Pennington, we thought that there was interest in this space of
doing

**[41:13]** things with matrices of counts, and how do you then get them to work well as word vectors
in the same way that word2vec worked well as word vectors. And so that's what led into the GloVe
algorithm, which was what I was actually showing you. And so what we wanted was to say, look, we want
a model in which linear components — sort of adding or subtracting a vector in a vector space —
correspond to a meaning difference. How can we do that? And Jeffrey did good thinking and math and
thought about that for a bit, and his solution

**[42:00]** was to say, well, if we think of it that ratios of co-occurrence probabilities can encode
meaning components — so if we can make a ratio of co-occurrence probabilities into something linear in
the vector space, we'll get the kind of result that word2vec or Doug Rohde got. So what does that
mean? Well, so if you start thinking of words occurring in the context of *ice*, you might think that
*solid* and *water* are likely to occur near *ice*, and *gas*, or a random word like *random*, aren't
likely to occur near *ice*. And similarly for *steam* you'd expect that *gas* and *water* are

**[42:46]** likely to occur near *steam*, but probably not *solid* or *random*. And well, if you're
just looking at one of these you don't really get meaning components, because you get something
that's large here or large here. But if you then look at the ratio of two of these co-occurrence
probabilities, then what you get out is that for *solid* it's going to be large, and for *gas* it's
going to be small. And so you're getting a direction in the space which will correspond to the
solid/liquid/gas dimension of physics, whereas for the other words it will be about one. This is just
the wave-your-hands —

**[43:32]** this was the conception of the idea. But if you actually do the counts, this actually
works out. So using real data, this is what you get for co-occurrence. And indeed you kind of get
these factors of 10 in both of these directions of these two, and the numbers over there are
approximately one. So Jeffrey's idea was, well, we're going to start with a co-occurrence count
matrix and we want to make this turn into a linear component. And well, how do you do that? Well,
first of all it makes sense immediately that you should be putting a log in, right, because once you
put a log in, this ratio will be being turned into something that's

**[44:19]** subtracted. And so simply all you have to do is have a log-bilinear model where the dot
product of two word vectors models this conditional probability, and then the difference between two
vectors will be corresponding to this log of the ratio of their co-occurrence probabilities. And so
that was basically the GloVe model. So you're wanting to model this dot product such that it's being
close to the log of the co-occurrence probability, but you do a little bit of extra work to have some
bias terms and some frequency thresholds, which aren't very important,

**[45:07]** so I'm going to skip past them. But I think that basic intuition as to what's the
important thing to get linear meaning components is a good one to know about. Okay, is everyone good
today? Cool. Yes.

*[Question: I noticed the original X matrix you showed was like 3 by 5 or something. Shouldn't it be
square?]*

So yeah, I mean, if you're doing — sorry, yeah, I maybe should have just shown you a square one. If
you're just doing vocabulary to vocabulary, yes, it should be square. But there was a bit in the
slides that I didn't mention, that there was another way you could do it where you did it words
versus documents, and then it would be non-square. But yeah,

**[45:52]** you're right, so let's just consider the square case. Okay. So, hey, I showed you that
demo of the GloVe vectors and they work great, didn't they? So these are good vectors. But in general
in NLP we'd like to have things that we can evaluate and know whether things are really good. And so
everywhere through the course we're going to want to evaluate things and work out how good they are,
and what's better and what's worse. And so one of the fundamental notions of evaluation that will
come up again and again is intrinsic and extrinsic evaluations. So

**[46:38]** an **intrinsic** evaluation is where you are doing a very specific internal subtask and
you just try and score whether it's good or bad. So normally intrinsic evaluations are fast to
compute, help you understand the component you're building, but they are sort of distant from your
downstream task, and improving the numbers internally may or may not help you. And that's the
contrast with an **extrinsic** evaluation, which is that you've got some real task you want to do —
question answering, or document summarization, or machine translation — and you want to know whether
some clever bit of internal

**[47:23]** modeling will help you on that task. So then you have to sort of run an entire system and
work out downstream accuracies, and find out whether it actually helps you at the end of the day. But
that often means it's kind of indirect, so harder to see exactly what's happening in your task. So
for something like word vectors, if we just sort of measure, are they modeling word similarity well,
that's an intrinsic evaluation. But we'd probably like to know whether they model word similarity well
for some downstream task, which might be doing web search, right — we'd like, when you say *cell

**[48:09]** phone* or *mobile phone*, that it comes out at about the same. So web search might be our
extrinsic evaluation. Okay, so for word vectors, two intrinsic evaluations, the ones we've already
seen. So there's the word vector analogies. You know, I cheated when I showed you the GloVe demo, I
only showed you ones that work. But if you play with it yourself you can find some that don't work. So
what we can do is sort of have a set of word analogies and find out which ones work. Now, in general
GloVe does work. Here's a set of word vectors showing you the

**[48:57]** female distinction — it's kind of good and linear. But in general for different ones it's
going to work and it's not going to work, and you're going to be able to score what percentage of the
time it works. Or we can do word similarity. How we do word similarity is we actually use human
judgments of similarity. So psychologists ask undergrads and they say, here is the word *plane* and
*car*, how similar are they on a scale of 1 to 10, or 0 to 10 — maybe actually I think it's 0 to 10
here — on a scale of 0 to 10. And the person says seven. And then they ask another person, and they
average

**[49:43]** what the undergrads say and they come out with these numbers. So *tiger*–*tiger* gets 10,
*book* and *paper* got an average of 7.46, *plane* and *car* got 5.77, *stock* and *phone* got 1.62,
and *stock* and *jaguar* got 0.92. Noisy process, but you roughly get to see how similar people think
words are. And so then we ask our models to also score how similar they think words are, and then we
get models of how well the scores are correlated between human judgments and our models' judgments.
And so here are a big table of numbers that we

**[50:29]** don't need to go through all of, but it sort of shows that a plain SVD works terribly;
simply doing SVD over log counts already starts to work reasonably. And then here's the two word2vec
algorithms, CBOW and skip-gram, and here are numbers from our GloVe vectors. And so you get these
kinds of scores that you can then score different models as to how good they are. And well, then you
can also — oh sorry, yeah, that's the only thing I have there. But what can you do for downstream
evaluation? Well then you want to pick some downstream task. And so a simple downstream task that's
been used a lot in NLP is what's called named

**[51:17]** entity recognition. And so that's recognizing names of things and what type they are. So
if the sentence is "Chris Manning lives in Palo Alto", you want to say *Chris* and *Manning*, that's
the name of a person, and *Palo* and *Alto*, that's the name of a place. So that can be the task. And
well, that's the kind of task which you might think word vectors would help you with. And it's indeed
the case, right. So here, what's labeled discrete was a baseline symbolic probabilistic named entity
recognition task, and by putting word vectors into it you can make the numbers go up. So these
numbers for GloVe are higher than

**[52:03]** the ones on the first line. And so I'm getting substantial improvements from adding word
vectors to my system, yay. Okay, I'll pile ahead into the next thing. This next one I think is
interesting, we should spend a minute on it, and it came up in your questions last time. Words have
lots of meanings. Most words have a whole bunch of meanings. Words that don't have a lot of different
meanings are only some very specialized scientific words. Okay, so my example of a word with multiple
meanings is probably not the first one you think of all the time.

**[52:49]** The most famous example of a word with a lot of meanings is *bank*, which already came up
last time, and I used *star*, which is another one. Here's a word that you probably don't use that
often, but it still has lots of meanings: the word *pike*. What are some things that the word *pike*
can mean? A fish — yes, it's a kind of fish, okay, we've got one. What else can a pike be? Yeah, a
spear — a spear, yeah, for the Dungeons and Dragons crowd, yeah, there's a long arm, right, yep,
that's another one. Yeah, a road, right, yes — so *pike* is used as a shorthand for a turnpike. Why
it's called a turnpike: originally you had this spiky looking thing at the start of it to sort of
count people. Okay,

**[53:35]** we've got three. Other meanings for *pike*? Yeah, is it also a frat, like a fraternity?
I'll believe you, I can't say I know that one. Are pikes sharp, as like a needle, something sharp
maybe? I mean, I think it's really the sort of *pike* as the weapon. Other — scratch your heads. One
that I think a lot of you will have seen in diving and swimming: you can do a pike. Olympics — if you
see Olympic diving there are pikes. Anyone seen

**[54:20]** those? Trust me, that's a pike. Okay. And we've sort of been doing the noun uses, but you
can also use *pike* as a verb, right — once you've got your medieval weapon you can pike somebody,
and that's a usage of *pike*. And you can do other ones. So here we go, here's ones I got from a
dictionary. We got most of those. There are sort of weirder usages, right, like *coming down the
pike* — that's a kind of metaphorical use that comes from the road sense, but it sort of ends up
meaning the future. Yeah, in Australia we also

**[55:07]** use *pike* to mean sort of chicken out of doing something, but I don't think that usage is
really used in the US. Anyway, words have lots of meanings. So how can you deal with that? Well, one
way you could deal with it is to say, okay, words have several meanings, and so we're just going to
say words have several meanings. And then we're going to take instances of words in text, we're going
to cluster them based on their similarity of occurrence to decide which sense of the word to regard
each token as, and then we're going to learn word vectors for those token clusters, which are our
senses. And you can do

**[55:52]** that. We did it in 2012, before word2vec came out. So you see here we have *bank one*, and
somewhere over here we have *bank two*, and here we have *jaguar one*, *jaguar 2*, *jaguar 3*,
*jaguar 4*. And this really works out great, right. So *jaguar one* picks out the sense of the kind of
car, and it's close to *luxury* and *convertible*. *Jaguar 2* comes right close to *software* and
*Microsoft* — and this one's a bit of a historical one, but when you were five or whatever, you might
remember Apple

**[56:39]** used to use large cats for versions of Mac OS, right, so Mac OS 10.3 or something like that
a long time ago was called Jaguar. So it's software, close to Microsoft. *Jaguar 3* — okay, *string*,
*keyboard*, *solo*, *musical*, *drum*, *bass* — that's because there's a Jaguar keyboard. And then
finally, the sort of what we think of as the basic sense, but it turns out turns up rather less in
text corpora normally, *jaguar* next to *hunter* is the animal. So it's done a good job at learning
the different senses. But that's not what's

**[57:26]** actually usually done these days. Instead, what's usually done is you do only have one
vector for *jaguar*, and when you do that — or *pike* here — the one vector you learn is a weighted
average of the vectors that you would have learned for the senses. It's often referred to as a
**superposition**, because somehow neural net math people like to use physics terms, and so they call
it a superposition, but it's a weighted average. So you're taking the relative frequency of the
different senses and multiplying the vectors you would have learned if you'd had sense

**[58:12]** vectors, and that's what you get as the representation as a whole. And I can make a sort of
a linguistic argument as to why you might want to do that, which is, although this model of *words
have senses* is very longstanding and common — I mean, it's essentially the way dictionaries are
built, right, you look up a word in the dictionary and it says sense one, sense 2, sense three, and
you get them for things like *bank* or *jaguar* as we're talking about — I mean, it's sort of really a
broken model, right, in that word meanings have a lot of nuance,

**[58:57]** they're used in a lot of different contexts. There are extreme examples like *bank*,
wherever it was, where we have finance bank and bank of a river over here, where it seems like the
senses are this far apart. But most words have sort of different meanings, but they're not actually
that far apart, and trying to cut them into senses seems actually very artificial. And if you look up
different dictionaries and you say, how many senses does this word have, pretty much everyone will
give you a different answer. So the kind of situation you have is a word like *field*. Well, a field
can be used for a place where you grow a crop, it

**[59:43]** can be used for sort of natural things like a rock field or an ice field, it can be used
for a sporting field, there's the mathematical sense of field. Now all of these things sort of have
something to do with each other — I mean, the math one's further away, but the physical ones are sort
of flat spaces. But the sense of it being a sporting field is clearly kind of different from the sense
of it being an ice field. Is the ice field and the rock field different, or am I just modifying? Are
they different senses, right? So really you sort of have a kind of a — what a math person would say is
sort of

**[1:00:29]** like some probability density distribution over things that can be meant by the meaning
of a word. So it sort of maybe makes sense to more use this model where you're just actually saying we
have a vector that's an average of all the contexts. And we'll see more of that when we get to
contextual word vectors later on. But one more surprising result on this is: since you have the vector
for *pike* overall being the sum of these different sense vectors, standard math would tell you that
if you just have the single vector there's no way that you can recover the individual

**[1:01:15]** sense vectors. But higher math tells you that actually these vector spaces are so high
dimensional and sparse that you can use ideas from **sparse coding theory** to reconstruct the sense
vectors out of the whole vector. And if you actually want to understand this, some of the people in
statistics — David Donoho I think is one of them — teach courses on sparse coding theory. But I'm not
going to try and teach that. But here's an example from this paper, Arora et al., where one of the
authors is Ma, who's now faculty in computer science here, where

**[1:02:03]** they, starting off with the word vector and using sparse coding to divide out sense
vectors from one word vector — and they work pretty well, right. So here's one sense of *tie* which is
a piece of clothing; another sense of *tie* which is ties in the game; this one is sort of similar to
that one, I'll admit; but this sense of *tie* here is then a tie as sort of you put on your electrical
cables; then you have the musical sense of *tie*. Right, at least four out of five, they've done a
pretty good job of getting senses out of this single word vector by sparse coding. So sparse coding
must be cool, if you want to go off and learn more about

**[1:02:49]** it. Okay. Okay, so that's everything I was going to say about word vectors and word
senses. Is everyone good to there? Any questions? I'll rush ahead for the last two pieces. Okay, so I
just wanted to start to introduce, in the last 15 minutes, the ideas of how we can build neural
classifiers, and how we start to build in general neural networks. I mean, in a sense we've already
built a very simple neural classifier: our word2vec model is predicting what words are

**[1:03:37]** likely to occur in the context of another word, and you can think of that as a
classifier. But let's look at a simple classifier like our named entity recognizer I mentioned before.
So for the named entity recognizer we want to label words with their class. So we want to say these two
words are a person, but the same words *Paris* and *Hilton* are then locations in this second
sentence. So words can be ambiguous as to what their class is. And the other state is that they're not
a named entity at all, they're just a word that is some other word. And this is something that's used
in lots of places as a bit of

**[1:04:22]** understanding. So if you've seen any of those web pages where they've sort of tagged
company names with a stock ticker, or there are links on a Wikipedia page to a Wikipedia page or
something like that, right, you've got named entities, where commonly after finding the named entities
you're doing this second stage of **entity linking**, where you're then linking the named entity to
some canonical form of it, like a Wikipedia page. But we're not going to talk about the second part of
it for the rest of the day. And so we could say that, building with our word vectors, we've got this
simple task where what we're going to do is

**[1:05:07]** we're going to look at a word in context, because sometimes *Paris* is a name of a person,
sometimes it's a location. And so we're going to want to look at this word in its context and say, aha,
this is a name of a location in this instance. And so the way that we're going to do it is we're going
to form a **window classifier**. So we're going to take a word with a couple of words of context on
each side, and for the words in our context window we're going to use our word vectors, because we
want to show they're useful for something. And then we want to feed this into something that is a
classifier. And our classifier, it's actually going to be a really simple logistic

**[1:05:53]** classifier — we're only here going to do location or not a location. So this one here we
want to say, for this window here, yes, it's a location; whereas if it had been "I love Paris Hilton
greatly", then we'd be saying no, because *Paris*, the word in the middle of the context, then isn't a
location. So that's the idea of a classification or classifier: we're assigning some set of classes to
things. Right, so in general for classifiers we do supervised learning, which means we have some
labeled

**[1:06:39]** examples, our training data set. So we have input items x_i, and for each one we've got a
class y_i. So I had, for my example training examples, ones like "I love Paris Hilton greatly" — that
was negative, not a location — and "I visit Paris every spring" — that's positive, that is a location
— where I'm actually classifying the middle word. Okay, so inputs, labels. And in general labels are a
set of classes. So my set here is simply location or not a location. But I could get fancier and I
could say I've got five classes: I've got location, person name,

**[1:07:26]** whatever other ones there are, company name, drug name. Right, I could be assigning a
bunch of, or *other*, not a name — a bunch of different classes. But I'm going to be doing it with only
two, because I'm using this example on next Tuesday's lecture as well and I'm wanting to keep it
simple. So that's what we're going to do. And so what we're going to be using in our class is neural
classifiers. And so I just wanted to go through quickly just the sort of food for thought as we go into
it. So for a typical stats machine learning classifier, you can build classifiers

**[1:08:13]** like logistic regression or softmax classifiers, or other ones like support vector
machines or naive Bayes, or whatever else you might have seen. The vast majority of these classifiers
are **linear classifiers**, meaning that they have a linear decision boundary. And when we're learning
these classifiers we're learning parameters here W, but our inputs are fixed — our inputs are
represented by symbols or quantities. So we have fixed inputs, we learn parameters as weights that are
used to multiply the inputs, and then we use a linear decision boundary. So when we have our neural
classifier

**[1:08:59]** we're kind of getting some more power. So first of all, we're not only learning weights W
for our classifier, we're also learning distributed representations for our words. So our word vectors
re-represent the actual words or symbols and can move them around in the space, so that in terms of the
original space we've got a nonlinear classifier that can represent much more complex functions. But we
will then sort of use the word vectors to re-represent those words to do a final classification. So at
the end of our deep network, which we're about to

**[1:09:45]** build, we will have a linear classifier in terms of our re-represented vectors, but not in
terms of our original space. Let me try and be concrete about that. Okay, so here's what I'm going to
use, and we'll use again next Tuesday, as my little neural network. And so I start with some words,
"museums in Paris are amazing". I first of all come up with the word embedding of those using my word
vectors. So now I've got this sort of high dimensional vector which is just a concatenation of five word
vectors. So if I have 100 dimensional word vectors, this is 500 dimensional. And then I'm going to put
it through a neural network layer, which is

**[1:10:32]** simply multiplying that vector by a matrix and adding on a bias vector, and then I'm going
to put it through some nonlinearity, which might be for example the logistic function that we've
already seen. So that will give me a new representation. And in particular, if the W is say 8 by 500,
I'll be reducing it to a much — or what, yeah, 8 by 500 — I'll be reducing it to a much smaller vector.
So then after that I can multiply my hidden representation, the middle of my neural network, by another
vector, and that will give me a score. And I'm going to put the score into the

**[1:11:19]** logistic function that we saw earlier, to say what's the probability this is the location.
So at this point my classifier is going to be a linear classifier in terms of this internal
representation that's used right at the end, but it's going to be a nonlinear classifier in terms of my
word vectors. Okay, great. Here's one other thing. This is just a note for learning ahead, since you
want to know this when we start doing the next assignments. I mean, up until now I've presented
everything as doing log

**[1:12:07]** likelihood and negative log likelihood for building our models. Very soon now, assignment
two, we're going to be starting to do things with PyTorch, and when you start working out your losses
with PyTorch, what you're going to be wanting to use is **cross entropy loss**. And so just let me
quickly say what cross entropy loss is. So cross entropy is from information theory. So if you have a
true probability distribution p and you're computing a probability distribution q, your cross entropy
loss is like this. So it's the log of your model probability,

**[1:12:56]** the expectation of that under your true probability distribution. But there's sort of a
special case, whereas if you have ground truth or gold or target data where things are labeled one
zero — so like for examples of "I visit Paris", right, I'm just labeling it one for location,
probability one it's a location, probability zero it's not a location — so if you're just labeling the
right class as probability one, then in this summation every other term goes to zero and the only thing
you're left with is, what probability is

**[1:13:43]** my model, what log probability is my model giving to the right class. And so that then is
your log likelihood, which we can use for the negative log likelihood. Little bit of a complication
here — just remember that you want to use cross entropy loss in PyTorch when building the model. Okay.
Before we end today, here is my obligatory one picture of human neurons. Don't miss it, because I'm not
going to show any more of these. Okay, these are human neurons. Human neurons were the inspiration for
neural networks,

**[1:14:29]** right. So human neurons have a single output which comes down this axon, and then when you
have these outputs they then feed into other neurons. I guess I don't really have an example here, but
in general one output can feed into multiple different neurons — you can see the different things
hanging into it. So you have the output connecting to the input, and sort of where you make this
connection, right, that's the synapses that people talk about. And so one neuron will normally have
many, many inputs where it picks things up from other neurons, and they all go into the nucleus of the
cell, and

**[1:15:17]** the nucleus combines together all those inputs, and kind of what happens is if there's
enough positive activation from all of these inputs, it then sends signals down its output. Now,
strictly, how neurons work is that they send spikes, so the level of activation of a neuron is its rate
of spiking. But that immediately got turned, in artificial neural networks, into just a real value as to
what is its level of activation. And so it does this. So this was kind of the genuine inspiration of all
of our neural networks, right. So a binary logistic regression is kind of a bit

**[1:16:05]** similar to a neuron, right. It has multiple inputs, you're working out your total level of
excitation, where in particular you can have inputs that are both exciting, positive inputs, and inputs
that are negative, which are then inhibitory inputs. You combine them all together and you get an
output that's your level of excitation, and you're then sort of converting that through some
nonlinearity. And so this was proposed as a very simple model of human neurons. Now, human neurons are
way more complex than this, and some people like neuroscientists think we maybe should be doing a better
model of actual human

**[1:16:53]** neurons. But in terms of what's being done in the current *neural networks eat the world*
revolution, everyone's forgotten about that and is just sticking with this very, very simple model,
which conveniently turns into linear algebra in a very simple way. So this gives us sort of like a
single neuron. But then, precisely, right, this single neuron, if you use the logistic function, is
identical to logistic regression, which you've probably seen in some stats class or something. But the
difference is that for neural networks we don't just have one logistic regression, we have a bunch of
logistic regressions at once.

**[1:17:42]** And well, it'd be tricky if we had to define what each of these logistic regressions was
calculating, but what we do is we just feed them into another logistic regression. And so we have some
eventual output that we want to be something like — we want it to say this is or isn't a location. But
then what will happen is that, by our machine learning, these intermediate logistic regressions will
figure out all by themselves something useful to do. That's the magic, right, that you get this sort of
self-learning property where the model has a lot of parameters

**[1:18:28]** and internally will work out useful things to do. So in general we can get more magic by
having more layers in the neural network, and we will build up functions. So effectively these
intermediate layers let us learn a model that re-represents the input data in ways that will make it
easier to classify, or easier to interpret and do things with downstream in our neural network. And it's
time, so I should stop there. Thank you.
