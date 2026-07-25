---
title: Lecture 3 — Backpropagation and Neural Networks
lecture: 3
video: https://www.youtube.com/watch?v=HnliVHU2g9U
source: YouTube auto-captions, copy-edited for readability
verbatim: false
original: original/03-backpropagation-and-neural-networks.md
slides: ../slides/03-backpropagation-and-neural-networks.md
---

# Lecture 3 — Backpropagation and Neural Networks — transcript

Timestamps mark the start of each paragraph and can be cited directly ("Manning
works the example around 51:13"). All 95 paragraph timestamps from the source
captions are preserved.

**This is an edited transcript.** The auto-generated captions had no punctuation and
mangled most technical vocabulary — *ReLU* arrived as "value" and "realu", *tanh* as
"10 H", *PyTorch* as "py torch" and "p i", *Swish and GELU* as "Swiss swis and Jello",
*affine* as "aine", *Hadamard* as "Hadad", *recurrent neural networks* as "current new
networks", and *neural* as "newal" throughout. They have been copy-edited into
sentences: punctuation added, false starts and filler removed, mis-heard terms restored
from context and checked against the slides. Mathematical expressions dictated aloud
("D S D W") are written in symbols (∂s/∂W). **No content was added, removed, or
reordered.** The verbatim captions are kept at
[`original/03-backpropagation-and-neural-networks.md`](original/03-backpropagation-and-neural-networks.md)
for reference — use this file unless you specifically need the raw output.

**Where the source is still unreliable**, the text carries an inline `[Ed:` note rather
than a silent guess. There are four, at 1:40, 22:26, 41:52 and 1:11:33 — an assignment
number, a decimal digit, a subscript, and one word that inverts a claim. In each case
the slide that settles it is named.

---

**[0:05]** Okay, hi everyone, I'll get started. So it's Tuesday of week two, so hopefully
that means everyone has done Assignment 1. Everyone done Assignment 1? You know, if I'm
saying this I'm probably saying it to the wrong people, but it seems like every year some
people blow some of their late days on Assignment 1, and it's really just the wrong place
to use them. So yeah, hopefully you've all done Assignment 1, and note that this is meant
to be the easy on-ramp, and then we go straight on from that. So out today we have
Assignment 2. Assignment 2 has two purposes.

**[0:55]** Purpose one is to make you do some math, to get some understanding of what
neural networks really compute and how they compute it — and that's what I'm going to
talk about today, is also going through that math. But then simultaneously — maybe it
does three things in Assignment 2 — we're going to be learning something about dependency
parsing, which will be actually something about language structure and linguistics. But
then thirdly, for Assignment 2 we're going to start using PyTorch. So PyTorch is one of
the leading software frameworks for deep learning, and the one that we're going to use
for this class.

**[1:40]** For the assignment, PyTorch is exceedingly scaffolded — so it's sort of, you
know, here's this thing and you have to write these two lines, use these two functions.
[Ed: the captions read "the assignment 3" here, but the surrounding discussion and slide
2 of the deck ("Ass 2 is quite scaffolded") are about Assignment 2.] But nevertheless, to
help people get up to speed and get started using PyTorch, on Friday at 3:30 in Gates B01
— and it will again be recorded — we have a tutorial on PyTorch. So that's a great way to
get more of a sense of PyTorch and how it works before doing Assignment 2. The other
things — yeah, so for nearly all the lectures we've got further reading, places that you
can look.

**[2:26]** Of all the classes in the entire — this, for many people, might be a really
good one to look at the suggested readings. We have several readings which are sort of
shorter tutorials and reviews of the kind of matrix calculus and linear algebra that we
need for this class. So I really encourage you to look at those. If you decide that one
is your favorite, you can tell us on Ed which one you think is the best one to choose
between them. I kind of like the one that's first on the list, but maybe you'll feel
differently. Conversely — yeah, so today will be sort of all math,

**[3:11]** and then Thursday will be kind of all language and linguistics. Some people
find the language and linguistics hard as well, so I guess different kinds of people.
Okay, so getting straight into it. So where we started last time: I'd sort of shown these
baby neural networks and sort of said, well, we can think of each of those orange things
as basically like a little logistic regression unit. And the crucial difference from the
kind of statistics machine learning you see in a stats class, 109, or wherever, is that
in those you have one logistic regression and you're defining the input features

**[3:56]** to it, and you've got some decision variable that you want to have at the
output. Here you're sort of building these cascades of little logistic regressions. And
so the idea is, right at the end we're going to define what we want — we're going to
capture that by our objective function or loss function. But for the stuff in the middle,
that stuff in the middle is going to be a chance for the neural network to learn by
itself what would be useful inputs to further downstream neurons. What kind of functions
should I come up with in terms of my inputs that will help me provide useful outputs to
help the final

**[4:43]** computation down the track? And if you haven't sort of seen and thought about
this much before, I think it's worth sitting with that idea for a moment, because this is
really a super powerful idea, which is what's made neural networks more powerful in most
circumstances than other forms of machine learning: the fact that you have this
self-organization of intermediate levels of representation that you use to compute things
that will be useful downstream for what you eventually want to do. The other reason I was
bringing back up this picture is I sort of wanted to go straight from here to matrices.

**[5:30]** So while you could sort of wire together neurons however you wanted to — and
arguably if you look at human brains, they look more like neurons wired together however
you wanted to — for what's done with neural networks, basically there's always this kind
of regular structure of layers. So once we have this regular structure of layers, we are
taking the output of one of our neurons at one layer and we're feeding them together with
weights to produce the inputs to the next layer. So we're taking the x₁, x₂, x₃ outputs,
we're multiplying them all by weights, we're

**[6:15]** adding a bias term, and then we're going to put it through a non-linearity, and
that will give us the value at the next layer. So if we then kind of collapse that to a
vector, and this to a vector, that then collapses into a computation where first of all
we're doing a matrix multiplication — we're calculating **Wx** of the inputs — and then
we're adding on the biases as a vector of biases, which gives us this intermediate value
**z**. And then we have this non-linearity or activation function which is applied to
that, which gives us the values in the next layer of the neural network. And the

**[7:01]** activation function is applied to a vector and produces a vector, but it's
operating on each of the individual components of that vector one at a time. So we've got
some scalar function that we're just applying to each element of the vector. And so that's
the kind of picture we saw when I did this example — and I'm going to continue to use this
example in today's class. Remember, we were going to decide whether the word in the middle
of the input window was a location or not. And so we were doing the matrix multiplication,
putting it through the non-linearity, we're then just doing a dot product here, and then
that got stuck into a sigmoid to

**[7:48]** predict yes or no. And the final thing I wanted to say a little bit about is
these *f*'s, the non-linearity or the activation function — and where did they come in?
Well, the starting point of where they came in in the history of neural networks is when
people came up with this idea that you could represent the operation of a basic neuron by
doing a matrix multiplication of the inputs and then having a bias term, or here a
threshold term, to see whether the neuron should fire or not. That was actually in the
very first implementation, which dates back to the

**[8:34]** 1940s, as done as a threshold. Right, so that if the activation was greater than
θ you output one, otherwise you output zero. And well, if you have a threshold, the two
lines are flat, right? So there is no slope, there is no gradient. So that actually makes
learning much harder. So the whole secret of what we build with neural networks — and an
alternative name that's popular in some circles these days is *gradient-based learning* —
and the entire idea of gradient-based learning is, if we actually have some slopes, then
it's

**[9:19]** like going skiing during spring break: you can work out where it's steeper and
you can head down where it's steeper, and that will allow us to optimize our function, to
learn much more quickly. And so that's one reason that we don't just want to have
threshold units — we want to have things with slopes, so we have gradient. So in
subsequent work people started using activation functions with slopes. And so the first
popular one was this sigmoidal logistic that we've seen, for mapping to probabilities.
But it's sort of imperfect, it seemed, because the output was always non-negative, so
that sort of

**[10:06]** tends to push things towards bigger numbers. So there was quite a bit of use
then of this tanh function. And you'll actually see tanh when we do Assignment 3 — we'll
be using tanh's in our recurrent neural networks. And so I've written there the formula
usually given for tanh in terms of exponentials. If your math is rusty it's not obvious
that tanh and logistic have much to do with each other, but if you want to treat this as
a math problem, tanh is literally just a rescaled logistic — you're stretching it by two
and moving it down by one. It's the same function.

**[10:54]** Okay. But you know, so that's nice, but if you're calculating tanh's you have
to do all of these exponentials, and exponentials are kind of slow on your computer and
things like that. You might wonder whether you couldn't get away with something much
cheaper. And so people thought about that, and thought, oh, maybe we could just use a
so-called *hard tanh*, where it has a slope of one in the middle and is then just flat
outside that area. And that seemed to work in many cases. And so that then led to the
popularity of the rectified linear unit. So the rectified linear unit is simply zero on
the negative region, and then is *y* = *x* in

**[11:40]** the positive region. Now, this seems kind of wonky and goes against what I was
saying about gradient-based learning, because once you're in the negative region there's
no gradient — you're just dead. But in the positive region there is gradient, and the
gradient is particularly simple: the slope is always one. And so this still feels slightly
perverse to me, but this really became the norm of what people use for a number of years,
because people found that although for an individual neuron it was dead half the time —
anytime it went negative — that overall for your neural network some things would be
alive. So it kind of gave

**[12:27]** sort of a form of specialization. And the fact that the slope was always one
meant that you got really easy, productive backward flow of gradients, in a way we'll talk
about later. And so learning with ReLU turned out to be very effective, and people started
using the ReLU non-linearity everywhere, and it sort of became the default and the norm.
And you'll see us using it in the assignments — in particular we use it in Assignment 2,
and so you get to see that it works. But nevertheless, at some point people sort of had
second thoughts and decided, you know, having it dead over half of its range maybe isn't
such a good idea after all, even though it seemed to

**[13:12]** work great for a few years. And so a lot of what's happened since then is to
come up with other functions which are in some sense ReLU-like but not actually dead. So —
okay, here we go. So one version of that is the so-called Leaky ReLU. So for the Leaky
ReLU you make the negative half a straight line as well, with a very minor slope, but
still it's got a little bit of slope. There is then a variant of that called the
Parametric ReLU, where you have one extra parameter, which is actually what the slope of
the negative part is, and people showed some

**[13:58]** positive results with that. More recently again — and this is what you often
see in recent Transformer models — you see non-linearities like Swish and GELU. So both of
these are sort of fancy functions, but what they both look like is basically *y* = *x* to
all intents and purposes — not quite, but approximately — and then you've got sort of some
funky bit of curve down here, which again gives you a bit of slope. The curve is going the
opposite way, which is sort of a bit funny, but they seem to work well, commonly used in
recent Transformer models. So that's a bit of a dump of all the non-linearities people

**[14:45]** use. I mean, the details of that aren't super important right now, but the
important thing to have in your head is: why do we need non-linearities? And the way to
think about that is that what we're doing with neural networks is function approximation.
There's some very complex function that we want to learn — you know, like maybe we want to
go from a piece of text to its meaning, or we want to be interpreting visual scenes or
something like that. And so we want to build really good function approximators. And well,
if you're just doing matrix multiplies, a matrix multiply of a vector is a linear
transform, so that doesn't let you model complex functions. I guess

**[15:32]** strictly, if you put a bias on the end, it's then an affine transform — but
let's keep it simple, linear transforms. So if you're doing multiple matrix multiplies,
you're doing multiple linear transforms, but they compose. So you could have just
multiplied these two matrices together and you'd have a single linear transform. So you
get no power in terms of representation by having multi-layer networks that are just
matrix multiplies. As a little aside: in terms of representational power, having
multi-layer matrix multiplies gives you no power, but if you think about it in terms of
learning, actually it does give you some power. So in the theoretical

**[16:18]** community looking at neural networks there are actually quite a few papers
that look at *linear* neural networks — meaning that they're just sequences of matrix
multiplies with no non-linearities — because they have interesting learning properties,
even though they give you no representational power. Okay, but we'd like to be able to
learn functions like this, not only functions like this. And to be able to learn functions
like this we need more than linear transforms, and we achieve those by having something
that makes us be calculating a non-linear function. And it's these activation functions
that give us non-linear functions. Okay, cool. So then, getting on to today:

**[17:04]** the whole thing we want to do now is gradient-based learning. This is our
stochastic gradient descent equation, where that upside-down triangle symbol, that's our
gradient. We're wanting to work out the slope of our objective function, and so this is
how we're going to learn, by calculating gradients. So what we want to know is: how do we
calculate the gradients for an arbitrary function? And so what I want to do today is,
first of all, do this by hand, with math, and then discuss how do we do it
computationally, which is effectively the famous thing that's taken as powering,
underpowering, all of neural

**[17:50]** nets, which is the backpropagation algorithm. But the backpropagation
algorithm is just automating the math. Okay. And so for the math, it's matrix calculus.
And at this point then there's a huge spectrum between people who know much more math than
me and people who barely ever learned this. But I hope to explain the essentials, or
remind people of them, enough that you're at least at a starting point for reading some
other stuff and doing homework 2. So let's get into that. And I'm going to spend about
half the time on those two halves. And the hope is that after this you'll feel like, oh, I
actually

**[18:35]** understand how neural networks work under the hood. Fingers crossed. Okay, so
here we go. So if you're a Stanford student you maybe did Math 51, or else you could have
done Math 51, which teaches linear algebra, multivariable calculus and modern
applications. Math 51 covers everything I'm going to talk about and way more stuff. So if
you actually know that and remember it, you can look at Instagram for the next 35 minutes.
But I think the problem is that, quite apart from the fact a lot of people do it as frosh,
this is a lot to get through in 10 weeks, and I think that a lot of the people who do

**[19:21]** this class, sort of by two years later, don't really have much ability to use
any of it. But if you actually looked at this book really hard and for a really long time,
you would have discovered that actually, right towards the end of the book in Appendix G,
there's actually an appendix on neural networks and the multivariable chain rule, which is
precisely what we're going to be using for doing our neural networks. But there are only
two problems. One problem is that this is page 697 of the book, and I'm not sure anyone
ever gets that far. And the problem is, even if you do get that far, I find these pages
that they're really dense, texty pages — it's not even easy to

**[20:08]** understand them if you have gone there. So here's my attempt on that. So the
mantra to have in your head is: if I can remember basic single-variable calculus — you
know that I've got 3*x*² and the derivative of that is 6*x*, that's all you sort of need to
know — the essence is, multivariable calculus is just like single-variable calculus except
you're using matrices. Okay, so that's our article of faith, and we're going to do that.
And so what we're wanting to do is to do matrix calculus, or the generalization of that,
tensor calculus, sort of using vectors, matrices and higher-order tensors. Because if we
can do things in what's

**[20:54]** referred to as *vectorized gradients* in the neural network world, that will be
sort of the fast, efficient way to do our operations. You know, so if you want to think it
all through you can do it single variable at a time and check that you're doing the right
thing — and I sort of tried to indicate that in the first lecture — but if we want to have
our networks go vroom, we want to be doing matrix calculus. Okay, so let's work up to
doing that. So this is the part that I trust everyone can remember. So we have
*f*(*x*) = *x*³, and we can do a single-variable derivative, and the

**[21:40]** derivative is 3*x*². Everyone remember that one? Okay, that's something we can
all start from. And remember, this derivative is saying the slope of things. And the slope
of things lets us work out where is something steep, so we'll be able to go skiing —
that's our goal. And so you can think of the slope of things as how much the output will
change if we change the input a bit. That's our measure of steepness. So since the
derivative is 3*x*², if we're at *x* = 1 that means the slope is about 3 × 1² = 3. So if I
work out the value of

**[22:26]** the function for 1.01, it's gone up by about three times — I move the *x* by
0.01 and the output moved by 0.03. Whereas if I go to *x* = 4, the derivative is
3 × 4² = 48, and so if I work out the value of the function at 4.01 I get approximately
64.4 [Ed: slide 13 gives 4.01³ = 64.48] versus 64. That small difference from 4 to 4.01
has been magnified 48 times in the output. Okay. So now we just remember the mantra: it's
going to be exactly the same single-variable calculus, but with more stuff. So if we have
a function with *n*

**[23:11]** inputs, we're then going to work out its gradient, which is its partial
derivative with respect to each input. So its gradient will now be a vector of the same
size as the number of inputs. And there's this funky symbol ∂ which people pronounce
various ways. I mean, this kind of originated as some kind of someone's weird way of
drawing a calligraphic *d*, right, so it is really a *d*. So I think I'll mainly just call
it *d*, but sometimes people call it "partial" or "funny *d*" or some other name. So you
have ∂*f*/∂*x*₁, ∂*f*/∂*x*₂, for each of the variables. Okay, so if we go beyond that and
then have a function

**[23:57]** with *n* inputs and *m* outputs, what we then get for the gradient is what's
referred to as the *Jacobian*. Now actually, the dude this is named after was a German Jew,
so it should really be "Yacobi", but no one says that in this country — Jacobian. Okay, so
the Jacobian is then a matrix of partial derivatives, where you're working out, for each
output and each input, the partial derivative between the component of the input and the
output. So this looks like the kind of thing that we're

**[24:43]** going to have when we have a neural network layer, because we're going to have
*n* inputs and *m* outputs for the two layers of our neural networks. So we'll be using
these kinds of Jacobians. Okay, so then the whole idea of neural networks is we've got
these multi-level computations, and they're going to correspond to composition of
functions. So we need to know how to compose things, both for calculating functions and
for calculating their gradients. So if we have a one-variable function and we want to work
out its derivative in terms of a composition of two functions,

**[25:31]** what we're doing is multiplying our computations. Okay, so if you compose
together *z* of *y* — that's the function that we did at the beginning, that gives you —
oh, was it? No, it's not, sorry, it's different. Okay. *z* of *y* gives you 3*x*², right?
And so we know that the derivative of that is 6*x*. Okay, if we do it in terms of the
pieces, we can work out d*z*/d*y*, which is just going to be 3, and d*y*/d*x* is 2*x*, and
we can work out the total derivative by multiplying these two pieces, and we get 6*x*, the
same

**[26:18]** answer. So, matrix calculus is exactly like single-variable calculus except
we're using tensors of different — so the word *tensor* is used to mean, as you go up that
spectrum in its size, so from sort of scalar to vector to matrix to then what in computer
science is normally still multidimensional arrays — that spectrum is then tensors of
different dimensions. Okay, so when we have multiple-variable functions we're going to
multiply Jacobians. So here we have a function **Wx** + **b**, and then we compose

**[27:04]** the non-linearity *f* to get **h**, and so we're going to be able to compute
that in the same way as a product of these partial derivatives, which are Jacobians. Okay,
so let's start looking at a few examples of what we get. So let's start with an
element-wise activation function. So when we have a vector that's being calculated as the
activation function of a previously computed quantity — well, we're computing that
component-wise, as I explained before. So h_i = *f*(z_i), where *f* is our activation
function, that actually applies to a scalar. But

**[27:51]** overall this layer is a function with *n* outputs and *n* inputs, and so it's
going to have an *n* × *n* Jacobian. And well, what that's going to — so this is our
definition of the Jacobian, but in this case this is sort of a special case, because if
*i* = *j* then we're going to have the output h_i depending on z_j, and otherwise it's
going to be zero. Because for the off-diagonal entries, it doesn't matter how you change
the value, it's not changing the output, because the output only depends on the
corresponding index. And so what we're going to get for this

**[28:37]** Jacobian of activation functions is a matrix where everything is zero apart
from the diagonal terms that correspond to where we're calculating the activation
function. And for those ones we're going to have to work out how to compute the derivative
of our activation function. That was one of the questions on Assignment 1, I do believe —
or was it on Assignment 2? No, no, it's Assignment 2. One of the questions on Assignment 2
— one of the ones on the new assignment — is to say, hey, can you work out the derivative
of a logistic function? Well, then we'd be able to plug that straight into *f*′. So I'm
not going to give that answer away today.

**[29:24]** Okay, so other things that we want to do with a Jacobian. Well, we have this
layer of our neural network where we're calculating **Wx** + **b**, and we want to work
out the partial derivative of that with respect to **x**. This is the kind of place where
it actually works to remember the mantra and say, matrix calculus is just like
single-variable calculus but with matrices. So if you just don't use your brain too hard
and think, oh, it's just like single-variable calculus, so what should the answer be —
it's obviously going to be **W**. And indeed it is. Similarly, if we want to do the same
thing for **Wx** + **b** and

**[30:10]** work out the partial derivative with respect to **b** — well, that would be one
in terms of single-variable calculus, and so in matrix calculus that becomes an identity
matrix. Okay, slightly different, the same idea, but that's reflecting the fact that **b**
is actually a vector, so we need it to be coming out as an identity matrix. Okay, so
higher up in my example picture I did this sort of vector dot product of **u**ᵀ**h**. And
well, what happens if we work out the derivative of that? What we end up with

**[30:57]** strictly is we come out with **h**ᵀ. And this is sort of like when you're
working out — well, we did this in the first class, right, when we did a dot product
calculation — that you kind of get, for each individual element, you get the opposite
term, and so you get the other vector coming out. These are sort of good ones to compute
at home for practice, to make sure you really do know the answers and why they work out
the way they do. Okay, so let's go back to our little neural net. This was most of our
neural net. Above our neural net there was the non-linearity. Now I'm going to leave

**[31:45]** that out this time. Oh, see, I got it wrong, it's on Assignment 2. But you
know, normally you'd be calculating the partials of the output, the loss function, with
respect to the inputs. But since the loss function is on Assignment 2, I'm going to leave
that out, and I'm just going to calculate derivatives with respect to this score that
feeds into the loss function. So first we've got the neural network layer, the
non-linearity, and then we're doing this dot product to work out a score for each position
which feeds into the logistic function. So if you want to work out ∂s/∂**b** — so that's
with respect to the bias, first. So the way we do it is, you know, we break up our
equations into

**[32:33]** our individual pieces that are composed together. And so that means we break
this up, so we first calculate the **z** = **Wx** + **b**, then we apply the activation
function to the different components. Okay, then after that, what we remember to do is,
okay, to work out our partial derivatives of *s* with respect to **b**, what we're going
to be doing is doing the product of the partial derivatives of the component pieces. So
we're applying the matrix calculus version of the chain rule. So ∂s/∂**b** =
∂s/∂**h** · ∂**h**/∂**z**

**[33:22]** · ∂**z**/∂**b**, which corresponds to these three layers that are composed
together. And so at that point we remember our useful Jacobians from the previous slide,
and we can just apply them. So the top one, ∂s/∂**h**, is **u**ᵀ — or else, or maybe it's
**u**; let's come back to that, there's a fine point on that that I will explain more
about later. Okay, then for the ∂**h**/∂**z**, that was the activation function, where we
got the diagonal of the derivative of *f*(**z**),

**[34:07]** and then for ∂**z**/∂**b**, that's where we got the identity function. Okay, so
we can simplify that down. And so what that's going to end up as is the **u**ᵀ transpose,
that funny symbol there, times the vector element-wise derivative of *f*. This symbol,
which doesn't normally turn up in your regular math course but turns up all the time in
neural networks, is referred to as the *Hadamard product*. And the Hadamard product means
element-wise multiplication. So it's not like a cross product, where you put two vectors
together and you get out one number, a scalar — you put two vectors together, you

**[34:55]** element-wise multiply them all, and you're left with another vector of the same
type. Okay, so that now gave us our working out of the partials of ∂s/∂**b**. And for a
neural network we want to work out all the other partials as well. So overall here in the
picture we had the **x**, the **W**, the **b** and the **u**, and we'd like to work out
partials with respect to all of those variables so we can change their values and learn,
so that our model predicts better. So suppose we now want to calculate

**[35:43]** ∂s/∂**W**. So again we can split it up with the same chain rule and say
∂s/∂**W** equals the product of these three things. And the important thing to notice is
that two of those three things were exactly the same ones that we calculated before. The
only bit that's different is that at the end we're now doing ∂**z**/∂**W** rather than
∂**z**/∂**b**. And so the first central idea that we'll come back to when we do
computation graphs is: oh, we really want to avoid doing repeated work. So we want to
realize that those two parts of things are the same, and since we're just sort of
multiplying these partial derivatives together, we can just compute what that part is and
reuse it.

**[36:30]** And so if we want to — wait, yeah. Okay, so if we're wanting to calculate
∂s/∂**W**, the part that's the same, this part here, we can refer to as δ. So δ is sort of
the *upstream gradient* or the *error signal* — the part that you've got from sort of
starting at the beginning, ∂s/∂**h** · ∂**h**/∂**z**, this sort of shared upstream part.
We can calculate that once and then we can use it to calculate both of these two things.
And for ∂s/∂**b**, because the ∂**z**/∂**b** just comes out as the identity matrix, the
answer is just δ. But

**[37:16]** for ∂s/∂**W** we need to work out the ∂**z**/∂**W** before we're finished.
Okay, so what do we get for that last piece? So one question you might start off with —
and it's normally a good thing to think about when you're doing assignment problems on
this and other things — the first thing to think about is, what do things look like? Should
the answer be a vector, a matrix, what size should it be, and things like that. So for
∂s/∂**W**: **W** is an *n* × *m* matrix, and *s* is a scalar. So therefore, since

**[38:03]** we have one output and *n* × *m* inputs, the answer according to math should be
that we've got a 1 × *nm* Jacobian — a big long row vector. But here's where things get a
teeny bit tricky, and there's sort of — we end up with this weird mess of math and
engineering convenience. Because immediately what we're wanting to do is, we're wanting to
take our old parameters, which will be stored in the form of matrices, vectors and so on
that we're using as coefficients, and we're going to

**[38:49]** want to subtract from them a fraction of our calculated gradient. So what we'd
like to do is have our calculated gradients in the same shapes as our parameters, because
then we can just do subtraction — whereas if they've turned into a God Almighty row
vector, that's not quite so convenient. So it turns out that what we end up doing is using
something that gets referred to as the *shape convention*, that we reshape our Jacobians so
they fit into things that are of the same shape

**[39:36]** as the parameters that we are using. So we're going to represent ∂s/∂**W** as
an *n* × *m* matrix laid out as follows. And that's a place that one people can get
confused. Okay, so that's what we want to calculate, that kind of matrix. And so that
matrix is going to be δ · ∂**z**/∂**W**. So δ is going to be part of the answer, and then
we want to know what ∂**z**/∂**W** is. And the answer is going to come out like this: so
∂s/∂**W** is going to be δᵀ**x**ᵀ. So it's going to be the product of the upstream
gradient — which was the same

**[40:21]** thing we calculated before for the other two quantities — and then a local
input signal, which is here coming out to **x**ᵀ. Okay, and so we're taking the transposes
of those two vectors, which means that we end up calculating an outer product of those two
vectors, which gives us our gradient. And so why is that the right answer? Well, it kind
of looks convenient, because that's giving us something of the right shape for what I was
arguing we want to find out, and we have the right number of terms. Now, I'm going to rush
through this, so I

**[41:07]** encourage you to read the lecture notes and do this more carefully. But let me
at least a little bit explain why it makes sense. So if you think of one weight in — so
all of these connections are our matrix, right, the matrix is being represented by all
these lines in your network. So if you think of one number in the matrix, so here is
W₂₃ — so it's connecting from input 3, or it's multiplying input 3, to give part of the
answer of h₂. So it's this line here. So for this line here, this weight is being used
only in the calculation of

**[41:52]** h₂, and the only thing it's dependent on is x₃. So that if you're then wanting
to work out the partial of h₂ — or z₂, sorry — yeah, sorry, z₂ — the partial of z₂ with
respect to W₂₃ [Ed: the captions read "with respect to x₃" here; slide 45 derives
∂z_i/∂W_ij = x_j, i.e. the partial is taken with respect to the weight, and x₃ is the
answer], it's sort of depending on these two pieces only. And that's what you're achieving
by working out the sort of outer product like that. Okay, so let me just come back one
more time to this

**[42:39]** sort of question of the shape of derivatives. You know, so I already sort of
fudged it when I was sort of talking about, oh, should I put the transpose there or should
I not, and get a row vector or a column vector. So there's sort of this disagreement
between whether you kind of have the Jacobian form, which is what actually makes the chain
rule work in terms of doing multiplication, versus the shape convention, which is how we
store everything for our computations and makes doing stochastic gradient descent, where

**[43:24]** you're subtracting whatever kind of tensor you have, easy. So this can be a
source of confusion. Since we're doing a computer science course, for the answers in the
assignment we expect you to follow the shape convention. So if you're working out the
derivatives with respect to some matrix, it should be shaped like a matrix with the same
parameters. But you may well want to think about Jacobian forms in computing your answers.
I mean, there are sort of two ways to go about doing this. One way of doing it is to sort
of work out all the math using Jacobians à la Math 51, and at the end just to reshape

**[44:10]** it so it fits into the same shape as the parameters, according to our shape
convention. The other way is to sort of do each stage following the shape convention, but
then you sort of have to be game to reshape things as needed, by doing transposing, to
have things work out at the different stages. Okay, that was my attempt to quickly review
the math. Most people are still here. I will now go on to the second half and go on to how
we do the computation. So most of — yeah, so the famous thing that powers neural

**[44:57]** networks is the backpropagation algorithm. So the backpropagation algorithm is
really only two things. Its invention made people famous because it gave an effective
learning algorithm, but at a fundamental level the backpropagation algorithm is only two
things. Thing one is you use the chain rule — you do calculus of complex functions. And
thing two is you store intermediate results, so you never recompute the same stuff again.
That's all there is to the backpropagation algorithm. And so let's just go through that.
So if we're

**[45:42]** computationally wanting to deal with functions and doing backpropagation, we
can think of them as being represented as a graph. And in some way or another, this kind
of graph is being used inside your neural network framework. So here is a representation
of my little neural network for finding whether the word at the center is a location. So
I'm taking the **x** vector input, I'm multiplying it by **W**, I'm adding **b** to it,
I'm putting it through the non-linearity, and then I'm doing the dot product with my
vector **u**. So that was my computation. And so the source nodes are the inputs in this

**[46:30]** graph, the interior nodes then the operations I do, and so then the edges that
connect those together then pass along the result of each operation. So I passed along
**Wx** to the addition function with **b**, then that gives me **z**, that I pass through
the non-linearity, which gives me **h**, which I then dot product with the **u** to get
*s*. Okay, so I do precisely this computation, and this is referred to as *forward
propagation* or the *forward pass* of a neural network. So the forward pass just calculates
functions. Okay, but then once we've done

**[47:16]** that, what we want to do is then work out gradients so we can do gradient-based
learning. And so that part is then referred to as *backpropagation* or the *backward pass*.
And then we run things backward. So for running things backward we're going to use the
same graph, and we're going to backwards pass along the gradients. And so we start at the
right-hand side and we have ∂s/∂s. So ∂s/∂s is just one, because if you change *s* you've
changed *s*. And then what we want to do is sort of then work further back, so we can work
out ∂s/∂**h**, ∂s/∂**z**, ∂s/∂**b**, ∂s/∂**W**, ∂s/∂**x** as we

**[48:05]** work back. And so this is what we want to work out with gradients. And so how
are we going to do that? Well, if we look at a single node — so for example our
non-linearity node, but any node where **h** = *f*(**z**) — what we can have is an
*upstream gradient*, ∂s/∂**h**, and what we want to do is calculate the *downstream
gradient* of the next variable down, the ∂s/∂**z**. And the way that we're going to do that
is we're going to say, well, let's look at *f* — what is *f*'s gradient? And that's going
to be our *local gradient*.

**[48:53]** And then this is immediately what gives us the chain rule: that ∂s/∂**z** is
going to be the product of our upstream gradient ∂s/∂**h** times the ∂**h**/∂**z**, the
local gradient that we calculate at that node. So downstream gradient equals upstream
gradient times local gradient. Oh yeah, that's what it says when I press that again. Okay,
so this is the single-input, single-output case, though those inputs might be vectors or
matrices or something like that. We

**[49:39]** then have sort of more complex graph cases. So — I think I should have retitled
this slide. Oh yeah, so still — so sorry, so the next case is, for our node it might have
multiple inputs. So this is where we're calculating **Wx**. So in that case we still have
a single upstream gradient, and then what we're going to do is we want to calculate the
downstream gradient with respect to each input. And the way we're going to do that is
we're going to work out the local gradient with respect to each input, and then we're
going to do the same kind of multiplication of upstream gradient times local gradient

**[50:27]** with respect to each input again — chain rule. Okay, so here's a little example
of this. This isn't really the kind of thing you normally see in a neural network, but
it's an easy example. So *f*(*x*, *y*, *z*) is going to be (*x* + *y*) × the max of
*y*, *z*, and we've got current values of *x*, *y* and *z* of 1, 2 and 0 respectively. So
here's our little computation graph. And so for forward propagation we're going to do this
addition, we're going to do this max function, and then we're going to multiply the two,
and that gives us the value of *f*. So we

**[51:13]** can run that with the current values of *x*, *y* and *z*, and this is what we
get. So the max of 2 and 0 is 2, the addition is 3, the answer is 6. Okay, so then after
having done that we run the backward propagation. And yeah, so this procedure is not
actually special to neural networks — you can use it for any piece of math, if you want to
just run your math on PyTorch rather than working it out in your head or with Mathematica.
Okay, so now we work out backwards. So we want to know the local gradient. So ∂a/∂z is
going to be one — sorry, I said that wrong — ∂a/∂x is going

**[52:00]** to be 1. So *a* = *x* + *y*, ∂a/∂y = 1. For the max function, that's going to
depend on which of the two is larger, because it's going to have a slope of one for the one
that's the biggest and zero for the one that's the smallest. And then for the product,
that's like what we saw with vectors: that ∂f/∂a is going to be *b* and ∂f/∂b is going to
be *a*. So those are all our local gradients. And so then we can use those to calculate out
the derivatives. So ∂f/∂f is one. We then multiply that by the two local gradients that
are

**[52:46]** calculated for *a* and *b*, so that gives us 2 and 3, where you're swapping
over the numbers. Then for the max, we're having the one that is biggest — we're taking the
upstream times one, so it gets 3, the other one gets zero. And then for the plus, we're
just sending the gradient down in both directions, and so both of them come out as 2. And
so that gives us ∂f/∂x, so the final function value is 2. ∂f/∂y, we're taking the 3 and
adding

**[53:32]** the 2 — I'll mention that again in a minute — which gives us 5. And then ∂f/∂z
is zero. And we should be able again to quickly check that we've got this right. So if we
consider the slope around *z*, as you change *z* a little — so if we make *z* 0.1, that
makes absolutely no difference to what the computed function value is, so the gradient
there is zero. That's correct. So if I change up the top — if I change *x* a little bit,
if I change *x* to 1.1, then I'll be calculating 1.1 + 2

**[54:22]** is 3.1, and then I'll be taking the max, which is 2, and I'll be calculating
5.1 — and so, wait, no, I did that wrong. Oh, times 2 — wait, I didn't do the
multiplication right. Sorry. Yeah, so we get the 3.1, that's multiplied by 2, that gives us
6.2. So a change of 0.1 in the *x* has moved things up by 2 — so that corresponds to the
gradient being 2. And so then the final case is, well, what if we change *y*? So *y*
started off as 2, and we made

**[55:12]** it 2.1. Then we're going to get 2.1 multiplied by 1 is 2.1 — 61, 6.5 — and right, and
then we've got the 2.1 here, the — oh, sorry, I keep doing this wrong. 2.1 + 1 = 3.1, and
then we've got 2.1 as the max, so we've got 2.1 × 3.1, and that comes out to be 6.51. So
our 0.1 difference has gone up to approximately 0.5 — this is just an estimate — and so
that corresponds to the gradient being 5. We get

**[55:59]** this five times multiplication of our changes. Okay, and so that illustrates the
fact that the right thing to do is, when you have outward branches in your computation
graph and you're running the backpropagation, what you do is you *sum* the gradients. So
for this case we had *y* sort of going into these two different things in our previous
chart, so once we've worked out the upstream gradients we sum them to get the total
gradient. And so that's what we did back here: we had two outward

**[56:44]** things, and we sort of took these calculated upstream gradients of 2 and 3 and
we just summed them to get 5, and that gave the right answer. Okay, and so you can think
about that for just generally how the sort of things to think about as gradients move
around in these pictures. So that when we have a plus operation, that plus just sort of
*distributes* gradient — so the same gradient that's the upstream gradient goes to each
input. When you have a max, it's kind of like a *router* of gradient, so the max is going

**[57:32]** to send the gradient to one of the inputs and send nothing at all to the other
inputs. And when you have a multiplication it's a little bit funky, because you're sort of
doing this sort of *switching* of the forward coefficient — so you're taking the upstream
gradient multiplied by the opposite forward coefficient, and that gives you your downstream
gradient. Okay, so we kind of have this systematic way of being able to forward-pass
calculate the values of functions, then run this backward to work out the gradients heading
down the

**[58:17]** network. And so the main other thing of the backpropagation algorithm is just
that we want to do this efficiently. So the wrong way to do it would be to say, well, gee,
I want to calculate ∂s/∂**b**, ∂s/∂**W**, ∂s/∂**x**, ∂s/∂**u**, so let me start doing those
one at a time and when I've done them all I will stop. Because that means, if you first
calculated ∂s/∂**b** you do all of the part that's in blue, but then if you went on to
∂s/∂**W** you'd be calculating all the part in red, and — well, just as we saw in the math
part when we were doing it as math,

**[59:02]** these parts are exactly the same. You're doing exactly the same computations. So
you only want to do that part once, and work out this upstream gradient or error signal that
is then being shared. So the picture that we want to have is you're doing together the
shared part, and then you're only sort of doing separately the little bits that you need to
do. Okay. Boy, I seem to have been rushing through today, and I'm going to actually end
early unless anyone is going to slow me down. But I did have just a few more slides to go
through. Yeah, so the sort of

**[59:49]** generalization of this as an algorithm is — in the general case we normally have
these sort of neural network layers and matrices, which you can represent as vectors and
matrices, and it's sort of nice and clean and it looks like doing that in calculus class. I
mean, strictly speaking that isn't necessary. So the algorithm for forward propagation and
backward propagation that I've outlined, you can have it work in a completely arbitrary
computation graph, providing it's a DAG that doesn't have cycles in it. So the general
algorithm is, well, you've got a whole bunch of

**[1:00:35]** variables that depend on other variables. There's some way in which we can
sort them so that each variable only depends on variables to the left of it — so that's
referred to as a *topological sort* of the outputs. And so that means there's a way we can
do a forward pass where we're calculating variables in terms of ones that have already been
calculated. But if we want to have some extra wonky arrangement, so it's not like nice
matrix multiplies or anything, we're totally allowed to do that. Or we can have things not
fully connected — right, there's no connections across here. We can have an arbitrary
computation graph. And so that gives us our

**[1:01:20]** forward propagation. And then once we've done the forward propagation we can
initialize the output gradient as one, and then we're going to visit the nodes in reverse
order, and for each node we're going to compute a gradient by using the upstream gradient
and the local gradient to compute the downstream gradient. And so then we can head back
down the computation graph and work out all of the downstream gradients. And so the crucial
thing to notice is that if you do it correctly, working out the gradients has the same
big-O

**[1:02:09]** complexity as working out the forward calculation. So that, in big-O terms,
you might have different functions depending on what the derivatives are, but in big-O
terms, if you're doing more work in the backward path than you're doing in the forward
path, that means that you're somehow failing to do this efficient computation and that
you're recomputing some of your work. Okay, so because we have such a good algorithm here,
you should be able to just work out the backward path automatically. And that gets referred
to as *automatic differentiation*. So if you had the symbolic form of what

**[1:02:59]** you're calculating with your forward pass, you should just be able to say, yo
computer, can you work out the backward pass for me? And kind of mathematically, it could
look at the symbolic form of all of your functions, work out their derivatives and do the
entire thing for you. So early on there was a pioneering deep learning framework, Theano,
principally from the Université de Montréal, which attempted to do precisely that: that you
had the entire forward path computation stated in symbolic form and it just did the entire
thing for you and worked out the

**[1:03:47]** backward pass automatically. But somehow that sort of proved to be too
heavyweight, or hard to deal with different things, or people just like to write their own
Python, or whatever it is. So that idea did not fully succeed. And so what in practice all
of the current main frameworks have fallen back on is something that's actually less
automated than that. So it's sort of like we've gone backwards in time — but the software's
got a lot better, really, it's a lot stabler and faster. So all of the modern deep learning
frameworks sort of say, look, I will manage the computation

**[1:04:34]** graph for you, and I can run the forward propagation path and the backward
propagation path, but you're going to have to work out the local derivatives yourself. So
if you're putting in a layer, or putting in a function like an activation function in a
neural network, your Python class that represents that — you're going to have to tell me
what the forward computation is and what the local gradient is, and I'm just going to call
your local gradient and assume it's correct. So there's a bit more that has to be done
manually. So the part

**[1:05:19]** that's automated then is that — you know, not precisely this code obviously,
but roughly — inside the deep learning software it's computing with a computation graph and
it's got a forward and a backward, and it's doing what I presented on the pictures before.
So for the forward pass it's topologically sorting all the nodes of the graph, and then
it's going through them, and for each node in the graph it's calling its forward function,
which will be able to compute its local value in terms of its inputs, which have already
been calculated because it's topologically sorted. And then it's

**[1:06:06]** running the backward pass, and in the backward pass you're reversing your
topological sort, and then you're working out the gradient, which is going to be the
multiplication of the upstream error signal times your local gradient. And so what a human
being has to implement is that, for anything — whether it's a single gate, here's a
multiply gate, or a neural network layer — you have to implement a forward pass and a
backward pass. So here for my baby example, since we're just doing multiplication, my
forward pass is that I just multiply the two numbers and return it. So I'm specifying that
for the local node, and

**[1:06:52]** then the other part is that I have to work out those gradients. And well, we
sort of know how to do that, because that's the examples that we've been doing here. But
notice that there's sort of a trick. For what I've got now, you kind of can't write down
what the gradients are, because backward is just taking as an input the upstream gradient,
and you can't work out what the downstream gradients are going to be unless you know what
function values you're calculating it at. So the standard trick — which is how everyone
writes this code — is you're

**[1:07:37]** relying on the fact that the forward is being calculated before the backward.
And so your forward method shoves into some local variables of the class what the values of
the inputs are, and then you have them available. So when you get to the backward pass you
can do what we did before: that the *dx* is going to be the upstream error signal times the
opposite input, and similarly for *dy*, and that's going to give us the answer. Okay, just
two last things then to mention. Yeah, so doing this, you need to

**[1:08:25]** get the math right for what's the derivative of your function, so you get the
right backward calculation. So the standard way to check that you've got the right backward
calculation is to do manual gradient checking with numeric gradients. So the way you do
that is sort of like for the couple of examples I did when I said, oh, let's check it by
going from 1 to 1.1, what should the slope be approximately. We're going to do that in an
automated way, and so we're going to say, at the value *x*, let's estimate what the
gradient should be. And the way to do that is to pick a small *h* —

**[1:09:12]** there isn't a magical number, because it depends on the function, but typically
for neural networks around 10⁻⁴ is good — a small *h*, and work out the function value, the
forward part, at *x* + *h* and *x* − *h*, divided by the run, which is 2*h*. And that
should give you an estimate of the slope, what the backward pass is calculating, and you
want those two numbers to be approximately equal — you know, within some 10⁻² of each other
— and then probably you're calculating the gradient right. And if they aren't equal, you
probably have made a mistake. Yeah, so

**[1:09:59]** note that this formula — the version I did for my examples, I just compared
*x* with *x* + *h*, right, I did a one-sided estimate, which is normally what you get
taught in a math class. If you're doing this to check your gradients numerically, you're
far, far better off doing this two-sided estimate, because it's much more accurate and
stable when you're doing it equally around both sides of your *h*. Yeah, so this looks easy
to do. If this was just so good, why doesn't everyone do this all the time and forget about
calculus? The reason you don't want to do this is that doing this is incredibly slow,

**[1:10:46]** because you have to repeat this computation for every parameter of your model,
so you're not getting the kind of speed-ups you're getting from the backpropagation
algorithm. But it's useful for checking your implementation is correct. In the old days,
before frameworks like PyTorch, we used to write everything by hand and people often got
things wrong. But nowadays it's less needed — but it's good to check that, if you've
implemented your own new layer, that it's doing the right thing. Okay, so that's everything
that we need to know about neural nets. Backpropagation is the chain rule applied
efficiently. The forward pass is just function application; the backward pass is

**[1:11:33]** the chain rule applied efficiently [Ed: the captions read "inefficiently"
here, which inverts the claim; slide 84 states "Backpropagation: recursively (and hence
efficiently) apply the chain rule"]. So we're going to inflict pain on our students by
making them do some math and calculate some of these things and do the homework, and I know
that'll be harder for some of you than others. In some sense you don't actually need to
know how to do this — the beauty of these modern deep learning frameworks is they'll do it
all for you. They predefine common layer types and you can just plug them together like
pieces of Lego and they'll be computed right. And this is precisely the reason that high
school students across the country and the world can now do deep learning projects for
their science fairs: because you don't actually

**[1:12:20]** have to understand any of this math, you can just use what's given to you. But
we kind of want to hope that you actually do understand something about what's going on
under the hood and how neural networks work. So therefore we make you suffer a little bit.
And of course, if you're wanting to look at and understand more complex things, you need to
have some sense of what's going on. So later on when we get on to recurrent neural networks
we'll talk a bit about things like exploding and vanishing gradients. And if you want to
have some understanding about why things aren't working and things are going wrong, then
you sort of want to know what it's actually calculating, rather than just thinking it's all
a black box

**[1:13:07]** magic. And so that's why we hope to have taught something about that. Okay, I
think I'm done, if the audience is sufficiently stunned, and we can stop for today. Okay,
thank you.
