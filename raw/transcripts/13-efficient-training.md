---
title: Efficient Training
lecture: 13
video: https://www.youtube.com/watch?v=UVX7SYGCKkA
source: edited from auto-generated YouTube transcript
verbatim: false
original: original/13-efficient-training.md
slides: ../slides/13-efficient-training.md
---

# Efficient Training — transcript

Copy-edited for punctuation, sentence boundaries, and mis-heard terms; cross-checked against
`raw/slides/13-efficient-training.md`. The verbatim auto-generated captions are kept at
`raw/transcripts/original/13-efficient-training.md`. Lecturer is Shikhar Murty. Student
questions and comments from the floor are set in *italics*. Timestamps mark the start of each
paragraph; all 81 are preserved in order.

**This is an edited transcript.** The captions had no punctuation and mangled most of the
systems vocabulary this lecture is built on: *Adam* arrived as "adom"; *bfloat16* as "B float
16", "B FL 16" and "b flat 16"; *FSDP* as "Fs DP" and "fsdp"; *shard* / *sharded* / *sharding*
as "shot", "shoted", "shouted" and "Shing"; *LoRA* as "Laura" and "Lura"; *batch size* as "bat
size", "bath size" and "bad size"; *DistilBERT* as "dist bir"; *GradScaler* as "grad scalar";
*autocast* as "AutoCast"; *reduce-scatter* as "reduce CER"; *layer-j* as "ler"; *gradients* as
"Radiance"; *PyTorch* as "pyos"; *GPT-3* as "gpt3" and "GPD 3"; *BitFit* as "bit fit"; *SVD* as
"SD"; *NaNs* as "n"; *FP16* as "fp6" in one place; *A100* as "a 100" and "A1 100"; *Ampere* as
"Amper"; *autodiff* as "Auto diff"; *efficient adaptation* as "f adaptation"; and *cache* /
*caching* as "cash" / "cashing". Terms were restored from context and checked against the
slides. **No content was added, removed, or reordered.**

**One number was corrected against the slides** rather than kept as heard, and is flagged
inline: at 4:47 the captions give the rounding example as "1.1 gets converted to one", but
slide 10 writes it as **1.0001 is 1 in half precision** — and 1.1 is in fact representable in
FP16, so the caption is wrong rather than the lecturer. Every other number is as spoken,
including the ones he rounds off the slide (slide 22's 24.59 min read as "about 25 minutes",
6.10352e-05 as "6e-5").

**Two spoken figures are not on the slides**, and are marked inline rather than silently
reconciled:

- At 41:12 he puts the carbon cost of training GPT-3 at "1.1 million tons", hedging it himself
  as "or some such number". Slide 54 gives no tonnage at all — it says only that Cornell
  scientists in 2021 estimated the emissions equivalent to *running a coal power plant for 10
  straight hours*, which is the part he then states correctly.
- At 29:37 he describes the FlatParameter example as "14 parameters plus some extra padding".
  Slide 43 draws 15 elements (a 12-entry weight matrix indexed 0–11 plus a 3-entry bias indexed
  12–14) and one padding slot, sharded across 16 ranks.

**Where the source is still unreliable**, the text carries an inline `[Ed:` note rather than a
silent guess. There are five, at 11:49, 12:36, 21:06, 21:56 and 54:25 — all of them student
questions or interjections picked up away from the microphone, plus the LoRA weight matrix at
45:51, whose subscript the captions destroy but slide 59 records as $W_0$.

---

**[0:06]** Okay, cool. Let's just get started. Welcome everyone to lecture 12. So far we've
learned a lot about how we convert words into vectors, how we convert sentences into vectors,
and basically take actions in the real world using that — so, classify documents. We learned
about Transformers, we learned about pre-training. Today is going to be a little bit different.
I'm going to be talking about how you can train large models on GPUs, and a few basics about
how these ML systems work. It has nothing to do with natural language at all, but hopefully
it's going to be useful for final

**[0:51]** projects. So I'm going to spend some time on mixed precision training, some time on
multi-GPU training with DDP and FSDP — and hopefully by the end of the lecture these terms will
make sense — and some time on parameter-efficient fine-tuning. But before we get into the
lecture, just some announcements. Proposal grades are going to be coming out shortly,
hopefully by the end of the day. Thank you so much for all the hard work. I know it's kind of
getting a little bit crammed with a lot of deadlines, for assignment 4 and the project
proposal, so thank you so much for all your hard work. The other thing is the project
milestone: details should be out shortly, if not

**[1:37]** already out on the website. It's worth 5% of the overall grade, it's due 12 days
from now, and it's a maximum of two pages. Really, the way to think about the milestone is to
use this as a forcing function to get work done for your final project. And yeah — with that
out of the way, let's just jump into the material. So I'm going to start by thinking about how
parameters and gradients, and generally numbers, are represented in computers. And I promise
it's going to be relevant to deep learning pretty soon. So let's start with floating point. How
many people here are familiar with this cartoon depiction of FP32? Can you just — okay, so some
of you. So yeah, let's

**[2:25]** kind of recap how floating points are represented in computers. So firstly, FP32 —
that's 32 bits, so the memory requirement is 4 bytes. Okay? And so if you're thinking about
neural networks, then for every single neural net parameter you need four bytes of GPU memory.
And the way to convert this cartoon into a real number is something like this. So the first bit
there is the sign, and then the stuff in green represents the range, and then the stuff in blue
represents precision. And for FP32 you can represent a pretty large range, and it's fairly
precise. And so the larger the

**[3:10]** stuff in green is, the more numbers you can represent — which means more smaller
numbers, and also larger numbers. And the more stuff in blue we have, the greater the precision
in representing actual numbers. So another popular data type, that takes half the memory of
FP32, is FP16. And the way we reduce memory is we're going to reduce the stuff in green, so
there's going to be less range, less dynamic range, and also the stuff in blue, which means
it's going to be less precision. But the good thing is that we can save memory — so we slash
memory requirements in half. So let's think of a scenario

**[3:59]** where you're trying to train a big neural network and your model parameters and
gradients are represented in FP32. You start training, and suddenly you get an out-of-memory
CUDA error. And so, just based on what we've seen so far, one possible solution is you cast
everything into FP16. And if you do that, you reduce memory usage by half. So let's kind of
work through what are some possible problems with doing something like that. So, like I said,
because there's less stuff in green there's going to be less range, and so that means a lot of
very small numbers will get converted to zero, and a lot of really large numbers will get
converted into NaNs.

**[4:47]** And there's also less precision, because you have fewer bits in blue, which means
you're going to get rounding errors. For example, 1.0001 gets converted to one in half
precision. [Ed: the captions say "1.1"; slide 10 writes the example as "1.0001 is 1 in half
precision", and 1.1 is representable in FP16.] And I have a little screenshot of how you can
test various properties of data types. So basically the things to look at are the epsilon —
the epsilon is like the smallest number such that if you add that to one you don't lose any
precision; if you add a number that's smaller than epsilon to one, that gets just rounded down
to one — and the smallest normal is the smallest number that can be represented in FP16.
Anything smaller than that, it goes

**[5:33]** straight to zero. And for neural network training, if a lot of small numbers get
rounded down to zero, that's actually not good. So here's a diagram that I took from an NVIDIA
blog post that's just showing some gradients during the course of training, and more than half
of these gradients will literally just get set to zero in FP16, which is kind of a problem. And
that has to do with the range of FP16. And the second problem is with precision — right, so we
have basically less precision, and so our updates are not going to be precise. So, the
solution. Here's one

**[6:19]** possible solution. So we're going to use FP16, but we're also going to use FP32.
That's sort of the high-level idea. And what we're going to do is, we're going to maintain a
copy of the model in FP32 — and let's call those master weights — and then you get a little bit
of data, you run a forward pass. And when you run your forward pass, you run it by converting
from FP32 into FP16. And then you run a backward pass and get your gradient in FP16. So
everything so far has happened in FP16. Then you take your gradients, upcast them into FP32,
and then update your master weights; and then once you update your master weights, you copy
them into the FP16

**[7:04]** version of the neural network. Okay, so this seems like a reasonable scheme. You
know, I'm using FP16 on my GPU, but I have the full 32-bit precision also lying around
somewhere, so I can have more precise updates. Can someone tell me why this is still
problematic? Any guesses? Yeah. *One would be, like, really slow, because you have to copy the
32-bit versions from GPU into, like, some—* Yeah, so that's a good point. So you can often
overlap I/O with forward and backward passes, so practically this is not a problem. But

**[7:50]** yeah, that's a good point — potentially, if your network is very, very small, this
would be a problem. Yeah. *Gradients are usually fairly small, like individual gradients are
usually fairly small, and when you copy the FP16-computed gradients onto FP32 you may be
sending your network somewhere else where you don't want it to.* So yeah, that's pretty much
the right answer. So let's kind of go back to this diagram that we had. So this shows gradients
in the backward pass, and I said that we're going to compute all our gradients in FP16. What's
going to happen? Most of them will just get converted to zero, which is something that we
really would like to avoid. So here's a possible solution. What you can do is, you get your
batch of data, you run your forward

**[8:35]** pass in FP16, you compute your gradient — but then when you have the — sorry, so, we
here. So you get a batch of data, you compute a forward pass in FP16, you get your loss, you
scale the loss by some large value — okay, let's say 100, let's say a thousand — and then you
compute gradients, and now you just scale your gradient by a large number. And so everything
that we had on the left-hand side of this red line just gets shifted to the right, and hopefully
there's less stuff that will get rounded down to zero. And then compute your gradient in FP16,
copy it into FP32, and then divide by the scaling factor, and then you update your master
weights. So

**[9:22]** this will solve both the problems that we talked about. And so this is basically what
we call mixed precision training. And it's relatively simple to implement this in PyTorch. All
you have to do is, you need to instantiate this GradScaler object, and then within the context
of this autocast you want to run your forward and backward passes, and then scale down your
gradient, and then update your model parameters. But then this seems a little complex — you
know, we have to deal with scaling the loss and then scaling it back down. What if you
multiplied it by 10,000 and that leads

**[10:08]** to NaNs? And so then you have to kind of scale — you have to update your scaler, you
have to, in the next iteration, multiply by a thousand, and you have to kind of adjust to
network dynamics. So we'd like to not do gradient scaling. So can we do something better? So the
reason why we have to do the scaling is — just recall the role of the bits in green. That kind
of tells you what is the dynamic range of the data type, and we needed scaling because FP16 has
a much smaller range compared to FP32. And so because of that, FP16 cannot represent very small
numbers. So how do we solve this? Any ideas?

**[11:03]** Yeah. So here's the problem. In FP16, because you have fewer bits for the exponent,
you can't represent very small numbers. So if you have something that's smaller than, I don't
know, 6e-5, it gets rounded down to zero, and that's because of the dynamic range of FP16. So
how do you solve that? *Sacrifice precision, so have more green.* Absolutely, yeah, so that's
the right answer. So what we're going to do is, we're going to sacrifice precision. So that's
the idea for bfloat16, which stands for brain float 16. So you're going to have exactly the
same number of bits for representing the range — so that's going to be eight bits,

**[11:49]** so it has the same dynamic range as FP32 — but a lot less precision. And it turns
out that this is okay for neural network training. And now if you use bfloat16 you don't need
to use grad scalers any more; it's as simple as wrapping your model forward pass and backward
pass within the right context. The one caveat about bfloat16 is that it's not available on all
GPUs, so you need to have the latest Ampere NVIDIA architectures, which the H100s, the A100, the
A6000 have. But if you have an older GPU then you might not be able to utilize bfloat16.
*Sorry, can you [Ed: question inaudible] why having less precision but the same amount of
bits—* Um, yeah, so

**[12:36]** it's [Ed: garbled — "it's B 6 and"] — oh, never mind. Sorry, I'm — so here are some
results. So someone fine-tuned DistilBERT for sentiment classification on a single A100. At the
very top is float64, which is a really, really rich 64-bit representation of floating points. It
takes about 25 minutes, and you get a pretty high accuracy, but it also takes a lot more memory.
And all the way down, we're using mixed precision training with bfloat16, and now we have
reduced training time by roughly a third, more or less have the same accuracy — a little

**[13:22]** bit better, actually, because there's some regularizing effect from the half
precision representation — and then a lot less memory. And the reason we see speedups for
training is because matrix multiplies tend to be faster when you are multiplying in half
precision. So, before we move on, are there any questions about this? Okay, cool. So let's keep
going, and let's sort of change the setting. So now we have more than one GPU, now we have
multiple GPUs, and we want to train a network on all of the multiple GPUs that we have. So

**[14:10]** let's start with some basics. So here's a cartoon showing basically a model and an
optimizer receiving some data from a dataset. And let's kind of work through what's stored on
GPU VRAM. And this is going to be somewhat of a lie, and I will point out what my lie is soon.
But just to keep things simple: we have the neural net parameters — so let's say we're doing
mixed precision training, and so it's stored in FP16 — and then we have an optimizer. And when I
first saw this a few years back, I was very surprised to see that optimizers also need memory.
But if you're using

**[14:57]** something like Adam, then you need to store the Adam momentum term and the Adam
variance, and every time you get a gradient you have to update Adam momentum and variance, and
that's what you use for updating your parameters. And because you're using mixed precision
training, these have to be represented in FP32. So that's what the picture looks like if you
have a single GPU. Now let's say we have multiple GPUs. And what we'd like to do is first divide
our dataset into — let's say we have four GPUs, right — so we'll divide our dataset into four
parts, and we'll maintain a synchronized copy of the model, and every model receives its own
slice of the dataset.

**[15:43]** In the beginning we have a synchronized model and everyone has their own copy. We
run a forward pass. So this forward pass receives different data points, and so every model is
going to have different activations, and correspondingly every model is going to have different
gradients. So you run a backward pass, every model has a different gradient because there's
different data points, and then we're going to run a synchronization step. And what
synchronization is going to do is communicate gradients between different workers. And so I'm
going to introduce the first MPI primitive in this lecture, and that primitive is called the
all-reduce operation. What all-reduce does is it takes

**[16:31]** four pieces of information — in this example, on four different GPUs — it sort of
merges everything together and then distributes it to all of the GPUs. And the communication
overhead of doing that is two bytes per parameter, because remember we have FP16 gradients. So
two bytes per gradient, and then this needs to be communicated, and so the overhead is 2 bytes
per parameter. So that's the all-reduce operation. And then once gradients have been
communicated — so they have to be communicated by sort of gathering on one worker and just
distributing the cumulative gradient — at that point every optimizer has the

**[17:17]** full gradient, and then the optimizer can update the model so that you maintain
synchronization. So that's the basic — that's known as distributed data parallel. That's good,
but it turns out that it has really poor memory scaling. So let's kind of go through our math
for how much memory is needed. So we have the model parameters — that's FP16, because we're
doing mixed precision training — and then for the gradient we also have the gradient in FP16, so
two bytes for the gradient. And then we have the stuff in green. The stuff in green is — let's
say we're doing Adam, so we need to — well, we need to store the master weights

**[18:03]** regardless of whether we're doing Adam or not, and then we need to store the
momentum and the variance. So that's 12 extra bytes per parameter. And this needs to be stored
on every single GPU. And so the question is, can we do better than this? And so now things are
going to get a little bit more tricky, so if you have questions just stop me and we can go from
there. So the way we're going to improve our memory scaling is, we have a set of techniques that
are together known as ZeRO — that stands for zero redundancy optimizer. So this was a set of
techniques

**[18:48]** released by Microsoft as part of their DeepSpeed project. And the idea is going to
be that, instead of having every GPU contain all of this state — and by the state I mean the
stuff in blue, the stuff in orange and the stuff in green — you're going to sort of shard it. So
there are going to be shards, so that not every GPU has all of the parameters or all of the
gradient, but by communication they can synchronize. So that's pretty much what the sketch for
this is going to look like. So let's look at stage one. So ZeRO has multiple stages — so there's
stage one, two and three. In stage one we're going to shard the stuff in green. So the stuff

**[19:33]** in green was the optimizer state. And so the way we're going to shard and still
maintain synchronization is something like this. So every GPU has the full set of parameters in
FP16, and every GPU has its gradient for its data, but it only has a sharded copy of the full
optimizer state. And the other requirement is that every GPU is responsible for updating the
parameters corresponding to its own shard. So if you go step by step, this is what it looks
like. Every GPU has its own data, every GPU gets a gradient on its subset of the data, then we
perform

**[20:20]** a reduce-scatter. So now this is the second MPI operation of the lecture — so we've
done all-reduce, this is the second one, this is called reduce-scatter. What a reduce-scatter
does is: every GPU has the full gradient on its data, and what you want to do is you want to
take the chunk corresponding to, let's say, GPU 1 — so let's say you're GPU 0 and you've
computed the full gradient for all the parameters, and you want to communicate the chunk for
GPU 1 to GPU 1, and same for GPU 2 and 3. So what you're going to do is, from the full gradient,
just communicate the bits that a different worker wants to that worker. And every GPU has to do
that. So that's called a reduce-

**[21:06]** scatter. And then once every worker gets the gradient corresponding to its shard,
they're going to update its parameters. And then once they have updated their shard, they're
going to perform an all-gather. So what that means is: let's say you have a neural network with,
let's say, eight parameters, two parameters on each GPU. At the end of this, each GPU has
updated their subset of parameters, and then they're going to do an all-gather to maintain
synchronization, so every GPU gets the full set of parameters that are all updated. *Yeah — [Ed:
question largely inaudible] is maintaining this and you're not merging them together in that
way, what is — what makes this more*

**[21:56]** *efficient?* Sorry, could you repeat your question? *Can you go over why this is
better than doing the [Ed: garbled — apparently "the previous one"]?* Right. So what we're going
to do is shard the optimizer state. So let's say, in a running example, we have a neural network
with eight parameters. Earlier, we needed the optimizer state for all of the eight parameters;
now every GPU has to maintain optimizer state for only two parameters. So after the
reduce-scatters are done, you have the full gradient corresponding to just two parameters. So
the optimizer state is just the gradient for two parameters; the model is going to update only
two

**[22:42]** parameters using the partial optimizer state. *But you have to have the entire set
of parameters to run?* So you'll eventually get the rest of the parameters back. So you have the
entire set of parameters, you have all the stuff in blue, and you have the full gradient for
your subset, but you don't have the full optimizer state. So what you can do is, you can only
update the parameters for the bits of optimizer state you have. So in the running example that I
just made up, GPU 0 updates two parameters, GPU 1 updates two parameters, and so on. And then
they communicate updated parameters to maintain synchronization. More questions

**[23:27]** about this? Okay, so let's keep going. So so far we have looked at three MPI
operations: we looked at all-gather, we looked at reduce-scatter, and we looked at all-reduce.
So it turns out that all-reduce is actually equivalent to running a reduce-scatter followed by
an all-gather operation. And just recall that for DDP, all we had to do was this all-reduce
operation, and we computed what's the communication overhead of that. And it turns out that when
you're doing this optimizer state sharding, you have to do exactly the same amount of
communication overhead, just

**[24:14]** because an all-reduce is equivalent to a reduce-scatter followed by an all-gather.
And so we basically saved memory for free. So you should just always use this, because you're
going to get memory savings and you don't have any additional communication overhead. So, we're
happy, we saved memory, and now we want to shard even more things. So let's start doing ZeRO
stage two. And now, along with sharding the stuff in green, which was my optimizer state, I'm
also going to shard gradients. And now this is going to be a little bit more complex, because we
kind of still need the full gradient for the worker's data slice. But

**[25:01]** each GPU only has enough memory for instantiating the gradient for a small subset of
parameters. So how are we going to deal with that? So we are actually never going to instantiate
the full gradient vector, and then whenever a GPU gets a gradient in the backward pass, you
instantiate a vector temporarily for the parameter for which you just received a gradient, and
then compute the gradient, and then just send it to the right worker, and then you destroy the
memory that you just created. That's kind of the sketch. And let's kind of go through this step
by step. So we have four workers. Each worker performs a backward pass, and

**[25:49]** the backward pass happens layer by layer, right? So recall the lecture on autodiff.
So you have the loss, and then you have this backward pass layer by layer, you compute
gradients. So now let's say you're at layer j. You take the upstream gradient, you compute the
gradient for the parameters at layer j. Immediately, the moment you compute those gradients,
send it to the right worker. So there exists some worker that is responsible for layer j. And
what's going to happen is, every GPU that's just computed the gradient for layer j for its data
slice sends it to the right worker. And then the moment you've done that, you deallocate this

**[26:34]** memory that you just created. And so this is kind of a fourth MPI operation, but
really not very different from a reduce-scatter — this is just a reduce. So there are four GPUs
that have a gradient, and then they just have to communicate it to whoever is responsible for
maintaining the gradient for that layer. And then, yeah, so there exists some worker that is
responsible for a given layer; they're going to update its parameter shard using the full
gradient that it received via this communication, along with the optimizer state. And then, at
the end, to synchronize everything, you have to perform an all-gather as

**[27:21]** before. Any questions about this high-level sketch? Okay, so let's keep moving.
Okay, so recall that for ZeRO stage one, it was basically free, because it turns out that an
all-reduce is equivalent to a reduce-scatter plus an all-gather. And we're kind of doing the
same thing here: we have a reduce followed by an all-gather, so this is practically also for
free. So we've gotten away with saving memory without any communication overhead compared to DDP
so far. So let's

**[28:06]** keep going. Let's try and see if we can shard even more things. And I think someone
sort of alluded to this in the audience early on: so what happens if you shard even your model
parameters? So let's say you run into a situation where — forget about the optimizer state —
even your model wouldn't fit on a single GPU. And so in that case, what you do is you split up
your model, so you split up your model across all the different GPUs. So you shard your model
parameters, which is the stuff in blue. But the caveat is that now we're not going to get memory
savings for free; there's going to be some communication overhead. And this is ZeRO stage three.
This is the final stage of ZeRO. This is

**[28:52]** also known as FSDP, fully sharded data parallel, for anyone who's heard that term
before. And here's the high-level sketch. And I feel like this is kind of the easiest to
understand compared to ZeRO stage one and two, just because there needs to be communication at
every step of the way — you can't get away without communicating. So the first thing we're going
to do is, we're going to take our model and we're going to convert the entire model into FSDP
units. So here's a sketch: a simple deep neural network. I'm going to convert that into multiple
FSDP units — three FSDP units

**[29:37]** here. So that's just a data structure, an FSDP unit. We've not done anything so far.
And then I have this FSDP unit, I'm going to convert this into another data structure called a
flat parameter, and then I'm going to assign a subset of these parameters to every single GPU.
So here we have 16 GPUs and a flat parameter consisting of 14 parameters plus some extra
padding, so that things divide properly, and I'm going to assign each parameter to a distinct
GPU. [Ed: slide 43 draws 15 elements — a 12-entry weight matrix indexed 0–11 plus a 3-entry bias
indexed 12–14 — and one padding slot.] And so that's basically just a complex way of saying that
we created some data structures and we just divided up model parameters to every GPU. So every
GPU gets a

**[30:24]** subset of model parameters. Okay, now let's start thinking about what my forward
pass would look like. So there's no GPU that has the full set of parameters. So you're running a
forward pass, let's say you're at layer 4. Now there's no GPU that has all of layer 4, so you
have to communicate. So we need to do an all-gather operation — that's the operation that we did
to accumulate multiple things that are on multiple GPUs so that every GPU has the full thing. So
you perform an all-gather so that you have all pieces of layer four, you run a forward pass. And
now you don't need layer four, so you now discard your parameter shards. And now you have to run
your

**[31:11]** backward pass, right? So you computed your loss and now you have to do a backward
pass. Again, let's say you are back at layer four. You have your upstream gradient, you don't
have layer four, so you need to do another all-gather, so you get all the parameters of layer
four. And then you run a backward pass for layer four, you compute the gradient for your subset
of parameters — so recall that every GPU has different data points, right, so there's going to
be different gradients for every GPU. So then, for layer four, you do an all-gather, get all
parameters, compute the gradient, every GPU has different gradients, and then you have to do a
reduce-scatter so that you can send the full gradient to the

**[31:56]** GPU that's responsible for whatever parts of layer 4 that you're sending. So yeah,
so that's basically full FSDP. And then once you sort of run the forward and backward pass, then
each GPU will update its own parameter shard using the full gradient that it received just now,
and then you do a synchronization. So let's kind of do a quick review of everything we've looked
at so far. So there was DDP, which was: you don't shard anything, you have the full model, full
gradient, the full optimizer state on every single

**[32:42]** GPU, and all you're going to divide up is the full dataset. So you have a big dataset
of 1,000 examples, every GPU gets 250 examples. And then you compute a forward pass and a
backward pass, every GPU has a different gradient, you need to communicate that gradient, and
then you synchronize. And so that was called an all-reduce operation in MPI terms. And then we
looked at ZeRO, which is: now we want to save some memory, we don't want the full memory
requirements of models, gradients and optimizer state on every single GPU. And in ZeRO stage one
we sharded the optimizer state, so that you don't have to maintain the full optimizer state for
every GPU — you kind of break that down between all

**[33:29]** the different GPUs that you have. And we saw that the communication overhead of
maintaining synchronization in ZeRO stage one boiled down to basically just doing an all-reduce,
through this identity that says that an all-reduce is a reduce-scatter plus an all-gather. And
we save memory for free with ZeRO stage one and two, so you should just do it. And then, with
ZeRO stage three, things got a little bit more complex, because you have to divide up your model
parameters, the optimizer state and the gradient. And so while you're running your forward pass,
you kind of have to do some communication to get the full parameters for any layer — for layer
four, in our example — and then you also have to do an all-

**[34:15]** gather in the backward pass so you get the full gradient, and then you have to do a
reduce-scatter so that you can send the full gradient for whatever chunk of the parameter to the
right GPU. And overall that's two all-gathers plus a reduce-scatter, so that's a lot more
overhead than stages one and two. But if you don't have enough GPU VRAM so that you can even
load your model onto a GPU, then this is kind of what you have to do. All right. Any questions
about MPI primitives, or stages of ZeRO, or FSDP? Okay, cool. So I'm going to fix the lie

**[35:02]** that I said earlier about the GPU VRAM calculation. So I said that there's just model
parameters and gradients and the optimizer state, but there's this final thing, the model
activations. So, you know, we've all seen that as you want to increase the batch size, there's a
point when the GPU says that it can't fit more stuff, and that's because you also need to store
model activations in the backward pass, right? And that scales linearly with the batch size. So
the larger the batch size, the more the number of model activations that need to be stored. And
by the way, if you're doing mixed precision, this is in FP16 or BF16, but it scales with the
batch size. And

**[35:50]** so that's sort of the other thing that you have to think about. And none of the
techniques that we've looked at so far help with sharding model activations. So okay, so we
looked at a bunch of basics of multi-GPU training and floating point, but it kind of boils down
to this very simple flowchart, which you can use for your final projects when you're fine-tuning
models. So the first thing is: always use mixed precision training. You barely ever see a hit in
performance — and by performance I mean generalization, or F1 or accuracy. And if you're using
the newer

**[36:35]** Ampere architectures, the H100s or the A100s or the A6000, always use bfloat16 —
it's just better. And you can check that with that torch command. Okay, so always use mixed
precision training. Now, ask yourself this question: does batch size one fit on a single GPU? If
it fits, try a larger batch size. Batch size one is too small — try a larger batch size and/or
use ZeRO stage two. ZeRO stage two is for free; just use ZeRO stage two and increase your batch
size. If you can't fit even batch size one, then you have to see if ZeRO stage three fixes your

**[37:22]** out-of-memory issues, because now you're going to shard your model parameters. And
all of this is in the context of full fine-tuning, right? So I'm fine-tuning all of my model
parameters. Sometimes the answer to that question is also no. So you can't full fine-tune your
model on four, whatever, A100s, or A6000s, and you've tried ZeRO stage three, you've tried mixed
precision training, you have a batch size of one, maybe you did gradient checkpointing,
activation checkpointing, and nothing works. And so now, basically, you can't do full
fine-tuning, and so the thing to do is to try parameter-efficient fine-tuning. And

**[38:08]** that's going to give you a lot more memory savings. So let's talk about
parameter-efficient fine-tuning. So why is it called parameter-efficient fine-tuning? So in full
fine-tuning, you run a forward pass and a backward pass and you update every single model
parameter; and in parameter-efficient fine-tuning you're only going to update a small subset of
the full set of parameters. And why would you want to do that? So maybe you're in a setting
where you cannot full fine-tune even with batch size one — you've tried all the tricks possible,
it just wouldn't fit — and so maybe you have to do

**[38:53]** parameter-efficient fine-tuning. Maybe the other possible reason why you want to do
it is kind of slightly more scientific. You know, these models these days are heavily
overparameterized, and you have a small dataset, and you believe that if you do
parameter-efficient fine-tuning then you can get better generalization. Or you believe that it's
going to match full fine-tuning. Sort of a second reason for wanting to do efficient adaptation:
so the plot on the right here shows, in red, the estimated growth in training compute for
training the largest AI

**[39:40]** models, and the line in blue is the global compute capacity. So very soon we are
going to overshoot the global compute capacity and are going to need a lot more compute than the
global capacity. And so this is kind of not sustainable. And there are arguments to be made
about how, if we keep going down this route, then AI development becomes concentrated in only
the hands of a few well-funded organizations, and as students we can't do it. And so that's a
problem. And then also, if there's only a small number of players that are training and
fine-tuning models, then they may bias the model in specific ways

**[40:26]** that reflect their value systems and not the broader public. And so that's another
reason to think about efficient adaptation. And there's sort of this paradigm in machine
learning in general, and NLP specifically, to focus a lot on accuracy instead of efficiency. And
so the plot on the right here shows the percentage of papers where the main contribution is a
method that produces just more accurate models, versus methods that produce the same accuracy
for more efficiency. And so we can see that for most

**[41:12]** conferences the vast majority of papers are about accuracy; there's very few papers
about efficiency. And so maybe this is kind of leading to this monoculture, and maybe that's why
we want to focus on efficiency. The second, maybe bigger concern is that there's this huge
hidden environmental cost of training and fine-tuning large language models. So I was just
reading some report where they said that the cost of training GPT-3 was equivalent to 1.1 million
tons of carbon emission, or some such number [Ed: slide 54 gives no tonnage — it says only that
Cornell scientists in 2021 estimated the emissions equivalent to running a coal power plant for
10 straight hours], and they kind of estimated that that's the cost of running a coal power plant
for 10 hours straight.

**[41:58]** All right. And, for an example closer to home, in the reinforcement learning class
there was, you know, the final project — no, not the final project, a homework assignment — and
a lot of students implemented kind of a common algorithm, one or two algorithms that sort of
outperformed everything else. It used a lot more power. And someone did this calculation, that
if everyone had used the most efficient algorithm that would have — sorry, if everyone had used
the more efficient algorithm, that would have reduced the power consumption of the class by
about 880 kilowatt-hours, which is what an American household uses in a

**[42:45]** month. So there's another — you know, these are all reasons to think about
efficiency, and how you can fine-tune your models with less resources. So let's kind of jump
back into parameter-efficient fine-tuning. And let's start by recapping what full fine-tuning
is. Any questions so far about any of this? Okay. So yeah, so let's recap full fine-tuning. So
let's say we have some large pre-trained autoregressive language model — let's say it's a GPT —
and maybe we want to use it for summarization, maybe we want it for semantic parsing, so like
converting

**[43:31]** natural language to SQL commands, or maybe we want it to answer questions about
paragraphs. And what do we do? We collect a dataset of (x, y) pairs and then we do full
fine-tuning. In full fine-tuning we are going to update all of the model parameters based on the
gradient for some loss function. And maybe that's not feasible: GPT-3 has 175 billion
parameters, and so there's just a lot more parameters to learn. And even once you have done full
fine-tuning, you kind of have to store all of the parameters, and if you're doing several tasks
you have to store parameters for every task. So can we do better?

**[44:17]** Okay. So the main idea is, instead of updating all of the parameters, I am going to
update a much smaller number of parameters. And then, instead of finding a delta theta which is
the same size as the entire set of parameters, I have to search over a much smaller space. And
then the added benefit is I can store this much smaller delta pretty easily on disk, and
hopefully it's going to require less compute, and hopefully it's going to generalize almost as
well as full fine-tuning. So there's many different ways

**[45:04]** of kind of operationalizing this high-level idea of parameter-efficient fine-tuning.
The one I'm going to talk about today is LoRA. So that stands for low-rank adaptation. And that
basically comes from this observation that when you have big language models that you fine-tune,
oftentimes when you look at the geometric structure of the gradients, they tend to have a low
intrinsic rank. Do people remember rank and SVD? All right. Okay. So these parameters — so the
gradients tend

**[45:51]** to have a low intrinsic rank. And so what the authors realized is, instead of
fine-tuning the entire set of parameters, you could instead fine-tune a much smaller, let's say
rank-r matrix, for every full-rank matrix that exists in the model. So let's say we have some
pre-trained weight matrix W [Ed: the captions render the subscript as "KN"; slide 59 writes it
$W_0 \in \mathbb{R}^{d \times k}$], and what I'm going to do is, instead of applying some kind of
arbitrary update, I'm going to make sure that the update has this following form. So it's going
to be the product of two low-rank matrices, B and A. So A is in R — is an r-by-k

**[46:40]** matrix, and B is a d-by-r matrix. And r is the rank, much much smaller than either
the incoming dimension and much much smaller than the outgoing dimension. And the term alpha,
you can think of that as some kind of trade-off between the knowledge that's already stored in
the pre-trained model versus some additional knowledge that you want to add into the model. So
if alpha is zero then you're not doing anything; if alpha is something really, really small then
you don't really want to change your model parameters all that much and you want to add some
really small task-specific knowledge. And then additionally, the only trainable parameters here
are

**[47:27]** going to be A and B. Okay. And then sort of the other thing to note about this is,
since I'm representing updates as this product B times A, as I increase r, that's going to
converge towards full fine-tuning, right? So you kind of have a slider that you can use to
control how much fine-tuning you want to do, essentially. And then the other important thing is
inference latency. So what you can do is, you can just store these learned matrices for every
task, and whenever you switch to a different task you can just remove

**[48:14]** the extra term that you've added to every matrix for that task, and add in the
task-specific terms for the new task that you want to run inference on. And the cost of storing
these much smaller matrices is also way lower than storing the full delta. And we'll kind of see
where you should apply LoRA, but generally you want to apply it to the weight matrices in
self-attention. So in code it actually looks fairly simple. So what you're going to do is, when
you're running the regular forward pass, then you just compute the hidden state as, let's

**[49:00]** say, the product of the matrix and the incoming feature vector. Now with LoRA what
you're going to do is, you're going to freeze your model parameters, you're going to compute the
h as before, and then to that you're going to add this additional offset term, and that's the
only thing that's going to be trainable. And that's pretty much all you have to do. We have to
do it for every single weight matrix, for every single layer. *But yeah, so there's an alpha
term in the second-to-last line — where do you define alpha in the rest, or do you just, like,
put it somewhere?* So you define it somewhere. If you set it to one, that's like saying that I
kind of want an equal trade-off

**[49:46]** between pre-trained knowledge and the new task-specific knowledge. Typically people
set it to one. You could set it to something larger than one if you believe your task is
something that the pre-trained model has no idea about, or something smaller than one if you
don't want to change the model too much. So that's basically LoRA. In practice — so I said
there's a bunch of different parameter-efficient fine-tuning methods, right? So I'm not even
going to name all of these. There's adapters, some of you might have heard about adapters; there
is BitFit, which is not here; and so there's lots of different, like, P-tuning. But it turns

**[50:33]** out that compared to a lot of these different methods, it's kind of pretty
high-performing on a bunch of different tasks, for these relatively smaller models. And then if
we try and look at some of the bigger — we're trying to fine-tune some of the bigger models like
GPT-3, and then compare it with other parameter-efficient fine-tuning methods — so full
fine-tuning is at the way top, then we have BitFit, which is you only fine-tune the bias terms,
and adapters. Compared to that, firstly, LoRA requires a lot fewer additional parameters that you
need to store, and

**[51:18]** it kind of gives you a good trade-off for accuracy compared to full fine-tuning. And
sometimes there's a regularizing effect from fine-tuning only a small subset of your model
parameters. Okay. So the question is, like, for every matrix you can apply LoRA — and I said that
you want to apply it to the various learned matrices in self-attention. The question is what
parameters you want to apply LoRA to. And generally the rule of thumb is that if you apply it to
the matrix that takes your hidden state and converts that into queries, and the

**[52:05]** matrix that converts your hidden state into values — apply LoRA to those, and that's
pretty much going to give you the best performance overall. The other hyperparameter for LoRA is
the optimal rank. So recall that these two matrices B and A that are both low rank — and it
turns out that already with a really small rank you can get a pretty high performance. And this
is much, much smaller than the hidden state dimensions of most of the matrices for most models
these days. All right. So we covered a bunch of things. We talked about floating points and
mixed precision training, multi-GPU training,

**[52:52]** DDP, FSDP, LoRA. It kind of boils down to a very simple flowchart that you can just
use for your project. So if you were sleeping through the entire lecture, maybe now it's the
time to wake up and just look at this flowchart. So: always use mixed precision training. If you
have the newer Ampere architectures, use bfloat16. Try with the batch size one. If batch size
one fits, try a larger batch size, and then always just use ZeRO stage two. Batch size one
doesn't fit — try ZeRO stage three, maybe try gradient checkpointing, activation checkpointing.
Sorry, there's a question. *This is assuming we have more than one GPU,*

**[53:39]** *because [ZeRO] doesn't help us [otherwise]?* Oh yeah, so all of this applies only if
you have more than one GPU, yeah. If you have a single GPU, yeah, you have to do other things —
maybe you have to heavily quantize the model, and even then I don't think you can fine-tune some
of the bigger models, yeah. So, assuming you have multiple GPUs, you can try ZeRO stage three if
you have out-of-memory errors with a batch size of one. If that doesn't work, you can try LoRA.
And the main hyperparameters in LoRA are the alpha, the rank, and what weight matrices to apply
LoRA to. Apply that to the query matrix, apply that to the value

**[54:25]** matrix, set rank to eight — that's a good starting point — set alpha to one. Just do
that and you should be good to go. So you can fine-tune your models and things should be
reasonably good. So I'm going to end now, unless there's questions. Oh, there's one question in
the back. *[Ed: opening words garbled — apparently "the diagram"] I was wondering if you could
just go back to it and walk through it a little bit, one step — sorry, on slide 48.* Yeah, this
diagram from the last — right. Okay. So let's go through this diagram. So basically what this
diagram shows

**[55:13]** is how the communication overhead is really not that bad if you have a fairly big
model, such that in the time it takes to do a forward pass you can already prefetch all of the
parameters for the next layer. So that's pretty much the idea. So that's kind of a standard idea
that I guess everyone should already be using: you want to make sure — and PyTorch does this by
default, by the way — you want to make sure that you sort of fully saturate your GPU, and then
make sure that you overlay that with any additional compute you're doing. And that's pretty much
what's going on here. But let's sort of go through this step by step. And so the starting point
here is

**[55:59]** FSDP units. So 0, 1 and 2 are different FSDP units. So what you start by doing is,
you want to run a forward pass on the first layer. You don't have the first layer — so let's say
you are GPU k, you don't have the first layer, so you have to do an all-gather to get all of the
parameters for the first layer. So that's AG0. At the end of AG0, every GPU has the full set of
parameters for the layers corresponding to FSDP unit zero. Let's just say that's layer one —
okay, or let's just say that's layer zero. So you have the full parameters for layer zero, you
run a forward pass, so that's the stuff in blue. And while you're running a forward

**[56:46]** pass through the first layer, you're going to be smart about communication overheads,
and while you're running that you're going to prefetch the parameters for the next FSDP unit. So
let's say layer two is a different FSDP unit, so that's AG1. And so you can see that there is a
little bit of overlap between FWD0 and AG1. At the end of getting all of the parameters for
layer one, you're going to do a forward pass, and so on. And then you're going to do AG2. And at
the same time — now let's say you just have way too many parameters on your GPU — so you're
going to do a little bit of

**[57:32]** memory free, you're going to free up some memory. So that's the stuff in yellow. And
so that's how that goes. So you basically overlay all-gather operations with the forward pass,
and that's how you run the forward pass. So the communication overhead is really not that bad if
you have a really big deep neural network, and assuming that you have sharded everything
properly. And then you start the backward pass. So in the backward pass, I guess it's a little
bit tricky, because you want to do these all-gather operations to get the full gradient. So
let's say it's a 10-layer neural network. So you want to

**[58:18]** compute the full gradient at layer 10. You need to do an all-gather operation to get
all of the gradients — or to get all of the parameters — at layer 10, and then you have to do a
reduce-scatter. So you have four GPUs, everyone has the full set of parameters at layer 10, they
have different gradients, and so they have to kind of merge their gradients and then split them
up to the right GPU. And so that's the reduce-scatter. But that's not too bad, because you can
still overlay reduce-scatter operations with the backward pass. And so that's what you see
happening in the backward pass there. And then, along with these forward and backward passes, at
regular intervals you have to make sure that you free up GPU memory. So for

**[59:04]** example, once you have run a forward pass through layer one, now you're on to layer
two, you don't need anything in layer one, so you just free up the memory in layer one. That's
pretty much the idea behind this diagram. So there's a few details here. One of the details is,
in FSDP, unit zero is treated differently, so you'll see that unit zero is never freed up.
That's just sort of an implementation detail in FSDP. I'll just quickly say one more thing about
FSDP and take a question. So the presentation here makes it seem like it's so simple and that it
can be applied to any neural network, right? But it turns out that that's not the full picture.
So you need to do this kind of — you need

**[59:51]** to kind of divide up your neural network into FSDP units. And depending on what
policy you use for dividing up your parameters into FSDP units, there's different communication
overheads. So for example, it makes sense to have multiple consecutive layers in the same FSDP
unit, and so on. And so this is very architecture-specific. So when you start to use this in
PyTorch, you'll see that the FSDP wrapper requires a sharding policy, and that is very
architecture-specific. So because everyone uses Transformers now, there are very sort of
handcrafted, fine-tuned policies for Transformer-like

**[1:00:38]** — for creating FSDP units and sharding strategies for Transformers. But let's say
you want to — you know, for your final project you came up with a new architecture, subquadratic
attention, whatever — maybe it's not going to be as efficient just because you don't have the
right shard policy. So that's one detail about FSDP that maybe you want to keep in mind. Okay,
you have a question. *Just a clarification: when you mentioned you can throw away the weights
that you don't need after each layer's forward pass, but then when you compute the backward pass
you stream them back in each time — or do you sort of cache some, or cache recent, or is there
any caching going on, or do you throw them all away and stream them all back?* So there might be
some caching, but

**[1:01:26]** in the system — but the idea is that you just sort of throw them away, or at least
to the user it seems like you've thrown it all away in terms of GPU RAM utilization, and then we
stream them each layer again. And so that's why it's important to shard it properly, right? So
for example, if every consecutive layer is sharded such that it's on multiple GPUs, then you
kind of always are communicating, as opposed to — you know, you kind of did an all-gather and
then all of the next three layers are loaded in. So that's why how you shard, and this sharding
policy, becomes important.

**[1:02:16]** Okay. Okay, so if there's no more questions, let's end early. Thank you so much.
[Applause]
