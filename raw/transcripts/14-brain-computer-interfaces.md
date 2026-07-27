---
title: Brain-Computer Interfaces
lecture: 14
video: https://www.youtube.com/watch?v=tfVgHsKpRC8
source: edited from auto-generated YouTube transcript
verbatim: false
original: original/14-brain-computer-interfaces.md
slides: ../slides/14-brain-computer-interfaces.md
---

# Brain-Computer Interfaces — transcript

Copy-edited for punctuation, sentence boundaries, and mis-heard terms; cross-checked against
`raw/slides/14-brain-computer-interfaces.md`. The verbatim auto-generated captions are kept at
`raw/transcripts/original/14-brain-computer-interfaces.md`. Lecturer is Chaofei Fan. Student
questions and comments from the floor are set in *italics*, and speech from the video clips
he plays is marked `[Video: …]`. Timestamps mark the start of each paragraph; all 94 are
preserved in order.

**This is an edited transcript.** The captions had no punctuation and mangled nearly every
technical and clinical term the lecture is built on: *brain-computer interface* arrived as
"bre computer interface", "breing computer interface", "green computer interface" and "Brin
computer interface", and *BCI* as "PCI", "BC" and "b"; *ALS* as "AOS" and "Asos"; *motor
cortex* as "modor cortex", "modal cortex", "model cortex" and "M cortex"; *orofacial* as
"artificial", "artifcial" and "auto facial"; *phoneme* / *phonemes* as "fum", "fums", "PHS",
"phum", "volume", "fores", "fonetic" and "phic"; *CTC* as "CDC" and *Connectionist Temporal
Classification* as "connection is temporal classification"; *RNN* as "RN"; *LSTM* as "ISM",
"LM" and "L lstm"; *GRU* / *gated recurrent unit* as "Gru Gator recurring unit"; *n-gram* as
"angram" and "engram"; *ECoG* as "EOG", "Eco" and "eug"; *EEG* as "EG"; *fMRI* as "Mi";
*Broca's area* as "Brokers area" and "broadcast area"; *NPTL* as "mptl"; *Neuralink* as "neur
link"; *Hans Berger* as "hansburg", "hburg" and "hansberger"; *Richard Caton* as "Richard
Kon"; *Frank Willett* as "Frank wet"; *letter board* as "lead board", "ladder board" and "Le
board"; *eye tracking* as "it tracking"; *cosine tuning curve* as "coign tuning curve";
*brain-to-text* as "spring to text"; *neuroprosthesis* as "neuroth neuro prothesis";
*word error rate* as "world a rate" and "word arate"; *attempted speech* as "template
speech"; *inner speech* as "UN speech" and "in decoding"; *restore walking* as "restore
working"; *tongue* as "towns"; *vowels* as "vows"; *axon* as "Exon"; *alpha wave* as "alha
wave"; *locked-in state* as "locking state"; *brainstem stroke* as "brain stamp stroke"; and
*CS224S* as "CS2 24s". Terms were restored from context and checked against the slides.
**No content was added, removed, or reordered.**

**Four passages are genuinely unclear and are marked inline rather than smoothed over:**

- At **0:51** the captions read that Howard "was shot," in the same sentence that attributes
  his condition to a severe stroke. The two readings conflict and the video itself is not in
  the deck, so the words are left as heard with a note.
- At **33:19** the institution behind the robotic-arm demonstration is captioned "ctech."
  *Caltech* is the obvious reading, but slide 25 does not name an institution and the
  lecturer hedges it himself, so his hedge is preserved.
- At **45:42** and **1:06:32** short fragments ("turning on", "the full work") sit inside
  the audio of video clips being played and do not parse; they are marked rather than
  reconstructed.

**Numbers are as spoken**, and the inventory was compared token by token against the verbatim
original. Where the lecturer states he has forgotten an exact figure — the alpha-wave range at
7:52 — that hedge is kept. Three number differences from the captions are deliberate and each
is either flagged inline or listed here: "0 five seconds" is written as **0.5 seconds** (24:01);
"assignment three" as **Assignment 3** (58:47); and the upper end of natural speech, captioned
"150 and 60 words per minute," is written as **150 and 160** with an inline note (35:40). The
stutters "T6 that code name T6" and "60 to 7 70" are collapsed to single figures. One number
differs from the slides and is noted at 49:30: he says "about 10,000 sentences," while slide 42
prints "~1,0000 sentences," which is the deck's own typo.

---

**[0:05]** So, thanks, y'all, for coming — I know it's a really busy time in the quarter,
everyone's busy with the homework, the project and midterms. Today I'm going to tell you about
something I'm really passionate about: the speech brain-computer interface, my research. Before
that, just some self-introduction. I'm Chaofei, from the Stanford NPTL lab. Our lab is trying to
build speech brain-computer interfaces to help people restore communication, or restore
movement. So today I'm just going to tell you guys how cool this brain-computer interface is,
given

**[0:51]** that we have so many recent developments in AI and machine learning, and I hope you
can enjoy this talk. All right, so let me first start with a video, to give you some motivation
for why we want to build a brain-computer interface. I think what the story tells is that we saw
this teenager, Howard, who was 21 at the time of this video, was shot [Ed: the captions read
"was shot" here, which conflicts with the severe stroke named in the same sentence; left as
heard], and he lost all his dreams because of a severe stroke, which also left him in this kind
of locked-in state

**[1:38]** where he can't move. He talks about how he used to like going out and playing
football, making friends, and just letting his emotions out. I think all of this is lost to him,
and I think the most important thing is that he couldn't really speak to express himself, to let
all the emotions out. So Howard is just one of those individuals who suffer from this kind of
neurological disease or disorder — such as brainstem stroke, or ALS — that can cause severe
speech and motor impairment, and even complete loss of speech. For these individuals, I think
life is really challenging for them. Just think about it: you cannot speak, you cannot move, you
still have a fully

**[2:25]** functioning brain, but everything is lost — all your dreams could be shattered. So I
think for people like Howard, as you just saw in this video, the way they can still communicate
with the outside world, with their loved ones, is through these assistive communication devices,
such as the one we just saw in the video, which is this kind of letter board that has the letters
organized physically. And for people like Howard, who may still have some residual eye movement,
they can use their gaze to tell his friend where he's looking, and then his friend can use the
gaze to tell what letter he's trying to say. Just imagine how slow this process is. If you want
to just say a sentence, it might take you a

**[3:12]** few minutes to express simple things like "how are you," or "I'm feeling not
comfortable today." An alternative here is that you can also use an eye-tracking device, so that
people can use eye tracking to type on the computer, on a virtual keyboard. But just think about
it: if you have to look at the computer screen all the time, all day, it's really tiring for
them. And also, these people are not like us — even if they still have some residual eye
movement, it's very hard for them to move their eyes, so it's very tiring as well. Maybe
something different here is

**[3:58]** that maybe some of you guys have already seen this recently: some videos published by
a company called Neuralink. For example, here's one video — let me see if I can play them,
hopefully I can play them. All right. So this company, Neuralink, is developing this kind of tiny
implantable device that can actually be placed inside your skull and read the brain signals. The
hope here is that, because for people like Howard their brain is still fully functioning, maybe
we can use this kind of direct interface with their brain

**[4:46]** so that they can still use their intact brain to control a computer, or even a robot,
to help them live a normal life. And here is a quote from their participant, Noland, and he's
pretty excited about being able to use this very state-of-the-art BCI to be able to connect with
his family and then being able to support himself. I think what I'm trying to say here is that
for people like Howard, and a lot of people who have lost control of their body and their
language, I think BCI can bring hope to them. So that's what I'm going to motivate here today:
we're trying to use BCI to really help these

**[5:33]** people. But I think before going into the details about how this works, I just want to
first go through a brief history of the brain-computer interface, just to help you guys
understand how this thing works — why we can put such tiny devices into the brain and then
suddenly interpret what the brain is doing. There are a lot of interesting stories here. So let
me start with a brief history of BCI. First, back in the 19th century, a British scientist called
Richard Caton started to do some experiments on animals, and one of the things he found is that
you can actually measure the brain activity — measure electricity — from the brain.

**[6:20]** Moreover, if you ask the animal — not ask, but let the animal do some task, say moving
their head — then you can see that the electricity changes somehow. So I think these are the very
first early experiments that scientists did to show that, okay, you can actually decode some
signals from the brain. But we still don't know exactly what those electric signals mean here. So
fast forward to 1924: a German scientist called Hans Berger invented this device called the —
yeah, I always forget how to read that word, but anyway, it's short for EEG. So basically, on the
right you can see that's this kind of

**[7:06]** electrode that you can place on the outside of your — basically on your scalp — and
then measure this kind of wave-like signal. So what the scientist, Berger, found is that — first,
he's the first scientist to find that you can actually measure these wave-like signals just from
this kind of electrode placed on the head. And then he found that this kind of signal has very
different frequencies depending on the state of the patient. For example, if the patient is in
this kind of very calm state, then it will generate this kind of slow alpha wave, around

**[7:52]** 10 to maybe 20 Hz — I forget the exact range. But if the patient opens their eyes and
then does some cognitive task, then you'll see really sharp beta waves. So he's the first
scientist to discover that you can use this kind of electrodes to measure the electric signal of
the brain. And there's also a funny story here, which is that Berger used to be a soldier, and
then one day he was training on a horse, and then he fell from the horse and suffered a
concussion. He also has a twin sister — not a twin, but he has a — yes, he has a twin sister. And
then the story is that on the same day, his sister felt that there was something weird, and she
just started to worry

**[8:39]** about her brother. So his sister sent a telegraph to their father, asking whether her
brother was okay. So this really intrigued Hans Berger — that maybe there's something called
telepathy that can connect two people through this kind of, I don't know, brain wave. So that's
his motivation to start to study psychology and neuroscience, and to invent this EEG, which we
are still using today to diagnose things like epilepsy. Okay. And then people started using this
kind of EEG device to — maybe, since we can somehow detect this kind of wave-like

**[9:25]** thing from the brain, and then we can also control the frequency of the wave — so
someone, a musician here, started to use these EEG devices to perform music. Anyway, so I guess
you guys already got the idea that someone is controlling, is trying to perform some music with
his brain waves. I think this is a really cool experiment; it was done, I think, in the 1950s.
And you can already see that people started to get the idea that you can actually bypass your
body, you can actually use your brain, directly connect your brain to some external device and
control that device. So I think the idea here is: what if we can also leverage the same idea, but
to help people like Howard? You know, you can help them to

**[10:12]** maybe control a robotic arm. But the problem with this kind of EEG, or external
measuring device, is that the signal you get is very weak. Just imagine that your brain is
generating — you probably know that the brain has a lot of neurons, right? The neurons are
actually generating a lot of signals, and if you just put some electrodes on the scalp, then what
you are actually measuring is like the average neuron firing of maybe millions of neurons. Which
is — if you think about an analogy, it's that if you are trying to hear what people are saying in
the room next to us, but we can only try to figure out what they are saying

**[10:58]** in this room, what we are hearing is kind of the mumbling of a lot of things. We can
probably just tell that maybe they are in this kind of happy mood, or maybe they have reached a
conclusion, but not exactly what they are trying to say. So the limitation here is that this kind
of EEG device can only give us a very low-precision, or low-resolution, signal. We want to get a
better signal. So how — I think the answer is trying to go inside the brain and then put this
kind of electrode next to a neuron, and then try to directly measure the neural activity of these
neurons. And for the purpose of this talk, we're mostly going to focus on the neurons in this
region of the

**[11:45]** brain called motor cortex. So this brain — as some of you may already know, the brain
has different regions that do different tasks. So in the center of the brain it's called motor
cortex; it's basically controlling all your body muscles. So the hope here is that if we can
understand the coding of the information that's encoded here in the neurons, then perhaps we can
decode this information and use this information to help people like Howard to be able to control
an external arm, or to be able to speak again. So here's some very basic neuroscience. So we know
that there's this kind

**[12:32]** of cell called neurons, right? So each one of these things is called a neuron, and
this is the body of the neuron, called the soma, and this is the axon. So this is another neuron.
So neurons connect through this tiny thing called a synapse. So if a neuron wants to transfer
some information to another neuron — just like in an artificial neural network, where you have
some neurons and then you want to send information to the next layer — this neuron will basically
generate some action potential, which is just some electricity here, to signal to the other
neuron that there is some information there. So if you put a tiny electrode, say, on the axon of
this neuron here

**[13:18]** and measure the membrane potential, what you will get is something like this. So
you'll have, on the x-axis, the time, and on the y-axis the measured electric potential, and then
you will see these very sharp spikes. And then if you zoom into these spikes, you will see this
kind of typical firing signature of the neuron, which is that the voltage suddenly goes up and
then goes down. So basically what you can measure at the neuron is that it's these very sharp
spikes; that's what you will get by putting electrodes next to a neuron. Okay, so how do we
figure out what kind of information is encoded in this

**[14:04]** what we call a spike train? We can perform some behavioral tasks here. So for
example, here, suppose that we're still listening in to a single neuron, and this neuron — let's
say we're using a monkey for this experiment, right? So we are instructing, training a monkey to
do two things. One thing is basically that we're trying to instruct the monkey to move his hand
either to the left or to the right, and then we measure the firing of the spikes of that single
neuron, and then try to get what kind of information is encoded by that neuron. So what you see
here is that each row here is basically a spike train of that neuron.

**[14:50]** As you just saw here, each vertical line is just a spike of the neuron; each row is a
trial, and a trial means that the monkey is just trying to move his hand in one direction. Then,
if you see here, the neuron seems to fire slightly differently across trials. So I think that's
one fundamental property of a neuron: it's very noisy. It's not like in an artificial neural
network, where if you put something in you always get something out; whereas in a real neural
network things are really noisy. So sometimes they fire a little bit faster, but sometimes they
fire a little bit slower

**[15:36]** under the same experimental conditions. And so here what we are trying to measure is
what kind of information this neuron is encoding when the monkey has moved his limb to the left
or his limb to the right. And then we can also split this information encoding into two phases:
one is preparation, and the other is the execution. So execution means that the monkey is
actually moving his arms, whereas preparation means that the monkey is preparing to move, but
he's holding his arm fixed. So he will actually move his

**[16:21]** arm at this "go" time here. So what you can see here is that it seems like this neuron
likes to fire a lot during the execution, when the monkey's hand is moving to the right, and it
also fires a little bit more when the monkey is preparing to move to the left. So this means that
maybe the neuron is encoding some movement direction here. So basically, if you repeat this
experiment for many different neurons, and then for a lot of different directions, eventually
what scientists find is that — if you fit, say for a single neuron, the firing rate of that

**[17:07]** neuron, basically how many spikes it's generating every second, to different movement
directions, you can fit this kind of cosine tuning curve to it. So what this tuning curve means
is that on the y-axis is the firing rate, and then on the horizontal axis is the movement
direction. So this neuron prefers to fire the most when the movement is, say, 180 degrees to some
reference, and then the firing gradually goes down. So that's the first thing scientists found —
how a single neuron encodes movement information. And then if you measure multiple neurons, you
will find that each neuron could encode very different information. For example, this green
neuron here, its tuning curve is

**[17:53]** slightly shifted to the right, and then the magnitude is shifted down, so its
preferred direction is around maybe 250. Now, with two neurons you can actually decode out more,
actually decode out what the intended movement direction is, right? So for example, with a single
neuron, suppose right now I measure the firing rate is around 30 spikes per second — then there
could be two movement directions, 120 and 240. However, with a second neuron here, you can see
that we can basically eliminate — suppose we measure the second neuron is around five spikes per
second, then we can exactly pinpoint that the movement direction is actually 120, instead of the

**[18:38]** other one, right? However, we know that neurons have some noise, so we actually cannot
really exactly tell the movement direction by using two neurons here. So for example, in the
third part here, due to — suppose the actual firing rate is those gray lines, but due to the
noise the firing rate is slightly shifted to those dashed lines. And you can see that originally
we could decode the movement direction as 120, but in this case the possibility becomes that
there are four possibilities; we cannot exactly, we cannot uniquely

**[19:24]** define it. However, you can see that maybe it's still more likely that the direction
the monkey tries to move is around 120, rather than the one that is around 50, and the other one
that is greater than 240, right? So how do we deal with this kind of noise in neurons? How can we
still more accurately decode this intended movement from this kind of multiple neuron recordings?
I think we can basically use machine learning here, right? So we can treat this as a kind of
classification problem. So here we are plotting each dot;

**[20:09]** here is basically a firing combination of two neurons, and the color here basically
represents the intended movement direction. And then if you somehow train a machine learning
classifier here, then you can basically see we can draw some decision boundaries, where, say, on
the right side, if a new measurement that we get — the firing rate — somehow falls onto this
region here, then we probably know that the monkey is trying to move to the left direction,
right? So okay, so I guess here we know that we can do this kind of single-neuron measurement, we
can measure firing rates of multiple neurons,

**[20:56]** and then by training a machine learning model we can use these machine learning models
with the neural data to infer what the likely movement direction is. So this is how we are going
to build up to actually building a brain-computer interface. Questions? *Yeah — for all these
data you mentioned, like neuron one has a very specific number, how do you pinpoint which neuron
you can start measuring?* Yeah, so here neuron one is basically — we make the assumption that
each tiny electrode you see here is measuring exactly one neuron, and then that electrode will be
fixed, always be fixed, always measuring the firing of that neuron.

**[21:41]** Yeah, but in the real case it's not always the case, because — you think about it —
the brain is this kind of soft structure, so if you put electrodes there they could move a little
bit and measure different neurons. So that's one of the challenging problems of BCI: how you can
deal with this kind of neural recording change. Yeah. All right, let's go back to here. So now we
can basically know that we can put some electrodes into the brain, into the motor cortex, measure
some signals, and then we know how the neurons encode those signals, and then we can also build a
machine learning decoder to decode those signals, right? So we can basically have some methods to
be able to

**[22:27]** build a brain-computer interface, so that we can interpret what the still
fully-functioning brain is trying to do. One more thing is: how can we record these signals? So
yeah, this is a very complicated figure, but don't worry about all the details. What I'm trying
to show here is that basically there are a lot of different technologies that you can use to
record brain signals, but when you think about this technology, you can think about it in this
kind of two-dimensional way. So on the y-axis, think about it as the spatial resolution. So the

**[23:15]** higher up you go on the y-axis, that means that you can basically measure — it
basically shows what the size of the region of the brain is that you can measure. So if you go
really high up there, that means that you can only measure, say, the average brain activity of a
very large brain area, whereas if you go down the y-axis, that means that you can actually
measure at very fine-grained scales, such as single neurons. Whereas the horizontal axis here
means the temporal resolution — that means that for technologies such as this kind of

**[24:01]** single-neuron recording, you can basically measure at exactly each time point, for
example one millisecond, what the electric potential is for that single neuron. Whereas for a
recording technology such as fMRI, which basically measures the blood flow in a small brain
region, it can only measure on average, say, around 0.5 seconds or 1 second, what the blood flow
change is in that small brain area. So that's really an average of a lot of information, because
we know that the neuron fires at this really fast speed, right? The electric potential change of
a neuron is usually on the order of one millisecond. If you can only measure things at around,
say, one second, you are

**[24:48]** basically averaging, smoothing out a lot of information. So ideally we want to have
something that has both high spatial resolution and also high temporal resolution here. So what
we use in most of this — right now, in a lot of the clinical trials in our lab — is this kind of
multielectrode array. So each electrode here is like a tiny needle that can measure maybe the
signal of a few neurons, and then you put these needles into a small tiny square the size of a
fingernail, and then you can measure maybe on the order of hundreds

**[25:34]** of neurons. All right, so now we have these devices to measure neurons. Let's go into
more examples of how we do this here. So here's an example: suppose someone has a spinal cord
injury and then has lost the connection to his body, so his mind is still fully functioning. So
the question here is whether we can — what kind of information we can still decode from his motor
cortex, such that we can decode that information and then use that information to

**[26:20]** either control his arm — his own arm — or an artificial arm, right? What we're going
to do is try to put this kind of tiny electrodes, microelectrode arrays, into his motor cortex,
really penetrating into the surface of his motor cortex. And each electrode here, as you see
here, is this kind of tiny needle, and those gray triangles are the size of a neuron. So each
electrode maybe is measuring the firing potential — the local field potential — of multiple
neurons around it. Then we can pass all this information

**[27:09]** in real time to a computer, through this kind of wire right now, and then what we get
on the computer here is that, for example, each block is basically the measurement of that one
electrode. And if we do some behavioral experiments as we just showed now, we can probably figure
out the tuning curve for each electrode. For example, this one's preferred direction is probably
to the left, right? So we can repeat the experiments,

**[27:54]** behavioral experiments, for other channels here, and probably train an ML decoder to
figure out what each channel is encoding, the preferred direction for each channel. So once we
have the decoder trained, then at test time we can basically ask our participant who has the
implant in his brain to try to imagine moving his hand in some direction, and then the decoder
tries to figure out the direction he's trying to move, right? So that's the basic idea here. Let
me go to a demo here. So this is one of the research projects coming out of our lab in 2017. So
here you see a

**[28:41]** participant is typing on a virtual keyboard with her mind, right? And then on the
bottom it shows the typing speed, measured as the correct characters per minute. So it peaks
around 40, and then on average maybe it's around 20. I think this is really amazing. Think about
people like Howard, who used to have to use this kind of letter board to communicate; now, with
this brain-computer

**[29:27]** interface, he can fully communicate by himself through, say, a computer. So that's a
huge improvement over the board. Yeah. *Does the person open their eyes, or close them?* He still
opens his eyes — I mean, she still opens her eyes. *So is there anything with eye tracking?* Not
for this experiment. *So even if she closed her eyes it would still work, right?* Yeah, it will
still work, but she won't have the visual feedback, you know — she won't know where she's typing.
*How about if she came up with a character in her mind, E or R, without looking at the keyboard?*
Okay, that's something I'm going to show next. So —

**[30:13]** *yeah, how do you know whether it's the person who messed up, or whether it's the
machine that's not capturing the correct character?* Like, I'm confused — what do you mean by
"correct"? Okay — oh, oh, here. So yeah, that's a good question. So let me clarify here. I think
the task here is — maybe it's not readable, but I think the task here is basically that she is
copying a sentence, so we know the ground truth, and then we can measure the error rate. Yeah.
*And the clicking motion, or the selection motion — is that*

**[30:58]** *easy to distinguish, or is there a certain way of knowing that the user is pressing
down? Or does she visualize a mouse, or actually —* That's really a good question. So as I just
mentioned, right, we can decode movements, and we can also decode different gestures — say, you
can use this kind of gesture, or move her elbow. So you can just imagine different motor
movements, and then we can basically decode those motor movements and map that to, say, a click
signal, or different signals. *So what if the person looks at the keyboard and then remembers the
keyboard in her — and then she closes her eyes, does it still*

**[31:47]** *work?* I think that's even hard for me to do, right? Can you remember the keyboard
and then just control, say, a mouse? I use a keyboard every day, so I definitely remember the
layout in my mind, and I just close my eyes — but this is a virtual keyboard, so it's not like a
physical keyboard where you can use your muscle memory. Yeah. So maybe one thing I have to
clarify here is that the mental image for her is, say, controlling a — what's the word — I mean,
just like controlling a mouse, right? So she's not actually doing touch typing, but she is
actually moving, say, a

**[32:33]** mouse. Let's move on. All right, so this is basically just a showcase that, building
up on all the knowledge we have learned about the brain, we can basically decode some attempted
movements from people like — I forgot her name, but I think it's the code name T6 — that we can
really help this kind of people to regain communication through this kind of BCI. And I think, as
I mentioned earlier, you can also use BCI to control robotic arms. So for example, in this case,
this is the participant at, I think,

**[33:19]** Caltech [Ed: captioned "ctech"; slide 25 names no institution, and the lecturer hedges
it himself]. He's using his mind to control this robotic arm, which grabs him a drink.
[applause] [Video: "All right — hey, you finish that thing off, that's good. And"

**[34:08]** that's — there you go."] All right. So, and also you can do things like restore
walking ability. I think that's something someone just mentioned right now, just now, which is
that maybe there's — we can try to restore different modalities of communication. For example,
just now we were just using the movements, and then by restoring movements we can control a
computer. But how about we directly restore the ability to do handwriting, right? Because
handwriting is a very natural way to communicate. So Frank — Frank Willett, a research scientist
from our lab — in 2021 published a paper to show that you can actually do this

**[34:55]** kind of handwriting BCI, and he showed that it's actually really, really fast compared
to the previous one. Okay, so now we have seen that there are different ways to restore
communication. Here's just a measurement of different ways of communicating, right? You can see
on the very left is the sip-and-puff interface. It's very slow — basically that means that for
someone who cannot really move but can still do some breathing, they can do this kind of sip and
puff to say yes and no to communicate. That's really slow, maybe around five words per minute.
For a normal person, I'm really surprised that on average a

**[35:40]** normal person can write maybe 13 or 14 words per minute — that's really slow, but
maybe that's just the average speed. And on the very far right side is the natural communication,
which can reach up to 150 and 160 words per minute [Ed: captions read "150 and 60"; slide 29 puts
conversational speech at about 150 WPM]. Just to put everything into context here: the
2D cursor I just showed you guys right now can do eight words per minute, and the handwriting can
do around 18 words per minute. So compared to, say, the letter board, or this kind of eye
tracking, we are really making a lot of advancement here, but still it's far away from

**[36:26]** the natural conversation speed. So the next question basically is: okay, how can we
get there? Can we actually restore speech with a brain-computer interface? To get there, I think
there's a huge barrier here. First is that the language processing in the brain is a really
complicated process. So for example, here it shows all the brain areas that are involved in
language, and we still don't know exactly how this happens, but this is just our best guess at
how language is processed in the brain. On the very right you see that there are a lot of brain
regions that are involved with

**[37:13]** knowledge and reasoning; in the center is maybe the area that's involved with
semantics and syntax; and on the very left is about the perception of speech and then the
production of speech. Language is really complex. So maybe the hope here is that we can start
with motor cortex, which does the motor planning of language, because — the thing I've just shown
you guys — we already know how the motor cortex can encode movements, right? And we can also know
that in order to produce language we need to speak, and then maybe we can put some electrodes
into that part of the motor cortex that actually controls our orofacial muscles, and then try to
decode some information there, and then see if we can actually

**[37:59]** restore speech. To actually be able to restore speech is, I think, more complicated
compared to restoring movement. So what I'm trying to say here is that the production of speech
is really a lot of complicated movements, and it's really rapid — it's just more than just moving
your hands to certain directions. So restoring speech is much harder than just decoding those
movements of each articulator. So instead of trying to decode the movements of each articulator,
because it's very hard, right — and also, for people who have lost

**[38:45]** speech, like Howard, it's basically very hard to actually measure their speech
articulator movements. Instead, maybe we can try to decode this kind of discrete phonemes,
instead of this kind of continuous speech articulator movements, because we know that all
languages can be decomposed into this kind of basic phonetic units. For example, for English we
know that there are different vowels and different consonants; they are correlated with how you
place your tongue in your mouth and how you place your different speech articulators. So here,
instead of decoding the actual articulator movements, we are trying to decode this kind of
discrete phonetic tokens. And then there's previous

**[39:32]** work showing that if you put some electrodes on the motor cortex, then you can
actually tell the differences of different phonemes by measuring the electric activity in the
motor cortex. So there's a hope of being able to restore speech just by putting electrodes in the
motor cortex here. And indeed, in 2021, researchers from UCSF actually demonstrated that it's
feasible to build this kind of small-vocabulary speech BCI with ECoG recording technology. The
difference between ECoG and the microelectrode array I just showed you guys is that, whereas the

**[40:18]** microelectrode array actually penetrates into the cortex, the ECoG stays on the
cortex, so it doesn't actually record single-neuron firing — it still records some average neural
activity over a small region. So compared to microelectrode arrays, they will have a slightly
lower resolution. So that's why their prototype is this kind of small-vocabulary BCI, which can
only decode 50 words at around maybe 75% accuracy. But this is still very exciting work that
showcases that you can actually achieve this kind of speech decoding by putting some electrodes
into motor cortex. All right, so I'll just right now go into the research that's done in our

**[41:04]** lab, which is to build a high-performance speech neuroprosthesis. So in 2022 we
recruited a participant, code-named T12, who has ALS. T12 used to be a very active person — she
likes to ride horses, likes to jog — but because of ALS a couple of years ago she basically
couldn't do all those things that she used to enjoy. And unlike most ALS patients, her symptoms
start with the orofacial movements first, so she still can move her hands a little bit, but she
cannot really speak

**[41:53]** intelligibly. So we decided to put four microelectrode arrays into her brain: two
arrays into her motor cortex, and then two arrays into the part of Broca's area, which is
supposed to be involved with language planning. So the hope here is that we want to decode the
execution of speech, the production of speech, which is how you control your speech articulators,
and also maybe decode some high-level planning about the speech. So that's why we wanted to put
arrays into two different brain regions here. So the first thing we do after we put the arrays in
her brain is that we did some behavioral tests to see

**[42:38]** what kind of information we can decode from those arrays. So here's the first result
we got. We're trying to classify different tasks here. The first plot is that we're trying to use
these four arrays to classify the orofacial movements. So this dashed line is the cue that she's
actually executing those orofacial movements, and then before this dashed line she is preparing
to do those orofacial movements. So you can see that these two red lines here show that you can
classify, you can predict those movements much better

**[43:24]** above chance using these two arrays in the motor cortex, whereas with these two in the
Broca's area you can't really predict too much above chance, especially during the execution of
those movements. And for single phonemes, which we instruct our participant to speak — single
English phonemes — you can also predict those things much higher above chance using these two
arrays from the motor cortex. And also for the words, for single words, right? So what these
results tell us is that for those two arrays we have put into T12's mind, these two arrays in the
motor cortex contain a lot of information about the phonemes being articulated and also the

**[44:10]** words being articulated. But those two arrays in the Broca's area, which is supposed to
help us to figure out the planning of the speech production, don't contain too much information.
So that's really intriguing to us, and we're still trying to figure out why that's true. So for
the rest of this talk here, we're mostly using only these two arrays in the motor cortex here. So
now we know that there's phonetic information being encoded in those two arrays. What we're going
to do next is actually try to build a real-time brain-to-text BCI here. So what we're going to do
is — let me just show you a video demo first

**[44:56]** to, you know, get a sense of what the BCI we're trying to build is. So here, this is
our participant. So she's connected to our decoding machine through this cable here, which
transmits her neural signals in real time to the decoding machine. And then on this screen you can
see that there's a sentence here that we instructed her to copy, to basically read out. And then
once this red square turns green, she will try to speak, and then what you will see below here is
what the machine has

**[45:42]** decoded. [Video: "I don't want to call for a babysitter." "That would be good." "I did
well in school." "I don't see much pollution."] [Ed: the captions then read "turning on", which
does not parse; left as heard.] All right. So that's the almost perfect decoding result from her.
And you can tell from the video that although she can vocalize, it's not

**[46:27]** really intelligible, because of her limited orofacial muscle movement. But we can still
decode out from her brain signal what she is trying to say. And this video — so the task I just
showed is that she's trying to copy a sentence, but this one is trying to answer a question here.
[Video: "I have a very good friend and sister."] And we also try different modalities, which is —
because when she tries to attempt to articulate, it's actually very

**[47:12]** tiring for her to actually articulate those sounds. So what we try here is that we only
instructed her to move her mouth, or move her articulators, but not vocalize. So what we call this
is silent speech, and we can still decode pretty well using this kind of silent speech modality.
[Video: "I do not have much to compare it to." "I — as much as I would like to

**[47:58]** either."] All right, okay. So let's just move on to more technical details about how we
build this speech BCI here. So as I just mentioned, right, the first thing we need to do is to try
to build a decoder, and before building that decoder we need to do some data collection. So here
is our research scientist Frank sitting next to T12 and asking her to read that sentence on the
screen, and then we'll record the neural activity of her seeing that sentence. So we'll have this
kind of paired data collected, where the input is the neural activity and the output is the target
sentence we want to decode. So we basically have to

**[48:44]** go to where T12 lives, and then we'll do some data collection sessions there, and then
test the decoder. The way we collect data is that, because we only have very limited time and we
cannot ask T12 to speak a lot of sentences, we'll divide our data collection into this kind of
block structure, where we instruct her to speak 40 sentences every block, and then she takes a
break, and then we collect another block. So the data collection will last about 100 minutes for
every research session. And then we train a decoder — maybe that takes 10 to 20 minutes, it's

**[49:30]** really quick. After training a decoder, we'll start actually evaluating the performance
of the decoder by asking our participant to speak some new sentences, and then see how well we can
decode on those new sets of sentences. So in total we did the experiment sessions over maybe three
months of time, and then we collected about 10,000 sentences from this Switchboard telephone
conversation corpus [Ed: slide 42 prints "~1,0000 sentences", the deck's own typo]. I really want
to emphasize that we want to decode this kind of conversational English. Once we have the data,
then we can try to see how we can design a decoder that can best solve this task. So let's first
define the problem here.

**[50:15]** So we have some neural feature inputs, right? So let's say we have some neural
features, which is a time series — you can think about it as maybe similar to audio, where at each
time point we'll get some feature vector. The output of this decoder is a set of words, right? So
we know that she's trying to speak some sentence, so we are trying to decode the words from this
input neural feature here. As I mentioned earlier, so instead of directly decoding words from the
input sentences, maybe we want to have this intermediate target of phonemes to decode. The reason
is that, first,

**[51:01]** we know that there are only 40 phonemes in English, so that's a much smaller set
compared to the number of words. So if you want to train a decoder that can actually decode words,
then you need to have much more data to cover all the possible words, whereas for phonemes you
probably need way less data to cover all the 40 phonemes here. So instead of directly decoding the
words, we decided to decode an intermediate representation of phonemes from the neural input
features. Okay, so basically there are two decoders we want to design: the first is a
neural-to-phoneme decoder, and the second is a phoneme-to-word decoder. So those are the two
decoders that we will have to design for this task. Let's focus on the neural-to-phoneme decoder

**[51:47]** first. So basically, I think at this point of the class we probably know that we can
treat this problem as a sequence-to-sequence problem, right? So the input is some feature sequence
and the output is a token sequence. And for a sequence-to-sequence problem, we probably know that
we can use some encoder and decoder models to solve this problem, right? However, the
encoder-decoder model is actually more powerful than we actually need, because the encoder-decoder
model allows this kind of arbitrary alignment between inputs and outputs. That's really helpful
for tasks such as machine translation, whereas, you know, some languages have different word
orders than other languages. But here

**[52:36]** we know that the alignment is kind of more monotonic compared to, say, machine
translation, where the alignment is arbitrary. But monotonic means that you know that, probably,
say for example, the first few neural features probably correspond to the first phonemes in the
output sentence rather than the last phoneme in the output sentence, right? So this is a kind of
monotonic alignment. So to solve this problem of monotonic alignment, we can actually borrow an
idea that people have developed for machine learning tasks such as handwriting recognition and
speech recognition, where the task is also trying to decode, say, a letter sequence

**[53:22]** or a phoneme sequence — also letter sequences — from some speech features. And the
technique we're going to use is called Connectionist Temporal Classification. So for people who
have taken CS224S, you probably already know what this means, but I'm going to do a little bit of
introduction about this thing. I guess I don't have too much time, but I'll just quickly go over
it here. So what CTC — Connectionist Temporal Classification — does is that, given some input
sequence, right, the goal is that we want to decode some output sequence, but we don't know the
exact alignment between them, and usually the input and output have some length mismatch. For
example, in the case of, say,

**[54:08]** speech recognition, where the input could have a length of several thousand frames,
right, where each frame corresponds to a very fine-level, high-temporal-resolution feature, such
as recorded at, say, 20 milliseconds, whereas the output only has a few tokens. So that's a huge
length mismatch here. What we can do is, we can still use, say, an RNN or a Transformer model to
predict what the output token is at each time step, right, and then we somehow have to figure out
a way to fill in between the output tokens some spacers, so that the output token

**[54:54]** can also have the same length as the input — so that the output sequence can have the
same length as the input sequence. So what the CTC loss does is introduce this additional blank
token as output. With this blank token, what you can actually do is — for example, here is an
example output of the CTC classifier; it's trying to produce this kind of sequence here, right.
What you can do is first you merge repeated tokens, and then you take out the blank tokens, so
what you get is a much shorter sequence that corresponds to the output. So what the CTC loss does
is it allows you to do a sequence-to-sequence problem that has a different input and output
length, and

**[55:40]** also has this kind of monotonic alignment property. Let me just skip through this
detail here, how you can train a CTC loss here — just skip through this thing. So now let's
suppose that we have this CTC loss that can actually be trained, be used to train our model,
right. The next problem is what kind of decoder we want to use for this task, what kind of neural
network decoder to use for this task. I think at this point of the class, I think most of you guys
are convinced that the Transformer is really powerful, right? There's no reason for me to say more
about it. But in this case we don't want to use a Transformer. The reason is that

**[56:26]** we don't have a large data set — as I mentioned previously, we only have 10,000
sentences, right. And also, the Transformer is really good at dealing with long-range
dependencies, but here, for speech production, there's no real requirement for long-range
dependency. So let's just go back to the very simple RNN. We know that RNNs work for small data
sets, and they can deal with short-range dependency pretty well. And another nice thing about RNNs
is that it's very efficient to run in real time; you can put a very complicated RNN and even run
it very efficiently on your mobile phone. One of the most popular RNNs we have learned is the
LSTM, right? It uses this

**[57:14]** kind of memory state — here's my cursor — it uses this memory state to try to store
some long-term, long-range information, and then uses these different input and forget gates,
input and output gates, to control how you can read and write into that memory state, right. But
the LSTM is also very complicated. There's a variant of the LSTM called GRU, gated recurrent unit.
I think the idea here is that it just tries to combine this memory state and hidden state into
just one hidden state, and by doing that you can also reduce some gates. So the GRU is basically a
more simple version of the LSTM that works really well when you have a small

**[58:00]** data set. So here we use the GRU instead of the LSTM for our task. So now we know how
we can decode phonemes, and then we have neural network models to decode phonemes, right, and we
know how to train the model. So at inference time — by inference time I mean at testing time — you
can pass in some neural activity into our decoder, and then you will decode out some phoneme
probabilities, right. So maybe at the first time stamp the highest probability is "I". The problem
here is: how do I figure out the most likely output sequence given these phoneme probabilities,
right? So basically the task is to find the most likely output sequence here. I think for this
problem, I think

**[58:47]** since we have already done something similar in Assignment 3, we can use beam search to
figure out the most likely sequence here. However, there is one caveat with the beam search when
you're applying it to the CTC loss, but I'm not going to expand on it too much here. Yeah, so
let's just skip over that. Now suppose that we can use the beam search to find the most likely
phoneme sequence. How do we convert that phoneme sequence into words, right? Because we eventually
want to decode sentences, but not just a sequence of phonemes. So one thing — you can modify the
beam search so that, if you have an English dictionary where you can map each word into its
pronunciation,

**[59:34]** then while doing the beam search we can basically see if you decode some phoneme
sequence that corresponds to a word, and can basically replace that phoneme sequence with that
word, right. However, you can actually do better by using a language model. For example, here —
that's the decoding equation. What I want to do here is that here the x is the input, the y is the
decoded word sequence, and then because not all word sequences have the same likelihood, right, so
some word sequences — say, suppose I decode a sentence called "I can spoke," that doesn't seem

**[1:00:19]** syntactically correct. So we can maybe use the language model to evaluate the
probabilities of each decoded hypothesis, and then use that as some sort of weight on the final
decoding probabilities. So we're adding this extra term here, called the probability of a
sentence, which you can just decompose into the probability of each token given its previous
tokens, and you can basically measure these things using any language model, right. Now another
thing we want to add here is another term, a word insertion bonus. So one problem about this
language model, this probability of sentences, is that actually longer sentences will have smaller
probabilities than

**[1:01:05]** shorter sentences. That's just the property of how you decompose this probability
here. So we want to balance the length of the decoded sequence here by adding some word insertion
bonus. So what we eventually try to optimize is this equation here, using both the probabilities
generated by the RNN decoder, and then using some sort of language model weights, and then the
word insertion bonus, and also some weights here you can optimize. Okay, just to start trying to
put everything together: suppose that you have a neural feature input here, which is — you get
these neural features every 20 milliseconds — you pass that through the GRU

**[1:01:50]** and now you get some phoneme probabilities, right. This all happens in real time, so
everything has to happen — all computation needs to be done — within 20 milliseconds. You do a
really quick beam search, and then you find, okay, maybe this phoneme corresponds to the word "I".
And here we want to use the n-gram language model instead of a more powerful Transformer language
model. The reason is that we want to really do a lot of evaluation really quickly, under 20
milliseconds. So suppose that you have, say, 100 hypotheses, right, and you want to evaluate the
probability of all of them — if you use a Transformer language model such as GPT-3, which is
really powerful, you cannot really do really fast inference

**[1:02:36]** under 20 milliseconds, right. Whereas with the n-gram language model, you can just
load everything into memory and all the evaluation is just a memory lookup, so it's really quick.
After that you can get some probabilities out, and then you'll just keep, say, the top-K
hypotheses for the next step of the beam search here. So that's how we use the n-gram language
model in the real-time decoding. After that, we'll use a Transformer language model to rerank all
the hypotheses generated by the n-gram language model. So this actually happens when you have
actually decoded out the entire sentence — say, I keep the most likely 100 sentences, and then at
this time I can use the Transformer language model,

**[1:03:22]** which can quickly evaluate the probabilities of, say, only 100 hypotheses in the time
of maybe half a second, right, and then can get a better probability measurement of the sentences
here. Yeah. So putting everything together, this is how the entire system works when I showed you
the previous video — that is, that we can right now use this complicated, not comp— like,
multi-stage machine learning model to accurately decode what the person is trying to say, and
build this high-performance speech BCI, right. We're almost out of time here, so I'll just skip
the evaluation part. Evaluation — how we measure the performance — is basically measured in the
word error rate.

**[1:04:09]** We also have all the data open as a competition, so if you guys are really curious
about this thing, you can try to play around with it. I think the most exciting thing about doing
this research is actually seeing how your research can impact people here. So this is a quote from
our participant T12, and then this is how she reacts when this thing first worked for her. It's
really exciting that she can speak after so many years of silence. Okay, so in the last five
minutes maybe I can just go a little bit into what I think is the future of BCIs here. So I think
what I've just shown you guys is that using BCI we can help people

**[1:04:57]** to either restore movement or restore communication. One exciting direction, I think,
is this kind of multimodal BCI. Here is a work published by a group at UCSF where they are trying
to decode not only the phonemes but also the actual speech, and then also some articulatory
gestures, so that you can actually move a 3D avatar here. And also, as I just mentioned, the final
goal of this BCI is that you want to actually deploy it for people to be able to use it every

**[1:05:44]** day, just as we use our phones, right. So here's a more recent development of speech
BCI by our collaborators at UC Davis. What they do is they actually put four arrays into the motor
cortex, meaning that they can actually have better signals than we do. So what they can actually
show is — so, just for reference, because previously I forgot to mention, the final performance of
our system is that we can get maybe around 25% word error rate, meaning that for every 100 words
the participant says, 25 of them maybe are wrong. So for this latest work at UC Davis, they show
that you can actually get close to zero word error rate in a

**[1:06:32]** few sessions, by training the system more and more continuously. So it's actually
being very close to being an actually usable system right now. And here's a video of their
participant using the system to type, to speak. It's very accurate. He's actually using the system
every day right now to communicate with his family, and [Video: "the full work — that really
cannot be understated, how important that is."] [Ed: "the full work" is garbled in the captions
and left as heard.] All right.

**[1:07:19]** So I think the most exciting direction — I think, at least personally, I think it's
happening in our lab — is that we're trying to maybe restore more effortless and natural
communication by decoding this kind of inner speech. So previously, all the speech BCIs I just
showed you — I think the maximum speed we can do is maybe 60 to 70 words per minute, but that's
still far slower compared to natural conversation, which happens at 150 words per minute. So one
of the reasons is that for all these participants, if we ask them to try to attempt to speak,
because they have lost speech for so many years, it's very hard for them to speak at a normal
rate. However, we know that a lot of people have this kind of inner speech, right? We're kind of
talking to ourselves in our mind. I think the research question here is whether we can

**[1:08:04]** decode this sort of inner speech. So there is some preliminary work from one of our
collaborators in our lab showing that you can actually do so. So for example, here the results
show that if you decode attempted speech, which is what I just showed you, for a small set of
words you can do, say, 90% accuracy; but if you ask the participant to imagine moving her mouth,
or imagine a voice in her head, you can do pretty well, right? So it's not as good as the
attempted speech, but still much higher than chance. So I think this shows that it's possible in
the future that we can decode this sort of

**[1:08:50]** inner speech to fully restore natural communication to people like Howard and T12.
But I think there's a more controversial issue regarding this kind of inner speech, because what
if you can decode something like your private thoughts or private memories that you don't want to
express, right? That's a very difficult question here. And also, as I just mentioned, not everyone
has inner speech. And also, when you think about it, it's kind of like speech, right? Speech is
just one external representation of your internal thoughts. It's just a linear representation that
you want to put out through this kind of medium of speech, whereas your thoughts could be more
complex, more multi-dimensional. So it's very hard to

**[1:09:38]** decide where you want to put arrays and where you want to decode all this inner
speech. But I think that's also a very exciting opportunity for us to learn more about the speech
processing in the brain. And I just mentioned, if you want to decode this kind of inner speech,
then you're also facing a lot of new ethical questions. That's really, I think, thought-provoking.
For example, should we allow BCIs to decode, to read the memories, right? Like, what if we decode
something you don't want to say, right? How can we deal with that? On the other hand, what if we
can actually use these things to help people who have lost their memories due to something like
Alzheimer's disease, right? Or

**[1:10:26]** we can read out some subconscious fear that can help people to do their
psychotherapy? How should we decide whether we want to allow this kind of inner speech decoding or
not, or memory decoding or not? And also, I think a deeper question is: what if one day we could
do this kind of cognitive enhancement with BCI — such as, you know, what if you can move a robotic
arm much faster than your real arm? Is that allowed? Or can you actually purchase a memory so that
you can skip this CS224N class? Yeah, I think that's really a hard question to answer, but it's
just to throw this question out so that, you know — it's not like only a BCI problem, but we're
also facing

**[1:11:12]** this problem right now, right? There are a lot of ways you can do enhancement of
yourself. So I guess what I'm trying to say here is that BCIs will raise a lot of new ethical
questions. So this is — I'm taking this quote from this textbook here. What it's trying to say is,
I think this question is — we're not really looking for an answer here, but I think the point here
is that maybe we just want to keep this in discussion, with scientists, with engineers, with
policymakers, and just to make sure that everything — we can use BCI to help people that really
need them, and also be aware that there could be a lot of potential issues

**[1:11:59]** here. Yeah, I think, just to give a summary here: I hope I can convince you guys that
BCI is a really cool new research direction. It's at the intersection of AI, machine learning,
neuroscience and neuroengineering. We'll soon have this kind of system that can really help people
to be able to communicate again, and also it's a really cool opportunity to understand how the
brain processes language. I think the most important thing is that we are bringing hope to people
like Howard and T12. All right, thank you everyone, and special thanks to the people in my lab.
