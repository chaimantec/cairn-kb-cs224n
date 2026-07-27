# Lecture 14 — Brain-computer interfaces

A guest research talk, and the only lecture in CS224N where the input to the model is not text,
audio or pixels but the firing of neurons in a person's motor cortex. Chaofei Fan, from
Stanford's Neural Prosthetics Translational Laboratory (NPTL) and the BrainGate consortium,
walks from Richard Caton's 1875 galvanometer to a system that reads a paralysed participant's
attempted speech off 128 electrodes and prints it on a screen in real time.

The reason it belongs in an NLP course is the middle of that system. Once the neural signal has
been turned into a stream of feature vectors, the problem is one CS224N has already taught:
a [sequence-to-sequence](seq2seq-and-encoder-decoder.md) task with a length mismatch, solved
with a [GRU](lstm.md), a [CTC](connectionist-temporal-classification.md) loss, a
[beam search](decoding-algorithms.md), an [n-gram language model](n-gram-language-models.md) in
the inner loop and a [Transformer](transformer.md) rescoring pass on top. The lecture's most
useful move for a student is showing why each of those choices is the *right* one under this
problem's constraints, several of which push against the course's usual defaults.

**Slide-by-slide text of this deck: [75 slides](../raw/slides/14-brain-computer-interfaces.md)**
— this deck prints no slide numbers at all, so slide *N* means PDF page *N*.

Slides PDF: [Speech brain-computer interfaces for restoring natural communication](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture13-speech-bci.pdf) ·
[Full transcript](../raw/transcripts/14-brain-computer-interfaces.md)

> **A note on numbering.** This lecture sits at **position 14** in the playlist this knowledge
> base follows; the video title and the deck's filename both call it "Lecture 13". Repo files
> use the position. The deck itself carries no lecture number at all — it is a research talk
> titled *Speech Brain-Computer Interfaces for Restoring Natural Communication: past, now, and
> future*, with no agenda slide and no course announcements.

## Why build one at all

The lecture opens with a news clip about Howard, a young man left unable to move or speak by a
severe stroke (≈0:51). The clinical term for his condition is the **locked-in state**: a fully
functioning brain with no motor route out of it. Fan's framing is that this is a communication
problem, not a cognition problem — "you still have a fully functioning brain, but everything is
lost" (≈2:25).

What such a person has today is slow. A **letter board** — a physical grid of letters that a
partner reads off the direction of the user's gaze — takes minutes to produce a sentence like
"I'm not feeling comfortable today" (≈3:12). Eye-tracking keyboards are faster but exhausting,
and residual eye movement is itself often impaired. Slide 29 puts numbers on the whole range:

| Method | Words per minute |
| --- | --- |
| Sip-and-puff interface | ~4 |
| BCI-driven 2D cursor control (Pandarinath et al. 2017) | ~7 |
| Handwriting (an able-bodied person, by hand) | ~13 |
| QWERTY touch screen / eye tracking with predictive text | ~19–20 |
| Brain-to-text iBCI (Willett et al. 2021) | ~20 |
| Touch screen with predictive text | ~60 |
| Professional typewriting | ~75 |
| Presentation-style speech | ~120 |
| Conversational speech | ~150 |

The entire lecture is about closing the gap between rows 2–5 and the last row. Fan states it as
150–160 words per minute for natural conversation (≈35:40).

## A brief history, in four steps

**1875 — there is electricity in the brain.** Richard Caton, working in Liverpool, put a
galvanometer on the exposed brains of rabbits and monkeys and found currents in every brain he
examined. Slide 8 reproduces the *British Medical Journal* page. The part that matters for BCI
is not that the currents exist but that they *relate to function*: Caton saw a negative
variation during motor acts — rotation of the head, mastication — at exactly the cortical areas
Ferrier had already mapped to those movements. That is the founding observation of the field:
brain electricity is not noise, it is about something.

**1924 — you can record it from outside the skull.** Hans Berger recorded the first human
**EEG** (slide 9). Scalp electrodes pick up wave-like signals whose frequency depends on the
subject's state — slow **alpha waves** when calm, sharp **beta waves** when the eyes are open
and the subject is engaged in a cognitive task (≈7:06). Fan tells the origin story: Berger fell
from a horse and suffered a concussion, and on the same day his sister, far away, felt something
was wrong and wired their father to ask whether he was all right. Berger went looking for the
physical basis of telepathy, and invented the EEG on the way (≈7:52).

**1965 — you can use it as an output channel.** Slide 10 shows Alvin Lucier's *Music for Solo
Performer*, in which amplified alpha waves drive percussion instruments. Fan's point is the
principle rather than the music: a person is bypassing their body and driving an external device
from brain activity directly (≈9:25).

**The limit of doing it from outside.** A scalp electrode averages the firing of millions of
neurons. Fan's analogy: it is like trying to hear a conversation in the next room — you can tell
they are cheerful, or that they have reached a conclusion, but not what they said (≈10:58). To
get more, you have to go inside. See
[neural recording technologies](neural-recording-technologies.md) for the full landscape of
methods and the spatial/temporal resolution trade-off that organizes it (slide 16).

## From spikes to intent

The BCI signal chain starts with a single fact about neurons: they communicate in **spikes**.
When a neuron sends information onward it fires an **action potential**, a brief sharp voltage
excursion, along its axon to a synapse (slide 12). An electrode placed next to a neuron records
a train of these — a **spike train**.

What the spikes mean is established by behavioural experiment. Record one neuron in a monkey's
dorsal premotor cortex while it reaches left or right, and split each trial into a
**preparation** phase (intention formed, arm held still) and an **execution** phase. Slide 13
shows the result: this neuron fires hard during rightward execution and only modestly during
leftward, so it is encoding *direction* (≈16:21).

Sweep across many directions and the neuron's firing rate traces a **cosine tuning curve**
(slide 14). One neuron is not enough — a firing rate of 30 spikes/s is consistent with both 120°
and 240° — but a second neuron with a different preferred direction disambiguates it, and a
handful of neurons plus a classifier does it robustly in the presence of neural noise
(slides 14–15, ≈17:53). This is the core of
[neural population decoding](neural-population-decoding.md), and it is where machine learning
enters: the decision boundaries on slide 15 are a trained classifier over firing rates.

Two properties of real neurons make this harder than it sounds, and Fan flags both:

- **Neurons are noisy.** "It's not like in an artificial neural network, where if you put
  something in you always get something out" — the same intended movement produces different
  spike counts on different trials (≈14:50).
- **The recording is not stable.** A student asks how you know which neuron you are listening to.
  The honest answer is that you assume one electrode reads one neuron, but the brain is soft, the
  array shifts, and over time you are reading different neurons. Fan names this as one of the
  open problems of the field (≈21:41).

## What this already buys, before speech

Slides 21–28 run through the BrainGate results that predate the speech work:

- **2D cursor control and typing.** A participant (code name **T6**) types on a virtual keyboard
  by imagining moving a mouse, peaking around **40 correct characters per minute** and averaging
  around **20** (≈28:41, Pandarinath et al. 2017). Notably, she is not doing touch typing — the
  mental image is cursor movement, and clicks are decoded from a separate imagined gesture such
  as moving an elbow (≈30:58).
- **Robotic arms.** A participant drinks unaided through a neurally controlled arm (≈33:19,
  slide 25).
- **Restoring walking** (slide 26).
- **Handwriting.** Frank Willett's 2021 work decodes the *imagined act of writing each letter*
  rather than cursor movement, and is much faster: slide 28 gives **90 characters per minute at
  95% accuracy**, about 18 words per minute (≈34:55).

The pattern worth extracting is that changing the imagined action changes the ceiling. Cursor
control tops out around 8 WPM; handwriting roughly doubles it. Speech is the next step up, and
it is the fastest thing a human body does with language.

## Why speech is harder, and the trick that makes it tractable

Language in the brain is not localized to one place. Slide 30 (Fedorenko et al. 2024) divides it
into perception, the language network itself, motor planning, and a wide knowledge-and-reasoning
system — with comprehension and production running in opposite directions through it. Nobody can
decode all of that (≈36:26).

So the strategy is to attack the end of the pipeline that is already understood: **motor
cortex**, which plans the movements that realize speech. Speech production is an enormously fast
coordination of articulators — tongue, lips, jaw, soft palate, vocal folds (slide 31) — and
trying to decode continuous articulator trajectories is hopeless, not least because a participant
who has lost speech cannot demonstrate them for you to measure against (≈38:45).

**The trick is to decode discrete phonemes instead of continuous movement.** Every language
decomposes into a small inventory of phonetic units; English has about **40**. Bouchard et al.
2013 (slide 32) had already shown that ventral sensorimotor cortex electrodes distinguish
different consonants by their articulatory gesture, so the information is there to read
(≈39:32).

An intermediate phoneme target also solves a data problem, which is the argument a CS224N student
should keep: 40 phoneme classes can be covered by far less training data than a vocabulary of
tens of thousands of words (≈51:01). The decoder is therefore split in two — a
**neural-to-phoneme decoder** and a **phoneme-to-word decoder** (slide 44).

Before Fan's own work, **Moses et al. 2021** at UCSF had demonstrated the idea with **ECoG**
electrodes, which sit *on* the cortical surface rather than penetrating it. Lower resolution, so
a smaller result: a **50-word vocabulary at roughly 75% accuracy** (≈40:18, slide 33).

## The T12 study

In 2022 the NPTL recruited a participant code-named **T12**, who has ALS. Unusually for ALS, her
symptoms began with the orofacial muscles, so she could still move her hands somewhat but could
not speak intelligibly (≈41:04).

**Four microelectrode arrays** were implanted: two in **area 6v** (ventral premotor cortex, the
speech-motor region) and two in **area 44** (part of Broca's area, expected to carry language
*planning*) (≈41:53).

The first result is a negative one and Fan presents it as genuinely surprising. Slide 36 plots
classification accuracy over time for orofacial movements, single phonemes and single words. Both
**6v** arrays rise far above chance after the go cue — roughly 55–60% for orofacial movements and
40–55% for words. Both **area 44** arrays stay near chance throughout. The region expected to
carry speech planning contributed almost nothing, and the lab does not yet know why (≈44:10). The
rest of the work uses the two 6v arrays only.

### Collecting the data

Slides 41–42 describe a protocol shaped entirely by the participant's stamina. Sentences are
prompted on screen in **blocks of 40**, with breaks between; a session collects **100 minutes**
of training data and **30 minutes** of evaluation data. Decoder training takes only 10–20
minutes, so it happens within the session and the new decoder is evaluated on fresh sentences
the same day (≈48:44). Across about **three months** of sessions the team collected roughly
**10,000 sentences**, drawn from the **Switchboard** corpus of telephone conversations —
deliberately, because the target is conversational English rather than read prose (≈49:30).

### The model, and four decisions worth studying

**The problem (slide 43).** Input is a sequence of neural feature vectors
$\{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$ with $\mathbf{x}_i \in \mathbb{R}^{d \times 1}$, one per
**20 ms** time bin. Output is a word sequence $\{\mathbf{y}_1, \ldots, \mathbf{y}_m\}$ with
$\mathbf{y}_i \in \mathbb{R}^{V \times 1}$. The intermediate representation is a phoneme string:
for "I can speak", `aɪ SIL k æ n SIL s p i k SIL`.

**Decision 1 — not an encoder-decoder.** This is a sequence-to-sequence problem, so the reflex
from [lecture 6](06-sequence-to-sequence-models.md) is an encoder-decoder with
[attention](attention.md). Fan rejects it as *too powerful*: encoder-decoder models permit
**arbitrary alignment** between input and output, which is exactly what
[machine translation](machine-translation.md) needs because word order differs across languages.
Here the alignment is **monotonic** — the first neural features correspond to the first phonemes,
never the last (≈52:36, slide 45). Buying arbitrary alignment you do not need costs data you do
not have.

**Decision 2 — CTC.** The right tool for a monotonic alignment with a large length mismatch is
**Connectionist Temporal Classification**, borrowed from handwriting and speech recognition
(slide 46). CTC adds a **blank** symbol $\epsilon$ to the output alphabet, lets the network emit
one symbol per input frame, and collapses the result by merging repeats and then deleting blanks:
`h h e ε ε l l l ε l l o` → `hello` (slide 47). Because many frame-level paths collapse to the
same string, training marginalizes over all of them:

$$p(Y \mid X) = \sum_{A \in \mathcal{A}_{X,Y}} \prod_{t=1}^{T} p_t(a_t \mid X)$$

where $\mathcal{A}_{X,Y}$ is the set of valid alignments of $Y$ to the $T$ input frames
(slide 48). Full treatment on the
[CTC topic page](connectionist-temporal-classification.md).

**Decision 3 — a GRU, not a Transformer.** Slide 50 makes the argument as a meme, but the three
reasons are serious and each is a constraint the course's usual defaults ignore:

- **Small data.** 10,000 sentences from one participant, not a web corpus.
- **No long-range dependency to model.** Speech production is locally determined; the
  Transformer's strength is irrelevant here.
- **Real-time inference.** Everything must run inside a 20 ms budget, and an RNN's per-step cost
  is fixed and small.

Between recurrent options, the [LSTM](lstm.md)'s separate cell state and third gate are more
machinery than the problem needs, so the system uses a **GRU**, which merges cell and hidden
state into one and drops a gate (slides 51–52, ≈57:14).

**Decision 4 — language models at two different speeds.** Beam search over phoneme probabilities
(slide 53) is the [same algorithm as Assignment 3](decoding-algorithms.md), with the CTC-specific
wrinkle that different extensions can merge into the same collapsed prefix (slide 55). Adding a
dictionary from words to pronunciations lets the search emit words; adding a language model does
much better, because "I can spoke" is a perfectly good phoneme sequence and a bad sentence
(≈59:34). The objective becomes

$$\mathbf{Y}^* = \arg\max_{\mathbf{Y}} P(\mathbf{Y} \mid \mathbf{X})^{\alpha} \times P(\mathbf{Y}) \times L(\mathbf{Y})^{\gamma}$$

where $P(\mathbf{Y} \mid \mathbf{X})$ is the acoustic — here neural — model's score,
$P(\mathbf{Y})$ is the language model's sentence probability factorized by the chain rule, and
$L(\mathbf{Y})^{\gamma}$ is a **word insertion bonus** that offsets the language model's
structural bias against long sentences (slide 56, ≈1:01:05).

The engineering answer to "which language model?" is **both, at different points**. Inside the
20 ms loop the system uses an **n-gram** LM, because scoring 100 hypotheses against it is a
memory lookup; a Transformer LM such as GPT-3 cannot answer in time (slide 57, ≈1:01:50). Once a
full sentence has been decoded, the **top ~100 hypotheses are rescored by a Transformer LM**,
which has about half a second to work (slide 58, ≈1:03:22). See
[language models in decoding](language-models-in-decoding.md).

Slide 59 shows the whole assembled system, including a text-to-speech stage that speaks the
result in a reconstruction of the participant's own voice.

### Results

Performance is measured as **word error rate**, the normalized edit distance between the decoded
and true word sequences (slide 60):

$$\mathrm{WER}(\mathbf{Y}, \hat{\mathbf{Y}}) = \frac{\mathrm{distance}(\mathbf{Y}, \hat{\mathbf{Y}})}{\mathrm{length}(\mathbf{Y})}$$

Across eight trial days, the 130,000-word-vocabulary system runs at roughly **20–28% WER** and a
50-word-vocabulary version at roughly **7.5–13%**, both at about **56–67 words per minute** —
against about 15 WPM for the earlier ECoG system. Fan summarizes the headline figure as "around
25% word error rate: for every 100 words, maybe 25 of them are wrong" (≈1:05:44), and elsewhere
gives the speed ceiling as 60–70 WPM (≈1:07:19).

Two modalities work: **attempted vocalized speech**, and **silent speech**, where the participant
moves her articulators without vocalizing. Silent speech matters because attempting to vocalize is
tiring, and the decoding is nearly as good (≈47:12).

The slide the lecture actually builds to is 61 — T12's own words when it first worked:

> "So many years of not being able to communicate and then suddenly the people in the room got
> what I said. […] it had to be along the lines of 'Holy shit, it worked, I'm so happy, and you
> guys did it.'"

## Where it goes next

**Multimodal decoding.** Metzger et al. 2023 (slide 63) decode phone probabilities, speech-sound
features and articulatory gestures in parallel from one signal, driving a talking avatar, on-screen
text and a synthesized voice simultaneously (≈1:04:57).

**Accuracy good enough for daily use.** Card et al. 2024 at UC Davis placed four arrays in motor
cortex and drove word error rate on a 125,000-word vocabulary from about 10% down to roughly
1% — slide 65 annotates one late session at **0.99%** — by continuing to train the system across
sessions. That participant uses the system every day to talk with his family (≈1:05:44).

**Inner speech.** The remaining gap to 150 WPM exists partly because participants who have not
spoken in years find *attempting* to speak slow and effortful. So the question is whether the
inner voice can be decoded directly. Slide 67 reports preliminary work across seven conditions:
attempted speech decodes at about **98%** on the ventral 6v array, and mimed speech, a "motoric
inner voice", an "auditory inner voice", imagined listening, listening and silent reading all
decode above a chance level of about 14%, declining to roughly 56% for listening and back up to
about 72% for silent reading (≈1:08:04). Well below attempted speech, well above chance.

Fan raises the conceptual problem himself: speech is a *linear* external representation of
thought, while thought is multi-dimensional, so it is not obvious where you would even put the
arrays (≈1:08:50). And the privacy problem is immediate — inner speech includes things you did
not choose to say. That leads directly into
[neuroethics](neuroethics.md), which closes the lecture.

## Related pages

- [Brain-computer interfaces](brain-computer-interfaces.md) — the concept, its history and the
  shape of a BCI system.
- [Neural recording technologies](neural-recording-technologies.md) — EEG, ECoG and
  microelectrode arrays, and the resolution trade-off between them.
- [Neural population decoding](neural-population-decoding.md) — spike trains, tuning curves, and
  reading intent off a population.
- [Connectionist Temporal Classification](connectionist-temporal-classification.md) — the loss
  that makes this decoder trainable.
- [Language models in decoding](language-models-in-decoding.md) — n-gram fusion, the word
  insertion bonus, and Transformer rescoring.
- [Neuroethics](neuroethics.md) — the questions the closing slides put to the room.
- [Sequence-to-sequence and encoder-decoder models](seq2seq-and-encoder-decoder.md) — the
  architecture this lecture deliberately declines to use.
- [LSTM](lstm.md) — and the GRU, which is what the decoder actually is.
- [Decoding algorithms](decoding-algorithms.md) — beam search, generalized.
- [n-gram language models](n-gram-language-models.md) — why a counting model is still the right
  tool inside a 20 ms budget.
