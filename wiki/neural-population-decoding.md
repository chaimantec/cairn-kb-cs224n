# Neural population decoding

The step that turns recorded spikes into a guess about what someone intended to do. This is where
machine learning enters a [brain-computer interface](brain-computer-interfaces.md), and
[lecture 14](14-brain-computer-interfaces.md) builds it up from a single neuron to a trained
classifier over slides 12–15 and 22–23.

## Neurons communicate in spikes

A neuron that has information to send fires an **action potential** — a brief, sharp voltage
excursion — down its axon to a synapse, where it reaches the next cell (slide 12). Put an
electrode next to the neuron and measure membrane potential against time and you see a train of
these: long quiet stretches punctuated by very sharp spikes, each with the characteristic shape
of a fast rise, a fall through baseline, and a slow return (≈13:18, slide 20).

So the raw material of decoding is a **spike train**: for each recorded channel, the times at
which spikes occurred. The quantity that carries information is usually the **firing rate**, how
many spikes per second.

## What a single neuron encodes: the tuning curve

You establish meaning by experiment, not by inspection. The canonical setup (slide 13, Shenoy &
Yu): a monkey reaches to a target, and each trial is split into a **preparation** phase — the
intention is formed but the arm is held still — and an **execution** phase, beginning at a "go"
cue. Record one neuron in dorsal premotor cortex across many trials and plot a raster, one row
per trial, one tick per spike.

For this neuron, the spike histogram is low and flat through preparation and during leftward
reaches, but rises to a tall sharp peak during rightward execution. The neuron is encoding
**movement direction** (≈16:21).

Sweep across many directions rather than two and the firing rate traces a smooth **cosine tuning
curve**: firing rate on the vertical axis, direction in degrees on the horizontal. Slide 14 shows
one peaking near 180° at about 37 spikes/s and falling to roughly 8 spikes/s at the extremes.
Each neuron has its own **preferred direction** — the angle at which it fires hardest — and its
own amplitude.

## Why one neuron is not enough

A tuning curve is not injective. Slide 14 draws a horizontal line at 30 spikes/s and it crosses
the curve **twice**, at about 120° and about 240°. Observing 30 spikes/s tells you the movement
was one of those two and nothing more (≈17:53).

A **second** neuron with a different preferred direction resolves it: if neuron 2 is also firing
at about 5 spikes/s, only 120° is consistent with both readings. This is the essential idea of a
**population code** — the intention is represented jointly across neurons, not by any one of
them.

## Why two neurons are not enough either

Real neurons are noisy. Fan is emphatic that this is the fundamental difference from artificial
networks: "it's not like in an artificial neural network, where if you put something in you always
get something out; whereas in a real neural network things are really noisy" (≈14:50). The same
intended movement yields a different firing rate on each trial.

With noise, the exact geometric intersection argument collapses — slide 14's clean crossing
points shift, and four directions become consistent with the observation instead of one (≈18:38).
What survives is a *likelihood*: 120° is still more probable than the alternatives, just not
certain.

## The machine learning formulation

So treat it as **classification** (≈19:24). Slide 15 plots each trial as a point in a
two-dimensional space — neuron 1's firing rate against neuron 2's — coloured by the direction the
subject was reaching. The four directions form four clusters, and a trained classifier draws
decision boundaries between them, "moved to optimize discrimination". At test time a new pair of
firing rates falls in some region, and that region names the decoded direction.

Everything scales from there. A 96-channel array gives a 96-dimensional feature vector rather
than two, and slide 22 shows the real version: eight raster plots arranged radially by reach
direction, a polar tuning plot summarizing one channel's preference across all eight, and that
channel identified on the 96-channel grid as "leftward". Repeat per channel, train a decoder over
all of them, and the participant can drive a cursor toward a target on screen (slide 23,
≈27:54).

## What generalizes to the speech case

The speech BCI in the second half of lecture 14 is the same idea with three changes:

- The feature vector is one **20 ms** bin of multi-channel neural features rather than a firing
  rate per neuron.
- The output is not one of four directions but a sequence of phonemes, which makes it a
  [sequence-to-sequence](seq2seq-and-encoder-decoder.md) problem and brings in
  [CTC](connectionist-temporal-classification.md).
- Calibration is done by having the participant attempt to read prompted sentences, so the label
  is a sentence rather than a cued direction (slide 41).

The dependence on a calibration phase, though, never goes away — and neither does the noise that
makes a statistical decoder necessary in the first place.

## Related pages

- [Brain-computer interfaces](brain-computer-interfaces.md)
- [Neural recording technologies](neural-recording-technologies.md) — where the spike trains come
  from.
- [Connectionist Temporal Classification](connectionist-temporal-classification.md) — the
  sequence version of this decoding problem.
- [Lecture 14 — Brain-computer interfaces](14-brain-computer-interfaces.md)
