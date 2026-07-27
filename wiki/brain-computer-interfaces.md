# Brain-computer interfaces

A **brain-computer interface** (BCI) reads activity directly from the brain and turns it into
control of an external device — a cursor, a robotic arm, or a stream of text — bypassing the
body entirely. The clinical motivation is people whose brains are intact but whose motor route
out is not: [lecture 14](14-brain-computer-interfaces.md) opens with the **locked-in state**
produced by a severe brainstem stroke or by ALS, where "you still have a fully functioning brain,
but everything is lost" (≈2:25).

Fan's framing is worth keeping because it explains what the machine learning is *for*: this is a
bandwidth problem. Existing assistive devices are slow. A letter board read off a partner's
interpretation of the user's gaze takes minutes per sentence; conversational speech runs at
about 150 words per minute. Everything a BCI does is an attempt to move an impaired person up
that scale (slide 29).

## The four parts of any BCI

Every system in the lecture has the same shape, and it is useful to hold the parts separately
because different research communities work on different ones.

1. **A recording device.** Electrodes somewhere on or in the head, producing a voltage time
   series. The choice determines everything downstream — see
   [neural recording technologies](neural-recording-technologies.md).
2. **A behavioural task that establishes what the signal means.** You cannot decode an intention
   you have never observed. Systems are *calibrated*: the participant is cued to attempt a known
   movement or sentence, and the neural data is paired with that known label.
3. **A decoder.** A trained model mapping the neural time series to the intended output. This is
   the machine learning, and in the speech case it is a full NLP pipeline — see
   [neural population decoding](neural-population-decoding.md) for the movement case and
   [lecture 14](14-brain-computer-interfaces.md#the-model-and-four-decisions-worth-studying) for
   the speech case.
4. **An effector.** The cursor, arm, avatar or text display the decoded intention drives.

## A short history

The field rests on four observations, each about forty to sixty years apart
([lecture 14](14-brain-computer-interfaces.md#a-brief-history-in-four-steps), slides 8–10):

- **1875, Richard Caton.** There are electric currents in the brain, and they *relate to
  function* — a negative variation appears during head rotation and mastication at the cortical
  areas already mapped to those movements. Not just electricity, but electricity that is about
  something.
- **1924, Hans Berger.** The first human **EEG**. Scalp electrodes record wave-like signals whose
  frequency reflects the subject's state: slow **alpha waves** when calm, sharp **beta waves**
  under cognitive load.
- **1965, Alvin Lucier.** *Music for Solo Performer* drives percussion from amplified alpha
  waves — a person operating an external device from brain activity, with the body out of the
  loop.
- **The intracortical era.** Once electrodes go *inside* cortex, single neurons become visible,
  and with them the tuning structure that makes real decoding possible.

## Invasiveness is the central trade-off

Scalp EEG averages millions of neurons. Fan's analogy: it is like listening to a conversation
through a wall — you can tell the mood, or that they reached a conclusion, but not what was said
(≈10:58). Resolution improves as you get closer to the neurons, and the cost is surgery.
[Neural recording technologies](neural-recording-technologies.md) lays out the three tiers
(EEG, ECoG, penetrating microelectrode arrays) and the space-versus-time chart from slide 16 that
organizes them.

## What has been demonstrated

From the BrainGate clinical trials and related work, as surveyed in lecture 14:

| Capability | Result | Source |
| --- | --- | --- |
| 2D cursor control, virtual-keyboard typing | ~40 correct characters/min peak, ~20 average | Pandarinath et al. 2017 (slide 24) |
| Robotic arm control | Participant drinks unaided | slide 25 |
| Restoring walking | slide 26 | |
| Handwriting BCI | 90 characters/min at 95% accuracy (~18 WPM) | Willett et al. 2021 (slide 28) |
| Small-vocabulary speech, ECoG | 50 words at ~75% accuracy | Moses et al. 2021 (slide 33) |
| Brain-to-text, intracortical | ~25% WER on 130k vocabulary at 60–70 WPM | Willett, Kunz, Fan et al. 2023 (slide 60) |
| Brain-to-text, four arrays, home use | down to 0.99% WER on 125k vocabulary | Card et al. 2024 (slide 65) |

The pattern across the rows is that **which imagined action you decode sets the ceiling**.
Decoding imagined cursor movement caps out around 8 WPM; decoding the imagined act of
handwriting roughly doubles it; decoding attempted speech triples it again, because speech is
the fastest thing the human body does with language.

## Two problems that do not go away

- **Neurons are noisy.** The same intended movement yields different spike counts on different
  trials — "it's not like in an artificial neural network, where if you put something in you
  always get something out" (≈14:50). Decoders must be statistical for this reason alone.
- **The recording drifts.** A student asks how you know which neuron an electrode is reading.
  You assume one electrode reads one neuron, but the brain is soft tissue, the array moves, and
  over days you are reading a different population. Fan names this as one of the field's open
  problems (≈21:41), and it is why systems are recalibrated session by session.

## Related pages

- [Lecture 14 — Brain-computer interfaces](14-brain-computer-interfaces.md) — the full lecture.
- [Neural recording technologies](neural-recording-technologies.md)
- [Neural population decoding](neural-population-decoding.md)
- [Connectionist Temporal Classification](connectionist-temporal-classification.md) — the loss
  the speech decoder is trained with.
- [Neuroethics](neuroethics.md) — what becomes contestable once inner speech is decodable.
