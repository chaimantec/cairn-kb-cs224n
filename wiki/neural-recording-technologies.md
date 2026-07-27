# Neural recording technologies

How you measure brain activity determines what you can decode from it, so the choice of
recording device is the first and most consequential decision in any
[brain-computer interface](brain-computer-interfaces.md). Covered in
[lecture 14](14-brain-computer-interfaces.md), slides 16, 19–21 and 33.

## The two axes that organize everything

Slide 16 (adapted from Belliveau 1992) plots every method on two logarithmic axes, with
invasiveness shaded underneath:

- **Spatial resolution** (vertical): how small a piece of brain a single measurement covers.
  High on the axis means you are averaging a large area — a whole cortical map or more. Low
  means single neurons, dendrites, individual synapses.
- **Temporal resolution** (horizontal): how finely you can resolve time. Fan's numbers: a single
  electrode reports the potential at each **millisecond**; fMRI, which measures blood flow, can
  only report an average over roughly **0.5 to 1 second** (≈24:01).

The second axis matters more than it looks. A neuron's action potential lasts on the order of one
millisecond. A method that integrates over a second is not merely coarse — it "averages, smooths
out a lot of information" that the code is actually carried in (≈24:48). The ideal device is
simultaneously high in both, and the practical answer for BCI is the chronic multi-site electrode
array, which slide 16 places at fine spatial resolution across long time scales.

## The three tiers, by how far in they go

Slide 33 (right panel) draws the comparison cleanly as a cross-section through scalp, skull and
cortex.

### EEG — on the scalp

Non-invasive, cheap, still the clinical standard for diagnosing conditions such as epilepsy. The
electrode sits outside the skull and records the summed activity of **millions** of neurons.

The limitation is the whole reason the rest of the field exists. Fan's analogy: it is like trying
to hear what people are saying in the next room. "What we are hearing is kind of the mumbling of
a lot of things. We can probably just tell that maybe they are in a happy mood, or maybe they
have reached a conclusion, but not exactly what they are trying to say" (≈10:58). Enough for
state detection or a coarse control signal — Alvin Lucier drove percussion instruments from
alpha waves in 1965 — not enough for language.

### ECoG — on the cortical surface

**Electrocorticography** places a grid of disc electrodes under the skull but *on top of* the
cortex, without penetrating it. It does not resolve individual neuron firing; it records averaged
neural activity over a small region, which puts its resolution between EEG and penetrating arrays
(≈40:18).

That intermediate position shows up directly in results. Moses et al. 2021 built the first
speech BCI with a **128-electrode ECoG array over speech sensorimotor cortex**, and got a
**50-word vocabulary at around 75% accuracy** (slide 33) — a real result, and visibly bounded by
the signal available.

### Intracortical microelectrode arrays — into the cortex

The **Utah array** and relatives: a small square bed of fine silicon needles, about the size of a
fingernail, pushed into the cortical surface. Each shank tip sits close enough to neurons to
record their spikes directly, and one array carries on the order of **96 channels**, each showing
its own sorted spike waveforms (slides 19–21). A pedestal connector on the skull carries the
signal out by cable.

This is what the NPTL speech work uses: **four arrays** implanted in participant T12, later four
arrays in the UC Davis study. It is the only tier in this list that resolves single-neuron
activity at millisecond precision, and correspondingly the only one that has produced
large-vocabulary speech decoding.

## What an electrode actually reports

- **A spike waveform.** Slide 20 shows the classic extracellular action potential recorded at one
  electrode tip: flat baseline, fast upward deflection to a sharp peak, rapid fall through
  baseline to a smaller trough, slow return. Detecting these is what turns a voltage trace into a
  spike train.
- **Local field potential.** Fan notes that each electrode may be picking up the combined firing
  of several neurons around it rather than exactly one (≈26:20) — an important caveat, because the
  clean "one electrode, one neuron" model is a teaching simplification.
- **Features on a fixed clock.** For the speech decoder the raw signal is reduced to one feature
  vector per **20 ms** bin, which is what the neural network consumes (slide 57).

## The stability problem

Nothing above is stationary. Because the brain is soft tissue and the array is rigid, an
electrode drifts relative to the neurons it was reading. Asked how you pinpoint which neuron
"neuron one" is, Fan answers that you assume a fixed one-to-one mapping and then admits it does
not hold: "the brain is this kind of soft structure, so if you put electrodes there they could
move a little bit and measure different neurons. So that's one of the challenging problems of
BCI" (≈21:41). This is why decoders are recalibrated at the start of a session rather than
trained once and deployed.

## Related pages

- [Brain-computer interfaces](brain-computer-interfaces.md) — where these devices sit in a full
  system.
- [Neural population decoding](neural-population-decoding.md) — what you do with the spikes once
  you have them.
- [Lecture 14 — Brain-computer interfaces](14-brain-computer-interfaces.md)
