---
title: Lecture 14 — Speech Brain-Computer Interfaces for Restoring Natural Communication (slide deck)
lecture: 14
slides: 75 printed / 75 pages in the PDF
source_pdf: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture13-speech-bci.pdf
note: |
  Lecturer is Chaofei Fan. The deck's own title is "Speech Brain-Computer Interfaces for
  Restoring Natural Communication" (subtitled "Past, now, and future"); the Cairn catalog lists
  it at **position 14**, and repo files use the catalog position. This deck prints no slide
  numbers at all — verified from the PDF text layer — so the headings below are PDF page
  positions: slide N is page N, 75 of each, straight through with no gaps and no offset.
---

# Lecture 14 — Speech Brain-Computer Interfaces for Restoring Natural Communication: slide-by-slide

Text and figures of
[`cs224n-spr2024-lecture13-speech-bci.pdf`](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture13-speech-bci.pdf),
transcribed from the deck. Diagrams and figures are described in prose since the KB is read as
text.

Companion pages: [wiki page for this lecture](../../wiki/14-brain-computer-interfaces.md) ·
[transcript](../transcripts/14-brain-computer-interfaces.md)

## Contents

The three section-title slides are 7, 34 and 62.

| Slides | Section |
| ------ | ------- |
| 1–6 | Title and motivation: the human cost of lost speech, assistive devices today, the Neuralink PRIME study, tapping the intact mind with BCIs |
| 7–21 | §1 A Brief History of BCI: electricity in the brain (Caton 1875), EEG (Berger, Lucier), single-neuron recording in motor cortex, spikes and cosine tuning, population decoding, the recording-technique landscape, Utah-array electrodes and spike waveforms |
| 22–30 | From single channels to real communication: a tuning curve linked to cursor control, BrainGate point-and-click and keyboard-typing demos, robotic-arm and walking demonstrations, handwriting BCI, the words-per-minute landscape, language processing in the brain |
| 31–33 | Speech production and its cortical encoding: vocal-tract anatomy, motor cortex's articulatory/phonemic code, an early ECoG small-vocabulary speech BCI |
| 34–38 | §2 A High-Performance Speech Neuroprosthesis: implanting participant T12's four Utah arrays, orofacial/phoneme/word decoding by array site, the real-time brain-to-text demo |
| 39–50 | The decoding pipeline: sentence-decoding task videos, the data-collection protocol, the neural-features → phonemes → words problem definition, CTC sequence modeling, CTC training, choosing an RNN over a Transformer |
| 51–60 | Decoder architecture and evaluation: LSTM and GRU internals, CTC inference and beam search, integrating n-gram and Transformer language models, the full system pipeline, word-error-rate and words-per-minute results |
| 61 | What T12 said |
| 62–68 | §3 Future of Speech BCIs: multimodal decoding (avatar, text, synthesized voice), a home-use speech BCI reaching under 1% word error rate, decoding inner and imagined speech, language processing in the brain (revisited) |
| 69–71 | Neuroethics considerations BCIs raise |
| 72–75 | Summary, a return to Howard Wicks, and acknowledgements |

---

## Slide 1 — Title

**Speech Brain-Computer Interfaces for Restoring Natural Communication.** Subtitle: *"Past, now,
and future."* Below, side by side: the **Stanford University / N·P·T·L (Neural Prosthetics
Translational Laboratory)** logo (a stylized profile head with radiating red-and-navy lines
converging on a blue dot near the temple) and the **BrainGate** logo ("BRAIN" in bold, "GATE" in
grey, tagline "Turning thought into action," beside a set of vertical bars in blue/green/grey).
Beneath that, a row of small institutional seals: Providence VA Medical Center, Massachusetts
General Hospital (MGH), Brown University, the Robert J. & Nancy D. Carney Institute for Brain
Science at Brown, Emory University, Harvard Medical School, Stanford University, and UC Davis
Health — the BrainGate consortium's member institutions. Byline at bottom left: **"Chaofei Fan,
CS224N."**

## Slide 2 — Video frame: a news segment about losing independence

A video still, letterboxed with black bars top and bottom (indicating an embedded video, not a
photo). It shows a teenage boy with curly brown hair, outdoors among green trees, holding up a
white dog so its face is near his own. A white caption bar burned into the bottom of the frame
reads **"DOING ACTIVITIES HAVING LESS ON YOUR MIND."** No other text appears on the slide itself
— it stands alone as a frame from a news broadcast, setting up the human stakes discussed on the
next slide.

## Slide 3 — What we take for granted is lost for some individuals

- Howard Wicks, 21, lost all his dreams after a sever [sic] stroke.
- Neurological disorders like brainstem stroke or Amyotrophic Lateral Sclerosis (ALS) can cause
  speech and motor impairment and even complete loss of speech.
- These individuals are facing extreme challenges in their lives.
- Communication with loved ones and caretakers is one of their most desperate needs.

## Slide 4 — Assistive communication devices

Title: **"Assistive communication devices."** Top right, a video still showing two people facing
each other; one is gesturing toward a large tablet mounted upright between them, displaying a
colorful keyboard-like grid of letters and numbers. Below it, a schematic diagram: an eye icon at
left with two lines radiating rightward — a red line labelled **"User's Gaze"** pointing to a
red-highlighted keyboard patch on a laptop-style screen, and a blue line labelled **"Camera's
View"** pointing down to a small eye-tracking camera clipped beneath the screen. The diagram
illustrates gaze-typing: the user looks at a letter on screen, and the camera below tracks the
gaze direction to register the selection.

## Slide 5 — Neuralink PRIME Study: BCI-controlled chess and gaming

Two side-by-side panels on a black background, captioned **"Neuralink Prime Study."** Left: a
digital chess board mid-game (flat colored piece icons), with one pawn (on e6) highlighted in
yellow and two grey dots marking candidate destination squares (c6 and d6) — implying a
cursor/selection being driven by the participant. Right: a split-screen capture of a kart-racing
video game resembling Mario Kart, two competing viewports shown side by side, the left player in
**1st** place and the right player in **2nd**, each with a coin count and lap counter ("3/3")
overlaid — illustrating a participant using the implant to play video games via cursor or button
control.

## Slide 6 — Tap into the intact mind with brain-computer interfaces (BCIs)

Large quotation: *"'[The Link] has helped me reconnect with the world, my friends, and my
family. It's given me the ability to do things on my own again without needing my family at all
hours of the day and night.'"* — **Noland Arbaugh, PRIME Study participant.**

Top right, a product-style exploded-view diagram labelled **NEURALINK**: five stacked
disc-shaped layers of an implant (outer case, a black electronics puck, a metallic ring, an
internal circuit board, and a thin ribbon/electrode film) shown pulled apart vertically above one
another, next to a photo of a hand holding the fully assembled coin-sized implant between two
fingers. Bottom right, a greyscale render of a human head and brain in profile with a small
coin-shaped implant device seated on the skull surface, showing where the device sits once
implanted.

## Slide 7 — Section title: A Brief History of BCI

## Slide 8 — How does BCI work? Start from the beginning… Electricity in the brain

Subtitle: **"Electricity in the brain."** Below, a scan of a page from *The British Medical
Journal*, dated August 28, 1875, page 278, headed **"The Electric Currents of the Brain. By
Richard Caton, M.D., Liverpool."** The reproduced text describes Caton's experiments on rabbit
and monkey brains: a galvanometer detects electric currents in every brain examined; the external
surface of the grey matter is usually positive relative to a section through it; and the currents
relate to function — showing negative variation during motor acts (rotation of the head,
mastication) at the cortical areas Dr. Ferrier had mapped to those movements, and responding to
sensory input (light stimulating the retina affecting the current of the area tied to eyelid
movement). Two passages are underlined in the original clipping to draw the audience's eye: one
in red/orange, about electrodes placed on the cortical surface or skull; one in magenta/pink,
about the current's relation to function and the head-rotation/mastication example.

## Slide 9 — Listen to the brain from the outside: Electroencephalogram (EEG)

Subtitle: **"Electroencephalogram (EEG)."** Text: "Hans Berger, a German psychiatrist, invented
EEG and succeeded in recording the first human EEG in 1924." Right, two archival images side by
side: a strip of early EEG paper tracing showing three stacked wavy-line channels (irregular
oscillations in the top two traces with a small calibration mark labelled "(5/8 × 3½ s?)" between
them, and a similar wavy trace at the bottom); beside it, a black-and-white photograph of a
seated subject wearing a scalp electrode headband, wired to a large boxy recording apparatus on a
table.

## Slide 10 — Perform with brain waves: Electroencephalogram (EEG)

Subtitle again: **"Electroencephalogram (EEG)."** Below, a video-still title card reading
**"Alvin Lucier / Amplified Brain Waves"** and **"Nicolas Collins / Electronics"** superimposed
over a grainy image of a bearded man wearing a dark headband fitted with electrodes, eyes closed
— a reference to Lucier's 1965 composition *Music for Solo Performer*, in which amplified alpha
brain waves were used to vibrate percussion instruments.

## Slide 11 — Listen to single neurons in motor cortex

Left, a colorized micrograph (credited **"Hubel 1988"**) of stained cortical neurons rendered in
orange and yellow, with a sharp metal microelectrode descending from the top of the frame toward
one cell body. Right, a lateral-view brain diagram with the motor cortex strip highlighted in
blue and labelled **"Motor cortex,"** the remainder of the brain shown in pink/red.

## Slide 12 — Neurons communicate with spikes

A hand-drawn diagram (credited **"Goodman, Spiking Neural Networks"**): a pre-synaptic
("sending") cell at left, with a red arrow labelled **"Action potential"** running along its
axon rightward toward a synapse — boxed and labelled **"SYNAPSE"** — where it meets the dendrites
of a post-synaptic ("receiving") cell at right.

## Slide 13 — Single neurons encode movement directions

Credited **"Shenoy & Yu, Brain Machine Interfaces."** Two side-by-side reach-task panels, each a
square outline containing a hand icon and an arrow to a target circle: left panel shows the hand
reaching leftward (arrow pointing left to a circle), right panel shows the hand reaching
rightward (arrow pointing right to a circle). Below each panel, labelled **"A Single neuron,
multiple trials"**: raster plots of a single neuron's spike times across roughly 10–20 trials,
split into **"Preparation"** and **"Execution"** epochs by vertical grey divider lines, and below
that a **"Spike histogram"** of firing rate (scale bar 25 spikes/s; time scale bar 200 ms). For
the leftward-reach condition, the histogram stays low and fairly flat through preparation with
only a modest bump during execution. For the rightward-reach condition, the histogram is
similarly low during preparation but rises to a tall, sharp peak during execution — showing this
neuron fires much more strongly for rightward reaches than leftward ones. A small brain diagram
at far right marks the recording site, labelled **"Dorsal premotor cortex,"** with an electrode
icon inserted into it.

## Slide 14 — Multiple neurons for accurate decoding: one neuron's tuning curve

Credited **"Shenoy & Yu, Brain Machine Interfaces."** Labelled **"A One neuron."** A single red,
bell-shaped (cosine) tuning curve: y-axis **"Firing rate (spikes/s)"** from 0 to a little over
30; x-axis **"Direction (degrees)"** from 0 to 360 in steps of 60. One data series only. The
curve peaks near 180° at roughly 37 spikes/s and falls off symmetrically toward 0° and 360°
(down to roughly 8 spikes/s at the edges). A dashed horizontal reference line at 30 spikes/s,
labelled **"Neuron 1 activity,"** crosses the curve at about 120° and 240°, with red downward
arrows marking both crossing points — illustrating that a single neuron's firing rate alone
cannot distinguish between two different reach directions that produce the same rate.

## Slide 15 — Multiple neurons for accurate decoding: population calibration

Credited **"Shenoy & Yu, Brain Machine Interfaces."** Labelled **"A Calibration phase."** A
scatter plot: x-axis **"Neuron 1 (spikes/s)"** from 0 to over 40; y-axis **"Neuron 2 (spikes/s)"**
from 0 to over 40. Four data series, one point-color per reach direction: dark-blue points
(labelled **"Down"**) cluster at low Neuron-1 / high Neuron-2 values; light-blue/cyan points
(labelled **"Right"**) cluster at low-to-moderate values on both axes; red points (labelled
**"Up"**) cluster near the origin, low on both axes; green points (labelled **"Left"**) cluster
at high Neuron-1 values across a range of Neuron-2 values. Dashed lines partition the plot into
four wedge-shaped decision regions, annotated **"Boundaries moved to optimize discrimination"** —
showing that combining just two neurons' firing rates lets a classifier separate all four
directions unambiguously, resolving the single-neuron ambiguity from the previous slide.

## Slide 16 — Techniques for measuring neural activity

Credited **"Adapted from Belliveau (1992) Invest. Radiol."** A 2D chart plotting **"Space (log
mm)"** (y-axis, roughly −3 to 3) against **"Time (log sec)"** (x-axis, roughly −3 to 7), with an
**"Invasiveness"** shading scale running beneath the chart from light (least invasive) to dark
(most invasive). Labelled regions — method-coverage boxes rather than data series — include:
EEG/MEG (upper left, least invasive); fMRI and PET occupying a large blue-outlined box in the
middle time/space range; Optical imaging (a lower-middle band); Single Electrode (an
orange-outlined narrow horizontal band around space ≈ −2, spanning most of the time axis); Patch
Clamp (a band at the very bottom); and a red-outlined region on the right labelled **"Chronic,
Multi-site, Electrode-Arrays,"** spanning coarse time scales (day to year) at fine spatial
resolution. A vertical list on the right edge gives the anatomical scale aligned to the y-axis,
top to bottom: Human brain/Macaque brain, Cortical map, Cortical column, Cortical layer/Neuron,
Dendrite, Synapse. Three photo insets connect methods to the chart: top left, an EEG-cap MRI-style
head scan (credited Harvard/MGH-NMR) linking to the EEG/MEG/PET/fMRI region; bottom left, the same
stained-neuron-and-microelectrode photo credited to Hubel (1988) as slide 11, linking to the
Single Electrode band; bottom right, a photo of a silicon microelectrode array (credited "Bionic
Technologies," scale bar 2 mm) linking to the Chronic/Multi-site/Electrode-Arrays region.

## Slide 17 — (untitled) A brain, alone in the dark

No title or body text. A black background holds a small, softly lit 3D-rendered human brain and
brainstem, viewed from behind/above, positioned in the upper-right of the frame. Functions as a
visual pause before the decoding-problem diagram on the next slide.

## Slide 18 — (untitled) The core BCI decoding problem

No title text; black background. A hand-drawn-style diagram: at left, a white outline sketch of a
human arm and open hand. At center, a sketched brain shown in profile, with a red question mark
on its surface where a blue dotted line — representing an inserted electrode — enters the cortex
and continues down through the brain toward the brainstem, ending at a red arrowhead/bracket
near the base. A yellow line runs from the electrode's cortical tip up and out of the brain to a
squiggly yellow trace (a spike/voltage waveform), which feeds into a desktop computer icon
(monitor, tower, and keyboard) at far right. Along the bottom of the slide, a white square-wave
pulse train runs from beneath the arm and hand at left to beneath the computer at right, visually
tying the arm's movement and the computer's recorded signal together as two readouts of the same
underlying intention — with the red question mark over the electrode marking the open question
of what, exactly, is being decoded, and from where in the brain.

## Slide 19 — Electrode arrays: a Utah-style array up close

No title text. A close-up photograph of a silicon microelectrode array: eight or more parallel
needle-like silicon shanks with polished, tapered tips, mounted on a mottled grey substrate,
shot from a low angle so the shanks recede toward the viewer — a macro view of a Utah-array-style
device of the kind pictured on slide 16.

## Slide 20 — Electrode arrays: a single-unit spike waveform

No title text. The same close-up array photograph as slide 19. One electrode tip near the center
of the frame is circled in red, with a leader line to an inset black box at right showing a
single yellow-traced action-potential waveform: a brief flat baseline, a fast upward deflection
to a sharp peak, a rapid fall through baseline to a smaller trough below it, then a slower return
to baseline — the classic biphasic extracellular spike shape recorded at that electrode's tip.

## Slide 21 — (untitled) A Utah array implanted, and its 96 channels of spikes

No title text. A close-up 3D render fills most of the frame: a glossy, skin-toned brain surface
with an implanted pedestal (a cylindrical metal connector, partially labelled "S/N93-…", with
pin sockets) sitting on the skull, connected by a short cable to a small tan microelectrode array
resting directly on the cortical surface near the upper-middle of the frame. To the right, a
light-grey inset grid of roughly 90 small waveform panels, each labelled with a channel number
(the visible labels run non-sequentially, e.g. 2, 1, 3, 4, 6, 8 … up to 96), each panel showing a
blue multi-trace overlay of many superimposed action-potential waveforms recorded on that
channel — a "spike sorting" style overview of simultaneous single-unit activity across the whole
array. A small L-shaped scale bar sits in the grid's bottom-left corner.

## Slide 22 — (untitled) Tuning across the array: relating a direction task to one channel

Left, eight small raster plots arranged radially like compass points, each labelled with an angle
(0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°) and each with x-axis **"time (s)"** from 0 to 0.5,
containing rows of black spike-time tick marks with a vertical blue dashed line marking a task
event (e.g. movement onset). In the center, a magenta polygon plotted on radial gridlines (rings
at 0–100) — a polar tuning plot summarizing this neuron's firing rate across the eight directions,
the same concept as the cosine tuning curve on slide 14 but shown as a radar chart across 8
sampled directions instead of a smooth curve. Eight data series (one per direction) feed the one
polygon shown. Right, the same 96-channel waveform grid as slide 21, with channel 67's panel
replaced by a blue left-pointing arrow icon in a white, red-outlined box; a large red arrow is
drawn across the slide connecting that box back to the raster/tuning plots at left — visually
tying this specific channel to a "leftward" preferred direction identified from the tuning
analysis.

## Slide 23 — (untitled) From a tuned channel to cursor control

Left, the same 96-channel waveform grid, with channel 67 still showing the blue left-arrow icon
in place (no connecting arrow this time). Right, an illustration of a desktop monitor on a stand:
a black screen holds a green filled circle (a target) in the upper-left area and a dark
mouse-cursor arrow icon near the screen's center — representing a move-cursor-to-target BCI task,
implicitly driven by the kind of directionally tuned channel just identified.

## Slide 24 — Video frame: a BrainGate2 point-and-click typing demonstration

A video still (no slide title). A woman reclines in a wheelchair, wearing glasses, a nasal
cannula/ventilator tubing, and headphones, with a pedestal connector visible on her head; she
looks toward a screen at upper right that shows a prompt, **"How did you encourage your sons to
practice music?,"** a partially typed answer, **"When they s_,"** and a QWERTY-style on-screen
keyboard grid of letters, a delete key, and a pause icon; a running timer in the corner reads
**00:30.5**. Below the video frame, a line chart plots **"Correct characters per minute"** (y-axis,
0–40) against **"Time (min)"** (x-axis, 0–9, gridlines at 3, 6, 9). One data series, a white
line: it climbs from 0 to roughly 35–38 correct characters/minute within the first minute or two,
then oscillates between roughly 15 and 38 for the remainder of the session, dropping sharply to 0
at the very end. A thin vertical yellow-green line near the start marks a reference point (e.g.
task onset). Caption: **"Pandarinath\*, Nuyujukian\*, et al. (2017) *eLife*."** Footer disclaimer:
"BrainGate2 Neural Interface System. CAUTION: Investigational Device. Limited by US Federal Law
to Investigational Use."

## Slide 25 — Beyond 2D control: control robotic arms

Subtitle: **"Control robotic arms."** A photograph shows two men: on the left, a man drinks from a
glass held to his own mouth by hand; on the right, a man in a wheelchair with a pedestal connector
on his head and a Santa hat perched on a nearby monitor sips through a straw held up to his mouth
by a black robotic arm mounted on a table, the arm's gripper also holding an orange cup. The image
illustrates a BrainGate participant using a neurally controlled robotic arm to feed/hydrate
himself.

## Slide 26 — Beyond 2D control: restore walking

Subtitle: **"Restore walking."** A photograph of a man walking through a glass-walled indoor
atrium (benches, tables, potted trees visible), wearing a dark cap-like headset with a wired
connector on top of his head, using a wheeled rollator/walker frame fitted with a small
electronic box for support and balance — illustrating a BCI-driven walking-assistance system.

## Slide 27 — Beyond 2D control: communication through handwriting

Subtitle: **"Communication through handwriting."** Credited **"Willett et al. 2021."** An
illustration: a person's head and brain in profile (grey line-art), with two electrode pedestals
on the skull connected by cables to a "computer tower," which connects to a monitor displaying the
word **"hello"** with a text cursor. A thought bubble above the head shows a hand gripping a pen,
mid-stroke, having written **"hello w"** on ruled paper — representing the participant imagining
the act of handwriting each letter. An inset circle at upper right zooms into the implanted device,
showing a dense bed of fine microelectrode needles (a Utah-array-style close-up, similar to slides
19–20).

## Slide 28 — 90 char/min with 95% accuracy: communication through handwriting

Subtitle: **"Communication through handwriting."** A photograph of the study setup: a man wearing
a dark headband with pedestal-style connectors sits facing a monitor, flanked by tripod-mounted
cameras and a boom microphone, in a home-like room decorated with framed photos and flowers. The
monitor displays scrolling decoded text, **"lowell > felt > like > a > soldier > on > a >
battlefield, > stripped > of > ammunition~,"** with a partially overwritten/underlined line below
it and a green square cue box beneath the text — the go-cue the participant watches while
imagining handwriting each character.

## Slide 29 — Restore natural speech?

Credited **"Chang & Anumanchipalli (2019) *JAMA*"** for the concept, with two specific studies
called out in boxes. A horizontal ruled axis labelled **"Words per Minute"** runs from 0 to 150.
Along it, labelled callout boxes connect down to specific points on the scale (approximate WPM
values): **Sip and puff interface** (~4), **BCI-driven cursor control** (~7), **Handwriting**
(~13), **QWERTY touch screen** (~19), **Eye tracking and predictive text** (~19–20), **Touch
screen and predictive text** (~60), **Professional typewriting** (~75), **Presentation-style
speech** (~120), **Conversational speech** (~150). Two additional annotated data points sit below
the main axis: a black dot at roughly 7 WPM labelled **"2D cursor iBCI — Pandarinath\*,
Nuyujukian\*, …, Hochberg, Shenoy\*\*, Henderson\*\* (2017) *eLife*"**; and a red dot at roughly 20
WPM labelled **"Brain-to-text iBCI — Willett, Avansino, Hochberg, Henderson\*\*, Shenoy\*\*
(2021) *Nature*,"** with a red arrow extending rightward from that point toward roughly 70 WPM —
signaling the gap this lecture's featured work aims to close between existing BCI communication
rates and natural conversational speech.

## Slide 30 — Language processing in the brain

Credited **"Fedorenko et al. 2024."** Four labelled brain-render panels connected by arrows (blue
= language comprehension, red = language production): **Perception** ("Perception of the surface
properties of linguistic input, for instance, speech perception area" — highlighted in blue on a
lateral brain view); **Language** ("Language knowledge and processing, language network" —
highlighted in purple/magenta across several patches of a lateral brain view); **Motor planning**
("Planning of the motor movements needed to realize linguistic output, for instance, Broca's
area" — highlighted in red on a lateral brain view); and **Knowledge and reasoning** ("Task
demands beyond language [multiple demand network], Pragmatics/social reasoning [theory of mind
network], Narratives/situation modelling [default mode network]" — shown as light-green,
mid-green, and dark-green patches across lateral and medial brain views), captioned **"Intended
meaning (multiple brain areas, including the above)."** Blue solid/dashed arrows link Perception
↔ Language and Language ↔ Knowledge-and-reasoning (comprehension direction); red arrows run
Knowledge-and-reasoning → Language → Motor planning (production direction).

## Slide 31 — How do we produce speech sound

Three panels side by side, the outer two branded **"INSIDE VOICE."** Left, **"Articulatory
Anatomy"**: a labelled sagittal head diagram showing the Nasal Cavity, Oral Cavity, Alveolar
Ridge, Hard Palate, Soft Palate, Lips, Blade/Apex/Back of the tongue, Pharynx, Epiglottis, and
Vocal Folds, with the oral and nasal cavities marked as "resonating chambers." Middle, a
grayscale video still from a real-time MRI of the vocal tract mid-speech (credited **"USC
SPAN"**), showing the tongue, jaw, and pharynx in profile. Right, **"Consonants — Point of
Articulation"**: the same vocal-tract outline with IPA consonant symbols (p, b, m, w, f, v, θ, ð,
t, d, s, z, tʃ, dʒ, n, l, ʃ, ʒ, r, j, k, g, ŋ, h) arranged along a curved legend by place of
articulation (Bilabial, Labio-dental, Lingua-dental, Lingua-alveolar, Lingua-palatal,
Lingua-velar, Glottal), each group connected by a dotted line down to its corresponding point on
the vocal-tract diagram.

## Slide 32 — Motor cortex encodes articulatory and phonemic information

Credited **"Bouchard et al. 2013."** Panel (a): a brain render with a grid of electrode dots
color-coded by a distance scale (0–45 mm) over the ventral sensorimotor cortex (vSMC). Panel (b):
a zoomed photo of the cortical surface labelling the precentral gyrus (PrCG), postcentral gyrus
(PoCG), central sulcus (CS), and Sylvian fissure (Sf); credited "Guenon." Panels (c)–(e): for the
syllables /ba/, /da/, /ga/ respectively — a small vocal-tract sketch with a red arrow marking the
articulator constriction point, a spectrogram (frequency axis, log kHz, 0.2–8) of the sound, and
below it roughly 15 stacked line traces of high-gamma activity (z-score) from individual vSMC
electrodes (colored black through red), time axis −500 to 600 ms relative to a dashed vertical
reference line. Panels (f)–(h): further high-gamma traces (z-score, 0–4) for three more syllable
sets each — /θa/, /sa/, /ʃa/ (f); /la/, /na/, /da/ (g); /ja/, /ji/, /ju/ (h) — each with the
sound's waveform envelope shown above the traces and a time axis of −400 to 600 ms. Across all
panels, electrodes closer to one end of the distance scale (red) consistently peak earlier or
differently than those at the other end (black), showing that different vSMC electrodes encode
distinct articulatory gestures and time their responses differently depending on which consonant
is produced.

## Slide 33 — Small vocabulary speech BCI with ECoG

Credited **"Moses et al. 2021."** Left, an illustration of a paralyzed man in a wheelchair with a
connector on his skull; labelled steps: **(A) Prompt** — a screen reads "How are you today?"; **(B)
Cortical signals** travel from the connector to a **"Neural processing system"** rack; **(G)
Decoded response** appears back on the screen as "I am very good." An inset diagram shows a
**"128-electrode array over speech sensorimotor cortex"** with its digital connector, seated on a
rendered brain. Right, a schematic cross-section comparing three recording methods relative to
Scalp, Skull, and Cortex layers: **EEG** (a disc electrode on the scalp), **ECoG** (a disc
electrode just under the skull, on the cortical surface), and **Intracortical Microelectrode**
(fine needles penetrating into the cortex itself).

## Slide 34 — Section title: A High-Performance Speech Neuroprosthesis

## Slide 35 — Implanting microelectrode arrays into BrainGate2 clinical trial participant T12

- T12 has bulbar-onset Amyotrophic Lateral Sclerosis (ALS)
- She retains some limited orofacial movement and an ability to vocalize, but is unable to
  produce intelligible speech.
- Four 64-channel Utah arrays
  - Two in area 6v (ventral motor cortex)
  - Two in area 44 (part of Broca's area)

Right, a photograph of a researcher placing a small array model onto an anatomical brain model on
a desk, brain renders visible on two monitors behind him. Below that, a small brain diagram
marking the central sulcus (CS, red line) with two light-blue squares showing the array
placements and a Utah-array icon connected by dashed lines, plus a macro photo of an actual
finger-sized array chip held between two fingertips. Credited **"Willett, Kunz, Fan, et al.
2023."**

## Slide 36 — Neural representation of orofacial movements and speech

Credited **"Willett, Kunz, Fan, et al. 2023."** Left, the same brain diagram as slide 35: central
sulcus marked in red, two light-blue squares marking the array sites, small Utah-array icon with
dashed leader lines. Right, three line-chart panels sharing the y-axis **"Classification accuracy
(%)"** and an x-axis **"Time (s)"** split into two 0–2 s windows (a **"Delay"** period and, after a
dashed divider, a **"Go"** period), titled **"Orofacial movements,"** **"Single phonemes,"** and
**"Words."** Four data series in each (legend below): **6v dorsal** (light red/orange), **6v
ventral** (dark red), **44 dorsal** (blue), **44 ventral** (purple), plus a fifth grey trace for **"Normalized speech
volume (indicates speech onset and offset)"** which appears in the "Single phonemes" and "Words"
panels but not in "Orofacial movements." A
dashed horizontal **"Chance"** line runs near the bottom of each panel. In all three panels the two
**6v** traces rise well above chance after the Go cue — peaking around 55–60% for orofacial
movements, around 20–35% for single phonemes, and around 40–55% for words, the words panel's peak
tracking the grey speech-volume trace — while both **44** traces stay close to chance throughout.
The figure argues that decodable speech-related information is concentrated in area 6v rather
than area 44.

## Slide 37 — Real-time brain-to-text BCI

Credited **"Willett, Kunz, Fan, et al. 2023."** An illustration: a person's head and brain shown
in profile (grey line art), with electrode pedestals on the skull connected by cables to a
**"Decoder PC"** tower, which connects to a monitor displaying the decoded word **"Hello"** and a
microphone. A thought bubble above the head reads **"Hello"** (the intended, silently attempted
speech), and a speech bubble beneath the microphone also reads **"Hello"** (the system's spoken
output) — illustrating the closed loop from attempted speech to decoded, spoken text.

## Slide 38 — Video frame: sentence-decoding task, "baby sitter" prompt

A video still (no slide title) showing a monitor displaying the prompt sentence **"I don't want to
call her a baby sitter"** above an orange/red-bordered square go-cue, with on-screen counters
**"Block: 17"** and **"Trial: 32."** Tripod-mounted cameras and a microphone stand surround the
setup; the participant (grey hair, glasses, a pedestal connector visible on the head) is seen from
behind facing the screen. Footer disclaimer: "BrainGate2 Neural Interface System. CAUTION:
Investigational Device. Limited by US Federal Law to Investigational Use."

## Slide 39 — Video frame: sentence-decoding task, "proud of" prompt

The same setup as slide 38, a different prompt: **"What are you proud of?"** above the orange
go-cue square, counters reading **"Block: 14"** and **"Trial: 15."** Same tripod cameras, and the
same BrainGate2 investigational-device footer disclaimer.

## Slide 40 — Video frame: sentence-decoding task, "compare it to" prompt

The same setup again, a third prompt: **"I do not have much to compare it to."** above the orange
go-cue square, counters reading **"Block: 17"** and **"Trial: 29."** Same room, same BrainGate2
investigational-device footer disclaimer.

## Slide 41 — Data collection

A photograph of a data-collection session: a seated male researcher in a grey sweater faces a
participant in a wheelchair (headband with a wired connector, oxygen tubing, a walker/rollator
frame with a basket beside her). Behind the researcher, two monitors display live neural signal
readouts (colorful multi-channel spectrogram-like panels and a code editor window). In front of
the participant, a third monitor shows the prompt **"Bring my glasses closer please."** above a
green go-cue square, with on-screen counters **"Block: 2"** and **"Trial: 40."** A tripod-mounted
video camera stands at the far left; an orange pill bottle and hand sanitizer sit on the desk.

## Slide 42 — Data collection

A schematic of the recording protocol: a row of orange boxes labelled **"40 sentences"** (several
shown, with an ellipsis for more) grouped under the heading **"Training data collection,"**
totalling **"100 minutes,"** feeding into a pink vertical bar labelled **"Train decoder,"** which
feeds a row of green boxes labelled **"40 sentences"** (again with an ellipsis) grouped under
**"Evaluation,"** totalling **"30 minutes."** Caption below: "Training and evaluation sentences
are randomly selected from the Switchboard corpus of telephone conversation (~1,0000 sentences)"
[the source's own count, printed with an apparent extra zero].

## Slide 43 — Problem definition

- Neural feature inputs: $\{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n\}$, $\mathbf{x}_i \in
  \mathbb{R}^{d\times 1}$
- Words outputs: $\{\mathbf{y}_1, \mathbf{y}_2, \ldots, \mathbf{y}_m\}$, $\mathbf{y}_i \in
  \mathbb{R}^{V\times 1}$

Below, a three-tier diagram for the example sentence "I can speak": a top row of blue boxes
spanning the words **I**, **can**, **speak**; a middle row of green boxes giving the phoneme
sequence **aɪ, SIL, k, æ, n, SIL, s, p, i, k, SIL**; and a bottom row of orange boxes for the
neural feature vectors $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n$ (first two shown
individually, the rest compressed under an ellipsis), sitting above a dense black-and-white
noise-like raster image representing the raw neural recording.

## Slide 44 — Problem definition

The same three-tier word/phoneme/neural-feature diagram as slide 43, now with two intermediate
processing-stage boxes inserted and connected by upward arrows: a **"Neural to phonemes decoder"**
box between the neural-feature row and the phoneme row, and a **"Phonemes to words decoder"** box
between the phoneme row and the word row — laying out the two-stage decoding pipeline the rest of
this section will develop.

## Slide 45 — Neural to phonemes decoder

- A Seq2Seq problem
- Encoder-decoder models allow arbitrary alignment between inputs and outputs.
- Neural to phonemes decoding only needs monotonic alignment.

Right, a classic neural machine translation attention diagram: a Chinese source sentence "我 喜欢
甜点" is fed through a CNN layer into a red **Encoder Bidirectional LSTM** (five encoder cells);
its hidden states feed **Attention Scores** $e_t$, an **Attention Distribution** $\alpha_t$ (shown
as a small bar chart), and an **Attention Output** $a_t$; this combines with a green **Decoder
LSTM** (fed the target-side tokens `<START>`, "I", "like") into a **Combined Output Vector**
$o_t$, producing an **Output Distribution** $P_t$ (bar chart) that assigns the highest probability
to the word **"desserts."** Below this reference diagram, the same phoneme-sequence row (**aɪ,
SIL, k, æ, n, SIL, s, p, i, k, SIL**) as slide 43, fed by a **"Neural to phonemes decoder"** box
from the neural-feature row $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n$ below it.

## Slide 46 — Sequence modeling with Connectionist Temporal Classification (CTC)

Credited **"A. Hannun 2017."** Two side-by-side examples of the CTC alignment problem, each
showing a row of individual output-letter tiles above a raw input signal, with an arrow marking
one aligned position: **Handwriting recognition** — letter tiles spelling "t h e _ q u i c k _ b r
o w n _ f o x" above an image of the handwritten phrase "The quick brown fox," captioned "The
input can be $(x,y)$ coordinates of a pen stroke or pixels in an image." **Speech recognition** —
letter tiles spelling "j u m p s _ o v e r _ t h e _ l a z y _ d o g" above an audio waveform,
captioned "The input can be a spectrogram or some other frequency based feature extractor."

## Slide 47 — Sequence modeling with CTC

Credited **"A. Hannun 2017."** Left, labelled **"Inputs":** a spectrogram-like image divided into
roughly nine vertical time-frame cells. Right, labelled **"CTC outputs post-processing":** a row
of per-frame output-symbol tiles reading **h, h, e, ε, ε, l, l, l, ε, l, l, o** — the raw,
frame-by-frame CTC output (repeated symbols and blank/ε tokens) that collapses, once repeats and
blanks are removed, to the word "hello."

## Slide 48 — CTC training

Credited **"A. Hannun 2017."** Top, the same spectrogram-style input as slide 47 feeding a chain
of RNN cells (small boxes connected left to right), each with a downward arrow into a column of
candidate output symbols (**h, e, l, o, ε**, stacked, shaded by probability). Three colored paths
(orange, red, purple) are traced through this symbol grid, each a different valid frame-level
alignment that collapses to the same word: three example alignments are spelled out below —
**"εheleεelleεo,"** **"heεllεlleεo,"** **"hhellεεloo …"** — all reducing to **"hello."** To the
right, the CTC objective is given:

$$p(Y \mid X) = \sum_{A \in \mathcal{A}_{X,Y}} \prod_{t=1}^{T} p_t(a_t \mid X)$$

annotated in three parts: "The CTC conditional probability" (left-hand side), "marginalizes over
the set of valid alignments" (the sum over $\mathcal{A}_{X,Y}$), and "computing the probability
for a single alignment step-by-step" (the product over $t$).

## Slide 49 — What neural network to use?

The same word/phoneme/neural-feature diagram fragment as the lower half of slide 45: the phoneme
row (**aɪ, SIL, k, æ, n, SIL, s, p, i, k, SIL**) fed by a **"Neural to phonemes decoder"** box from
the neural-feature row $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n$, shown alone as a lead-in
to the architecture comparison on the next slide.

## Slide 50 — What neural network to use? Transformer vs. RNN

A two-panel "Drake meme" comparison. Left, labelled **"Transformer":** an image of Drake in his
rejecting pose (hand raised, palm out) composited onto the body of a blue Transformers-style toy
robot (credited "Guido 2005"), with bullets **"✋ Large datasets"** and **"✋ Long-range
dependency"** — framing the Transformer as a poor fit here because it demands large training data.
Right, labelled **"RNN":** an image of Drake in his approving pose (pointing, nodding along),
above a small labelled box fed by an input $x_t$, with bullets **"👉 Small datasets,"** **"👉
Short-range dependency,"** and **"👉 Efficient real-time processing"** — framing the RNN as the
better fit for this BCI setting, which has little training data per participant and needs
low-latency streaming decoding.

## Slide 51 — LSTM

The standard Long Short-Term Memory diagram (credited "colah.github.io/posts/2015-08-Understanding-LSTMs/"; a leftover page number "24" from the source material is printed at bottom left). Shows a cell
receiving $c_{t-1}$ and $h_{t-1}$: a **forget gate** ($\sigma$) multiplies into $c_{t-1}$; an
**input gate** ($\sigma$) and **new cell content** ($\tanh$) combine and add into the cell state to
produce $c_t$; an **output gate** ($\sigma$) combines with a $\tanh$ of $c_t$ to produce $h_t$,
which also feeds $\hat{y} = \mathrm{softmax}(Uh + b_2) \in \mathbb{R}^{|V|}$. Callout boxes label
each step ("Compute the forget/input/output gate," "Compute the new cell content," "Forget some
cell content," "Write some new cell content," "Output some cell content to the hidden state"), and
a pink annotation reads **"The + sign is the secret!"** pointing at the addition operation that
carries $c_{t-1}$ forward into $c_t$. A legend at bottom identifies box/circle/arrow conventions:
Neural Network Layer, Pointwise Operation, Vector Transfer, Concatenate, Copy.

## Slide 52 — Gated Recurrent Units (GRU)

Two simplified cell diagrams side by side. **LSTM cell:** takes $c_{t-1}$ and $h_{t-1}$ (plus
$x_t$ into a "gate controllers" box) and outputs $c_t$, $h_t$, and $y_t$, with labelled forget
(blue), input (red), and output (green) gate operations. **GRU cell:** takes only $h_{t-1}$ (plus
$x_t$ into its own "gate controllers" box) and outputs $h_t$ and $y_t$, with only forget (blue) and
input (red) gate operations and no separate cell state — illustrating that the GRU merges the
LSTM's cell state and hidden state into one and drops the output gate.

## Slide 53 — CTC inference

A left-to-right pipeline: the neural-feature row $\mathbf{x}_1, \mathbf{x}_2, \ldots,
\mathbf{x}_n$ feeds a blue **GRU** bar, which produces, at each of several time steps, a column
labelled **"Phoneme probability"** stacked with candidate phonemes (**SIL, k, n, s, …, aɪ, æ,
ε**), each shaded by probability (three example columns are shown). These columns feed a pink
**Beam Search** bar, which outputs the final phoneme sequence row (**aɪ, SIL, k, æ, n, SIL, s, p,
i, k, SIL**) at top. The objective is given as $\mathbf{Y}^* = \arg\max_{\mathbf{Y}} P(\mathbf{Y}
\mid \mathbf{X})$.

## Slide 54 — CTC beam search

Credited **"A. Hannun 2017."** A beam-search tree over the toy alphabet $\{\epsilon, a, b\}$ with
beam size three, shown across three time steps (**T=1, T=2, T=3**), each with a "current
hypotheses" column and a "proposed extensions" column. At T=1, the empty string $\lambda$ (blue)
branches into three proposed extensions, $\epsilon$, $a$, $b$ (red). At T=2, each of those becomes
a current hypothesis (blue) and again branches into $\epsilon$/$a$/$b$ extensions, with the beam
keeping only the top three across all branches (kept ones in red, pruned ones in grey) — dashed
lines trace which extensions survive into T=2's current-hypotheses column. The same branch-and-prune
pattern repeats into T=3. Caption: "A standard beam search algorithm with an alphabet of $\{\epsilon,
a, b\}$ and a beam size of three."

## Slide 55 — CTC beam search (continued)

Credited **"A. Hannun 2017."** The same beam-search tree extended one more step, now showing
**T=1** through **T=4**. By T=4, the surviving three beam hypotheses have collapsed to short
strings such as "a," "a a," and "b a" (repeated symbols and $\epsilon$s merged/removed per the CTC
collapsing rule). An annotation with a leader line reads **"Multiple extensions merge to the same
prefix,"** pointing at a spot where two different T=3 hypotheses (one ending in "…$\epsilon$ a,"
one ending in "…a a") both collapse into the same surviving "b a" hypothesis at T=4. Caption: "The
CTC beam search algorithm with an output alphabet $\{\epsilon, a, b\}$ and a beam size of three."

## Slide 56 — CTC inference with language models

The same phoneme-probability-to-beam-search pipeline as slide 53, but the beam search now outputs
the words **"I can speak"** at top instead of a raw phoneme string. To the right, the objective is
extended with a language model:

$$\mathbf{Y}^* = \arg\max_{\mathbf{Y}} P(\mathbf{Y} \mid \mathbf{X}) \approx \arg\max_{\mathbf{Y}}
P(\mathbf{Y}\mid\mathbf{X})^{\alpha} \times P(\mathbf{Y}) \times L(\mathbf{Y})^{\gamma}$$

with **"Word insertion bonus"** annotated pointing at the $L(\mathbf{Y})^\gamma$ term and
**"Probability of a sentence"** pointing at the $P(\mathbf{Y})$ term, which is expanded below as
the chain-rule factorization $P(\mathbf{Y}) = P(y_1)P(y_2\mid y_1)P(y_3\mid y_2,y_1)\ldots$

## Slide 57 — Integrating language models in real-time decoding

A left-to-right streaming pipeline: a spectrogram-like input $\mathbf{x}_t$ (one **"20ms time
bin"**) feeds a blue **GRU** bar, producing a **phoneme-probability** column (SIL, k, n, s, …, aɪ,
æ, ε, shaded by probability); this feeds a pink **Beam search** bar, producing a short candidate
word list (**"I,"** **"Eye,"** "…"); this feeds a purple **n-gram LM** bar, which scores each
candidate (**P(I) = 0.9,** **P(EYE) = 0.01,** "…"). A feedback arrow labelled **"Keep top-k
hypotheses for next time bin"** loops the scored hypotheses back into the beam-search stage for the
next 20 ms time bin, illustrating the online/incremental nature of the decoding loop.

## Slide 58 — Transformer LM for 2nd pass rescoring

A short pipeline: a list of **"n-best hypotheses"** (**"I can speak,"** **"I can spoke,"** "…")
feeds a purple **Transformer LM** bar, which rescores them (**P(I can speak) = 0.95,** **P(I can
spoke) = 0.01,** "…"), and the highest-scoring hypothesis, **"I can speak,"** is emitted as the
final output — a second, more powerful rescoring pass applied after the real-time n-gram-based beam
search.

## Slide 59 — Putting everything together

Credited **"N. Card et al. 2023."** A full end-to-end system illustration: a person's head and
brain in profile, with a thought bubble reading **"Hello"** (intended speech) above it; an inset
zooms into the implant, labelled **"4x"** and **"3.2 mm,"** showing a dense bed of microelectrode
needles. The recorded **Neural features** (a raster-style image) feed a small **Neural network**
icon (connected nodes), which outputs **Phoneme probabilities** (a small time-course plot with
labelled curves for phonemes /h/, /ə/, /l/, /u/, and silence). These feed a **5-gram LM**, then a
**Transformer LM**, narrowing to the **"100 most probable word sequences"** and finally the
**"Highest probability word sequence,"** which is displayed on a monitor as **"Hello …"** and
routed to an **"Own-voice Text-to-speech"** module, which speaks it aloud through a small speaker/
microphone icon, shown as a matching speech bubble reading **"Hello."**

## Slide 60 — Evaluation

**Word error rate:** normalized edit distance between predicted words and ground truth words.

$$WER(\mathbf{Y}, \hat{\mathbf{Y}}) = \frac{\mathrm{distance}(\mathbf{Y}, \hat{\mathbf{Y}})}{\mathrm{length}(\mathbf{Y})}$$

A QR code bearing the NPTL logo sits below the formula (link target not legible at this
resolution). Right, two line charts sharing an x-axis of **"Trial Day"** (categorical days 113,
119, 125, 134, 136 bracketed **"Vocalizing,"** then 141, 146, 148 bracketed **"Silent"**), each
with two data series — orange circles for **"130,000 word vocab"** and blue circles for **"50 word
vocab,"** each point with vertical error bars. Left chart, y-axis **"Word Error Rate (%)"** (0–35):
the 130,000-word series runs roughly 20–28%, edging upward in the Silent days toward a dashed
reference line at about 25%; the 50-word series runs consistently lower, roughly 7.5–13%. Right
chart, y-axis **"Words Per Minute"** (0–70): the 130,000-word series sits around 64–67 WPM across
the Vocalizing days, dips to about 61 on day 141 and climbs back to about 67 by day 148, while the
50-word series runs a few WPM lower throughout at roughly 56–61; a dashed horizontal line at
about 15 WPM is labelled **"Moses et al. 2021 (50 word vocab),"** marking the much lower speed of
the earlier ECoG-based system from slide 33 for comparison. Footer credit: "Brain-to-Text
Benchmark '24."

## Slide 61 — What T12 said

Block quote: *"So many years of not being able to communicate and then suddenly the people in the
room got what I said. I don't remember what I exactly said after the prescribed script finished,
but it had to be along the lines of 'Holy shit, it worked, I'm so happy, and you guys did it.'"*

Right, a photograph of the moment: a man with arms crossed smiles in the background; a woman
wearing a surgical mask stands beside him; in the foreground, the participant (T12 — grey hair,
glasses, a wired headband connector, a tie-dyed shirt, seated with a walker frame beside her)
laughs with visible emotion; a fourth person, in a striped orange-and-white top, leans in to touch
a nearby monitor bearing a red "Z" logo. Credited **"Stanford magazine."**

## Slide 62 — Section title: Future of Speech BCIs

## Slide 63 — Multimodal Speech BCI

Credited **"Metzger et al. 2023."** A diagram: a person's head in profile, labelled **"Brainstem
stroke"** (a red mark on the rendered brain) as the cause of paralysis, with an electrode array on
the skull connected by a green cable. A thought bubble reads **"Attempted silent speech"** with the
text "Good to see you aga[in]" inside it. Three parallel decoder streams (each a small
feedforward-network icon, grouped under the label **"Deep-learning models"**) take the neural
signal and separately output: **Phone probabilities** (a colorful cloud of overlapping IPA
symbols); **Speech-sound features** (a spectrogram image); and **Articulatory gestures** (a
vocal-tract outline with colored traces for different articulators). These three streams combine
to drive three simultaneous outputs shown on a display: **Avatar animation** (a rendered talking
digital face), **Text decoding** (on-screen text reading "Good to see you again"), and **Speech
synthesis** (a waveform icon feeding a small speaker) — illustrating one decode driving a talking
avatar, live captions, and a synthesized voice all at once.

## Slide 64 — (untitled) Raw multi-channel neural recording software

No slide title. A screenshot of a clinical neural-recording application: a control-panel window at
top shows the file path **"D:\bravo3_chang\20230411-113726\20230411-113726,"** fields for "Remote
Recording Control," "Setup," "Disable File Splitting," Record/Stop buttons, and status readouts
including "Spike Count 0," "Packet Count 88459334," "File Size (MB) 1883.0," "Section 1,"
"Available (MB) 7177665," and "Current Time 12:49:06." Below it, filling the rest of the frame, a
dense multi-channel scroll of raw oscillating signal traces (rendered as tightly packed black and
white waveforms) — a live view of the ongoing recording. The file path names the recording
"bravo3_chang," identifying this as data from the UCSF BRAVO trial (Chang lab) participant BRAVO3.

## Slide 65 — An accurate speech BCI for personal use

Credited **"Card et al. 2024."** Left, a brain render marking three colored cortical regions along
the precentral gyrus, separated from the parietal side by a dashed **"Central Sulcus"** line:
**Middle precentral gyrus (55b)** (magenta), **Ventral premotor cortex (6v)** (green, with "d" and
"v" sub-sites marked), and **Primary motor cortex (4)** (blue) — small black squares mark
individual array placements within each region. Right, a line chart: y-axis **"Word error rate
(%)"** (0–30); x-axis **"Session number"** (1, 2, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, then a
break, 69, 80), with **"Days since implant"** and **"Hours"** given as extra rows beneath each
session tick. Two data series: a red **"50 words"** line (only two early points, both essentially
at **0%**, annotated); and a blue/purple **"125k words"** line with diamond markers, starting
around 10% at session 2, peaking near 14% at session 4, then declining steadily through the
teens-numbered sessions to roughly 1–3% by sessions 69–80, with an annotation **"0.99%"** marking
one of the later low points. Two dashed grey horizontal reference lines near the top of the chart
mark prior benchmarks: **"Ref. 19 (1024 words)"** (~26%) and **"Ref. 21 (125k words)"** (~24%) —
showing this system's word error rate falling far below both, especially as sessions accumulate.

## Slide 66 — (untitled) Video frame: a cursor/menu control interface

No slide title. A video still of a task display: a black screen with a red dot near its center
(a cursor or fixation target) and four labelled circular buttons at the screen's corners —
**"100% CORRECT"** (top left, green outline), **"DONE"** (top right, purple outline), **"MOSTLY
CORRECT"** and **"INCORRECT"** (stacked at left, below the top-left button), and **"Activate
spelling mode"** (bottom right, green outline). A participant (seen from behind, bald head, an
over-ear device visible) sits facing the screen; a tree and patio are visible through a window
behind the display.

## Slide 67 — Restoring effortless and natural communication by decoding inner speech

Credited **"Erin, unpublished."** Left, a bar chart titled **"t12,"** y-axis **"Decoding Accuracy
(%)"** (0–100), x-axis categories **Attempted, Mimed, Motoric Inner Voice, Auditory Inner Voice,
Imagined Listening, Listening, Silent Reading.** Two data series (legend): **6v ventral** (salmon/
red) and **6v dorsal** (yellow), with a dashed horizontal chance-level reference line around 14%.
The 6v ventral bars are consistently much higher than 6v dorsal across every category — roughly
98% for Attempted, declining gradually across the middle categories to about 56% for Listening,
then rising back to about 72% for Silent Reading — while 6v dorsal stays low throughout, roughly
20–30% for most categories (peaking near 52% only for Attempted). Right, an illustration: a
stylized human head split vertically into two profiles — one tan-toned showing an eye, one black
silhouette showing a second eye — captioned **"Cope & Kalantzis / Transpositional Grammar •
meaningpatterns.net/inner-speech,"** an artistic depiction of inner speech/duality of thought.

## Slide 68 — Language processing in the brain

The same figure as slide 30, credited **"Fedorenko et al. 2024,"** repeated here: the four
labelled brain-render panels (Perception, Language, Motor planning, Knowledge and reasoning /
Intended meaning) connected by blue comprehension and red production arrows, revisited as context
for the inner-speech and future-BCI discussion in this section.

## Slide 69 — BCIs raise new neuroethics considerations

Credited **"Shenoy & Yu, Brain Machine Interfaces."**

- Should BCIs be allowed to read inner thoughts and memories?
  - Read out inner speech that would not naturally be enacted?
  - Read out memories that may otherwise be lost to Alzheimer's disease?
  - Read out subconscious fears to assist desensitization psychotherapy?

## Slide 70 — BCIs raise new neuroethics considerations

Credited **"Shenoy & Yu, Brain Machine Interfaces."**

- Should BCIs be allowed to enhance cognitive function beyond natural levels?
  - Move a robotic arm faster and more accurately than a native arm?
  - Purchase a memory to skip a grade of mathematics in high school?
- We are currently grappling with the same questions:
  - Steroids, stimulants, elective plastic surgery …

## Slide 71 — BCIs raise new neuroethics considerations

Credited **"Shenoy & Yu, Brain Machine Interfaces."** Block text: "Although some of these ideas
and questions may appear farfetched at present, as brain function and dysfunction continues to be
revealed, BCI systems could build on these discoveries and create even more daunting ethical
quandaries. But equally important is the immediate need to help people suffering from profound
neurological disease and injury through restorative BMIs. In order to achieve the right balance it
is imperative that we as physicians, scientists and engineers proceed in close conversation and
partnership with ethicists, government oversight agencies, and patient advocacy groups."

## Slide 72 — Summary

- Recent advancements in AI and NLP, combined with years of neuroscience and neuroengineering
  research, show potential for restoring natural communication to people with speech impairments.
- We will soon have systems to assist people with communication disorders and paralysis.
- And understand better how the brain processes language!
- This brings hope to people like Howard and T12!

## Slide 73 — (untitled) Video frame: a news segment on ALS speech BCI research

No slide title. A video still from a "TODAY" show segment: a close-up profile of a woman with a
wired headband connector, looking off-camera; in the background, a man stands near a desk holding
papers, with two monitors showing colorful multi-channel signal traces and a text document, and an
orange cup on the desk. A caption bar reads **"NEW HOPE FOR ALS PATIENTS"** beside the **TODAY**
show logo.

## Slide 74 — (untitled) Photo collage: Howard Wicks

No slide title. Three photographs captioned **"@howard_wicks":** left, three young men on a beach
boardwalk — one in a wheelchair, embraced by two friends wearing matching "LOCKED-IN" polo shirts;
middle, a close-up selfie of a young man and a woman (cheek to cheek); right, the same young man in
a wheelchair outdoors at what appears to be an equestrian event, horses and riders visible behind
him, a dog draped across his lap. The collage brings the lecture's opening story (slides 2–3) full
circle, showing Howard living his life.

## Slide 75 — Acknowledgements

A group photograph of roughly fourteen lab members standing outdoors on a lawn in front of a
building, with a reddish-brown dog in front of the group, captioned **"Stanford Neural Prosthetics
Translational Lab (NPTL)."** At left, a separate photo and caption memorialize **"Krishna Shenoy
(9/3/1968 – 1/21/2023)"** beside the Howard Hughes Medical Institute (HHMI) logo. Text: "Participants
T5, T6 and T12." A **"Funding"** list at bottom credits the ALS Association; Stanford Bio-X, Wu
Tsai Neuroscience, and OPA; NIH (NINDS, NIDCD, NICHD); the VA Rehab. R&D Service; Massachusetts
General Hospital ECOR; and individual donors Larry and Pamela Garlick, Samuel and Betsy Reeves, and
John Gunn — alongside institutional logos for MGH, Brown University, Harvard, Stanford, the VA, and
the Brown Institute for Brain Science.
