# Neuroethics

The closing section of [lecture 14](14-brain-computer-interfaces.md) (slides 69–71, ≈1:08:50–
1:11:59). Fan is explicit that he is not offering answers: "we're not really looking for an
answer here, but I think the point is that maybe we just want to keep this in discussion."

The reason it belongs in a technical lecture is that the questions are made live by a specific
technical result. As long as a [brain-computer interface](brain-computer-interfaces.md) decodes
*attempted* speech or *attempted* movement, it reads only what the user chose to try to do — the
intention was already on its way out of the body. The moment it decodes **inner speech**, that
stops being true.

## The trigger: inner speech decoding

Speech BCIs currently top out around 60–70 words per minute, against 150 for natural conversation.
Part of the gap is that participants who have not spoken in years find *attempting* to speak slow
and effortful. So the natural next step is to decode the inner voice directly, and slide 67
reports preliminary evidence that this is possible: attempted speech decodes at about 98% on one
array, while mimed speech, a "motoric inner voice", an "auditory inner voice", imagined listening,
listening and silent reading all decode well above a chance level near 14% (≈1:08:04).

That result is what makes everything below a practical question rather than a thought experiment.

## Question 1 — should BCIs read inner thoughts and memories?

Slide 69 poses it as three sub-questions that deliberately do not point the same way:

- Should a BCI **read out inner speech that would not naturally be enacted** — the things you
  thought and chose not to say?
- Should it **read out memories that would otherwise be lost** to Alzheimer's disease?
- Should it **read out subconscious fears** to support desensitization psychotherapy?

The first is a privacy violation with no obvious upside. The second and third are plausibly
enormous goods. They are the same capability. Fan puts the tension plainly: "what if you can
decode something like your private thoughts or private memories that you don't want to express?"
against the possibility of helping people who have lost their memories (≈1:09:38).

There is also a conceptual obstacle that is not ethical at all but bears on the ethics. Speech is
a **linear** external representation of thought; thought itself is multi-dimensional. So it is
not clear that "reading inner speech" and "reading thought" are the same thing, or where you
would even place the arrays to attempt the latter (≈1:08:50). And not everyone experiences inner
speech in the first place.

## Question 2 — should BCIs enhance beyond natural function?

Slide 70 moves from restoration to enhancement:

- Should a BCI let someone **move a robotic arm faster and more accurately than a native arm**?
- Should someone be able to **purchase a memory** to skip a grade of mathematics? (Fan's version
  in the room: "can you actually purchase a memory so that you can skip this CS224N class?",
  ≈1:10:26.)

The slide's own answer to anyone who finds this science-fictional is that **we already grapple
with exactly these questions**: steroids, stimulants, elective plastic surgery. The line between
restoring a function and exceeding it is one society negotiates continually, and BCIs do not
introduce the problem so much as sharpen it.

## The stance the lecture recommends

Slide 71 quotes Shenoy & Yu's textbook at length, and it is the closest thing to a
recommendation the lecture makes:

> "Although some of these ideas and questions may appear farfetched at present, as brain function
> and dysfunction continues to be revealed, BCI systems could build on these discoveries and
> create even more daunting ethical quandaries. But equally important is the immediate need to
> help people suffering from profound neurological disease and injury through restorative BMIs.
> In order to achieve the right balance it is imperative that we as physicians, scientists and
> engineers proceed in close conversation and partnership with ethicists, government oversight
> agencies, and patient advocacy groups."

Two things are held together there. The speculative harms are real and worth anticipating; and
the present need is not speculative at all. The recommended posture is not a moratorium but a
partnership — engineers in continuous conversation with ethicists, regulators and patient
advocacy groups, rather than handing the question off after the fact.

## Related pages

- [Brain-computer interfaces](brain-computer-interfaces.md) — the capability these questions are
  about.
- [Lecture 14 — Brain-computer interfaces](14-brain-computer-interfaces.md) — the inner-speech
  results that prompt them.
- [Evaluating LLMs](evaluating-llms.md) and [RLHF](rlhf.md) — where the course's other questions
  about deploying systems that act on human intent are discussed.
