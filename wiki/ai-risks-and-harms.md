# AI risks and harms

[Lecture 18](18-nlp-linguistics-philosophy.md) closes CS224N with a section on "our AI future"
(slides 49–64), and its argument is about **ordering**: which risks deserve the attention they are
getting, and which deserve more. Christopher Manning's position is that existential risk is
overweighted and the concentration of power is underweighted, and he states it as a position
rather than a consensus.

## Automation and jobs: an old fear

Two press clippings do the work here (≈1:04:03–1:05:37).

The *New York Times*, **1928**: "March of the Machine Makes Idle Hands — Prevalence of
Unemployment with Greatly Increased Industrial Output Points to the Influence of Labor-Saving
Devices as an Underlying Cause." *Time*, **1961**: "In the past, new industries hired far more
people than those they put out of business. But this is not true of many of today's new
industries. Today's new industries have comparatively few jobs for the unskilled or semiskilled,
just the class of workers whose jobs are being eliminated by automation."

Manning's read: "this is a longstanding fear which, at least so far, has not been realized — here
we are, in a country in which not everyone might have the work that they wish they had, but,
overall, almost everybody has a job, and many people are working a lot of hours a week." The
promised three-day working week never arrived. He also notes that many labour-saving machines —
washing machines, dishwashers, sewing machines — turned out to be things people wanted.

The historical framing he attaches to the 1928 clipping is the one that carries into the next
section: it was published "at a time when a small group of immensely powerful and rich men
dominated the United States, just before the Great Depression," and what followed was policy
change that "distributed wealth and work much more evenly across the country."

## Concentration of power: the risk he does weight

"Will almost all the money go to five to ten enormous technology giants? I actually think this is a
more serious worry — this seems to be the direction that we're headed in at the moment"
(≈1:05:37–1:07:11).

The mechanism is network effects plus a concentration of AI talent. The analogy is the early
twentieth century, where "domination of the new transportation networks, like railways … led to a
few people dominating the economic system," and the resolution was political: after the Depression,
"countries successfully dealt with the monopolistic power of a small number of companies, and, with
political leadership, we could do that again."

The conclusion is deliberately not technological: "that's a political problem to solve, rather than
actually being a technological problem to solve" — with the caveat that "there's not much sign of
political leadership right at the moment."

## The existential-risk debate

Slide 54 collects the headlines that made x-risk mainstream: "Pausing AI Developments Isn't Enough,
We Need to Shut It All Down," "How Rogue AIs May Arise," "AI 'Godfather' Geoffrey Hinton Warns of
Dangers as He Quits Google," "We Must Slow Down the Race to God-Like AI" — and Manning notes these
concerns motivated the creation of AI safety institutes in the US and UK.

His own view: "I don't personally give these concerns too much credence, and I think there's
started to be increasing pushback against them" (≈1:08:47). Slide 55 presents three lines of
criticism.

**François Chollet** (the architect of Keras):

> There does not exist any AI model or technique that could represent an extinction risk for
> humanity, not even if you extrapolate capabilities far into the future via scaling laws. Most
> arguments boil down to: this is a new type of technology, it could happen.

**Joelle Pineau** (a senior Meta AI leader) calls the discourse "unhinged," and Manning spells out
the specific logical flaw he thinks matters (≈1:09:34): if the elimination of humanity is treated
as *infinitely* bad, "that means any nonzero chance, multiplied by infinity, will be bigger than
the badness of anything else that could happen in the world — but that isn't actually a sensible
way to have rational discussion about the outcomes." The objection is to the structure of the
utilitarian argument, not to caring about the outcome.

**Timnit Gebru** and others argue that the focus on existential risk distracts — "and, if you're
more cynical, a lot of the *purpose* of this focus on existential risk — is to distract away from
the immediate harms that are arising from companies deploying automated systems, including their
biases, worker exploitation, copyright violation, disinformation, growing concentration of power,
and regulatory capture by leading AI companies" (≈1:10:19).

## Present-day harms

Slide 56's list, of what sits "behind all the discussions of our amazing AIs": disinformation,
deception, hallucinations, homogeneity of decision-making, violation of copyright and people's
creativity, carbon emissions, erosion of rich human practices (≈1:11:05).

For NLP specifically (slide 57): generating offensive content, generating untruthful content, and
enabling disinformation. Manning sharpens the third into a research question worth taking seriously
(≈1:11:51):

> if models can reason well about text, can they also be persuasive in communicating incorrect
> information or opinions to users — perhaps there are new possibilities for doing very
> personalized misinformation propagation that more easily persuades human beings than traditional
> methods of political advertising.

His read of the evidence: "it's still being debated in the literature, but there are now multiple
studies suggesting that humans can be influenced by disinformation generated by AIs." And the
expectation that text is not the worst of it — "it's likely that visual fakes are going to be even
more compelling in a political context," with major electoral incidents likely (≈1:12:36).

Two concrete harms appear earlier in the same lecture rather than in this section:

- **Hallucination in high-stakes domains.** A Stanford RegLab study of legal NLP systems found
  made-up material in roughly **one answer in six** — "which isn't a very good accuracy rate if
  you're someone who's wanting to rely on these systems for legal advice" (≈14:50).
- **Systematic bias.** NLP systems "remain very biased against various cultures and religions. They
  have certain social norms, you could say, that they pick up from somewhere, but those social
  norms are very biased against certain groups" (≈15:35). Compounded by the
  [multilingual gap](18-nlp-linguistics-philosophy.md#open-problems): whatever holds for English is
  worse for every other language, and thousands of low-resource languages are out of reach
  entirely.

## The reframing

The sentence the section builds to (≈1:12:36):

> what we should be doing is worrying not about existential risks, but worrying about what people
> and organizations with power will use AI to do.

The precedent offered is social media: it "was meant to lead to new freedoms for people across the
globe, bringing the positives of free political thought and improved human lives in large measure.
That isn't what's happened — new technologies get captured by powerful people and organizations who
master the new technological options, and AI and machine learning is being increasingly used for
surveillance and control, and we're seeing that around the world at the moment" (≈1:13:24).

The lecture ends with Carl Sagan, from *The Demon-Haunted World* (≈1:14:58):

> I have a foreboding of a world in my children's or grandchildren's time — when awesome
> technological powers are in the hands of a very few, and no one representing the public interest
> can even grasp the issues; when the people have lost the ability to set their own agendas or
> knowledgeably question those in authority; when, clutching our crystals and nervously consulting
> our horoscopes, our critical faculties in decline, unable to distinguish between what feels good
> and what's true, we slide, almost without noticing, back into superstition and darkness.

Manning's final claim to the class is that this — not extinction — "is actually much more the risk
that humanity is facing," and that it is the argument for valuing education and for open source,
"that supports the broad dissemination of learning" (≈1:15:44).

## See also

- [Lecture 18](18-nlp-linguistics-philosophy.md) — the source lecture.
- [Neuroethics](neuroethics.md) — lecture 14's treatment of risk, for a technology where the
  privacy stakes are more immediate.
- [Benchmark contamination](benchmark-contamination.md) — the evaluation integrity problem, which
  is where "we cannot tell how good these systems are" meets "we are deploying them."
- [Evaluating LLMs](evaluating-llms.md) — what we can and cannot currently measure.
- [RLHF](rlhf.md) and [instruction finetuning](instruction-finetuning.md) — the main technical
  levers for reducing offensive and untruthful output.
