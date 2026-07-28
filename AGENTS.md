# How this knowledge base is organized

This repo is the knowledge base for **CS224N — Natural Language Processing with Deep Learning (Stanford, Spring 2024)**. It is read
by Cairn's in-extension AI chat, which fetches files over
raw.githubusercontent.com and follows relative markdown links.

Every page in it was written by **Claude Opus 5** and **Claude Sonnet 5** running as agents.
Opus 5 writes the wiki prose and makes the editorial calls — what a lecture establishes, how a
garbled passage resolves against the slides, which numbering claims are safe to assert. Sonnet 5
does the script-checkable bulk work: reading each deck page by page, and copy-editing the
auto-generated captions. If you extend this KB, keep that split — the transcript edit is safe to
delegate because the checks below verify it mechanically, and the wiki prose is not, because
nothing scores it.

## Layout

| Path                | Contents                                                      |
| ------------------- | ------------------------------------------------------------- |
| `INDEX.md`          | Entry point. Course summary + annotated table of contents.    |
| `wiki/`             | Durable pages: one per lecture, plus cross-lecture topics.    |
| `raw/slides/`       | Full text of every slide, numbered. See the caveat below.     |
| `raw/transcripts/`  | Edited lecture transcripts with `[MM:SS]` paragraph marks.    |
| `raw/transcripts/original/` | Verbatim captions. Reference only — prefer the edited ones. |
| `sources.md`        | Every course document with its canonical URL and fetch date.  |
| `TODO.md`           | Build tracker. Unchecked boxes are outstanding work.          |

Slide and handout **PDFs are not committed to this repo** — `sources.md` holds
their canonical `web.stanford.edu` URLs instead. This KB is consumed by an agent
that reads markdown and cannot extract text from a PDF blob, so the URL is the
useful artifact and the decks would have added over 100MB to every clone. Link
slides at their source URL, not at a local path.

## Conventions

- **INDEX.md is the front door.** The chat reads it first on every conversation.
  Every wiki page must appear there with a one-line description of what it holds.
  An unindexed page is effectively invisible.
- **Relative links between KB pages** (`[gradient descent](gradient-descent.md)`,
  `[transcript](../raw/transcripts/05-recurrent-neural-networks.md)`). Absolute GitHub
  URLs break when the repo is renamed or forked. Course PDFs are the exception:
  link those at their canonical course-site URL, since they are not committed.
- **Cite everything, and pick the right citation.** Use a **slide number** for what is
  written on a slide (equations, tables, definitions) and an **`[MM:SS]` timestamp** for
  what is said aloud (asides, worked reasoning, answers to students).
- **Printed slide numbers do not always equal PDF page numbers.** They do for lectures 1–3, 5–9,
  12, 13 and 15. They do *not* for lecture 4, whose printed numbers run 1–49 over a 45-page PDF
  because four slides were hidden in the source deck and never exported, nor for lecture 10,
  whose printed numbers run 1–76 over a 71-page PDF with five hidden slides, nor for lecture 11,
  whose printed numbers run 1–99 over a 94-page PDF, also with five hidden — so in those two the
  offset **grows** through the deck rather than being constant. **Lecture 14 is a different case
  again: its deck prints no slide number on any page at all**, so its "slide N" is a PDF page
  position by construction rather than a number read off the slide, and its file header says so.
  **Lecture 16 maps 1:1 but prints its number in the bottom-right corner** rather than the
  bottom-left every other deck uses, which initially fooled `slide_number_map.py`'s
  corner-scanning heuristic into reporting no printed numbers at all — check both corners before
  trusting a null result. When a deck yields no printed numbers, say that plainly instead of
  reporting a clean 1:1 mapping — a wrong 1:1 claim is the most damaging thing this pipeline can
  emit, because everything downstream trusts it. Always cite the **printed** number, and check
  the header of the relevant `raw/slides/` file, which states the mapping for that deck. When
  transcribing a new deck, verify the printed numbers against the page count rather than
  assuming, and do not assume a constant offset once you find one.
- **Not every lecture in the playlist is from the course the site describes.** Lectures 1–8 and
  11–18 are Spring 2024; lectures 9 and 10 are Winter 2023 recordings by John Hewitt and Xiang
  Lisa Li, whose decks come from the `cs224n.1234` archive. Before transcribing a deck for a new
  lecture, check that it matches the video — announcements ("Assignment 5 is out on Thursday"),
  the lecturer's name, and the day-of-week references on the title and reminder slides are the
  reliable tells. `sources.md` records which site each deck came from.
- **A lecture may carry several conflicting numbers.** Catalog position, video title, deck title
  slide and deck filename can all disagree; lecture 10 has four different numbers, and lectures
  11, 12 and 13 are titled "Lecture 10", "Lecture 11" and "Lecture 12" by both the video and the
  deck — the off-by-one is now systematic from position 11 onward, so expect it to continue.
  Position **14** continues it in the video title and the deck filename
  (`…lecture13-speech-bci.pdf`), but its deck carries **no lecture number at all**: it is a guest
  research talk titled only *Speech Brain-Computer Interfaces for Restoring Natural
  Communication*. Position **15** continues it again — video and deck both title it "Lecture 14"
  (Reasoning and Agents, Shikhar Murty) — and position **16** goes one step further, titled
  "Lecture 15" by the video while the deck's own title page reads simply "Life after DPO" (Nathan
  Lambert, AI2). This repo names files by the **Cairn catalog position**, and the slide file
  records the others in a table so any of them can be resolved.
- **Slides beat transcripts on conflict.** The transcripts are corrected auto-captions;
  the slides are what the instructor wrote. Where the captions are garbled or the
  dictated notation is unreliable, cite the slide.
- **Mathematics in the wiki is LaTeX**, `$...$` inline and `$$...$$` displayed on its own
  lines. Both render in Cairn's chat and on github.com. Never put an equation in a code
  fence or an indented block — those render as source, so the reader sees `\frac{...}`
  instead of a fraction. Match the course's own notation ($u_o^{\top} v_c$, $h^{(t)}$,
  $W_h$, $\theta$), define every symbol on first use because pages are read out of order,
  and keep prose around the equation saying what it does. A literal dollar sign in prose
  must be escaped as `\$`, or it will pair with a delimiter and swallow the text between.
  `raw/transcripts/` is exempt: it is a verbatim record and stays spelled out.
  `raw/slides/` reproduces the deck's own layout in Unicode and is left as transcribed.
- **Never invent course content.** If a source is unclear at some point, say so on the
  page. Do not fill the gap from outside knowledge — the chat presents these pages as
  authoritative material from this course. Recovering a mangled term from unambiguous
  context is reading the source; supplying a number or a name the sources do not
  contain is not, and the KB says so explicitly where that happens (see the mini-batch
  size in [gradient descent](wiki/gradient-descent.md)).
- **Prose over fragments.** The chat quotes these pages to learners; bullet
  fragments quote badly.

## Rebuilding

Built and updated by the `cairn-kb` skill. To add newly released lectures,
append entries to `TODO.md` and re-run the skill — it only does unchecked work.
