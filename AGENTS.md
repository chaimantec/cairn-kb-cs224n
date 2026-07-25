# How this knowledge base is organized

This repo is the knowledge base for **CS224N — Natural Language Processing with Deep Learning (Stanford, Spring 2024)**. It is read
by Cairn's in-extension AI chat, which fetches files over
raw.githubusercontent.com and follows relative markdown links.

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
- **Printed slide numbers do not always equal PDF page numbers.** They do for lectures 1–3,
  5 and 6. They do *not* for lecture 4, whose printed numbers run 1–49 over a 45-page PDF
  because four slides were hidden in the source deck and never exported. Always cite the
  **printed** number, and check the header of the relevant `raw/slides/` file, which states
  the mapping for that deck. When transcribing a new deck, verify the printed numbers
  against the page count rather than assuming.
- **Slides beat transcripts on conflict.** The transcripts are corrected auto-captions;
  the slides are what the instructor wrote. Where the captions are garbled or the
  dictated notation is unreliable, cite the slide.
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
