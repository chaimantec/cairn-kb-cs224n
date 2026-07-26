# KB build — CS224N (Natural Language Processing with Deep Learning, Stanford, Spring 2024)

The course has 23 lectures, built incrementally.

- **Run 1** (complete): lectures 1–2, plus the site crawl and `sources.md`.
- **Run 2** (complete): lectures 3–4.
- **Run 3** (complete): lectures 5–6.
- **Run 4** (complete): no new lectures — converted all mathematics in the wiki to LaTeX.
- **Run 5** (complete): lectures 7–8.
- **Run 6** (in progress): lectures 9–10.
- Lectures 11–23 remain deferred; video ids are listed at the bottom.

---

# Run 6 — lectures 9–10

Catalog positions 9 and 10. Note the catalog's own titles: position 9 is
"Lecture 9 - Pretraining", position 10 is "Lecture 11 - Natural Language Generation".

## Provenance — these two videos are Winter 2023, not Spring 2024

Lectures 1–8 were Spring 2024 recordings that matched the Spring 2024 decks. These two
are not. Evidence, gathered before any transcription:

- The Spring 2024 schedule has **no** Natural Language Generation lecture at all. Its
  week-5 Thursday slot is Post-training (Archit Sharma).
- Lecture 9's lecturer says "this lecture, the Transformers lecture, and then to a lesser
  extent **Thursday's lecture on natural language generation**" (≈0:05) — a Thursday NLG
  lecture only exists in Winter 2023 (Thu Feb 9).
- Both lecturers refer to **Assignment 5**. Spring 2024 ran only A1–A4; Winter 2023's site
  links `assignments/a5.pdf` from the NLG row.
- The decks confirm it outright. `cs224n-2023-lecture9-pretraining.pdf` page 1 reads "John
  Hewitt / Lecture 9: Pretraining" and page 2 reads "Assignment 5 is out on Thursday! It
  covers lecture 8 and lecture 9 (Today)!", both matching the transcript word for word.
  The NLG deck's announcement slide matches Lisa Li's spoken announcements item for item
  (AWS by midnight, proposals Tuesday, A4 just due, A5 due Fri Feb 17, HuggingFace tutorial
  Friday).

So the decks for this run come from the **Winter 2023** archive
(`cs224n.1234`), not from the Spring 2024 site the run-1 crawl covered.

## Lecture numbering for NLG

This one lecture carries four different numbers. Stated here and in the slide file so a
citation is never ambiguous:

| Where | Number |
|---|---|
| Cairn catalog position | 10 |
| Catalog / YouTube title | "Lecture 11" |
| Deck's own title slide | "Lecture 12: Neural Language Generation" |
| Deck filename on the W23 site | `cs224n-2023-lecture10-nlg.pdf` |

Repo files use the catalog position (`10-…`), per the convention set in run 3 for
deck-vs-catalog title disagreements.

## Transcripts
- [ ] 09 Pretraining — video DGfCRXuNA2w (102 paragraphs)
- [ ] 10 Natural Language Generation — video N9L32bFieEY (102 paragraphs)
- [ ] Copy-edit both by hand, verbatim originals kept in `raw/transcripts/original/`.

## Slides
- [x] raw/slides/09-pretraining.md — all 54 slides. Printed numbers match PDF pages 1:1;
      pages 38, 39 and 46 print no number but sit in sequence. Deck title is "Lecture 9:
      Pretraining" (John Hewitt), matching the catalog. Noted that slide 2's lecture plan
      lists decoders/encoders/encoder-decoders but the deck presents them in the order
      encoders → encoder-decoders → decoders, which is what the lecture follows.
- [x] raw/slides/10-natural-language-generation.md — all 71 pages, cited by the deck's
      **printed** numbers, which run 1–76. Printed 35, 41, 47, 54 and 66 are absent from the
      PDF (hidden in the source deck); 76 − 5 = 71, so every page is accounted for. The
      offset accumulates rather than being constant, so the file carries a full page↔slide
      mapping table. Pages printing no number: the title page and printed 10, 11, 12, 53, 64.
      Slides 39, 40 and part of 42 re-show figures from 22, 24 and 26; transcribed with a
      pointer rather than twice in full, per run 3's convention.
- [x] The three ethics slides (73, 74, and the jailbreak screenshot on 70) reproduce hate
      speech, sexual violence and profanity in full on the deck. The slide file states
      precisely what each figure shows and what the finding is, and cites the source paper,
      but does not reproduce the passages — this repo is public and the pedagogical content
      is the finding, not the invective. Slide 72's stereotype table is transcribed in full,
      since the stereotypes *are* the data.

## Wiki
- [ ] wiki/09-pretraining.md
- [ ] wiki/10-natural-language-generation.md
- [ ] Topic pages (new)
- [ ] Update existing topic pages that these lectures extend
- [ ] Update INDEX.md
- [ ] Update sources.md with the two Winter 2023 deck URLs
- [ ] Link sweep + KaTeX validation

## Publish
- [ ] Commit and push (kbUrl already set, no re-link needed)

---

# Run 5 — lectures 7–8

## Transcripts
- [x] 07 Attention, Final Projects and LLM Intro — video J7ruSOIzhrE (100 paragraphs)
- [x] 08 Self-Attention and Transformers — video LWMzyfvuehA (100 paragraphs)
- [x] Copy-edited both by hand, verbatim originals kept in `raw/transcripts/original/`.
      All 100+100 `[MM:SS]` timestamps verified in order; number inventories compared,
      differences all explained (garbled duplicates merged — "201 2015"→2015,
      "assignments 34"→"3, 4"; numbers spelled as words; ASR-spaced "4 000"→"4,000").
      Restored against slides: BLEU = Bilingual Evaluation Understudy; "boff
      evaluations"→bake-off (slide 43's own phrase "Kaggle/bake-off/shared task");
      "tanglong and me"→Luong (and Pham) and Manning 2015, bilinear/multiplicative
      attention; "B hour Al paper"/"dimma bad now K huno and Yoshua Benjo"→Bahdanau,
      Cho, and Bengio 2014, additive attention; French toy sentence "exiler"→"il
      a m'entarté" (slide 7, "he pied me"); "bir"→BERT/minBERT; project names/authors
      (Deep Poetry — Xie, Rastogi, Chang; Carol Hsin's differentiable-neural-computer
      reimplementation; Word2Bits — Lam; CodeLlama-Fortran-PEFT — Govande, Kang, Shi;
      AI-driven fashion cataloging — Ma, Gopinath), all confirmed against slides 41–50.
      Math in the transcripts stays spelled out (bold/Unicode-subscript notation, not
      LaTeX) per the repo's established convention — LaTeX is wiki-only. Two literal
      dollar amounts escaped (`\$50`).

## Slides
- [x] raw/slides/07-attention-final-projects-and-llm-intro.md — all 73 slides.
      Printed numbers match PDF pages 1:1, no gaps or offset.
- [x] raw/slides/08-self-attention-and-transformers.md — all 62 slides. Printed
      numbers match PDF pages 1:1, no gaps or offset.

## Wiki
- [x] wiki/07-attention-final-projects-and-llm-intro.md
- [x] wiki/08-self-attention-and-transformers.md
- [x] Topic pages — five new: attention, self-attention, transformer,
      evaluating-machine-translation (BLEU), final-project-guidance.
- [x] Updated existing topic pages — seq2seq-and-encoder-decoder.md's and
      machine-translation.md's forward-pointers to "attention" turned into real links,
      plus new Related-pages entries.
- [x] Updated INDEX.md — coverage note, 2 lecture entries, 5 topic entries, raw
      materials section (slide/transcript counts, garbled-term examples for run 5).
- [x] Link sweep (scripted): all relative links resolve, all in-page anchors resolve,
      no wiki page points into gitignored raw/pdfs/. Caught and fixed a sed/perl bug
      that had merged several display-math lines with following prose in
      attention.md, self-attention.md and transformer.md.

## Publish
- [x] Commit and push (kbUrl already set on the catalog entry, no re-link needed) —
      https://github.com/chaimantec/cairn-kb-cs224n/commit/2c4d3c6

## Notes for run 6

- **Deck-vs-catalog title matching held for lectures 7–8**, unlike the divergence past
  lecture ~9 noted after run 3: `lecture07-final-project.pdf` and
  `lecture08-transformers.pdf` matched catalog lectures 7 and 8 by number as well as
  title. Re-verify by title once lectures resume past 9, per
  [[cairn-kb-cs224n]]'s existing caveat.
- **LaTeX is wiki-only — do not use it in `raw/transcripts/`.** This run initially wrote
  the edited transcripts with `$...$` LaTeX math (following the run-4 "write math in
  LaTeX" habit), then discovered lecture 3's transcript header explicitly documents the
  actual convention: mathematical expressions dictated aloud go into the transcript as
  Unicode symbols and **bold** variable names (`∂s/∂W`, **W**, **x**ᵢ), not LaTeX — LaTeX
  is reserved for the wiki. Both lecture 7 and 8 transcripts were rewritten to match
  before this run finished. Read `raw/transcripts/03-backpropagation-and-neural-networks.md`'s
  header before hand-editing any future lecture's transcript.
- **A `sed`/`perl` regex bug ate trailing newlines and merged lines.** While de-indenting
  display-math lines pulled under numbered-list items in the new wiki pages, a
  `perl -pe 's/^\s+(\$\$.*\$\$)\s*$/$1/'` one-liner used `\s*` at the end of the
  pattern, which in Perl matches `\n` too — silently stripping the line's trailing
  newline and merging it with the next line whenever the line matched. Caught by
  grepping for `$$...$$` not followed immediately by end-of-line. If de-indenting or
  otherwise reflowing text near LaTeX blocks with a regex one-liner again, verify with a
  targeted grep afterward rather than trusting a clean diff — the corruption reads fine
  in a `diff` summary line-count check but breaks the equation rendering.
- Lectures 9–23 remain deferred; see the video-id list at the bottom. Several later
  catalog entries (11 Post-training, 19–23 the guest lectures and tutorials) may have no
  deck at all and would be transcript-only — see run 1's note.

---

# Run 4 — LaTeX mathematics pass (no new lectures)

The skill gained a "Write mathematics in LaTeX" rule. This run applied it to everything
already built; no transcripts, slides or lectures were added.

## Wiki
- [x] All 28 wiki pages converted — `$...$` inline, `$$...$$` displayed. Equations that
      had been spelled out in prose, written in Unicode (`h⁽ᵗ⁾ = σ(W_h h⁽ᵗ⁻¹⁾ + …)`), or
      parked in indented code blocks are now real math, in the course's own notation, with
      every symbol defined on first use.
- [x] Math pulled out of code blocks. Three fenced blocks (the gradient-clipping
      pseudo-code in gradient-descent and vanishing-and-exploding-gradients) and roughly
      twenty 4-space indented blocks were rendering as source — the reader saw the markup,
      not the formula. The clipping algorithm became an `aligned` block; the LSTM gate
      equations, the arc-standard transition system and the perplexity derivation likewise.
      `backpropagation.md`'s Python fence stayed a fence: it is code.
- [x] Some equations added where the page previously only described one in words and the
      slide had it written out — word2vec's likelihood and objective (slides 28–29), the
      sigmoid definition (lecture 2, slide 12), the SVD factorization (slide 18), the
      negative-sampling objective on lecture 2's page. Each is cited to its slide; nothing
      was supplied that the sources do not contain.
- [x] Notation matched to the decks rather than tidied: `u_o^\top v_c` not `u_o \cdot v_c`,
      `h^{(t)}` not `h_t`, `W_h`/`W_e`/`U_f`, `\theta`, `\ell = i - j`. Where a slide reuses
      an index (slide 58 sums over `i` while `J^{(i)}` also uses `i`), the page keeps the
      slide's notation and says so, rather than silently improving it.

## Dollar signs
- [x] Escaped the literal currency dollars that would otherwise pair with a math delimiter
      and swallow the text between them — `for \$27 a share` in wiki/04-dependency-parsing,
      wiki/syntactic-ambiguity, and (twice) raw/transcripts/04-dependency-parsing. The
      ASCII bracket diagram in syntactic-ambiguity became a fenced block for the same
      reason. `raw/transcripts/original/` was left untouched, as always.

## Conventions
- [x] AGENTS.md — new convention stating the LaTeX rule, the no-equations-in-code-fences
      rule, match-the-course-notation, define-symbols-on-first-use, the `\$` escape, and
      that `raw/transcripts/` (verbatim) and `raw/slides/` (transcribed as printed) are
      exempt.
- [x] INDEX.md — a "Mathematics" note in the header block telling the chat that wiki
      equations are quotable LaTeX while the transcripts spell notation out in words, so
      quote the wiki page when a learner wants a formula.

## Verification
- [x] Every math span extracted and run through **KaTeX** — 837 spans, 0 failures, and the
      unusual ones (`\mathbb{1}`, `\overrightarrow{h}^{(t)}`, `\lVert\cdot\rVert`,
      `\mathbf{8.9}`) also pass in KaTeX's strict mode, which is what GitHub uses.
- [x] Delimiter sweep: no unbalanced `$` or `$$`, no `$$` mid-line, no LaTeX left inside a
      code fence or indented block.
- [x] Link sweep re-run: every relative link still resolves.

## Notes for run 5

- **`raw/slides/` was deliberately left in Unicode.** Those files transcribe what is
  printed on the deck, including its layout; the skill's rule is scoped to the wiki and
  explicitly exempts the verbatim transcripts. Converting the slide files would mean
  re-typesetting a source record, with a real chance of introducing errors into the thing
  the wiki cites as authoritative. If a later run does want them in LaTeX, do it deck by
  deck against the PDF, not by find-and-replace.
- **Display math inside a blockquote does not render on GitHub.** One `$$` block had to be
  moved out from under a `>` quote in vanishing-and-exploding-gradients. Keep equations at
  the top level.
- **Multi-line `$$ … \begin{aligned} … \end{aligned} … $$` works** in both KaTeX and on
  github.com, and is the right shape for the LSTM equations and any algorithm box.
- Markdown tables take inline `$...$` fine, but avoid `\begin{cases}` in a table cell —
  write the piecewise definition on one line instead (see activation-functions).

---

# Run 3 — lectures 5–6

## Transcripts
- [x] 05 Recurrent Neural Networks — video fyc0Jzr74y4 (102 paragraphs)
- [x] 06 Sequence to Sequence Models — video Ba6Fn1-Jsfw (100 paragraphs)
- [x] Copy-edit both into readable sentences, keeping verbatim originals. Read and
      rewritten by hand: punctuation and sentence boundaries added, filler and false
      starts removed, mis-heard terms restored and checked against the slides, student
      questions marked in italics. Lecture 5's captions destroyed *n-gram* ("engram",
      "NRS", "Byram", "5 G"), *Xavier* ("harier"), *AdaGrad* ("adrad"), *Hadamard*
      ("hadam mod") and *Bengio* ("Benjo"); lecture 6's destroyed *Jelinek* ("Fred
      gelan"), *Feigenbaum* ("Ed Fen bam"), *Hochreiter* ("HW", "hot crater"), *Gers*
      ("gz"), *Olah* ("Chris oler"), *Kneser-Ney* ("kessi") and *eigenvalue* ("ion
      value"). All 102 + 100 timestamps preserved in order (verified by diff); number
      inventories compared and every difference accounted for.
- [x] Eight residual ambiguities left marked inline as `[Ed: ...]` rather than guessed —
      an indistinct student answer and a garbled student question (L5); an unrecoverable
      word in a backprop aside, a publication start year, and four heavily garbled
      student questions (L6). Two dropped leading zeros in L5 were restored to 0.35 and
      0.2 and verified against the cumulative distributions on slides 22-24.

## Slides
- [x] raw/slides/05-recurrent-neural-networks.md — all 72 slides. Printed numbers match
      PDF pages 1:1; six pages (3–6, 11, 12) print no number but sit in sequence.
      Deck title is "Language Models and Recurrent Neural Networks"; the catalog calls
      the lecture "Recurrent Neural Networks".
- [x] raw/slides/06-sequence-to-sequence-models.md — all 56 slides. Printed numbers
      match PDF pages 1:1; four full-bleed image pages (19, 43, 44, 48) print no
      number. Slides 4–18 re-run lecture 5's slides 49–63 as a recap; transcribed in
      brief with pointers, except slide 15, which adds the "~7 tokens back" rule of
      thumb that slide 25 contrasts LSTMs against. Deck title is "LSTM RNNs and Neural
      Machine Translation"; the catalog calls the lecture "Sequence to Sequence Models".

## Wiki
- [x] wiki/05-recurrent-neural-networks.md
- [x] wiki/06-sequence-to-sequence-models.md
- [x] Topic pages — nine new: language-modeling, n-gram-language-models,
      recurrent-neural-networks, perplexity, vanishing-and-exploding-gradients, lstm,
      machine-translation, seq2seq-and-encoder-decoder, plus regularization-and-dropout.
      The last was not in the plan but earned a page: lecture 5 slides 7-9 teach it
      properly, including Manning's claim that modern practitioners no longer believe the
      textbook overfitting picture. (Run 2 had dropped a regularization page on the
      grounds that lectures 1-4 do not teach it; lecture 5 does.) Bidirectional and
      multi-layer RNNs were folded into recurrent-neural-networks rather than split out —
      they are variants of one architecture, not separate concepts.
- [x] Update existing topic pages that lectures 5–6 extend — gradient-descent (Xavier
      initialization, the adaptive optimizer family, gradient clipping),
      backpropagation (backpropagation through time and truncation),
      softmax-and-cross-entropy (the LM output layer, its loss, and perplexity as its
      exponential), activation-functions (tanh in RNNs; sigmoid-as-gate vs tanh-as-content
      in the LSTM)
- [x] Update INDEX.md table of contents — coverage note now says lectures 1-6, 2 lecture
      entries, 9 topic entries, slides section including lecture 6's recap caveat,
      transcripts section listing the new garbles
- [x] Update AGENTS.md — replaced the stale "Slide N = PDF page N" line with a convention
      that states which decks are 1:1 and tells the reader to check the slide file header
- [x] Link sweep: all 50 markdown files checked; every live relative link resolves, none
      point into gitignored raw/pdfs/, every wiki page is reachable from INDEX.md, and all
      three course PDF URLs verified against sources.md. Also fixed a stale illustrative
      path in AGENTS.md left over from the skill's CS229 example.

## Publish
- [x] Commit and push — https://github.com/chaimantec/cairn-kb-cs224n (kbUrl already
      set from run 1, so no link_kb.sh needed)

## Notes for run 4

- **Decks repeat each other.** Lecture 6's slides 4-18 re-run lecture 5's slides 49-63
  verbatim as a recap. Transcribing them twice in full would bloat the KB and split
  citations across two files; transcribing them in brief with a pointer to the first
  occurrence, and calling out only what is *new* (slide 15's "~7 tokens back" bullet),
  worked well. Check for this overlap before transcribing.
- **Deck title and catalog title can disagree.** Lecture 5's deck is titled "Language
  Models and Recurrent Neural Networks" but the catalog calls it "Recurrent Neural
  Networks"; lecture 6's deck is "LSTM RNNs and Neural Machine Translation" against the
  catalog's "Sequence to Sequence Models". File names follow the catalog (so transcript,
  slides and wiki slugs match); both titles are stated in the slide file's front matter.
- **Verify the number inventory on the transcript body only.** Diffing numbers between the
  original and edited transcript picks up everything in the new header too, which is pure
  noise. Slice from the first `**[` marker before comparing.
- Unicode sub/superscripts (h⁽ᵗ⁾, b₂, x₁) legitimately remove ASCII digits, so expect them
  in the number diff and account for them rather than reverting.
- **Deck filenames stop matching catalog lecture numbers past roughly lecture 9.** The
  crawled decks run `lecture01`-`lecture16` plus `lecture18` — there is no `lecture17`
  file — while the catalog lists 23 lectures. Catalog 17 "ConvNets and TreeRNNs" is
  `slides-cs224n-spr2024-lecture16-CNN-TreeRNN.pdf`, and catalog 18 is `lecture18-...`.
  Match each catalog lecture to its deck **by title**, not by number, before transcribing,
  and state the mapping in the slide file. Several later catalog entries (11 Post-training,
  19-23 the guest lectures and tutorials) may have no deck at all and would be
  transcript-only.

---

# Run 2 — lectures 3–4

## Transcripts
- [x] 03 Backpropagation and Neural Networks — video HnliVHU2g9U (95 paragraphs)
- [x] 04 Dependency Parsing — video KVKvde-_MYc (102 paragraphs)
- [x] Copy-edit both into readable sentences, keeping verbatim originals. Read and
      rewritten by hand: punctuation and sentence boundaries added, filler and false
      starts removed, mis-heard terms restored and checked against the slides, student
      questions marked in italics. Lecture 3's captions destroyed *ReLU* ("value",
      "realu"), *tanh* ("10 H") and *Swish/GELU* ("Swiss swis and Jello"); lecture 4's
      destroyed *parsing* itself ("paing", "paa", "parza"), *Pāṇini*, *Nivre* and
      *Tesnière*. All 95 + 102 timestamps preserved in order (verified by diff);
      number inventories compared and every difference accounted for.
- [x] Seven residual ambiguities left marked inline as `[Ed: …]` rather than guessed —
      an assignment number, a decimal digit, a subscript, one word that inverted a
      claim (L3); Danqi Chen's name, which the captions drop entirely, two parse
      counts that contradict the Catalan formula on the slide, and a magazine name
      (L4). Each names the slide that settles it, or says it is unrecoverable.

## Slides
- [x] raw/slides/03-backpropagation-and-neural-networks.md — all 85 slides.
      Printed numbers match PDF pages 1:1. Deck has two internal numbering quirks
      (slide 10 headed "Lecture 4", slide 6 headed "7. Neural computation") —
      leftovers from an earlier year's ordering, noted in the file.
- [x] raw/slides/04-dependency-parsing.md — all 45 pages. Printed slide numbers run
      1–49 but **4, 5, 8 and 13 are absent from the PDF** (hidden in the source deck,
      never exported). 49 − 4 = 45 pages, so every page is accounted for. The file
      cites printed numbers and states the offset.

## Wiki
- [x] wiki/03-backpropagation-and-neural-networks.md
- [x] wiki/04-dependency-parsing.md
- [x] Topic pages — six new: backpropagation, matrix-calculus, activation-functions,
      dependency-grammar, transition-based-parsing, syntactic-ambiguity.
      (Planned "neural-network-basics" was split into backpropagation +
      matrix-calculus + activation-functions, which is how lecture 3 actually divides.
      "named-entity-recognition" and "regularization-and-training-practices" dropped:
      NER appears only as the running example and is already covered under
      evaluating-word-vectors; regularization is not taught in lectures 1-4 at all.)
- [x] Update existing topic pages that lectures 3–4 extend — gradient-descent
      (general gradients, backprop), softmax-and-cross-entropy (parser output layer),
      distributional-semantics (embedding POS tags and dependency labels),
      evaluating-word-vectors (UAS/LAS, the arrival of evaluation)
- [x] Update INDEX.md table of contents — coverage note, 2 lecture entries, 6 topic
      entries, slides/transcripts sections including lecture 4's page-number caveat
- [x] Link sweep: all relative links resolve; no wiki page points into gitignored
      raw/pdfs/; all six course PDF URLs verified against sources.md

## Publish
- [x] Commit and push — https://github.com/chaimantec/cairn-kb-cs224n (kbUrl already
      set from run 1, so no link_kb.sh needed)

## Notes for run 3

- **Slide decks do not all number cleanly.** Lecture 4's PDF has 45 pages but printed
  numbers 1–49; slides 4, 5, 8 and 13 were hidden in the source deck and never
  exported. Always check printed number against PDF page and state the mapping —
  `mdls -name kMDItemNumberOfPages -raw <pdf>` gives a trustworthy page count
  (counting `/Type /Page` with a regex does not).
- **Reading a PDF in 20-page chunks can silently skip pages.** Verify by checking the
  printed numbers on the returned slides against the page count, and re-request the
  gaps individually.
- Lecture decks get long: lecture 3 is 85 slides. Budget for that.

---

# Run 1 — lectures 1–2 (complete)

## Transcripts
- [x] 01 Intro and Word Vectors — video DzpHeXVSC5I
- [x] 02 Word Vectors and Language Models — video nBor4jfWetQ
- [x] Copy-edit both transcripts into readable sentences (word2vec was arriving as
      "word Tove", and the captions had no punctuation at all). Read and rewritten by
      hand, not by find-and-replace: punctuation and sentence boundaries added, filler
      and false starts removed, mis-heard terms and names restored, student questions
      marked. All 103 + 102 timestamps preserved in order; verified no numbers lost.
      Verbatim originals kept in raw/transcripts/original/ for reference.

## Slides
- [x] raw/slides/01-intro-and-word-vectors.md — all 40 slides transcribed
- [x] raw/slides/02-word-vectors-and-language-models.md — all 47 slides transcribed
- [x] Cite slide numbers from the wiki pages so the chat can answer "which slide?"

## Crawl
- [x] Fetch course site index: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/
- [x] Download linked PDFs and slides — 33 files (slides for lectures 1–18,
      readings, assignment + project handouts). 43 off-site papers recorded as
      links only. Past student final-project reports excluded deliberately:
      student work, not course material.
- [x] Write sources.md

## Wiki
- [x] wiki/01-intro-and-word-vectors.md
- [x] wiki/02-word-vectors-and-language-models.md
- [x] Topic pages (cross-lecture concepts) — word2vec, distributional-semantics,
      glove, gradient-descent, softmax-and-cross-entropy,
      word-senses-and-polysemy, evaluating-word-vectors
- [x] INDEX.md table of contents

## Publish
- [x] Commit and push — https://github.com/chaimantec/cairn-kb-cs224n (public)
- [x] PATCH kbUrl onto the catalog entry — verified set on course
      b102c48c-0c1b-4fc5-be6d-4c6d1e7211d1

## Build decisions worth remembering for the next run

- **PDF binaries are not committed.** `sources.md` carries the canonical
  `web.stanford.edu` URL for all 76 documents instead. The KB is read by an agent
  that navigates markdown and cannot use a PDF blob, and the decks totalled 163MB
  (one was 62MB). Wiki pages link slides at their source URL. `raw/pdfs/` is
  gitignored; the downloaded copies remain on disk locally, so building
  lectures 3-23 needs no re-crawl.
- **Student final-project reports are excluded** from the crawl
  (`--exclude final-reports --exclude reports_20`) — student work, not course
  material, and inappropriate to redistribute in a public repo.
- Slides for lectures 1-18 are already inventoried, so the next run only needs
  transcripts plus wiki pages.

## Deferred (not in this run)

Lectures 3–23, for a later run. Video ids kept so the next build does not need
to re-query the catalog:

- 03 Backpropagation, Neural Network — HnliVHU2g9U
- 04 Dependency Parsing — KVKvde-_MYc
- 05 Recurrent Neural Networks — fyc0Jzr74y4
- 06 Sequence to Sequence Models — Ba6Fn1-Jsfw
- 07 Attention, Final Projects and LLM Intro — J7ruSOIzhrE
- 08 Self-Attention and Transformers — LWMzyfvuehA
- 09 Pretraining — DGfCRXuNA2w
- 10 Natural Language Generation — N9L32bFieEY
- 11 Post-training (Archit Sharma) — 35X6zlhoCy4
- 12 Benchmarking (Yann Dubois) — TO0CqzqiArM
- 13 Efficient Training (Shikhar Murty) — UVX7SYGCKkA
- 14 Brain-Computer Interfaces (Chaofei Fan) — tfVgHsKpRC8
- 15 Reasoning and Agents (Shikhar Murty) — I0tj4Y7xaOQ
- 16 After DPO (Nathan Lambert) — dnF463_Ar9I
- 17 ConvNets and TreeRNNs — S8d-7v3f5MQ
- 18 NLP, Linguistics, Philosophy — NxH0Y78xcF4
- 19 Multimodal Deep Learning (Douwe Kiela) — 5vfIT5LOkR0
- 20 Model Interpretability & Editing (Been Kim) — cd3pRpEtjLs
- 21 Python Tutorial (Manasi Sharma) — 8j4wpU98Q74
- 22 PyTorch Tutorial (Drew Kaul) — Uv0AIRr3ptg
- 23 Hugging Face Tutorial (Eric Frankel) — b80by3Xk_A8
