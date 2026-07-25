# KB build — CS224N (Natural Language Processing with Deep Learning, Stanford, Spring 2024)

The course has 23 lectures, built incrementally.

- **Run 1** (complete): lectures 1–2, plus the site crawl and `sources.md`.
- **Run 2** (complete): lectures 3–4.
- **Run 3** (in progress): lectures 5–6.
- Lectures 7–23 remain deferred; video ids are listed at the bottom.

---

# Run 3 — lectures 5–6

## Transcripts
- [x] 05 Recurrent Neural Networks — video fyc0Jzr74y4 (102 paragraphs)
- [x] 06 Sequence to Sequence Models — video Ba6Fn1-Jsfw (100 paragraphs)
- [ ] Copy-edit both into readable sentences, keeping verbatim originals

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
- [ ] wiki/05-recurrent-neural-networks.md
- [ ] wiki/06-sequence-to-sequence-models.md
- [ ] Topic pages — candidates: language-modeling, n-gram-language-models,
      recurrent-neural-networks, perplexity, vanishing-and-exploding-gradients, lstm,
      machine-translation, seq2seq-and-encoder-decoder
- [ ] Update existing topic pages that lectures 5–6 extend
- [ ] Update INDEX.md table of contents
- [ ] Update AGENTS.md — the "Slide N = PDF page N" line is stale since lecture 4
- [ ] Link sweep

## Publish
- [ ] Commit and push (kbUrl already set from run 1, so no link_kb.sh needed)

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
