# KB build — CS224N (Natural Language Processing with Deep Learning, Stanford, Spring 2024)

The course has 23 lectures, built incrementally.

- **Run 1** (complete): lectures 1–2, plus the site crawl and `sources.md`.
- **Run 2** (in progress): lectures 3–4.
- Lectures 5–23 remain deferred; video ids are listed at the bottom.

---

# Run 2 — lectures 3–4

## Transcripts
- [x] 03 Backpropagation and Neural Networks — video HnliVHU2g9U (95 paragraphs)
- [x] 04 Dependency Parsing — video KVKvde-_MYc (102 paragraphs)
- [ ] Copy-edit both into readable sentences, keeping verbatim originals

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
- [ ] wiki/03-backpropagation-and-neural-networks.md
- [ ] wiki/04-dependency-parsing.md
- [ ] Topic pages: backpropagation, neural-network-basics, named-entity-recognition,
      dependency-parsing, regularization-and-training-practices
- [ ] Update existing topic pages that lectures 3–4 extend
- [ ] Update INDEX.md table of contents

## Publish
- [ ] Commit and push (kbUrl already set — no link_kb.sh needed)

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
