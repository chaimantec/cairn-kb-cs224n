# KB build — CS224N (Natural Language Processing with Deep Learning, Stanford, Spring 2024)

Scope for this run: lectures 1–2 only (smoke test). The course has 23 lectures;
the remaining 21 are listed under "Deferred" and are not part of this build.

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
