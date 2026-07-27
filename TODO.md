# KB build — CS224N (Natural Language Processing with Deep Learning, Stanford, Spring 2024)

The course has 23 lectures, built incrementally.

- **Run 1** (complete): lectures 1–2, plus the site crawl and `sources.md`.
- **Run 2** (complete): lectures 3–4.
- **Run 3** (complete): lectures 5–6.
- **Run 4** (complete): no new lectures — converted all mathematics in the wiki to LaTeX.
- **Run 5** (complete): lectures 7–8.
- **Run 6** (complete): lectures 9–10.
- **Run 7** (complete): lectures 11–12 — Post-training (Archit Sharma) and
  Benchmarking (Yann Dubois).
- **Run 8** (complete): lecture 13 — Efficient Training (Shikhar Murty).
- **Run 9** (complete): lecture 14 — Brain-Computer Interfaces (Chaofei Fan).
- **Run 10** (in progress): lecture 15 — Reasoning and Agents (Shikhar Murty).
- Lectures 16–23 remain deferred; video ids are listed at the bottom.

---

# Run 10 — lecture 15

Catalog position 15. The catalog's own title is "Lecture 14 - Reasoning and Agents by Shikhar
Murty". Repo files use the catalog **position** (`15-…`), per the convention set in run 3. The
off-by-one between catalog position and the deck's own numbering, systematic since position 11,
continues here.

## Provenance — Spring 2024

Position 15 ↔ `slides-cs224n-spr2024-lecture14-agents-shikhar-updated.pdf`, already on disk from
the run-1 crawl. Confirmed by the title slide: Shikhar Murty, *Lecture 14: Reasoning and Agents*.
Same lecturer as position 13 (Efficient Training).

| Where | Reasoning and Agents |
|---|---|
| Cairn catalog position | 15 |
| Catalog / YouTube title | "Lecture 14" |
| Deck's own title | "Lecture 14: Reasoning and Agents" |
| Deck filename | `…lecture14-agents-shikhar-updated.pdf` |
| Repo file prefix | `15-` |

Slide 38 credits "some slides borrowed from Frank Xu (CMU)" for the agents half.

## Transcripts
- [ ] 15 Reasoning and Agents — video I0tj4Y7xaOQ (82 paragraphs, 64 min)

## Slides
- [x] raw/slides/15-reasoning-and-agents.md — 75 pages, printed numbers 1:1 with PDF pages,
      no gaps and no offset (only page 1, the title, prints no number).
      `slide_number_map.py --verify` passes: 75 headings, sequence matches the deck exactly.
- [x] Delegated to a Step 1b subagent (model sonnet), foreground. **The first agent was killed
      by a session limit mid-run**, having written slides 1–60 — because it was told to append
      each chunk as it went, that work survived intact and verified (60 headings, in order, no
      gaps). Its transcript could not be resumed, so a second agent read pages 61–75 and built
      the contents table. The append-as-you-go instruction is what turned a fatal interruption
      into a 15-page top-up; keep giving it.
- [x] Figure spot-check delegated to a third agent (model sonnet), so no PDF pages entered the
      main context. Audited pages 3, 14, 36, 37, 67, 72, 73. **One error found and fixed**:
      slide 36 called the counterfactual synthetic alphabet "20-letter" while quoting all 26
      letters of it. Everything else agreed — page 14's two full results tables cell by cell,
      the three-series legend on 37, the printed averages on 67, the missing GPT-3 bar in 72's
      Medium panel, and 73's "(~2mins/task)" margin note correctly described as an annotation
      rather than a second data series.
- [x] Contents table's claim that slides 3 and 38 are the deck's only two section titles
      confirmed against the pages — the one thing `--verify` cannot see.

## Wiki
- [ ] wiki/15-reasoning-and-agents.md
- [ ] Topic pages
- [ ] INDEX.md, README.md, AGENTS.md
- [ ] Link sweep

## Publish
- [ ] Commit and push (`kbUrl` already set from run 1; no `link_kb.sh` needed)

---

# Run 9 — lecture 14

Catalog position 14. The catalog's own title is "Lecture 13 - Brain-Computer Interfaces".
Repo files use the catalog **position** (`14-…`), per the convention set in run 3. The
off-by-one between catalog position and the deck's own numbering, systematic since position
11, continues here.

## Provenance — Spring 2024

Position 14 ↔ `slides-cs224n-spr2024-lecture13-speech-bci.pdf`, already on disk from the
run-1 crawl (62MB, the largest deck in the course). Confirmed by the title slide: Chaofei
Fan, Stanford NPTL / BrainGate.

| Where | Brain-Computer Interfaces |
|---|---|
| Cairn catalog position | 14 |
| Catalog / YouTube title | "Lecture 13" |
| Deck's own title | *Speech Brain-Computer Interfaces for Restoring Natural Communication* (unnumbered) |
| Deck filename | `…lecture13-speech-bci.pdf` |
| Repo file prefix | `14-` |

This is a guest research talk rather than a numbered course lecture: the deck carries no
lecture number, no agenda slide of the usual form, and no announcements.

## Transcripts
- [x] 14 Brain-Computer Interfaces — video tfVgHsKpRC8 (94 paragraphs, 72 min)
- [x] Copy-edited by hand; verbatim original kept in `raw/transcripts/original/`.
      All 94 `[MM:SS]` timestamps verified identical and in order.
- [x] Number inventory compared body-to-body. Every difference investigated and accounted
      for: "0 five seconds" → **0.5 seconds**, "assignment three" → **Assignment 3**,
      "CS2 24s"/"cs2 24n" → **CS224S**/**CS224N**, the stutters "T6 that code name T6" and
      "60 to 7 70" collapsed, and "150 and 60 words per minute" → **150 and 160**, which is
      the only one that adds a digit and is therefore flagged inline against slide 29.
- [x] Restored against the slides: *brain-computer interface* ("bre/breing/green/Brin
      computer interface"), *BCI* ("PCI", "BC"), *ALS* ("AOS", "Asos"), *motor cortex*
      ("modor/modal/model/M cortex"), *orofacial* ("artificial", "auto facial"), *phoneme*
      ("fum", "PHS", "volume", "fores", "phic"), *CTC* ("CDC"), *RNN* ("RN"), *LSTM* ("ISM"),
      *GRU* ("Gru Gator recurring unit"), *n-gram* ("angram", "engram"), *ECoG* ("EOG",
      "Eco"), *fMRI* ("Mi"), *Broca's area* ("Brokers area", "broadcast area"), *NPTL*
      ("mptl"), *Hans Berger* ("hansburg"), *Richard Caton* ("Richard Kon"), *Frank Willett*
      ("Frank wet"), *letter board* ("lead board", "ladder board"), *cosine tuning curve*
      ("coign tuning curve"), *neuroprosthesis* ("neuroth neuro prothesis"), *word error
      rate* ("world a rate"), *attempted speech* ("template speech"), *inner speech* ("UN
      speech"), *locked-in state* ("locking state"), *brainstem stroke* ("brain stamp
      stroke"), *tongue* ("towns"), *vowels* ("vows").
- [x] Names confirmed from the slides rather than from memory: **Richard Caton** (slide 8
      reproduces his 1875 *BMJ* paper), **Noland Arbaugh** (slide 6 quote), **Frank Willett**
      (slides 27, 36).
- [x] Four residual ambiguities left marked `[Ed: …]` rather than guessed: 0:51 "was shot",
      which conflicts with the stroke named in the same sentence; 33:19 "ctech" for the
      robotic-arm institution, where slide 25 names none and the lecturer hedges himself;
      and two fragments inside video-clip audio at 45:42 and 1:06:32.

## Slides
- [x] raw/slides/14-brain-computer-interfaces.md — 75 pages.
      `slide_number_map.py --verify` passes: 75 headings, sequence matches the deck exactly.
- [x] **This deck prints no slide number on any page** — the first in the course to do so.
      `slide_number_map.py` warned rather than silently claiming a clean 1:1, which is the
      failure mode that warning exists for. Slide N = page N by construction, and the file's
      front matter says so explicitly instead of implying the deck printed them.
- [x] Delegated to a Step 1b subagent (model sonnet), read in four contiguous chunks.
      Hand-checked four figure-heavy pages against the PDF: *29* (the words-per-minute
      scale — all nine labelled points and both annotated study markers correct), *17*,
      *36* and *60*. Two slips found and fixed, both small and both about series
      attribution: slide 36 confined the grey speech-volume trace to the "Words" panel when
      it also appears in "Single phonemes", and slide 60 gave the 130,000-word WPM series as
      a flat 63–67 when it dips to about 61 on day 141. No invented series, no invented
      cells.
- [x] Contents table checked by hand against the deck's actual section-title slides (7, 34,
      62) — the one thing `--verify` cannot see.

## Wiki
- [x] wiki/14-brain-computer-interfaces.md
- [x] Topic pages (6 new): `brain-computer-interfaces`, `neural-recording-technologies`,
      `neural-population-decoding`, `connectionist-temporal-classification`,
      `language-models-in-decoding`, `neuroethics`.
- [x] Back-links added from the existing pages this lecture reuses: `lstm` (the GRU, and the
      small-net-over-Transformer argument), `seq2seq-and-encoder-decoder` (why arbitrary
      alignment is a cost), `decoding-algorithms` (LM fusion, and CTC's merging-prefix
      wrinkle), `n-gram-language-models` (why a counting model is still right inside 20 ms).
- [x] INDEX.md table of contents — lecture entry, six topic entries, coverage note updated to
      1–14, the no-printed-numbers caveat added to the `raw/slides/` description, and lecture
      14's caption garbles added to the transcript description.
- [x] AGENTS.md — recorded lecture 14 as a **fourth kind** of slide-numbering exception (a deck
      that prints no numbers at all, distinct from the hidden-slide offsets in lectures 4, 10
      and 11) and as a guest talk whose deck carries no lecture number.
- [x] All 1,122 relative links in the repo re-checked programmatically; every one resolves.
      (Three apparent hits are illustrative examples inside code spans in AGENTS.md/TODO.md,
      the same known false positives as previous runs.)

## Publish
- [x] Commit and push — https://github.com/chaimantec/cairn-kb-cs224n
- [x] `kbUrl` already set on the catalog entry from run 1; no `link_kb.sh` needed.

---

# Run 8 — lecture 13

Catalog position 13. The catalog's own title is "Lecture 12 - Efficient Training, Shikhar
Murty". Repo files use the catalog **position** (`13-…`), per the convention set in run 3.

## Provenance — Spring 2024

Position 13 ↔ `slides-cs224n-spr2024-lecture12-training-shikhar.pdf`, already on disk from
the run-1 crawl. The lecturer opens with "welcome to lecture 12", matching the deck.

| Where | Efficient Training |
|---|---|
| Cairn catalog position | 13 |
| Catalog / YouTube title | "Lecture 12" |
| Deck's own title | "Lecture 12" |
| Repo file prefix | `13-` |

## Transcripts
- [x] 13 Efficient Training — video UVX7SYGCKkA (81 paragraphs, 62 min)
- [x] Copy-edited by hand; verbatim original kept in `raw/transcripts/original/`.
      All 81 `[MM:SS]` timestamps verified identical and in order; number inventory compared
      body-to-body, and only four tokens differ, each a caption artifact: "fp32 that's like
      **32 bytes**" (the lecturer's own false start, self-corrected to 32 bits in the same
      breath), "a **th000**" ×2 (spelled out as "a thousand"), "the **fp6** computed
      Radiance" (garble of FP16), and "stage **one 1** two and three" (duplicated numeral).
- [x] Restored against the slides: *Adam* ("adom"), *bfloat16* ("B float 16", "B FL 16",
      "b flat 16"), *FSDP* ("Fs DP"), *shard/sharded/sharding* ("shot", "shoted", "shouted",
      "Shing"), *LoRA* ("Laura", "Lura"), *batch size* ("bat size", "bath size", "bad size"),
      *DistilBERT* ("dist bir"), *GradScaler* ("grad scalar"), *autocast* ("AutoCast"),
      *reduce-scatter* ("reduce CER"), *layer-j* ("ler"), *gradients* ("Radiance"), *PyTorch*
      ("pyos"), *GPT-3* ("gpt3", "GPD 3"), *BitFit* ("bit fit"), *SVD* ("SD"), *NaNs* ("n"),
      *A100* ("a 100", "A1 100"), *Ampere* ("Amper"), *autodiff* ("Auto diff"), *efficient
      adaptation* ("f adaptation"), *cache/caching* ("cash").
- [x] One number corrected against the slides and flagged inline: 4:47 "1.1 gets converted to
      one" → **1.0001** (slide 10 writes the example that way, and 1.1 *is* representable in
      FP16, so the caption is wrong rather than the lecturer).
- [x] Two spoken figures that the slides do not carry are marked inline rather than quietly
      reconciled: 41:12's "1.1 million tons" of carbon for GPT-3 (slide 54 gives no tonnage —
      only Cornell's 2021 "coal power plant for 10 straight hours", which he then states
      correctly and which he himself hedges as "or some such number"), and 29:37's
      FlatParameter example of "14 parameters plus padding" (slide 43 draws 15 elements plus
      one padding slot across 16 ranks).
- [x] Five residual ambiguities left marked `[Ed: …]` rather than guessed — all student
      questions or interjections off-mic (11:49, 12:36, 21:06, 21:56, 54:25), plus the LoRA
      weight matrix at 45:51 whose subscript the captions destroy ("W KN"); slide 59 records
      it as $W_0 \in \mathbb{R}^{d \times k}$.
- [x] Useful cross-check: the student at 54:25 asks about "slide 48", and printed slide 48 is
      indeed the FSDP communication timeline with the full overhead recap — independent
      confirmation of the deck's 1:1 numbering.

## Slides
- [x] raw/slides/13-efficient-training.md — 65 pages, printed numbers 1:1 with PDF pages
      (pages 1, 53, 54, 55, 62 print no number but sit in an unambiguous run).
      `slide_number_map.py --verify` passes: 65 headings, sequence matches the deck exactly.
- [x] **First run of the delegated slide-reading path** (Step 1b subagent, model sonnet) —
      it works. Hand-checked five pages against the PDF rather than trusting the script:
      *22* (DistilBERT precision table — all 15 cells correct), *64* (both LoRA ablation
      tables — all 41 cells correct), *53/54* (the two charts — both line series correct
      with no invented third, all twelve bar values correct), *46–48* (the FSDP
      communication timeline — CPU/comp/comm tracks correct block by block). The known
      failure modes did not appear: no fabricated data series, no thin "a chart showing
      results" prose. It also flagged three genuine uncertainties unprompted rather than
      guessing, including the slide's own garbled "It's quite that simple…" line, which is
      really on slide 54 and was left verbatim.
      *Caveat:* the first attempt died on an infrastructure fault (background fork could not
      be resumed) having written nothing. Re-run in the foreground. Worth telling the agent
      to append each chunk to the file as it goes, so a crash leaves partial work.

## Wiki
- [x] wiki/13-efficient-training.md
- [x] Topic pages (7 new): mixed-precision-training, gpu-memory-for-training,
      collective-communication, distributed-data-parallel, zero-and-fsdp,
      parameter-efficient-finetuning, lora. Topic pages now total 51.
- [x] Cross-links added to the pages this lecture extends: pretraining-and-finetuning →
      parameter-efficient-finetuning + lora (with a note that the two decks **swap the letters
      $A$ and $B$** and name the rank differently — lecture 9 writes $W + AB$ with rank $k$,
      lecture 13 writes $W_0 + \alpha BA$ with rank $r$); final-project-guidance → the whole
      flowchart, since this lecture is explicitly aimed at final projects; gradient-descent →
      gpu-memory-for-training, because Adam's per-parameter state is what costs 12 bytes.
- [x] Update INDEX.md — coverage note now says lectures 1–13, the numbering caveat covers 13,
      one lecture entry, seven topic entries, and the slides/transcripts sections list the new
      deck and transcript.
- [x] README.md — lecture table row, 44 → 51 topic pages, "lectures 14–23 not built yet",
      guest-lecturer note extended to Shikhar Murty.
- [x] AGENTS.md — lecture 13 added to the 1:1-numbering list and to the Spring 2024
      provenance list; the multiple-numbering note now records that the off-by-one between
      catalog position and video/deck title is **systematic from position 11 onward**.
- [x] Link sweep: every relative link in the repo resolves; no wiki page points into
      gitignored `raw/pdfs/`; the deck URL matches `sources.md` exactly. The only three
      unresolvable strings are the illustrative examples inside backticks in AGENTS.md and
      TODO.md's own quotation of them, as in previous runs.
- [x] LaTeX check: no equation inside a code fence, no unbalanced `$` on any line. One inline
      expression that had wrapped across a line break was promoted to a display equation,
      since inline math split by a newline does not reliably render.

## Publish
- [x] Commit and push — https://github.com/chaimantec/cairn-kb-cs224n (3 commits: slides,
      transcripts, wiki). `kbUrl` was already set on catalog course
      b102c48c-0c1b-4fc5-be6d-4c6d1e7211d1, so no `link_kb.sh` was needed.

## Notes for the next run

- **The delegated slide-reading path (Step 1b) is now proven** — see the Slides section above
  for what was hand-checked. Two operational lessons: pass `model: sonnet` explicitly, and run
  the agent in the **foreground**. The first background attempt died on a checkpoint/resume
  fault having written nothing, and a background agent that dies takes its whole context with
  it. Also tell the agent to append each chunk of pages to the output file as it goes, so an
  interruption leaves partial work rather than none.
- **The numbering off-by-one is systematic from position 11 onward.** Positions 11, 12 and 13
  are titled "Lecture 10", "Lecture 11" and "Lecture 12" by both the video and the deck. Expect
  position 14 to be "Lecture 13" (Brain-Computer Interfaces, Chaofei Fan) and to match
  `slides-cs224n-spr2024-lecture13-speech-bci.pdf`, but verify rather than assume.
- Lectures 14–23 remain. Spring 2024 decks for 14–18 are already on disk in `raw/pdfs/` from the
  run-1 crawl, so no re-crawl is needed. **Lectures 19–23 have no deck on the Spring 2024 site**
  (multimodal, interpretability, and the three tutorials) and will need either a
  Winter-2023-style hunt or a transcript-only build.
- `slides-cs224n-spr2024-lecture13-speech-bci.pdf` — the deck for the *next* lecture — is
  **62MB**, over GitHub's 50MB warning threshold. It is gitignored like the rest, so this only
  matters if anyone ever reverses the no-PDF-commits decision.
- Lecture 13's transcript was unusually clean: 81 paragraphs for a 62-minute lecture, and only
  five residual `[Ed: …]` marks, all of them off-mic student questions. Its garbles are
  systems vocabulary rather than proper nouns, which the slides resolve easily.

---

# Run 7 — lectures 11–12

Catalog positions 11 and 12. The catalog's own titles are "Lecture 10 - Post-training by
Archit Sharma" and "Lecture 11 - Benchmarking by Yann Dubois". Repo files use the catalog
**position** (`11-…`, `12-…`), per the convention set in run 3.

## Provenance — both are Spring 2024, unlike run 6

Lectures 9–10 were Winter 2023 recordings. These two are back on the Spring 2024 track and
match the Spring 2024 decks crawled in run 1:

- Position 11 ↔ `cs224n-spr2024-lecture10-prompting-rlhf.pdf`, whose title slide reads
  "Archit Sharma / Lecture 10: Prompting, Instruction Finetuning, and DPO/RLHF (Based on
  slides from Jesse Mu)".
- Position 12 ↔ `cs224n-spr2024-lecture11-evaluation-yann.pdf`, whose title slide reads
  "Yann Dubois / Lecture 11: Benchmarking and Evaluation".

## Lecture numbering

| Where | Post-training | Benchmarking |
|---|---|---|
| Cairn catalog position | 11 | 12 |
| Catalog / YouTube title | "Lecture 10" | "Lecture 11" |
| Deck's own title slide | "Lecture 10" | "Lecture 11" |
| Repo file prefix | `11-` | `12-` |

## Transcripts
- [x] 11 Post-training — video 35X6zlhoCy4 (104 paragraphs)
- [x] 12 Benchmarking — video TO0CqzqiArM (110 paragraphs)
- [x] Copy-edited both by hand, verbatim originals kept in `raw/transcripts/original/`.
      All 104+110 `[MM:SS]` timestamps verified identical and in order; number inventories
      compared paragraph by paragraph and every difference explained in the file headers.
- [x] Restored against the slides: L11 — *ChatGPT* ("CH GPD", "chargy GPD", "chbt"),
      *GPT-2/3/4* ("gpd2", "gp22", "gbd3", "gbd4"), *InstructGPT* ("instruct gbt"), *RLHF*
      ("rlf", "rhf", "rft"), *DPO*/*PPO* ("DP", "BP", "po"), *Bradley–Terry* ("Brad lary",
      "broadly ter"), *Kullback–Leibler* ("cbak LI Li"), *Boltzmann* ("boltzman"), *MMLU*
      ("mlu"), *Flan-T5* ("FL T5"), *PaLM* ("power models"), *LIMA* / *Less Is More for
      Alignment* ("LMA", "paperless as more for alignment"), *Mistral* ("mistol"), *TL;DR*
      ("TLD drr"). L12 — *SuperGLUE* ("super clue"), *BoolQ* ("bull Q"), *BLEU*/*ROUGE*
      ("blur", "rou", "RK"), *BERTScore*/*BLEURT* ("bir", "blurt"), *AlpacaEval*/*AlpacaFarm*
      ("Paka eval", "back a EV", "alpaca form"), *MT-Bench* ("empty bench"), *Chatbot Arena*
      ("chadbad Arena"), *Penn Treebank* ("pentry bank"), *CoNLL* ("caral"), *scikit-learn*
      ("es learn"), *Codeforces* ("code Force"), *Phi-1.5* ("F 1.5"), *DynaBench* ("Dynam
      Ben"), *MLPerf* ("ml puff"), *DiscrimEval* ("dis remal"), *Anthropic* ("entropic"),
      *Horace He* ("harass he"), *Zygote* ("Z zigot"), and Chris Potts ("Chris Peach").
- [x] Numbers corrected against the slides rather than kept as heard, each flagged in the
      header: L11 22:20 "60-year-old"→six-year-old (slide 36), 19:17 "5 40 billion"→540
      billion, 11:35 "gp22"→GPT-2, 1:12:32 "th000 tasks"→a thousand tasks, 1:10:11 "199"→the
      student's "plus one versus plus 99". L12 24:43 "7%"→67% for "Heck no!" (slide 18 scores
      it 0.67, the same as "Yes!", which is the point of the false-positive example) and
      1:06:13 "63 3.7"→63.7 (slide 51 gives 0.637).
- [x] Eleven residual ambiguities left marked inline as `[Ed: …]` rather than guessed — L11
      has five (four garbled student questions plus the benchmark a student names at 27:00,
      almost certainly ImageNet), L12 has six (five garbled student questions or
      interjections, and the phrase the captions render as "selfie station" at 1:16:58 inside
      the lecturer's account of the OpinionQA result).

## Slides
- [x] raw/slides/11-post-training.md — all 94 pages transcribed (printed 1–99;
      printed 58, 59, 60, 64 and 84 are hidden slides absent from the PDF)
- [x] raw/slides/12-benchmarking.md — all 65 pages transcribed (printed numbers 1:1)

## Wiki
- [x] wiki/11-post-training.md
- [x] wiki/12-benchmarking.md
- [x] Topic pages (9 new): prompting, chain-of-thought, instruction-finetuning,
      reward-modeling, rlhf, direct-preference-optimization, evaluating-llms,
      llm-as-a-judge, benchmark-contamination
- [x] Cross-links added to the pages these supersede or extend:
      gpt-and-in-context-learning → prompting + chain-of-thought;
      pretraining-and-finetuning → instruction-finetuning;
      evaluating-nlg → evaluating-llms + llm-as-a-judge + benchmark-contamination
- [x] Update INDEX.md table of contents — coverage note now says lectures 1–12, the
      numbering caveat covers 11/12, and the raw-materials sections list both new decks
      and both new transcripts
- [x] README.md brought up to date — it had gone stale at run 5 and still claimed
      "Lectures 1–6, in full" and "Lectures 7–23 are not built yet". Now lists all 12
      lectures with slide and paragraph counts, and 44 topic pages.
- [x] AGENTS.md updated — lecture 11 added to the list of decks whose printed-number
      offset grows, and the provenance and multiple-numbering notes extended to 11–12.
- [x] Verified every relative link in the repo resolves (only two known non-links remain,
      the illustrative `[gradient descent](gradient-descent.md)` examples inside AGENTS.md)

## Publish
- [x] Commit and push — https://github.com/chaimantec/cairn-kb-cs224n (3 commits: slides,
      transcripts, wiki). `kbUrl` was already set on catalog course
      b102c48c-0c1b-4fc5-be6d-4c6d1e7211d1, so no `link_kb.sh` was needed.

## Notes for the next run

- Lectures 13–23 remain. The Spring 2024 decks for 13–18 are already on disk in
  `raw/pdfs/` from the run-1 crawl, so no re-crawl is needed; lectures 19–23
  (multimodal, interpretability, and the three tutorials) have **no deck on the
  Spring 2024 site** and will need either a Winter-2023-style hunt or a
  transcript-only build.
- `slides-cs224n-spr2024-lecture13-speech-bci.pdf` is **62MB** — over GitHub's 50MB
  warning threshold. It is gitignored like the rest, so this only matters if anyone
  ever reverses the no-PDF-commits decision.
- **Step 1b changed after this run: slide decks are now transcribed by parallel subagents,
  not in the main session.** The path has never executed — lectures 11-12 were done by hand.
  Treat run 8 as its trial: run `slide_number_map.py <pdf> --verify <slides.md>` on every
  returned file (it catches skipped pages, merged slides and wrong numbering), and then open
  two or three figure-heavy pages against the PDF yourself. **Pass `model: sonnet`
  explicitly** — with no model argument a subagent inherits Opus. Tested on 13 hard pages:
  both Haiku and Sonnet got 100% of table cells, but on charts Haiku recovered 79% of values
  and invented a data series out of two marginal labels, while Sonnet recovered 91% and got
  the structure right. Fabricated series and thin figure prose pass every automated check,
  and are the failure mode delegation is most prone to. Also: `Read` silently drops disjoint
  `pages` lists (`"2,4"` returns one page) — request contiguous ranges and check the count.
- Two lecture-11 and lecture-12 residuals worth revisiting if better captions appear:
  the benchmark a student names at L11 27:00 (almost certainly ImageNet) and the
  phrase the captions render as "selfie station" at L12 1:16:58.

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
- [x] 09 Pretraining — video DGfCRXuNA2w (102 paragraphs)
- [x] 10 Natural Language Generation — video N9L32bFieEY (102 paragraphs)
- [x] Copy-edited both by hand, verbatim originals kept in `raw/transcripts/original/`.
      All 102+102 `[MM:SS]` timestamps verified identical and in order; number inventories
      compared and every difference explained (ASR-spaced "17 000"→17,000, "40 000"→40,000,
      "7 000"→7,000; ".03"→0.03; spelled-out numbers rendered as digits — the Fibonacci
      sequence on slide 17, "assignment five", "homework one"; dictated notation written in
      symbols — `w_{1:t−1}`, `y_{t+1}`; garbles that happened to contain digits removed —
      "12K decoding"→top-k, "chaiji 50"→ChatGPT; one false start dropped, L10 at 37:50,
      where "you will get 10 — you will very likely get 10 different outputs" repeats the
      same number).
- [x] Restored against the slides: L9 — *word2vec* ("word to VEC", "where to back"), *UNK*
      ("ankh", "UNC"), *BERT* ("Birch", "burp model"), *RoBERTa* ("Brita"), *Iroh* ("Ira",
      "IRL"), *Adam*/*AdaGrad* ("atom", "add a grad"), *salient span masking* ("salience
      band masking"), GPT input format `[START]`/`[DELIM]`/`[EXTRACT]` from slide 44. L10 —
      *NLG* itself ("an LG", "analogy", "energy"), *autoregressive* ("other aggressive"),
      *n-gram* ("unground", "engram"), *softmax* ("solve Max"), *nucleus sampling* ("nuclear
      sampling"), *BLEU*/*ROUGE*/*BERTScore*/*BLEURT*/*MAUVE* ("Google score", "root score",
      "Bird score", "Port", "mouth score"), *DAgger*, *RLHF* ("rlhs"), *Dirac delta*
      ("direct Delta"), the slide-53 exchange ("hack yes"→"Heck yes!"), and slide 56's Word
      Mover's Distance example ("the president Grace the press"→"The President greets the
      press in Chicago").
- [x] Ten residual ambiguities left marked inline as `[Ed: …]` rather than guessed — L9 has
      four (two garbled student questions, one unrecoverable word inside a third, and the
      BERT corpus sizes at 52:30, where the lecturer corrects himself and the captions
      mangle the figures he lands on; slide 27 named as the figure of record). L10 has six
      (four garbled student questions, one unrecoverable example word at 27:01, and one
      place at 27:46 where the captions say "quickly" where slide 30 says "slowly" — keeping
      the caption would have made the sentence contradict itself).
- [x] One reading resolved by date rather than by slide: L9's captions give "people using
      GPT uh three four simple programming things" (1:14:49). The lecture is February 2023
      and GPT-4 shipped the following month, so this is "GPT-3 **for** simple programming
      things", not a GPT-4 reference. Noted in the transcript header.

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
- [x] wiki/09-pretraining.md
- [x] wiki/10-natural-language-generation.md
- [x] Topic pages — eight new. From lecture 9: subword-modeling,
      pretraining-and-finetuning, bert, gpt-and-in-context-learning. From lecture 10:
      natural-language-generation, decoding-algorithms, exposure-bias-and-teacher-forcing,
      evaluating-nlg. The split for lecture 9 puts the *paradigm* (why pretraining works,
      the three architecture classes, span corruption/T5, parameter-efficient fine-tuning)
      on one page and the two model families that get asked about by name on their own,
      rather than one enormous "pretraining" page.
- [x] Updated existing topic pages these lectures extend — word2vec and
      word-senses-and-polysemy (the Firth 1935 quote and the *record*/*record* example, i.e.
      the limitation that motivates contextual models), distributional-semantics,
      language-modeling (LM reused as a pretraining objective), perplexity (its two later
      uses, and the trap that repetitive text scores well), softmax-and-cross-entropy
      (temperature), transformer and seq2seq-and-encoder-decoder (which objective each shape
      can be pretrained with; when a bare decoder beats an encoder-decoder),
      evaluating-machine-translation (where BLEU stops working, and why it makes a bad RL
      reward), final-project-guidance (read your own generations; adapt cheaply).
- [x] Updated INDEX.md — coverage note now says lectures 1–10 and carries the Winter 2023
      provenance caveat and lecture 10's four numbers; 2 lecture entries; 8 topic entries;
      raw-materials section updated with both slide counts, lecture 10's accumulating
      page/slide offset, the ethics-slide note, and the new garbled-term examples.
- [x] Updated AGENTS.md — the slide-numbering convention now covers lecture 10's
      *accumulating* offset (previous decks were either 1:1 or constant-offset), plus two new
      conventions: verify a deck actually matches its video before transcribing, and name
      files by catalog position when a lecture carries several conflicting numbers.
- [x] Updated sources.md — new "Winter 2023 decks" section with both URLs, and a note that
      the Spring 2024 `lecture09-pretraining-updated.pdf` is a *different* 64-page deck whose
      numbering does not match this video.
- [x] Link sweep (scripted): 80 markdown files, every relative link and in-page anchor
      resolves, no wiki page points into gitignored `raw/pdfs/`, and all 45 wiki pages are
      reachable from INDEX.md. Two apparent failures in AGENTS.md are illustrative examples
      inside backticks, not links.
- [x] KaTeX validation: 1,101 math spans, 0 render failures, 0 issues in strict mode (which
      is what GitHub uses). Delimiter sweep clean — no `$$` mid-line, no display math inside
      a blockquote, no LaTeX left in a code fence or indented block. The one unescaped `$` in
      `raw/slides/04-dependency-parsing.md` ("for $27 a share") sits inside a 4-space indented
      block, so it renders literally and is safe; it is pre-existing.

## Publish
- [x] Commit and push (kbUrl already set on the catalog entry, no re-link needed)

## Notes for run 7

- **Check the deck against the video before transcribing anything.** Run 5 recorded that
  deck-vs-catalog title matching "held for lectures 7–8"; it broke immediately at lecture 9,
  and not in the way the earlier note anticipated. The problem was not a title mismatch but a
  *different course offering* — lectures 9 and 10 are Winter 2023 recordings spliced into a
  Spring 2024 playlist. The tells that settled it, in order of usefulness: the Spring 2024
  schedule has no NLG lecture at all; lecture 9's lecturer refers to "Thursday's lecture on
  natural language generation," which only exists in Winter 2023; both lecturers refer to
  Assignment 5, which Spring 2024 did not have; and each deck's announcement slide matches the
  spoken announcements item for item. Do this check *first* on every remaining lecture — the
  guest lectures (catalog 11–16, 19–20) are the likeliest to diverge.
- **Do not assume a page/slide offset is constant.** Run 2 found lecture 4 had four hidden
  slides and a fixed offset. Lecture 10 has five hidden slides scattered through the deck, so
  the offset grows from 0 to +5 as you go. Extract the printed footer number from every page
  with `pdftotext -f N -l N` and build the mapping table before transcribing; do not
  extrapolate from the first gap you find. Watch for pages that print no number at all (the
  title page and full-bleed figures), which look like gaps but are not.
- **A lecture can carry four different numbers.** Catalog position 10, video title "Lecture
  11", deck title slide "Lecture 12", filename `lecture10-nlg`. Files follow the catalog
  position; state every variant in the slide file so a learner quoting any of them can be
  matched.
- **Some slides should not be transcribed verbatim into a public repo.** Lecture 10's slides
  70, 73 and 74 reproduce a profanity-laced jailbreak, racist and antisemitic hate speech, and
  sexually violent generations, all in full, as evidence about model behaviour. The slide file
  states precisely what each figure shows, what the finding is, and which paper it comes from,
  without reproducing the passages — the pedagogical content is the finding, not the invective.
  Slide 72's stereotype table *is* transcribed, because there the stereotypes are the data.
- **Date can settle a caption ambiguity that the slides can't.** Lecture 9's captions read
  "using GPT uh three four simple programming things"; GPT-4 shipped a month after this
  lecture, so it is "GPT-3 **for**". Check release dates before writing a model name into a
  transcript.
- Lectures 11–23 remain deferred; see the video-id list at the bottom. Note that the deferred
  list's own titles use catalog positions, which for several later entries also disagree with
  the video titles.

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
- 14 Brain-Computer Interfaces (Chaofei Fan) — tfVgHsKpRC8  ✔ done in run 9
- 15 Reasoning and Agents (Shikhar Murty) — I0tj4Y7xaOQ
- 16 After DPO (Nathan Lambert) — dnF463_Ar9I
- 17 ConvNets and TreeRNNs — S8d-7v3f5MQ
- 18 NLP, Linguistics, Philosophy — NxH0Y78xcF4
- 19 Multimodal Deep Learning (Douwe Kiela) — 5vfIT5LOkR0
- 20 Model Interpretability & Editing (Been Kim) — cd3pRpEtjLs
- 21 Python Tutorial (Manasi Sharma) — 8j4wpU98Q74
- 22 PyTorch Tutorial (Drew Kaul) — Uv0AIRr3ptg
- 23 Hugging Face Tutorial (Eric Frankel) — b80by3Xk_A8
