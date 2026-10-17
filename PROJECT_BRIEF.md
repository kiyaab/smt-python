# ANTIGRAVITY MASTER PROMPT: Real-Time Sign Motion Translator & Multilingual Converter

## 0. ROLE AND WORKING MODE

You are a senior software engineer with deep experience in computer vision, sequence modeling, NLP, and accessibility-focused desktop applications. You are building a complete, working, demo-ready project from scratch in this workspace.

Working rules:
1. **Plan first.** Before writing code, produce an Implementation Plan artifact (architecture, folder tree, phase breakdown, risks). Wait for my approval, then execute phase by phase.
2. **One phase at a time.** After each phase, produce a Walkthrough artifact: what was built, how to run it, what tests pass, known gaps. Do not start the next phase until the current phase's acceptance criteria are met.
3. **Verify, don't assume.** Run the code and the tests yourself in the terminal. Report real outputs. Never claim something works without executing it.
4. **Be honest about limits.** If a requirement cannot be met as stated (for example, no public dataset exists for a sign language), say so, propose the closest workable alternative, and continue. Never fake accuracy numbers or fabricate datasets.
5. **Ask when blocked** on a decision only I can make (see Section 12). Otherwise choose sensible defaults and document them.
6. Keep code typed, documented, modular, and PEP 8 compliant. Prefer small, testable functions.

---

## 1. PROJECT SUMMARY

**Title:** Hand Motion-to-Sentence & Multilingual Translator for Inclusive Communication  
**Tagline:** "Disability is Not Inability."  
**Goal:** A desktop application that uses a standard PC webcam (no special hardware) to track a signer's hand motion in real time, recognize sign gestures, assemble them into grammatical sentences, translate the sentence into six languages, display it on screen, and speak it aloud with text-to-speech.  

**Submission context:** STEM Center Innovation Grant, 8-week project, deliverables are a working prototype, a final demonstration, and a final report. Budget is $0, so use only free and open-source tools.  

**Target languages (6):** English, Amharic, Spanish, French, Arabic, Mandarin Chinese. Make the language list configurable in one config file.  

**Pipeline:**  
`Webcam frame -> Hand/pose landmarks -> Temporal gesture classifier -> Gloss sequence -> Sentence generator -> Translator (6 languages) -> On-screen text + Text-to-Speech`

---

## 2. HARD CONSTRAINTS

- Runs on a standard consumer PC: Intel Core i3 / Ryzen 3 class CPU, **4 GB RAM minimum**, 720p webcam, **no GPU required** at inference time. Windows 10/11, macOS, and Ubuntu 20.04+.
- Python 3.10+. All core dependencies free and open source.
- Real-time target: end-to-end sign-to-text latency under ~500 ms after a gesture completes; UI at 20+ FPS on CPU.
- Offline-capable core (tracking, recognition, sentence building, UI). Translation and TTS may use an online fallback, but the app must degrade gracefully with a clear message when offline.
- Privacy: video is processed locally and never saved or uploaded unless the user explicitly records training data. Show a visible camera-active indicator.

---

## 3. TECH STACK (use these unless you justify a change in the plan)

| Layer | Choice |
|---|---|
| Language | Python 3.10+ |
| Capture and CV | OpenCV, MediaPipe (use the current **MediaPipe Tasks** HandLandmarker / PoseLandmarker / FaceLandmarker APIs, not the deprecated legacy `solutions` API, unless the plan documents why) |
| ML | PyTorch (preferred) or TensorFlow/Keras, NumPy, scikit-learn |
| Sequence model | Start with a small GRU/LSTM; evaluate a light 1D-CNN/Transformer alternative; export the best to ONNX for fast CPU inference |
| Translation | Primary: an offline model, preferably `facebook/nllb-200-distilled-600M` via HuggingFace Transformers (covers all six languages, including Amharic `amh_Ethi`), with a lighter Helsinki-NLP/opus-mt option for low-RAM machines. Fallback: `deep-translator` (online). Make backend selectable in config |
| TTS | `gTTS` (online, supports Amharic) and `pyttsx3` (offline; check which languages the installed OS voices cover and warn the user when a language has no offline voice) |
| UI | PyQt6 (preferred) with a clean, high-contrast, accessible design |
| Testing | pytest, pytest-qt where useful |
| Packaging | `pyproject.toml`, `requirements.txt`, PyInstaller spec for a one-click executable |

---

## 4. ARCHITECTURE AND REPOSITORY LAYOUT

Create a modular codebase like this (adjust only with justification):

```
sign-translator/
├── README.md
├── pyproject.toml
├── requirements.txt
├── config/
│   ├── default.yaml            # languages, thresholds, model paths, backends
│   └── vocabulary.yaml         # sign glosses, display names, category tags
├── data/
│   ├── raw/                    # recorded landmark sequences (not video by default)
│   ├── processed/
│   └── splits/
├── models/                     # trained weights + ONNX exports + label maps
├── src/signtranslator/
│   ├── __init__.py
│   ├── capture/                # webcam thread, frame queue, FPS control
│   ├── tracking/               # MediaPipe wrappers, landmark normalization
│   ├── recognition/            # dataset, model, trainer, realtime inference, segmenter
│   ├── language/               # gloss->sentence engine, grammar templates
│   ├── translation/            # backend interface, NLLB, opus-mt, online fallback
│   ├── speech/                 # TTS backends, language-to-voice mapping
│   ├── ui/                     # PyQt6 main window, widgets, themes
│   ├── tools/                  # data recorder, evaluator, vocabulary editor
│   └── app.py                  # entry point wiring everything together
├── scripts/                    # train.py, evaluate.py, export_onnx.py, benchmark.py
├── tests/
└── docs/                       # architecture.md, user_guide.md, evaluation_report.md
```

Design rules:
- Each stage sits behind a small interface (for example `Translator.translate(text, src, tgt)`), so backends can be swapped without touching the UI.
- Capture, inference, and UI run on separate threads/queues so the UI never blocks.
- All thresholds, paths, and language codes come from config, never hard-coded.

---

## 5. PHASE-BY-PHASE BUILD PLAN (maps to the 8-week proposal timeline)

### Phase 1: Foundation, data tooling, landmark tracking (Weeks 1-2)
1. Scaffold the repo, virtual environment setup, linting, and a working `README`.
2. Build `capture/` and `tracking/`: webcam stream, MediaPipe hand (21 joints, x/y/z per hand, both hands) plus upper-body pose and key face landmarks (eyebrows, mouth) for non-manual markers.
3. Landmark preprocessing: wrist-relative translation, scale normalization by palm/shoulder width, left/right handling, missing-hand padding, and mirrored-hand augmentation.
4. Build the **data recorder tool** (`tools/recorder`): a small PyQt window where a contributor picks a gloss, sees a countdown, records N repetitions, and the tool saves **landmark sequences** (not raw video) with metadata (gloss, signer ID, timestamp, lighting note). Include review/delete of bad takes.
5. Define a starter vocabulary in `vocabulary.yaml` of **30 to 50 high-value conversational glosses** (greetings, yes/no, please/thank you, help, water, food, toilet, school, work, name, I/you, want, need, go, where, what, when, family terms, numbers 1-10, emergency words, and the like), grouped by category. **Make the sign language configurable** (ASL, Ethiopian Sign Language, or other) because sign vocabularies differ by language; document that the recorded data defines what the model recognizes.
6. Add a **synthetic/bootstrap dataset generator** clearly labeled as synthetic (landmark jitter, speed, and rotation augmentation of recorded samples) so the pipeline can be tested before real data exists. Never present synthetic-data accuracy as real-world accuracy.

**Acceptance:** live overlay of landmarks at 20+ FPS on CPU; recorder produces valid, versioned `.npz` sequences; unit tests for normalization pass.

### Phase 2: Gesture recognition model (Weeks 3-4)
1. Build `recognition/`: dataset loader, stratified train/val/test splits **split by signer** (to avoid leakage), augmentation, class-balance handling.
2. Train and compare at least two sequence models (GRU/LSTM baseline vs. a light alternative). Log accuracy, per-class F1, confusion matrix, model size, and CPU inference time.
3. **Continuous recognition**, not just isolated clips: implement a sliding-window segmenter with a motion-energy/hand-presence gate, a "no sign / idle" class, confidence thresholding, temporal smoothing, and debouncing so one sign produces one token.
4. Export the best model to ONNX; run inference with ONNX Runtime on CPU.
5. Produce `docs/evaluation_report.md` with real measured numbers, the lighting and distance conditions tested, and failure analysis.

**Acceptance:** `scripts/evaluate.py` reproduces the reported metrics; live demo emits a stream of gloss tokens with confidence; inference latency documented.

### Phase 3: Sentence construction and multilingual translation (Weeks 5-6)
1. Build `language/`: a **gloss-sequence-to-sentence engine**. Sign languages have their own grammar, so do not translate word for word. Implement:
   - A rule/template engine (reordering, auto-inserting articles, copulas, tense markers, pronoun handling, question detection from a question-sign or non-manual marker such as raised eyebrows) producing grammatical English as the **pivot language**.
   - A clean extension point to later plug in a small local language model for fluency polishing (optional, off by default).
   - A sentence-boundary policy: finalize on a pause longer than X ms or a "done/period" sign, plus a manual "Speak now" button.
2. Build `translation/`: translate the pivot English sentence into the six languages through the selectable backend (NLLB offline, opus-mt, online fallback). Include language-code mapping (`eng_Latn`, `amh_Ethi`, `spa_Latn`, `fra_Latn`, `arb_Arab`, `zho_Hans`), caching of repeated translations, and a **memory-aware loader** that picks a smaller model or the online fallback if RAM is low.
3. Handle right-to-left rendering for Arabic and correct font support for Amharic (Ethiopic script) and Chinese in the UI.
4. Test translation quality with a fixed test set of phrases covering every vocabulary category; record results, and flag low-confidence translations in the UI rather than hiding them.

**Acceptance:** a test suite feeds gloss sequences and verifies the English output and all six translations are produced, non-empty, and in the correct script.

### Phase 4: UI, speech output, integration (Week 7)
1. Build the PyQt6 main window:
   - Live webcam preview with landmark overlay toggle and a clear "camera on" indicator.
   - Recognized-sign ticker (current gloss + confidence), running sentence area, and a **6-language output panel** (tabs or cards) with a one-click language switcher.
   - Large-text mode, high-contrast theme, adjustable font size, full keyboard navigation, and screen-reader-friendly labels.
   - Buttons: Start/Stop, Clear, Speak, Copy text, Save transcript.
   - A conversation mode with a two-way text area so a hearing person can type a reply that is displayed large for the deaf user.
2. Build `speech/`: TTS for the selected language with graceful fallback and a clear message when a voice is unavailable; non-blocking playback with a stop button.
3. Wire the full pipeline end to end with threading, back-pressure handling, and error recovery (camera lost, model missing, network down).
4. Settings dialog (camera index, language set, backends, thresholds) persisted to config.

**Acceptance:** a complete flow works live: sign -> gloss ticker -> English sentence -> six translations -> spoken audio, with no UI freezes during a 10-minute run.

### Phase 5: Testing, evaluation, packaging, report (Week 8)
1. Full test run (unit + integration), plus a soak test and a memory/CPU benchmark on a 4 GB-class configuration (document how it was measured).
2. Robustness checks: varied lighting, background clutter, distance, left/right-handed signers, partial occlusion. Report measured accuracy per condition.
3. **End-user evaluation kit:** a short consent-based usability protocol and questionnaire (task completion, perceived accuracy, effort, comfort) to run with deaf and hard-of-hearing volunteers and with hearing participants who know no sign language, plus a script that aggregates the results. Do not invent results.
4. Package with PyInstaller (or document the exact build steps) for Windows, with a Linux/macOS run guide.
5. Write `README.md`, `docs/user_guide.md`, `docs/architecture.md`, and a **final project report draft** that follows the structure of the STEM Center proposal (Introduction, Objectives, Methodology, Results, Limitations, Conclusion, References), filled only with real, measured results.
6. Prepare a 5-minute demo script and a list of fallback steps if the live demo fails (for example, a pre-recorded landmark replay mode).

**Acceptance:** fresh-machine install following the README works; final report and demo script exist.

---

## 6. QUALITY, ETHICS, AND ACCESSIBILITY REQUIREMENTS

- **Community first:** state in the docs that the recognized vocabulary is limited, that the tool is a communication aid and **not a replacement for qualified human interpreters** in medical, legal, or emergency settings, and that real signers should validate vocabulary and data. Include a visible "limited vocabulary" notice in the UI.
- **Consent and privacy:** recorder requires explicit consent; only landmarks are stored by default; document data handling.
- **Bias and fairness:** report accuracy across signers, skin tones, lighting, and hand dominance; do not claim generalization beyond what was measured.
- **Honest metrics:** every number in docs must come from a script in the repo.
- **Error transparency:** show confidence, and never silently output a low-confidence guess as fact.

---

## 7. DEFINITION OF DONE

- [ ] `pip install -r requirements.txt && python -m signtranslator.app` launches the app on a clean machine.
- [ ] Live sign recognition works for the defined vocabulary using only a webcam.
- [ ] Sentences are generated (not just single words) and translated into all six languages with correct scripts and RTL for Arabic.
- [ ] Spoken output works for each language, or a clear fallback message is shown.
- [ ] Zero paid dependencies; total cost $0.
- [ ] Test suite passes; evaluation report contains real measured metrics.
- [ ] README, user guide, architecture doc, final report draft, and demo script are delivered.
