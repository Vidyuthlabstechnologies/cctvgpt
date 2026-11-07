# Forensic footage search

Find specific people, vehicles, objects and moments across photo galleries, video files and CCTV/NVR exports, including dark, noisy and IR night footage. Investigators can search in four ways:

- **Description**, in English or an Indian language (translated automatically), e.g. "man in a red shirt on a motorcycle", "फर्श पर लेटा हुआ व्यक्ति".
- **Example photo.**
- **Number plate**: full, partial, or with OCR errors.
- **Face matching**: opt-in, case-scoped, for human verification.

Results can be filtered by place type and vehicle colour, and footage shot at the same place is grouped automatically. Results are time-stamped segments; any of them can be exported from the untouched original with a chain-of-custody record.

It runs fully on-premises on CPU (no cloud calls), which suits police IT. A GPU speeds up ingest.

## How it works

```
footage folder / upload
  -> SHA-256 of every original (never modified)
  -> synthetic check: CCTVGPT watermark/provenance -> flagged and excluded from search
  -> frames sampled (1 per second + scene changes)
  -> person / two-wheeler / car / bus / truck / bag detection (on an enhanced copy of dark frames)
  -> CLIP embeddings of each frame and each detected object crop
  -> place-type tags per source; colour tag per vehicle; background image per source (for same-place grouping)
  -> number plates read on every vehicle crop (and on whole photos)
  -> faces detected and embedded (only when face search is enabled)
  -> search: text (translated if needed), photo, plate or face -> ranked segments (nearby hits merged)
  -> export: stream copy from the original + hashes + custody record
```

| Component | Choice | Licence notes |
|---|---|---|
| Embeddings | OpenCLIP ViT-B-32 (LAION-2B); optional multilingual XLM-R ViT-B-32 | Check model and dataset terms before commercial deployment **[VERIFY]** |
| Detection | torchvision Faster R-CNN MobileNetV3 (COCO) | BSD code; COCO weights |
| Index | SQLite + float16 vectors, brute-force cosine | Fine to about 1M items; FAISS/pgvector beyond that |
| Query translation | IndicTrans2 (AI4Bharat, MIT; 22 languages) when its gated weights are available, else Helsinki-NLP opus-mt-hi-en (Apache-2.0, Hindi) | IndicTrans2 needs a Hugging Face account to download once (`HF_TOKEN`, or `CCTVGPT_INDICTRANS2_PATH` for air-gapped installs) |
| Plate localisation / reading | PP-OCRv3 text detector (Apache-2.0) + fast-plate-ocr `cct-xs-v2-global` (MIT); CRNN fallback | Indian-format normalisation and look-alike correction in `plates.py` |
| Face detection / matching | YuNet (MIT) + SFace (Apache-2.0), OpenCV Zoo | Off unless `CCTVGPT_FACE_SEARCH_ENABLED=true` |
| Same-place grouping | ORB keypoints + RANSAC homography on each source's median background | Fixed cameras; overlapping views |

Commands:

```bash
pip install -e '.[search]'
```

```bash
cctvgpt search ingest /evidence/FIR-0123 --case FIR-0123/2026 --camera "Market Gate 2" --officer "SI Sharma"
```

```bash
cctvgpt search query "man in a red shirt on a motorcycle" --case FIR-0123/2026 --purpose "trace suspect after chain snatching" --officer "SI Sharma"
```

```bash
cctvgpt search export src_xxx,41.0,55.0 --case FIR-0123/2026 --officer "SI Sharma"
```

The web page is `/search`. The API lives under `/v1/search/*` and needs an `investigator` or `admin` key. Server-side ingest only reads folders listed in `CCTVGPT_SEARCH_ROOTS` plus the upload inbox.

## Measured accuracy: plates, faces, translation (benchmarks/plates_faces_eval.py)

| Capability | Data | Clean | CCTV-like degradation |
|---|---|---|---|
| Plate reading (exact / within 1 char / plate search finds it first) | 965 real Indian plates (internal evaluation only; no licence stated) | 60% / 68% / 71% | Half resolution: 21% / 27% / 30% · Night: 15% / 20% / 22% · Night + enhancement: 18% / 23% / 26% |
| Face matching (accuracy, true-match rate, false-match rate) | LFW, 1,000 verification pairs | 99.0%, 98.0%, 0.0% | Face 45 px: 98.8% · Face 20 px: 90.8% (true match 82%, false match 0.4%) · JPEG q15: 98.4% · Night: 50.5% (98.6% of faces not detected) · Night + enhancement: 64.5% (true match 29%, false match 0%) |
| Hindi query, translated (AP / P@10), URFD "lying" frames | Real footage | "फर्श पर लेटा हुआ व्यक्ति" → "The man lying on the floor": **0.51 / 1.0** (untranslated 0.23; English 0.60) | Night: 0.39 / 0.7 |

What this means:

- **Plates are usable** on reasonably sized plates. They are weak at low resolution and at night, where dedicated ANPR cameras (IR illuminated, high shutter speed) or plate-specific fine-tuning on Indian plates are needed. The synthetic generator can render Indian plates for that training.
- **Face matching is accurate on visible faces**, down to about 20 px, and **rarely matches the wrong person** at the default threshold (false matches 0–0.4% in every condition). It **fails in the dark**, because faces are not even detected. LFW is mostly frontal photos of public figures and is not representative of Indian demographics or CCTV angles. Accuracy must be re-measured on Indian, CCTV-angle data before any operational use **[VERIFY]**.
- **Translation fixes Hindi search** (AP 0.23 → 0.51 for the best phrasing). The fallback model is sensitive to phrasing ("गिरा हुआ" became "dropped"), which is why IndicTrans2 is preferred and the translated text is always shown.
- **Same-place grouping** correctly grouped the same office by day and by (degraded) night, with 53 matched keypoints, and kept different rooms apart. Embedding similarity alone could not: two different rooms scored 0.90.
- **Place-type tags** are zero-shot and indicative only. They are not evaluated quantitatively.

## Measured accuracy: description search (benchmarks/search_eval.py)

### Real footage

UR Fall Detection Dataset: 752 frames at 2 fps, 163 of them labelled "lying". Query: "a person lying on the floor". Random ranking would score AP 0.22.

| Model | Day: AP / P@10 | Poor night (dark, noisy, JPEG): AP / P@10 |
|---|---|---|
| English CLIP | **0.60 / 1.0** | 0.42 / 0.7 |
| English CLIP + classical enhancement + crops | 0.60 / 1.0 | 0.40 / 0.8 |
| Multilingual CLIP, English query | 0.47 / 1.0 | **0.46 / 1.0** |
| Multilingual CLIP, Hindi query ("फर्श पर लेटा हुआ व्यक्ति") | 0.23 / 0.1 | 0.26 / 0.2 |

### Synthetic footage with exact clothing ground truth

Query: "a person wearing a <colour> shirt", over 7 colours and 165–235 person crops per condition.

| Condition | Mean AP (random ≈ 0.12) | Notes |
|---|---|---|
| Day | 0.60 | red 0.91, purple 0.86, green 0.76; white 0.09, black 0.37 |
| Dusk | 0.56 | |
| Night with lighting (colour camera) | 0.60 | |
| Rain | 0.48 | |
| Fog | 0.46 | |
| **Night IR (monochrome)** | **0.18** | Colour cannot be recovered from IR. The product now warns |

### What this means

- **Search works on real footage.** At P@10 = 1.0, every one of the top ten results is relevant.
- **Poor night footage costs about a third of AP** with the English model. Classical denoise/CLAHE/gamma enhancement did **not** help the embedding model, so enhanced-frame indexing is off by default. Enhancement is kept for the detector and for display.
- **The multilingual model holds up better at night** but is weaker by day.
- **Hindi queries do not work yet** (chance level). Indian-language search needs a translation step (for example IndicTrans2) or a stronger multilingual model. The product flags non-English queries rather than giving false confidence.
- **Colour search is unusable on IR footage**, and white and black clothing is unreliable in any light. Results from IR sources carry a warning.

### Limits

- One indoor real dataset and one query type.
- Simulated night degradation, not real night recordings.
- The synthetic colour test uses ground-truth crops, so it isolates retrieval from detection.

Real CCTV night archives from a pilot partner, labelled for evaluation, are the next step.

## Roadmap to stronger search

1. **Robustness to night footage via synthetic data.** Fine-tune the embedding model on paired day/night/IR renders from the CCTVGPT generator. This is where the generator becomes the training engine.
2. **Attributes that survive IR.** Train clothing-pattern, carried-object, two-wheeler and helmet attributes, and use upper/lower clothing *brightness* instead of hue on IR footage.
3. **Indian languages.** Done with translation: enable IndicTrans2 on the server, then evaluate each language with native-speaker query sets.
4. **Plates.** Fine-tune the recogniser on Indian plates (real + synthetic), add dedicated ANPR camera feeds, and add state/RTO decoding.
5. **Faces at night.** Add a low-light face detector, and evaluate on Indian CCTV-angle data with demographic breakdowns before any operational use.
6. **Scale.** GPU ingest, FAISS/pgvector, NVR/RTSP connectors (ONVIF), and DAV/proprietary DVR formats via vendor SDKs.
7. **Integration** with CCTNS/ICJS case systems and MHA's e-Sakshya evidence app **[VERIFY APIs and procurement route]**.

## Legal and ethical design (research summary, not legal advice)

Police search over CCTV involves personal data at scale. The design follows the constitutional proportionality test for privacy intrusions (*K.S. Puttaswamy v. Union of India*, 2017): a legal basis, a legitimate aim, necessity, and procedural safeguards.

| Safeguard | Implementation |
|---|---|
| Purpose limitation | Every search requires a **case reference and a purpose**; both are audited |
| Accountability | Hash-chained audit of every ingest, search, media view and export, with officer names |
| Role-based access | Only `investigator` and `admin` keys can search; customer (synthetic-data) keys cannot |
| Evidence integrity | Originals hashed and never modified; exports are stream copies with custody records for a **BSA 2023 s.63** certificate **[VERIFY with counsel]** |
| No synthetic contamination | CCTVGPT-generated footage is detected by its watermark/provenance, flagged and excluded |
| Human review | Results are labelled as leads, not identifications; scores are relative |
| Face matching, tightly limited | **Off by default.** It must be enabled per deployment (`CCTVGPT_FACE_SEARCH_ENABLED`) after legal review; until then faces are not even indexed. When enabled, it compares an officer-supplied probe photo against **that case's footage only**. Cross-case search is admin-only. Each search needs a written justification and logs a hash of the probe image. Results are "possible matches for human verification", with weak and small-face results flagged |
| No gender classification | Not built. Automated gender labelling of every person is error-prone, misgenders people and adds little: witnesses' descriptions ("woman in a red saree") already work as description search |
| No community profiling | Place tags deliberately exclude religious sites |
| Data protection | DPDP Act 2023 obligations apply to processors; exemptions for State investigation under s.17 depend on government notification **[VERIFY]**. Retention of indexes and exports must follow the department's records policy |

Not built, and not to be built without leadership and legal sign-off:
- face identification against population databases (Aadhaar, passport, CCTNS photo galleries)
- watch-lists
- live alerting on people
- tracking named individuals across cameras in real time
- gait recognition
- gender or ethnicity classification

Legal points for counsel before enabling face matching **[VERIFY]**:
- the statutory basis for the police use of face matching in the deploying state
- DPDP Act 2023 treatment of facial data, and any State exemption under s.17
- retention and deletion of face embeddings after a case closes
- disclosure of accuracy limits to courts
- independent audit of the audit log
