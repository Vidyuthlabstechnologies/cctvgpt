# Forensic footage search

Find specific people, vehicles, objects and moments across photo galleries, video files and CCTV/NVR exports, including dark, noisy and IR night footage. Investigators search in plain words ("man in a red shirt on a motorcycle", "person lying on the floor") or with an example photo. Results are time-stamped segments; any of them can be exported from the untouched original with a chain-of-custody record.

It runs fully on-premises on CPU (no cloud calls), which suits police IT. A GPU speeds up ingest.

## How it works

```
footage folder / upload
  -> SHA-256 of every original (never modified)
  -> synthetic check: CCTVGPT watermark/provenance -> flagged and excluded from search
  -> frames sampled (1 per second + scene changes)
  -> person / two-wheeler / car / bus / truck / bag detection (on an enhanced copy of dark frames)
  -> CLIP embeddings of each frame and each detected object crop
  -> search: text or photo -> ranked segments (nearby hits merged)
  -> export: stream copy from the original + hashes + custody record
```

| Component | Choice | Licence notes |
|---|---|---|
| Embeddings | OpenCLIP ViT-B-32 (LAION-2B); optional multilingual XLM-R ViT-B-32 | Check model and dataset terms before commercial deployment **[VERIFY]** |
| Detection | torchvision Faster R-CNN MobileNetV3 (COCO) | BSD code; COCO weights |
| Index | SQLite + float16 vectors, brute-force cosine | Fine to about 1M items; FAISS/pgvector beyond that |

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

## Measured accuracy (benchmarks/search_eval.py)

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
3. **Indian languages.** Translate queries with IndicTrans2 or similar, or adopt a stronger multilingual model; evaluate per language.
4. **Vehicle search.** Number-plate OCR for Indian plates (HSRP and older formats), plus vehicle type and colour.
5. **Scale.** GPU ingest, FAISS/pgvector, NVR/RTSP connectors (ONVIF), and DAV/proprietary DVR formats via vendor SDKs.
6. **Integration** with CCTNS/ICJS case systems and MHA's e-Sakshya evidence app **[VERIFY APIs and procurement route]**.

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
| No biometric identification | **No face recognition.** Adding it is a separate policy decision needing a clear legal basis, accuracy testing across demographics, and oversight |
| Data protection | DPDP Act 2023 obligations apply to processors; exemptions for State investigation under s.17 depend on government notification **[VERIFY]**. Retention of indexes and exports must follow the department's records policy |

Not built, and not to be built without leadership and legal sign-off:
- tracking named individuals across cameras
- watch-lists
- live alerting on people
- face or gait recognition
