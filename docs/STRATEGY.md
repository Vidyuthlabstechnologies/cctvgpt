# From feature to company: strategy for the synthetic-video vertical

**Status:** a draft for the CEO, written alongside the Phase 1 build. Market figures are third-party estimates that disagree with one another, so they are marked **[VERIFY]**. Nothing here substitutes for the Phase 0 customer interviews.

## 0. Update: footage search is the lead product

The founders' goal is **search**: finding specific photos, clips and moments across galleries, videos and CCTV records, including noisy night footage, mainly for Indian police. The synthetic generator becomes the engine behind that product, not the product itself:

| Role | What it does |
|---|---|
| **Search (lead product, for police and enterprises)** | Plain-language or photo search over footage; case-linked and audited; custody-preserving export (docs/SEARCH.md). On real footage the top-10 precision was 1.0 by day and 0.7–1.0 on degraded night footage. Weak spots measured: Hindi queries (chance level today), colour on IR footage, white/black clothing |
| **Generator as test bench** | Exact ground truth for night, IR, rain, fog and occlusion, used to *prove* search and analytics accuracy to buyers. This is Certify, below |
| **Generator as training engine** | Paired day/night/IR renders to fine-tune search for night robustness. This is ScenarioLab, below |
| **Verify** | Keeps synthetic footage out of investigations: search flags CCTVGPT-generated clips automatically |

The competitive set for search includes BriefCam (Canon), Genetec and Milestone add-ons, and Indian analytics vendors on Safe City projects. We would differentiate on:
- Indian conditions (night, IR, low bitrate, DAV exports)
- Indian languages (once fixed)
- on-premises CPU deployment at state-police budgets
- custody-grade exports
- measured accuracy published per condition

**[VERIFY]** pricing and procurement through state police modernisation budgets and GeM.

## 1. The honest starting point

**What we have now (this repository):**
- **Generator:** a working, safeguarded synthetic CCTV generator with exact ground truth.
- **Acceptance testing:** a scorecard service that measures any detector against controlled scenario suites.
- **Real-data evidence** (`benchmarks/README.md`):
  - A fall classifier trained on **zero real fall footage** reached **F1 0.91 (precision 1.0, AUROC 0.99) on real URFD falls**. That beats a hand-tuned rule (0.80) and approaches in-domain training (0.96).
  - With only one real example per class, adding synthetic data lifted F1 from 0.68 to 0.87.
- **Honest negatives:**
  - An off-the-shelf COCO detector recognises only about 9% of our synthetic people. The *appearance* gap is large, so today's value comes from motion and ground truth, not pixels.
  - Synthetic data stops helping on F1 once 3 or more real examples per class exist.

**What the market tells us:**
- **Synthetic data is a small market.** Global estimates for synthetic data across all modalities are about **$0.3–0.8B in 2024–25, growing to about $1.8–3.7B by 2030** [VERIFY].
- **The pure-play vision companies have consolidated or struggled.**
  - Datagen is reported closed.
  - Synthesis AI moved into a Globant partnership.
  - Gretel (tabular and text) was bought by NVIDIA.
  - NVIDIA itself gives away simulation and world-model tooling.
  - "Sell synthetic clips" is therefore not a billion-dollar business on its own.
- **The adjacent markets are larger:**
  - India's CCTV market is about **$4.2B (2024)** [VERIFY].
  - India's video-analytics market is about **$0.2B (2023), forecast at $0.78B by 2028** [VERIFY].
  - Content authenticity and provenance became a *legal requirement* in 2026, in India (IT Rules, February 2026) and the EU (AI Act Article 50, August 2026).

**The thesis:** VidyuthLabs should build the **trust infrastructure for video AI**. That means three products on one engine. The synthetic generator is the engine, not the product.

## 2. Three products, one engine

### A. ScenarioLab: rare-event data for training (the PRD as written)

* **What it is:** labelled, privacy-safe clips of falls, intrusions, fire, crowd surges and abandoned objects, plus the *confusers* (lie-down, crouch, walk-through) that cause false alarms. Delivered by API and batch sweeps.
* **Proof:** our URFD benchmark, plus the literature.
  * OmniFall (2025/26): synthetic-only training gave the best real-world fall F1 of any single source.
  * MOTSynth: synthetic-only pedestrian detection beat COCO on MOT17.
* **Buyers:** video-analytics companies (in India, the companies supplying analytics to Safe City projects), camera OEMs adding on-camera AI, and elder-care and hospital fall-detection vendors.
* **Pricing:** per-clip credits and enterprise data subscriptions.
* **Weakness:** commoditisation by open video models (Wan 2.x) and NVIDIA tooling. It must be bundled with B and C.

### B. Certify: independent acceptance testing of video analytics (the wedge)

* **The problem:** buyers (police, cities, metros, airports, factories) cannot check vendor accuracy claims. Real test footage of rare events doesn't exist, can't legally be staged at scale, and contains personal data.
* **What it is:** standard scenario suites with exact ground truth. Vendors submit outputs; buyers get a condition-by-condition scorecard, for example "person recall at night in rain at 15 m: 0.62; fall alarms: 9/10; false alarms on lie-down: 3".
  * The PRD explicitly allows this for public-safety buyers ("pre-deployment testing of analytics").
  * It is implemented in `cctvgpt/evaluation.py` and `POST /v1/batches/{id}/evaluate`.
* **Why it can become a standard:** suites are versioned and reproducible from a seed, so results are comparable across vendors and years. The goal is that **tenders cite "CCTVGPT Suite v1, night/rain subset, recall ≥ 0.8"**, like an "STQC/ARAI for video AI". Partnering with STQC, BIS or a state agency on a public test standard would be the moat [VERIFY feasibility with someone who has sold to government].
* **Buyers:** procurement teams, system integrators that need to prove compliance, insurers underwriting AI deployments, and vendors that want a badge.
* **Pricing:** per-certification fees, annual re-certification, and private suites for enterprises.

### C. Verify: footage authenticity and provenance (the scale play)

* **The problem:** cheap generative video makes fake "CCTV evidence" a real risk. India's new rules require labels and metadata on synthetic content and fast takedowns. Courts and police need to know whether a clip is genuine.
* **What it is:**
  1. **Verify-synthetic.** This already exists: the public `/verify` page detects *our* outputs. Extend it to read C2PA manifests and the watermarks of other generators that publish detectors.
  2. **Sign-at-capture.** An SDK and firmware partnership with Indian camera and NVR makers so that *real* footage is signed at the camera with C2PA. "This clip came from camera X at time T and has not been altered" becomes checkable. That is the stronger long-term product, because verifying real footage beats detecting fakes.
  3. **Consumer checker.** A free web/WhatsApp tool: "is this viral CCTV clip real?". It reads provenance and watermarks and gives an honest answer, including "unknown", for distribution and brand. Monetise through B2B APIs for newsrooms, platforms (whose SSMI declaration-verification duty creates demand) and insurers.
* **Note:** general deepfake *detection* is an adversarial arms race. We should not promise it. Provenance (signing) is the defensible part.

### Why one company

The engine (controllable generation plus exact labels plus provenance) feeds all three products. ScenarioLab's realism investment improves Certify's suites. Certify's scorecards generate demand for ScenarioLab data to fix the failures they reveal. Verify's provenance stack is the same code that keeps ScenarioLab compliant.

## 3. Who it serves (within the PRD's limits)

| Segment | Permitted uses | Not permitted |
|---|---|---|
| **Police and public safety** | Control-room operator training and drills; crowd-management rehearsal; **acceptance testing of analytics before deployment** (Certify); authenticity checks on footage they receive (Verify) | Investigation, evidence, suspect identification, interrogation, any public use as real |
| **Businesses** | Warehouse and factory safety (falls, fire, intrusion); retail (abandoned objects, crowding); hospitals and elder care (falls, long-lies); metros and airports (crowd surge); integrators' demos; insurers' risk assessment | Monitoring named employees or customers (no real people) |
| **Consumers** | Free authenticity checker (Verify); safer home and elder-care cameras built by OEMs trained on our data | **No consumer generation.** The PRD's non-goals (anonymous users, realistic fake CCTV) stay in force. A consumer generator would be the misuse vector the IT Rules target. |

## 4. Path to scale (assumption-driven, [VERIFY] everything)

A billion-dollar valuation typically needs roughly **$50–100M ARR** at venture multiples, or a dominant standard position. One illustrative path:

| Stage | Main revenue | Indicative goal |
|---|---|---|
| Year 1 (Phases 0–2) | ScenarioLab pilots with 3–5 tier-1 analytics vendors; first Certify engagements through an integrator | Proof: measurable detector gains at 2+ customers; 1 tender citing a Certify suite |
| Years 2–3 | Certify as a de facto standard in Indian public procurement; ScenarioLab subscriptions; sign-at-capture pilots with 1–2 camera OEMs | Several $M ARR; gross margin > 70% (CPU-cheap simulation; GPU only for realism passes) |
| Years 4–6 | Export Certify and Verify to markets with the same regulatory pull (the EU after the AI Act, the Gulf and Southeast Asia safe-city programmes); provenance licensing per camera | Tens of $M ARR; per-device provenance royalties are the scale lever |

The biggest unknowns are willingness to pay (Phase 0 interviews), government sales cycles, and whether a standards body will co-own a test suite.

## 5. Technical roadmap (what the evidence says to build next)

1. **Realism (the gating item, measured).** A COCO detector finds only about 9% of our synthetic people, so a realism pass is required before Certify can be sold. Add an AI video-to-video realism pass (for example Wan 2.x, Apache-2.0 [VERIFY licence]) *conditioned on our renders* (depth, pose, segmentation). This keeps exact labels and camera control while closing the appearance gap. OmniFall shows that diffusion-generated video transfers well; our simulator supplies the control and ground truth those models lack.
2. **Scene and behaviour library.** Add sitting and falls from chairs (missing, and visible in URFD's real falls), stairs, vehicles, crowds by Indian context (markets, railway platforms, festivals), and Indian clothing and skin-tone diversity. Diversity mattered more than volume in MOTSynth.
3. **Provenance.** Ship a conformant C2PA manifest and evaluate a second neural watermark (Meta Video Seal, MIT [VERIFY]). Commission an independent attack test of SR-2.
4. **Certify.** Publish versioned public suites with leaderboards; add event-level timelines, a latency SLA and a calibration report.
5. **Benchmarks as a habit.** Every release runs `benchmarks/` against licensed real sets, and results go into the release-gate record.

## 6. Risks specific to this strategy

* **Becoming a "test lab" with low margins.** Mitigate by productising suites rather than doing bespoke testing, and charging per certification and per device.
* **Regulatory dependence.** A compliance-driven product can stall if enforcement is weak. Mitigate by leading with real accuracy problems (false alarms cost money) and treating compliance as the accelerant, not the reason to buy.
* **Government concentration.** Lead with tier-1 vendors and enterprises; treat government as a later, integrator-led channel (as the PRD says).
* **Misuse and reputation.** The safeguards and release gate stay non-negotiable. Never offer consumer generation. Withdraw SR-9 clean exports for India (see docs/COMPLIANCE.md).

## 7. Decisions needed from leadership

1. Endorse the three-product framing (ScenarioLab / Certify / Verify), or keep scope to the PRD's data product.
2. Approve withdrawing SR-9 clean exports for India, pending counsel.
3. Approve outreach to one standards body or state agency about a public analytics test suite.
4. Budget a GPU realism-pass experiment (Phase 1b) and an independent watermark attack test.
5. Run the Phase 0 interviews with Certify questions added: "How do you verify analytics accuracy before acceptance today? What would an independent scorecard be worth?"

## Sources

- Market estimates: [Grand View Research](https://www.grandviewresearch.com/press-release/global-synthetic-data-generation-market), [ResearchAndMarkets via BusinessWire](https://www.businesswire.com/news/home/20250113130135/en/), [MarketsandMarkets India video analytics](https://www.marketsandmarkets.com/Market-Reports/geography/intelligent-video-analytics-market/India), [IMARC India CCTV](https://www.imarcgroup.com/India-CCTV-Market)
- Industry consolidation: [Pebblous on Datagen / Synthesis AI / Gretel](https://blog.pebblous.ai/project/SyntheticData/synthetic-data-companies-rise-fall/en/), [Turing Post synthetic data platforms 2026](https://www.turingpost.com/p/synthetic-data-companies)
- Indian deployments: [Awiros, Bangalore Safe City](https://cxotoday.com/media-coverage/awiros-powers-artificial-intelligence-based-video-analytics-for-bangalore-safe-city-project/), [Analytics India Magazine](https://analyticsindiamag.com/bangalore-police-taps-into-ai-based-video-analytics-to-make-the-city-safer/)
- Research: [OmniFall (arXiv 2505.19889)](https://www.alphaxiv.org/abs/2505.19889), [MOTSynth (arXiv 2108.09518)](https://www.alphaxiv.org/abs/2108.09518), [UBnormal (arXiv 2111.08644)](https://www.alphaxiv.org/abs/2111.08644), [GTA-Crime (arXiv 2509.08232)](https://www.alphaxiv.org/abs/2509.08232), [Video Seal (arXiv 2412.09492)](https://arxiv.org/pdf/2412.09492)
- Regulation: see [docs/COMPLIANCE.md](COMPLIANCE.md)
