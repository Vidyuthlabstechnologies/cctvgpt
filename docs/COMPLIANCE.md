# Compliance analysis (research summary, not legal advice)

**Prepared:** 6 November 2026, from public sources. This is the PRD section 7 checklist updated with what the final rules say. Every item needs counsel's confirmation before external release. Sources are listed at the end.

## 1. India: IT Rules amendment on synthetically generated information (SGI)

The Information Technology (Intermediary Guidelines and Digital Media Ethics Code) Amendment Rules, 2026 were notified on 10 February 2026 (G.S.R. 120(E)) and came into force on 20 February 2026.

### What the final rules require, per law-firm summaries

| Obligation (final rules) | How CCTVGPT meets it | Status |
|---|---|---|
| SGI defined as audio-visual content artificially created or altered that "reasonably appears to be real, authentic or true". Our outputs are designed to look like real CCTV, so they are in scope. | Treated as in scope | Assumed in scope **[VERIFY whether VidyuthLabs is an "intermediary" for its own B2B generation service]** |
| Intermediaries offering SGI-creation tools must label it with "prominent visibility", "easily noticeable and adequately perceivable". The draft's 10%-of-surface-area rule was dropped in the final text. | SR-1: two corner badges plus a full-frame tiled `SYNTHETIC` pattern, checked on every frame | Meets the stated standard **[VERIFY: is "prominent" satisfied by corner badges? Consider a larger default badge]** |
| Embed "permanent metadata or other appropriate technical provenance mechanisms, to the extent technically feasible", including a **unique identifier of the intermediary's computer resource** | SR-3 signed manifest, now with `platform_uid` (`in.vidyuthlabs.cctvgpt:<key id>`), plus the SR-2 invisible watermark carrying the clip token | Implemented in this release |
| Intermediaries must "**prevent any modification, suppression or removal** of the label, permanent metadata, or unique identifier" | Label is burned into pixels; watermark is robust to common edits; metadata is signed, and tampering is detected on verification | Implemented for platform outputs |
| Deploy "reasonable and appropriate technical measures" to block unlawful SGI. This covers CSAM, non-consensual intimate imagery, obscene content, **false documents or false electronic records**, explosives/arms/ammunition, and **content that falsely depicts a person or real-world event in a deceptive way** | SR-6 filter categories cover each; closed-vocabulary renderer; faceless actors; no real places or people | Implemented (rule-based) **[VERIFY coverage]** |
| Takedown in 3 h on a court order or government notice; 2 h for intimate or impersonation content; grievances resolved in 7 days | Misuse-report channel and admin workflow (docs/MISUSE_POLICY.md); audit log identifies the clip and customer even from re-encoded copies | Process exists. **Response targets must be tightened to hours** (the current proposal says 1 business day) |
| Users told about the consequences of violations at least every 3 months | Not implemented | **To do:** quarterly terms reminder to customers |

### Consequences for the PRD

1. **SR-9 (clean export) very likely conflicts with the "prevent removal of label" duty.** It remains disabled by default and should be withdrawn for India unless counsel signs off. This answers CEO open question 4 in the negative for now.
2. **CCTV footage is an "electronic record".** A fake clip passed off as real could be a "false electronic record", which is exactly the risk the safeguards address. The terms' ban on evidence use, and the labelling, are central to the legal position. They are not optional extras.
3. **Weapons.** The rules list arms and ammunition among the SGI categories to block. Weapon-detection training data is a large market served by foreign competitors that publish synthetic gun and knife CCTV datasets. **[VERIFY]** whether purpose-limited, labelled training data for weapon *detection* is permitted. Until then, it stays out of the v1 library (the filter's `unsupported_violence` category).

## 2. India: DPDP Act 2023 and DPDP Rules 2025

The DPDP Rules were notified on 13–14 November 2025 with a phased start:
- **Immediately:** definitions and the Data Protection Board.
- **About November 2026:** consent-manager registration.
- **About May 2027:** most data-fiduciary obligations, including notice, security safeguards, breach notification and retention.

Where the product touches personal data:
- **Customer contact data and user identities** (KYC records, API-key principals): VidyuthLabs is the data fiduciary. A privacy notice, a retention schedule and breach procedures are needed by May 2027.
- **Prompts:** the filter blocks real names, but prompts may still contain personal data. They are stored in the audit log (SR-7). The retention period is set to [OPEN] until counsel decides.
- **Outputs:** they contain no real people by construction (faceless procedural actors), which is the product's privacy advantage over real footage.
- **Customer-supplied footage** (future features such as scene capture or domain adaptation): this would make VidyuthLabs a processor of footage containing personal data. **Keep it out of scope** until a DPDP-compliant processing agreement exists.

## 3. Evidence law (India)

Evidence law is why outputs must never be presented as real:
- The Bharatiya Sakshya Adhiniyam, 2023 (Section 63) governs the admissibility of electronic records, including CCTV, and requires certification.
- The Bharatiya Nyaya Sanhita, 2023 criminalises fabricating false evidence.

The verification service (SR-4) is also useful in reverse. Courts, police and insurers can check whether a clip presented as real is in fact a platform output. **[VERIFY section numbers and applicability with counsel]**

## 4. Exports: EU AI Act, Article 50

* Article 50 applies from **2 August 2026**. Providers of generative systems must mark outputs "in a machine-readable format and detectable as artificially generated". The Commission's **Code of Practice on marking and labelling** (final version June/July 2026) expects *multi-layered* marking, because no single technique is robust enough.
* CCTVGPT already layers a visible label, an invisible watermark and signed metadata, which matches the Code's approach. To claim interoperability, adopt **C2PA** manifests (via the c2pa SDK) and consider a second, independently evaluated neural watermark, such as Meta's open-source Video Seal **[VERIFY licence and robustness]**.
* Deployers of deepfakes have disclosure duties under Article 50(4). EU customers need contract terms that pass these on.

## 5. Dataset and model licences used in evaluation

| Asset | Licence | How we use it |
|---|---|---|
| UR Fall Detection Dataset (URFD) | CC BY-NC-SA 4.0, non-commercial academic | Internal evaluation only (`benchmarks/`). Never redistributed or used to train shipped models. Commercial benchmarking claims need the authors' permission. |
| torchvision Faster R-CNN MobileNetV3 weights (COCO) | BSD-3 code; COCO annotations are CC BY 4.0, while COCO images carry Flickr licences | Used as an independent third-party detector in evaluation only |
| OmniFall OF-Syn and OF-ItW (candidates for the next benchmark) | Check before use; OF-ItW derives from OOPS (YouTube) clips | Not used yet |
| FFmpeg / libx264 | LGPL/GPL depending on build; x264 is GPL or commercially licensed | **[VERIFY]** before distributing binaries in a commercial product |

## 6. Public-sector sales (police and Safe City)

* Permitted (PRD section 4):
  * control-room operator training and drills
  * crowd-management rehearsal
  * **pre-deployment acceptance testing of analytics**, which `cctvgpt/evaluation.py` implements
* Prohibited: investigations, evidence, suspect identification, interrogation, and any public presentation as real.
* Government procurement runs through GeM and state tenders, so expect empanelment, CERT-In-empanelled audits and data-localisation clauses **[VERIFY with someone who has sold to government]**.

## Sources

- [Chandhiok & Mahajan: e-alert on the IT Rules amendment 2026](https://chandhiok.com/insights/cm-e-alert-ministry-of-electronics-and-information-technology-notifies-the-information-technology-intermediary-guidelines-and-digital-media-ethics-code-amendment-rules-2026)
- [DSCI: IT (Amendment) Rules, 2026](https://www.dsci.in/resource/content/it-amendment-rules-2026)
- [MediaNama: MeitY notifies amendments on synthetic media](https://www.medianama.com/2026/02/223-meity-amendments-it-rules-synthetic-media-deepfakes/)
- [HLC: mandatory AI labelling and 3-hour takedown](https://www.hlc.com/en/publications/india-introduces-mandatory-labelling-for-ai-and-3hour-takedown-for-illegal-content)
- [YourStory: DPDP Rules 2025 rollout over 18 months](https://yourstory.com/2025/11/digital-personal-data-protection-dpdp-rules-2025-rollout-over-18-months)
- [Mondaq: analysing the DPDP Rules 2025](https://www.mondaq.com/india/data-protection/1708434/operationalising-indias-data-protection-regime-analysing-the-newly-notified-dpdp-rules-2025)
- [Jones Day: final Code of Practice on marking and labelling AI-generated content](https://www.jonesday.com/en/insights/2026/06/european-commission-publishes-final-code-of-practice-on-marking-and-labelling-aigenerated-content)
- [Cooley: first draft Code of Practice on transparency](https://www.cooley.com/news/insight/2025/2025-12-18-eu-ai-act-first-draft-code-of-practice-on-transparency-and-watermarking-released)
- [Video Seal paper (Meta)](https://arxiv.org/pdf/2412.09492)
- [URFD dataset page and licence](http://fenix.ur.edu.pl/~mkepski/ds/uf.html)
