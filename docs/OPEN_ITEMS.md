# Open decisions and items to verify

The PRD marks items **[OPEN]** (decisions for leadership) and **[VERIFY]** (facts to confirm). This register shows where each one touches the code and what the default is until it is decided. Nothing below is resolved by the software.

## Configuration waiting on a decision

| Item | PRD ref | Where | Current default | Owner |
|---|---|---|---|---|
| Audit-log retention period | SR-7 [OPEN] | `CCTVGPT_AUDIT_RETENTION_DAYS`, `cctvgpt audit purge` | Retain indefinitely | Counsel |
| Legality of clean exports | SR-9 [VERIFY] | `CCTVGPT_CLEAN_EXPORT_ENABLED`, plus the `feature/clean_export` release approval | Disabled. The final IT Rules require intermediaries to "prevent ... removal of the label", so withdrawal for India is recommended (docs/COMPLIANCE.md) | Counsel, CEO (open question 4) |
| Terms of use text | SR-8 | `docs/TERMS_OF_USE.md`, `TERMS_VERSION` in `context.py` | Draft `2026-11-draft-1`, which customers must accept | Counsel |
| Misuse triage and response targets | SR-8 | `docs/MISUSE_POLICY.md` | 1 business day (proposed) | Compliance owner |
| Maximum clip length and batch size | FR-1, 6.3 | `CCTVGPT_MAX_DURATION_S`, `CCTVGPT_MAX_BATCH` | 30 s; 50 clips | Technical lead, after compute quotes |
| Named safeguards owner | SR-10 | `cctvgpt init --owner` | Whoever is bootstrapped | CEO |
| Pursue police and public-safety at all | Section 4, open question 6 | `public_safety_training` use case needs an enterprise contract | Allowed only with contract | CEO |

## Items to verify

| Item | PRD ref | Status in this build |
|---|---|---|
| Watermark robustness under attack | SR-2 | Internal evaluation in `docs/SAFEGUARDS.md`; independent attack test still needed |
| Conformance with the current C2PA specification | SR-3 | C2PA-aligned fields only; migrate to a conformant manifest via the c2pa SDK |
| Key custody (KMS/HSM) | SR-3 | Keys stored as 0600 files under `data/keys` |
| Coverage of the prompt filter's names, places and incidents | SR-6 | Sample lists in `cctvgpt/safeguards/data/`; add a learned classifier before Phase 3 |
| IT Rules 2026 amendment: labelling, metadata, takedown timelines, intermediary status | Section 7 | Researched in docs/COMPLIANCE.md. Label, metadata and the platform unique identifier are implemented. Still open: intermediary status, the 2–3 hour takedown targets, quarterly user notices, and whether weapon-detection training data is permitted |
| DPDP Act applicability to prompts and customer data | Section 7 | Prompts and contact data are stored; needs counsel review and a privacy notice |
| Encryption at rest | 6.3 | Expected from the host volume; not implemented in-app |
| Compute cost per clip | 6.3, section 12 | About 4 s of CPU per 8 s 640×360 clip with the simulation backend; Option A costs unknown |
| Licences for any third-party model or dataset | Section 7 | None used. Runtime dependencies: NumPy, OpenCV, cryptography, FastAPI, uvicorn and FFmpeg. FFmpeg/libx264 licensing for commercial distribution needs checking |
| Customer problem validation | Section 2 | Phase 0 interviews still to run |

## Technical decisions for the technical lead

* Option A, B or C (PRD section 8). The simulation backend (B) is implemented. `cctvgpt/backends` defines the adapter for A or for an AI realism pass.
* Additional scenes and events: violence, vehicles, and crowd types beyond the v1 library.
* Production datastore (Postgres) and job workers (a separate queue service) for Phase 2 scale.
