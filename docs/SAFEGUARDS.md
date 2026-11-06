# Safeguards design (SR-1 to SR-10)

Realistic synthetic surveillance video can be misused to fabricate evidence or spread misinformation. In this product, safeguards are part of the generation path itself, not an add-on. A clip cannot leave the pipeline unless every check passes.

## Generation path

1. **Customer gate (SR-5).** The API key must belong to an active customer with verified KYC who has accepted the current terms version. Restricted use (`public_safety_training`) also needs an enterprise tier and a contract reference on file.
2. **Release gate (SR-10).** The renderer version must currently be approved by the safeguards owner. Clean export is a separately gated feature.
3. **Prompt filter (SR-6).** The prompt and the free-text camera label are screened. Refusals are audited with the prompt and the categories that matched.
4. **Render.** The renderer understands only a closed vocabulary of scenes and events. Unrecognised words have no effect, so a prompt cannot make it depict a specific person, place or incident.
5. **Visible label (SR-1),** applied to every frame after the realism stage, so compression cannot blur it out of existence.
6. **Invisible watermark (SR-2),** carrying a random 32-bit clip token plus a CRC-16.
7. **QA gate.** The encoded file is decoded again. The label must be detectable, and the watermark must decode to this clip's token. On failure the clip is deleted and the failure audited.
8. **Provenance (SR-3).** The SHA-256 of the encoded video stream goes into a manifest. The manifest is signed with Ed25519, embedded in the MP4 `comment` tag, and also written as a sidecar file.
9. **Audit (SR-7).** The log records principal, customer, prompt, parameters, output stream hash and file hash, in a hash-chained log.

## SR-1 Visible label

The label has three layers:

* a solid amber-on-black badge, bottom right: `SYNTHETIC - AI GENERATED`
* a smaller `SYNTHETIC` badge, top right
* a faint tiled `SYNTHETIC` pattern across the whole frame (7% opacity)

The on-screen timestamp is also marked `SYN` and uses the fixed simulation date 2000-01-01, so it never asserts a real time. `label_score()` correlates the badge region with a reference rendering. The pipeline requires at least 0.6 before encoding and 0.5 after decoding.

## SR-2 Invisible watermark

The scheme is `vl-ss-dct-v1`, a keyed spread-spectrum mark in the luma channel:

* **Patterns.** There are 49 patterns of ±1 values on a 64×64 grid, upsampled to 256×256: one sync pattern plus 48 payload bits. The patterns come from HMAC(watermark key).
* **Amplitude and timing.** The embedding has a luma standard deviation of 2.0 and is texture-masked. Its sign flips every 0.25 s of clip time. The flip is tied to time, not frame index, so it survives frame-rate changes.
* **Detection.** Each frame is resized to 256×256, high-pass filtered and correlated with the patterns. Frames are split into the + and − groups with a 1-D two-means on the sync correlation. The midpoint of the two groups estimates the static scene content, which is then cancelled. The bits are decided by t-statistics across frames.
* **Presence.** A mark counts as present only if the CRC passes (2⁻¹⁶ chance by accident) and every bit has |t| ≥ 3.

Measured with `scripts/eval_watermark.py` on 6 clips of 6 s each, across all scenes:

| Attack | Detection |
|---|---|
| None (platform encode) | 100% |
| Re-encode H.264 CRF 28 | 100% |
| Re-encode H.264 CRF 35 | 83% |
| Resize to 75% | 100% |
| Resize to 50% + CRF 28 | 83% |
| Resize to 50% + CRF 32 | 50% |
| Half frame rate | 100% |
| MPEG-4 Part 2 transcode | 100% |
| Brightness/contrast change | 100% |
| Grayscale | 100% |
| Trim first second | 100% |
| Crop 10% | 0% (not designed for) |
| Rotate 3° | 0% (not designed for) |

Mean PSNR was 40.7 dB, and there were no false positives on the unmarked versions.

**[VERIFY]** These figures come from a small internal set. Before Phase 2 the safeguards owner should commission an attack evaluation by someone who did not build the scheme. It should cover adversarial removal, averaging and collusion, and a geometric re-synchronisation step should be considered for crops.

## SR-3 Provenance

The manifest fields are:

* schema and claim generator
* clip ID and creation time
* `synthetic: true`
* IPTC digital source type `trainedAlgorithmicMedia`
* disclaimer
* backend, renderer and package versions, watermark scheme
* customer ID and declared use
* SHA-256 of the parameters and of the prompt (the prompt itself stays in the audit log)
* visible-label flag and clean-export contract
* video stream hash, dimensions, fps and frame count
* a `c2pa.created` action

The video stream hash is computed over the encoded packets (`ffmpeg -f hash`), so it is unaffected by container metadata edits.

Signing keys sit under `data/keys/` with 0600 permissions. Retired public keys stay in `trusted_public_keys.json` so older clips still verify. Production custody should move to a KMS/HSM **[VERIFY]**. The field vocabulary follows C2PA so a later move to a conformant C2PA manifest is mechanical **[VERIFY against current spec]**.

## SR-4 Verification

`POST /v1/verify` and the `/verify` page combine four checks:

1. the signature and stream hash of the embedded manifest
2. an exact stream-hash lookup in the registry
3. watermark token lookup
4. visible label detection

Possible verdicts:

* `verified_platform_original`
* `platform_original_metadata_altered`
* `platform_origin_modified`
* `platform_manifest_stream_modified`
* `invalid_provenance_claim`
* `no_platform_marks_found`, which always comes with the caveat that it does not prove a video is real

Uploaded files are processed in a temporary file and not retained. The verdict and clip ID are audited. Public results truncate the customer ID; the admin endpoint shows it in full.

## SR-5 Customer verification

Customers are created by an admin, and KYC decisions are recorded with a document reference, the approver and a timestamp. API keys are stored only as SHA-256 hashes and can be revoked. Customers can be suspended or terminated, which blocks generation immediately.

## SR-6 Filtering

Prompts are screened by the rule-based categories in `content_filter.py`:

* evidence and real-footage misuse
* label removal
* real people (lists, a capitalised-name heuristic, handles, emails and phone numbers)
* real locations (addresses, PIN codes, a gazetteer)
* real incidents
* political and communal content
* sexual content
* minors
* harassment and impersonation
* weapons and violence, which are outside the v1 event library

On the output side, the closed vocabulary means outputs can only contain library content, and the QA gate guarantees the label and watermark are present. A learned classifier and gazetteer expansion are tracked in OPEN_ITEMS.

## SR-7 Audit log

Each entry stores `sha256(ts, actor, customer, action, details, prev_hash)`. `cctvgpt audit verify` and `GET /v1/admin/audit/verify` detect any altered or removed entry. Retention purging (`cctvgpt audit purge`) is itself audited. The retention period is **[OPEN]** for counsel.

## SR-9 Clean export

Clean export is unavailable unless all four conditions hold:

1. `CCTVGPT_CLEAN_EXPORT_ENABLED=true` (off by default, pending legal confirmation)
2. the customer is enterprise tier
3. the customer has a clean-export contract reference on file
4. the safeguards owner has approved the `feature/clean_export` release

Clean clips omit only the visible label. The watermark, signed metadata (which records the contract reference) and audit logging stay in place.

## SR-10 Release gate

`cctvgpt release approve` and `POST /v1/releases/decide` run the safeguards self-test first:

* the label is on every frame
* the watermark survives the platform encode and a re-encode with resize
* there are no false positives
* the provenance signature is valid and tampering is detected
* the filter blocks the reference prohibited set and allows the in-scope set

Approval is refused if any check fails. A `blocked` decision takes effect on the next generation request. Every decision is audited with the owner's name.

## Known limitation (to be stated to customers and regulators)

> Watermarks and labels can be degraded, cropped or covered by a determined bad actor. The safeguards reduce misuse and make provenance checkable; they do not eliminate it.
