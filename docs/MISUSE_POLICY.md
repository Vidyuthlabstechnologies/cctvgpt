# Misuse policy and response process (DRAFT)

The owner of this document is the compliance owner (PRD section 11). It covers the Phase 0 deliverables "misuse policy and banned-use list" and "misuse reporting and response process". It needs counsel review **[VERIFY]**.

## Banned uses

The banned-use list is section 4 of [TERMS_OF_USE.md](TERMS_OF_USE.md). Where possible, each item is enforced technically:

| Banned use | Technical control |
|---|---|
| Evidence, investigations, court and media use as real | Visible label, watermark, signed provenance, public verification, the `evidence_misuse` filter category, declared-use field |
| Real named people, locations or incidents | Filter categories `real_person`, `real_location` and `real_incident`; closed-vocabulary renderer with faceless figures |
| Impersonation, defamation, harassment | `harassment` filter category; KYC-verified customers only |
| Political or communal content | `political_communal` filter category |
| Sexual content or harm to minors | `sexual_content` and `minors` filter categories; the renderer has no such content |
| Removing the safeguards | `label_removal` filter category; watermark and provenance survive many edits; verification flags modified copies |

## Reporting channels

* The public form on `/verify`, or `POST /v1/misuse-reports`. No account is needed.
* Internal: anyone at VidyuthLabs who sees possible misuse files a report through the same channel, so it is logged.

## Response process

1. **Triage, within 1 business day [target; OPEN].** The compliance owner reviews new reports via `GET /v1/admin/misuse-reports`.
2. **Identify the clip.** Run the reported file through `POST /v1/admin/verify`. This gives the clip ID and customer even when metadata has been stripped, as long as the watermark survives. Then pull the audit trail with `GET /v1/admin/audit?customer_id=...`.
3. **Contain.** If misuse is likely, suspend the customer (`PATCH /v1/admin/customers/{id}` with `status=suspended`). This blocks generation immediately.
4. **Escalate.** Notify counsel. Where the IT Rules impose takedown or reporting timelines, follow them **[VERIFY exact timelines from the Gazette notification]**.
5. **Resolve.** Record the outcome (`PATCH /v1/admin/misuse-reports/{id}`). The outcome may be terminating the customer, dismissing the report, or improving a safeguard. Both the record and the outcome are in the hash-chained audit log.
6. **Learn.** The quarterly safeguards review (PRD section 11) covers all reports and any filter or watermark changes they triggered.

## Metrics (PRD section 9)

* Number of known misuse incidents. Target: zero.
* Time from misuse report to triage and to containment.
* Unlabelled outputs outside approved exceptions. This is zero by construction: the QA gate blocks release of any unlabelled, non-exception clip.
