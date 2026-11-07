"""Queue-based generation (FR-6, NFR scalability).

Jobs are persisted in the store and executed by a worker pool. Batch requests
expand a parameter sweep (e.g. weather x time of day x seed) into one job per
combination, which is how analytics teams test a detector across controlled
conditions (use case 2).
"""

from __future__ import annotations

import io
import itertools
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .context import Platform
from .params import GenerationRequest, ParamError, RealismParams, SceneParams
from .pipeline import GenerationRefused, generate
from .prompt import resolve_request
from .store import new_id

_SCENE_FIELDS = set(SceneParams.__dataclass_fields__)
_REALISM_FIELDS = set(RealismParams.__dataclass_fields__)


def expand_sweep(base: dict, sweep: dict, limit: int) -> list[dict]:
    """Cartesian product of sweep values applied on top of the base request payload."""
    if not sweep:
        return [base]
    keys = list(sweep)
    for k in keys:
        if k not in _SCENE_FIELDS and k not in _REALISM_FIELDS:
            raise ParamError(f"cannot sweep unknown field {k!r}")
        if not isinstance(sweep[k], list) or not sweep[k]:
            raise ParamError(f"sweep[{k!r}] must be a non-empty list")
    combos = list(itertools.product(*(sweep[k] for k in keys)))
    if len(combos) > limit:
        raise ParamError(f"sweep expands to {len(combos)} clips; the limit is {limit}")
    out = []
    for combo in combos:
        p = json.loads(json.dumps(base))
        p.setdefault("scene", {})
        p.setdefault("realism", {})
        for k, v in zip(keys, combo):
            (p["scene"] if k in _SCENE_FIELDS else p["realism"])[k] = v
        out.append(p)
    return out


class JobQueue:
    def __init__(self, platform: Platform, workers: int | None = None):
        self.platform = platform
        self.pool = ThreadPoolExecutor(max_workers=workers or platform.settings.worker_threads, thread_name_prefix="gen")
        # Re-queue anything left over from a previous run.
        for j in platform.store.q("SELECT id FROM jobs WHERE status IN ('queued','running')"):
            platform.store.update_job(j["id"], status="queued")
            self.pool.submit(self._run, j["id"])

    def submit(self, customer_id: str, principal: str, payload: dict, batch_id: str | None = None) -> str:
        # Validate eagerly so callers get parameter errors immediately.
        req, interp = resolve_request(payload)
        req.validate(self.platform.settings.max_duration_s)
        jid = self.platform.store.create_job(customer_id, principal, payload, batch_id)
        self.platform.store.update_job(jid, interpretation_json=json.dumps(interp.to_dict()))
        self.pool.submit(self._run, jid)
        return jid

    def submit_batch(self, customer_id: str, principal: str, base: dict, sweep: dict) -> tuple[str, list[str]]:
        payloads = expand_sweep(base, sweep, self.platform.settings.max_batch_size)
        for p in payloads:  # validate all before queueing any
            req, _ = resolve_request(p)
            req.validate(self.platform.settings.max_duration_s)
        batch_id = new_id("bat")
        self.platform.store.audit(principal, "batch_submitted", {"batch_id": batch_id, "size": len(payloads), "sweep": sweep}, customer_id)
        return batch_id, [self.submit(customer_id, principal, p, batch_id) for p in payloads]

    def run_sync(self, customer_id: str, principal: str, payload: dict) -> dict:
        jid = self.platform.store.create_job(customer_id, principal, payload)
        self._run(jid)
        return self.platform.store.job(jid)

    def _run(self, jid: str) -> None:
        st = self.platform.store
        job = st.job(jid)
        if job is None:
            return
        st.update_job(jid, status="running")
        try:
            req, interp = resolve_request(json.loads(job["request_json"]))
            st.update_job(jid, interpretation_json=json.dumps(interp.to_dict()))
            res = generate(self.platform, job["customer_id"], job["principal"], req, interp, job_id=jid)
            st.update_job(jid, status="done", clip_id=res.clip_id)
        except GenerationRefused as e:
            st.update_job(jid, status="blocked", error=json.dumps({"reason": e.reason, "details": e.details}))
        except Exception as e:  # noqa: BLE001 - surface any failure on the job record
            st.update_job(jid, status="failed", error=json.dumps({"reason": type(e).__name__, "details": str(e)}))

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True)


def job_view(platform: Platform, job: dict) -> dict:
    out = {k: job[k] for k in ("id", "status", "batch_id", "clip_id", "created_at", "updated_at")}
    out["request"] = json.loads(job["request_json"])
    out["interpretation"] = json.loads(job["interpretation_json"]) if job.get("interpretation_json") else None
    out["error"] = json.loads(job["error"]) if job.get("error") else None
    if job.get("clip_id"):
        clip = platform.store.clip(job["clip_id"])
        if clip:
            out["qa"] = json.loads(clip["qa_json"]) if clip.get("qa_json") else None
            out["links"] = {
                "video": f"/v1/clips/{clip['id']}/video",
                "annotations": f"/v1/clips/{clip['id']}/annotations",
                "tracks": f"/v1/clips/{clip['id']}/tracks",
                "provenance": f"/v1/clips/{clip['id']}/provenance",
            }
    return out


def clip_files(clip: dict) -> list[Path]:
    if not clip.get("path"):
        return []
    video = Path(clip["path"])
    d = video.parent
    return [p for p in (video, d / f"{clip['id']}.provenance.json", d / f"{clip['id']}.annotations.json", d / f"{clip['id']}.tracks.csv") if p.exists()]


def batch_zip(platform: Platform, batch_id: str, customer_id: str) -> bytes:
    jobs = platform.store.q("SELECT * FROM jobs WHERE batch_id=? AND customer_id=?", (batch_id, customer_id))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        index = []
        for j in jobs:
            entry = {"job_id": j["id"], "status": j["status"], "clip_id": j["clip_id"], "request": json.loads(j["request_json"])}
            if j["clip_id"]:
                clip = platform.store.clip(j["clip_id"])
                for f in clip_files(clip or {}):
                    z.write(f, f"{j['clip_id']}/{f.name}")
            index.append(entry)
        z.writestr("batch_index.json", json.dumps({"batch_id": batch_id, "jobs": index, "notice": "All clips are synthetic. See provenance files."}, indent=2))
        z.writestr("README_SYNTHETIC.txt", "Every video in this archive is SYNTHETIC, AI-generated simulation footage.\nIt must not be used or presented as evidence or as a record of real events.\nVerify any clip at /verify.\n")
    return buf.getvalue()


__all__ = ["JobQueue", "expand_sweep", "job_view", "batch_zip", "clip_files", "GenerationRequest"]
