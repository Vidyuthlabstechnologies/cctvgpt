"""HTTP API and web dashboard (FR-6, SR-4, SR-5, SR-7, SR-8, SR-10).

Roles (API keys, header ``X-API-Key``):
  * customer          - a verified business customer's user; no anonymous generation (SR-5)
  * admin             - onboarding, KYC decisions, audit access, misuse handling
  * safeguards_owner  - release gate decisions (SR-10)
Public (no key): the verification page and API (SR-4), misuse reporting (SR-8),
terms of use, and the provenance public keys.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from .. import config, release_gate
from ..context import TERMS_VERSION, Platform
from ..jobs import JobQueue, batch_zip, job_view
from ..params import ParamError, option_schema
from ..pipeline import GenerationRefused, screen_prompt
from ..prompt import resolve_request
from ..verify import verify_upload

STATIC = Path(__file__).parent / "static"
DOCS = Path(__file__).resolve().parents[2] / "docs"
MAX_UPLOAD = 300 * 1024 * 1024


class Principal(BaseModel):
    key_id: str
    role: str
    principal: str
    customer_id: str | None = None


class GenerateBody(BaseModel):
    prompt: str = ""
    use_case: str = "analytics_testing"
    scene: dict = Field(default_factory=dict)
    realism: dict = Field(default_factory=dict)
    annotations: bool = True
    clean_export: bool = False


class BatchBody(BaseModel):
    base: GenerateBody
    sweep: dict = Field(default_factory=dict)


class EvaluateBody(BaseModel):
    predictions: dict = Field(description="{clip_id: {frames: [...], events: [...]}} in the format documented in cctvgpt/evaluation.py")


class InterpretBody(BaseModel):
    prompt: str


class MisuseBody(BaseModel):
    description: str = Field(min_length=10, max_length=5000)
    clip_reference: str | None = Field(default=None, max_length=200)
    contact: str | None = Field(default=None, max_length=200)


class CustomerBody(BaseModel):
    name: str
    org_type: str
    contact_email: str
    country: str = "IN"
    tier: str = "standard"


class KycBody(BaseModel):
    status: str
    reference: str


class CustomerPatch(BaseModel):
    tier: str | None = None
    clean_export_contract: str | None = None
    restricted_use_contract: str | None = None
    status: str | None = None


class KeyBody(BaseModel):
    principal: str


class TermsBody(BaseModel):
    version: str


class ReleaseBody(BaseModel):
    component: str = "renderer"
    version: str = config.RENDERER_VERSION
    decision: str
    notes: str = ""


class MisusePatch(BaseModel):
    status: str
    resolution: str


def create_app(platform: Platform | None = None, workers: int | None = None, search_engine=None) -> FastAPI:
    platform = platform or Platform.open()
    queue = JobQueue(platform, workers)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        queue.shutdown()

    app = FastAPI(title="CCTVGPT - Synthetic CCTV-style video", version=config.PACKAGE_VERSION, lifespan=lifespan)
    app.state.platform = platform
    app.state.queue = queue
    st = platform.store

    # ------------------------------------------------------------------ auth
    def principal(x_api_key: str | None = Header(default=None)) -> Principal:
        row = st.resolve_api_key(x_api_key or "")
        if row is None:
            raise HTTPException(401, "a valid X-API-Key is required; anonymous use is not permitted")
        return Principal(key_id=row["id"], role=row["role"], principal=row["principal"], customer_id=row["customer_id"])

    def customer_user(p: Principal = Depends(principal)) -> Principal:
        if p.role != "customer":
            raise HTTPException(403, "customer key required")
        return p

    def admin_user(p: Principal = Depends(principal)) -> Principal:
        if p.role != "admin":
            raise HTTPException(403, "admin key required")
        return p

    def owner_user(p: Principal = Depends(principal)) -> Principal:
        if p.role != "safeguards_owner":
            raise HTTPException(403, "safeguards owner key required (SR-10)")
        return p

    @app.exception_handler(ParamError)
    async def _param_error(_: Request, exc: ParamError):
        return JSONResponse({"error": "invalid_parameters", "detail": str(exc)}, status_code=422)

    @app.exception_handler(GenerationRefused)
    async def _refused(_: Request, exc: GenerationRefused):
        return JSONResponse({"error": "refused", "reason": exc.reason, "details": exc.details}, status_code=403)

    # ------------------------------------------------------------------ public pages
    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return (STATIC / "index.html").read_text()

    @app.get("/search", response_class=HTMLResponse)
    def search_page():
        return (STATIC / "search.html").read_text()

    from .search_routes import register as register_search

    register_search(app, platform, principal, search_engine)

    @app.get("/verify", response_class=HTMLResponse)
    def verify_page():
        return (STATIC / "verify.html").read_text()

    @app.get("/v1/health")
    def health():
        return {"ok": True, "version": config.PACKAGE_VERSION, "renderer": config.RENDERER_VERSION,
                "renderer_approved": release_gate.is_approved(st, *release_gate.renderer_component())}

    @app.get("/v1/public-key")
    def public_key():
        return platform.keys.public_info()

    @app.get("/v1/terms")
    def terms():
        p = DOCS / "TERMS_OF_USE.md"
        return {"version": TERMS_VERSION, "text": p.read_text() if p.exists() else "See docs/TERMS_OF_USE.md"}

    @app.post("/v1/verify")
    async def verify_public(file: UploadFile = File(...)):
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "file too large")
        report = verify_upload(platform, data, file.filename or "upload.mp4", full_details=False)
        st.audit("public", "verification_performed", {"verdict": report["verdict"], "clip_id": (report.get("clip") or {}).get("clip_id")})
        return report

    @app.post("/v1/misuse-reports", status_code=201)
    def misuse_report(body: MisuseBody):
        rid = st.create_misuse_report(body.description, body.clip_reference, body.contact)
        return {"report_id": rid, "message": "Thank you. The compliance team will review this report."}

    # ------------------------------------------------------------------ customer
    @app.get("/v1/me")
    def me(p: Principal = Depends(principal)):
        out = {"role": p.role, "principal": p.principal, "terms_version_current": TERMS_VERSION}
        if p.customer_id:
            c = st.customer(p.customer_id)
            out["customer"] = {k: c[k] for k in ("id", "name", "org_type", "kyc_status", "tier", "status", "terms_accepted_version")}
            out["customer"]["clean_export_available"] = bool(platform.settings.clean_export_enabled and c["clean_export_contract"])
            out["customer"]["restricted_use_available"] = bool(c["restricted_use_contract"])
        return out

    @app.post("/v1/me/accept-terms")
    def accept_terms(body: TermsBody, p: Principal = Depends(customer_user)):
        if body.version != TERMS_VERSION:
            raise HTTPException(409, f"current terms version is {TERMS_VERSION}")
        st.accept_terms(p.customer_id, body.version, p.principal)
        return {"accepted": body.version}

    @app.get("/v1/options")
    def options():
        from ..render.events import EVENT_BUILDERS

        return {**option_schema(), "event_library": sorted(EVENT_BUILDERS)}

    @app.post("/v1/interpret")
    def interpret(body: InterpretBody, p: Principal = Depends(customer_user)):
        req, interp = resolve_request({"prompt": body.prompt})
        return {"interpretation": interp.to_dict(), "filter": screen_prompt(req), "resolved_scene": req.scene.__dict__}

    @app.post("/v1/generate", status_code=202)
    def generate(body: GenerateBody, wait: bool = Query(False), p: Principal = Depends(customer_user)):
        payload = body.model_dump()
        if wait:
            job = queue.run_sync(p.customer_id, p.principal, payload)
            return JSONResponse(job_view(platform, job), status_code=200 if job["status"] == "done" else 403 if job["status"] == "blocked" else 500)
        jid = queue.submit(p.customer_id, p.principal, payload)
        return job_view(platform, st.job(jid))

    @app.post("/v1/batches", status_code=202)
    def batch(body: BatchBody, p: Principal = Depends(customer_user)):
        batch_id, jids = queue.submit_batch(p.customer_id, p.principal, body.base.model_dump(), body.sweep)
        return {"batch_id": batch_id, "jobs": jids}

    @app.get("/v1/batches/{batch_id}")
    def batch_status(batch_id: str, p: Principal = Depends(customer_user)):
        jobs = st.q("SELECT * FROM jobs WHERE batch_id=? AND customer_id=? ORDER BY created_at", (batch_id, p.customer_id))
        if not jobs:
            raise HTTPException(404, "batch not found")
        counts: dict = {}
        for j in jobs:
            counts[j["status"]] = counts.get(j["status"], 0) + 1
        return {"batch_id": batch_id, "counts": counts, "jobs": [job_view(platform, j) for j in jobs]}

    @app.get("/v1/batches/{batch_id}/export.zip")
    def batch_export(batch_id: str, p: Principal = Depends(customer_user)):
        data = batch_zip(platform, batch_id, p.customer_id)
        st.audit(p.principal, "batch_exported", {"batch_id": batch_id, "bytes": len(data)}, p.customer_id)
        return Response(data, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{batch_id}.zip"'})

    @app.post("/v1/batches/{batch_id}/evaluate")
    def batch_evaluate(batch_id: str, body: EvaluateBody, format: str = Query("json"), p: Principal = Depends(customer_user)):
        """Acceptance testing: score the customer's detector outputs against the batch's exact ground truth."""
        from ..evaluation import scorecard, scorecard_html

        jobs = st.q("SELECT clip_id FROM jobs WHERE batch_id=? AND customer_id=? AND status='done'", (batch_id, p.customer_id))
        if not jobs:
            raise HTTPException(404, "no finished clips in this batch")
        anns = {}
        for j in jobs:
            clip = st.clip(j["clip_id"])
            path = Path(clip["path"]).parent / f"{clip['id']}.annotations.json"
            if path.exists():
                anns[clip["id"]] = json.loads(path.read_text())
        unknown = sorted(set(body.predictions) - set(anns))
        if unknown:
            raise HTTPException(422, f"predictions for clips not in this batch: {unknown[:5]}")
        card = scorecard(anns, body.predictions)
        st.audit(p.principal, "evaluation_scored", {"batch_id": batch_id, "clips": len(anns), "missing": len(card["missing_predictions"])}, p.customer_id)
        if format == "html":
            return HTMLResponse(scorecard_html(card, f"Acceptance test - batch {batch_id}"))
        return card

    @app.get("/v1/jobs")
    def list_jobs(limit: int = Query(50, le=500), p: Principal = Depends(customer_user)):
        jobs = st.q("SELECT * FROM jobs WHERE customer_id=? ORDER BY created_at DESC LIMIT ?", (p.customer_id, limit))
        return [job_view(platform, j) for j in jobs]

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str, p: Principal = Depends(customer_user)):
        job = st.job(job_id)
        if not job or job["customer_id"] != p.customer_id:
            raise HTTPException(404, "job not found")
        return job_view(platform, job)

    def _owned_clip(clip_id: str, p: Principal) -> dict:
        clip = st.clip(clip_id)
        if not clip or clip["customer_id"] != p.customer_id or clip["status"] != "ready":
            raise HTTPException(404, "clip not found")
        return clip

    @app.get("/v1/clips/{clip_id}/video")
    def clip_video(clip_id: str, p: Principal = Depends(customer_user)):
        clip = _owned_clip(clip_id, p)
        st.audit(p.principal, "clip_downloaded", {"clip_id": clip_id}, p.customer_id)
        return FileResponse(clip["path"], media_type="video/mp4", filename=Path(clip["path"]).name)

    def _sidecar(clip: dict, suffix: str) -> Path:
        path = Path(clip["path"]).parent / f"{clip['id']}{suffix}"
        if not path.exists():
            raise HTTPException(404, "not available for this clip")
        return path

    @app.get("/v1/clips/{clip_id}/annotations")
    def clip_annotations(clip_id: str, p: Principal = Depends(customer_user)):
        return json.loads(_sidecar(_owned_clip(clip_id, p), ".annotations.json").read_text())

    @app.get("/v1/clips/{clip_id}/tracks")
    def clip_tracks(clip_id: str, p: Principal = Depends(customer_user)):
        return FileResponse(_sidecar(_owned_clip(clip_id, p), ".tracks.csv"), media_type="text/csv")

    @app.get("/v1/clips/{clip_id}/provenance")
    def clip_provenance(clip_id: str, p: Principal = Depends(customer_user)):
        return json.loads(_sidecar(_owned_clip(clip_id, p), ".provenance.json").read_text())

    # ------------------------------------------------------------------ admin
    @app.post("/v1/admin/customers", status_code=201)
    def admin_create_customer(body: CustomerBody, p: Principal = Depends(admin_user)):
        if body.tier not in ("standard", "enterprise"):
            raise HTTPException(422, "tier must be standard or enterprise")
        return st.create_customer(body.name, body.org_type, body.contact_email, body.country, body.tier, actor=p.principal)

    @app.get("/v1/admin/customers")
    def admin_list_customers(p: Principal = Depends(admin_user)):
        return st.q("SELECT * FROM customers ORDER BY created_at DESC")

    @app.post("/v1/admin/customers/{cid}/kyc")
    def admin_kyc(cid: str, body: KycBody, p: Principal = Depends(admin_user)):
        if not st.customer(cid):
            raise HTTPException(404, "customer not found")
        try:
            return st.set_kyc(cid, body.status, body.reference, p.principal)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.patch("/v1/admin/customers/{cid}")
    def admin_patch_customer(cid: str, body: CustomerPatch, p: Principal = Depends(admin_user)):
        if not st.customer(cid):
            raise HTTPException(404, "customer not found")
        fields = {k: v for k, v in body.model_dump().items() if v is not None}
        if fields.get("tier") not in (None, "standard", "enterprise"):
            raise HTTPException(422, "tier must be standard or enterprise")
        if fields.get("status") not in (None, "active", "suspended", "terminated"):
            raise HTTPException(422, "status must be active, suspended or terminated")
        return st.update_customer(cid, p.principal, **fields)

    @app.post("/v1/admin/customers/{cid}/api-keys", status_code=201)
    def admin_create_key(cid: str, body: KeyBody, p: Principal = Depends(admin_user)):
        if not st.customer(cid):
            raise HTTPException(404, "customer not found")
        return {"api_key": st.create_api_key("customer", body.principal, cid, p.principal), "note": "Shown once. Store it securely."}

    @app.get("/v1/admin/audit")
    def admin_audit(limit: int = Query(100, le=2000), customer_id: str | None = None, p: Principal = Depends(admin_user)):
        if customer_id:
            rows = st.q("SELECT * FROM audit WHERE customer_id=? ORDER BY seq DESC LIMIT ?", (customer_id, limit))
        else:
            rows = st.q("SELECT * FROM audit ORDER BY seq DESC LIMIT ?", (limit,))
        for r in rows:
            r["details"] = json.loads(r.pop("details_json"))
        return rows

    @app.get("/v1/admin/audit/verify")
    def admin_audit_verify(p: Principal = Depends(admin_user)):
        return st.verify_audit_chain()

    @app.get("/v1/admin/misuse-reports")
    def admin_misuse(p: Principal = Depends(admin_user)):
        return st.q("SELECT * FROM misuse_reports ORDER BY ts DESC")

    @app.patch("/v1/admin/misuse-reports/{rid}")
    def admin_misuse_update(rid: str, body: MisusePatch, p: Principal = Depends(admin_user)):
        st.resolve_misuse_report(rid, body.status, body.resolution, p.principal)
        return st.q1("SELECT * FROM misuse_reports WHERE id=?", (rid,))

    @app.post("/v1/admin/verify")
    async def admin_verify(file: UploadFile = File(...), p: Principal = Depends(admin_user)):
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "file too large")
        return verify_upload(platform, data, file.filename or "upload.mp4", full_details=True)

    # ------------------------------------------------------------------ release gate
    @app.get("/v1/releases")
    def releases(p: Principal = Depends(principal)):
        if p.role not in ("admin", "safeguards_owner"):
            raise HTTPException(403, "admin or safeguards owner key required")
        rows = st.q("SELECT id, component, version, status, decided_by, decided_at, notes FROM releases ORDER BY decided_at DESC")
        return {"current_renderer": config.RENDERER_VERSION, "decisions": rows}

    @app.post("/v1/releases/selftest")
    def run_selftest(p: Principal = Depends(owner_user)):
        res = release_gate.selftest(platform.keys)
        st.audit(p.principal, "selftest_run", {"passed": res["passed"], "checks": res["checks"]})
        return res

    @app.post("/v1/releases/decide")
    def decide(body: ReleaseBody, p: Principal = Depends(owner_user)):
        selftest = release_gate.selftest(platform.keys) if body.decision == "approved" else None
        try:
            return release_gate.decide(st, body.component, body.version, body.decision, p.principal, body.notes, selftest)
        except (release_gate.ReleaseBlocked, ValueError) as e:
            raise HTTPException(409, str(e)) from e

    return app


def app_factory() -> FastAPI:  # for `uvicorn cctvgpt.api.app:app_factory --factory`
    return create_app()
