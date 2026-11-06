"""Forensic search API (investigators and admins only).

Every query requires a case reference and a purpose and is written to the
hash-chained audit log, as is every ingest, media view and export.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..search.engine import IMAGE_EXT, VIDEO_EXT, SearchEngine

MAX_UPLOAD = 2 * 1024 * 1024 * 1024  # 2 GB per file
CASE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _./-]{1,63}$")


class IngestBody(BaseModel):
    path: str
    case_ref: str
    camera: str | None = None
    start_time: str | None = None
    every_s: float = Field(1.0, ge=0.2, le=30)


class QueryBody(BaseModel):
    text: str = Field(min_length=2, max_length=300)
    case_ref: str
    purpose: str = Field(min_length=3, max_length=300)
    k: int = Field(30, ge=1, le=200)
    classes: list[str] | None = None
    source_ids: list[str] | None = None
    include_synthetic: bool = False


class ExportBody(BaseModel):
    source_id: str
    t0: float = Field(ge=0)
    t1: float = Field(ge=0)
    case_ref: str


def _check_case(case_ref: str) -> None:
    if not CASE_RE.match(case_ref or ""):
        raise HTTPException(422, "case_ref must be 2-64 characters: letters, digits, space, _ . / -")


def register(app, platform, principal_dep, engine: SearchEngine | None = None) -> None:
    st = platform.store
    settings = platform.settings
    holder: dict = {"engine": engine}
    lock = threading.Lock()

    def eng() -> SearchEngine:
        with lock:
            if holder["engine"] is None:
                holder["engine"] = SearchEngine(settings.search_dir, st, platform.watermarker, platform.keys, variant=settings.search_model)
            return holder["engine"]

    def investigator(p=Depends(principal_dep)):
        if p.role not in ("investigator", "admin"):
            raise HTTPException(403, "investigator or admin key required")
        return p

    def _allowed(path: Path) -> bool:
        rp = path.resolve()
        return any(rp == root or root in rp.parents for root in settings.allowed_search_roots())

    @app.post("/v1/search/ingest")
    def ingest(body: IngestBody, p=Depends(investigator)):
        _check_case(body.case_ref)
        path = Path(body.path)
        if not path.exists():
            raise HTTPException(404, "path not found on the server")
        if not _allowed(path):
            raise HTTPException(403, "path is outside the configured evidence folders (CCTVGPT_SEARCH_ROOTS)")
        return eng().ingest(path, p.principal, body.case_ref, body.camera, body.start_time, body.every_s)

    @app.post("/v1/search/upload")
    async def upload(case_ref: str = Form(...), camera: str | None = Form(None), files: list[UploadFile] = File(...), p=Depends(investigator)):
        _check_case(case_ref)
        dest = settings.evidence_inbox / re.sub(r"[^A-Za-z0-9_.-]", "_", case_ref)
        dest.mkdir(parents=True, exist_ok=True)
        saved = []
        for f in files:
            name = Path(f.filename or "upload.bin").name
            if Path(name).suffix.lower() not in IMAGE_EXT | VIDEO_EXT:
                raise HTTPException(422, f"unsupported file type: {name}")
            target = dest / name
            n = 0
            with open(target, "wb") as out:
                while chunk := await f.read(1 << 20):
                    n += len(chunk)
                    if n > MAX_UPLOAD:
                        out.close()
                        target.unlink(missing_ok=True)
                        raise HTTPException(413, f"{name} is larger than 2 GB")
                    out.write(chunk)
            saved.append(target)
        summary = {"files": 0, "indexed": 0, "skipped_duplicates": 0, "synthetic_flagged": 0, "frames": 0, "objects": 0, "errors": []}
        for t in saved:
            r = eng().ingest(t, p.principal, case_ref, camera)
            for k in summary:
                summary[k] = summary[k] + r[k] if k != "errors" else summary[k] + r[k]
        return summary

    @app.get("/v1/search/sources")
    def sources(p=Depends(investigator)):
        return [{**s, "quality": json.loads(s.pop("quality_json") or "{}")} for s in eng().sources()]

    @app.post("/v1/search/query")
    def query(body: QueryBody, p=Depends(investigator)):
        _check_case(body.case_ref)
        try:
            return eng().search(p.principal, body.case_ref, body.purpose, text=body.text, k=body.k, classes=body.classes,
                                source_ids=body.source_ids, include_synthetic=body.include_synthetic)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.post("/v1/search/query-image")
    async def query_image(case_ref: str = Form(...), purpose: str = Form(...), text: str | None = Form(None), k: int = Form(30),
                          image: UploadFile = File(...), p=Depends(investigator)):
        _check_case(case_ref)
        data = await image.read(50 * 1024 * 1024)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(422, "could not read the query image")
        try:
            return eng().search(p.principal, case_ref, purpose, text=text or None, image=img, k=max(1, min(k, 200)))
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.get("/v1/search/thumb/{item_id}")
    def thumb(item_id: int, p=Depends(investigator)):
        path = eng().thumb_path(item_id)
        if path is None or not path.exists():
            raise HTTPException(404, "no thumbnail")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/v1/search/media/{source_id}")
    def media(source_id: str, case_ref: str = Query(...), p=Depends(investigator)):
        _check_case(case_ref)
        src = eng().source(source_id)
        if src is None:
            raise HTTPException(404, "unknown source")
        st.audit(p.principal, "search_media_viewed", {"case_ref": case_ref, "source_id": source_id})
        return FileResponse(src["path"])

    @app.post("/v1/search/export")
    def export(body: ExportBody, p=Depends(investigator)):
        _check_case(body.case_ref)
        if body.t1 < body.t0:
            raise HTTPException(422, "t1 must be >= t0")
        try:
            return eng().export_segment(body.source_id, body.t0, body.t1, p.principal, body.case_ref)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e

    @app.get("/v1/search/exports/{export_id}/download")
    def export_download(export_id: str, p=Depends(investigator)):
        rows = eng()._q("SELECT * FROM exports WHERE id=?", (export_id,))
        if not rows:
            raise HTTPException(404, "unknown export")
        st.audit(p.principal, "search_export_downloaded", {"export_id": export_id, "case_ref": rows[0]["case_ref"]})
        return FileResponse(rows[0]["path"], filename=Path(rows[0]["path"]).name)

    app.state.search_engine = holder
