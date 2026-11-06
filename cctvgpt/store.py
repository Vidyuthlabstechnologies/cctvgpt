"""Persistence (SQLite): customers & KYC (SR-5), API keys, clips, jobs,
hash-chained audit log (SR-7), misuse reports (SR-8) and release approvals
(SR-10).

SQLite keeps the Phase 1 build dependency-free. The access layer is narrow so
it can move to Postgres for Phase 2. Encryption at rest is expected from the
host volume [VERIFY with the infrastructure plan, section 6.3].
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    org_type TEXT,
    contact_email TEXT,
    country TEXT,
    kyc_status TEXT NOT NULL DEFAULT 'pending',   -- pending | verified | rejected
    kyc_reference TEXT,
    kyc_verified_by TEXT,
    kyc_verified_at TEXT,
    tier TEXT NOT NULL DEFAULT 'standard',          -- standard | enterprise
    terms_accepted_version TEXT,
    terms_accepted_at TEXT,
    clean_export_contract TEXT,                     -- SR-9 contract reference, enterprise only
    restricted_use_contract TEXT,                   -- public-safety training contract reference
    status TEXT NOT NULL DEFAULT 'active',          -- active | suspended | terminated
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    customer_id TEXT,
    role TEXT NOT NULL,                             -- customer | admin | safeguards_owner
    principal TEXT NOT NULL,
    key_hash TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL,
    revoked INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS clips (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    job_id TEXT,
    token INTEGER UNIQUE NOT NULL,
    created_at TEXT NOT NULL,
    request_json TEXT NOT NULL,
    manifest_json TEXT,
    stream_sha256 TEXT,
    file_sha256 TEXT,
    path TEXT,
    status TEXT NOT NULL,
    renderer_version TEXT,
    labelled INTEGER NOT NULL DEFAULT 1,
    qa_json TEXT
);
CREATE INDEX IF NOT EXISTS clips_stream ON clips(stream_sha256);
CREATE INDEX IF NOT EXISTS clips_file ON clips(file_sha256);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    principal TEXT NOT NULL,
    batch_id TEXT,
    status TEXT NOT NULL,                           -- queued | running | done | failed | blocked
    request_json TEXT NOT NULL,
    interpretation_json TEXT,
    clip_id TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_batch ON jobs(batch_id);
CREATE TABLE IF NOT EXISTS audit (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    customer_id TEXT,
    action TEXT NOT NULL,
    details_json TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    entry_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS misuse_reports (
    id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    reporter_contact TEXT,
    clip_reference TEXT,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    resolution TEXT
);
CREATE TABLE IF NOT EXISTS releases (
    id TEXT PRIMARY KEY,
    component TEXT NOT NULL,
    version TEXT NOT NULL,
    status TEXT NOT NULL,                           -- approved | blocked
    decided_by TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    notes TEXT,
    selftest_json TEXT
);
"""

GENESIS = "0" * 64


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    @contextmanager
    def tx(self):
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def q(self, sql: str, args=()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    def q1(self, sql: str, args=()) -> dict | None:
        rows = self.q(sql, args)
        return rows[0] if rows else None

    def x(self, sql: str, args=()) -> None:
        with self.tx() as c:
            c.execute(sql, args)

    # ------------------------------------------------------------------ audit (SR-7)
    def audit(self, actor: str, action: str, details: dict | None = None, customer_id: str | None = None) -> dict:
        details_json = json.dumps(details or {}, sort_keys=True, default=str)
        with self.tx() as c:
            row = c.execute("SELECT entry_hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row["entry_hash"] if row else GENESIS
            ts = now_iso()
            body = json.dumps([ts, actor, customer_id, action, details_json, prev], separators=(",", ":"))
            entry_hash = hashlib.sha256(body.encode()).hexdigest()
            c.execute(
                "INSERT INTO audit (ts, actor, customer_id, action, details_json, prev_hash, entry_hash) VALUES (?,?,?,?,?,?,?)",
                (ts, actor, customer_id, action, details_json, prev, entry_hash),
            )
        return {"ts": ts, "action": action, "entry_hash": entry_hash}

    def verify_audit_chain(self) -> dict:
        rows = self.q("SELECT * FROM audit ORDER BY seq")
        prev = rows[0]["prev_hash"] if rows else GENESIS
        for r in rows:
            if r["prev_hash"] != prev:
                return {"ok": False, "broken_at_seq": r["seq"], "reason": "prev_hash mismatch", "entries": len(rows)}
            body = json.dumps([r["ts"], r["actor"], r["customer_id"], r["action"], r["details_json"], r["prev_hash"]], separators=(",", ":"))
            if hashlib.sha256(body.encode()).hexdigest() != r["entry_hash"]:
                return {"ok": False, "broken_at_seq": r["seq"], "reason": "entry hash mismatch (entry altered)", "entries": len(rows)}
            prev = r["entry_hash"]
        return {"ok": True, "entries": len(rows), "head": prev}

    def purge_audit(self, retention_days: int, actor: str) -> int:
        """Delete entries older than the retention period. The chain stays verifiable
        from the oldest retained entry, and the purge itself is audited."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat().replace("+00:00", "Z")
        with self.tx() as c:
            n = c.execute("SELECT COUNT(*) AS n FROM audit WHERE ts < ?", (cutoff,)).fetchone()["n"]
            c.execute("DELETE FROM audit WHERE ts < ?", (cutoff,))
        self.audit(actor, "audit_purged", {"retention_days": retention_days, "entries_removed": n, "cutoff": cutoff})
        return n

    # ------------------------------------------------------------------ customers (SR-5)
    def create_customer(self, name: str, org_type: str, contact_email: str, country: str = "IN", tier: str = "standard", actor: str = "system") -> dict:
        cid = new_id("cus")
        self.x(
            "INSERT INTO customers (id, name, org_type, contact_email, country, tier, created_at) VALUES (?,?,?,?,?,?,?)",
            (cid, name, org_type, contact_email, country, tier, now_iso()),
        )
        self.audit(actor, "customer_created", {"name": name, "org_type": org_type, "tier": tier}, cid)
        return self.customer(cid)

    def customer(self, cid: str) -> dict | None:
        return self.q1("SELECT * FROM customers WHERE id=?", (cid,))

    def set_kyc(self, cid: str, status: str, reference: str, actor: str) -> dict:
        if status not in ("verified", "rejected", "pending"):
            raise ValueError("kyc status must be verified|rejected|pending")
        self.x("UPDATE customers SET kyc_status=?, kyc_reference=?, kyc_verified_by=?, kyc_verified_at=? WHERE id=?", (status, reference, actor, now_iso(), cid))
        self.audit(actor, "kyc_updated", {"status": status, "reference": reference}, cid)
        return self.customer(cid)

    def accept_terms(self, cid: str, version: str, actor: str) -> dict:
        self.x("UPDATE customers SET terms_accepted_version=?, terms_accepted_at=? WHERE id=?", (version, now_iso(), cid))
        self.audit(actor, "terms_accepted", {"version": version}, cid)
        return self.customer(cid)

    def update_customer(self, cid: str, actor: str, **fields) -> dict:
        allowed = {"tier", "clean_export_contract", "restricted_use_contract", "status", "name", "contact_email"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"cannot update {sorted(bad)}")
        for k, v in fields.items():
            self.x(f"UPDATE customers SET {k}=? WHERE id=?", (v, cid))
        self.audit(actor, "customer_updated", fields, cid)
        return self.customer(cid)

    # ------------------------------------------------------------------ API keys
    def create_api_key(self, role: str, principal: str, customer_id: str | None, actor: str) -> str:
        if role not in ("customer", "admin", "safeguards_owner"):
            raise ValueError("invalid role")
        if role == "customer" and not customer_id:
            raise ValueError("customer keys need a customer_id")
        raw = f"vl_{role[:3]}_{secrets.token_urlsafe(24)}"
        kid = new_id("key")
        self.x("INSERT INTO api_keys (id, customer_id, role, principal, key_hash, created_at) VALUES (?,?,?,?,?,?)", (kid, customer_id, role, principal, hash_key(raw), now_iso()))
        self.audit(actor, "api_key_created", {"key_id": kid, "role": role, "principal": principal}, customer_id)
        return raw

    def resolve_api_key(self, raw: str) -> dict | None:
        if not raw:
            return None
        return self.q1("SELECT * FROM api_keys WHERE key_hash=? AND revoked=0", (hash_key(raw),))

    def revoke_api_key(self, key_id: str, actor: str) -> None:
        self.x("UPDATE api_keys SET revoked=1 WHERE id=?", (key_id,))
        self.audit(actor, "api_key_revoked", {"key_id": key_id})

    # ------------------------------------------------------------------ jobs
    def create_job(self, customer_id: str, principal: str, request: dict, batch_id: str | None = None) -> str:
        jid = new_id("job")
        ts = now_iso()
        self.x(
            "INSERT INTO jobs (id, customer_id, principal, batch_id, status, request_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (jid, customer_id, principal, batch_id, "queued", json.dumps(request), ts, ts),
        )
        return jid

    def update_job(self, jid: str, **fields) -> None:
        fields["updated_at"] = now_iso()
        sets = ", ".join(f"{k}=?" for k in fields)
        self.x(f"UPDATE jobs SET {sets} WHERE id=?", (*fields.values(), jid))

    def job(self, jid: str) -> dict | None:
        return self.q1("SELECT * FROM jobs WHERE id=?", (jid,))

    # ------------------------------------------------------------------ clips
    def clip(self, clip_id: str) -> dict | None:
        return self.q1("SELECT * FROM clips WHERE id=?", (clip_id,))

    def clip_by_token(self, token: int) -> dict | None:
        return self.q1("SELECT * FROM clips WHERE token=?", (int(token),))

    def clip_by_stream_hash(self, h: str) -> dict | None:
        return self.q1("SELECT * FROM clips WHERE stream_sha256=?", (h,))

    def clip_by_file_hash(self, h: str) -> dict | None:
        return self.q1("SELECT * FROM clips WHERE file_sha256=?", (h,))

    def token_in_use(self, token: int) -> bool:
        return self.q1("SELECT 1 AS x FROM clips WHERE token=?", (int(token),)) is not None

    # ------------------------------------------------------------------ misuse reports (SR-8)
    def create_misuse_report(self, description: str, clip_reference: str | None, contact: str | None) -> str:
        rid = new_id("rpt")
        self.x("INSERT INTO misuse_reports (id, ts, reporter_contact, clip_reference, description) VALUES (?,?,?,?,?)", (rid, now_iso(), contact, clip_reference, description))
        self.audit("public", "misuse_report_received", {"report_id": rid, "clip_reference": clip_reference})
        return rid

    def resolve_misuse_report(self, rid: str, status: str, resolution: str, actor: str) -> None:
        self.x("UPDATE misuse_reports SET status=?, resolution=? WHERE id=?", (status, resolution, rid))
        self.audit(actor, "misuse_report_updated", {"report_id": rid, "status": status, "resolution": resolution})
