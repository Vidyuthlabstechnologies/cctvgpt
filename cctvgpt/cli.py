"""Command-line interface for operators.

    cctvgpt init --admin NAME --owner NAME      bootstrap keys and the first admin/safeguards-owner API keys
    cctvgpt customer create|kyc|terms|update|key|list
    cctvgpt release selftest|approve|block|list
    cctvgpt generate --customer ID --prompt "..." [--set scene.weather=rain ...] [--out DIR]
    cctvgpt verify FILE [--full]
    cctvgpt audit verify|tail|purge
    cctvgpt serve [--host 127.0.0.1 --port 8000]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import config, release_gate
from .context import TERMS_VERSION, Platform


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _parse_sets(items) -> dict:
    payload: dict = {"scene": {}, "realism": {}}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        if key.startswith("scene.") or key.startswith("realism."):
            group, field = key.split(".", 1)
            payload[group][field] = value
        else:
            payload[key] = value
    return payload


def cmd_init(p: Platform, a) -> None:
    out = {"data_dir": str(p.settings.data_dir), "signing_key_id": p.keys.key_id}
    existing = p.store.q("SELECT role FROM api_keys WHERE revoked=0 AND role IN ('admin','safeguards_owner')")
    roles = {r["role"] for r in existing}
    if "admin" not in roles or a.force:
        out["admin_api_key"] = p.store.create_api_key("admin", a.admin, None, "cli:init")
    if "safeguards_owner" not in roles or a.force:
        out["safeguards_owner_api_key"] = p.store.create_api_key("safeguards_owner", a.owner, None, "cli:init")
    out["note"] = "API keys are shown once. The renderer must be approved by the safeguards owner before generation: `cctvgpt release approve --owner NAME`."
    _print(out)


def cmd_customer(p: Platform, a) -> None:
    st = p.store
    if a.action == "create":
        if not a.name or not a.email:
            raise SystemExit("customer create needs --name and --email")
        _print(st.create_customer(a.name, a.org_type, a.email, a.country, a.tier or "standard", actor=a.actor))
    elif a.action == "kyc":
        _print(st.set_kyc(a.id, a.status, a.reference, a.actor))
    elif a.action == "terms":
        _print(st.accept_terms(a.id, TERMS_VERSION, a.actor))
    elif a.action == "update":
        fields = {k: v for k, v in (("tier", a.tier), ("clean_export_contract", a.clean_export_contract), ("restricted_use_contract", a.restricted_use_contract), ("status", a.status)) if v is not None}
        _print(st.update_customer(a.id, a.actor, **fields))
    elif a.action == "key":
        _print({"api_key": st.create_api_key("customer", a.principal, a.id, a.actor)})
    elif a.action == "list":
        _print(st.q("SELECT id, name, org_type, kyc_status, tier, status, terms_accepted_version FROM customers ORDER BY created_at"))


def cmd_release(p: Platform, a) -> None:
    comp = (a.component, a.version or config.RENDERER_VERSION)
    if a.action == "selftest":
        res = release_gate.selftest(p.keys)
        _print(res)
        sys.exit(0 if res["passed"] else 1)
    elif a.action == "approve":
        res = release_gate.selftest(p.keys)
        if not res["passed"]:
            _print({"error": "self-test failed; release not approved", "selftest": res})
            sys.exit(1)
        _print(release_gate.decide(p.store, *comp, "approved", a.owner, a.notes, res))
    elif a.action == "block":
        _print(release_gate.decide(p.store, *comp, "blocked", a.owner, a.notes))
    elif a.action == "list":
        _print(p.store.q("SELECT component, version, status, decided_by, decided_at, notes FROM releases ORDER BY decided_at"))


def cmd_generate(p: Platform, a) -> None:
    from .pipeline import GenerationRefused, generate
    from .prompt import resolve_request

    payload = _parse_sets(a.set)
    payload["prompt"] = a.prompt or ""
    if a.use_case:
        payload["use_case"] = a.use_case
    req, interp = resolve_request(payload)
    try:
        res = generate(p, a.customer, f"cli:{a.actor}", req, interp)
    except GenerationRefused as e:
        _print({"refused": e.reason, "details": e.details})
        sys.exit(2)
    out = {"clip_id": res.clip_id, "video": str(res.video_path), "qa": res.qa, "interpretation": interp.to_dict()}
    if a.out:
        dest = Path(a.out)
        dest.mkdir(parents=True, exist_ok=True)
        for f in res.video_path.parent.iterdir():
            shutil.copy2(f, dest / f.name)
        out["copied_to"] = str(dest)
    _print(out)


def cmd_verify(p: Platform, a) -> None:
    from .verify import verify_file

    _print(verify_file(p, Path(a.file), full_details=a.full))


def cmd_audit(p: Platform, a) -> None:
    if a.action == "verify":
        _print(p.store.verify_audit_chain())
    elif a.action == "tail":
        rows = p.store.q("SELECT seq, ts, actor, customer_id, action, details_json FROM audit ORDER BY seq DESC LIMIT ?", (a.n,))
        _print(list(reversed(rows)))
    elif a.action == "purge":
        days = a.days or p.settings.audit_retention_days
        if not days:
            raise SystemExit("no retention period set (CCTVGPT_AUDIT_RETENTION_DAYS is [OPEN], to be set by counsel)")
        _print({"removed": p.store.purge_audit(days, "cli:purge")})


def cmd_evaluate(_p: Platform, a) -> None:
    from .evaluation import load_json, scorecard, scorecard_html

    anns = {}
    for f in sorted(Path(a.annotations).glob("**/*.annotations.json")):
        d = load_json(f)
        anns[d["clip_id"]] = d
    preds = load_json(a.predictions)
    card = scorecard(anns, preds)
    if a.html:
        Path(a.html).write_text(scorecard_html(card))
    _print(card["overall"])


def cmd_serve(_p: Platform, a) -> None:
    import uvicorn

    from .api.app import create_app

    uvicorn.run(create_app(_p), host=a.host, port=a.port)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cctvgpt", description="Synthetic CCTV-style video generation with built-in safeguards")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create keys and bootstrap admin / safeguards-owner API keys")
    s.add_argument("--admin", required=True, help="name of the platform admin")
    s.add_argument("--owner", required=True, help="name of the safeguards owner (SR-10)")
    s.add_argument("--force", action="store_true", help="issue new keys even if some exist")

    s = sub.add_parser("customer", help="manage customers and KYC (SR-5)")
    s.add_argument("action", choices=["create", "kyc", "terms", "update", "key", "list"])
    s.add_argument("--id")
    s.add_argument("--name")
    s.add_argument("--org-type", default="video_analytics")
    s.add_argument("--email")
    s.add_argument("--country", default="IN")
    s.add_argument("--tier", default=None)
    s.add_argument("--status", default=None)
    s.add_argument("--reference", help="KYC document reference (e.g. GSTIN / CIN)")
    s.add_argument("--clean-export-contract", default=None)
    s.add_argument("--restricted-use-contract", default=None)
    s.add_argument("--principal", default="user")
    s.add_argument("--actor", default="cli")

    s = sub.add_parser("release", help="release gate (SR-10)")
    s.add_argument("action", choices=["selftest", "approve", "block", "list"])
    s.add_argument("--component", default="renderer")
    s.add_argument("--version", default=None)
    s.add_argument("--owner", default="safeguards-owner")
    s.add_argument("--notes", default="")

    s = sub.add_parser("generate", help="generate one clip for a customer")
    s.add_argument("--customer", required=True)
    s.add_argument("--prompt", default="")
    s.add_argument("--use-case", default=None)
    s.add_argument("--set", action="append", help="override, e.g. scene.weather=rain realism.compression=high")
    s.add_argument("--out", help="copy outputs to this directory")
    s.add_argument("--actor", default="operator")

    s = sub.add_parser("verify", help="verify a clip (SR-4)")
    s.add_argument("file")
    s.add_argument("--full", action="store_true", help="include customer details (internal use)")

    s = sub.add_parser("audit", help="audit log (SR-7)")
    s.add_argument("action", choices=["verify", "tail", "purge"])
    s.add_argument("-n", type=int, default=20)
    s.add_argument("--days", type=int, default=None)

    s = sub.add_parser("evaluate", help="acceptance-test a detector's outputs against clip ground truth")
    s.add_argument("--annotations", required=True, help="directory containing *.annotations.json files")
    s.add_argument("--predictions", required=True, help="JSON file {clip_id: {frames: [...], events: [...]}}")
    s.add_argument("--html", help="also write an HTML scorecard here")

    s = sub.add_parser("serve", help="run the API and dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    return ap


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    p = Platform.open()
    {"init": cmd_init, "customer": cmd_customer, "release": cmd_release, "generate": cmd_generate,
     "verify": cmd_verify, "audit": cmd_audit, "evaluate": cmd_evaluate, "serve": cmd_serve}[a.cmd](p, a)


if __name__ == "__main__":
    main()
