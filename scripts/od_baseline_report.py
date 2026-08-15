"""Stage-0 baseline: measure observed OD flows for every configured metro.

Gate G0 requires ``internal_share`` and the aux-sourced I-I share recorded for
**every** configured metro — the config set, not a hand-picked subset. This
script walks the config tree, assembles observed LODES OD per region, and writes
one row per metro plus a per-metro diagnostics JSON.

It needs only the LODES OD files (downloaded/cached via pygris) and the state
FIPS → abbreviation mapping, so it runs without the region's DB being populated.

Usage:
    python scripts/od_baseline_report.py
    python scripts/od_baseline_report.py --configs config/USA/TwinCities/config_twincities.json
    python scripts/od_baseline_report.py --out docs/od_matrix/baseline_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")

from data_sources.lodes_od import assemble_lodes_od, LodesODUnavailable  # noqa: E402

# State FIPS → USPS abbreviation. LODES URLs are keyed by abbreviation, and the
# usual lookup goes through the region DB; this table keeps the baseline
# runnable for metros whose DB has not been populated yet.
_STATE_ABBR: Dict[str, str] = {
    "01": "al", "02": "ak", "04": "az", "05": "ar", "06": "ca", "08": "co",
    "09": "ct", "10": "de", "11": "dc", "12": "fl", "13": "ga", "15": "hi",
    "16": "id", "17": "il", "18": "in", "19": "ia", "20": "ks", "21": "ky",
    "22": "la", "23": "me", "24": "md", "25": "ma", "26": "mi", "27": "mn",
    "28": "ms", "29": "mo", "30": "mt", "31": "ne", "32": "nv", "33": "nh",
    "34": "nj", "35": "nm", "36": "ny", "37": "nc", "38": "nd", "39": "oh",
    "40": "ok", "41": "or", "42": "pa", "44": "ri", "45": "sc", "46": "sd",
    "47": "tn", "48": "tx", "49": "ut", "50": "vt", "51": "va", "53": "wa",
    "54": "wv", "55": "wi", "56": "wy", "72": "pr",
}


def find_configs(root: Path) -> List[Path]:
    """Every metro config under config/USA/<Metro>/*.json."""
    return sorted(p for p in root.glob("*/*.json") if p.is_file())


def measure(config_path: Path, data_dir: Optional[str]) -> Dict[str, Any]:
    """Assemble observed OD for one metro and return its baseline row."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if data_dir:
        config.setdefault("data", {})["data_dir"] = data_dir

    counties = config.get("region", {}).get("counties", [])
    states = sorted({g[:2] for g in counties})
    metro = config_path.parent.name

    row: Dict[str, Any] = {
        "metro": metro,
        "config": str(config_path),
        "counties": len(counties),
        "states": [_STATE_ABBR.get(s, s).upper() for s in states],
        "multi_state": len(states) > 1,
    }

    mapping = {s: _STATE_ABBR[s] for s in states if s in _STATE_ABBR}
    unknown = [s for s in states if s not in _STATE_ABBR]
    if unknown:
        row["error"] = f"unknown state FIPS {unknown}"
        return row

    try:
        observed = assemble_lodes_od(config, state_abbr_mapping=mapping)
    except LodesODUnavailable as e:
        row["error"] = str(e)
        return row

    t = observed.totals
    row.update({
        "internal_ii": t.internal_ii,
        "outbound_ie": t.outbound_ie,
        "inbound_ei": t.inbound_ei,
        "internal_share": t.internal_share,
        "job_side_share": t.job_side_share,
        "two_sided_share": t.two_sided_share,
        "aux_sourced_ii": observed.aux_sourced_ii,
        "aux_sourced_ii_share": observed.aux_sourced_ii_share,
        "diagnostics": observed.as_diagnostics(),
    })
    return row


def build_markdown(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Stage 0 baseline — observed OD flows per metro",
        "",
        "Gate **G0**: `internal_share` and the aux-sourced I-I share, recorded for",
        "every configured metro. Measured from LODES OD (`main` + `aux` for every",
        "state a region touches, then filtered to the configured counties).",
        "",
        "`internal_share` is resident-side — `I-I / (I-I + I-E)` — the share of the",
        "region's employed residents who also work inside it. That is the quantity",
        "that explains the agent drop (E1): observed OD gives a work trip only to",
        "those working in-region, while the gravity matrix gives one to every",
        "employed resident.",
        "",
        "| metro | counties | states | I-I | I-E | E-I | internal_share | job-side | aux-sourced I-I | aux share |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in rows:
        if "error" in r:
            lines.append(
                f"| {r['metro']} | {r['counties']} | {'+'.join(r['states'])} | "
                f"— | — | — | — | — | — | _{r['error'][:60]}_ |"
            )
            continue
        lines.append(
            f"| {r['metro']} | {r['counties']} | {'+'.join(r['states'])} | "
            f"{r['internal_ii']:,} | {r['outbound_ie']:,} | {r['inbound_ei']:,} | "
            f"{r['internal_share']:.4f} | {r['job_side_share']:.4f} | "
            f"{r['aux_sourced_ii']:,} | {r['aux_sourced_ii_share']:.2%} |"
        )

    ok = [r for r in rows if "error" not in r]
    multi = [r for r in ok if r["multi_state"]]
    single = [r for r in ok if not r["multi_state"]]

    lines += ["", "## Observations", ""]

    if ok:
        lo = min(ok, key=lambda r: r["internal_share"])
        hi = max(ok, key=lambda r: r["internal_share"])
        lines.append(
            f"- `internal_share` spans **{lo['internal_share']:.1%} "
            f"({lo['metro']})** to **{hi['internal_share']:.1%} ({hi['metro']})** — "
            f"it varies widely between configs and must be reported per run, never assumed."
        )
    if multi:
        lines.append(
            "- **Multi-state regions and the `aux` rule (E4).** These regions have "
            "internal commutes filed under `aux` because they cross a state line "
            "inside the region. Building I-I from `main` alone would silently drop them:"
        )
        for r in sorted(multi, key=lambda r: -r["aux_sourced_ii_share"]):
            lines.append(
                f"  - **{r['metro']}** ({'+'.join(r['states'])}): "
                f"{r['aux_sourced_ii']:,} trips = **{r['aux_sourced_ii_share']:.2%}** of its I-I"
            )
    if single:
        names = ", ".join(r["metro"] for r in single)
        all_zero = all(r["aux_sourced_ii"] == 0 for r in single)
        lines.append(
            f"- Single-state regions ({names}) draw "
            f"{'no' if all_zero else 'some'} I-I from `aux`, as expected — the same "
            f"rule is simply a no-op for them, so no region-specific branching is needed."
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-root", type=Path, default=Path("config/USA"),
                    help="root holding <Metro>/<config>.json (default config/USA)")
    ap.add_argument("--configs", type=Path, nargs="*",
                    help="explicit config paths instead of walking the root")
    ap.add_argument("--data-dir", default="data",
                    help="override data.data_dir (pygris cache lives here)")
    ap.add_argument("--out", type=Path, help="write the markdown report here")
    ap.add_argument("--json-out", type=Path, help="write per-metro rows as JSON here")
    args = ap.parse_args()

    configs = args.configs or find_configs(args.config_root)
    if not configs:
        raise SystemExit(f"No configs found under {args.config_root}")

    rows = []
    for path in configs:
        print(f"=== {path} ===", file=sys.stderr)
        try:
            rows.append(measure(path, args.data_dir))
        except Exception as e:  # noqa: BLE001 - one bad metro must not stop the sweep
            print(f"  FAILED: {e!r}", file=sys.stderr)
            rows.append({"metro": path.parent.name, "config": str(path),
                         "counties": 0, "states": [], "multi_state": False,
                         "error": repr(e)})

    report = build_markdown(rows)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(report)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
