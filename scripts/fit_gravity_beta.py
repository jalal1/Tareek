"""Stage 4 — fit the gravity model's distance-decay beta against observed flows.

The gravity model is the fallback for regions LODES does not cover, so it has to
be defensible on its own. Today beta is a guess (1.5) applied to a functional
form that is unbounded near zero. This script fits beta per metro for each
bounded form and reports whether a single constant transfers across regions
(open question Q6).

Fitting target: the observed trip-length distribution. Beta controls decay, so
the honest objective is to match how far people actually travel — median trip
length and intrazonal share — rather than cell-by-cell correlation, which is
dominated by the margins IPF already enforces.

Usage:
    python scripts/fit_gravity_beta.py --config config/USA/TwinCities/config_twincities.json
    python scripts/fit_gravity_beta.py --all --out docs/od_matrix/stage4_beta_fit.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")

from data_sources.lodes_od import (  # noqa: E402
    LodesODUnavailable, assemble_lodes_od, flows_to_zone_matrix,
)
from models.od_matrix_v3 import (  # noqa: E402
    FRICTION_EXPONENTIAL, FRICTION_GAMMA, FRICTION_POWER, compute_friction,
)
from utils.od_diagnostics import cosine_corrected_km, zone_sqrt_area_km  # noqa: E402


def load_zones(config: Dict[str, Any], geo_level: str = "block_group"):
    """Zone-level residents, jobs, centroids and extents from the region DB."""
    from models.home_locs_v2 import load_home_locations_by_counties
    from models.work_locs_v2 import load_work_locations_by_counties

    prefix = 11 if geo_level == "tract" else 12
    homes = load_home_locations_by_counties(config)
    works = load_work_locations_by_counties(config)

    def agg(blocks, emp_key="n_employees"):
        acc: Dict[str, Dict[str, Any]] = {}
        for bid, d in blocks.items():
            if d.get("lat") is None or d.get("lon") is None:
                continue
            z = bid[:prefix]
            e = acc.setdefault(z, {"n": 0.0, "pts": []})
            e["n"] += float(d.get(emp_key) or 0)
            e["pts"].append((d["lon"], d["lat"]))
        return acc

    h, w = agg(homes), agg(works)
    home_zones = sorted(h)
    work_zones = sorted(w)

    Oi = np.array([h[z]["n"] for z in home_zones], dtype=float)
    Dj = np.array([w[z]["n"] for z in work_zones], dtype=float)
    hc = np.array([np.mean(h[z]["pts"], axis=0) for z in home_zones], dtype=float)
    wc = np.array([np.mean(w[z]["pts"], axis=0) for z in work_zones], dtype=float)
    sqrt_area = zone_sqrt_area_km({z: h[z]["pts"] for z in home_zones})

    return home_zones, work_zones, Oi, Dj, hc, wc, sqrt_area


def build_distances(home_zones, work_zones, hc, wc, sqrt_area,
                    intrazonal_factor: float = 0.5) -> np.ndarray:
    """Stage-1 geometry: cosine-corrected km with area-based intrazonal cells."""
    d = cosine_corrected_km(hc, wc)
    widx = {z: j for j, z in enumerate(work_zones)}
    for i, z in enumerate(home_zones):
        j = widx.get(z)
        if j is not None and z in sqrt_area:
            d[i, j] = intrazonal_factor * sqrt_area[z]
    return np.maximum(d, 0.05)


def run_ipf(friction: np.ndarray, Oi: np.ndarray, Dj: np.ndarray,
            max_iterations: int = 200, threshold: float = 0.03) -> np.ndarray:
    """Doubly-constrained IPF, matching the production routine."""
    Dj = Dj * (Oi.sum() / Dj.sum()) if Dj.sum() else Dj
    M = friction.copy()
    for _ in range(max_iterations):
        rs = M.sum(axis=1, keepdims=True)
        M = M * np.divide(Oi.reshape(-1, 1), rs, out=np.ones_like(rs), where=rs != 0)
        cs = M.sum(axis=0, keepdims=True)
        M = M * np.divide(Dj.reshape(1, -1), cs, out=np.ones_like(cs), where=cs != 0)
        if max((np.abs(M.sum(1) - Oi) / (Oi + 1e-10)).max(),
               (np.abs(M.sum(0) - Dj) / (Dj + 1e-10)).max()) < threshold:
            break
    return M


def trip_metrics(M: np.ndarray, d: np.ndarray, diag: np.ndarray) -> Dict[str, float]:
    """Trip-weighted length percentiles and intrazonal share."""
    w = M.ravel()
    dd = d.ravel()
    total = w.sum()
    if total <= 0:
        return {}
    order = np.argsort(dd)
    cum = np.cumsum(w[order]) / total
    return {
        "mean_km": float((dd * w).sum() / total),
        "median_km": float(dd[order][np.searchsorted(cum, 0.5)]),
        "p25_km": float(dd[order][np.searchsorted(cum, 0.25)]),
        "p75_km": float(dd[order][np.searchsorted(cum, 0.75)]),
        "intrazonal_share": float(M[diag[:, 0], diag[:, 1]].sum() / total),
    }


def objective(model: Dict[str, float], target: Dict[str, float]) -> float:
    """Distance between a candidate's trip-length profile and the observed one.

    Median and intrazonal share are weighted equally: the median captures
    whether trips are long enough, the intrazonal share whether the model
    concentrates them on the diagonal. Beta that fixes one and breaks the other
    is not a fit — that failure is exactly why the power law is being replaced.
    """
    if not model or not target:
        return float("inf")
    med = abs(model["median_km"] - target["median_km"]) / max(target["median_km"], 1e-9)
    intra = abs(model["intrazonal_share"] - target["intrazonal_share"])
    return med + 5.0 * intra


def fit_metro(config_path: Path, data_dir: Optional[str],
              forms: Tuple[str, ...]) -> Dict[str, Any]:
    """Fit beta for one metro against its observed LODES flows."""
    config = json.load(open(config_path, encoding="utf-8"))
    if data_dir:
        config.setdefault("data", {})["data_dir"] = data_dir

    metro = config_path.parent.name
    out: Dict[str, Any] = {"metro": metro, "config": str(config_path)}

    try:
        observed = assemble_lodes_od(config)
    except LodesODUnavailable as e:
        out["error"] = str(e)
        return out

    home_zones, work_zones, Oi, Dj, hc, wc, sqrt_area = load_zones(config)
    d = build_distances(home_zones, work_zones, hc, wc, sqrt_area)

    widx = {z: j for j, z in enumerate(work_zones)}
    diag = np.array([[i, widx[z]] for i, z in enumerate(home_zones) if z in widx])

    obs_matrix = flows_to_zone_matrix(observed.internal_flows, home_zones, work_zones)
    target = trip_metrics(obs_matrix.to_numpy(dtype=float), d, diag)
    out["observed"] = target

    results = {}
    for form in forms:
        # Power betas are dimensionless and small; bounded betas are 1/km.
        grid = (np.arange(0.5, 2.6, 0.1) if form == FRICTION_POWER
                else np.arange(0.02, 0.42, 0.01))
        best = None
        curve = []
        for beta in grid:
            M = run_ipf(compute_friction(d, float(beta), form=form), Oi, Dj)
            m = trip_metrics(M, d, diag)
            score = objective(m, target)
            curve.append({"beta": round(float(beta), 3), "score": round(score, 4), **{
                k: round(v, 4) for k, v in m.items()}})
            if best is None or score < best["score"]:
                best = {"beta": round(float(beta), 3), "score": round(score, 4), **m}
        results[form] = {"best": best, "curve": curve}
        if best:
            print(f"  {metro:14s} {form:12s} beta={best['beta']:<6.2f} "
                  f"median {best['median_km']:6.2f} (obs {target['median_km']:6.2f})  "
                  f"intra {best['intrazonal_share']:6.2%} (obs {target['intrazonal_share']:6.2%})",
                  file=sys.stderr)

    out["fits"] = results
    return out


def build_report(rows: List[Dict[str, Any]], forms: Tuple[str, ...]) -> str:
    lines = [
        "# Stage 4 — gravity beta fit",
        "",
        "Beta fitted per metro against observed LODES trip lengths, on the",
        "stage-1 geometry (cosine-corrected km, area-based intrazonal distance).",
        "",
        "Objective: match the observed median trip length and intrazonal share.",
        "Cell correlation is not used as the target because IPF already pins the",
        "margins — what beta actually controls is how far trips travel.",
        "",
    ]

    for form in forms:
        lines += [
            f"## {form}",
            "",
            "| metro | beta | median km (fit / obs) | intrazonal (fit / obs) | score |",
            "|---|---:|---|---|---:|",
        ]
        for r in rows:
            if "error" in r or form not in r.get("fits", {}):
                lines.append(f"| {r['metro']} | — | — | — | _{r.get('error','no fit')[:40]}_ |")
                continue
            b = r["fits"][form]["best"]
            o = r["observed"]
            lines.append(
                f"| {r['metro']} | **{b['beta']}** | "
                f"{b['median_km']:.2f} / {o['median_km']:.2f} | "
                f"{b['intrazonal_share']:.2%} / {o['intrazonal_share']:.2%} | "
                f"{b['score']:.3f} |"
            )
        betas = [r["fits"][form]["best"]["beta"] for r in rows
                 if "fits" in r and form in r.get("fits", {})]
        if len(betas) > 1:
            lines += [
                "",
                f"- fitted beta ranges **{min(betas)} – {max(betas)}** "
                f"(mean {np.mean(betas):.3f}, sd {np.std(betas):.3f})",
                f"- **Q6 — does one constant transfer?** "
                + ("Yes: the spread is small enough to ship a single default."
                   if np.std(betas) < 0.02 * max(1e-9, np.mean(betas)) or np.std(betas) < 0.015
                   else "Not cleanly: the spread is wide enough that a single "
                        "default would misfit some regions. Prefer a per-region "
                        "beta, or accept the mean and record the error it costs."),
            ]
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="single metro config")
    ap.add_argument("--all", action="store_true", help="fit every config under config/USA")
    ap.add_argument("--config-root", type=Path, default=Path("config/USA"))
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--forms", nargs="*",
                    default=[FRICTION_EXPONENTIAL, FRICTION_GAMMA, FRICTION_POWER])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    if args.all:
        configs = sorted(p for p in args.config_root.glob("*/*.json"))
    elif args.config:
        configs = [args.config]
    else:
        ap.error("pass --config or --all")

    forms = tuple(args.forms)
    rows = []
    for path in configs:
        print(f"=== {path} ===", file=sys.stderr)
        try:
            rows.append(fit_metro(path, args.data_dir, forms))
        except Exception as e:  # noqa: BLE001 - one metro must not stop the sweep
            print(f"  FAILED: {e!r}", file=sys.stderr)
            rows.append({"metro": path.parent.name, "config": str(path), "error": repr(e)})

    report = build_report(rows, forms)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(report)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
        print(f"wrote {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
