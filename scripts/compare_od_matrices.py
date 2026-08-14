"""Compare two OD matrices — the tool that answers "did it get better".

Standalone by design: it reads saved artefacts (``combined_od_matrix.csv``,
``base_od_matrix.csv``) rather than hooking into the pipeline, so any two runs
can be compared after the fact — including runs from different branches.

Metrics, and what each one is there to catch (see ``docs/od_matrix/design.md``):

    total trips, non-zero pairs, density      the agent-count drop; sparsity
    flow correlation (Pearson + Spearman)     overall agreement
    mean / median / p25 / p75 trip length     the distance-decay defect
    intrazonal share                          the dominant symptom
    top-N OD pairs, symmetric difference      where the two disagree most
    per-zone inflow/outflow correlation       whether marginals still hold
    zero-row / zero-col counts                zones stranded with no flow

Usage:
    # two experiment folders (reads combined_od_matrix.csv from each)
    python scripts/compare_od_matrices.py --a experiments/run_A --b experiments/run_B

    # explicit matrices, with a config supplying zone coordinates for distances
    python scripts/compare_od_matrices.py \
        --a path/to/one.csv --b path/to/two.csv \
        --config config/USA/TwinCities/config_twincities.json

    # compare the gravity baseline against observed LODES OD directly
    python scripts/compare_od_matrices.py --a experiments/run_A --b-observed \
        --config config/USA/TwinCities/config_twincities.json

Outputs a markdown summary to stdout (or --out), and with --plots writes a
trip-length distribution overlay plus a log-log cell scatter.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.od_diagnostics import cosine_corrected_km  # noqa: E402

# Reports contain em-dashes and other non-ASCII punctuation. A Windows console
# defaults to cp1252 and raises UnicodeEncodeError on them, which would kill the
# run at the final print after all the work is done. Replace unencodable
# characters instead of failing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")


def load_matrix(path: Path) -> pd.DataFrame:
    """Read an OD matrix CSV, keeping zone IDs as strings.

    Zone IDs are zero-padded GEOID prefixes; letting pandas infer them as
    integers would strip leading zeros and silently mismatch zones between the
    two matrices being compared.
    """
    df = pd.read_csv(path, index_col=0, dtype={0: str})
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def resolve_matrix_arg(arg: str) -> Path:
    """Accept either an experiment folder or a direct CSV path."""
    p = Path(arg)
    if p.is_dir():
        candidate = p / "combined_od_matrix.csv"
        if not candidate.exists():
            raise SystemExit(f"No combined_od_matrix.csv in {p}")
        return candidate
    if not p.exists():
        raise SystemExit(f"Matrix not found: {p}")
    return p


def align(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Put both matrices on the union of their zone grids, filling 0.

    Comparing on the intersection would hide exactly the disagreement that
    matters — zones one source covers and the other does not.
    """
    rows = sorted(set(a.index) | set(b.index))
    cols = sorted(set(a.columns) | set(b.columns))
    return (a.reindex(index=rows, columns=cols, fill_value=0).astype(float),
            b.reindex(index=rows, columns=cols, fill_value=0).astype(float))


def zone_coords_from_config(config_path: Path) -> Tuple[Dict[str, Tuple[float, float]],
                                                        Dict[str, Tuple[float, float]]]:
    """Load home/work zone centroids from the DB the config points at.

    Needed for the trip-length metrics; without a config the comparison still
    runs but skips every distance-based row.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    from models.home_locs_v2 import load_home_locations_by_counties
    from models.work_locs_v2 import load_work_locations_by_counties

    homes = load_home_locations_by_counties(config)
    works = load_work_locations_by_counties(config)

    geo_level = config.get("od_matrix", {}).get("geo_level", "block_group")
    prefix = 11 if geo_level == "tract" else 12

    def _centroids(blocks: Dict[str, Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
        acc: Dict[str, list] = {}
        for bid, d in blocks.items():
            if d.get("lat") is None or d.get("lon") is None:
                continue
            acc.setdefault(bid[:prefix], []).append((d["lon"], d["lat"]))
        return {z: (float(np.mean([c[0] for c in v])), float(np.mean([c[1] for c in v])))
                for z, v in acc.items()}

    return _centroids(homes), _centroids(works)


def observed_matrix_from_config(config_path: Path) -> pd.DataFrame:
    """Assemble the observed LODES OD matrix for the configured region."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    from data_sources.lodes_od import assemble_lodes_od, flows_to_zone_matrix
    from models.home_locs_v2 import load_home_locations_by_counties
    from models.work_locs_v2 import load_work_locations_by_counties

    observed = assemble_lodes_od(config)

    geo_level = config.get("od_matrix", {}).get("geo_level", "block_group")
    prefix = 11 if geo_level == "tract" else 12

    home_zones = {b[:prefix] for b in load_home_locations_by_counties(config)}
    work_zones = {b[:prefix] for b in load_work_locations_by_counties(config)}

    return flows_to_zone_matrix(observed.internal_flows, sorted(home_zones),
                                sorted(work_zones), geo_level=geo_level)


def basic_stats(m: pd.DataFrame) -> Dict[str, Any]:
    arr = m.to_numpy(dtype=float)
    nonzero = int(np.count_nonzero(arr))
    row_sums = arr.sum(axis=1)
    col_sums = arr.sum(axis=0)
    return {
        "total_trips": float(arr.sum()),
        "nonzero_pairs": nonzero,
        "density": nonzero / arr.size if arr.size else 0.0,
        "zero_rows": int((row_sums == 0).sum()),
        "zero_cols": int((col_sums == 0).sum()),
    }


def distance_stats(m: pd.DataFrame,
                   home_coords: Dict[str, Tuple[float, float]],
                   work_coords: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    """Trip-weighted distance percentiles and intrazonal share."""
    rows = [z for z in m.index if z in home_coords]
    cols = [z for z in m.columns if z in work_coords]
    if not rows or not cols:
        return {}

    sub = m.loc[rows, cols]
    w = sub.to_numpy(dtype=float)
    total = w.sum()
    if total <= 0:
        return {}

    d = cosine_corrected_km(
        np.array([home_coords[z] for z in rows], dtype=float),
        np.array([work_coords[z] for z in cols], dtype=float),
    )

    flat_d, flat_w = d.ravel(), w.ravel()
    keep = flat_w > 0
    flat_d, flat_w = flat_d[keep], flat_w[keep]

    order = np.argsort(flat_d)
    d_s, w_s = flat_d[order], flat_w[order]
    cum_frac = np.cumsum(w_s) / w_s.sum()

    def pct(p: float) -> float:
        return float(d_s[np.searchsorted(cum_frac, p)])

    shared = [z for z in rows if z in set(cols)]
    intra = float(sum(m.at[z, z] for z in shared)) if shared else 0.0

    return {
        "mean_km": float((flat_d * flat_w).sum() / flat_w.sum()),
        "p25_km": pct(0.25),
        "median_km": pct(0.5),
        "p75_km": pct(0.75),
        "intrazonal_share": intra / total,
        "_distances": flat_d,
        "_weights": flat_w,
    }


def correlations(a: pd.DataFrame, b: pd.DataFrame) -> Dict[str, Any]:
    """Cell-level and marginal agreement between two aligned matrices."""
    x = a.to_numpy(dtype=float).ravel()
    y = b.to_numpy(dtype=float).ravel()

    out: Dict[str, Any] = {}
    if x.std() > 0 and y.std() > 0:
        out["pearson"] = float(np.corrcoef(x, y)[0, 1])
        # Spearman on ranks; ties are pervasive here because both matrices
        # carry large blocks of zeros, so use average ranks.
        out["spearman"] = float(pd.Series(x).corr(pd.Series(y), method="spearman"))

    a_out, b_out = a.sum(axis=1), b.sum(axis=1)
    a_in, b_in = a.sum(axis=0), b.sum(axis=0)
    if a_out.std() > 0 and b_out.std() > 0:
        out["outflow_corr"] = float(a_out.corr(b_out))
    if a_in.std() > 0 and b_in.std() > 0:
        out["inflow_corr"] = float(a_in.corr(b_in))

    return out


def top_pair_overlap(a: pd.DataFrame, b: pd.DataFrame, n: int = 100) -> Dict[str, Any]:
    """How much the two matrices agree on where the biggest flows are."""
    a_top = set(a.stack().nlargest(n).index)
    b_top = set(b.stack().nlargest(n).index)
    shared = a_top & b_top
    return {
        "top_n": n,
        "shared": len(shared),
        "symmetric_difference": len(a_top ^ b_top),
        "jaccard": len(shared) / len(a_top | b_top) if (a_top | b_top) else 0.0,
    }


def _fmt(v: Any, spec: str = "{:,.2f}") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return spec.format(v)
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def build_report(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str,
                 home_coords: Optional[Dict] = None,
                 work_coords: Optional[Dict] = None,
                 top_n: int = 100) -> Tuple[str, Dict[str, Any]]:
    """Produce the markdown summary and the raw metric dict."""
    sa, sb = basic_stats(a), basic_stats(b)
    corr = correlations(a, b)
    tops = top_pair_overlap(a, b, top_n)

    da = distance_stats(a, home_coords, work_coords) if home_coords else {}
    db = distance_stats(b, home_coords, work_coords) if home_coords else {}

    lines = [
        f"# OD matrix comparison",
        "",
        f"- **A** — {label_a}",
        f"- **B** — {label_b}",
        f"- Compared on the union grid: {a.shape[0]:,} × {a.shape[1]:,} zones",
        "",
        "## Size and density",
        "",
        "| metric | A | B | B - A |",
        "|---|---:|---:|---:|",
        f"| total trips | {_fmt(sa['total_trips'], '{:,.0f}')} | {_fmt(sb['total_trips'], '{:,.0f}')} | "
        f"{_fmt(sb['total_trips'] - sa['total_trips'], '{:+,.0f}')} |",
        f"| non-zero pairs | {_fmt(sa['nonzero_pairs'])} | {_fmt(sb['nonzero_pairs'])} | "
        f"{_fmt(sb['nonzero_pairs'] - sa['nonzero_pairs'], '{:+,}')} |",
        f"| density | {sa['density']:.2%} | {sb['density']:.2%} | "
        f"{(sb['density'] - sa['density']) * 100:+.2f} pp |",
        f"| zero rows | {_fmt(sa['zero_rows'])} | {_fmt(sb['zero_rows'])} | "
        f"{sb['zero_rows'] - sa['zero_rows']:+,} |",
        f"| zero cols | {_fmt(sa['zero_cols'])} | {_fmt(sb['zero_cols'])} | "
        f"{sb['zero_cols'] - sa['zero_cols']:+,} |",
        "",
        "## Agreement",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| cell correlation (Pearson) | {_fmt(corr.get('pearson'), '{:.4f}')} |",
        f"| cell correlation (Spearman) | {_fmt(corr.get('spearman'), '{:.4f}')} |",
        f"| per-zone outflow correlation | {_fmt(corr.get('outflow_corr'), '{:.4f}')} |",
        f"| per-zone inflow correlation | {_fmt(corr.get('inflow_corr'), '{:.4f}')} |",
        f"| top-{tops['top_n']} pairs shared | {tops['shared']} / {tops['top_n']} "
        f"(Jaccard {tops['jaccard']:.3f}) |",
        f"| top-{tops['top_n']} symmetric difference | {tops['symmetric_difference']} |",
    ]

    if da and db:
        lines += [
            "",
            "## Trip length",
            "",
            "| metric | A | B | B - A |",
            "|---|---:|---:|---:|",
            f"| mean km | {da['mean_km']:.2f} | {db['mean_km']:.2f} | {db['mean_km'] - da['mean_km']:+.2f} |",
            f"| p25 km | {da['p25_km']:.2f} | {db['p25_km']:.2f} | {db['p25_km'] - da['p25_km']:+.2f} |",
            f"| median km | {da['median_km']:.2f} | {db['median_km']:.2f} | {db['median_km'] - da['median_km']:+.2f} |",
            f"| p75 km | {da['p75_km']:.2f} | {db['p75_km']:.2f} | {db['p75_km'] - da['p75_km']:+.2f} |",
            f"| intrazonal share | {da['intrazonal_share']:.2%} | {db['intrazonal_share']:.2%} | "
            f"{(db['intrazonal_share'] - da['intrazonal_share']) * 100:+.2f} pp |",
        ]
    elif home_coords:
        lines += ["", "_Trip-length metrics skipped: no zones matched the supplied coordinates._"]
    else:
        lines += ["", "_Trip-length metrics skipped: pass --config to supply zone coordinates._"]

    metrics = {
        "label_a": label_a, "label_b": label_b,
        "a": sa, "b": sb, "correlations": corr, "top_pairs": tops,
        "distance_a": {k: v for k, v in da.items() if not k.startswith("_")},
        "distance_b": {k: v for k, v in db.items() if not k.startswith("_")},
    }
    return "\n".join(lines), metrics


def make_plots(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str,
               home_coords: Dict, work_coords: Dict, out_dir: Path) -> None:
    """Trip-length distribution overlay and a log-log cell scatter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    da = distance_stats(a, home_coords, work_coords)
    db = distance_stats(b, home_coords, work_coords)

    if da and db:
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(0, 80, 81)
        for stats, label, color in ((da, label_a, "tab:blue"), (db, label_b, "tab:orange")):
            ax.hist(stats["_distances"], bins=bins, weights=stats["_weights"],
                    density=True, histtype="step", linewidth=2, label=label, color=color)
        ax.set_xlabel("trip length (km)")
        ax.set_ylabel("share of trips")
        ax.set_title("Trip-length distribution")
        ax.legend()
        fig.tight_layout()
        path = out_dir / "trip_length_distribution.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"wrote {path}", file=sys.stderr)

    # Log-log cell scatter, matching the style of generate_counts_loglog.py.
    x = a.to_numpy(dtype=float).ravel()
    y = b.to_numpy(dtype=float).ravel()
    keep = (x > 0) & (y > 0)
    if keep.any():
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x[keep], y[keep], s=4, alpha=0.15, edgecolors="none")
        lo = max(min(x[keep].min(), y[keep].min()), 1e-3)
        hi = max(x[keep].max(), y[keep].max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="1:1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{label_a} (trips per OD pair)")
        ax.set_ylabel(f"{label_b} (trips per OD pair)")
        ax.set_title(f"Cell-level comparison ({keep.sum():,} pairs non-zero in both)")
        ax.legend()
        fig.tight_layout()
        path = out_dir / "od_cell_loglog.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"wrote {path}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True,
                    help="experiment folder or OD matrix CSV (arm A)")
    ap.add_argument("--b", help="experiment folder or OD matrix CSV (arm B)")
    ap.add_argument("--b-observed", action="store_true",
                    help="use observed LODES OD as arm B (requires --config)")
    ap.add_argument("--config", type=Path,
                    help="config JSON supplying zone coordinates (and the region "
                         "for --b-observed). Without it, distance metrics are skipped.")
    ap.add_argument("--top-n", type=int, default=100,
                    help="how many top OD pairs to compare (default 100)")
    ap.add_argument("--out", type=Path, help="write the markdown report here")
    ap.add_argument("--json-out", type=Path, help="write raw metrics as JSON here")
    ap.add_argument("--plots", type=Path, metavar="DIR",
                    help="write comparison plots into this directory")
    args = ap.parse_args()

    if not args.b and not args.b_observed:
        ap.error("pass either --b or --b-observed")
    if args.b_observed and not args.config:
        ap.error("--b-observed requires --config to know the region")

    path_a = resolve_matrix_arg(args.a)
    a = load_matrix(path_a)
    label_a = str(path_a)

    if args.b_observed:
        b = observed_matrix_from_config(args.config)
        label_b = f"observed LODES OD ({args.config.stem})"
    else:
        path_b = resolve_matrix_arg(args.b)
        b = load_matrix(path_b)
        label_b = str(path_b)

    a, b = align(a, b)

    home_coords = work_coords = None
    if args.config:
        try:
            home_coords, work_coords = zone_coords_from_config(args.config)
        except Exception as e:  # noqa: BLE001 - distances are optional
            print(f"warning: could not load zone coordinates ({e}); "
                  f"skipping distance metrics", file=sys.stderr)

    report, metrics = build_report(a, b, label_a, label_b,
                                   home_coords, work_coords, args.top_n)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(report)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}", file=sys.stderr)

    if args.plots:
        if not home_coords:
            print("warning: --plots needs --config for coordinates; "
                  "only the log-log scatter will be drawn", file=sys.stderr)
            make_plots(a, b, label_a, label_b, {}, {}, args.plots)
        else:
            make_plots(a, b, label_a, label_b, home_coords, work_coords, args.plots)


if __name__ == "__main__":
    main()
