"""Stage 5 — the ablation that decides whether the OD change actually helped.

Four arms per metro, one variable at a time, so an improvement can be attributed
to the change that caused it:

    A  gravity, power, legacy degrees   today's behaviour (the baseline)
    B  gravity, power, corrected km     stage 1 alone
    C  gravity, bounded, calibrated     stage 4
    D  observed LODES OD                the target

Two things this script does that a plain "compare the totals" would miss:

**Fixed seed across arms.** Sparsity raises seed variance sharply — at
scaling_factor=0.15 the large majority of demand rides on Bernoulli draws (E10),
so two runs of the *same* arm can differ. Comparing arms under different seeds
would measure noise.

**Per-station shortfall, not just the aggregate (G6).** Whether the leftover gap
is uniform across stations or concentrated on particular corridors is what
decides the next move: uniform means a `scaling_factor` question, concentrated
on truck routes or boundary radials means a freight or boundary question. The
aggregate cannot tell those apart, and E1 step 2 cannot be decided without it.

Usage:
    # print the arm matrix and the config each arm needs
    python scripts/od_ablation.py --plan

    # write ready-to-run config files, one per arm
    python scripts/od_ablation.py --emit-configs out/ablation \
        --config config/USA/TwinCities/config_twincities.json

    # after the runs, compare the arms' matrices and count performance
    python scripts/od_ablation.py --compare \
        --arm A=experiments/armA --arm D=experiments/armD \
        --config config/USA/TwinCities/config_twincities.json
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

# Arm definitions. Each entry is the od_matrix overrides that arm needs; every
# other config value stays at the metro's own setting so the arms differ only
# in the variable under test.
ARMS: Dict[str, Dict[str, Any]] = {
    "A": {
        "_what": "today's behaviour — gravity, power law, raw degrees",
        "source": "gravity",
        "friction": "power",
        "distance": "legacy_degrees",
    },
    "B": {
        "_what": "stage 1 — same model, corrected geometry",
        "source": "gravity",
        "friction": "power",
        "distance": "cosine_km",
    },
    "C": {
        "_what": "stage 4 — bounded friction on corrected geometry",
        "source": "gravity",
        "friction": "exponential",
        "distance": "cosine_km",
        "_beta_note": "refit per metro with scripts/fit_gravity_beta.py — beta is in 1/km here",
    },
    "D": {
        "_what": "target — observed LODES OD flows",
        "source": "lodes_od",
        "distance": "cosine_km",
    },
}


def print_plan() -> None:
    print("Stage 5 ablation — four arms per metro\n")
    print(f"{'arm':4s} {'source':10s} {'friction':12s} {'distance':16s} what")
    print("-" * 88)
    for name, cfg in ARMS.items():
        print(f"{name:4s} {cfg.get('source','-'):10s} {cfg.get('friction','-'):12s} "
              f"{cfg.get('distance','-'):16s} {cfg['_what']}")
    print("\nHold these fixed across all four arms, or the comparison measures noise:")
    print("  - plan_generation.random_seed  (E10: sparsity raises seed variance)")
    print("  - plan_generation.scaling_factor (NOT tuned during the ablation — E1/G6)")
    print("  - region.counties, data.lodes.year, geo level")
    print("\nReport per arm: GEH pass rates / RMSE / correlation vs FHWA counts, the §4 matrix")
    print("metrics, and the per-station shortfall breakdown (G6). Non-work is")
    print("reported separately — stage 1 touches it, stage 2 does not.")


def emit_configs(base_config: Path, out_dir: Path, seed: Optional[int],
                 betas: Optional[Dict[str, float]] = None) -> None:
    """Write one config per arm, differing only in the arm's variable."""
    import collections

    base = json.load(open(base_config, encoding="utf-8"),
                     object_pairs_hook=collections.OrderedDict)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, overrides in ARMS.items():
        cfg = json.loads(json.dumps(base))  # deep copy
        od = cfg.setdefault("od_matrix", {})
        for k, v in overrides.items():
            if k.startswith("_"):
                continue
            od[k] = v

        # Arm C runs a bounded form, whose beta is in 1/km and must be refitted.
        if name == "C" and betas and "C" in betas:
            od["beta"] = betas["C"]

        if seed is not None:
            cfg.setdefault("plan_generation", {})["random_seed"] = seed

        cfg["_ablation_arm"] = {"arm": name, "what": overrides["_what"]}

        path = out_dir / f"arm_{name}_{base_config.stem}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"wrote {path}", file=sys.stderr)

    print(f"\nRun each with:  python run_experiment.py --config {out_dir}/arm_<X>_*.json",
          file=sys.stderr)


# --------------------------------------------------------------------------
# Per-station shortfall — the G6 question
# --------------------------------------------------------------------------

def find_countscompare(folder: Path) -> Optional[Path]:
    """Highest-iteration countscompare file in an experiment folder."""
    iters = folder / "ITERS"
    if iters.exists():
        candidates = [p for p in iters.glob("it.*/*.countscompare.txt")
                      if "AWTV" not in p.name]
    else:
        candidates = [p for p in folder.glob("**/*.countscompare.txt")
                      if "AWTV" not in p.name]
    if not candidates:
        return None

    def iter_num(path: Path) -> int:
        for part in path.parts:
            if part.startswith("it."):
                try:
                    return int(part[3:])
                except ValueError:
                    pass
        return -1

    return max(candidates, key=iter_num)


def load_station_volumes(experiment_dir: Path,
                         hours: Optional[List[int]] = None) -> Optional[pd.DataFrame]:
    """Per-station simulated and observed volumes, summed over *hours*."""
    path = find_countscompare(experiment_dir)
    if path is None:
        print(f"  no countscompare in {experiment_dir}", file=sys.stderr)
        return None

    df = pd.read_csv(path, sep="\t")
    required = {"Count Station Id", "Hour", "MATSIM volumes", "Count volumes"}
    if not required.issubset(df.columns):
        print(f"  unexpected columns in {path}: {sorted(df.columns)}", file=sys.stderr)
        return None

    if hours:
        df = df[df["Hour"].isin(hours)]

    grouped = df.groupby("Count Station Id").agg(
        sim=("MATSIM volumes", "sum"),
        obs=("Count volumes", "sum"),
    ).reset_index()
    grouped = grouped[grouped["obs"] > 0]
    return grouped.rename(columns={"Count Station Id": "station"})


def geh(sim: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """GEH statistic — the standard traffic-count goodness-of-fit measure.

    Below 5 is conventionally a good match; it tolerates larger absolute errors
    on high-volume links than a plain percentage would.
    """
    denom = (sim + obs) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(np.where(denom > 0, 2 * (sim - obs) ** 2 / denom, np.nan))


def station_shortfall(volumes: pd.DataFrame) -> Dict[str, Any]:
    """Is the leftover demand gap uniform across stations, or concentrated?

    This is the G6 question, and it decides what happens to `scaling_factor`.
    A flat shortfall across stations means demand is uniformly short and a
    uniform multiplier is an honest fix. A shortfall concentrated on a subset
    means specific missing demand — freight on truck corridors, boundary
    commuters on the radials — and scaling everything up would inflate the
    stations that are already correct while still missing the ones that matter.
    """
    sim = volumes["sim"].to_numpy(dtype=float)
    obs = volumes["obs"].to_numpy(dtype=float)

    ratio = np.divide(sim, obs, out=np.full_like(sim, np.nan), where=obs > 0)
    g = geh(sim, obs)

    finite = ratio[np.isfinite(ratio)]
    # Coefficient of variation of the per-station ratio is the discriminator:
    # a uniform shortfall has every station short by roughly the same factor.
    cv = float(np.std(finite) / np.mean(finite)) if len(finite) and np.mean(finite) else float("nan")

    out = {
        "stations": int(len(volumes)),
        "total_sim": float(sim.sum()),
        "total_obs": float(obs.sum()),
        "aggregate_ratio": float(sim.sum() / obs.sum()) if obs.sum() else None,
        "median_station_ratio": float(np.nanmedian(ratio)),
        "p10_station_ratio": float(np.nanpercentile(ratio, 10)),
        "p90_station_ratio": float(np.nanpercentile(ratio, 90)),
        "ratio_cv": cv,
        # Pass rates only. These GEH values are computed on each station's
        # total over the selected hours, so each one is a real GEH of a real
        # count — but averaging them across stations of different size would
        # measure station size as much as model error, and the mean would not
        # be the GEH of anything. Counting how many stations clear a threshold
        # has no such defect.
        "pct_geh_under_5": float(np.nanmean(g < 5) * 100),
        "pct_geh_under_10": float(np.nanmean(g < 10) * 100),
        "rmse": float(np.sqrt(np.nanmean((sim - obs) ** 2))),
        "correlation": float(np.corrcoef(sim, obs)[0, 1]) if len(sim) > 1 else None,
    }

    # A tight spread means one number describes every station; a wide one means
    # the gap lives in specific places and a uniform bump is the wrong tool.
    out["shortfall_pattern"] = (
        "uniform" if cv == cv and cv < 0.35 else "concentrated"
    )
    out["scaling_factor_verdict"] = (
        "A uniform scaling_factor increase is defensible — the shortfall is "
        "spread evenly across stations."
        if out["shortfall_pattern"] == "uniform" else
        "Do NOT raise scaling_factor yet — the shortfall is concentrated on a "
        "subset of stations, so a uniform multiplier would inflate stations that "
        "already match while still missing the ones that don't. Identify what "
        "those stations have in common (truck corridors? boundary radials?) first."
    )
    return out


def worst_stations(volumes: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """The stations contributing most of the absolute volume gap."""
    v = volumes.copy()
    v["gap"] = v["obs"] - v["sim"]
    v["ratio"] = v["sim"] / v["obs"]
    v["geh"] = geh(v["sim"].to_numpy(float), v["obs"].to_numpy(float))
    return v.reindex(v["gap"].abs().sort_values(ascending=False).index).head(n)


def compare_arms(arms: Dict[str, Path], config: Optional[Path],
                 hours: Optional[List[int]]) -> str:
    """Build the stage-5 comparison report across whichever arms have run."""
    lines = ["# Stage 5 — ablation results", ""]

    if config:
        lines += [f"Region config: `{config}`", ""]

    # --- OD matrix metrics per arm (from each run's diagnostics) ---
    lines += ["## Matrix metrics", "",
              "| arm | source | friction | distance | total trips | non-zero pairs | density | intrazonal | median km |",
              "|---|---|---|---|---:|---:|---:|---:|---:|"]

    for name, folder in sorted(arms.items()):
        diag_path = Path(folder) / "od_matrix_diagnostics.json"
        if not diag_path.exists():
            lines.append(f"| {name} | _no diagnostics file in {folder}_ | | | | | | | |")
            continue
        d = json.loads(diag_path.read_text(encoding="utf-8"))
        m = d.get("matrix") or {}
        t = d.get("trip_length") or {}
        fr = d.get("friction") or {}
        di = d.get("distance") or {}
        lines.append(
            f"| {name} | {d.get('source_used','—')} | {fr.get('form','—')} | "
            f"{di.get('mode','—')} | {m.get('total_trips',0):,.0f} | "
            f"{m.get('nonzero_pairs',0):,} | {m.get('density',0):.2%} | "
            f"{(t.get('intrazonal_share') or 0):.2%} | {t.get('median_km','—')} |"
        )

    # --- Counts performance per arm ---
    lines += ["", "## Versus FHWA counts", "",
              "| arm | stations | sim/obs | % GEH<5 | % GEH<10 | RMSE | corr | shortfall |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]

    shortfalls: Dict[str, Dict[str, Any]] = {}
    for name, folder in sorted(arms.items()):
        vols = load_station_volumes(Path(folder), hours)
        if vols is None or vols.empty:
            lines.append(f"| {name} | _no counts data_ | | | | | | |")
            continue
        s = station_shortfall(vols)
        shortfalls[name] = s
        lines.append(
            f"| {name} | {s['stations']} | {s['aggregate_ratio']:.3f} | "
            f"{s['pct_geh_under_5']:.0f}% | {s['pct_geh_under_10']:.0f}% | "
            f"{s['rmse']:,.0f} | {s['correlation']:.3f} | {s['shortfall_pattern']} |"
        )

    # --- G6: the per-station question ---
    if shortfalls:
        lines += ["", "## G6 — is the shortfall uniform or concentrated?", "",
                  "This is what decides whether `scaling_factor` may be touched (E1 step 2).",
                  ""]
        for name, s in sorted(shortfalls.items()):
            lines += [
                f"### Arm {name}",
                "",
                f"- aggregate sim/obs: **{s['aggregate_ratio']:.3f}**",
                f"- per-station ratio: median {s['median_station_ratio']:.3f}, "
                f"p10 {s['p10_station_ratio']:.3f}, p90 {s['p90_station_ratio']:.3f}",
                f"- coefficient of variation: **{s['ratio_cv']:.3f}** → "
                f"**{s['shortfall_pattern']}**",
                "",
                f"> {s['scaling_factor_verdict']}",
                "",
            ]

    # --- G5 verdict ---
    if "A" in shortfalls and "D" in shortfalls:
        a, d = shortfalls["A"], shortfalls["D"]
        better = d["pct_geh_under_5"] > a["pct_geh_under_5"]
        lines += [
            "## G5 — does arm D beat arm A?",
            "",
            f"- % stations GEH<5: A {a['pct_geh_under_5']:.0f}% → "
            f"D {d['pct_geh_under_5']:.0f}% ({'improved' if better else 'worse'})",
            f"- correlation: A {a['correlation']:.3f} → D {d['correlation']:.3f}",
            "",
        ]
        if not better:
            lines += [
                "> **A first-pass failure here is expected and informative.** Arm D "
                "carries fewer agents (E1) and, under `boundary_policy: drop`, no "
                "boundary trips (E5) — both can worsen raw counts while the demand "
                "*pattern* is more correct. Expected order of fixes: boundary "
                "anchoring first, then the per-station analysis above, then "
                "`scaling_factor` last. Do not conclude the source change failed "
                "until those are done.",
                "",
            ]

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print the arm matrix")
    ap.add_argument("--emit-configs", type=Path, metavar="DIR",
                    help="write one config per arm into DIR")
    ap.add_argument("--config", type=Path, help="base metro config")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed held fixed across arms (E10)")
    ap.add_argument("--beta-c", type=float,
                    help="refitted beta for arm C (1/km) from fit_gravity_beta.py")
    ap.add_argument("--compare", action="store_true", help="compare finished runs")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=DIR",
                    help="an arm's experiment folder, e.g. --arm A=experiments/armA")
    ap.add_argument("--hours", type=int, nargs="*",
                    help="restrict count comparison to these hours (1-24)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.plan:
        print_plan()
        return

    if args.emit_configs:
        if not args.config:
            ap.error("--emit-configs requires --config")
        betas = {"C": args.beta_c} if args.beta_c else None
        emit_configs(args.config, args.emit_configs, args.seed, betas)
        return

    if args.compare:
        arms = {}
        for spec in args.arm:
            if "=" not in spec:
                ap.error(f"--arm expects NAME=DIR, got {spec!r}")
            name, path = spec.split("=", 1)
            arms[name.strip()] = Path(path)
        if not arms:
            ap.error("--compare needs at least one --arm NAME=DIR")

        report = compare_arms(arms, args.config, args.hours)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(report, encoding="utf-8")
            print(f"wrote {args.out}", file=sys.stderr)
        else:
            print(report)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
