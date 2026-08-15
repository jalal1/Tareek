"""Generate log-log observed-vs-simulated count comparisons across experiments.

Reads MATSim's ``*.countscompare.txt`` (written by the counts module: one row
per count station per hour with scale-corrected, iteration-averaged MATSIM and
Count volumes) and draws log-log scatter plots of simulated vs. observed volumes
at selected hours.

Two layouts (``--mode``):
  * ``overlay``  — one figure per hour, every experiment drawn on the same axes
                   in a different colour/marker (compare experiments directly).
  * ``separate`` — one figure per (experiment, hour), like the paper figures.

Station aggregation:
  This branch names FHA count stations per direction as
  ``FHA_<state>_<id>_<dir>`` (dir in {1,3,5,7}). By default each per-direction
  station is plotted as its own point. Pass ``--sum-directions`` to sum the two
  carriageways back into one point per physical sensor (the evaluator's
  ``_station_base`` convention: strip a trailing ``_<digits>`` suffix, and split
  a combined ``A+B`` id on ``+`` first).

Experiments are passed on the command line, so the pipeline can call this per
run without the file being edited first.

Comparing across experiments:
  In overlay mode, pass ``--connect`` to join the same station's points across
  experiments with a line so you can tell whether a station moved toward the
  1:1 line (green = improved) or away from it (red = regressed). Without this,
  overlaid circles and squares can't be matched by eye to the same station.
  Stations are matched on their full Count Station Id (the station_base when
  ``--sum-directions`` collapses directions).

Usage:
    # one run — figures land in that run's evaluation/ folder
    python scripts/generate_counts_loglog.py experiments/experiment_20260812_233102

    # several runs overlaid, with labels for the legend
    python scripts/generate_counts_loglog.py \\
        "drop=experiments/exp_A" "anchor=experiments/exp_B" --connect

    python scripts/generate_counts_loglog.py <exp> --mode separate --sum-directions
    python scripts/generate_counts_loglog.py <exp_a> <exp_b> --hours 7 8 16 17
    python scripts/generate_counts_loglog.py <exp> --out-dir some/other/place
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Clock hours to plot, 0..23 — the same convention the evaluator and the rest
# of the report use ("hour 7" is 07:00-08:00, matching count_error_h07.png).
#
# countscompare.txt numbers its Hour column 1..24 for the SAME bins, so clock
# hour h is row Hour == h + 1. Everything below works in clock hours and
# converts only at the point of reading the file; getting this wrong silently
# shifts every figure by one hour against the maps it sits beside.
DEFAULT_HOURS = [7, 8, 16, 17]


def clock_to_countscompare(hour: int) -> int:
    """Clock hour (0..23) -> the Hour value used in countscompare.txt (1..24)."""
    return hour + 1

# Distinct colours/markers for overlaying experiments.
_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
            "#8c564b", "#e377c2"]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def find_countscompare(folder: Path) -> Path | None:
    """Return the highest-iteration ``*.countscompare.txt`` (excluding *AWTV*).

    Searches the experiment's MATSim output ITERS tree, falling back to a flat
    glob in the folder itself.
    """
    iters = folder / "output" / "ITERS"
    candidates: list[Path] = []
    if iters.exists():
        candidates = [p for p in iters.glob("it.*/*.countscompare.txt")
                      if not p.name.endswith("AWTV.txt")]
    if not candidates:
        candidates = [p for p in folder.glob("**/*.countscompare.txt")
                      if not p.name.endswith("AWTV.txt")]
    if not candidates:
        return None

    def iter_num(path: Path) -> int:
        try:
            return int(path.parent.name.split(".")[-1])
        except (ValueError, IndexError):
            return -1

    candidates.sort(key=iter_num)
    return candidates[-1]


def station_base(cs_id: str) -> str:
    """Strip the trailing _<dir> from an FHA per-direction station id.

    Mirrors SimulationEvaluator._station_base so this script aggregates exactly
    like the evaluator does. 'FHA_27_000026_1' -> 'FHA_27_000026'. A combined
    'A+B' id (two stations on one link) keeps its first segment's base. Ids
    without a numeric direction suffix are returned unchanged.
    """
    sid = str(cs_id).split("+")[0]
    parts = sid.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return sid


def load_hour(counts_file: Path, hour: int, sum_directions: bool) -> pd.DataFrame | None:
    """Load one hour from a countscompare file.

    Returns a DataFrame with columns: station, sim, obs. When *sum_directions*
    is set, per-direction rows are summed into one row per physical station.
    """
    df = pd.read_csv(counts_file, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    required = {"Count Station Id", "Hour", "MATSIM volumes", "Count volumes"}
    missing = required - set(df.columns)
    if missing:
        print(f"  SKIP {counts_file}: missing columns {sorted(missing)}")
        return None

    hour_df = df[df["Hour"] == clock_to_countscompare(hour)]
    if hour_df.empty:
        return None

    out = pd.DataFrame({
        "station": hour_df["Count Station Id"].astype(str).values,
        "sim": hour_df["MATSIM volumes"].astype(float).values,
        "obs": hour_df["Count volumes"].astype(float).values,
    })

    if sum_directions:
        out["station"] = out["station"].apply(station_base)
        out = out.groupby("station", as_index=False)[["sim", "obs"]].sum()

    return out


def _plot_coords(plot_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-station plotting coordinates on the log-log axes.

    Zero-observed points are clipped to obs=1 and sim>=1 so they stay on the
    log axes — the same transform used for the scatter, so connector lines line
    up exactly with the drawn markers. Columns: station, px (obs), py (sim).
    """
    obs = plot_df["obs"].values.astype(float)
    sim = plot_df["sim"].values.astype(float)
    px = np.where(obs == 0, 1.0, obs)
    py = np.where(obs == 0, np.clip(sim, 1, None), sim)
    return pd.DataFrame({"station": plot_df["station"].astype(str).values,
                         "px": px, "py": py})


def _scatter(ax, plot_df: pd.DataFrame, color: str, marker: str, label: str):
    """Plot one experiment's (obs, sim) points on log-log axes.

    Non-zero-observed points are drawn directly; zero-observed points are
    clipped to obs=1 so they remain visible on the log axis.
    """
    sim = plot_df["sim"].values
    obs = plot_df["obs"].values
    mask_zero = obs == 0

    sim_nz, obs_nz = sim[~mask_zero], obs[~mask_zero]
    sim_z = sim[mask_zero]

    if len(obs_nz) > 0:
        ax.scatter(obs_nz, sim_nz, s=20, color=color, alpha=0.75,
                   marker=marker, linewidths=0, zorder=3, label=label)
    if len(sim_z) > 0:
        ax.scatter(np.ones_like(sim_z), np.clip(sim_z, 1, None), s=20,
                   color=color, alpha=0.45, marker=marker, linewidths=0,
                   zorder=3)


def _draw_connectors(ax, coords_by_exp: list[pd.DataFrame]):
    """Join each station's points across experiments with a thin line.

    coords_by_exp is a list of per-experiment DataFrames (station, px, py) in
    experiment order. A station present in >=2 experiments gets a polyline
    through its points, colored by whether the LAST experiment is closer to the
    1:1 line than the FIRST: green = improved, red = regressed, gray = flat/
    only-some-present. Distance to 1:1 is |log10(sim) - log10(obs)| (constant
    on a log-log plot), so it is scale-independent.

    Returns the number of stations connected (drawn with >=2 points).
    """
    # station -> ordered list of (exp_index, px, py)
    per_station: dict[str, list[tuple[int, float, float]]] = {}
    for ei, coords in enumerate(coords_by_exp):
        for _, r in coords.iterrows():
            per_station.setdefault(r["station"], []).append((ei, r["px"], r["py"]))

    def dist_to_diagonal(px: float, py: float) -> float:
        # |log10(sim) - log10(obs)|; px is plotted-obs, py is plotted-sim.
        return abs(np.log10(max(py, 1e-9)) - np.log10(max(px, 1e-9)))

    connected = 0
    labeled = {"improved": False, "regressed": False, "flat": False}
    color_for = {"improved": "#2ca02c", "regressed": "#d62728", "flat": "#999999"}
    for pts in per_station.values():
        if len(pts) < 2:
            continue  # station not in >=2 experiments — nothing to connect
        pts.sort(key=lambda t: t[0])  # by experiment order
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]

        d_first = dist_to_diagonal(pts[0][1], pts[0][2])
        d_last = dist_to_diagonal(pts[-1][1], pts[-1][2])
        if d_last < d_first - 1e-9:
            kind = "improved"
        elif d_last > d_first + 1e-9:
            kind = "regressed"
        else:
            kind = "flat"

        label = None
        if not labeled[kind]:
            label = {"improved": "Improved (toward 1:1)",
                     "regressed": "Regressed (away from 1:1)",
                     "flat": "Unchanged"}[kind]
            labeled[kind] = True

        ax.plot(xs, ys, color=color_for[kind], linewidth=0.8, alpha=0.55,
                zorder=1, label=label)
        connected += 1

    return connected


def _finalize_axes(ax, all_obs: np.ndarray, all_sim: np.ndarray, title: str):
    """Add 1:1 / 2x / 0.5x reference lines, log scales, labels, and title.

    Axis limits auto-fit the data: the log-log range is derived from the actual
    min/max of all observed+simulated volumes (with a small margin) so points
    are always visible, whatever the volume scale of the run.
    """
    # Only positive values are plottable on a log axis. The 1.0 fallback is a
    # guard for the degenerate all-zero case, NOT a data point — including it
    # unconditionally would pin the axis floor at 1 even when the smallest real
    # volume is far above it, wasting most of the frame on empty decades.
    vals = np.concatenate([all_obs, all_sim])
    vals = vals[vals > 0]
    if len(vals) == 0:
        vals = np.array([1.0])
    data_lo = vals.min()
    data_hi = vals.max()
    # Reference lines span a little beyond the data on both ends.
    lo = max(data_lo * 0.5, 1.0)
    hi = data_hi * 2.0
    x_ref = np.array([lo, hi])
    ax.plot(x_ref, x_ref,       color="#444444", linewidth=1.4, linestyle="-",
            zorder=2, label="1:1 line")
    ax.plot(x_ref, x_ref * 2,   color="#888888", linewidth=1.0, linestyle="--", zorder=2)
    ax.plot(x_ref, x_ref * 0.5, color="#888888", linewidth=1.0, linestyle="--", zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    # Auto-fit limits to the data with a ~20% margin on the log axis.
    ax.set_xlim(left=lo, right=hi)
    ax.set_ylim(bottom=lo, top=hi)
    ax.set_xlabel("Observed Volumes [veh/h]", fontsize=11)
    ax.set_ylabel("Simulated Volumes [veh/h]", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)


def _new_fig():
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def generate_overlay(loaded: dict[str, Path], hours: list[int], sum_directions: bool,
                     out_dir: Path, connect: bool = False, fmt: str = "pdf"):
    """One figure per hour; every experiment overlaid on shared axes.

    When *connect* is set, the same station's points are joined across
    experiments so you can see whether a station moved toward or away from the
    1:1 line. Stations are matched by their full Count Station Id (the
    station_base when --sum-directions collapses directions).
    """
    for hour in hours:
        fig, ax = _new_fig()
        all_obs, all_sim = [], []
        coords_by_exp = []
        plotted = 0
        for i, (label, counts_file) in enumerate(loaded.items()):
            plot_df = load_hour(counts_file, hour, sum_directions)
            if plot_df is None or plot_df.empty:
                print(f"  SKIP {label} hour {hour}: no data")
                continue
            # Connectors are drawn first (lower zorder) so markers sit on top.
            coords_by_exp.append(_plot_coords(plot_df))
            _scatter(ax, plot_df, _PALETTE[i % len(_PALETTE)],
                     _MARKERS[i % len(_MARKERS)], label)
            all_obs.append(plot_df["obs"].values)
            all_sim.append(plot_df["sim"].values)
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            continue

        if connect:
            if plotted < 2:
                print(f"  hour {hour}: --connect needs >=2 experiments with data; "
                      f"only {plotted} present, drawing points only")
            else:
                n = _draw_connectors(ax, coords_by_exp)
                print(f"  hour {hour}: connected {n} station(s) across experiments")

        agg = "summed" if sum_directions else "per-dir"
        _finalize_axes(ax, np.concatenate(all_obs), np.concatenate(all_sim),
                       f"Counts comparison — hour {hour:02d}:00 ({agg})")
        out_dir.mkdir(parents=True, exist_ok=True)
        # A single experiment gets the stable name the report looks for; only a
        # multi-experiment overlay needs the aggregation/connect suffixes to
        # tell its variants apart.
        if len(loaded) == 1:
            out_path = out_dir / f"counts_loglog_h{hour:02d}.{fmt}"
        else:
            suffix = "summed" if sum_directions else "perdir"
            if connect:
                suffix += "_connected"
            out_path = out_dir / f"counts_overlay_h{hour:02d}_{suffix}.{fmt}"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Figure: {out_path}")


def generate_separate(loaded: dict[str, Path], hours: list[int], sum_directions: bool,
                      out_dir: Path, fmt: str = "pdf"):
    """One figure per (experiment, hour)."""
    for label, counts_file in loaded.items():
        safe = label.replace(", ", "_").replace(" ", "_")
        for hour in hours:
            plot_df = load_hour(counts_file, hour, sum_directions)
            if plot_df is None or plot_df.empty:
                print(f"  SKIP {label} hour {hour}: no data")
                continue

            fig, ax = _new_fig()
            _scatter(ax, plot_df, _PALETTE[0], _MARKERS[0], "Count stations")
            agg = "summed" if sum_directions else "per-dir"
            _finalize_axes(ax, plot_df["obs"].values, plot_df["sim"].values,
                           f"{label} — hour {hour:02d}:00 ({agg})")
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix = "summed" if sum_directions else "perdir"
            out_path = out_dir / f"counts_h{hour:02d}_{safe}_{suffix}.{fmt}"
            fig.savefig(out_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"  Figure: {out_path}")


def parse_experiment_arg(arg: str) -> tuple[str, Path]:
    """Split a ``LABEL=path`` argument into its parts.

    A bare path is allowed; the folder name then doubles as the label. Only the
    first ``=`` splits, so a path containing one still works — and a Windows
    drive letter ('E:/...') is never mistaken for a label, because the split is
    on '=' rather than ':'.
    """
    if "=" in arg:
        label, _, raw = arg.partition("=")
        label = label.strip()
        if label and raw.strip():
            return label, Path(raw.strip())
    path = Path(arg)
    return path.name or str(path), path


def resolve_experiments(specs: list[str]) -> dict[str, Path]:
    """Resolve CLI arguments to {label: countscompare_path}, skipping missing ones.

    Paths are used as given (relative ones resolve against the working
    directory), so the same command works on the server and on Windows without
    a checkout-specific root baked into the file.
    """
    loaded: dict[str, Path] = {}
    for spec in specs:
        label, folder = parse_experiment_arg(spec)
        if not folder.exists():
            print(f"  ERROR {label}: folder not found: {folder}")
            continue
        counts_file = find_countscompare(folder)
        if counts_file is None:
            print(f"  SKIP {label}: no countscompare.txt under {folder}")
            continue
        if label in loaded:
            # Two runs sharing a folder name would otherwise silently collapse
            # into one series.
            label = f"{label} ({len(loaded)})"
        loaded[label] = counts_file
        print(f"  {label}: {counts_file}")
    return loaded


def default_output_dir(specs: list[str]) -> Path:
    """Where figures go when --out-dir is not given.

    A single run writes into its own ``evaluation/`` folder, next to the other
    per-run figures, which is what lets the pipeline call this per run and have
    the report pick the output up. Comparing several runs has no single home,
    so those land in ``experiments/_figures`` beside the runs themselves.
    """
    if len(specs) == 1:
        _, folder = parse_experiment_arg(specs[0])
        return folder / "evaluation"
    _, first = parse_experiment_arg(specs[0])
    parent = first.resolve().parent
    return parent / "_figures"


def main():
    parser = argparse.ArgumentParser(
        description="Log-log observed-vs-simulated count comparison across experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Each EXPERIMENT is a run folder, optionally prefixed with a "
               "legend label: 'anchor=experiments/exp_B'. Without a label the "
               "folder name is used.",
    )
    parser.add_argument(
        "experiments", nargs="+", metavar="EXPERIMENT",
        help="Experiment folder(s), each optionally 'LABEL=path'.",
    )
    parser.add_argument(
        "--format", choices=["pdf", "png"], default="pdf",
        help="Figure format. PDF for papers (default); PNG for the run report, "
             "which embeds images and cannot show a PDF.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Where to write figures. Default: the run's own evaluation/ "
             "folder for a single experiment, else experiments/_figures.",
    )
    parser.add_argument(
        "--mode", choices=["overlay", "separate"], default="overlay",
        help="overlay = all experiments on one figure per hour; "
             "separate = one figure per experiment per hour (default: overlay).",
    )
    parser.add_argument(
        "--hours", type=int, nargs="+", default=DEFAULT_HOURS,
        help=f"MATSim hour indices to plot (default: {DEFAULT_HOURS}).",
    )
    parser.add_argument(
        "--sum-directions", action="store_true",
        help="Sum per-direction stations into one point per physical sensor. "
             "Default: plot each direction separately.",
    )
    parser.add_argument(
        "--connect", action="store_true",
        help="Overlay mode only: join the same station's points across "
             "experiments with a line (green=moved toward 1:1, red=away) so you "
             "can see per-station improvement. Matches on full station id.",
    )
    args = parser.parse_args()

    if args.connect and args.mode != "overlay":
        print("  NOTE: --connect only applies to --mode overlay; ignoring it.")
        args.connect = False

    loaded = resolve_experiments(args.experiments)
    if not loaded:
        print("No experiments with countscompare data found. Exiting.")
        raise SystemExit(1)

    out_dir = args.out_dir or default_output_dir(args.experiments)
    agg = "summed per station" if args.sum_directions else "per direction"
    print(f"\nMode: {args.mode} | hours: {args.hours} | stations: {agg}")
    print(f"Output: {out_dir}\n")

    if args.mode == "overlay":
        generate_overlay(loaded, args.hours, args.sum_directions, out_dir,
                         connect=args.connect, fmt=args.format)
    else:
        generate_separate(loaded, args.hours, args.sum_directions, out_dir,
                          fmt=args.format)

    print("\nDone.")


if __name__ == "__main__":
    main()
