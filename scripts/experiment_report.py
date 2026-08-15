#!/usr/bin/env python3
"""Single-file experiment report: Markdown -> styled HTML -> PDF.

Collects what is otherwise scattered across experiment_summary.json, the
evaluation/ folder and the MATSim output into one document:

  - a verdict block: the four headline metrics against their thresholds
  - the run's configuration and agent counts
  - a plain-language reading of what the numbers mean
  - the evaluation figures, grouped into sections, two per row, each with a
    caption saying what to look for

Three artifacts are written next to each other, so you can use whichever fits:
  report.md    the source — readable, diffable, greppable
  report.html  styled, opens in any browser
  report.pdf   rendered via headless Edge/Chrome (Windows/macOS/Linux)

The PDF step needs no extra Python package: it drives an installed
Edge/Chrome with --headless --print-to-pdf. If no browser is found the
Markdown and HTML are still written and the PDF step is skipped with a note.

Usage:
    python scripts/experiment_report.py experiments/experiment_20260812_233102
    python scripts/experiment_report.py <exp> --baseline experiments/<other>
    python scripts/experiment_report.py <exp> --no-pdf      # md + html only
"""
from __future__ import annotations

import argparse
import base64
import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


# ---------------------------------------------------------------------------
# Tolerances. Every band the report uses to call a number "good" or flag it is
# defined here, so none is buried in the prose below and all can be overridden
# per region.
#
# These are working targets, not universal constants. Only two have a claim to
# being industry standards: GEH < 5 on individual hourly link counts (target
# >=85% of counts) and +/-10% on screenline totals. The rest are this project's
# judgement, chosen against US metros — a region with genuinely different travel
# behaviour may need different bands, which is why they are overridable.
#
# Override per region in config under ``evaluation.report_tolerances``:
#     "evaluation": { "report_tolerances": { "trips_per_person_band": 0.15 } }
# ---------------------------------------------------------------------------
TOLERANCES: Dict[str, float] = {
    # Ratio bands: |value - 1.0| within this counts as "about right".
    "volume_ratio_band": 0.10,       # aggregate sim/obs, iqr_mean
    "trips_per_person_band": 0.10,   # demand generation vs survey
    "trip_length_band": 0.10,        # median/mean trip km vs survey
    # Per-station spread below this means the error is uniform enough that a
    # single global lever (scaling_factor) is a valid correction.
    "station_cv_uniform": 0.35,
    "correlation_min": 0.85,
    # Mode share difference, in percentage points, worth flagging.
    "mode_share_flag_pp": 2.0,
    # Network distance exceeds the straight-line or reported distance a survey
    # records, because a vehicle follows the road rather than the crow. The
    # simulated side is MATSim ``traveled_distance`` (on-network) and the survey
    # side is its own reported distance, so the raw ratio of the two is not a
    # like-for-like comparison. This factor is the expected network/reported
    # ratio for a trip of correct length; the report divides the raw ratio by it
    # before grading. 1.0 disables the correction.
    "distance_detour_factor": 1.25,
    # Percent of hourly link counts scoring GEH < 5. The one genuinely
    # standard target here, and it applies to the pass rate rather than to any
    # average: a GEH is defined for one count over one period.
    "geh_lt5_pct_min": 85.0,
}

# Retired thresholds, kept named so a config that still sets them is ignored
# rather than silently treated as a live band:
#   station_daily_geh_max  graded an average of per-station GEH values
#   hourly_spread_max      a hand-chosen cut on the hourly ratio spread
# Both were this project's inventions rather than established criteria. The
# hourly ratio spread is still reported, just not graded pass/fail.
_RETIRED_TOLERANCES = frozenset({"station_daily_geh_max", "hourly_spread_max"})


def load_tolerances(summary: Dict[str, Any]) -> Dict[str, float]:
    """Merge any per-region overrides from the run's config over the defaults."""
    out = dict(TOLERANCES)
    overrides = _dig(summary, ["parameters", "evaluation", "report_tolerances"]) or {}
    if not isinstance(overrides, dict):
        return out
    for key, value in overrides.items():
        if key in _RETIRED_TOLERANCES:
            # Silently ignored rather than an error: an existing region config
            # may still carry one of these, and it should not fail the report.
            continue
        if key in out and isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _thresholds(tol: Dict[str, float]) -> List[Tuple[str, str, str, float, str]]:
    """Metric table rows, built from the active tolerances.

    The last field defines the metric — what the number is, in one or two
    sentences — rather than telling the reader what to conclude from it. A
    verdict that holds in one metro can be wrong in the next, so the wording
    stays regional-neutral and lets the value speak.
    """
    return [
    ("aggregate_sim_obs_ratio", "Overall volume (sim/obs)", "~1", tol["volume_ratio_band"],
     "Total simulated vehicles divided by total observed vehicles, summed over "
     "every count station and hour. 1.0 means the totals match."),
    ("station_ratio_cv", "Per-station ratio CV", "<", tol["station_cv_uniform"],
     "CV is the coefficient of variation: the standard deviation of the "
     "per-station sim/obs ratios divided by their mean. Near 0 means every "
     "station is off by the same factor; larger means the stations differ."),
    # The one industry-standard criterion in this table. It grades the pass
    # RATE — each hourly count tested against GEH < 5 on its own — rather than
    # an average of GEH values, which would measure station size as much as
    # model error.
    ("geh_lt_5_pct", "% hourly counts GEH < 5", ">", tol["geh_lt5_pct_min"],
     "GEH is a volume-aware error statistic that tolerates a larger absolute "
     "gap on a busy link than a quiet one. This is the share of individual "
     "station-hours scoring under 5, the usual acceptance threshold."),
    ("correlation", "Correlation", ">", tol["correlation_min"],
     "Pearson correlation between observed and simulated volume across all "
     "station-hours. It measures whether the pattern rises and falls together, "
     "not whether the levels agree."),
    ("interquartile_mean_ratio", "Volume level (iqr_mean)", "~1", tol["volume_ratio_band"],
     "The mean sim/obs ratio over the middle 50% of stations, after dropping "
     "the highest and lowest quarter. It gives the typical station, so a few "
     "extreme stations cannot move it."),
]


# One or two sentences for each name the report uses that a reader may not
# know. Rendered once, as small print under the section that first uses them,
# so no metric name appears without a definition within reach.
GLOSSARY: List[Tuple[str, str]] = [
    ("sim/obs", "Simulated volume divided by observed volume. Above 1.0 the "
                "model puts more vehicles on the link than the counter recorded; "
                "below 1.0, fewer."),
    ("Count station", "A roadside sensor that records how many vehicles pass. "
                      "Most report each direction separately, so one physical "
                      "station usually yields two directional counts."),
    ("Station-hour", "One count station in one hour of the day. A station with "
                     "24 hours of data contributes 24 station-hours."),
    ("GEH", "An error statistic used for traffic counts. It scales with volume, "
            "so the same percentage error scores worse on a busy link than a "
            "quiet one. Under 5 is the usual acceptance threshold for an hourly "
            "count."),
    ("CV", "Coefficient of variation — the standard deviation divided by the "
           "mean. It expresses spread relative to size, so values from stations "
           "of different volume can be compared."),
    ("iqr_mean", "Interquartile mean. The highest and lowest quarter of values "
                 "are dropped and the middle half is averaged, which keeps a "
                 "few extreme stations from moving the result."),
    ("p10 / p90", "The 10th and 90th percentile. 10% of stations fall below the "
                  "p10 value and 10% rise above the p90 value."),
    ("MAE / RMSE", "Mean absolute error and root mean square error, both in "
                   "vehicles per hour. RMSE weights large misses more heavily, "
                   "so RMSE far above MAE means the error sits in a few big "
                   "misses rather than spread evenly."),
    ("pp", "Percentage points — the arithmetic difference between two "
           "percentages. 92% against 87% is a gap of 5 pp."),
]

# Survey names, expanded where the report cites one. Keyed by the lowercase
# prefix of the survey's own label, so a region that loads a different survey
# simply contributes no entry rather than showing a wrong expansion.
SURVEY_GLOSSARY: Dict[str, str] = {
    "nhts": "The National Household Travel Survey, a federal survey of US "
            "household travel. It describes the country as a whole, not any "
            "one metro.",
    "tbi": "A Travel Behavior Inventory — a household travel survey run by a "
           "metropolitan planning organisation for its own region.",
}


def _glossary_block(terms: List[Tuple[str, str]]) -> List[str]:
    """Definitions as small print — present but out of the way of the numbers."""
    if not terms:
        return []
    out = ['<div class="glossary">']
    for term, text in terms:
        out.append(f"<p><strong>{term}</strong> — {text}</p>")
    out.append("</div>")
    out.append("")
    return out


# Figures grouped into sections. Captions say what the reader should look for.
FIGURE_SECTIONS: List[Tuple[str, str, List[Tuple[str, str]]]] = [
    # The "Spatial Distribution" section was removed, both figures with it.
    #
    # spatial_overview.png coloured each station by the GEH of its 24-hour
    # TOTAL. That is a real GEH of a real count, but a daily total is not the
    # quantity the GEH<5 convention is about, and collapsing a station's day
    # into one colour hides when it goes wrong — a station can be badly over
    # at the peak and under at midday and still show one mid-range colour.
    # The informative view is GEH per hour per station, which the per-station
    # device reports in evaluation/device_reports/ already carry in full.
    #
    # heatmap_daily.png was captioned "daily volume across the network", but
    # its renderer (_plot_traffic_heatmap_clean) is documented as a plain grey
    # network background and never reads the total_volume passed to it. The
    # figure is the road layout and the county boundary — the name and caption
    # promised a heatmap the code was never written to draw.
    ("Error Through the Day",
     "The same two figures MATSim draws from its own counts output, rebuilt "
     "from the run's comparison table. Both are relative error, which is what "
     "MATSim reports throughout its counts dashboard.",
     [("hourly_relative_error_box.png",
       "Signed relative error per hour, one box per hour across all count "
       "stations. The block and hourly ratios elsewhere are computed on summed "
       "volumes, so they show only the region-wide level; this shows how much "
       "the individual stations disagree within the hour. A box straddling "
       "zero with long whiskers means over- and under-simulating stations are "
       "cancelling in the totals."),
      ("hourly_bias.png",
       "Mean |relative error| (red, left) against mean bias in vehicles "
       "(blue, right). The red line cannot cancel — it rises whenever "
       "stations are wrong in either direction. The blue line can, so where "
       "red stays high while blue sits near zero, the errors are offsetting "
       "rather than absent.")]),
    ("Peak Hours",
     "Peaks stress the network hardest and expose capacity problems that "
     "daily averages hide.",
     [("heatmap_8am_highways.png",
       "Simulated highway load at 08:00, coloured by volume."),
      ("heatmap_5pm_highways.png",
       "Simulated highway load at 17:00, on the same colour scale, so the two "
       "peaks can be compared directly.")]),
]


# The three per-hour views, rendered as one tab per hour. Same moment, three
# questions: where the error sits, whether the network was congested, and how
# the stations scatter against the counts.
HOUR_FIGURES: List[Tuple[str, str, str]] = [
    ("count_error_h{hh}.png", "Count station % error",
     "Each station-direction coloured by how far its simulated volume is from "
     "the count. Clusters of one colour mean a corridor problem; a scatter of "
     "both means the error is spread."),
    ("counts_loglog_h{hh}.png", "Observed vs simulated",
     "One point per station-direction on log-log axes, with the 1:1 line and "
     "the 2x / 0.5x bands. Points above the line over-simulate. This is the "
     "view that shows whether errors are proportional or concentrated at one "
     "end of the volume range."),
]

# Congestion heatmaps are deliberately NOT here. They exist only for 8 AM and
# 5 PM — the most expensive figure to draw and the least variable hour to hour
# — so including them would make two tabs different from the other 22. They
# keep their own "Peak Hours" section above, where two figures side by side is
# the point rather than an inconsistency.


def _hour_figure_name(pattern: str, hour: int) -> str:
    return pattern.format(hh=f"{hour:02d}")


# Counter making each lightbox anchor unique. The same figure can legitimately
# appear more than once in a report, and duplicate ids would make every copy
# open the first one's overlay.
_figure_seq = itertools.count()


def _figure(src: str, alt: str, caption: str) -> str:
    """A figure whose image opens full-size when clicked.

    CSS-only, using :target — clicking the image jumps to the figure's own id,
    which promotes that same <img> to a full-screen overlay; clicking again
    returns to '#_'. No JavaScript, so this survives the report being opened
    straight off disk with no server, and without CSS it degrades to a plain
    inline image.

    The image is emitted ONCE. A separate overlay copy would be simpler to
    write but doubles the file: these are base64 data URIs, so a second copy of
    every figure took the Birmingham report from 5.4 MB to 10.8 MB for pixels
    already present on the page.
    """
    fid = f"fig{next(_figure_seq)}"
    return (
        f'<figure id="{fid}" class="zoomable">'
        f'<a class="zoom-in" href="#{fid}"><img src="{src}" alt="{alt}"/></a>'
        f'<a class="zoom-out" href="#_"></a>'
        f'<figcaption>{caption}</figcaption>'
        f'</figure>'
    )


def build_hour_tabs(eval_dir: Path, embed_dir: Optional[Path]) -> Tuple[List[str], set]:
    """Per-hour figures as CSS-only tabs, with a plain-Markdown fallback.

    Radio inputs drive the tabs, so the HTML needs no JavaScript — it works
    offline, when printed to PDF, and under a strict CSP. Readers of report.md
    see a plain heading per hour instead, which keeps the Markdown greppable.

    Returns the lines and the set of figure filenames consumed.
    """
    hours: List[int] = []
    for h in range(24):
        if any((eval_dir / _hour_figure_name(p, h)).is_file()
               for p, _, _ in HOUR_FIGURES):
            hours.append(h)
    if not hours:
        return [], set()

    used: set = set()
    L: List[str] = []
    L.append("## Hour by hour")
    L.append("")
    L.append("Three views of the same hour. Pick an hour to see where the error "
             "sat, whether the network was congested, and how the stations "
             "scattered against the counts.")
    L.append("")
    L.append('<div class="hour-tabs">')

    # Radio inputs first: a CSS sibling selector can only style elements that
    # come after the checked input, so every input must precede every panel.
    for i, h in enumerate(hours):
        checked = " checked" if i == 0 else ""
        L.append(f'<input type="radio" name="hourtab" id="hour-{h:02d}"{checked}/>')
    L.append('<div class="hour-labels">')
    for h in hours:
        L.append(f'<label for="hour-{h:02d}">{h:02d}</label>')
    L.append("</div>")

    for h in hours:
        present = [(p, lbl, cap) for p, lbl, cap in HOUR_FIGURES
                   if (eval_dir / _hour_figure_name(p, h)).is_file()]
        # Plain HTML for the figures, not Markdown-in-HTML: md_in_html does not
        # reliably recurse into a <figure> nested inside a <div>, and when it
        # silently fails the images are emitted as literal text — the HTML
        # loses every per-hour figure while report.md still looks correct.
        # An HTML comment carries the hour for anyone grepping the Markdown.
        L.append(f'<div class="hour-panel" id="panel-{h:02d}">')
        L.append(f'<h3>{h:02d}:00</h3>')
        # Column count follows what this hour actually has, so an hour without
        # a congestion heatmap does not leave a hole in the grid.
        L.append(f'<div class="fig-grid cols-{len(present)}">')
        for pattern, label, caption in present:
            name = _hour_figure_name(pattern, h)
            used.add(name)
            src = _image_src(eval_dir / name, embed_dir)
            L.append(_figure(src, name,
                             f"<strong>{label}.</strong> {caption}"))
        L.append("</div>")
        L.append("</div>")
    L.append("</div>")
    L.append("")
    return L, used


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _dig(d: Any, path: List[str]) -> Any:
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def load_run(exp_dir: Path) -> Dict[str, Any]:
    summary_file = exp_dir / "experiment_summary.json"
    if not summary_file.is_file():
        raise SystemExit(f"ERROR: no experiment_summary.json in {exp_dir}")
    with open(summary_file, encoding="utf-8") as f:
        summary = json.load(f)
    evaluation = summary.get("evaluation") or {}
    if not evaluation:
        legacy = exp_dir / "evaluation" / "summary_metrics.json"
        if legacy.is_file():
            with open(legacy, encoding="utf-8") as f:
                evaluation = json.load(f)
    return {"summary": summary, "evaluation": evaluation, "dir": exp_dir}


def aggregate_ratio(exp_dir: Path) -> Optional[float]:
    """Total simulated / total observed, from the per-station comparison."""
    path = exp_dir / "evaluation" / "volume_comparison.csv"
    if not path.is_file():
        return None
    sim = obs = 0.0
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for rec in csv.DictReader(f):
                try:
                    o = float(rec.get("observed") or 0)
                    s = float(rec.get("simulated") or 0)
                except (TypeError, ValueError):
                    continue
                if o > 0:
                    obs += o
                    sim += s
    except OSError:
        return None
    return sim / obs if obs > 0 else None


def _passes(value: Optional[float], comp: str, threshold: float) -> Optional[bool]:
    if value is None:
        return None
    if comp == "<":
        return value < threshold
    if comp == ">":
        return value > threshold
    if comp == "~1":
        return abs(value - 1.0) <= threshold
    return None


def _reading(ev: Dict[str, Any], tol: Dict[str, float]) -> List[str]:
    """What each headline number says, stated as a measurement.

    Deliberately descriptive rather than prescriptive. An earlier version named
    a cause and a config lever for each pattern ("a global multiplier would drag
    the already-correct stations off target"), but those inferences hold only
    under assumptions that vary by region — network coverage, how the counts
    were matched, whether external traffic is modelled. Read in another metro
    they were confidently wrong, so the wording now reports what was measured
    and leaves the diagnosis to the reader, who knows their own region.

    Every band still comes from *tol*, so a region that widens a band sees the
    wording follow.
    """
    band = tol["volume_ratio_band"]
    out: List[str] = []
    cv = ev.get("station_ratio_cv")
    iqr = ev.get("interquartile_mean_ratio")
    geh_lt5 = ev.get("geh_lt_5_pct")

    # 1. Level: how the totals compare, and how that splits across stations.
    agg = ev.get("aggregate_sim_obs_ratio")
    over = ev.get("stations_over_simulated")
    under = ev.get("stations_under_simulated")
    ok = ev.get("stations_within_10pct")
    if agg is not None and agg > 0:
        n = sum(v for v in (over, under, ok) if isinstance(v, (int, float)))
        split = ""
        if n:
            # The station split uses fixed 0.9/1.1 cutoffs in the evaluator, so
            # the wording states them literally rather than echoing the
            # configurable band, which they do not follow.
            split = (f" Of {n} station-directions, {over} are above 1.1, "
                     f"{ok} are between 0.9 and 1.1, and {under} are below 0.9.")
        direction = ("above" if agg > 1.0 + band
                     else "below" if agg < 1.0 - band else "within")
        if direction == "within":
            out.append(f"**Total volume is {agg:.3f}x observed**, inside the "
                       f"±{band:.0%} band.{split}")
        else:
            out.append(f"**Total volume is {agg:.3f}x observed**, {direction} the "
                       f"±{band:.0%} band.{split}")

    # 2. Shape: the same ratio computed within each part of the day.
    blocks = {k: ev.get(f"ratio_{k}") for k in ("night", "morning", "midday", "evening")}
    if all(v is not None and v > 0 for v in blocks.values()):
        worst_name = max(blocks, key=lambda k: abs(blocks[k] - 1.0))
        detail = (f"night {blocks['night']:.2f}, morning {blocks['morning']:.2f}, "
                  f"midday {blocks['midday']:.2f}, evening {blocks['evening']:.2f}")
        if abs(blocks[worst_name] - 1.0) > 2 * band:
            out.append(
                f"**The ratio differs by time of day** ({detail}). The block furthest "
                f"from 1.0 is *{worst_name}*, at {blocks[worst_name]:.2f}.")
        else:
            out.append(f"**The ratio is similar across the day** ({detail}).")

    # 3. Spread: whether the stations agree with each other.
    if cv is not None:
        p10, p90 = ev.get("station_ratio_p10"), ev.get("station_ratio_p90")
        range_note = (f" The middle 80% of stations run from {p10:.2f} to {p90:.2f}."
                      if p10 is not None and p90 is not None else "")
        comparison = "below" if cv < tol["station_cv_uniform"] else "above"
        out.append(
            f"**Per-station ratios have a CV of {cv:.3f}**, {comparison} the "
            f"{tol['station_cv_uniform']:.2f} threshold.{range_note}")

    if iqr is not None:
        pct = abs(iqr - 1.0) * 100
        if abs(iqr - 1.0) <= band:
            out.append(f"**The typical station is at {iqr:.3f}x observed** "
                       f"(iqr_mean), inside the ±{band:.0%} band.")
        else:
            side = "above" if iqr > 1.0 else "below"
            out.append(f"**The typical station is at {iqr:.3f}x observed** "
                       f"(iqr_mean), {pct:.0f}% {side} the counts.")

    if geh_lt5 is not None:
        target = tol["geh_lt5_pct_min"]
        verdict = "meets" if geh_lt5 >= target else "is short of"
        out.append(
            f"**{geh_lt5:.1f}% of station-hours score GEH < 5**, which {verdict} the "
            f"{target:.0f}% target.")
    return out


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _fmt(value: Any, decimals: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.{decimals}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def build_markdown(run: Dict[str, Any], baseline: Optional[Dict[str, Any]],
                   embed_dir: Optional[Path]) -> str:
    summary, ev, exp_dir = run["summary"], run["evaluation"], run["dir"]
    params = summary.get("parameters", {})
    mo = summary.get("matsim_output", {})
    base_ev = baseline["evaluation"] if baseline else {}
    # Bands come from this run's config where it overrides them, so the same
    # code produces region-appropriate wording rather than US-tuned constants.
    tol = load_tolerances(summary)

    L: List[str] = []
    L.append(f"# Experiment Report")
    L.append("")
    L.append(f"**`{exp_dir.name}`**  ·  created {summary.get('created_at', 'unknown')}")
    if baseline:
        L.append(f"  ·  compared against `{baseline['dir'].name}`")
    L.append("")

    # ---- headline tiles ------------------------------------------------
    L.append('<div class="kpi-row">')
    for key, label, comp, threshold, _note in _thresholds(tol):
        value = ev.get(key)
        ok = _passes(value, comp, threshold)
        cls = "kpi ok" if ok else ("kpi warn" if ok is False else "kpi na")
        shown = "—" if value is None else f"{value:.3f}"
        status = "PASS" if ok else ("CHECK" if ok is False else "n/a")
        L.append(f'<div class="{cls}"><div class="kpi-value">{shown}</div>'
                 f'<div class="kpi-label">{label}</div>'
                 f'<div class="kpi-status">{status}</div></div>')
    L.append("</div>")
    L.append("")

    # ---- verdict -------------------------------------------------------
    L.append("## Verdict")
    L.append("")
    for line in _reading(ev, tol):
        L.append(f"- {line}")
    L.append("")
    L.append("> Totals and spread are separate questions. Over- and under-simulating "
             "stations cancel in the total, so a ratio near 1.0 is consistent with "
             "both a uniformly accurate model and a widely scattered one. The CV "
             "and the p10/p90 range distinguish them.")
    L.append("")

    # ---- metrics table -------------------------------------------------
    # ---- demand validation ---------------------------------------------
    # Demand comes before counts: counts validate how demand was assigned to
    # the network, so reading them first invites blaming assignment for a
    # demand defect (or the reverse).
    dv = summary.get("demand_validation") or {}
    surveys = (dv.get("surveys") or {}) if dv else {}
    if surveys:
        # Column order: local surveys first. For its own metro a local
        # household travel survey is the authoritative reference; the national
        # one is context, and the gap between them is a property of the metro
        # rather than a model error.
        names = sorted(surveys, key=lambda n: (not surveys[n].get("is_local"), n))
        primary = dv.get("primary_survey") or names[0]
        multi = len(names) > 1

        L.append("## Demand validation (vs household survey)")
        L.append("")
        L.append("Counts validate how demand is **assigned** to the network. This "
                 "validates the **demand itself** — how many trips, by what mode, "
                 "how far, and when.")
        L.append("")
        for n in names:
            e = surveys[n]
            kind = "local survey" if e.get("is_local") else "national survey"
            mark = " — **reference**" if n == primary and multi else ""
            L.append(f"- **{e.get('label', n)}** ({kind}, "
                     f"{_fmt(e.get('survey_trips'), 0)} trips){mark}")
        if multi:
            L.append("")
            L.append("Both are shown because absolute levels depend on how each survey "
                     "defines and counts a trip, so their figures are not "
                     "interchangeable. Where they disagree, the local survey describes "
                     "this metro and the national one describes the country.")
        elif not surveys[names[0]].get("is_local"):
            L.append("")
            L.append("No local household travel survey is loaded for this region. A "
                     "metro-specific survey would be the better reference — add one to "
                     "`data.surveys` with `weight > 0`.")
        L.append("")
        # Expand whichever survey acronyms this run actually cites. A region
        # loading a survey with no entry simply contributes nothing here.
        survey_terms: List[Tuple[str, str]] = []
        for n in names:
            label = str(surveys[n].get("label", n)).lower()
            for prefix, text in SURVEY_GLOSSARY.items():
                if label.startswith(prefix):
                    term = prefix.upper()
                    if not any(t == term for t, _ in survey_terms):
                        survey_terms.append((term, text))
        L.extend(_glossary_block(survey_terms))

        def _row(label: str, key: str, decimals: int = 2) -> str:
            cells = "".join(f" {_fmt(surveys[n].get(key), decimals)} |" for n in names)
            return f"| {label} |{cells}"

        head = "| Quantity |" + "".join(f" {surveys[n].get('label', n)} |" for n in names)
        div = "|---|" + "---:|" * len(names)
        L.append("### Survey values")
        L.append("")
        L.append(head)
        L.append(div)
        L.append(_row("Trips per person per day", "trips_per_person_survey"))
        L.append(_row("Trip length, median km", "survey_median_km"))
        L.append(_row("Trip length, mean km", "survey_mean_km"))
        L.append("")

        sim_e = surveys[primary]
        detour = tol["distance_detour_factor"]
        L.append("### Simulated vs each survey")
        L.append("")
        L.append("Ratio is simulated / survey. The simulated value is the same in "
                 "every column — only the reference changes.")
        L.append("")
        L.append("| Quantity | Simulated |" +
                 "".join(f" vs {surveys[n].get('label', n)} |" for n in names))
        L.append("|---|---:|" + "---:|" * len(names))
        for label, sim_key, ratio_key in (
                ("Trips per person per day", "trips_per_person_simulated",
                 "trips_per_person_ratio"),
                ("Trip length, median km", "simulated_median_km", "median_km_ratio"),
                ("Trip length, mean km", "simulated_mean_km", "mean_km_ratio")):
            cells = "".join(f" **{_fmt(surveys[n].get(ratio_key), 2)}** |" for n in names)
            L.append(f"| {label} | {_fmt(sim_e.get(sim_key), 2)} |{cells}")
        L.append("")
        if detour and detour != 1.0:
            L.append(f"The two sides measure distance differently: the simulated "
                     f"figure is distance travelled along the network, the survey "
                     f"figure is the distance the survey itself records. A trip of "
                     f"the correct length therefore reads about {detour:.2f}x on "
                     f"this table. The row below divides the ratios by "
                     f"{detour:.2f} before comparing them with the band.")
            L.append("")

        # ---- demand readings ------------------------------------------
        # All three quantities are graded, not only trip count. Trips and
        # length multiply into vehicle-km, so a run can match trips per person
        # exactly and still put far too much traffic on the network. An earlier
        # version tested trips_per_person_ratio alone and reported demand as
        # sound while both length ratios were well outside their band.
        dband = tol["trips_per_person_band"]
        lband = tol["trip_length_band"]
        ref_note = f" (against {sim_e.get('label', primary)})" if multi else ""
        tpp_r = sim_e.get("trips_per_person_ratio")
        med_r = sim_e.get("median_km_ratio")
        mean_r = sim_e.get("mean_km_ratio")

        def _adj(raw: Optional[float]) -> Optional[float]:
            """Ratio corrected for the network/reported distance difference."""
            if raw is None or not detour:
                return raw
            return raw / detour

        notes: List[str] = []
        if tpp_r is not None:
            state = ("within" if abs(tpp_r - 1.0) <= dband
                     else "above" if tpp_r > 1.0 else "below")
            notes.append(f"**Trips per person: {tpp_r:.2f}{ref_note}** — {state} the "
                         f"±{dband:.0%} band.")
        adj_med, adj_mean = _adj(med_r), _adj(mean_r)
        for label, raw, adj in (("Median trip length", med_r, adj_med),
                                ("Mean trip length", mean_r, adj_mean)):
            if raw is None or adj is None:
                continue
            state = ("within" if abs(adj - 1.0) <= lband
                     else "above" if adj > 1.0 else "below")
            shown = (f"{raw:.2f} raw, {adj:.2f} after the distance correction"
                     if detour and detour != 1.0 else f"{raw:.2f}")
            notes.append(f"**{label}: {shown}** — {state} the ±{lband:.0%} band.")

        # Vehicle-km is trips x length, so the two combine. Stating the product
        # keeps a reader from reading a passing trip count as demand being
        # settled when the length ratio moves the total a long way.
        if tpp_r is not None and adj_mean is not None:
            product = tpp_r * adj_mean
            notes.append(f"**Distance travelled per person: {product:.2f}** — trips "
                         f"per person multiplied by the corrected mean length.")
        for note in notes:
            L.append(f"- {note}")
        if notes:
            L.append("")

        # Mode share: one column pair per survey.
        all_modes: List[str] = []
        for n in names:
            for mode in (surveys[n].get("mode_share") or {}):
                if mode not in all_modes:
                    all_modes.append(mode)
        if all_modes:
            L.append("### Mode share")
            L.append("")
            L.append("Every transit sub-mode is `pt` on both sides — MATSim writes "
                     "only `pt`, so each survey is collapsed the same way.")
            L.append("")
            L.append("| Mode | Simulated % |" +
                     "".join(f" {surveys[n].get('label', n)} % | Diff (pp) |" for n in names))
            L.append("|---|---:|" + "---:|---:|" * len(names))
            for mode in all_modes:
                sim_pct = ((surveys[primary].get("mode_share") or {})
                           .get(mode, {}).get("simulated_pct"))
                cells = ""
                for n in names:
                    row = (surveys[n].get("mode_share") or {}).get(mode, {})
                    diff = row.get("diff_pp")
                    flag = (" ⚠" if diff is not None
                            and abs(diff) >= tol["mode_share_flag_pp"] else "")
                    cells += (f" {_fmt(row.get('survey_pct'), 2)} |"
                              f" **{_fmt(diff, 2)}**{flag} |")
                L.append(f"| {mode} | {_fmt(sim_pct, 2)} |{cells}")
            L.append("")

        # Departure profile against each survey.
        if surveys[primary].get("departure_blocks"):
            L.append("### Departure times vs survey")
            L.append("")
            L.append("| Block | Hours | Simulated % |" +
                     "".join(f" {surveys[n].get('label', n)} % | Diff (pp) |" for n in names))
            L.append("|---|---|---:|" + "---:|---:|" * len(names))
            for i, b in enumerate(surveys[primary]["departure_blocks"]):
                cells = ""
                for n in names:
                    blocks_n = surveys[n].get("departure_blocks") or []
                    bn = blocks_n[i] if i < len(blocks_n) else {}
                    cells += (f" {_fmt(bn.get('survey_pct'), 2)} |"
                              f" **{_fmt(bn.get('diff_pp'), 2)}** |")
                L.append(f"| {b.get('block')} | {b.get('hours')} "
                         f"| {_fmt(b.get('simulated_pct'), 2)} |{cells}")
            L.append("")
            L.append("> This table is when trips **start**. The Time of day table "
                     "under Count validation is when vehicles **pass a counter**, "
                     "which a trip does some way into its journey. The two describe "
                     "different moments and need not agree.")
            L.append("")

    L.append("## Count validation")
    L.append("")
    header = "| Metric | Value |"
    divide = "|---|---:|"
    if baseline:
        header += " Baseline |"
        divide += "---:|"
    header += " Target | Status | What the number is |"
    divide += "---|:--:|---|"
    L.append(header)
    L.append(divide)
    for key, label, comp, threshold, note in _thresholds(tol):
        value = ev.get(key)
        ok = _passes(value, comp, threshold)
        target = f"{comp} {threshold}" if comp != "~1" else f"1.0 ± {threshold:.2f}"
        status = "**PASS**" if ok else ("**CHECK**" if ok is False else "n/a")
        row = f"| {label} | **{_fmt(value)}** |"
        if baseline:
            row += f" {_fmt(base_ev.get(key))} |"
        row += f" `{target}` | {status} | {note} |"
        L.append(row)
    L.append("")

    # ---- time of day ---------------------------------------------------
    # A single daily ratio can look fine while individual parts of the day are
    # badly wrong in opposite directions, so the profile gets its own table.
    blocks = [("Night", "night", "0-3"), ("Morning", "morning", "4-9"),
              ("Midday", "midday", "10-17"), ("Evening", "evening", "18-23")]
    if any(ev.get(f"ratio_{k}") for _, k, _ in blocks):
        L.append("### Time of day")
        L.append("")
        L.append("The same sim/obs ratio, computed within each part of the day. "
                 "The blocks divide all 24 hours with no overlap, so their "
                 "volumes sum to the daily totals.")
        L.append("")
        head = "| Block | Hours | Observed | Simulated | Sim/Obs |"
        div = "|---|---|---:|---:|---:|"
        if baseline:
            head += " Baseline |"
            div += "---:|"
        L.append(head)
        L.append(div)
        for label, key, hours in blocks:
            ratio = ev.get(f"ratio_{key}")
            obs = ev.get(f"observed_{key}")
            sim = ev.get(f"simulated_{key}")
            flag = ("" if ratio is None
                    or abs(ratio - 1.0) <= tol["volume_ratio_band"] else " ⚠")
            row = (f"| {label} | {hours} | {_fmt(obs, 0)} | {_fmt(sim, 0)} | "
                   f"**{_fmt(ratio, 2)}**{flag} |")
            if baseline:
                row += f" {_fmt(base_ev.get(f'ratio_{key}'), 2)} |"
            L.append(row)
        L.append("")
        # The worst single hour is deliberately NOT reported here. With a night
        # block near zero it is almost always the same fact as the hourly ratio
        # spread, stated twice in different words, and the hourly error figures
        # show the whole 24-hour profile rather than one extracted hour.

    # ---- supporting numbers -------------------------------------------
    # Only what is not already stated above or drawn in a figure:
    #   * "Aggregate sim/obs" repeated the headline metric, computed a second
    #     way (from volume_comparison.csv, which drops zero-observed rows). Two
    #     near-identical numbers under different names invite the reader to
    #     treat them as independent evidence.
    #   * The three station-split counts are now in the verdict, in a sentence
    #     that also states the 0.9/1.1 cutoffs they use.
    #   * "Hourly ratio spread" duplicated the time-of-day table and the
    #     hourly error figures, both of which show the profile itself.
    #   * "% stations with daily GEH < 10" is a strictly weaker test than the
    #     < 5 row beside it, so it can never disagree in a way worth acting on.
    L.append("### Supporting numbers")
    L.append("")
    L.append("Detail not shown above or in the figures.")
    L.append("")
    rows: List[Tuple[str, Any, Any]] = [
        ("Physical stations", ev.get("num_devices"), base_ev.get("num_devices")),
        ("Station-directions", ev.get("num_directional_counts"),
         base_ev.get("num_directional_counts")),
        ("Station-hours compared", ev.get("num_comparisons"),
         base_ev.get("num_comparisons")),
        ("Ratio p10", ev.get("station_ratio_p10"), base_ev.get("station_ratio_p10")),
        ("Ratio p90", ev.get("station_ratio_p90"), base_ev.get("station_ratio_p90")),
        ("% stations with daily GEH < 5", ev.get("station_daily_geh_lt5_pct"),
         base_ev.get("station_daily_geh_lt5_pct")),
        ("Stations under 10% of observed", ev.get("num_stations_below_10pct"),
         base_ev.get("num_stations_below_10pct")),
        ("MAE (vehicles/hour)", ev.get("mae"), base_ev.get("mae")),
        ("RMSE (vehicles/hour)", ev.get("rmse"), base_ev.get("rmse")),
    ]
    L.append("| Metric | Value |" + (" Baseline |" if baseline else ""))
    L.append("|---|---:|" + ("---:|" if baseline else ""))
    for label, value, bvalue in rows:
        decimals = 1 if "%" in label else 3
        line = f"| {label} | {_fmt(value, decimals)} |"
        if baseline:
            line += f" {_fmt(bvalue, decimals)} |"
        L.append(line)
    L.append("")
    L.extend(_glossary_block(GLOSSARY))

    # ---- configuration -------------------------------------------------
    L.append("## Configuration")
    L.append("")
    L.append("| Parameter | Value |" + (" Baseline |" if baseline else ""))
    L.append("|---|---:|" + ("---:|" if baseline else ""))
    bparams = baseline["summary"].get("parameters", {}) if baseline else {}
    bmo = baseline["summary"].get("matsim_output", {}) if baseline else {}
    cfg_rows: List[Tuple[str, Any, Any]] = [
        ("scaling_factor", params.get("scaling_factor"), bparams.get("scaling_factor")),
        ("flowCapacityFactor", params.get("flow_capacity_factor"),
         bparams.get("flow_capacity_factor")),
        ("storageCapacityFactor", params.get("storage_capacity_factor"),
         bparams.get("storage_capacity_factor")),
        ("Iterations", params.get("iterations"), bparams.get("iterations")),
        ("Agents simulated", mo.get("output_persons_count"),
         bmo.get("output_persons_count")),
        ("Stuck agents", mo.get("total_stuck_agents"), bmo.get("total_stuck_agents")),
    ]
    for label, value, bvalue in cfg_rows:
        line = f"| {label} | {_fmt(value)} |"
        if baseline:
            line += f" {_fmt(bvalue)} |"
        L.append(line)
    L.append("")

    # ---- figures -------------------------------------------------------
    eval_dir = exp_dir / "evaluation"
    used: set[str] = set()
    for title, blurb, figures in FIGURE_SECTIONS:
        present = [(n, c) for n, c in figures if (eval_dir / n).is_file()]
        if not present:
            continue
        L.append(f"## {title}")
        L.append("")
        L.append(blurb)
        L.append("")
        L.append('<div class="fig-grid">')
        for name, caption in present:
            used.add(name)
            src = _image_src(eval_dir / name, embed_dir)
            L.append(_figure(src, name, caption))
        L.append("</div>")
        L.append("")

    # Per-hour figures, after the whole-day ones: read the day's shape first,
    # then drill into an hour.
    hour_lines, hour_used = build_hour_tabs(eval_dir, embed_dir)
    L.extend(hour_lines)
    used |= hour_used

    # Figures the evaluator still writes but the report deliberately omits.
    # Without this they would come back through the catch-all below, which
    # picks up every PNG not already placed in a section.
    suppressed = {"spatial_overview.png", "heatmap_daily.png"}
    extras = sorted(p for p in eval_dir.glob("*.png")
                    if p.name not in used and p.name not in suppressed) \
        if eval_dir.is_dir() else []
    if extras:
        L.append("## Other figures")
        L.append("")
        L.append('<div class="fig-grid">')
        for path in extras:
            src = _image_src(path, embed_dir)
            L.append(_figure(src, path.name, path.stem.replace("_", " ")))
        L.append("</div>")
        L.append("")

    L.append("---")
    L.append("")
    L.append("<small>Except for GEH &lt; 5 on individual hourly counts, thresholds are "
             "this project's working targets rather than universal standards, and are "
             "overridable per region under <code>evaluation.report_tolerances</code>. "
             "Correlation is measured on station-hours, so it is lower than the same "
             "statistic computed on daily totals. GEH is reported only as a pass rate "
             "and per station, never averaged, because GEH grows with volume and an "
             "average across stations of different size would measure station size as "
             "much as model error.</small>")
    L.append("")
    return "\n".join(L)


def _image_src(path: Path, embed_dir: Optional[Path]) -> str:
    """Image reference for the Markdown/HTML.

    Relative paths keep report.md readable and the files small — the figures
    already live next to it in evaluation/. Base64 data URIs are used only when
    *embed_dir* is set, which makes a single self-contained file at the cost of
    roughly a 4/3 size increase per image.
    """
    if embed_dir is None:
        return path.as_uri()
    try:
        return os.path.relpath(path, embed_dir).replace("\\", "/")
    except ValueError:
        return path.as_uri()


def _image_data_uri(path: Path, max_width: int = 1400) -> str:
    """Base64 data URI for an image, downscaled to a sensible report width.

    The evaluation maps are high-DPI (~3.4 MB each); embedded at full size the
    HTML runs to tens of megabytes for no visible benefit at print resolution.
    Pillow ships with matplotlib, so downscaling needs no extra dependency; if
    it is somehow unavailable the original bytes are embedded unchanged.
    """
    raw = path.read_bytes()
    try:
        import io
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as img:
            if img.width > max_width:
                height = round(img.height * max_width / img.width)
                img = img.convert("RGB").resize((max_width, height), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=86, optimize=True)
                return ("data:image/jpeg;base64,"
                        + base64.b64encode(buf.getvalue()).decode("ascii"))
    except Exception:
        pass
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# HTML + PDF
# ---------------------------------------------------------------------------

CSS = """
@page { size: A4; margin: 14mm 12mm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", -apple-system, Helvetica, Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.5; color: #1a1a1a;
  max-width: 980px; margin: 0 auto; padding: 12px 18px;
}
h1 { font-size: 25pt; font-weight: 650; margin: 0 0 2px; letter-spacing: -0.4px; }
h1 + p { color: #666; font-size: 9.5pt; margin: 0 0 18px; }
h2 {
  font-size: 14pt; font-weight: 620; margin: 26px 0 10px; padding-bottom: 5px;
  border-bottom: 2px solid #e3e6ea; page-break-after: avoid;
}
h3 { font-size: 11.5pt; font-weight: 600; margin: 18px 0 8px; page-break-after: avoid; }
code { background: #f4f5f7; padding: 1px 5px; border-radius: 3px; font-size: 9.2pt; }
blockquote {
  margin: 14px 0; padding: 10px 14px; background: #fffbe9;
  border-left: 3px solid #e0b400; color: #5a4a00; font-size: 9.8pt;
}
ul { padding-left: 20px; }
li { margin: 5px 0; }

/* KPI tiles */
.kpi-row { display: flex; gap: 10px; margin: 20px 0 6px; page-break-inside: avoid; }
.kpi {
  flex: 1; border: 1px solid #e0e3e8; border-top: 4px solid #9aa3ad;
  border-radius: 6px; padding: 12px 10px; text-align: center; background: #fcfcfd;
}
.kpi.ok   { border-top-color: #2e9e5b; }
.kpi.warn { border-top-color: #d9822b; }
.kpi.na   { border-top-color: #b9bfc7; }
.kpi-value { font-size: 24pt; font-weight: 680; letter-spacing: -1px; line-height: 1.1; }
.kpi-label { font-size: 8.4pt; color: #5a6470; margin-top: 3px; }
.kpi-status { font-size: 8pt; font-weight: 700; margin-top: 6px; letter-spacing: 0.4px; }
.kpi.ok .kpi-status   { color: #2e9e5b; }
.kpi.warn .kpi-status { color: #d9822b; }
.kpi.na .kpi-status   { color: #9aa3ad; }

/* tables */
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9.6pt; }
th {
  text-align: left; background: #f4f6f8; font-weight: 620; font-size: 8.8pt;
  text-transform: uppercase; letter-spacing: 0.4px; color: #4a5560;
  padding: 8px 9px; border-bottom: 2px solid #dfe3e8;
}
td { padding: 7px 9px; border-bottom: 1px solid #edf0f3; }
tr:nth-child(even) td { background: #fafbfc; }
tbody tr:hover td { background: #f2f7fd; }

/* figures */
.fig-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 14px 0;
}
figure {
  margin: 0; border: 1px solid #e3e6ea; border-radius: 6px; padding: 8px;
  background: #fff; page-break-inside: avoid;
}
figure img { width: 100%; height: auto; display: block; border-radius: 3px; }
figcaption { font-size: 8.4pt; color: #5a6470; margin-top: 7px; line-height: 1.4; }
small { color: #8a929b; font-size: 8.4pt; }

/* Definitions: present for a reader who needs them, quiet for one who does
   not. Smaller and greyer than body text so they never compete with the
   numbers they explain. */
.glossary {
  margin: 10px 0 18px; padding: 10px 14px; background: #fafbfc;
  border: 1px solid #edf0f3; border-radius: 5px;
  page-break-inside: avoid;
}
.glossary p {
  margin: 0 0 5px; font-size: 8.6pt; line-height: 1.45; color: #5a6470;
}
.glossary p:last-child { margin-bottom: 0; }
.glossary strong { color: #3a434e; }
hr { border: none; border-top: 1px solid #e3e6ea; margin: 22px 0 12px; }

/* ---- Hour tabs -------------------------------------------------------
   CSS only: radio inputs hold the state and a sibling selector reveals the
   matching panel, so the report needs no JavaScript and still works offline
   and under a strict CSP. The inputs are visually hidden but remain
   focusable, so the tabs stay keyboard-reachable.                        */
.hour-tabs > input[type="radio"] {
  position: absolute; opacity: 0; pointer-events: none;
}
.hour-labels {
  display: flex; flex-wrap: wrap; gap: 4px; margin: 10px 0 14px;
  border-bottom: 2px solid #e3e6ea; padding-bottom: 8px;
}
.hour-labels label {
  font-size: 9pt; font-variant-numeric: tabular-nums; cursor: pointer;
  padding: 4px 9px; border: 1px solid #d7dbe0; border-radius: 4px;
  background: #f6f7f9; color: #5a6470; user-select: none;
}
.hour-labels label:hover { background: #eceef1; color: #2b3138; }
.hour-panel { display: none; }
.hour-panel h3 { margin: 0 0 4px; font-size: 11pt; }
.fig-grid.cols-1 { grid-template-columns: minmax(0, 620px); }
.fig-grid.cols-2 { grid-template-columns: repeat(2, 1fr); }
.fig-grid.cols-3 { grid-template-columns: repeat(3, 1fr); }
/* Hour-panel figures wrap a Markdown image, which the converter puts inside a
   <p>; strip that paragraph's spacing so the card matches the others. */
.hour-panel figure p { margin: 0; }

/* ---- Click-to-zoom ---------------------------------------------------
   CSS only, via :target. The thumbnail is a link to the overlay's id; the
   overlay is a link back to '#_', so clicking anywhere closes it. No
   JavaScript, so it works from a file:// path with no server. Without CSS
   the page still shows every image inline, just without the zoom.        */
a.zoom-in { display: block; cursor: zoom-in; }
a.zoom-out { display: none; }

/* When a figure is the :target, it becomes the full-screen overlay itself —
   the same <img> is scaled up rather than a second copy being loaded. */
figure.zoomable:target {
  position: fixed; inset: 0; z-index: 100; margin: 0; border: none;
  border-radius: 0; padding: 24px;
  background: rgba(16, 19, 23, 0.9);
  display: flex; align-items: center; justify-content: center;
}
figure.zoomable:target a.zoom-in {
  cursor: default; max-width: 100%; max-height: 100%;
}
figure.zoomable:target img {
  /* Natural size, capped to the viewport: a large figure fills the screen,
     a small one is not stretched past its real resolution. */
  max-width: 100%; max-height: calc(100vh - 48px);
  width: auto; height: auto;
  border-radius: 4px; background: #fff;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
}
/* Caption is noise at full size, and the close target must cover the backdrop
   so clicking anywhere outside the image dismisses it. */
figure.zoomable:target figcaption { display: none; }
figure.zoomable:target a.zoom-out {
  display: block; position: absolute; inset: 0; cursor: zoom-out; z-index: -1;
}

/* Printing must not leave a figure stuck in its overlay state. */
@media print {
  figure.zoomable:target {
    position: static; background: none; padding: 8px; display: block;
  }
  figure.zoomable:target figcaption { display: block; }
  a.zoom-out { display: none !important; }
}

/* One rule per hour: the checked input highlights its label and shows its
   panel. Generated rather than hand-written so the two never drift.      */
__HOUR_TAB_RULES__

/* Print/PDF: no way to click a tab on paper, so every hour is laid out in
   sequence instead of only the selected one.                             */
@media print {
  .hour-labels { display: none; }
  .hour-panel { display: block !important; page-break-inside: avoid; }
}
"""


def _hour_tab_rules() -> str:
    """Per-hour CSS rules pairing each radio input with its label and panel."""
    rules = []
    for h in range(24):
        hh = f"{h:02d}"
        rules.append(
            f'.hour-tabs > #hour-{hh}:checked ~ .hour-labels label[for="hour-{hh}"] '
            '{ background: #2b6cb0; border-color: #2b6cb0; color: #fff; }')
        rules.append(
            f'.hour-tabs > #hour-{hh}:checked ~ #panel-{hh} '
            '{ display: block; }')
    return "\n".join(rules)


CSS = CSS.replace("__HOUR_TAB_RULES__", _hour_tab_rules())


def markdown_to_html(md_text: str, title: str,
                     base_dir: Optional[Path] = None) -> str:
    """Render Markdown to a styled, standalone HTML document.

    Relative image paths are inlined as data URIs so the HTML (and the PDF
    printed from it) works when moved away from the experiment directory.
    """
    import re
    import markdown as md_lib
    body = md_lib.markdown(
        md_text,
        extensions=["tables", "attr_list", "md_in_html", "sane_lists"],
    )
    if base_dir is not None:
        def inline(match: "re.Match[str]") -> str:
            src = match.group(1)
            if src.startswith(("data:", "http:", "https:", "file:")):
                return match.group(0)
            path = (base_dir / src).resolve()
            if not path.is_file():
                return match.group(0)
            # Per-hour figures are displayed two or three across inside a tab
            # panel, so they never need the full-width budget the whole-day
            # figures get. With 48 of them in a run, embedding each at 1400 px
            # put the HTML above 30 MB for pixels no one can see.
            per_hour = path.name.startswith(("count_error_h", "counts_loglog_h"))
            return f'src="{_image_data_uri(path, max_width=620 if per_hour else 1400)}"'
        body = re.sub(r'src="([^"]+)"', inline, body)
    return (f"<!DOCTYPE html>\n<html><head><meta charset='utf-8'/>"
            f"<title>{title}</title><style>{CSS}</style></head>"
            f"<body>\n{body}\n</body></html>\n")


def find_browser() -> Optional[str]:
    """An installed Chromium-family browser that can print to PDF."""
    candidates = [
        os.environ.get("CHROME_PATH"),
        shutil.which("msedge"), shutil.which("chrome"),
        shutil.which("chromium"), shutil.which("chromium-browser"),
        shutil.which("google-chrome"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


def html_to_pdf(html_path: Path, pdf_path: Path) -> bool:
    browser = find_browser()
    if not browser:
        return False
    with tempfile.TemporaryDirectory() as profile:
        cmd = [
            browser, "--headless", "--disable-gpu", "--no-sandbox",
            f"--user-data-dir={profile}",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=180)
        except (subprocess.TimeoutExpired, OSError) as e:
            print(f"  PDF step failed: {e}", file=sys.stderr)
            return False
    if not pdf_path.is_file():
        err = (proc.stderr or b"").decode("utf-8", "replace").strip()
        if err:
            print(f"  PDF step failed: {err.splitlines()[-1][:200]}", file=sys.stderr)
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a Markdown + HTML + PDF report for an experiment.",
        epilog="The server has no browser, so it writes report.md and report.html "
               "only. To get the PDF, copy report.html to a machine with "
               "Edge/Chrome and run:  python scripts/experiment_report.py "
               "--html-to-pdf report.html")
    ap.add_argument("experiment", nargs="?", help="experiment directory")
    ap.add_argument("--html-to-pdf", metavar="HTML",
                    help="render an existing report.html to PDF and exit "
                         "(for turning a server-generated report into a PDF locally)")
    ap.add_argument("--baseline", help="optional experiment to compare against")
    ap.add_argument("--output-dir", help="where to write (default: the experiment dir)")
    ap.add_argument("--no-pdf", action="store_true", help="write .md and .html only")
    args = ap.parse_args()

    # Standalone renderer: turn an existing (server-generated) report.html
    # into a PDF on a machine that has a browser.
    if args.html_to_pdf:
        html_path = Path(args.html_to_pdf)
        if not html_path.is_file():
            raise SystemExit(f"ERROR: not a file: {html_path}")
        pdf_path = html_path.with_suffix(".pdf")
        if html_to_pdf(html_path, pdf_path):
            print(f"wrote {pdf_path}  ({pdf_path.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            raise SystemExit(
                "ERROR: no Chromium-family browser found. Set CHROME_PATH, or "
                "open the HTML and print to PDF.")
        return

    if not args.experiment:
        ap.error("an experiment directory is required (or use --html-to-pdf)")
    exp_dir = Path(args.experiment)
    if not exp_dir.is_dir():
        raise SystemExit(f"ERROR: not a directory: {exp_dir}")
    run = load_run(exp_dir)
    baseline = load_run(Path(args.baseline)) if args.baseline else None

    out_dir = Path(args.output_dir) if args.output_dir else exp_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # The Markdown keeps relative image paths so it stays small and readable;
    # the HTML inlines them so the file (and the PDF printed from it) is
    # self-contained.
    md_text = build_markdown(run, baseline, embed_dir=out_dir)
    md_path = out_dir / "report.md"
    md_path.write_text(md_text, encoding="utf-8")
    print(f"wrote {md_path}  ({md_path.stat().st_size / 1024:.0f} KB)")

    html_path = out_dir / "report.html"
    html_path.write_text(markdown_to_html(md_text, exp_dir.name, base_dir=out_dir),
                         encoding="utf-8")
    print(f"wrote {html_path}  ({html_path.stat().st_size / 1024 / 1024:.1f} MB)")

    if args.no_pdf:
        return
    pdf_path = out_dir / "report.pdf"
    if html_to_pdf(html_path, pdf_path):
        print(f"wrote {pdf_path}")
    else:
        print("  no Chromium-family browser found — skipped the PDF step.")
        print("  Set CHROME_PATH, or open report.html and print to PDF.")


if __name__ == "__main__":
    main()
