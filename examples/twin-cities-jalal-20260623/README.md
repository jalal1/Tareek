# Twin Cities (Minneapolis–St. Paul) experiment — Jalal, 2026-06-23

A full Tareek run for the 15-county Minneapolis–St. Paul metro, compared against
FHA directional traffic counts using MATSim's `countscompare` output.

- **Experiment ID:** `experiment_20260623_053227`
- **Region:** Twin Cities metro — 13 Minnesota counties (incl. Hennepin/Minneapolis
  `27053` and Ramsey/St. Paul `27123`) + 2 Wisconsin counties (St. Croix `55109`,
  Pierce `55093`).
- **Scaling factor:** `0.15` (15% population sample) · **MATSim iterations:** 100
- **Runtime:** ~10 h total (MATSim ~9.6 h)

---

## How to reproduce

This run was produced by the standard Tareek pipeline using
[`config_used.json`](config_used.json). To re-run the same experiment locally, after
completing the project setup, copy this config into the repo's `config/` folder and
launch it:

```bash
# from the repository root, with the virtualenv activated
cp examples/twin-cities-jalal-20260623/config_used.json config/twin-cities.json
python run_experiment.py --config config/twin-cities.json
```

Output lands in `experiments/<experiment-id>/`. This is a large scenario
(~3.7M population, 100 MATSim iterations) and took ~10 h to run; use
`--skip-simulation` to generate plans only without running MATSim.

The bulky intermediate artifacts (`network.xml` ≈ 329 MB, `plans.xml` ≈ 210 MB, the
MATSim `output/` directory) are **not** committed here — they are regenerated from the
config by the run above.

For full prerequisites and environment setup, see the
**[project README](../../README.md#quick-start)**. The two API keys referenced in the
config are **optional** — see [Optional API keys](#optional-api-keys) below.

---

## Configuration

The exact configuration this run used:

- [`config_used.json`](config_used.json) — full Tareek config snapshot
- [`config.xml`](config.xml) — generated MATSim config
- [`counts.xml`](counts.xml) — observed counts fed to MATSim
- [`matched_devices.csv`](matched_devices.csv) — FHA count devices matched to network links
- [`experiment_summary.json`](experiment_summary.json) — full machine-readable run summary
- [`experiment_20260623_053227.log`](experiment_20260623_053227.log) — run log

### Key parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `scaling_factor` (population) | **0.15** | 15% sample of the full population |
| MATSim iterations | **100** | route/plan optimization |
| `flow_capacity_factor` | 0.10 | QSim flow capacity (≈ scaling factor) |
| `storage_capacity_factor` | 0.12 | QSim storage capacity (≥ flow factor) |
| volume scaling multiplier | 10.0× | applied to sim volumes for comparison |

### Scenario scale

| | |
|---|---|
| Total population (15 counties) | 3,689,612 |
| Plans generated (after scaling) | 467,216 (99.98% success) |
| Network | 275,034 nodes · 634,245 links |
| Simulated trips / legs | 1,273,382 / 1,309,158 |
| Generated mode split | car 89.7% · walk 8.3% · pt 2.0% |
| Transit supply | 3,134 routes · 8,681 departures |

---

## Headline results

Simulated vs. observed link volumes at **47 matched count devices** (94 directional
counts, 2,256 hourly comparison points). The most informative single signal is the
**interquartile mean sim/obs ratio**, which is robust to boundary stations capturing
through-traffic from outside the modeled area.

| Metric | Value | Read as |
|--------|------:|---------|
| **Interquartile mean ratio** | **0.974** | ~1.0 = well matched on the middle 50% of stations |
| Median station ratio | 0.985 | half of stations within 1.5% of observed totals |
| Correlation (sim vs obs) | 0.768 | strong positive |
| Peak-hour correlation | 0.683 | |
| MAE | 766 veh/h | mean absolute hourly error |
| RMSE | 1,119 veh/h | |
| Mean % error | −20.0% | slight under-count (boundary-station sensitive) |

> **On GEH:** a single network-wide *mean* GEH (or % of comparisons with GEH < 5)
> is a poor headline indicator here — it's dominated by a handful of boundary
> stations carrying through-traffic from outside the modeled area, which inflates
> the aggregate regardless of how well the interior matches. For a real read on
> match quality, look at the **[per-device count reports](#per-device-count-reports)
> below**, where GEH and the hourly sim-vs-observed profile can be judged
> station-by-station. (The raw aggregates are in
> [`evaluation/summary_metrics.json`](evaluation/summary_metrics.json) if you need them.)

Full per-hour comparison data: [`evaluation/volume_comparison.csv`](evaluation/volume_comparison.csv) ·
metrics JSON: [`evaluation/summary_metrics.json`](evaluation/summary_metrics.json)

### Count error map (7 AM)

Spatial distribution of simulated-vs-observed error at each count station during the
7 AM hour:

![Count error at 7 AM](evaluation/count_error_h07.png)

---

## MATSim simulation graphs

MATSim's own output graphs for this run (sim vs. real volumes by hour, error stats).

> **Viewing the interactive graphs:** GitHub does **not** render HTML in the browser —
> clicking the `.html` files below just shows their source. To use the interactive
> version, **download or clone** the [`graphs/`](graphs/) folder and open
> `graphs/start.html` locally. For an in-browser preview without downloading, use the
> **PNGs** in [`graphs/png/`](graphs/png/), which display inline on GitHub.

Interactive (local viewing): `graphs/start.html` is the index — it links to the daily
sim-vs-real chart, per-hour charts (`simVsRealVolumesHour{N}Iteration100.html`, hours
1–24), and the error/bias-error pages.

PNG previews (display on GitHub) — daily sim-vs-real scatter (iteration 100):

![Sim vs real daily volumes](graphs/png/simVsRealVolumes24Iteration100.png)

---

## Per-device count reports

Observed vs. simulated hourly profiles for each of the **94 directional count stations**
(MN = Minnesota, WI = Wisconsin; `dir` = FHA direction code, `link` = matched network
link). Click any thumbnail to open it full size.

<table>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_000040_1_linkid_182997.png"><img src="evaluation/device_reports/device_FHA_27_000040_1_linkid_182997.png" width="100%"></a><br><sub>MN 000040 dir1 (link 182997)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_000040_5_linkid_350001.png"><img src="evaluation/device_reports/device_FHA_27_000040_5_linkid_350001.png" width="100%"></a><br><sub>MN 000040 dir5 (link 350001)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_000382_1_linkid_279370.png"><img src="evaluation/device_reports/device_FHA_27_000382_1_linkid_279370.png" width="100%"></a><br><sub>MN 000382 dir1 (link 279370)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_000382_5_linkid_599453.png"><img src="evaluation/device_reports/device_FHA_27_000382_5_linkid_599453.png" width="100%"></a><br><sub>MN 000382 dir5 (link 599453)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_000420_3_linkid_251590.png"><img src="evaluation/device_reports/device_FHA_27_000420_3_linkid_251590.png" width="100%"></a><br><sub>MN 000420 dir3 (link 251590)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_000420_7_linkid_481082.png"><img src="evaluation/device_reports/device_FHA_27_000420_7_linkid_481082.png" width="100%"></a><br><sub>MN 000420 dir7 (link 481082)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_006461_3_linkid_216638.png"><img src="evaluation/device_reports/device_FHA_27_006461_3_linkid_216638.png" width="100%"></a><br><sub>MN 006461 dir3 (link 216638)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_006461_7_linkid_437849.png"><img src="evaluation/device_reports/device_FHA_27_006461_7_linkid_437849.png" width="100%"></a><br><sub>MN 006461 dir7 (link 437849)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_009110_1_linkid_86860.png"><img src="evaluation/device_reports/device_FHA_27_009110_1_linkid_86860.png" width="100%"></a><br><sub>MN 009110 dir1 (link 86860)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_009110_5_linkid_86871.png"><img src="evaluation/device_reports/device_FHA_27_009110_5_linkid_86871.png" width="100%"></a><br><sub>MN 009110 dir5 (link 86871)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_009556_3_linkid_273233.png"><img src="evaluation/device_reports/device_FHA_27_009556_3_linkid_273233.png" width="100%"></a><br><sub>MN 009556 dir3 (link 273233)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_009556_7_linkid_273234.png"><img src="evaluation/device_reports/device_FHA_27_009556_7_linkid_273234.png" width="100%"></a><br><sub>MN 009556 dir7 (link 273234)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010069_1_linkid_614318.png"><img src="evaluation/device_reports/device_FHA_27_010069_1_linkid_614318.png" width="100%"></a><br><sub>MN 010069 dir1 (link 614318)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010069_5_linkid_518418.png"><img src="evaluation/device_reports/device_FHA_27_010069_5_linkid_518418.png" width="100%"></a><br><sub>MN 010069 dir5 (link 518418)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010205_1_linkid_599475.png"><img src="evaluation/device_reports/device_FHA_27_010205_1_linkid_599475.png" width="100%"></a><br><sub>MN 010205 dir1 (link 599475)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010205_5_linkid_485401.png"><img src="evaluation/device_reports/device_FHA_27_010205_5_linkid_485401.png" width="100%"></a><br><sub>MN 010205 dir5 (link 485401)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010206_3_linkid_330203.png"><img src="evaluation/device_reports/device_FHA_27_010206_3_linkid_330203.png" width="100%"></a><br><sub>MN 010206 dir3 (link 330203)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010206_7_linkid_611478.png"><img src="evaluation/device_reports/device_FHA_27_010206_7_linkid_611478.png" width="100%"></a><br><sub>MN 010206 dir7 (link 611478)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010225_3_linkid_296940.png"><img src="evaluation/device_reports/device_FHA_27_010225_3_linkid_296940.png" width="100%"></a><br><sub>MN 010225 dir3 (link 296940)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010225_7_linkid_118471.png"><img src="evaluation/device_reports/device_FHA_27_010225_7_linkid_118471.png" width="100%"></a><br><sub>MN 010225 dir7 (link 118471)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010310_1_linkid_2867.png"><img src="evaluation/device_reports/device_FHA_27_010310_1_linkid_2867.png" width="100%"></a><br><sub>MN 010310 dir1 (link 2867)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010310_5_linkid_2220.png"><img src="evaluation/device_reports/device_FHA_27_010310_5_linkid_2220.png" width="100%"></a><br><sub>MN 010310 dir5 (link 2220)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010338_1_linkid_63099.png"><img src="evaluation/device_reports/device_FHA_27_010338_1_linkid_63099.png" width="100%"></a><br><sub>MN 010338 dir1 (link 63099)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010338_5_linkid_559393.png"><img src="evaluation/device_reports/device_FHA_27_010338_5_linkid_559393.png" width="100%"></a><br><sub>MN 010338 dir5 (link 559393)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010390_1_linkid_614815.png"><img src="evaluation/device_reports/device_FHA_27_010390_1_linkid_614815.png" width="100%"></a><br><sub>MN 010390 dir1 (link 614815)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010390_5_linkid_396159.png"><img src="evaluation/device_reports/device_FHA_27_010390_5_linkid_396159.png" width="100%"></a><br><sub>MN 010390 dir5 (link 396159)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010398_3_linkid_388817.png"><img src="evaluation/device_reports/device_FHA_27_010398_3_linkid_388817.png" width="100%"></a><br><sub>MN 010398 dir3 (link 388817)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010398_7_linkid_256197.png"><img src="evaluation/device_reports/device_FHA_27_010398_7_linkid_256197.png" width="100%"></a><br><sub>MN 010398 dir7 (link 256197)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010730_1_linkid_132947.png"><img src="evaluation/device_reports/device_FHA_27_010730_1_linkid_132947.png" width="100%"></a><br><sub>MN 010730 dir1 (link 132947)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010730_5_linkid_192350.png"><img src="evaluation/device_reports/device_FHA_27_010730_5_linkid_192350.png" width="100%"></a><br><sub>MN 010730 dir5 (link 192350)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010794_3_linkid_17842.png"><img src="evaluation/device_reports/device_FHA_27_010794_3_linkid_17842.png" width="100%"></a><br><sub>MN 010794 dir3 (link 17842)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010794_7_linkid_96910.png"><img src="evaluation/device_reports/device_FHA_27_010794_7_linkid_96910.png" width="100%"></a><br><sub>MN 010794 dir7 (link 96910)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010800_3_linkid_559300.png"><img src="evaluation/device_reports/device_FHA_27_010800_3_linkid_559300.png" width="100%"></a><br><sub>MN 010800 dir3 (link 559300)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010800_7_linkid_362774.png"><img src="evaluation/device_reports/device_FHA_27_010800_7_linkid_362774.png" width="100%"></a><br><sub>MN 010800 dir7 (link 362774)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010808_1_linkid_24632.png"><img src="evaluation/device_reports/device_FHA_27_010808_1_linkid_24632.png" width="100%"></a><br><sub>MN 010808 dir1 (link 24632)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010808_5_linkid_221155.png"><img src="evaluation/device_reports/device_FHA_27_010808_5_linkid_221155.png" width="100%"></a><br><sub>MN 010808 dir5 (link 221155)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010830_1_linkid_50648.png"><img src="evaluation/device_reports/device_FHA_27_010830_1_linkid_50648.png" width="100%"></a><br><sub>MN 010830 dir1 (link 50648)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010830_5_linkid_594011.png"><img src="evaluation/device_reports/device_FHA_27_010830_5_linkid_594011.png" width="100%"></a><br><sub>MN 010830 dir5 (link 594011)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010840_3_linkid_442423.png"><img src="evaluation/device_reports/device_FHA_27_010840_3_linkid_442423.png" width="100%"></a><br><sub>MN 010840 dir3 (link 442423)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010840_7_linkid_556282.png"><img src="evaluation/device_reports/device_FHA_27_010840_7_linkid_556282.png" width="100%"></a><br><sub>MN 010840 dir7 (link 556282)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010899_3_linkid_39132.png"><img src="evaluation/device_reports/device_FHA_27_010899_3_linkid_39132.png" width="100%"></a><br><sub>MN 010899 dir3 (link 39132)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010899_7_linkid_618468.png"><img src="evaluation/device_reports/device_FHA_27_010899_7_linkid_618468.png" width="100%"></a><br><sub>MN 010899 dir7 (link 618468)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010919_3_linkid_346143.png"><img src="evaluation/device_reports/device_FHA_27_010919_3_linkid_346143.png" width="100%"></a><br><sub>MN 010919 dir3 (link 346143)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_010919_7_linkid_456898.png"><img src="evaluation/device_reports/device_FHA_27_010919_7_linkid_456898.png" width="100%"></a><br><sub>MN 010919 dir7 (link 456898)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011191_3_linkid_366598.png"><img src="evaluation/device_reports/device_FHA_27_011191_3_linkid_366598.png" width="100%"></a><br><sub>MN 011191 dir3 (link 366598)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011191_7_linkid_437753.png"><img src="evaluation/device_reports/device_FHA_27_011191_7_linkid_437753.png" width="100%"></a><br><sub>MN 011191 dir7 (link 437753)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011196_3_linkid_324718.png"><img src="evaluation/device_reports/device_FHA_27_011196_3_linkid_324718.png" width="100%"></a><br><sub>MN 011196 dir3 (link 324718)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011196_7_linkid_602210.png"><img src="evaluation/device_reports/device_FHA_27_011196_7_linkid_602210.png" width="100%"></a><br><sub>MN 011196 dir7 (link 602210)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011205_3_linkid_113976.png"><img src="evaluation/device_reports/device_FHA_27_011205_3_linkid_113976.png" width="100%"></a><br><sub>MN 011205 dir3 (link 113976)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011205_7_linkid_304090.png"><img src="evaluation/device_reports/device_FHA_27_011205_7_linkid_304090.png" width="100%"></a><br><sub>MN 011205 dir7 (link 304090)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011228_1_linkid_124592.png"><img src="evaluation/device_reports/device_FHA_27_011228_1_linkid_124592.png" width="100%"></a><br><sub>MN 011228 dir1 (link 124592)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011228_5_linkid_120450.png"><img src="evaluation/device_reports/device_FHA_27_011228_5_linkid_120450.png" width="100%"></a><br><sub>MN 011228 dir5 (link 120450)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011236_3_linkid_589221.png"><img src="evaluation/device_reports/device_FHA_27_011236_3_linkid_589221.png" width="100%"></a><br><sub>MN 011236 dir3 (link 589221)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011236_7_linkid_543038.png"><img src="evaluation/device_reports/device_FHA_27_011236_7_linkid_543038.png" width="100%"></a><br><sub>MN 011236 dir7 (link 543038)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011238_3_linkid_500373.png"><img src="evaluation/device_reports/device_FHA_27_011238_3_linkid_500373.png" width="100%"></a><br><sub>MN 011238 dir3 (link 500373)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011238_7_linkid_327078.png"><img src="evaluation/device_reports/device_FHA_27_011238_7_linkid_327078.png" width="100%"></a><br><sub>MN 011238 dir7 (link 327078)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011273_1_linkid_456448.png"><img src="evaluation/device_reports/device_FHA_27_011273_1_linkid_456448.png" width="100%"></a><br><sub>MN 011273 dir1 (link 456448)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011273_5_linkid_525557.png"><img src="evaluation/device_reports/device_FHA_27_011273_5_linkid_525557.png" width="100%"></a><br><sub>MN 011273 dir5 (link 525557)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011280_1_linkid_24596.png"><img src="evaluation/device_reports/device_FHA_27_011280_1_linkid_24596.png" width="100%"></a><br><sub>MN 011280 dir1 (link 24596)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011280_5_linkid_58756.png"><img src="evaluation/device_reports/device_FHA_27_011280_5_linkid_58756.png" width="100%"></a><br><sub>MN 011280 dir5 (link 58756)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011283_3_linkid_102886.png"><img src="evaluation/device_reports/device_FHA_27_011283_3_linkid_102886.png" width="100%"></a><br><sub>MN 011283 dir3 (link 102886)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011283_7_linkid_498942.png"><img src="evaluation/device_reports/device_FHA_27_011283_7_linkid_498942.png" width="100%"></a><br><sub>MN 011283 dir7 (link 498942)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011464_1_linkid_603866.png"><img src="evaluation/device_reports/device_FHA_27_011464_1_linkid_603866.png" width="100%"></a><br><sub>MN 011464 dir1 (link 603866)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011464_5_linkid_92394.png"><img src="evaluation/device_reports/device_FHA_27_011464_5_linkid_92394.png" width="100%"></a><br><sub>MN 011464 dir5 (link 92394)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011510_1_linkid_21327.png"><img src="evaluation/device_reports/device_FHA_27_011510_1_linkid_21327.png" width="100%"></a><br><sub>MN 011510 dir1 (link 21327)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011510_5_linkid_330960.png"><img src="evaluation/device_reports/device_FHA_27_011510_5_linkid_330960.png" width="100%"></a><br><sub>MN 011510 dir5 (link 330960)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011516_1_linkid_359576.png"><img src="evaluation/device_reports/device_FHA_27_011516_1_linkid_359576.png" width="100%"></a><br><sub>MN 011516 dir1 (link 359576)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011516_5_linkid_557307.png"><img src="evaluation/device_reports/device_FHA_27_011516_5_linkid_557307.png" width="100%"></a><br><sub>MN 011516 dir5 (link 557307)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011517_1_linkid_464961.png"><img src="evaluation/device_reports/device_FHA_27_011517_1_linkid_464961.png" width="100%"></a><br><sub>MN 011517 dir1 (link 464961)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011517_5_linkid_109783.png"><img src="evaluation/device_reports/device_FHA_27_011517_5_linkid_109783.png" width="100%"></a><br><sub>MN 011517 dir5 (link 109783)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011726_3_linkid_184806.png"><img src="evaluation/device_reports/device_FHA_27_011726_3_linkid_184806.png" width="100%"></a><br><sub>MN 011726 dir3 (link 184806)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011726_7_linkid_184805.png"><img src="evaluation/device_reports/device_FHA_27_011726_7_linkid_184805.png" width="100%"></a><br><sub>MN 011726 dir7 (link 184805)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011747_3_linkid_193768.png"><img src="evaluation/device_reports/device_FHA_27_011747_3_linkid_193768.png" width="100%"></a><br><sub>MN 011747 dir3 (link 193768)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011747_7_linkid_591375.png"><img src="evaluation/device_reports/device_FHA_27_011747_7_linkid_591375.png" width="100%"></a><br><sub>MN 011747 dir7 (link 591375)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011749_3_linkid_273580.png"><img src="evaluation/device_reports/device_FHA_27_011749_3_linkid_273580.png" width="100%"></a><br><sub>MN 011749 dir3 (link 273580)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011749_7_linkid_273628.png"><img src="evaluation/device_reports/device_FHA_27_011749_7_linkid_273628.png" width="100%"></a><br><sub>MN 011749 dir7 (link 273628)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011759_1_linkid_149608.png"><img src="evaluation/device_reports/device_FHA_27_011759_1_linkid_149608.png" width="100%"></a><br><sub>MN 011759 dir1 (link 149608)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_011759_5_linkid_184708.png"><img src="evaluation/device_reports/device_FHA_27_011759_5_linkid_184708.png" width="100%"></a><br><sub>MN 011759 dir5 (link 184708)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_042507_3_linkid_479639.png"><img src="evaluation/device_reports/device_FHA_27_042507_3_linkid_479639.png" width="100%"></a><br><sub>MN 042507 dir3 (link 479639)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_27_042507_7_linkid_479057.png"><img src="evaluation/device_reports/device_FHA_27_042507_7_linkid_479057.png" width="100%"></a><br><sub>MN 042507 dir7 (link 479057)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_55_550002_3_linkid_436126.png"><img src="evaluation/device_reports/device_FHA_55_550002_3_linkid_436126.png" width="100%"></a><br><sub>WI 550002 dir3 (link 436126)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_55_550002_7_linkid_325449.png"><img src="evaluation/device_reports/device_FHA_55_550002_7_linkid_325449.png" width="100%"></a><br><sub>WI 550002 dir7 (link 325449)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_55_550006_3_linkid_600152.png"><img src="evaluation/device_reports/device_FHA_55_550006_3_linkid_600152.png" width="100%"></a><br><sub>WI 550006 dir3 (link 600152)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_55_550006_7_linkid_137440.png"><img src="evaluation/device_reports/device_FHA_55_550006_7_linkid_137440.png" width="100%"></a><br><sub>WI 550006 dir7 (link 137440)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_55_550008_3_linkid_478973.png"><img src="evaluation/device_reports/device_FHA_55_550008_3_linkid_478973.png" width="100%"></a><br><sub>WI 550008 dir3 (link 478973)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_55_550008_7_linkid_478972.png"><img src="evaluation/device_reports/device_FHA_55_550008_7_linkid_478972.png" width="100%"></a><br><sub>WI 550008 dir7 (link 478972)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_55_550153_3_linkid_266168.png"><img src="evaluation/device_reports/device_FHA_55_550153_3_linkid_266168.png" width="100%"></a><br><sub>WI 550153 dir3 (link 266168)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_55_550153_7_linkid_505330.png"><img src="evaluation/device_reports/device_FHA_55_550153_7_linkid_505330.png" width="100%"></a><br><sub>WI 550153 dir7 (link 505330)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_55_550154_3_linkid_314887.png"><img src="evaluation/device_reports/device_FHA_55_550154_3_linkid_314887.png" width="100%"></a><br><sub>WI 550154 dir3 (link 314887)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_55_550154_7_linkid_424822.png"><img src="evaluation/device_reports/device_FHA_55_550154_7_linkid_424822.png" width="100%"></a><br><sub>WI 550154 dir7 (link 424822)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_55_550216_1_linkid_220059.png"><img src="evaluation/device_reports/device_FHA_55_550216_1_linkid_220059.png" width="100%"></a><br><sub>WI 550216 dir1 (link 220059)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_55_550216_5_linkid_449002.png"><img src="evaluation/device_reports/device_FHA_55_550216_5_linkid_449002.png" width="100%"></a><br><sub>WI 550216 dir5 (link 449002)</sub></td>
</tr>
<tr>
<td width="25%"><a href="evaluation/device_reports/device_FHA_55_550880_1_linkid_103323.png"><img src="evaluation/device_reports/device_FHA_55_550880_1_linkid_103323.png" width="100%"></a><br><sub>WI 550880 dir1 (link 103323)</sub></td>
<td width="25%"><a href="evaluation/device_reports/device_FHA_55_550880_5_linkid_628070.png"><img src="evaluation/device_reports/device_FHA_55_550880_5_linkid_628070.png" width="100%"></a><br><sub>WI 550880 dir5 (link 628070)</sub></td>
</tr>
</table>

---

## Optional API keys

`config_used.json` references two external services. They are **optional** — the
pipeline runs without them; you'll just see warnings and some missing data (no ACS
mode-share calibration, and any `wmata.com`-hosted transit feed is skipped). The keys
in the committed config are **redacted** (`YOUR_..._KEY` placeholders). Both are free:

| Field in config | Service | Register (free) |
|-----------------|---------|-----------------|
| `data.census_api_key` | U.S. Census API — ACS commute data | https://api.census.gov/data/key_signup.html |
| `gtfs.api_keys["wmata.com"]` | WMATA GTFS feed (auth-required transit feed) | https://developer.wmata.com/ |

Most GTFS feeds need no key; `wmata.com` is the only one this region's feed discovery
required. If your region uses different authenticated feeds, add their keys under
`gtfs.api_keys` keyed by domain. **Never commit real keys** — keep placeholders in any
config you share.
