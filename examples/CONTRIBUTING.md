# Contributing an example

Sharing an experiment is easy.

1. **Fork the repo** and create a folder named `examples/<region>-<author>-<YYYYMMDD>/`
   (e.g. `examples/seattle-jalal-20260623/`).

2. **Add a `README.md`** showing whatever you want about your experiment — the region,
   what you ran, your configuration, and your results (plots and numbers welcome).
   The best guide is the existing examples in this folder: open one and follow the same
   shape.

3. **Add any small supporting files** you reference — `config_used.json`, small result
   CSVs, figures. Keep it small (target under 50 MB): commit the recipe and results, not
   large or regenerable inputs like OSM extracts or full MATSim networks/plans.

   > ⚠️ **Redact your secrets first.** `config_used.json` can contain real API keys
   > (e.g. `data.census_api_key`, anything under `gtfs.api_keys`). Replace each with a
   > `YOUR_..._KEY` placeholder before committing, and list the services contributors
   > need in your README. Never commit a live key.

4. **Add a row** to the index table in [`examples/README.md`](README.md).

5. **Open a pull request** against `main`. We'll take a quick look and merge.
