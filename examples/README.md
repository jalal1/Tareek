# Examples

Shared experiments for Tareek. Each example documents a real run for a
specific region — what was run, the configuration used, and the results.

Each example is a folder with a `README.md` and whatever supporting files the author
wants to show (config, small result CSVs, plots). Keep them small: commit the recipe
and the results, not large or regenerable inputs (OSM extracts, full MATSim
networks/plans).

## Index

| Region | Author | Date | Highlights |
|--------|--------|------|------------|
| [Twin Cities (Minneapolis–St. Paul)](twin-cities-jalal-20260623/) | Jalal | 2026-06-23 | 15 counties (MN+WI), 0.15 scaling, 100 iters; IQM sim/obs ratio 0.97, corr 0.77 |

## Contributing

It's simple — see [CONTRIBUTING.md](CONTRIBUTING.md):

1. Fork the repo and add a folder `examples/<region>-<author>-<YYYYMMDD>/`.
2. Write a `README.md` showing whatever you want about your experiment. Use the
   existing examples in this folder as a guide.
3. Open a pull request.
